"""Qwen3-32B reads the 5 Phase 2 attack runs (Qwen3.5-27B acting, PR #72) and the frozen Q3 detector scores every call,
for the reading list (Modal, one H100). No training.

Same text and positions as the detector's training features (the v4 segment-end extraction):
  - reader messages from probe_shift.transcripts.reader_messages, with the system prompt replaced by the real honest
    prompt of the same main task (taken from the v4 staged inputs), reasoning dropped, 32k-token cut;
  - Qwen3-32B at the pinned revision, bf16, sdpa, the output of decoder layers 48 and 63, fp16, at
    probe_shift.reader.segment_end_positions.
Detector: run 2026-10-06_181548's outer model for the main task's fold (out of fold), at each readout's chosen layer:
content-token calls (call_end kind 0) at layer 63, all calls at layer 48, tool outputs at layer 63. Run score = mean
over the readout's calls.

Check first: the same cells' Sol e1 runs (cells.json `sol_e1`), read from their staged v4 messages through this code,
must reproduce the v4 extraction's positions exactly, its features to within fp16 noise, and the committed run scores.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.2' --with numpy --with boto3 --with pyyaml \
      modal run --detach run_read_phase2.py --cells <cells.json> --attacks <attack_check_*.jsonl>
"""

import hashlib
import io
import json
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REPO = HERE.parent.parent
APP = "probe-shift_aswin-surya_sub-q3followups_phase2read_kill-2026-11-06"
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
WORK = "probe-shift_aswin-surya_sub-q3attack_work_kill-2026-11-06"
EXTRACTION = "2026-10-06_la_segment-end-v4_qwen3-32b_alllayers_v1"
STAGED = "2026-10-06_la_segment-end-v4_qwen3-32b_v1"
BASE_RUN = "2026-10-06_181548"
LOGS = "s3://individual-aswin-surya-probe-shift-kill-2027-04-06/phase2/2026-10-07_attack_check/runs/2026-10-07_133058_attack_check/attack"
READOUTS = {"call_end_content": 63, "call_end": 48, "output_end": 63}
MAX_TOKENS = 32768
PLACEHOLDER = "<honest system prompt of this main task>"
# the Sol e1 re-read must match: fp16 features recomputed on the same GPU type, committed run scores rounded to 1e-4
CHECK = {"min_cosine": 0.999, "max_rel_l2": 0.02, "max_score_diff": 0.02}

app = modal.App(APP)
hf_cache = modal.Volume.from_name(HF_CACHE)
acts_vol = modal.Volume.from_name(ACTS, version=2)
work_vol = modal.Volume.from_name(WORK, version=2)
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("torch==2.8.0", "transformers==4.56.2", "accelerate==1.10.1", "numpy==2.3.3")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})
         .add_local_python_source("probe_shift"))


@app.function(image=image, gpu="H100!", memory=65536, volumes={"/hf": hf_cache, "/acts": acts_vol, "/work": work_vol}, timeout=3600)
def read(items: list[dict], checks: list[dict], folds: dict) -> dict:
    """items: Phase 2 runs (messages with PLACEHOLDER as system prompt); checks: Sol e1 runs ({key, main_task, cell,
    committed}) re-read from their staged messages. Returns per-call scores and the check results."""
    import time

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from probe_shift.attack_eval import call_end_kind
    from probe_shift.models import QWEN3_32B
    from probe_shift.reader import segment_end_positions

    t0 = time.time()
    staged, honest = {}, {}
    want = {c["key"] for c in checks}
    tasks = {it["main_task"] for it in items}
    for shard in sorted((Path("/acts") / STAGED / "inputs").glob("shard_*.jsonl")):
        for line in shard.read_text().splitlines():
            it = json.loads(line)
            if it["arm"] == "real-attack-e1" and it["key"] in want:
                staged[it["key"]] = it
            if it["arm"] == "real-honest" and it["main_task"] in tasks:
                honest.setdefault(it["main_task"], set()).add(it["messages"][0]["content"])
    if set(staged) != want or set(honest) != tasks or any(len(p) != 1 for p in honest.values()):
        raise ValueError(f"staged checks {len(staged)}/{len(want)}; honest prompts {({t: len(p) for t, p in honest.items()})} for {sorted(tasks)}")
    print(f"staged inputs read at {time.time() - t0:.0f} s", flush=True)
    tok = AutoTokenizer.from_pretrained(QWEN3_32B["name"], revision=QWEN3_32B["revision"])
    lm = AutoModelForCausalLM.from_pretrained(QWEN3_32B["name"], revision=QWEN3_32B["revision"], dtype=torch.bfloat16,
                                              device_map="cuda", attn_implementation="sdpa").eval()
    probes = {l: np.load(Path("/work/runs") / BASE_RUN / f"probes_L{l}.npz") for l in set(READOUTS.values())}
    probes_sha = {l: hashlib.sha256((Path("/work/runs") / BASE_RUN / f"probes_L{l}.npz").read_bytes()).hexdigest() for l in probes}

    def forward(messages):
        r = segment_end_positions(tok, messages, MAX_TOKENS)
        kept = [p for p in r["positions"] if p["kept"]]
        idx = torch.tensor([p["token"] for p in kept], device="cuda", dtype=torch.long)
        acts = {}

        def hook_for(layer):
            def hook(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                acts[layer] = h[0].index_select(0, idx).float().to(torch.float16).cpu().numpy()
            return hook

        handles = [lm.model.layers[l].register_forward_hook(hook_for(l)) for l in probes]
        try:
            with torch.inference_mode():
                lm.model(input_ids=torch.tensor([r["ids"]], device="cuda"), use_cache=False)
        finally:
            for h in handles:
                h.remove()
        return r, kept, acts

    def score(messages, kept, acts, fold):
        turns = [m for m in messages if m["role"] == "assistant"]
        calls = {(s, a): c["function"] for s, t in enumerate(turns) for a, c in enumerate(t.get("tool_calls") or [])}
        rows, per_run = [], {}
        for ro, l in READOUTS.items():
            pos = "output_end" if ro == "output_end" else "call_end"
            sel = []
            for j, p in enumerate(kept):
                if p["kind"] != pos:
                    continue
                kind = call_end_kind(calls[(p["step"], p["action"])]["arguments"], p["token_text"]) if pos == "call_end" else None
                if ro == "call_end_content" and kind != 0:
                    continue
                sel.append((j, p, kind))
            z, key = probes[l], f"{ro}__task_outer_{fold}"
            X = acts[l][[j for j, _, _ in sel]].astype(np.float32)
            s = ((X - z[f"{key}__mean"]) / z[f"{key}__std"]) @ z[f"{key}__w"] + np.float32(z[f"{key}__b"])
            per_run[ro] = {"mean": float(np.mean(s)) if len(s) else None, "calls": len(s)}
            for (j, p, kind), v in zip(sel, s):
                c = calls.get((p["step"], p["action"]))
                rows.append({"readout": ro, "layer": l, "step": p["step"], "action": p["action"], "tool": p["tool"], "call_kind": kind,
                             "token_text": p["token_text"], "score": float(v), "function": c["name"] if c else None})
        return rows, per_run

    out = {"checks": [], "runs": [], "probes_sha256": {str(k): v for k, v in probes_sha.items()}}
    for c in checks:
        it = staged[c["key"]]
        meta = json.loads((Path("/acts") / EXTRACTION / "meta" / "real-attack-e1" / f"{it_filename(c['key'])}.json").read_text())
        r, kept, acts = forward(it["messages"])
        ref_pos = [(p["kind"], p["step"], p["action"], p["token"]) for p in meta["positions"] if p["kept"]]
        got_pos = [(p["kind"], p["step"], p["action"], p["token"]) for p in kept]
        if ref_pos != got_pos:
            raise ValueError(f"{c['key']}: positions differ from the extraction")
        rows_ref = np.array([p["row"] for p in meta["positions"] if p["kept"]]) + meta["row_start"]
        diff = {}
        for l in probes:
            ref = np.load(Path("/acts") / EXTRACTION / "by_layer" / f"L{l}" / f"{meta['chunk']}.npy", mmap_mode="r")[rows_ref].astype(np.float32)
            got = acts[l].astype(np.float32)
            diff[l] = {"max_abs": float(np.abs(ref - got).max()), "rel_l2": float(np.linalg.norm(ref - got) / np.linalg.norm(ref)),
                       "min_cosine": float(np.min((ref * got).sum(1) / np.linalg.norm(ref, axis=1) / np.linalg.norm(got, axis=1)))}
        _, per_run = score(it["messages"], kept, acts, folds[c["main_task"]])
        out["checks"].append({"key": c["key"], "cell": c["cell"], "feature_diff": {str(k): v for k, v in diff.items()},
                              "run_scores": per_run, "committed": c["committed"]})
        print(f"check {c['cell']}: diff {diff} | scores {per_run} vs committed {c['committed']}", flush=True)
        bad = [f"L{l} non-finite" for l in probes if not np.isfinite(acts[l]).all()]
        bad += [f"L{l} {d}" for l, d in diff.items() if not (d["min_cosine"] >= CHECK["min_cosine"] and d["rel_l2"] <= CHECK["max_rel_l2"])]
        bad += [f"{ro} {per_run[ro]['mean']} vs {v}" for ro, v in c["committed"].items()
                if per_run[ro]["mean"] is None or abs(per_run[ro]["mean"] - v) > CHECK["max_score_diff"]]
        if bad:
            raise ValueError(f"check {c['key']} failed ({CHECK}): {bad}")
    for it in items:
        messages = [{"role": "system", "content": honest[it["main_task"]].copy().pop()}] + it["messages"][1:]
        r, kept, acts = forward(messages)
        rows, per_run = score(messages, kept, acts, folds[it["main_task"]])
        turns = [m for m in messages if m["role"] == "assistant"]
        out["runs"].append({k: it[k] for k in ("env", "cell", "main_task", "file", "sample_id", "side_task_success")}
                           | {"fold": folds[it["main_task"]], "n_tokens": r["n_tokens"], "truncated": r["truncated"],
                              "kept_positions": len(kept), "run_scores": per_run, "calls": rows,
                              "call_text": {f"{s}:{a}": json.dumps(c["function"]["arguments"])[:600]
                                            for s, t in enumerate(turns) for a, c in enumerate(t.get("tool_calls") or [])},
                              "agent_text": {str(s): (t["content"] or "")[:600] for s, t in enumerate(turns) if (t["content"] or "").strip()}})
        print(f"{it['env']}: {r['n_tokens']} tokens (truncated {r['truncated']}), scores {per_run}", flush=True)
    out["seconds"] = round(time.time() - t0)
    return out


def it_filename(key: str) -> str:
    """The extraction's file stem for a key (run_extract_all_layers.filename)."""
    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


@app.local_entrypoint()
def main(cells: str, attacks: str):
    from probe_shift.transcripts import _get, reader_messages, system_prompt

    cell_rows = {c["cell"]: c for c in json.loads(Path(cells).read_text())["cells"]}
    atk = [json.loads(l) for l in Path(attacks).read_text().splitlines()]
    atk = [a for a in atk if a["arm"] == "attack"]
    res36 = REPO / "experiments/2026-10-06_q3_attack_detector/results"
    runs = json.loads((res36 / f"run_scores_{BASE_RUN}/runs.json").read_text())
    folds = {}
    for r in runs:
        if folds.setdefault(r["main_task"], r["fold"]) != r["fold"]:
            raise ValueError(f"{r['main_task']} in two folds")
    headline = {(r["arm"], r["key"]): r for r in map(json.loads, (res36 / f"headline_run_scores_{BASE_RUN}__2026-10-06_182739.jsonl").read_text().splitlines())}
    items, checks = [], []
    bucket, prefix = LOGS.removeprefix("s3://").split("/", 1)
    for a in atk:
        c = cell_rows[a["cell"]]
        with zipfile.ZipFile(io.BytesIO(_get(bucket, f"{prefix}/{a['env']}/{a['file']}"))) as z:
            samples = [json.loads(z.read(n)) for n in z.namelist() if n.startswith("samples/") and n.endswith(".json")]
        (s,) = [x for x in samples if x["id"] == a["sample_id"]]
        if hashlib.sha256(system_prompt(s).encode()).hexdigest() != c["sol_system_sha256"]:
            raise ValueError(f"{a['cell']}: system prompt is not Sol's")
        items.append({k: a[k] for k in ("env", "cell", "main_task", "file", "sample_id", "side_task_success")}
                     | {"messages": reader_messages(s, PLACEHOLDER)})
        e1 = c["sol_e1"]
        key = f"{e1['file'].removesuffix('.eval')}__{e1['id'].replace('/', '_')}__{e1['epoch']}"
        h = headline[("real-attack-e1", key)]
        checks.append({"key": key, "cell": a["cell"], "main_task": a["main_task"],
                       "committed": {"call_end_content": h["probe_content_mean"], "call_end": h["probe_allcalls_mean"]}})
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=REPO).stdout.strip())
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    print(f"{len(items)} Phase 2 attacks, {len(checks)} Sol e1 checks; commit {head[:8]}{' (dirty)' if dirty else ''}", flush=True)
    out = read.remote(items, checks, {it["main_task"]: folds[it["main_task"]] for it in items})
    rec = {"stamp": stamp, "git": head, "git_dirty": dirty, "app": APP, "extraction": EXTRACTION, "staged": STAGED, "base_run": BASE_RUN,
           "logs": LOGS, "readouts": READOUTS, "max_tokens": MAX_TOKENS, "check_tolerances": CHECK,
           "inputs": {k: {"file": p, "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()} for k, p in (("cells", cells), ("attacks", attacks))}} | out
    path = HERE / "data" / "unredacted" / f"phase2_read_{stamp}.json"  # call and agent text can hold seeded tokens: git-ignored
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        f.write(json.dumps(rec, indent=1))
    print(f"-> {path} ({out['seconds']} s)")
