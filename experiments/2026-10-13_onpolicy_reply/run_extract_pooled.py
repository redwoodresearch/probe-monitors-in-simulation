"""The averaged-token readout (RULE.md Amendment 4): a copy of experiments/2026-10-08_q3_onpolicy_figures/run_extract_replay.py
(main @ 8ef61b08: the same fidelity gate, leases, chunking, resume and padding) that hooks layers 39, 51 and 63 only and,
per kept tool output, stores the MEAN of the residual stream over the output's content tokens beside its last-token row.

Content span of an output_end readout r: from p + 2 through r, where p is the nearest `<tool_response>` before r. Checks
(Amendment 4): no `</tool_response>` in (p, r]; the token at p + 1 decodes to a string starting with the newline symbol
(the template's newline, never wholly inside the content); r >= p + 2. A failing output is excluded and counted; more than
1% failing in a shard stops it. Rows: by_layer/L<l>/<chunk>.npy (mean, fp16) and last/L<l>/<chunk>.npy (last token, fp16);
meta/<arm>/<filename>.json lists each output's call, row and span.

Configs are built from the original extraction configs on main (inputs, inputs_sha256, shards, chunk_size, pad_to), with
an arm filter (the staged files are hash-checked whole, then only the listed arms are read). `--test`: one shard's
longest epoch of one config, read twice, gated on the last-token rows equalling the original extraction's stored rows
for the same calls (max |diff| <= 1e-2).

  PYTHONPATH=src uv run --no-project --python 3.14 --with 'modal==1.6.1' --with pyyaml modal run [--detach] \\
      experiments/2026-10-13_onpolicy_reply/run_extract_pooled.py [--test --only <config>] [--attempt 1]
"""

import hashlib
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-onpol-reply_pooled_kill-2026-11-13"
ACTS = "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08"
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
LEASES = "probe-shift_aswin-surya_sub-q3onpol_leases_kill-2026-11-08"
PIP = ["torch==2.8.0", "transformers==5.19.0", "accelerate>=1.1.0", "numpy", "huggingface_hub", "flash-linear-attention==0.5.2"]
TIMEOUT_S, LAYERS, SPAN_FAIL_MAX = 4 * 3600, [39, 51, 63], 0.01
BASE4 = ["attack", "honest_a", "honest_b", "sim_attack@september"]
CONFIGS = {  # original config on main -> arms read (RULE.md Amendment 4: option C)
    "extract_rerun_segends.yaml": BASE4, "extract_scaleup_canary.yaml": BASE4, "extract_scaleup_l2.yaml": BASE4, "extract_scaleup_topup.yaml": BASE4,
    "extract_second_attack.yaml": ["attack"],
    "extract_retrain_rerun_strict_segends_scrub1.yaml": BASE4, "extract_retrain_scaleup_canary_segends_scrub1.yaml": BASE4,
    "extract_retrain_scaleup_l2_segends_scrub1.yaml": BASE4, "extract_retrain_scaleup_topup_segends_scrub1.yaml": BASE4,
    "extract_retrain_second_attack_segends_scrub1.yaml": ["attack"],
    "extract_swapsys_rerun_attack.yaml": ["attack"], "extract_swapsys_canary_attack.yaml": ["attack"], "extract_swapsys_l2_attack.yaml": ["attack"],
    "extract_swapsys_topup_attack.yaml": ["attack"], "extract_swapsys_second_attack.yaml": ["attack"],
    "extract_swapsys2_rerun_sim.yaml": ["sim_attack@september"], "extract_swapsys2_canary_sim.yaml": ["sim_attack@september"],
    "extract_swapsys2_l2_sim.yaml": ["sim_attack@september"], "extract_swapsys2_topup_sim.yaml": ["sim_attack@september"]}

app = modal.App(APP)
acts_vol = modal.Volume.from_name(ACTS, create_if_missing=False, version=2)
leases = modal.Dict.from_name(LEASES, create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PIP).env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})


def ids_sha(ids) -> str:
    import numpy as np

    return hashlib.sha256(np.asarray(ids, np.int32).tobytes()).hexdigest()


def gate(ep: dict) -> None:  # run_extract_replay.gate, unchanged
    bad = [c["irid"] for c in ep["calls"] if c["n"] > len(ep["ids"]) or ids_sha(ep["ids"][: c["n"]]) != c["sha256"]
           or (c["readout"] is not None and not 0 <= c["readout"] < c["n"])]
    if bad:
        raise ValueError(f"fidelity gate: {len(bad)} calls of {ep['key']} differ from the logged tokens, e.g. {bad[:3]}")


def save_npy(path: Path, x) -> None:
    import numpy as np

    with open(path, "wb") as fh:
        np.save(fh, x)


def filename(key: str) -> str:  # run_extract_replay.filename, unchanged (the original meta's file name)
    return hashlib.sha1(key.encode()).hexdigest()[:20]


@app.function(image=image, gpu="H200", memory=65536, timeout=TIMEOUT_S, retries=0, max_containers=40,
              volumes={"/hf": modal.Volume.from_name(HF_CACHE).read_only(), "/acts": acts_vol})
def extract_shard(cfg: dict, shard: int) -> dict:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    me = f"{os.environ.get('MODAL_TASK_ID', '?')}@{time.time():.0f}"
    key = f"{cfg['run']}/shard{shard:03d}"
    if not leases.put(f"{key}/a{cfg['attempt']}", me, skip_if_exists=True):
        raise RuntimeError(f"shard {shard} attempt {cfg['attempt']} already has a writer; rerun with a new --attempt")
    held = leases.get(key)
    if held is not None and held["attempt"] >= cfg["attempt"]:
        raise RuntimeError(f"shard {shard} is held by attempt {held['attempt']}; rerun with a later --attempt")
    leases[key] = {"writer": me, "attempt": cfg["attempt"]}

    def check():
        if (leases.get(key) or {}).get("writer") != me:
            raise RuntimeError(f"shard {shard}: a later attempt took over; stopping before writing")

    def put(path: Path, write):
        tmp = path.with_name(f".{path.name}.{cfg['attempt']}.tmp")
        write(tmp)
        check()
        os.replace(tmp, path)

    acts_vol.reload()
    run_dir, src = Path("/acts") / cfg["run"], Path("/acts") / cfg["inputs"] / "inputs"
    # the files the original config pins, each hash-checked (a staging folder may have gained other arms' files since)
    raw = {n: (src / n).read_bytes() for n in sorted(cfg["inputs_sha256"])}
    if {n: hashlib.sha256(b).hexdigest() for n, b in raw.items()} != cfg["inputs_sha256"]:
        raise ValueError(f"the staged inputs in {src} differ from the config's sha256s")
    if missing := set(cfg["arms"]) - {n[:-6] for n in raw}:
        raise ValueError(f"arms {sorted(missing)} are not among the config's pinned files")
    eps = sorted((json.loads(line) for n, b in raw.items() if n[:-6] in cfg["arms"] for line in b.decode().splitlines()),
                 key=lambda e: (-len(e["ids"]), e["key"]))
    mine = eps[:1] if cfg.get("test") else eps[shard:: cfg["shards"]]
    for ep in mine:
        gate(ep)

    def done(ep) -> bool:
        f = run_dir / "meta" / ep["arm"] / f"{filename(ep['key'])}.json"
        if not f.exists():
            return False
        m = json.loads(f.read_text())
        idx = json.loads((run_dir / "chunks" / f"{m['chunk']}.json").read_text())
        if m["inputs_sha256"] != cfg["inputs_sha256"][f"{ep['arm']}.jsonl"] or m["key"] not in {e["key"] for e in idx["epochs"]}:
            raise ValueError(f"{f} was written from other inputs or its chunk does not list it")
        return True

    todo = [ep for ep in mine if not done(ep)]
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    TR, TRC = tok.convert_tokens_to_ids("<tool_response>"), tok.convert_tokens_to_ids("</tool_response>")
    if (TR, TRC) != (248066, 248067):
        raise ValueError(f"unexpected tool_response ids {TR}, {TRC}")
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    lm = AutoModelForCausalLM.from_pretrained(cfg["model"], revision=cfg["revision"], dtype=torch.bfloat16, device_map="cuda",
                                              attn_implementation="sdpa").eval()
    base = lm.model
    load_s = round(time.time() - t0)

    def spans_of(ep):
        ok, fail = [], 0
        for c in ep["calls"]:
            if c.get("kind") != "output_end" or c["readout"] is None:
                continue
            r = c["readout"]
            p = next((i for i in range(r - 1, -1, -1) if ep["ids"][i] == TR), None)
            if p is None or TRC in ep["ids"][p + 1: r + 1] or r < p + 2 or not tok.convert_ids_to_tokens(ep["ids"][p + 1]).startswith("Ċ"):
                fail += 1
                continue
            ok.append((c, p, r))
        return ok, fail

    def read(ids, P, R):
        acts = {}

        def hook(l):
            def f(m, a, out):
                h = (out[0] if isinstance(out, tuple) else out)[0]
                cs = torch.cumsum(h.float(), 0)
                mean = (cs.index_select(0, R) - cs.index_select(0, P + 1)) / (R - P - 1).unsqueeze(1).float()
                acts[l] = (mean.to(torch.float16).cpu().numpy(), h.index_select(0, R).to(torch.float16).cpu().numpy())
            return f
        hooks = [base.layers[l].register_forward_hook(hook(l)) for l in LAYERS]
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        try:
            with torch.inference_mode():
                base(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)
            torch.cuda.synchronize()
        finally:
            for h in hooks:
                h.remove()
        return acts, round(time.time() - t, 2), round(torch.cuda.max_memory_allocated() / 2**30, 1)

    for d in ("by_layer", "last"):
        for l in LAYERS:
            (run_dir / d / f"L{l}").mkdir(parents=True, exist_ok=True)
    (run_dir / "chunks").mkdir(parents=True, exist_ok=True)
    buf, metas, stats, peaks, chunk, n_out, n_fail = {(d, l): [] for d in ("by_layer", "last") for l in LAYERS}, [], [], [], 0, 0, 0
    test_gate = None

    def flush():
        nonlocal chunk
        if not metas:
            return
        name = f"shard{shard:03d}_a{cfg['attempt']}_chunk{chunk:03d}"
        xs = {k: np.concatenate(v) for k, v in buf.items()}
        nonfinite = sum(int((~np.isfinite(x)).sum()) for x in xs.values())
        if nonfinite:
            raise ValueError(f"{name}: {nonfinite} non-finite activations; nothing of it is published")
        for (d, l), x in xs.items():
            put(run_dir / d / f"L{l}" / f"{name}.npy", lambda t, x=x: save_npy(t, x))
            buf[(d, l)].clear()
        start = 0
        for m in metas:
            m |= {"chunk": name, "row_start": start, "inputs_sha256": cfg["inputs_sha256"][f"{m['arm']}.jsonl"]}
            start += m["n_rows"]
        index = {"layers": LAYERS, "rows": start, "epochs": [{k: m[k] for k in ("arm", "key", "row_start", "n_rows")} for m in metas]}
        put(run_dir / "chunks" / f"{name}.json", lambda t: t.write_text(json.dumps(index)))
        for m in metas:
            dd = run_dir / "meta" / m["arm"]
            dd.mkdir(parents=True, exist_ok=True)
            put(dd / f"{filename(m['key'])}.json", lambda t, m=m: t.write_text(json.dumps(m)))
        stats.append({"chunk": name, "epochs": len(metas), "rows": start})
        metas.clear()
        chunk += 1
        acts_vol.commit()

    for ep in todo:
        check()
        gate(ep)
        kept, fail = spans_of(ep)
        n_out, n_fail = n_out + len(kept) + fail, n_fail + fail
        ids = ep["ids"] + [pad_id] * (-len(ep["ids"]) % cfg["pad_to"])
        P = torch.tensor([p for _, p, _ in kept], device="cuda", dtype=torch.long)
        R = torch.tensor([r for _, _, r in kept], device="cuda", dtype=torch.long)
        acts, seconds, peak = read(ids, P, R) if kept else ({l: (np.zeros((0, 5120), np.float16),) * 2 for l in LAYERS}, 0.0, 0.0)
        peaks.append(peak)
        timing = {"forward_s": seconds, "peak_gib": peak, "padded_tokens": len(ids)}
        if cfg.get("test"):
            again, s2, _ = read(ids, P, R)
            timing |= {"forward_s_repeat": s2, "repeat_max_abs_diff": max(float(np.abs(acts[l][0].astype(np.float32) - again[l][0].astype(np.float32)).max(initial=0)) for l in LAYERS)}
            om = json.loads((Path("/acts") / cfg["orig_run"] / "meta" / ep["arm"] / f"{filename(ep['key'])}.json").read_text())
            orow = {(c["irid"], c["call"]): c["row"] for c in om["calls"] if c.get("kind") == "output_end" and c.get("row") is not None}
            diffs = []
            for l in LAYERS:
                O = np.load(Path("/acts") / cfg["orig_run"] / "by_layer" / f"L{l}" / f"{om['chunk']}.npy", mmap_mode="r")
                for j, (c, _, _) in enumerate(kept):
                    if (c["irid"], c["call"]) in orow:
                        diffs.append(float(np.abs(acts[l][1][j].astype(np.float32) - np.asarray(O[om["row_start"] + orow[(c["irid"], c["call"])]], np.float32)).max()))
            test_gate = {"outputs_compared": len(diffs) // len(LAYERS), "max_abs_diff_last_token_vs_original": max(diffs) if diffs else None,
                         "passes": bool(diffs) and max(diffs) <= 1e-2, "span_fail": fail, "outputs": len(kept) + fail}
        for l in LAYERS:
            buf[("by_layer", l)].append(acts[l][0])
            buf[("last", l)].append(acts[l][1])
        metas.append({k: ep[k] for k in ("key", "arm", "run", "task", "epoch", "attempt_index", "attempt_kind", "redrawn_because", "policy_called")}
                     | {"n_tokens": len(ep["ids"]), "n_rows": len(kept), "span_fail": fail, **timing,
                        "calls": [{"irid": c["irid"], "call": c["call"], "turn": c.get("turn"), "call_index": c.get("call_index", 0),
                                   "readout": r, "span": [p + 2, r], "row": j} for j, (c, p, r) in enumerate(kept)]})
        if len(metas) == cfg["chunk_size"]:
            flush()
    if n_out and n_fail / n_out > SPAN_FAIL_MAX:
        raise ValueError(f"shard {shard}: {n_fail} of {n_out} outputs failed the span checks (> 1%); nothing more is published")
    flush()
    return {"run": cfg["run"], "shard": shard, "epochs": len(mine), "done_before": len(mine) - len(todo), "chunks": stats, "load_s": load_s,
            "seconds": round(time.time() - t0), "max_peak_gib": max(peaks, default=None), "outputs": n_out, "span_fail": n_fail, "test_gate": test_gate}


@app.local_entrypoint()
def main(test: bool = False, only: str = "", attempt: int = 1):
    import io

    import yaml

    git = {"commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip(),
           "dirty": bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=HERE).stdout.strip())}
    if git["dirty"]:
        raise SystemExit("uncommitted tracked changes")
    from probe_shift.models import QWEN35_27B

    jobs, cfgs = [], []
    for name, arms in CONFIGS.items():
        if only and name != only:
            continue
        orig = yaml.safe_load(subprocess.run(["git", "show", f"origin/main:experiments/2026-10-08_q3_onpolicy_figures/config/{name}"],
                                             capture_output=True, text=True, check=True, cwd=HERE).stdout)
        cfg = {k: orig[k] for k in ("inputs", "inputs_sha256", "model", "shards", "chunk_size", "pad_to")} | {
            "orig_run": orig["run"], "run": f"2026-10-13_onpolreply_pooled_{orig['run']}" + ("_test" if test else ""), "arms": arms,
            "layers": LAYERS, "revision": QWEN35_27B["revision"], "attempt": attempt, "test": test, "source_config": name}
        if cfg["model"] != QWEN35_27B["name"]:
            raise ValueError(f"{name}: {cfg['model']} is not the pinned Qwen3.5-27B")
        settings = {k: v for k, v in cfg.items() if k != "attempt"}
        try:
            stored = json.loads(b"".join(acts_vol.read_file(f"{cfg['run']}/config.json")))
        except (FileNotFoundError, modal.exception.NotFoundError):
            stored = None
        if stored is not None and stored != settings:
            raise ValueError(f"{cfg['run']} exists with other settings; use a new run name")
        with acts_vol.batch_upload(force=True) as b:
            if stored is None:
                b.put_file(io.BytesIO(json.dumps(settings, indent=1).encode()), f"{cfg['run']}/config.json")
            b.put_file(io.BytesIO(json.dumps({"attempt": attempt, "git": git}, indent=1).encode()), f"{cfg['run']}/attempt_{attempt}.json")
        cfgs.append(cfg)
        jobs += [(cfg, s) for s in range(1 if test else cfg["shards"])]
    res = list(extract_shard.starmap(jobs, return_exceptions=True))
    out = {"git": git, "app": APP, "configs": cfgs, "shards": [r if isinstance(r, dict) else {"error": repr(r)[:500]} for r in res]}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"extract_pooled_{'test_' if test else ''}a{attempt}_{stamp}.json"
    path.open("x").write(json.dumps(out, indent=1))
    print(json.dumps(out["shards"], indent=1)[:4000])
    print("->", path)
    if not test:
        for cfg in cfgs:
            mine = [s for s in out["shards"] if s.get("run") == cfg["run"]]
            if len(mine) == cfg["shards"] and not any("error" in s for s in mine):
                with acts_vol.batch_upload(force=True) as b:
                    b.put_file(io.BytesIO(json.dumps({"epochs": sum(s["epochs"] for s in mine), "attempt": attempt}).encode()), f"{cfg['run']}/complete.json")
        failed = [s for s in out["shards"] if "error" in s]
        if failed:
            raise SystemExit(f"{len(failed)} shards failed; rerun with --attempt {attempt + 1} (done epochs are kept)")
