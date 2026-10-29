"""Mean-over-edit readout on LinuxArena (Modal H100s): the SafeCoder-trained vulnerability probe read the way it
works best, as the mean activation over each editor edit's new code, instead of at call_end.

Selection (seed 0, from the call table): `cells` paired cells whose e1, e2 and sim runs each have at least one
text_editor edit (str_replace, create, insert), and one honest run with an edit per chosen cell's main task. Messages
come from the segment-end extraction's staged inputs (Volume, read-only), cut to 32,768 tokens as there.
Per transcript, one forward pass stopped after the last needed layer; per edit (probe_shift.vuln.edit_code_spans:
tokens entirely inside the new-code string, within the cut) the mean residual stream at each layer, saved fp16 to the
Q3 Volume (edit_means/<run>/), and scored with the SafeCoder mean-over-code probes (probes/vuln_sc_v1_mean).
Writes data/scores_vulnerability_safecoder_mean_edit_mean_<stamp>.parquet (kind "edit_mean", one row per edit, the
format analyze_shift.py reads with --kind edit_mean --subset) and results/edit_mean_run_<stamp>.json.

  PYTHONPATH=src uv run --no-project --python 3.13 --with 'modal>=1.2' --with pyyaml --with numpy --with pandas \
      --with pyarrow modal run experiments/2026-10-06_q3_offpolicy_probes/run_edit_mean_readout.py \
      --calls data/call_table_<stamp>.jsonl.gz [--cells 300] [--limit 2]
"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

from probe_shift.models import QWEN3_32B

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q3_edit-mean_kill-2026-11-06"
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
OUTPUT = "probe-shift_aswin-surya_sub-q3_outputs_kill-2026-11-06"
GPU = "H100!"
PIP = ["torch==2.8.0", "transformers==4.56.2", "accelerate==1.10.1", "numpy"]
CFG = {
    "run": "edit_means_v1", "model": QWEN3_32B["name"], "dtype": "bfloat16", "attn_implementation": "sdpa",
    "revision": QWEN3_32B["revision"],  # pinned; the only snapshot on the HF cache Volume, so edit_means_v1 used it too
    "segment_end_run": "2026-10-06_la_segment-end_qwen3-32b_v1", "max_tokens": 32768,
    "layers": [16, 24, 32, 40, 48], "probe_dir": "probes/vuln_sc_v1_mean", "seed": 0, "containers": 8,
}

app = modal.App(APP)
hf_cache = modal.Volume.from_name(HF_CACHE).read_only()
acts_vol = modal.Volume.from_name(ACTS).read_only()
out_vol = modal.Volume.from_name(OUTPUT)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PIP)
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}).add_local_python_source("probe_shift"))


@app.function(image=image, volumes={"/acts": acts_vol}, cpu=4, memory=32768, timeout=1800)
def select(cfg: dict, keys: list[list[str]]) -> list[dict]:
    """The staged items for these (arm, key) pairs, in the given order."""
    want = {tuple(k) for k in keys}
    found = {}
    for p in sorted((Path("/acts") / cfg["segment_end_run"] / "inputs").glob("shard_*.jsonl")):
        for line in p.read_text().splitlines():
            it = json.loads(line)
            if (it["arm"], it["key"]) in want:
                found[(it["arm"], it["key"])] = it
    missing = want - set(found)
    if missing:
        raise ValueError(f"{len(missing)} selected runs are not in the staged inputs, e.g. {sorted(missing)[:2]}")
    return [found[tuple(k)] for k in keys]


class Stop(Exception):
    pass


@app.function(image=image, gpu=GPU, volumes={"/hf": hf_cache, "/out": out_vol}, timeout=4 * 3600,
              retries=modal.Retries(max_retries=2, initial_delay=10.0))
def read(cfg: dict, shard: int, items: list[dict], probes: dict, shard_fingerprint: str) -> list[dict]:
    import time

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from probe_shift.offpolicy import probe_scores
    from probe_shift.vuln import edit_code_spans

    out = Path("/out") / "edit_means" / cfg["run"] / f"shard_{shard}"
    if (out / "rows.json").exists():  # a retried shard returns what it wrote, but only for the same inputs and probes
        if (out / "fingerprint").read_text() != shard_fingerprint:
            raise FileExistsError(f"{out} holds results for other inputs, settings or probes; change cfg['run']")
        return json.loads((out / "rows.json").read_text())
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    base = AutoModelForCausalLM.from_pretrained(cfg["model"], revision=cfg["revision"], dtype=getattr(torch, cfg["dtype"]),
                                                device_map="cuda", attn_implementation=cfg["attn_implementation"]).model.eval()
    L, last = cfg["layers"], max(cfg["layers"])
    rows, means = [], {l: [] for l in L}
    for it in items:
        r = edit_code_spans(tok, it["messages"], cfg["max_tokens"])
        if not r["edits"]:
            rows.append({k: it[k] for k in ("arm", "key")} | {"kind": "no_edits"})
            continue
        got = {}

        def hook_for(l):
            def hook(m, a, o):
                h = (o[0] if isinstance(o, tuple) else o)[0]
                got[l] = torch.stack([h[e["start"]:e["end"]].float().mean(0) for e in r["edits"]])
                if l == last:
                    raise Stop  # later layers are not needed
            return hook

        hs = [base.layers[l].register_forward_hook(hook_for(l)) for l in L]
        t = time.time()
        try:
            with torch.inference_mode():
                base(input_ids=torch.tensor([r["ids"]], device="cuda"), use_cache=False)
        except Stop:
            pass
        finally:
            for h in hs:
                h.remove()
        torch.cuda.synchronize()  # the forward and pooling run asynchronously
        seconds = time.time() - t
        acts = {l: got[l].cpu().numpy() for l in L}
        scores = {f"L{l}": probe_scores(acts[l], probes[l]) for l in L}
        for j, e in enumerate(r["edits"]):
            rows.append({k: it[k] for k in ("arm", "key", "cell", "main_task", "side_task", "environment")}
                        | {"kind": "edit_mean", "step": e["step"], "action": e["action"], "tool": "text_editor",
                           "token": e["start"], "token_text": None, "n_code_tokens": e["end"] - e["start"],
                           "truncated": r["n_tokens"] > cfg["max_tokens"], "seconds": round(seconds, 2)}
                        | {c: float(v[j]) for c, v in scores.items()})
        for l in L:
            means[l].append(acts[l].astype(np.float16))
    bad = sum(int((~np.isfinite(m.astype(np.float32))).sum()) for v in means.values() for m in v)
    bad += sum(not np.isfinite(r[f"L{l}"]) for r in rows if r["kind"] == "edit_mean" for l in L)  # no_edits rows have no scores
    if bad:  # never cache a bad shard: a retry would return it as done
        raise ValueError(f"shard {shard}: {bad} non-finite means or scores; nothing saved")
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "means.npz", **{f"L{l}": np.concatenate(v) if v else np.zeros((0, 5120), np.float16) for l, v in means.items()})
    (out / "fingerprint").write_text(shard_fingerprint)
    (out / "rows.json").write_text(json.dumps(rows))  # written last: marks the shard done
    out_vol.commit()
    return rows


@app.local_entrypoint()
def main(calls: str, cells: int = 300, limit: int = 0):
    """`--limit N`: a test on N cells (and their honest runs) in one container, under run <run>_test."""
    import random

    import numpy as np
    import pandas as pd
    import yaml

    from probe_shift.offpolicy import load_probe, provenance

    cfg = CFG | ({"run": CFG["run"] + "_test", "containers": 1} if limit else {})
    q3 = yaml.safe_load((HERE / "config" / "q3.yaml").read_text())
    ex = json.loads((HERE.parent.parent / "splits" / q3["exclusions_file"]).read_text())
    ct = pd.read_json(HERE / calls, lines=True)
    has_edit = ct[(ct.tool == "text_editor") & ct.editor_command.isin(["str_replace", "create", "insert"])].groupby(["arm", "key"]).size()
    valid = {(r["arm"], r["key"]): r for r in ex["samples"] if r["reason"] is None}
    by_cell = {}
    for (arm, key), r in valid.items():
        by_cell.setdefault(r["cell"], {})[arm] = key
    attack = ("real-attack-e1", "real-attack-e2", "sim-attack")
    honest_tasks = {r["main_task"] for (a, k), r in valid.items() if a == "real-honest" and (a, k) in has_edit.index}
    ok = sorted(c for c in ex["paired_attack_cells"]["cells"]
                if all(a in by_cell.get(c, {}) and (a, by_cell[c][a]) in has_edit.index for a in attack)
                and valid[("real-attack-e1", by_cell[c]["real-attack-e1"])]["main_task"] in honest_tasks)  # honest counterpart
    rng = random.Random(cfg["seed"])
    chosen = rng.sample(ok, limit or min(cells, len(ok)))
    tasks = sorted({valid[("real-attack-e1", by_cell[c]["real-attack-e1"])]["main_task"] for c in chosen})
    honest = sorted(k for (a, k), r in valid.items() if a == "real-honest" and r["main_task"] in tasks and (a, k) in has_edit.index)
    rng.shuffle(honest)
    one_per_task, seen = [], set()
    for k in honest:
        t = valid[("real-honest", k)]["main_task"]
        if t not in seen:
            one_per_task.append(k)
            seen.add(t)
    keys = [[a, by_cell[c][a]] for c in chosen for a in attack] + [["real-honest", k] for k in one_per_task]
    print(f"{len(ok)} paired cells have an edit in every attack arm; reading {len(chosen)} cells and {len(one_per_task)} "
          f"honest runs ({len(keys)} transcripts)", flush=True)

    import hashlib

    pvol = modal.Volume.from_name(OUTPUT)
    raw = {l: b"".join(pvol.read_file(f"{cfg['probe_dir']}/probe_L{l}.npz")) for l in cfg["layers"]}
    probes = {l: load_probe(b) for l, b in raw.items()}
    items = select.remote(cfg, keys)
    n = cfg["containers"]
    order = sorted(range(len(items)), key=lambda i: -items[i]["tokens"])  # longest first, dealt round-robin
    shards = [[items[i] for i in order[s::n]] for s in range(n)]
    probe_hash = hashlib.sha256(b"".join(raw[l] for l in cfg["layers"])).hexdigest()
    # Full items (messages included), settings and probes. The revision is left out: edit_means_v1 was read from the
    # pinned snapshot (the only one cached) before the pin, and its cached shards must stay reusable.
    cfg_fp = {k: v for k, v in cfg.items() if k != "revision"}
    fps = [hashlib.sha256(json.dumps([cfg_fp, probe_hash, sh]).encode()).hexdigest() for sh in shards]
    parts = list(read.map([cfg] * n, range(n), shards, [probes] * n, fps))
    rows = [r for p in parts for r in p]
    df = pd.DataFrame([r for r in rows if r["kind"] == "edit_mean"])
    no_edits = [r for r in rows if r["kind"] == "no_edits"]
    forwarded = df.drop_duplicates(["arm", "key"])  # every transcript read, before the cohort filter below
    # A run can lose all its edits (empty new code, or every edit past the 32k cut): drop its whole cell, so every
    # remaining cell is paired, and keep honest runs only on the remaining cells' main tasks.
    # Likewise a task whose honest run lost all its edits loses its attack cells, so attack and honest runs always
    # cover the same main tasks.
    att = df[df.arm.isin(attack)]
    n_arms = att.groupby("cell").arm.nunique()
    honest_left = set(df.loc[df.arm == "real-honest", "main_task"])
    complete = {c for c in n_arms[n_arms == 3].index if att.loc[att.cell == c, "main_task"].iloc[0] in honest_left}
    dropped_cells = sorted(set(chosen) - complete)
    keep_tasks = set(att.loc[att.cell.isin(complete), "main_task"])
    df = df[(df.arm.isin(attack) & df.cell.isin(complete)) | ((df.arm == "real-honest") & df.main_task.isin(keep_tasks))]
    score_cols = [f"L{l}" for l in cfg["layers"]]
    if not np.isfinite(df[score_cols].to_numpy()).all():
        raise ValueError("non-finite scores")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S") + ("_test" if limit else "")
    path = HERE / "data" / f"scores_vulnerability_safecoder_mean_edit_mean_{stamp}.parquet"
    path.parent.mkdir(exist_ok=True)
    df.to_parquet(path, index=False)
    with pvol.batch_upload() as batch:
        batch.put_file(str(path), f"scores/{path.name}")
    exclusions = HERE.parent.parent / "splits" / q3["exclusions_file"]
    summary = {"config": cfg | {"modal": {"app": APP, "gpu": GPU, "pip": PIP}},
               "provenance": provenance(__file__, HERE.parent.parent / "src/probe_shift/vuln.py", HERE.parent.parent / "src/probe_shift/offpolicy.py"),
               "inputs": {"calls": calls, "calls_sha256": hashlib.sha256((HERE / calls).read_bytes()).hexdigest(),
                          "exclusions": q3["exclusions_file"], "exclusions_sha256": hashlib.sha256(exclusions.read_bytes()).hexdigest(),
                          "probe_sha256": probe_hash},
               "selected": keys,  # the resolved (arm, key) cohort, in read order of selection
               "cells": len(chosen), "honest_runs": len(one_per_task), "transcripts": len(keys),
               "transcripts_forwarded_with_edits": len(forwarded), "transcripts_without_scoreable_edits": len(no_edits),
               "forward_seconds": {"total": round(float(forwarded.seconds.sum())), "mean": round(float(forwarded.seconds.mean()), 2)},
               "cells_dropped_for_a_run_without_scoreable_edits": dropped_cells, "cells_kept": len(complete),
               "honest_runs_kept": int(df[df.arm == "real-honest"].key.nunique()),
               "runs_kept": int(df.groupby("arm").key.nunique().sum()), "edits_per_arm_kept": df.groupby("arm").size().to_dict(),
               "file": str(path.relative_to(HERE)), "volume": f"{OUTPUT}/edit_means/{cfg['run']}/"}
    (HERE / "results" / f"edit_mean_run_{stamp}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("cells", "cells_kept", "honest_runs", "honest_runs_kept", "transcripts_forwarded_with_edits",
                                              "transcripts_without_scoreable_edits", "edits_per_arm_kept", "forward_seconds")}))
    print(f"-> {path}")
