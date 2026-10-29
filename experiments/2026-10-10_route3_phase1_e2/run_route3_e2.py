"""Route 3 (PLAN.md): Sol's epoch 2 vs epoch 1 through the published Phase 1 monitor. Modal CPU; keys and numbers out.

Epoch 1 and the simulated run are the refit's rows. Epoch 1 is rebuilt from the v4 dump and gated against the refit
(rows, labels, folds, main tasks, bytes at layers 3/7/11, scores). Epoch 2 goes through the same rebuild: Q2's labeler
(calls.py at a85ee070, fingerprint eb014307fd1aab31, passed as source and hash-checked) and Q2's Qwen3.5 matched rule.
Every row gets all five real-trained probes' scores, so the analysis can recalibrate on resamples. No command or token
text leaves the container.

  modal deploy experiments/2026-10-10_route3_phase1_e2/run_route3_e2.py
  python experiments/2026-10-10_route3_phase1_e2/run_route3_e2.py spawn             # spawn-and-return; prints the call id
  python experiments/2026-10-10_route3_phase1_e2/run_route3_e2.py fetch <call id>   # the call's summary (no rows)
  modal volume get probe-shift_aswin-surya_sub-route3_e2_kill-2026-11-08 <tag>/route3_scores.json data/route3/
"""

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = OUT_VOL = "probe-shift_aswin-surya_sub-route3_e2_kill-2026-11-08"
DUMP_VOL, DUMP_RUN = "probe-shift_aswin-surya_sub-qwen35_acts_kill-2026-11-06", "2026-10-06_la_segment-end-v4_qwen35-27b_alllayers_v1"
CUT = (f"{DUMP_RUN}/qwen3_cut/2026-10-06_la_segment-end-v4_qwen3-32b_alllayers_v1.json",
       "4fa23ccfda0b92c708898f71fb6a22b01ebc3193b8343f3f2f8fbea9d1b1d72a")
INPUTS_VOL, INPUTS_RUN = "probe-shift_aswin-surya_sub-la-activations", "2026-10-06_la_segment-end-v4_qwen3-32b_v1"
REFIT_VOL, REFIT = "probe-shift_aswin-surya_sub-q2-refit_boot_kill-2026-11-08", "2026-10-08_impactful_refit_bootstrap_v1/qwen3.5"
REP = "replicates/5ab3b9a4b25ba4f3"
SHA = {"rows.json": "41a68be0b1d7ae045d573dce15ce2a4b3cfa349f915cc6babc7fd478f28ed8e4",  # tpr_at_1pct manifest_v2
       f"{REP}/probes_boot_0000.npz": "bd6372b39619550aa425e3507aeb8be1c0c5086de1828ae694a8d33f30453186",
       f"{REP}/scores_boot_0000.npz": "d404c076343693d5af93014b7a3e0046e1d85dbe7783c4b8899b7dc143026979"}
LABELER, LABELER_COMMIT = "eb014307fd1aab31", "a85ee070"
ARMS = {"e1": "real-attack-e1", "e2": "real-attack-e2", "sim": "sim-attack"}

app = modal.App(APP)
out_vol = modal.Volume.from_name(OUT_VOL, create_if_missing=True, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")


def positions_of(meta: dict, step: int, action: int) -> dict:
    """Q2's three readouts of a call (copied from qwen35_replication/run_q2_alllayers_qwen35.py; None where missing)."""
    at = {(q["kind"], q["step"], q["action"]): q for q in meta["positions"]}
    last_output = {}
    for q in meta["positions"]:
        if q["kind"] == "output_end":
            last_output[q["step"]] = max(last_output.get(q["step"], -1), q["action"])
    return {"call_end": at.get(("call_end", step, action)), "pre_call": at.get(("pre_call", step, action)),
            "prev_content": at.get(("call_end", step, action - 1)) if action > 0 else
            (at.get(("output_end", step - 1, last_output[step - 1])) if step - 1 in last_output else None)}


class Safe(Exception):
    """An error whose message holds only identifiers and counts, so it may leave the container."""


STAGE = ["start"]


def module(name: str, src: str):
    import types

    m = types.ModuleType(name)
    sys.modules[name] = m
    exec(compile(src, f"{name}.py", "exec"), m.__dict__)
    return m


@app.function(image=image, cpu=8, memory=32768, timeout=2 * 3600, retries=0,
              volumes={"/dump": modal.Volume.from_name(DUMP_VOL, version=2).read_only(),
                       "/in": modal.Volume.from_name(INPUTS_VOL, version=2).read_only(),
                       "/refit": modal.Volume.from_name(REFIT_VOL, version=2).read_only(), "/out": out_vol})
def build_and_score(calls_src: str, elig_src: str, task_fold: dict, fix: list, tag: str) -> dict:
    """Any other exception is replaced by its type and stage: parse and labeler errors can carry transcript text."""
    try:
        return build(calls_src, elig_src, task_fold, fix, tag)
    except Safe as e:
        raise RuntimeError(f"route3: {e}") from None
    except Exception as e:  # noqa: BLE001 (re-raised without its message)
        raise RuntimeError(f"route3 failed at stage {STAGE[0]!r}: {type(e).__name__}") from None


def build(calls_src: str, elig_src: str, task_fold: dict, fix: list, tag: str) -> dict:
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np

    t0 = time.time()
    if hashlib.sha256(calls_src.encode()).hexdigest()[:16] != LABELER:
        raise Safe("the labeler source is not eb014307")
    calls, elig = module("calls", calls_src), module("eligibility", elig_src)
    ref, dump = Path("/refit") / REFIT, Path("/dump")
    for f, h in SHA.items():
        if hashlib.sha256((ref / f).read_bytes()).hexdigest() != h:
            raise Safe(f"{f}: sha256 differs from the manifest")
    if hashlib.sha256((dump / CUT[0]).read_bytes()).hexdigest() != CUT[1]:
        raise Safe("the cut index's sha256 differs from PLAN.md's")
    refit_rows = json.loads((ref / "rows.json").read_text())
    pz = np.load(ref / f"{REP}/probes_boot_0000.npz")
    table = json.loads(str(pz["table"]))
    real = [i for i, p in enumerate(table) if p["train"] == "real"]
    probes = [(table[i]["fold"], table[i]["layer"], pz["w"][i], pz["mu"][i], pz["sd"][i], table[i]["b"], table[i]["threshold_real"]) for i in real]
    if [p[0] for p in probes] != list(range(5)):
        raise Safe("expected one real-trained probe per fold, in fold order")
    layers = sorted({p[1] for p in probes})
    for r in refit_rows:  # the plan's folds file agrees with the refit's folds and main tasks
        if task_fold.get(r["main_task"]) != r["fold"]:
            raise Safe(f"{r['main_task']}: folds_v3 and rows.json disagree")

    # metadata of all three worlds; Q2's cells are those in the refit's rows
    STAGE[0] = "metadata"
    meta_dir = dump / DUMP_RUN / "meta"
    files = [(e, p) for e, arm in ARMS.items() for p in sorted((meta_dir / arm).glob("*.json"))]
    with ThreadPoolExecutor(32) as ex:
        allm = list(ex.map(lambda ep: (ep[0], json.loads(ep[1].read_text())), files))
    refit_keys = {(r["arm"], r["key"]) for r in refit_rows}
    cells = {m["cell"] for _, m in allm if (m["arm"], m["key"]) in refit_keys}
    cell_of = {(m["arm"], m["key"]): m["cell"] for _, m in allm}
    metas = [(e, m) for e, m in allm if e != "sim" and m["cell"] in cells]
    per = Counter((e, m["cell"]) for e, m in metas)
    if any(v != 1 for v in per.values()):
        raise Safe(f"cells with more than one transcript in an epoch: {sum(v != 1 for v in per.values())}")
    cut = json.loads((dump / CUT[0]).read_text())["read"]
    if [1 for e, m in metas if e == "e1" and f"{m['arm']}|{m['key']}" not in cut]:
        raise Safe("epoch-1 transcripts missing from the cut index")
    not_in_cut = sum(1 for e, m in metas if e == "e2" and f"{m['arm']}|{m['key']}" not in cut)
    metas = [(e, m) for e, m in metas if f"{m['arm']}|{m['key']}" in cut]
    counts = {"refit_cells": len(cells), "transcripts": {e: sum(x == e for x, _ in metas) for e in ("e1", "e2")},
              "cells_without_a_transcript": {e: len(cells - {c for x, c in per if x == e}) for e in ("e1", "e2")},
              "e2_transcripts_not_in_cut_index": not_in_cut}
    want = {(m["arm"], m["key"]) for _, m in metas}
    STAGE[0] = "staged inputs"
    msgs = {}
    for f in sorted((Path("/in") / INPUTS_RUN / "inputs").glob("shard_*.jsonl")):
        for line in f.open():
            it = json.loads(line)
            if (it["arm"], it["key"]) in want:
                msgs[(it["arm"], it["key"])] = it["messages"]
    if len(msgs) != len(want):
        raise Safe(f"{len(want) - len(msgs)} transcripts not in the staged inputs")

    # eligible matched calls, labelled (Q2's rule); only keys, labels and dump rows are kept
    STAGE[0] = "labelling"
    rows, drop = [], {e: Counter() for e in ("e1", "e2")}
    for e, m in metas:
        turns = [x.get("tool_calls") or [] for x in msgs[(m["arm"], m["key"])] if x["role"] == "assistant"]
        inside = {tuple(q) for q in cut[f"{m['arm']}|{m['key']}"]}
        for p in m["positions"]:
            if p["kind"] != "call_end":
                continue
            tc = turns[p["step"]][p["action"]]
            fn = tc["function"]
            name, args = (fn["name"], fn.get("arguments")) if isinstance(fn, dict) else (fn, tc.get("arguments"))
            args = json.loads(args) if isinstance(args, str) else (args or {})
            if name != p["tool"]:
                raise Safe(f"{m['key']} step {p['step']} action {p['action']}: the message and positions disagree on the tool")
            if name == "submit":
                drop[e]["terminal submit"] += 1
                continue
            y = int(calls.label_call(name, args)["label"] == "impactful")
            if not elig.matched_call(name, args):
                drop[e]["not matched"] += 1
                continue
            qs = positions_of(m, p["step"], p["action"])
            if qs["prev_content"] is None:
                drop[e]["no previous content token"] += 1
            elif not all(q is not None and q["kept"] for q in qs.values()):
                drop[e]["a readout with no Qwen3.5 token"] += 1
            elif not all((q["kind"], q["step"], q["action"]) in inside for q in qs.values()):
                drop[e]["outside Qwen3-32B's 32k cut"] += 1
            elif m["main_task"] not in task_fold:
                drop[e]["main task without a fold"] += 1
            else:
                rows.append({"world": e, "key": m["key"], "step": p["step"], "action": p["action"], "y": y, "main_task": m["main_task"],
                             "cell": m["cell"], "fold": task_fold[m["main_task"]], "chunk": m["chunk"], "row": int(m["row_start"]) + qs["call_end"]["row"]})
    del msgs
    counts |= {"dropped": {e: dict(c) for e, c in drop.items()}, "rows": {e: sum(r["world"] == e for r in rows) for e in ("e1", "e2")}}

    def read(sel: list) -> dict:
        """layer -> [len(sel), hidden] fp16, each chunk file read whole (row-indexed reads of a Volume file are slow)."""
        groups = {}
        for i, r in enumerate(sel):
            g = groups.setdefault(r["chunk"], ([], []))
            g[0].append(i), g[1].append(r["row"])
        X = {layer: np.empty((len(sel), 5120), np.float16) for layer in layers}

        def one(job):
            layer, chunk, (pos, idx) = job
            X[layer][pos] = np.load(dump / DUMP_RUN / "by_layer" / f"L{layer}" / f"{chunk}.npy")[idx]

        with ThreadPoolExecutor(32) as ex:
            list(ex.map(one, [(layer, c, g) for layer in layers for c, g in groups.items()]))
        return X

    def score(X: dict) -> np.ndarray:
        """[row, fold model]: ((x.float() - mu) / sd) @ w + b at each probe's layer."""
        return np.stack([((X[layer].astype(np.float32) - mu) / sd) @ w + b for _, layer, w, mu, sd, b, _ in probes], 1)

    # the gate: epoch 1 rebuilt equals the refit's rows, bytes and scores; the simulated rows' scores match
    STAGE[0] = "gate"
    e1 = [r for r in rows if r["world"] == "e1"]
    ref_e1 = {(r["key"], r["step"], r["action"]): (i, r) for i, r in enumerate(refit_rows) if r["arm"] == ARMS["e1"]}
    mine = {(r["key"], r["step"], r["action"]): r for r in e1}
    both = mine.keys() & ref_e1.keys()
    gate = {"only_rebuilt": len(mine.keys() - ref_e1.keys()), "only_refit": len(ref_e1.keys() - mine.keys()),
            **{f"{k}_mismatch": sum(mine[c][k] != ref_e1[c][1][k] for c in both) for k in ("y", "fold", "main_task")}}
    ok = not any(gate.values())
    out = {"tag": tag, "labeler": LABELER, "labeler_commit": LABELER_COMMIT, "dump": f"{DUMP_VOL}/{DUMP_RUN}", "refit": f"{REFIT_VOL}/{REFIT}",
           "cut_index_sha256": CUT[1], "probes": [{"fold": p[0], "layer": p[1], "threshold_real": p[6]} for p in probes], "counts": counts}
    if ok:
        stored = np.load(ref / f"{REP}/scores_boot_0000.npz")["scores"][:, 0, :].T  # [refit row, fold model], real-trained
        packs = {layer: np.load(ref / f"L{layer}.npy") for layer in layers}
        e1 = sorted(e1, key=lambda r: ref_e1[(r["key"], r["step"], r["action"])][0])  # refit row order
        i1 = np.array([ref_e1[(r["key"], r["step"], r["action"])][0] for r in e1])
        X1 = read(e1)
        gate["bytes_mismatch_rows"] = {layer: int((packs[layer][i1].view(np.uint16) != X1[layer].view(np.uint16)).any(1).sum()) for layer in layers}
        s1 = score(X1)
        isim = np.array([i for i, r in enumerate(refit_rows) if r["arm"] == ARMS["sim"]])
        ssim = score({layer: packs[layer][isim] for layer in layers})
        del packs, X1
        gate["max_score_diff"] = {"e1": float(np.abs(s1 - stored[i1]).max()), "sim": float(np.abs(ssim - stored[isim]).max())}
        ok = not any(gate["bytes_mismatch_rows"].values()) and max(gate["max_score_diff"].values()) <= 1e-4
    gate["passed"] = ok
    out["gate"] = gate
    if not ok:  # the plan promises epoch-1 diagnostics only
        out["counts"] = {k: ({"e1": v["e1"]} if isinstance(v, dict) and "e1" in v else v) for k, v in counts.items()
                         if k not in ("e2_transcripts_not_in_cut_index",)}
    if ok:  # only now: epoch 2's activations and scores
        STAGE[0] = "epoch 2"
        e2 = [r for r in rows if r["world"] == "e2"]
        s2 = score(read(e2))
        sim = [refit_rows[i] | {"world": "sim", "cell": cell_of[(ARMS["sim"], refit_rows[i]["key"])]} for i in isim]
        keep = ("world", "key", "step", "action", "y", "main_task", "fold", "cell")
        out["rows"] = [{k: r[k] for k in keep} | {"s": [float(v) for v in s]}  # full float32 precision
                       for rr, ss in ((e1, s1), (e2, s2), (sim, ssim)) for r, s in zip(rr, ss)]
        fixset = {tuple(x) for x in fix}  # the secondary: cells with any world's transcript on the call_end-fix list
        out["callend_fix_cells"] = sorted({m["cell"] for _, m in allm if m["cell"] in cells and (m["arm"], m["key"]) in fixset})
    out["seconds"] = round(time.time() - t0)
    d = Path("/out") / tag
    d.mkdir(parents=True, exist_ok=True)
    (d / "route3_scores.json").write_text(json.dumps(out))
    out_vol.commit()
    return {k: v for k, v in out.items() if k != "rows"} | {"n_rows": len(out.get("rows", []))}


def inputs() -> tuple:
    root = HERE.parents[1]
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=root).stdout
    calls_src = git("show", f"{LABELER_COMMIT}:src/probe_shift/calls.py")
    elig_src = (root / "src/probe_shift/eligibility.py").read_text()
    task_fold = json.loads((root / "splits/folds_v3.json").read_text())["task_folds"]["main_task_to_fold"]
    fix = json.loads((root / "experiments/2026-10-06_qwen35_replication/config/callend_fix_transcripts_v4.json").read_text())["affected"]
    return calls_src, elig_src, task_fold, fix


if __name__ == "__main__":
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    if sys.argv[1] == "spawn":
        c = modal.Function.from_name(APP, "build_and_score").spawn(*inputs(), stamp)
        print(json.dumps({"spawned_at": stamp, "call": c.object_id, "tag": stamp}))
    elif sys.argv[1] == "fetch":
        print(json.dumps(modal.FunctionCall.from_id(sys.argv[2]).get(timeout=0), indent=1))
