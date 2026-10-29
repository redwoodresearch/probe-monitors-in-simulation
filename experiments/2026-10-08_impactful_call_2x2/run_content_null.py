"""Label-shuffle null for the content-call slice of the impactful-command 2x2 (#119), from saved fits (Modal CPU).

For each reader's attack-only all-layer Q2 run, the saved fits at all 64 layers: the real labels and the
within-transcript shuffle draws (Qwen3-32B: 20, Qwen3.5-27B: 5). Per draw, as Q2's summary does: per outer fold, the
earliest layer within the tolerance of the best validation AUROC; that fit's out-of-fold scores, scored against the
draw's own (shuffled) labels. Here only the attack calls whose call_end is the command's content (`call_end_content`, as
in plot_2x2.py) are scored. Per training world and test world: AUROC, and catch at 1% false alarms calibrated per fold
on the test world's own read-only calls. The real-label row should reproduce plot_2x2.py's numbers (a check).

  uv run modal run experiments/2026-10-08_impactful_call_2x2/run_content_null.py
"""
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
READERS = {"Qwen3.5-27B": ("probe-shift_aswin-surya_sub-qwen35_q2feats_kill-2026-11-06", "2026-10-06_q2_qwen35_matched_eb014307fd1aab31_as_labeled"),
           "Qwen3-32B": ("probe-shift_aswin-surya_sub-q2_probe-features_kill-2026-11-06", "2026-10-06_q2_features_eb014307fd1aab31_as_labeled")}
ARMS = {"real-attack-e1": "real", "sim-attack": "sim"}
TOL, FPR = 0.005, 0.01
app = modal.App("probe-shift_aswin-surya_sub-q2-content-null_kill-2026-11-08")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy", "scikit-learn", "joblib")
vols = {f"/v{i}": modal.Volume.from_name(v, version=2) for i, (v, _) in enumerate(READERS.values())}


@app.function(image=image, volumes=vols, cpu=8, memory=16384, timeout=3600, retries=0)
def null_for(i: int) -> dict:
    import numpy as np
    from joblib import Parallel, delayed
    from sklearn.metrics import roc_auc_score

    reader, (_, run) = list(READERS.items())[i]
    d = Path(f"/v{i}") / run
    import hashlib

    raw = (d / "manifest.json").read_bytes()
    # The fits score only the test domains (the two attack arms), in manifest order: keep those rows, in order.
    manifest = [m for m in json.loads(raw) if m["arm"] in ARMS]
    logs = [d / "alllayers" / "fits_index.jsonl"] + sorted((d / "alllayers" / "fits_index").glob("*.jsonl"))
    fits = [json.loads(l) for log in logs if log.exists() for l in log.read_text().splitlines()]
    inputs = {"manifest sha256": hashlib.sha256(raw).hexdigest(),
              **{f"{log.relative_to(d)} sha256": hashlib.sha256(log.read_bytes()).hexdigest() for log in logs if log.exists()}}
    by = {(f["train_set"], f["layer"], f["seed"]): f for f in fits if f["position"] == "call_end" and f.get("view", "") == ""}
    layers = sorted({k[1] for k in by})
    seeds = [None] + sorted({k[2] for k in by if k[2] is not None})
    arm = np.array([m["arm"] for m in manifest])
    content = np.array([bool(m.get("call_end_content")) for m in manifest])
    fold = np.array([m["fold"] for m in manifest])
    test = np.isin(arm, list(ARMS)) & content
    if "call_end_content" not in manifest[0]:
        raise ValueError(f"{reader}: the manifest has no call_end_content flag")

    def draw(seed):
        out = {}
        for t in ARMS:  # noqa: PLC0206
            score, ys = np.zeros(len(manifest)), None
            for f in sorted(set(fold)):
                val = {l: next(fo["val_auroc"] for fo in by[(t, l, seed)]["folds"] if fo["fold"] == f) for l in layers}
                pick = min(l for l in layers if val[l] >= max(val.values()) - TOL)
                a = np.load(d / "alllayers" / "fits" / by[(t, pick, seed)]["file"])
                if len(a["score"]) != len(manifest):
                    raise ValueError(f"{reader} {t} seed {seed}: {len(a['score'])} scores for {len(manifest)} rows")
                score[fold == f] = a["score"][fold == f]
                if seed is not None:
                    ys = np.array(a["y"]) if ys is None else ys  # the draw's shuffled labels (one per draw)
            y = np.array([m["y"] for m in manifest]) if seed is None else ys
            for te_arm, te in ARMS.items():
                m = test & (arm == te_arm)
                hits = pos = 0
                for f in sorted(set(fold[m])):
                    mf = m & (fold == f)
                    neg = np.sort(score[mf & (y == 0)])[::-1]
                    thr = neg[min(int(np.floor(FPR * len(neg) + 1e-9)), len(neg) - 1)]
                    hits += int((score[mf & (y == 1)] > thr).sum())
                    pos += int((mf & (y == 1)).sum())
                out[f"{ARMS[t]}->{te}"] = {"auroc": float(roc_auc_score(y[m], score[m])),
                                           "pauc_0_5pct": float(roc_auc_score(y[m], score[m], max_fpr=0.05)),  # standardized (McClish), amendment 40
                                           "catch, calibrated in test world": hits / pos}
        return out

    res = Parallel(n_jobs=8)(delayed(draw)(s) for s in seeds)
    return {"reader": reader, "run": run, "layers": len(layers), "null_seeds": len(seeds) - 1, "seed_ids": seeds[1:], "inputs": inputs,
            "real": res[0], "null": res[1:], "content_calls": {te: int((test & (arm == a)).sum()) for a, te in ARMS.items()}}


@app.local_entrypoint()
def main():
    import numpy as np

    out = {}
    for r in null_for.map(range(len(READERS))):
        summary = {}
        for cell, real in r["real"].items():
            summary[cell] = {}
            for metric, v in real.items():
                nulls = np.array([n[cell][metric] for n in r["null"]])
                summary[cell][metric] = {"real labels": round(v, 4), "null mean": round(float(nulls.mean()), 4), "null max": round(float(nulls.max()), 4),
                                         "p (share of null draws >= real, with +1)": round(float((1 + (nulls >= v).sum()) / (1 + len(nulls))), 4)}
        out[r["reader"]] = {"run": r["run"], "null_seeds": r["null_seeds"], "seed_ids": r["seed_ids"], "layers": r["layers"],
                            "inputs": r["inputs"], "content_calls": r["content_calls"], "cells": summary,
                            "per_draw": {"real labels": r["real"], "null": dict(zip(map(str, r["seed_ids"]), r["null"]))},
                            "settings": {"tolerance": TOL, "fpr": FPR, "readout": "call_end", "slice": "call_end_content"}}
        print(r["reader"], r["null_seeds"], {c: {m: (s["real labels"], s["null mean"], s["null max"]) for m, s in v.items()} for c, v in summary.items()})
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"content_null_{stamp}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(path)
