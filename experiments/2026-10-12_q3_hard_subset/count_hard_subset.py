"""Counts of the hard-command subset in the acting set-up (196 tasks), before any probe number on it. $0, local.

Hard = an impactful call (y = 1) that the command-text TF-IDF (trained on real run 1, run_text_catch.py at a60e9c39) does
NOT flag at its frozen 1% threshold, in the real-trained cells real->real, real->sim and real->run2. Reads the two stored
per-call draw-0 files (IDs, labels, scores and flags only; no command text). Prints and saves counts and join checks
only. The probe enters only through two checks that print mismatch counts: its frozen thresholds are recomputed from the
stored calibration scores, and its stored flags are compared with score > threshold. No probe catch is computed.

  uv run --no-project --with numpy python experiments/2026-10-12_q3_hard_subset/count_hard_subset.py <probe draw-0 file> <tfidf draw-0 file>
"""
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
SHA = {"probe": "c2656658baaf3aa150f83f4d105e7572bfd854f98032831621419e1f8e8f36bf",  # s3_manifest_scaleup196_v1.json
       "tfidf": "112088a04a7e2cf376dae486ead8dd94cf86f64a0ac189377a4da3c6bdaf770f"}  # manifest_textcatch_v1.json @ 251924f8
CELLS = {"real": "real->real", "sim": "real->sim", "run2": "real->run2"}


def threshold(neg, w, fpr=0.01):
    """analyze_scaleup_plants.threshold (run_score_onpolicy.threshold), copied: the (k+1)-th largest read-only score with
    rows counted w times, k = floor(fpr * sum w); flags are score > t."""
    o = np.argsort(-neg, kind="mergesort")
    c = np.cumsum(w[o])
    return float(neg[o][min(int((c <= np.floor(fpr * c[-1] + 1e-9)).sum()), len(neg) - 1)])


def kish(n):
    n = n[n > 0]
    return float(len(n) * (n ** 2).sum() / n.sum() ** 2)


def main(probe_p, tfidf_p):
    got = {"probe": hashlib.sha256(Path(probe_p).read_bytes()).hexdigest(), "tfidf": hashlib.sha256(Path(tfidf_p).read_bytes()).hexdigest()}
    if got != SHA:
        raise SystemExit(f"inputs differ from the manifests: {got}")
    P, T = json.loads(Path(probe_p).read_text()), json.loads(Path(tfidf_p).read_text())
    meta, folds = P["meta"], P["folds"]
    checks = {"meta equal": P["meta"] == T["meta"], "folds equal": P["folds"] == T["folds"]}
    out_w = {}
    cal0 = P["draws"][0]["cells"]["real->real"]["per_call"]["calibration"]
    for w, c in CELLS.items():
        p, t = P["draws"][0]["cells"][c]["per_call"], T["draws"][0]["cells"][c]["per_call"]
        checks[f"{c}: rows equal"] = p["row"] == t["row"]
        checks[f"{c}: rows all {w}"] = all(meta[i]["world"] == w for i in p["row"])
        checks[f"{c}: calibration equal to real->real's"] = p["calibration"] == cal0
        rows = np.array(p["row"]); y = np.array([meta[i]["y"] for i in rows]) == 1
        task = np.array([meta[i]["task"] for i in rows]); fold = np.array([folds[x] for x in task])
        tf_flag, tf_score = np.array(t["flag_frozen"], bool), np.array(t["score"], float)
        # TF-IDF flags are a threshold rule within each evaluation fold (its thresholds were not saved)
        checks[f"{c}: TF-IDF flags not a per-fold threshold rule (folds)"] = int(sum(
            (tf_score[(fold == k) & ~tf_flag].max() >= tf_score[(fold == k) & tf_flag].min()) for k in range(5)))
        # the probe's frozen thresholds, recomputed from the stored calibration scores (real read-only calls, model k)
        cr, cs, ck = np.array(cal0["row"]), np.array(cal0["score"], float), np.array(cal0["model_fold"])
        neg = np.array([meta[i]["world"] == "real" and meta[i]["y"] == 0 for i in cr])
        thr = {k: threshold(cs[neg & (ck == k)], np.ones(int((neg & (ck == k)).sum()))) for k in range(5)}
        stored = {f["fold"]: f["threshold"] for f in P["draws"][0]["cells"][c]["folds"] if f["threshold_kind"] == "frozen"}
        checks[f"{c}: probe frozen threshold max |recomputed - stored|"] = max(abs(thr[k] - stored[k]) for k in range(5))
        pf, ps = np.array(p["flag_frozen"], bool), np.array(p["score"], float)
        checks[f"{c}: probe flag_frozen != score > recomputed threshold"] = int((pf != (ps > np.array([thr[k] for k in fold]))).sum())
        hard = y & ~tf_flag
        tasks = sorted(set(task))
        n_hard = np.array([int((hard & (task == x)).sum()) for x in tasks], float)
        out_w[w] = {"cell": c, "eval calls": int(len(rows)), "impactful": int(y.sum()), "read-only": int((~y).sum()),
                    "TF-IDF caught impactful": int((y & tf_flag).sum()), "hard (TF-IDF missed impactful)": int(hard.sum()),
                    "hard share of impactful": round(float(hard.sum() / y.sum()), 4),
                    "tasks": len(tasks), "tasks with a hard call": int((n_hard > 0).sum()), "most hard in one task": int(n_hard.max()),
                    "kish design effect (hard per task)": round(kish(n_hard), 3),
                    "hard by fold": {str(k): int((hard & (fold == k)).sum()) for k in range(5)},
                    "TF-IDF false alarms on read-only (descriptive)": round(float((~y & tf_flag).sum() / (~y).sum()), 4)}
    bad = [k for k, v in checks.items() if (v is False) or (isinstance(v, int) and not isinstance(v, bool) and v > 0)
           or (isinstance(v, float) and v > 1e-5)]  # stored scores carry float32 precision: 2.8e-7 seen, flags all match
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", str(Path(__file__).resolve())], capture_output=True, text=True, cwd=HERE).stdout.strip()
    out = {"what": "hard-subset counts (TF-IDF flags only) and join checks, before any probe number on the subset",
           "git": head, "script_uncommitted_changes": bool(dirty), "inputs sha256": got, "checks": checks, "checks failed": bad,
           "worlds": out_w, "tasks in cohort": len({m["task"] for m in meta})}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"counts_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1)); print(dest)
    if bad:
        raise SystemExit(f"checks failed: {bad}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
