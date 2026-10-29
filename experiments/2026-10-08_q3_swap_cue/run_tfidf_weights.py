"""Which words does the tool-output text detector rely on? Refits control 1's 5 outer TF-IDF models (same texts, recipe
and folds as run_tfidf_outputs.py, which saved scores but not weights) and splits the e1 − simulated score difference
into per-feature contributions. Modal CPU (Aswin's go via the orchestrator: about $0.05, cap $0.50).

- Gate: each refit must reproduce the saved per-output scores of its outer model (results/tfidf_outputs_<stamp>.json.gz).
- Contribution of feature j under outer model f: w_j × D_j, where D_j is the mean over cells of fold f of (the e1 run's
  mean feature value over its tool outputs − the simulated run's). Summed over features it equals the mean e1 −
  simulated run-score difference (checked); folds are combined weighted by their cell count.
- Every feature of every model is saved (n-gram, weight, e1 and simulated mean values), so the analysis can combine a
  feature's contributions over all folds. The n-grams are transcript text: they go to data/ (git-ignored) and S3. The
  summary in results/ holds ranks, numbers and counts only.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.2' --with numpy modal run \
      experiments/2026-10-08_q3_swap_cue/run_tfidf_weights.py --saved results/tfidf_outputs_2026-10-08_140722.json.gz
"""

import gzip
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q3swapcue_tfidf-weights_kill-2026-11-06"
WORK = "probe-shift_aswin-surya_sub-q3attack_work_kill-2026-11-06"
DATA_ID = "307cbfee925c"
TFIDF = {"ngram_range": [1, 2], "min_df": 3, "max_features": 200000, "sublinear_tf": True, "C": 1.0, "max_iter": 2000}
GATE_MAX_ABS = 1e-4

app = modal.App(APP)
work_vol = modal.Volume.from_name(WORK, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "scikit-learn==1.7.2", "scipy==1.18.1")  # the version the original run resolved


@app.function(image=image, cpu=4, memory=49152, timeout=3600, volumes={"/work": work_vol.read_only()})
def fit(work_run: str, f: int, saved_rows: list, saved_scores: list) -> dict:
    import numpy as np
    import scipy.sparse as sp
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    t0 = time.time()
    d = Path("/work/data") / DATA_ID
    runs = json.loads((d / "runs.json").read_text())
    r_run = np.load(d / "tables.npz")["output_end__run"]
    with gzip.open(Path("/work/runs") / work_run / "texts.json.gz", "rt") as fh:
        base = np.array(json.load(fh)["base"], object)
    arm = np.array([r["arm"] for r in runs]); fold = np.array([r["fold"] for r in runs])
    y = np.where(arm == "real-attack-e1", 1.0, np.where(arm == "real-honest", 0.0, np.nan))
    tr = np.flatnonzero((~np.isnan(y) & (fold != f))[r_run])
    te = np.flatnonzero((fold == f)[r_run])
    vec = TfidfVectorizer(ngram_range=tuple(TFIDF["ngram_range"]), min_df=TFIDF["min_df"], max_features=TFIDF["max_features"],
                          sublinear_tf=TFIDF["sublinear_tf"], dtype=np.float32)
    clf = LogisticRegression(C=TFIDF["C"], max_iter=TFIDF["max_iter"]).fit(vec.fit_transform(base[tr]), y[r_run[tr]])
    Xte = vec.transform(base[te])
    s = clf.decision_function(Xte)
    if not np.array_equal(te, np.array(saved_rows)):
        raise ValueError(f"fold {f}: scored rows differ from the saved run")
    gate = float(np.abs(s - np.array(saved_scores)).max())
    if gate > GATE_MAX_ABS:
        raise ValueError(f"fold {f}: refit scores differ from the saved ones by up to {gate}")
    # per-run mean feature vectors (runs of fold f), then the mean over cells of e1 − simulated
    runs_f = np.unique(r_run[te]); pos = np.searchsorted(runs_f, r_run[te])
    cnt = np.bincount(pos, minlength=len(runs_f))
    A = sp.csr_matrix((1.0 / cnt[pos], (pos, np.arange(len(te)))), shape=(len(runs_f), len(te)))
    M = (A @ Xte.astype(np.float64)).tocsr()
    row_of = {int(r): i for i, r in enumerate(runs_f)}
    e1_of = {runs[i]["cell"]: i for i in range(len(runs)) if arm[i] == "real-attack-e1"}
    cells = [(row_of[e1_of[runs[i]["cell"]]], row_of[i]) for i in runs_f if arm[i] == "sim-attack"]
    ie, isim = np.array([c[0] for c in cells]), np.array([c[1] for c in cells])
    D = np.asarray(M[ie].mean(0) - M[isim].mean(0)).ravel()
    w = clf.coef_[0].astype(np.float64)
    contrib = w * D
    run_mean = np.bincount(pos, weights=s, minlength=len(runs_f)) / cnt
    gap = float(np.mean(run_mean[ie] - run_mean[isim]))
    if abs(contrib.sum() - gap) > 1e-6 * max(1.0, abs(gap)):
        raise ValueError(f"fold {f}: contributions sum to {contrib.sum()}, the score gap is {gap}")
    names = vec.get_feature_names_out()
    sim_mean, e1_mean = np.asarray(M[isim].mean(0)).ravel(), np.asarray(M[ie].mean(0)).ravel()
    return {"f": f, "cells": len(cells), "gap_logits": gap, "gate_max_abs": gate, "vocabulary": len(names),
            "contrib_pos_sum": float(contrib[contrib > 0].sum()), "contrib_neg_sum": float(contrib[contrib < 0].sum()),
            "features": {"ngram": names.tolist(), "weight": w.tolist(), "e1_mean": e1_mean.tolist(), "sim_mean": sim_mean.tolist()},
            "seconds": round(time.time() - t0)}


@app.local_entrypoint()
def main(saved: str):
    with gzip.open(HERE / saved if not Path(saved).is_absolute() else saved, "rt") as fh:
        S = json.load(fh)
    outer = {x["f"]: x for x in S["fits"] if x["v"] is None}
    t0 = datetime.now(ZoneInfo("America/Los_Angeles"))
    print("start", t0.isoformat(), flush=True)
    res = list(fit.starmap([(S["work_run"], f, outer[f]["base_rows"], outer[f]["base_scores"]) for f in range(5)]))
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    data = HERE.parents[1] / "data" / "q3_swap_cue"
    data.mkdir(parents=True, exist_ok=True)
    path = data / f"tfidf_weights_{stamp}.json.gz"  # n-grams: git-ignored, archived to S3
    with gzip.open(path, "xt") as fh:
        json.dump({"stamp": stamp, "start": t0.isoformat(), "saved": Path(saved).name, "work_run": S["work_run"], "tfidf": TFIDF, "fits": res}, fh)
    for r in res:
        print(f"fold {r['f']}: {r['cells']} cells, gap {r['gap_logits']:+.3f} logits, gate {r['gate_max_abs']:.1e}, {r['seconds']} s")
    print("stop", datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(), "->", path)
