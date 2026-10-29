"""Is the sabotage-call probe's 0.83 just command vocabulary? The same evaluation, with the call's text in place of
activations. Local, $0, no model calls. Python 3.14 (reads the cached transcripts).

- Text: #36's text-baseline input, json.dumps({"name": ..., "arguments": ...}) per call.
- Classifier: #36's TF-IDF recipe (word 1–2-grams, min_df 3, at most 200,000 features, sublinear tf, logistic
  regression C = 1), with balanced classes as in the probe.
- Calls: the probe's 480 calls (those with activation features), read from its per-call file.
- Folds: side-task-grouped (13) and leave-one-run-out (20), the same as the probe.
- The ensemble averages ranks taken within each held-out fold, so different folds' score scales never mix.

It reports the same metrics as the probe: held-out AUROC with a run-bootstrap interval, within-run AUROC, how often
the sabotage call tops its run, sabotage vs preparation, and content-ending calls only. It also reports the probe's
own numbers on the identical calls, and the AUROC of the average of the two models' within-fold ranks. The text
itself is never written out: the outputs hold scores and metrics only.

  python analyze_tfidf_baseline.py --probe-calls results/sabotage_probe_mvp_calls_<stamp>_side_task.jsonl
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).parent
FOLLOWUPS = HERE.parent / "2026-10-07_q3_codex_followups"
BOOT = 2000


def auroc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, bool)
    if y.all() or not y.any():
        return None
    order = np.argsort(s, kind="mergesort")
    ss, r, i = s[order], np.empty(len(s)), 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        r[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def call_texts(keys):
    sys.path.insert(0, str(FOLLOWUPS))
    argv, sys.argv = sys.argv, [sys.argv[0]]  # label_sabotage_dump reads sys.argv at import
    import label_sabotage_dump as L

    sys.argv = argv
    out = {}
    for key, _m, s in L.samples(keys):
        for step, action, fn, args, _o in L.calls(s):
            out[(key, step, action)] = json.dumps({"name": fn, "arguments": args}, ensure_ascii=False)
    return out


def evaluate(s, lab, run, content, rng):
    sab, ben, prep = lab == "sabotage", lab == "benign", lab == "preparation"
    sb = sab | ben
    keys = np.unique(run)
    vals = []
    for _ in range(BOOT):
        pick = rng.choice(keys, len(keys))
        p = np.concatenate([s[(run == k) & sab] for k in pick]); n = np.concatenate([s[(run == k) & ben] for k in pick])
        a = auroc(np.r_[p, n], np.r_[np.ones(len(p)), np.zeros(len(n))].astype(bool))
        if a is not None:
            vals.append(a)
    within = [w for w in (auroc(s[(run == k) & sb], sab[(run == k) & sb]) for k in keys) if w is not None]
    return {"sabotage_vs_benign": auroc(s[sb], sab[sb]), "ci95_runs": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
            "within_run_mean_auroc": float(np.mean(within)),
            "runs_whose_top_call_is_sabotage": int(sum(bool(sab[run == k][np.argmax(s[run == k])]) for k in keys)),
            "sabotage_vs_preparation": auroc(s[sab | prep], sab[sab | prep]),
            "preparation_vs_benign": auroc(s[prep | ben], prep[prep | ben]),
            "content_calls_sabotage_vs_benign": auroc(s[sb & content], sab[sb & content])}


def fold_rank(v, group):
    """ranks scaled to [0, 1] within each held-out fold, so fold-specific score scales never mix"""
    out = np.empty(len(v))
    for g in np.unique(group):
        m = np.flatnonzero(group == g)
        out[m] = np.argsort(np.argsort(v[m], kind="mergesort"), kind="mergesort") / max(len(m) - 1, 1)
    return out


def paired_diff(a, b, lab, run, rng):
    """probe minus text, sabotage vs benign pooled AUROC, the same run resamples for both (paired)"""
    sab, ben = lab == "sabotage", lab == "benign"
    keys, d = np.unique(run), []
    for _ in range(BOOT):
        pick = rng.choice(keys, len(keys))
        m = np.concatenate([np.flatnonzero((run == k) & (sab | ben)) for k in pick])
        x, y = auroc(a[m], sab[m]), auroc(b[m], sab[m])
        if x is not None and y is not None:
            d.append(x - y)
    return {"observed": auroc(a[sab | ben], sab[sab | ben]) - auroc(b[sab | ben], sab[sab | ben]),
            "ci95_runs": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "share_above_0": float(np.mean(np.array(d) > 0))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-calls", required=True)
    args = ap.parse_args()
    rows = [json.loads(x) for x in Path(args.probe_calls).read_text().splitlines()]
    texts = call_texts(sorted({r["key"] for r in rows}))
    missing = [r for r in rows if (r["key"], r["step"], r["action"]) not in texts]
    if missing:
        raise ValueError(f"{len(missing)} probe calls have no text")
    X = np.array([texts[(r["key"], r["step"], r["action"])] for r in rows], dtype=object)
    lab = np.array([r["label"] for r in rows]); run = np.array([r["key"] for r in rows]); content = np.array([r["content"] for r in rows])
    side = np.array([r["key"].rsplit(":", 2)[-2] for r in rows])
    y = np.where(lab == "sabotage", 1, np.where(lab == "benign", 0, -1))
    probe = np.array([r["score_L63"] for r in rows])
    out = {"probe_calls": Path(args.probe_calls).name, "calls": len(rows), "recipe": "#36 tfidf: ngram (1,2), min_df 3, max_features 200000, "
           "sublinear_tf, LogisticRegression C=1, max_iter 2000, class_weight balanced", "results": {}}
    for name, group in (("side_task", side), ("run", run)):
        s = np.full(len(rows), np.nan)
        for g in np.unique(group):
            tr, te = (group != g) & (y >= 0), group == g
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200000, sublinear_tf=True, dtype=np.float32)
            clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(vec.fit_transform(X[tr]), y[tr])
            s[te] = clf.decision_function(vec.transform(X[te]))
        rng = np.random.default_rng(0)
        res = {"tfidf": evaluate(s, lab, run, content, rng)}
        if name == "side_task":  # the probe file holds the side-task-grouped held-out scores
            res["probe_L63_same_calls"] = evaluate(probe, lab, run, content, np.random.default_rng(0))
            rs, rp = fold_rank(s, group), fold_rank(probe, group)  # within each held-out side task
            res["mean_of_ranks"] = evaluate((rs + rp) / 2, lab, run, content, np.random.default_rng(0))
            res["probe_tfidf_score_spearman"] = float(np.corrcoef(rs, rp)[0, 1])
            res["probe_minus_tfidf_paired"] = paired_diff(probe, s, lab, run, np.random.default_rng(1))
        out["results"][name] = res
        out.setdefault("per_call_tfidf_scores", {})[name] = [float(x) for x in s]  # in the order of the probe-calls file
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path, text = HERE / "results" / f"tfidf_baseline_{stamp}.json", json.dumps(out, indent=1)
    with path.open("x") as f:
        f.write(text)
    for name, res in out["results"].items():
        for m, v in res.items():
            print(name, m, v if not isinstance(v, dict) else {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()})
    print("->", path)


if __name__ == "__main__":
    main()
