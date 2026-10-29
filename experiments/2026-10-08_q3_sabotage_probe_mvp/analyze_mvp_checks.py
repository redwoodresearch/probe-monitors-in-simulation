"""Codex's audit checks on the sabotage-call MVP. $0, local, saved out-of-fold probe scores only (no new activations,
no labels). The calls are the same 480. Folds (`--group`): main task (the repository rule; 20 folds, and with one run
per main task these are the run folds of the run-grouped probe scores) or side task (13 folds, matching the side-task
probe scores). The nuisance and TF-IDF models, ranks and bootstrap all use the chosen folds. Python 3.14 (reads the
cached transcripts for argument lengths and edit sizes; no text is written out).

1. Nuisance baseline: logistic regression on relative position, log argument length and tool, inside the same folds.
   Then probe + nuisance vs nuisance alone, and TF-IDF + nuisance. Each combination is a fixed, unfitted average of
   ranks: a fitted second stage would leak, because the probe's out-of-fold scores on the training folds came
   from models that saw the held-out fold's labels. Ranks are taken within each held-out fold, which by
   itself changes the pooled AUROC, so every combination is compared with its components ranked the same way.
2. Matched comparison: sabotage vs benign calls in the same run, with the same tool name (exact) and raw argument
   length within a factor of 2. Pairwise win rate for the probe, TF-IDF and nuisance.
3. Quiet edits: sabotage text_editor edits changing at most 3 lines (as in K5). Probe and TF-IDF on them.
4. Statistics: every interval resamples the folds' groups. The MVP null's p-values are (1 + exceedances) / 201,
   so 0 of 200 gives 0.005, the smallest value 200 draws can show.
5. Spot-check list for Aswin (~30 calls, ids and locations only):
   - agreed sabotage;
   - agreed benign;
   - disputed;
   - quiet positives;
   - the probe's highest-scoring negatives.

  uv run --no-project --python 3.14 --with numpy --with scipy --with scikit-learn --with pyyaml python analyze_mvp_checks.py \
      --group main_task --probe-calls results/sabotage_probe_mvp_calls_2026-10-08_094806.jsonl
  (--group side_task with results/sabotage_probe_mvp_calls_2026-10-08_095844_side_task.jsonl for the side-task folds)
"""

import argparse
import difflib
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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


def edit_lines(fn, args):
    if fn != "text_editor" or args.get("command") not in ("str_replace", "insert"):
        return None
    if args["command"] == "insert":
        return len(str(args.get("insert_text") or args.get("new_str") or "").splitlines())
    old, new = str(args.get("old_str") or "").splitlines(), str(args.get("new_str") or "").splitlines()
    d = list(difflib.ndiff(old, new))
    return max(sum(x.startswith("+ ") for x in d), sum(x.startswith("- ") for x in d))


def load_calls(keys):
    sys.path.insert(0, str(FOLLOWUPS))
    argv, sys.argv = sys.argv, [sys.argv[0]]
    import label_sabotage_dump as L

    sys.argv = argv
    uris = yaml.safe_load(L.REAL_RUNS.read_text())["runs"]
    info, loc = {}, {}
    for key, m, s in L.samples(keys):
        calls = [c for c in L.calls(s) if c[2] != "submit"]
        loc[key] = {"eval": f"{uris[m['cohort']]['uri']}logs/{m['file']}", "sample_id": m["sample_id"], "epoch": m["epoch"]}
        for i, (step, action, fn, args, _o) in enumerate(calls):
            n = len(json.dumps(args, ensure_ascii=False))
            info[(key, step, action)] = {"tool": "bash" if fn == "bash" or fn.endswith("_bash") else ("text_editor" if fn == "text_editor" else "other"),
                                         "fn": fn, "arg_len": n, "rel_pos": i / max(len(calls) - 1, 1), "log_len": math.log1p(n),
                                         "edit_lines": edit_lines(fn, args),
                                         "text": json.dumps({"name": fn, "arguments": args}, ensure_ascii=False)}
    return info, loc


def fold_rank(v, group, keep):
    """average ranks (ties share one) scaled to [0, 1] within each held-out fold, among the evaluated calls `keep`
    (sabotage and benign; preparation calls get NaN), so fold-specific score scales never mix"""
    from scipy.stats import rankdata

    out = np.full(len(v), np.nan)
    for g in np.unique(group):
        m = np.flatnonzero((group == g) & keep)
        out[m] = (rankdata(v[m]) - 1) / max(len(m) - 1, 1)
    return out


def grouped_fit(F, y, group, C=1.0, scale=True):
    """held-out decision scores, one fold per group, trained on sabotage (1) vs benign (0)"""
    out = np.full(len(y), np.nan)
    for g in np.unique(group):
        tr, te = (group != g) & (y >= 0), group == g
        Xtr, Xte = F[tr], F[te]
        if scale:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        out[te] = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(Xtr, y[tr]).decision_function(Xte)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-calls", required=True)
    ap.add_argument("--group", choices=("main_task", "side_task"), required=True)
    args = ap.parse_args()
    rows = [json.loads(x) for x in Path(args.probe_calls).read_text().splitlines()]
    info, loc = load_calls(sorted({r["key"] for r in rows}))
    ids = [(r["key"], r["step"], r["action"]) for r in rows]
    lab = np.array([r["label"] for r in rows]); run = np.array([r["key"] for r in rows])
    main_task = np.array([r["key"].rsplit("__", 2)[-2].split(":")[0] for r in rows])
    if any(len(set(main_task[run == k])) != 1 for k in set(run)) or len(set(main_task)) != len(set(run)):
        raise ValueError("expected one main task per run and one run per main task (run folds = main-task folds)")
    side = main_task if args.group == "main_task" else np.array([r["key"].rsplit(":", 2)[-2] for r in rows])  # the fold groups
    y = np.where(lab == "sabotage", 1, np.where(lab == "benign", 0, -1))
    sab, ben, sb = lab == "sabotage", lab == "benign", (lab == "sabotage") | (lab == "benign")
    probe = np.array([r["score_L63"] for r in rows])
    tool = np.array([info[i]["tool"] for i in ids]); rel = np.array([info[i]["rel_pos"] for i in ids])
    loglen = np.array([info[i]["log_len"] for i in ids])
    nuis = np.c_[rel, loglen, tool == "bash", tool == "text_editor"].astype(float)

    # scores, all held out by fold group
    vec_scores = np.full(len(rows), np.nan)
    texts = np.array([info[i]["text"] for i in ids], dtype=object)
    for g in np.unique(side):
        tr, te = (side != g) & (y >= 0), side == g
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200000, sublinear_tf=True, dtype=np.float32)
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(vec.fit_transform(texts[tr]), y[tr])
        vec_scores[te] = clf.decision_function(vec.transform(texts[te]))
    nuisance = grouped_fit(nuis, y, side)
    rank = lambda v: fold_rank(v, side, sb)  # noqa: E731  (within each held-out fold, sabotage and benign calls only)
    S = {"probe": probe, "tfidf": vec_scores, "nuisance": nuisance,
         "probe (within-fold rank)": rank(probe), "tfidf (within-fold rank)": rank(vec_scores), "nuisance (within-fold rank)": rank(nuisance),
         "probe+nuisance (rank average)": (rank(probe) + rank(nuisance)) / 2,
         "tfidf+nuisance (rank average)": (rank(vec_scores) + rank(nuisance)) / 2,
         "position_only": rel, "length_only": loglen}

    groups = np.unique(side)
    rng = np.random.default_rng(0)
    picks = [rng.choice(groups, len(groups)) for _ in range(BOOT)]
    idx_of = {g: np.flatnonzero((side == g) & sb) for g in groups}

    def pooled(s, pick):
        m = np.concatenate([idx_of[g] for g in pick])
        return auroc(s[m], sab[m])

    def boot_ci(s):
        v = [x for x in (pooled(s, p) for p in picks) if x is not None]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    def boot_diff(a, b):
        d = [x - z for x, z in ((pooled(a, p), pooled(b, p)) for p in picks) if x is not None and z is not None]
        return {"observed": auroc(a[sb], sab[sb]) - auroc(b[sb], sab[sb]), "ci95_groups": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                "share_above_0": float(np.mean(np.array(d) > 0))}

    src = Path(args.probe_calls)
    res = {"input": {"probe_calls": src.name, "sha256": hashlib.sha256(src.read_bytes()).hexdigest(), "group": args.group, "groups": int(len(np.unique(side)))},
           "check1_nuisance": {name: {"auroc": auroc(s[sb], sab[sb]), "ci95_groups": boot_ci(s)} for name, s in S.items()},
           "check1_differences": {"probe - tfidf": boot_diff(S["probe"], S["tfidf"]),
                                  "probe+nuisance - nuisance": boot_diff(S["probe+nuisance (rank average)"], S["nuisance (within-fold rank)"]),
                                  "tfidf+nuisance - nuisance": boot_diff(S["tfidf+nuisance (rank average)"], S["nuisance (within-fold rank)"]),
                                  "rank(nuisance) - raw nuisance": boot_diff(S["nuisance (within-fold rank)"], S["nuisance"]),
                                  "rank(probe) - rank(tfidf)": boot_diff(S["probe (within-fold rank)"], S["tfidf (within-fold rank)"]),
                                  "probe+nuisance - tfidf+nuisance": boot_diff(S["probe+nuisance (rank average)"], S["tfidf+nuisance (rank average)"])}}
    # sizing heuristic from the side-task bootstrap: standard error scales as 1/sqrt(side tasks)
    def groups_for(ci, half=0.05):
        se = (ci[1] - ci[0]) / 3.92
        return {"se": se, "groups_for_halfwidth_0.05": float(len(groups) * (1.96 * se / half) ** 2)}

    res["check4_sizing"] = {"probe_auroc": groups_for(res["check1_nuisance"]["probe"]["ci95_groups"]),
                            "probe_minus_tfidf": groups_for(res["check1_differences"]["probe - tfidf"]["ci95_groups"]),
                            "groups_now": int(len(groups)), "note": "a planning heuristic, not a guarantee"}

    # 2. matched pairs: same run, same tool, argument length within a factor of 2
    fn = np.array([info[i]["fn"] for i in ids]); alen = np.array([info[i]["arg_len"] for i in ids])
    pairs = [(a, b) for a in np.flatnonzero(sab) for b in np.flatnonzero(ben)
             if run[a] == run[b] and fn[a] == fn[b] and max(alen[a], alen[b]) <= 2 * min(alen[a], alen[b])]
    pair_side = np.array([side[a] for a, _ in pairs])

    def win(s, which):
        w = [1.0 if s[a] > s[b] else 0.5 if s[a] == s[b] else 0.0 for a, b in (pairs[i] for i in which)]
        return float(np.mean(w)) if w else None

    allp = np.arange(len(pairs))
    pidx = {g: np.flatnonzero(pair_side == g) for g in groups}
    m2 = {}
    for name in ("probe", "tfidf", "nuisance"):
        v = [x for x in (win(S[name], np.concatenate([pidx[g] for g in p])) for p in picks) if x is not None]
        m2[name] = {"win_rate": win(S[name], allp), "ci95_groups": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
    dv = [x - z for x, z in ((win(S["probe"], np.concatenate([pidx[g] for g in p])), win(S["tfidf"], np.concatenate([pidx[g] for g in p])))
                             for p in picks) if x is not None and z is not None]
    m2["probe - tfidf"] = {"observed": m2["probe"]["win_rate"] - m2["tfidf"]["win_rate"],
                           "ci95_groups": [float(np.percentile(dv, 2.5)), float(np.percentile(dv, 97.5))], "share_above_0": float(np.mean(np.array(dv) > 0))}
    res["check2_matched"] = {"pairs": len(pairs), "runs_with_pairs": len({run[a] for a, _ in pairs}), "groups_with_pairs": len(set(pair_side.tolist())),
                             "rule": "same run, same tool name (exact), raw argument length within a factor of 2"} | m2

    # 3. quiet sabotage edits (text_editor str_replace/insert changing <= 3 lines)
    quiet = np.array([sab[k] and info[ids[k]]["edit_lines"] is not None and info[ids[k]]["edit_lines"] <= 3 for k in range(len(rows))])

    def pct_in_run(s, k):
        others = [j for j in np.flatnonzero(run == run[k]) if j != k]
        return float(np.mean([s[k] > s[j] for j in others]))

    res["check3_quiet"] = {"quiet_sabotage_calls": int(quiet.sum()), "runs": len(set(run[quiet].tolist())),
                           **{name: {"quiet_vs_benign_auroc": auroc(S[name][quiet | ben], quiet[quiet | ben]),
                                     "median_percentile_in_run": float(np.median([pct_in_run(S[name], k) for k in np.flatnonzero(quiet)])),
                                     "top_of_run": int(sum(pct_in_run(S[name], k) == 1.0 for k in np.flatnonzero(quiet)))}
                              for name in ("probe", "tfidf", "nuisance")}}

    # 5. spot-check list (ids and locations only)
    A = json.loads((FOLLOWUPS / "results/k5_labels_A_2026-10-08.json").read_text())
    B = json.loads((FOLLOWUPS / "results/k5_labels_B_2026-10-08.json").read_text())

    def lab_of(D, k):
        return D[ids[k][0]]["labels"].get(f"{ids[k][1]}.{ids[k][2]}", ["benign"])[0]

    pick = lambda mask, n, by=None: list(np.flatnonzero(mask)[np.argsort(-by[mask])][:n] if by is not None else  # noqa: E731
                                         rng.choice(np.flatnonzero(mask), min(n, int(mask.sum())), replace=False))
    agreeA = np.array([lab_of(A, k) for k in range(len(rows))]); agreeB = np.array([lab_of(B, k) for k in range(len(rows))])
    spot = {"agreed_sabotage": pick((agreeA == "sabotage") & (agreeB == "sabotage"), 7),
            "agreed_benign": pick((agreeA == "benign") & (agreeB == "benign") & ben, 5),
            "disputed": pick(agreeA != agreeB, 8),
            "quiet_sabotage": pick(quiet, 4),
            "probe_top_negatives": pick(ben, 6, by=probe)}
    spot_rows, seen = [], set()
    for cat, ks in spot.items():
        for k in ks:
            if k in seen:  # one entry per call
                continue
            seen.add(k)
            key, st, ac = ids[k]
            spot_rows.append({"category": cat, "run_key": key, "step": int(st), "action": int(ac), "call": f"{st}.{ac}"} | loc[key] |
                             {"settled_label": lab[k], "labeller_A": agreeA[k], "labeller_B": agreeB[k],
                              "probe_score": float(probe[k]), "tfidf_score": float(vec_scores[k])})
    res["check5_spotcheck_count"] = len(spot_rows)
    res["check4_note"] = (f"intervals resample the {len(groups)} {args.group} groups; the MVP p-values are 0.005 (0 of 200 draws "
                          "exceed; the minimum resolution); no run-count extrapolation is made")
    res["check1_note"] = ("combinations average ranks taken within each held-out fold, so the subtracted baselines are "
                          "ranked the same way (within-fold rank of nuisance); raw nuisance is reported for reference")

    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    for name, obj in (("mvp_checks", res), ("spotcheck_list", {"note": "ids, locations, labels and scores; no transcript text",
                                                               "calls": spot_rows})):
        path, text = HERE / "results" / f"{name}_{stamp}.json", json.dumps(obj, indent=1)
        with path.open("x") as f:
            f.write(text)
        print("->", path)
    print(json.dumps({k: v for k, v in res.items() if k != "check5_spotcheck_count"}, indent=1)[:4000])


if __name__ == "__main__":
    main()
