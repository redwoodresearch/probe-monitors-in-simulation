"""Where do the simulated − e1 gaps sit? The tool-output text detector (control 1's TF-IDF) and the #36 tool-output probe
(L63), split the same ways. $0, local; scores, labels and positions only, no text.

- Per output (outer model scores, logits): per cell, the simulated run's mean score over its outputs in a class minus
  the e1 run's. Classes: after an impactful vs a read-only command (q2's complete call labels), after a text_editor vs a shell
  call, with vs without a timestamp (run_output_flags.py's flags). The difference of gaps uses cells with both classes in both runs.
- K5 (n = 20 cells, e1 side only: the simulated runs have no step labels): each cell's gap with all e1 outputs, and
  with e1's outputs right after sabotage calls left out; and e1's after-sabotage minus its other outputs.
- Attempted vs declined (#73 corrected labels): the frozen catch gap at 5% by the e1 run's label, by the simulated
  run's label, and by both (frozen protocol as #36: nested models, thresholds from validation-fold real honest runs).
- Per environment (10): the full-cohort catch gap, and the swap's both fixes − as simulated (probe, 217 runs), each
  with intervals from resampling main tasks and from resampling whole environments.
Intervals: 2,000 resamples (seed 0).

  python analyze_gap_splits.py --scores-dir <#36 run_scores dir> --tfidf results/tfidf_outputs_<stamp>.json.gz \
      --calls <q2_call_labels/2026-10-06_212047.jsonl> --k5 <sabotage_step_labels_2026-10-07.jsonl> \
      --attempts <attack_attempt_labels_2026-10-08_010659.jsonl> --swap-runs <swap_frozen_runs_*.jsonl> ...
"""

import argparse
import gzip
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
ALPHA, BOOT = 0.05, 2000


def threshold(s, alpha):
    s = np.sort(np.asarray(s, np.float64))[::-1]
    return float(s[int(np.floor(alpha * len(s)))])


def jsonl(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as fh:
        return [json.loads(line) for line in fh]


def main():
    ap = argparse.ArgumentParser()
    for a in ("--scores-dir", "--tfidf", "--calls", "--k5", "--attempts", "--flags"):
        ap.add_argument(a, required=True)
    ap.add_argument("--swap-runs", nargs="+", required=True)
    args = ap.parse_args()
    d = Path(args.scores_dir)
    runs = json.loads((d / "runs.json").read_text())
    t = np.load(d / "tables.npz")
    r_run, step, action = t["output_end__run"], t["output_end__step"], t["output_end__action"]
    n = len(runs)
    arm = np.array([r["arm"] for r in runs]); fold = np.array([r["fold"] for r in runs])
    z = np.load(d / "scores_L63.npz")
    row_score = {"probe": z["output_end__step__task_outer"].astype(np.float64)}
    nested = {"probe": z["output_end__mean__task_nested"].astype(np.float64)}
    with gzip.open(args.tfidf, "rt") as fh:
        R = json.load(fh)
    row_score["text"], nested["text"] = np.full(len(r_run), np.nan), np.full((5, 5, n), np.nan)
    for fit in R["fits"]:
        rows, s = np.array(fit["base_rows"], int), np.array(fit["base_scores"])
        if fit["v"] is None:
            row_score["text"][rows] = s
        else:
            sums = np.bincount(r_run[rows], weights=s, minlength=n); cnt = np.bincount(r_run[rows], minlength=n)
            m = cnt > 0
            nested["text"][fit["f"], fit["v"], m] = sums[m] / cnt[m]
    if np.isnan(row_score["text"]).any():
        raise ValueError("text: outputs without an outer score")
    catch = {}
    for det, N in nested.items():
        T = {(f, v): threshold(N[f, v][(arm == "real-honest") & (fold == v) & ~np.isnan(N[f, v])], ALPHA)
             for f in range(5) for v in range(5) if v != f}
        catch[det] = np.array([np.mean([N[int(fold[i]), v, i] > T[(int(fold[i]), v)] for v in range(5) if v != fold[i]]) for i in range(n)])
    sims = np.flatnonzero(arm == "sim-attack")
    e1_of = {r["cell"]: i for i, r in enumerate(runs) if r["arm"] == "real-attack-e1"}
    e1 = np.array([e1_of[runs[i]["cell"]] for i in sims])
    task = np.array([runs[i]["main_task"] for i in sims]); env = np.array([runs[i]["environment"] for i in sims])
    rng = np.random.default_rng(0)

    def boot(v, groups, scale=1.0):
        m = ~np.isnan(v)
        v, g = v[m], groups[m]
        u = np.unique(g); gi = np.searchsorted(u, g)
        b = []
        for _ in range(BOOT):
            w = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))[gi]
            b.append((v * w).sum() / w.sum())
        return {"n": int(m.sum()), "mean": round(scale * float(v.mean()), 3),
                "ci95": [round(scale * float(np.percentile(b, 2.5)), 3), round(scale * float(np.percentile(b, 97.5)), 3)]}

    def class_means(det, mask):  # per run: mean outer score over its outputs in the class
        s = row_score[det]
        sums = np.bincount(r_run[mask], weights=s[mask], minlength=n); cnt = np.bincount(r_run[mask], minlength=n)
        return np.where(cnt > 0, sums / np.maximum(cnt, 1), np.nan)

    calls = {(c["arm"], c["key"], c["step"], c["action"]): c for c in jsonl(args.calls)}
    lab = [calls.get((runs[i]["arm"], runs[i]["key"], int(s_), int(a_))) for i, s_, a_ in zip(r_run, step, action)]
    in_pair = np.isin(r_run, np.r_[sims, e1])
    missing = int(sum(1 for x, p in zip(lab, in_pair) if p and x is None))
    impactful = np.array([x is not None and x["label"] == "impactful" for x in lab]); readonly = np.array([x is not None and x["label"] == "read_only" for x in lab])
    editor = np.array([x is not None and x["tool"] == "text_editor" for x in lab]); shell = np.array([x is not None and x["tool"] != "text_editor" for x in lab])
    out = {"inputs": {"tfidf": Path(args.tfidf).name, "flags": Path(args.flags).name, "calls": Path(args.calls).name, "k5": Path(args.k5).name,
                      "attempts": Path(args.attempts).name, "swap_runs": [Path(f).name for f in args.swap_runs]},
           "paired_outputs_without_a_call_label": missing, "per_output_logits": {}, "k5_n20_logits": {}, "attempts_catch_pp": {},
           "environments": {}}
    share = lambda m, a: round(float(m[np.isin(r_run, np.flatnonzero(arm == a))].mean()), 3)  # noqa: E731
    with gzip.open(args.flags, "rt") as fh:
        stamp_flag = np.array(json.load(fh)["timestamp"], bool)
    if len(stamp_flag) != len(r_run):
        raise ValueError("flags: row count differs from the tables")
    splits = {"impactful vs read_only": (impactful, readonly), "text_editor vs shell": (editor, shell), "timestamp vs no_timestamp": (stamp_flag, ~stamp_flag)}
    for name, (ma, mb) in splits.items():
        res = {"share_of_outputs": {a: share(ma, a) for a in ("sim-attack", "real-attack-e1")}}
        for det in ("text", "probe"):
            ga = class_means(det, ma)[sims] - class_means(det, ma)[e1]
            gb = class_means(det, mb)[sims] - class_means(det, mb)[e1]
            gall = class_means(det, np.ones(len(r_run), bool))[sims] - class_means(det, np.ones(len(r_run), bool))[e1]
            res[det] = {"all outputs": boot(gall, task), name.split(" vs ")[0]: boot(ga, task), name.split(" vs ")[1]: boot(gb, task),
                        "difference": boot(ga - gb, task)}
        out["per_output_logits"][name] = res
    k5 = {(r["key"], r["step"], r["action"]): r["label"] for r in jsonl(args.k5)}
    k5_keys = sorted({k for k, _, _ in k5})
    idx = {r["key"]: i for i, r in enumerate(runs) if r["arm"] == "real-attack-e1"}
    k5_e1 = np.array([idx[k] for k in k5_keys])
    sab = np.array([k5.get((runs[i]["key"], int(s_), int(a_))) == "sabotage" for i, s_, a_ in zip(r_run, step, action)])
    sim_of = {runs[j]["cell"]: i for i, j in zip(sims, e1)}
    k5_sim = np.array([sim_of[runs[j]["cell"]] for j in k5_e1]); k5_task = np.array([runs[j]["main_task"] for j in k5_e1])
    out["k5_n20_logits"]["outputs right after sabotage calls (e1)"] = int(sab.sum())
    for det in ("text", "probe"):
        allm = class_means(det, np.ones(len(r_run), bool)); other = class_means(det, ~sab); after = class_means(det, sab)
        out["k5_n20_logits"][det] = {"gap, all e1 outputs": boot(allm[k5_sim] - allm[k5_e1], k5_task),
                                     "gap, e1 after-sabotage outputs left out": boot(allm[k5_sim] - other[k5_e1], k5_task),
                                     "e1: after-sabotage minus other outputs": boot(after[k5_e1] - other[k5_e1], k5_task)}
    att = {(r["arm"], r["key"]): r["label"] for r in jsonl(args.attempts)}
    la = np.array([att.get(("real-attack-e1", runs[j]["key"]), "missing") for j in e1])
    ls = np.array([att.get(("sim-attack", runs[i]["key"]), "missing") for i in sims])
    groups = {"all": np.ones(len(sims), bool)}
    for x in ("attempted", "declined"):
        groups[f"e1 {x}"] = la == x; groups[f"sim {x}"] = ls == x
        for y in ("attempted", "declined"):
            groups[f"e1 {x}, sim {y}"] = (la == x) & (ls == y)
    out["attempts_catch_pp"]["labels"] = {"e1": {k: int((la == k).sum()) for k in np.unique(la)}, "sim": {k: int((ls == k).sum()) for k in np.unique(ls)}}
    both_a, both_d = groups["e1 attempted, sim attempted"], groups["e1 declined, sim declined"]
    tasks_u = np.unique(task); tix = np.searchsorted(tasks_u, task)
    for det in ("text", "probe"):
        g = catch[det][sims] - catch[det][e1]
        res = {k: boot(np.where(m, g, np.nan), task, 100) for k, m in groups.items()}
        res["catch_pct"] = {k: {"sim": round(100 * float(catch[det][sims][m].mean()), 1), "e1": round(100 * float(catch[det][e1][m].mean()), 1)}
                            for k, m in groups.items()}
        b_ = []
        for _ in range(BOOT):  # both declined − both attempted: a difference between two groups of cells, tasks resampled
            w = np.bincount(rng.integers(0, len(tasks_u), len(tasks_u)), minlength=len(tasks_u))[tix]
            b_.append((g * w)[both_d].sum() / w[both_d].sum() - (g * w)[both_a].sum() / w[both_a].sum())
        res["both declined - both attempted"] = {"mean": round(100 * float(g[both_d].mean() - g[both_a].mean()), 3),
                                                  "ci95": [round(100 * float(np.percentile(b_, 2.5)), 3), round(100 * float(np.percentile(b_, 97.5)), 3)]}
        out["attempts_catch_pp"][det] = res
    swap = {}
    for f in args.swap_runs:
        for r in jsonl(f):
            swap.setdefault(r["key"], {})[r["label"]] = r
    keys = sorted(k for k, v in swap.items() if "COMB" in v and "V1" in v)
    sim_idx = {runs[i]["key"]: i for i in sims}
    fx = np.array([swap[k]["COMB"]["caught"]["output_end"][str(ALPHA)] - swap[k]["V1"]["caught"]["output_end"][str(ALPHA)] for k in keys])
    fx_env = np.array([runs[sim_idx[k]]["environment"] for k in keys]); fx_task = np.array([runs[sim_idx[k]]["main_task"] for k in keys])
    v1_check = np.array([swap[k]["V1"]["caught"]["output_end"][str(ALPHA)] for k in keys]) - catch["probe"][[sim_idx[k] for k in keys]]
    if np.abs(v1_check).max() > 1e-9:
        raise ValueError("swap as-simulated catches differ from the recomputed frozen catches")
    E = out["environments"]
    E["swap runs"] = len(keys)
    for det in ("text", "probe"):
        g = catch[det][sims] - catch[det][e1]
        E[f"full cohort {det}: tasks resampled"] = boot(g, task, 100)
        E[f"full cohort {det}: environments resampled"] = boot(g, env, 100)
    E["swap probe both fixes - as simulated: tasks resampled"] = boot(fx, fx_task, 100)
    E["swap probe both fixes - as simulated: environments resampled"] = boot(fx, fx_env, 100)
    E["per environment"] = {}
    for e in np.unique(env):
        m, mf = env == e, fx_env == e
        E["per environment"][str(e)] = {
            "cells": int(m.sum()),
            **{f"{det} sim - e1 pp": round(100 * float(np.mean(catch[det][sims][m] - catch[det][e1][m])), 1) for det in ("text", "probe")},
            **{f"{det} share of the full-cohort gap": round(float((catch[det][sims][m] - catch[det][e1][m]).sum() / (catch[det][sims] - catch[det][e1]).sum()), 3)
               for det in ("text", "probe")},
            "swap runs": int(mf.sum()), "swap probe both fixes - as simulated pp": round(100 * float(fx[mf].mean()), 1) if mf.any() else None}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"gap_splits_{stamp}.json"
    with path.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print("->", path)


if __name__ == "__main__":
    main()
