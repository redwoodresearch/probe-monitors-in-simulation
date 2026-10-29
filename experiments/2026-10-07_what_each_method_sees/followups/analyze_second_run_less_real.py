"""Q1 open question: on same-batch pairs (both real honest runs from overlap800), why does the real-vs-simulated TF-IDF
model score a task's second real run as slightly more simulated than its first? Saved out-of-fold scores only ($0).

Run 1 is the first run by key, so within overlap800 it is always the lower batch index. The real-vs-simulated model is
trained on run 1s (real) against simulated runs; run 2s are never trained on. Audits, per split, level and outputs
variant, on the 163 same-batch tasks:
- the paired difference run 2 − run 1 in run score (mean of output log-odds), with a 95% main-task bootstrap;
- the same using only each run's first 10 scored outputs (coverage), and with both runs cut to their common number of
  scored outputs;
- length and truncation: an OLS of the pair difference on the pair's UNcentred differences in tokens (thousands), scored
  outputs and truncation; its intercept is the expected difference for two runs equal on all three, reported with the
  coefficients and pair-bootstrap intervals (2026-10-08 fix: the first version centred the predictors, so its
  intercept was the raw mean by construction and ruled nothing out); plus stratified means for pairs with the same
  truncation status and pairs within 2 scored outputs of each other;
- ordering: the difference by tercile of the pair's batch-index gap (run 1 always has the lower index);
- per tool, the difference in mean output score;
- the batch index: correlation of run score with index among all overlap800 real runs.

  uv run python experiments/2026-10-07_what_each_method_sees/followups/analyze_second_run_less_real.py
"""

import gzip
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from scipy.stats import spearmanr

from probe_shift.provenance import capture

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent
B = Path("~/probe-shift-baselines/experiments/2026-10-06_q1_text_baselines").expanduser()
STAGING = B / ".run_cache/examples_2026-10-06_213745"
SETS = {"main-task folds": "2026-10-06_213855", "held-out environments": "2026-10-06_213857"}
R1, R2 = "real-honest", "real-honest-2"


def boot(d, groups, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    u, inv = np.unique(groups, return_inverse=True)
    means = []
    for _ in range(n):
        w = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))[inv]
        means.append((w * d).sum() / w.sum())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": round(float(d.mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)], "n": int(len(d))}


def main():
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    runs = {r["key"]: r for r in map(json.loads, (STAGING / "runs.jsonl").read_text().splitlines())}
    pair = defaultdict(dict)
    for r in runs.values():
        if r["arm"] in (R1, R2):
            pair[r["main_task"]][r["arm"]] = r["key"]
    same = {t: d for t, d in pair.items() if len(d) == 2 and all(k.startswith("overlap800") for k in d.values())}
    idx = lambda k: int(k.split("_")[1])
    out = {"stamp": stamp, "provenance": capture(ROOT, [Path(__file__)]), "same_batch_tasks": len(same),
           "run1_lower_index_share": float(np.mean([idx(d[R1]) < idx(d[R2]) for d in same.values()])), "rows": []}
    for split, s in SETS.items():
        for level in (0, 7):
            for outputs in ("all", "no_editor"):
                per = defaultdict(list)  # key -> [(step, action, tool, score)]
                for r in map(json.loads, gzip.open(B / f"results/level{level}_scores_{s}.jsonl.gz", "rt")):
                    if r["comparison"] == "real_vs_sim" and r["outputs"] == outputs and r["arm"] in (R1, R2):
                        per[r["key"]].append((r["step"], r["action"], r["tool"], r["score"]))
                for v in per.values():
                    v.sort()
                tasks = [t for t, d in same.items() if d[R1] in per and d[R2] in per]
                mean = lambda k, cut=None: np.mean([x[3] for x in per[k][:cut]])
                d_all = np.array([mean(same[t][R2]) - mean(same[t][R1]) for t in tasks])
                d_10 = np.array([mean(same[t][R2], 10) - mean(same[t][R1], 10) for t in tasks])
                common = [min(len(per[same[t][R1]]), len(per[same[t][R2]])) for t in tasks]
                d_common = np.array([mean(same[t][R2], c) - mean(same[t][R1], c) for t, c in zip(tasks, common)])
                X = np.array([[(runs[same[t][R2]]["tokens"] - runs[same[t][R1]]["tokens"]) / 1000,
                               runs[same[t][R2]]["kept_outputs"] - runs[same[t][R1]]["kept_outputs"],
                               float(runs[same[t][R2]]["truncated"]) - float(runs[same[t][R1]]["truncated"])] for t in tasks], float)
                A = np.c_[np.ones(len(tasks)), X]  # uncentred: the intercept is the difference at zero length differences
                rng = np.random.default_rng(0)
                betas = np.array([np.linalg.lstsq(A[i], d_all[i], rcond=None)[0]
                                  for i in (rng.integers(0, len(tasks), len(tasks)) for _ in range(2000))])
                point = np.linalg.lstsq(A, d_all, rcond=None)[0]
                ci = np.percentile(betas, [2.5, 97.5], axis=0)
                adjusted = {nm: [round(float(point[j]), 4), round(float(ci[0, j]), 4), round(float(ci[1, j]), 4)]
                            for j, nm in enumerate(["intercept_equal_length", "per_1k_tokens", "per_scored_output", "run2_truncated_not_run1"])}
                same_trunc = np.array([runs[same[t][R1]]["truncated"] == runs[same[t][R2]]["truncated"] for t in tasks])
                near_len = np.abs(X[:, 1]) <= 2
                gap = np.array([idx(same[t][R2]) - idx(same[t][R1]) for t in tasks])
                terc = np.digitize(gap, np.percentile(gap, [100 / 3, 200 / 3]))
                by_tool = {}
                for tool in sorted({x[2] for k in per for x in per[k]}):
                    dt = [np.mean([x[3] for x in per[same[t][R2]] if x[2] == tool]) - np.mean([x[3] for x in per[same[t][R1]] if x[2] == tool])
                          for t in tasks if any(x[2] == tool for x in per[same[t][R1]]) and any(x[2] == tool for x in per[same[t][R2]])]
                    if len(dt) >= 20:
                        by_tool[tool] = boot(np.array(dt), np.arange(len(dt)))
                ov = [k for k in per if k.startswith("overlap800")]
                run_mean = {k: mean(k) for k in ov}
                out["rows"].append({
                    "split": split, "level": level, "outputs": outputs,
                    "run2_minus_run1": boot(d_all, np.array(tasks)), "share_run2_more_sim": round(float((d_all > 0).mean()), 3),
                    "first_10_outputs": boot(d_10, np.array(tasks)), "common_length": boot(d_common, np.array(tasks)),
                    "length_truncation_ols": adjusted,
                    "same_truncation_status": boot(d_all[same_trunc], np.array(tasks)[same_trunc]),
                    "within_2_scored_outputs": boot(d_all[near_len], np.array(tasks)[near_len]),
                    "by_index_gap_tercile": {f"{['small', 'middle', 'large'][k]} (gap {int(gap[terc == k].min())}-{int(gap[terc == k].max())})":
                                             boot(d_all[terc == k], np.array(tasks)[terc == k]) for k in range(3)},
                    "by_tool": by_tool,
                    "spearman_index_vs_score_overlap800": round(float(spearmanr([idx(k) for k in ov], [run_mean[k] for k in ov])[0]), 3)})
    f = lambda v: f"{v['mean']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]"
    md = [f"# Second real honest run vs first: the real-vs-simulated TF-IDF model's run score ({stamp})", "",
          f"Population: the {out['same_batch_tasks']} same-batch real-honest task pairs (both runs overlap800); a run's score is "
          "the mean of its output log-odds under the honest real-vs-simulated TF-IDF model (saved out-of-fold scores, levels 0 "
          f"and 7). Run 1 is the first by key, so the lower batch index in {out['run1_lower_index_share']:.0%} of pairs.", "",
          "| split | level | outputs | run 2 − run 1 | share run 2 more simulated | first 10 outputs | common length | same truncation | within 2 outputs |",
          "|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['split']} | {r['level']} | {r['outputs']} | {f(r['run2_minus_run1'])} | {r['share_run2_more_sim']} | {f(r['first_10_outputs'])} | "
           f"{f(r['common_length'])} | {f(r['same_truncation_status'])} | {f(r['within_2_scored_outputs'])} |" for r in out["rows"]]
    g = lambda v: f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]"
    md += ["", "Length and truncation (OLS on uncentred pair differences; the intercept is the difference for equal runs):", "",
           "| split | level | outputs | intercept (equal length) | per 1k tokens | per scored output | run 2 truncated, run 1 not |", "|---|---|---|---|---|---|---|"]
    md += [f"| {r['split']} | {r['level']} | {r['outputs']} | " + " | ".join(g(r['length_truncation_ols'][k]) for k in
                                                                     ("intercept_equal_length", "per_1k_tokens", "per_scored_output", "run2_truncated_not_run1")) + " |"
           for r in out["rows"]]
    md += ["", "By tercile of the batch-index gap (run 2 index − run 1 index):", ""]
    md += [f"- {r['split']}, level {r['level']}, {r['outputs']}: " + "; ".join(f"{k} {f(v)}" for k, v in r["by_index_gap_tercile"].items())
           for r in out["rows"]]
    md += ["", "Per tool (main-task folds, level 7, all outputs): " + "; ".join(f"{k} {f(v)} (n {v['n']})" for k, v in
                                                                            next(r for r in out["rows"] if r["split"] == "main-task folds" and r["level"] == 7 and r["outputs"] == "all")["by_tool"].items())]
    (HERE / "results" / f"second_run_less_real_{stamp}.json").write_text(json.dumps(out, indent=1))
    (HERE / "results" / f"second_run_less_real_{stamp}.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
