"""The honest real-vs-real controls, recomputed by machine batch from saved per-output scores (no refit).

A task's first real honest run by key is compared with its second. Keys sort `base200_…` before `overlap800_…`, so in 95
of the 258 two-run tasks the first run is from the base200 batch (group-writable files) and the second from the
overlap800 batch; the other 163 pairs are both overlap800. Each control is recomputed on all pairs (reproducing the
stored number), on same-batch pairs (overlap800/overlap800) and on mixed pairs (base200/overlap800), as run AUROC with
a 95% main-task bootstrap. Controls:
- `null_e1_vs_e2`: the model fit on first (0) vs second (1) real run;
- `scored_by_real_vs_sim`: the real-vs-simulated model's scores, second run vs first ("real-honest-2_vs_real-honest_scored
  _by_this_model" in the summaries).
Also real honest vs simulated honest split by the batch of the task's first real run (every honest simulation
references an overlap800 run, so base200 cells cross batches). Score sets: own output (`level<N>_scores_<stamp>`, main-task folds and held-out environments), running mean
(`tfidf_context_*honest_all_levels_scores_<stamp>`), and the honest pilot (`level{0,7}_scores_2026-10-06_213900`).

  uv run python experiments/2026-10-06_q1_text_baselines/analyze_honest_floor_batch.py --scores-dir <dir with the score files>
Writes results/honest_floor_by_batch_<stamp>.{json,md}.
"""

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from probe_shift.provenance import capture
from probe_shift.scoring import auroc_ci

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
R1, R2 = "real-honest", "real-honest-2"
# (name, split, score file pattern with {level}, the baseline name in the file, summary file pattern, summary key prefix)
SETS = [
    ("own output", "main-task folds", "level{level}_scores_2026-10-06_213855.jsonl.gz", "level{level}", "level{level}_summary_2026-10-06_213855.json", ""),
    ("own output", "held-out environments", "level{level}_scores_2026-10-06_213857.jsonl.gz", "level{level}", "level{level}_summary_2026-10-06_213857.json", ""),
    ("running mean", "main-task folds", "tfidf_context_honest_all_levels_scores_2026-10-06_220459.jsonl.gz", "level{level}_running_mean",
     "tfidf_context_honest_all_levels_summary_2026-10-06_220459.json", "level{level}_running_mean/"),
    ("running mean", "held-out environments", "tfidf_context_heldout_env_honest_all_levels_scores_2026-10-06_220502.jsonl.gz",
     "level{level}_running_mean", "tfidf_context_heldout_env_honest_all_levels_summary_2026-10-06_220502.json", "level{level}_running_mean/"),
    ("pilot (20 tasks held out)", "main-task folds", "level{level}_scores_2026-10-06_213900.jsonl.gz", "level{level}", "level{level}_summary_2026-10-06_213900.json", ""),
]
PILOT_LEVELS = (0, 7)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", default=str(HERE / "results"), help="where the (git-ignored) score files are")
    args = ap.parse_args()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    sdir = Path(args.scores_dir).expanduser()
    out = {"stamp": stamp, "provenance": capture(ROOT, [Path(__file__)]), "inputs": {}, "rows": []}
    cache = {}
    for name, split, pattern, baseline, summary, prefix in SETS:
        for level in (PILOT_LEVELS if name.startswith("pilot") else range(8)):
            path = sdir / pattern.format(level=level)
            if path not in cache:  # one pass per file: {(baseline, comparison, outputs): {key: [arm, task, sum, n]}}
                runs = defaultdict(dict)
                for r in map(json.loads, gzip.open(path, "rt")):
                    if r["arm"] not in (R1, R2):
                        continue
                    d = runs[(r["baseline"], r["comparison"], r["outputs"])].setdefault(r["key"], [r["arm"], r["main_task"], 0.0, 0])
                    d[2] += r["score"]
                    d[3] += 1
                cache[path] = runs
                out["inputs"][path.name] = sha(path)
            stored = json.loads((HERE / "results" / summary.format(level=level)).read_text())["results"]
            for outputs in ("all", "no_editor"):
                for control, comparison in (("null_e1_vs_e2", "null_e1_vs_e2"), ("scored_by_real_vs_sim", "real_vs_sim")):
                    runs = cache[path].get((baseline.format(level=level), comparison, outputs))
                    if not runs:
                        continue
                    first = {}
                    for k, (a, t, _, _) in runs.items():
                        first.setdefault(t, {})[a] = k.split("_")[0]
                    pattern_of = {t: f"{d[R1]}/{d[R2]}" for t, d in first.items() if len(d) == 2}
                    row = {"set": name, "split": split, "level": level, "outputs": outputs, "control": control}
                    for subset, keep in (("all pairs", lambda p: True), ("same batch", lambda p: p == "overlap800/overlap800"),
                                         ("mixed batch", lambda p: p == "base200/overlap800")):
                        ks = [k for k, (a, t, _, _) in runs.items() if t in pattern_of and keep(pattern_of[t])]
                        y = np.array([runs[k][0] == R2 for k in ks])
                        s = np.array([runs[k][2] / runs[k][3] for k in ks])
                        row[subset] = auroc_ci(y, s, np.array([runs[k][1] for k in ks]))
                    # The all-pairs number must be the stored one, where one was stored (the running-mean summaries
                    # do not report the real-vs-sim model's score on the second run).
                    key = f"{prefix.format(level=level)}{comparison}/{outputs}"
                    s_runs = (stored[key]["all_cells"]["runs"] if control == "null_e1_vs_e2" else
                              stored[key].get("real-honest-2_vs_real-honest_scored_by_this_model", {}).get("cells_with_real-honest-2", {}).get("runs"))
                    if s_runs is not None:
                        assert abs(s_runs["auroc"] - row["all pairs"]["auroc"]) < 1e-3, (row, s_runs)
                    row["stored"] = None if s_runs is None else s_runs["auroc"]
                    out["rows"].append(row)
    # Real honest vs simulated honest, split by the batch of the task's first real run (the run that comparison uses):
    # every honest simulation references an overlap800 run, so base200 cells cross batches.
    out["real_vs_sim_by_real_batch"] = []
    for name, split, pattern, baseline, _, _ in SETS:
        if name != "own output":
            continue
        for level in range(8):
            for outputs in ("all", "no_editor"):
                runs = {}
                for r in map(json.loads, gzip.open(sdir / pattern.format(level=level), "rt")):
                    if r["comparison"] == "real_vs_sim" and r["outputs"] == outputs and r["arm"] in (R1, "sim-honest"):
                        d = runs.setdefault(r["key"], [r["arm"], r["main_task"], 0.0, 0])
                        d[2] += r["score"]
                        d[3] += 1
                real_batch = {t: k.split("_")[0] for k, (a, t, _, _) in runs.items() if a == R1}
                row = {"split": split, "level": level, "outputs": outputs}
                for subset, keep in (("all cells", lambda b: True), ("same batch", lambda b: b == "overlap800"), ("base200 real", lambda b: b == "base200")):
                    ks = [k for k, (a, t, _, _) in runs.items() if keep(real_batch.get(t))]
                    row[subset] = auroc_ci(np.array([runs[k][0] == "sim-honest" for k in ks]), np.array([runs[k][2] / runs[k][3] for k in ks]),
                                           np.array([runs[k][1] for k in ks]))
                out["real_vs_sim_by_real_batch"].append(row)
    fmt = lambda v: f"{v['auroc']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]"
    md = [f"# Honest real-vs-real controls by machine batch ({stamp})", "",
          ("Run AUROC, second real honest run vs first, 95% main-task bootstrap. All pairs: 258 tasks (pilot: fewer); "
           + "same batch: 163 overlap800/overlap800; mixed: 95 base200/overlap800."), "",
          "| set | split | level | outputs | control | all pairs (as reported) | same batch | mixed batch |", "|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['set']} | {r['split']} | {r['level']} | {r['outputs']} | {r['control']} | {fmt(r['all pairs'])} | "
           f"{fmt(r['same batch'])} | {fmt(r['mixed batch'])} |" for r in out["rows"]]
    md += ["", "## Real honest vs simulated honest, by the first real run's batch (own output)", "",
           "| split | level | outputs | all cells | same batch (overlap800 real) | base200 real |", "|---|---|---|---|---|---|"]
    md += [f"| {r['split']} | {r['level']} | {r['outputs']} | {fmt(r['all cells'])} | {fmt(r['same batch'])} | {fmt(r['base200 real'])} |"
           for r in out["real_vs_sim_by_real_batch"]]
    (HERE / "results" / f"honest_floor_by_batch_{stamp}.json").write_text(json.dumps(out, indent=1))
    (HERE / "results" / f"honest_floor_by_batch_{stamp}.md").write_text("\n".join(md))
    print(f"wrote results/honest_floor_by_batch_{stamp}.json and .md ({len(out['rows'])} rows)")


if __name__ == "__main__":
    main()
