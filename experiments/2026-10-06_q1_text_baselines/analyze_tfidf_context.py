"""TF-IDF (all text up to this output, running mean of segment scores): the probe at a tool output has read the whole
transcript before it, so this baseline gives each output the mean of the out-of-fold TF-IDF scores of the segments in its
run up to and including it.

Variants (config `variants`): the running mean over output scores only (the headline), the same on times-masked output
scores, and over output and tool-call scores together. Inputs are out-of-fold score files from analyze_tfidf.py /
run_tfidf_modal.py on the same staged examples. Order within a run: an agent turn's calls all come before its outputs,
outputs in call order (as the reader renders them). A run's score is the mean over all its kept segments. With all
segments, and with text_editor segments dropped (each from its own no-editor model). Metrics as in analyze_tfidf.py, for
real epoch 1 vs simulated and for the e1-vs-e2 null; a run's score is the mean of its per-output running-mean scores, as
for the probe (the mean over all its segment scores is reported separately). Plus per-output AUROC by position in run.

  uv run python experiments/2026-10-06_q1_text_baselines/analyze_tfidf_context.py [config/tfidf_context.yaml]
  (on Modal, one container per variant: run_tfidf_modal.py::context --config <config>)
"""

import gzip
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import yaml

from analyze_tfidf import SPLITS, add_slices, comparisons, on_side, evaluate_scores, load, output_variants, provenance, with_cells
from probe_shift.scoring import auroc_ci

HERE = Path(__file__).parent
BUCKETS = {"1": (0, 1), "2-3": (1, 3), "4-10": (3, 10), "11-25": (10, 25), ">25": (25, 10**9)}  # 1-based positions


def run(cfg: dict, fold_of: dict, cache: Path, results: Path, out_dir: Path, stamp: str, part: str = "") -> dict:
    """The analysis for cfg's variants: stagings read from `cache`, score files from `results`; writes
    <name>_scores_<stamp>[__<part>].jsonl.gz to out_dir and returns the summary (run_tfidf_modal.py::context runs one
    variant per container as parts and joins them)."""
    d = load(cache / cfg["examples"], fold_of, cells=cfg.get("cells"))  # kept outputs: the rows scored
    add_slices(d, cfg)
    index = {(r["key"], r["step"], r["action"]): i for i, r in enumerate(d["rows"])}
    selected = set(cfg.get("cells") or [])
    files = {f for v in cfg["variants"].values() for k, f in v.items() if k != "running" and f}
    loaded = {f: [json.loads(l) for l in gzip.open(results / f, "rt")] for f in files}
    summary = {"config": cfg, "counts": d["counts"], "results": {}, "segments_used": {}}  # git: main() or the Modal launch
    out_path = out_dir / f"{cfg['name']}_scores_{stamp}{'__' + part if part else ''}.jsonl.gz"
    with gzip.open(out_path, "wt") as fh:
        for variant, sources in cfg["variants"].items():
            for outputs in cfg.get("output_variants", ["all", "no_editor"]):
                for comparison, (neg, pos, scored) in comparisons(cfg).items():
                    by_run = {}  # key -> [((step, kind, action), kind, score)]; a turn's calls sort before its outputs
                    fold = d["fold"].copy()  # replaced by the output scores' own folds (main-task or environment)
                    for kind, f in sources.items():
                        if kind == "running" or not f:
                            continue
                        for r in loaded[f]:
                            if cfg.get("cells") is not None and r["cell"] not in selected:
                                continue  # the score file may cover more cells than this analysis
                            if r["comparison"] == comparison and r["outputs"] == outputs:
                                by_run.setdefault(r["key"], []).append(((r["step"], int(kind == "outputs"), r["action"]), kind, r["score"]))
                                if kind == "outputs":
                                    fold[index[(r["key"], r["step"], r["action"])]] = r["fold"]
                    name = f"{variant}/{comparison}/{outputs}"
                    summary["segments_used"][name] = dict(Counter(k for v in by_run.values() for _, k, _ in v))
                    score, position = np.full(len(d["rows"]), np.nan), np.full(len(d["rows"]), -1)
                    run_total = {}
                    for key, segs in by_run.items():
                        segs.sort()
                        total, n_out = 0.0, 0
                        for n, (order, kind, s) in enumerate(segs, 1):
                            total += s
                            if kind == "outputs":
                                i = index[(key, order[0], order[2])]
                                # running mean up to here, or (variant `running: false`) the output's own score
                                score[i], position[i] = (total / n if sources.get("running", True) else s), n_out
                                n_out += 1
                        run_total[key] = total / len(segs)

                    keep = output_variants(d, {"output_variants": [outputs]})[outputs]
                    # Run level as for the probe: the mean of the run's per-output (running-mean) scores.
                    res = evaluate_scores(d, score, keep, neg, pos, cfg)
                    m = keep & (on_side(d["arm"], neg) | on_side(d["arm"], pos)) & ~np.isnan(score)
                    # Alternative run score, reported separately: the mean over all the run's segment scores.
                    keys, first = np.unique(d["key"][m], return_index=True)
                    first = np.flatnonzero(m)[first]
                    res["runs_mean_over_segments"] = auroc_ci(on_side(d["arm"][first], pos), np.array([run_total[k] for k in keys]),
                                                              d["main_task"][first], cfg["bootstrap"], cfg["seed"])
                    res["by_position"] = {b: auroc_ci(on_side(d["arm"][m & (position >= lo) & (position < hi)], pos),
                                                      score[m & (position >= lo) & (position < hi)],
                                                      d["main_task"][m & (position >= lo) & (position < hi)], cfg["bootstrap"], cfg["seed"])
                                          for b, (lo, hi) in BUCKETS.items()}
                    summary["results"][name] = res
                    for i in np.flatnonzero(~np.isnan(score) & keep & np.isin(d["arm"], scored)):
                        r = d["rows"][i]
                        fh.write(json.dumps({"baseline": variant, "comparison": comparison, "outputs": outputs, "arm": r["arm"],
                                             "key": r["key"], "cell": r["cell"], "main_task": r["main_task"], "step": r["step"],
                                             "action": r["action"], "tool": r["tool"], "fold": int(fold[i]),
                                             "position": int(position[i]), "score": round(float(score[i]), 5)}) + "\n")
                    a = res["all_cells"]
                    print(f"{variant:32} {comparison:14} {outputs:9} outputs {a['outputs']['auroc']:.3f} {a['outputs']['ci95']}  "
                          f"runs {a['runs']['auroc']:.3f}  pairs {a['pairs']['accuracy']:.3f}  by position "
                          + " ".join(f"{b}:{v['auroc']:.3f}" for b, v in res["by_position"].items()), flush=True)
    summary["score_file"] = out_path.name
    return summary


def main():
    cfg = with_cells(yaml.safe_load((Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "config" / "tfidf_context.yaml").read_text()))
    fold_of = json.loads((SPLITS / cfg["folds_file"]).read_text())["task_folds"]["main_task_to_fold"]
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    summary = provenance(cfg, {}) | run(cfg, fold_of, HERE / ".run_cache", HERE / "results", HERE / "results", stamp)
    out = HERE / "results" / f"{cfg['name']}_summary_{stamp}.json"
    out.write_text(json.dumps(summary, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
