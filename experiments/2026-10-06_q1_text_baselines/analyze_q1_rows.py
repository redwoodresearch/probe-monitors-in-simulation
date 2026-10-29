"""Rows for the shared Q1 table (experiments/2026-10-06_q1_segment_end/report_q1.py --external <file>).

Reads one TF-IDF summary and one paper-judge summary from results/ and writes `{"rows": [...]}`, one row per baseline:
name, metric ("auroc" or "pairwise accuracy"), and each slice as {"auroc"|"accuracy": x, "ci95": [lo, hi]} or null.
"run" means a run's score is the mean of its per-output scores, as in the probe.

  uv run python experiments/2026-10-06_q1_text_baselines/analyze_q1_rows.py <tfidf_summary.json> <paper_judge_summary.json>
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
NAMES = {"tfidf_cleaned": "TF-IDF (cleaned text)", "tfidf_cleaned_times_masked": "TF-IDF (cleaned text, times masked)",
         "tfidf_raw": "TF-IDF (raw text)", "tfidf_raw_times_masked": "TF-IDF (raw text, times masked)"}
NOTES = {"tfidf_cleaned": "the tool-output text exactly as Qwen3-32B reads it",
         "tfidf_cleaned_times_masked": "the same, with clock times and dates replaced by placeholders",
         "tfidf_raw": "the tool message as stored in the log: output, error record, tool name",
         "tfidf_raw_times_masked": "raw text with clock times and dates replaced by placeholders"}
# The scrub ladder's steps 0 and 1 are the same models as tfidf_cleaned and tfidf_cleaned_times_masked (same text,
# features and settings), so a ladder summary can stand in for those two rows; its steps 2-3 are not Q1 table rows.
NAMES |= {"ladder_step0": NAMES["tfidf_cleaned"], "ladder_step1": NAMES["tfidf_cleaned_times_masked"]}
NOTES |= {"ladder_step0": NOTES["tfidf_cleaned"], "ladder_step1": NOTES["tfidf_cleaned_times_masked"]}
JUDGES = {"gpt-5.6-sol_medium": "Paper judge (gpt-5.6-sol, medium effort)", "gpt-5.6-sol_high": "Paper judge (gpt-5.6-sol, high effort)"}


def value(d, metric):
    return None if d is None else {metric: d[metric], "ci95": d["ci95"]}


def main():
    tfidf_path, judge_path = (HERE / "results" / Path(p).name for p in sys.argv[1:3])
    tfidf, judge = json.loads(tfidf_path.read_text()), json.loads(judge_path.read_text())
    assert tfidf["config"]["examples"] == judge["config"]["examples"], \
        f"different staging runs (cells): {tfidf['config']['examples']} vs {judge['config']['examples']}"
    rows = []
    # Summaries from before the per-baseline refactor keep results under "results"; later ones under "baselines".
    by_baseline = ({b: v["results"] for b, v in tfidf["baselines"].items()} if "baselines" in tfidf else tfidf["results"])
    for level, metric, label in (("runs", "auroc", "auroc"), ("pairs", "accuracy", "pairwise accuracy")):
        for b, res in by_baseline.items():
            if b not in NAMES:
                continue
            main_, noed, null = res["real_vs_sim/all"], res["real_vs_sim/no_editor"], res["null_e1_vs_e2/all"]
            rows.append({"name": NAMES[b], "metric": label,
                         "run": value(main_["all_cells"][level], metric),
                         "step": value(main_["all_cells"]["outputs"], "auroc") if level == "runs" else None,
                         "no_editor_run": value(noed["all_cells"][level], metric),
                         "no_errors_run": (value(res["real_vs_sim/no_errors"]["all_cells"][level], metric)
                                           if "real_vs_sim/no_errors" in res else None),
                         "base200_run": value(main_["base200"][level], metric),
                         "overlap800_run": value(main_["overlap800"][level], metric),
                         "under_32k_run": value(main_["no_run_over_32k"][level], metric),
                         "null_e1_vs_e2_run": value(null["all_cells"][level], metric),
                         "note": NOTES[b] + ("; pairwise: share of cells whose simulated run scores above its real run"
                                             if level == "pairs" else "")})
    for j, res in judge["judges"].items():
        ours = res["our_cells"]
        rows.append({"name": JUDGES[j], "metric": "pairwise accuracy", "run": value(ours["all_cells"], "accuracy"), "step": None,
                     "no_editor_run": None, "no_errors_run": None, "base200_run": value(ours["base200"], "accuracy"),
                     "overlap800_run": value(ours["overlap800"], "accuracy"), "under_32k_run": value(ours["no_run_over_32k"], "accuracy"),
                     "null_e1_vs_e2_run": None,
                     "note": (f"the LinuxArena paper's stored verdicts (no new calls), {ours['coverage']['matched']} of our cells; "
                              "shown one real and one simulated transcript of a cell in both orders, names the real one; sees whole "
                              "transcripts (tool calls and outputs; no 32k cut, no system prompt), so under_32k_run is the "
                              "comparison with the probe's context")})
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = HERE / "results" / f"q1_baseline_rows_{stamp}.json"
    out.write_text(json.dumps({"sources": {"tfidf": tfidf_path.name, "paper_judge": judge_path.name}, "rows": rows}, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
