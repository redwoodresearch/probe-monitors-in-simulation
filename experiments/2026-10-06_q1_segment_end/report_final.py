"""The final Q1 table: for every text level, split and output set, the probe's headline (the frozen rule: the layer
chosen by inner CV on training folds) against TF-IDF at the same text level, own-output and running mean, as paired
task-bootstrap gaps on identical outputs; and the frozen rule's verdict on the primary row. Every number names its
text level, split and TF-IDF variant.

Inputs are protocol_report outputs (results/q1_protocol_<stamp>.json), one per text level and TF-IDF variant:

  uv run python experiments/2026-10-06_q1_segment_end/report_final.py \
      --own L0=results/q1_protocol_<a>.json ... L7=... --running-mean L0=results/q1_protocol_<b>.json ... \
      [--readout output_end] > results/final_table_<stamp>.md
"""

import argparse
import json
from pathlib import Path

SPLITS = {"A_held_out_environment": "held-out environments", "B_main_task": "main-task folds"}
VARIANTS = {"all_outputs": "all outputs", "no_editor_outputs": "no editor outputs"}


def ci(d, key="auroc"):
    return "–" if not d else f"{d[key]:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]"


def gap(d):
    return "–" if not d else f"{d['gap']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"


def verdict(g, sn):
    """The frozen rule on the gap; a probe advantage stands only with its selection-aware null passed."""
    if not g:
        return "–"
    lo, hi = g["ci95"]
    v = "no probe advantage" if lo <= 0 <= hi else ("probe advantage" if lo > 0 else "TF-IDF better")
    if v == "probe advantage":
        v += " (selection-aware null: " + ("missing" if not sn else "passed" if sn["headline_above_p95"] else "FAILED") + ")"
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--own", nargs="+", required=True, help="label=protocol json with TF-IDF own-output")
    ap.add_argument("--running-mean", nargs="*", default=[], help="label=protocol json with TF-IDF running mean")
    ap.add_argument("--readout", default="output_end")
    args = ap.parse_args()
    own = {x.split("=", 1)[0]: json.loads(Path(x.split("=", 1)[1]).read_text()) for x in args.own}
    rm = {x.split("=", 1)[0]: json.loads(Path(x.split("=", 1)[1]).read_text()) for x in args.running_mean}
    md = [f"# Q1: probe vs TF-IDF at every text level ({args.readout}, real vs simulated, run level)", "",
          "Probe: the headline layer chosen per outer fold by inner CV on that fold's training folds (protocol_level7.md). "
          "TF-IDF: baselines-q3 at the same text level, each output's own text (own) or the running mean over the outputs "
          "so far (running mean), both averaged over the same outputs as the probe. Gaps: probe minus TF-IDF, paired "
          "task bootstrap (1,000 resamples). Verdict: the frozen rule, on the gap against own-output TF-IDF. Fixed slot: "
          "the gap vs own with outputs ending in `<n>` or `<END_TOOL_OUTPUT>` dropped from both (no refit).", "",
          "| Text level | Split | Outputs | Probe (all its outputs) | Probe (outputs TF-IDF also scored) | TF-IDF own | "
          "Gap vs own | TF-IDF running mean | Gap vs running mean | Verdict | Clean cells: gap vs own | "
          "Fixed slot dropped: gap vs own | Selection-aware null p95 | Real vs real |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for label, p in own.items():
        for split, sname in SPLITS.items():
            for variant, vname in VARIANTS.items():
                k = f"{args.readout}__real_vs_sim__{variant}"
                r = p["results"][split].get(k)
                if not r:
                    continue
                n = p["results"][split].get(f"{args.readout}__null_e1_vs_e2__{variant}") or {}
                g = r.get("gap_probe_minus_tfidf")
                rg = (rm.get(label, {}).get("results", {}).get(split, {}).get(k) or {}).get("gap_probe_minus_tfidf")
                sn = r.get("selection_null")
                rm_tf = "–" if not rg else format(rg["baseline"], ".3f")
                null95 = "–" if not sn else format(sn["p95"], ".3f")
                clean = gap((r.get("clean_cells") or {}).get("gap_probe_minus_tfidf"))
                fixed = gap((r.get("no_fixed_slot_outputs") or {}).get("gap_probe_minus_tfidf"))
                md.append(f"| {label} | {sname} | {vname} | {ci(r['probe'])} | {ci(r.get('probe_on_shared_outputs'))} | "
                          f"{ci(r.get('tfidf'))} | {gap(g)} | {rm_tf} | {gap(rg)} | {verdict(g, sn)} | {clean} | {fixed} | {null95} | "
                          f"{ci(n.get('probe'))} |")
    print("\n".join(md))


if __name__ == "__main__":
    main()
