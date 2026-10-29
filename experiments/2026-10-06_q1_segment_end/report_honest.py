"""The honest arm at every text level (held-out environments, output_end): the probe on real honest vs simulated honest
runs next to baselines-q3's TF-IDF at the same level. Full analyses give the frozen rule's headline (inner CV over 64
layers); lite analyses have no headline, so they give the range over the 64 layers, which is not a verdict.

  uv run python experiments/2026-10-06_q1_segment_end/report_honest.py --tfidf <heldout-env honest summary.json> \
      --analysis L0=results/q1_gpu_<id>.json ... > results/honest_table_<stamp>.md
"""

import argparse
import json
from pathlib import Path

VARIANTS = {"all_outputs": "all", "no_editor_outputs": "no_editor"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfidf", required=True, help="baselines-q3 held-out-environment honest summary (all levels)")
    ap.add_argument("--analysis", nargs="+", required=True, help="L<level>=results/q1_gpu_<id>.json")
    args = ap.parse_args()
    tf = json.loads(Path(args.tfidf).read_text())["results"]
    md = ["| Level | Analysis | Outputs | Probe (held-out environments) | Real vs real (probe) | TF-IDF own | TF-IDF running mean |",
          "|---|---|---|---|---|---|---|"]
    for x in args.analysis:
        label, path = x.split("=", 1)
        res = json.loads(Path(path).read_text())
        n = int(label.lstrip("L"))
        head = res.get("headline_held_out_environment") or {}
        for variant, short in VARIANTS.items():
            h = ((head.get("output_end") or {}).get("real_vs_sim") or {}).get(variant)
            nh = ((head.get("output_end") or {}).get("null_e1_vs_e2") or {}).get(variant)
            if h:
                probe = f"{h['run']['auroc']:.3f} [{h['run']['ci95'][0]:.3f}, {h['run']['ci95'][1]:.3f}] (headline)"
            else:  # lite: no headline; the spread over layers
                a = [r["results"]["output_end"]["real_vs_sim"]["held_out_environment"][variant]["run"]["auroc"]
                     for r in res["by_layer"].values() if "output_end" in r["results"]
                     and (r["results"]["output_end"]["real_vs_sim"]["held_out_environment"].get(variant) or {}).get("run")]
                probe = f"{min(a):.3f}–{max(a):.3f} (range over {len(a)} layers, lite)" if a else "–"
            null = "–" if not nh else f"{nh['run']['auroc']:.3f}"
            own = tf.get(f"level{n}_own_output/real_vs_sim/{short}", {}).get("all_cells", {}).get("runs", {}).get("auroc")
            rm = tf.get(f"level{n}_running_mean/real_vs_sim/{short}", {}).get("all_cells", {}).get("runs", {}).get("auroc")
            md.append(f"| {n} | `{Path(path).stem.removeprefix('q1_gpu_')}` | {variant.replace('_', ' ')} | {probe} | {null} | "
                      f"{'–' if own is None else format(own, '.3f')} | {'–' if rm is None else format(rm, '.3f')} |")
    print("\n".join(md))


if __name__ == "__main__":
    main()
