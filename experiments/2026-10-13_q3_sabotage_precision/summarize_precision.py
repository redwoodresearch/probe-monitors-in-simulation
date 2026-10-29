"""Compact summary and README table from a precision_POSTHOC_*.json (numbers by key, no hand-typed values).

  uv run --no-project --python 3.12 python summarize_precision.py results/precision_POSTHOC_<stamp>.json [full]

With `full` (Amendment 3), the table's 16-task columns come from the all-full-population design (D2 at K = 16), not D1.
"""
import hashlib
import json
import sys
from pathlib import Path

NAMES = {"panel_call_end": "panel labels, call_end (catch 25.5%)", "adjudicated": "adjudicated labels (recall 12.4%)",
         "command_endpoint": "command endpoint (catch 36.6%)"}


def pts(x):
    return f"±{100 * x:.1f}"


def det(x):
    return "not reached on the tested grid" if isinstance(x, str) else f"{100 * x['gap']:.1f}"


def main(p, full=False):
    d = json.loads(Path(p).read_text())
    out = {"source": Path(p).name, "source_sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest(), "git": d["git"],
           "sigma_ml": d["sigma"]["ml"], "sigma_fit": d["sigma_fit"], "observed": d["observed"], "model_check_k16": d["model_check_k16"], "settings": {}}
    lines = ["| detector | real 1 − sim (no control) | with run 2: difference in differences | with run 2: pooled real reference | detectable gap at 80% (sim lower / sim higher), points | tasks for ±15 (pooled) | cost for ±15, 3 runs per new task: raw / +30% (central; high) |",
             "|---|---|---|---|---|---|---|"]
    for k, s in d["settings"].items():
        c, pm = (s["D2_real_full"][0] if full else s["D1_real_sampled_k16"]), s["pm15"]
        cg = pm["cost_at_K_grid"]
        cost = (f"${cg['3_runs_central']['raw']:.0f} / ${cg['3_runs_central']['plus30']:.0f}; ${cg['3_runs_high']['raw']:.0f} / ${cg['3_runs_high']['plus30']:.0f}"
                if isinstance(cg, dict) else cg)
        row = {"D1_k16": {q: c[q]["sd_x1.96"] for q in ("without_control_real1_minus_sim", "did_equals_real2_minus_sim", "pooled_real_minus_sim",
                                                          "yardstick_real2_minus_real1", "real1_alone", "real2_alone", "real_pooled_alone", "sim_alone")},
               "D1_k16_detectable_80": {q: c[f"detectable_80_two_arm_{q}"] for q in ("sim_lower", "sim_higher")},
               "D2_by_K": {c2["K"]: {"two_arm": c2["without_control_real1_minus_sim"]["sd_x1.96"], "pooled": c2["pooled_real_minus_sim"]["sd_x1.96"],
                                     "detectable_80_sim_lower": c2["detectable_80_two_arm_sim_lower"], "detectable_80_sim_higher": c2["detectable_80_two_arm_sim_higher"],
                                     "equiv15_fixed_width_approx": c2["equiv15_fixed_width_approx"]} for c2 in s["D2_real_full"]},
               "pm15": pm, "sensitivity_k16": {q: v["sd_x1.96"] for q, v in s["sensitivity_k16"].items()}}
        out["settings"][k] = row
        lines.append(f"| {NAMES[k]} | {pts(c['without_control_real1_minus_sim']['sd_x1.96'])} | {pts(c['did_equals_real2_minus_sim']['sd_x1.96'])} | "
                     f"{pts(c['pooled_real_minus_sim']['sd_x1.96'])} | {det(c['detectable_80_two_arm_sim_lower'])} / {det(c['detectable_80_two_arm_sim_higher'])} | "
                     f"{pm['K_grid']} ({pm['K_grid_pooled']}) | {cost} |")
    q = Path(p).with_name(Path(p).name.replace("precision_POSTHOC_", "precision_summary_POSTHOC_"))
    with q.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print("\n".join(lines))
    print("->", q)


if __name__ == "__main__":
    main(sys.argv[1], full=len(sys.argv) > 2 and sys.argv[2] == "full")
