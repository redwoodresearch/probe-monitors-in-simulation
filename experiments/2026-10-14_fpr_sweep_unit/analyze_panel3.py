"""Readings of the panel-3 record by script (RULE.md Add-on (0d 02:1x, panel 3)): A own-real on the shared grid, B fixed vs joint
intervals at the low-FPR end with degenerate-draw counts, C sim-honest vs real honest at the real command-A thresholds, and the
three predictions' outcomes. Reads the pinned record and part (b)'s record; Modal CPU.

  PYTHONPATH=src <modal 1.6.1> run --detach experiments/2026-10-14_fpr_sweep_unit/analyze_panel3.py --record <file> --sha <sha256>
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-fprsweep_unit_kill-2026-11-14"
REC_B = ("fprsweep_unit_2026-10-15_003643.json", "d979413913b988ff6f29fbc0a5910aa01d3bb048387bc41b1a848e039ebba7d2")
CMPS = ("sim_minus_run1", "sim_minus_real_mean", "sim_minus_run2", "run2_minus_run1")
STRICT = (0.001, 0.0018)  # the band 1e-3 .. 1.78e-3
app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12")


@app.function(image=image, cpu=(1, 1), memory=(4096, 4096), timeout=300, retries=0)
def read(blob: bytes, blob_b: bytes) -> dict:
    rec, recb = json.loads(blob), json.loads(blob_b)
    res = lambda s: ("up" if s["ci95"][0] > 0 else ("down" if s["ci95"][1] < 0 else None)) if s.get("ci95") else "withheld"  # noqa: E731
    hw = lambda s: 100 * (s["ci95"][1] - s["ci95"][0]) / 2 if s.get("ci95") else None  # noqa: E731
    inb = lambda f: STRICT[0] - 1e-12 <= f <= STRICT[1]  # noqa: E731
    out = {"A": {}, "B": {}, "C": {}}
    for lab, r in rec["A"].items():
        rd = [q for q in r["points"] if q["estimable"] and not q["shaded_lt10_above"]]
        band = [q for q in rd if inb(q["fpr_target"])]
        out["A"][lab] = {"readable_floor": min(q["fpr_target"] for q in rd), "readable_points": len(rd),
                         "resolved_all_readable": {c: {"up": sum(res(q["diff"][c]) == "up" for q in rd), "down": sum(res(q["diff"][c]) == "down" for q in rd)} for c in CMPS},
                         "band_1e-3_to_1.78e-3": [{"fpr_target": q["fpr_target"], "run_false_alarm_pool": q["realized_run_false_alarm_any_unit"]["pool"],
                                                   "real_mean_catch": q["tpr"]["real_mean"]["point"], "sim_catch": q["tpr"]["sim"]["point"],
                                                   "sim_minus_run1_pts": 100 * q["diff"]["sim_minus_run1"]["point"], "sim_minus_run1_ci95_pts": [100 * x for x in q["diff"]["sim_minus_run1"]["ci95"]],
                                                   "resolved": {c: res(q["diff"][c]) for c in CMPS}} for q in band]}
    fixed = {"Sol (a)": recb["probes"]["sol_a|avg_top2|corrected|none|tight"]["points"], "FIBS": recb["probes"]["fibs|reply_mean|corrected|none"]["points"],
             "own_a_tight avg_top2": rec["A"]["own_a_tight avg_top2"]["points"], "own_a_tight cmd_A": rec["A"]["own_a_tight cmd_A"]["points"]}
    for lab, r in rec["B"].items():
        rows = []
        for q, fq in zip(r["points"], fixed[lab]):
            if q["fpr_target"] > STRICT[1]:
                continue
            assert q["fpr_target"] == fq["fpr_target"]
            hf, hj = hw(fq["diff"]["sim_minus_run1"]), hw(q["diff"]["sim_minus_run1"])
            rows.append({"fpr_target": q["fpr_target"], "fixed_shaded": fq["shaded_lt10_above"], "k_draw0": q["k_draw0"], "draws_k0": q["draws_k0"],
                         "draws_k_lt10": q["draws_k_lt10"], "k_draws_p5_p50_p95": q["k_draws_p5_p50_p95"],
                         "sim_minus_run1_fixed_ci95_pts": [100 * x for x in fq["diff"]["sim_minus_run1"]["ci95"]],
                         "sim_minus_run1_joint_ci95_pts": [100 * x for x in q["diff"]["sim_minus_run1"]["ci95"]] if q["diff"]["sim_minus_run1"].get("ci95") else None,
                         "half_width_ratio_joint_over_fixed": (hj / hf) if (hf and hj is not None) else None,
                         "joint_finite_draws": q["diff"]["sim_minus_run1"]["finite_draws"], "joint_resolved": {c: res(q["diff"][c]) for c in CMPS}})
        out["B"][lab] = rows
    for lab, r in rec["C"].items():
        rows = [{"fpr_target": q["fpr_target"], "shaded": q["shaded_lt10_above"], "sim_honest_unit_fpr": q["sim_honest"]["unit_fpr"],
                 "sim_honest_run_fa": q["sim_honest"]["run_false_alarm_any_unit"], "real_a_unit_fpr": q["real_honest"]["honest_a"]["unit_fpr"],
                 "real_a_run_fa": q["real_honest"]["honest_a"]["run_false_alarm_any_unit"], "pool_unit_fpr": q["real_honest"]["pool"]["unit_fpr"],
                 "pool_run_fa": q["real_honest"]["pool"]["run_false_alarm_any_unit"]} for q in r["points"]]
        ge = [x for x in rows if x["fpr_target"] >= 1e-3 - 1e-12]
        rat = [x["sim_honest_unit_fpr"] / x["real_a_unit_fpr"] for x in ge if x["real_a_unit_fpr"]]
        out["C"][lab] = {"sim_honest_units_runs": [r["points"][0]["sim_honest"]["units"], r["points"][0]["sim_honest"]["runs"]], "rows": rows,
                         "ratio_simh_over_real_a_unit_fpr_at_ge_1e-3": {"min": min(rat), "max": max(rat), "zero_denominators": sum(1 for x in ge if not x["real_a_unit_fpr"])}}
    band_ok = all(0.70 <= x["real_mean_catch"] <= 0.98 and x["resolved"]["sim_minus_run1"] is None for a in out["A"].values() for x in a["band_1e-3_to_1.78e-3"])
    b1e3 = [x for rows in out["B"].values() for x in rows if abs(x["fpr_target"] - 1e-3) < 1e-12]
    out["predictions"] = {
        "PA (own-real real catch 70-98% and no sim - run1 resolved, readable 1e-3..1.78e-3, every row)": band_ok,
        "PB (at 1e-3: joint sim - run1 interval 0-40% wider; <30% of draws k_b < 10; no k_b = 0)": {
            "ratios": [x["half_width_ratio_joint_over_fixed"] for x in b1e3], "draws_k_lt10": [x["draws_k_lt10"] for x in b1e3], "draws_k0": [x["draws_k0"] for x in b1e3],
            "holds": all(x["half_width_ratio_joint_over_fixed"] is not None and 1.0 <= x["half_width_ratio_joint_over_fixed"] <= 1.4 and x["draws_k_lt10"] < 3000 and x["draws_k0"] == 0 for x in b1e3)},
        "PC (sim-honest unit FPR within a factor 2 of real honest A at f >= 1e-3)": {lab: (0.5 <= v["ratio_simh_over_real_a_unit_fpr_at_ge_1e-3"]["min"] and v["ratio_simh_over_real_a_unit_fpr_at_ge_1e-3"]["max"] <= 2.0
                                                                                           and v["ratio_simh_over_real_a_unit_fpr_at_ge_1e-3"]["zero_denominators"] == 0) for lab, v in out["C"].items()}}
    return out


@app.local_entrypoint()
def main(record: str, sha: str):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("uncommitted tracked changes")
    b, bb = (HERE / "results" / record).read_bytes(), (HERE / "results" / REC_B[0]).read_bytes()
    if hashlib.sha256(b).hexdigest() != sha or hashlib.sha256(bb).hexdigest() != REC_B[1]:
        raise SystemExit("record sha256")
    out = read.remote(b, bb) | {"record": {"file": record, "sha256": sha}, "git": git("rev-parse", "HEAD"), "label": "from raw data, by script; PRELIMINARY until 09 and Codex review"}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dest = HERE / "results" / f"panel3_readings_{stamp}.json"
    dest.open("x").write(json.dumps(out, indent=1))
    print(dest.name)
