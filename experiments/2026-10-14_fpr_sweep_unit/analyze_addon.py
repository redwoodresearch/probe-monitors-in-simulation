"""Readings of the add-on record (RULE.md Add-on, 0d 01:4x) by script: the readable floor, resolved counts per comparison (95%
interval excludes 0) over the readable points and the strict band, interval half-widths, and the four pre-registered
predictions' outcomes. Reads the pinned record only; Modal CPU.

  PYTHONPATH=src <modal 1.6.1> run --detach experiments/2026-10-14_fpr_sweep_unit/analyze_addon.py --record <file> --sha <sha256>
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
CMPS = ("sim_minus_run1", "sim_minus_real_mean", "sim_minus_run2", "run2_minus_run1")
app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12")


@app.function(image=image, cpu=(1, 1), memory=(2048, 2048), timeout=300, retries=0)
def read(blob: bytes) -> dict:
    rec = json.loads(blob)
    out = {}
    for pid, p in rec["probes"].items():
        pts = [q for q in p["enlarged"]["points"] if q["estimable"]]
        rd = [q for q in pts if not q["shaded_lt10_above"]]
        floor = min(q["fpr_target"] for q in rd)
        res = lambda q, c: ("up" if q["diff"][c]["ci95"][0] > 0 else ("down" if q["diff"][c]["ci95"][1] < 0 else None))  # noqa: E731
        band = [q for q in rd if q["fpr_target"] <= 1.8e-3]  # the readable points from the floor up to 1.78e-3
        hw = lambda q, c: 100 * (q["diff"][c]["ci95"][1] - q["diff"][c]["ci95"][0]) / 2  # noqa: E731
        out[p["label"]] = {
            "n_pool_units": p["enlarged"]["n_pool_units"], "readable_floor": floor, "readable_points": len(rd),
            "run_false_alarm_pool_at_floor": next(q["realized_run_false_alarm_any_unit"]["pool"] for q in rd if q["fpr_target"] == floor),
            "resolved_all_readable": {c: {"up": sum(res(q, c) == "up" for q in rd), "down": sum(res(q, c) == "down" for q in rd)} for c in CMPS},
            "floor_to_1.78e-3": {"points": len(band), "resolved": {c: [(q["fpr_target"], res(q, c)) for q in band if res(q, c)] for c in CMPS},
                                 "sim_minus_run1_half_width_pts": [min(hw(q, "sim_minus_run1") for q in band), max(hw(q, "sim_minus_run1") for q in band)],
                                 "sim_minus_run1_points_pts": [100 * q["diff"]["sim_minus_run1"]["point"] for q in band],
                                 "run1_catch_pts": [100 * q["tpr"]["run1"]["point"] for q in band],
                                 "run_false_alarm_pool": [q["realized_run_false_alarm_any_unit"]["pool"] for q in band]},
            "sim_minus_run1_resolved_up_between_1e-3_and_1e-2": [q["fpr_target"] for q in rd if 1e-3 - 1e-12 <= q["fpr_target"] <= 1e-2 + 1e-12 and res(q, "sim_minus_run1") == "up"],
            "relaunch_vs_bc_unit_fpr_at_ge_1e-3": [{"fpr_target": q["fpr_target"], "relaunch": q["relaunch"]["realized_unit_fpr"],
                                                   "b_u_c": (q["realized_unit_fpr"]["honest_b"] * 6179 + q["realized_unit_fpr"]["honest_c"] * 6391) / 12570}  # B 6,179 and C 6,391 units (count call)
                                                  for q in rd if q["fpr_target"] >= 1e-3 - 1e-12]}
    sol, pc = out["Sol (a)"], out["c_asis (positive control)"]
    ratios = [r["relaunch"] / r["b_u_c"] for o in (sol, pc) for r in o["relaunch_vs_bc_unit_fpr_at_ge_1e-3"] if r["b_u_c"]]
    out["predictions"] = {
        "P1 (pool 38,000-48,000 units; first readable point 2.4e-4 or 3.2e-4)": 38000 <= sol["n_pool_units"] <= 48000 and (abs(sol["readable_floor"] - 10 ** -3.625) < 1e-12 or abs(sol["readable_floor"] - 10 ** -3.5) < 1e-12),
        "P2 (Sol (a): no sim - run1 interval excludes 0 from the floor to 1.78e-3, half-widths 5-11 points)": {
            "no_interval_excludes_0": not sol["floor_to_1.78e-3"]["resolved"]["sim_minus_run1"],
            "half_widths_within_5_to_11": 5 <= sol["floor_to_1.78e-3"]["sim_minus_run1_half_width_pts"][0] and sol["floor_to_1.78e-3"]["sim_minus_run1_half_width_pts"][1] <= 11},
        "P3 (c_asis: sim - run1 resolved above 0 at >= 1 readable point in 1e-3..1e-2)": len(pc["sim_minus_run1_resolved_up_between_1e-3_and_1e-2"]) >= 1,
        "P4 (relaunch unit FPR within a factor 2 of B u C at >= 1e-3)": {"min_ratio": min(ratios), "max_ratio": max(ratios), "holds": all(0.5 <= x <= 2 for x in ratios)}}
    return out


@app.local_entrypoint()
def main(record: str, sha: str):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("uncommitted tracked changes")
    b = (HERE / "results" / record).read_bytes()
    if hashlib.sha256(b).hexdigest() != sha:
        raise SystemExit("record sha256")
    out = read.remote(b) | {"record": {"file": record, "sha256": sha}, "git": git("rev-parse", "HEAD"), "label": "from raw data, by script; PRELIMINARY until 09 and Codex review"}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dest = HERE / "results" / f"addon_readings_{stamp}.json"
    dest.open("x").write(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
