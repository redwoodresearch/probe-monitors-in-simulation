"""Counts for the caption revision (2e/0d, 10-15): which unit-level points of part (b)'s record have a resolved comparison (its 95%
interval excludes 0), scoped by readable points, alarm band and threshold range. Reads the pinned record only; Modal CPU.

  PYTHONPATH=src <modal 1.6.1> run --detach experiments/2026-10-14_fpr_sweep_unit/analyze_readings.py
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
CMPS = ("sim_minus_real_mean", "sim_minus_run1", "sim_minus_run2", "run2_minus_run1")
STRICT = [10 ** (-4 + 0.125 * i) for i in (8, 9, 10)]  # 1e-3, 1.33e-3, 1.78e-3
app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12")


@app.function(image=image, cpu=(1, 1), memory=(2048, 2048), timeout=300, retries=0)
def count(blob: bytes) -> dict:
    rec = json.loads(blob)

    def res(q, c):
        iv = q["diff"][c]["ci95"]
        return "up" if iv[0] > 0 else ("down" if iv[1] < 0 else None)

    def tally(pts):
        out = {}
        for c in CMPS:
            r = [res(q, c) for _, q in pts]
            out[c] = {"up": r.count("up"), "down": r.count("down")}
        return out

    main = [(pid, q) for pid, p in rec["probes"].items() if not pid.startswith("c_asis") for q in p["points"] if not q["shaded_lt10_above"]]
    fa = lambda q: q["realized_run_false_alarm_any_unit"]["pool"]  # noqa: E731
    hi = [(pid, q) for pid, q in main if fa(q) >= 0.22]
    low = [(pid, q) for pid, q in main if fa(q) <= 0.05]
    strict = [(pid, q) for pid, q in main if any(abs(q["fpr_target"] - f) <= 1e-12 * f for f in STRICT)]
    srm = [(pid, q) for pid, q in main if res(q, "sim_minus_real_mean")]
    r2r1 = {"lower": 0, "equal": 0, "higher": 0}
    for _, q in srm:
        d = q["tpr"]["run2"]["point"] - q["tpr"]["run1"]["point"]
        r2r1["lower" if d < 0 else ("equal" if d == 0 else "higher")] += 1
    lst = lambda pts, c: [{"probe": pid.split("|")[0], "fpr_target": q["fpr_target"], "point_pts": 100 * q["diff"][c]["point"], "ci95_pts": [100 * x for x in q["diff"][c]["ci95"]],  # noqa: E731
                           "run_false_alarm_pool": fa(q)} for pid, q in pts if res(q, c)]
    pc = [q for q in rec["probes"]["c_asis|avg_top2|corrected|none"]["points"] if not q["shaded_lt10_above"]]
    return {"readable_main_points": len(main), "all_readable": tally(main),
            "alarm_ge_22pct": {"points": len(hi)} | tally(hi), "alarm_le_5pct": {"points": len(low)} | tally(low),
            "strict_1e-3_to_1.8e-3": {"points": len(strict)} | tally(strict) | {"resolved_list": {c: lst(strict, c) for c in CMPS}},
            "sim_minus_run1_resolved": lst(main, "sim_minus_run1"),
            "of_sim_minus_real_mean_resolved_run2_vs_run1_catch": {"points": len(srm)} | r2r1,
            "c_asis_readable_sim_minus_real_mean": [{"fpr_target": q["fpr_target"], "resolved": res(q, "sim_minus_real_mean"),
                                                    "ci95_pts": [100 * x for x in q["diff"]["sim_minus_real_mean"]["ci95"]]} for q in pc]}


@app.local_entrypoint()
def main():
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("uncommitted tracked changes")
    b = (HERE / "results" / REC_B[0]).read_bytes()
    if hashlib.sha256(b).hexdigest() != REC_B[1]:
        raise SystemExit("record sha256")
    out = count.remote(b) | {"record": {"file": REC_B[0], "sha256": REC_B[1]}, "git": git("rev-parse", "HEAD"),
                              "label": "from raw data, by script; PRELIMINARY until 09 and Codex review"}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dest = HERE / "results" / f"readings_{stamp}.json"
    dest.open("x").write(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
