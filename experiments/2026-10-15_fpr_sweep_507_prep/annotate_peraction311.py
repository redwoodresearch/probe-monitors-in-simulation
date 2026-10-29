"""Labels and derived fields for a run_peraction311 record, by script (0d's recheck via 2e, 13:4x; no recompute; the record is unchanged).
Writes results/<record stem>_annotations.json: per row x point, Delta_full (vs the REAL MEAN, 25h's primary) with its joint and fixed-tau
95% intervals; the exclusion scenarios' LEVELS and MOVES from the primary (high: point and fixed interval; its joint interval withheld when
fewer than 95% of draws are available); the H_sim pool of the row; and the note that the record's `registered` field (sim - run 1) is the
common table's comparator, not 25h's Delta_full.

  python3 -I annotate_peraction311.py results/peraction311_all_C_<stamp>.json
"""

import hashlib
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
b = p.read_bytes()
rec = json.loads(b)
pt = lambda s: None if s is None or s.get("point") is None or s["point"] != s["point"] else round(100 * s["point"], 2)  # noqa: E731
ci = lambda s: [round(100 * x, 2) for x in s["ci95"]] if s and s.get("ci95") else "withheld (fewer than 95% of draws available)"  # noqa: E731
out = {"record": p.name, "record_sha256": hashlib.sha256(b).hexdigest(), "units": "percentage points of catch",
       "notes": {"exclusions": "6 excluded runs (Q1 t237, t300, t389; Q4 t361, t411, t413): M = 135 + 0 + 168 = 303 imputed units; t298 has its own ruling (not applicable to the roots read)",
                 "comparators": "delta_full = sim at tau_sim - mean(run 1, run 2) at tau_B (25h primary). The row field `registered` (sim_minus_run1) is the common table's registered comparator; sim - run 1 and sim - run 2 at the same thresholds are stored as gap.sim_minus_run1 / sim_minus_run2.",
                 "intervals": "joint: tau and moments re-estimated per draw; fixed: tau and moments at draw 0, attack tasks resampled. Every JOINT interval of delta_full contains 0 only where so marked below; FIXED-tau intervals are narrower and are listed beside."},
       "rows": {}}
for lb, v in rec["rows"].items():
    r = {"H_sim": {"runs": v["coverage"]["H_sim_runs"], "by_batch": v["n_per_batch"]["H_sim_runs"], "units": v["coverage"]["H_sim_units"]}, "registered_field": v["registered"], "points": {}}
    for f in ("0.01", "0.05", "0.001"):
        J, Fx = v["points"]["joint"][f]["gap"], v["points"]["fixed"][f]["gap"]
        prim = pt(J["delta_full"])
        hi, lo = pt(J["delta_full_high"]), pt(J["delta_full_low"])
        r["points"][f] = {"delta_full": {"point": prim, "joint_ci95": ci(J["delta_full"]), "fixed_ci95": ci(Fx["delta_full"]),
                                         "joint_contains_0": isinstance(ci(J["delta_full"]), list) and ci(J["delta_full"])[0] <= 0 <= ci(J["delta_full"])[1],
                                         "fixed_contains_0": isinstance(ci(Fx["delta_full"]), list) and ci(Fx["delta_full"])[0] <= 0 <= ci(Fx["delta_full"])[1]},
                          "high": {"level": hi if hi is not None else "unavailable (threshold among imputed units)", "move": round(hi - prim, 2) if hi is not None else None,
                                   "fixed_ci95": ci(Fx["delta_full_high"]), "joint_ci95": ci(J["delta_full_high"])},
                          "low": {"level": lo, "move": round(lo - prim, 2) if lo is not None else None, "fixed_ci95": ci(Fx["delta_full_low"]), "joint_ci95": ci(J["delta_full_low"])},
                          "delta_rr": {"point": pt(J["delta_rr"]), "joint_ci95": ci(J["delta_rr"]), "fixed_ci95": ci(Fx["delta_rr"])},
                          "keys": [f"rows.{lb}.points.{m}.{f}.gap.{k}" for m in ("joint", "fixed") for k in ("delta_full", "delta_full_high", "delta_full_low", "delta_rr")]}
        print(lb, f, r["points"][f]["delta_full"], "| high", r["points"][f]["high"]["level"], r["points"][f]["high"]["move"], "| low move", r["points"][f]["low"]["move"])
    out["rows"][lb] = r
dest = p.with_name(p.stem + "_annotations.json")
dest.open("x").write(json.dumps(out, indent=1))
print(dest.name, hashlib.sha256(dest.read_bytes()).hexdigest())
