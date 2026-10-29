"""Per-record counts for the without-S4 correction (read-only, by key from each record's own rows[label][origin].coverage): cohort cells and the
calibration pools (H_sim, H_A, H_B runs and units) of the with-S4 row vs its "without S4" row. Usage: nos4_counts.py <out.json> <record>..."""
import json
import sys

KS = ("cohort_cells", "H_sim_runs", "H_sim_units", "H_A_runs", "H_A_units", "H_B_runs", "H_B_units")
out = {}
for f in sys.argv[2:]:
    d = json.load(open(f))
    rec = {}
    for lb, r in d["rows"].items():
        for w in ("S_newmains", "union"):
            wo = f"{w} without S4"
            if w in r and wo in r and "coverage" in r[w] and "coverage" in r[wo]:
                rec.setdefault(lb, {})[w] = {k: [r[w]["coverage"].get(k), r[wo]["coverage"].get(k)] for k in KS}
    out[f] = rec
json.dump(out, open(sys.argv[1], "x"), indent=1)
for f, rec in out.items():
    pools_moved = sum(1 for lb in rec for w in rec[lb] for k in KS[1:] if rec[lb][w][k][0] != rec[lb][w][k][1])
    print(f, "labels", len(rec), "pairs", sum(len(v) for v in rec.values()), "pool counts that moved", pools_moved)
