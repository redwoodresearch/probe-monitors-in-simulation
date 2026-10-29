"""Captions by key for the extension record's figure title (r1 of plant_2026-10-15_130848): every title number recomputed from the record, with key paths.
  python3 make_captions_v2.py results/<record>.json <record commit>"""
import json
import statistics
import sys

rec_p, commit = sys.argv[1], sys.argv[2]
rec = json.load(open(rec_p))
key_, probes, deltas = "unit0.01", rec["design"]["figure_probes"], rec["design"]["deltas"]
BH, S = rec["baseline_and_first_own_flag"][key_], rec["summary"][key_]
figl = sorted(lab for lab in BH if lab.rsplit(" ", 1)[0] in probes)
kp = lambda lab, f: f"baseline_and_first_own_flag.{key_}[{lab!r}].{f}"  # noqa: E731
hw = {lab: 100 * 0 + BH[lab]["halfwidth_points"] for lab in figl}
elig = [lab for lab in figl if BH[lab]["eligible_base_interval_includes_0"]]
first = {lab: BH[lab]["first_delta_own_interval_flags"] for lab in elig}
ent = [{"what": "half-width range and median (points), 10 figure rows", "min": min(hw.values()), "max": max(hw.values()), "median": statistics.median(hw.values()),
        "keys": [kp(lab, "halfwidth_points") for lab in figl]},
       {"what": "eligible rows (uninjected interval includes 0)", "value": len(elig), "of": len(figl), "not_eligible": [lab for lab in figl if lab not in elig],
        "keys": [kp(lab, "eligible_base_interval_includes_0") for lab in figl]},
       {"what": "first delta at which the own interval flags, per eligible row", "value": first, "keys": [kp(lab, "first_delta_own_interval_flags") for lab in elig]},
       {"what": "rows flagged by delta = 8, and of those already at delta = 3",
        "value": [lab for lab in elig if lab in S.get("att_8", {}) and S["att_8"][lab]["own_interval_detects"]],
        "at3": [lab for lab in elig if lab in S.get("att_3", {}) and S["att_3"][lab]["own_interval_detects"] and lab in S.get("att_8", {}) and S["att_8"][lab]["own_interval_detects"]],
        "keys": [f"summary.{key_}.att_{d}[{lab!r}].own_interval_detects" for d in (3, 8) for lab in elig if lab in S.get(f"att_{d}", {})]},
       {"what": "never-flagging rows and their uninjected e2e point (fraction)",
        "value": {lab: rec["rows"][lab]["points"][key_]["arms"]["base"]["metric"]["e2e_vs_real_mean"]["point"] for lab, v in first.items() if v is None},
        "keys": [f"rows[{lab!r}].points.{key_}.arms.base.metric.e2e_vs_real_mean.point" for lab, v in first.items() if v is None]}]
for arm, m in (("com_8", "B_shift_fires"), ("com_8", "C_flag_outside_half_to_twice_nominal"), ("com_16", "C_flag_outside_half_to_twice_nominal")):
    labs = [lab for lab in figl if lab in S.get(arm, {})]
    ent.append({"what": f"{arm} {m}", "value": sum(S[arm][lab][m] for lab in labs), "of": len(labs), "keys": [f"summary.{key_}.{arm}[{lab!r}].{m}" for lab in labs]})
mx = max(abs(rec["rows"][lab]["points"][key_]["arms"][f"com_{d}"]["shift_vs_base"]["e2e_vs_real_mean"]["point"]) for lab in figl for d in deltas
         if f"com_{d}" in rec["rows"][lab]["points"][key_]["arms"])
ent.append({"what": "max |e2e shift| under the common-shift control (fraction)", "value": mx, "key_pattern": f"rows[*].points.{key_}.arms.com_*.shift_vs_base.e2e_vs_real_mean.point"})
json.dump({"record": f"experiments/2026-10-15_e2e_planted_shift/results/{rec_p.split('/')[-1]} @ {commit}", "numbers": ent},
          open(rec_p.replace(".json", "_captions.json"), "x"), indent=1, default=str)
for e in ent:
    print(e["what"], "->", {k: v for k, v in e.items() if k not in ("what", "keys", "key_pattern")})
