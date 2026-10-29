"""Captions by key for the planted-shift figure: every number in the title, recomputed from the record and cited with its key paths. Local, reads one JSON.
  python3 make_captions.py results/<record>.json <record commit>"""
import json
import sys

rec_p, commit = sys.argv[1], sys.argv[2]
rec = json.load(open(rec_p))
key_, probes, deltas = "unit0.01", rec["design"]["figure_probes"], rec["design"]["deltas"]
S = rec["summary"][key_]
sel = lambda arm: sorted(lab for lab in S.get(arm, {}) if lab.rsplit(" ", 1)[0] in probes)  # noqa: E731
ent = []


def count(arm, metric):
    labs = sel(arm)
    v = sum(S[arm][lab][metric] for lab in labs)
    ent.append({"what": f"{arm} {metric}: count of figure-probe rows", "value": v, "of": len(labs),
                "keys": [f"summary.{key_}.{arm}[{lab!r}].{metric}" for lab in labs]})
    return v, len(labs)


count("att_1", "e2e_shift_fires")
count("att_5", "e2e_shift_fires")
count("att_8", "own_interval_detects")
count("com_8", "B_shift_fires")
count("com_8", "C_flag_outside_half_to_twice_nominal")
mx, where = 0.0, None
for lab, r in rec["rows"].items():
    if r["family"] not in probes:
        continue
    for d in deltas:
        a = r["points"][key_]["arms"].get(f"com_{d:g}")
        if a and abs(a["shift_vs_base"]["e2e_vs_real_mean"]["point"]) >= mx:
            mx, where = abs(a["shift_vs_base"]["e2e_vs_real_mean"]["point"]), f"rows[{lab!r}].points.{key_}.arms.com_{d:g}.shift_vs_base.e2e_vs_real_mean.point"
ent.append({"what": "max |e2e shift| under the common-shift negative control (fraction; title shows x100 points)", "value": mx, "key": where})
json.dump({"record": f"experiments/2026-10-15_e2e_planted_shift/results/{rec_p.split('/')[-1]} @ {commit}", "numbers": ent}, open(rec_p.replace(".json", "_captions.json"), "x"), indent=1)
print(json.dumps(ent, indent=1))
