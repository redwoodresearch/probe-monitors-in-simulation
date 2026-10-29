"""Keys and counts only: the declared E12 pairs gate 2 never compared (no label x combo where BOTH keys have units), with each key's state per
label x combo: units > 0, zero units (admitted, 0), or absent (not in that label x combo's pools). Same inputs as count_e12_identical.py."""
import json, sys, hashlib, collections
blk_p, man_p = sys.argv[1:3]
for p in (blk_p, man_p):
    print(p.rsplit("/", 1)[-1], hashlib.sha256(open(p, "rb").read()).hexdigest()[:16])
B = json.load(open(blk_p))["pools"]
where = collections.defaultdict(set)
for o, pools in B.items():
    for nm, v in pools.items():
        subs = [(nm, v)] if isinstance(v, dict) and "e2" in v else [(f"{nm} / {g}", x) for g, x in (v.items() if isinstance(v, dict) else []) if isinstance(x, dict) and "e2" in x]
        for lab, x in subs:
            for e in x["e2"]:
                where[(e["key_e2"], e["replicates"])].add((o, lab))
M = json.load(open(man_p))["per_label_combo"]
def st(eu, k):
    if k not in eu:
        return "absent"
    return "units" if any((n or 0) > 0 for n in eu[k].values()) else "zero(" + ",".join(f"{r}" for r in sorted(eu[k])) + ")"
for (k2, k1), sps in sorted(where.items()):
    states = {lc: (st(v["sizes"]["eligible_units_by_run"], k2), st(v["sizes"]["eligible_units_by_run"], k1)) for lc, v in M.items()}
    if any(a == "units" and b == "units" for a, b in states.values()):
        continue
    print(f"\npair: key_e2 {k2} | replicates {k1}")
    print("  pools:", sorted(sps))
    by = collections.defaultdict(list)
    for lc, s in states.items():
        by[s].append(lc)
    for s, lcs in sorted(by.items()):
        print(f"  key_e2 {s[0]:<12} replicate {s[1]:<12} in {len(lcs)} label x combos: {sorted(lcs)}")
