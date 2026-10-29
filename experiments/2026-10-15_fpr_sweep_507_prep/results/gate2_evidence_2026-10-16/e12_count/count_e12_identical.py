"""Keys and counts only: declared E12 pairs (key_e2, replicates) in the E12 block by set and pool, and which of them appear as byte-identical
groups in the held catalogue attempt #3 manifest (sizes.content_identical_keys per label x combo). Prints counts and pool names, no values."""
import json, sys, hashlib, collections
blk_p, man_p = sys.argv[1:3]
for p in (blk_p, man_p):
    print(p.rsplit("/", 1)[-1], hashlib.sha256(open(p, "rb").read()).hexdigest()[:16])
B = json.load(open(blk_p))["pools"]
where = collections.defaultdict(set)  # pair -> {(set, pool)}
for o, pools in B.items():
    for nm, v in pools.items():
        subs = [(nm, v)] if isinstance(v, dict) and "e2" in v else [(f"{nm} / {g}", x) for g, x in (v.items() if isinstance(v, dict) else []) if isinstance(x, dict) and "e2" in x]
        for lab, x in subs:
            for e in x["e2"]:
                where[frozenset((e["key_e2"], e["replicates"]))].add((o, lab))
print("\n(2) declared E12 pairs in the block: distinct", len(where))
tot = collections.Counter(sp for sps in where.values() for sp in sps)
for sp, n in sorted(tot.items()):
    print(f"   {sp[0]:<11} {sp[1]:<70} {n}")
M = json.load(open(man_p))["per_label_combo"]
groups = collections.defaultdict(set)  # group -> label x combos
for lc, v in M.items():
    for g in v["sizes"]["content_identical_keys"]:
        groups[frozenset(g)].add(lc)
e12g = {g: l for g, l in groups.items() if len(g) == 2 and g in where}
other = {g: l for g, l in groups.items() if g not in e12g}
print(f"\n(1) byte-identical groups in the manifest: distinct {len(groups)} over {len(M)} label x combos; declared E12 pairs among them: {len(e12g)}; other groups: {len(other)}")
c1 = collections.Counter(sp for g in e12g for sp in where[g])
for sp, n in sorted(c1.items()):
    print(f"   {sp[0]:<11} {sp[1]:<70} {n} of {tot[sp]}")
lab_n = collections.Counter(len(l) for l in e12g.values())
print("   label x combos per identical E12 pair (how many of the", len(M), "list it):", dict(sorted(lab_n.items())))
print("   identical E12 pairs by world prefix of key_e2:", dict(collections.Counter(next(k for k in g if "_e2|" in k).split("|")[0] for g in e12g)))
print("\n   other groups (keys):")
for g, l in sorted(other.items(), key=lambda x: sorted(x[0])):
    print("   ", sorted(g), f"in {len(l)} label x combos")

# declared pairs CONTAINED in any identical group (a pair inside a larger group counts too), and how many pairs were comparable at all
import itertools
inside = collections.defaultdict(set)
for g, l in groups.items():
    for a, b in itertools.combinations(sorted(g), 2):
        if frozenset((a, b)) in where:
            inside[frozenset((a, b))] |= l
print(f"\n(1') declared E12 pairs whose two keys sit in one byte-identical group (any size): {len(inside)}")
for p, l in inside.items():
    print("   ", sorted(p), "in", len(l), "label x combos;", sorted(where[p]))
comp = collections.defaultdict(set)  # pair -> label x combos where BOTH keys have units (in any role): compared by gate 2
for lc, v in M.items():
    eu = v["sizes"]["eligible_units_by_run"]
    has = {k for k, rs in eu.items() if any((n or 0) > 0 for n in rs.values())}
    for p in where:
        if p <= has:
            comp[p].add(lc)
cc = collections.Counter(sp for p in comp for sp in where[p])
print(f"\n(2') declared pairs COMPARED by gate 2 (both keys with units in at least one label x combo): {len(comp)} of {len(where)}")
for sp, n in sorted(cc.items()):
    print(f"   {sp[0]:<11} {sp[1]:<70} {n} of {tot[sp]}")
