"""FINAL E12 / gate-2 count for 09 (keys and counts only; no score is read). Inputs: the E12 block (declared pairs = pools[*][*].e2 entries'
(key_e2, replicates)), and the held catalogue attempt 3's manifest and catalogue (stem ..._catalogue_e12_2026-10-16_174048). The header records
every input's sha256 and this script's own sha256. For EACH label x combo it reports whether gate 2 SCANNED it (pw_union writes
sizes.content_identical_keys for every label x combo it unions, after comparing every run WITH units; zero-unit runs are counted, not compared),
how many runs it compared, its identical groups (keys), the declared E12 pairs inside them, and how many declared pairs had both keys compared.
It reconciles '0 vs 1' (2-key groups that ARE a declared pair vs declared pairs INSIDE any group) and lists the declared pairs never compared,
with why (which of their main's worlds the catalogue holds at all).
usage: python3 count_e12_final.py <E12 block> <manifest> <catalogue>"""
import collections, hashlib, itertools, json, sys
from pathlib import Path

blk_p, man_p, cat_p = sys.argv[1:4]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
print("# E12 / gate-2 count, FINAL (keys and counts only)")
print(f"script {Path(__file__).name} sha256 {sha(__file__)}")
for n, p in (("e12_block", blk_p), ("manifest", man_p), ("catalogue", cat_p)):
    print(f"input {n} {Path(p).name} sha256 {sha(p)}")
B, M, CT = json.loads(Path(blk_p).read_bytes())["pools"], json.loads(Path(man_p).read_bytes())["per_label_combo"], json.loads(Path(cat_p).read_bytes())["runs"]
where = collections.defaultdict(set)  # declared pair -> {(set, pool)}
for o, pools in B.items():
    for nm, v in pools.items():
        subs = [(nm, v)] if isinstance(v, dict) and "e2" in v else [(f"{nm} / {g}", x) for g, x in (v.items() if isinstance(v, dict) else []) if isinstance(x, dict) and "e2" in x]
        for lab, x in subs:
            for e in x["e2"]:
                where[frozenset((e["key_e2"], e["replicates"]))].add((o, lab))
print(f"\ndeclared E12 pairs (distinct): {len(where)}")
print("\n## per label x combo (scanned = sizes.content_identical_keys present; compared = runs with units; zero-unit runs not compared)")
print("label x combo | scanned | runs compared | zero-unit not compared | S_newmains-only keys | groups | declared pairs inside groups | declared pairs compared")
comp_any, inside_any, two_key_any = set(), collections.defaultdict(set), set()
no_group = []
for lc in sorted(M):
    sz = M[lc]["sizes"]
    eu = sz["eligible_units_by_run"]
    has = {k for k, rs in eu.items() if any((n or 0) > 0 for n in rs.values())}
    nm_only = sum(1 for k in eu if CT.get(k, {}).get("sets") == ["S_newmains"])
    grps = [frozenset(g) for g in sz.get("content_identical_keys", [])]
    ins = {p for g in grps for p in map(frozenset, itertools.combinations(sorted(g), 2)) if p in where}
    for g in grps:
        if len(g) == 2 and g in where:
            two_key_any.add(g)
    for p in ins:
        inside_any[p].add(lc)
    cp = {p for p in where if p <= has}
    comp_any |= cp
    if not grps:
        no_group.append(lc)
    print(f"{lc} | {'content_identical_keys' in sz} | {len(has)} | {sz.get('gate2_zero_unit_keys_not_compared')} | {nm_only} | {len(grps)} "
          f"{[sorted(g) for g in grps]} | {[sorted(p) for p in ins]} | {len(cp)}")
print(f"\nlabel x combos with NO identical group: {len(no_group)}: {no_group}")
print("  (each was scanned; see 'scanned' and 'runs compared' above)")
print("\n## reconcile '0 vs 1'")
print(f"(i) 2-key identical groups that ARE exactly one declared pair (count_out.txt section (1)): {len(two_key_any)}")
print(f"(ii) declared pairs whose two keys sit INSIDE any identical group (count_out2.txt section (1')): {len(inside_any)}: "
      + "; ".join(f"{sorted(p)} in {len(l)} label x combos, pools {sorted(where[p])}" for p, l in inside_any.items()))
print(f"\ndeclared pairs compared in at least one label x combo: {len(comp_any)} of {len(where)}")
nc = sorted((sorted(p), sorted(where[p])) for p in where if p not in comp_any)
print(f"declared pairs never compared: {len(nc)}")
for keys, pools in nc:
    mains = {k.split("|", 1)[1] for k in keys}
    held = sorted(k for k in CT if k.split("|", 1)[1] in mains)
    print(f"  {keys} pools {pools}; catalogue keys on this main: {held} "
          f"(sim-honest key present: {any(k.split('|', 1)[0].startswith(('simh', 'sp493', 't196fill')) for k in held)}; "
          f"honest_a key present: {any(k.split('|', 1)[0].startswith('honest_a') for k in held)})")
