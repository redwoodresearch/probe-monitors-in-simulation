"""S4 orphan check (read-only, by key): every outside_source_map row of d9's S4 source map (4r part @ b68ab0a53) vs the S_newmains membership
(folder -> keys) that each nm360 record used. A row is used iff the membership assigns its key to the orphan's own root folder."""
import collections, json, subprocess, sys
REPO = sys.argv[1]
g = lambda c, p: subprocess.run(["git", "show", f"{c}:{p}"], capture_output=True, check=True, cwd=REPO).stdout
SM = json.loads(g("b68ab0a53", "experiments/2026-10-14_q3_overnight_chain/results/s4_source_map/s4_source_map_4r_2026-10-15.json"))
orph = SM["outside_source_map"]
staged = [r for r in orph if r["staging"] == "staged"]
print("outside rows", len(orph), "staged", len(staged), "staged cells", len({r["cell"] for r in staged}),
      "by world", dict(collections.Counter(r["world"] for r in staged)), "roots", sorted({r["root"] for r in staged}))
cells4r = {c["cell"]: c for c in SM["cells"]}
out = {}
for mc in ("0f9d96158", "f3e57dc98"):
    mem = json.loads(g(mc, "experiments/2026-10-15_fpr_sweep_507_prep/config/membership710.json"))
    import hashlib; sh = hashlib.sha256(g(mc, "experiments/2026-10-15_fpr_sweep_507_prep/config/membership710.json")).hexdigest()
    fk = mem["folder_keys"]
    owners = collections.defaultdict(list)
    for f, ks in fk.items():
        for k in ks:
            owners[k].append(f)
    multi = {k: v for k, v in owners.items() if len(v) > 1}
    used, absent, other = [], [], []
    for r in orph:
        own = owners.get(r["key"], [])
        is_orphan_root = [f for f in own if f.split("__")[0] == r["root"]]
        (used if is_orphan_root else absent if not own else other).append((r["key"], r["root"], r["staging"], own))
    # every 4r cell x world: owner must be a 4r folder (or absent = status row)
    bad4r = []
    for c, e in cells4r.items():
        for w in ("honest_a", "honest_b", "run1", "run2", "sim"):
            k = f"{w}|{c}"
            for f in owners.get(k, []):
                if "s4r" not in f:
                    bad4r.append((k, f))
    res = {"membership_commit": mc, "sha256": sh, "keys_in_two_folders": len(multi), "orphan_rows_used": len(used), "used": used[:60],
           "orphan_rows_key_absent": len(absent), "orphan_rows_key_owned_elsewhere": len(other),
           "owned_elsewhere_by_folder": dict(collections.Counter(o[0] for *_, o in other)), "cells_4r": len(cells4r), "cells_4r_with_non4r_owner": bad4r}
    print(json.dumps({k: v for k, v in res.items() if k != "used"}, indent=1))
    if used:
        print("USED:", json.dumps(used, indent=0)[:3000])
    out[mc] = res
json.dump({"source_map": "b68ab0a53", "staged": len(staged), "results": out}, open(sys.argv[2], "x"), indent=1)
