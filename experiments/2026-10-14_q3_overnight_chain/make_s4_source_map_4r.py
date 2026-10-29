"""S4 source map, 4r part (d9 10-15 ~10:55; 09's §3 requires the two S4 mains' cell sets to be disjoint). Keys and SHA-256s only.
  - Every one of 4r's 61 registered cells (s4r_selection.json @ 1bbab93a2, RULE.md Amendment 8 of the new-mains launch) with the owning
    launch of each arm: honest_a, run1, honest_b, run2 from 2026-10-14_newmains_s4r and sim from 2026-10-14_newmains_s4r_fu (both launch.json
    SHA-256s; from probe_s4r_cells.py's read-only box-4 output).
  - Asserts: the 61 and the s4d part's 19 cells (s4_source_map_s4d_2026-10-15.json @ 30f547dbd) are disjoint and together make S4's 80; every
    4r cell has all five worlds in 4r.
  - Every 2026-10-14_newmains_s4 key in the four earlier scored roots (nm360_s4_main_v1, nm360_s4_main_s1_v1, nm360_s4_run2_v1,
    nm360_s4_run2_s1_v1: their staging manifests at the commits the chain's attestations pin) whose cell is not one of s4d's 19:
    "outside S4 source map (cell owned by 4r); not used", with its staging state (staged or excluded, from chain_stage_nm360_s4_*.json).
  python3 make_s4_source_map_4r.py <probe txt> <out json>"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
probe_p, dest = Path(sys.argv[1]), Path(sys.argv[2])
NM = "experiments/2026-10-14_q3_new_mains_launch"
CH = "experiments/2026-10-14_q3_overnight_chain"
show = lambda c, p: subprocess.run(["git", "show", f"{c}:{p}"], capture_output=True, check=True, cwd=HERE, timeout=60).stdout
sha = lambda b: hashlib.sha256(b).hexdigest()
lines = probe_p.read_text().splitlines()
probe = json.loads(next(l for l in reversed(lines) if l.startswith("{")))
sel_raw = show("1bbab93a2", f"{NM}/results/s4r_selection.json")
sel = json.loads(sel_raw)
r4 = set(sel["keys"])
s4d_raw = show("30f547dbd", f"{CH}/results/s4_source_map/s4_source_map_s4d_2026-10-15.json")
s4d = json.loads(s4d_raw)
d19 = {r["cell"] for r in s4d["rows"]}
assert len(r4) == 61 and len(d19) == 19, (len(r4), len(d19))
assert not (r4 & d19), sorted(r4 & d19)
assert len(r4 | d19) == sel["of_80"] == 80
cells = probe["cells"]
assert set(cells) == r4, sorted(set(cells) ^ r4)
WANT = {"honest_a": "2026-10-14_newmains_s4r", "run1": "2026-10-14_newmains_s4r", "honest_b": "2026-10-14_newmains_s4r",
        "run2": "2026-10-14_newmains_s4r", "sim": "2026-10-14_newmains_s4r_fu"}
for c, w in cells.items():
    assert {k: v["launch"] for k, v in w.items()} == WANT, (c, w)
rows = [{"cell": c, "owner": "4r",
         "arms": {w: cells[c][w] for w in ("honest_a", "run1", "honest_b", "run2", "sim")}} for c in sorted(r4)]
# the four earlier scored roots: manifests pinned by the chain's attestations
ROOTS = [("nm360_s4_main_v1", "47d88a8d0c326500cffe2f73ef6d4cfd34c7163d", "manifest_nm360_s4_main.json", "chain_stage_nm360_s4_main_2026-10-14_234202.json", False),
         ("nm360_s4_main_s1_v1", "fb3c5511ba6148c942db238e97047b8cd7a76197", "manifest_nm360_s4_main_s1.json", "chain_stage_nm360_s4_main_s1_2026-10-14_231801.json", False),
         ("nm360_s4_run2_v1", "47d88a8d0c326500cffe2f73ef6d4cfd34c7163d", "manifest_nm360_s4_run2.json", "chain_stage_nm360_s4_run2_2026-10-14_234337.json", True),
         ("nm360_s4_run2_s1_v1", "fb3c5511ba6148c942db238e97047b8cd7a76197", "manifest_nm360_s4_run2_s1.json", "chain_stage_nm360_s4_run2_s1_2026-10-14_231934.json", True)]
orphans, inputs, in_s4d = [], {}, 0
for root, commit, man, stage, run2 in ROOTS:
    mraw = show(commit, f"{CH}/config/{man}")
    st_raw = (HERE / "results" / stage).read_bytes()
    st = json.loads(st_raw)
    excl = {}
    for part, v in st.items():
        if isinstance(v, dict):
            excl |= {k.split("#")[0]: why for k, why in (v.get("c_gate") or {}).get("excluded", {}).items()}
    inputs[root] = {"manifest": f"{CH}/config/{man} @ {commit[:9]}", "manifest_sha256": sha(mraw), "staging_record": f"{CH}/results/{stage}",
                    "staging_record_sha256": sha(st_raw)}
    for f in json.loads(mraw)["files"]:
        assert f["launch"] == "2026-10-14_newmains_s4", f["launch"]
        assert f["world"] in (("run2", "honest_a", "honest_b") if run2 else ("run1", "honest_a", "honest_b")), (root, f["world"])
        cell = f["task"][:-3] if f["task"].endswith("_r2") else f["task"]
        if cell in d19:
            in_s4d += 1
            continue
        assert cell in r4, cell  # every non-s4d cell of S4 is one of 4r's 61
        world = f["world"]  # Codex r54: the manifest's own world (run2 manifests also carry honest donors)
        target = world == "run2" if run2 else world in ("run1", "honest_a", "honest_b")
        orphans.append({"key": f"{world}|{cell}", "root": root, "cell": cell, "world": world, "launch": f["launch"], "record": f["path"],
                        "sha256": f["sha256"],
                        "staging": ("excluded" if f["path"] in excl else "staged") if target else "donor (not a staging target of this root)",
                        "staging_reason": excl.get(f["path"]),
                        "status": "outside S4 source map (cell owned by 4r); not used",
                        "cites": "4r's registration: new-mains RULE.md Amendment 8 (make_s4r.py; s4r_selection.json @ 1bbab93a2, 61 cells) and "
                                 "Amendment 9 (armed by Aswin's decision); 09's Append 21c amendment item 1 @ 0b234d283; s4d map @ 30f547dbd"})
doc = {"part": "4r", "rule": "09's §3 (bffe22e3e) and Append 21c amendment item 1 (0b234d283): S4 = union by cell of 4r (61) and s4d (19), disjoint",
       "launch_json_sha256": probe["launch_json_sha256"] | {"2026-10-14_newmains_s4": s4d["launch_json_sha256"]["2026-10-14_newmains_s4"]},
       "four_r": {"selection": {"commit": "1bbab93a2", "path": f"{NM}/results/s4r_selection.json", "sha256": sha(sel_raw), "cells": len(r4),
                                 "reasons_counts": sel.get("reasons")}},
       "s4d_part": {"commit": "30f547dbd", "path": f"{CH}/results/s4_source_map/s4_source_map_s4d_2026-10-15.json", "sha256": sha(s4d_raw), "cells": len(d19)},
       "probe": {"path": f"{CH}/results/s4_source_map/{probe_p.name}", "sha256": sha(probe_p.read_bytes()), "read": lines[0]},
       "checks": {"disjoint_4r_s4d": True, "union_is_80": True, "every_4r_cell_all_five_worlds_in_4r": True, "every_orphan_cell_in_4r": True},
       "cells": rows, "earlier_roots": inputs, "outside_source_map": orphans,
       "counts": {"cells_4r": len(rows), "earlier_root_keys_in_s4d_cells": in_s4d, "outside_keys": len(orphans),
                  "outside_keys_staged": sum(o["staging"] == "staged" for o in orphans),
                  "outside_keys_excluded": sum(o["staging"] == "excluded" for o in orphans),
                  "outside_keys_donor_only": sum(o["staging"].startswith("donor") for o in orphans),
                  "outside_cells": len({o["cell"] for o in orphans}),
                  "outside_cells_with_staged_keys": len({o["cell"] for o in orphans if o["staging"] == "staged"})}}
dest.open("x").write(json.dumps(doc, indent=1) + "\n")
print(json.dumps(doc["counts"]))
print("cells with staged keys outside the map:", sorted({o["cell"].split("_")[0] for o in orphans if o["staging"] == "staged"}))
print("sha256", sha(dest.read_bytes()))
