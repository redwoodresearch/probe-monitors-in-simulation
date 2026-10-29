"""d9's CUT fork: the all-names collision test for the (b) cutfull folders (names and keys only; local). Every planned cutfull routed
folder must have exactly ONE original routed folder that has runs (more is a collision, none an orphan). Per 09 (RULE @ 47565ecc2 §2), a registered name with 0 runs (no
row in the inventory, so no stage record and no staged set) is "registered name; 0 runs; not an original" and does not collide.

Inputs: 2e's inventory_b_v1 (git ref), naming addendum 2's 149 names. Writes results/cutfull_collisions_all_<stamp>.json; exits
nonzero on any collision or orphan.

  python3 experiments/2026-10-14_q3_overnight_chain/cutfull/cutfull_collisions_all.py <commit>:<inventory_b_v1 path>
"""
import collections
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
CHAIN = HERE.parent
REPO = CHAIN.parents[1]


def token(arm):
    return arm.split("@")[0]


def main():
    ref = sys.argv[1]
    raw = subprocess.run(["git", "show", ref], capture_output=True, check=True, cwd=REPO).stdout
    inv = json.loads(raw)
    a2 = json.loads((HERE / "cutfull_naming_v1_addendum_2.json").read_text())
    runs = collections.Counter()  # original routed folder -> runs naming it
    cut_of = collections.defaultdict(set)  # cutfull folder -> original folders
    for r in inv["runs"]:
        orig = f"{r['cut']['staged_set']}__{token(r['arm'])}"
        runs[orig] += 1
        up = r.get("uncut_planned")
        if isinstance(up, dict):
            cut_of[up["folder"]].add(orig)
        elif isinstance(up, str) and "collision" in up:
            cut_of["simh_nm360_s3_sim_cutfull_v1__sim_honest"].add(orig)
    for n in a2["names"]:
        cut_of[n["cutfull_folder"]].add(n["original_folder"])
    for c, origs in inv.get("collisions", {}).items():  # the inventory's listed collision partners (they may have no rows)
        cut_of[c] |= set(origs)
    zero = sorted({o for v in cut_of.values() for o in v if runs[o] == 0})
    coll = {c: sorted(o for o in v if runs[o] > 0) for c, v in cut_of.items() if len([o for o in v if runs[o] > 0]) > 1}
    orphan = sorted(c for c, v in cut_of.items() if not any(runs[o] > 0 for o in v))  # a cutfull folder with NO original that has runs
    evidence = {c: {o: runs[o] for o in sorted(v)} for c, v in sorted(cut_of.items())}  # every folder's originals with their row counts
    out = {"what": "all-names collision test for the (b) cutfull folders; a 0-run registered name is not an original (09, RULE @ 47565ecc2 §2)",
           "inventory": {"ref": ref, "sha256": hashlib.sha256(raw).hexdigest()}, "addendum_2_sha256": hashlib.sha256((HERE / "cutfull_naming_v1_addendum_2.json").read_bytes()).hexdigest(),
           "cutfull_folders": len(cut_of), "zero_run_registered_names": zero, "collisions": coll, "n_collisions": len(coll),
           "folders_without_an_original_with_runs": orphan, "originals_with_row_counts": evidence, "passes": not coll and not orphan}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dest = CHAIN / "results" / f"cutfull_collisions_all_{stamp}.json"
    dest.open("x").write(json.dumps(out, indent=1))
    print(dest.name, "PASS" if out["passes"] else "FAIL", len(cut_of), "folders,", len(coll), "collisions,", len(orphan), "without an original; zero-run names", zero)
    if not out["passes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
