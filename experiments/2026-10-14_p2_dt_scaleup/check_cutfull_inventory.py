"""By-key check of the CUT fork's (a'') inventory against DT (names and keys only; no compute):
for each inventory run: its cutfull folder (uncut_planned.folder) passes 09's grammar (a) no _cutfull_gate_, (b) the cutfull regex,
(c) the derived <target>_v1__<arm> is accepted by DT's registered grammar (FOLDER for su311, SIMH simh311 folder regex), and the (a'') scope
(su311 + simh311 only); its original folder is a DT-scored registered folder whose manifest_dt.json holds the run's (world|task) key;
the original folder's staged key set equals the inventory's runs for that folder (one-to-one); the gate folder is never a scoring target.
Usage: python3 -I check_cutfull_inventory.py <wt> <out json> <manifest dir> [<manifest dir> ...]"""
import collections
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

WT, OUT, MDIRS = Path(sys.argv[1]), Path(sys.argv[2]), [Path(x) for x in sys.argv[3:]]
INV = ("b8a39f329", "experiments/2026-10-15_cut_sensitivity/results/inventory_v3_2026-10-15.json")
raw = subprocess.run(["git", "-C", str(WT), "show", f"{INV[0]}:{INV[1]}"], capture_output=True, check=True).stdout
inv = json.loads(raw)
src = (WT / "experiments/2026-10-14_p2_dt_scaleup/run_dt_q4b.py").read_text()
FOLDER = re.compile(eval(re.search(r'^FOLDER = re\.compile\((r"[^"]*")\)', src, re.M)[1]))
SIMH311 = re.compile(r"^simh311_q[1-4]_sim(?:_s[0-9]+|_rlr(?:[2-9]|[1-9][0-9]+)?)?_v1(?:_try[0-9]+)?__sim_honest$")  # DT's SIMH["simh311"] folder regex
CUT = re.compile(r"^([a-z0-9_]+?)_cutfull_v1(?:_try[0-9]+)?__([A-Za-z0-9_@.]+)$")  # 09 @ bfc8742a6 (b)
assert SIMH311.pattern in src, "SIMH simh311 regex differs from the job's"
man = {}
for d in MDIRS:
    for f in d.glob("*.json"):
        if "__" in f.stem:
            man[f.stem.replace(".manifest_dt", "")] = f
fails, per_folder = collections.defaultdict(list), collections.defaultdict(lambda: {"inventory_keys": set(), "cutfull": set(), "gate": set()})
for r in inv["runs"]:
    up, of = r["uncut_planned"], r.get("original_folder") or r.get("cut", {}).get("original_folder")
    cf, gf = up["folder"], up.get("gate_folder", "")
    m = CUT.fullmatch(cf)
    derived = f"{m[1]}_v1__{m[2]}" if m else None
    reg = FOLDER.fullmatch(derived or "") if derived else None
    ok_c = bool(derived) and ((reg is not None and reg[1] == "su311") or SIMH311.fullmatch(derived) is not None)
    if "_cutfull_gate_" in cf or not m:
        fails["09_a_or_b"].append(cf)
    if not ok_c:
        fails["09_c_or_scope"].append(cf)
    if "_cutfull_gate_" not in gf:
        fails["gate_folder_not_a_gate_name"].append(gf)
    if of is None or not re.fullmatch(rf"{re.escape(m[1]) if m else '#'}_v1(?:_try[0-9]+)?__{re.escape(m[2]) if m else '#'}", of or ""):
        fails["original_folder_not_the_target's"].append((cf, of))
    key = f"{r['world']}|{r['task']}"
    per_folder[of]["inventory_keys"].add(key)
    per_folder[of]["cutfull"].add(cf)
    per_folder[of]["gate"].add(gf)
for of, v in sorted(per_folder.items()):
    f = man.get(of)
    if f is None:
        fails["original_folder_without_a_DT_manifest"].append(of)
        continue
    dk = set(json.loads(f.read_text())["runs"])
    dt_tasks, inv_tasks = {k.split("|", 1)[1] for k in dk}, {k.split("|", 1)[1] for k in v["inventory_keys"]}  # the inventory names sim-honest's world by arm
    if dt_tasks != inv_tasks or len(dt_tasks) != len(dk) or len(inv_tasks) != len(v["inventory_keys"]):
        fails["original_keys_differ_from_inventory"].append({"folder": of, "dt_only": len(dt_tasks - inv_tasks), "inventory_only": len(inv_tasks - dt_tasks)})
    if len(v["cutfull"]) != 1:
        fails["several_cutfull_folders_for_one_original"].append(of)
res = {"what": "the CUT fork's (a'') inventory v3 against DT, by key (names and keys only); PRELIMINARY",
       "written_pt": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S"),
       "git": subprocess.run(["git", "-C", str(WT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
       "inventory": {"commit": INV[0], "path": INV[1], "sha256": hashlib.sha256(raw).hexdigest(), "n_runs": len(inv["runs"]), "original_folders": len(per_folder)},
       "checks": ["09 (a)(b) on uncut_planned.folder", "09 (c): derived <target>_v1__<arm> accepted by DT's FOLDER (su311) or SIMH simh311 regex; (a'') scope su311 + simh311",
                  "gate_folder carries _cutfull_gate_ (never a scoring target)", "original_folder is the target's registered folder (the original's _try allowed)",
                  "original_folder has a DT manifest_dt.json", "DT's scored tasks of the original folder == the inventory's tasks for it, one run per task each (one-to-one; one world per folder)",
                  "exactly one cutfull folder per original folder"],
       "fails": {k: v for k, v in fails.items()}, "pass": not fails}
OUT.write_text(json.dumps(res, indent=1, default=sorted))
print("inventory", res["inventory"]["sha256"][:16], "runs", res["inventory"]["n_runs"], "original folders", res["inventory"]["original_folders"])
print("PASS" if res["pass"] else f"FAIL: {[(k, len(v)) for k, v in fails.items()]}")
for k, v in fails.items():
    print(" ", k, v[:4])
