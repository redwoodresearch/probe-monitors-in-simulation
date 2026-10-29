"""By-key checks of the simh_t196fill and ilnm_s4rgap DT records (keys and counts only): each folder's manifest_dt.json (read-only from the
work Volume) holds exactly the routing's staged keys (manifest minus staging exclusions); t196fill: world t196fill, stratum simh_t196fill,
T196 cells on a held-out fold 0-4; s4rgap: world sim2, stratum S_sim2gap, fold -1, nm145/nm242 absent; rows = call sum; record git = manifest
git; gate (a) passed. Usage: python3 -I check_t196fill_s4rgap.py <wt> <t196fill|s4rgap> <launch dir>"""
import json, subprocess, sys
from pathlib import Path
WT, W, LD = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
P2 = Path(__file__).resolve().parent.parent
MODAL, WVOL = "/Users/aswinsurya/.local/bin/modal", "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14"
RT = Path("/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad/chain_run_state/routing")
cfg = {"t196fill": ("simh_t196fill", "simh_t196fill_v1__sim_honest", "simh/simh_t196fill/dt/simh_t196fill_v1__sim_honest", "simh-t196fill__simh_t196fill.json", "t196fill", "simh_t196fill", 8),
       "s4rgap": ("ilnm_s4rgap_sim2", "ilnm_s4rgap_sim2_v1__sim_attack", "scaleup/ilnm_s4rgap_sim2_v1__sim_attack", "il-s4rgap__ilnm_s4rgap_sim2.json", "sim2", "S_sim2gap", 57)}[W]
t, folder, vpath, rtf, world, stratum, n = cfg
recs = sorted((WT / "experiments/2026-10-14_p2_dt_scaleup/results").glob(f"dt_q4b_arm_{t}_2*.json"))
assert len(recs) == 1, recs
d = json.loads(recs[0].read_text())
dst = P2 / "dt4f" / f"{folder}.manifest_dt.json"
subprocess.run([MODAL, "volume", "get", "--force", WVOL, f"{vpath}/manifest_dt.json", str(dst)], check=True, capture_output=True)
m = json.loads(dst.read_text())
r = json.loads((RT / rtf).read_text())
files = json.loads((LD / f"manifest_{t}.json").read_text())["files"]
excl, arm = r.get("staging_exclusions", {}), r["arms"][0]["meta_arm"]
staged = {f"{f['world']}|{f['task']}" for f in files if f["arm_dir"] == arm and f"{f['launch']}/{f['rel']}#e0" not in excl}
runs = m["runs"]
checks = {"runs_equal_staged": set(runs) == staged and len(runs) == n, "world": all(k.startswith(f"{world}|") for k in runs),
          "stratum": all(v["stratum"] == stratum for v in runs.values()),
          "folds": all((v["fold"] in range(5)) if W == "t196fill" else v["fold"] == -1 for v in runs.values()),
          "exclusions_absent": not any(k.split("|")[1].split("_")[0] in ("nm145", "nm242") for k in runs) if W == "s4rgap" else True,
          "rows_equal_call_sum": m["rows"] == sum(v["calls"] for v in runs.values()), "git": m["git"] == d["git"], "gate_a": bool(d.get("gate_a", {}).get("pass")),
          "record_folder": list(d["folders"]) == [folder]}
print(recs[0].name, "runs", len(runs), "rows", m["rows"], "zero-call", sum(v["calls"] == 0 for v in runs.values()))
print("ALL OK" if all(checks.values()) else f"FAIL {[k for k, v in checks.items() if not v]}")
