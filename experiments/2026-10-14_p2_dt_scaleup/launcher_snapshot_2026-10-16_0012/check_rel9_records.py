"""After rel9 launches: by-key checks of every rel9 record (keys and counts only). For each arm record (sim-honest or IL q4c) and each of its
folders: the folder's manifest_dt.json (fetched read-only from the work Volume into dt4f/rel9_manifests/) has runs exactly equal to the routing
manifest's staged rows (rows of the routed arm minus d9's staging exclusions), rows = the call sum, zero-call runs listed, folds as routed
(fold -1 for +311/nm360 mains), gate (a) passing and the record's git. primary_v3: the two folders' runs equal the counted keys split 145 / 4,
folds 0..4, unmapped runs listed. Usage: python3 -I check_rel9_records.py <wt> <out>"""
import json
import re
import subprocess
import sys
from pathlib import Path

WT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
P2 = Path(__file__).resolve().parents[1]
RT = Path("/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad/chain_run_state/routing")
MODAL, WVOL = "/Users/aswinsurya/.local/bin/modal", "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14"
MD = P2 / "dt4f" / "rel9_manifests"
MD.mkdir(exist_ok=True)
git = lambda *a: subprocess.run(["git", "-C", str(WT), *a], capture_output=True, check=True).stdout  # noqa: E731
routes = {}
for f in RT.glob("*__*.json"):
    r = json.loads(f.read_text())
    routes[r.get("staged_set")] = r
res = {}
for rf in sorted((WT / "experiments/2026-10-14_p2_dt_scaleup/results").glob("dt_q4b_*.json")):
    r = json.loads(rf.read_text())
    if r.get("job") not in ("arm", "simh_primary") or not (r.get("message", "").startswith(("simh", "primary_v3", "il311_q4c"))):
        continue
    for g, v in r["folders"].items():
        if r["job"] == "simh_primary":
            vol = f"simh/primary_v3/dt/{g}/manifest_dt.json"
        elif g.startswith("simh"):
            vol = f"simh/{re.match(r'^(simh311_q[1-4]|simh_nm360_(?:s1|s2|s3|s4d|s4r|s5))_sim', g)[1]}/dt/{g}/manifest_dt.json"
        else:
            vol = f"scaleup/{g}/manifest_dt.json"
        dst = MD / f"{g}.json"
        if not dst.exists():
            subprocess.run([MODAL, "volume", "get", WVOL, vol, str(dst)], capture_output=True, check=True)
        m = json.loads(dst.read_text())
        runs = m["runs"]
        out = {"record": rf.name, "git": r["git"][:9], "gate_a": r["gate_a"]["pass"], "folder": vol.rsplit("/", 1)[0], "runs": len(runs),
               "rows": m["rows"], "rows_equal_call_sum": m["rows"] == sum(x["calls"] for x in runs.values()),
               "zero_call_runs": sorted(j for j, x in runs.items() if x["calls"] == 0), "folds": sorted({x["fold"] for x in runs.values()})}
        if r["job"] == "simh_primary":
            ck = json.loads(git("show", "572f67dbb:experiments/2026-10-14_q3_q35_simhonest_purchase/results/simh_counted_keys_primary_v3.json"))
            out["runs_n_expected"] = {"simh_full_b1": 145, "simh_full_b2": 4}[g]
            out["keys_subset_of_counted"] = all(j.split("|")[1] in {k["task"] for k in ck["keys"]} and j.startswith("simh_full|") for j in runs)
            out["unmapped"] = sorted(m.get("unmapped_runs", {}))
            out["ok"] = out["gate_a"] and out["runs"] == out["runs_n_expected"] and out["keys_subset_of_counted"] and out["rows_equal_call_sum"] \
                and all(f in range(5) for f in out["folds"])
        else:
            ss = g.split("__")[0]
            rt = routes[ss]
            p_, c_ = (y.strip() for y in rt["manifest"].split(" @ "))
            files = json.loads(git("show", f"{c_}:{p_}"))["files"]
            arms = {a["meta_arm"] for a in rt["arms"]}
            excl = rt.get("staging_exclusions", {})
            want = {f"{f['world']}|{f['task']}" for f in files if f["arm_dir"] in arms and f"{f['launch']}/{f['rel']}#e0" not in excl}
            out["expected_staged"] = len(want)
            out["runs_equal_staged_by_key"] = set(runs) == want
            out["excluded_not_scored"] = not ({f"{f['world']}|{f['task']}" for f in files if f"{f['launch']}/{f['rel']}#e0" in excl} & set(runs))
            out["ok"] = out["gate_a"] and out["runs_equal_staged_by_key"] and out["excluded_not_scored"] and out["rows_equal_call_sum"] and out["folds"] == [-1]
        res[g] = out
OUT.write_text(json.dumps(res, indent=1))
for g, o in res.items():
    print(g, o["runs"], o.get("expected_staged", o.get("runs_n_expected")), o["rows"], "zero-call", len(o["zero_call_runs"]), "OK" if o["ok"] else "NOT OK")
print("ALL OK" if res and all(o["ok"] for o in res.values()) else "NOT ALL OK", len(res), "folders")
