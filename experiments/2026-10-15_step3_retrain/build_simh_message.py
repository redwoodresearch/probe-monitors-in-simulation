"""Adapter (metadata only; no activation or score is read): build a pass-2-schema message for ONE routed sim-honest root from d9's
committed records, gate it, and write the message plus an adapter record that score_step3_simh_batch.py accepts as its committed
source record. Gates (all must pass): d9's attestations for the staged sha (every check "pass"); the staging record (staged set,
staged sha, c_gate counts and exclusions); the extraction record (model, inputs sha, every shard NVIDIA H200, shard count); the
repro-c comparison (no failing shard, every shard covered, passing == epochs, min cosine >= 0.9999); the format check (passes, same
staged sha); the manifest (the routed arm's files = staged + excluded, one per cell, all first attempts); the run-table snapshot
(same cells, sim_status ran). Epoch key = "<launch>/<rel>#e0" (make_pass2_message.py's rule).

  python3 -I build_simh_message.py --folder <d9 routed folder> --volume <vol> --root <extraction root> --staged-set <set> \\
      --staged-sha <sha256> --rec manifest=<commit>:<path> --rec staging=... --rec device=... --rec repro=... --rec format=... \\
      --rec attest=... --rec run_table=... --out-dir <dir>
"""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[2]
ARM = "sim_honest@september"


def main():
    ap = argparse.ArgumentParser()
    for a in ("folder", "volume", "root", "staged_set", "staged_sha", "out_dir"):
        ap.add_argument("--" + a.replace("_", "-"), required=True)
    ap.add_argument("--rec", action="append", required=True)
    ap.add_argument("--run-table-scope", choices=("exact", "superset", "toolerr"), default="exact",
                    help="exact: the run table is this snapshot's; superset: a FINAL run table over the whole batch (this root's cells must all have ran); "
                         "toolerr: A25 fill-in only, whose hygiene has no run table: d9's tool-error file (one row and one source .eval per cell) stands in")
    a = ap.parse_args()
    if "_cutfull_" in a.staged_set or "_cutfull_" in a.folder:  # d9's cutfull routing records differ (top-level original_folders, step-0 gate): fail closed
        raise SystemExit("cutfull sets are not built here: they need the cutfull adapter over d9's routing record (cutfull_routing_v1.json)")
    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, check=True).stdout  # noqa: E731
    pins, recs = {}, {}
    for r in a.rec:
        name, cp = r.split("=", 1)
        c, p = cp.split(":", 1)
        if not any(b.strip().startswith("origin/") for b in git("branch", "-r", "--contains", c).decode().splitlines()):
            raise SystemExit(f"{name}: commit {c} is not on origin")
        b = git("show", f"{c}:{p}")
        pins[name] = {"commit": git("rev-parse", c).decode().strip(), "path": p, "sha256": hashlib.sha256(b).hexdigest()}
        recs[name] = json.loads(b)
    need = {"manifest", "staging", "device", "repro", "format", "attest", "run_table"}
    if set(recs) != need:
        raise SystemExit(f"records needed: {sorted(need)}")
    gate = {}
    att = recs["attest"]["attestations"].get(a.staged_sha) or []
    gate["attestations_all_pass"] = {"checks": [x["check"] for x in att], "pass": bool(att) and all(str(x["pass"]) == "True" for x in att)}
    st = recs["staging"]["main"]
    excluded = st["c_gate"]["excluded"]
    gate["staging"] = {"staged": st["c_gate"]["staged"], "excluded": len(excluded),
                       "pass": st["out"] == a.staged_set and st["assembled"]["staged_inputs_sha256"] == {f"{ARM}.jsonl": a.staged_sha}
                       and st["assembled"]["epochs"] == {ARM: st["c_gate"]["staged"]} and st["c_gate"]["manifest_keys"] == st["c_gate"]["staged"] + len(excluded)}
    dv = recs["device"]
    gate["extraction_h200"] = {"shards": len(dv["shards"]), "pass": dv["config"]["run"] == a.root and dv["config"]["inputs"] == a.staged_set
                               and dv["config"]["inputs_sha256"].get(f"{ARM}.jsonl") == a.staged_sha and dv["config"]["model"] == "Qwen/Qwen3.5-27B"
                               and dv["volume"] == a.volume and len(dv["shards"]) == dv["config"]["shards"] and all(s.get("gpu_name") == "NVIDIA H200" for s in dv["shards"])}
    rp = recs["repro"]
    gate["repro_c"] = {"epochs": rp["epochs"], "pass": rp["run"] == a.root and rp["staged_sha"].get(f"{ARM}.jsonl") == a.staged_sha and not rp["failing_shards"]
                       and rp["passing"] == rp["epochs"] and rp["coverage"]["every_shard_one_of_each"] is True and rp["min_cosine"] >= rp["bar"]["cosine"]}
    fc = recs["format"]
    gate["format_check"] = {"pass": str(fc["passes"]) == "True" and fc["staged_sha"] == a.staged_sha and fc["arm"] == ARM}
    files = [f for f in recs["manifest"]["files"] if f["arm_dir"] == ARM]
    keys = {f"{f['launch']}/{f['rel']}#e0": f for f in files}
    cells = [f["task"] for f in files]
    gate["manifest"] = {"files": len(files), "pass": len(keys) == len(files) == st["c_gate"]["manifest_keys"] and len(set(cells)) == len(cells)
                        and set(excluded) <= set(keys) and all(f["attempt_kind"] == "first" for f in files)}
    rt = recs["run_table"]
    if a.run_table_scope == "toolerr":
        if not a.folder.startswith("simh_t196fill_"):
            raise SystemExit("--run-table-scope toolerr is for the A25 fill-in root only")
        rk = [r_["key"] for r_ in rt["rows"]]
        gate["run_table"] = {"cells": len(rk), "scope": "toolerr", "pass": rt["arm"] == ARM and rt["world"] == "t196fill" and rt["staged_sha"] == a.staged_sha
                             and sorted(rk) == sorted(f"t196fill|{c}" for c in cells) and len(rt["source_evals"]) == len(cells) == rt["totals"]["runs"]
                             and sorted(x["path"] for x in rt["source_evals"]) == sorted(f"{f['launch']}/{f['rel']}" for f in files)}  # one distinct .eval per manifest file
        rt = {"cells": [{"cell": c, "sim_status": "ran"} for c in cells], "staged_sha": a.staged_sha}  # the ran-check below then holds by construction
    rcells = {x["cell"]: x for x in rt["cells"]}
    exc_cells = {keys[k]["task"] for k in excluded}  # a staged cell must have run; a staging-excluded one may be "ran" or "excluded" (e.g. an _rlr replacement)
    st_ok = all(c in rcells and (rcells[c]["sim_status"] == "ran" or (c in exc_cells and rcells[c]["sim_status"] == "excluded")) for c in cells)
    if a.run_table_scope == "exact":
        rt_ok = set(rcells) == set(cells) and st_ok
    else:
        rt_ok = set(cells) <= set(rcells) and st_ok
    if a.run_table_scope != "toolerr":
        gate["run_table"] = {"cells": len(rcells), "scope": a.run_table_scope, "pass": rt_ok and json.dumps(rt["staged_sha"]).find(a.staged_sha) >= 0}
    if not all(g["pass"] for g in gate.values()):
        raise SystemExit(f"adapter gates failed: {json.dumps({k: g for k, g in gate.items() if not g['pass']})}")
    staged = f"{a.staged_set}/inputs/{ARM}.jsonl"
    world, stratum = ("simh_nm360", "S_newmains") if a.folder.startswith("simh_nm360_") else ("t196fill", "S_old") if a.folder.startswith("simh_t196fill_") \
        else ("simh_311", "S_new")
    lab = {"label": "A25 fill-in (POST HOC, generated after A25 results were read); registered A25 rows unchanged"} if world == "t196fill" else {}  # 09 @ 286b2f518 s4
    epochs = [{"key": k, "cell": f["task"], "world": world, "stratum": stratum, "run": a.root, "arm": ARM, "staged": staged, "staged_sha256": a.staged_sha,
               "folder": a.folder, "launch_task": f.get("launch_task")} | lab for k, f in sorted(keys.items()) if k not in excluded]
    exc = [{"key": k, "world": world, "cell": keys[k]["task"], "arm": ARM, "folder": a.folder, "reason": excluded[k]} for k in sorted(excluded)]
    msg = {"d9_arm": a.folder.split("__")[0], "volume": a.volume, "provenance": [{"kind": "simh", "run": a.root, "staged_set": a.staged_set,
           "device_record": {"commit": pins["device"]["commit"], "path": pins["device"]["path"]}, "staging_record": {"commit": pins["staging"]["commit"], "path": pins["staging"]["path"]},
           "excluded_by_staging": exc, "rows": len(epochs)}], "epochs": epochs}
    mb = json.dumps(msg, indent=1, sort_keys=True).encode()
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    mp = od / f"msg_{a.folder}.json"
    mp.open("xb").write(mb)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rec = {"label": "adapter record: d9's committed records -> pass-2 message for one routed sim-honest root (metadata only)", "stamp (PT)": stamp,
           "git": git("rev-parse", "HEAD").decode().strip(), "folder": a.folder, "message_path": str(mp.relative_to(REPO)) if mp.is_relative_to(REPO) else str(mp),
           "message_sha256": hashlib.sha256(mb).hexdigest(), "n_epochs": len(epochs), "excluded_by_staging": exc, "pins": pins, "gate": gate,
           "d9_stratum_label": recs["attest"].get("stratum")}
    rp_ = od / f"adapter_{a.folder}_{stamp}.json"
    rp_.open("x").write(json.dumps(rec, indent=1))
    print(rp_, rec["message_sha256"], rec["n_epochs"], len(exc), {k: g["pass"] for k, g in gate.items()})


if __name__ == "__main__":
    main()
