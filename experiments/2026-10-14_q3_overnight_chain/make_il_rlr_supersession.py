"""Chain-side supersession record for one IL paired-rerun family (09 Append 32 @ 9fd14891a, Append 35 B; d9 10-15 ~04:25), written before
its roots are routed. Keys and hashes only. It checks the design fork's supersession rows (supersession_rows.py output, relayed
beside the run as `<run>.supersession_rows.json`) against three independent sources and records the result per key:
- the pinned replacement list (config/ilrr_rerun_of_<h>.json @ f368ebab, or an append-only revision such as the rlr2 one):
  the same cells, and each sim2 original's failed .eval sha256;
- the chain's own manifests of the ORIGINAL IL family (every snapshot and the final): each original arm's .eval sha256;
- the chain's own staged manifests of the RERUN family: each replacement arm's .eval sha256 (a replacement without an .eval is
  recorded as such; it is still authoritative, with no fallback).
Every listed cell must have exactly one record per arm (run3, sim2), all authoritative "replacement"; a missing, extra or duplicate
record, or any hash mismatch, is an ERROR (exit 2).
Run: python make_il_rlr_supersession.py --rows <rows json> --pinned <commit:path> --orig-prefix il311_q2 --rerun-prefix il311_q2r --out <json>"""
import argparse
import glob
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def canon(launch_id):
    """09's launch_id comparison rule @ c065957e2: remove exactly one literal leading `runs/`, nothing else; then compare bytes."""
    return launch_id[len("runs/"):] if isinstance(launch_id, str) and launch_id.startswith("runs/") else launch_id


def manifest_rows(pattern: str) -> dict:
    docs = {p: json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(HERE / "config" / pattern)))}
    gone = {d.get("supersedes") for d in docs.values()} - {None}  # a later version (e.g. _v2, bare launch ids) replaces the file it names
    out = {}
    for p, d in docs.items():
        if Path(p).name in gone:
            continue
        for f in d["files"]:
            if f["arm_dir"] in ("attack", "sim_attack@september"):
                arm = "run3" if f["arm_dir"] == "attack" else "sim2"
                out.setdefault((arm, f["task"]), []).append({"sha256": f["sha256"], "path": f["path"], "manifest": Path(p).name, "launch_id": f.get("launch_id")})
    return out


def main():
    ap = argparse.ArgumentParser()
    for x in ("--rows", "--pinned", "--orig-prefix", "--rerun-prefix", "--out"):
        ap.add_argument(x, required=True)
    ap.add_argument("--revision", default="", help="commit:path of an append-only revision that moves some keys of THIS launch to a later launch")
    a = ap.parse_args()
    errors = []
    rows_raw = Path(a.rows).read_bytes()
    rows = json.loads(rows_raw)
    commit, path = a.pinned.split(":", 1)
    pin_raw = subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, check=True, cwd=HERE).stdout
    pinned = {x["cell"]: x for x in json.loads(pin_raw)["rerun_of"]}
    orig = manifest_rows(f"manifest_{a.orig_prefix}_run3*.json") | manifest_rows(f"manifest_{a.orig_prefix}_sim2*.json")
    repl = manifest_rows(f"manifest_{a.rerun_prefix}_run3*.json") | manifest_rows(f"manifest_{a.rerun_prefix}_sim2*.json")
    moved, rev_info = {}, None
    if a.revision:  # e.g. old's rlr3 revision: keys whose designation in this launch is retired (never started) and moved on
        rc, rp = a.revision.split(":", 1)
        rev_raw = subprocess.run(["git", "show", f"{rc}:{rp}"], capture_output=True, check=True, cwd=HERE).stdout
        for x in json.loads(rev_raw)["rows"]:
            if canon(x["retired_designation"]["launch"]) == canon(rows.get("rerun_launch")):
                moved[(x["arm"], x["cell"])] = x["authoritative"]
        rev_info = {"commit": rc, "path": rp, "sha256": hashlib.sha256(rev_raw).hexdigest(), "moved_keys": len(moved)}
    seen, out = {}, []
    for r in rows["records"]:
        k = (r["arm"], r["cell"])
        if k in seen:
            errors.append(f"duplicate record {r['key']}")
        seen[k] = r
        if r.get("authoritative") != "replacement":
            errors.append(f"{r['key']}: authoritative is {r.get('authoritative')}")
        o_sh = {e["sha256"] for e in r["original"]["evals"]}
        o_man = orig.get(k, [])
        # exact identity: at most one .eval per run, and the rows' hash set equals the chain's manifest hash set
        o_ok = len(r["original"]["evals"]) <= 1 and len(o_man) <= 1 and o_sh == {m["sha256"] for m in o_man}
        if not o_ok:
            errors.append(f"{r['key']}: original .eval hashes {len(o_sh)} differ from the chain's original manifests ({len(o_man)} rows)")
        if r["arm"] == "sim2":  # the pinned list against the ACTUAL original (rows and chain manifest), not only the rows' copied field
            p = pinned.get(r["cell"])
            if (p is None or p["run_key"] != r["original"]["run_key"] or o_sh != {p["eval_sha256"]} or {m["sha256"] for m in o_man} != {p["eval_sha256"]}
                    or r["original"].get("failed_eval_sha256") != p["eval_sha256"]):
                errors.append(f"{r['key']}: the pinned failed sim2 (run key, .eval sha) does not match the original's actual .eval")
        rp_sh = {e["sha256"] for e in r["replacement"]["evals"]}
        rp_man = repl.get(k, [])
        rp_ok = len(r["replacement"]["evals"]) <= 1 and len(rp_man) <= 1 and rp_sh == {m["sha256"] for m in rp_man}  # no .eval: none staged either
        if not rp_ok:
            errors.append(f"{r['key']}: replacement .eval sha does not match the chain's rerun manifest ({len(rp_man)} rows)")
        for side, man in (("original", o_man), ("replacement", rp_man)):  # 09 @ c065957e2: canonical launch_id equality with the chain's manifests
            if any(canon(m["launch_id"]) != canon(r[side]["launch_id"]) for m in man):
                errors.append(f"{r['key']}: {side} launch_id differs from the chain's manifest (canonical comparison)")
        mv = moved.get(k)
        if mv is not None and (r["replacement"]["state"] == "done" or rp_sh or rp_man):
            errors.append(f"{r['key']}: moved to {mv['launch']} by the revision, but this launch has a run or .eval for it")
        out.append({"key": r["key"], "cell": r["cell"], "arm": r["arm"],
                    "authoritative": "replacement" if mv is None else f"moved to {mv['launch']} ({mv['run_key']}) by the revision; this launch's run is retired",
                    "rerun_reason": r.get("rerun_reason"),
                    "original": {"launch_id": r["original"]["launch_id"], "run_key": r["original"]["run_key"], "state": r["original"]["state"],
                                 "eval_sha256": sorted(o_sh), "chain_manifest": [m["manifest"] for m in o_man]},
                    "replacement": {"launch_id": r["replacement"]["launch_id"], "run_key": r["replacement"]["run_key"], "state": r["replacement"]["state"],
                                    "eval_sha256": sorted(rp_sh), "chain_manifest": [m["manifest"] for m in rp_man]},
                    "replacement_state_done": r.get("replacement_state_done"), "no_fallback": True})
    want = {(arm, c) for c in pinned for arm in ("run3", "sim2")}
    if set(seen) != want:
        errors.append(f"records cover {len(seen)} (arm, cell); the pinned list needs {len(want)}: missing {sorted(want - set(seen))[:6]}, extra {sorted(set(seen) - want)[:6]}")
    doc = {"rule": "09 Append 32 @ 9fd14891a / Append 35 B (paired rate-limit reruns): both replacement arms authoritative, no fallback",
           "rows": {"path": a.rows, "sha256": hashlib.sha256(rows_raw).hexdigest(), "rerun_launch": rows.get("rerun_launch"), "original_launch": rows.get("original_launch")},
           "pinned": {"commit": commit, "path": path, "sha256": hashlib.sha256(pin_raw).hexdigest(), "cells": len(pinned)}, "revision": rev_info,
           "original_family": a.orig_prefix, "rerun_family": a.rerun_prefix, "records": sorted(out, key=lambda x: x["key"]),
           "counts": {"cells": len(pinned), "records": len(out), "replacement_done": sum(bool(x["replacement_state_done"]) for x in out)},
           "errors": errors, "pass": not errors}
    Path(a.out).open("x").write(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: doc[k] for k in ("counts", "errors", "pass")}, indent=1))
    if errors:
        sys.exit(2)


if __name__ == "__main__":
    main()
