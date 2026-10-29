"""Chain-side supersession record for one IL completion family (09 @ 92d21b8f6 / e46ba05d0; per-arm maps @ 54823ddbf), written
before its roots are routed. Keys and hashes only. It checks the design fork's filled per-arm rows (completion_rows.py output,
`<run>.supersession_rows.json`) against:
- the registered per-arm map (commit:path): the same 66-style key set, and every map field unchanged (cell, arm, key, tier, il_order,
  original, authoritative);
- the tier-3 map (commit:path): the rows' tier-3 cells are exactly its cells;
- the chain's manifests of the ORIGINAL IL family (every snapshot and the final, superseded versions skipped): a tier-2 key, and a
  tier-3 sim2 key, has no original run there ("not run", 0 attempts); a tier-3 run3 key has exactly one original .eval there,
  which the completion supersedes;
- the chain's staged manifests of the COMPLETION family: each replacement's .eval sha256 set equals the rows' (exact identity, at
  most one .eval), and its launch id equals the rows' and the authoritative launch (09's canonical comparison @ c065957e2).
Any mismatch is an ERROR (pass false).
Run: python make_ilc_supersession.py --rows <json> --map <commit:path> --tier3 <commit:path> --orig-prefix il311_q3 --repl-prefix il311_q3c --out <json>"""
import argparse
import glob
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from make_il_rlr_supersession import HERE, canon, manifest_rows

MAP_FIELDS = ("cell", "arm", "key", "tier", "il_order", "original", "authoritative")


def git_json(spec: str) -> tuple[bytes, dict]:
    commit, path = spec.split(":", 1)
    raw = subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, check=True, cwd=HERE).stdout
    return raw, json.loads(raw)


def main():
    ap = argparse.ArgumentParser()
    for x in ("--rows", "--map", "--tier3", "--orig-prefix", "--repl-prefix", "--out"):
        ap.add_argument(x, required=True)
    a = ap.parse_args()
    errors = []
    rows_raw = Path(a.rows).read_bytes()
    rows = json.loads(rows_raw)
    map_raw, pmap = git_json(a.map)
    t3_raw, t3 = git_json(a.tier3)
    want = {r["key"]: r for r in pmap["rows"]}
    orig = manifest_rows(f"manifest_{a.orig_prefix}_run3*.json") | manifest_rows(f"manifest_{a.orig_prefix}_sim2*.json")
    no_eval = set()  # (arm folder, cell) of the original family's counted attempts without an .eval: such a run is not "not run"
    for pat in (f"manifest_{a.orig_prefix}_run3*.json", f"manifest_{a.orig_prefix}_sim2*.json"):
        for p in glob.glob(str(HERE / "config" / pat)):
            for paths in json.loads(Path(p).read_text()).get("counted_attempts_without_eval", {}).values():
                for x in paths:
                    parts = x.split("/")
                    no_eval.add((parts[-2], parts[-1].split("_redraw")[0]))
    repl = manifest_rows(f"manifest_{a.repl_prefix}_run3*.json") | manifest_rows(f"manifest_{a.repl_prefix}_sim2*.json")
    seen, out = set(), []
    for r in rows["rows"]:
        k = r["key"]
        if k in seen:
            errors.append(f"duplicate row {k}")
        seen.add(k)
        m = want.get(k)
        if m is None:
            errors.append(f"{k}: not in the registered map")
            continue
        changed = [f for f in MAP_FIELDS if r.get(f) != m.get(f)]
        if changed:
            errors.append(f"{k}: map fields changed {changed}")
        mk = (r["arm"], r["cell"])
        o_man = orig.get(mk, [])
        arm_dir = "attack" if r["arm"] == "run3" else "sim_attack@september"
        if any(canon(x["launch_id"]) != canon(r["original"]["launch"]) for x in o_man):  # an artifact of another launch never counts as the original
            errors.append(f"{k}: an original manifest row is not from {r['original']['launch']} (canonical comparison)")
        if r["tier"] == "tier 3" and r["arm"] == "run3":  # the original ran: exactly one original .eval, which this key supersedes
            if r["original"]["state"] != "done" or len(o_man) != 1:
                errors.append(f"{k}: tier-3 run3 original should be done with one .eval in the chain's manifests ({r['original']['state']}, {len(o_man)} rows)")
        elif r["original"]["state"] != "not run" or r["original"].get("attempts") != 0 or o_man or (arm_dir, r["cell"]) in no_eval:  # tier 2, tier-3 sim2: no run
            errors.append(f"{k}: original should be 'not run' with 0 attempts and no chain manifest row ({r['original']}, {len(o_man)} rows)")
        rp = r.get("replacement") or {}
        rp_sh = {e["sha256"] for e in rp.get("evals") or []}
        # explicit accounting: a stated replacement state, the row's status equal to it, replacement_state_done consistent with it, and
        # an artifact-less replacement only with an explicit null eval_sha256 (registration allows it; it stays authoritative, no fallback)
        if not rp.get("state") or r.get("status") != rp["state"] or r.get("replacement_state_done") is not (rp["state"] == "done"):
            errors.append(f"{k}: replacement state not explicit or inconsistent (state {rp.get('state')}, status {r.get('status')}, done {r.get('replacement_state_done')})")
        if not rp_sh and ("eval_sha256" not in r or r["eval_sha256"] is not None):
            errors.append(f"{k}: a replacement without an .eval must say eval_sha256 null")
        rp_man = repl.get(mk, [])
        if not (len(rp.get("evals") or []) <= 1 and len(rp_man) <= 1 and rp_sh == {x["sha256"] for x in rp_man} and rp_sh == set(r.get("eval_sha256") or [])):
            errors.append(f"{k}: replacement .eval sha does not match the chain's completion manifest ({len(rp_man)} rows) or the row's eval_sha256")
        ids = {canon(rp.get("launch_id")), canon(r["authoritative"]["launch"]), canon(rows.get("completion_launch"))} | {canon(x["launch_id"]) for x in rp_man}
        if len(ids) != 1:
            errors.append(f"{k}: launch ids differ (canonical comparison): {sorted(map(str, ids))}")
        if rp.get("run_key") != r["authoritative"]["run_key"]:
            errors.append(f"{k}: replacement run key {rp.get('run_key')} != authoritative {r['authoritative']['run_key']}")
        out.append({"key": k, "cell": r["cell"], "arm": r["arm"], "tier": r["tier"], "authoritative": "replacement", "no_fallback": True,
                    "supersedes_original_run": bool(o_man),
                    "original": {"launch_id": r["original"]["launch"], "run_key": r["original"]["run_key"], "state": r["original"]["state"],
                                 "eval_sha256": sorted(x["sha256"] for x in o_man), "chain_manifest": [x["manifest"] for x in o_man]},
                    "replacement": {"launch_id": rp.get("launch_id"), "run_key": rp.get("run_key"), "state": rp.get("state"), "eval_sha256": sorted(rp_sh),
                                    "chain_manifest": [x["manifest"] for x in rp_man]},
                    "replacement_state_done": r.get("replacement_state_done")})
    if seen != set(want):
        errors.append(f"rows cover {len(seen)} keys; the map has {len(want)}: missing {sorted(set(want) - seen)[:6]}")
    t3_cells = {x["cell"] for x in t3["rows"]}
    r3_cells = {r["cell"] for r in rows["rows"] if r["tier"] == "tier 3"}
    if t3_cells != r3_cells:
        errors.append(f"tier-3 cells {len(r3_cells)} != the tier-3 map's {len(t3_cells)}")
    doc = {"rule": "09 @ 92d21b8f6 / e46ba05d0 (IL completions; per-arm maps @ 54823ddbf): every listed key's completion run is authoritative, no fallback",
           "rows": {"path": a.rows, "sha256": hashlib.sha256(rows_raw).hexdigest(), "completion_launch": rows.get("completion_launch"), "original_launch": rows.get("original_launch")},
           "map": {"spec": a.map, "sha256": hashlib.sha256(map_raw).hexdigest(), "keys": len(want)},
           "tier3": {"spec": a.tier3, "sha256": hashlib.sha256(t3_raw).hexdigest(), "cells": len(t3_cells)},
           "original_family": a.orig_prefix, "completion_family": a.repl_prefix, "records": sorted(out, key=lambda x: x["key"]),
           "counts": {"keys": len(out), "tier2": sum(x["tier"] == "tier 2" for x in out), "tier3": sum(x["tier"] == "tier 3" for x in out),
                      "supersedes_original_run": sum(x["supersedes_original_run"] for x in out), "replacement_done": sum(bool(x["replacement_state_done"]) for x in out)},
           "errors": errors, "pass": not errors}
    Path(a.out).open("x").write(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: doc[k] for k in ("counts", "errors", "pass")}, indent=1))
    if errors:
        sys.exit(2)


if __name__ == "__main__":
    main()
