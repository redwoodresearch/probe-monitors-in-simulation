"""S5r supersession record (09 @ bffe22e3e via d9, 10-15 ~04:00), written before any 5r sim run is routed for scoring. Keys and hashes only.
Per key sim|<cell> of the pinned map (new-mains fork, s5r_supersession_map_filled_2026-10-15.json @ 75fef359f):
- the original: launch, run key, .eval sha256 (the map's, checked against the committed S5 follow-up manifests) and how the chain's
  staging treated it (staged set and the stager's reason);
- the replacement: launch, run key, .eval sha256 (the chain's staged manifest row, checked against the map) and its staging outcome;
- authoritative: replacement, rerun_reason "infra: RateLimitError"; no fallback (a failed or unstaged replacement stays a failed replacement).
Resolution order: staging retries within each launch first (the routing file's staged set), then supersession, then uniqueness per
(world, cell). The originals stay in Append 35 accounting (6 of 69 attrition). Also: the 6-key list for the mandatory sensitivity
without the replacement cells, and the length check at completion (@ 6bdad83c9) cited. Any identity mismatch, a missing or extra
row, or a duplicate key is an ERROR (exit 2).
Run: python make_s5r_supersession.py --routing <chain routing dir> --out <json>"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAP = ("75fef359f", "experiments/2026-10-14_q3_new_mains_launch/results/s5r_supersession_map_filled_2026-10-15.json")
LENGTH = ("6bdad83c9", "experiments/2026-10-14_q3_new_mains_launch/results/length_check_s5_resolved_2026-10-15.json")
ORIG_MANIFESTS = ["manifest_nm360_s5_sim_s1.json", "manifest_nm360_s5_sim.json"]
REPL = "nm360_s5r_sim"


def canon(launch_id):
    """09's launch_id comparison rule @ c065957e2: remove exactly one literal leading `runs/`, nothing else; then compare bytes."""
    return launch_id[len("runs/"):] if isinstance(launch_id, str) and launch_id.startswith("runs/") else launch_id


def git_show(commit, path):
    return subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, check=True, cwd=HERE).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routing", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    errors = []
    mraw = git_show(*MAP)
    mp = json.loads(mraw)["map"]
    lraw = git_show(*LENGTH)
    orig_rows = {}
    for m in ORIG_MANIFESTS:
        for f in json.loads((HERE / "config" / m).read_text())["files"]:
            if f["arm_dir"] == "sim_attack@september":
                orig_rows.setdefault(f["task"], []).append(f | {"manifest": m})
    repl_man = json.loads((HERE / "config" / f"manifest_{REPL}.json").read_text())
    repl_rows = {}
    for f in repl_man["files"]:
        if f["arm_dir"] == "sim_attack@september":
            repl_rows.setdefault(f["task"], []).append(f)
    route = json.loads((Path(a.routing) / f"nm360-s5rfu__{REPL}.json").read_text())
    staged_set = route["staged_set"]
    repl_ex = {k.split("#")[0].split("/")[-2]: v for k, v in (route.get("staging_exclusions") or {}).items()}
    orig_ex = {}
    for rf in ("nm360-s5fu_s1__nm360_s5_sim_s1.json", "nm360-s5fu__nm360_s5_sim.json"):
        r = json.loads((Path(a.routing) / rf).read_text())
        for k, v in (r.get("staging_exclusions") or {}).items():
            orig_ex[k.split("#")[0].split("/")[-2]] = {"reason": v, "staged_set": r["staged_set"]}
    keys = [r["key"] for r in mp]
    if len(keys) != len(set(keys)) or len(keys) != 6:
        errors.append(f"map rows: {len(keys)} keys, {len(set(keys))} unique (expected 6)")
    cells = sorted({k.split("/")[-1] for k in keys})
    if sorted(repl_rows) != cells:
        errors.append(f"replacement launch cells {sorted(repl_rows)} differ from the map's {cells}")
    rows = []
    for r in sorted(mp, key=lambda x: x["key"]):
        c = r["key"].split("/")[-1]
        o, rp = orig_rows.get(c, []), repl_rows.get(c, [])
        if len(o) != 1 or len(rp) != 1:
            errors.append(f"sim|{c}: {len(o)} original and {len(rp)} replacement manifest rows (expected 1 each)")
            continue
        o, rp = o[0], rp[0]
        if o["sha256"] != r["original"]["eval_sha256"]:
            errors.append(f"sim|{c}: original sha differs from the map")
        if rp["sha256"] != r["replacement"]["eval_sha256"]:
            errors.append(f"sim|{c}: replacement sha differs from the map")
        if canon(o["launch"]) != canon(r["original"]["launch"]) or canon(rp["launch"]) != canon(r["replacement"]["launch"]):  # 09 @ c065957e2
            errors.append(f"sim|{c}: launch ids differ from the map")
        rows.append({"key": f"sim|{c}", "authoritative": "replacement", "rerun_reason": "infra: RateLimitError",
                     "original": {"launch": o["launch"], "run_key": r["original"]["run_key"], "eval_sha256": o["sha256"], "manifest": o["manifest"],
                                  "chain_staging": orig_ex.get(c, {"reason": None, "staged_set": None}), "map_classification": r["original"].get("classification"),
                                  "accounting": "Append 35 attrition (6 of 69): counted, never scored"},
                     "replacement": {"launch": rp["launch"], "run_key": r["replacement"]["run_key"], "eval_sha256": rp["sha256"], "staged_set": staged_set,
                                     "staged": c not in repl_ex, "stager_reason": repl_ex.get(c), "map_usable_reference": r["replacement"].get("usable_reference"),
                                     "map_clean": r["replacement"].get("clean")},
                     "no_fallback": True})
    if any(not x["original"]["chain_staging"]["reason"] for x in rows):
        errors.append("an original was staged by the chain (expected: every original excluded at staging)")
    doc = {"rule": "09 @ bffe22e3e (S5 supersession; relayed by d9)", "resolution_order": "staging retries within each launch, then supersession, then uniqueness",
           "map": {"commit": MAP[0], "path": MAP[1], "sha256": hashlib.sha256(mraw).hexdigest()},
           "length_check_at_completion": {"commit": LENGTH[0], "path": LENGTH[1], "sha256": hashlib.sha256(lraw).hexdigest()},
           "replacement_manifest": {"path": f"experiments/2026-10-14_q3_overnight_chain/config/manifest_{REPL}.json",
                                    "sha256": hashlib.sha256((HERE / "config" / f"manifest_{REPL}.json").read_bytes()).hexdigest()},
           "staged_set": staged_set, "sensitivity_without_replacement_cells": sorted(x["key"] for x in rows), "rows": rows, "errors": errors, "pass": not errors}
    Path(a.out).open("x").write(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(rows), "staged_replacements": sum(x["replacement"]["staged"] for x in rows), "errors": errors, "pass": not errors}, indent=1))
    if errors:
        sys.exit(2)


if __name__ == "__main__":
    main()
