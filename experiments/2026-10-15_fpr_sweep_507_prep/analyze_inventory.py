"""Metadata inventory of the 507 scale-up arms: KEYS AND COUNTS ONLY, no score value is read.

Inputs are the scorers' manifests (Sol manifest_sol.json, 4f's manifest_p2.json and manifest_famexp.json), fetched with
`modal volume get` from probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14:/scaleup/<arm>/<sub>/ into --man-dir as
<arm>@<sub>.json, plus the `modal volume ls --json` listings in --ls-dir (<arm>@<sub>.json). The script reads only key lists,
counts, flags and file names from them; manifests carry no score arrays (reproduction-gate diffs are skipped, never printed).

  python3 -I analyze_inventory.py --man-dir <dir> --ls-dir <dir> --out results/inventory_507_<stamp>.json
"""

import argparse
import glob
import hashlib
import json
import os
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--man-dir", required=True)
ap.add_argument("--ls-dir", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()


def load(p):
    b = open(p, "rb").read()
    return json.loads(b), hashlib.sha256(b).hexdigest()


def listing(arm, sub):
    p = os.path.join(a.ls_dir, f"{arm}@{sub}.json")
    return sorted(x["filename"].rsplit("/", 1)[1] for x in json.load(open(p))) if os.path.exists(p) else None


arms = sorted({os.path.basename(p).split("@")[0] for p in glob.glob(os.path.join(a.ls_dir, "*@*.json"))})
rows, reply_keys = {}, defaultdict(set)
for arm in arms:
    r = {"stratum": {"su311": "S_new", "nm360": "S_newmains"}.get(arm.split("_")[0], "other")}
    sol_dirs = [s for s in ("sol_r2", "sol") if listing(arm, s) is not None]
    used = next((s for s in sol_dirs if "SUPERSEDED.json" not in listing(arm, s)), None)
    r["sol"] = {"dirs": sol_dirs, "used": used}
    if used:
        m, sha = load(os.path.join(a.man_dir, f"{arm}@{used}.json"))
        files = listing(arm, used)
        folds = m["folds"]
        miss = {rd: set(m["runs_without_rows"].get(rd, [])) for rd in ("avg", "cmd_A")}
        r["sol"] |= {
            "manifest_sha256": sha, "scorer_git": m["scorer_git"], "arm_to_world": m["arm_to_world"],
            "runs_staged": len(folds), "fold_counts": dict(sorted(Counter(map(str, folds.values())).items())),
            "fold_minus1_runs": sum(v == -1 for v in folds.values()), "routing_entries": len(m.get("routing", {})),
            "unit_files": {f: f in files for f in ("a_tight__avg_top2.npz", "a_tight__cmd_A.npz", "c_asis_tight__avg_top2.npz", "c_asis_tight__cmd_A.npz")},
            "runs_with_reply_rows": len(folds) - len(miss["avg"]), "runs_with_cmd_A_rows": len(folds) - len(miss["cmd_A"]),
            "runs_without_rows": {k: len(v) for k, v in miss.items()},
            "row_counts": "not in the manifest (counted by key in the own_a job / the sweep's G2)",
        }
        for k in folds:
            if k not in miss["avg"]:
                reply_keys[k.split("|", 1)[0]].add(k.split("|", 1)[1])
    if listing(arm, "p2") is not None and os.path.exists(os.path.join(a.man_dir, f"{arm}@p2.json")):
        m, sha = load(os.path.join(a.man_dir, f"{arm}@p2.json"))
        r["p2"] = {"manifest_sha256": sha, "world": m["world"], "verdict": m["verdict"],
                   "counts": {p: {rd: {k: c[k] for k in ("rows", "runs", "dropped")} for rd, c in m["counts"][p].items() if rd in ("reply_mean", "command")}
                              for p in ("fibs", "harmfulness", "refusal", "highstakes")}}
    else:
        r["p2"] = None
    fx = next((s for s in ("famexp_r2", "famexp") if listing(arm, s) is not None and "SUPERSEDED.json" not in listing(arm, s)), None)
    if fx:
        m, sha = load(os.path.join(a.man_dir, f"{arm}@{fx}.json"))
        names = listing(arm, fx)
        r["own_frac100_L48"] = {"dir": fx, "manifest_sha256": sha, "complete": m.get("complete"), "arm_to_world": m.get("arm_to_world"),
                                "files": {x: f"own_frac100_L48__{x}.npz" in names for x in ("R", "C")},
                                "failed_layers": len(m.get("failed_layers", [])),
                                "incomplete_entries_L48": sum("own_frac100_L48" in json.dumps(e) for e in m.get("incomplete_variant_readouts", []))}
    else:
        r["own_frac100_L48"] = None
    rows[arm] = r

out = {"label": "inventory by keys and counts only; no score value read", "arms": rows}
for st in ("S_new", "S_newmains"):
    pre = {"S_new": "su311", "S_newmains": "nm360"}[st]
    cells = {w: {c for c in reply_keys[w] if c.startswith(("t" if pre == "su311" else "nm"))} for w in ("run1", "run2", "sim", "honest_a", "honest_b")}
    out[f"sol_reply_coverage_{st}"] = {"cells_with_reply_rows": {w: len(v) for w, v in cells.items()},
                                       "cells_with_run1_run2_sim": len(cells["run1"] & cells["run2"] & cells["sim"]),
                                       "note": "union over the arm folders read (sol_r2 where sol is SUPERSEDED); before attempt labels, counted-key and staging checks"}
blob = json.dumps(out, indent=1, sort_keys=True)
open(a.out, "x").write(blob)
print(a.out, hashlib.sha256(blob.encode()).hexdigest())
