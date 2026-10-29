"""ug comparability map (09's ug note @ 48363962a, Q1; Append 34 §7): for each of the 149 registered ug cells, keyed `sim_ug|<cell>`, the
exact run1, run2 and sim1 artifacts FINAL (@ 3ce6ea55e) used, per world {source (path @ commit), row key, eval sha256, volume path} or null
with a reason. Committed before ug staging; the comparability job (provenance_chain.py ug mode) hashes these and sim_ug's own artifact.
- S_new cells (fork source from refs_ug/q<k>): FINAL's spec.counted entries per world (run1, run2, sim; the sim entries include the Q2
  rerun's designated replacements, and the q2fu exclusion record drops its stopped runs), read at their pinned commits; rows by world and task.
- S_old cells (fork source from il_refindex_old): correctedb_manifest_v1.json @ 46ca25fc4, epochs with join "<world>|<cell>" (sim1 -> the
  stored world `sim`); the key's file (without #e0) joined to files[].path of the upload manifest @ 581ffb229 for its sha256 and attempt
  fields. A counted attempt without an epoch (e.g. t88) is null with the upload manifest's own reason; never a fallback.
A cell whose artifact is not unique stops the build (exit 2, nothing written). Keys, paths and hashes only.

  python make_ug_comparability_map.py <out json>
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FINAL = ("3ce6ea55e", "experiments/2026-10-14_xerr_scaleup_09/specs/final_spec_3_launch.json")
FROZEN = ("origin/aswin/q3-ungrounded-sim", "experiments/2026-10-15_q3_ungrounded_sim/results/ug_frozen_manifest_2026-10-15_005643.json")
# 09's ruling @ 6a36b4a27 (via d9 ~13:52): excluded from ug's analysis (both arms of G's paired set); status rows, never a fallback
RULED_OUT = {c: "reference mismatch: pinned reference is a reference-rerun attempt, counted honest_a is the first attempt; no fallback (09 @ 6a36b4a27)"
             for c in ("t389_model_registry", "t83_auto_workflow")}
CORRB = ("46ca25fc4", "experiments/2026-10-14_q35_sol_avg_POSTHOC/results/correctedb_manifest_v1.json")
UPLOAD = ("581ffb229", "experiments/2026-10-13_q4_why_weaker_POSTHOC/results/manifest_q4b_eval_upload_v1.json")
CHAIN_VOL, SRC_VOL = "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14", "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13"


def show(c: str, p: str) -> bytes:
    return subprocess.run(["git", "show", f"{c}:{p}"], capture_output=True, check=True, cwd=REPO).stdout


def full(c: str) -> str:
    return subprocess.run(["git", "rev-parse", c], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()


def main():
    out = Path(sys.argv[1])
    errors = []
    fin_raw, fr_raw, cb_raw, up_raw = show(*FINAL), show(*FROZEN), show(*CORRB), show(*UPLOAD)
    counted = json.loads(fin_raw)["counted"]
    new = {}  # (world, cell) -> artifact
    for w in ("run1", "run2", "sim"):
        for e in counted[w]:
            man = json.loads(show(e["commit"], e["path"]))
            drop = set()
            if e.get("exclusions"):
                xr = json.loads(show(e["exclusions"]["commit"], e["exclusions"]["path"]))
                drop = {x["manifest_key"] for x in xr["excluded"] if x.get("excluded") is True}
            for r in man["files"]:
                if r["world"] != w or f"{r['launch']}/{r['rel']}" in drop:
                    continue
                art = {"source": f"{e['path']} @ {e['commit']}", "row_key": f"{r['launch']}/{r['rel']}", "eval_sha256": r["sha256"],
                       "volume": CHAIN_VOL, "volume_path": f"evals_v1/{r['path']}", "launch_task": r.get("launch_task")}
                if (w, r["task"]) in new:
                    errors.append(f"{w}|{r['task']}: two FINAL counted rows")
                new[(w, r["task"])] = art
    up = {f["path"]: f for f in json.loads(up_raw)["files"]}
    old = {}
    for ep in json.loads(cb_raw)["epochs"]:
        w, cell = ep["join"].split("|", 1)
        if w not in ("run1", "run2", "sim"):
            continue
        path = ep["key"].split("#", 1)[0]
        f = up.get(path)
        if f is None:
            errors.append(f"{ep['join']}: corrected-B epoch file not in the upload manifest")
            continue
        if (w, cell) in old:
            errors.append(f"{ep['join']}: two corrected-B epochs")
        old[(w, cell)] = {"source": f"{CORRB[1]} @ {CORRB[0]} (epoch {ep['key']}) + {UPLOAD[1]} @ {UPLOAD[0]}", "row_key": path,
                          "eval_sha256": f["sha256"], "attempt_index": f.get("attempt_index"), "attempt_kind": f.get("attempt_kind"),
                          "volume": SRC_VOL, "volume_path": f"evals_v1/{path}"}
    ne = {}  # the upload manifest's counted attempts without an .eval: (world, cell) -> its path
    for set_name, paths in json.loads(up_raw)["counted_attempts_without_eval"].items():
        for p in paths:
            arm, task = p.split("/")[1:3]
            w = "run2" if set_name == "run2" else "run1" if arm == "attack" else "sim" if arm.startswith("sim_attack") else None
            if w:
                ne[(w, task.split("_redraw")[0])] = p
    rows = {}
    for c in json.loads(fr_raw)["cells"]:
        k, src = c["key"], (c.get("reference_record") or {}).get("index_source", "")
        origin = "S_new" if "/refs_ug/q" in src else "S_old" if src.endswith("il_refindex_old/launch.json") else None
        if origin is None:
            errors.append(f"{k}: origin not identifiable")
            continue
        row = {"cell": k, "origin": origin}
        for label, w in (("run1", "run1"), ("run2", "run2"), ("sim1", "sim")):
            a = (new if origin == "S_new" else old).get((w, k))
            if a is None:
                why = (f"counted attempt without an .eval ({ne[(w, k)]}, upload manifest @ {UPLOAD[0]})" if origin == "S_old" and (w, k) in ne
                       else "no FINAL counted row" if origin == "S_new" else "no corrected-B epoch")
                row[label] = None
                row[f"{label}_null_reason"] = why
            else:
                row[label] = a
        if k in RULED_OUT:
            row["analysis_status"] = RULED_OUT[k]
        rows[f"sim_ug|{k}"] = row
    if errors:
        print(json.dumps({"errors": errors[:50], "n": len(errors)}, indent=1))
        sys.exit(2)
    doc = {"what": "ug comparability map (09's ug note Q1; Append 34 §7): FINAL's run1 / run2 / sim1 artifacts per registered ug cell",
           "inputs": {n: {"commit": full(c)[:12], "path": p, "sha256": hashlib.sha256(raw).hexdigest()}
                      for n, (c, p), raw in (("final", FINAL, fin_raw), ("frozen_manifest", FROZEN, fr_raw), ("correctedb", CORRB, cb_raw), ("upload", UPLOAD, up_raw))},
           "sim_ug": "the ug family's own counted artifact, from the chain's committed ug manifest (added by the comparability job)",
           "cells": rows}
    with out.open("x") as f:
        f.write(json.dumps(doc, indent=1) + "\n")
    nulls = {k: [x for x in ("run1", "run2", "sim1") if v[x] is None] for k, v in rows.items() if any(v[x] is None for x in ("run1", "run2", "sim1"))}
    from collections import Counter
    print(json.dumps({"cells": len(rows), "origin": Counter(v["origin"] for v in rows.values()), "null_worlds": nulls}, indent=1))


if __name__ == "__main__":
    main()
