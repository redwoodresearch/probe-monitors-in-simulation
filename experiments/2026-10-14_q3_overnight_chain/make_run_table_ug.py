"""run_table_ug.json (09's ug note @ 48363962a, Q2; Append 34): one row per registered ug cell, keys and launch names only, built after the
chain has read the ug launch on its box (boxfiles_<batch>.json of the family's final remainder and any snapshot, in the chain state). Per row:
- cell, origin (S_new for a cell whose fork-source reference came from refs_ug/q<k>_launch.json; S_old for il_refindex_old), canonical
  main task (Append 34's frozen manifest `main`) and canonical main launch (the counted run1's launch), env;
- run1_launch, run2_launch, attack_pair_launch, sim1_launch: copied from the scale-up table (run_table_311.json @ 806b7a63a6, the one FINAL
  pins) for S_new cells; for S_old cells from the corrected-B manifest (correctedb_manifest_v1.json @ 46ca25fc4: the run1 / run2 / sim
  epochs, the T196 artifacts FINAL used), else, for a counted attempt with no epoch (e.g. t88), the launch named by the upload manifest
  @ 581ffb229 (its counted_attempts_without_eval list), with the world listed under t196_counted_without_eval; attack_pair_launch
  "separate" (T196's second attack is its own launch);
- ug_launch and ug_status, in this order: no usable honest_a fork source -> "not run: no usable honest_a fork source"; a counted attempt
  without an .eval -> "not run" ("counted, no .eval", with the run's launcher state); a counted attempt with an .eval -> "ran" (with its
  launcher state, so an interrupted run stays visible); never counted -> "not run"
  with the launcher's own state, and "not run (cap)" ONLY for a run that never started (state ready / waiting / not run) when a cap or
  failure-stop marker is present, with Append 34's "partial, selected by collection/completion/admission"; a started or interrupted run
  keeps its launcher state and never becomes "not run (cap)"; a cell the launch does not list is "not run: absent from the launch";
- fork_source: Append 34's manifest reference key (copy path) and verified sha256.
Ambiguity (a cell in two epochs of one world, two rows for one cell, an origin that cannot be read) stops the build: exit 2, nothing written.

  python make_run_table_ug.py <chain state dir> <ug family id> <out json>
"""
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FROZEN = ("origin/aswin/q3-ungrounded-sim", "experiments/2026-10-15_q3_ungrounded_sim/results/ug_frozen_manifest_2026-10-15_005643.json")
# 09's ruling @ 6a36b4a27 (via d9 ~13:52): excluded from ug's analysis (both arms of G's paired set); status rows, never a fallback
RULED_OUT = {c: "reference mismatch: pinned reference is a reference-rerun attempt, counted honest_a is the first attempt; no fallback (09 @ 6a36b4a27)"
             for c in ("t389_model_registry", "t83_auto_workflow")}
RT311 = ("806b7a63a6", "experiments/2026-10-14_q3_overnight_chain/results/hygiene/run_table_311.json")
CORRB = ("46ca25fc4", "experiments/2026-10-14_q35_sol_avg_POSTHOC/results/correctedb_manifest_v1.json")
UPLOAD = ("581ffb229", "experiments/2026-10-13_q4_why_weaker_POSTHOC/results/manifest_q4b_eval_upload_v1.json")
NEVER_STARTED = ("ready", "waiting", "not run")


def show(c: str, p: str) -> bytes:
    return subprocess.run(["git", "show", f"{c}:{p}"], capture_output=True, check=True, cwd=REPO).stdout


def pin(c: str, p: str, raw: bytes) -> dict:
    full = subprocess.run(["git", "rev-parse", c], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()
    return {"commit": full[:12], "path": p, "sha256": hashlib.sha256(raw).hexdigest()}


def main():
    st_dir, fam_id, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    st = json.loads((st_dir / "state.json").read_text())["batches"]
    b = st[fam_id]
    errors = []
    if b["kind"] != "fu" or "boxfiles" not in b["done"]:
        sys.exit(f"{fam_id}: the family's final remainder must have its box read done")
    fam = [b] + [x for x in st.values() if x.get("family") == fam_id and x["id"] != fam_id and x["kind"] == "fus"]
    boxes = {x["id"]: json.loads((st_dir / f"boxfiles_{x['id']}.json").read_text()) for x in fam if "boxfiles" in x["done"]}
    no_eval = {x["task"] for bx in boxes.values() for x in bx["counted_without_eval"]}  # across every root of the family
    with_eval = {f["task"] for bx in boxes.values() for m in bx["manifests"].values() for f in m["files"] if f["world"] == b.get("sim_world")}
    run_state = {}  # every counted run's launcher state, from each root's launch_subset (the members' launch.json records)
    for bx in boxes.values():
        for r in json.loads(bx["launch_subset"])["runs"]:
            if r.get("arm") and "sim" in r["arm"] and run_state.setdefault(r["task"], r.get("state")) != r.get("state"):
                errors.append(f"{r['task']}: two launcher states across roots")
    final = boxes[fam_id]
    not_counted = {}
    for x in final["facts"]["not_counted"]:
        if not_counted.setdefault(x["task"], x["state"]) != x["state"]:
            errors.append(f"{x['task']}: two launcher states")
    roster = set(final["facts"]["roster"])
    markers = final.get("markers", [])
    cap = [m for m in markers if "CAP" in m.upper() or "STOP" in m.upper()]
    fr_raw, rt_raw, cb_raw, up_raw = show(*FROZEN), show(*RT311), show(*CORRB), show(*UPLOAD)
    frozen, up = json.loads(fr_raw), json.loads(up_raw)
    rt = {}
    for r in json.loads(rt_raw)["cells"]:
        if r["cell"] in rt:
            errors.append(f"{r['cell']}: two run_table_311 rows")
        rt[r["cell"]] = r
    old = {}
    for e in json.loads(cb_raw)["epochs"]:
        w, cell = e["join"].split("|", 1)
        if w in ("run1", "run2", "sim"):
            if (w, cell) in old:
                errors.append(f"{w}|{cell}: two corrected-B epochs")
            old[(w, cell)] = e["key"].split("/", 1)[0]
    ne_old = {}  # T196 counted attempts without an .eval: (world, cell) -> launch (upload manifest's list; e.g. t88)
    for set_name, paths in up["counted_attempts_without_eval"].items():
        for p in paths:
            launch, arm, task = p.split("/")[:3]
            cell = task.split("_redraw")[0]
            w = "run2" if set_name == "run2" else "run1" if arm == "attack" else "sim" if arm.startswith("sim_attack") else None
            if w:
                ne_old.setdefault((w, cell), launch)
    cells = frozen["cells"]
    if len(cells) != len({c["key"] for c in cells}):
        errors.append("duplicate cells in the frozen manifest")
    rows = []
    for c in cells:
        k, ref = c["key"], c.get("reference_record") or {}
        src = ref.get("index_source", "")
        origin = "S_new" if "/refs_ug/q" in src else "S_old" if src.endswith("il_refindex_old/launch.json") else None
        row = {"cell": k, "origin": origin, "canonical_main_task": c["main"], "env": c["env"],
               "fork_source": {"reference_key": ref.get("copy"), "sha256": c.get("reference_sha256_verified")}}
        if origin is None:
            errors.append(f"{k}: origin not identifiable from its fork-source index")
            continue
        if origin == "S_new":
            r = rt.get(k)
            if r is None:
                errors.append(f"{k}: S_new cell absent from run_table_311")
                continue
            row |= {"run1_launch": r["run1_launch"], "run2_launch": r["run2_launch"], "attack_pair_launch": r["attack_pair_launch"],
                    "sim1_launch": r["sim_launch"], "sim1_status": r["sim_status"]}
        else:
            ne = []
            for w, f_ in (("run1", "run1_launch"), ("run2", "run2_launch"), ("sim", "sim1_launch")):
                row[f_] = old.get((w, k)) or ne_old.get((w, k))
                if (w, k) not in old and (w, k) in ne_old:
                    ne.append(w)
            row["attack_pair_launch"] = "separate"
            row["t196_counted_without_eval"] = ne
            missing = [f_ for f_ in ("run1_launch", "run2_launch", "sim1_launch") if row[f_] is None]
            if missing:
                row["t196_missing"] = missing  # a genuine missing-artifact state, kept as it is (09 Q1)
        row["canonical_main_launch"] = row["run1_launch"]
        row["ug_launch"] = b["run"].rsplit("/", 1)[-1]
        if k in RULED_OUT:
            row["analysis_status"] = RULED_OUT[k]
        if not (ref.get("copy") and c.get("reference_sha256_verified")):
            row["ug_status"] = "not run: no usable honest_a fork source"
        elif k in no_eval:
            row["ug_status"], row["ug_status_detail"] = "not run", {"launcher_state": run_state.get(k, "unknown"), "counted": "counted, no .eval"}
        elif k in with_eval:
            row["ug_status"], row["ug_status_detail"] = "ran", {"launcher_state": run_state.get(k, "unknown")}
        elif k in not_counted:
            s = not_counted[k]
            if cap and s in NEVER_STARTED:
                row["ug_status"], row["ug_status_detail"] = "not run (cap)", {"launcher_state": s, "markers": cap,
                                                                              "selection": "partial, selected by collection/completion/admission"}
            else:
                row["ug_status"], row["ug_status_detail"] = "not run", {"launcher_state": s}
        elif k not in roster:
            row["ug_status"], row["ug_status_detail"] = "not run", {"launcher_state": "absent from the launch"}
        else:
            errors.append(f"{k}: listed by the launch but neither counted nor uncounted")
            continue
        rows.append(row)
    if errors:
        print(json.dumps({"errors": errors}, indent=1))
        sys.exit(2)
    doc = {"pool": "ug", "world": "sim_ug", "family": fam_id, "launch": b["run"], "markers": markers, "cap_or_stop_markers": cap,
           "inputs": {"frozen_manifest": pin(FROZEN[0], FROZEN[1], fr_raw), "run_table_311": pin(*RT311, rt_raw), "correctedb": pin(*CORRB, cb_raw),
                      "upload_manifest": pin(*UPLOAD, up_raw),
                      "boxfiles": {x: hashlib.sha256((st_dir / f"boxfiles_{x}.json").read_bytes()).hexdigest() for x in boxes}},
           "rule": "09's ug note @ 48363962a Q2 (Append 34); keys and launch names only", "cells": rows}
    with out.open("x") as f:
        f.write(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"cells": len(rows), "origin": Counter(r["origin"] for r in rows), "ug_status": Counter(r["ug_status"] for r in rows),
                      "t196_missing": {r["cell"]: r["t196_missing"] for r in rows if r.get("t196_missing")}}, indent=1))


if __name__ == "__main__":
    main()
