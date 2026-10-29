"""inventory_b_fix_v2 (09 route C, option (ii), via 2e / 507-prep 10-17): inventory_b_fix_v1 @ 09b070617 + EXACTLY the 20 pilot sim-honest rows,
by key. Keys, names and shas only; no Modal. Two files from the same 20 row objects, both exclusive-create:
  (1) inventory_b_pilot_rows_v1_<stamp>.json: {"runs": [the 20 rows]} for CUT's scorers (beside the pinned fix_v1; no key overlap);
  (2) inventory_b_fix_v2_<stamp>.json: fix_v1's rows byte-identical, then the same 20 rows (INPUTS_FULL pins only this one).
(2) names (1) by sha256. Exit 0 only when COMPLETE.

The pilot rows mirror fix_v1's simh_q4f_v1_full_b1 rows field for field (same names, same order):
- key <launch>/<rel>#e0 and eval_rel / launch / eval_sha256 from the pilot upload manifest @ 4af62db74, by key equal to the counted keys
  (simh_counted_keys_pilot_v3 @ 344829af6): same task, same eval sha256;
- world "sim_honest" (the ROLE world: inputs_full.roles_from_inventory roles any other world as REAL), asserted in both files; the counted
  keys' (scorer) world is simh_pilot, asserted too; arm sim_honest@september; task_set T196 (every task a T196 cell);
- original_folder simh_q4f_v1_pilot__sim_honest (registered by key in pilot_v3); cut = the registered 32k extraction (staged set
  simh_q4f_v1_pilot, root simh_q4b_extract_v1_pilot_c, its extraction record by sha, which must name that run and inputs);
- uncut_planned = CUT's pilot staging: folder simh_q4f_v1_pilot_cutfull_v1__sim_honest, staged set simh_pilot_cutfull_v1, root
  simh_pilot_cutfull_extract_v1, staged sha ca037a4d..., format_check (--format) and routing (--routing) as inputs_full.format_problems
  reads them: the format record passes on exactly the 20 keys at that staged sha; the routing record binds this original folder to this
  uncut folder and staged set, lists no failing set, and is the highest revision of its name at its commit;
- full_len = tokens_before = the format record's full_len for the key (an int); in_S_old_cohort_cell from FINAL's cohort cells (as v1).
  Unlike v2 (which compared the census's tokens_before with the format record's full_len), both come from ONE source here: the pilot has no
  v1 census row. CUT's staging record is not read.
- "registered" has two senses here: the inventory's `cut` block is the pilot's registered 32k extraction (the files the 32k pack read), while
  INPUTS_FULL's "n/a" inside_32k_equal for this root means the pinned 32k CUT MAP has no counterpart (the pilot was never in the registered
  32k S_old H_sim list). Nothing reads the inventory's `cut` block for that rule.
STOPs: fix_v1 not at its pinned sha; any pilot key, original folder or uncut folder already in fix_v1; not exactly the 20 counted keys
(a 21st or a missing key, in the manifest, the upload manifest or the format record); a world other than sim_honest; any mismatch above.

  uv run python make_inventory_b_fix_v2.py --format <commit>:<path>@<sha256> --routing <commit>:<path>@<sha256> [--extraction-record ...] [--dry]
"""

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIX_V1 = ("09b070617", "experiments/2026-10-15_cut_sensitivity/results/inventory_b_fix_v1_2026-10-16_025058.json",
          "4a5d0fb3cad32740f849df5def3c7a03a40a18aa4c42bc360709f0c740973d03")
PILOT = ("344829af6", "experiments/2026-10-14_q3_q35_simhonest_purchase/results/simh_counted_keys_pilot_v3.json",
         "2149c5cde368aef9e5012c8d9cda5b12fe5a93db3d27ad2d525225bb9f511b27")
UPLOAD = ("4af62db74", "experiments/2026-10-14_q3_q35_simhonest_purchase/results/manifest_simh_eval_upload_pilot.json",
          "875167b9916560444f80be78bf516beba6711837a811bd677b82f66e704d182b")
T196 = ("debef0af2", "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json",
        "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb")
FINAL = ("3ce6ea55e", "experiments/2026-10-14_xerr_scaleup_09/results/xerr_scaleup_final_2026-10-15_013255.json",
         "7dceb8bb25752aa3619cfde64add702fd4f7d13a1309a56dbc8880a6b27af9f5")  # v1's cohort source (= run_common455.FINAL); only cohort_cells is used
EXTRACTION = ("c389286cf", "experiments/2026-10-14_q3_q35_simhonest_purchase/results/extract_simh_simh_q4b_extract_v1_pilot_c_a1_2026-10-14_171107.json",
              "9ba914b378369bd40f09db16fea82d99b717c7f9bed73a9f3c28c14c8981b76e")
N = 20
ROLE_WORLD, KEY_WORLD, ARM, TASK_SET = "sim_honest", "simh_pilot", "sim_honest@september", "T196"
ORIG, UNCUT = "simh_q4f_v1_pilot__sim_honest", "simh_q4f_v1_pilot_cutfull_v1__sim_honest"  # 2e (10-17 ~02:0x): CUT's rename (b1's pattern; cutfull.original())
CUT = {"staged_set": "simh_q4f_v1_pilot", "extraction_root": "simh_q4b_extract_v1_pilot_c",
       "volume": "probe-shift_aswin-surya_sub-q3onpol_work_kill-2026-11-13"}  # = b1's cut volume (v1's VOL["a25"])
UP = {"folder": UNCUT, "staged_set": "simh_pilot_cutfull_v1", "extraction_root": "simh_pilot_cutfull_extract_v1",
      "volume": "probe-shift_aswin-surya_sub-q3cutfull_acts_kill-2026-11-14",
      "staged_sha256": "ca037a4d5566b24cd4270f1eab92c127a396afc4cc1fc1c3d4e0fd2692abe8cb", "revision": 1, "staged_inputs_file": "sim_honest@september.jsonl"}
B1 = "simh_q4f_v1_full_b1__sim_honest"  # the mirror: fix_v1's b1 rows give the field names and order


def sha(b):
    return hashlib.sha256(b).hexdigest()


def rev_of(name):
    mo = re.match(r"^(.*?)(?:_r(\d+))?\.json$", name)
    return mo.group(1), int(mo.group(2) or 1)


def build(fix_b, pilot, upload, t196_cells, cohort, ext, fmt, fmt_ptr, rt, rt_ptr, rt_tree, ext_ptr, pins):
    """Pure: parsed inputs -> (pilot_rows_out, fix_v2_out_without_pilot_file_sha, stops). fix_b = fix_v1's bytes (sha checked here)."""
    stops = []
    if sha(fix_b) != FIX_V1[2]:
        return None, None, ["fix_v1 is not at its pinned sha256 (a changed row)"]
    fix = json.loads(fix_b)
    keys_ = pilot.get("keys") or []
    if pilot.get("n") != N or len(keys_) != N or len({k.get("task") for k in keys_}) != N or len({k.get("eval_sha256") for k in keys_}) != N:
        stops.append(f"counted keys: not {N} distinct tasks / eval sha256s (n {pilot.get('n')}, keys {len(keys_)})")
    if any(k.get("world") != KEY_WORLD or k.get("stratum") != "S_old" for k in keys_):
        stops.append(f"counted keys: scorer world not {KEY_WORLD} or stratum not S_old")
    if any(k.get("task") not in t196_cells for k in keys_):
        stops.append("counted keys: a task that is not a T196 cell")
    files = upload.get("files") or []
    if len(files) != N:
        stops.append(f"upload manifest: {len(files)} files, not {N}")
    up_by = {(f.get("launch"), f.get("rel")): f for f in files}
    ec_ = ext.get("config") or {}
    if ec_.get("run") != CUT["extraction_root"] or ec_.get("inputs") != CUT["staged_set"]:
        stops.append("extraction record: not the run / inputs of the pilot's registered 32k extraction")
    # format record, as inputs_full.format_problems reads it
    fkeys_ = [str(x.get("key")).removeprefix("runs/") for x in fmt.get("rows") or []]  # some stagers prefix keys with runs/ (inputs_full does too)
    if len(set(fkeys_)) != len(fkeys_):  # Astra r1 R1: one format row per (normalized) key, else full_len would be picked by row order
        stops.append(f"format record: {len(fkeys_) - len(set(fkeys_))} duplicate rows by normalized key")
    frows = dict(zip(fkeys_, fmt.get("rows") or []))
    if not (fmt.get("passes") is True and fmt.get("failing_keys") == [] and fmt.get("shard_errors") == [] and UP["staged_sha256"] in (fmt.get("staged_sha") or [])
            and fmt.get("epochs_checked") == len(fmt.get("rows") or [])):
        stops.append("format record: does not pass at the pilot's staged sha (passes / failing_keys / shard_errors / staged_sha / epochs_checked)")
    # routing record, as inputs_full.format_problems reads it
    cf, of = rt.get("cutfull") or {}, rt.get("original_folders") or {}
    if not (cf.get("staged_set") == UP["staged_set"] and any(of.get(w) == ORIG and (cf.get("folders") or {}).get(w) == UNCUT for w in of)):
        stops.append("routing record: does not bind the pilot's original folder to its uncut folder and staged set")
    if rt.get("failing", {}) not in ({}, None):
        stops.append("routing record: lists failing sets")
    base_, k_ = rev_of(Path(rt_ptr.split("@")[0].strip()).name)
    top_ = max((rev_of(Path(x).name)[1] for x in rt_tree if Path(x).name.endswith(".json") and rev_of(Path(x).name)[0] == base_), default=-1)
    if not (k_ == UP["revision"] == top_ and rt.get("revision", k_) == k_):
        stops.append(f"routing record: revision r{k_} is not r{UP['revision']}, the highest at its commit (r{top_}), or the record's own")
    if cf.get("extraction_root") not in (None, UP["extraction_root"]) or cf.get("volume") not in (None, UP["volume"]):
        stops.append("routing record: extraction root / volume differ from the planned ones")
    b1_ = next((r for r in fix["runs"] if r["original_folder"] == B1), None)
    if b1_ is None:
        return None, None, stops + ["fix_v1 has no b1 row to mirror"]
    rows = []
    for k in keys_:
        launch, task = k.get("launch"), k.get("task")
        pre_ = f"simh_evals_v1/{launch}/"
        vp_ = str(k.get("volume_path") or "")
        rel = vp_.removeprefix(pre_)
        f = up_by.get((launch, rel))
        if not vp_.startswith(pre_) or f is None:
            stops.append(f"{task}: no upload-manifest file for the counted key's volume path")
            continue
        if f.get("sha256") != k.get("eval_sha256") or f.get("task") != task or not rel.startswith(f"{ARM}/{task}/") or not k.get("run_key", "").startswith(ARM + "/"):
            stops.append(f"{task}: upload manifest and counted key disagree (sha256 / task / arm)")
        key = f"{launch}/{rel}#e0"
        fr = frows.get(key)
        fl = (fr or {}).get("full_len")
        if not isinstance(fl, int) or isinstance(fl, bool) or fl <= 0:
            stops.append(f"{task}: no int full_len in the format record for its key")
        rows.append({"key": key, "task_set": TASK_SET, "world": ROLE_WORLD, "arm": ARM, "task": task, "full_len": fl, "eval_rel": rel, "launch": launch,
                     "eval_sha256": k.get("eval_sha256"), "upload_manifest": f"{UPLOAD[1]} @ {UPLOAD[0]}",
                     "cut": {"staged_set": CUT["staged_set"], "extraction_root": CUT["extraction_root"], "extraction_record": ext_ptr, "volume": CUT["volume"]},
                     "original_folder": ORIG, "original_folder_registered": True,
                     "uncut_planned": {"folder": UP["folder"], "staged_set": UP["staged_set"], "extraction_root": UP["extraction_root"], "volume": UP["volume"],
                                       "staged_sha256": UP["staged_sha256"], "format_check": fmt_ptr, "routing": rt_ptr, "revision": UP["revision"],
                                       "staged_inputs_file": UP["staged_inputs_file"]},
                     "in_counted_membership710": None, "in_S_old_cohort_cell": task in cohort, "in_S_old_pool (FINAL pool_keys_old)": None,
                     "tokens_before": fl, "registered_source": {"source": f"{PILOT[1]} @ {PILOT[0]}", "sha256": PILOT[2], "matched_by_key": True}})
    pk_ = {r["key"] for r in rows}
    if set(frows) != pk_:
        stops.append(f"format record keys != the {N} pilot keys ({len(set(frows) - pk_)} extra, {len(pk_ - set(frows))} missing)")
    # the mirror and the asserts 507-prep / 09 named
    for r in rows:
        if list(r) != list(b1_) or any(list(r[x]) != list(b1_[x]) for x in ("cut", "uncut_planned", "registered_source")):
            stops.append(f"{r['task']}: fields differ from fix_v1's b1 rows (names or order)")
            break
    if len(rows) != N or len(pk_) != N:
        stops.append(f"{len(rows)} pilot rows, not {N}")
    if any(r["world"] != "sim_honest" for r in rows):
        stops.append("a pilot row's inventory world is not sim_honest (it would be roled REAL)")
    if {r["eval_sha256"] for r in rows} & {r.get("eval_sha256") for r in fix["runs"]}:  # 507-prep's review (optional): no .eval shared with fix_v1
        stops.append("a pilot .eval sha256 is already a fix_v1 row's")
    fk_ = {r["key"] for r in fix["runs"]}
    if pk_ & fk_ or any(r["original_folder"] == ORIG or (r.get("uncut_planned") or {}).get("folder") == UNCUT for r in fix["runs"]):
        stops.append("a pilot key, original folder or uncut folder is already in fix_v1")
    status = "STOP" if stops else "COMPLETE"
    lab_ = "09 route C option (ii) (2e / 507-prep 10-17): the 20 pilot sim-honest rows (S_old, T196); POST HOC; keys/names/shas only"
    pilot_out = {"label": lab_ + "; pilot rows only (CUT's scorers, beside inventory_b_fix_v1)", "status": status, "stops": stops, "problems": [],
                 "inputs": pins, "n_runs": len(rows), "by_world": {ROLE_WORLD: len(rows)}, "scorer_key_world": KEY_WORLD, "runs": rows}
    fix2 = dict(fix, label=lab_ + "; inventory_b_fix_v2 = inventory_b_fix_v1 + these 20 rows", status=status, stops=list(fix.get("stops") or []) + stops,
                supersedes={"path": Path(FIX_V1[1]).name, "sha256": FIX_V1[2], "pinned_as": f"inventory_b_fix_v1 @ {FIX_V1[0]}"},
                n_runs=len(fix["runs"]) + len(rows), by_world=dict(fix.get("by_world") or {}, **{ROLE_WORLD: (fix.get("by_world") or {}).get(ROLE_WORLD, 0) + len(rows)}),
                n_routed_folders=(fix.get("n_routed_folders") or 0) + 1,
                fix_v2={"rule": "09 route C option (ii): fix_v1's rows byte-identical, then the 20 pilot rows (by key)", "rows_added": len(rows),
                        "original_folder": ORIG, "uncut_folder": UNCUT, "inputs": pins, "scorer_key_world": KEY_WORLD, "stops": stops},
                runs=fix["runs"] + rows)
    return pilot_out, fix2, stops


CHANGED_TOP = ("label", "status", "stops", "supersedes", "n_runs", "by_world", "n_routed_folders", "fix_v2", "runs")  # fix_v2's own fields


def check_written(fix_b, pilot_b, fix2_b):
    """After serialization: fix_v1's rows equal in fix_v2 and the 20 rows equal across the two files, each compared as the canonical JSON of the
    parsed row (value and key order; fix_v1's own bytes are pinned by its sha256 above, not compared as source slices), world sim_honest in both, and
    every top-level field outside CHANGED_TOP equal to fix_v1's (507-prep's review: no per-folder summary silently carried as if updated)."""
    v1, p1, v2 = json.loads(fix_b), json.loads(pilot_b), json.loads(fix2_b)
    n1 = len(v1["runs"])
    bad = []
    if set(v2) - set(v1) - set(CHANGED_TOP) or any(k not in v2 or json.dumps(v1[k]) != json.dumps(v2[k]) for k in v1 if k not in CHANGED_TOP):
        bad.append("fix_v2 changes a top-level field outside the listed ones, or adds one")
    if [json.dumps(r) for r in v2["runs"][:n1]] != [json.dumps(r) for r in v1["runs"]]:
        bad.append("fix_v2's first rows are not fix_v1's, byte for byte")
    if [json.dumps(r) for r in v2["runs"][n1:]] != [json.dumps(r) for r in p1["runs"]] or len(p1["runs"]) != N:
        bad.append("the pilot rows differ between the two files")
    if any(r["world"] != "sim_honest" for r in p1["runs"] + v2["runs"][n1:]):
        bad.append("a pilot row's inventory world is not sim_honest")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", required=True)  # CUT: <commit>:<path>@<sha256> of format_full_simh_pilot_cutfull_sim_honest_<stamp>.json
    ap.add_argument("--routing", required=True)  # CUT: <commit>:<path>@<sha256> of the pilot's routing / attestation record
    ap.add_argument("--extraction-record", default=f"{EXTRACTION[0]}:{EXTRACTION[1]}@{EXTRACTION[2]}")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    git = lambda *x: subprocess.run(["git", *x], capture_output=True, check=True, cwd=REPO).stdout  # noqa: E731

    def pinned(ref):
        cp, h = ref.rsplit("@", 1)
        b = git("show", cp)
        if sha(b) != h:
            raise SystemExit(f"{cp}: sha256 differs")
        c, p = cp.split(":", 1)
        return json.loads(b), f"{p} @ {c}", {"commit": c, "path": p, "sha256": h}

    fix_b = git("show", f"{FIX_V1[0]}:{FIX_V1[1]}")
    pilot, _, _ = pinned(f"{PILOT[0]}:{PILOT[1]}@{PILOT[2]}")
    up_b = git("show", f"{UPLOAD[0]}:{UPLOAD[1]}")
    if sha(up_b) != UPLOAD[2]:
        raise SystemExit("upload manifest: sha256 differs")
    t196 = {c["key"] for c in pinned(f"{T196[0]}:{T196[1]}@{T196[2]}")[0]["cells"]}
    fin_b = git("show", f"{FINAL[0]}:{FINAL[1]}")
    if sha(fin_b) != FINAL[2]:
        raise SystemExit("FINAL: sha256 differs")
    cohort = set(json.loads(fin_b)["result"]["cohort"]["cohort_cells"])
    ext, ext_ptr, ext_pin = pinned(a.extraction_record)
    fmt, fmt_ptr, fmt_pin = pinned(a.format)
    rt, rt_ptr, rt_pin = pinned(a.routing)
    rt_tree = git("ls-tree", "--full-tree", "--name-only", rt_pin["commit"], str(Path(rt_pin["path"]).parent) + "/").decode().split()
    pins = {"fix_v1": dict(zip(("commit", "path", "sha256"), FIX_V1)), "counted_keys": dict(zip(("commit", "path", "sha256"), PILOT)),
            "upload_manifest": dict(zip(("commit", "path", "sha256"), UPLOAD)), "t196_cells": dict(zip(("commit", "path", "sha256"), T196)),
            "cohort_cells": dict(zip(("commit", "path", "sha256"), FINAL)) | {"field": "result.cohort.cohort_cells"},
            "extraction_record": ext_pin, "format_check": fmt_pin, "routing": rt_pin}
    pilot_out, fix2, stops = build(fix_b, pilot, json.loads(up_b), t196, cohort, ext, fmt, fmt_ptr, rt, rt_ptr, rt_tree, ext_ptr, pins)
    if pilot_out is None:
        raise SystemExit(f"STOP: {stops}")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    p1 = HERE / "results" / f"inventory_b_pilot_rows_v1_{stamp}.json"
    p2 = HERE / "results" / f"inventory_b_fix_v2_{stamp}.json"
    b1 = json.dumps(pilot_out, indent=1).encode()
    fix2["fix_v2"]["pilot_rows_file"] = {"path": p1.name, "sha256": sha(b1)}
    b2 = json.dumps(fix2, indent=1).encode()
    bad = check_written(fix_b, b1, b2)
    status = "STOP" if stops or bad else "COMPLETE"
    if status == "COMPLETE" and not a.dry:
        with p1.open("xb") as fh:
            fh.write(b1)
        with p2.open("xb") as fh:
            fh.write(b2)
    print(json.dumps({"status": status, "dry": a.dry, "pilot_rows": {"file": p1.name, "sha256": sha(b1), "n": pilot_out["n_runs"]},
                      "fix_v2": {"file": p2.name, "sha256": sha(b2), "n": fix2["n_runs"]}, "stops": stops[:8], "post_write_checks": bad}, indent=1))
    return status


if __name__ == "__main__":
    import sys

    sys.exit(0 if main() == "COMPLETE" else 1)
