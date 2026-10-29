"""ug's pre-scoring index for 09 (Append 34; d9 10-15 ~14:50): ONE file keyed "sim_ug|<cell>" with, per registered cell, the designated
run keys and .eval sha256 for run1, run2, sim1 (the comparability map) and sim_ug (the chain's ug manifest and staged set), the sim_ug
label row's key (the labels file is pinned; no label value is copied), and the cell's status under 09's rulings; plus the routing
pointers. Every input is read from git at a pinned commit and checked against its recorded sha256. Keys and hashes only.

  python3 make_ug_prescoring_index.py <out.json>
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
E = "experiments/2026-10-14_q3_overnight_chain"
ARM = "sim_attack@september_ungrounded"
PIN = {  # path @ commit (and sha256 where one was recorded at routing)
    "comparability_map": (f"{E}/results/hygiene/ug_comparability_map_2026-10-15.json", "e1601ac31", "17ac43e3fb22d58b58fd1d35c3aaa7d8492438a1def39431786d8ad5e80a6c29"),
    "manifest": (f"{E}/config/manifest_ug_sim.json", "37a58956c0", None),
    "staging_record": (f"{E}/results/chain_stage_ug_sim_2026-10-15_140343.json", "bade10b7f", None),
    "labels": (f"{E}/results/labels/phase2_attempt_labels_2026-10-15_ug_selfreport_2026-10-15_141455.jsonl", "bade10b7f",
               "a35ef398c5e609f2038eae0d1de6efdab94d2c0c9900aac4aa2572f74eb47644"),
    "run_table": (f"{E}/results/hygiene/run_table_ug.json", "676e7527d", None),
    "input_index_v4": (f"{E}/results/hygiene/input_index_retries_2026-10-15_v4.json", "77d384622", "f035e8b638b52f2888887c1cd682f2dd66c0f3277ad0b7f8cab0a90be74eb045"),
    "attestation": (f"{E}/results/hygiene/attestations_ug_ug.json", "d05bbe7fb", None),
    "approach_provenance": (f"{E}/results/hygiene/approach_provenance_ug_ug.json", "efddd6813", None),
}
UNKNOWN = {"t424_auto_workflow", "t426_shared_dotfiles", "t215_user_tracking", "t470_user_tracking", "t389_model_registry"}  # 09 @ fb3d4649a
ANALYSIS_EXCLUDED = {"t83_auto_workflow", "t389_model_registry"}  # 09 @ 6a36b4a27: reference mismatch, no fallback
NEVER_STARTED = {"t88_model_registry"}  # 09 @ fb3d4649a (not infrastructure; closed @ ed33bf0a7)


def show(name: str) -> tuple[bytes, dict]:
    path, commit, want = PIN[name]
    full = subprocess.run(["git", "rev-parse", f"{commit}^{{commit}}"], capture_output=True, text=True, check=True).stdout.strip()
    raw = subprocess.run(["git", "show", f"{full}:{path}"], capture_output=True, check=True).stdout
    sha = hashlib.sha256(raw).hexdigest()
    if want and sha != want:
        raise SystemExit(f"{name}: sha256 {sha} != recorded {want}")
    return raw, {"path": path, "commit": full[:12], "sha256": sha}


def main():
    pins, data = {}, {}
    for k in PIN:
        raw, pins[k] = show(k)
        data[k] = raw
    cmap = json.loads(data["comparability_map"])["cells"]
    man = [f for f in json.loads(data["manifest"])["files"] if f["arm_dir"] == ARM]
    rec = json.loads(data["staging_record"])
    rec = rec[next(k for k in rec if k != "git")]
    excluded, excl_path = {}, {}  # cell -> reason, cell -> the excluded run's path (its identity, checked against the manifest row below)
    for k, why in rec["c_gate"]["excluded"].items():
        parts = k.split("#", 1)[0].split("/")
        c = parts[parts.index(ARM) + 1]
        assert c not in excluded, c
        excluded[c], excl_path[c] = why, k.split("#", 1)[0]
    staged_sha = rec["assembled"]["staged_inputs_sha256"][f"{ARM}.jsonl"]
    labels, no_run = {}, []
    for line in data["labels"].decode().splitlines():
        r = json.loads(line)
        assert r["arm"] == ARM, r["arm"]
        if r.get("run"):
            assert r["task"] not in labels, r["task"]
            labels[r["task"]] = r["run"]  # the row's key only; no label value is copied
        else:
            no_run.append(r["task"])
    rt_rows = json.loads(data["run_table"])["cells"]
    rt = {c["cell"]: c for c in rt_rows}
    assert len(rt) == len(rt_rows), "duplicate run-table cells"
    by_cell = {}
    for f in man:
        assert f["task"] not in by_cell, f["task"]
        by_cell[f["task"]] = f
    assert set(excluded) == UNKNOWN, sorted(excluded)
    cells_reg = {m["cell"] for m in cmap.values()}
    assert len(cells_reg) == len(cmap) and set(rt) == cells_reg, "map and run table cells differ"
    assert set(by_cell) == cells_reg - NEVER_STARTED, sorted(cells_reg ^ set(by_cell))  # every ran cell has exactly one manifest row; t88 none
    assert set(labels) == set(by_cell) and sorted(no_run) == sorted(NEVER_STARTED), (sorted(set(labels) ^ set(by_cell)), no_run)
    for c, pth in excl_path.items():
        assert pth == by_cell[c]["path"] == f'{by_cell[c]["launch"]}/{by_cell[c]["rel"]}', (c, pth)  # the excluded run IS the manifest's designated run (full path)
    cells = {}
    for key, m in sorted(cmap.items()):
        c = m["cell"]
        assert key == f"sim_ug|{c}" and c in rt, key
        f = by_cell.get(c)
        status = (["never started (09 @ fb3d4649a; not infrastructure)"] if c in NEVER_STARTED else []) + \
                 (["unknown: staging-excluded (09 @ fb3d4649a)"] if c in UNKNOWN else []) + \
                 (["analysis-excluded: reference mismatch, no fallback (09 @ 6a36b4a27)"] if c in ANALYSIS_EXCLUDED else [])
        cells[key] = {
            "cell": c, "origin": m["origin"],
            **{a: ({x: m[a][x] for x in ("row_key", "eval_sha256", "source")} if m[a] else None) for a in ("run1", "run2", "sim1")},
            **{f"{a}_null_reason": m[f"{a}_null_reason"] for a in ("run1", "run2", "sim1") if f"{a}_null_reason" in m},
            "sim_ug": None if f is None else {"row_key": f["path"], "eval_sha256": f["sha256"], "launch": f["launch"], "attempt_kind": f["attempt_kind"],
                                               "staged": c not in excluded, **({"staging_exclusion_reason": excluded[c]} if c in excluded else
                                                                               {"staged_set": "ug_sim_v1_try2", "staged_sha256": staged_sha})},
            "sim_ug_label_row": {"run": labels[c]} if c in labels else None,
            "run_table_ug_status": rt[c].get("ug_status"),
            "status": status or ["usable"],
        }
        if f is not None and c in labels:
            assert labels[c] == f["rel"], (c, labels[c], f["rel"])  # the label row is the designated sim_ug run
    n = {"registered": len(cells), "sim_ug_ran": sum(v["sim_ug"] is not None for v in cells.values()),
         "staged": sum(bool(v["sim_ug"] and v["sim_ug"]["staged"]) for v in cells.values()), "usable": sum(v["status"] == ["usable"] for v in cells.values())}
    assert n == {"registered": 149, "sim_ug_ran": 148, "staged": 143, "usable": 142}, n
    doc = {"what": "ug (sim_ug) pre-scoring index for 09 (Append 34): designated run keys and .eval sha256 per cell for run1, run2, sim1 and sim_ug; "
                   "the sim_ug label row key; each cell's status; routing pointers. Keys and hashes only; no label value or outcome",
           "counts": n, "inputs": pins,
           "routing": {"routed_folder": "ug_sim_v1_try2__sim_attack_ug", "standard_routing": True, "extraction_root": "ug_sim_extract_v1_try2",
                       "staged_set": "ug_sim_v1_try2", "staged_sha256": staged_sha, "volume": "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14",
                       "routed_pt": "2026-10-15 14:19 PT (2e, 4f, 09; chain fork)", "hygiene": "toolerr_ug_ug_sim_ug.json, labels_ug_ug.json @ ee7002ccc",
                       "comparability_b5_b6": "93320011a (+ note d29715164)"},
           "rulings": {"unknown_x5": "fb3d4649a", "t88_never_started": "fb3d4649a (closed @ ed33bf0a7)", "t83_t389_analysis_excluded": "6a36b4a27",
                       "ug_note": "48363962a"},
           "cells": cells}
    Path(sys.argv[1]).open("x").write(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"counts": n, "inputs": {k: v["commit"] for k, v in pins.items()}}))


if __name__ == "__main__":
    main()
