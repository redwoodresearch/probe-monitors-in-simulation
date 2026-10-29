"""ug B5 (render-donor candidate mapping with identity binding) and B6 (gate (a1)/(a2) against the frozen fork-source
reference): 09's Append 32 format note §2 (ug), d9 10-15 ~13:30, the chain fork's answers ~13:40. Laptop; reads only
committed inputs (git show at pinned commits) and the committed box output of ug_b5_index_pin.py. Keys, paths, hashes,
attempt kinds and booleans only.

  python3 ug_b5_b6_mapping.py <ug_b5_index_pin json> --out <json>

Inputs (each recorded with its sha256):
  - the index pin (il_refindex_ug, sha256 36c2b273...): per cell the index copy, its sha256 and source, and the main's pin;
  - the frozen launch manifest (Append 34 @ 71c308096): each cell's fork-source reference record;
  - counted candidates. S_new: the su311 main manifests on aswin/q3-overnight-chain @ CHAIN (rows arm_dir honest_a /
    honest_b). S_old: correctedb_manifest_v1.json @ 46ca25fc4 epochs join honest_a|<cell> / honest_b|<cell>, key without
    "#e0" joined to manifest_q4b_eval_upload_v1.json @ 581ffb229 files[].path for sha256 and attempt fields (09's Q1 note);
  - Census E @ 86b8f2129 (the IL (a1)/(a2) census): the reference's stager-extracted system-prompt sha256, reused for a
    cell only where its grounding_sha256 equals ug's index copy sha256 (the chain fork's condition).

Per cell:
  B5: the index honest_a reference reconciled with the counted honest_a by HASH (equal) or IDENTITY (same launch, arm
      honest_a and task, a different attempt directory: e.g. the launcher's later usable attempt or a reference rerun,
      run_pilot.py:490-500), else an error: counted candidate missing, ambiguous (more than one counted row), or a mismatch
      (a different launch, arm or task). honest_b is listed with its hash and identity; ug renders from the index copy
      only (the chain reads ug's main_run @index, honest_a rows), so honest_b is never selected.
  B6: (a1) the frozen fork-source reference sha256 == the main's pinned sha256 == the index copy sha256.
      (a2) the render donor is the index copy; the fork-source reference is the frozen manifest's reference_record.copy.
      Same file (path and sha256) reduces (a2) to "the same file, with a non-empty stager-extracted system prompt", the
      prompt sha256 taken from Census E where its grounding_sha256 equals the index copy's. A cell with no such Census E
      row is an explicit "own read needed" row (an error unless the cell has no ug run to stage, listed as such).
Fail-closed (Codex r28): the box output's pins (PINS) and empty `problems` are enforced, and the frozen 149 cells must
each appear exactly once. A counted candidate must be exactly ONE row (S_old: exactly one epoch, its key ending "#e0",
with exactly one upload match). honest_b problems are errors too. Identity is checked on both branches: hash-equal requires
the same launch, arm, task, attempt directory and file; a hash difference reconciles only for the same launch, arm and
task with a DIFFERENT attempt directory or file. Census E evidence must be verified bytes (file_sha256_ok) with one
non-empty prompt sha256 across its matching rows.
Exit 1 if any cell or input check fails; the output file is written either way.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import subprocess

CHAIN = "a918466e1"
CHAIN_CFG = "experiments/2026-10-14_q3_overnight_chain/config"
CORRB = ("46ca25fc4", "experiments/2026-10-14_q35_sol_avg_POSTHOC/results/correctedb_manifest_v1.json")
UPLOAD = ("581ffb229", "experiments/2026-10-13_q4_why_weaker_POSTHOC/results/manifest_q4b_eval_upload_v1.json")
CENSUS_E = ("86b8f2129", "experiments/2026-10-14_q3_overnight_chain/results/il_donor_check/il_census_e_2026-10-15_022645.json")
FROZEN = ("71c308096", "experiments/2026-10-15_q3_ungrounded_sim/results/ug_frozen_manifest_2026-10-15_005643.json")
RUNS = "/home/aswin/phase2/runs/"
PINS = {"ug_index": "36c2b2736908d7d6a028e101fe95cd1ff17da35343c0678d62b6b69ac5f6ad8f",
        "tasks": "cf89e595a37b85375ba1a84796718211d07b48a6e6f697c69a5faad72f76b80b",
        "/home/aswin/phase2/il_refindex_old/launch.json": "79acbfc1ef4523f9a57cc19dc876252159c6c1d62521001f1ac71e2644036940",
        "/home/aswin/phase2/refs_ug/q1_launch.json": "7e8227cd859502e60ad718fc60215410ece9d4b86eea04040bd7bee0a82c53ac",
        "/home/aswin/phase2/refs_ug/q2_launch.json": "fa489bf06d2be9c4a2b5c432ded7fb6065df49763618fb19e96a5f5d01cf03fb",
        "/home/aswin/phase2/refs_ug/q3_launch.json": "289ffb1030625f24666b1c9fb86a2ad577cf66a55407b9f18db125f0ed121701",
        "/home/aswin/phase2/refs_ug/q4_launch.json": "f12ed4cd86ed516036bcc47857c7a4d90ba95e21595a8c8f484653b9055a9ad6"}
NOT_STAGED = {"t88_model_registry": "never started (09 @ fb3d4649a); no sim_ug artifact to render"}


def blob(commit: str, path: str) -> tuple[dict, str]:
    raw = subprocess.run(["git", "show", f"{commit}:{path}"], check=True, capture_output=True).stdout
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def ident(rel: str | None):
    """(launch, arm, task, attempt_dir) from '<launch>/<arm>/<task dir>/<file>.eval' (a runs-relative path)."""
    if not rel:
        return None
    p = rel.split("/")
    if len(p) < 4:
        return None
    task = re.sub(r"_(redraw|rerun)\d+$", "", p[2])
    return {"launch": p[0], "arm": p[1], "task": task, "attempt_dir": p[2], "file": p[3]}


def one(rows: list[dict]):
    """Exactly one counted row (Codex r28: several rows are ambiguous even when their hashes agree)."""
    if not rows:
        return None, "missing"
    if len(rows) > 1:
        return None, "ambiguous"
    return rows[0], "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("index_pin")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    raw_pin = open(a.index_pin, "rb").read()
    pin = json.loads(raw_pin)
    frozen, frozen_sha = blob(*FROZEN)
    fz = {c["key"]: c for c in frozen["cells"]}
    mains, inputs = {}, {"index_pin": hashlib.sha256(raw_pin).hexdigest(), f"{FROZEN[1]} @ {FROZEN[0]}": frozen_sha}
    for k in (1, 2, 3, 4):
        p = f"{CHAIN_CFG}/manifest_su311_q{k}_main.json"
        mains[k], inputs[f"{p} @ {CHAIN}"] = blob(CHAIN, p)
    corrb, inputs[f"{CORRB[1]} @ {CORRB[0]}"] = blob(*CORRB)
    upload, inputs[f"{UPLOAD[1]} @ {UPLOAD[0]}"] = blob(*UPLOAD)
    ce, inputs[f"{CENSUS_E[1]} @ {CENSUS_E[0]}"] = blob(*CENSUS_E)
    input_errors = []
    if pin["ug_index"]["sha256"] != PINS["ug_index"] or frozen["reference_index"]["sha256"] != PINS["ug_index"]:
        input_errors.append("il_refindex_ug sha256 is not the frozen pin")
    if pin["tasks_file"]["sha256"] != PINS["tasks"]:
        input_errors.append("the ug task file sha256 is not its pin")
    for path, h in pin["launch_jsons"].items():
        if PINS.get(path) != h:
            input_errors.append(f"index source {path}: sha256 is not its pin")
    if pin.get("problems"):
        input_errors.append(f"the box output lists {len(pin['problems'])} problems")
    keys = [r["cell"] for r in pin["rows"]]
    if len(keys) != len(set(keys)) or set(keys) != set(fz) or len(fz) != 149:
        input_errors.append("coverage: the box rows are not exactly the frozen 149 cells, once each")
    up_by_path = collections.defaultdict(list)
    for f in upload["files"]:
        up_by_path[f["path"]].append(f)
    ce_by = collections.defaultdict(list)
    for c in ce["result"]["cells"]:
        ce_by[c["cell"]].append(c)

    def counted(cell: str, origin: str, arm: str) -> tuple[dict | None, str]:
        if origin.startswith("S_new"):
            k = int(origin[-1])
            rows = [f for f in mains[k]["files"] if f.get("task") == cell and f.get("arm_dir") == arm]
            got, st = one(rows)
            return (got and {k2: got.get(k2) for k2 in ("path", "sha256", "attempt_index", "attempt_kind", "launch")}), st
        eps = [e for e in corrb["epochs"] if e.get("join") == f"{arm}|{cell}"]
        if len(eps) != 1:
            return None, "missing" if not eps else "ambiguous"
        key = eps[0]["key"]
        if not key.endswith("#e0"):
            return None, "unreadable"  # not the specified .eval#e0 form
        got, st = one(up_by_path.get(key[: -len("#e0")], []))
        return (got and {k2: got.get(k2) for k2 in ("path", "sha256", "attempt_index", "attempt_kind", "launch")}), st

    out_rows, summary = [], collections.Counter()
    for r in pin["rows"]:
        cell, idx, errors = r["cell"], r["index"], []
        src = idx["index_source"]
        m = re.search(r"refs_ug/q([1-4])_launch\.json$", src)
        origin = f"S_new q{m.group(1)}" if m else ("S_old" if src.endswith("il_refindex_old/launch.json") else None)
        if origin is None:
            errors.append("origin not determinable from the index source")
        ref_rel = (idx.get("source") or "").removeprefix(RUNS) if (idx.get("source") or "").startswith(RUNS) else None
        cand = {}
        for arm in ("honest_a", "honest_b"):
            got, st = counted(cell, origin or "", arm)
            cand[arm] = {"status": st, **(got or {})}
        ha = cand["honest_a"]
        if cand["honest_b"]["status"] != "ok":
            errors.append(f"counted honest_b {cand['honest_b']['status']}")
        ri, ci = ident(ref_rel), ident(ha.get("path"))
        same_run = bool(ri and ci and ri["launch"] == ci["launch"] and ri["arm"] == ci["arm"] == "honest_a"
                        and ri["task"] == ci["task"] == cell)
        same_attempt = same_run and ri["attempt_dir"] == ci["attempt_dir"] and ri["file"] == ci["file"]
        if ha["status"] != "ok":
            recon = ha["status"]
            errors.append(f"counted honest_a {ha['status']}")
        elif ha["sha256"] == idx["sha256"]:
            recon = "hash_equal" if same_attempt else "mismatch"
            if not same_attempt:
                errors.append("index reference and counted honest_a: equal hash but a different identity")
        elif same_run and not same_attempt:
            recon = "same_run_different_attempt"
        else:
            recon = "mismatch"
            errors.append("index reference vs counted honest_a: a hash difference not explained by a different attempt")
        fzr = (fz.get(cell) or {}).get("reference_record") or {}
        a1 = bool(fzr.get("sha256")) and fzr.get("sha256") == r["main_pinned"]["sha256"] == idx["sha256"] == r["copy_sha256"]
        if not a1:
            errors.append("(a1) fails")
        same_file = fzr.get("copy") == idx.get("copy") and fzr.get("sha256") == idx.get("sha256")
        ce_rows = [c for c in ce_by.get(cell, []) if c.get("grounding_sha256") == idx["sha256"]]
        prompts = [c["reference"].get("system_sha256") for c in ce_rows]
        verified = bool(ce_rows) and all(c["reference"].get("file_sha256_ok") is True for c in ce_rows) and all(prompts)
        prompt = set(prompts) if verified else set()
        if ce_rows and not verified:
            errors.append("Census E: a matching row with unverified bytes or an empty prompt")
        if same_file and len(prompt) == 1:
            a2 = "pass: same file, non-empty system prompt (Census E)"
        elif not same_file:
            a2 = "fail: render donor is not the fork-source reference file"
            errors.append("(a2) fails")
        elif len(prompt) > 1:
            a2 = "fail: Census E rows disagree on the reference prompt"
            errors.append("(a2) ambiguous")
        elif cell in NOT_STAGED:
            a2 = f"not evaluated: {NOT_STAGED[cell]}; no Census E row with this reference"
        else:
            a2 = "own read needed: no Census E row with this reference"
            errors.append("(a2) own read needed")
        row = {"cell": cell, "origin": origin, "index": {**idx, "source_rel": ref_rel}, "source_kind": r["source_kind"],
               "main_pinned": r["main_pinned"], "frozen_reference": {"copy": fzr.get("copy"), "sha256": fzr.get("sha256")},
               "candidates": cand, "b5_reconcile_honest_a": recon,
               "honest_b_equals_reference": cand["honest_b"].get("sha256") == idx["sha256"] if cand["honest_b"]["status"] == "ok" else None,
               "render_donor": "the index copy (honest_a; the chain reads ug's main_run @index)",
               "b6_a1": a1, "b6_a2": a2, "b6_a2_same_file": same_file,
               "census_e": [{"launch": c["launch"], "reference_system_sha256": c["reference"].get("system_sha256")} for c in ce_rows],
               "errors": errors}
        out_rows.append(row)
        summary[f"b5 {recon}"] += 1
        summary[f"a1 {'pass' if a1 else 'fail'}"] += 1
        summary[f"a2 {a2.split(':')[0]}"] += 1
        summary["cells with errors"] += bool(errors)
    summary["input errors"] = len(input_errors)
    out = {"spec": "09 Append 32 format note §2 (ug render donors with identity binding); d9 B5/B6", "input_errors": input_errors,
           "ug_index_sha256": pin["ug_index"]["sha256"], "inputs_sha256": inputs, "n_cells": len(out_rows),
           "summary": dict(sorted(summary.items())), "rows": out_rows}
    with open(a.out, "x") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["summary"]))
    raise SystemExit(1 if summary["cells with errors"] or input_errors else 0)


if __name__ == "__main__":
    main()
