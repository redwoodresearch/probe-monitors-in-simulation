"""sel97 donor gate (d9 for the design fork, 10-15 ~13:00; the selection-half IL add-on, registration @ 273470963): Census E's method
(census_il_gate_a.py @ 86b8f2129) for the 96 keys of tasks_sel96.json @ b246ffb9c, before the add-on starts. Hashes only:
- (a1) the grounding reference's sha256 (il_grounding_references @ 193a8cfc4, launch 2026-10-14_il_old, read from il_refindex_old) equals
  the pin the chain's box read verified: the cell's honest_a donor row in the il_old family's committed manifests (box_files re-hashes
  each reference copy against the index before it writes the row). Missing or differing is an ERROR.
- (a2) the system-prompt sha256 of every honest donor row of the cell equals the grounding reference's (both non-empty), extracted with
  the stager's own functions (stage_q4b.load, transcripts.system_prompt); an empty prompt or an unreadable file is an ERROR.
- (b) within-cell attack system-prompt equality over the cell's T196 attack runs from batch B's upload manifest
  (config/manifest_q4b_eval_upload_v1.json): run1 (the main launches' attack), run2 (the second attack), sim1 (sim_attack@september);
  every prompt non-empty and all equal; each file's sha256 checked against the manifest.
Prompts are hashed inside the Modal job; only keys, hashes and pass/fail leave it.

  PYTHONPATH=src modal run experiments/2026-10-14_q3_overnight_chain/census_sel97_gate.py --out <json>
"""
import hashlib
import json
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE  # the container imports this file from /root (no repo there)
VOL = "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14"
SRC = "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13"  # 2e's Volume (batch B's .eval files), read only
SEL = ("b246ffb9c", "experiments/2026-10-15_q3_interleaved_run3_sim2/results/tasks_sel96.json")
GROUNDING = ("193a8cfc4", "experiments/2026-10-15_q3_interleaved_run3_sim2/results/il_grounding_references_2026-10-15.json")
B_MAN = HERE / "config" / "manifest_q4b_eval_upload_v1.json"
MAIN = ["2026-10-09_000523_rerun_strict", "2026-10-09_173815_scaleup_canary", "2026-10-09_182505_scaleup", "2026-10-10_001314_scaleup_topup"]
RUN2 = "2026-10-10_001314_second_attack"
app = modal.App("probe-shift_aswin-surya_sub-q3su311_sel97gate_kill-2026-10-22")
image = (modal.Image.debian_slim(python_version="3.13").uv_pip_install("numpy", "jinja2", "zstandard", "boto3", "pyyaml")
         .add_local_python_source("probe_shift").add_local_file(HERE / "stage_q4b.py", "/root/repo/q4/stage_q4b.py")
         .add_local_file(HERE / "evalzip.py", "/root/repo/q4/evalzip.py"))


def show(c: str, p: str) -> bytes:
    return subprocess.run(["git", "show", f"{c}:{p}"], capture_output=True, check=True, cwd=REPO).stdout


@app.function(image=image, cpu=8, memory=32768, timeout=3600, retries=0,
              volumes={"/ev": modal.Volume.from_name(VOL, version=2).read_only(), "/src": modal.Volume.from_name(SRC, version=2).read_only()})
def census(cells: list) -> list:
    import sys as _s

    _s.path.insert(0, "/root/repo/q4")
    import stage_q4b as S

    def read(root: str, rel: str):  # (file sha256, system-prompt sha256 or None if empty); the prompt itself never leaves
        p = Path(root) / rel
        raw = p.read_bytes()
        sp = S.transcripts.system_prompt(S.load(p))
        return hashlib.sha256(raw).hexdigest(), (hashlib.sha256(sp.encode()).hexdigest() if sp else None)

    out = []
    for c in cells:
        errs, row = [], {"key": c["key"], "grounding_sha256": c["grounding_sha256"], "pinned_sha256": c["pinned_sha256"]}
        row["a1"] = ("pass" if c["grounding_sha256"] and c["grounding_sha256"] == c["pinned_sha256"] else
                     "ERROR: missing sha" if not (c["grounding_sha256"] and c["pinned_sha256"]) else "FAIL")
        if row["a1"] != "pass":
            errs.append(f"a1 {row['a1']}")
        ref_sp, cands = None, {}
        for arm, d in sorted(c["donors"].items()):
            try:
                fs, sp = read("/ev/evals_v1", d["path"])
            except Exception as e:  # unreadable: an ERROR (class name only)
                errs.append(f"{arm} donor unreadable: {type(e).__name__}")
                continue
            cands[arm] = {"file_sha256": fs, "file_sha256_ok": fs == d["sha256"], "is_grounding_reference": fs == c["grounding_sha256"], "system_sha256": sp}
            if fs != d["sha256"]:
                errs.append(f"{arm} donor file differs from its manifest row")
            if fs == c["grounding_sha256"]:
                ref_sp = sp
        if ref_sp is None:
            errs.append("grounding reference not among the donor rows, or its system prompt is empty")
        for arm, v in cands.items():
            v["a2"] = ("ERROR: no reference prompt" if ref_sp is None else "empty donor prompt" if v["system_sha256"] is None
                       else "pass" if v["system_sha256"] == ref_sp else "FAIL")
            if v["a2"] != "pass":
                errs.append(f"{arm} a2 {v['a2']}")
        row["donors"] = cands
        att = {}
        for w, d in sorted(c["attack"].items()):
            try:
                fs, sp = read("/src/evals_v1", d["path"])
            except Exception as e:
                errs.append(f"{w} attack unreadable: {type(e).__name__}")
                continue
            att[w] = {"file_sha256": fs, "file_sha256_ok": fs == d["sha256"], "system_sha256": sp}
            if fs != d["sha256"]:
                errs.append(f"{w} attack file differs from batch B's manifest")
        shas = {v["system_sha256"] for v in att.values()}
        row["attack"] = att
        missing = [w for w in ("run1", "run2", "sim1") if w not in att]
        row["b"] = (f"ERROR: missing {missing}" if missing else "ERROR: empty attack prompt" if None in shas
                    else "pass" if len(shas) == 1 else "FAIL")
        if row["b"] != "pass":
            errs.append(f"b {row['b']}")
        row["errors"] = errs
        row["pass"] = not errs
        out.append(row)
    return out


@app.local_entrypoint()
def main(out: str):
    sel_raw, g_raw, b_raw = show(*SEL), show(*GROUNDING), B_MAN.read_bytes()
    keys = [c if isinstance(c, str) else (c.get("key") or c.get("cell")) for c in json.loads(sel_raw)["cells"]]
    assert len(keys) == len(set(keys)) == 96, len(keys)
    (gl,) = [x for x in json.loads(g_raw)["launches"] if x["launch_run_dir"].rstrip("/").endswith("2026-10-14_il_old")]
    donors, pins = {}, {}
    for m in sorted((HERE / "config").glob("manifest_ilold_*.json")):
        for r in json.loads(m.read_text())["files"]:
            if r["task"] in keys and r["arm_dir"] in ("honest_a", "honest_b"):
                prev = donors.setdefault(r["task"], {}).setdefault(r["arm_dir"], {"path": r["path"], "sha256": r["sha256"]})
                assert prev["sha256"] == r["sha256"], (r["task"], r["arm_dir"], "two donor rows differ across manifests")
                if r["arm_dir"] == "honest_a":
                    pins[r["task"]] = r["sha256"]
    att = {}
    for r in json.loads(b_raw)["files"]:
        if r["task"] not in keys:
            continue
        w = ("run2" if r["launch"] == RUN2 and r["arm_dir"] == "attack" else "run1" if r["launch"] in MAIN and r["arm_dir"] == "attack"
             else "sim1" if r["launch"] in MAIN and r["arm_dir"] == "sim_attack@september" else None)
        if w:
            assert w not in att.setdefault(r["task"], {}), (r["task"], w, "two counted attack runs")
            att[r["task"]][w] = {"path": r["path"], "sha256": r["sha256"]}
    cells = [{"key": k, "grounding_sha256": (gl["references"].get(k) or {}).get("sha256"), "pinned_sha256": pins.get(k),
              "donors": donors.get(k, {}), "attack": att.get(k, {})} for k in keys]
    res = census.remote(cells)
    failing = {r["key"]: r["errors"] for r in res if not r["pass"]}
    doc = {"check": "sel97 donor gate (a1, a2, b), Census E's method, hashes only",
           "inputs": {"keys": {"commit": SEL[0], "path": SEL[1], "sha256": hashlib.sha256(sel_raw).hexdigest()},
                      "grounding": {"commit": GROUNDING[0], "path": GROUNDING[1], "sha256": hashlib.sha256(g_raw).hexdigest()},
                      "batch_b_manifest": {"path": str(B_MAN.relative_to(REPO)), "sha256": hashlib.sha256(b_raw).hexdigest()},
                      "il_old_manifests": sorted(p.name for p in (HERE / "config").glob("manifest_ilold_*.json"))},
           "counts": {"keys": len(res), "pass": sum(r["pass"] for r in res), "a1_pass": sum(r["a1"] == "pass" for r in res),
                      "b_pass": sum(r["b"] == "pass" for r in res), "attack_worlds": {w: sum(w in r["attack"] for r in res) for w in ("run1", "run2", "sim1")}},
           "failing": failing, "cells": res}
    Path(out).open("x").write(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"counts": doc["counts"], "failing": failing}, indent=1))
    if failing:
        raise SystemExit(f"{len(failing)} keys fail the gate (record written)")
