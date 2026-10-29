"""09's basis for reading the su311 Q1 famexp/ folders as they are (09 chose (a): no re-issue, no SUPERSEDED.json): for the 4
folders of pass2_su311_q1_2026-10-14_202036.json, every stored `fold` (per row) and `run_fold` (per run) in all 26 npz equals
famexp_r2's per-probe routing recomputed from the pinned inputs (own_frac / ad_k: Sol #36 fold of the main; b_asis: SA ∪ SH fold
of the main; -1 outside), and the npz field names equal famexp_r2's. KEYS, FOLD IDS AND COUNTS ONLY: score arrays are never
loaded (np.load reads only the named members); each npz is sha256-hashed as bytes against its manifest.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run <this file> --r36 <runs36.json> --message <message_q1.json>
"""

import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-p2probes_q1foldcheck_kill-2026-11-13"
RECORD = ("341ff6cac", "experiments/2026-10-14_p2_family_expansion/results/pass2_su311_q1_2026-10-14_202036.json")
Q2 = ("0caadbfb5", "experiments/2026-10-14_q2_scaleup_cohort/results/scaleup_tasks_plus311_2026-10-14_125135.json",
      "69f1f75313c9610f7d16b18225c1166268c0b6e084859c641d0213f293896a8b")
SIMH = ("573054a4d", "experiments/2026-10-06_q3_attack_detector/results/sim_honest_scores_2026-10-07_124717/runs.json",
        "f420ed2401dcc776cbbcfd414bbe9d6d1132a3bbd012b529a33aba7fa5ecd068")
R36_SHA = "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96"
R2_FIELDS = ["key", "unit", "call_index", "score_by_model", "fold", "run_key", "run_fold", "run_stratum"]  # run_famexp_pass2.py savez
app = modal.App(APP)
WORK = modal.Volume.from_name("probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14", version=2).read_only()
img = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")


@app.function(image=img, cpu=(2, 2), memory=(4096, 4096), timeout=600, retries=0, scaledown_window=2, volumes={"/work": WORK})
def check(folders: dict, expected: dict) -> dict:
    import numpy as np

    out = {}
    for fo, msha in folders.items():
        mp = Path("/work/scaleup") / fo / "famexp" / "manifest_famexp.json"
        raw = mp.read_bytes()
        man = json.loads(raw)
        r = {"manifest": str(mp), "manifest_sha256_equal_record": hashlib.sha256(raw).hexdigest() == msha,
             "superseded_json_present": (mp.parent / "SUPERSEDED.json").exists(), "files": len(man["files"]), "by_variant_readout": {}}
        for p, fsha in sorted(man["files"].items()):
            v, readout = Path(p).stem.split("__")
            exp = expected["b_asis" if v == "b_asis" else "own_ad"]
            b = Path(p).read_bytes()
            with np.load(Path(p), allow_pickle=False) as z:
                names = list(z.files)
                key, fold, rk, rf = (z[n].tolist() for n in ("key", "fold", "run_key", "run_fold"))
            row_bad = sorted({k for k, f in zip(key, fold) if exp.get(k) != f})
            run_bad = sorted(k for k, f in zip(rk, rf) if exp.get(k) != f)
            r["by_variant_readout"][f"{v}|{readout}"] = {
                "npz_sha256_equal_manifest": hashlib.sha256(b).hexdigest() == fsha, "fields": names, "rows": len(key),
                "runs": len(rk), "run_keys": sorted(rk), "row_fold_mismatch_keys": row_bad, "run_fold_mismatch_keys": run_bad}
        out[fo] = r
    return out


@app.local_entrypoint()
def main(r36: str, message: str):
    show = lambda c, p: subprocess.run(["git", "-C", str(HERE), "show", f"{c}:{p}"], capture_output=True, check=True).stdout  # noqa: E731
    rec = json.loads(show(*RECORD))
    rb, mb, q2, sh = Path(r36).read_bytes(), Path(message).read_bytes(), show(*Q2[:2]), show(*SIMH[:2])
    if hashlib.sha256(rb).hexdigest() != R36_SHA or hashlib.sha256(mb).hexdigest() != rec["message_sha256"] \
            or hashlib.sha256(q2).hexdigest() != Q2[2] or hashlib.sha256(sh).hexdigest() != SIMH[2]:
        raise SystemExit("an input differs from its pin")
    msg, main_of = json.loads(mb), {c["key"]: c["main"] for c in json.loads(q2)["cells"]}
    fom = {r["main_task"]: r["fold"] for r in json.loads(rb)}
    fb = defaultdict(set)
    for r in [r for r in json.loads(rb) if r["arm"] == "sim-attack"] + [r for r in json.loads(sh) if r["arm"] == "sim-honest"]:
        fb[r["main_task"]].add(r["fold"])
    if any(len(x) != 1 for x in fb.values()):
        raise SystemExit("b_asis's SA and SH folds disagree for a main")
    fom_b = {m: next(iter(x)) for m, x in fb.items()}
    runs = {f"{e['world']}|{e['cell']}": e["cell"] for e in msg["epochs"]}
    expected = {"own_ad": {k: fom.get(main_of[c], -1) for k, c in runs.items()}, "b_asis": {k: fom_b.get(main_of[c], -1) for k, c in runs.items()}}
    folders = {fo: m["sha256"] for fo, m in rec["result"]["manifests"].items()}
    got = check.remote(folders, expected)
    cells = {e["cell"] for e in msg["epochs"]}
    by_fo = defaultdict(set)
    for e in msg["epochs"]:
        by_fo[e["folder"]].add(f"{e['world']}|{e['cell']}")
    summ = {}
    for fo, r in got.items():
        vr = r["by_variant_readout"]
        summ[fo] = {"manifest_sha256": folders[fo], "manifest_sha256_equal_record": r["manifest_sha256_equal_record"],
                    "superseded_json_present": r["superseded_json_present"], "variant_readouts": len(vr),
                    "variants": sorted({k.split("|")[0] for k in vr}),
                    "all_npz_sha256_equal_manifest": all(x["npz_sha256_equal_manifest"] for x in vr.values()),
                    "message_runs": len(by_fo[fo]), "runs_per_variant_readout": sorted({x["runs"] for x in vr.values()}),
                    "run_keys_subset_of_message_runs": all(set(x["run_keys"]) <= by_fo[fo] for x in vr.values()),
                    "runs_checked": sum(x["runs"] for x in vr.values()), "rows_checked": sum(x["rows"] for x in vr.values()),
                    "row_fold_mismatch_keys": {k: x["row_fold_mismatch_keys"] for k, x in vr.items() if x["row_fold_mismatch_keys"]},
                    "run_fold_mismatch_keys": {k: x["run_fold_mismatch_keys"] for k, x in vr.items() if x["run_fold_mismatch_keys"]},
                    "field_sets": sorted({json.dumps(x["fields"]) for x in vr.values()})}
    git = lambda *a: subprocess.run(["git", "-C", str(HERE), *a], capture_output=True, text=True, check=True).stdout.strip()  # noqa: E731
    res = {"label": "keys, fold ids and counts only; score arrays never loaded",
           "basis_for": "09 (a): read the su311 Q1 famexp/ folders as they are (no re-issue, no SUPERSEDED.json)",
           "stamp (PT)": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S"), "git": git("rev-parse", "HEAD"),
           "app": APP, "record": {"commit": RECORD[0], "path": RECORD[1]},
           "inputs": {"runs36": {"file": "runs36.json (Sol #36)", "sha256": R36_SHA}, "message_sha256": rec["message_sha256"],
                      "q2_list": dict(zip(("commit", "path", "sha256"), Q2)), "simh": dict(zip(("commit", "path", "sha256"), SIMH))},
           "rule": "own_frac*/ad_k*: Sol #36 fold of the cell's main (-1 if absent); b_asis: SA (runs36 sim-attack) ∪ SH (simh sim-honest) fold of the main (-1 if absent); = run_famexp_pass2.py famexp_r2",
           "cells": len(cells), "message_runs": len(runs),
           "cells_where_b_asis_fold_differs_from_own_ad_fold": sorted(c for k, c in runs.items() if expected["own_ad"][k] != expected["b_asis"][k]),
           "expected_fold_minus1_runs": {p: sum(f < 0 for f in e.values()) for p, e in expected.items()},
           "famexp_r2_fields": R2_FIELDS,
           "field_names_equal_famexp_r2": all(s == [json.dumps(R2_FIELDS)] for s in (x["field_sets"] for x in summ.values())),
           "total_mismatches": sum(len(v) for x in summ.values() for d_ in (x["row_fold_mismatch_keys"], x["run_fold_mismatch_keys"]) for v in d_.values()),
           "folders": summ}
    out = HERE / "results" / f"q1_famexp_routing_equivalence_{res['stamp (PT)']}.json"
    out.write_text(json.dumps(res, indent=1))
    print(out, json.dumps({k: res[k] for k in ("cells", "message_runs", "cells_where_b_asis_fold_differs_from_own_ad_fold", "expected_fold_minus1_runs", "field_names_equal_famexp_r2", "total_mismatches")}))
    for fo, x in summ.items():
        print(fo, {k: x[k] for k in ("manifest_sha256_equal_record", "superseded_json_present", "variant_readouts", "all_npz_sha256_equal_manifest", "message_runs", "runs_per_variant_readout", "run_keys_subset_of_message_runs", "runs_checked", "rows_checked")},
              "mismatch entries", len(x["row_fold_mismatch_keys"]) + len(x["run_fold_mismatch_keys"]))
