"""Q4 (B) review checks, second sweep (Codex threads on #1512 and #1513), counts only, no text, CPU only:
- coverage: Sol cells, #36's runs (honest, both real attack runs, simulated attack) with and without a meta in the dump
  (run_q4b_cells.py skips a run without one); own cells, meta files against staged epochs and claim C's labels, and
  output_end calls without a row;
- cut: Qwen3.5 reads Sol's runs uncut; what the other cells' 32,768-token cut would remove from it (rows past the cut, cohort
  runs whose first 3 or 21 output_end rows reach past it, units and honest runs that would leave the cohort); the same
  rebuild on Qwen3-32B's Sol dump, which is already cut, checks the rebuild;
- false alarms: the observed honest false-alarm rate at the nominal 1% and 5% thresholds, read from the four cell results.
Every rebuilt cohort and packed count must equal the as-run cell result, or the check stops.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 --with numpy modal run --detach \\
      experiments/2026-10-13_q4_why_weaker_POSTHOC/check_q4b_review2.py --labels <claimc_labels_reduced.json>
"""

import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-q4own_review_kill-2026-11-13"
VOLS = {"la": "probe-shift_aswin-surya_sub-la-activations", "q35": "probe-shift_aswin-surya_sub-qwen35_acts_kill-2026-11-06",
        "xread": "probe-shift_aswin-surya_sub-q3xread_acts_kill-2026-11-13", "own": "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13"}
H_, E1, E2, SA = "real-honest", "real-attack-e1", "real-attack-e2", "sim-attack"
ARMS, ATT = (H_, E1, E2, SA), (E1, E2, SA)
WORLD_OF = {E1: "run1", E2: "run2", SA: "sim"}
DUMPS = {"q35_sol": ("q35", "2026-10-06_la_segment-end-v4-scrub1_qwen35-27b_alllayers_v1"),
         "q3_sol": ("la", "2026-10-06_la_segment-end-v4-scrub1_qwen3-32b_alllayers_v1")}
OWN = {"q3_own": ("xread", [("2026-10-13_xread_main_v1", "attack", E1), ("2026-10-13_xread_run2_v1", "attack", E2),
                            ("2026-10-13_xread_main_v1", "sim_attack@september", SA), ("2026-10-13_xread_main_v1", "honest_a", H_)]),
       "q35_own": ("own", [("q4b_extract_main_v1", "attack", E1), ("q4b_extract_run2_v1", "attack", E2),
                           ("q4b_extract_main_v1", "sim_attack@september", SA), ("q4b_extract_main_v1", "honest_a", H_)])}
CELLS = {"q35_sol": "q4b_cell_q35_sol_POSTHOC_2026-10-14_002939.json", "q3_sol": "q4b_cell_q3_sol_POSTHOC_2026-10-14_002919.json",
         "q3_own": "q4b_cell_q3_own_POSTHOC_2026-10-14_002759.json", "q35_own": "q4b_cell_q35_own_POSTHOC_2026-10-14_003924.json"}
KIND, CUT, MIN_ROWS = "output_end", 32768, 21
R36 = "experiments/2026-10-06_q3_attack_detector/results/run_scores_2026-10-06_181548/runs.json"
LABELS_SHA = "d00e88773fc2a71bef51e1ff0ccc2559932f5c3a306a8f5604c5248e797dc995"

app = modal.App(APP)
vols = {k: modal.Volume.from_name(v, version=2) for k, v in VOLS.items()}
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")
         .add_local_file(REPO / R36, "/root/runs36.json"))
MOUNTS = {f"/{k}": v.read_only() for k, v in vols.items()}


class Gate(Exception):
    """A rebuilt count differs from the as-run result; its message holds numbers only."""


def cohort(arm, unit, n):  # run_q4b_cells.run_cell's cohort rule
    import numpy as np

    full = {u for u in set(unit[np.isin(arm, ATT)]) if all(((unit == u) & (arm == a) & (n >= MIN_ROWS)).any() for a in ATT)}
    return full, np.where(np.isin(arm, ATT), np.isin(unit, sorted(full)), (arm == H_) & (n >= MIN_ROWS))


@app.function(image=image, cpu=4, memory=16384, timeout=1800, retries=0, volumes=MOUNTS)
def sol(cell: str, as_run: dict) -> dict:
    import numpy as np

    vk, dump = DUMPS[cell]
    root = Path(f"/{vk}") / dump
    base = [r for r in json.loads(Path("/root/runs36.json").read_text()) if r["arm"] in ARMS]
    meta = {}
    for a in ARMS:
        for p in (root / "meta" / a).glob("*.json"):
            m = json.loads(p.read_text())
            meta[(a, m["key"])] = m
    k36 = {(r["arm"], r["key"]) for r in base}
    cov = {a: {"runs_in_36": sum(r["arm"] == a for r in base), "with_meta": sum(r["arm"] == a and (a, r["key"]) in meta for r in base),
               "meta_not_in_36": sum(k[0] == a and k not in k36 for k in meta)} for a in ARMS}
    kept = [(r["arm"], r["cell"], [p["token"] for p in meta[(r["arm"], r["key"])]["positions"] if p["kind"] == KIND and p["kept"]])
            for r in base if (r["arm"], r["key"]) in meta]
    arm, unit = np.array([k[0] for k in kept]), np.array([k[1] for k in kept])
    tok = [np.array(k[2], np.int64) for k in kept]
    n = np.array([len(t) for t in tok])
    full, coh = cohort(arm, unit, n)
    rebuilt = {"units": len(full), "runs": {a: int((coh & (arm == a)).sum()) for a in ARMS}}
    if rebuilt != as_run["cohort"] or [cov[a]["with_meta"] for a in ARMS] != [as_run["packs"][0]["runs"][a] for a in ARMS]:
        raise Gate(f"{cell}: rebuilt {rebuilt} {[cov[a]['with_meta'] for a in ARMS]}")
    n_cut = np.array([(t < CUT).sum() for t in tok])
    full_c, coh_c = cohort(arm, unit, n_cut)
    past = lambda t, j: len(t) >= j and t[j - 1] >= CUT  # noqa: E731
    per_arm = {a: {"runs": int((arm == a).sum()), "rows": int(n[arm == a].sum()), "rows_past_cut": int((n - n_cut)[arm == a].sum()),
                   "runs_with_rows_past_cut": int(((n > n_cut) & (arm == a)).sum()),
                   "cohort_runs": int((coh & (arm == a)).sum()), "cohort_runs_with_rows_past_cut": int(((n > n_cut) & coh & (arm == a)).sum()),
                   "cohort_runs_3rd_row_past_cut": sum(past(tok[i], 3) for i in np.flatnonzero(coh & (arm == a))),
                   "cohort_runs_21st_row_past_cut": sum(past(tok[i], 21) for i in np.flatnonzero(coh & (arm == a))),
                   "cohort_runs_leaving_under_cut": int((coh & ~coh_c & (arm == a)).sum()),
                   "runs_joining_under_cut": int((~coh & coh_c & (arm == a)).sum())} for a in ARMS}
    return {"cell": cell, "dump": dump, "coverage": cov, "cohort_as_run_rebuilt": rebuilt,
            "max_kept_token": int(max(int(t.max()) for t in tok if len(t))), "runs_tokens_not_ascending": int(sum(bool(np.any(np.diff(t) <= 0)) for t in tok)),
            "runs_truncated_flag": {a: dict(Counter(str(meta[(r["arm"], r["key"])].get("truncated")) for r in base if r["arm"] == a and (a, r["key"]) in meta)) for a in ARMS},
            "cut": CUT, "units_under_cut": len(full_c), "units_leaving_under_cut": len(full - full_c), "per_arm": per_arm}


@app.function(image=image, cpu=4, memory=16384, timeout=1800, retries=0, volumes=MOUNTS)
def own(lab: dict, as_run: dict, cells: dict) -> dict:
    out = {"staged_epochs": {f"{s}/{p.name}": sum(1 for _ in p.open()) for s in ("q4b_main_v1", "q4b_run2_v1")
                             for p in sorted(Path(f"/own/{s}/inputs").glob("*.jsonl"))}}
    for cell, (vk, runs) in OWN.items():
        rows, packed = [], {}
        for run, arm_dir, arm in runs:
            ms = [json.loads(p.read_text()) for p in sorted(Path(f"/{vk}/{run}/meta/{arm_dir}").glob("*.json"))]
            tasks = [m["task"] for m in ms]
            w = WORLD_OF.get(arm)
            att = {k.split("|", 1)[1] for k, v in lab.items() if w and k.split("|", 1)[0] == w and v == "attempted"}
            packed[arm] = len(ms) if arm == H_ else sum(t in att for t in tasks)
            rows.append({"run": run, "arm_dir": arm_dir, "arm": arm, "meta_files": len(ms), "distinct_tasks": len(set(tasks)),
                         "packed": packed[arm], "attempted_labels": len(att) if w else None,
                         "attempted_labels_without_meta": len(att - set(tasks)) if w else None,
                         "output_end_calls": sum(c["kind"] == KIND for m in ms for c in m["calls"]),
                         "output_end_calls_without_row": sum(c["kind"] == KIND and c.get("row") is None for m in ms for c in m["calls"])})
        if [packed[a] for a in ARMS] != [as_run[cell]["packs"][0]["runs"][a] for a in ARMS]:
            raise Gate(f"{cell}: packed {[packed[a] for a in ARMS]}")
        out[cell] = rows
    fa = {}
    for cell, d in cells.items():
        for key, m in d["metrics"].items():
            if "__cohort_" in key:
                fa[f"{cell}__{key}"] = {f"{a:g}": m[f"fpr_{a:g}"]["real_honest_fa"] for a in (0.01, 0.05)}
    out["observed_honest_fa"] = fa
    return out


@app.local_entrypoint()
def main(labels: str):
    if hashlib.sha256(Path(labels).read_bytes()).hexdigest() != LABELS_SHA:
        raise SystemExit("labels sha256 mismatch")
    lab = json.loads(Path(labels).read_text())["labels"]
    cells = {c: json.loads((HERE / "results" / f).read_text()) for c, f in CELLS.items()}
    o = own.spawn(lab, {c: {"packs": cells[c]["packs"]} for c in OWN}, cells)
    res = {"cells": CELLS, "sol": list(sol.starmap([(c, {"cohort": cells[c]["cohort"], "packs": cells[c]["packs"]}) for c in DUMPS])),
           "own": o.get()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dest = HERE / "results" / f"q4b_review2_checks_{stamp}.json"
    dest.open("x").write(json.dumps(res, indent=1, default=int))
    print(json.dumps(res, indent=1, default=int))
    print(dest.name)
