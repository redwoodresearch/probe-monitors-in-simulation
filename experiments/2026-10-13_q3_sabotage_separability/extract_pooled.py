"""Q3(a) POST HOC (Codex round 1, item 2): read the pooled out-of-fold AUROCs that RULE.md promises "beside" from the run's per-draw
file on the work Volume (sab_separability_draws_2026-10-13_233027.json, written by run_sab_separability.py @ 4476548d). Reads
only; no fit. Every set, contrast and C view, at each readout; also the full-sample combined AUROC, as a cross-check against the
summary file.

  cd <repo root>; modal run --detach experiments/2026-10-13_q3_sabotage_separability/extract_pooled.py
"""

import hashlib
import json
from pathlib import Path

import modal

APP = "probe-shift_aswin-surya_sub-q3sabsep_pooled_kill-2026-11-13"
WORK = "probe-shift_aswin-surya_sub-q3sabsep_work_kill-2026-11-13"
DRAWS = "sab_separability_draws_2026-10-13_233027.json"
HERE = Path(__file__).parent

app = modal.App(APP)
work = modal.Volume.from_name(WORK, version=2)


@app.function(image=modal.Image.debian_slim(python_version="3.12"), cpu=1, memory=2048, timeout=300, retries=0,
              volumes={"/work": work.read_only()})
def extract() -> dict:
    b = Path(f"/work/{DRAWS}").read_bytes()
    R = json.loads(b)
    out = {ro: {s: {c: {v: {"pooled": r[v]["pooled_full"], "combined_full": r[v]["combined_draws"][0]} for v in ("selected_C", "fixed_C_1e-4")}
                    for c, r in S.items()} for s, S in R[ro]["sets"].items()} for ro in R}
    return {"draws_file": DRAWS, "draws_sha256": hashlib.sha256(b).hexdigest(), "values": out}


@app.local_entrypoint()
def main():
    res = extract.remote()
    dst = HERE / "results" / "pooled_auroc_POSTHOC_2026-10-13_233027.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(res, indent=1) + "\n")
    print("->", dst, hashlib.sha256(dst.read_bytes()).hexdigest())
