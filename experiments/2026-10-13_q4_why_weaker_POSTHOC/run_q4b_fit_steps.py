"""Q4 (B) convergence audit, step 1 (no refits): the per-fit L-BFGS closure-evaluation counts that run_q4b_cells_null.py
stored for every nested model (the real column refit exactly as run_q4b_cells.py, C = 0.01, and the joint 100-column null
fit), read from the Volume on one Modal CPU container. Per cell, layer and probe: min / median / max over the 20 nested
models, and the number of fits under 10 evaluations or at the evaluation cap (1,250 = max_iter * 5 // 4).

  uv run --no-project --python 3.12 --with modal==1.6.1 modal run --detach \\
      experiments/2026-10-13_q4_why_weaker_POSTHOC/run_q4b_fit_steps.py
"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-q4own_steps_kill-2026-11-13"
VOL = "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13"
DIRS = {"q35_own": "q4b_null_work/2026-10-14_022926/q35_own", **{c: f"q4b_null_work/2026-10-14_023056/{c}" for c in ("q35_sol", "q3_sol", "q3_own", "q35_sol_cut32k")}}
CAP = 1000 * 5 // 4
app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")


@app.function(image=image, cpu=1, memory=4096, timeout=900, retries=0, volumes={"/v": modal.Volume.from_name(VOL, version=2).read_only()})
def steps() -> dict:
    import numpy as np

    out = {}
    for cell, d in DIRS.items():
        for layer in (48, 63):
            for p in ("a", "c"):
                s = np.load(Path("/v", d, f"L{layer}_{p}_scores.npz"))["steps"]  # [20, 2]: real, null, per nested (f, v)
                out[f"{cell}|L{layer}|{p}"] = {k: {"min": int(x.min()), "median": float(np.median(x)), "max": int(x.max()), "under_10": int((x < 10).sum()),
                                                   "at_cap": int((x >= CAP).sum()), "n": int(len(x))} for k, x in (("real", s[:, 0]), ("null", s[:, 1]))}
    return out


@app.local_entrypoint()
def main():
    res = {"label": "Q4 (B) convergence audit, step 1: stored closure-evaluation counts (no refits)", "C": 0.01, "eval_cap": CAP, "steps": steps.remote()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dest = HERE / "results" / f"q4b_fit_steps_{stamp}.json"
    dest.open("x").write(json.dumps(res, indent=1))
    for k, v in res["steps"].items():
        r, n = v["real"], v["null"]
        print(f"{k:24s} real min {r['min']:4d} median {r['median']:6.1f} max {r['max']:4d} (<10: {r['under_10']}, cap: {r['at_cap']}) | "
              f"null min {n['min']:4d} median {n['median']:6.1f} max {n['max']:4d} (<10: {n['under_10']}, cap: {n['at_cap']})")
    print(dest.name)
