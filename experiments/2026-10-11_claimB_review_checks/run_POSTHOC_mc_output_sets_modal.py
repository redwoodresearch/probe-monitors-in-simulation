"""POST HOC (PR #526 review thread N0W7; Aswin's approval relayed by 0d just before 12:48 PT 10-11, under his standing
under-$5 rule): do the meaning control's four versions of each run (none, sim, prod, neutral) score the same set of
outputs? A planted line can push later outputs past the 32k cut, so the kept output sets can differ across versions.
One Modal CPU container reads only each run file's `meta` (the activation arrays are never loaded) and returns counts:
runs whose kept (step, action) lists differ from `none`'s, by version, and the outputs gained or lost. Numbers only.

  uv run --no-project --python 3.12 --with 'modal>=1.4.3' modal run \\
      experiments/2026-10-11_claimB_review_checks/run_POSTHOC_mc_output_sets_modal.py
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
RUNS = "claimB/meaning_control_v1/runs"
app = modal.App("probe-shift_aswin-surya_sub-claimb_mc-sets_kill-2026-11-11")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")
vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)


@app.function(image=image, cpu=2, memory=4096, volumes={"/acts": vol}, timeout=10 * 60, retries=0)
def count() -> dict:
    import numpy as np

    files = sorted(f for f in (Path("/acts") / RUNS).glob("*.npz") if not f.name.endswith(".tmp.npz"))
    if not files:
        raise ValueError(f"no run files under {RUNS}")
    VS = ("none", "sim", "prod", "neutral")
    out = {"runs": 0, "versions": list(VS), "runs_differing_from_none": {v: 0 for v in VS[1:]}, "outputs_only_in_none": {v: 0 for v in VS[1:]},
           "outputs_only_in_version": {v: 0 for v in VS[1:]}, "runs_differing_in_any_version": 0, "max_outputs_lost": {v: 0 for v in VS[1:]}}
    for f in files:
        with np.load(f) as z:  # only the meta member is read
            meta = json.loads(z["meta"].item())
        if not all(v in meta and "steps" in meta[v] for v in VS):
            raise ValueError(f"{f.name} lacks a version")
        vs = VS
        base = {tuple(x) for x in meta["none"]["steps"]}
        out["runs"] += 1
        any_diff = False
        for v in vs:
            if v == "none":
                continue
            s = {tuple(x) for x in meta[v]["steps"]}
            if s != base:
                any_diff = True
                out["runs_differing_from_none"][v] = out["runs_differing_from_none"].get(v, 0) + 1
            out["outputs_only_in_none"][v] = out["outputs_only_in_none"].get(v, 0) + len(base - s)
            out["outputs_only_in_version"][v] = out["outputs_only_in_version"].get(v, 0) + len(s - base)
            out["max_outputs_lost"][v] = max(out["max_outputs_lost"].get(v, 0), len(base - s))
        out["runs_differing_in_any_version"] += any_diff
    return out


@app.local_entrypoint()
def main():
    res = count.remote()
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    res |= {"label": "POST HOC (#526 N0W7): kept-output sets across the meaning control's four versions (counts only)", "git": git,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    dst = HERE / "results" / f"mc_output_sets_POSTHOC_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print("->", dst)
