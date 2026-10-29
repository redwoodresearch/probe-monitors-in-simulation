"""Replay input (RULE.md v3, open input 4): own_a tight command-A per-fold moments on the run MAX. The recorded
correctedb_decision moments are on the run mean (2e), so they are recomputed here with the same definition as `_zmom`:
ddof 0 over fold f's honest_b run scores, draw 0. Modal CPU; scores only, no text.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 modal run \\
      experiments/2026-10-15_replay_sim_outputs/run_owna_cmd_max_moments.py
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REPO = HERE.parent.parent
OWN_RECORD = ("cc53afc3a", "experiments/2026-10-14_q35_sol_avg_POSTHOC/results/correctedb_scores_2026-10-14_164934.json",
              "c002d5f822e49d7877f4e8b79a80ca9dd2e2f16c664b921d3ad12cb3da4c49ac")
app = modal.App("probe-shift_aswin-surya_sub-replay_kill-2026-11-14")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")
sol = modal.Volume.from_name("probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13").read_only()


@app.function(image=image, cpu=1, memory=4096, timeout=600, retries=0, volumes={"/sol": sol})
def moments(path: str, sha: str) -> dict:
    import io

    import numpy as np

    raw = (Path("/sol") / path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError("rows file sha256 differs from the record")
    z = np.load(io.BytesIO(raw), allow_pickle=False)
    key, score, fold = z["key"].astype(str), z["score"].astype(np.float64), z["fold"].astype(np.int64)
    if not (len(key) == len(score) == len(fold)) or not np.isfinite(score).all():
        raise ValueError("unequal arrays or non-finite scores")
    runs, fo = {}, {}
    for k, s, f in zip(key, score, fold, strict=True):
        runs.setdefault(k, []).append(float(s))
        fo.setdefault(k, set()).add(int(f))
    if any(len(v) != 1 for v in fo.values()):
        raise ValueError("a run scored by more than one fold model")
    hb = {k: max(v) for k, v in runs.items() if k.startswith("honest_b|")}
    out = {}
    for f in range(5):
        x = np.array([v for k, v in hb.items() if next(iter(fo[k])) == f])
        if len(x) < 2 or not np.isfinite(x.std()) or x.std() <= 0:
            raise ValueError(f"fold {f}: too few or flat honest_b runs")
        out[str(f)] = {"mu": float(x.mean()), "sd": float(x.std()), "n_honest_b_runs": int(len(x))}
    return {"statistic": "run max of content-only call_end rows (command A)", "pool": "honest_b", "ddof": 0, "draw": 0,
            "n_rows": int(len(key)), "n_runs": len(runs), "n_honest_b_runs": len(hb), "moments": out}


@app.local_entrypoint()
def main():
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=REPO, check=True).stdout.strip()  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    rec = subprocess.run(["git", "show", f"{OWN_RECORD[0]}:{OWN_RECORD[1]}"], capture_output=True, check=True, cwd=REPO).stdout
    if hashlib.sha256(rec).hexdigest() != OWN_RECORD[2]:
        raise SystemExit("own_a record sha256 differs")
    spec = json.loads(rec)["saved"]["cmd"]["cmdA"]["rows|own_a_tight"]
    res = moments.remote(spec["path"], spec["sha256"])
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    head = {"git": git("rev-parse", "HEAD"), "rows": f"{spec['path']} (sha256 {spec['sha256']})",
            "record": f"{OWN_RECORD[1]}@{OWN_RECORD[0]}", "label": "NEW (df): run-max moments; not 4018c645f's run-mean ones"}
    p = HERE / "results" / f"owna_cmd_max_moments_{stamp}.json"
    p.parent.mkdir(exist_ok=True)
    with p.open("x") as fh:
        json.dump(head | res, fh, indent=1)
    print(f"wrote {p.name} sha256 {hashlib.sha256(p.read_bytes()).hexdigest()}")
    print("runs", res["n_runs"], "honest_b runs", res["n_honest_b_runs"], "per fold", {f: m["n_honest_b_runs"] for f, m in res["moments"].items()})
