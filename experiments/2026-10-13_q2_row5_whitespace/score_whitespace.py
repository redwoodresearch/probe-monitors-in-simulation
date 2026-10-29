"""Gate row 5 (PLAN.md here): frozen-probe scoring. A copy of P4's q2_reply_score.py (../2026-10-12_plantslip_transfer,
as run) for one reader, #36 (Qwen3-32B, last-token run 2026-10-06_181548), at layer 48 in every fold. One Modal CPU
container per call; saved activations and frozen probes only, no fit, no model, no text.

Per run in test fold f: each nested model (f, v), v != f, scores the run's kept output_end rows as
((x - mean) / std) . w + b in float32 from the float16 rows; the run score is the mean over those rows.
  --which controls   the 20 controls; the record holds the gate: 80 re-scored nested scores against #36's stored
                     scores_L48.npz, tolerance 1e-3 logit.
  --which planted    the planted runs; needs --gate, a passing controls record from the same staging.
The probe file, the stored scores and runs.json are checked against their pinned sha256 on every call.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with numpy modal run \\
      experiments/2026-10-13_q2_row5_whitespace/score_whitespace.py --which <controls|planted> --extract-run <run> --stage <stage json> [--gate <json>]
"""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-row5ws_score_kill-2026-11-13"
WORK, RUN, ACTS, LAYER = "probe-shift_aswin-surya_sub-q3attack_work_kill-2026-11-06", "2026-10-06_181548", "probe-shift_aswin-surya_sub-la-activations", 48
SCORES = Path("/Users/aswinsurya/probe-distribution-shift-from-sims/.claude/worktrees/agent-a41654b9677ba3292/experiments/"
              "2026-10-06_q3_attack_detector/results/run_scores_2026-10-06_181548")
PIN = {"probes_L48.npz": "9fd9989b88eacf8652cc6ccf653e105b03496cff19f8fca0f5a13843015bd104",  # sim_honest_scoring_2026-10-07_124717.json
       "scores_L48.npz": "f6cf939e20134218cc2f0f7d9ecc1bdaf21aadfdb05b921d7361551017e70fc4",
       "runs.json": "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96"}
TOL = 1e-3

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")
VOLS = {"/work": modal.Volume.from_name(WORK, version=2).read_only(), "/acts": modal.Volume.from_name(ACTS, version=2).read_only()}


def filename(key: str) -> str:  # the extractors' file stems
    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


@app.function(image=image, volumes=VOLS, cpu=8, memory=32768, timeout=3600, retries=0)
def score(extract_run: str, runs: list[dict]) -> dict:
    """runs: {j, key, fold, arm}. Nested scores [len(runs), 5] (NaN off the run's fold and where a run has no meta or
    no kept row), the probe file's sha256, and the count of runs with no meta."""
    import numpy as np
    root = Path("/acts") / extract_run
    metas, missing = {}, 0
    for i, r in enumerate(runs):
        p = root / "meta" / r["arm"] / f"{filename(r['key'])}.json"
        if not p.exists():
            missing += 1
            continue
        m = json.loads(p.read_text())
        metas[i] = (m["chunk"], m["row_start"], sorted(q["row"] for q in m["positions"] if q["kind"] == "output_end" and q["kept"]))
    f = Path("/work") / "runs" / RUN / f"probes_L{LAYER}.npz"
    sha = hashlib.sha256(f.read_bytes()).hexdigest()
    if sha != PIN["probes_L48.npz"]:
        raise ValueError("probes_L48.npz differs from its pin")
    with np.load(f) as zf:
        z = {k: zf[k] for k in zf.files}
    S, by_chunk = np.full((len(runs), 5), np.nan), {}
    for i, (c, _, _) in metas.items():
        by_chunk.setdefault(c, []).append(i)
    for c, idx in by_chunk.items():
        X = np.load(root / "by_layer" / f"L{LAYER}" / f"{c}.npy")  # whole file: row-indexed mmap is slow on a Volume
        for i in idx:
            _, s0, rows = metas[i]
            if not rows:
                continue
            x = X[[s0 + q for q in rows]].astype(np.float32); f_ = runs[i]["fold"]
            for v in range(5):
                if v != f_:
                    k = f"output_end__task_nested_{f_}_{v}__"
                    S[i, v] = float(np.mean((((x - z[k + "mean"]) / z[k + "std"]) @ z[k + "w"] + np.float32(z[k + "b"])).astype(np.float64)))
    return {"scores": S.tolist(), "probes sha256": sha, "runs with no meta": missing}


@app.local_entrypoint()
def main(which: str, extract_run: str, stage: str, gate: str = ""):
    import numpy as np
    for f in (Path(__file__).resolve(), HERE / "PLAN.md"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    if any(sha(SCORES / n) != PIN[n] for n in ("scores_L48.npz", "runs.json")):
        raise SystemExit("#36's stored scores or runs.json differ from their pins")
    if which not in ("controls", "planted"):
        raise SystemExit("--which controls|planted")
    st = json.loads(Path(stage).read_text())
    if which == "planted":  # the gate must have passed, from the same staging, before any planted score
        g = json.loads(Path(gate).read_text()) if gate else {}
        if g.get("gate") != "pass" or g.get("stage sha256") != sha(stage):
            raise SystemExit("planted scoring needs a passing controls record (--gate) from the same staging")
    runs_all = json.loads((SCORES / "runs.json").read_text())
    j_of = {(r["arm"], r["key"]): j for j, r in enumerate(runs_all)}
    todo = [{"j": j_of[(x["orig_arm"], x["key"])], "key": x["key"], "fold": int(runs_all[j_of[(x["orig_arm"], x["key"])]]["fold"]), "arm": x["arm"]}
            for x in st[which]]
    res = score.remote(extract_run, todo)
    S = np.array(res["scores"])
    rec = {"plan": "PLAN.md (this folder)", "git": head, "reader": "Qwen/Qwen3-32B (#36, last-token)", "frozen run": RUN, "layer": LAYER,
           "extract run": extract_run, "which": which, "stage file": Path(stage).name, "stage sha256": sha(stage),
           "probes sha256": res["probes sha256"], "stored scores sha256": {n: PIN[n] for n in ("runs.json", "scores_L48.npz")},
           "runs": len(todo), "runs with no meta": res["runs with no meta"]}
    if which == "controls":
        stored = np.load(SCORES / "scores_L48.npz")["output_end__mean__task_nested"]
        diffs = [abs(S[i, v] - stored[t["fold"], v, t["j"]]) for i, t in enumerate(todo) for v in range(5) if v != t["fold"]]
        distinct = len({t["key"] for t in todo})
        ok = bool(distinct == 20 == len(todo) and len(diffs) == 80 and np.isfinite(diffs).all() and max(diffs) <= TOL)
        rec |= {"gate: max |re-scored - stored| nested score (tolerance 1e-3)": float(np.nanmax(diffs)) if diffs else None,
                "gate: comparisons (20 controls x 4 nested models)": len(diffs), "gate: distinct controls": distinct, "gate": "pass" if ok else "fail"}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    npz = HERE / "results" / f"scores_ws_{which}_{stamp}.npz"
    np.savez(npz, j=np.array([t["j"] for t in todo]), arm=np.array([t["arm"] for t in todo]), L48=S)
    rec |= {"scores file": npz.name, "scores sha256": sha(npz)}
    with open(dest := HERE / "results" / f"scores_ws_{which}_{stamp}.json", "x") as fh:
        fh.write(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1)); print(dest)
    if rec.get("gate") == "fail":
        raise SystemExit("the gate failed: nothing more runs")
