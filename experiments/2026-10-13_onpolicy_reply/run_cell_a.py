"""Cell (a), off-policy (RULE.md step 1 with Amendment 2.1): q2's frozen Qwen3.5 reply probes, fit on Sol's runs (e1 vs
real honest, L63 in every outer fold), score Qwen3.5's own runs. One Modal CPU container; returns run scores only.

- Parity gate first: 200 of q2's own stored per-output outer scores (`output_end__step__task_outer`), 40 per q2 fold,
  seed 20261013, recomputed from q2's packed L63 features as ((x - mean_f) / std_f) . w_f + b_f with f the run's outer
  fold; every one within 1e-3, else the job stops.
- Fold join: own task key -> the 196-task file's `cells[].main` -> q2's main_task -> q2's outer fold f (one fold per main,
  asserted). Each own run is scored by q2's outer model f only (it and its layer choice never saw that main task). A key
  without a main, or a main outside q2's folds, stops the job (no averaging fallback).
- Own runs: output_end rows at L63 of claim C's swapped cache (level 1) and the retrain caches (unswapped, levels 1 and
  0); run score = the mean over its outputs (#36's pooling). Worlds run1, run2, sim, honest_a, honest_b.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with numpy modal run \\
      experiments/2026-10-13_onpolicy_reply/run_cell_a.py --probes <q2 probes_L63.npz> --q2-runs <q2 runs.json>
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE  # inside the container the module sits in /root
APP = "probe-shift_aswin-surya_sub-onpol-reply_cella_kill-2026-11-13"
SCORE, Q2WORK, Q2DATA = "probe-shift_aswin-surya_sub-q3onpol_score_kill-2026-11-08", "probe-shift_aswin-surya_sub-q2-qwen35-reply_work_kill-2026-11-12", "d403ff8ad9c6"
SHA = {"probes": "6cf38abe14c5f3badc1d", "q2_runs": "3c028aca246181950e46"}  # sha256 prefixes of the S3 copies (run 2026-10-12_011944)
T196 = "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"
CACHES = {"swapped_L1": "claimc_cache_v1", "unswapped_L1": "retrain_cache_v1/level1", "unswapped_L0": "retrain_cache_v1/level0"}
WORLDS = ("run1", "run2", "sim", "honest_a", "honest_b")
app = modal.App(APP)
score_vol = modal.Volume.from_name(SCORE, version=2).read_only()
q2_vol = modal.Volume.from_name(Q2WORK).read_only()
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")


@app.function(image=image, cpu=8, memory=65536, timeout=1800, retries=0, volumes={"/score": score_vol, "/q2": q2_vol})
def score(P: dict, q2_fold_of_run: list, fold_of_task: dict) -> dict:
    import numpy as np

    P = {k: np.asarray(v) for k, v in P.items()}
    w = {f: (P[f"output_end__task_outer_{f}__w"].astype(np.float64), float(P[f"output_end__task_outer_{f}__b"]),
             P[f"output_end__task_outer_{f}__mean"].astype(np.float64), P[f"output_end__task_outer_{f}__std"].astype(np.float64)) for f in range(5)}
    sc = lambda X, f: ((X.astype(np.float64) - w[f][2]) / w[f][3]) @ w[f][0] + w[f][1]  # noqa: E731
    d = Path("/q2/data") / Q2DATA
    t = np.load(d / "tables.npz")
    X = np.load(d / "packed" / "L63_output_end.npy", mmap_mode="r")
    run_of_row, stored = t["output_end__run"], P["__stored_step"]
    if not len(run_of_row) == X.shape[0] == len(stored):
        raise ValueError("q2's table, packed rows and stored per-output scores disagree")
    rng, picks = np.random.default_rng(20261013), []
    rf = np.asarray(q2_fold_of_run)[run_of_row]
    for f in range(5):
        picks += sorted(rng.choice(np.flatnonzero(rf == f), 40, replace=False).tolist())
    got = np.array([sc(np.asarray(X[i])[None], int(rf[i]))[0] for i in picks])
    diff = np.abs(got - stored[picks])
    gate = {"n": len(picks), "per_fold": 40, "max_abs_diff": float(diff.max()), "passes": bool(diff.max() <= 1e-3)}
    if not gate["passes"]:
        return {"parity_gate": gate}
    out = {"parity_gate": gate, "variants": {}}
    for name, cache in CACHES.items():
        rows = json.loads((Path("/score") / cache / "on_rows.L63.json").read_text())
        A = np.load(Path("/score") / cache / "on_L63.npy", mmap_mode="r")
        keep = [i for i, r in enumerate(rows) if r["kind"] == "output_end" and r["world"] in WORLDS]
        missing = sorted({rows[i]["task"] for i in keep} - set(fold_of_task))
        if missing:
            raise ValueError(f"{name}: tasks without a q2 fold {missing[:5]}")
        tot, cnt = {}, {}
        for f in range(5):
            idx = [i for i in keep if fold_of_task[rows[i]["task"]] == f]
            s = sc(np.asarray(A[idx]), f) if idx else []
            for i, v in zip(idx, s):
                k = f"{rows[i]['world']}|{rows[i]['task']}"
                tot[k] = tot.get(k, 0.0) + float(v)
                cnt[k] = cnt.get(k, 0) + 1
        out["variants"][name] = {"run_scores": {k: tot[k] / cnt[k] for k in sorted(tot)}, "outputs": cnt,
                                 "rows_by_world": {wd: sum(rows[i]["world"] == wd for i in keep) for wd in WORLDS}}
    return out


@app.local_entrypoint()
def main(probes: str, q2_runs: str):
    import numpy as np

    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=HERE).stdout:
        raise SystemExit("uncommitted tracked changes")
    for k, p in (("probes", probes), ("q2_runs", q2_runs)):
        if not hashlib.sha256(Path(p).read_bytes()).hexdigest().startswith(SHA[k]):
            raise SystemExit(f"{k} sha256 mismatch")
    runs = json.loads(Path(q2_runs).read_text())
    fold_of_main = {}
    for r in runs:
        if fold_of_main.setdefault(r["main_task"], r["fold"]) != r["fold"]:
            raise SystemExit(f"q2 main task {r['main_task']} in two folds")
    cells = json.loads((REPO / T196).read_text())["cells"]
    fold_of_task = {}
    for c in cells:
        if not c.get("main") or c["main"] not in fold_of_main:
            raise SystemExit(f"fold join failed for {c.get('key')}: main {c.get('main')!r} not in q2's folds")
        fold_of_task[c["key"]] = fold_of_main[c["main"]]
    P = {k: v for k, v in np.load(probes).items() if k.startswith("output_end__task_outer_")}
    q2_scores = np.load(Path(q2_runs).parent / "scores_L63.npz")
    P["__stored_step"] = q2_scores["output_end__step__task_outer"]
    res = score.remote({k: v.tolist() for k, v in P.items()}, [r["fold"] for r in runs], fold_of_task)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dest = HERE / "results" / f"cell_a_run_scores_POSTHOC_{stamp}.json"
    dest.open("x").write(json.dumps({"label": "POST HOC (RULE.md step 1 (a), Amendment 2.1); run scores only", "git": git,
                                     "q2_run": "2026-10-12_011944", "fold_of_task": fold_of_task} | res, indent=1))
    print(json.dumps(res["parity_gate"]), {n: v["rows_by_world"] for n, v in res.get("variants", {}).items()})
    print(dest.name)
