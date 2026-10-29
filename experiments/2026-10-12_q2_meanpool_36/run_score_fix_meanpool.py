"""The repaired arm's scoring (PLAN.md here): the mean-pooled fit's frozen nested probes score fix939's 823 re-rendered
runs (the fix939 mean-pool dump, arm sim-attack-note-sr-both) and, as the gate, re-score the 939 as-simulated runs
(the main mean-pool dump). One Modal CPU container; saved activations and probes only, no fit, no model. Built on
../2026-10-12_q2_olmo_reader/run_score_fix_olmo.py; here a transcript's block holds every readout kind, so only its
kept output_end_mean rows are read (row_start + each position's row).

Per run in test fold f and layer l: each nested model (f, v), v != f, scores the run's output_end_mean rows as
((x - mean) / std) . w + b in float32 from the float16 rows, and the run score is the mean over its rows, as the fit pools
them. The local side checks the re-scored as-simulated runs against the stored nested scores
(output_end_mean__mean__task_nested) within 1e-3 at every layer and nested model, or writes nothing.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with numpy modal run \\
      experiments/2026-10-12_q2_meanpool_36/run_score_fix_meanpool.py --run-id <fit run id> \\
      --scores-dir <results/run_scores_<run id>> --fix <fix939 per-run jsonl>
"""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-meanpool36_fixscore_kill-2026-11-12"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
WORK = "probe-shift_aswin-surya_sub-q2-meanpool36_work_kill-2026-11-12"
MAIN, FIXRUN = "2026-10-12_la_mean-pool36_qwen3-32b_l7_v1", "2026-10-12_la_mean-pool36-fix939_qwen3-32b_l7_v1"
FIX_ARM, KIND = "sim-attack-note-sr-both", "output_end_mean"
LAYERS, TOL = (16, 24, 32, 40, 48, 56, 63), 1e-3
PIN = {"fix939_frozen_runs_POSTHOC_2026-10-11_020458.jsonl": "5f6ecbc42b1c6bb9586cb58e474d74c5c1d914bedb95d80a5a07dd078b0e5ebd"}

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2).read_only()
work = modal.Volume.from_name(WORK, version=2).read_only()
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")


@app.function(image=image, volumes={"/acts": acts, "/work": work}, cpu=8, memory=32768, timeout=3600, retries=0)
def score(sim: list[dict], run_id: str) -> dict:
    """sim: the as-simulated runs, {i, filename, fold}. Per layer, nested scores [5, 5, len(sim)] for the as-simulated
    and the re-rendered runs (NaN where a run has no re-rendered version)."""
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    dirs = {"sim-attack": Path("/acts") / MAIN, FIX_ARM: Path("/acts") / FIXRUN}
    metas = {}
    for lab, d in dirs.items():
        for r in sim:
            p = d / "meta" / lab / f"{r['filename']}.json"
            if p.exists():
                m = json.loads(p.read_text())
                rows = [m["row_start"] + q["row"] for q in m["positions"] if q["kind"] == KIND and q["kept"]]
                metas[(lab, r["filename"])] = (m["chunk"], rows, m["key"])
    if any(("sim-attack", r["filename"]) not in metas for r in sim):
        raise ValueError("an as-simulated run has no meta in the main dump")
    fix_found = sorted(f for (lab, f) in metas if lab == FIX_ARM)

    def one_layer(layer):
        with np.load(Path("/work") / "runs" / run_id / f"probes_L{layer}.npz") as zf:
            z = {k: zf[k] for k in zf.files}
        out = {lab: np.full((5, 5, len(sim)), np.nan) for lab in dirs}
        by_chunk = {}
        for j, r in enumerate(sim):
            for lab in dirs:
                if (lab, r["filename"]) in metas:
                    c, rows, _ = metas[(lab, r["filename"])]
                    by_chunk.setdefault((lab, c), []).append((j, lab, rows, r["fold"]))
        for (lab_, c), items in by_chunk.items():
            X = np.load(dirs[lab_] / "by_layer" / f"L{layer}" / f"{c}.npy")  # whole file: row-indexed mmap is slow on a Volume
            for j, lab, rows, f in items:
                if not rows:
                    continue
                x = X[rows].astype(np.float32)
                for v in range(5):
                    if v == f:
                        continue
                    k = f"{KIND}__task_nested_{f}_{v}__"
                    s = ((x - z[k + "mean"]) / z[k + "std"]) @ z[k + "w"] + np.float32(z[k + "b"])
                    out[lab][f, v, j] = float(np.mean(s.astype(np.float64)))
        return layer, out
    with ThreadPoolExecutor(len(LAYERS)) as pool:
        res = dict(pool.map(one_layer, LAYERS))
    return {"fix keys": sorted(metas[(FIX_ARM, f)][2] for f in fix_found),
            "scores": {str(l): {lab: res[l][lab].tolist() for lab in res[l]} for l in LAYERS}}


@app.local_entrypoint()
def main(run_id: str, scores_dir: str, fix: str):
    import numpy as np
    me = Path(__file__).resolve()
    for f in (me, HERE / "PLAN.md"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    d, fix_p = Path(scores_dir), Path(fix)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    if sha(fix_p) != PIN[fix_p.name] or not d.name.endswith(run_id):
        raise SystemExit("not the pinned fix939 per-run file, or the scores dir is not this run's")
    runs = json.loads((d / "runs.json").read_text())
    sim = [{"i": i, "filename": r["filename"], "fold": int(r["fold"]), "key": r["key"]} for i, r in enumerate(runs) if r["arm"] == "sim-attack"]
    want = sorted({json.loads(l)["key"] for l in fix_p.open() if json.loads(l)["label"] == "FIX"})
    if len(sim) != 939 or len(want) != 823 or not set(want) <= {r["key"] for r in sim}:
        raise SystemExit("not the 939 as-simulated runs and fix939's 823 re-rendered keys")
    res = score.remote([{k: r[k] for k in ("i", "filename", "fold")} for r in sim], run_id)
    if res["fix keys"] != want:
        raise SystemExit(f"the fix939 dump's runs are not fix939's 823: {len(res['fix keys'])} found")
    gate, fixed = {}, {}
    idx = np.array([r["i"] for r in sim]); fold = np.array([r["fold"] for r in sim])
    for l in LAYERS:
        stored = np.load(d / f"scores_L{l}.npz")[f"{KIND}__mean__task_nested"][:, :, idx]
        mine = np.array(res["scores"][str(l)]["sim-attack"])
        m = np.zeros_like(mine, bool)
        for j, f in enumerate(fold):
            m[f, :, j] = True; m[f, f, j] = False
        diff = np.abs(mine[m] - stored[m])
        if not np.isfinite(stored[m]).all() or not np.isfinite(diff).all():
            raise SystemExit(f"L{l}: a missing stored or re-scored nested score")
        gate[str(l)] = float(diff.max())
        fixed[f"L{l}"] = np.array(res["scores"][str(l)][FIX_ARM])
    if max(gate.values()) > TOL:
        raise SystemExit(f"gate failed: max |re-scored - stored| by layer {gate}")
    has_fix = np.isfinite(fixed[f"L{LAYERS[-1]}"][fold, (fold + 1) % 5, np.arange(len(sim))])
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    npz = HERE / "results" / f"fix_scores_meanpool_{stamp}.npz"
    np.savez(npz, run_index=idx, has_fix=has_fix, **fixed)
    rec = {"plan": "PLAN.md (this folder)", "git": head, "run": run_id, "extraction": [MAIN, FIXRUN],
           "inputs sha256": {fix_p.name: sha(fix_p), "runs.json": sha(d / "runs.json")} | {f"scores_L{l}.npz": sha(d / f"scores_L{l}.npz") for l in LAYERS},
           "gate: max |re-scored - stored| nested score, by layer (tolerance 1e-3)": gate,
           "as-simulated runs": len(sim), "with re-rendered scores": int(has_fix.sum()), "unchanged (as-simulated scores stand)": int((~has_fix).sum()),
           "scores file": npz.name, "scores sha256": sha(npz)}
    with open(dest := HERE / "results" / f"fix_scores_meanpool_{stamp}.json", "x") as fh:
        fh.write(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1)); print(dest)
