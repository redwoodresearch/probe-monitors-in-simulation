"""Gate row 1 for Llama, the repaired arm (PLAN.md here): ../2026-10-12_q2_llama_reader/run_score_fix_llama.py with only its
app, Volumes, extraction (one run holding both cohorts), layers (60, 70; never row 2's), readout (output_end_mean) and output
names changed. Its docstring follows.

Llama part 2's scoring (PLAN.md step C here): ../2026-10-12_q2_olmo_reader/run_score_fix_olmo.py with only its names,
layers (20..79 of 80) and the run (an argument: Llama's fit run id) changed. The frozen probes of that run score fix939's
823 re-rendered runs (this extraction's arm sim-attack-fix939) and, as the gate, re-score the 939 as-simulated runs. One Modal CPU
container; saved activations and probes only, no fit, no model.

Per run in test fold f and layer l: each nested model (f, v), v != f, scores the run's R2 rows as
((x - mean) / std) . w + b in float32 from the float16 rows, and the run score is the mean over its rows, as the fit pools
them. The local side checks the re-scored as-simulated runs against the stored nested scores (scores_L<l>.npz,
output_end__mean__task_nested) within 1e-3 at every layer and nested model, or writes nothing; then it writes the fixed
runs' nested scores (npz) and a record (json) to results/.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with numpy modal run \\
      experiments/2026-10-12_q2_meanpool_readers/run_score_fix_mpllama.py --run-id <mean-pooled Llama fit run id> \\
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
APP = "probe-shift_aswin-surya_sub-q2-mpllama_fixscore_kill-2026-11-12"
ACTS = "probe-shift_aswin-surya_sub-q2-mpllama_acts_kill-2026-11-12"
WORK = "probe-shift_aswin-surya_sub-q2-mpllama_work_kill-2026-11-12"
EXTRACT, FIX_LABEL, KIND = "2026-10-12_la_output-end-mean_llama33-70b_l2_v1", "sim-attack-fix939", "output_end_mean"
LAYERS, ROW2, TOL = (60, 70), (20, 30, 40, 50, 79), 1e-3
assert not set(LAYERS) & set(ROW2)
PIN = {"fix939_frozen_runs_POSTHOC_2026-10-11_020458.jsonl": "5f6ecbc42b1c6bb9586cb58e474d74c5c1d914bedb95d80a5a07dd078b0e5ebd"}

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2).read_only()
work = modal.Volume.from_name(WORK, version=2).read_only()
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")


@app.function(image=image, volumes={"/acts": acts, "/work": work}, cpu=8, memory=32768, timeout=3600, retries=0)
def score(sim: list[dict], run_id: str) -> dict:
    """sim: the as-simulated runs, {i, filename, fold}. Returns, per layer, nested scores [5, 5, len(sim)] for the
    as-simulated rows and for the re-rendered rows (NaN where a run has no re-rendered version), and the runs found."""
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    dirs = {"sim-attack": Path("/acts") / EXTRACT, FIX_LABEL: Path("/acts") / EXTRACT}  # one run holds both cohorts (--scope all)
    metas = {}
    for label in ("sim-attack", FIX_LABEL):
        for r in sim:
            p = dirs[label] / "meta" / label / f"{r['filename']}.json"
            if p.exists():
                m = json.loads(p.read_text())
                metas[(label, r["filename"])] = (m["chunk"], m["row_start"], m["n_rows"], m["key"])
    fix_found = sorted(f for (lab, f) in metas if lab == FIX_LABEL)
    if any(("sim-attack", r["filename"]) not in metas for r in sim):
        raise ValueError("an as-simulated run has no meta in the extraction")

    def one_layer(layer):
        with np.load(Path("/work") / "runs" / run_id / f"probes_L{layer}.npz") as zf:
            z = {k: zf[k] for k in zf.files}  # once: an npz re-reads its zip member on every access
        out = {lab: np.full((5, 5, len(sim)), np.nan) for lab in ("sim-attack", FIX_LABEL)}
        by_chunk = {}
        for j, r in enumerate(sim):
            for lab in ("sim-attack", FIX_LABEL):
                if (lab, r["filename"]) in metas:
                    c, s0, n, _ = metas[(lab, r["filename"])]
                    by_chunk.setdefault((lab, c), []).append((j, lab, s0, n, r["fold"]))
        for (lab_, c), items in by_chunk.items():
            X = np.load(dirs[lab_] / "by_layer" / f"L{layer}" / f"{c}.npy")  # whole file: row-indexed mmap is slow on a Volume
            for j, lab, s0, n, f in items:
                if n == 0:
                    continue
                x = X[s0:s0 + n].astype(np.float32)
                for v in range(5):
                    if v == f:
                        continue
                    k = f"{KIND}__task_nested_{f}_{v}__"
                    s = ((x - z[k + "mean"]) / z[k + "std"]) @ z[k + "w"] + np.float32(z[k + "b"])
                    out[lab][f, v, j] = float(np.mean(s.astype(np.float64)))
        return layer, out
    with ThreadPoolExecutor(len(LAYERS)) as pool:
        res = dict(pool.map(one_layer, LAYERS))
    return {"fix filenames found": fix_found, "fix keys": sorted(metas[(FIX_LABEL, f)][3] for f in fix_found),
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
    if sha(fix_p) != PIN[fix_p.name]:
        raise SystemExit("not the pinned fix939 per-run file")
    runs = json.loads((d / "runs.json").read_text())
    sim = [{"i": i, "filename": r["filename"], "fold": int(r["fold"]), "key": r["key"]} for i, r in enumerate(runs) if r["arm"] == "sim-attack"]
    want = sorted({json.loads(l)["key"] for l in fix_p.open() if json.loads(l)["label"] == "FIX"})
    if len(sim) != 939 or len(want) != 823 or not set(want) <= {r["key"] for r in sim}:
        raise SystemExit("not the 939 as-simulated runs and fix939's 823 re-rendered keys")
    res = score.remote([{k: r[k] for k in ("i", "filename", "fold")} for r in sim], run_id)
    if res["fix keys"] != want:
        raise SystemExit(f"the extraction's re-rendered runs are not fix939's 823: {len(res['fix keys'])} found")
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
        fixed[f"L{l}"] = np.array(res["scores"][str(l)][FIX_LABEL])
    if max(gate.values()) > TOL:
        raise SystemExit(f"gate failed: max |re-scored - stored| by layer {gate}")
    has_fix = np.isfinite(fixed[f"L{LAYERS[-1]}"][fold, (fold + 1) % 5, np.arange(len(sim))])
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    npz = HERE / "results" / f"fix_scores_mpllama_{stamp}.npz"
    np.savez(npz, run_index=idx, has_fix=has_fix, **fixed)
    rec = {"plan": "PLAN.md (this folder)", "git": head, "run": run_id, "extraction": EXTRACT,
           "inputs sha256": {fix_p.name: sha(fix_p), "runs.json": sha(d / "runs.json")} | {f"scores_L{l}.npz": sha(d / f"scores_L{l}.npz") for l in LAYERS},
           "gate: max |re-scored - stored| nested score, by layer (tolerance 1e-3)": gate,
           "as-simulated runs": len(sim), "with re-rendered scores": int(has_fix.sum()), "unchanged (as-simulated scores stand)": int((~has_fix).sum()),
           "scores file": npz.name, "scores sha256": sha(npz)}
    with open(dest := HERE / "results" / f"fix_scores_mpllama_{stamp}.json", "x") as fh:
        fh.write(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1)); print(dest)
