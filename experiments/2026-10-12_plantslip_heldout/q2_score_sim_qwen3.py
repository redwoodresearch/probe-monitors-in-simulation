"""Row 2, step 5 (RULE.md here): Qwen3-32B's simulated arm, scored only now by the real-only fit's frozen per-fold probes (run
2026-10-12_192842, layers 20, 28, 36, 44, 52, 60) on 2026-10-06_la_segment-end_qwen3-32b_alllayers_v1's sim-attack rows. Each run's
kept output_end rows, ((x - mean) / std) . w + b, averaged, as the runner pools them (q2_score_frozen_qwen3.py's arithmetic).
Gate, in the same job: the same code re-scores the 939 cells' real e2 runs from the same dump, which must reproduce the fit's
stored nested scores within 1e-3 logits at every layer and nested model, or nothing is written. A simulated run with no meta, no
kept row, or a non-finite own-fold nested score is counted per layer and listed (never dropped); q2_y.py reports that
configuration as missing. It refuses to run unless the predictions commits and the B2 commit are
ancestors of HEAD and the predictions files and b2.json match their recorded sha256. One Modal CPU container.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with numpy modal run experiments/2026-10-12_plantslip_heldout/q2_score_sim_qwen3.py \\
      --qwen-runs <#36 runs.json> --fit-run 2026-10-12_192842
"""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-heldout_simscore_kill-2026-11-12"
ACTS, DUMP = "probe-shift_aswin-surya_sub-la-activations", "2026-10-06_la_segment-end_qwen3-32b_alllayers_v1"
WORK = "probe-shift_aswin-surya_sub-q2-heldout_work_kill-2026-11-12"
LAYERS, TOL = (20, 28, 36, 44, 52, 60), 1e-3
QWEN_RUNS_SHA = "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96"
ANCESTORS = ("c5a2fd2f", "434b1f2c", "d20db1ec")
RECORDED = {"predictions_2026-10-12_204016.json": "15d4bcfef8c6575b4a6f2e8d271b592bff18a51c44f06de06dbd259d95a0cad8",
            "predictions_2026-10-12_212633.json": "5b617fcca97bae442f7b9ac1dda407cbd17c415161a65aad1b52caf676486ee3",
            "b2_2026-10-12_214616.json": "cbf3f7e658a6e410fa1c1811d663c953f734b47a97d54412101e0f1f9d2bcdd7"}

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")
acts = modal.Volume.from_name(ACTS, version=2).read_only()
work = modal.Volume.from_name(WORK, version=2).read_only()


def filename(key: str) -> str:
    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


@app.function(image=image, volumes={"/acts": acts, "/work": work}, cpu=8, memory=32768, timeout=3600, retries=0)
def score(fit_run: str, items: list[dict]) -> dict:
    """items: {key, arm, fold}. Per layer, nested scores [5, 5, len(items)]."""
    try:
        return _score(fit_run, items)
    except Exception as e:  # noqa: BLE001 — sanitized and re-raised, never swallowed
        raise RuntimeError(f"score failed: {type(e).__name__}") from None


def _score(fit_run, items):
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    root = Path("/acts") / DUMP
    metas = []
    for it in items:
        mp = root / "meta" / it["arm"] / f"{filename(it['key'])}.json"
        if not mp.exists():  # recorded as unavailable (NaN), never dropped
            metas.append(None); continue
        m = json.loads(mp.read_text())
        rows = [m["row_start"] + q["row"] for q in m["positions"] if q["kind"] == "output_end" and q["kept"]]
        if m["key"] != it["key"]:
            raise ValueError("a meta key mismatch")
        metas.append((m["chunk"], rows) if rows else None)
    by_chunk = {}
    for j, mt in enumerate(metas):
        if mt is not None:
            by_chunk.setdefault(mt[0], []).append((j, mt[1]))
    probe_sha = {}

    def one_layer(layer):
        p = Path("/work") / "runs" / fit_run / f"probes_L{layer}.npz"
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        with np.load(p) as zf:
            z = {k: zf[k] for k in zf.files}
        S = np.full((5, 5, len(items)), np.nan)
        for c, js in by_chunk.items():
            X = np.load(root / "by_layer" / f"L{layer}" / f"{c}.npy")  # whole file: row-indexed mmap is slow on a Volume
            for j, rows in js:
                x, f = X[rows].astype(np.float32), items[j]["fold"]
                for v in range(5):
                    if v != f:
                        k = f"output_end__task_nested_{f}_{v}__"
                        S[f, v, j] = float(np.mean((((x - z[k + "mean"]) / z[k + "std"]) @ z[k + "w"] + np.float32(z[k + "b"])).astype(np.float64)))
        return layer, h, S.tolist()
    with ThreadPoolExecutor(3) as pool:
        res = list(pool.map(one_layer, LAYERS))
    return {"scores": {str(l): s for l, _, s in res}, "probes sha256": {str(l): h for l, h, _ in res}}


@app.local_entrypoint()
def main(qwen_runs: str, fit_run: str):
    import numpy as np
    for f in (Path(__file__).resolve(), HERE / "RULE.md"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    for a in ANCESTORS:
        if subprocess.run(["git", "merge-base", "--is-ancestor", a, "HEAD"], cwd=HERE).returncode:
            raise SystemExit(f"{a} is not an ancestor of HEAD")
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    for n, h in RECORDED.items():
        if sha(HERE / "results" / n) != h:
            raise SystemExit(f"{n} differs from its recorded sha256")
    if sha(qwen_runs) != QWEN_RUNS_SHA:
        raise SystemExit("not #36's runs.json")
    rec_fit = json.loads((HERE / "results" / f"run_{fit_run}.json").read_text())
    fdir = HERE / "results" / f"run_scores_{fit_run}"
    if sha(fdir / "runs.json") != rec_fit["archive"]["objects"]["data/runs.json"]["sha256"]:
        raise SystemExit("the fit's runs.json differs from its archive")
    fruns = json.loads((fdir / "runs.json").read_text())
    fe2 = {r["cell"]: (i, int(r["fold"]), r["key"]) for i, r in enumerate(fruns) if r["arm"] == "real-attack-e2"}
    runs = json.loads(Path(qwen_runs).read_text())
    sim = {r["cell"]: r for r in runs if r["arm"] == "sim-attack"}
    cells = sorted(fe2)
    if len(cells) != 939 or set(sim) != set(cells):
        raise SystemExit("not 939 paired cells")
    if any(int(sim[c]["fold"]) != fe2[c][1] for c in cells):
        raise SystemExit("a simulated run's fold differs from its cell's real e2 fold")
    items = [{"key": fe2[c][2], "arm": "real-attack-e2", "fold": fe2[c][1]} for c in cells] + [{"key": sim[c]["key"], "arm": "sim-attack", "fold": fe2[c][1]} for c in cells]
    res = score.remote(fit_run, items)
    if any(res["probes sha256"][str(l)] != rec_fit["archive"]["objects"][f"probes_L{l}.npz"]["sha256"] for l in LAYERS):
        raise SystemExit("the probes read differ from the fit's archived ones")
    n = len(cells); gate, unavailable = {}, {}
    out_sim = {}
    for l in LAYERS:
        Sx = np.array(res["scores"][str(l)]); stored = np.load(fdir / f"scores_L{l}.npz")["output_end__mean__task_nested"]
        dl = [abs(Sx[f, v, j] - stored[f, v, fe2[c][0]]) for j, c in enumerate(cells) for f in [fe2[c][1]] for v in range(5) if v != f]
        if not np.isfinite(dl).all():
            raise SystemExit(f"L{l}: a non-finite re-scored real e2 score")
        gate[str(l)] = float(max(dl))
        Ssim = Sx[:, :, n:]
        unavailable[str(l)] = [c for j, c in enumerate(cells) if not np.isfinite(np.delete(Ssim[fe2[c][1], :, j], fe2[c][1])).all()]
        out_sim[f"L{l}"] = Ssim
    if max(gate.values()) > TOL:
        raise SystemExit(f"gate failed: {gate}")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    npz = HERE / "results" / f"q2_sim_scores_qwen3_{stamp}.npz"
    np.savez(npz, cells=np.array(cells), sim_keys=np.array([sim[c]["key"] for c in cells]), folds=np.array([fe2[c][1] for c in cells]), **out_sim)
    rec = {"rule": "RULE.md step 5 (this folder)", "git": head, "fit run": fit_run, "dump": DUMP, "arm": "sim-attack", "cells": n,
           "gate: max |re-scored real e2 - stored| nested score, by layer (tolerance 1e-3)": gate, "comparisons per layer": n * 4, "gate passes": True,
           "simulated runs without four finite nested scores, by layer": {l: len(v) for l, v in unavailable.items()}, "unavailable cells, by layer": unavailable,
           "probes sha256": res["probes sha256"], "inputs sha256": {**RECORDED, "#36 runs.json": QWEN_RUNS_SHA, f"run_{fit_run}.json": sha(HERE / "results" / f"run_{fit_run}.json")},
           "scores file": npz.name, "scores sha256": sha(npz)}
    with open(dest := HERE / "results" / f"q2_sim_scores_qwen3_{stamp}.json", "x") as fh:
        fh.write(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k not in ("probes sha256", "inputs sha256")}, indent=1)); print(dest)
