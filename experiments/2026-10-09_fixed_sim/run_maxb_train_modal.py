"""Max-over-tokens version (b) (PLAN.md amendment 7): training on pass 1's sampled tokens (run_maxb_read_modal.py --mode
sample). Every sampled token carries its run's label; one H100 container per layer fits, per comparison, C and outer
fold (an environment; planted checks: the 5 test environments, a pair's copies in one fold), d9's fitter on the training
rows (standardized x sqrt(C)) for 1 + 100 label columns at once (maxb_core.labels: B's null units), and the same fit
without the fold's validation tasks (task fold = outer fold mod 5) to score them. The choice criterion is the validation
run-level AUROC with run score = the max over the run's sampled-token scores, per column, so the null reruns the choice.
Converged fits only (B's gradient check, refits up to 80,000 iterations). Saved per (comparison, layer, C): every fold's
weights, intercepts, standardization and gradients (pass 2 needs the chosen probes of every column), and a pass-1 preview
(test-fold run max over the sampled tokens). Stages:
  --stage prepare  (CPU) runs.json (canonical run order: arm, key) and rows.npz (each sampled row's run, output, token)
  --stage grid     (H100, one container per layer; --layers, --comps, --cs restrict it: the pilot is one cell of the real
                   grid, e.g. --layers 63 --comps plant1 --cs 0.001, timed and then kept, since a finished cell is skipped)
  --stage select   (CPU) per comparison, fold and column the best converged (layer, C), compiled into per-layer banks
                   (`bank/L<l>.npz`: raw-activation weights and intercepts; index.json; env_fold.json; choices.json)

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy modal run --detach \\
      experiments/2026-10-09_fixed_sim/run_maxb_train_modal.py --stage prepare|grid|select [--tag <tag>] [--layers 63 --comps plant1 --cs 0.001]
"""

import json
from pathlib import Path

import modal

HERE = Path(__file__).parent
ROOT = HERE.parents[1] if modal.is_local() else Path("/root")
APP = "probe-shift_aswin-surya_sub-claimb_maxb-train_kill-2026-11-10"
RUN1 = "2026-10-10_la_fixed-sim-v2-maxb-sample-scrub1_qwen3-32b_l8_v1"

app = modal.App(APP)
vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy", "scipy")
         .add_local_python_source("probe_shift").add_local_file(HERE / "maxb_core.py", "/root/maxb_core.py"))


def base(tag: str) -> Path:
    return Path(f"/acts/claimB/level1/maxb_{tag}")


def chunk_names() -> list[str]:
    """Pass 1's production chunks (the --limit pilot writes RUN1_test, which is never trained on)."""
    names = sorted(p.stem for p in (Path("/acts") / RUN1 / "chunks").glob("*.json"))
    if not names:
        raise FileNotFoundError(f"no pass-1 chunks under {RUN1}: run the full pass 1 first")
    return names


@app.function(image=image, cpu=4, memory=32768, volumes={"/acts": vol}, timeout=1800, retries=0)
def prepare(tag: str, folds: dict) -> dict:
    import numpy as np

    vol.reload()
    metas = [(name, m) for name in chunk_names() for m in json.loads((Path("/acts") / RUN1 / "chunks" / f"{name}.json").read_text())]
    task_fold = folds["task_folds"]["main_task_to_fold"]
    order = sorted(range(len(metas)), key=lambda i: (metas[i][1]["arm"], metas[i][1]["key"]))
    runs = [{k: metas[i][1][k] for k in ("arm", "key", "cell", "main_task", "environment")} | {
        "fold": task_fold[metas[i][1]["main_task"]], "outputs": metas[i][1]["outputs"], "line_tokens": metas[i][1].get("line_tokens"),
        "site_outputs": metas[i][1].get("site_outputs")} for i in order]
    pos = {(m["arm"], m["key"]): i for i, m in enumerate(runs)}
    if len(pos) != len(runs):
        raise ValueError("a transcript appears in two chunks")
    run_of, out_of, token = [], [], []
    for _, m in metas:  # chunk order, as the per-layer arrays are concatenated
        run_of += [pos[(m["arm"], m["key"])]] * len(m["rows"])
        out_of += m["row_output"]
        token += m["rows"]
    b = base(tag)
    b.mkdir(parents=True, exist_ok=True)
    (b / "runs.json").write_text(json.dumps(runs))
    (b / "env_fold.json").write_text(json.dumps(folds["environment_folds"]["environment_to_fold"]))
    np.savez(b / "rows.npz", run=np.array(run_of), output=np.array(out_of), token=np.array(token))
    vol.commit()
    return {"runs": len(runs), "rows": len(run_of), "chunks": len(chunk_names())}


@app.function(image=image, gpu="H100!", cpu=8, memory=131072, volumes={"/acts": vol}, timeout=3 * 3600, retries=0, scaledown_window=5)
def grid(tag: str, layer: int, comps: list[str], Cs: list[float]) -> dict:
    import time

    import numpy as np
    import torch

    import maxb_core as K

    t0, b = time.time(), base(tag)
    runs = json.loads((b / "runs.json").read_text())
    env_fold = json.loads((b / "env_fold.json").read_text())
    ru = np.load(b / "rows.npz")["run"]
    X = torch.from_numpy(np.concatenate([np.load(Path("/acts") / RUN1 / "by_layer" / f"L{layer}" / f"{n}.npy") for n in chunk_names()])).cuda()
    if len(X) != len(ru):
        raise ValueError(f"L{layer}: {len(X)} rows vs {len(ru)} in rows.npz")
    efold, tfold = np.array([env_fold[m["environment"]] for m in runs]), np.array([m["fold"] for m in runs])
    out = {}
    for comp in comps:
        use, Y = K.labels(runs, comp)
        folds = sorted(set(efold[use]))
        for C in Cs:
            p = b / "grid" / comp / f"L{layer}_C{C:g}.npz"
            if p.exists():
                continue
            r = {k: [] for k in ("W", "b", "mean", "std", "grad_full", "grad_val", "val_auroc")}
            preview = np.full((len(runs), Y.shape[1]), np.nan)
            for f in folds:
                tr_run, te_run = use & (efold != f), use & (efold == f)
                va_run = tr_run & (tfold == f % 5)
                for name, fit_run, score_run in (("val", tr_run & ~va_run, va_run), ("full", tr_run, te_run)):
                    a, s = np.flatnonzero(fit_run[ru]), np.flatnonzero(score_run[ru])
                    Xa = X[torch.from_numpy(a).cuda()].double()
                    mean, std = Xa.mean(0), Xa.std(0, unbiased=False)
                    std = std.where(std > 0, std.new_ones(()))
                    Xa = (Xa - mean) / std * C**0.5
                    W, bb, g, _ = K.fit(Xa, torch.from_numpy(Y[ru[a]]).cuda().double(), C)
                    del Xa
                    S = ((X[torch.from_numpy(s).cuda()].double() - mean) / std * C**0.5) @ W + bb
                    k, inv = np.unique(ru[s], return_inverse=True)
                    M = K.run_max(S, torch.from_numpy(inv).cuda(), len(k)).cpu().numpy()
                    r[f"grad_{name}"].append(g)
                    if name == "val":
                        r["val_auroc"].append(K.col_auc(Y[k], M))
                    else:
                        preview[k] = M
                        r["W"].append(W.float().cpu().numpy()), r["b"].append(bb.float().cpu().numpy())
                        r["mean"].append(mean.float().cpu().numpy()), r["std"].append(std.float().cpu().numpy())
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p.with_suffix(".tmp"), "wb") as fh:
                np.savez(fh, folds=np.array(folds), C=C, tol=K.tol_of(C), preview=preview, **{k: np.stack(v) for k, v in r.items()})
            p.with_suffix(".tmp").replace(p)
            vol.commit()
            out[f"{comp} {C:g}"] = round(time.time() - t0)
            print(f"L{layer} {comp} C {C:g} done at {round(time.time() - t0)} s", flush=True)
    return {"layer": layer, "done_at_s": out}


@app.function(image=image, cpu=8, memory=65536, volumes={"/acts": vol}, timeout=3600, retries=0)
def select(tag: str) -> dict:
    import numpy as np

    import maxb_core as K

    vol.reload()
    b = base(tag)
    bank, index, choices = {l: ([], []) for l in K.LAYERS}, {l: [] for l in K.LAYERS}, {}
    for comp in K.COMPS:
        R = {(l, C): dict(np.load(b / "grid" / comp / f"L{l}_C{C:g}.npz")) for l in K.LAYERS for C in K.C_GRID}  # each read once
        folds = R[(K.LAYERS[0], K.C_GRID[0])]["folds"].tolist()
        V = np.stack([np.stack([R[(l, C)]["val_auroc"] for C in K.C_GRID]) for l in K.LAYERS])
        ok = np.stack([np.stack([np.maximum(R[(l, C)]["grad_full"], R[(l, C)]["grad_val"]) <= R[(l, C)]["tol"] for C in K.C_GRID]) for l in K.LAYERS])
        li, ci = K.choose(V, ok)
        choices[comp] = {"folds": folds, "layer": [[K.LAYERS[x] if x >= 0 else None for x in row] for row in li],
                         "C": [[K.C_GRID[x] if x >= 0 else None for x in row] for row in ci], "unchosen_columns": int((li < 0).sum())}
        for i, f in enumerate(folds):
            for k in range(V.shape[3]):
                if li[i, k] < 0:
                    continue
                l, C = K.LAYERS[li[i, k]], K.C_GRID[ci[i, k]]
                z = R[(l, C)]
                W2, b2 = K.compile_probe(z["W"][i][:, k].astype(np.float64), z["b"][i][k], z["mean"][i].astype(np.float64), z["std"][i].astype(np.float64), C)
                bank[l][0].append(W2.astype(np.float32)), bank[l][1].append(np.float32(b2))
                index[l].append([comp, k, int(f)])
    (b / "bank").mkdir(exist_ok=True)
    for l, (Ws, bs) in bank.items():
        if Ws:
            np.savez(b / "bank" / f"L{l}.npz", W=np.stack(Ws, 1), b=np.array(bs))
    (b / "bank" / "index.json").write_text(json.dumps({str(l): v for l, v in index.items() if v}))
    (b / "bank" / "env_fold.json").write_text((b / "env_fold.json").read_text())
    (b / "bank" / "choices.json").write_text(json.dumps(choices))
    vol.commit()
    return {"probes_per_layer": {l: len(v) for l, v in index.items() if v}, "unchosen": {c: v["unchosen_columns"] for c, v in choices.items()}}


@app.local_entrypoint()
def main(stage: str, tag: str = "", layers: str = "", comps: str = "", cs: str = "", no_wait: bool = False):
    """`--no-wait` (grid, with `modal run --detach`): spawn the layers and return, so a laptop network drop cannot stop the
    app; finished cells are skipped on a rerun and select reads them from the Volume."""
    import subprocess

    import numpy  # noqa: F401

    import maxb_core as K

    for f in (Path(__file__), HERE / "maxb_core.py"):
        if subprocess.run(["git", "status", "--porcelain", str(f)], capture_output=True, text=True, check=True).stdout:
            raise SystemExit(f"commit {f.name} first: results record the commit that ran")
    tag = tag or subprocess.run(["git", "rev-parse", "--short=8", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    print("tag", tag, flush=True)
    if stage == "prepare":
        print(json.dumps(prepare.remote(tag, json.loads((ROOT / "splits" / "folds_v2.json").read_text()))))
    elif stage == "grid":
        ls = [int(x) for x in layers.split(",")] if layers else K.LAYERS
        calls = [grid.spawn(tag, l, comps.split(",") if comps else list(K.COMPS), [float(x) for x in cs.split(",")] if cs else K.C_GRID) for l in ls]
        if no_wait:
            print(f"spawned {len(calls)} grid containers; not waiting", flush=True)
            return
        for c in calls:
            print(json.dumps(c.get()), flush=True)
    elif stage == "select":
        print(json.dumps(select.remote(tag)))
