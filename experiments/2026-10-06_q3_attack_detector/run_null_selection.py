"""Selection-aware label-shuffle null for the Q3 headline (probe on content-token calls, mean over calls), on Modal CPU.

The headline picks its layer per test fold from the nested models' validation AUROCs, but the committed null refit one
layer at a time (fixed-layer), so it did not cover that choice. Here every shuffle repeats the whole procedure:
  1. fit_layer (CPU, one container per layer and block of label columns): the headline readout's 25 models per layer
     (5 outer, 20 nested) refit from the base run's packed features, exactly as run_q3_attack.probe_layer does:
     - the real labels as single-column fits;
     - the shuffles as one batched fit per model, as the committed null did.
     Call scores are pooled to run scores and saved to the work Volume with the input files' SHA-256s.
  2. select (CPU): per label column, the per-fold layer choice (best mean validation AUROC, honest vs real attack e1,
     earliest layer on ties) and the AUROC of the chosen layers' outer scores on the runs the headline uses. Also:
     - each layer's fixed-layer null on the same draws;
     - the same selection on the base run's saved scores, which must reproduce the committed headline exactly.
Settings (layers, C, iteration cap, seed, readout, folds, data) come from the base run's committed record. No GPU, no
API calls, no activations off Modal, no credentials in the containers.

  uv run modal run --detach experiments/2026-10-06_q3_attack_detector/run_null_selection.py [--test]
"""

import hashlib
import json
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REPO = HERE.parent.parent
APP = "probe-shift_aswin-surya_sub-q3attack_nullsel_kill-2026-11-06"
WORK = "probe-shift_aswin-surya_sub-q3attack_work_kill-2026-11-06"
CPU, MEMORY_MB = 16, 32768  # must match config/null_selection.yaml (checked at launch)

app = modal.App(APP)
work_vol = modal.Volume.from_name(WORK, version=2)
image = (modal.Image.debian_slim(python_version="3.12")  # the run_q3_attack image: same torch and numpy
         .uv_pip_install("torch==2.8.0", "numpy==2.3.3", "scikit-learn==1.7.2", "scipy==1.16.2")
         .add_local_python_source("probe_shift"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def labels(runs: list[dict], seed: int, shuffles: int):
    """Real labels (column 0) and the shuffles (columns 1..), drawn exactly as run_q3_attack.probe_layer draws its
    null: shuffle_within_task with numpy default_rng(seed), one draw after another."""
    import numpy as np

    from probe_shift.attack_eval import E1, HONEST, shuffle_within_task

    arm = np.array([r["arm"] for r in runs])
    task = np.array([r["main_task"] for r in runs])
    y = np.where(arm == E1, 1.0, np.where(arm == HONEST, 0.0, np.nan))
    rng = np.random.default_rng(seed)
    return np.column_stack([y] + [shuffle_within_task(y, task, ~np.isnan(y), rng) for _ in range(shuffles)])


@app.function(image=image, volumes={"/work": work_vol}, cpu=CPU, memory=MEMORY_MB, timeout=3 * 3600)
def fit_layer(cfg: dict, data_id: str, null_id: str, layer: int, block: int, cols: list[int]) -> dict:
    """Outer and nested run scores at one layer for label columns `cols` (0 = real labels, fit one column at a time
    as in probe_layer; shuffles batched)."""
    import numpy as np
    import torch

    from probe_shift import torch_probe as tp
    from probe_shift.attack_eval import pool

    torch.set_num_threads(CPU)
    t0 = time.time()
    work_vol.reload()
    d = Path("/work/data") / data_id
    runs = json.loads((d / "runs.json").read_text())
    t = np.load(d / "tables.npz")
    n = len(runs)
    Y = labels(runs, cfg["seed"], cfg["shuffles"])
    trainable = ~np.isnan(Y[:, 0])
    fold = np.array([r["fold"] for r in runs])
    spec = cfg["readouts"][cfg["readout"]]
    if spec["position"] != "call_end":
        raise ValueError("this null covers call_end readouts only")
    packed = d / "packed" / f"L{layer}_call_end.npy"
    sha = {"packed": sha256_file(packed), "tables": sha256_file(d / "tables.npz"), "runs": sha256_file(d / "runs.json")}
    sel = np.arange(len(t["call_end__run"])) if "kinds" not in spec else np.flatnonzero(np.isin(t["call_end__kind"], spec["kinds"]))
    r_run = t["call_end__run"][sel]
    full = np.load(packed)
    if full.shape[0] != len(t["call_end__run"]):
        raise ValueError(f"L{layer}: {full.shape[0]} packed rows, {len(t['call_end__run'])} in the table")
    X = torch.from_numpy(full[sel]).float()
    del full
    print(f"L{layer} block {block}: {X.shape[0]} rows, columns {cols[0]}..{cols[-1]}, loaded at {time.time() - t0:.0f} s",
          flush=True)

    def fit_score(train_runs, score_runs, Ycols):
        tr, te = np.flatnonzero(train_runs[r_run]), np.flatnonzero(score_runs[r_run])
        Xtr = X[torch.from_numpy(tr)]
        mean, std = Xtr.mean(0), Xtr.std(0, unbiased=False)  # tp.standardize, as probe_layer computes it
        std = std.where(std > 0, std.new_ones(()))
        W, b, n_eval = tp.fit((Xtr - mean) / std, torch.from_numpy(Ycols[r_run[tr]]).float(), cfg["fit"]["C"],
                              cfg["fit"]["max_iter"])
        s = tp.decision((X[torch.from_numpy(te)] - mean) / std, W, b).double().numpy()
        return pool(r_run[te], s, n, cfg["pooling"]), n_eval  # [n, k], NaN outside score_runs

    groups = ([[0]] if 0 in cols else []) + ([[c for c in cols if c != 0]] if any(c != 0 for c in cols) else [])
    outer = np.full((n, len(cols)), np.nan, np.float64)
    nested = np.full((5, 5, n, len(cols)), np.nan, np.float64)
    evals = []
    for group in groups:  # the real column alone (single-column fits, as probe_layer), then the shuffles batched
        idx = [cols.index(c) for c in group]
        for f in range(5):
            rs, ne = fit_score(trainable & (fold != f), fold == f, Y[:, group])
            outer[np.ix_(fold == f, idx)] = rs[fold == f]
            evals.append({"model": f"outer_{f}", "columns": len(group), "n_eval": int(ne)})
        for f in range(5):
            for v in range(5):
                if v == f:
                    continue
                te = (fold == f) | (fold == v)
                rs, ne = fit_score(trainable & ~te, te, Y[:, group])
                nested[f, v][np.ix_(te, idx)] = rs[te]
                evals.append({"model": f"nested_{f}_{v}", "columns": len(group), "n_eval": int(ne)})
        print(f"L{layer} block {block}: {len(group)} columns done at {time.time() - t0:.0f} s", flush=True)
    out = Path("/work/nullsel") / null_id
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"L{layer}_block{block}.npz", cols=np.array(cols), outer=outer, nested=nested)
    info = {"layer": layer, "block": block, "columns": [cols[0], cols[-1]], "inputs_sha256": sha, "fits": evals,
            "seconds": round(time.time() - t0)}
    (out / f"L{layer}_block{block}.json").write_text(json.dumps(info))  # written last: marks the block done
    work_vol.commit()
    return info


@app.function(image=image, volumes={"/work": work_vol}, cpu=8, memory=32768, timeout=3600)
def select(cfg: dict, data_id: str, null_id: str, blocks: int) -> dict:
    """Per label column: the per-fold layer choice and the AUROC (honest vs e1) of the chosen layers' outer scores on
    the runs the headline uses; each layer's fixed-layer AUROC; and the same selection on the base run's saved scores."""
    import numpy as np

    from probe_shift.attack_eval import E1, HONEST, auc

    work_vol.reload()
    d = Path("/work/data") / data_id
    runs = json.loads((d / "runs.json").read_text())
    n, layers = len(runs), [str(x) for x in cfg["layers"]]
    Y = labels(runs, cfg["seed"], cfg["shuffles"])
    k = Y.shape[1]
    arm = np.array([r["arm"] for r in runs])
    cell = np.array([r["cell"] for r in runs])
    fold = np.array([r["fold"] for r in runs])
    trainable = ~np.isnan(Y[:, 0])
    src = Path("/work/nullsel") / null_id
    outer, nested = {}, {}
    for layer in layers:
        o, ne = np.full((n, k), np.nan), np.full((5, 5, n, k), np.nan)
        for b in range(blocks):
            z = np.load(src / f"L{layer}_block{b}.npz")
            o[:, z["cols"]], ne[..., z["cols"]] = z["outer"], z["nested"]
        if np.isnan(o).all(0).any():
            raise ValueError(f"L{layer}: some label columns have no scores")
        outer[layer], nested[layer] = o, ne

    def paired(ok):  # as analyze_q3_attack: honest runs that pass; attack cells passing in all three attack arms
        failing = set(cell[(arm != HONEST) & ~ok])
        return np.where(arm == HONEST, ok, ~np.isin(cell, list(failing)))

    def one_auc(s, y):
        return float(auc(s, y, np.ones((1, len(s))))[0])

    def run(outer_l, nested_l, y):
        """analyze_q3_attack.evaluate's layer choice and outer scores for one label vector, then the e1 AUROC on its
        evaluation runs (eval_mask: runs with a score, paired, honest and e1)."""
        chosen, score = {}, np.full(n, np.nan)
        for f in range(5):
            inner = {}
            for layer in layers:
                vals = []
                for v in range(5):
                    if v == f:
                        continue
                    s = nested_l[layer][f, v]
                    m = trainable & (fold == v) & ~np.isnan(s)
                    if len(np.unique(y[m])) == 2:
                        vals.append(one_auc(s[m], y[m]))
                inner[layer] = float(np.mean(vals)) if vals else np.nan
            ok = {l: x for l, x in inner.items() if not np.isnan(x)}
            chosen[f] = max(ok, key=ok.get)
            score[fold == f] = outer_l[chosen[f]][fold == f]
        caught_ok = np.all([~np.isnan(nested_l[chosen[f]][f, v]) | (fold != f) for f in range(5) for v in range(5) if v != f], 0)
        use = paired(~np.isnan(score) & caught_ok) & np.isin(arm, [HONEST, E1])
        return one_auc(score[use], y[use]), chosen, score, use

    # 1. The base run's saved (GPU) scores, real labels: must reproduce the committed headline exactly.
    base = Path("/work/runs") / cfg["base_run"]
    saved_o, saved_n, saved_sha = {}, {}, {}
    for layer in layers:
        z = np.load(base / f"scores_L{layer}.npz")
        saved_o[layer] = z[f"{cfg['readout']}__{cfg['pooling']}__task_outer"].astype(np.float64)
        saved_n[layer] = z[f"{cfg['readout']}__{cfg['pooling']}__task_nested"].astype(np.float64)
        saved_sha[layer] = sha256_file(base / f"scores_L{layer}.npz")
    s_auc, s_chosen, s_score, s_use = run(saved_o, saved_n, Y[:, 0])
    # 2. Every label column on the CPU refits: column 0 (real labels) checks the refit, 1.. are the null.
    sel_auc, sel_chosen = [], []
    for j in range(k):
        a, ch, score, use = run({l: outer[l][:, j] for l in layers}, {l: nested[l][..., j] for l in layers}, Y[:, j])
        sel_auc.append(a)
        sel_chosen.append([ch[f] for f in range(5)])
        if j == 0:
            refit_score, refit_use = score, use
    fixed = {l: [one_auc(outer[l][s_use, j], Y[s_use, j]) for j in range(k)] for l in layers}
    diff = {l: {"outer_max_abs": float(np.nanmax(np.abs(outer[l][:, 0] - saved_o[l]))),
                "nested_max_abs": float(np.nanmax(np.abs(nested[l][..., 0] - saved_n[l])))} for l in layers}
    return {"saved_scores": {"auroc": s_auc, "chosen_layer_by_fold": [s_chosen[f] for f in range(5)],
                             "n_runs": {"honest": int((s_use & (arm == HONEST)).sum()), "e1": int((s_use & (arm == E1)).sum())},
                             "selected_score_sha256": hashlib.sha256(np.ascontiguousarray(s_score).tobytes()).hexdigest(),
                             "eval_mask_sha256": hashlib.sha256(s_use.tobytes()).hexdigest(), "files_sha256": saved_sha},
            "refit_real": {"auroc": sel_auc[0], "chosen_layer_by_fold": sel_chosen[0],
                           "same_eval_runs_as_saved": bool((refit_use == s_use).all()),
                           "max_abs_diff_vs_saved": diff,
                           "selected_score_max_abs_diff_vs_saved": float(np.nanmax(np.abs(refit_score - s_score)))},
            "null_selection_aware": {"auroc": sel_auc[1:], "chosen_layer_by_fold": sel_chosen[1:]},
            "null_fixed_layer": {l: x[1:] for l, x in fixed.items()},
            "fixed_layer_real": {l: x[0] for l, x in fixed.items()}}


def wait(call):
    for _ in range(120):  # our network: retry for up to an hour; remote errors are raised
        try:
            return call.get()
        except (socket.gaierror, ConnectionError, TimeoutError, modal.exception.ConnectionError,
                modal.exception.ServiceError) as e:
            print(f"lost the connection ({e!r}); retrying in 30 s", flush=True)
            time.sleep(30)
    raise TimeoutError("no connection to Modal for an hour; block outputs are on the work Volume")


@app.local_entrypoint()
def main(test: bool = False, resume: str = ""):
    """`--test`: a timing pilot on one layer. `--resume <null_id>`: reuse that launch's settings and every block already
    marked done on the work Volume, refit only the rest, then select."""
    import io

    import numpy as np
    import yaml

    if resume:
        launch = json.loads(b"".join(work_vol.read_file(f"nullsel/{resume}/launch.json")))
    else:
        own = yaml.safe_load((HERE / "config" / "null_selection.yaml").read_text())
        if (own["cpu"], own["memory_mb"]) != (CPU, MEMORY_MB):
            raise ValueError("config cpu/memory differ from the script's")
        record = json.loads((HERE / "results" / f"run_{own['base_run']}.json").read_text())
        base = record["config"]
        cfg = {k: base[k] for k in ("layers", "fit", "seed", "readouts")} | {
            k: own[k] for k in ("readout", "pooling", "shuffles", "base_run")}
        blocks = own["column_blocks"]
        if test:
            cfg |= {"layers": own["test"]["layers"], "shuffles": own["test"]["shuffles"]}
            blocks = own["test"]["column_blocks"]
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=REPO).stdout.strip())
        null_id = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S") + ("_test" if test else "")
        launch = {"null_id": null_id, "cfg": cfg, "blocks": blocks, "data_id": record["data_id"], "commit": head,
                  "dirty": dirty, "test": test}
        with work_vol.batch_upload() as batch:  # the launch record first, so an interrupted sweep can be resumed
            batch.put_file(io.BytesIO(json.dumps(launch).encode()), f"nullsel/{null_id}/launch.json")
    cfg, blocks, data_id, null_id = launch["cfg"], launch["blocks"], launch["data_id"], launch["null_id"]
    head, dirty, test = launch["commit"], launch["dirty"], launch["test"]
    k = cfg["shuffles"] + 1
    edges = np.linspace(0, k, blocks + 1).round().astype(int)
    col_blocks = [list(range(edges[b], edges[b + 1])) for b in range(blocks)]
    done = {e.path.rsplit("/", 1)[-1] for e in work_vol.listdir(f"nullsel/{null_id}")}
    print(f"null {null_id}: base {cfg['base_run']} (data {data_id}), layers {cfg['layers']}, {cfg['shuffles']} shuffles "
          f"in {blocks} blocks, C {cfg['fit']['C']}, seed {cfg['seed']} | commit {head[:8]}{' (dirty)' if dirty else ''}",
          flush=True)
    started = time.time()
    calls = [(layer, b, fit_layer.spawn(cfg, data_id, null_id, layer, b, cols))
             for layer in cfg["layers"] for b, cols in enumerate(col_blocks) if f"L{layer}_block{b}.json" not in done]
    infos = [json.loads(b"".join(work_vol.read_file(f"nullsel/{null_id}/{name}"))) for name in sorted(done)
             if name.endswith(".json") and name != "launch.json"]
    for layer, b, call in calls:
        info = wait(call)
        infos.append(info)
        capped = sum(x["n_eval"] >= cfg["fit"]["max_iter"] for x in info["fits"])
        print(f"L{layer} block {b}: {info['seconds']} s, {len(info['fits'])} fits, {capped} may have hit the cap", flush=True)
    sel = wait(select.spawn(cfg, data_id, null_id, blocks))
    null = np.array(sel["null_selection_aware"]["auroc"])
    observed = sel["saved_scores"]["auroc"]  # the committed headline, recomputed from the saved scores
    summary = {"observed_headline_auroc": observed, "refit_real_auroc": sel["refit_real"]["auroc"],
               "null_selection_aware": {"n": int(len(null)), "mean": float(null.mean()), "p95": float(np.percentile(null, 95)),
                                        "p99": float(np.percentile(null, 99)), "max": float(null.max()),
                                        "p_value": float((1 + (null >= observed).sum()) / (1 + len(null)))},
               "null_fixed_layer": {l: {"mean": float(np.mean(x)), "p95": float(np.percentile(x, 95)), "max": float(np.max(x))}
                                    for l, x in sel["null_fixed_layer"].items()}}
    out = {"null_id": null_id, "git": head, "git_dirty": dirty, "test": test, "data_id": data_id, "config": cfg,
           "column_blocks": blocks, "modal": {"app": APP, "work_volume": WORK, "cpu": CPU, "memory_mb": MEMORY_MB,
                                              "raw": f"{WORK}/nullsel/{null_id}"},
           "wall_seconds": round(time.time() - started), "cpu_seconds": sum(x["seconds"] for x in infos) * CPU,
           "summary": summary, "selection": sel, "fits": infos}
    path = HERE / "results" / f"null_selection_{null_id}.json"
    if path.exists():
        raise FileExistsError(f"{path} exists; results/ is append-only")
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"results -> {path}")
