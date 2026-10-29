"""Selection-aware null for the nested fresh-probe headline (Modal CPU, saved features only; no GPU, no API).

The nested fresh probe scores each held-out environment at the layer that inner leave-one-environment-out CV picks
among all 64 (the earliest layer within 0.01 of the best inner AUROC), with standardized L2 logistic regression,
C = 0.01. Its committed null was fixed-layer. Here every label shuffle reruns the whole pipeline: the inner CV over the
64 layers, the layer choice, and the outer fit at the chosen layer, on the same environment folds as the real run.

Isolation, as asked:
1. `--mode reproduce`: the new code with labels NOT shuffled. It mirrors the committed run's numerics
   (analyze_probes.fresh_inner: one label column per inner fit; analyze_probes.fresh: outer fits with the real labels
   plus the same 100 seed-0 swap columns), so it must give the committed chosen layers and nested AUROC. The features'
   sha256 is recorded per readout.
2. `--mode null`: n shuffles (within-pair label swaps, fixed seed, separate from the seed-0 draws). Each one gets its
   own inner CV, layer choice and outer score, all fit as batched label columns on the same features and folds.
3. The nested headline vs the selection-aware null's p95 and p, next to the fixed-layer null and the best-of-64-layers
   null already in results/.

Hardware differs from the committed run (CPU float32 here, A10G float32 there), so the reproduction is checked to the
reported precision: the same chosen layer for every environment and the nested AUROC equal to 4 decimals.

  PYTHONPATH=src python -m modal run experiments/2026-10-06_matched_step_pairs/analyze_selection_null.py \\
    --run 2026-10-06_matched_read_all200 --pairs data/pairs_<stamp>.jsonl --metrics results/metrics_<stamp>.json \\
    --mode reproduce|null [--n-shuffles 200]
"""

import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-matched_selnull_kill-2026-11-06"
PAIRS_VOL = "probe-shift_aswin-surya_sub-matched_pairs_kill-2026-11-06"
KINDS = ["output_end", "output_marker", "pre_call", "call_end", "call_marker"]
C, MAX_ITER, N_NULL_COMMITTED, TOL = 0.01, 2000, 100, 0.01
SHUFFLE_SEED = 20261007  # the selection-aware null's draws; the committed fixed-layer null used seed 0
PT = ZoneInfo("America/Los_Angeles")

app = modal.App(APP)
vol = modal.Volume.from_name(PAIRS_VOL, version=2)
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cpu")
         .uv_pip_install("numpy", "scikit-learn==1.7.2"))


def fit(X, Y, C: float, max_iter: int):
    """analyze_probes.fit, unchanged: scikit-learn's L2 logistic objective for every label column at once."""
    import torch
    import torch.nn.functional as F

    Y = Y.reshape(len(Y), -1).to(X.dtype)
    W = torch.zeros(X.shape[1], Y.shape[1], device=X.device, dtype=X.dtype, requires_grad=True)
    b = torch.zeros(Y.shape[1], device=X.device, dtype=X.dtype, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=max_iter, tolerance_grad=1e-5, tolerance_change=1e-12,
                            history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (C * F.binary_cross_entropy_with_logits(X @ W + b, Y, reduction="sum") + 0.5 * (W * W).sum()) / len(X)
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), b.detach()


def standardize(train, *others):
    mean, sd = train.mean(0), train.std(0, unbiased=False)
    sd = sd.where(sd > 0, sd.new_ones(()))
    return mean, sd, [(x - mean) / sd for x in (train, *others)]


def flips(pair, y0, n: int, seed: int):
    """analyze_probes.within_pair_flips, unchanged: n columns, per pair real and sim swap with probability 1/2."""
    import numpy as np

    rng = np.random.default_rng(seed)
    up = np.unique(pair)
    cols = []
    for _ in range(n):
        flip = dict(zip(up, rng.random(len(up)) < 0.5))
        cols.append(np.where([flip[p] for p in pair], 1 - y0, y0))
    return cols


def choose(auroc_by_layer) -> int:
    """analyze_pairs.choose: the earliest layer within TOL of the best AUROC."""
    best = max(auroc_by_layer)
    return min(l for l, a in enumerate(auroc_by_layer) if a >= best - TOL)


@app.function(image=image, volumes={"/pairs": vol}, cpu=8, memory=32768, timeout=3600)
def prep(run: str, out: str) -> dict:
    """Stack the saved activations per readout (items sorted by pair and version, as analyze_probes.pair_acts) into
    one file each, with the sha256 of every source file and of each stacked array."""
    import hashlib

    import numpy as np

    vol.reload()
    d = Path("/pairs") / run / "acts"
    metas = sorted((json.loads(p.read_text()) for p in d.glob("*.json")), key=lambda m: (m["pair"], m["version"]))
    versions = {}
    for m in metas:
        versions.setdefault(m["pair"], set()).add(m["version"])
    complete = {p for p, v in versions.items() if v == {"real", "sim"}}  # as analyze_probes.pair_acts
    metas = [m for m in metas if m["pair"] in complete]
    files = [d / f"{m['pair']:03d}_{m['version']}.npz" for m in metas]
    file_sha = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    X = np.stack([np.load(f)["acts"] for f in files])  # [item, kind, layer, hidden] fp16
    assert np.isfinite(X).all()
    o = Path("/pairs") / out
    o.mkdir(parents=True, exist_ok=True)
    sha = {}
    for kind in ("output_end", "call_end"):
        A = np.ascontiguousarray(X[:, KINDS.index(kind)])
        np.save(o / f"features_{kind}.npy", A)
        sha[kind] = hashlib.sha256(A.tobytes()).hexdigest()
    vol.commit()
    return {"items": [[m["pair"], m["version"]] for m in metas], "features_sha256": sha,
            "source_files_sha256": hashlib.sha256(json.dumps(file_sha, sort_keys=True).encode()).hexdigest(),
            "n_source_files": len(files)}


@app.function(image=image, volumes={"/pairs": vol}, cpu=16, memory=32768, timeout=6 * 3600,
              retries=modal.Retries(max_retries=1, initial_delay=10.0))
def outer_env(out: str, readout: str, env_of: list, pair_of: list, y0: list, outer: str, mode: str, n_shuffles: int) -> dict:
    """One held-out environment. reproduce: the committed real-label path. null: n_shuffles shuffled pipelines."""
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score

    torch.set_num_threads(16)
    vol.reload()
    A_all = np.load(Path("/pairs") / out / f"features_{readout}.npy", mmap_mode="r")
    env, pair, y0 = np.array(env_of), np.array(pair_of), np.array(y0, dtype=float)
    inside, te_outer = env != outer, env == outer
    inner_envs = np.unique(env[inside])
    if mode == "reproduce":
        Y_inner = torch.from_numpy(y0)  # fresh_inner: one column
        Y_outer = torch.from_numpy(np.stack([y0] + flips(pair, y0, N_NULL_COMMITTED, 0), 1))  # fresh: real + 100 swaps
        n_cols = 1
    else:
        # column 0: the real labels through this same batched pipeline; columns 1..n: the shuffles
        Y_inner = Y_outer = torch.from_numpy(np.stack([y0] + flips(pair, y0, n_shuffles, SHUFFLE_SEED), 1))
        n_cols = 1 + n_shuffles
    labels = Y_inner.reshape(len(y0), -1).numpy()
    inner_auroc = np.zeros((n_cols, 64))
    outer_scores = np.zeros((64, n_cols, int(te_outer.sum())))
    t0 = time.time()
    for layer in range(64):
        A = torch.from_numpy(np.asarray(A_all[:, layer], dtype=np.float32))
        s = np.full((len(y0), n_cols), np.nan)
        for e in inner_envs:
            tr, te = inside & (env != e), env == e
            _, _, (Atr, Ate) = standardize(A[torch.from_numpy(tr)], A[torch.from_numpy(te)])
            W, b = fit(Atr, Y_inner[torch.from_numpy(tr)], C, MAX_ITER)
            s[te] = (Ate @ W + b).numpy()[:, :n_cols]
        inner_auroc[:, layer] = [roc_auc_score(labels[inside, c], s[inside, c]) for c in range(n_cols)]
        _, _, (Atr, Ate) = standardize(A[torch.from_numpy(inside)], A[torch.from_numpy(te_outer)])
        W, b = fit(Atr, Y_outer[torch.from_numpy(inside)], C, MAX_ITER)
        outer_scores[layer] = (Ate @ W + b).numpy()[:, :n_cols].T
        if layer % 16 == 15:
            print(f"{readout} {outer} {mode}: layer {layer + 1}/64, {time.time() - t0:.0f}s", flush=True)
    chosen = [choose(list(inner_auroc[c])) for c in range(n_cols)]  # per label column: its own inner-CV layer
    nested = np.stack([outer_scores[chosen[c], c] for c in range(n_cols)])  # [column, outer item]
    import io

    buf = io.BytesIO()  # every layer's outer scores, for the best-of-64-layers statistic on the same shuffles
    np.save(buf, outer_scores.astype(np.float32))
    return {"outer": outer, "readout": readout, "mode": mode, "chosen": chosen, "nested_scores": nested.tolist(),
            "inner_auroc": inner_auroc.round(5).tolist(), "outer_idx": np.flatnonzero(te_outer).tolist(),
            "all_layer_scores_npy": buf.getvalue(), "seconds": round(time.time() - t0)}


@app.local_entrypoint()
def main(run: str, pairs: str, metrics: str, probes: str, mode: str, n_shuffles: int = 200):
    """`--probes` is the committed probe-score file the metrics were computed from (scores only): it gives the
    committed nested scores, so the observed AUROC is compared unrounded."""
    import hashlib

    import numpy as np
    from sklearn.metrics import roc_auc_score

    assert mode in ("reproduce", "null")
    pairs_path, metrics_path, probes_path = Path(pairs), Path(metrics), Path(probes)
    env_by_pair = {json.loads(l)["pair"]: json.loads(l)["environment"] for l in pairs_path.read_text().splitlines()}
    M = json.loads(metrics_path.read_text())
    P = json.loads(probes_path.read_text())
    pairs_sha = hashlib.sha256(pairs_path.read_bytes()).hexdigest()
    read_cfg = json.loads(b"".join(vol.read_file(f"{run}/config.json")))
    # One selection throughout: the read run, the probe scores and the metrics all come from this pairs file.
    assert read_cfg["pairs_sha256"] == pairs_sha, f"read run {run} was made from another pairs file"
    assert M["pairs_file"] == P["pairs_file"] == pairs_path.name and P["run"] == run, "inputs from different selections"
    assert M["probes_file"] == probes_path.name, "the metrics were not computed from these probe scores"
    stamp = datetime.now(PT).strftime("%Y-%m-%d_%H%M%S")
    out = f"{run}__selnull"
    info = prep.remote(run, out)
    items = info["items"]
    assert items == P["fresh"]["output_end"]["items"] == P["fresh"]["call_end"]["items"], "items differ from the committed run"
    pair = [p for p, _ in items]
    y0 = [float(v == "sim") for _, v in items]
    env = [env_by_pair[p] for p in pair]
    envs = sorted(set(env))
    t0 = time.time()
    jobs = [(out, readout, env, pair, y0, e, mode, n_shuffles) for readout in ("output_end", "call_end") for e in envs]
    parts = list(outer_env.starmap(jobs))
    y0a, paira = np.array(y0), np.array(pair)
    res = {"created": datetime.now(PT).isoformat(timespec="seconds"), "mode": mode, "run": run,
           "pairs_file": pairs_path.name, "pairs_sha256": hashlib.sha256(pairs_path.read_bytes()).hexdigest(),
           "metrics_file": metrics_path.name, "features": info, "C": C, "max_iter": MAX_ITER, "layer_rule_tolerance": TOL,
           "hardware": "Modal CPU, float32, torch 2.8.0 (cpu wheel), 16 threads", "readouts": {}}
    for readout in ("output_end", "call_end"):
        mine = [p for p in parts if p["readout"] == readout]
        n_cols = len(mine[0]["chosen"])
        if mode == "reproduce":
            labels = y0a[:, None]
        else:
            labels = np.stack([y0a] + flips(paira, y0a, n_shuffles, SHUFFLE_SEED), 1)
        nested = np.full((len(y0), n_cols), np.nan)
        chosen = {}
        for p in mine:
            chosen[p["outer"]] = p["chosen"]
            nested[p["outer_idx"]] = np.array(p["nested_scores"]).T
        assert not np.isnan(nested).any(), "an item was not scored by its held-out environment"
        aurocs = np.array([roc_auc_score(labels[:, c], nested[:, c]) for c in range(n_cols)])
        committed = M["methods"]["probe_fresh"][readout]
        # The committed nested score, unrounded: each environment's items at its committed chosen layer.
        committed_nested = np.array([P["fresh"][readout]["layers"][str(committed["chosen_layer_by_env"][e])]["scores"][i]
                                     for i, e in enumerate(env)])
        committed_auroc = float(roc_auc_score(y0a, committed_nested))
        assert round(committed_auroc, 4) == committed["nested"]["all"]["auroc"]
        entry = {"seconds_per_env": {p["outer"]: p["seconds"] for p in mine}}
        if mode == "reproduce":
            mine_layers = {e: v[0] for e, v in chosen.items()}
            # For comparing with the committed run item by item and layer by layer (scores only; no activations).
            entry |= {"inner_auroc_by_env": {p["outer"]: p["inner_auroc"][0] for p in mine},
                      "nested_scores": nested[:, 0].round(6).tolist(), "items": items}
            entry |= {"chosen_layer_by_env": mine_layers, "committed_chosen_layer_by_env": committed["chosen_layer_by_env"],
                      "nested_auroc": float(aurocs[0]), "committed_nested_auroc": committed_auroc,
                      "nested_score_max_abs_diff": float(np.abs(nested[:, 0] - committed_nested).max()),
                      "reproduced": mine_layers == committed["chosen_layer_by_env"]
                      and round(float(aurocs[0]), 4) == round(committed_auroc, 4)}
        else:
            import io

            # The same shuffles' best-of-64-layers statistic: per shuffle, the highest pooled AUROC over all layers.
            all_layers = np.full((64, n_cols, len(y0)), np.nan)
            for p in mine:
                all_layers[:, :, p["outer_idx"]] = np.load(io.BytesIO(p["all_layer_scores_npy"]))
            best_of_64 = np.array([max(roc_auc_score(labels[:, c], all_layers[l, c]) for l in range(64)) for c in range(n_cols)])
            committed_best = max(roc_auc_score(y0a, P["fresh"][readout]["layers"][str(l)]["scores"]) for l in range(64))

            def test(observed, draws):  # ties count as exceedances; 1e-12 absorbs float noise in equal AUROCs
                return {"observed_auroc": float(observed), "null_mean": float(draws.mean()),
                        "null_p95": float(np.percentile(draws, 95)),
                        "p_value": float((1 + (draws >= observed - 1e-12).sum()) / (1 + len(draws)))}

            null_nested, null_best = aurocs[1:], best_of_64[1:]  # column 0 is the real labels
            entry |= {"n_shuffles": n_shuffles, "shuffle_seed": SHUFFLE_SEED,
                      # observed = the committed (GPU) headline, unrounded; and the same statistic from column 0 of this
                      # very batched CPU pipeline, so the comparison does not rest on cross-hardware agreement alone
                      "nested": test(committed_auroc, null_nested),
                      "nested_this_pipeline": test(aurocs[0], null_nested) | {"chosen_layer_by_env": {e: v[0] for e, v in chosen.items()}},
                      "best_of_64_layers": test(committed_best, null_best),
                      "best_of_64_layers_this_pipeline": test(best_of_64[0], null_best),
                      "null_aurocs_nested": null_nested.round(5).tolist(), "null_aurocs_best_of_64": null_best.round(5).tolist(),
                      "chosen_layers_by_env_per_shuffle": {e: v[1:] for e, v in chosen.items()}}
        res["readouts"][readout] = entry
        print(readout, {k: v for k, v in entry.items() if k not in ("null_aurocs_nested", "null_aurocs_best_of_64", "chosen_layers_by_env_per_shuffle", "nested_scores", "items", "inner_auroc_by_env")}, flush=True)
    res["wall_seconds"] = round(time.time() - t0)
    path = HERE / "results" / f"selection_null_{mode}_{stamp}.json"
    with open(path, "x") as f:
        f.write(json.dumps(res, indent=1))
    print(f"wrote {path}")
