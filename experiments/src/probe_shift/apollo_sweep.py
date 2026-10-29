"""Apollo's deception probe (the `probe_shift.apollo` port) fit at every layer, with both readouts, from one forward
pass per dialogue.

Readouts, for layer index L of a model with N decoder blocks:
- "block_output": the raw output of decoder block L, i.e. the residual stream after L + 1 blocks.
- "apollo": Apollo's hidden_states[L] from a model cut at L blocks (what the Llama check reproduced): the final
  RMSNorm applied to the residual stream after L blocks (to the embeddings when L = 0).
`depth` puts both on one axis: blocks applied / N.

The probe is the port's: per-token logistic regression on standardized activations (no intercept, C = 1/reg_coeff),
fit on the instructed pairs minus a held-out share of pairs. A dialogue's score is the mean of its per-token scores.
That mean is linear in the activations, so eval sets keep only each dialogue's mean activation per layer.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from probe_shift.apollo import render_detect

READOUTS = ("block_output", "apollo")


def depth(readout: str, layer: int, n_blocks: int) -> float:
    return (layer + 1 if readout == "block_output" else layer) / n_blocks


def roleplaying_from_generations(generations: list[dict]) -> list[dict]:
    """Port-format Roleplaying dialogues from responses another model wrote to the same prompts (unlabeled)."""
    return [{"id": g["id"], "label": None, "grade": None,
             "messages": [{"role": "system", "content": g["system"]}, {"role": "user", "content": g["user"]},
                          {"role": "assistant", "content": g["prefix"] + g["response"]}],
             "detect": [len(g["prefix"]), len(g["prefix"]) + len(g["response"])]}
            for g in generations]


def collect(base_model, tokenizer, dialogues: list[dict], template_kwargs: dict, per_token: bool, log=print) -> tuple[dict, list[int]]:
    """Activations over each dialogue's detected tokens at every layer, both readouts, and each dialogue's count of
    detected tokens.

    per_token: {readout: list (one per dialogue) of [N, n_detect, H] float32 arrays}.
    Otherwise: {readout: [n_dialogues, N, H] float32 per-dialogue means}, NaN for dialogues with no detected tokens.
    """
    import torch

    n = len(base_model.layers)
    norm = base_model.norm
    device = base_model.embed_tokens.weight.device
    hidden = base_model.config.hidden_size
    out = {r: [] for r in READOUTS} if per_token else {r: np.full((len(dialogues), n, hidden), np.nan, np.float32) for r in READOUTS}
    captured, det, counts = {}, [], []

    def grab(k):  # k = -1: the embeddings; else decoder block k
        def hook(module, args, output):
            h = (output[0] if isinstance(output, tuple) else output)[0, det]
            feats = {}
            if k >= 0:
                feats["block_output", k] = h
            if k + 1 < n:
                feats["apollo", k + 1] = norm(h.to(norm.weight.device))  # the model may span GPUs
            for key, x in feats.items():
                captured[key] = x.float() if per_token else x.float().mean(0)
        return hook

    hooks = [base_model.embed_tokens.register_forward_hook(grab(-1))]
    hooks += [block.register_forward_hook(grab(k)) for k, block in enumerate(base_model.layers)]
    t0 = time.time()
    try:
        for i, d in enumerate(dialogues):
            ids, det[:] = render_detect(tokenizer, d, template_kwargs)
            counts.append(len(det))
            if not det:  # statements of five words or fewer, as in Apollo
                if per_token:
                    for r in READOUTS:
                        out[r].append(np.zeros((n, 0, hidden), np.float32))
                continue
            captured.clear()
            with torch.inference_mode():
                base_model(input_ids=torch.tensor([ids], device=device), use_cache=False)
            for r in READOUTS:
                stacked = torch.stack([captured[r, l].to("cpu") for l in range(n)]).numpy()
                if per_token:
                    out[r].append(stacked)
                else:
                    out[r][i] = stacked
            if (i + 1) % 100 == 0:
                log(f"  {i + 1}/{len(dialogues)} dialogues, {time.time() - t0:.0f}s")
    finally:
        for h in hooks:
            h.remove()
    return out, counts


def split_pairs(train: list[dict], val_fraction: float, seed: int) -> np.ndarray:
    """The port's held-out split: a random share of statement pairs, both members held out together."""
    pairs = sorted({d["pair"] for d in train})
    rng = np.random.default_rng(seed)
    val_pairs = set(rng.choice(pairs, int(round(val_fraction * len(pairs))), replace=False).tolist())
    return np.array([d["pair"] in val_pairs for d in train])


def auroc(labels, scores) -> float | None:
    """AUROC over labeled dialogues with a score. Used for the run's printout; analyze_* recomputes everything from
    the saved scores and fails loudly on labeled dialogues without one."""
    from sklearn.metrics import roc_auc_score

    labels, scores = np.asarray(labels, float), np.asarray(scores, float)
    ok = ~np.isnan(labels) & ~np.isnan(scores)
    return float(roc_auc_score(labels[ok], scores[ok])) if len(set(labels[ok])) == 2 else None


def sweep(cfg: dict, base_model, tokenizer, train: list[dict], evals: dict[str, list[dict]], log=print) -> tuple[dict, dict]:
    """Fit the probe at every layer and readout; score the held-out pairs and every eval set.

    Returns (result, probes). result["by_probe"] has one row per (readout, layer) with its AUROCs, and
    result["scores"] the per-dialogue scores. probes[readout] holds coef, scaler_mean, scaler_scale as [N, H]."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    kwargs = cfg.get("chat_template") or {}
    n = len(base_model.layers)
    t0 = time.time()
    log(f"instructed pairs: {len(train)} dialogues")
    train_acts, _ = collect(base_model, tokenizer, train, kwargs, per_token=True, log=log)
    eval_means, eval_counts = {}, {}
    for name, ds in evals.items():
        log(f"{name}: {len(ds)} dialogues")
        eval_means[name], eval_counts[name] = collect(base_model, tokenizer, ds, kwargs, per_token=False, log=log)
    forward_s = time.time() - t0

    is_val = split_pairs(train, cfg["val_fraction"], cfg["split_seed"])
    y = np.array([d["label"] for d in train])
    fit_rows = [i for i, a in enumerate(train_acts[READOUTS[0]]) if not is_val[i] and a.shape[1]]
    val_rows = [i for i, a in enumerate(train_acts[READOUTS[0]]) if is_val[i] and a.shape[1]]

    def fit(job):
        r, l = job
        X = np.concatenate([train_acts[r][i][l] for i in fit_rows])
        yt = np.concatenate([np.full(train_acts[r][i].shape[1], y[i]) for i in fit_rows])
        scaler = StandardScaler().fit(X)
        lr = LogisticRegression(C=1 / cfg["reg_coeff"], fit_intercept=False, random_state=42, max_iter=cfg["max_iter"])
        lr.fit(scaler.transform(X), yt)
        return job, lr.coef_[0], scaler.mean_, scaler.scale_, int(lr.n_iter_[0]), int(len(yt))

    t1 = time.time()
    jobs = [(r, l) for r in READOUTS for l in range(n)]
    with threadpool_limits(limits=cfg["blas_threads_per_fit"]), ThreadPoolExecutor(cfg["fit_workers"]) as pool:
        fitted = list(pool.map(fit, jobs))
    fit_s = time.time() - t1
    log(f"fit {len(jobs)} probes in {fit_s:.0f}s")

    probes = {r: {k: np.zeros((n, base_model.config.hidden_size), np.float32) for k in ("coef", "scaler_mean", "scaler_scale")} for r in READOUTS}
    rows, scores = [], {"instructed_val": {}, **{name: {} for name in evals}}
    for (r, l), coef, mean, scale, n_iter, n_tok in fitted:
        probes[r]["coef"][l], probes[r]["scaler_mean"][l], probes[r]["scaler_scale"][l] = coef, mean, scale
        val = np.array([(((train_acts[r][i][l] - mean) / scale) @ coef).mean() for i in val_rows])
        row = {"readout": r, "layer": l, "depth": depth(r, l, n), "train_tokens": n_tok, "n_iter": n_iter,
               "converged": n_iter < cfg["max_iter"], "instructed_val_auroc": auroc(y[val_rows], val)}
        scores["instructed_val"][f"{r}:{l}"] = val.round(5).tolist()
        for name, ds in evals.items():
            s = ((eval_means[name][r][:, l] - mean) / scale) @ coef
            row[f"{name}_auroc"] = auroc([np.nan if d["label"] is None else d["label"] for d in ds], s)
            scores[name][f"{r}:{l}"] = [None if np.isnan(x) else round(float(x), 5) for x in s]
        rows.append(row)
    if not all(row["converged"] for row in rows):
        log(f"WARNING: {sum(not row['converged'] for row in rows)} probes hit max_iter")
    result = {
        "n_blocks": n, "forward_seconds": round(forward_s, 1), "fit_seconds": round(fit_s, 1),
        "n_train_dialogues": len(fit_rows), "n_val_dialogues": len(val_rows),
        "n_dialogues_without_tokens": int(sum(a.shape[1] == 0 for a in train_acts[READOUTS[0]])),
        "val_ids": [train[i]["id"] for i in val_rows], "val_labels": y[val_rows].tolist(),
        "eval_sets": {name: {"ids": [d["id"] for d in ds], "labels": [d["label"] for d in ds], "grades": [d.get("grade") for d in ds],
                             "n_detect": eval_counts[name]}
                      for name, ds in evals.items()},
        "by_probe": rows, "scores": scores,
    }
    return result, probes
