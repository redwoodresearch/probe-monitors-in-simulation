"""Fit step 1's two probe readouts, choose each one's layer on dev, and test on the held-out implied set (Modal CPU).

For each extracted layer (the run manifest's: 16 for step 1, 8 for the retrain) and both readouts:
- (a) last token: one row per dialogue, the span's last token;
- (b) per token (Apollo): every span token is a row with its dialogue's label, and a dialogue's score is the mean of
  its tokens' logits.
Standardize on the train rows, then fit logistic regression (C = 0.1, no intercept; Apollo's settings). Score dev and
held-out dialogues. The true labels and the 50 null draws share one feature matrix, so they are one batched L-BFGS
fit on scikit-learn's objective (`probe_shift.torch_probe`, CPU): one scikit-learn fit per draw took over 12 minutes per
layer. `--pilot` fits one layer and compares the solver with scikit-learn on 100 train dialogues, writing nothing to
the Volume.

Per readout, the layer is the earliest within 0.01 of the best dev AUROC, or `--layer L` fixes it for the true and
the null fits alike (N1). The readout carried forward is the one with the higher dev AUROC (README).

Null: 50 draws that flip labels at random within each train and dev pair (--null-unit task: one coin per task for all
its pairs), then redo the fits and the layer choice. The
held-out AUROC at each draw's chosen layer, with the true labels, gives the selection-aware null for bar (a-i).

Writes the frozen probes (both readouts, at their chosen layers: weights, means, scales) and every AUROC to the
Volume, and a summary to results/.

  PYTHONPATH=src uv run modal run experiments/2026-10-10_eval_awareness_probe/run_fit.py --run step1_<stamp> [--pilot]
"""

import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-evalaware_fit_kill-2026-11-09"
VOL = "probe-shift_aswin-surya_sub-evalaware_acts_kill-2026-11-09"
LAYERS = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 63]
READOUTS, C, TOL, SHUFFLES, SEED = ("last_token", "per_token"), 0.1, 0.01, 50, 20261010
MAX_ITER, GRAD_TOL, CPUS, PILOT_LAYER = 2000, 1e-5, 16, 32
ACCEPT_GRAD = 1e-4 * C  # converged = scikit-learn's default stopping rule: max |gradient| <= 1e-4 on its objective (the
# mean loss plus ||w||^2 / (2 C n)), which is torch_probe's objective divided by C

app = modal.App(APP)
vol = modal.Volume.from_name(VOL, version=2)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy", "scikit-learn==1.7.2")
         .uv_pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cpu").add_local_python_source("probe_shift"))


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def standardized(X, rows, tr):
    """float64: in float32 the batched L-BFGS stopped on loss changes below float32 resolution (pilot, layer 32)."""
    Xr = X[rows].astype("float64")
    mu, sd = Xr[tr].mean(0), Xr[tr].std(0) + 1e-6
    return (Xr - mu) / sd, mu, sd


def fit_readout(X, rows, owner, train, Y):
    """Fit every label draw (Y: dialogues x draws) at once on one readout's rows. Returns the dialogue scores (dialogues x
    draws, the mean of each dialogue's row logits), draw 0's weights and standardization, and the solver's stats."""
    import numpy as np
    import torch

    from probe_shift import torch_probe

    tr = np.isin(owner, train)
    Z, mu, sd = standardized(X, rows, tr)
    Z = torch.from_numpy(Z)
    Ztr, Ytr = Z[torch.from_numpy(tr)], torch.from_numpy(Y[owner[tr]].astype(np.float64))
    t = time.time()
    W, _, evals = torch_probe.fit(Ztr, Ytr, C=C, max_iter=MAX_ITER, tol=GRAD_TOL, intercept=False)
    seconds = time.time() - t
    with torch.no_grad():  # gradient of torch_probe's objective (scikit-learn's, divided by n) at the solution
        grad = (C * Ztr.T @ (torch.sigmoid(Ztr @ W) - Ytr) + W) / len(Ztr)
        logit = (Z @ W).numpy()
    max_grad = float(grad.abs().max())
    if not (np.isfinite(logit).all() and max_grad <= ACCEPT_GRAD):  # fail loudly: an unconverged fit never feeds the gate
        raise RuntimeError(f"fit did not converge: max |gradient| {max_grad:.2e} after {evals} evaluations, {seconds:.0f} s")
    n = len(Y)
    cnt = np.maximum(np.bincount(owner, minlength=n), 1)
    score = np.stack([np.bincount(owner, weights=logit[:, k], minlength=n) for k in range(Y.shape[1])], 1) / cnt[:, None]
    return score, (W[:, 0].numpy(), mu, sd), {"evals": evals, "seconds": round(seconds, 1), "max_grad": max_grad}


def sklearn_check(X, rows, owner, train, Y):
    """Fit all draws jointly on the first 100 train dialogues with torch_probe, and draws 0-2 one by one with
    scikit-learn; compare every row's logits per draw."""
    import numpy as np
    import torch
    from sklearn.linear_model import LogisticRegression

    from probe_shift import torch_probe

    tr = np.isin(owner, train[:100])
    Z, _, _ = standardized(X, rows, tr)
    W, _, evals = torch_probe.fit(torch.from_numpy(Z[tr]), torch.from_numpy(Y[owner[tr]].astype(np.float64)), C=C,
                                  max_iter=MAX_ITER, tol=GRAD_TOL, intercept=False)
    out = {"train_rows": int(tr.sum()), "torch_evals": evals}
    for k in (0, 1, 2):
        sk = LogisticRegression(C=C, fit_intercept=False, max_iter=10000, tol=1e-6).fit(Z[tr], Y[owner[tr], k])
        a, b = Z @ sk.coef_[0], (torch.from_numpy(Z) @ W[:, k]).numpy()
        out[f"draw {k}"] = {"sklearn_n_iter": int(sk.n_iter_[0]), "logit_corr": float(np.corrcoef(a, b)[0, 1]),
                            "logit_max_abs_diff": float(np.abs(a - b).max()), "logit_sd": float(a.std())}
    return out


@app.function(image=image, cpu=CPUS, memory=16384, volumes={"/acts": vol}, timeout=3600, retries=0)
def fit(run: str, fit_id: str, pilot: bool = False, null_unit: str = "pair", layer: int = -1, expect_sha256: str = "") -> dict:
    import numpy as np
    import torch

    torch.set_num_threads(CPUS)
    import hashlib

    own = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if expect_sha256 and own != expect_sha256:  # a spawned fit must run the launching commit's code
        raise ValueError("the deployed run_fit.py is not the launching commit's: redeploy")
    vol.reload()  # a warm container sees the latest commits
    d = Path("/acts") / run
    meta = json.loads((d / "meta.json").read_text())
    layers = json.loads((d / "manifest.json").read_text())["layers"]  # the run's own layers (step 1: LAYERS)
    if pilot and PILOT_LAYER not in layers:
        raise ValueError(f"the pilot layer {PILOT_LAYER} was not extracted")
    if layer >= 0 and layer not in layers:
        raise ValueError(f"the fixed layer {layer} was not extracted")
    split = np.array([m["split"] for m in meta])
    y = np.array([m["label"] for m in meta])
    pair = np.array([m["pair"] for m in meta])
    rng = np.random.default_rng(SEED)
    labels = [y]
    task = np.array([m["task"] for m in meta])
    unit = pair if null_unit == "pair" else task  # "task": one coin per task, for all its train and dev pairs
    for _ in range(SHUFFLES):  # flip within train and dev pairs; held-out labels stay true
        flip = {p: rng.random() < 0.5 for p in sorted(set(unit[split != "heldout"]))}  # sorted: the seed reproduces the draws
        labels.append(np.array([1 - l if s != "heldout" and flip[u] else l for l, s, u in zip(y, split, unit)]))
    Y = np.stack(labels, 1)  # dialogues x draws; column 0 = the true labels
    train = np.flatnonzero(split == "train")
    sel = {"last_token": (np.array([m["row_start"] + m["n_rows"] - 1 for m in meta]), np.arange(len(meta))),
           "per_token": (np.concatenate([np.arange(m["row_start"], m["row_start"] + m["n_rows"]) for m in meta]),
                         np.concatenate([np.full(m["n_rows"], i) for i, m in enumerate(meta)]))}
    dev, held = np.flatnonzero(split == "dev"), np.flatnonzero(split == "heldout")
    res = {r: {"dev": {}, "heldout": {}, "null_dev": {}, "null_heldout": {}} for r in READOUTS}
    probes = {r: {} for r in READOUTS}
    mech = np.array([m["mechanism"] or "" for m in meta])
    by_mech = {r: {} for r in READOUTS}
    true_scores = {r: {} for r in READOUTS}  # dialogue scores with the true labels, per layer
    solver = {r: {} for r in READOUTS}
    for L in [PILOT_LAYER] if pilot else layers:
        t = time.time()
        X = np.load(d / f"L{L}.npy").astype(np.float32)
        load_s = time.time() - t
        for r in READOUTS:
            score, wts, solver[r][L] = fit_readout(X, *sel[r], train, Y)
            solver[r][L]["load_s"] = round(load_s, 1)
            for k in range(SHUFFLES + 1):
                dv, hv = auroc(Y[dev, k], score[dev, k]), auroc(y[held], score[held, k])
                if k == 0:
                    res[r]["dev"][L], res[r]["heldout"][L] = dv, hv
                    probes[r][L] = {"w": wts[0].tolist(), "mu": wts[1].tolist(), "sd": wts[2].tolist()}
                    by_mech[r][L] = {mm: auroc(y[held][mech[held] == mm], score[held, 0][mech[held] == mm]) for mm in sorted(set(mech[held]))}
                    true_scores[r][L] = score[:, 0]
                else:
                    res[r]["null_dev"].setdefault(k, {})[L] = dv
                    res[r]["null_heldout"].setdefault(k, {})[L] = hv
        print(f"L{L}: dev " + ", ".join(f"{r} {res[r]['dev'][L]:.3f}" for r in READOUTS) + " | solver " +
              ", ".join(f"{r} {solver[r][L]}" for r in READOUTS), flush=True)
    if pilot:
        return {"pilot_layer": PILOT_LAYER, "cpus": CPUS, "solver": {r: solver[r][PILOT_LAYER] for r in READOUTS},
                "dev_auroc": {r: res[r]["dev"][PILOT_LAYER] for r in READOUTS},
                "sklearn_check": {r: sklearn_check(X, *sel[r], train, Y) for r in READOUTS}}
    pick = (lambda dv: layer) if layer >= 0 else (lambda dv: min(L for L in layers if dv[L] >= max(dv.values()) - TOL))  # --layer: fixed, for true and null fits
    summary = {}
    for r in READOUTS:
        L = pick(res[r]["dev"])
        null = np.array([res[r]["null_heldout"][k][pick(res[r]["null_dev"][k])] for k in range(1, SHUFFLES + 1)])
        summary[r] = {"layer": L, "dev_auroc": res[r]["dev"][L], "heldout_auroc": res[r]["heldout"][L],
                      "heldout_by_mechanism": by_mech[r][L], "null_heldout_p95": float(np.percentile(null, 95)),
                      "null_heldout_mean": float(null.mean())}
    chosen = max(READOUTS, key=lambda r: summary[r]["dev_auroc"])
    pipeline = []  # the whole selection per null draw: each readout's layer, then the readout with the higher dev AUROC
    for k in range(1, SHUFFLES + 1):
        lay = {r: pick(res[r]["null_dev"][k]) for r in READOUTS}
        rk = max(READOUTS, key=lambda r: res[r]["null_dev"][k][lay[r]])
        pipeline.append(res[rk]["null_heldout"][k][lay[rk]])
    pipeline_p95 = float(np.percentile(pipeline, 95))
    frozen = {r: {"layer": summary[r]["layer"], **probes[r][summary[r]["layer"]]} for r in READOUTS}
    fd = d / "fits" / fit_id  # one directory per fit: a refit never overwrites the probes an earlier score used
    fd.mkdir(parents=True, exist_ok=False)
    blob = json.dumps({"chosen_readout": chosen, "C": C, "probes": frozen}).encode()
    (fd / "probes.json").write_bytes(blob)
    (fd / "fit_all.json").write_text(json.dumps({"res": res, "by_mechanism": by_mech, "solver": solver}))
    vol.commit()
    heldout_scores = {r: [{"id": meta[i]["id"], "label": int(y[i]), "mechanism": meta[i]["mechanism"],
                           "score": float(true_scores[r][summary[r]["layer"]][i])} for i in held] for r in READOUTS}
    manifest = json.loads((d / "manifest.json").read_text())
    return {"fit_id": fit_id, "settings": {"layer": layer, "null_unit": null_unit, "shuffles": SHUFFLES, "C": C, "seed": SEED, "max_iter": MAX_ITER,
                                           "grad_tol": GRAD_TOL, "run_fit_sha256": own},
            "probes_sha256": hashlib.sha256(blob).hexdigest(), "sets_sha256": manifest["sets_sha256"],
            "chosen_readout": chosen, "null_heldout_p95 (selected pipeline)": pipeline_p95, "summary": summary,
            "heldout_scores_at_chosen_layer": heldout_scores,
            "dev_by_layer": {r: res[r]["dev"] for r in READOUTS}, "heldout_by_layer": {r: res[r]["heldout"] for r in READOUTS},
            "solver_by_layer": solver, "max_grad (all fits)": max(v["max_grad"] for r in READOUTS for v in solver[r].values())}


@app.local_entrypoint()
def main(run: str, pilot: bool = False, null_unit: str = "pair", layer: int = -1):
    if null_unit not in ("pair", "task"):
        raise ValueError("null_unit must be pair or task")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = fit.remote(run, f"fit_{stamp}", pilot, null_unit, layer)
    out = {"run": run, "volume": VOL, "shuffles": SHUFFLES, "seed": SEED, "C": C, "max_iter": MAX_ITER, "grad_tol": GRAD_TOL, "null_unit": null_unit,
           **({"fixed_layer": layer} if layer >= 0 else {}), **out}
    path = HERE / "results" / f"fit_{'pilot_' if pilot else ''}{stamp}.json"
    if path.exists():  # two fits launched in the same second overwrote one result on 2026-10-11; results/ is append-only
        path = path.with_name(f"fit_{'pilot_' if pilot else ''}{stamp}_{run}.json")
    if path.exists():
        raise FileExistsError(f"{path} exists")
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out if pilot else {k: out[k] for k in ("chosen_readout", "summary", "max_grad (all fits)")}, indent=1))
    print(f"-> {path}")
