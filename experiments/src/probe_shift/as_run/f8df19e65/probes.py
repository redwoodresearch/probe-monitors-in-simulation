"""Linear probes on GPU (torch): L2 logistic regression and per-fold layer and C selection.

Runs inside GPU containers; torch and sklearn are imported lazily so this module imports anywhere.
"""


def fit_logreg(X, y, C: float, max_iter: int, w0=None):
    """Minimize mean log loss + |w - prior|^2 / (2 C n), i.e. sklearn's L2 objective divided by C n, with the intercept
    unpenalized. With w0 (weights with the intercept last), the prior centre and the starting point are w0: updating
    a trained probe with a few new examples. Returns (weights with the intercept last, max |gradient|)."""
    import torch
    import torch.nn.functional as F

    n, h = X.shape
    prior = torch.zeros(h, device=X.device) if w0 is None else w0[:-1].detach()
    w = (torch.zeros(h + 1, device=X.device) if w0 is None else w0.detach().clone()).requires_grad_(True)
    opt = torch.optim.LBFGS([w], lr=1, max_iter=max_iter, tolerance_grad=1e-7, tolerance_change=1e-10,
                            history_size=20, line_search_fn="strong_wolfe")
    lam = 1.0 / (C * n)

    def closure():
        opt.zero_grad()
        d = w[:-1] - prior
        loss = F.binary_cross_entropy_with_logits(X @ w[:-1] + w[-1], y) + 0.5 * lam * (d @ d)
        loss.backward()
        return loss

    opt.step(closure)
    closure()
    return w.detach(), float(w.grad.abs().max())


def select(X, y, train, val, layers: list[int], grid: list[float], tol: float, max_iter: int) -> dict:
    """Fit every layer and C on `train` and choose on `val` AUROC: the earliest layer within `tol` of the best, with
    that layer's best C. X is [layer, call, hidden] on the GPU; y, train and val are numpy. Returns the chosen probe:
    layer index ("li") and id, C, val_auroc, grad_max, weights "w" (intercept last), standardization "mean" and "sd"
    (training rows), "scores" for every call, and the per-layer validation curve."""
    import torch
    from sklearn.metrics import roc_auc_score

    ytr = torch.tensor(y[train], dtype=torch.float32, device=X.device)
    best = {}
    for li in range(len(layers)):
        Xl = X[li].float()
        mu, sd = Xl[train].mean(0), Xl[train].std(0).clamp_min(1e-6)
        Z = (Xl - mu) / sd
        for C in grid:
            w, g = fit_logreg(Z[train], ytr, C, max_iter)
            s = (Z @ w[:-1] + w[-1]).cpu().numpy()
            auc = float(roc_auc_score(y[val], s[val]))
            if li not in best or auc > best[li]["val_auroc"]:
                best[li] = {"li": li, "layer": layers[li], "C": C, "val_auroc": auc, "grad_max": g, "scores": s,
                            "w": w, "mean": mu, "sd": sd}
        del Xl, Z
    top = max(b["val_auroc"] for b in best.values())
    chosen = best[min(l for l, b in best.items() if b["val_auroc"] >= top - tol)]
    return chosen | {"layer_curve": {f"L{layers[l]}": round(b["val_auroc"], 4) for l, b in sorted(best.items())}}
