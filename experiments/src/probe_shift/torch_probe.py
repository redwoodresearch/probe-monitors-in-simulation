"""Logistic-regression probes on GPU, with scikit-learn's objective, fit many label sets at once.

`fit` minimizes, for each label column k independently, the C-free scaling of scikit-learn's objective,
G = (sum_i logloss(x_i . w_k + b_k, y_ik) + ||w_k||^2 / (2C)) / n (the intercept is not penalized), whose minimizer is
`sklearn.linear_model.LogisticRegression(C=C)`'s. The columns share the feature matrix, so a label-shuffle null with 100
draws is one batched fit. torch is imported lazily.

Convergence is checked, not assumed (10-14): by default `fit` works in float64 with tolerance_grad 1e-8 and
tolerance_change 0, then certifies the result, max|dG| <= 1e-6 over w and b on the training rows, and raises
`UncertifiedFit` otherwise unless `allow_uncertified=True`. `certify` applies the same bar to any logistic fit (sklearn's
too). Results made before this change keep their as-run commits; the new default applies to new runs only, and
reproductions of earlier fits call `fit_as_run`, the pre-change optimizer kept verbatim.
"""

CERT_BAR = 1e-6


class UncertifiedFit(RuntimeError):
    pass


def standardize(X_train, *others):
    """Center and scale by the training rows' mean and standard deviation (as StandardScaler; zero std -> 1)."""
    mean = X_train.mean(0)
    std = X_train.std(0, unbiased=False)
    std = std.where(std > 0, std.new_ones(()))
    return [(X - mean) / std for X in (X_train, *others)]


def certify(
    X, Y, W, b, C: float, intercept: bool = True, chunk_rows: int | None = None
) -> dict:
    """max|dG| (float64) at (W, b) on rows X with labels Y, G as above. Accepts torch tensors or numpy arrays (sklearn:
    W = coef_.T, b = intercept_). Returns {n, max_grad, max_grad_per_column, certified}. The gradient is accumulated over
    row chunks, so certification never holds a full float64 copy of X (large GPU fits run with dtype=None)."""
    import torch

    def as64(a, device=None):
        t = torch.as_tensor(a)
        return t.to(device=device or t.device, dtype=torch.float64)

    Xt = torch.as_tensor(X)
    dev = Xt.device
    n = Xt.shape[0]
    W64 = as64(W, dev).reshape(Xt.shape[1], -1)
    b64 = as64(b, dev).reshape(-1)
    Yt = torch.as_tensor(Y).reshape(n, -1)
    if (
        chunk_rows is None
    ):  # about 256 MiB of float64 rows per chunk (d = 5,120 -> 6,553 rows)
        chunk_rows = max(1, (256 * 2**20) // (8 * max(1, Xt.shape[1])))
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    gW = torch.zeros_like(W64)
    gb = W64.new_zeros(W64.shape[1])
    for i in range(0, n, chunk_rows):
        Xc = as64(Xt[i : i + chunk_rows], dev)
        r = torch.sigmoid(Xc @ W64 + b64) - as64(Yt[i : i + chunk_rows], dev)
        gW += Xc.T @ r
        gb += r.sum(0)
        del Xc, r
    g = (
        torch.cat([(gW + W64 / C) / n] + ([gb[None, :] / n] if intercept else []), 0)
        .abs()
        .amax(0)
    )
    m = float(g.max())
    return {
        "n": n,
        "max_grad": m,
        "max_grad_per_column": g.cpu().tolist(),
        "certified": m <= CERT_BAR,
    }


def fit(
    X,
    Y,
    C: float,
    max_iter: int = 5000,
    tol: float = 1e-8,
    intercept: bool = True,
    tolerance_change: float = 0.0,
    dtype="float64",
    allow_uncertified: bool = False,
    return_certificate: bool = False,
):
    """X: [n, d] float tensor (standardized). Y: [n] or [n, k] {0, 1}.

    Returns (W [d, k], b [k], n_evals) in X's dtype, or (W, b, certificate) with return_certificate=True; the
    certificate is `certify`'s dict plus n_evals. Raises UncertifiedFit if any column's max|dG| > 1e-6, unless
    allow_uncertified. intercept=False fixes b at 0 (`LogisticRegression(fit_intercept=False)`).

    All columns share one L-BFGS run, so one column that does not certify (e.g. near-separable data at large C) makes the
    call raise; to read each column's pass or fail, pass allow_uncertified=True, return_certificate=True and use
    `max_grad_per_column`. Certified fits cost far more than the old default: on real data (n = 35,071, d = 5,120,
    C = 0.01, 16 CPU cores) 876 s and 5,776 evaluations against 7.6 s for the old default; float64 is slow on GPUs with little float64
    throughput (A10G, L4). Time one fit before scaling."""
    import torch
    import torch.nn.functional as F

    out_dtype = X.dtype
    if dtype is not None:
        X = X.to(getattr(torch, dtype))
    Y = Y.reshape(len(Y), -1).to(X.dtype)
    n, d = X.shape
    W = torch.zeros(d, Y.shape[1], device=X.device, dtype=X.dtype, requires_grad=True)
    b = torch.zeros(Y.shape[1], device=X.device, dtype=X.dtype, requires_grad=intercept)
    opt = torch.optim.LBFGS(
        [W, b] if intercept else [W],
        lr=1,
        max_iter=max_iter,
        tolerance_grad=tol,
        tolerance_change=tolerance_change,
        history_size=20,
        line_search_fn="strong_wolfe",
    )
    evals = [0]

    def closure():
        opt.zero_grad()
        z = X @ W + b
        loss = (
            F.binary_cross_entropy_with_logits(z, Y, reduction="sum")
            + (W * W).sum() / (2 * C)
        ) / n  # G
        loss.backward()
        evals[0] += 1
        return loss

    opt.step(closure)
    W, b = (
        W.detach().to(out_dtype),
        b.detach().to(out_dtype),
    )  # the certificate is of what is returned
    cert = certify(X, Y, W, b, C, intercept) | {"n_evals": evals[0]}
    if not cert["certified"] and not allow_uncertified:
        raise UncertifiedFit(
            f"max|dG| {cert['max_grad']:.3g} > {CERT_BAR} after {evals[0]} evaluations"
            f" (C={C}, returned as {out_dtype}; pass a float64 X to keep full precision)"
        )
    return (W, b, cert) if return_certificate else (W, b, evals[0])


def fit_as_run(
    X,
    Y,
    C: float,
    max_iter: int = 1000,
    tol: float = 1e-5,
    intercept: bool = True,
    tolerance_change: float = 1e-12,
):
    """Reproduces fits made before 2026-10-14; never for new results. The pre-change `fit`, verbatim: the L = C*loss +
    ||w||^2/2 scale (divided by n), the input dtype, tolerance_grad 1e-5, tolerance_change 1e-12, no certificate.

    X: [n, d] float tensor (standardized). Y: [n] or [n, k] {0, 1}. Returns (W [d, k], b [k], n_iter).

    intercept=False fixes b at 0 (`LogisticRegression(fit_intercept=False)`)."""
    import torch
    import torch.nn.functional as F

    Y = Y.reshape(len(Y), -1).to(X.dtype)
    n, d = X.shape
    W = torch.zeros(d, Y.shape[1], device=X.device, dtype=X.dtype, requires_grad=True)
    b = torch.zeros(Y.shape[1], device=X.device, dtype=X.dtype, requires_grad=intercept)
    opt = torch.optim.LBFGS(
        [W, b] if intercept else [W],
        lr=1,
        max_iter=max_iter,
        tolerance_grad=tol,
        tolerance_change=tolerance_change,
        history_size=20,
        line_search_fn="strong_wolfe",
    )
    steps = [0]

    def closure():
        opt.zero_grad()
        z = X @ W + b
        # Divided by n for conditioning; the minimizer is unchanged.
        loss = (
            C * F.binary_cross_entropy_with_logits(z, Y, reduction="sum")
            + 0.5 * (W * W).sum()
        ) / n
        loss.backward()
        steps[0] += 1
        return loss

    opt.step(closure)
    return W.detach(), b.detach(), steps[0]


def decision(X, W, b):
    """Scores [n, k] (the logit), as sklearn's decision_function."""
    return X @ W + b
