"""Logistic-regression probes on GPU, with scikit-learn's objective, fit many label sets at once.

`fit` minimizes, for each label column k independently, C * sum_i logloss(x_i . w_k + b_k, y_ik) + 0.5 * ||w_k||^2
(the intercept is not penalized), which is `sklearn.linear_model.LogisticRegression(C=C)`. The columns share the
feature matrix, so a label-shuffle null with 100 draws is one batched fit. torch is imported lazily.
"""


def standardize(X_train, *others):
    """Center and scale by the training rows' mean and standard deviation (as StandardScaler; zero std -> 1)."""
    mean = X_train.mean(0)
    std = X_train.std(0, unbiased=False)
    std = std.where(std > 0, std.new_ones(()))
    return [(X - mean) / std for X in (X_train, *others)]


def fit(X, Y, C: float, max_iter: int = 1000, tol: float = 1e-5, intercept: bool = True, tolerance_change: float = 1e-12):
    """X: [n, d] float tensor (standardized). Y: [n] or [n, k] {0, 1}. Returns (W [d, k], b [k], n_iter).

    intercept=False fixes b at 0 (`LogisticRegression(fit_intercept=False)`)."""
    import torch
    import torch.nn.functional as F

    Y = Y.reshape(len(Y), -1).to(X.dtype)
    n, d = X.shape
    W = torch.zeros(d, Y.shape[1], device=X.device, dtype=X.dtype, requires_grad=True)
    b = torch.zeros(Y.shape[1], device=X.device, dtype=X.dtype, requires_grad=intercept)
    opt = torch.optim.LBFGS([W, b] if intercept else [W], lr=1, max_iter=max_iter, tolerance_grad=tol, tolerance_change=tolerance_change,
                            history_size=20, line_search_fn="strong_wolfe")
    steps = [0]

    def closure():
        opt.zero_grad()
        z = X @ W + b
        # Divided by n for conditioning; the minimizer is unchanged.
        loss = (C * F.binary_cross_entropy_with_logits(z, Y, reduction="sum") + 0.5 * (W * W).sum()) / n
        loss.backward()
        steps[0] += 1
        return loss

    opt.step(closure)
    return W.detach(), b.detach(), steps[0]


def decision(X, W, b):
    """Scores [n, k] (the logit), as sklearn's decision_function."""
    return X @ W + b
