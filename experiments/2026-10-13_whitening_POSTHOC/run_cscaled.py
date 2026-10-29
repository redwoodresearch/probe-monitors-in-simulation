"""Q1(a) whitening, Amendment 11 (RULE.md; PR review): the inner-CV C choice with small C fit properly. Amendment 9's inner-fold
procedure for all 10 training sets x plain and ZCA: the 11 original values in one batched fit exactly as Amendment 9 (gated
on its losses), and every C <= 1e-3 plus 1e-5 and 10^-4.5 refit alone on the objective divided by C, (log loss + |w|^2/(2C))/n,
so L-BFGS's absolute gradient tolerance acts at a usable scale. Per fit: validation and training log loss, iterations, final
gradient. The job stops if any scaled fit's training loss is not below log 2 - 1e-3. No refit of any cell. One detached Modal
H100 function. Nothing computes on the laptop.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 modal run --detach \\
      experiments/2026-10-13_whitening_POSTHOC/run_cscaled.py --scratch <dir with the result files> --out <results json>
"""

import hashlib
import json
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-whitening_cscaled_kill-2026-11-13"
DFV = "probe-shift_aswin-surya_sub-whysteps_work_kill-2026-11-13"
STAMP, KIND, LAYER = "2026-10-13_185308", "ffavg_output_end_mean", 48
SHA_FEATURES = "7b7b1d579312b3ccdf002a243dc8fdf7610eb3531969fd176f14dce124fc5ae2"
INNER_JSON = ("innerfold/innerfold_POSTHOC_2026-10-14_053143.json", "d43eb5ccec1a18ff584e09642712df64bced31f9376f18b4351290a736d2ee5a")  # manifest innerfold v1
REFIT_JSON = ("refit/refit_innerc_POSTHOC_2026-10-14_054120.json", "a55a3ad0284706292cc08e24c1a2434f08213da1d8f01da244d66727b0acb53d")  # manifest refit_innerc v1
SMALL = [10.0 ** e for e in (-5, -4.5, -4, -3.5, -3)]  # refit alone on the scaled objective
LOG2 = 0.6931471805599453
R36 = REPO / "experiments/2026-10-06_q3_attack_detector/results"
H, SA, SH = "real-honest", "sim-attack", "sim-honest"
MAX_ITER = 1000
GRID = [10.0 ** e for e in (-4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1)]
PAIRS = [(f, v) for f in range(5) for v in range(5) if v != f]
KEYS = sorted({tuple(sorted(p)) for p in PAIRS})  # all 10 training sets, as run_zca_c.py
app = modal.App(APP)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy==2.3.3", "scikit-learn==1.7.2")
         .add_local_python_source("probe_shift"))


@app.function(image=image, gpu="H100!", cpu=(8, 8), memory=(65536, 65536), timeout=1800, retries=0,
              volumes={"/df": modal.Volume.from_name(DFV, version=2).read_only()})
def run(runs: list, a9_loss: dict, a10_C: dict) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F

    arm, fold = np.array([r["arm"] for r in runs]), np.array([r["fold"] for r in runs], int)
    raw = np.load(f"/df/packs/{STAMP}/{KIND}_L{LAYER}.npy")
    if hashlib.sha256(raw.tobytes()).hexdigest() != SHA_FEATURES:
        raise RuntimeError("features differ from df's record")
    r_run = np.load(f"/df/packs/{STAMP}/rows_{KIND}_L{LAYER}.npz")["run"].astype(np.int64)
    X = torch.from_numpy(raw).cuda().float()
    y = np.where(arm == SA, 1.0, np.where(arm == SH, 0.0, np.nan))
    lab, row_fold = (~np.isnan(y))[r_run], fold[r_run]
    cuda = lambda a: torch.from_numpy(a).cuda()  # noqa: E731

    def transform(method, rows_tr, rh):  # run_zca_c.py's transforms
        if method == "plain":
            Xt = X[cuda(rows_tr)]
            mean, std = Xt.mean(0), Xt.std(0, unbiased=False)
            std = std.where(std > 0, std.new_ones(()))
            return lambda Z: (Z - mean) / std
        Xc = X[cuda(rh)].double()
        m = Xc.mean(0)
        Xc -= m
        nr, p = Xc.shape
        Sm = Xc.T @ Xc / nr
        mu = torch.trace(Sm) / p
        delta_ = (Sm * Sm).sum()
        X2 = Xc * Xc
        beta = ((X2.T @ X2).sum() / nr - delta_) / (p * nr)
        delta = (delta_ - 2 * mu * torch.trace(Sm) + p * mu ** 2) / p
        shrink = 0.0 if float(beta) == 0 else float(torch.minimum(beta, delta) / delta)
        del X2, Xc
        lam, U = torch.linalg.eigh((1 - shrink) * Sm + shrink * mu * torch.eye(p, device=Sm.device, dtype=Sm.dtype))
        A, m = ((U * lam.rsqrt()) @ U.T).float(), m.float()
        return lambda Z: (Z - m) @ A

    def fit_multi(Z, yv, Cs):  # run_zca_c.py's batched fit, unchanged
        Cv = torch.tensor(Cs, device=Z.device, dtype=Z.dtype)
        Y = yv.reshape(-1, 1).expand(len(Z), len(Cs))
        W = torch.zeros(Z.shape[1], len(Cs), device=Z.device, dtype=Z.dtype, requires_grad=True)
        b = torch.zeros(len(Cs), device=Z.device, dtype=Z.dtype, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], lr=1, max_iter=MAX_ITER, tolerance_grad=1e-5, tolerance_change=1e-12, history_size=20, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = ((Cv * F.binary_cross_entropy_with_logits(Z @ W + b, Y, reduction="none").sum(0)).sum() + 0.5 * (W * W).sum()) / len(Z)
            loss.backward()
            return loss

        opt.step(closure)
        return W.detach(), b.detach()

    def fit_scaled(Z, yv, C):  # tp.fit's minimizer, on the objective divided by C: (log loss + |w|^2 / (2C)) / n
        W = torch.zeros(Z.shape[1], 1, device=Z.device, dtype=Z.dtype, requires_grad=True)
        b = torch.zeros(1, device=Z.device, dtype=Z.dtype, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], lr=1, max_iter=MAX_ITER, tolerance_grad=1e-5, tolerance_change=1e-12, history_size=20, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = (F.binary_cross_entropy_with_logits((Z @ W + b)[:, 0], yv, reduction="sum") + 0.5 / C * (W * W).sum()) / len(Z)
            loss.backward()
            return loss

        opt.step(closure)
        closure()
        g = float(torch.maximum(W.grad.abs().max(), b.grad.abs().max()))
        return W.detach(), b.detach(), int(opt.state[opt._params[0]]["n_iter"]), g

    out = []
    for f, v in KEYS:
        te = (fold == f) | (fold == v)
        tr = np.flatnonzero(lab & ~te[r_run])
        yt = cuda(y[r_run[tr]]).float()
        for method in ("plain", "zca"):
            k = f"{method}|{f}{v}"
            batched, val, trn, iters, grads = [], [], [], [], []
            for j in [j for j in range(5) if j not in (f, v)]:  # Amendment 9's inner-fold transforms
                a, va = row_fold[tr] != j, row_fold[tr] == j
                tf = transform(method, tr[a], np.flatnonzero(((arm == H) & ~te & (fold != j))[r_run]))
                Za, Zv = tf(X[cuda(tr[a])]), tf(X[cuda(tr[va])])
                ya, yv = yt[cuda(a)], yt[cuda(va)]
                W, b = fit_multi(Za, ya, GRID)  # as run (Amendment 9)
                batched.append(F.binary_cross_entropy_with_logits(Zv @ W + b, yv.reshape(-1, 1).expand(-1, len(GRID)), reduction="none").mean(0).double().cpu().numpy())
                rv, rt, ri, rg = [], [], [], []
                for C in SMALL:
                    Ws, bs, it, g = fit_scaled(Za, ya, C)
                    rv.append(float(F.binary_cross_entropy_with_logits((Zv @ Ws + bs)[:, 0], yv)))
                    rt.append(float(F.binary_cross_entropy_with_logits((Za @ Ws + bs)[:, 0], ya)))
                    ri.append(it)
                    rg.append(g)
                    if rt[-1] >= LOG2 - 1e-3:
                        raise RuntimeError(f"{k} fold {j} C {C:g}: training loss {rt[-1]:.6f} did not move below log 2 - 1e-3")
                val.append(rv)
                trn.append(rt)
                iters.append(ri)
                grads.append(rg)
                del Za, Zv
            mb, ms = np.mean(batched, 0), np.mean(val, 0)
            gate = float(np.abs(mb - np.array(a9_loss[k])).max())
            if gate > 1e-6:
                raise RuntimeError(f"{k}: the batched 11-value losses differ from Amendment 9's by {gate}")
            loss = {c: float(x) for c, x in zip(GRID, mb)} | {c: float(x) for c, x in zip(SMALL, ms)}  # scaled fits replace C <= 1e-3
            grid13 = sorted(loss)
            best = grid13[int(np.argmin([loss[c] for c in grid13]))]
            moved = {f"{c:.3g}": float(ms[i] - mb[GRID.index(c)]) for i, c in enumerate(SMALL) if c in GRID and c <= 10 ** -3.5 + 1e-12}
            out.append({"key": f"{f}{v}", "method": method, "a10_C": a10_C[k], "chosen_C": best, "edge_bound": best == SMALL[0],
                        "batched_mean_loss": mb.tolist(), "scaled_C": SMALL, "scaled_val_mean_loss": ms.tolist(), "scaled_train_loss": trn,
                        "scaled_iters": iters, "scaled_final_grad": grads, "scaled_minus_batched_small": moved, "gate_vs_a9": gate})
    rows = out
    flags = {"below_1e4_wins": [f"{r['method']}|{r['key']}" for r in rows if r["chosen_C"] < 1e-4 - 1e-12],
             "chosen_differs_from_a10": [f"{r['method']}|{r['key']}" for r in rows if r["chosen_C"] != r["a10_C"]],
             "small_C_moved_over_1e-3": [f"{r['method']}|{r['key']}" for r in rows if any(abs(d) > 1e-3 for d in r["scaled_minus_batched_small"].values())]}
    return {"rows": rows, "stop_and_report": flags, "any_stop": any(flags.values())}


@app.local_entrypoint()
def main(scratch: str = "", out: str = ""):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=REPO, check=True).stdout.strip()  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=all"):
        raise SystemExit("refusing to run from a tree with changes or untracked files")
    files = {}
    for k, (rel, h) in (("a9", INNER_JSON), ("a10", REFIT_JSON)):
        b = (Path(scratch) / rel).read_bytes()
        if hashlib.sha256(b).hexdigest() != h:
            raise SystemExit(f"{rel}: sha256 differs from its manifest")
        files[k] = json.loads(b)
    a9_loss = {f"{r['method']}|{r['key']}": r["inner_mean_loss"] for r in files["a9"]["rows"]}
    a10_C = {k: v["chosen_C"] for k, v in files["a10"]["selection"].items()}
    keep = ("arm", "fold")
    base = [{k: r[k] for k in keep} for r in json.loads((R36 / "run_scores_2026-10-06_181548/runs.json").read_text())]
    sh = [{k: r[k] for k in keep} for r in json.loads((R36 / "sim_honest_scores_2026-10-07_124717/runs.json").read_text()) if r["arm"] == SH]
    runs = base + sh
    if len(runs) != 4244:
        raise SystemExit("unexpected runs")
    sha_at_launch = git("rev-parse", "HEAD")
    r = run.remote(runs, a9_loss, a10_C)
    Path(out).open("x").write(json.dumps({"label": "POST HOC (RULE.md Amendment 11), preliminary until Codex agrees", "git": sha_at_launch, "app": APP} | r, indent=1))
    lines = [f"git at launch {sha_at_launch} | stop-and-report: {r['stop_and_report']}", "scaled C: " + " ".join(f"{c:.3g}" for c in SMALL)]
    for x in r["rows"]:
        it = [max(f[i] for f in x["scaled_iters"]) for i in range(len(SMALL))]
        lines.append(f"{x['method']:5s} {x['key']}: A10 C {x['a10_C']:.3g} -> chosen {x['chosen_C']:.3g}{' (edge)' if x['edge_bound'] else ''} | scaled-minus-batched {x['scaled_minus_batched_small']} | max iters {it} | gate {x['gate_vs_a9']:.1e}")
        lines.append("   scaled val " + " ".join(f"{v:.4f}" for v in x["scaled_val_mean_loss"]) + " | batched (11) " + " ".join(f"{v:.4f}" for v in x["batched_mean_loss"]))
    txt = "\n".join(lines)
    print(txt)
    Path(out[:-5] + ".printout.txt").open("x").write(txt + "\n")
