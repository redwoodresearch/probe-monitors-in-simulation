"""Q1(a) whitening, Amendment 9 (RULE.md; PR review of #1478): Amendment 3's inner-CV C choice redone with the transform refit
inside each inner fold (plain: the 2 inner-training folds' fake/fake rows; ZCA: their real honest rows), beside the as-run
outer-transform inner CV (gated on the committed inner losses), for all 10 training sets × plain and ZCA. The batched fit
and transforms are run_cequiv.py's (= run_zca_c.py's). One detached Modal H100 function. Nothing computes on the laptop.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 modal run --detach \\
      experiments/2026-10-13_whitening_POSTHOC/run_innerfold.py --scratch <dir with the result files> --out <results json>
"""

import hashlib
import json
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-whitening_innerfold_kill-2026-11-13"
DFV = "probe-shift_aswin-surya_sub-whysteps_work_kill-2026-11-13"
STAMP, KIND, LAYER = "2026-10-13_185308", "ffavg_output_end_mean", 48
SHA_FEATURES = "7b7b1d579312b3ccdf002a243dc8fdf7610eb3531969fd176f14dce124fc5ae2"
RET_JSON = ("zcac/zca_c_POSTHOC_2026-10-13_233835.json", "6fcaa7d776d5ab2f7708e71ce46ab91d2d29daf20f7074aeb4fba205ee4799ab")  # manifest zca_c v1
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
def run(runs: list, committed: dict, chosen: dict) -> dict:
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

    def inner_cv(f, v, method, inner_transform):
        te = (fold == f) | (fold == v)
        tr = np.flatnonzero(lab & ~te[r_run])
        yt = cuda(y[r_run[tr]]).float()
        if not inner_transform:  # as run: the outer model's transform for every inner fold
            Zt = transform(method, tr, np.flatnonzero(((arm == H) & ~te)[r_run]))(X[cuda(tr)])
        losses = []
        for j in [j for j in range(5) if j not in (f, v)]:
            a, va = row_fold[tr] != j, row_fold[tr] == j
            if inner_transform:  # refit on the 2 inner-training folds only (fold j excluded, as are f and v)
                tf = transform(method, tr[a], np.flatnonzero(((arm == H) & ~te & (fold != j))[r_run]))
                Za, Zv = tf(X[cuda(tr[a])]), tf(X[cuda(tr[va])])
            else:
                Za, Zv = Zt[cuda(a)], Zt[cuda(va)]
            W, b = fit_multi(Za, yt[cuda(a)], GRID)
            losses.append(F.binary_cross_entropy_with_logits(Zv @ W + b, yt[cuda(va)].reshape(-1, 1).expand(-1, len(GRID)), reduction="none").mean(0).double().cpu().numpy())
            del Za, Zv
        return np.mean(losses, 0)

    out = []
    for f, v in KEYS:
        for method in ("plain", "zca"):
            k = f"{method}|{f}{v}"
            lo, li = inner_cv(f, v, method, False), inner_cv(f, v, method, True)
            gate = float(np.abs(lo - np.array(committed[k])).max())
            if gate > 1e-6:
                raise RuntimeError(f"outer-transform inner CV for {k} differs from the committed losses by {gate}")
            if GRID[int(np.argmin(lo))] != chosen[k]:
                raise RuntimeError(f"outer-transform argmin for {k} differs from the committed chosen C")
            out.append({"key": f"{f}{v}", "method": method, "committed_C": chosen[k], "outer_mean_loss": lo.tolist(), "inner_mean_loss": li.tolist(),
                        "argmin_inner": GRID[int(np.argmin(li))], "max_abs_diff": float(np.abs(lo - li).max()), "gate_vs_committed": gate})
    return {"rows": out, "same_C_everywhere": all(r["argmin_inner"] == r["committed_C"] for r in out)}


@app.local_entrypoint()
def main(scratch: str = "", out: str = ""):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=REPO, check=True).stdout.strip()  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=all"):
        raise SystemExit("refusing to run from a tree with changes or untracked files")
    b = (Path(scratch) / RET_JSON[0]).read_bytes()
    if hashlib.sha256(b).hexdigest() != RET_JSON[1]:
        raise SystemExit("the retune result's sha256 differs from its manifest")
    sel = json.loads(b)["selection"]
    committed, chosen = {k: v["inner_loss"] for k, v in sel.items()}, {k: v["chosen_C"] for k, v in sel.items()}
    keep = ("arm", "fold")
    base = [{k: r[k] for k in keep} for r in json.loads((R36 / "run_scores_2026-10-06_181548/runs.json").read_text())]
    sh = [{k: r[k] for k in keep} for r in json.loads((R36 / "sim_honest_scores_2026-10-07_124717/runs.json").read_text()) if r["arm"] == SH]
    runs = base + sh
    if len(runs) != 4244:
        raise SystemExit("unexpected runs")
    r = run.remote(runs, committed, chosen)
    Path(out).open("x").write(json.dumps({"label": "POST HOC (RULE.md Amendment 9), preliminary until Codex agrees", "git": git("rev-parse", "HEAD"), "app": APP} | r, indent=1))
    lines = [f"same C everywhere (inner-fold transforms vs committed): {r['same_C_everywhere']}", "grid: " + " ".join(f"{c:.4g}" for c in GRID)]
    for x in r["rows"]:
        lines.append(f"{x['method']:5s} {x['key']}: committed C {x['committed_C']:.4g} | inner-fold argmin {x['argmin_inner']:.4g} | max |diff| {x['max_abs_diff']:.2e} | gate {x['gate_vs_committed']:.1e}")
        lines.append("   outer      " + " ".join(f"{v:.4f}" for v in x["outer_mean_loss"]))
        lines.append("   inner-fold " + " ".join(f"{v:.4f}" for v in x["inner_mean_loss"]))
    txt = "\n".join(lines)
    print(txt)
    Path(out[:-5] + ".printout.txt").open("x").write(txt + "\n")
