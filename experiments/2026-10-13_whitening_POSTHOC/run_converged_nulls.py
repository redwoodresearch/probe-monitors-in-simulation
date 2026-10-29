"""Q1(a) whitening, Amendment 12, job 2 (RULE.md; POST HOC): the retuned nulls redone with converged fits. Three shuffle
designs, one H100 worker each under a CPU orchestrator: within task (Amendment 4, seed 20261014), global (G, 20261017) and
within task without single-class tasks (W, 20261018), 20 shuffles each, for plain and ZCA retuned. Every fit minimizes the
scaled objective (log loss + |w|^2/(2C))/n with a fail-loudly training-loss gate. C is reselected per column on the scaled
grid 1e-5..1e-2 by the inner-fold procedure (Amendment 9); the real-label column is fit alone and, where its training set is
job 1's, gated against job 1's converged choice and scores. Each worker saves its own shard to a Volume. Point catches.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 modal run --detach \\
      experiments/2026-10-13_whitening_POSTHOC/run_converged_nulls.py --conv <job 1 result json> --out <results json>
"""

import hashlib
import json
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-whitening_convnulls_kill-2026-11-13"
OUTV = modal.Volume.from_name("probe-shift_aswin-surya_sub-whitening_convnulls_kill-2026-11-13", create_if_missing=True, version=2)
DFV = "probe-shift_aswin-surya_sub-whysteps_work_kill-2026-11-13"
STAMP, KIND, LAYER = "2026-10-13_185308", "ffavg_output_end_mean", 48
SHA_FEATURES = "7b7b1d579312b3ccdf002a243dc8fdf7610eb3531969fd176f14dce124fc5ae2"
R36 = REPO / "experiments/2026-10-06_q3_attack_detector/results"
H, E2, SA, SH = "real-honest", "real-attack-e2", "sim-attack", "sim-honest"
MAX_ITER, N_SHUF = 1000, 20
SEEDS = {"T": 20261014, "G": 20261017, "W": 20261018}
SCALED = [10.0 ** e for e in (-5, -4.5, -4, -3.5, -3, -2.5, -2)]
LOG2 = 0.6931471805599453
PAIRS = [(f, v) for f in range(5) for v in range(5) if v != f]
KEYS = sorted({tuple(sorted(p)) for p in PAIRS})
app = modal.App(APP)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy==2.3.3", "scikit-learn==1.7.2")
         .add_local_python_source("probe_shift"))


@app.function(image=image, gpu="H100!", cpu=(8, 8), memory=(65536, 65536), timeout=1800, retries=0,
              volumes={"/df": modal.Volume.from_name(DFV, version=2).read_only(), "/out": OUTV})
def work(runs: list, design: str, conv: dict, tag: str, smoke: bool = False) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F

    from probe_shift.attack_eval import pool, shuffle_within_task, threshold

    arm, task = np.array([r["arm"] for r in runs]), np.array([r["main_task"] for r in runs])
    fold, cell = np.array([r["fold"] for r in runs], int), np.array([r["cell"] for r in runs])
    n = len(arm)
    raw = np.load(f"/df/packs/{STAMP}/{KIND}_L{LAYER}.npy")
    if hashlib.sha256(raw.tobytes()).hexdigest() != SHA_FEATURES:
        raise RuntimeError("features differ from df's record")
    r_run = np.load(f"/df/packs/{STAMP}/rows_{KIND}_L{LAYER}.npz")["run"].astype(np.int64)
    X = torch.from_numpy(raw).cuda().float()
    y = np.where(arm == SA, 1.0, np.where(arm == SH, 0.0, np.nan))
    lab = ~np.isnan(y)
    rng = np.random.default_rng(SEEDS[design])
    ns = 1 if smoke else N_SHUF  # the smoke test: one shuffle column, one training set, ZCA only, same path
    keys, methods = (KEYS[:1], ("zca",)) if smoke else (KEYS, ("plain", "zca"))
    if design == "T":  # run_null.py's shuffles
        Y = np.column_stack([y] + [shuffle_within_task(y, task, lab, rng) for _ in range(ns)])
    elif design == "G":  # run_null_leak.py's global shuffles
        cols = [y]
        for _ in range(ns):
            o, ix = y.copy(), np.flatnonzero(lab)
            o[ix] = rng.permutation(y[ix])
            cols.append(o)
        Y = np.column_stack(cols)
    else:  # run_null_leak.py's (W): single-class tasks dropped in every column
        single = [t for t in np.unique(task[lab]) if len(np.unique(y[lab & (task == t)])) == 1]
        yW = np.where(np.isin(task, single), np.nan, y)
        Y = np.column_stack([yW] + [shuffle_within_task(yW, task, ~np.isnan(yW), rng) for _ in range(ns)])
    labc = ~np.isnan(Y[:, 0])
    if not (np.isnan(Y) == ~labc[:, None]).all() or any((Y[labc & (task == t)].sum(0) != Y[labc & (task == t), 0].sum()).any() for t in np.unique(task[labc])) and design != "G":
        raise RuntimeError("a shuffle changed the labelled runs or a task's label counts")
    if design == "G" and len(set(np.nansum(Y, 0))) != 1:
        raise RuntimeError("a global shuffle changed the label count")
    row_fold = fold[r_run]
    cuda = lambda a: torch.from_numpy(a).cuda()  # noqa: E731
    zc = {}

    def transform(method, rows_tr, rh, key):  # Amendment 2's transforms; ZCA cached by its real honest row set
        if method == "plain":
            Xt = X[cuda(rows_tr)]
            mean, std = Xt.mean(0), Xt.std(0, unbiased=False)
            std = std.where(std > 0, std.new_ones(()))
            return lambda Z: (Z - mean) / std
        if key not in zc:
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
            zc[key] = (m.float(), ((U * lam.rsqrt()) @ U.T).float())
        m, A = zc[key]
        return lambda Z: (Z - m) @ A

    def fit_scaled(Z, Yk, C):  # columns share C: (sum log loss + |W|^2/(2C))/n, separable by column; fails loudly
        Yk = Yk.reshape(len(Z), -1)
        W = torch.zeros(Z.shape[1], Yk.shape[1], device=Z.device, dtype=Z.dtype, requires_grad=True)
        b = torch.zeros(Yk.shape[1], device=Z.device, dtype=Z.dtype, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], lr=1, max_iter=MAX_ITER, tolerance_grad=1e-5, tolerance_change=1e-12, history_size=20, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = (F.binary_cross_entropy_with_logits(Z @ W + b, Yk, reduction="sum") + 0.5 / C * (W * W).sum()) / len(Z)
            loss.backward()
            return loss

        opt.step(closure)
        closure()
        g = torch.maximum(W.grad.abs().max(0).values, b.grad.abs()).double().cpu().numpy()  # final gradient, per column
        tl = F.binary_cross_entropy_with_logits(Z @ W + b, Yk, reduction="none").mean(0).detach().double().cpu().numpy()
        if not (np.isfinite(tl).all() and np.isfinite(g).all() and (tl < LOG2 - 1e-3).all()):
            raise RuntimeError(f"a fit at C {C:g} did not move: training losses {tl}")
        return W.detach(), b.detach(), {"iters": int(opt.state[opt._params[0]]["n_iter"]), "grad": [float(x) for x in g], "train_loss": [float(x) for x in tl]}

    nested = {m: np.full((1 + ns, 5, 5, n), np.nan, np.float32) for m in methods}
    chosen, iters, c_mismatch, fits = {}, [], [], []
    for f, v in keys:
        te = (fold == f) | (fold == v)
        tr, sc = np.flatnonzero((labc & ~te)[r_run]), np.flatnonzero(te[r_run])
        rh = np.flatnonzero(((arm == H) & ~te)[r_run])
        Yt = cuda(Y[r_run[tr]]).float()
        inner = [j for j in range(5) if j not in (f, v)]
        for method in methods:
            loss = np.zeros((len(inner), len(SCALED), 1 + ns))
            for ji, j in enumerate(inner):  # Amendment 9's inner-fold transforms
                a, va = row_fold[tr] != j, row_fold[tr] == j
                tf = transform(method, tr[a], np.flatnonzero(((arm == H) & ~te & (fold != j))[r_run]), (f, v, j))
                Za, Zv = tf(X[cuda(tr[a])]), tf(X[cuda(tr[va])])
                for ci_, C in enumerate(SCALED):
                    for cols in ([0], list(range(1, 1 + ns))):  # the real column alone, the shuffles as one batch
                        W, b, info = fit_scaled(Za, Yt[cuda(a)][:, cols], C)
                        iters.append(info["iters"])
                        fits.append({"key": f"{f}{v}", "method": method, "inner_fold": j, "C": C, "cols": "real" if cols == [0] else "shuffles"} | info)
                        loss[ji, ci_, cols] = F.binary_cross_entropy_with_logits(Zv @ W + b, Yt[cuda(va)][:, cols], reduction="none").mean(0).double().cpu().numpy()
                del Za, Zv
            pick = np.argmin(loss.mean(0), 0)  # per column; ascending grid, so ties go to the smaller C
            chosen[f"{method}|{f}{v}"] = [SCALED[i] for i in pick]
            if design != "W":  # the observed probe is job 1's: its C is used, and this job's own pick is reported if it differs
                c1 = conv["C"][f"{method}|{f}{v}"]
                if SCALED[pick[0]] != c1:
                    c_mismatch.append(f"{method}|{f}{v}: own pick {SCALED[pick[0]]:.3g}, job 1 {c1:.3g}")
                pick[0] = SCALED.index(c1)
            tf = transform(method, tr, rh, (f, v))
            Zt, Zs = tf(X[cuda(tr)]), tf(X[cuda(sc)])
            groups = [(pick[0], [0])] + [(i, list(1 + np.flatnonzero(pick[1:] == i))) for i in np.unique(pick[1:])]
            for i, cols in groups:
                W, b, info = fit_scaled(Zt, Yt[:, cols], SCALED[i])
                iters.append(info["iters"])
                fits.append({"key": f"{f}{v}", "method": method, "final": True, "C": SCALED[i], "cols": [int(c_) for c_ in cols]} | info)
                s = (Zs @ W + b).double().cpu().numpy()
                if not np.isfinite(s).all():
                    raise RuntimeError("non-finite scores")
                Sx = pool(r_run[sc], s, n, "mean")
                Sx[~te] = np.nan
                for a_, b_ in ((f, v), (v, f)):
                    nested[method][cols, a_, b_] = Sx.T
            del Zt, Zs
    gates = {"real_label_C_differs_from_job1": c_mismatch}
    if design != "W":
        for method in methods:
            ref = np.array(conv["nested"][f"{method}_conv"], np.float64)
            for f, v in keys:  # every set must reproduce job 1's converged scores
                for a_, b_ in ((f, v), (v, f)):
                    r_, m_ = ref[a_, b_], nested[method][0][a_, b_]
                    d = float(np.nanmax(np.abs(r_ - m_)))
                    if not (np.array_equal(np.isnan(r_), np.isnan(m_)) and d <= 1e-3):
                        raise RuntimeError(f"{method}|{f}{v}: real-label scores differ from job 1's converged ones by {d}")
                    gates[f"{method}_vs_job1_max"] = max(gates.get(f"{method}_vs_job1_max", 0.0), d)
    pairing = [a for a in np.unique(arm) if "attack" in a]

    def caught(nest, alpha):  # run_whitening.py's rule at the point estimate
        ok = np.ones(n, bool)
        for f, v in PAIRS:
            ok &= ~((fold == f) & ~np.isfinite(nest[f, v]))
        bad = set(cell[np.isin(arm, pairing) & ~ok])
        keep = np.where(np.isin(arm, pairing), ~np.isin(cell, list(bad)), ok)
        hits = np.zeros(n)
        for f, v in PAIRS:
            s = nest[f, v]
            T = threshold(s[(arm == H) & (fold == v) & np.isfinite(s)], alpha)
            hits[fold == f] += s[fold == f] > T
        return keep, hits / 4

    rate = lambda kh, a: float(kh[1][kh[0] & (arm == a)].mean())  # noqa: E731
    cells = {}
    for method in methods:
        cells[method] = {}
        for alpha in (0.01, 0.05):
            kh = [caught(nested[method][col], alpha) for col in range(1 + ns)] if not smoke else None
            if smoke:  # the smoke test checks the path, not the statistics: catches need all 10 training sets
                cells[method][f"fpr_{alpha:g}"] = {"smoke": True, "finite_scores_fold01": int(np.isfinite(nested[method][:, 0, 1]).sum())}
                continue
            e2, fk = [rate(x, E2) for x in kh], [rate(x, SA) for x in kh]
            if design != "W" and alpha == 0.01 and round(e2[0], 3) != round(conv["e2_1pct"][method], 3):
                raise RuntimeError(f"{method}: observed 1% e2 catch {e2[0]} differs from job 1's {conv['e2_1pct'][method]}")
            exc = {k: int(sum(x >= o[0] for x in o[1:])) for k, o in (("e2", e2), ("fake", fk))}
            cells[method][f"fpr_{alpha:g}"] = {"observed": {"e2": e2[0], "fake": fk[0]}, "null_e2": e2[1:], "null_fake": fk[1:],
                                               "null_median": {"e2": float(np.median(e2[1:])), "fake": float(np.median(fk[1:]))},
                                               "null_range": {"e2": [min(e2[1:]), max(e2[1:])], "fake": [min(fk[1:]), max(fk[1:])]},
                                               "exceedances": exc, "p": {k: (1 + x) / (1 + ns) for k, x in exc.items()}}
    res = {"design": design, "seed": SEEDS[design], "smoke": smoke, "gates": gates, "chosen_C": chosen, "iterations": [min(iters), max(iters)], "cells": cells, "fits": fits}
    shard = Path("/out") / tag / f"shard_{design}.json"
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.open("x").write(json.dumps(res))
    OUTV.commit()
    return res


@app.function(image=image, cpu=(1, 1), memory=(4096, 4096), timeout=1800, retries=0, volumes={"/out": OUTV})
def orchestrate(runs: list, conv: dict, git_sha: str, stamp: str, smoke: bool = False) -> dict:
    tag = f"{stamp}_{git_sha[:9]}" + ("_smoke" if smoke else "")
    designs = ["T"] if smoke else ["T", "G", "W"]
    out = list(work.map([runs] * len(designs), designs, [conv] * len(designs), [tag] * len(designs), [smoke] * len(designs)))
    if sorted(r["design"] for r in out) != sorted(designs):
        raise RuntimeError("missing or duplicate designs")
    res = {"tag": tag, "git": git_sha, "designs": {r["design"]: r for r in out}}
    OUTV.reload()
    comb = Path("/out") / tag / "combined.json"  # saved before returning, so a client failure loses nothing
    comb.parent.mkdir(parents=True, exist_ok=True)
    comb.open("x").write(json.dumps(res))
    OUTV.commit()
    return res


@app.local_entrypoint()
def main(conv: str = "", out: str = "", smoke: bool = False):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=REPO, check=True).stdout.strip()  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=all"):
        raise SystemExit("refusing to run from a tree with changes or untracked files")
    from datetime import datetime
    from zoneinfo import ZoneInfo

    j1 = json.loads(Path(conv).read_text())
    nested = json.loads(Path(conv[:-5] + ".nested_conv.json").read_text())
    c = {"C": {k: v["chosen_C"] for k, v in j1["selection"].items()}, "nested": {k: nested[k] for k in ("plain_conv", "zca_conv")},
         "e2_1pct": {m: j1["rates"]["fpr_0.01"][f"conv_{m}_conv"]["e2"]["value"] for m in ("plain", "zca")}}
    keep = ("arm", "key", "cell", "main_task", "fold")
    base = [{k: r[k] for k in keep} for r in json.loads((R36 / "run_scores_2026-10-06_181548/runs.json").read_text())]
    sh = [{k: r[k] for k in keep} for r in json.loads((R36 / "sim_honest_scores_2026-10-07_124717/runs.json").read_text()) if r["arm"] == SH]
    runs = base + sh
    if len(runs) != 4244:
        raise SystemExit("unexpected runs")
    sha_at_launch = git("rev-parse", "HEAD")
    r = orchestrate.remote(runs, c, sha_at_launch, datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S"), smoke)
    rec = {"label": "POST HOC (RULE.md Amendment 12, job 2), preliminary until Codex agrees", "git": sha_at_launch, "app": APP,
           "job1": {"file": Path(conv).name, "sha256": hashlib.sha256(Path(conv).read_bytes()).hexdigest()}} | r
    Path(out).open("x").write(json.dumps(rec, indent=1))
    lines = [f"git at launch {sha_at_launch} | smoke {smoke} | shards and combined.json under /out/{r['tag']}/ on the Volume"]
    for d in sorted(r["designs"]):
        x = r["designs"][d]
        lines.append(f"== design {d} (seed {x['seed']}): gates {x['gates']} | iterations {x['iterations']}")
        for m in x["cells"]:
            picks = [c_ for k, v in x["chosen_C"].items() if k.startswith(m + "|") for c_ in v[1:]]
            obs = sorted({v[0] for k, v in x["chosen_C"].items() if k.startswith(m + "|")})
            lines.append(f"  {m}: real-label C {obs} | shuffled C " + ", ".join(f"{g:.3g} x{picks.count(g)}" for g in SCALED if g in picks))
            for a, q in x["cells"][m].items():
                if q.get("smoke"):
                    lines.append(f"    {a}: smoke {q}")
                    continue
                lines.append(f"    {a}: real e2 observed {100 * q['observed']['e2']:.1f} | null median {100 * q['null_median']['e2']:.1f} "
                             f"(range {100 * q['null_range']['e2'][0]:.1f}-{100 * q['null_range']['e2'][1]:.1f}) | exceedances {q['exceedances']['e2']}/{N_SHUF}, p {q['p']['e2']:.3f} || "
                             f"fake observed {100 * q['observed']['fake']:.1f} | null median {100 * q['null_median']['fake']:.1f} | exceedances {q['exceedances']['fake']}/{N_SHUF}")
    txt = "\n".join(lines)
    print(txt)
    Path(out[:-5] + ".printout.txt").open("x").write(txt + "\n")
