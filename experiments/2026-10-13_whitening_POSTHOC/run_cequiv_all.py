"""Q1(a) whitening, Amendment 8 (RULE.md; PR review of #1533): Amendment 7's check on the other 8 training sets. The worker
is run_cequiv.py's function unchanged except for its key list and a per-container shard written to a Volume; a CPU
orchestrator fans 2 H100 workers out (4 training sets each) and gathers them. Git is read at launch. Nothing computes on
the laptop.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 modal run --detach \\
      experiments/2026-10-13_whitening_POSTHOC/run_cequiv_all.py --scratch <dir with the result files> --out <results json>
"""

import hashlib
import json
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-whitening_cequivall_kill-2026-11-13"
OUTV = modal.Volume.from_name("probe-shift_aswin-surya_sub-whitening_cequivall_kill-2026-11-13", create_if_missing=True, version=2)
DFV = "probe-shift_aswin-surya_sub-whysteps_work_kill-2026-11-13"
STAMP, KIND, LAYER = "2026-10-13_185308", "ffavg_output_end_mean", 48
SHA_FEATURES = "7b7b1d579312b3ccdf002a243dc8fdf7610eb3531969fd176f14dce124fc5ae2"
RET_JSON = ("zcac/zca_c_POSTHOC_2026-10-13_233835.json", "6fcaa7d776d5ab2f7708e71ce46ab91d2d29daf20f7074aeb4fba205ee4799ab")  # manifest zca_c v1
R36 = REPO / "experiments/2026-10-06_q3_attack_detector/results"
H, SA, SH = "real-honest", "sim-attack", "sim-honest"
MAX_ITER = 1000
GRID = [10.0 ** e for e in (-4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1)]
REST = [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4)]  # the 8 training sets Amendment 7 skipped
GROUPS = [REST[:4], REST[4:]]
app = modal.App(APP)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy==2.3.3", "scikit-learn==1.7.2")
         .add_local_python_source("probe_shift"))


@app.function(image=image, gpu="H100!", cpu=(8, 8), memory=(65536, 65536), timeout=1800, retries=0,
              volumes={"/df": modal.Volume.from_name(DFV, version=2).read_only(), "/out": OUTV})
def work(runs: list, committed: dict, keys: list, tag: str) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F

    from probe_shift import torch_probe as tp

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

    out = []
    for f, v in [tuple(k) for k in keys]:
        te = (fold == f) | (fold == v)
        tr = np.flatnonzero(lab & ~te[r_run])
        rh = np.flatnonzero(((arm == H) & ~te)[r_run])
        yt = cuda(y[r_run[tr]]).float()
        for method in ("plain", "zca"):
            Zt = transform(method, tr, rh)(X[cuda(tr)])
            batched, separate, evals = [], [], []
            for j in [j for j in range(5) if j not in (f, v)]:
                a, va = cuda(row_fold[tr] != j), cuda(row_fold[tr] == j)
                W, b = fit_multi(Zt[a], yt[a], GRID)
                batched.append(F.binary_cross_entropy_with_logits(Zt[va] @ W + b, yt[va].reshape(-1, 1).expand(-1, len(GRID)), reduction="none").mean(0).double().cpu().numpy())
                row, ev = [], []
                for C in GRID:
                    Ws, bs, ne = tp.fit(Zt[a], yt[a], C, MAX_ITER)
                    row.append(float(F.binary_cross_entropy_with_logits((Zt[va] @ Ws + bs)[:, 0], yt[va]).double()))
                    ev.append(int(ne))
                separate.append(row)
                evals.append(ev)
            mb, ms = np.mean(batched, 0), np.mean(separate, 0)
            ref = np.array(committed[f"{method}|{f}{v}"], np.float64)
            gate = float(np.abs(mb - ref).max())
            if gate > 1e-6:
                raise RuntimeError(f"batched losses for {method}|{f}{v} differ from the committed ones by {gate}")
            out.append({"key": f"{f}{v}", "method": method, "batched_mean_loss": mb.tolist(), "separate_mean_loss": ms.tolist(),
                        "max_abs_diff": float(np.abs(mb - ms).max()), "argmin_batched": GRID[int(np.argmin(mb))], "argmin_separate": GRID[int(np.argmin(ms))],
                        "gate_vs_committed": gate, "separate_evals": evals})
            del Zt
    res = {"tag": tag, "keys": keys, "rows": out}
    shard = Path("/out") / tag / ("shard_" + "_".join(f"{a}{b}" for a, b in keys) + ".json")
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.open("x").write(json.dumps(res))
    OUTV.commit()
    return res


@app.function(image=image, cpu=(1, 1), memory=(2048, 2048), timeout=1800, retries=0)
def orchestrate(runs: list, committed: dict, git_sha: str, stamp: str) -> dict:
    tag = f"{stamp}_{git_sha[:9]}"
    shards = list(work.map([runs] * len(GROUPS), [committed] * len(GROUPS), [[list(k) for k in g] for g in GROUPS], [tag] * len(GROUPS)))
    rows = [r for s in shards for r in s["rows"]]
    if sorted(r["key"] for r in rows) != sorted(f"{a}{b}" for a, b in REST for _ in range(2)):
        raise RuntimeError("missing or duplicate training sets")
    return {"tag": tag, "rows": rows, "same_argmin_everywhere": all(r["argmin_batched"] == r["argmin_separate"] for r in rows)}


@app.local_entrypoint()
def main(scratch: str = "", out: str = ""):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=REPO, check=True).stdout.strip()  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=all"):
        raise SystemExit("refusing to run from a tree with changes or untracked files")
    b = (Path(scratch) / RET_JSON[0]).read_bytes()
    if hashlib.sha256(b).hexdigest() != RET_JSON[1]:
        raise SystemExit("the retune result's sha256 differs from its manifest")
    committed = {k: v["inner_loss"] for k, v in json.loads(b)["selection"].items()}
    keep = ("arm", "fold")
    base = [{k: r[k] for k in keep} for r in json.loads((R36 / "run_scores_2026-10-06_181548/runs.json").read_text())]
    sh = [{k: r[k] for k in keep} for r in json.loads((R36 / "sim_honest_scores_2026-10-07_124717/runs.json").read_text()) if r["arm"] == SH]
    runs = base + sh
    if len(runs) != 4244:
        raise SystemExit("unexpected runs")
    from datetime import datetime
    from zoneinfo import ZoneInfo

    r = orchestrate.remote(runs, committed, git("rev-parse", "HEAD"), datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S"))
    Path(out).open("x").write(json.dumps({"label": "POST HOC (RULE.md Amendment 8), preliminary until Codex agrees", "git": git("rev-parse", "HEAD"), "app": APP} | r, indent=1))
    lines = [f"same argmin everywhere: {r['same_argmin_everywhere']}", "grid: " + " ".join(f"{c:.4g}" for c in GRID)]
    for x in r["rows"]:
        lines.append(f"{x['method']:5s} {x['key']}: argmin batched {x['argmin_batched']:.4g} separate {x['argmin_separate']:.4g} | max |diff| {x['max_abs_diff']:.2e} | gate {x['gate_vs_committed']:.1e}")
        lines.append("   batched  " + " ".join(f"{v:.4f}" for v in x["batched_mean_loss"]))
        lines.append("   separate " + " ".join(f"{v:.4f}" for v in x["separate_mean_loss"]))
    txt = "\n".join(lines)
    print(txt)
    Path(out[:-5] + ".printout.txt").open("x").write(txt + "\n")
