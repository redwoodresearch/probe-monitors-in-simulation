"""H2 dev-environment robustness check (PLAN.md "Dev-environment robustness check (Aswin ~13:56 PT)"; approved by Aswin
just before 13:56 PT 10-11 via 0d): H2's runner (run_h2_search_modal.py at 267e438a) with only these changes: the
held-out outer folds are the 5 dev environments instead of the 5 test environments, the families are A0 and M2 only,
the comparison is fixed sim vs e1 only, the hard container timeout is 90 minutes (H2: 150; the in-code deadline is
checked only between fits), and there is no gate, pack or planted mode (H2's gate reproduced step A on the
test folds through the same real-comparison code path; the dev folds have no step A number to reproduce). Everything
else (labels and shuffle seeds, folds and validation tasks, step A's fitter and C grid, the MLP and PCA, seeds,
checkpoints, deadlines) is H2's code unchanged.
A robustness check, not an untouched confirmation: dev runs helped select M2 in H2. Keys and numbers only.

  `modal deploy` this file, then
  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy --with pyyaml \\
      modal run experiments/2026-10-11_claimB_h2_probe_best_found/run_h2_devcheck_modal.py --mode spawn
  (spawn and return; each container stops itself at 1.5x its family's expected time; watch_cost.py --stop 10)
"""

import hashlib
import json
import os
import time
from pathlib import Path

import modal

HERE = Path(__file__).parent
CLAIMB = Path.home() / "probe-shift-claimb"  # d9's worktree pinned at a1821be0 (step A's fitter, scorer, pack code)
APP = "probe-shift_aswin-surya_sub-claimb-h2dev_kill-2026-11-11"
VOL = "probe-shift_aswin-surya_sub-la-activations"
PACK_A, PACK_B, PACK_MAX = "claimB/level1/packed_39f842231a46", "claimB/level1/packedB_2cfd1aec393c", "claimB/level1/packedB_3adf85556e82"
CALLS, OUT = "claimB/level1/h2_calls_v1", "claimB/level1/h2_devcheck_v1"
LAYERS = [19, 32, 41, 44, 47, 52, 58, 63]
C_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
TEST_ENVS = ["auto_workflow", "file_indexing", "model_registry", "slack_clone", "web_scraping"]  # the DEV environments: held out in turn here
K, MAX_ITER, MAX_ITER_CAP = 101, 5000, 80000
HIDDEN, DROPOUT, LR, STEPS = 256, 0.1, 1e-3, 200
FAMILIES = {"A0": ["A0"], "M": ["M2"]}  # step A's pipeline and M2 frozen (one sub-configuration)
JOBS = [("fixed_vs_e1", f) for f in FAMILIES]
SCRIPT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
EXPECTED_MIN = {"A0": 50, "M": 10, "gate": 15}  # PRICE.md's rates (M2 alone: a fifth of the M family, rounded up); each job stops at 1.5x
STEP_A_AUROC, GATE_TOL = 0.8221, 0.005

app = modal.App(APP)
vol = modal.Volume.from_name(VOL, version=2)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy", "scikit-learn==1.7.2", "scipy", "pyyaml")
         .add_local_dir(CLAIMB / "src/probe_shift", "/root/probe_shift")
         .add_local_file(CLAIMB / "experiments/2026-10-10_claimB_probe/run_claimB_grid.py", "/root/run_claimB_grid.py")
         .add_local_file(CLAIMB / "experiments/2026-10-10_claimB_probe/run_claimB_pack.py", "/root/run_claimB_pack.py"))


def seed_of(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def setup(cfg, comparison):
    """Step A's runs, scorer, step table and folds; the 101 label columns with main-task (or planted-pair) shuffles."""
    import numpy as np

    import run_claimB_grid as G
    from probe_shift import q1_eval
    from probe_shift.q1_eval import Scorer, step_table

    q1_eval.DESIGNS["claimB"] = {"comparisons": G.COMPS}
    runs = json.loads((Path("/acts") / PACK_A / "runs.L63.json").read_text())
    sc = Scorer(cfg | {"design": "claimB"}, runs, comparison)
    unit = [runs[i]["cell"] if comparison.startswith("plant") else runs[i]["main_task"] for i in range(len(runs))]
    units = sorted({unit[i] for i in np.flatnonzero(sc.runs_in)})
    Y = np.zeros((len(runs), K), np.float32)
    Y[:, 0] = sc.y_run
    for k in range(1, K):
        rng = np.random.default_rng(seed_of("h2-shuffle", comparison, k))
        flip = dict(zip(units, rng.random(len(units)) < 0.5))
        Y[:, k] = [1 - v if (sc.runs_in[i] and flip[unit[i]]) else v for i, v in enumerate(sc.y_run)]
    env = np.array([cfg["environment_to_fold"][m["environment"]] for m in runs])
    return runs, sc, step_table(runs, "output_end"), Y, env


def features(name, runs, t):
    """Row matrix (float16, numpy) for a configuration's input, aligned to step A's output_end step table, and extra
    call_end rows (matrix, run index) when the configuration reads agent turns."""
    import numpy as np

    load = lambda p: np.load(Path("/acts") / p)  # noqa: E731  (whole files: row-indexed mmap reads are slow on Volumes)
    calls = None
    if name in ("L3", "P3"):
        meta = json.loads((Path("/acts") / CALLS / "rows.json").read_text())
        key_ix = {(m["arm"], m["key"]): i for i, m in enumerate(runs)}
        calls = (load(f"{CALLS}/L63_call_end.npy"), np.array([key_ix[(a, k)] for a, k in meta["run"]]))
    if name in ("A0",):
        return {l: load(f"{PACK_A}/L{l}_output_end.npy") for l in LAYERS}, calls
    if name in ("L1", "M5"):
        return np.concatenate([load(f"{PACK_A}/L{l}_output_end.npy") for l in LAYERS], 1), calls
    if name == "L2":
        def aligned(pack, files):
            pr = json.loads((Path("/acts") / pack / "runs.L63.json").read_text())
            from probe_shift.q1_eval import step_table
            pt = step_table(pr, "output_end")
            pos = {(pr[r]["arm"], pr[r]["key"], s, a): k for k, (r, s, a) in enumerate(zip(pt["run"], pt["step"], pt["action"]))}
            ix = np.array([pos[(runs[r]["arm"], runs[r]["key"], s, a)] for r, s, a in zip(t["run"], t["step"], t["action"])])
            return [load(f"{pack}/{f}")[ix] for f in files]
        parts = [load(f"{PACK_A}/L63_output_end.npy")]
        parts += aligned(PACK_B, ["L63_output_end_mean.npy", "L63_output_end_ema0.8.npy", "L63_output_end_ema0.95.npy"])
        parts += aligned(PACK_MAX, ["L63_output_end_max.npy"])
        return np.concatenate(parts, 1), calls
    layer = 58 if name in ("M1", "M3") else 63
    return load(f"{PACK_A}/L{layer}_output_end.npy"), calls


def subconfigs(name):
    if name == "A0":
        return [{"layer": l, "C": C} for l in LAYERS for C in C_GRID]
    if name in ("L1", "L2", "L3"):
        return [{"C": C} for C in C_GRID]
    if name in ("M1", "M2", "M3", "M4"):
        return [{"wd": 1e-4 if name in ("M1", "M2") else 1e-2}]
    if name in ("M5", "P1", "P2"):
        return [{"wd": w} for w in ((1e-4, 1e-2) if name == "M5" else ((1e-4,) if name == "P1" else (1e-2,)))]
    return [{"wd": 1e-4}]  # P3 = P1 with call_end rows


def code_sha() -> str:
    """This script plus the d9 code it runs: local paths here, the image's copies in the container."""
    if modal.is_local():
        d9 = [CLAIMB / "src/probe_shift/torch_probe.py", CLAIMB / "src/probe_shift/q1_eval.py",
              CLAIMB / "experiments/2026-10-10_claimB_probe/run_claimB_grid.py", CLAIMB / "experiments/2026-10-10_claimB_probe/run_claimB_pack.py"]
    else:
        d9 = [Path("/root/probe_shift/torch_probe.py"), Path("/root/probe_shift/q1_eval.py"), Path("/root/run_claimB_grid.py"), Path("/root/run_claimB_pack.py")]
    return hashlib.sha256(b"".join(f.read_bytes() for f in [Path(__file__)] + d9)).hexdigest()


def ident_of(cfg, code) -> str:
    """Code + folds + inputs: what a cached file or a passing gate must match."""
    return hashlib.sha256(json.dumps({k: cfg[k] for k in ("packs", "main_task_to_fold", "environment_to_fold")}
                                     | {"code": code, "inputs": [PACK_A, PACK_B, PACK_MAX, CALLS]}, sort_keys=True).encode()).hexdigest()


def preflight(runs, t, names):
    """Before any fit: every input a configuration needs covers every step-A output row (L2's B and max packs)."""
    import numpy as np

    from probe_shift.q1_eval import step_table

    if "L2" in names:
        want = {(runs[r]["arm"], runs[r]["key"], s, a) for r, s, a in zip(t["run"], t["step"], t["action"])}
        for pack in (PACK_B, PACK_MAX):
            pr = json.loads((Path("/acts") / pack / "runs.L63.json").read_text())
            pt = step_table(pr, "output_end")
            have = {(pr[r]["arm"], pr[r]["key"], s, a) for r, s, a in zip(pt["run"], pt["step"], pt["action"])}
            if want - have:
                raise ValueError(f"{pack} lacks {len(want - have)} of step A's output rows: L2 can't run as frozen")
    if {"L3", "P3"} & set(names) and not (Path("/acts") / CALLS / "rows.json").exists():
        raise ValueError("call_end rows are not packed yet")
    return np.True_


@app.function(image=image, gpu="H100!", cpu=8, memory=131072, volumes={"/acts": vol}, timeout=90 * 60, retries=0)  # hard stop (H2: 150)
def search(cfg: dict, comparison: str, family: str, k_cols: int = K) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F

    import run_claimB_grid as G
    from probe_shift import torch_probe as tp

    t0 = time.time()
    code = code_sha()
    if code != cfg["code_sha"]:
        raise ValueError("the code in the container differs from the code committed locally")
    ident = ident_of(cfg, code)
    runs, sc, t, Y, env = setup(cfg, comparison)
    Y = Y[:, :k_cols]
    names = sum(FAMILIES.values(), []) if family == "all" else FAMILIES.get(family, ["A0"])
    preflight(runs, t, names)
    deadline = t0 + 1.5 * EXPECTED_MIN["gate" if (k_cols == 1 and comparison == "fixed_vs_e1" and family == "A0") else family] * 60
    use, efold, tfold = sc.runs_in[t["run"]], env[t["run"]], sc.fold[t["run"]]
    folds = sorted({cfg["environment_to_fold"][e] for e in TEST_ENVS})
    is_gate = k_cols == 1 and comparison == "fixed_vs_e1" and family == "A0"
    out_dir = Path("/acts") / OUT / ("gate" if is_gate else comparison if k_cols == K else f"{comparison}_k{k_cols}")
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    KK = Y.shape[1]
    log = lambda m: print(f"[{comparison}|{family}] {time.time() - t0:7.0f}s {m}", flush=True)  # noqa: E731

    def run_means(row_runs, S):
        k, inv = np.unique(row_runs, return_inverse=True)
        tot = torch.zeros(len(k), S.shape[1], dtype=torch.float64, device=dev).index_add_(0, torch.from_numpy(inv).to(dev), S.double())
        return k, (tot / torch.from_numpy(np.bincount(inv)).to(dev)[:, None]).cpu().numpy()

    cap = MAX_ITER if comparison.startswith("plant") else MAX_ITER_CAP  # planted: no refit chain (12:40 PT)

    def linear(Xa, Xb, Ya, C):
        Xa, Xb = tp.standardize(Xa.double(), Xb.double())
        Xa, Xb = Xa * C**0.5, Xb * C**0.5
        mi, prev = MAX_ITER, None
        while True:  # step A's convergence loop
            W, b, evals = tp.fit(Xa, Ya, 1.0, mi, tol=G.tol_of(C))
            Wg, bg = W.clone().requires_grad_(True), b.clone().requires_grad_(True)
            loss = (F.binary_cross_entropy_with_logits(Xa @ Wg + bg, Ya, reduction="sum") + 0.5 * (Wg * Wg).sum()) / len(Xa)
            loss.backward()
            g = float(torch.maximum(Wg.grad.abs().amax(0), bg.grad.abs()).max())
            if g <= G.tol_of(C) or evals < mi or evals == prev or mi * 4 > cap:
                return tp.decision(Xb, W, b), g <= G.tol_of(C)
            mi, prev = mi * 4, evals

    def mlp(Za, Zb, Ya, wd, tag):
        """101 independent one-hidden-layer MLPs; returns logits [rows_b, K]."""
        d = Za.shape[1]
        W1 = torch.stack([torch.randn(d, HIDDEN, generator=torch.Generator().manual_seed(seed_of("h2-init", tag, k))) / d**0.5 for k in range(KK)]).to(dev)
        ps = [W1.requires_grad_(True), torch.zeros(KK, 1, HIDDEN, device=dev, requires_grad=True), torch.zeros(KK, HIDDEN, 1, device=dev, requires_grad=True),
              torch.zeros(KK, 1, device=dev, requires_grad=True)]
        opt = torch.optim.AdamW(ps, lr=LR, weight_decay=wd)
        torch.manual_seed(seed_of("h2-dropout", tag))
        for _ in range(STEPS):
            H = F.dropout(torch.relu(torch.einsum("nd,cdh->cnh", Za, ps[0]) + ps[1]), DROPOUT, training=True)
            loss = F.binary_cross_entropy_with_logits((H @ ps[2]).squeeze(-1) + ps[3], Ya.T, reduction="none").mean(1).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            return ((torch.relu(torch.einsum("nd,cdh->cnh", Zb, ps[0]) + ps[1]) @ ps[2]).squeeze(-1) + ps[3]).T  # [rows_b, K]

    def pool(Za, ra, Zb, rb, Yruns_a, wd, tag):
        """101 independent attention-pooled run classifiers: per run, softmax(v.z) over its rows, then u.pooled + c."""
        ka, ia = np.unique(ra, return_inverse=True)
        kb, ib = np.unique(rb, return_inverse=True)
        ia, ib = torch.from_numpy(ia).to(dev), torch.from_numpy(ib).to(dev)
        init = lambda part: torch.stack([torch.randn(Za.shape[1], generator=torch.Generator().manual_seed(seed_of("h2-init", tag, part, k))) * 0.01  # noqa: E731
                                         for k in range(KK)], 1)  # each column its own seed
        ps = [init("v").to(dev).requires_grad_(True), init("u").to(dev).requires_grad_(True), torch.zeros(KK, device=dev, requires_grad=True)]

        def forward(Z, inv, n):
            a = Z @ ps[0]  # [rows, K]
            m = torch.full((n, KK), -1e30, device=dev).index_reduce_(0, inv, a, "amax", include_self=True)
            e = torch.exp(a - m[inv])
            w = e / torch.zeros(n, KK, device=dev).index_add_(0, inv, e)[inv]
            pooled = torch.zeros(n, KK, Z.shape[1], device=dev).index_add_(0, inv, w[:, :, None] * Z[:, None, :])  # [n, K, d]
            return torch.einsum("nkd,dk->nk", pooled, ps[1]) + ps[2]

        Yr = torch.from_numpy(Yruns_a[ka]).to(dev)
        opt = torch.optim.AdamW(ps, lr=LR, weight_decay=wd)
        for _ in range(STEPS):
            loss = F.binary_cross_entropy_with_logits(forward(Za, ia, len(ka)), Yr, reduction="none").mean(0).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            return kb, forward(Zb, ib, len(kb)).double().cpu().numpy()

    def pca(Xa, Xb, q, tag):
        Xa, Xb = tp.standardize(Xa.float(), Xb.float())
        torch.manual_seed(seed_of("h2-pca", tag))  # pca_lowrank's random projection, seeded per comparison, input, fold and fit
        _, _, V = torch.pca_lowrank(Xa, q=q, center=False, niter=4)
        Za, Zb = Xa @ V, Xb @ V
        s = Za.std(0) + 1e-6
        return Za / s, Zb / s

    def valid(f, subs):
        z = np.load(f)
        return str(z["ident"]) == ident and json.loads(str(z["subs"])) == subs and z["val_auroc"].shape[-1] == KK

    done = {}
    for name in names:
        dst = out_dir / f"{name}.npz"
        subs = subconfigs(name)
        if dst.exists():
            if not valid(dst, subs):
                raise ValueError(f"{dst} was made by other code or settings; use a new output folder")
            done[name] = "exists"
            continue
        X, calls = features(name, runs, t)
        r = {"val_auroc": np.full((len(subs), len(folds), KK), np.nan), "test_run_score": np.full((len(subs), len(runs), KK), np.nan),
             "converged": np.ones((len(subs), len(folds), 2), bool)}
        rows_run, rows_use, rows_ef, rows_tf = t["run"], use, efold, tfold
        if calls is not None:  # agent turns: call_end rows join the output rows, labelled by their run
            cX, crun = calls
            keep_c = sc.runs_in[crun]
            rows_run = np.concatenate([t["run"], crun])
            rows_use = np.concatenate([use, keep_c])
            rows_ef, rows_tf = np.concatenate([efold, env[crun]]), np.concatenate([tfold, sc.fold[crun]])
        for i, f in enumerate(folds):
            ck = out_dir / f"{name}.fold{f}.npz"
            if ck.exists():  # a completed fold from an earlier attempt, same code and settings
                if not valid(ck, subs):
                    raise ValueError(f"{ck} was made by other code or settings")
                z = np.load(ck)
                r["val_auroc"][:, i], r["converged"][:, i] = z["val_auroc"][:, i], z["converged"][:, i]
                fold_runs = np.flatnonzero(np.isin(env, [f]))
                r["test_run_score"][:, fold_runs] = z["test_run_score"][:, fold_runs]
                log(f"{name} fold {f} resumed")
                continue
            if time.time() > deadline:
                raise TimeoutError(f"{comparison} {family}: past 1.5x the expected time")
            tr, te = rows_use & (rows_ef != f), rows_use & (rows_ef == f)
            va = tr & (rows_tf == f % 5)
            for j, (part, a, b_) in enumerate((("full", tr, te), ("val", tr & ~va, va))):
                ia_, ib_ = np.flatnonzero(a), np.flatnonzero(b_)
                def rowsX(ix, layer=None):
                    src = X[layer] if layer is not None else X
                    if calls is None:
                        return torch.from_numpy(np.asarray(src[ix], np.float32)).to(dev)
                    n_o = len(t["run"])
                    return torch.from_numpy(np.concatenate([np.asarray(src[ix[ix < n_o]], np.float32), np.asarray(cX[ix[ix >= n_o] - n_o], np.float32)])).to(dev)
                if calls is not None:  # keep row order consistent with rowsX (output rows first, then call rows)
                    n_o = len(t["run"])
                    ia_, ib_ = np.concatenate([ia_[ia_ < n_o], ia_[ia_ >= n_o]]), np.concatenate([ib_[ib_ < n_o], ib_[ib_ >= n_o]])
                ra, rb = rows_run[ia_], rows_run[ib_]
                Ya = torch.from_numpy(Y[ra]).to(dev).double()
                ptag = f"{comparison}|{name}|{f}|{part}"
                if name in ("L1", "M5") or name.startswith("M"):
                    Za, Zb = pca(rowsX(ia_), rowsX(ib_), 512, ptag)
                elif name.startswith("P"):
                    Za, Zb = pca(rowsX(ia_), rowsX(ib_), 256, ptag)
                for s, sub in enumerate(subs):
                    if time.time() > deadline:
                        raise TimeoutError(f"{comparison} {family}: past 1.5x the expected time, inside {name} fold {f}")
                    tag = f"{comparison}|{name}|{s}|{f}|{part}"
                    if s and s % 24 == 0:
                        log(f"{name} fold {f} {part}: {s} of {len(subs)} sub-configurations")
                    if name == "A0":
                        S, ok = linear(rowsX(ia_, sub["layer"]), rowsX(ib_, sub["layer"]), Ya, sub["C"])
                        k, M = run_means(rb, S)
                    elif name in ("L1",):
                        S, ok = linear(Za, Zb, Ya, sub["C"])
                        k, M = run_means(rb, S)
                    elif name in ("L2", "L3"):
                        S, ok = linear(rowsX(ia_), rowsX(ib_), Ya, sub["C"])
                        k, M = run_means(rb, S)
                    elif name.startswith("M"):
                        S, ok = mlp(Za, Zb, Ya.float(), sub["wd"], tag), True
                        k, M = run_means(rb, S)
                    else:
                        (k, M), ok = pool(Za, ra, Zb, rb, Y, sub["wd"], tag), True
                    r["converged"][s, i, j] = ok
                    if part == "full":
                        r["test_run_score"][s, k] = M
                    else:
                        r["val_auroc"][s, i] = G.col_auc(Y[k], M)
                torch.cuda.empty_cache()
            with open(ck.with_suffix(".tmp"), "wb") as fh:  # per-fold checkpoint
                np.savez(fh, subs=json.dumps(subs), ident=ident, **r)
            os.replace(ck.with_suffix(".tmp"), ck)
            vol.commit()
            log(f"{name} fold {f} done ({len(subs)} sub-configurations)")
        tested = sc.runs_in & np.isin(env, folds)
        if not np.isfinite(r["test_run_score"][:, tested]).all():
            raise ValueError(f"{comparison} {name}: a compared test run got no score")
        with open(dst.with_suffix(".tmp"), "wb") as fh:
            np.savez(fh, folds=np.array(folds), subs=json.dumps(subs), ident=ident, **r)
        os.replace(dst.with_suffix(".tmp"), dst)
        vol.commit()
        done[name] = round(time.time() - t0)
        log(f"{name} saved")
    if is_gate:  # the gate, from the saved A0 file (fresh or cached): step A's selection, best layer and C per fold
        z = np.load(out_dir / "A0.npz")
        r, subs = {k: z[k] for k in ("val_auroc", "test_run_score")}, json.loads(str(z["subs"]))
        if True:
            best = r["val_auroc"][:, :, 0].argmax(0)
            S = np.full(len(runs), np.nan)
            for i, f in enumerate(folds):
                fr = np.flatnonzero(env == f)
                S[fr] = r["test_run_score"][best[i], fr, 0]
            from sklearn.metrics import roc_auc_score
            m = sc.runs_in & np.isin(env, folds)
            auc = float(roc_auc_score(Y[m, 0], S[m]))
            done["gate"] = {"auroc": auc, "step_a": STEP_A_AUROC, "pass": abs(auc - STEP_A_AUROC) <= GATE_TOL,
                            "picks": [subs[b] for b in best], "ident": ident}
            (out_dir / "gate.json").write_text(json.dumps(done["gate"]))
            vol.commit()
    return {"comparison": comparison, "family": family, "done": done, "seconds": round(time.time() - t0)}


@app.local_entrypoint()
def main(mode: str = "spawn"):
    import subprocess
    import sys

    if subprocess.run(["git", "status", "--porcelain", str(Path(__file__))], capture_output=True, text=True, check=True).stdout:
        raise SystemExit(f"commit {Path(__file__).name} first")
    folds = json.loads((HERE.parents[1] / "splits/folds_v2.json").read_text())
    cfg = {"packs": {"level1": "39f842231a46"}, "seed": 0, "bootstrap": 1000, "tag": "h2", "code_sha": code_sha(),
           "main_task_to_fold": folds["task_folds"]["main_task_to_fold"], "environment_to_fold": folds["environment_folds"]["environment_to_fold"]}
    if mode == "spawn":
        f = modal.Function.from_name(APP, "search")  # no gate here (see the docstring)
        for c, fam in JOBS:
            call = f.spawn(cfg, c, fam)
            print(c, fam, call.object_id, flush=True)
    else:
        sys.exit(f"unknown mode {mode}")
