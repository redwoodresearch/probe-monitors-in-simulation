"""Item (A), probes after all four fixes with an environment-half split (PLAN.md in this folder; Aswin, relayed by 0d
before 15:30 PT 10-11). Train step A's pipeline on one half of the environments and score the other half, both
directions (dev -> test, test -> dev), on the four-fix (evened) runs everywhere. Saves score arrays only (numbers and
keys); the summary is analyze_halves.py.

- Rows: H3's assembly. Changed runs take their evened reads (test: H3's extraction `claimB/h3_small_v1/04c3953b139e`,
  dev: B-dev's `claimB/slips_evened_dev_v1`), with "Invalid view_range" outputs left out of the run mean; unchanged runs
  keep step B's level-1 pack rows. Plant 2 (R2 undone): test's evened plant-2 copies and B-dev's 100 evened dev
  planted reads. Plant 1 has no dev copies, so it can't be checked across halves.
- Probe: step A's pipeline, linear on output_end at L58 and L63 x 9 C, chosen per label column on validation main tasks
  (task fold 0) inside the training half; run score = the mean over the run's kept outputs.
- Columns: fixed sim vs e1 and e2 vs e1 get the real labels plus 100 refit shuffles, one coin per main task, the same
  coins for both comparisons; plant 2 gets the real labels plus 20 refit per-pair shuffles, linear fits capped at 5,000
  iterations (H2's and H3's cap).
- Nothing H3's refit or summary wrote is read.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy \\
      modal run experiments/2026-10-11_claimB_halves/run_halves_modal.py --mode refit
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).parent
H3 = HERE.parent / "2026-10-11_claimB_h3_fix_to_plateau"
sys.path[:0] = [str(H3), str(HERE.parent / "2026-10-11_claimB_slips_evened_probe")] if modal.is_local() else ["/root"]
import run_h3_small_modal as R  # noqa: E402  (H3's runner at 23b5c37a: B-dev's code, constants, the evened test reads)

H3_RUNNER_SHA12 = "04c3953b139e"  # run_h3_small_modal.py at 23b5c37a: its extraction wrote the evened test reads
B, M = R.B, R.M
SIM, E1, E2, PL2 = M.SIM, M.REAL, "real-attack-e2", "sim-attack-fixed-plant2"
TEST = list(R.TEST)
DEV = ["auto_workflow", "file_indexing", "model_registry", "slack_clone", "web_scraping"]
DIRS = {"dev_to_test": (DEV, TEST), "test_to_dev": (TEST, DEV)}
COMPS = {"fixed_vs_e1": (E1, SIM), "e2_vs_e1": (E1, E2), "plant2": (SIM, PL2)}  # (negative arm, positive arm)
K, KP, VAL_FOLD = 101, 21, 0
SCRIPT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
OUT = f"claimB/halves_v1/{SCRIPT_SHA[:12]}"

app = modal.App("probe-shift_aswin-surya_sub-claimb-halves_kill-2026-11-11")
image = R.image.add_local_file(H3 / "run_h3_small_modal.py", "/root/run_h3_small_modal.py")


@app.function(image=image, gpu="H100!", cpu=8, memory=131072, volumes={"/acts": B.P.vol}, timeout=120 * 60, retries=0)
def refit(cfg: dict, model: str, revision: str) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scipy.stats import rankdata

    from probe_shift import torch_probe as tp
    from probe_shift.q1_eval import step_table

    R.check_b()
    if R.SCRIPT_SHA[:12] != H3_RUNNER_SHA12:
        raise ValueError("H3's runner differs from the one whose extraction wrote the evened test reads")
    B.P.vol.reload()
    t0 = time.time()
    log = lambda m: print(f"[halves] {time.time() - t0:6.0f}s {m}", flush=True)  # noqa: E731
    for base in (R.OUT, R.DEV_READS):  # both reads' gates passed (4 runs bitwise at L58 and L63)
        g = json.loads((Path("/acts") / base / "gate.json").read_text())
        if not g.get("passed"):
            raise ValueError(f"{base}: gate not passed")
    pruns = json.loads((Path("/acts") / M.PACK / "runs.L63.json").read_text())
    t = step_table(pruns, "output_end")
    XP = {l: np.load(Path("/acts") / M.PACK / f"L{l}_output_end.npy") for l in R.LAYERS}
    rows_of = {}
    for k in range(len(t["run"])):
        rows_of.setdefault(int(t["run"][k]), {})[(int(t["step"][k]), int(t["action"][k]))] = k

    def files(base, version):
        out = {}
        for f in (Path("/acts") / base / "runs").glob("*.npz"):
            z = np.load(f)
            m = json.loads(str(z["meta"]))
            if m["version"] != version:
                continue
            if (m["script_sha256"], m.get("deps_sha256"), m["model"], m["revision"]) != (B.SCRIPT_SHA, B.DEPS_SHA, model, revision):
                raise ValueError(f"{f.name} was made by another version or model")
            out[(m["cell"], m["arm"])] = (m, z)
        return out

    test_reads, dev_reads, dev_planted = files(R.OUT, "evened"), files(R.DEV_READS, "evened"), files(R.DEV_READS, "planted")
    h3_plan = json.loads((Path("/acts") / R.OUT / "cells.json").read_text())
    dev_plan = json.loads((Path("/acts") / R.DEV_READS / "cells.json").read_text())
    if h3_plan["script_sha256"] != R.SCRIPT_SHA:
        raise ValueError("H3's cells.json is from another runner version")
    want_t = {tuple(x) for x in h3_plan["changed"]}
    want_d = {tuple(x) for x in dev_plan["changed"]}
    if not want_t <= set(test_reads) or not want_d <= set(dev_reads) or set(dev_plan["planted"]) - {c for c, _ in dev_planted}:
        raise ValueError("evened or planted reads are missing")

    def read_rows(meta, z):
        rng = {tuple(x) for x in meta["range"]}
        keep = [k for k, s in enumerate(meta["steps"]) if tuple(s) not in rng]
        return {l: np.asarray(z[f"L{l}"][keep], np.float16) for l in R.LAYERS}, len(keep)

    runs, rr, Xl = [], [], {l: [] for l in R.LAYERS}
    env_of = {m["cell"]: m["environment"] for m in pruns}

    def add(entry, rows, n):
        if n == 0:
            raise ValueError(f"{entry['cell']} {entry['arm']}: no rows")
        for l in R.LAYERS:
            Xl[l].append(rows[l])
        rr.extend([len(runs)] * n)
        runs.append(entry | {"rows": n})

    for i, m in enumerate(pruns):
        if m["arm"] not in (SIM, E1, E2, PL2):
            continue
        src = (test_reads if m["environment"] in TEST else dev_reads).get((m["cell"], m["arm"]))
        if src:
            rows, n = read_rows(*src)
            how = "evened read"
        else:
            ix = sorted(rows_of.get(i, {}).values())
            rows, n = {l: XP[l][ix] for l in R.LAYERS}, len(ix)
            how = "pack rows"
        add({"cell": m["cell"], "arm": m["arm"], "environment": m["environment"], "main_task": m["main_task"], "source": how}, rows, n)
    for cell in dev_plan["planted"]:  # B-dev's evened dev planted reads (R2 undone), as plant-2 copies
        rows, n = read_rows(*dev_planted[(cell, SIM)])
        add({"cell": cell, "arm": PL2, "environment": env_of[cell], "main_task": cell.split(":")[0], "source": "B-dev planted read"}, rows, n)
    rr = np.array(rr)
    X = {l: np.concatenate(Xl[l]) for l in R.LAYERS}
    del XP
    log(f"rows assembled: {len(rr)} rows, {len(runs)} runs")
    env = np.array([r["environment"] for r in runs])
    arm = np.array([r["arm"] for r in runs])
    task = np.array([r["main_task"] for r in runs])
    cell = np.array([r["cell"] for r in runs])
    tfold = np.array([cfg["main_task_to_fold"][x] for x in task])
    pl2_cells = set(cell[arm == PL2])
    dev = torch.device("cuda")
    out_dir = Path("/acts") / OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "runs.json").write_text(json.dumps({"runs": [{k: r[k] for k in ("cell", "arm", "environment", "main_task", "source", "rows")} for r in runs],
                                                   "script_sha256": SCRIPT_SHA}))
    B.P.vol.commit()

    def linear(Xa, Xb, Ya, C, cap):  # step A's fitter and convergence loop (H3's copy)
        Xa, Xb = tp.standardize(Xa.double(), Xb.double())
        Xa, Xb = Xa * C**0.5, Xb * C**0.5
        tol, mi, prev = 1e-4 * min(C**0.5, 1.0), R.MAX_ITER, None
        while True:
            W, b, evals = tp.fit(Xa, Ya, 1.0, mi, tol=tol)
            Wg, bg = W.clone().requires_grad_(True), b.clone().requires_grad_(True)
            loss = (F.binary_cross_entropy_with_logits(Xa @ Wg + bg, Ya, reduction="sum") + 0.5 * (Wg * Wg).sum()) / len(Xa)
            loss.backward()
            gk = torch.maximum(Wg.grad.abs().amax(0), bg.grad.abs()).cpu().numpy()
            if float(gk.max()) <= tol or evals < mi or evals == prev or mi * 4 > cap:
                return tp.decision(Xb, W, b), gk
            mi, prev = mi * 4, evals

    def labels(comp):
        neg, pos = COMPS[comp]
        if comp == "plant2":  # pairs: each plant-2 copy and its own evened fixed-sim twin
            use = ((arm == PL2) | (arm == SIM)) & np.isin(cell, list(pl2_cells))
            unit, kk, tag = cell, KP, "plant2"
        else:
            use = (arm == neg) | (arm == pos)
            unit, kk, tag = task, K, "real"  # the same coins for both real comparisons
        y = (arm == pos).astype(np.float32)
        Y = np.zeros((len(runs), kk), np.float32)
        Y[:, 0] = y
        # the real comparisons draw their coins over one task list (all three real arms), so column k is the same world
        units = sorted(set(unit[use])) if comp == "plant2" else sorted(set(task[np.isin(arm, [SIM, E1, E2])]))
        for k in range(1, kk):
            rngk = np.random.default_rng(R.seed_of("halves-shuffle", tag, k))
            flip = dict(zip(units, rngk.random(len(units)) < 0.5))
            Y[:, k] = [1 - v if (use[i] and flip[unit[i]]) else v for i, v in enumerate(y)]
        return use, Y

    subs = [(l, C) for l in R.LAYERS for C in R.C_GRID]
    res = {"rows": int(len(rr)), "runs": len(runs), "planted_dev": len(dev_plan["planted"]), "planted_test": int(((arm == PL2) & np.isin(env, TEST)).sum())}
    for dname, (train_envs, test_envs) in DIRS.items():
        for comp in COMPS:
            use, Y = labels(comp)
            KK = Y.shape[1]
            cap = R.MAX_ITER if comp == "plant2" else R.MAX_ITER_CAP
            u_r, e_r, f_r = use[rr], env[rr], tfold[rr]
            tr, te = u_r & np.isin(e_r, train_envs), u_r & np.isin(e_r, test_envs)
            va = tr & (f_r == VAL_FOLD)
            r = {"val_auroc": np.full((len(subs), KK), np.nan), "test_run_score": np.full((len(subs), len(runs), KK), np.nan),
                 "grad": np.full((len(subs), 2, KK), np.nan)}
            for j, (part, a, b_) in enumerate((("full", tr, te), ("val", tr & ~va, va))):
                ia, ib = np.flatnonzero(a), np.flatnonzero(b_)
                Ya = torch.from_numpy(Y[rr[ia]]).to(dev).double()
                for s, (l, C) in enumerate(subs):
                    S, ok = linear(torch.from_numpy(np.asarray(X[l][ia], np.float32)).to(dev), torch.from_numpy(np.asarray(X[l][ib], np.float32)).to(dev), Ya, C, cap)
                    kk_, inv = np.unique(rr[ib], return_inverse=True)
                    tot = torch.zeros(len(kk_), KK, dtype=torch.float64, device=dev).index_add_(0, torch.from_numpy(inv).to(dev), S.double())
                    Mv = (tot / torch.from_numpy(np.bincount(inv)).to(dev)[:, None]).cpu().numpy()
                    r["grad"][s, j] = ok
                    if part == "full":
                        r["test_run_score"][s, kk_] = Mv
                    else:
                        R_, Yv = rankdata(Mv, axis=0), Y[kk_]
                        n1 = Yv.sum(0)
                        r["val_auroc"][s] = ((R_ * Yv).sum(0) - n1 * (n1 + 1) / 2) / (n1 * (len(Yv) - n1))
            held = use & np.isin(env, test_envs)
            if not np.isfinite(r["test_run_score"][:, held]).all():
                raise ValueError(f"{dname} {comp}: a held-out run got no score")
            (out_dir / dname).mkdir(exist_ok=True)
            with open(out_dir / dname / f"{comp}.tmp", "wb") as fh:
                np.savez(fh, subs=json.dumps(subs), Y=Y, use=use, script_sha256=SCRIPT_SHA, train_envs=json.dumps(train_envs), test_envs=json.dumps(test_envs), **r)
            os.replace(out_dir / dname / f"{comp}.tmp", out_dir / dname / f"{comp}.npz")
            B.P.vol.commit()
            res[f"{dname}|{comp}"] = {"train_runs": int(len(np.unique(rr[tr]))), "held_runs": int(held.sum())}
            log(f"{dname} {comp} done")
    return res | {"seconds": round(time.time() - t0)}


@app.function(image=image, cpu=1, memory=1024, timeout=120, retries=0)
def version() -> str:
    return SCRIPT_SHA


@app.local_entrypoint()
def main(mode: str = "refit"):
    from probe_shift.models import QWEN3_32B

    if subprocess.run(["git", "status", "--porcelain", str(Path(__file__))], capture_output=True, text=True, check=True).stdout:
        raise SystemExit(f"commit {Path(__file__).name} first")
    folds = json.loads((HERE.parents[1] / "splits/folds_v2.json").read_text())
    cfg = {"main_task_to_fold": folds["task_folds"]["main_task_to_fold"]}
    if modal.Function.from_name(app.name, "version").remote() != SCRIPT_SHA:
        raise SystemExit("the deployed app runs other code: `modal deploy` this revision first")
    if mode == "refit":
        print(modal.Function.from_name(app.name, "refit").spawn(cfg, QWEN3_32B["name"], QWEN3_32B["revision"]).object_id)
    else:
        raise SystemExit(f"unknown mode {mode}")
