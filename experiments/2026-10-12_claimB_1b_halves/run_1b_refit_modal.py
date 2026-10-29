"""Llama-3.2-1B through the halves test (RULE.md here): halves' refit with the reader swapped to Llama. Keys, numbers and
score arrays only.

  --mode refit  1 H100: the rows are assembled as the Qwen3.5 halves runner assembled them
                (aswin/claimb-halves-qwen35 run_halves_qwen35_modal.py @2358b925), with Llama sources: changed runs (H3's
                test, B-dev's dev) take their evened Llama reads (run_1b_read_modal.py) without "Invalid `view_range`"
                outputs; unchanged runs keep the stored Llama level-1 pack rows (claimB/level1/packedB_091e169bff86, the
                small-model baseline's); B-dev's 100 dev planted copies are plant-2 copies. The analysis has exactly 3,067
                runs (3 x 939 cells + 150 test and 100 dev plant-2 copies). Layers: Llama modules 14 and 15, the relative
                depths of halves' Qwen3-32B L58 and L63 (run_small_lm_modal.py's module map; RULE.md amendment 1). Fitting,
                label columns, seeds and folds are halves' code unchanged; the fixed-vs-e1 probes also score the held-out
                e2 runs (the direct yardstick) and plant-2 copies, which changes no fit.

probe_shift comes from the as-run commit 23b5c37a (PYTHONPATH=<asrun>/src), so H3's check_b() holds, as in the halves runners.

  PYTHONPATH=<asrun>/src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy modal deploy \\
      experiments/2026-10-12_claimB_1b_halves/run_1b_refit_modal.py
  PYTHONPATH=<asrun>/src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy python \\
      experiments/2026-10-12_claimB_1b_halves/run_1b_refit_modal.py refit
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
H3_DIR = HERE.parent / "2026-10-11_claimB_h3_fix_to_plateau"
sys.path[:0] = [str(HERE), str(H3_DIR), str(HERE.parent / "2026-10-11_claimB_slips_evened_probe")] if modal.is_local() else ["/root"]
import run_1b_read_modal as RD  # noqa: E402  (the reads: OUT, fid, the staged run)
import run_h3_small_modal as R  # noqa: E402  (H3's runner: B-dev's code and constants; hash-checked below)

H3_SHA12 = "04c3953b139e"
B, M = R.B, R.M
SIM, E1, E2, PL2 = M.SIM, M.REAL, "real-attack-e2", "sim-attack-fixed-plant2"
TEST = list(R.TEST)
DEV = ["auto_workflow", "file_indexing", "model_registry", "slack_clone", "web_scraping"]
DIRS = {"dev_to_test": (DEV, TEST), "test_to_dev": (TEST, DEV)}
COMPS = {"fixed_vs_e1": (E1, SIM), "e2_vs_e1": (E1, E2), "plant2": (SIM, PL2)}  # (negative arm, positive arm)
K, KP, VAL_FOLD, LAYERS = 101, 21, 0, (14, 15)
PACK = "claimB/level1/packedB_091e169bff86"  # the small-model baseline's Llama level-1 pack (output_end, 8 modules)
N_RUNS = 3 * 939 + 150 + 100
READ_SHA = RD.SCRIPT_SHA
SCRIPT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
OUT = f"claimB/halves_1b_v1/refit_{READ_SHA[:8]}_{SCRIPT_SHA[:12]}"

app = modal.App("probe-shift_aswin-surya_sub-claimb-1b-refit_kill-2026-11-12")
image = (R.image.add_local_file(H3_DIR / "run_h3_small_modal.py", "/root/run_h3_small_modal.py")
         .add_local_file(HERE / "run_1b_read_modal.py", "/root/run_1b_read_modal.py"))


def check_code():
    R.check_b()
    if R.SCRIPT_SHA[:12] != H3_SHA12:
        raise ValueError("H3's runner differs from the one whose reads defined the changed sets")


@app.function(image=image, gpu="H100!", cpu=8, memory=131072, volumes={"/acts": B.P.vol}, timeout=120 * 60, retries=0, scaledown_window=10)
def refit(cfg: dict) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scipy.stats import rankdata

    from probe_shift import torch_probe as tp
    from probe_shift.q1_eval import step_table

    check_code()
    B.P.vol.reload()
    t0 = time.time()
    log = lambda m: print(f"[halves-1b] {time.time() - t0:6.0f}s {m}", flush=True)  # noqa: E731
    g = json.loads((Path("/acts") / RD.OUT / "gate.json").read_text())
    if not g["passed"] or g["script_sha256"] != READ_SHA:
        raise ValueError("no passed Llama gate from the read runner's version")
    man = json.loads((Path("/acts") / RD.STAGED / "manifest.json").read_text())
    if man["script_sha256"] != RD.STAGE_SHA:
        raise ValueError("the staged run was made by another staging script")
    ev = {}  # (kind, cell, arm) -> (rows by layer without range outputs, n)
    for it in man["items"]:
        if it["kind"] == "h3_gate_unchanged":
            continue
        z = np.load(Path("/acts") / RD.OUT / "reads" / f"{RD.fid(it['arm'], it['key'])}.npz")
        m = json.loads(str(z["meta"]))
        if (m["script_sha256"], m["model"], m["revision"], m["key"], m["arm"]) != (READ_SHA, RD.MODEL, RD.REVISION, it["key"], it["arm"]):
            raise ValueError(f"{it['key']}: read by other code, model or for another item")
        outs = [p for p in m["positions"] if p["kind"] == "output_end"]
        if it["range_tool_ordinals"] and max(it["range_tool_ordinals"]) >= len(outs):
            raise ValueError(f"{it['key']}: a range output has no output_end position")
        rng = {(outs[k]["step"], outs[k]["action"]) for k in it["range_tool_ordinals"]}
        keep = [p["row"] for p in outs if p["kept"] and (p["step"], p["action"]) not in rng]
        rows = {l: np.asarray(z[f"L{l}"][keep], np.float16) for l in LAYERS}
        if not keep or not all(np.isfinite(rows[l].astype(np.float32)).all() for l in LAYERS):
            raise ValueError(f"{it['key']}: no kept rows after range outputs, or non-finite values")
        ev[(it["kind"], it["cell"], it["arm"])] = (rows, len(keep), it)
    pruns = json.loads((Path("/acts") / PACK / "runs.L15.json").read_text())
    t = step_table(pruns, "output_end")
    XP = {l: np.load(Path("/acts") / PACK / f"L{l}_output_end.npy") for l in LAYERS}
    rows_of = {}
    for k in range(len(t["run"])):
        rows_of.setdefault(int(t["run"][k]), {})[(int(t["step"][k]), int(t["action"][k]))] = k
    ev_by = {(c, a): v for (kind, c, a), v in ev.items() if kind in ("h3_test_changed", "bdev_dev_changed")}
    planted_dev = [v for (kind, c, a), v in ev.items() if kind == "bdev_dev_planted"]

    runs, rr, Xl = [], [], {l: [] for l in LAYERS}

    def add(entry, rows, n):
        if n == 0:
            raise ValueError(f"{entry['cell']} {entry['arm']}: no rows")
        for l in LAYERS:
            Xl[l].append(rows[l])
        rr.extend([len(runs)] * n)
        runs.append(entry | {"rows": n})

    used = set()
    for i, m in enumerate(pruns):
        if m["arm"] not in (SIM, E1, E2, PL2):
            continue
        src = ev_by.get((m["cell"], m["arm"]))
        if src:
            rows, n, _ = src
            how = "evened Llama read"
            used.add((m["cell"], m["arm"]))
        else:
            ix = sorted(rows_of.get(i, {}).values())
            rows, n = {l: XP[l][ix] for l in LAYERS}, len(ix)
            how = "Llama level-1 pack rows"
        add({"cell": m["cell"], "arm": m["arm"], "environment": m["environment"], "main_task": m["main_task"], "source": how}, rows, n)
    if used != set(ev_by):
        raise ValueError(f"{len(set(ev_by) - used)} evened reads matched no pack run")
    for rows, n, it in planted_dev:  # B-dev's evened dev planted reads (R2 undone), as plant-2 copies
        add({"cell": it["cell"], "arm": PL2, "environment": it["environment"], "main_task": it["main_task"], "source": "B-dev planted (Llama)"}, rows, n)
    if len(runs) != N_RUNS:
        raise ValueError(f"{len(runs)} runs assembled, not {N_RUNS}")
    rr = np.array(rr)
    X = {l: np.concatenate(Xl[l]) for l in LAYERS}
    del XP
    log(f"rows assembled: {len(rr)} rows, {len(runs)} runs")
    # --- from here to the end of the direction loop: run_halves_qwen35_modal.py refit() at 2358b925, unchanged except where marked ---
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
                                                   "script_sha256": SCRIPT_SHA, "read_script_sha256": READ_SHA}))
    B.P.vol.commit()

    def linear(Xa, Xb, Ya, C, cap):  # step A's fitter and convergence loop (halves' copy)
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
        units = sorted(set(unit[use])) if comp == "plant2" else sorted(set(task[np.isin(arm, [SIM, E1, E2])]))
        for k in range(1, kk):
            rngk = np.random.default_rng(R.seed_of("halves-shuffle", tag, k))
            flip = dict(zip(units, rngk.random(len(units)) < 0.5))
            Y[:, k] = [1 - v if (use[i] and flip[unit[i]]) else v for i, v in enumerate(y)]
        return use, Y

    subs = [(l, C) for l in LAYERS for C in R.C_GRID]
    res = {"rows": int(len(rr)), "runs": len(runs), "planted_dev": len(planted_dev), "planted_test": int(((arm == PL2) & np.isin(env, TEST)).sum())}
    for dname, (train_envs, test_envs) in DIRS.items():
        for comp in COMPS:
            use, Y = labels(comp)
            KK = Y.shape[1]
            cap = R.MAX_ITER if comp == "plant2" else R.MAX_ITER_CAP
            u_r, e_r, f_r = use[rr], env[rr], tfold[rr]
            tr, te = u_r & np.isin(e_r, train_envs), u_r & np.isin(e_r, test_envs)
            extra = np.isin(arm, [E2, PL2])  # (1B: the direct yardstick and the plant-2 copies, scored by the sim-trained probe)
            te_full = te | (extra[rr] & np.isin(e_r, test_envs)) if comp == "fixed_vs_e1" else te
            va = tr & (f_r == VAL_FOLD)
            r = {"val_auroc": np.full((len(subs), KK), np.nan), "test_run_score": np.full((len(subs), len(runs), KK), np.nan),
                 "grad": np.full((len(subs), 2, KK), np.nan)}
            for j, (part, a, b_) in enumerate((("full", tr, te_full), ("val", tr & ~va, va))):
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
            tols = np.array([1e-4 * min(C**0.5, 1.0) for _, C in subs])  # linear()'s tolerance per candidate
            r["converged"] = r["grad"] <= tols[:, None, None]  # (1B: per candidate, part (full, val) and column; RULE.md amendment 2)
            held = (use | (extra if comp == "fixed_vs_e1" else False)) & np.isin(env, test_envs)  # (1B: + e2 and plant-2 copies)
            if not np.isfinite(r["test_run_score"][:, held]).all():
                raise ValueError(f"{dname} {comp}: a held-out run got no score")
            (out_dir / dname).mkdir(exist_ok=True)
            with open(out_dir / dname / f"{comp}.tmp", "wb") as fh:
                np.savez(fh, subs=json.dumps(subs), Y=Y, use=use, script_sha256=SCRIPT_SHA, read_script_sha256=READ_SHA,
                         train_envs=json.dumps(train_envs), test_envs=json.dumps(test_envs), **r)
            os.replace(out_dir / dname / f"{comp}.tmp", out_dir / dname / f"{comp}.npz")
            B.P.vol.commit()
            res[f"{dname}|{comp}"] = {"train_runs": int(len(np.unique(rr[tr]))), "held_runs": int(held.sum())}
            log(f"{dname} {comp} done")
    return res | {"seconds": round(time.time() - t0)}


@app.function(image=image, cpu=1, memory=1024, timeout=120, retries=0)
def version() -> str:
    return f"{SCRIPT_SHA}|{READ_SHA}"  # both runners: a stale read runner in the deployed image fails the handshake


def main(mode: str):
    for f in (Path(__file__), HERE / "run_1b_read_modal.py"):
        if subprocess.run(["git", "status", "--porcelain", str(f)], capture_output=True, text=True, check=True).stdout:
            raise SystemExit(f"commit {f.name} first")
    check_code()  # locally too: probe_shift must be the as-run version
    if modal.Function.from_name(app.name, "version").remote() != f"{SCRIPT_SHA}|{READ_SHA}":
        raise SystemExit("the deployed app runs other code: `modal deploy` this revision first")
    if mode != "refit":
        raise SystemExit(f"unknown mode {mode}")
    folds = json.loads((HERE.parents[1] / "splits/folds_v2.json").read_text())
    call = modal.Function.from_name(app.name, "refit").spawn({"main_task_to_fold": folds["task_folds"]["main_task_to_fold"]})
    print(f"refit: {call.object_id}", flush=True)
    print(json.dumps(call.get(timeout=130 * 60), indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
