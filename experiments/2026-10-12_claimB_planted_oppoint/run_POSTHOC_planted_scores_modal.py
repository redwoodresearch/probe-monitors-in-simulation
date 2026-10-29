"""POST HOC (RULE.md @ 859fbaad; Aswin about 11:00 PT 10-12 "run the proper planted check", 0d's go via b7): score the
plant-2 copies with each direction's frozen sim-trained fixed-vs-e1 detector, for halves (Qwen3-32B) and its Qwen3.5
rerun. The stored arrays score only each comparison's own rows, so copies have no fixed-vs-e1 score there.

Each function re-runs the identical fixed-vs-e1 fits of the run that made the stored arrays, with code copied unchanged:
  qwen3  rows assembled as run_POSTHOC_halves_e2_modal.py @ 53da7f48 (halves' rows; Qwen3-32B reads), stored arrays in
         claimB/halves_v1/5b07509a98fc (fixed_vs_e1) and its POSTHOC_e2 (the direct-e2 refit);
  qwen35 rows assembled as run_halves_qwen35_modal.py @ 2358b925 (Qwen3.5 evened reads + its level-1 pack), stored arrays
         in claimB/halves_qwen35_v1/a273e40df0db.
The one addition: the held-out plant-2 copies (and e2, as both earlier refits did) are scored too. Standardization uses
the training rows only, so scoring more rows changes no fit. Reproduction gate before anything is saved: every
validation AUROC (101 columns) within 1e-3, every stored held-out test score within 1e-4, the same column-0 pick, and the
copies' and twins' scores finite. Saves score arrays only, to <stored folder>/POSTHOC_planted/<direction>/fixed_vs_e1.npz.

probe_shift comes from the as-run commit 23b5c37a (PYTHONPATH=<asrun>/src), so H3's check_b() holds, as in both runs.
  PYTHONPATH=<asrun>/src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy modal deploy \\
      experiments/2026-10-12_claimB_planted_oppoint/run_POSTHOC_planted_scores_modal.py
  PYTHONPATH=<asrun>/src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy python \\
      experiments/2026-10-12_claimB_planted_oppoint/run_POSTHOC_planted_scores_modal.py
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
sys.path[:0] = [str(H3_DIR), str(HERE.parent / "2026-10-11_claimB_slips_evened_probe")] if modal.is_local() else ["/root"]
import run_h3_small_modal as R  # noqa: E402  (H3's runner at 23b5c37a, byte-identical on main: B-dev's code, constants)

H3_SHA12 = "04c3953b139e"
B, M = R.B, R.M
SIM, E1, E2, PL2 = M.SIM, M.REAL, "real-attack-e2", "sim-attack-fixed-plant2"
TEST = list(R.TEST)
DEV = ["auto_workflow", "file_indexing", "model_registry", "slack_clone", "web_scraping"]
DIRS = {"dev_to_test": (DEV, TEST), "test_to_dev": (TEST, DEV)}
K, VAL_FOLD = 101, 0
Q3_DIR, Q3_SHA = "claimB/halves_v1/5b07509a98fc", "5b07509a98fc"  # run_halves_modal.py @ 53da7f48 (its sha's first 12)
Q35_DIR, Q35_SHA = "claimB/halves_qwen35_v1/a273e40df0db", "a273e40df0db"  # run_halves_qwen35_modal.py @ 2358b925
PACK_Q35, Q35_LAYERS = "claimB/level1/packedB_06a0329937c1", (58, 63)
SCRIPT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

app = modal.App("probe-shift_aswin-surya_sub-claimb-plantop_kill-2026-11-12")
image = R.image.add_local_file(H3_DIR / "run_h3_small_modal.py", "/root/run_h3_small_modal.py")


def fit_and_save(X, rr, runs, layers, stored: str, cfg: dict, refs: dict, log) -> dict:
    """The fixed-vs-e1 fits of halves' refit (linear and labels copied unchanged), held-out e2 and plant-2 copies scored
    too; the reproduction gate against the stored arrays; then save."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scipy.stats import rankdata

    from probe_shift import torch_probe as tp

    env = np.array([r["environment"] for r in runs])
    arm = np.array([r["arm"] for r in runs])
    task = np.array([r["main_task"] for r in runs])
    tfold = np.array([cfg["main_task_to_fold"][x] for x in task])
    dev = torch.device("cuda")

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

    def labels():  # halves' labels("fixed_vs_e1"), copied
        use = (arm == E1) | (arm == SIM)
        y = (arm == SIM).astype(np.float32)
        Y = np.zeros((len(runs), K), np.float32)
        Y[:, 0] = y
        units = sorted(set(task[np.isin(arm, [SIM, E1, E2])]))
        for k in range(1, K):
            rngk = np.random.default_rng(R.seed_of("halves-shuffle", "real", k))
            flip = dict(zip(units, rngk.random(len(units)) < 0.5))
            Y[:, k] = [1 - v if (use[i] and flip[task[i]]) else v for i, v in enumerate(y)]
        return use, Y

    subs = [(l, C) for l in layers for C in R.C_GRID]
    use, Y = labels()
    out = {}
    for dname, (train_envs, test_envs) in DIRS.items():
        u_r, e_r, f_r = use[rr], env[rr], tfold[rr]
        tr = u_r & np.isin(e_r, train_envs)
        te = (u_r | np.isin(arm[rr], [E2, PL2])) & np.isin(e_r, test_envs)  # + held-out e2 and plant-2 copies, never trained on
        va = tr & (f_r == VAL_FOLD)
        r = {"val_auroc": np.full((len(subs), K), np.nan), "test_run_score": np.full((len(subs), len(runs), K), np.nan),
             "grad": np.full((len(subs), 2, K), np.nan)}
        for j, (part, a, b_) in enumerate((("full", tr, te), ("val", tr & ~va, va))):
            ia, ib = np.flatnonzero(a), np.flatnonzero(b_)
            Ya = torch.from_numpy(Y[rr[ia]]).to(dev).double()
            for s, (l, C) in enumerate(subs):
                S, ok = linear(torch.from_numpy(np.asarray(X[l][ia], np.float32)).to(dev), torch.from_numpy(np.asarray(X[l][ib], np.float32)).to(dev), Ya, C, R.MAX_ITER_CAP)
                kk_, inv = np.unique(rr[ib], return_inverse=True)
                tot = torch.zeros(len(kk_), K, dtype=torch.float64, device=dev).index_add_(0, torch.from_numpy(inv).to(dev), S.double())
                Mv = (tot / torch.from_numpy(np.bincount(inv)).to(dev)[:, None]).cpu().numpy()
                r["grad"][s, j] = ok
                if part == "full":
                    r["test_run_score"][s, kk_] = Mv
                else:
                    R_, Yv = rankdata(Mv, axis=0), Y[kk_]
                    n1 = Yv.sum(0)
                    r["val_auroc"][s] = ((R_ * Yv).sum(0) - n1 * (n1 + 1) / 2) / (n1 * (len(Yv) - n1))
        held = (use | np.isin(arm, [E2, PL2])) & np.isin(env, test_envs)
        if not np.isfinite(r["test_run_score"][:, held]).all():
            raise ValueError(f"{dname}: a held-out run got no score")
        rep = {}
        for name, (rel, arms) in refs.items():  # every reference file is required; its expected rows are named by arm
            z0 = np.load(Path("/acts") / stored / rel.format(d=dname))
            want = np.isin(arm, arms) & np.isin(env, test_envs)
            if not np.isfinite(z0["test_run_score"][:, want]).all():
                raise ValueError(f"{dname} {name}: the reference lacks scores for its expected rows")
            rep[name] = {"compared_runs": int(want.sum()), "val_auroc_max_abs_diff": float(np.abs(r["val_auroc"] - z0["val_auroc"]).max()),
                         "test_score_max_abs_diff": float(np.abs(r["test_run_score"][:, want] - z0["test_run_score"][:, want]).max()),
                         "same_column0_pick": bool(r["val_auroc"][:, 0].argmax() == z0["val_auroc"][:, 0].argmax())}
        log(f"{dname} reproduction: {rep}")
        if any(not x["same_column0_pick"] or not x["test_score_max_abs_diff"] <= 1e-4 or not x["val_auroc_max_abs_diff"] <= 1e-3
               or x["compared_runs"] == 0 for x in rep.values()):
            raise ValueError(f"{dname}: the rerun does not reproduce the stored fits: {rep}")
        od = Path("/acts") / stored / "POSTHOC_planted" / dname
        od.mkdir(parents=True, exist_ok=True)
        with open(od / "fixed_vs_e1.tmp", "wb") as fh:
            np.savez(fh, subs=json.dumps(subs), Y=Y, use=use, script_sha256=SCRIPT_SHA, train_envs=json.dumps(train_envs), test_envs=json.dumps(test_envs), **r)
        os.replace(od / "fixed_vs_e1.tmp", od / "fixed_vs_e1.npz")
        B.P.vol.commit()
        out[dname] = {"held_runs": int(held.sum()), "held_copies": int((held & (arm == PL2)).sum()), "reproduction": rep}
        log(f"{dname} saved")
    return out


def check_runs(runs, stored):
    meta = json.loads((Path("/acts") / stored / "runs.json").read_text())
    if [(r["cell"], r["arm"], r["rows"]) for r in meta["runs"]] != [(r["cell"], r["arm"], r["rows"]) for r in runs]:
        raise ValueError(f"the assembled runs differ from {stored}/runs.json")
    return meta["script_sha256"]


@app.function(image=image, gpu="H100!", cpu=8, memory=131072, volumes={"/acts": B.P.vol}, timeout=40 * 60, retries=0)
def qwen3(cfg: dict, model: str, revision: str) -> dict:
    """halves' rows (Qwen3-32B), assembled exactly as run_POSTHOC_halves_e2_modal.py @ 53da7f48."""
    import numpy as np

    from probe_shift.q1_eval import step_table

    R.check_b()
    if R.SCRIPT_SHA[:12] != H3_SHA12:
        raise ValueError("H3's runner differs from the one whose extraction wrote the evened test reads")
    B.P.vol.reload()
    t0 = time.time()
    log = lambda m: print(f"[qwen3] {time.time() - t0:6.0f}s {m}", flush=True)  # noqa: E731
    for base in (R.OUT, R.DEV_READS):
        if not json.loads((Path("/acts") / base / "gate.json").read_text()).get("passed"):
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
        else:
            ix = sorted(rows_of.get(i, {}).values())
            rows, n = {l: XP[l][ix] for l in R.LAYERS}, len(ix)
        add({"cell": m["cell"], "arm": m["arm"], "environment": m["environment"], "main_task": m["main_task"]}, rows, n)
    for cell in dev_plan["planted"]:
        rows, n = read_rows(*dev_planted[(cell, SIM)])
        add({"cell": cell, "arm": PL2, "environment": env_of[cell], "main_task": cell.split(":")[0]}, rows, n)
    del XP
    stored_sha = check_runs(runs, Q3_DIR)
    if stored_sha[:12] != Q3_SHA:
        raise ValueError("halves' runs.json comes from another runner version")
    X = {l: np.concatenate(Xl[l]) for l in R.LAYERS}
    log(f"rows assembled: {sum(r['rows'] for r in runs)} rows, {len(runs)} runs")
    return fit_and_save(X, np.array(rr), runs, R.LAYERS, Q3_DIR, cfg, {"fixed_vs_e1": ("{d}/fixed_vs_e1.npz", [SIM, E1]),
                                                                       "direct_e2": ("POSTHOC_e2/{d}/fixed_vs_e1.npz", [SIM, E1, E2])}, log) | {"seconds": round(time.time() - t0)}


@app.function(image=image, gpu="H100!", cpu=8, memory=131072, volumes={"/acts": B.P.vol}, timeout=40 * 60, retries=0)
def qwen35(cfg: dict) -> dict:
    """The Qwen3.5 rerun's rows, assembled exactly as run_halves_qwen35_modal.py @ 2358b925 refit()."""
    import numpy as np

    from probe_shift.q1_eval import step_table

    R.check_b()
    if R.SCRIPT_SHA[:12] != H3_SHA12:
        raise ValueError("H3's runner differs from the one whose reads defined the changed sets")
    B.P.vol.reload()
    t0 = time.time()
    log = lambda m: print(f"[qwen35] {time.time() - t0:6.0f}s {m}", flush=True)  # noqa: E731
    ev = json.loads((Path("/acts") / Q35_DIR / "evened" / "index.json").read_text())
    if not ev.get("passed") or ev["script_sha256"][:12] != Q35_SHA:
        raise ValueError("no passed Qwen3.5 reproduction gate from the runner that made the stored arrays")
    EV = {l: np.load(Path("/acts") / Q35_DIR / "evened" / f"L{l}.npy") for l in Q35_LAYERS}
    pruns = json.loads((Path("/acts") / PACK_Q35 / "runs.L63.json").read_text())
    t = step_table(pruns, "output_end")
    XP = {l: np.load(Path("/acts") / PACK_Q35 / f"L{l}_output_end.npy") for l in Q35_LAYERS}
    rows_of = {}
    for k in range(len(t["run"])):
        rows_of.setdefault(int(t["run"][k]), {})[(int(t["step"][k]), int(t["action"][k]))] = k
    ev_by = {(x["cell"], x["arm"]): x for x in ev["runs"] if x["kind"] in ("h3_test_changed", "bdev_dev_changed")}
    planted_dev = [x for x in ev["runs"] if x["kind"] == "bdev_dev_planted"]

    def ev_rows(x):
        rng = {tuple(s) for s in x["range"]}
        keep = [x["row_start"] + j for j, s in enumerate(x["steps"]) if tuple(s) not in rng]
        return {l: np.asarray(EV[l][keep], np.float16) for l in Q35_LAYERS}, len(keep)

    runs, rr, Xl = [], [], {l: [] for l in Q35_LAYERS}

    def add(entry, rows, n):
        if n == 0:
            raise ValueError(f"{entry['cell']} {entry['arm']}: no rows")
        for l in Q35_LAYERS:
            Xl[l].append(rows[l])
        rr.extend([len(runs)] * n)
        runs.append(entry | {"rows": n})

    used = set()
    for i, m in enumerate(pruns):
        if m["arm"] not in (SIM, E1, E2, PL2):
            continue
        src = ev_by.get((m["cell"], m["arm"]))
        if src:
            rows, n = ev_rows(src)
            used.add((m["cell"], m["arm"]))
        else:
            ix = sorted(rows_of.get(i, {}).values())
            rows, n = {l: XP[l][ix] for l in Q35_LAYERS}, len(ix)
        add({"cell": m["cell"], "arm": m["arm"], "environment": m["environment"], "main_task": m["main_task"]}, rows, n)
    if used != set(ev_by):
        raise ValueError(f"{len(set(ev_by) - used)} evened reads matched no pack run")
    for x in planted_dev:
        rows, n = ev_rows(x)
        add({"cell": x["cell"], "arm": PL2, "environment": x["environment"], "main_task": x["main_task"]}, rows, n)
    del XP, EV
    check_runs(runs, Q35_DIR)
    X = {l: np.concatenate(Xl[l]) for l in Q35_LAYERS}
    log(f"rows assembled: {sum(r['rows'] for r in runs)} rows, {len(runs)} runs")
    return fit_and_save(X, np.array(rr), runs, Q35_LAYERS, Q35_DIR, cfg, {"fixed_vs_e1": ("{d}/fixed_vs_e1.npz", [SIM, E1, E2])}, log) | {"seconds": round(time.time() - t0)}


@app.function(image=image, cpu=1, memory=1024, timeout=120, retries=0)
def version() -> str:
    return SCRIPT_SHA


def main():
    from probe_shift.models import QWEN3_32B

    if subprocess.run(["git", "status", "--porcelain", str(Path(__file__))], capture_output=True, text=True, check=True).stdout:
        raise SystemExit(f"commit {Path(__file__).name} first")
    R.check_b()  # locally too: probe_shift must be the as-run version
    if modal.Function.from_name(app.name, "version").remote() != SCRIPT_SHA:
        raise SystemExit("the deployed app runs other code: `modal deploy` this revision first")
    folds = json.loads((HERE.parents[1] / "splits/folds_v2.json").read_text())
    cfg = {"main_task_to_fold": folds["task_folds"]["main_task_to_fold"]}
    c3 = modal.Function.from_name(app.name, "qwen3").spawn(cfg, QWEN3_32B["name"], QWEN3_32B["revision"])
    c35 = modal.Function.from_name(app.name, "qwen35").spawn(cfg)
    print(f"qwen3: {c3.object_id}\nqwen35: {c35.object_id}", flush=True)
    res = {"qwen3": c3.get(timeout=45 * 60), "qwen35": c35.get(timeout=45 * 60)}
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
