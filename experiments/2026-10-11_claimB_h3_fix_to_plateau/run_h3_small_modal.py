"""H3 amendment 1, the smaller H3 (PLAN.md on branch aswin/claimb-h-plans; approved before 11:41 PT 10-11 via 0d):
B-dev's four evenings applied once to the test environments' runs (both arms and the planted copies), the changed
ones read once by B-dev's reader, then step A's pipeline refit at L58/L63 x 9 C on evened data from all 10
environments (dev: B-dev's evened reads), scored on the 5 test environments with a main-task-grouped null. B-dev's
code (reader, evening, frozen refit, staged inputs) is imported unchanged and hash-checked. Keys and numbers only.

  --mode cells    CPU: per test run, whether the evening changes it (plant 1 is re-inserted into the evened twin at its
                  staged site, so plants come after the evening; plant 2's R2 undo commutes with the evening); the jobs
                  and 4 unchanged gate runs; cells.json.
  --mode gate     1 H100: the 4 gate runs read unevened must equal the pack rows bitwise at L58 and L63, before any
                  bulk extraction.
  --mode extract  10 H100 shards (spawned, only after the gate passed): evened copies of the changed test runs.
  --mode refit    1 H100 (spawned): gate check, rows assembled per run, the refit grid on 101 label columns per
                  comparison, and B-dev's frozen L63 probes on the evened test runs; arrays saved for the summary.
                  H2 read "inconclusive" (h2_summary_2026-10-11_132926.json), so H2's winning configuration M2 (an MLP
                  on L63 output_end, PCA 512, weight decay 1e-4; H2's code copied unchanged) is refit beside it, after
                  M2 reproduces H2's fixed-vs-e1 test AUROC on the unevened rows within 0.005. Planted checks: the real
                  labels plus 20 refit per-pair shuffles; planted linear fits capped at 5,000 iterations (H2's cap).

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy \\
      modal run experiments/2026-10-11_claimB_h3_fix_to_plateau/run_h3_small_modal.py --mode cells
"""

import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).parent
BDEV = HERE.parent / "2026-10-11_claimB_slips_evened_probe"
sys.path.insert(0, str(BDEV) if modal.is_local() else "/root")
import run_slips_evened_modal as B  # noqa: E402

B_SCRIPT = "0abee4a4d8730c0d3330e4702d179dfd8d3a11a228279e18e84fc4baa1d4cd06"
B_DEPS = "33d539e9a646c5adfeb3b8069c708c3f696f1ac794e09c26a281448bf0c6a65d"
M = B.M
DEV_READS, STAGED_A = "claimB/slips_evened_dev_v1", "2026-10-10_la_fixed-sim-v2_probe-inputs-l1_v1"
TEST = ["clinical_trial", "media_processing", "shared_dotfiles", "tiktok", "user_tracking"]
ARMS = (M.SIM, M.REAL, "real-attack-e2", "sim-attack-fixed-plant1", "sim-attack-fixed-plant2")
COMPS = ("fixed_vs_e1", "e2_vs_e1", "plant1", "plant2")
LAYERS, C_GRID = (58, 63), [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
K, MAX_ITER, MAX_ITER_CAP, SHARDS, N_GATE = 101, 5000, 80000, 10, 4
M2_H2_AUROC, M2_TOL = 0.8407671781485635, 0.005  # H2's M2, fixed sim vs e1, test (h2_summary_2026-10-11_132926.json)
PLANT_SHUFFLES = 20  # refit per-pair shuffles per planted check (Aswin ~13:56 PT 10-11, PR #722 review QRWy7)
SCRIPT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
OUT = f"claimB/h3_small_v1/{SCRIPT_SHA[:12]}"  # one folder per script version: no read made by other H3 code is reused

app = modal.App("probe-shift_aswin-surya_sub-claimb-h3_small_kill-2026-11-11")
image = B.image.add_local_file(BDEV / "run_slips_evened_modal.py", "/root/run_slips_evened_modal.py")


def seed_of(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def check_b():
    if (B.SCRIPT_SHA, B.DEPS_SHA) != (B_SCRIPT, B_DEPS):
        raise ValueError("B-dev's code here differs from the code that ran")


def staged():
    """Every staged level-1 run (all 10 environments, the five arms), by (cell, arm)."""
    return {(r["cell"], r["arm"]): r for r in B.MP.inputs(M.STAGED_B) if r["arm"] in ARMS}


def evened_version(runs, cell, arm):
    """(evened messages, range indices) of a run. Plant 1 is re-made on the evened twin at its staged site (the plant
    goes in after the evening); every other run, plant 2 included (R2 undo and the evening touch different outputs),
    is evened directly."""
    if arm != "sim-attack-fixed-plant1":
        e, _, rng = B.evened(runs[(cell, arm)]["messages"])
        return e, rng
    twin = runs[(cell, M.SIM)]
    site = {x["key"]: x for x in json.loads((Path("/acts") / STAGED_A / "manifest.json").read_text())["plant1"]["sites"]}[twin["key"]]
    e, _, rng = B.evened(twin["messages"])
    e = [dict(m) for m in e]
    lines = e[site["msg"]]["content"].split("\n")
    line = json.loads((Path("/acts") / STAGED_A / "manifest.json").read_text())["plant1"]["line"]
    e[site["msg"]] = e[site["msg"]] | {"content": "\n".join(lines[:site["after_line"]] + [line] + lines[site["after_line"]:])}
    return e, rng


@app.function(image=image, cpu=8, memory=32768, volumes={"/acts": B.P.vol}, timeout=20 * 60, retries=0)
def cells() -> dict:
    check_b()
    runs = staged()
    changed, unchanged, ranges = [], [], 0
    for (cell, arm), r in sorted(runs.items()):
        if r["environment"] not in TEST:
            continue
        e, rng = evened_version(runs, cell, arm)
        (changed if (e != r["messages"] or rng) else unchanged).append([cell, arm])
        ranges += bool(rng)
    gate = random.Random("h3-small-gate").sample(sorted(c for c, a in unchanged if a == M.SIM), N_GATE)
    plan = {"script_sha256": SCRIPT_SHA, "changed": changed, "gate": gate, "counts": {
        "test_runs": len(changed) + len(unchanged), "changed": len(changed), "runs_with_range_outputs": ranges,
        "changed_by_arm": {a: sum(1 for _, x in changed if x == a) for a in ARMS}}}
    (Path("/acts") / OUT).mkdir(parents=True, exist_ok=True)
    (Path("/acts") / OUT / "cells.json").write_text(json.dumps(plan))
    B.P.vol.commit()
    return plan["counts"] | {"gate": len(gate)}


def load_plan():
    plan = json.loads((Path("/acts") / OUT / "cells.json").read_text())
    if plan["script_sha256"] != SCRIPT_SHA:
        raise ValueError("cells.json was made by another version of this script")
    return plan


def jobs_for(shard):
    plan, runs = load_plan(), staged()
    if shard == "gate":
        return [(c, M.SIM, "original", runs[(c, M.SIM)]["messages"], []) for c in plan["gate"]]
    g = json.loads((Path("/acts") / OUT / "gate.json").read_text())
    if not g["passed"] or g["script_sha256"] != SCRIPT_SHA:
        raise ValueError("the gate has not passed for this script version: no bulk extraction")
    jobs = []
    for cell, arm in plan["changed"]:
        e, rng = evened_version(runs, cell, arm)
        jobs.append((cell, arm, "evened", e, list(rng)))
    return jobs[shard::SHARDS]


@app.function(image=image, gpu="H100!", cpu=8, memory=65536, volumes={"/acts": B.P.vol, "/hf": B.P.hf}, timeout=15 * 60, retries=0)
def gate(model: str, revision: str) -> dict:
    """The 4 unchanged test runs, read unevened, must equal the pack rows bitwise at L58 and L63."""
    import numpy as np

    from probe_shift.q1_eval import step_table

    check_b()
    B.OUT = OUT
    stats = B._read(jobs_for("gate"), model, revision, "gate")
    plan = load_plan()
    pruns = json.loads((Path("/acts") / M.PACK / "runs.L63.json").read_text())
    t = step_table(pruns, "output_end")
    key_of = {(m["cell"], m["arm"]): i for i, m in enumerate(pruns)}
    rows_of = {}
    for k in range(len(t["run"])):
        rows_of.setdefault(int(t["run"][k]), {})[(int(t["step"][k]), int(t["action"][k]))] = k
    ok = []
    for c in plan["gate"]:
        z = np.load(Path("/acts") / OUT / "runs" / f"{hashlib.sha256(f'{c}|{M.SIM}|original'.encode()).hexdigest()[:16]}.npz")
        m = json.loads(str(z["meta"]))
        steps, pr = [tuple(x) for x in m["steps"]], rows_of[key_of[(c, M.SIM)]]
        ok.append(set(steps) == set(pr) and all(np.array_equal(z[f"L{l}"], np.load(Path("/acts") / M.PACK / f"L{l}_output_end.npy", mmap_mode="r")[[pr[s] for s in steps]])
                                                 for l in LAYERS))
    res = {"passed": bool(len(ok) == N_GATE and all(ok)), "runs_bitwise": int(sum(ok)), "script_sha256": SCRIPT_SHA} | stats
    (Path("/acts") / OUT / "gate.json").write_text(json.dumps(res))
    B.P.vol.commit()
    return res


@app.function(image=image, gpu="H100!", cpu=8, memory=65536, volumes={"/acts": B.P.vol, "/hf": B.P.hf}, timeout=60 * 60, retries=0)
def extract(shard: int, model: str, revision: str) -> dict:
    check_b()
    B.OUT = OUT  # B-dev's reader writes under OUT/runs, keeping B-dev's script hash in each file
    return B._read(jobs_for(shard), model, revision, f"shard{shard}")


@app.function(image=image, gpu="H100!", cpu=8, memory=131072, volumes={"/acts": B.P.vol}, timeout=150 * 60, retries=0)
def refit(cfg: dict, model: str, revision: str) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score

    from probe_shift import q1_eval
    from probe_shift import torch_probe as tp
    from probe_shift.q1_eval import Scorer, step_table

    check_b()
    B.P.vol.reload()
    t0 = time.time()
    log = lambda m: print(f"[h3 refit] {time.time() - t0:6.0f}s {m}", flush=True)  # noqa: E731
    plan = load_plan()
    g = json.loads((Path("/acts") / OUT / "gate.json").read_text())
    if not g["passed"] or g["script_sha256"] != SCRIPT_SHA:
        raise ValueError("the gate has not passed for this script version")
    pruns = json.loads((Path("/acts") / M.PACK / "runs.L63.json").read_text())
    t = step_table(pruns, "output_end")
    XP = {l: np.load(Path("/acts") / M.PACK / f"L{l}_output_end.npy") for l in LAYERS}
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

    test_reads, dev_reads = files(OUT, "evened"), files(DEV_READS, "evened")
    want = {tuple(x) for x in plan["changed"]}
    dev_want = {tuple(x) for x in json.loads((Path("/acts") / DEV_READS / "cells.json").read_text())["changed"]}
    if not want <= set(test_reads) or not dev_want <= set(dev_reads):
        raise ValueError(f"reads missing: {len(want - set(test_reads))} evened test runs, {len(dev_want - set(dev_reads))} B-dev dev runs")
    gate = [True] * N_GATE  # checked by the gate step before extraction (gate.json)
    # rows per run: evened reads (range outputs out) for changed runs, else the pack's rows
    rr, Xl, changed_runs = [], {l: [] for l in LAYERS}, 0
    for i, m in enumerate(pruns):
        src = test_reads.get((m["cell"], m["arm"])) or (dev_reads.get((m["cell"], m["arm"])) if m["environment"] not in TEST else None)
        if src:
            meta, z = src
            rng = {tuple(x) for x in meta["range"]}
            keep = [k for k, s in enumerate(meta["steps"]) if tuple(s) not in rng]
            for l in LAYERS:
                Xl[l].append(np.asarray(z[f"L{l}"][keep], np.float16))
            rr += [i] * len(keep)
            changed_runs += 1
        else:
            ix = sorted(rows_of.get(i, {}).values())
            for l in LAYERS:
                Xl[l].append(XP[l][ix])
            rr += [i] * len(ix)
    rr = np.array(rr)
    X = {l: np.concatenate(Xl[l]) for l in LAYERS}
    no_rows = sorted(set(range(len(pruns))) - set(rr.tolist()))
    log(f"rows assembled: {len(rr)} rows, {changed_runs} runs from evened reads")
    q1_eval.DESIGNS["claimB"] = {"comparisons": {"fixed_vs_e1": ((M.REAL,), (M.SIM,)), "e2_vs_e1": ((M.REAL,), ("real-attack-e2",)),
                                                 "plant1": ((M.SIM,), ("sim-attack-fixed-plant1",)), "plant2": ((M.SIM,), ("sim-attack-fixed-plant2",))}}
    env = np.array([cfg["environment_to_fold"][m["environment"]] for m in pruns])
    folds = sorted({cfg["environment_to_fold"][e] for e in TEST})
    dev = torch.device("cuda")
    out_dir = Path("/acts") / OUT / "refit"
    out_dir.mkdir(parents=True, exist_ok=True)

    def linear(Xa, Xb, Ya, C, cap):  # step A's fitter and convergence loop; planted comparisons get no refit chain (H2's cap)
        Xa, Xb = tp.standardize(Xa.double(), Xb.double())
        Xa, Xb = Xa * C**0.5, Xb * C**0.5
        tol, mi, prev = 1e-4 * min(C**0.5, 1.0), MAX_ITER, None
        while True:
            W, b, evals = tp.fit(Xa, Ya, 1.0, mi, tol=tol)
            Wg, bg = W.clone().requires_grad_(True), b.clone().requires_grad_(True)
            loss = (F.binary_cross_entropy_with_logits(Xa @ Wg + bg, Ya, reduction="sum") + 0.5 * (Wg * Wg).sum()) / len(Xa)
            loss.backward()
            gk = torch.maximum(Wg.grad.abs().amax(0), bg.grad.abs()).cpu().numpy()  # per label column
            g = float(gk.max())
            if g <= tol or evals < mi or evals == prev or mi * 4 > cap:
                return tp.decision(Xb, W, b), gk
            mi, prev = mi * 4, evals

    def pca(Xa, Xb, tag):  # H2's M2 input, copied unchanged: standardize, PCA 512 on the training rows, unit variance
        Xa, Xb = tp.standardize(Xa.float(), Xb.float())
        torch.manual_seed(seed_of("h2-pca", tag))
        _, _, V = torch.pca_lowrank(Xa, q=512, center=False, niter=4)
        Za, Zb = Xa @ V, Xb @ V
        sd = Za.std(0) + 1e-6
        return Za / sd, Zb / sd

    def mlp(Za, Zb, Ya, tag):  # H2's M2, copied unchanged: independent one-hidden-layer MLPs, one per label column
        KK, d = Ya.shape[1], Za.shape[1]
        W1 = torch.stack([torch.randn(d, 256, generator=torch.Generator().manual_seed(seed_of("h2-init", tag, k))) / d**0.5 for k in range(KK)]).to(dev)
        ps = [W1.requires_grad_(True), torch.zeros(KK, 1, 256, device=dev, requires_grad=True), torch.zeros(KK, 256, 1, device=dev, requires_grad=True),
              torch.zeros(KK, 1, device=dev, requires_grad=True)]
        opt = torch.optim.AdamW(ps, lr=1e-3, weight_decay=1e-4)
        torch.manual_seed(seed_of("h2-dropout", tag))
        for _ in range(200):
            Hd = F.dropout(torch.relu(torch.einsum("nd,cdh->cnh", Za, ps[0]) + ps[1]), 0.1, training=True)
            loss = F.binary_cross_entropy_with_logits((Hd @ ps[2]).squeeze(-1) + ps[3], Ya.T, reduction="none").mean(1).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            return ((torch.relu(torch.einsum("nd,cdh->cnh", Zb, ps[0]) + ps[1]) @ ps[2]).squeeze(-1) + ps[3]).T

    def m2(X63, row_run, use_r, ef, tf, Y, comp):
        """M2 per test fold, as in H2 (same seed tags): full fit -> test run means; validation fit -> validation AUROC."""
        from scipy.stats import rankdata

        KK = Y.shape[1]
        r = {"val_auroc": np.full((1, len(folds), KK), np.nan), "test_run_score": np.full((1, len(pruns), KK), np.nan)}
        for i, f in enumerate(folds):
            tr, te = use_r & (ef != f), use_r & (ef == f)
            va = tr & (tf == f % 5)
            for part, a, b_ in (("full", tr, te), ("val", tr & ~va, va)):
                ia, ib = np.flatnonzero(a), np.flatnonzero(b_)
                Za, Zb = pca(torch.from_numpy(np.asarray(X63[ia], np.float32)).to(dev), torch.from_numpy(np.asarray(X63[ib], np.float32)).to(dev), f"{comp}|M2|{f}|{part}")
                S = mlp(Za, Zb, torch.from_numpy(Y[row_run[ia]]).to(dev).double().float(), f"{comp}|M2|0|{f}|{part}")
                kk, inv = np.unique(row_run[ib], return_inverse=True)
                tot = torch.zeros(len(kk), KK, dtype=torch.float64, device=dev).index_add_(0, torch.from_numpy(inv).to(dev), S.double())
                Mv = (tot / torch.from_numpy(np.bincount(inv)).to(dev)[:, None]).cpu().numpy()
                if part == "full":
                    r["test_run_score"][0, kk] = Mv
                else:
                    R_, Yv = rankdata(Mv, axis=0), Y[kk]
                    n1 = Yv.sum(0)
                    r["val_auroc"][0, i] = ((R_ * Yv).sum(0) - n1 * (n1 + 1) / 2) / (n1 * (len(Yv) - n1))
            torch.cuda.empty_cache()
        return r

    def labels(comp):
        sc = Scorer(cfg | {"design": "claimB"}, pruns, comp)
        unit = [m["main_task"] if not comp.startswith("plant") else m["cell"] for m in pruns]
        units = sorted({unit[i] for i in np.flatnonzero(sc.runs_in)})
        Y = np.zeros((len(pruns), K), np.float32)
        Y[:, 0] = sc.y_run
        for k in range(1, K):  # the same coins for both real comparisons (Step 0's same-coin worlds); per pair for plants
            rngk = np.random.default_rng(seed_of("h3-shuffle", "real" if not comp.startswith("plant") else comp, k))
            flip = dict(zip(units, rngk.random(len(units)) < 0.5))
            Y[:, k] = [1 - v if (sc.runs_in[i] and flip[unit[i]]) else v for i, v in enumerate(sc.y_run)]
        if comp.startswith("plant"):  # the real labels plus 20 refit per-pair shuffles, same fitting, selection and caps
            Y = Y[:, :1 + PLANT_SHUFFLES]
        return sc, Y

    # M2 reproduction before any evened M2 fit: H2's rows (the pack's kept output_end rows, same order), fixed sim vs e1
    sc0, Y0 = labels("fixed_vs_e1")
    tr_run = np.asarray(t["run"])
    r0 = m2(XP[63], tr_run, sc0.runs_in[tr_run], env[tr_run], sc0.fold[tr_run], Y0, "fixed_vs_e1")
    m_te = sc0.runs_in & np.isin(env, folds)
    auc0 = float(roc_auc_score(Y0[m_te, 0], r0["test_run_score"][0, m_te, 0]))
    repro = {"auroc": auc0, "h2": M2_H2_AUROC, "pass": abs(auc0 - M2_H2_AUROC) <= M2_TOL}
    (out_dir / "m2_reproduction.json").write_text(json.dumps(repro | {"script_sha256": SCRIPT_SHA}))
    B.P.vol.commit()
    log(f"M2 reproduction: {repro}")
    if not repro["pass"]:
        raise ValueError(f"M2 does not reproduce H2: {auc0:.4f} vs {M2_H2_AUROC:.4f}")

    res = {"gate_runs_bitwise": int(sum(gate)), "rows": int(len(rr)), "runs_from_evened_reads": changed_runs, "m2_reproduction": repro}
    for comp in COMPS:
        sc, Y = labels(comp)
        KK = Y.shape[1]
        use, efold, tfold = sc.runs_in[rr], env[rr], sc.fold[rr]
        subs = [(l, C) for l in LAYERS for C in C_GRID]
        if set(np.flatnonzero(sc.runs_in)) & set(no_rows):
            raise ValueError(f"{comp}: compared runs with no rows after range exclusion")
        r = {"val_auroc": np.full((len(subs), len(folds), KK), np.nan), "test_run_score": np.full((len(subs), len(pruns), KK), np.nan),
             "grad": np.full((len(subs), len(folds), 2, KK), np.nan)}
        for i, f in enumerate(folds):
            tr, te = use & (efold != f), use & (efold == f)
            va = tr & (tfold == f % 5)
            for j, (part, a, b_) in enumerate((("full", tr, te), ("val", tr & ~va, va))):
                ia, ib = np.flatnonzero(a), np.flatnonzero(b_)
                Ya = torch.from_numpy(Y[rr[ia]]).to(dev).double()
                for s, (l, C) in enumerate(subs):
                    S, ok = linear(torch.from_numpy(np.asarray(X[l][ia], np.float32)).to(dev), torch.from_numpy(np.asarray(X[l][ib], np.float32)).to(dev), Ya, C,
                                   MAX_ITER if comp.startswith("plant") else MAX_ITER_CAP)
                    kk, inv = np.unique(rr[ib], return_inverse=True)
                    tot = torch.zeros(len(kk), KK, dtype=torch.float64, device=dev).index_add_(0, torch.from_numpy(inv).to(dev), S.double())
                    Mv = (tot / torch.from_numpy(np.bincount(inv)).to(dev)[:, None]).cpu().numpy()
                    r["grad"][s, i, j] = ok  # per-column final gradients (step A had no convergence filter; reported)
                    if part == "full":
                        r["test_run_score"][s, kk] = Mv
                    else:
                        from scipy.stats import rankdata
                        R_, Yv = rankdata(Mv, axis=0), Y[kk]
                        n1 = Yv.sum(0)
                        r["val_auroc"][s, i] = ((R_ * Yv).sum(0) - n1 * (n1 + 1) / 2) / (n1 * (len(Yv) - n1))
            log(f"{comp} fold {f} done")
        tested = sc.runs_in & np.isin(env, folds)
        if not np.isfinite(r["test_run_score"][:, tested]).all():
            raise ValueError(f"{comp}: a compared test run got no score")
        with open(out_dir / f"{comp}.tmp", "wb") as fh:
            np.savez(fh, folds=np.array(folds), subs=json.dumps(subs), Y=Y, script_sha256=SCRIPT_SHA, **r)
        os.replace(out_dir / f"{comp}.tmp", out_dir / f"{comp}.npz")
        B.P.vol.commit()
        rm = m2(X[63], rr, use, efold, tfold, Y, comp)  # H2's M2 on the evened rows
        if not np.isfinite(rm["test_run_score"][:, tested]).all():
            raise ValueError(f"{comp}: a compared test run got no M2 score")
        with open(out_dir / f"{comp}_M2.tmp", "wb") as fh:
            np.savez(fh, folds=np.array(folds), subs=json.dumps([{"config": "M2", "wd": 1e-4}]), Y=Y, script_sha256=SCRIPT_SHA, **rm)
        os.replace(out_dir / f"{comp}_M2.tmp", out_dir / f"{comp}_M2.npz")
        B.P.vol.commit()
        log(f"{comp} M2 done")
    # B-dev's frozen L63 probes (the meaning control's refits on the unevened pack) on the evened test runs, descriptive
    arm_of = np.array([pruns[i]["arm"] for i in t["run"]])
    efold_t = np.array([cfg["environment_to_fold"][pruns[i]["environment"]] for i in t["run"]])
    frozen, frozen_fits = {}, {}
    for e in TEST:
        (mean, std, W, b), fit = B.refit(e, 63, cfg["environment_to_fold"], arm_of, efold_t, XP)
        frozen_fits[e] = fit
        sel = np.flatnonzero(np.array([pruns[i]["environment"] == e for i in rr]))
        S = ((((torch.from_numpy(np.asarray(X[63][sel], np.float32)).cuda().double() - mean) / std * M.C**0.5) @ W + b)[:, 0]).cpu().numpy()
        for i_run in np.unique(rr[sel]):
            frozen[int(i_run)] = float(S[rr[sel] == i_run].mean())
    (out_dir / "frozen_run_scores.json").write_text(json.dumps({"scores": frozen, "fits": frozen_fits, "script_sha256": SCRIPT_SHA}))
    B.P.vol.commit()
    return res | {"seconds": round(time.time() - t0)}


@app.function(image=image, cpu=1, memory=1024, timeout=120, retries=0)
def version() -> str:
    return SCRIPT_SHA


@app.local_entrypoint()
def main(mode: str = "cells"):
    from probe_shift.models import QWEN3_32B

    if subprocess.run(["git", "status", "--porcelain", str(Path(__file__))], capture_output=True, text=True, check=True).stdout:
        raise SystemExit(f"commit {Path(__file__).name} first")
    folds = json.loads((HERE.parents[1] / "splits/folds_v2.json").read_text())
    cfg = {"packs": {"level1": "2cfd1aec393c"}, "seed": 0, "bootstrap": 1000, "tag": "h3",
           "main_task_to_fold": folds["task_folds"]["main_task_to_fold"], "environment_to_fold": folds["environment_folds"]["environment_to_fold"]}
    model, rev = QWEN3_32B["name"], QWEN3_32B["revision"]
    if mode in ("gate", "extract", "refit") and modal.Function.from_name(app.name, "version").remote() != SCRIPT_SHA:
        raise SystemExit("the deployed app runs other code: `modal deploy` this revision first")
    if mode == "cells":
        print(json.dumps(cells.remote(), indent=1))
    elif mode == "gate":
        res = modal.Function.from_name(app.name, "gate").remote(model, rev)
        print(json.dumps(res, indent=1))
        if not res["passed"]:
            raise SystemExit("gate failed: no extraction")
    elif mode == "extract":  # spawn and return; each shard writes its own files
        f = modal.Function.from_name(app.name, "extract")
        for s in range(SHARDS):
            print(s, f.spawn(s, model, rev).object_id, flush=True)
    elif mode == "refit":
        print(modal.Function.from_name(app.name, "refit").spawn(cfg, model, rev).object_id)
    else:
        raise SystemExit(f"unknown mode {mode}")
