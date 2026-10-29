"""Coverage of d9's 20-split pooled interval for recipe #1 P2 minus a_m (RULE.md Amendment 3; POST HOC). Gate first
(d9's stored per-row D reproduces the result JSON; my recomputed D equals the stored rows), then R simulated datasets in
the smoothed world, worlds F and E, intervals V1 and V2 (O+P). Inputs read from the work Volume, sha256-checked.

  PYTHONPATH=src:experiments/2026-10-13_countmatched_coverage_POSTHOC uv run --no-project --python 3.12 \\
      --with 'modal==1.6.1' --with numpy modal run --detach \\
      experiments/2026-10-13_countmatched_coverage_POSTHOC/run_cmcov_splits_modal.py [--pilot] [--n 300] [--chunks 2]
"""
import hashlib
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal
import numpy as np
from run_cmcov_modal import B_IN, DRAWS, E2, R1, R1MET, SEED, SIM_SEED, STORED, B, G, Gate, H, kept, pct, vol, work_vol

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
SPLITS, PAIRING = 20, ("real-attack-e1", E2, "sim-attack", "sim-attack-fixed")
R1RUN = ("recipe1/2026-10-13_173403/recipe1_run_2026-10-13_173403.json", "88aea1b89a339e8e34a557423fcba145a79bc7ee9b888c4431f4f19c22850aa4")
RESEED = ("recipe1_reseed/2026-10-13_reseed_v2/recipe1_reseed_POSTHOC_2026-10-13_182501.json", "83d2f7f1032e207f")
AMS, AM0 = "recipe1_amsplits/2026-10-13_184928/", "recipe1_am/2026-10-13_175506/P2/"
RES = (AMS + "recipe1_amsplits_v2_POSTHOC_2026-10-13_185609.json", "03f0c0a2")
ROWS = {0.01: (AMS + "D_G_rows_fpr_0.01_2026-10-13_185609.npz", "31445ede38b5cb69"), 0.05: (AMS + "D_G_rows_fpr_0.05_2026-10-13_185609.npz", "4bc6680339ea9ffb")}

app = modal.App("probe-shift_aswin-surya_sub-cmcov_splits_kill-2026-11-13")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "scipy==1.16.2").add_local_python_source("probe_shift", "run_cmcov_modal")


def vpre(rel, prefix):
    p = Path("/work") / rel
    if not hashlib.sha256(p.read_bytes()).hexdigest().startswith(prefix):
        raise Gate(f"{rel}: sha256 differs from the pin")
    return p


def load():
    """Runs, per-split per-fold configs (d9's splits_and_runs), and the unique base models: #1's [5 v, n] and a_m's [5 d, 5 v, n] per (fold, config)."""
    Z, Zc = np.load(vol(*R1)), np.load(vol(*STORED["avg"]))
    arm, key, task = (Z[f"runs_P2__{k}"].astype(str) for k in ("arm", "key", "main_task"))
    if not (np.array_equal(arm, Zc["runs__arm"].astype(str)) and np.array_equal(key, Zc["runs__key"].astype(str))):
        raise Gate("P2 runs are not in the stored order")
    rec, rs, res = json.loads(vol(*R1RUN).read_text()), json.loads(vpre(*RESEED).read_text()), json.loads(vpre(*RES).read_text())
    cfg = lambda l, ro, C: (int(l), str(ro), float(C))
    ch = {(0, f): cfg(*rec["choice"]["P2"][str(f)]["chosen"]) for f in range(5)}
    for u in rs["parts"]["P2"]["unit_list"]:
        ch[u["seed"], u["fold"]] = cfg(u["layer"], u["readout"], u["C"])
    if sorted(ch) != [(s, f) for s in range(SPLITS) for f in range(5)]:
        raise Gate("a split or fold has no config")
    amrec = json.loads(vol(*R1MET).read_text())
    chp = {f: cfg(*c) for f, c in enumerate(amrec["parts"]["P2"]["choice_per_fold"])}
    files = {(f, *chp[f]): (AM0 + x["file"], x["sha256"]) for x in amrec["fits"] if x["part"] == "P2" for f in x["folds"]}
    files |= {(r["fold"], *cfg(r["layer"], r["readout"], r["C"])): (AMS + r["file"], r["sha256"]) for x in res["fits"] for r in x["files"]}
    keys = sorted({(f, *ch[s, f]) for s in range(SPLITS) for f in range(5)})
    Xb, Ab, cache = {}, {}, {}
    for k in keys:
        f, l, ro, C = k
        Xb[k] = Z[f"P2__L{l}__{ro}__C{C:g}__sim"][f]
        if k not in files:
            raise Gate(f"fold {f}: no a_m file at its config")
        if files[k][0] not in cache:
            cache[files[k][0]] = np.load(vol(*files[k]))["nested"]
        Ab[k] = cache[files[k][0]][:, f]
    return arm, task, Z["runs_P2__fold"].astype(int), Zc["runs__cell"].astype(str), ch, keys, Xb, Ab, res


def hits(base, f, W, rate, present):
    """[Bw, n_e2(f)] share of fold f's 4 nested models (rows v of base) whose redrawn threshold each e2 run exceeds."""
    from probe_shift import tail_threshold as tt

    e = G["e2f"][f]
    out = np.zeros((W.shape[0], len(e)), np.float32)
    for v in [v for v in range(5) if v != f]:
        s = base[v]
        hv = np.flatnonzero(G["hon"][v] & np.isfinite(s) & present)
        out += s[e][None, :] > tt.threshold(s[hv], W[:, hv], rate, "order")[:, None]
    return out / 4


def dmatrix(XH, AH, W):
    """[20, 5, Bw] D per split, draw and row: #1's composed catch minus a_m's, on e2 runs kept by both."""
    e2, pos = G["e2"], G["pos"]
    We, D = W[:, e2], np.zeros((SPLITS, DRAWS, W.shape[0]))
    for s in range(SPLITS):
        cx, ca = np.empty((W.shape[0], len(e2)), np.float32), np.empty((DRAWS, W.shape[0], len(e2)), np.float32)
        for f in range(5):
            k = (f, *G["ch"][s, f])
            cx[:, pos[f]], ca[:, :, pos[f]] = XH[k], AH[k]
        for d in range(DRAWS):
            m = G["both"][s, d]
            w = We[:, m]
            D[s, d] = ((w * cx[:, m]).sum(1) - (w * ca[d][:, m]).sum(1)) / w.sum(1)
    return D


def v1v2(D, rng):  # V1: all 100 pairs per row; V2: per row 20 splits, then 5 draws each, with replacement (row 0 all)
    Bw = D.shape[2]
    ss, dd = rng.integers(0, SPLITS, (Bw, SPLITS)), rng.integers(0, DRAWS, (Bw, SPLITS, DRAWS))
    ss[0], dd[0] = np.arange(SPLITS), np.arange(DRAWS)
    return D.mean((0, 1)), D[ss[:, :, None], dd, np.arange(Bw)[:, None, None]].mean((1, 2))


def one_dataset(args):
    rate, r = args
    rng = np.random.default_rng([SIM_SEED, 99, round(rate * 100), r])
    nt, tix, n = G["n_tasks"], G["tix"], len(G["fold"])
    c = np.zeros(nt)
    for f in range(5):  # main tasks resampled within each fold; copies share their noise
        tf = G["tasks_of_fold"][f]
        c += np.bincount(tf[rng.integers(0, len(tf), len(tf))], minlength=nt)
    W = np.vstack([c[tix]] + [rng.multinomial(nt, c / nt)[tix] for _ in range(B_IN)]).astype(float)
    present = W[0] > 0
    noisy = lambda base, h, f: np.stack([base[v] + rng.normal(0, h[v], n) if v != f else base[v] for v in range(5)])
    XH = {k: hits(noisy(G["Xb"][k], G["hX"][k], k[0]), k[0], W, rate, present) for k in G["keys"]}
    AH = {k: np.stack([hits(noisy(G["Ab"][k][d], G["hA"][k][d], k[0]), k[0], W, rate, present) for d in range(DRAWS)]) for k in G["keys"]}
    D = dmatrix(XH, AH, W)
    ssE, ddE = rng.integers(0, SPLITS, SPLITS), rng.integers(0, DRAWS, (SPLITS, DRAWS))
    out = {}
    for world, Dw in (("F", D), ("E", D[ssE[:, None], ddE])):
        for ver, x in zip(("V1", "V2"), v1v2(Dw, rng)):
            lo, hi = pct(x)
            out[f"{world}|{ver}"] = (bool(lo <= G["mu"][rate] <= hi), float(hi - lo)) if np.isfinite([lo, hi]).all() else None
    return out


@app.function(image=image, cpu=32, memory=32768, timeout=1800, retries=0, volumes={"/work": work_vol.read_only()})
def shard(rate: float, r0: int, r1: int) -> dict:
    try:
        return _shard(rate, r0, r1)
    except Gate as e:
        return {"rate": rate, "r0": r0, "gate_failed": str(e)}


def _shard(rate, r0, r1):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    from probe_shift import tail_threshold as tt

    t0 = time.time()
    arm, task, fold, cell, ch, keys, Xb, Ab, res = load()
    tasks, tix = np.unique(task, return_inverse=True)
    if any(len(set(fold[tix == t])) != 1 for t in range(len(tasks))):
        raise Gate("a main task spans folds")
    e2 = np.flatnonzero(arm == E2)
    G.update(arm=arm, fold=fold, cell=cell, pairing=list(PAIRING), ch=ch, keys=keys, Xb=Xb, Ab=Ab, tix=tix, n_tasks=len(tasks), e2=e2,
             pos={f: np.flatnonzero(fold[e2] == f) for f in range(5)}, e2f={f: e2[fold[e2] == f] for f in range(5)},
             hon={v: (arm == H) & (fold == v) for v in range(5)},
             tasks_of_fold={f: np.flatnonzero(np.array([fold[tix == t][0] for t in range(len(tasks))]) == f) for f in range(5)})
    comp = lambda get, s: np.stack([get((f, *ch[s, f])) for f in range(5)])
    kx = [kept(comp(lambda k: Xb[k], s))[e2] for s in range(SPLITS)]
    G["both"] = {(s, d): kx[s] & kept(comp(lambda k, d=d: Ab[k][d], s))[e2] for s in range(SPLITS) for d in range(DRAWS)}
    out = {"rate": rate, "r0": r0, "r1": r1, "n_unique_models": len(keys), "gate": {}}
    # gate 1: d9's stored per-row D reproduces the result JSON with d9's seeds
    k = f"fpr_{rate:g}"
    Ds, ref = np.load(vpre(*ROWS[rate]))["D"], res["by_alpha"][k]
    rs, rd = np.random.default_rng(20261015), np.random.default_rng(20261016)
    V2 = np.array([Ds[ss[:, None], dd, b].mean() for b in range(1, B + 1) for ss, dd in [(rs.integers(0, SPLITS, SPLITS), rd.integers(0, DRAWS, (SPLITS, DRAWS)))]])
    r4 = lambda x: [round(float(np.percentile(x, q)), 4) for q in (2.5, 97.5)]
    mine = [round(float(Ds[:, :, 0].mean()), 4), r4(V2), r4(Ds[:, :, 1:].mean((0, 1)))]
    want = [ref["D_point_mean_over_splits_and_draws"], ref["V2_primary"]["D_ci95"], ref["V1_beside"]["D_ci95"]]
    out["gate"]["stored_rows_vs_json"] = {"mine": mine, "stored": want}
    if mine != want:
        raise Gate(f"{k}: stored per-row D does not reproduce the result JSON")
    # gate 2: my D from the scores equals the stored per-row D
    W = tt.task_weights(tix, len(tasks), B, SEED).astype(float)
    every = np.ones(len(arm), bool)
    XH = {q: hits(Xb[q][:], q[0], W, rate, every) for q in keys}
    AH = {q: np.stack([hits(Ab[q][d], q[0], W, rate, every) for d in range(DRAWS)]) for q in keys}
    gap = float(np.abs(dmatrix(XH, AH, W) - Ds).max())
    out["gate"]["recomputed_vs_stored_max_abs"] = gap
    if not np.isfinite(gap) or gap > 1e-9:
        raise Gate(f"{k}: recomputed D differs from the stored rows by {gap:.3g}")
    del W, XH, AH
    out["gate_seconds"] = round(time.time() - t0, 1)
    # the smoothed world: per-model bandwidths, truths, then the datasets
    G["hX"] = {q: {v: tt.silverman(Xb[q][v][G["hon"][v] & np.isfinite(Xb[q][v])]) for v in range(5) if v != q[0]} for q in keys}
    G["hA"] = {q: [{v: tt.silverman(Ab[q][d][v][G["hon"][v] & np.isfinite(Ab[q][d][v])]) for v in range(5) if v != q[0]} for d in range(DRAWS)] for q in keys}

    def truth(base, f, h):
        S = {(f, v): base[v] for v in h}
        cal = {(f, v): np.flatnonzero(G["hon"][v] & np.isfinite(base[v])) for v in h}
        return tt.smoothed_truth(S, fold, cal, {(f, v): x for v, x in h.items()}, rate)[G["e2f"][f]]
    TX = {q: truth(Xb[q], q[0], G["hX"][q]) for q in keys}
    TA = {q: np.stack([truth(Ab[q][d], q[0], G["hA"][q][d]) for d in range(DRAWS)])[:, None, :] for q in keys}
    tD = dmatrix({q: x[None, :] for q, x in TX.items()}, TA, np.ones((1, len(arm))))[:, :, 0]
    G["mu"] = {rate: float(tD.mean())}
    out["truth"] = {"mu": G["mu"][rate], "split_means": tD.mean(1).tolist()}
    t1 = time.time()
    with ProcessPoolExecutor(32, mp_context=multiprocessing.get_context("fork")) as ex:
        rows = list(ex.map(one_dataset, [(rate, r) for r in range(r0, r1)]))
    out["dataset_seconds_wall"] = round(time.time() - t1, 1)
    out["rows"] = rows
    print("RESULT_JSON " + json.dumps(out), flush=True)  # recoverable from the app logs if the client drops
    return out


@app.local_entrypoint()
def main(pilot: bool = False, n: int = 300, chunks: int = 2):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=REPO, check=True).stdout
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("uncommitted tracked changes")
    head = git("rev-parse", "HEAD").strip()
    jobs = [(0.01, 0, 2)] if pilot else [(r, i * n // chunks, (i + 1) * n // chunks) for r in (0.01, 0.05) for i in range(chunks)]
    res = list(shard.starmap(jobs, return_exceptions=True))
    out = {"label": "POST HOC (RULE.md Amendment 3)" + (" pilot" if pilot else "; preliminary until Codex agrees"), "git": head,
           "shards": [x if isinstance(x, dict) else {"error": repr(x)[:500]} for x in res]}
    for rate in (0.01, 0.05):
        sh = [x for x in out["shards"] if x.get("rate") == rate and "rows" in x]
        rows = [r for x in sh for r in x["rows"]]
        if rows:
            out[f"coverage_fpr_{rate:g}"] = {c: {"coverage": float(np.mean([r[c][0] for r in rows if r[c]])), "mean_width": float(np.mean([r[c][1] for r in rows if r[c]])),
                                                 "n": sum(1 for r in rows if r[c]), "n_invalid": sum(1 for r in rows if not r[c])} for c in rows[0]}
    now = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    p = HERE / "results" / f"cmcov_splits{'_pilot' if pilot else ''}_{now}.json"
    with p.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(f"wrote {p}")
