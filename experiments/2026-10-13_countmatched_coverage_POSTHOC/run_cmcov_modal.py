"""Coverage of the count-matched training comparisons (RULE.md; POST HOC). One Modal CPU shard per target and rate:
the gate, the real-data intervals (V1 fixed draws = d9's Amendment 21b, V2 draws resampled; O+P, O+S, HD+P), then R
simulated datasets in the smoothed world (worlds F and E). Inputs are read-only from the work Volume, sha256-checked.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' modal run --detach \\
      experiments/2026-10-13_countmatched_coverage_POSTHOC/run_cmcov_modal.py --n 300
"""
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE  # inside the container the module sits at /root
APP = "probe-shift_aswin-surya_sub-cmcov_coverage_kill-2026-11-13"
WORK = "probe-shift_aswin-surya_sub-q3attack_work_kill-2026-11-06"
B, SEED, B_IN, SIM_SEED, DRAWS = 2000, 20261013, 400, 20261015, 5
H, E2 = "real-honest", "real-attack-e2"
PAIRS = [(f, v) for f in range(5) for v in range(5) if v != f]
STORED = {"avg": ("recipe1/stored/sce_meanpool_b_scores_2026-10-13_140912.npz", "6a099d93875096c3b9ed3369c66b0afeb93cf20a54b0a3e39367a118d3ce4385"),
          "lt": ("/inputs/sce_scores_2026-10-13_121538.npz", "c8f81042965ec1305dad34b35c38702423f6617679f1c02b367cc36c761c94cd")}  # b, b_fixed live only here
NEG = "sce_negcount/2026-10-13_173339/"
AM = {("avg", 48): ("nested_avg_L48_b45a028407cf.npz", "261669eab57de8c1b7ea86405e234f670f2f1a12208aecf8e5e9cf2aa172930f"),
      ("lt", 48): ("nested_lt_L48_6db524163524.npz", "07b254b90cef27fe58e3d69dde935792a997e78427a2f6132e33d43ff619d2c4"),
      ("lt", 63): ("nested_lt_L63_5b0887a37c15.npz", "ab9a909d51565bf4ede9eaf55b01d6f27a7d312d5b0a957ea151f560740bc672")}
R1 = ("recipe1/2026-10-13_173403/recipe1_scores_2026-10-13_173403.npz", "3e42da3f206c8154d2defc442ac63722addb9affedf29f2f058b74a6a77eb5b3")
R1AM = {40: ("recipe1_am/2026-10-13_175506/P2/am_P2_L40_cfffab825b4c.npz", "2f86379c8946bb9833e9c7b02c9ba22965497b378a2266d298b10ee43e7fd25f"),
        48: ("recipe1_am/2026-10-13_175506/P2/am_P2_L48_f0fd647ac2be.npz", "74f17e5bfcc50002bb1e88f0260a78a03ea6460bda823cbeef5d941cc8401433"),
        63: ("recipe1_am/2026-10-13_175506/P2/am_P2_L63_44e809bf1b35.npz", "c510782b46d5f651cdd1857273c3d6298aea3df1db7194c7f714cff7c9a27273")}
R1MET = ("recipe1_am/2026-10-13_175506/recipe1_am_metrics_POSTHOC_2026-10-13_175506.json", "093e209f4e5eefa9d242a5d8ba40fc8cb9b3a7ce05987ba2270e64c513c47cad")
R1SCHED = {0: 40, 1: 48, 2: 48, 3: 63, 4: 48}  # #1's P2 per-fold layer choice
D9MET = ("ec605eb75bbd0a8fa5c5f20f44f1f133117f4833", "experiments/2026-10-13_q3_sce_reply_probe/results/sce_negcount_metrics_2026-10-13_174005.json")
TARGETS = ["avg48", "lt48", "lt63", "recipeP2"]
METHODS = ("O+P", "O+S", "HD+P")
G = {}  # data shared with forked worker processes

app = modal.App(APP)
work_vol = modal.Volume.from_name(WORK, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "scipy==1.16.2").add_local_python_source("probe_shift")
LT_LOCAL = Path(os.environ.get("CMCOV_IN", "/nonexistent")) / Path(STORED["lt"][0]).name  # the last-token file isn't on the Volume
if LT_LOCAL.is_file():
    image = image.add_local_file(LT_LOCAL, STORED["lt"][0])


class Gate(Exception):
    """A failed check; its message holds numbers only."""


def vol(rel, sha):
    p = Path("/work") / rel
    if hashlib.sha256(p.read_bytes()).hexdigest() != sha:
        raise Gate(f"{rel}: sha256 differs from the pin")
    return p


def load(target, d9met):
    """arm, fold, tix, n_tasks, cell, pairing, cells {name: nested [5, 5, n]}, draws [5 x nested], ref {(cell, rate): stored}."""
    if target == "recipeP2":
        Z, Zc = np.load(vol(*R1)), np.load(vol(*STORED["avg"]))
        arm, key, task = (Z[f"runs_P2__{k}"].astype(str) for k in ("arm", "key", "main_task"))
        fold = Z["runs_P2__fold"]
        if not (np.array_equal(arm, Zc["runs__arm"].astype(str)) and np.array_equal(key, Zc["runs__key"].astype(str))):
            raise Gate("P2 runs are not in the stored order")
        cell, get = Zc["runs__cell"].astype(str), lambda L: Z[f"P2__L{L}__output_end_mean__C0.01__sim"]
        X = np.full_like(get(48), np.nan)
        am = np.full((DRAWS,) + X.shape, np.nan, np.float32)
        for f, L in R1SCHED.items():
            X[f] = get(L)[f]
            am[:, f] = np.load(vol(*R1AM[L]))["nested"][:, f]
        if any(np.isnan(am[:, f, v]).all(-1).any() for f, v in PAIRS):
            raise Gate("an a_m model is missing")
        pairing = [a for a in ("real-attack-e1", E2, "sim-attack", "sim-attack-fixed") if (arm == a).any()]
        met = json.loads(vol(*R1MET).read_text())["parts"]["P2"]["D_vs_a_m"]
        cells, draws, ref = {"P2": X}, list(am), {("P2", r): met[f"fpr_{r:g}"] for r in (0.01, 0.05)}
    else:
        pack, L = target[:-2], int(target[-2:])
        Zp = np.load(vol(*STORED[pack]))
        arm, task, cell = (Zp[f"runs__{k}"].astype(str) for k in ("arm", "main_task", "cell"))
        fold = Zp["runs__fold"]
        n = len(arm)
        Na = np.load(vol(NEG + AM[pack, L][0], AM[pack, L][1]))
        cells = {c: Zp[f"L{L}__{c}__nested_real"] for c in ("b", "b_fixed")}
        draws = [Na[f"a|1.0|{d}"][..., :n] for d in range(DRAWS)]
        pairing = [a for a in np.unique(arm) if "attack" in a]
        ref = {(c, r): d9met["D_vs_a_m"][f"{pack}|{L}|{c}|fpr_{r:g}"] for c in cells for r in (0.01, 0.05)}
    fold = fold.astype(int)
    tasks, tix = np.unique(task, return_inverse=True)
    if any(len(set(fold[tix == t])) != 1 for t in range(len(tasks))):
        raise Gate("a main task spans folds")
    return arm, fold, tix, len(tasks), cell, pairing, cells, draws, ref


def kept(nested):  # d9's keep rule: a pairing arm's cell is dropped if any of its runs lacks a model
    arm, fold, cell, pairing = G["arm"], G["fold"], G["cell"], G["pairing"]
    ok = np.ones(len(arm), bool)
    for f, v in PAIRS:
        ok &= ~((fold == f) & ~np.isfinite(nested[f, v]))
    bad = set(cell[np.isin(arm, pairing) & ~ok])
    return np.where(np.isin(arm, pairing), ~np.isin(cell, list(bad)), ok)


def catch(nested, W, rate, method, present, zfun):
    """[Bw, n_e2] share of the 4 nested models whose threshold each second-real-run attack run exceeds."""
    from probe_shift import tail_threshold as tt

    fold, e = G["fold"], G["e2"]
    out = np.zeros((W.shape[0], len(e)), np.float32)
    for f, v in PAIRS:
        s = nested[f, v]
        hv = np.flatnonzero((G["arm"] == H) & (fold == v) & np.isfinite(s) & present)
        if method == "O+S":  # d9's Amendment 21b jitter: Silverman with ddof 1, the same noise for every detector
            x = s[hv]
            xr = np.repeat(x, W[0, hv].astype(int))  # Amendment 1: the bandwidth counts each run's outer copies (all 1 on real data)
            h = 0.9 * min(xr.std(ddof=1), np.subtract(*np.percentile(xr, [75, 25])) / 1.34) * len(xr) ** (-0.2)
            sj = x[None, :] + h * zfun(f, v, hv)
            o = np.argsort(-sj, axis=1, kind="mergesort")
            cw = np.cumsum(np.take_along_axis(W[:, hv], o, 1), 1)
            i = np.argmax(cw > np.floor(rate * cw[:, -1:] + 1e-9), 1)
            T = np.take_along_axis(np.take_along_axis(sj, o, 1), i[:, None], 1)[:, 0]
        else:
            T = tt.threshold(s[hv], W[:, hv], rate, "order" if method == "O+P" else "hd")
        te = fold[e] == f
        out[:, te] += s[e[te]][None, :] > T[:, None]
    return out / 4


def dmat(dets, keeps, W, rate, method, present, zfun):
    """{cell: [5, Bw]} D per draw: cell minus a_m draw d on second-real-run runs kept by both."""
    We, e = W[:, G["e2"]], G["e2"]
    hA = [catch(a, W, rate, method, present, zfun) for a in dets["draws"]]
    out = {}
    for c, X in dets["cells"].items():
        hX = catch(X, W, rate, method, present, zfun)
        rows = []
        for d in range(DRAWS):
            m = keeps[c][e] & keeps[d][e]
            w = We[:, m]
            rows.append(((w * hX[:, m]).sum(1) - (w * hA[d][:, m]).sum(1)) / w.sum(1))
        out[c] = np.array(rows)
    return out


def pct(x):
    return [float(np.percentile(x[1:], 2.5)), float(np.percentile(x[1:], 97.5))]


def versions(Dm, idx2):  # V1: mean over the draws held; V2: the draws resampled per row too
    return Dm.mean(0), Dm[idx2, np.arange(Dm.shape[1])[:, None]].mean(1)


def one_dataset(args):
    rate, r = args
    rng = np.random.default_rng([SIM_SEED, TARGETS.index(G["target"]), round(rate * 100), r])
    nt, tix, n = G["n_tasks"], G["tix"], len(G["arm"])
    c = np.zeros(nt)
    for f in range(5):  # main tasks resampled within each fold; copies share their noise
        tf = G["tasks_of_fold"][f]
        c += np.bincount(tf[rng.integers(0, len(tf), len(tf))], minlength=nt)
    Wout = c[tix]
    W = np.vstack([Wout] + [rng.multinomial(nt, c / nt)[tix] for _ in range(B_IN)]).astype(float)
    noisy = lambda X, hw: np.stack([np.stack([X[f, v] + rng.normal(0, hw[f, v], n) if f != v else X[f, v] for v in range(5)]) for f in range(5)])
    dets = {"cells": {k: noisy(X, G["hw"][k]) for k, X in G["dets"]["cells"].items()},
            "draws": [noisy(X, G["hw"][d]) for d, X in enumerate(G["dets"]["draws"])]}
    Zs = {(f, v): rng.standard_normal((B_IN + 1, len(G["cand"][f, v]))) for f, v in PAIRS}
    for z in Zs.values():
        z[0] = 0.0
    zfun = lambda f, v, hv: Zs[f, v][:, np.searchsorted(G["cand"][f, v], hv)]
    idxE = rng.integers(0, DRAWS, DRAWS)
    idx2 = np.vstack([np.arange(DRAWS), rng.integers(0, DRAWS, (B_IN, DRAWS))])
    out = {}
    for meth in METHODS:
        for c, Dm in dmat(dets, G["keeps"], W, rate, meth, Wout > 0, zfun).items():
            mu, td = G["truth"][c, rate]
            for world, D in (("F", Dm), ("E", Dm[idxE])):
                for ver, x in zip(("V1", "V2"), versions(D, idx2)):
                    lo, hi = pct(x)
                    out[f"{c}|{meth}|{world}|{ver}"] = (lo <= mu <= hi, hi - lo) if np.isfinite([lo, hi]).all() else None
            ends = np.array([pct(Dm[d]) for d in range(DRAWS)])
            out[f"{c}|{meth}|per_draw_mean"] = (float(np.mean((ends[:, 0] <= td) & (td <= ends[:, 1]))), float(np.mean(ends[:, 1] - ends[:, 0]))) if np.isfinite(ends).all() else None
    return out


@app.function(image=image, cpu=32, memory=32768, timeout=1800, retries=0, volumes={"/work": work_vol.read_only()})
def shard(target: str, rate: float, R: int, d9met: dict) -> dict:
    try:
        return _shard(target, rate, R, d9met)
    except Gate as e:
        return {"target": target, "rate": rate, "gate_failed": str(e)}


def _shard(target, rate, R, d9met):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    from probe_shift import tail_threshold as tt

    arm, fold, tix, nt, cell, pairing, cells, draws, ref = load(target, d9met)
    n = len(arm)
    G.update(target=target, arm=arm, fold=fold, tix=tix, n_tasks=nt, cell=cell, pairing=pairing, e2=np.flatnonzero(arm == E2),
             dets={"cells": cells, "draws": draws}, cand={(f, v): np.flatnonzero((arm == H) & (fold == v)) for f, v in PAIRS},
             tasks_of_fold={f: np.flatnonzero(np.array([fold[tix == t][0] for t in range(nt)]) == f) for f in range(5)})
    G["keeps"] = {**{c: kept(X) for c, X in cells.items()}, **{d: kept(X) for d, X in enumerate(draws)}}
    res = {"target": target, "rate": rate, "R": R, "real": {}, "gate": {}, "truth": {}, "coverage": {}}
    # gate, then the real-data intervals (B = 2,000, d9's draws and jitter seeds)
    W = tt.task_weights(tix, nt, B, SEED).astype(float)
    every = np.ones(n, bool)

    def z_real(f, v, hv):
        z = np.random.default_rng([SEED + 7, f, v]).standard_normal((B + 1, len(hv)))
        z[0] = 0.0
        return z
    idx2 = np.vstack([np.arange(DRAWS), np.random.default_rng(SEED + 11).integers(0, DRAWS, (B, DRAWS))])
    for meth in METHODS:
        for c, Dm in dmat(G["dets"], G["keeps"], W, rate, meth, every, z_real).items():
            if meth == "O+P":
                st, lo = ref[c, rate], [pct(Dm[d])[0] for d in range(DRAWS)]
                hi = [pct(Dm[d])[1] for d in range(DRAWS)]
                mine = [min(lo), max(lo)] + [min(hi), max(hi)]
                if not np.isfinite(np.array(st["D_per_draw"] + st["lower_end_min_max"] + st["upper_end_min_max"], float)).all():
                    raise Gate(f"{c} at {rate:g}: a stored reference value is not finite")  # Amendment 2 (after the rerun)
                gap = max(np.abs(np.array(mine) - np.array(st["lower_end_min_max"] + st["upper_end_min_max"])).max(),
                          np.abs(Dm[:, 0] - np.array(st["D_per_draw"])).max())
                res["gate"][c] = float(gap)
                if not np.isfinite(Dm).all() or not np.isfinite(gap) or gap > 1e-4:
                    raise Gate(f"{c} at {rate:g}: per-draw values differ from the stored ones by {gap:.6f}")
            v1, v2 = versions(Dm, idx2)
            res["real"][f"{c}|{meth}"] = {"point": float(v1[0]), "V1_ci95": pct(v1), "V2_ci95": pct(v2),
                                          "per_draw": [[float(Dm[d, 0])] + pct(Dm[d]) for d in range(DRAWS)]}
    del W
    # the smoothed world's truths
    hw, Tc = {}, {}
    for k, X in [*cells.items(), *enumerate(draws)]:
        S = {(f, v): X[f, v] for f, v in PAIRS}
        cal = {(f, v): np.flatnonzero((arm == H) & (fold == v) & np.isfinite(X[f, v])) for f, v in PAIRS}
        hw[k] = {(f, v): tt.silverman(S[f, v][cal[f, v]]) for f, v in PAIRS}
        Tc[k] = tt.smoothed_truth(S, fold, cal, hw[k], rate)
    G["hw"] = hw
    for c in cells:
        td = []
        for d in range(DRAWS):
            m = G["keeps"][c] & G["keeps"][d] & (arm == E2)
            td.append(float(Tc[c][m].mean() - Tc[d][m].mean()))
        G.setdefault("truth", {})[c, rate] = (float(np.mean(td)), td)
        res["truth"][c] = {"mu": float(np.mean(td)), "per_draw": td}
    with ProcessPoolExecutor(32, mp_context=multiprocessing.get_context("fork")) as ex:
        rows = list(ex.map(one_dataset, [(rate, r) for r in range(R)], chunksize=2))
    for k in rows[0]:  # nonfinite intervals are counted apart, not as misses
        ok = [x[k] for x in rows if x[k] is not None]
        res["coverage"][k] = {"coverage": float(np.mean([v[0] for v in ok])), "mean_width": float(np.mean([v[1] for v in ok])), "n_invalid": len(rows) - len(ok)}
    print("RESULT_JSON " + json.dumps(res), flush=True)  # recoverable from the app logs if the client drops
    return res


@app.local_entrypoint()
def main(n: int = 300, out: str = "", targets: str = ""):  # n simulated datasets per target and rate; targets: comma list
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=REPO, check=True).stdout
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("uncommitted tracked changes")
    head = git("rev-parse", "HEAD").strip()
    d9met = json.loads(git("show", f"{D9MET[0]}:{D9MET[1]}"))
    jobs = [(t, r, n, d9met) for t in (targets.split(",") if targets else TARGETS) for r in (0.01, 0.05)]
    res = list(shard.starmap(jobs, return_exceptions=True))
    now = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    m = {"label": "POST HOC (RULE.md); preliminary until Codex agrees", "git": head, "d9_metrics": list(D9MET),
         "shards": [x if isinstance(x, dict) else {"error": repr(x)[:500]} for x in res], "written": now}
    p = Path(out or HERE / "results" / f"cmcov_{now}.json")
    p.parent.mkdir(exist_ok=True)
    with p.open("x") as fh:
        fh.write(json.dumps(m, indent=1))
    print(f"wrote {p}")
