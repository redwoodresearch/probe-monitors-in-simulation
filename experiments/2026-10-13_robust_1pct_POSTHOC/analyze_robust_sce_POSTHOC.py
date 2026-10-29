"""More robust 1% intervals on d9's SCE targets (RULE.md; POST HOC; local, $0). Targets (i) averaged (b') L48 D and gap,
(ii) last-token (a) L63 gap; v2 (Amendment 3) adds single-arm catch, detector-specific truths, within-fold
resampling, the gate first, and drops the band. Gate: the committed values and intervals to 4 decimals (O+P, analyze_sce_meanpool_b.py's
draws). Then every method on the real data, then simulated coverage in the smoothed world (R datasets, inner draws).

  PYTHONPATH=src uv run --no-project --python 3.12 --with numpy --with scipy python \\
      experiments/2026-10-13_robust_1pct_POSTHOC/analyze_robust_sce_POSTHOC.py <averaged npz> <last-token npz> <committed metrics json> <out json> [R]
"""

import hashlib
import json
import multiprocessing
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from probe_shift import tail_threshold as tt

HERE = Path(__file__).resolve().parent
PIN = {"averaged_npz": "6a099d93875096c3b9ed3369c66b0afeb93cf20a54b0a3e39367a118d3ce4385", "last_position_npz": "c8f81042965ec1305dad34b35c38702423f6617679f1c02b367cc36c761c94cd",
       "metrics": "010e8f66febffc40cf931f92ff765831540e948cd08379c564e41d2087f9cfc2"}
B, SEED, B_IN, SIM_SEED = 2000, 20261013, 400, 20261015
H, E1, E2, SA, SAF = "real-honest", "real-attack-e1", "real-attack-e2", "sim-attack", "sim-attack-fixed"
CELLS = ("a", "b", "c", "b_fixed", "c_fixed")
PAIRS = [(f, v) for f in range(5) for v in range(5) if v != f]
TARGETS = {"i_D": ("averaged", 48, "D"), "i_gap": ("averaged", 48, "gap"), "ii_gap": ("last_token", 63, "gap"),
           "i_catch": ("averaged", 48, "catch"), "ii_catch": ("last_token", 63, "catch")}  # catch: the second real run, single arm
VOTE = ("O+P", "HD+P", "G+P", "O+S")
G = {}  # data shared with worker processes


def load(avg_path, lt_path):
    Z, Y = np.load(avg_path), np.load(lt_path)
    arm, fold, task, cell = (Z[f"runs__{k}"] for k in ("arm", "fold", "main_task", "cell"))
    fold = fold.astype(int)
    where = {(a, k): i for i, (a, k) in enumerate(zip(Y["runs__arm"], Y["runs__key"]))}
    idx = np.array([where[(a, k)] for a, k in zip(arm, Z["runs__key"])])
    nested = {("averaged", L, c): Z[f"L{L}__{c}__nested_real"] for L in (48, 63) for c in CELLS}
    nested |= {("last_token", L, c): Y[f"L{L}__{c}__nested_real"][:, :, idx] for L in (48, 63) for c in CELLS}

    def kept(s3):
        ok = np.ones(len(arm), bool)
        for f, v in PAIRS:
            ok &= ~((fold == f) & ~np.isfinite(s3[f, v]))
        bad = set(cell[np.isin(arm, [E1, E2, SA, SAF]) & ~ok])
        return np.where(np.isin(arm, [E1, E2, SA, SAF]), ~np.isin(cell, list(bad)), ok)

    keep = {L: np.logical_and.reduce([kept(nested[(ro, L, c)]) for ro in ("averaged", "last_token") for c in CELLS]) for L in (48, 63)}
    tasks, tix = np.unique(task, return_inverse=True)
    return arm, fold, tix, len(tasks), nested, keep


def cells_of(tg):
    ro, L, kind = TARGETS[tg]
    return [(ro, L, "b_fixed"), (ro, L, "a")] if kind == "D" else [(ro, L, "b_fixed" if ro == "averaged" else "a")]


def stat(tg, C, W, arm, keep):
    ro, L, kind = TARGETS[tg]
    k = keep[L]
    if kind == "D":
        return tt.rate_of(C[(ro, L, "b_fixed")], k & (arm == E2), W) - tt.rate_of(C[(ro, L, "a")], k & (arm == E2), W)
    c = cells_of(tg)[0]
    if kind == "catch":
        return tt.rate_of(C[c], k & (arm == E2), W)
    own = SAF if c[2] == "b_fixed" else SA
    return tt.rate_of(C[c], k & (arm == own), W) - tt.rate_of(C[c], k & (arm == E2), W)


def S_cal(s3, arm, fold):
    S = {(f, v): s3[f, v] for f, v in PAIRS}
    cal = {(f, v): np.flatnonzero((arm == H) & (fold == v) & np.isfinite(s3[f, v])) for f, v in PAIRS}
    return S, cal


def meanz(S, cal, fold, arm, W, rate, pool):  # pooled mean-z detector: thresholds from honest runs of one/other/all folds
    n, Bw = len(fold), W.shape[0]
    zb, cnt = np.zeros((Bw, n)), np.zeros(n)
    for (f, v), s in S.items():
        hv, w = cal[(f, v)], W[:, cal[(f, v)]]
        mu = (w * s[hv]).sum(1) / w.sum(1)
        sd = np.sqrt((w * (s[hv][None, :] - mu[:, None]) ** 2).sum(1) / w.sum(1))
        te = (fold == f) & np.isfinite(s)
        zb[:, te] += (s[te][None, :] - mu[:, None]) / sd[:, None]
        cnt[te] += 1
    zb /= cnt
    hit = np.zeros((Bw, n))
    for f in range(5):
        pf = {"one": (fold == (f + 1) % 5), "other": (fold != f), "all": np.ones(n, bool)}[pool]
        hv = np.flatnonzero((arm == H) & pf & np.isfinite(zb[0]))
        o = np.argsort(-zb[:, hv], 1, kind="mergesort")
        zs, ws = np.take_along_axis(zb[:, hv], o, 1), np.take_along_axis(W[:, hv], o, 1)
        cw = np.cumsum(ws, 1)
        T = zs[np.arange(Bw), np.argmax(cw > np.floor(rate * cw[:, -1:] + 1e-9), 1)]
        te = fold == f
        hit[:, te] = zb[:, te] > T[:, None]
    return hit


def methods(tg, Sd, W, rate, rng, h, only=None):
    """{method: [rows] statistic} on data Sd = {cell: (S, cal)} with weights W."""
    arm, fold, keep = G["arm"], G["fold"], G["keep"]
    out = {}
    for name, meth in (("O+P", "order"), ("HD+P", "hd"), ("G+P", "gauss"), ("O+S", "smooth")):
        if only and name != only:
            continue
        C = {c: tt.catch(S, fold, cal, W, rate, meth, h=h[c], rng=rng)[0] for c, (S, cal) in Sd.items()}
        out[name] = stat(tg, C, W, arm, keep)
    if only:
        return out
    for pool in ("one", "other", "all"):
        out[f"meanz_{pool}+P"] = stat(tg, {c: meanz(S, cal, fold, arm, W, rate, pool) for c, (S, cal) in Sd.items()}, W, arm, keep)
    return out


def meanz_truth(S, cal, h, rate, pool):  # the mean-z detector's smoothed-world true catch per run (Amendment 3)
    fold, arm = G["fold"], G["arm"]
    n = len(fold)
    zs, t2, cnt = np.zeros(n), np.zeros(n), np.zeros(n)
    for (f, v), s in S.items():
        x, hh = s[cal[(f, v)]], h[(f, v)]
        mu, sd = x.mean(), np.sqrt(x.var() + hh ** 2)
        te = (fold == f) & np.isfinite(s)
        zs[te] += (s[te] - mu) / sd
        t2[te] += (hh / sd) ** 2
        cnt[te] += 1
    zb, tau = zs / cnt, np.sqrt(t2) / cnt
    out = np.full(n, np.nan)
    for f in range(5):
        pf = {"one": (fold == (f + 1) % 5), "other": (fold != f), "all": np.ones(n, bool)}[pool]
        hv = (arm == H) & pf & np.isfinite(zb)
        F = lambda t: norm.cdf((t - zb[hv]) / tau[hv]).mean() - (1 - rate)  # noqa: E731
        T = brentq(F, zb[hv].min() - 10 * tau[hv].max(), zb[hv].max() + 10 * tau[hv].max())
        te = fold == f
        out[te] = norm.cdf((zb[te] - T) / tau[te])
    return out


def summarize(out):
    res = {}
    for k, x in out.items():
        if k == "O+B":
            res[k] = {"value": float(x[2]), "ci95": [float(x[0]), float(x[1])], "width": float(x[1] - x[0])}
        else:
            lo, hi = tt.interval(x)
            res[k] = {"value": float(x[0]), "ci95": [lo, hi], "width": hi - lo}
    return res


def one_dataset(args):  # coverage: one simulated dataset in the smoothed world
    tg, rate, r = args
    rng = np.random.default_rng([SIM_SEED, r])
    nt, tix = G["n_tasks"], G["tix"]
    c = np.zeros(nt)
    for f in range(5):  # Amendment 3: main tasks resampled within each fold
        tf = G["tasks_of_fold"][f]
        c += np.bincount(tf[rng.integers(0, len(tf), len(tf))], minlength=nt)
    Wout = c[tix]
    Win = np.vstack([Wout] + [rng.multinomial(nt, c / nt)[tix] for _ in range(B_IN)]).astype(float)
    Sd = {}
    for cc in cells_of(tg):
        S, cal = G["Sd"][cc]
        Sn = {k: s + rng.normal(0, G["h"][cc][k], len(s)) for k, s in S.items()}
        cal_n = {k: v[Wout[v] > 0] for k, v in cal.items()}
        Sd[cc] = (Sn, cal_n)
    res = summarize(methods(tg, Sd, Win, rate, rng, G["h"]))
    tru = lambda k: G["truth"][(tg, rate, "vote" if k in VOTE else k.split("+")[0])]  # noqa: E731  detector-specific truth
    return {k: (v["ci95"][0] <= tru(k) <= v["ci95"][1], v["width"]) for k, v in res.items()}


def main(avg_path, lt_path, met_path, out, R=300, git=None):  # git: the commit, verified clean by the Modal caller
    for p, k in ((avg_path, "averaged_npz"), (lt_path, "last_position_npz"), (met_path, "metrics")):
        if hashlib.sha256(Path(p).read_bytes()).hexdigest() != PIN[k]:
            raise SystemExit(f"{p}: not the pinned {k} file")
    if git is None and subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=HERE).stdout:
        raise SystemExit("uncommitted tracked changes")
    arm, fold, tix, n_tasks, nested, keep = load(avg_path, lt_path)
    met = json.loads(Path(met_path).read_text())["by_layer"]
    W = tt.task_weights(tix, n_tasks, B, SEED)
    Sd = {cc: S_cal(nested[cc], arm, fold) for tg in TARGETS for cc in cells_of(tg)}
    h = {cc: {k: tt.silverman(S[k][cal[k]]) for k in S} for cc, (S, cal) in Sd.items()}
    tf = np.array([np.bincount(fold[tix == t]).argmax() for t in range(n_tasks)])  # each main task's fold (task-grouped folds)
    assert all(len(set(fold[tix == t])) == 1 for t in range(n_tasks)), "a main task spans folds"
    G.update(arm=arm, fold=fold, keep=keep, tix=tix, n_tasks=n_tasks, Sd=Sd, h=h, tasks_of_fold={f: np.flatnonzero(tf == f) for f in range(5)})
    res = {"label": "POST HOC diagnostic (RULE.md)", "pins": PIN, "git": git or subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip(),
           "real": {}, "coverage": {}, "tail_check": {}}
    rng = np.random.default_rng(SEED)
    for tg, (ro, L, kind) in TARGETS.items():
        for rate in (0.01, 0.05):
            com = met[str(L)]["readouts"][ro]["b_fixed" if ro == "averaged" else "a"][f"fpr_{rate:g}"]
            cv = (com["criterion"]["D_e2_cell_minus_a"] if kind == "D" else com["sce_reported_minus_real"] if kind == "gap"
                  else com["catch"][E2])
            o = summarize(methods(tg, {cc: Sd[cc] for cc in cells_of(tg)}, W, rate, rng, h, only="O+P"))["O+P"]  # gate first
            if [round(o["value"], 4)] + [round(x, 4) for x in o["ci95"]] != [cv["value"]] + cv["ci95"]:
                raise SystemExit(f"gate failed: {tg} at {rate}: {o} vs {cv}")
            res["real"][f"{tg} fpr_{rate:g}"] = summarize(methods(tg, {cc: Sd[cc] for cc in cells_of(tg)}, W, rate, rng, h))
    # Gaussian tail check: achieved false alarms of the G threshold on out-of-sample (test-fold) honest runs, and QQ correlation on the upper 25%
    for cc, (S, cal) in Sd.items():
        fa, qq = [], []
        for (f, v), s in S.items():
            x = s[cal[(f, v)]]
            t = tt.threshold(x, np.ones((1, len(x))), 0.01, "gauss")[0]
            test_h = s[(arm == H) & (fold == f) & np.isfinite(s)]
            fa.append(float((test_h > t).mean()))
            up = np.sort(x)[int(0.75 * len(x)):]
            qq.append(float(np.corrcoef(up, norm.ppf((np.arange(int(0.75 * len(x)), len(x)) + 0.5) / len(x)))[0, 1]))
        res["tail_check"]["/".join(map(str, cc))] = {"gauss_1pct_achieved_FA_mean": float(np.mean(fa)), "range": [min(fa), max(fa)], "upper25_qq_corr_mean": float(np.mean(qq))}
    # truths in the smoothed world, then coverage
    for tg in TARGETS:
        for rate in (0.01, 0.05):
            Ct = {cc: tt.smoothed_truth(*Sd[cc][:1], fold, Sd[cc][1], h[cc], rate)[None, :] for cc in cells_of(tg)}
            G.setdefault("truth", {})[(tg, rate, "vote")] = float(stat(tg, Ct, np.ones((1, len(arm))), arm, keep)[0])
            for pool in ("one", "other", "all"):
                Cz = {cc: meanz_truth(*Sd[cc], h[cc], rate, pool)[None, :] for cc in cells_of(tg)}
                G["truth"][(tg, rate, f"meanz_{pool}")] = float(stat(tg, Cz, np.ones((1, len(arm))), arm, keep)[0])
    with ProcessPoolExecutor(mp_context=multiprocessing.get_context("fork")) as ex:  # workers inherit G
        for tg in TARGETS:
            for rate in (0.01, 0.05):
                rows = list(ex.map(one_dataset, [(tg, rate, r) for r in range(R)]))
                res["coverage"][f"{tg} fpr_{rate:g}"] = {"truths": {k[2]: v for k, v in G["truth"].items() if k[:2] == (tg, rate)}, "R": R,
                                                         **{m: {"coverage": float(np.mean([x[m][0] for x in rows])), "mean_width": float(np.mean([x[m][1] for x in rows]))} for m in rows[0]}}
                print(f"coverage {tg} {rate:g}: " + " | ".join(f"{m} {100 * v['coverage']:.0f}% w{100 * v['mean_width']:.1f}" for m, v in res["coverage"][f"{tg} fpr_{rate:g}"].items() if m not in ("truths", "R")), flush=True)
    Path(out).open("x").write(json.dumps(res, indent=1, default=float))
    for k, o in res["real"].items():
        print(f"REAL {k}: " + " | ".join(f"{m} {100 * v['value']:+.1f} [{100 * v['ci95'][0]:+.1f},{100 * v['ci95'][1]:+.1f}] w{100 * v['width']:.1f}" for m, v in o.items()))
    for k, v in res["tail_check"].items():
        print(f"TAIL {k}: {v}")
    print("written", Path(out).name)


if __name__ == "__main__":
    main(*sys.argv[1:5], *(int(a) for a in sys.argv[5:6]), *sys.argv[6:7])
