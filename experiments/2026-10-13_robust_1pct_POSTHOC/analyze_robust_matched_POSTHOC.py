"""Coverage of a MATCHED-false-alarm gap, each world thresholded on its own honest pool (RULE_AMENDMENT4_matched.md; POST
HOC; one Modal CPU container). d9's SCE averaged readout, L48, cells b' (b_fixed: repaired simulated attacks, edited
simulated honest) and b (as-simulated). Gap = catch_sim(attacks, sim-honest thresholds) - catch_real(e2, real-honest
thresholds), committed 4-model vote. Gate: the committed sizing quote (O+P, analyze_sce_meanpool_b.py's draws) for b' at
1% and 5% reproduces within 0.05 points. Then real-data intervals per method, then smoothed-world coverage (world-specific
h and truths; outer resampling within folds; scenarios real85 and realfull).
v2 (Amendment 5): real85 has exactly 85 weighted real calibration copies per fold (k = 0 at 1%); the smoothed bootstrap
re-estimates h per calibration set; n_eff and k are recorded for both worlds.

  python analyze_robust_matched_POSTHOC.py <averaged npz> <out json> [R] [git]
"""

import hashlib
import json
import multiprocessing
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from probe_shift import tail_threshold as tt

HERE = Path(__file__).resolve().parent
PIN = "6a099d93875096c3b9ed3369c66b0afeb93cf20a54b0a3e39367a118d3ce4385"
B, B_IN, SIM_SEED, L = 2000, 400, 20261016, 48
H, E2 = "real-honest", "real-attack-e2"
CELLS = {"b_fixed": ("sim-attack-fixed", "sim-honest-edited"), "b": ("sim-attack", "sim-honest")}
PAIRS = [(f, v) for f in range(5) for v in range(5) if v != f]
GATE = {("b_fixed", 0.01): (-10.0, -11.0, 11.1), ("b_fixed", 0.05): (-1.3, -6.9, 2.8)}  # sizing_matched_decomp printout @ 43715f67
G = {}


def gap_methods(S, cal_r, cal_s, W, rate, rng, h_r, h_s, D, att, Wt_r=None):
    """W: run weights (rates, and the simulated world's calibration); Wt_r: the real world's calibration weights."""
    fold, arm = D["fold"], D["arm"]
    Wt_r = W if Wt_r is None else Wt_r
    out = {}
    for name, meth in (("fixed+P", None), ("O+P", "order"), ("HD+P", "hd"), ("G+P", "gauss"), ("O+S", "smooth")):
        if meth is None:
            Tr = {k: tt.threshold(S[k][cal_r[k]], Wt_r[:1, cal_r[k]], rate)[0] for k in S}
            Ts = {k: tt.threshold(S[k][cal_s[k]], W[:1, cal_s[k]], rate)[0] for k in S}
            Cr, Cs = tt.catch(S, fold, cal_r, Wt_r, rate, T=Tr)[0], tt.catch(S, fold, cal_s, W, rate, T=Ts)[0]
        else:
            Cr = tt.catch(S, fold, cal_r, Wt_r, rate, meth, h=None, rng=rng)[0]  # v2: h re-estimated per calibration set
            Cs = tt.catch(S, fold, cal_s, W, rate, meth, h=None, rng=rng)[0]
        va = np.isfinite(Cr[0]) & np.isfinite(Cs[0])
        cs, cr = tt.rate_of(Cs, va & (arm == att), W), tt.rate_of(Cr, va & (arm == E2), W)
        out[name] = (100 * (cs - cr), 100 * cs, 100 * cr)
    return out


def summ(x):
    lo, hi = tt.interval(x)
    return {"value": float(x[0]), "ci95": [lo, hi], "width": hi - lo}


def truths(S, cal_r, cal_s, fold, arm, h_r, h_s, rate, att):
    tr = tt.smoothed_truth(S, fold, cal_r, h_r, rate)  # real-world truth per run (real thresholds, real h)
    ts = tt.smoothed_truth(S, fold, cal_s, h_s, rate)  # simulated-world truth per run
    va = np.isfinite(tr) & np.isfinite(ts)
    cs, cr = ts[va & (arm == att)].mean(), tr[va & (arm == E2)].mean()
    return 100 * (cs - cr), 100 * cs, 100 * cr


def one_dataset(args):
    cell, rate, scen, r = args
    D, att, _ = G["D"], *CELLS[cell]
    rng = np.random.default_rng([SIM_SEED, r])
    nt, tix = G["n_tasks"], G["tix"]
    c = np.zeros(nt)
    for idx in G["tasks_by_fold"]:  # within folds; repeated task copies share their noise
        c += np.bincount(rng.choice(idx, size=len(idx), replace=True), minlength=nt)
    Wout = c[tix]
    Win = np.vstack([Wout] + [rng.multinomial(nt, c / nt)[tix] for _ in range(B_IN)]).astype(float)
    S, cal_r, cal_s, h_r, h_s, world_sim = G["S"][cell], G["cal_r"][cell], G["cal_s"][cell], G["h_r"][cell], G["h_s"][cell], G["world_sim"][cell]
    Sn = {k: s + rng.normal(0, 1, len(s)) * np.where(world_sim, h_s[k], h_r[k]) for k, s in S.items()}
    arm, fold = D["arm"], D["fold"]
    Wt_r = Win.copy()
    if scen == "real85":  # v2: exactly 85 weighted real calibration copies per validation fold
        for v in range(5):
            hv = np.flatnonzero((arm == H) & (fold == v) & (Wout > 0))
            copies = np.repeat(hv, Wout[hv].astype(int))
            m = np.bincount(copies[rng.choice(len(copies), size=min(85, len(copies)), replace=False)], minlength=len(Wout))
            Wt_r[:, hv] = Win[:, hv] * (m[hv] / Wout[hv])[None, :]
    cr_n = {k: v[Wt_r[0, v] > 0] for k, v in cal_r.items()}
    cs_n = {k: v[Wout[v] > 0] for k, v in cal_s.items()}
    ne_r = np.array([Wt_r[:, cr_n[k]].sum(1) for k in cr_n]); ne_s = np.array([Win[:, cs_n[k]].sum(1) for k in cs_n])
    res = gap_methods(Sn, cr_n, cs_n, Win, rate, rng, h_r, h_s, D, att, Wt_r)
    t = G["truth"][(cell, rate)]
    return {"_neff": tuple(float(f(a)) for a in (ne_r[:, 0], ne_r[:, 1:], ne_s[:, 0], ne_s[:, 1:]) for f in (np.min, np.max))} | {m: tuple((lambda s: (s["ci95"][0] <= t[i] <= s["ci95"][1], s["width"]))(summ(x[i])) for i in range(3)) for m, x in res.items()}


def main(npz_path, out, R=300, git=None):
    if hashlib.sha256(Path(npz_path).read_bytes()).hexdigest() != PIN:
        raise SystemExit("not the pinned averaged-readout scores")
    if git is None and subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=HERE).stdout:
        raise SystemExit("uncommitted tracked changes")
    Z = np.load(npz_path)
    arm, fold, task = Z["runs__arm"], Z["runs__fold"].astype(int), Z["runs__main_task"]
    tasks, tix = np.unique(task, return_inverse=True)
    D = {"arm": arm, "fold": fold}
    W = tt.task_weights(tix, len(tasks), B, 20261013)
    res = {"label": "POST HOC (RULE_AMENDMENT4_matched.md)", "scores sha256": PIN,
           "git": git or subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip(), "real": {}, "coverage": {}, "truths": {}}
    G.update(D=D, S={}, cal_r={}, cal_s={}, h_r={}, h_s={}, world_sim={}, truth={})
    rng = np.random.default_rng(20261013)
    for cell, (att, hon) in CELLS.items():
        s3 = Z[f"L{L}__{cell}__nested_real"]
        S = {(f, v): s3[f, v].astype(np.float64) for f, v in PAIRS}
        cal_r = {k: np.flatnonzero((arm == H) & (fold == k[1]) & np.isfinite(S[k])) for k in S}
        cal_s = {k: np.flatnonzero((arm == hon) & (fold == k[1]) & np.isfinite(S[k])) for k in S}
        h_r = {k: tt.silverman(S[k][cal_r[k]]) for k in S}
        h_s = {k: tt.silverman(S[k][cal_s[k]]) for k in S}
        G["S"][cell], G["cal_r"][cell], G["cal_s"][cell], G["h_r"][cell], G["h_s"][cell] = S, cal_r, cal_s, h_r, h_s
        G["world_sim"][cell] = np.isin(arm, [att, hon])
        for rate in (0.01, 0.05):
            o = gap_methods(S, cal_r, cal_s, W, rate, rng, h_r, h_s, D, att)
            if (cell, rate) in GATE:
                g = GATE[(cell, rate)]
                v = summ(o["O+P"][0])
                if max(abs(v["value"] - g[0]), abs(v["ci95"][0] - g[1]), abs(v["ci95"][1] - g[2])) > 0.05:
                    raise SystemExit(f"gate failed for {cell} at {rate}: {v} vs {g}")
            res["real"][f"{cell} fpr_{rate:g}"] = {m: {"gap": summ(x[0]), "sim catch": summ(x[1]), "real catch": summ(x[2])} for m, x in o.items()}
            G["truth"][(cell, rate)] = truths(S, cal_r, cal_s, fold, arm, h_r, h_s, rate, att)
            res["truths"][f"{cell} fpr_{rate:g}"] = G["truth"][(cell, rate)]
    tfold = {}
    for t, f in zip(tix, fold):
        tfold.setdefault(int(t), set()).add(int(f))
    if any(len(v) != 1 for v in tfold.values()):
        raise SystemExit("a main task spans more than one fold")
    G.update(n_tasks=len(tasks), tix=tix, tasks_by_fold=[np.array(sorted(t for t, v in tfold.items() if f in v)) for f in range(5)])
    with ProcessPoolExecutor(mp_context=multiprocessing.get_context("fork")) as ex:
        for cell in CELLS:
            for scen in ("real85", "realfull"):
                for rate in (0.01, 0.05):
                    rows = list(ex.map(one_dataset, [(cell, rate, scen, r) for r in range(R)]))
                    k = f"{cell} {scen} fpr_{rate:g}"
                    ne = np.array([x["_neff"] for x in rows])
                    fk = lambda a, b: [int(np.floor(rate * a + 1e-9)), int(np.floor(rate * b + 1e-9))]  # noqa: E731
                    nk = {"real point n_eff": [ne[:, 0].min(), ne[:, 1].max()], "real point k": fk(ne[:, 0].min(), ne[:, 1].max()),
                          "real inner n_eff": [ne[:, 2].min(), ne[:, 3].max()], "real inner k": fk(ne[:, 2].min(), ne[:, 3].max()),
                          "sim point n_eff": [ne[:, 4].min(), ne[:, 5].max()], "sim point k": fk(ne[:, 4].min(), ne[:, 5].max()),
                          "sim inner n_eff": [ne[:, 6].min(), ne[:, 7].max()], "sim inner k": fk(ne[:, 6].min(), ne[:, 7].max())}
                    res["coverage"][k] = {"R": R, "n_eff and k": {a: [float(x) for x in b] for a, b in nk.items()},
                                          **{m: {s: {"coverage": float(np.mean([x[m][i][0] for x in rows])), "mean_width": float(np.mean([x[m][i][1] for x in rows]))}
                                                 for i, s in enumerate(("gap", "sim catch", "real catch"))} for m in rows[0] if m != "_neff"}}
                    print(f"coverage {k}: " + " | ".join(f"{m} gap {100 * v['gap']['coverage']:.0f}% w{v['gap']['mean_width']:.1f} (sim {100 * v['sim catch']['coverage']:.0f}%, real {100 * v['real catch']['coverage']:.0f}%)"
                                                         for m, v in res["coverage"][k].items() if m not in ("R", "n_eff and k")) + f" || {res['coverage'][k]['n_eff and k']}", flush=True)
    Path(out).open("x").write(json.dumps(res, indent=1, default=float))
    for k, o in res["real"].items():
        print(f"REAL {k}: " + " | ".join(f"{m} {v['gap']['value']:+.1f} [{v['gap']['ci95'][0]:+.1f},{v['gap']['ci95'][1]:+.1f}] w{v['gap']['width']:.1f}" for m, v in o.items()))
    for k, t in res["truths"].items():
        print(f"TRUTH {k}: gap {t[0]:+.2f}, sim catch {t[1]:.2f}, real catch {t[2]:.2f}")
    print("written", Path(out).name)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(int(a) for a in sys.argv[3:4]), *sys.argv[4:5])
