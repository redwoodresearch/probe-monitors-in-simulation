"""Is the halves result run length? Applies RULE_length.md (committed before any number, 18f6e764) to item (A)'s stored
scores: A. the frozen strict alarm within coarse length strata; B. out-of-fold length-only logistic baselines vs
baseline + detector score. EXPLORATORY; numbers only; nothing sealed is read.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy --with scikit-learn \\
      python experiments/2026-10-11_claimB_halves_confirm/analyze_length_control.py
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import analyze_power_confirm as P  # noqa: E402  (load, thr, DIRS, arms, SHA-checked inputs)

SEED, DRAWS, FOLDS = 20261012, 2000, 5
ARM_TOK = {P.SIM: ("sim-attack", None), P.E1: ("real-attack-e1", None), P.E2: ("real-attack-e2", None)}


def tokens_for(cell, arm):
    """exclusions_v4 token counts by (cell, arm)."""
    d = json.loads((HERE.parents[1] / "splits/exclusions_v4.json").read_text())
    tok = {}
    for s in d["samples"]:
        tok[(s["cell"], s["arm"])] = s.get("tokens")
    return np.array([tok.get((c, ARM_TOK[a][0]), np.nan) if a in ARM_TOK else np.nan for c, a in zip(cell, arm)], dtype=float)


def strat_gap(sf, ef, sb, eb, w):
    """Weighted mean over strata of (sim rate - e2 rate); strata missing an arm dropped, weights renormalized."""
    num = den = 0.0
    used = 0
    for k in range(len(w)):
        ms, me = sb == k, eb == k
        if ms.sum() == 0 or me.sum() == 0:
            continue
        num += w[k] * (sf[ms].mean() - ef[me].mean())
        den += w[k]
        used += 1
    return (num / den if den > 0 else np.nan), used


def oof(X, y, groups):
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=FOLDS).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        pred[te] = m.decision_function(sc.transform(X[te]))
    return pred


def boot_auc_diff(y, pa, pb, groups, rng):
    tasks = np.array(sorted(set(groups)))
    idx = {t: np.flatnonzero(groups == t) for t in tasks}
    out = []
    for _ in range(DRAWS):
        pick = np.concatenate([idx[t] for t in tasks[rng.integers(len(tasks), size=len(tasks))]])
        yy = y[pick]
        if yy.min() == yy.max():
            continue
        out.append(roc_auc_score(yy, pb[pick]) - roc_auc_score(yy, pa[pick]))
    return [float(x) for x in np.percentile(out, [2.5, 97.5])], len(out)


def runs_rows():
    """n_outputs per run: the `rows` field of item (A)'s runs.json (sha256-checked, same order as the score arrays)."""
    import modal
    vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
    raw = b"".join(vol.read_file(f"{P.OUT}/runs.json"))
    if hashlib.sha256(raw).hexdigest() != P.SHA["runs.json"]:
        raise SystemExit("runs.json sha256 differs")
    return np.array([r["rows"] for r in json.loads(raw)["runs"]], dtype=float)


def main():
    env, arm, task, cell, score = P.load()
    nout = runs_rows()
    ntok = tokens_for(cell, arm)
    rng = np.random.default_rng(SEED)
    out = {"label": "EXPLORATORY: is the halves result run length? (RULE_length.md @ 18f6e764)", "A": {}, "B": {}, "descriptive": {}}
    pooledA = {}
    for d, (_, held_envs) in P.DIRS.items():
        m = np.isin(env, held_envs)
        s, a, t, no, nt = score[d][m], arm[m], task[m], nout[m], ntok[m]
        th = P.thr(s[a == P.E1])
        gate = (int((s[a == P.SIM] > th).sum()), int((a == P.SIM).sum()), int((s[a == P.E2] > th).sum()), int((a == P.E2).sum()))
        if gate != P.EXPECT[d]:
            raise SystemExit(f"{d}: gate failed {gate} vs {P.EXPECT[d]}")
        flag = (s > th).astype(float)
        out["descriptive"][d] = {arm_: {"n": int((a == arm_).sum()), "mean_n_outputs": float(no[a == arm_].mean()),
                                        "median_n_outputs": float(np.median(no[a == arm_])), "mean_n_tokens": float(np.nanmean(nt[a == arm_])),
                                        "median_n_tokens": float(np.nanmedian(nt[a == arm_])), "missing_tokens": int(np.isnan(nt[a == arm_]).sum())}
                                 for arm_ in (P.SIM, P.E1, P.E2)}
        tasks = np.array(sorted(set(t)))
        tix = {x: np.flatnonzero(t == x) for x in tasks}
        draws = [np.concatenate([tix[x] for x in tasks[rng.integers(len(tasks), size=len(tasks))]]) for _ in range(DRAWS)]
        out["A"][d] = {}
        for meas, L in (("n_tokens", nt), ("n_outputs", no)):
            ok = np.isfinite(L) & (L > 0)
            se = (a == P.SIM) & ok
            ee = (a == P.E2) & ok
            base = np.concatenate([L[se], L[ee]])
            cuts = np.quantile(base, [0.2, 0.4, 0.6, 0.8])
            b = np.searchsorted(cuts, L, side="right")
            w = np.bincount(np.concatenate([b[se], b[ee]]), minlength=5) / (se.sum() + ee.sum())
            adj, used = strat_gap(flag[se], flag[ee], b[se], b[ee], w)
            both = [k for k in range(5) if (b[se] == k).any() and (b[ee] == k).any()]
            cs, ce = se & np.isin(b, both), ee & np.isin(b, both)  # the population the adjusted gap covers
            sa, ea = a == P.SIM, a == P.E2
            raw_all = flag[sa].mean() - flag[ea].mean()
            raw_cov = flag[cs].mean() - flag[ce].mean()
            ga, gall, gcov, drops = [], [], [], 0
            for dr in draws:
                ds, de = dr[se[dr]], dr[ee[dr]]
                g, u = strat_gap(flag[ds], flag[de], b[ds], b[de], w)
                drops += 5 - u
                ga.append(g)
                gall.append(flag[dr[sa[dr]]].mean() - flag[dr[ea[dr]]].mean())
                gcov.append(flag[dr[cs[dr]]].mean() - flag[dr[ce[dr]]].mean())
            ga = np.array(ga)
            valid = int(np.isfinite(ga).sum())
            if valid < 0.95 * DRAWS:
                lo = hi = float("nan")
                reading = f"undefined ({valid} of {DRAWS} draws valid)"
            else:
                lo, hi = (float(x) for x in np.percentile(ga[np.isfinite(ga)], [2.5, 97.5]))
                reading = ("persists after this coarse length adjustment (this proxy)" if lo > 0 else
                           "reversed" if hi < 0 else "adjusted gap not established")
            rlo, rhi = (float(x) for x in np.percentile(gall, [2.5, 97.5]))
            clo, chi = (float(x) for x in np.percentile(gcov, [2.5, 97.5]))
            per_stratum = [{"stratum": k, "weight": float(w[k]), "sim_n": int((b[se] == k).sum()), "sim_flagged": int(flag[se][b[se] == k].sum()),
                            "e2_n": int((b[ee] == k).sum()), "e2_flagged": int(flag[ee][b[ee] == k].sum())} for k in range(5)]
            out["A"][d][meas] = {"cuts": [float(c) for c in cuts], "excluded_runs": int((~ok & np.isin(a, [P.SIM, P.E2])).sum()),
                                 "raw_gap_all": float(raw_all), "raw_gap_all_ci95": [rlo, rhi], "raw_gap_covered": float(raw_cov),
                                 "raw_gap_covered_ci95": [clo, chi], "adjusted_gap": float(adj), "adjusted_ci95": [lo, hi],
                                 "valid_draws": valid, "strata_used": used,
                                 "stratum_drops_over_draws": int(drops), "per_stratum": per_stratum, "reading": reading}
            pooledA.setdefault(meas, []).append((adj, list(ga), int(se.sum() + ee.sum())))
        # B: sim vs e1 (and the e2 vs e1 yardstick), out-of-fold
        out["B"][d] = {}
        for name, pos in (("fixed_vs_e1", P.SIM), ("e2_vs_e1", P.E2)):
            k = np.isin(a, [pos, P.E1]) & np.isfinite(nt) & (nt > 0) & (no > 0)
            y = (a[k] == pos).astype(int)
            lt, lo_ = np.log(nt[k]), np.log(no[k])
            XL = np.c_[lt, lo_]
            XF = np.c_[lt, lo_, lt ** 2, lo_ ** 2, lt * lo_]
            S = s[k][:, None]
            g = t[k]
            pL, pF = oof(XL, y, g), oof(XF, y, g)
            pLS, pFS = oof(np.c_[XL, S], y, g), oof(np.c_[XF, S], y, g)
            r = {"n": int(k.sum()), "excluded": int((np.isin(a, [pos, P.E1]) & ~k).sum()),
                 "auroc_score_alone": float(roc_auc_score(y, s[k])), "auroc_L": float(roc_auc_score(y, pL)), "auroc_L_flex": float(roc_auc_score(y, pF)),
                 "auroc_L_plus_S": float(roc_auc_score(y, pLS)), "auroc_L_flex_plus_S": float(roc_auc_score(y, pFS))}
            if name == "fixed_vs_e1":
                ci, nd = boot_auc_diff(y, pL, pLS, g, rng)
                cif, ndf = boot_auc_diff(y, pF, pFS, g, rng)
                r |= {"delta": r["auroc_L_plus_S"] - r["auroc_L"], "delta_ci95": ci, "delta_flex": r["auroc_L_flex_plus_S"] - r["auroc_L_flex"],
                      "delta_flex_ci95": cif, "valid_draws": [nd, ndf],
                      "reading": ("the detector improves over these length-only baselines" if ci[0] > 0 and cif[0] > 0 else "not shown")}
            out["B"][d][name] = r
    out["A"]["pooled"] = {}
    for meas, parts in pooledA.items():
        ws = np.array([p[2] for p in parts], dtype=float)
        adj = float(sum(p[0] * w for p, w in zip(parts, ws)) / ws.sum())
        G = np.array([[p[1][i] for p in parts] for i in range(DRAWS)])
        gp = (G * ws).sum(1) / ws.sum()
        valid = int(np.isfinite(gp).sum())
        if valid < 0.95 * DRAWS:
            lo = hi = float("nan")
            reading = f"undefined ({valid} of {DRAWS} paired draws valid)"
        else:
            lo, hi = (float(x) for x in np.percentile(gp[np.isfinite(gp)], [2.5, 97.5]))
            reading = ("persists after this coarse length adjustment (this proxy)" if lo > 0 else
                       "reversed" if hi < 0 else "adjusted gap not established")
        out["A"]["pooled"][meas] = {"adjusted_gap": adj, "adjusted_ci95": [lo, hi], "valid_paired_draws": valid, "reading": reading}
    dd = [out["B"][d]["fixed_vs_e1"] for d in P.DIRS]
    out["B"]["limitation"] = "B's intervals resample fixed out-of-fold predictions; they omit model-fitting uncertainty"
    out["B"]["pooled_descriptive"] = {"delta_run_weighted": float(sum(x["delta"] * x["n"] for x in dd) / sum(x["n"] for x in dd)),
                                      "delta_flex_run_weighted": float(sum(x["delta_flex"] * x["n"] for x in dd) / sum(x["n"] for x in dd))}
    out |= {"inputs": P.SHA, "rule": "RULE_length.md @ 18f6e764",
            "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"length_control_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    for d in P.DIRS:
        for meas in ("n_tokens", "n_outputs"):
            x = out["A"][d][meas]
            print(f"A {d} {meas}: raw(all) {x['raw_gap_all']:.4f} [{x['raw_gap_all_ci95'][0]:.4f}, {x['raw_gap_all_ci95'][1]:.4f}] "
                  f"raw(covered) {x['raw_gap_covered']:.4f} [{x['raw_gap_covered_ci95'][0]:.4f}, {x['raw_gap_covered_ci95'][1]:.4f}] adjusted {x['adjusted_gap']:.4f} "
                  f"[{x['adjusted_ci95'][0]:.4f}, {x['adjusted_ci95'][1]:.4f}] valid {x['valid_draws']} strata {x['strata_used']} drops {x['stratum_drops_over_draws']} -> {x['reading']}")
    for meas, x in out["A"]["pooled"].items():
        print(f"A pooled {meas}: adjusted {x['adjusted_gap']:.4f} [{x['adjusted_ci95'][0]:.4f}, {x['adjusted_ci95'][1]:.4f}] -> {x['reading']}")
    for d in P.DIRS:
        for name, r in out["B"][d].items():
            print(f"B {d} {name}: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}))
    print("B pooled (descriptive):", json.dumps(out["B"]["pooled_descriptive"]), "|", out["B"]["limitation"])
    print("descriptive:", json.dumps(out["descriptive"]))
    print("->", dst)


if __name__ == "__main__":
    main()
