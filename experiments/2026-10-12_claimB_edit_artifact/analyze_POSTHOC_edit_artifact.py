"""POST HOC (RULE.md in this folder, committed first): does the halves probe flag simulated runs more when the fixed-sim
rules (R1/R2/R5/R6) edited more of their tool outputs, once run length is controlled? Numbers only: the fixed-sim
records are read for key, cell, msg, rule and changed; scores come from local copies of halves' Volume arrays whose
sha256s are checked against the hidden-errors result's recorded inputs. The halves code (not on main) is imported
unchanged from a `git archive` of PIN into --tree; the gate reproduces halves' direct counts before anything else.

  uv run --no-project --python 3.12 --with numpy --with scipy --with scikit-learn --with pyyaml --with 'modal>=1.4.3' python \\
      experiments/2026-10-12_claimB_edit_artifact/analyze_POSTHOC_edit_artifact.py --tree <scratch dir> \\
      --inputs <dir with runs.json, {dev_to_test,test_to_dev}/fixed_vs_e1.npz, POSTHOC_e2/*/fixed_vs_e1.npz> \\
      --records <dir with manifest.json and records/records_*.jsonl.gz> --pack <packed_39f842231a46 runs.L63.json>
"""

import argparse
import gzip
import hashlib
import io
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
PIN = "53da7f48"
SIM, E1, E2 = "sim-attack-fixed", "real-attack-e1", "real-attack-e2"
INPUTS = {"runs.json": "1f55e447610e669e3d27726b8bcec24027798aa5d7fc477898833b684664bc27",
          "dev_to_test/fixed_vs_e1.npz": "c68284a61ba2bcf5c657d5f55509b018d3af751b413dad9eabb43c9007ca8e0c",
          "POSTHOC_e2/dev_to_test/fixed_vs_e1.npz": "9210c7391bc4b8cf89650a52b0e63e75caea034e0047c0f25194b12757246fd6",
          "test_to_dev/fixed_vs_e1.npz": "42d8b2455b7c228e5b50fab108e761ba91e84fdf336665715da2d2fc47a984f3",
          "POSTHOC_e2/test_to_dev/fixed_vs_e1.npz": "4ab399cc937b01098b5f44100642e94c4d9c26c13b58c3729330026094339687"}
GATE_COUNTS = {"dev_to_test": (33, 473, 8, 473), "test_to_dev": (18, 466, 7, 466)}  # sim caught, sims, e2 flagged, e2 runs
GATE_POOLED_CI = (0.019190469651194725, 0.07047052657630772)  # halves direct pooled gap interval
PACK_SHA = "6957bc6019ac4ab9df3fe9712a797b8c41290918fab0a6a601d49d1520a39c28"
RULES, STRATA, RHO_SEED = ("R1", "R2", "R5", "R6"), 5, 20261012


def ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return [float(x) for x in np.percentile(v, [2.5, 97.5])] if len(v) else [None, None]


def clean(o):
    """NaN -> None so the JSON stays standard."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def ok(x):
    return x is not None and np.isfinite(x)


def main():
    p = argparse.ArgumentParser()
    for a_ in ("--tree", "--inputs", "--records", "--pack"):
        p.add_argument(a_, type=Path, required=True)
    a = p.parse_args()
    if not (a.tree / "experiments/2026-10-11_claimB_halves").exists():
        a.tree.mkdir(parents=True, exist_ok=True)
        arc = subprocess.run(["git", "archive", PIN, "experiments", "splits", "src"], capture_output=True, check=True, cwd=ROOT).stdout
        subprocess.run(["tar", "-x", "-C", str(a.tree)], input=arc, check=True)
    sys.path[:0] = [str(a.tree / "experiments/2026-10-11_claimB_halves"), str(a.tree / "src")]
    import analyze_halves as A  # noqa: E402  (thr, SEED, DRAWS)
    import run_halves_modal as R  # noqa: E402  (DIRS)
    for mod, rel in ((A, "analyze_halves.py"), (R, "run_halves_modal.py")):
        want = subprocess.run(["git", "show", f"{PIN}:experiments/2026-10-11_claimB_halves/{rel}"], capture_output=True, check=True, cwd=ROOT).stdout
        if Path(mod.__file__).read_bytes() != want:
            raise SystemExit(f"{rel}: the imported copy differs from {PIN}")

    out = {"label": "POST HOC (RULE.md): edit-count artifact check on the halves test; preliminary until reviewed", "inputs": {}, "pin": PIN}
    raw = {}
    for name, sha in INPUTS.items():
        raw[name] = (a.inputs / name).read_bytes()
        if hashlib.sha256(raw[name]).hexdigest() != sha:
            raise SystemExit(f"{name}: sha256 differs from the recorded input")
        out["inputs"][name] = sha
    meta = json.loads(raw["runs.json"])
    if meta["script_sha256"] != R.SCRIPT_SHA:
        raise SystemExit("runs.json comes from another runner version")
    runs = meta["runs"]
    env, arm, task, cell = (np.array([r[k] for r in runs]) for k in ("environment", "arm", "main_task", "cell"))
    rows = np.array([r["rows"] for r in runs])
    score = {}
    for d in R.DIRS:
        zo, zp = np.load(io.BytesIO(raw[f"{d}/fixed_vs_e1.npz"])), np.load(io.BytesIO(raw[f"POSTHOC_e2/{d}/fixed_vs_e1.npz"]))
        b = int(zo["val_auroc"][:, 0].argmax())
        if int(zp["val_auroc"][:, 0].argmax()) != b:
            raise SystemExit(f"{d}: the rerun picks another sub-configuration")
        score[d] = zp["test_run_score"][b, :, 0]
    held = {d: np.isin(env, R.DIRS[d][1]) for d in R.DIRS}
    rng = np.random.default_rng(A.SEED)  # the same draws as analyze_halves.py and the hidden-errors check
    halves = {d: sorted(set(task[held[d] & np.isin(arm, [SIM, E1, E2])])) for d in R.DIRS}
    draws = {d: [rng.integers(0, len(halves[d]), len(halves[d])) for _ in range(A.DRAWS)] for d in R.DIRS}

    def flags(d, dr=None):
        w = np.zeros(len(runs))
        if dr is None:
            w[held[d]] = 1.0
        else:
            pos = {t: j for j, t in enumerate(halves[d])}
            cnt = np.bincount(dr, minlength=len(halves[d]))
            for i in np.flatnonzero(held[d] & np.isin(arm, [SIM, E1, E2])):
                w[i] = cnt[pos[task[i]]]
        e1 = held[d] & (arm == E1) & (w > 0)
        th = A.thr(np.repeat(score[d][e1], w[e1].astype(int)))
        return score[d] > th, w

    def rate(mask, fl):
        num = sum(float((w * f * mask * held[d]).sum()) for d, (f, w) in fl.items())
        den = sum(float((w * mask * held[d]).sum()) for d, (f, w) in fl.items())
        return num / den if den else float("nan")

    point = {d: flags(d) for d in R.DIRS}
    for d, (sc, ns, ec, ne) in GATE_COUNTS.items():
        f, _ = point[d]
        got = (int((f & held[d] & (arm == SIM)).sum()), int((held[d] & (arm == SIM)).sum()),
               int((f & held[d] & (arm == E2)).sum()), int((held[d] & (arm == E2)).sum()))
        if got != (sc, ns, ec, ne):
            raise SystemExit(f"gate failed for {d}: {got} vs {(sc, ns, ec, ne)}")
    boot = [{d: flags(d, draws[d][b]) for d in R.DIRS} for b in range(A.DRAWS)]
    gci = ci([rate(arm == SIM, fl) - rate(arm == E2, fl) for fl in boot])
    if not np.allclose(gci, GATE_POOLED_CI, rtol=0, atol=1e-12):
        raise SystemExit(f"gate failed: pooled gap interval {gci}")
    out["gate"] = {"pass": True, "counts": GATE_COUNTS, "pooled_gap_ci95": gci}

    # edit counts (metadata fields only) and length
    man = json.loads((a.records / "manifest.json").read_text())
    man = man.get("files", man)
    recs = []
    listed = sorted(k for k in man if k.startswith("records/records_"))
    have = sorted(f"records/{f.name}" for f in (a.records / "records").glob("records_*.jsonl.gz"))
    if listed != have:
        raise SystemExit(f"records: {len(have)} local shards vs {len(listed)} in the manifest")
    for f in sorted((a.records / "records").glob("records_*.jsonl.gz")):
        b_ = f.read_bytes()
        want = man[f"records/{f.name}"]
        want = want["sha256"] if isinstance(want, dict) else want
        if hashlib.sha256(b_).hexdigest() != want:
            raise SystemExit(f"{f.name}: sha256 differs from the fixed-sim manifest")
        recs += [{k: x[k] for k in ("key", "cell", "msg", "rule", "changed")} for x in map(json.loads, gzip.decompress(b_).splitlines())]
    out["inputs"]["fixed_sim_records"] = {"files": len(list((a.records / "records").glob("records_*.jsonl.gz"))), "rows": len(recs)}
    keys_of = {}
    for r in recs:
        keys_of.setdefault(r["cell"], set()).add(r["key"])
    if any(len(v) != 1 for v in keys_of.values()):
        raise SystemExit("records: a cell maps to more than one simulated run")
    edited = {}
    for r in recs:
        if r["changed"]:
            edited.setdefault(r["cell"], {}).setdefault(r["msg"], set()).add(r["rule"])
    pack_raw = a.pack.read_bytes()
    if hashlib.sha256(pack_raw).hexdigest() != PACK_SHA:
        raise SystemExit("pack: sha256 differs")
    out["inputs"]["pack_runs.L63.json"] = PACK_SHA
    L_of = {r["cell"]: sum(1 for q in r["positions"] if isinstance(q, dict) and q.get("kind") == "output_end")
            for r in json.loads(pack_raw) if r["arm"] == SIM}
    sims = arm == SIM
    E = np.full(len(runs), np.nan)
    ER = {rl: np.full(len(runs), np.nan) for rl in RULES}
    L = np.full(len(runs), np.nan)
    for i in np.flatnonzero(sims):
        msgs = edited.get(cell[i], {})
        E[i] = len(msgs)
        for rl in RULES:
            ER[rl][i] = sum(1 for v in msgs.values() if rl in v)
        if cell[i] not in L_of:
            raise SystemExit(f"{cell[i]}: no simulated run in the pack")
        L[i] = L_of[cell[i]]
    out["counts"] = {"sim_runs": int(sims.sum()), "sim_runs_without_any_records": int(sum(cell[i] not in keys_of for i in np.flatnonzero(sims))),
                     "sim_runs_without_changed_records": int(sum(cell[i] not in edited for i in np.flatnonzero(sims))),
                     "E_median": float(np.median(E[sims])), "E_mean": float(np.mean(E[sims])), "L_median": float(np.median(L[sims])),
                     "spearman_E_L": float(spearmanr(E[sims], L[sims]).statistic),
                     "E_rule_runs_with_any": {rl: int((ER[rl][sims] > 0).sum()) for rl in RULES}}

    def group_stats(hi, lo):
        """Pooled sim flag rates in two groups (boolean masks over sim runs), with the e2 runs of the same cells."""
        def e2_of(m):
            return (arm == E2) & np.isin(cell, cell[m])
        res = {}
        for nm, m in (("high", hi), ("low", lo)):
            v = [rate(m, fl) for fl in boot]
            res[nm] = {"n_sim": int(m.sum()), "cells": int(len(set(cell[m]))), "tasks": int(len(set(task[m]))),
                       "flagged_sim": int(sum((point[d][0] & held[d] & m).sum() for d in R.DIRS)),
                       "sim_rate": rate(m, point), "sim_rate_ci95": ci(v),
                       "e2_rate_same_cells": rate(e2_of(m), point), "E_median": float(np.median(E[m])) if m.any() else None,
                       "L_median": float(np.median(L[m])) if m.any() else None, "rows_median": float(np.median(rows[m])) if m.any() else None}
        dv = [rate(hi, fl) - rate(lo, fl) for fl in boot]
        res["diff_high_minus_low"] = {"value": res["high"]["sim_rate"] - res["low"]["sim_rate"], "ci95": ci(dv)}
        return res

    def matched(X):
        """Length-matched difference: within L quintiles, high vs low X at the stratum median; size-weighted mean."""
        edges = np.quantile(L[sims], np.linspace(0, 1, STRATA + 1))
        st = np.clip(np.searchsorted(edges, L, side="right") - 1, 0, STRATA - 1)
        parts = []
        for s in range(STRATA):
            m = sims & (st == s)
            med = np.median(X[m])
            hi, lo = m & (X > med), m & (X <= med)
            parts.append({"stratum": s, "L_range": [float(edges[s]), float(edges[s + 1])], "n": int(m.sum()), "X_median": float(med),
                          "n_high": int(hi.sum()), "n_low": int(lo.sum()), "hi": hi, "lo": lo})
        use = [q for q in parts if q["n_high"] and q["n_low"]]
        for q in parts:
            q |= {"rate_high": rate(q["hi"], point), "rate_low": rate(q["lo"], point), "tasks": int(len(set(task[q["hi"] | q["lo"]]))),
                  "e2_rate_high_cells": rate((arm == E2) & np.isin(cell, cell[q["hi"]]), point),
                  "e2_rate_low_cells": rate((arm == E2) & np.isin(cell, cell[q["lo"]]), point)}
        def comb(fl):  # a draw that empties either group of a used stratum is undefined as a whole
            vals = [(q["n"], rate(q["hi"], fl) - rate(q["lo"], fl)) for q in use]
            if not vals or not all(np.isfinite(v) for _, v in vals):
                return float("nan")
            return sum(n * v for n, v in vals) / sum(n for n, _ in vals)
        dv = [comb(fl) for fl in boot]
        return {"value": comb(point), "ci95": ci(dv), "finite_draws": int(np.isfinite(dv).sum()),
                "strata": [{k: v for k, v in q.items() if k not in ("hi", "lo")} for q in parts],
                "strata_used": len(use)}

    def rhos(X):
        """Per-direction Spearman rho (n-weighted mean over directions): residualised score (on log L) vs X, raw score vs X,
        raw score vs X/L; 95% intervals from task draws (default_rng(RHO_SEED)), the residual fit redone per draw."""
        g = np.random.default_rng(RHO_SEED)
        idx = {d: np.flatnonzero(held[d] & sims) for d in R.DIRS}
        tk = {d: sorted(set(task[idx[d]])) for d in R.DIRS}
        by_t = {d: {t: idx[d][task[idx[d]] == t] for t in tk[d]} for d in R.DIRS}
        def one(sel):
            r = {"residual_vs_X": [], "score_vs_X": [], "score_vs_X_per_L": []}
            for d in R.DIRS:
                ii = sel[d]
                y, x, lg = score[d][ii], X[ii], np.log(L[ii])
                beta = np.polyfit(lg, y, 1)
                res = y - np.polyval(beta, lg)
                n = len(idx[d])  # fixed weights (473, 466)
                r["residual_vs_X"].append((n, spearmanr(res, x).statistic))
                r["score_vs_X"].append((n, spearmanr(y, x).statistic))
                r["score_vs_X_per_L"].append((n, spearmanr(y, x / L[ii]).statistic))
            return {k: sum(n * v for n, v in w) / sum(n for n, _ in w) for k, w in r.items()}
        pt = one(idx)
        bs = []
        for _ in range(2000):
            sel = {d: np.concatenate([by_t[d][tk[d][j]] for j in g.integers(0, len(tk[d]), len(tk[d]))]) for d in R.DIRS}
            bs.append(one(sel))
        return {k: {"value": float(pt[k]), "ci95": ci([b_[k] for b_ in bs]), "finite_draws": int(np.isfinite([b_[k] for b_ in bs]).sum())} for k in pt}

    def split_median(X):
        med = np.median(X[sims])
        return sims & (X > med), sims & (X <= med), float(med)

    hi, lo, med = split_median(E)
    out["primary"] = {"E_median_split_at": med, **group_stats(hi, lo)}
    out["length_matched"] = matched(E)
    out["correlations"] = rhos(E)
    pm, rr = out["length_matched"], out["correlations"]["residual_vs_X"]
    need = [pm["value"], pm["ci95"][0], rr["value"], rr["ci95"][0]]
    if not all(ok(x) for x in need):
        out["reading"] = "mixed / inconclusive (a required estimate or interval is undefined)"
    else:
        sup = pm["value"] > 0 and pm["ci95"][0] > 0 and rr["value"] > 0 and rr["ci95"][0] > 0
        nos = pm["ci95"][0] <= 0 and rr["ci95"][0] <= 0
        rd, rs = out["primary"]["diff_high_minus_low"]["ci95"][0], out["correlations"]["score_vs_X"]["ci95"][0]
        raw_pos = (ok(rd) and rd > 0) or (ok(rs) and rs > 0)
        out["reading"] = ("supports the artifact hypothesis" if sup else
                          ("does not support (the raw association does not survive these length adjustments)" if raw_pos else "does not support")
                          if nos else "mixed / inconclusive")
    by_rule = {}
    for rl in ("R6", "R2", "R1", "R5"):
        X = ER[rl]
        pres, absn = sims & (X > 0), sims & (X == 0)
        ent = {"runs_with_any": int(pres.sum()), "present_vs_absent": group_stats(pres, absn) if pres.any() and absn.any() else None}
        if rl in ("R6", "R2"):
            h, l_, m_ = split_median(X)
            ent["median_split"] = {"split_at": m_, **group_stats(h, l_)} if h.any() and l_.any() else None
            ent["length_matched"] = matched(X)
            ent["correlations"] = rhos(X)
        by_rule[rl] = ent
    out["by_rule_descriptive"] = by_rule
    out |= {"draws": A.DRAWS, "seed_flags": A.SEED, "seed_rho": RHO_SEED,
            "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (HERE / "results").mkdir(exist_ok=True)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"edit_artifact_POSTHOC_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(clean(out), indent=1) + "\n")
    print(json.dumps(clean({k: out[k] for k in ("gate", "counts", "primary", "reading")}), indent=1))
    print("length_matched", json.dumps({k: out["length_matched"][k] for k in ("value", "ci95", "strata_used")}))
    print("correlations", json.dumps(out["correlations"]))
    print("->", dst)


if __name__ == "__main__":
    main()
