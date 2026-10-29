"""H2 summary (PLAN.md frozen at ece4fa14 plus the #680 fixes; written before any search result exists). Reads the
search's saved score arrays from the Volume (numbers only) and applies the plan's rules:

- Per configuration, its own selection (best sub-configuration per fold and label column on validation AUROC) gives
  test run scores; its planted checks (real labels) must reach 0.833 (check 1) and 0.910 (check 2), or it is excluded.
- Best found: the same selection across every eligible configuration's sub-configurations, per fold and column, so
  the 100 shuffled columns repeat the whole search (selection-aware null). A0 alone is step A's pipeline.
- Gate: A0's real-label test AUROC (fixed sim vs e1) reproduces step A's 0.8221 within 0.005.
- D = AUROC(best found) - AUROC(A0), fixed sim vs e1, test runs, 2,000 main-task draws (probes fixed); upper bound <
  0.03: "within 0.03"; lower bound > 0.03: "richer separate more"; otherwise inconclusive (H3 carries both).
- The best found must beat its null (p < 0.05). Catch at 1% (and 5%) is its own conclusion: best found and A0, the
  1% table's two-way cross-fit, each with its selection-aware null, and their paired difference over 2,000 task draws.
- Real vs real: e2 vs e1 best found and its null. delta = best(sim vs e1) - best(e2 vs e1) with its margin from H2's own
  shuffle columns by H3 Step 0's method (column k of each comparison is one world; X80 = the 80th percentile of the
  one-sided 95% bound, rounded up to 0.005, with a Monte Carlo interval). Calibration note: H2's shuffles draw separate
  coins per comparison, so these worlds are Step 0's "independent coins" kind, slightly wider than its same-coin refit
  worlds (conservative for an equivalence claim).
- Planted checks come from the one-column rerun (`<comparison>_k1/`, made by c3b745f6's spawn_planted): only the
  real-label column, since no rule uses the planted nulls. The real comparisons come from the 101-column jobs (c13aff24).
  Each file is checked against the identity of the version that made it.
- Readings are gated: a decision is read only if the best found beats its null and passes its own planted checks; if no
  richer configuration passes its planted checks, the reading is "no richer probe was shown to work".

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy --with scikit-learn --with scipy \\
      --with pyyaml python experiments/2026-10-11_claimB_h2_probe_best_found/analyze_h2_search.py
"""

import hashlib
import io
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal
import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
CLAIMB = Path.home() / "probe-shift-claimb"
sys.path[:0] = [str(CLAIMB / "src"), str(CLAIMB / "experiments/2026-10-10_claimB_probe"), str(HERE),
                str(Path.home() / "probe-shift-sims/experiments/2026-10-09_fixed_sim")]
import analyze_tpr_at_fpr as T  # noqa: E402  (crossfit, TARGETS, SEED: the 1% table's two-way cross-fit)
import run_claimB_grid as G  # noqa: E402
import run_h2_search_modal as H  # noqa: E402  (seed_of, FAMILIES, subconfigs, paths, K)

sys.path.insert(0, str(HERE.parent / "2026-10-11_claimB_h3_fix_to_plateau"))
from analyze_power_margin import bound  # noqa: E402  (H3 Step 0's one-sided bound over main-task draws)
from probe_shift import q1_eval  # noqa: E402
from probe_shift.q1_eval import Scorer  # noqa: E402

COMPS = ("fixed_vs_e1", "e2_vs_e1", "plant1", "plant2")
# Pre-stated contingency (1), agreed by 0d: L2 at full width could not be fit within its cap. Each large-C fit
# (fp64 LBFGS, 25,600 dims, 101 columns, refit chain to 105,000 evaluations) takes about 50 minutes, so its jobs were
# stopped at 12:40 PT before the 13:24 cap with no fold saved. H2 is read over the other 11 configurations.
SKIPPED = {"L2": "could not be fit within its cap (full-width fp64 LBFGS at large C); stopped 12:40 PT; contingency (1)"}
PLANT_MIN = {"plant1": 0.833, "plant2": 0.910}
STEP_A, GATE_TOL, MARGIN_D, DRAWS, SEED = 0.8221, 0.005, 0.03, 2000, 20261011


def labels(cfg, runs, comparison):
    """The search's label columns, rebuilt exactly (H.setup's rule, runs from the Volume)."""
    q1_eval.DESIGNS["claimB"] = {"comparisons": G.COMPS}
    sc = Scorer(cfg | {"design": "claimB"}, runs, comparison)
    unit = [runs[i]["cell"] if comparison.startswith("plant") else runs[i]["main_task"] for i in range(len(runs))]
    units = sorted({unit[i] for i in np.flatnonzero(sc.runs_in)})
    Y = np.zeros((len(runs), H.K), np.float32)
    Y[:, 0] = sc.y_run
    for k in range(1, H.K):
        rng = np.random.default_rng(H.seed_of("h2-shuffle", comparison, k))
        flip = dict(zip(units, rng.random(len(units)) < 0.5))
        Y[:, k] = [1 - v if (sc.runs_in[i] and flip[unit[i]]) else v for i, v in enumerate(sc.y_run)]
    return sc, Y > 0.5


def select(V, S, env, folds):
    """V [subs, folds, K] validation AUROCs, S [subs, runs, K] test run scores: per fold and column the best
    sub-configuration's scores on that fold's runs (ties: the first). Non-finite validation AUROCs are an error."""
    if not np.isfinite(V).all():
        raise SystemExit(f"{int((~np.isfinite(V)).sum())} validation AUROCs are not finite")
    out = np.full(S.shape[1:], np.nan)
    best = V.argmax(axis=0)  # [folds, K]
    for i, f in enumerate(folds):
        fr = np.flatnonzero(env == f)
        out[fr] = np.take_along_axis(S[:, fr, :], best[i][None, None, :].repeat(len(fr), 1), 0)[0]
    return out


def col_aucs(y, S):
    return np.array([roc_auc_score(y[:, k], S[:, k]) for k in range(S.shape[1])])


def boot_paired(y, a, b, task, rng):
    """AUROC(a) - AUROC(b) on the same runs over main-task draws (probes fixed)."""
    tasks, inv = np.unique(task, return_inverse=True)
    members = [np.flatnonzero(inv == j) for j in range(len(tasks))]
    d = []
    for _ in range(DRAWS):
        i = np.concatenate([members[j] for j in rng.integers(0, len(tasks), len(tasks))])
        if y[i].all() or not y[i].any():
            continue
        d.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), len(d)


def catch_diff(y, b, a, task, pct, rng):
    """Catch share at pct, best found minus A0, over main-task draws that redo the two-way cross-fit (as the 1% table)."""
    tasks, inv = np.unique(task, return_inverse=True)
    members = [np.flatnonzero(inv == j) for j in range(len(tasks))]
    d = []
    for _ in range(DRAWS):
        counts = np.bincount(rng.integers(0, len(tasks), len(tasks)), minlength=len(tasks))
        perm = rng.permutation(np.flatnonzero(counts))
        cb, ca = T.crossfit(y, b, members, counts, perm), T.crossfit(y, a, members, counts, perm)
        if cb and ca:
            d.append(cb[pct]["caught"] / cb[pct]["positives"] - ca[pct]["caught"] / ca[pct]["positives"])
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], len(d)


def catch(y, s, task):
    tasks, inv = np.unique(task, return_inverse=True)
    members = [np.flatnonzero(inv == j) for j in range(len(tasks))]
    return T.crossfit(y, s, members, np.ones(len(tasks), int), np.random.default_rng(T.SEED).permutation(len(tasks)))


def main():
    vol = modal.Volume.from_name(H.VOL, version=2)
    read = lambda p: b"".join(vol.read_file(p))  # noqa: E731
    folds_file = json.loads((CLAIMB / "splits/folds_v2.json").read_text())
    cfg = {"packs": {"level1": "39f842231a46"}, "seed": 0, "bootstrap": 1000, "tag": "h2",
           "main_task_to_fold": folds_file["task_folds"]["main_task_to_fold"], "environment_to_fold": folds_file["environment_folds"]["environment_to_fold"]}
    runs_raw = read(f"{H.PACK_A}/runs.L63.json")
    runs = json.loads(runs_raw)
    env = np.array([cfg["environment_to_fold"][m["environment"]] for m in runs])
    folds = sorted({cfg["environment_to_fold"][e] for e in H.TEST_ENVS})
    names = [n for n in sum(H.FAMILIES.values(), []) if n not in SKIPPED]
    out = {"label": "H2 best-found search summary (preliminary until reviewed)", "configs": {}, "rows": {}, "skipped_configurations": SKIPPED,
           "inputs": {"runs_json": hashlib.sha256(runs_raw).hexdigest()}}
    data, per_cfg, idents = {}, {}, set()
    d9 = [CLAIMB / "src/probe_shift/torch_probe.py", CLAIMB / "src/probe_shift/q1_eval.py",
          CLAIMB / "experiments/2026-10-10_claimB_probe/run_claimB_grid.py", CLAIMB / "experiments/2026-10-10_claimB_probe/run_claimB_pack.py"]
    old = subprocess.run(["git", "show", "c13aff24:experiments/2026-10-11_claimB_h2_probe_best_found/run_h2_search_modal.py"],
                         capture_output=True, check=True, cwd=HERE).stdout
    ident_v1 = H.ident_of(cfg, hashlib.sha256(old + b"".join(f.read_bytes() for f in d9)).hexdigest())  # the 101-column jobs
    ident_v2 = H.ident_of(cfg, H.code_sha())  # the one-column planted rerun
    for comp in COMPS:
        planted = comp.startswith("plant")
        kk, folder, want_ident = (1, f"{comp}_k1", ident_v2) if planted else (H.K, comp, ident_v1)
        sc, Y = labels(cfg, runs, comp)
        Y = Y[:, :kk]
        test = sc.runs_in & np.isin(env, folds)
        data[comp] = {"Y": Y[test], "task": sc.main[test], "test": test}
        for name in names:
            raw = read(f"{H.OUT}/{folder}/{name}.npz")
            z = np.load(io.BytesIO(raw))
            idents.add(str(z["ident"]))
            if str(z["ident"]) != want_ident or json.loads(str(z["subs"])) != H.subconfigs(name) or list(z["folds"]) != folds \
                    or z["val_auroc"].shape != (len(H.subconfigs(name)), len(folds), kk) or z["test_run_score"].shape != (len(H.subconfigs(name)), len(runs), kk):
                raise SystemExit(f"{folder}/{name}.npz does not match its version, folds, sub-configurations or shapes")
            S = select(z["val_auroc"], z["test_run_score"], env, folds)[test]
            per_cfg[(comp, name)] = {"V": z["val_auroc"], "S": z["test_run_score"], "sel": S, "converged": z["converged"]}
            out["inputs"][f"{comp}/{name}.npz"] = hashlib.sha256(raw).hexdigest()
    if idents != {ident_v1, ident_v2}:
        raise SystemExit("search files come from unexpected code/fold/input identities")
    # per-configuration planted checks (real labels) and test AUROCs
    eligible = []
    for name in names:
        row = {c: float(roc_auc_score(data[c]["Y"][:, 0], per_cfg[(c, name)]["sel"][:, 0])) for c in COMPS}
        row["planted_pass"] = all(row[c] >= PLANT_MIN[c] for c in PLANT_MIN)
        row["unconverged_fits"] = {c: int((~per_cfg[(c, name)]["converged"]).sum()) for c in COMPS}
        out["configs"][name] = row
        if row["planted_pass"]:
            eligible.append(name)
    out["eligible"] = eligible
    a0 = out["configs"]["A0"]["fixed_vs_e1"]
    out["gate"] = {"A0_auroc": a0, "step_a": STEP_A, "pass": abs(a0 - STEP_A) <= GATE_TOL}
    if not out["gate"]["pass"]:
        raise SystemExit(f"A0 gate failed: {a0:.4f} vs {STEP_A}")
    rng = np.random.default_rng(SEED)
    if not eligible:  # every configuration, A0 included, failed its planted checks: save the verdict and stop
        out["decision"] = "no configuration passed its planted checks: nothing read"
        stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
        with open(HERE / "results" / f"h2_summary_{stamp}.json", "x") as fh:
            fh.write(json.dumps(out, indent=1))
        raise SystemExit(out["decision"])
    best = {}
    for comp in COMPS:  # the search pools exactly the eligible configurations (A0 too, when eligible)
        V = np.concatenate([per_cfg[(comp, n)]["V"] for n in eligible])
        S = np.concatenate([per_cfg[(comp, n)]["S"] for n in eligible])
        best[comp] = select(V, S, env, folds)[data[comp]["test"]]
    for comp in COMPS:
        y, b, a = data[comp]["Y"], best[comp], per_cfg[(comp, "A0")]["sel"]
        ab, aa = col_aucs(y, b), col_aucs(y, a)
        row = {"best_found_auroc": float(ab[0]), "A0_auroc": float(aa[0])}
        if len(ab) > 1:  # the real comparisons carry the 100-shuffle null; the planted rerun has the real column only
            row |= {"best_null_p95": float(np.percentile(ab[1:], 95)), "best_p": float((1 + (ab[1:] >= ab[0]).sum()) / H.K),
                    "A0_null_p95": float(np.percentile(aa[1:], 95)), "A0_p": float((1 + (aa[1:] >= aa[0]).sum()) / H.K)}
        if comp in ("fixed_vs_e1", "e2_vs_e1"):
            lo, hi, n = boot_paired(y[:, 0], b[:, 0], a[:, 0], data[comp]["task"], rng)
            row["D"] = {"value": float(ab[0] - aa[0]), "ci95": [lo, hi], "draws": n}
            for pct in T.TARGETS:
                cb, ca = catch(y[:, 0], b[:, 0], data[comp]["task"]), catch(y[:, 0], a[:, 0], data[comp]["task"])
                nb = [catch(y[:, k], b[:, k], data[comp]["task"]) for k in range(1, H.K)]
                na = [catch(y[:, k], a[:, k], data[comp]["task"]) for k in range(1, H.K)]
                share = lambda c: c[pct]["caught"] / c[pct]["positives"]  # noqa: E731
                diff_ci, n_d = catch_diff(y[:, 0], b[:, 0], a[:, 0], data[comp]["task"], pct, rng)
                row[f"catch_{pct}pct"] = {"best_minus_A0": share(cb) - share(ca), "best_minus_A0_ci95": diff_ci, "draws": n_d,
                                          "best": share(cb), "best_fpr": cb[pct]["false_alarms"] / cb[pct]["negatives"],
                                          "best_p": float((1 + sum(share(c) >= share(cb) for c in nb if c)) / (1 + sum(1 for c in nb if c))),
                                          "A0": share(ca), "A0_fpr": ca[pct]["false_alarms"] / ca[pct]["negatives"],
                                          "A0_p": float((1 + sum(share(c) >= share(ca) for c in na if c)) / (1 + sum(1 for c in na if c)))}
        out["rows"][comp] = row
    # delta = best(sim vs e1) - best(e2 vs e1), its one-sided 95% bound, and its margin from H2's own shuffle worlds
    # (H3 Step 0's method: column k of both comparisons is one world; X80 = the 80th percentile of the bound)
    f_, e_ = data["fixed_vs_e1"], data["e2_vs_e1"]
    tasks = np.unique(np.concatenate([f_["task"], e_["task"]]))
    cnt = np.zeros((DRAWS, len(tasks)))
    np.add.at(cnt, (np.repeat(np.arange(DRAWS), len(tasks)), np.random.default_rng(SEED).integers(0, len(tasks), DRAWS * len(tasks))), 1)
    Wf, We = cnt[:, np.searchsorted(tasks, f_["task"])], cnt[:, np.searchsorted(tasks, e_["task"])]
    def delta(sf, se_):
        ub = np.array([bound(f_["Y"][:, k], sf[:, k], Wf, e_["Y"][:, k], se_[:, k], We)[0] for k in range(H.K)])
        mc = np.random.default_rng(SEED + 1)
        x80 = float(np.percentile(ub[1:], 80))
        boots = [np.percentile(ub[1:][mc.integers(0, H.K - 1, H.K - 1)], 80) for _ in range(DRAWS)]
        margin = float(np.ceil(x80 / 0.005) * 0.005)
        return {"value": float(col_aucs(f_["Y"][:, :1], sf[:, :1])[0] - col_aucs(e_["Y"][:, :1], se_[:, :1])[0]), "upper_bound_95": float(ub[0]),
                "X80": x80, "X80_mc_ci95": [float(x) for x in np.percentile(boots, [2.5, 97.5])], "margin_rounded_up": margin,
                "within_margin": bool(ub[0] < margin)}
    out["delta"] = {"best_found": delta(best["fixed_vs_e1"], best["e2_vs_e1"]),
                    "A0": delta(per_cfg[("fixed_vs_e1", "A0")]["sel"], per_cfg[("e2_vs_e1", "A0")]["sel"])}
    d = out["rows"]["fixed_vs_e1"]["D"]
    best_planted = {c: float(roc_auc_score(data[c]["Y"][:, 0], best[c][:, 0])) for c in PLANT_MIN}
    out["best_found_planted"] = best_planted
    out["best_beats_null"] = out["rows"]["fixed_vs_e1"]["best_p"] < 0.05
    out["no_richer_config_passed"] = not [n for n in eligible if n != "A0"]
    if out["no_richer_config_passed"]:
        out["decision"] = "no richer probe was shown to work"
    elif not all(best_planted[c] >= PLANT_MIN[c] for c in PLANT_MIN):
        out["decision"] = "search broken (the selected best found fails its planted checks): nothing read"
    elif not out["best_beats_null"]:
        out["decision"] = "best found not above its null: nothing read"
    else:
        out["decision"] = ("within 0.03" if d["ci95"][1] < MARGIN_D else "richer separate more" if d["ci95"][0] > MARGIN_D else "inconclusive")
    readable = out["decision"] in ("within 0.03", "richer separate more", "inconclusive")
    out["h3_endpoint"] = {"within 0.03": "step A's pipeline", "richer separate more": "add the best-found pipeline",
                          "inconclusive": "carry both pipelines; each its own margin; H3 equivalence needs both bounds below"}.get(
        out["decision"], "step A's pipeline; H2 gave no reading (" + out["decision"] + ")")
    if not readable:
        out["delta"]["best_found"]["within_margin"] = None  # no reading from an invalid search
    out["real_vs_real_above_null"] = out["rows"]["e2_vs_e1"]["best_found_auroc"] > out["rows"]["e2_vs_e1"]["best_null_p95"]
    out["selected_unconverged_note"] = "step A had no convergence filter; per-configuration unconverged counts are in configs[*].unconverged_fits"
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    out |= {"git": git, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "search_idents": {"v1_101_columns": ident_v1, "v2_planted_k1": ident_v2}}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    dst = HERE / "results" / f"h2_summary_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("gate", "eligible", "decision", "h3_endpoint", "best_beats_null", "best_found_planted",
                                         "real_vs_real_above_null", "delta")}, indent=1))
    for c, r in out["rows"].items():
        print(c, json.dumps(r))
    print("->", dst)


if __name__ == "__main__":
    main()
