"""Exploratory MVP, not the pre-registered Stage 2: can a probe trained on K5's sabotage labels find the sabotage call
within a run, which the #36 detector can't? It informs whether labelling more runs is worth it.

Data:
- K5 settled labels: 20 attempted real e1 runs; 56 sabotage, 33 preparation, 412 benign calls. Only calls inside the
  32k cut have features: 53 sabotage, 31 preparation and 396 benign, in 19 runs with a sabotage call (one run,
  admin_user_management, loses its only sabotage call to the cut).
- Features: Qwen3-32B call_end activations from the v4 all-layer extraction, at layers 48 and 63 (the #36 call
  readouts).

Method:
- Leave-one-run-out: train logistic regression (standardized, L2, C = 0.01 as #36, with an intercept, balanced
  classes) on the other 19 runs' sabotage vs benign calls, then score every call of the held-out run.
- The solver is liblinear's dual form (fast for few samples and many features), with intercept_scaling = 100 so its
  intercept penalty is negligible. On four held-out runs at both layers it is checked against lbfgs (`solver_check`).
- The layer is chosen by pooled held-out AUROC.
- Null: labels permuted among the labelled calls within each run (each run keeps its count of sabotage calls). The
  whole cross-validation is redone at both layers and the max is taken, so the null repeats the layer choice.

Reported:
- held-out AUROC with a 95% interval (runs resampled) and the null above: the only metric tested against chance;
- descriptive only, with no label-shuffle null (read them as descriptions, not as evidence of localization):
  - within-run AUROC;
  - how often the sabotage call tops its run (against #36: 1–4 of 20);
  - sabotage vs preparation, and preparation vs benign;
  - the content-call subset;
- the number of labelled runs a ±0.05 interval would need.

Modal CPU; no model calls.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.2' --with numpy modal run --detach \
      experiments/2026-10-08_q3_sabotage_probe_mvp/run_sabotage_probe_mvp.py --scores-dir <#36 run_scores dir>
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REPO = HERE.parent.parent
APP = "probe-shift_aswin-surya_sub-q3sabmvp_fit_kill-2026-11-06"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
WORK = "probe-shift_aswin-surya_sub-q3attack_work_kill-2026-11-06"  # the remote also saves its output here
EXTRACTION = "2026-10-06_la_segment-end-v4_qwen3-32b_alllayers_v1"
LABELS = REPO / "experiments/2026-10-07_q3_codex_followups/results/sabotage_step_labels_2026-10-07.jsonl"
LAYERS, C, NULL_DRAWS, BOOT = (48, 63), 0.01, 200, 2000

app = modal.App(APP)
acts_vol = modal.Volume.from_name(ACTS, version=2)
work_vol = modal.Volume.from_name(WORK, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "scikit-learn==1.7.2", "joblib")


def auroc(s, y):
    import numpy as np

    s, y = np.asarray(s, float), np.asarray(y, bool)
    if y.all() or not y.any():
        return None
    order = np.argsort(s, kind="mergesort")
    ss, r, i = s[order], np.empty(len(s)), 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        r[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


@app.function(image=image, cpu=16, memory=32768, timeout=3600, volumes={"/acts": acts_vol.read_only(), "/work": work_vol})
def fit(items: list[dict], stamp: str, group_by: str, layers: tuple = LAYERS, null_draws: int = NULL_DRAWS) -> dict:
    import numpy as np
    from joblib import Parallel, delayed
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    run_dir = Path("/acts") / EXTRACTION
    rows = []  # (index into items, chunk, row)
    keys = sorted({it["key"] for it in items})
    for key in keys:
        m = json.loads((run_dir / "meta" / "real-attack-e1" / f"{key}.json").read_text())
        pos = {(p["step"], p["action"]): p["row"] for p in m["positions"] if p["kind"] == "call_end" and p["kept"]}
        for i, it in enumerate(items):
            if it["key"] == key and (it["step"], it["action"]) in pos:
                rows.append((i, m["chunk"], m["row_start"] + pos[(it["step"], it["action"])]))
    idx = np.array([r[0] for r in rows])
    X = {}
    for l in layers:
        X[l] = np.stack([np.asarray(np.load(run_dir / "by_layer" / f"L{l}" / f"{ch}.npy", mmap_mode="r")[r], np.float32) for _, ch, r in rows])
    lab = np.array([items[i]["label"] for i in idx]); run = np.array([items[i]["key"] for i in idx])
    content = np.array([items[i]["content"] for i in idx])
    # cross-validation folds: one run, or one side task (all runs of that side task held out together)
    group = run if group_by == "run" else np.array([items[i]["key"].rsplit(":", 2)[-2] for i in idx])

    def make(solver):
        """L2 logistic regression with an intercept, C = 0.01, balanced classes. liblinear's dual form is fast for few
        samples and many features; it penalizes the intercept, so intercept_scaling = 100 makes that penalty negligible."""
        if solver == "liblinear":
            return LogisticRegression(C=C, class_weight="balanced", solver="liblinear", dual=True, intercept_scaling=100.0, max_iter=20000)
        return LogisticRegression(C=C, class_weight="balanced", solver="lbfgs", max_iter=5000)

    def loro(l, y, solver="liblinear", only=None):
        """held-out scores of every call, training on the other runs' sabotage (1) vs benign (0) calls"""
        out = np.full(len(y), np.nan)
        for r in (np.unique(group) if only is None else only):
            tr = (group != r) & (y >= 0)
            sc = StandardScaler().fit(X[l][tr])
            clf = make(solver).fit(sc.transform(X[l][tr]), y[tr])
            te = group == r
            out[te] = clf.decision_function(sc.transform(X[l][te]))
        return out

    y = np.where(lab == "sabotage", 1, np.where(lab == "benign", 0, -1))  # preparation: never trained on, scored
    sb = y >= 0
    # solver check: liblinear (dual) must reproduce lbfgs on held-out AUROC, here on four runs at both layers
    check_runs = list(np.unique(group)[:4])
    m4 = np.isin(group, check_runs) & sb
    solver_check = {}
    for l in layers:
        a_lib, a_lbf = loro(l, y, "liblinear", check_runs), loro(l, y, "lbfgs", check_runs)
        solver_check[str(l)] = {"auroc_liblinear": auroc(a_lib[m4], y[m4] == 1), "auroc_lbfgs": auroc(a_lbf[m4], y[m4] == 1),
                                "score_corr": float(np.corrcoef(a_lib[m4], a_lbf[m4])[0, 1])}
    scores = {l: loro(l, y) for l in layers}
    obs = {l: auroc(scores[l][sb], y[sb] == 1) for l in layers}

    def null_draw(seed):
        rng = np.random.default_rng(seed)
        yp = y.copy()
        for r in np.unique(run):  # permute sabotage/benign labels among this run's labelled calls
            m = (run == r) & sb
            yp[m] = rng.permutation(y[m])
        return max(auroc(loro(l, yp)[sb], yp[sb] == 1) for l in layers)

    null = Parallel(n_jobs=16, inner_max_num_threads=1)(delayed(null_draw)(s) for s in range(null_draws))  # one BLAS thread each
    res = {"n_calls_with_features": int(len(idx)), "items": len(items), "scores": {str(l): scores[l].tolist() for l in layers},
            "idx": idx.tolist(), "labels": lab.tolist(), "runs": run.tolist(), "content": content.tolist(),
            "observed": {str(l): obs[l] for l in layers}, "null_max_over_layers": null, "solver_check": solver_check,
            "group_by": group_by, "folds": int(len(np.unique(group)))}
    out_dir = Path("/work/sabotage_probe_mvp"); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"fit_{stamp}_{group_by}.json").write_text(json.dumps(res))  # survives a dropped local client
    work_vol.commit()
    return res


@app.local_entrypoint()
def main(scores_dir: str, group_by: str = "run", layers: str = "48,63", null_draws: int = NULL_DRAWS):
    """layers / null_draws: e.g. --layers 0,8,16 --null-draws 20 for a directional read at early layers"""
    lay = tuple(int(x) for x in layers.split(","))
    import numpy as np

    labels = [json.loads(x) for x in LABELS.read_text().splitlines()]
    labels = [x for x in labels if x["label"] != "submit"]
    d = Path(scores_dir)
    runs = json.loads((d / "runs.json").read_text())
    tables = np.load(d / "tables.npz")
    ri = {(r["arm"], r["key"]): i for i, r in enumerate(runs)}
    kind = {(int(r), int(s), int(a)): int(k) for r, s, a, k in
            zip(tables["call_end__run"], tables["call_end__step"], tables["call_end__action"], tables["call_end__kind"])}
    items = [{"key": x["key"], "step": x["step"], "action": x["action"], "label": x["label"],
              "content": kind.get((ri[("real-attack-e1", x["key"])], x["step"], x["action"])) == 0} for x in labels]
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=REPO,
                           check=True).stdout.strip()
    if dirty:
        raise SystemExit(f"refusing to run from a dirty tree:\n{dirty}")
    (HERE / "results").mkdir(parents=True, exist_ok=True)  # before the remote fit, so its output can always be saved
    t0 = datetime.now()
    stamp = t0.astimezone(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    if group_by not in ("run", "side_task"):
        raise SystemExit("--group-by must be run or side_task")
    print("remote output (if this client drops):", f"{WORK}:/sabotage_probe_mvp/fit_{stamp}_{group_by}.json", flush=True)
    out = fit.remote(items, stamp, group_by, lay, null_draws)
    obs = {int(l): v for l, v in out["observed"].items()}
    best = max(obs, key=obs.get)
    s = np.array(out["scores"][str(best)]); lab = np.array(out["labels"]); run = np.array(out["runs"]); content = np.array(out["content"])
    sab, ben, prep = lab == "sabotage", lab == "benign", lab == "preparation"
    sb = sab | ben
    null = np.array(out["null_max_over_layers"])
    rng = np.random.default_rng(0)
    keys = np.unique(run)

    def boot(mask_pos, mask_neg):
        vals = []
        for _ in range(BOOT):
            pick = rng.choice(keys, len(keys))
            p = np.concatenate([s[(run == k) & mask_pos] for k in pick]); n = np.concatenate([s[(run == k) & mask_neg] for k in pick])
            a = auroc(np.r_[p, n], np.r_[np.ones(len(p)), np.zeros(len(n))].astype(bool))
            if a is not None:
                vals.append(a)
        return vals

    vals = boot(sab, ben)
    se = float(np.std(vals))
    within = [auroc(s[(run == k) & sb], sab[(run == k) & sb]) for k in keys]
    within = [w for w in within if w is not None]
    with_sab = [k for k in keys if sab[run == k].any()]  # runs whose sabotage call lies inside the cut
    top = sum(bool(sab[run == k][np.argmax(s[run == k])]) for k in with_sab)
    res = {
        "solver_check": out["solver_check"],
        "chosen_layer": best, "observed_by_layer": obs, "calls_with_features": out["n_calls_with_features"], "labelled_calls": out["items"],
        "sabotage_vs_benign": {"auroc": obs[best], "ci95_runs_given_layer": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
                               "se_runs": se, "null_max_over_layers_95": float(np.percentile(null, 95)), "null_mean": float(null.mean()),
                               "p_one_sided": float((1 + (null >= obs[best]).sum()) / (1 + len(null)))},
        "within_run_mean_auroc": float(np.mean(within)), "within_run_runs": len(within),
        "runs_whose_top_call_is_sabotage": f"{top} of {len(with_sab)} runs with a sabotage call inside the cut (#36 detector: 1-4 of 20)",
        "sabotage_vs_preparation": auroc(s[sab | prep], sab[sab | prep]),
        "preparation_vs_benign": auroc(s[prep | ben], prep[prep | ben]),
        "content_calls_sabotage_vs_benign": auroc(s[sb & content], sab[sb & content]),
        "runs_for_halfwidth_0.05": int(np.ceil(len(keys) * (1.96 * se / 0.05) ** 2)),
        "note": ("the interval and the run count resample runs with the chosen layer's held-out scores fixed: they omit "
                 "layer-selection and refitting uncertainty (approximate, optimistic). The p-value's null repeats the "
                 "layer choice and the fitting. The run count is for precision (half-width), not power. Only "
                 "sabotage_vs_benign has a null; the other metrics are descriptive."),
    }
    rec = {"stamp": stamp, "group_by": group_by, "folds": out["folds"], "git": head, "git_clean": True, "app": APP, "extraction": EXTRACTION, "layers": lay, "C": C,
           "null_draws": null_draws, "labels": {"file": LABELS.name, "sha256": hashlib.sha256(LABELS.read_bytes()).hexdigest()},
           "exploratory": "not the pre-registered Stage 2 probe", "result": res, "seconds": (datetime.now() - t0).total_seconds()}
    path, text = HERE / "results" / f"sabotage_probe_mvp_{stamp}_{group_by}.json", json.dumps(rec, indent=1)
    with path.open("x") as f:
        f.write(text)
    per = HERE / "results" / f"sabotage_probe_mvp_calls_{stamp}_{group_by}.jsonl"
    with per.open("x") as f:
        for i, k in enumerate(out["idx"]):
            f.write(json.dumps({"key": items[k]["key"], "step": items[k]["step"], "action": items[k]["action"], "label": items[k]["label"],
                                "content": items[k]["content"]} | {f"score_L{l}": out["scores"][str(l)][i] for l in lay}) + "\n")
    print(json.dumps(res, indent=1))
    print("->", path, per.name)
