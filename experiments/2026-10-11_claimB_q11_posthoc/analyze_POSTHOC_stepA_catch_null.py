"""POST HOC (PLAN.md check 1; 0d's skeptic-panel request, 2026-10-11): step A's own selection-aware null for its catch
at the 1% (and 5%) false-alarm target, from step A's stored grid cells. Local and free: reads the cells (score arrays,
no activations) from the Volume into memory, redoes step A's `summarize` selection (best layer and C on validation
AUROC per outer fold, no convergence filter) on all 101 label columns, then the 1% table's two-way cross-fit catch on
every column. Gate: column 0 reproduces step A's stored summary (test AUROC, AUROC-null p95 and p) or the script stops.

  uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy --with scikit-learn --with scipy --with pyyaml \\
      python experiments/2026-10-11_claimB_q11_posthoc/analyze_POSTHOC_stepA_catch_null.py
"""

import hashlib
import io
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal
import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
FIXED = HERE.parent / "2026-10-09_fixed_sim"
CLAIMB = Path.home() / "probe-shift-claimb"  # the worktree at d9's pinned commit a1821be0
GRID_FILE = CLAIMB / "experiments/2026-10-10_claimB_probe/run_claimB_grid.py"
PINNED, TAG, LEVEL, PACK = "a1821be0", "e402bcc321a8", "level1", "39f842231a46"
SUMMARY = FIXED / "results/claimB_probe_summary_2026-10-10_021700.json"
PUBLISHED = FIXED / "results/tpr_at_fpr_2026-10-10_235259.json"
sys.path[:0] = [str(CLAIMB / "src"), str(GRID_FILE.parent), str(FIXED)]
import analyze_tpr_at_fpr as T  # noqa: E402  (crossfit, TARGETS, SEED)
import run_claimB_grid as G  # noqa: E402  (d9's module, unchanged)
from probe_shift import q1_eval  # noqa: E402  (a1821be0's)

PLANT2_LAYERS = [19, 32, 41, 44, 47, 52, 58]  # step A's summary: planted check 2 on its complete grid
PLAN = {c: (PLANT2_LAYERS, [C for C in G.C_GRID if C != 100.0]) if c == "plant2" else (G.LAYERS, G.C_GRID)
        for c in G.LEVEL_COMPS[LEVEL]}
KEEP = ("val_auroc", "test_run_score", "grad_full", "grad_val", "tol")


def setup(cfg, runs, comparison):
    """G.setup's labels, shuffles and folds (copied; the runs come from the Volume instead of /acts)."""
    q1_eval.DESIGNS["claimB"] = {"comparisons": G.COMPS}
    sc = q1_eval.Scorer(cfg | {"design": "claimB"}, runs, comparison)
    rng = np.random.default_rng(G.SEED + 1)
    cells = sorted({runs[i]["cell"] for i in np.flatnonzero(sc.runs_in)})
    Y = np.zeros((len(runs), 1 + G.N_NULL), np.float32)
    Y[:, 0] = sc.y_run
    for d in range(G.N_NULL):  # swap the two labels within a cell (a planted pair, or a real and a simulated run)
        flip = dict(zip(cells, rng.random(len(cells)) < 0.5))
        Y[:, 1 + d] = [1 - v if (sc.runs_in[i] and flip[runs[i]["cell"]]) else v for i, v in enumerate(sc.y_run)]
    env = np.array([cfg["environment_to_fold"][m["environment"]] for m in runs])
    return sc, Y, env


def read(vol, path):
    raw = b"".join(vol.read_file(path))
    z = np.load(io.BytesIO(raw))
    return {k: z[k] for k in KEEP} | {"folds": z["folds"], "sha256": hashlib.sha256(raw).hexdigest()}


def main():
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=CLAIMB, capture_output=True, text=True, check=True).stdout.strip()
    assert head.startswith(PINNED), f"the claimb worktree is at {head}, not {PINNED}"
    tag = hashlib.sha256(GRID_FILE.read_bytes() + (CLAIMB / "src/probe_shift/torch_probe.py").read_bytes()
                         + (CLAIMB / "src/probe_shift/q1_eval.py").read_bytes()).hexdigest()[:12]
    assert tag == TAG, f"grid code tag {tag}, expected {TAG}"
    folds = json.loads((CLAIMB / "splits/folds_v2.json").read_text())
    cfg = {"packs": {LEVEL: PACK}, "seed": G.SEED, "bootstrap": G.BOOT, "tag": TAG,
           "main_task_to_fold": folds["task_folds"]["main_task_to_fold"], "environment_to_fold": folds["environment_folds"]["environment_to_fold"]}
    vol = modal.Volume.from_name(G.ACTS, version=2)
    runs_raw = b"".join(vol.read_file(f"claimB/{LEVEL}/packed_{PACK}/runs.L{G.LAYERS[0]}.json"))
    runs = json.loads(runs_raw)
    summary = json.loads(SUMMARY.read_text())["levels"][LEVEL]
    published = {r["id"]: r for r in json.loads(PUBLISHED.read_text())["rows"]}
    out = {"label": "POST HOC (0d skeptic-panel check 1, 2026-10-11); step A's own selection-aware catch null; preliminary until reviewed",
           "grid": {"code": head, "tag": TAG, "level": LEVEL, "pack": PACK}, "rows": [],
           "config": {"layers_and_C": {c: [L, Cs] for c, (L, Cs) in PLAN.items()}, "environment_to_fold": cfg["environment_to_fold"],
                      "test_envs": G.TEST_ENVS, "shuffle_seed": G.SEED + 1, "catch_seed": T.SEED, "targets_pct": list(T.TARGETS),
                      "shuffles": G.N_NULL, "selection": "best_layer_and_C on validation AUROC per outer fold, no convergence filter"},
           "inputs": {"runs_json": hashlib.sha256(runs_raw).hexdigest(), SUMMARY.name: hashlib.sha256(SUMMARY.read_bytes()).hexdigest(),
                      PUBLISHED.name: hashlib.sha256(PUBLISHED.read_bytes()).hexdigest(),
                      "folds_v2.json": hashlib.sha256((CLAIMB / "splits/folds_v2.json").read_bytes()).hexdigest()}}
    for comp, (L, Cs) in PLAN.items():
        sc, Y, env = setup(cfg, runs, comp)
        base = f"claimB/{LEVEL}/grid_{PACK}_{TAG}/{comp}"
        with ThreadPoolExecutor(16) as ex:
            R = dict(zip([(l, C) for l in L for C in Cs], ex.map(lambda lc: read(vol, f"{base}/L{lc[0]}_C{lc[1]:g}.npz"), [(l, C) for l in L for C in Cs])))
        fold_ids = R[(L[0], Cs[0])]["folds"].tolist()
        # step A's summarize selection, every column (best_layer_and_C)
        V = np.stack([np.stack([R[(l, C)]["val_auroc"] for C in Cs]) for l in L])  # [layer, C, fold, column]
        c_ix, top = V.argmax(1), V.max(1)
        l_of = top.argmax(0)
        c_of = c_ix[l_of, np.arange(len(fold_ids))[:, None], np.arange(Y.shape[1])[None, :]]
        S = np.full(Y.shape, np.nan)
        unconverged = 0
        for i, f in enumerate(fold_ids):
            m = env == f
            for k in range(Y.shape[1]):
                v = R[(L[l_of[i, k]], Cs[c_of[i, k]])]
                S[m, k] = v["test_run_score"][m, k]
                unconverged += int(max(v["grad_full"][i, k], v["grad_val"][i, k]) > v["tol"])
        r_in = np.flatnonzero(sc.runs_in)
        r = r_in[np.isin([runs[i]["environment"] for i in r_in], G.TEST_ENVS)]
        # gate: column 0 reproduces step A's stored summary
        if not np.isfinite(S[r]).all():
            raise SystemExit(f"{comp}: a compared run has no selected score in some column")
        test, nul = {"auroc": round(float(roc_auc_score(Y[r, 0], S[r, 0])), 4)}, G.null_against(G.col_auc(Y[r], S[r]))
        if not all(np.isfinite([test["auroc"], nul["null_p95"], nul["p"]])):
            raise SystemExit(f"{comp}: non-finite gate statistic")
        ref = summary[comp]["best_layer_and_C"]["test_envs"]
        for a, b, what in ((test["auroc"], ref["test"]["auroc"], "AUROC"), (nul["null_p95"], ref["null"]["null_p95"], "AUROC-null p95"),
                           (nul["p"], ref["null"]["p"], "AUROC-null p")):
            if abs(a - b) > 1e-4:
                raise SystemExit(f"{comp}: {what} {a} does not reproduce step A's summary {b}; stop")
        n_gate = len(r)
        if comp.startswith("plant"):  # each planted copy with its own twin (whole cells), as the published planted rows
            arms = {}
            for i in r:
                arms.setdefault(runs[i]["cell"], []).append(sc.y_run[i])
            r = np.array([i for i in r if sorted(arms[runs[i]["cell"]]) == [0.0, 1.0]])
        task = sc.main[r]
        tasks, inv = np.unique(task, return_inverse=True)
        members = [np.flatnonzero(inv == j) for j in range(len(tasks))]
        perm = np.random.default_rng(T.SEED).permutation(len(tasks))
        stat = {}
        for col in range(Y.shape[1]):
            cf = T.crossfit(Y[r, col].astype(bool), S[r, col], members, np.ones(len(tasks), int), perm)
            if cf is not None:
                stat[col] = {p: (cf[p]["caught"], cf[p]["positives"], cf[p]["false_alarms"], cf[p]["negatives"]) for p in T.TARGETS}
        if 0 not in stat:
            raise SystemExit(f"{comp}: no column-0 catch")
        pid = f"step A|{LEVEL} · run mean|{comp}|test 5"
        row = {"comparison": comp, "scope": "test 5", "runs": len(r), "runs_in_gate": n_gate, "main_tasks": len(tasks), "null_columns": len(stat) - 1,
               "selected_fit_pairs_over_tol": unconverged, "of_selected_fit_pairs": len(fold_ids) * Y.shape[1],  # pair = full + validation fit
               "gate": {"auroc": test["auroc"], "auroc_null_p95": nul["null_p95"], "auroc_null_p": nul["p"], "matches_summary": True},
               "cells_sha256": hashlib.sha256("".join(v["sha256"] for _, v in sorted(R.items())).encode()).hexdigest()}
        for p in T.TARGETS:
            c, npos, fa, nneg = stat[0][p]
            obs = c / npos
            nulls = np.array([v[p][0] / v[p][1] for k, v in stat.items() if k != 0])
            pv = (1 + int((nulls >= obs).sum())) / (1 + len(nulls))
            pub = published.get(pid, {}).get(f"at_{p}pct", {})
            row[f"at_{p}pct"] = {"caught": c, "positives": npos, "false_alarms": fa, "negatives": nneg, "observed": obs,
                                 "null_p95": float(np.percentile(nulls, 95)), "null_median": float(np.median(nulls)), "p": pv,
                                 "above_null": pv < 0.05, "published_refit": [pub.get("caught"), pub.get("positives")]}
        out["rows"].append(row)
        a1, a5 = row["at_1pct"], row["at_5pct"]
        print(f"{comp} [test 5, {len(r)} runs]: gate OK (AUROC {test['auroc']}, null p95 {nul['null_p95']}) | 1%: {a1['caught']}/{a1['positives']} "
              f"= {100 * a1['observed']:.2f}% (published refit {a1['published_refit'][0]}/{a1['published_refit'][1]}), null p95 "
              f"{100 * a1['null_p95']:.2f}%, p {a1['p']:.4f} | 5%: {100 * a5['observed']:.2f}%, null p95 {100 * a5['null_p95']:.2f}%, "
              f"p {a5['p']:.4f} | null cols {row['null_columns']}, selected fit pairs over tol {unconverged}/{row['of_selected_fit_pairs']}", flush=True)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    out |= {"git": git, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    dst = HERE / "results" / f"stepA_catch_null_POSTHOC_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print("->", dst)


if __name__ == "__main__":
    main()
