"""Q4 Amendment 1 (RULE.md): on-policy Qwen3.5-27B on FULL transcripts (claim C's variant) at early reply positions, one
Modal CPU container, no fitting:
- (a) real-trained: D3's full-transcript records claimck26_5eb808288f50 read by D3's `columns` (copied unchanged from
  ../2026-10-13_rr_2x2_POSTHOC/run_rr_early.py), plus the whole-run column. Gate: run 1 catch at 1% equals D3's full result.
- (c) mixed: cell (c)'s grid onpolreply_c_bc89eeb2869c/swapped_L1 (claimc_cache_v1). The selection recomputed from its records
  must equal stage 4's choices; each fold's held-out output_end rows are rescored with the saved column-0 weights, sorted
  per run by (turn, call index); gate: run means equal the stored column-0 scores within 1e-6.
Both on D3's fixed cohort, then analyze_onpolicy_cells.stats at j = 3, 21 and the whole run.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 --with numpy modal run --detach \\
      experiments/2026-10-13_q4_why_weaker_POSTHOC/run_q4full_onpolicy_positions.py --labels <claimc_labels_reduced.json> --stage4 <stage4_cell_c json>
"""

import hashlib
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-q4-full-pos_kill-2026-11-13"
SCORE_VOL = "probe-shift_aswin-surya_sub-q3onpol_score_kill-2026-11-08"
CACHE, A_REC, C_REC = "claimc_cache_v1", "claimck26_5eb808288f50", "onpolreply_c_bc89eeb2869c/swapped_L1"
D3_FULL = ("de458d7a", "experiments/2026-10-08_q3_onpolicy_figures/results/d3_position_POSTHOC_2026-10-11_112843.json")
LABELS_SHA256 = "d00e88773fc2a71bef51e1ff0ccc2559932f5c3a306a8f5604c5248e797dc995"
T196 = "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"
MODS = ["experiments/2026-10-13_onpolicy_reply/analyze_onpolicy_cells.py", "experiments/2026-10-08_q3_onpolicy_figures/analyze_retrain.py"]
LAYERS, C_GRID, PREFIX_J, POS = [39, 51, 63], [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], (1, 2, 3, 5, 8, 13, 21), (3, 21)
KIND, WORLDS, GATE_D3, GATE_C = "output_end", ("run1", "run2", "sim", "honest_a", "honest_b"), 1e-9, 1e-6

app = modal.App(APP)
score_vol = modal.Volume.from_name(SCORE_VOL, version=2)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy", "scipy")
         .add_local_python_source("probe_shift").add_local_file(REPO / T196, f"/root/{T196}"))
for m in MODS:
    image = image.add_local_file(REPO / m, f"/root/{m}")


@app.function(image=image, cpu=8, memory=65536, timeout=1800, retries=0, volumes={"/score": score_vol.read_only()})
def analyze(lab: dict, c_choices: list, d3_run1: dict) -> dict:
    import sys

    import numpy as np

    sys.path.insert(0, "/root/experiments/2026-10-13_onpolicy_reply")
    import analyze_onpolicy_cells as A

    f196 = A.frozen_folds(json.loads(A.T196.read_text()))
    tasks = sorted(f196)
    ti = {t: i for i, t in enumerate(tasks)}
    W = [A.task_weights([f196[t] for t in tasks], b) for b in range(1, A.B + 1)]
    cands = [(l, C) for C in C_GRID for l in LAYERS]

    def records(prefix, cols0_only=False):  # D3's columns, first half: claim C's selection over converged candidates
        R = {c: dict(np.load(Path("/score") / prefix / f"L{c[0]}_{KIND}_C{c[1]:g}.npz")) for c in cands}
        r0 = R[cands[0]]
        if any(not all(np.array_equal(r[k], r0[k]) for k in ("on_runs", "Y", "on_fold", "train")) for r in R.values()):
            raise ValueError(f"{prefix}: records disagree")
        ok_f, ok_v = (np.stack([R[c][f"grad_{n}"] <= R[c]["tol"] for c in cands]) for n in ("full", "val"))
        share = float(1 - np.concatenate([ok_f.ravel(), ok_v.ravel()]).mean())
        V = np.stack([np.where(ok_f[i] & ok_v[i] & np.isfinite(R[c]["val_auroc"]), R[c]["val_auroc"], -np.inf) for i, c in enumerate(cands)])
        pick, usable = V.argmax(0), np.isfinite(V.max(0))
        u = usable[:, 0] if cols0_only else usable
        if share > A.MAX_FAIL or not u.all():
            raise ValueError(f"{prefix}: unconverged share {share:.4f}, unusable fold-columns {int((~u).sum())}")
        complete = np.all(np.stack([np.isfinite(R[c]["on_score"]).all(1) for c in cands]), 0)
        return R, r0, pick, complete, share

    def selected(M, r0, pick, complete, K):
        fold, n = r0["on_fold"], len(r0["on_runs"])
        Z = np.full((n, K), np.nan)
        for f in range(5):
            for j in range(K):
                m = (fold == f) & complete
                Z[m, j] = M[cands[pick[f, j]]][m, j]
        return Z

    # (a): D3's columns unchanged (selection, prefix-j columns from the sidecars, per-output reproduction check) + the whole run
    R, r0, pick, complete, share_a = records(A_REC)
    n, K = len(r0["on_runs"]), r0["Y"].shape[1]
    side = {c: dict(np.load(Path("/score") / A_REC / f"L{c[0]}_{KIND}_C{c[1]:g}.outscores.npz")) for c in cands}
    n_rows = np.bincount(side[cands[0]]["row_run"], minlength=n)
    chk = 0.0
    for c in cands:
        s = side[c]
        full = np.stack([np.bincount(s["row_run"], weights=s["out_scores"][:, j].astype(float), minlength=n) for j in range(K)], 1) / np.maximum(n_rows, 1)[:, None]
        m = np.isfinite(R[c]["on_score"][:, 0]) & (n_rows > 0)
        chk = max(chk, float(np.abs(full[m] - R[c]["on_score"][m]).max()))
    if chk > 1e-5:
        raise ValueError(f"(a): per-output scores do not reproduce the saved run scores ({chk})")
    cols = {"a": {"runs": r0["on_runs"].astype(str).tolist(), "Y": r0["Y"], "train_world": "run1",
                  "P": {j: selected({c: side[c][f"prefix_j{j}"] for c in cands}, r0, pick, complete, K) for j in PREFIX_J}
                  | {"whole": selected({c: R[c]["on_score"] for c in cands}, r0, pick, complete, K)}}}
    meta = {"a": {"unconverged_share": share_a, "reproduction_max_abs_diff": chk, "choices_real_column": [[*cands[pick[f, 0]]] for f in range(5)]}}

    # (c): the same selection on cell (c)'s records (column 0), then each fold's rows rescored from the saved weights
    R, r0, pick, complete, share_c = records(C_REC, cols0_only=True)
    picked = [[*cands[pick[f, 0]]] for f in range(5)]
    if picked != [list(x) for x in c_choices]:
        raise ValueError(f"(c): recomputed selection {picked} differs from stage 4's {c_choices}")
    runs_c = r0["on_runs"].astype(str).tolist()
    idx = {u: i for i, u in enumerate(runs_c)}
    fold = r0["on_fold"]
    per_run, worst = {}, 0.0
    for f, (l, C) in enumerate(c_choices):
        w = np.load(Path("/score") / C_REC / f"W_L{l}_C{C:g}.npz")
        rows = json.loads((Path("/score") / CACHE / f"on_rows.L{l}.json").read_text())
        X = np.load(Path("/score") / CACHE / f"on_L{l}.npy", mmap_mode="r")
        sel = [i for i, r in enumerate(rows) if r["kind"] == KIND and f"{r['world']}|{r['task']}" in idx and fold[idx[f"{r['world']}|{r['task']}"]] == f]
        s = (((np.asarray(X[sel], np.float64) - w["mean"][f]) / w["std"][f]) @ w["W"][f]) * float(w["sqrtC"]) + w["b"][f]
        s = np.asarray(s, np.float64).reshape(len(sel), -1)[:, 0]
        for i, v in zip(sel, s):
            r = rows[i]
            per_run.setdefault(f"{r['world']}|{r['task']}", []).append((r["order"][0], r["order"][1], float(v)))
        rec = R[(l, C)]
        for u in [u for u in runs_c if fold[idx[u]] == f and u in per_run]:
            st = rec["on_score"][idx[u], 0]
            if np.isfinite(st):
                worst = max(worst, abs(np.mean([v for *_, v in per_run[u]]) - float(st)))
    if worst > GATE_C:
        raise ValueError(f"(c): rescored run means differ from the stored scores (max {worst:.3g})")
    P = {j: np.full((len(runs_c), 1), np.nan) for j in (*POS, "whole")}
    nr_c = np.zeros(len(runs_c), int)
    for u, lst in per_run.items():
        v = [x for *_, x in sorted(lst)]
        i = idx[u]
        nr_c[i] = len(v)
        if not complete[i]:
            continue
        P["whole"][i, 0] = np.mean(v)
        for j in POS:
            if len(v) >= j:
                P[j][i, 0] = np.mean(v[:j])
    cols["c"] = {"runs": runs_c, "Y": r0["Y"][:, :1], "train_world": None, "P": P}
    meta["c"] = {"unconverged_share": share_c, "rescore_max_abs_diff": worst, "choices_real_column": picked}

    # D3's fixed cohort from (a)'s prefix-21 column; (c) must cover the same tasks
    def cohort(d):
        world = np.array([u.split("|", 1)[0] for u in d["runs"]])
        task = np.array([u.split("|", 1)[1] for u in d["runs"]])
        return {t for t in tasks if all(((world == w) & (task == t) & np.isfinite(d["P"][21][:, 0])).any() for w in WORLDS)}

    fixed = cohort(cols["a"])
    if cohort(cols["c"]) != fixed:
        raise ValueError(f"(c)'s 21-row coverage differs from (a)'s ({len(cohort(cols['c']))} vs {len(fixed)} tasks)")

    def masked(d, j):
        task = np.array([u.split("|", 1)[1] for u in d["runs"]])
        Z = d["P"][j].copy()
        Z[~np.isin(task, sorted(fixed))] = np.nan
        return {"runs": d["runs"], "Y": d["Y"], "Z": Z}

    out = {"fixed_cohort_tasks": len(fixed), "meta": meta, "cells": {}, "gate_d3": {}}
    for row, d in cols.items():
        for j in (*POS, "whole"):
            st = A.stats(masked(d, j), lab, tasks, ti, W, d["train_world"])
            for fpr in ("0.01", "0.05"):
                o = st[f"at_{fpr}"]
                o["share_of_room_used"] = None if o["catch_run2"] in (None, 1.0) else (o["catch_sim"] - o["catch_run2"]) / (1 - o["catch_run2"])
            out["cells"][f"{row}_j{j}"] = st
            if row == "a" and j in POS:
                got, want = st["at_0.01"]["catch_run1"], d3_run1[str(j)]
                out["gate_d3"][f"j{j}"] = {"got": got, "d3": want, "passes": got is not None and abs(got - want) <= GATE_D3}
    if not all(g["passes"] for g in out["gate_d3"].values()):
        return {"gate_d3": out["gate_d3"], "STOP": "(a) does not reproduce D3's full result; nothing is read"}
    return out


@app.local_entrypoint()
def main(labels: str, stage4: str):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("uncommitted tracked changes")
    if hashlib.sha256(Path(labels).read_bytes()).hexdigest() != LABELS_SHA256:
        raise SystemExit("labels sha256 mismatch")
    head, t0 = git("rev-parse", "HEAD"), time.time()
    lab = json.loads(Path(labels).read_text())["labels"]
    s4 = json.loads(Path(stage4).read_text())
    c_choices = s4["cells"]["c_swapped_L1"]["choices_real_column"]
    d3 = json.loads(git("show", f"{D3_FULL[0]}:{D3_FULL[1]}"))["readouts"]["reply_end"]["fixed"]
    d3_run1 = {str(j): d3[str(j)]["observed"]["tpr_run1"] for j in POS}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    print(json.dumps({"stamp": stamp, "git": head, "c_choices": c_choices, "d3_run1": d3_run1}), flush=True)
    res = analyze.remote(lab, c_choices, d3_run1)
    (HERE / "results").mkdir(exist_ok=True)
    dest = HERE / "results" / f"q4full_onpolicy_positions_POSTHOC_{stamp}.json"
    dest.open("x").write(json.dumps({"label": "POST HOC (RULE.md Amendment 1); preliminary until Codex", "git": head, "stamp": stamp,
                                     "stage4_sha256": hashlib.sha256(Path(stage4).read_bytes()).hexdigest(), "d3_full": D3_FULL,
                                     "client_seconds": round(time.time() - t0)} | res, indent=1, default=float))
    if "STOP" in res:
        print(json.dumps(res, default=float))
        print(dest.name)
        return
    pct = lambda x: "  n/a" if x is None else f"{100 * x:6.2f}"  # noqa: E731
    print("fixed cohort tasks:", res["fixed_cohort_tasks"], "| gate (a) vs D3 full:", res["gate_d3"], "| meta:", json.dumps(res["meta"]))
    for row in ("c", "a"):
        for j in (*POS, "whole"):
            c = res["cells"][f"{row}_j{j}"]
            for fpr in ("0.01", "0.05"):
                o = c[f"at_{fpr}"]
                iv = o["sim_minus_run2_interval"]["ci95"]
                print(f"({row}) j{j!s:<5} @{float(fpr) * 100:.0f}% sim {pct(o['catch_sim'])} run2 {pct(o['catch_run2'])} run1 {pct(o['catch_run1'])} "
                      f"| sim-run2 {pct(o['sim_minus_run2'])} [{pct(iv[0])},{pct(iv[1])}] | share {pct(o['share_of_room_used'])}")
    print(dest.name)
