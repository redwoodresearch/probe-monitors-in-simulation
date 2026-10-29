"""The attack-detector retrain's fits (plan_attack_detector_retrain_2026-10-10.md, frozen in #342, Amendments 1-2 at
d4a8bbaa): the converged C search with its selection-aware null, trained off-policy on Sol's runs read by Qwen3.5-27B,
cross-fit by main-task fold, scoring Qwen's own runs. Modal: a CPU cache, then one H100 per (layer, readout).

Cells: {all attack runs, attempted-only} x {level 0 (unmasked), level 1 (dates and times masked)}. Training rows: Sol's
real attack (epoch 1) and real honest runs, every output's readout (output_end, or call_end) labelled with its run;
run score = mean over the run's outputs. Attempted-only drops real attack runs not labelled attempted (Phase 1 labels,
`label()` at 8950b34, results/attack_attempt_labels_2026-10-08_010659.jsonl of the nano monitor). Fitter: copied from
run_claimB_grid.py at 336da339 (the rescaled problem at C = 1 on sqrt(C)·X, float64, converged when each column's largest
gradient entry <= 1e-4·min(sqrt(C), 1), refits up to 80,000 iterations). 1 + 100 label columns: the real labels, then
100 shuffles of the attack/honest labels among each main task's runs (seed [20261010, d], the same in every container).
Outer fold k (folds_v3 main-task folds): the full model trains on the other folds and scores (a) Qwen's runs whose task's
main task is in fold k, by all outputs and by its first k_match readouts, and (b) Sol's real honest training runs, whose
mean and SD freeze that fold's standardization for every label column (Amendments 1, 3). The validation model trains
without fold (k + 1) % 5 and scores it: validation AUROC, on which the analysis chooses C, layer and readout. Each
(level, cell, C) record is written as it finishes. k_match (plan, matched-length check; Amendment 3): one k for both
readouts, the smallest over Qwen's real attack runs 1 and 2, honest_a and honest_b of each world's 10th percentile of
tool-output counts (level 0's run inventory: every counted run, every tool output with or without a readout), rounded down, fixed here before any score; a run with fewer tool
outputs gets no first-k score. `--timing`: Aswin's rule (10:51 PT): one container, level 0, all runs, C 1e-6, 1e-2 and 100, every label column a
shuffle (no real labels), records under a separate tag that the analysis never reads.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' modal run --detach \
      experiments/2026-10-08_q3_onpolicy_figures/run_retrain_grid.py::main [--cache-only] [--levels level0] [--timing]
"""

import hashlib
import json
import os
import time
from pathlib import Path

import modal

HERE = Path(__file__).parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-q3onpol_retrain_kill-2026-11-10"
OFF_VOL, ON_VOL = "probe-shift_aswin-surya_sub-qwen35_acts_kill-2026-11-06", "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08"
SCORE_VOL = "probe-shift_aswin-surya_sub-q3onpol_score_kill-2026-11-08"
ON_RUNS = ["2026-10-09_rerun_strict_segends_v1", "2026-10-10_scaleup_canary_segends_v1", "2026-10-10_scaleup_l2_segends_v1",
           "2026-10-10_scaleup_topup_segends_v1", "2026-10-10_second_attack_segends_v1"]
SECOND = "2026-10-10_second_attack_segends_v1"
LEVELS = {"level0": ("2026-10-06_la_segment-end-v4_qwen35-27b_alllayers_v1", ON_RUNS),
          "level1": ("2026-10-06_la_segment-end-v4-scrub1_qwen35-27b_alllayers_v1", [f"{r}_scrub1" for r in ON_RUNS])}
WORLD = {"attack": "run1", "honest_a": "honest_a", "honest_b": "honest_b", "sim_attack@september": "sim",
         "sim_honest@september": "sim_honest", "sim_attack@today": "sim_today", "sim_honest@today": "sim_honest_today"}
OFF_ARMS = {"real-attack-e1": 1, "real-honest": 0}
LAYERS, KINDS, CELLS = [39, 51, 63], ["output_end", "call_end"], ["all", "attempted"]
C_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
N_NULL, MAX_ITER, MAX_ITER_CAP, CACHE = 100, 5000, 80000, "retrain_cache_v1"
FOLDS, TASKS = "splits/folds_v3.json", "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"
P1_LABELS = "experiments/2026-10-07_q3_nano_monitor/results/attack_attempt_labels_2026-10-08_010659.jsonl"

app = modal.App(APP)
off_vol, on_vol, score_vol = (modal.Volume.from_name(v, version=2) for v in (OFF_VOL, ON_VOL, SCORE_VOL))
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy", "scipy")
         .add_local_python_source("probe_shift")
         .add_local_file(REPO / FOLDS, f"/root/{FOLDS}").add_local_file(REPO / TASKS, f"/root/{TASKS}")
         .add_local_file(REPO / P1_LABELS, f"/root/{P1_LABELS}"))
VOLS = {"/off": off_vol.read_only(), "/on": on_vol.read_only(), "/score": score_vol}


def tol_of(C: float) -> float:
    return 1e-4 * min(C**0.5, 1.0)


def read_rows(root: Path, picks, layer: int):
    """fp16 [len(picks), 5120]: each chunk file read whole (row-indexed memory maps fetch page by page from a Volume)."""
    import numpy as np

    by = {}
    for i, (chunk, row) in enumerate(picks):
        by.setdefault(chunk, []).append((i, row))
    X = np.empty((len(picks), 5120), np.float16)
    for chunk, items in by.items():
        a = np.load(root / "by_layer" / f"L{layer}" / f"{chunk}.npy")
        X[[i for i, _ in items]] = a[[r for _, r in items]]
    if not np.isfinite(X.astype(np.float32)).all():
        raise ValueError(f"non-finite activations in {root.name} L{layer}")
    return X


@app.function(image=image, cpu=8, memory=65536, timeout=3600, retries=0, volumes=VOLS)
def cache(level: str, layer: int) -> dict:
    """Off-policy (Sol, read by Qwen3.5) and on-policy (Qwen's own runs) rows of both readouts at one layer."""
    import numpy as np

    d = Path("/score") / CACHE / level
    if (d / f"done.L{layer}.json").exists():
        return json.loads((d / f"done.L{layer}.json").read_text())
    d.mkdir(parents=True, exist_ok=True)
    off_name, on_names = LEVELS[level]
    off_rows, off_picks = [], []
    for arm in OFF_ARMS:
        for f in sorted((Path("/off") / off_name / "meta" / arm).glob("*.json")):
            m = json.loads(f.read_text())
            for p in m["positions"]:
                if p["kind"] in KINDS and p.get("kept") and p.get("row") is not None:
                    off_rows.append({"arm": arm, "key": m["key"], "main_task": m["main_task"], "kind": p["kind"]})
                    off_picks.append((m["chunk"], m["row_start"] + p["row"]))
    on_rows, on_picks, counted, inventory = [], [], {}, {}
    for run in on_names:
        root = Path("/on") / run
        json.loads((root / "complete.json").read_text())  # written only when every shard finished
        for f in sorted((root / "meta").glob("*/*.json")):
            m = json.loads(f.read_text())
            world = "run2" if m["arm"] == "attack" and run.startswith(SECOND) else WORLD.get(m["arm"])
            if world is None or m.get("redrawn_because") or m.get("attempt_kind") == "rerun":
                continue
            counted.setdefault((world, m["task"]), set()).add(m["run"])
            # every counted run and its tool outputs, readout or not (the matched-length check's counts; Amendment 3)
            inventory[f"{world}|{m['task']}"] = sum(c.get("kind") == "output_end" for c in m["calls"])
            for c in m["calls"]:
                if c.get("kind") in KINDS and c.get("row") is not None:
                    on_rows.append({"world": world, "task": m["task"], "kind": c["kind"], "order": [c["turn"], c.get("call_index", 0)]})
                    on_picks.append((run, m["chunk"], m["row_start"] + c["row"]))
    dup = {f"{w}|{t}": sorted(v) for (w, t), v in counted.items() if len(v) > 1}
    if dup:
        raise ValueError(f"{len(dup)} world/task pairs have several counted runs, e.g. {next(iter(dup.items()))}")
    Xo = read_rows(Path("/off") / off_name, off_picks, layer)
    Xn = np.concatenate([read_rows(Path("/on") / r, [(c, i) for r_, c, i in on_picks if r_ == r], layer) for r in on_names])
    order = [i for r in on_names for i, p in enumerate(on_picks) if p[0] == r]
    on_rows = [on_rows[i] for i in order]
    np.save(d / f"off_L{layer}.npy", Xo)
    np.save(d / f"on_L{layer}.npy", Xn)
    (d / f"off_rows.L{layer}.json").write_text(json.dumps(off_rows))
    (d / f"on_rows.L{layer}.json").write_text(json.dumps(on_rows))
    (d / f"inventory.L{layer}.json").write_text(json.dumps(inventory))
    with_rows = {f"{r['world']}|{r['task']}" for r in on_rows}
    info = {"level": level, "layer": layer, "off_rows": len(off_rows), "on_rows": len(on_rows), "on_runs": len(inventory),
            "on_runs_without_readout_rows": sorted(set(inventory) - with_rows),
            "rows_sha256": hashlib.sha256(json.dumps([off_rows, on_rows]).encode()).hexdigest()}
    (d / f"done.L{layer}.json").write_text(json.dumps(info))
    score_vol.commit()
    return info


def k_match(counts: dict) -> int:
    """One k for both readouts: the smallest, over Qwen's real runs 1 and 2, honest_a and honest_b, of each world's 10th
    percentile of tool-output counts, rounded down (plan, matched-length check; Amendment 3, item 4)."""
    import numpy as np

    return int(min(np.floor(np.percentile([v for u, v in counts.items() if u.split("|")[0] == w], 10))
                   for w in ("run1", "run2", "honest_a", "honest_b")))


@app.function(image=image, gpu="H100!", cpu=8, memory=98304, timeout=3 * 3600, retries=0, volumes=VOLS, scaledown_window=5)
def grid(tag: str, layer: int, kind: str, timing: bool = False, levels: tuple = tuple(LEVELS)) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scipy.stats import rankdata

    from probe_shift import torch_probe as tp

    score_vol.reload()
    fold_of = json.loads((Path("/root") / FOLDS).read_text())["task_folds"]["main_task_to_fold"]
    main_of = {c["key"]: c["main"] for c in json.loads((Path("/root") / TASKS).read_text())["cells"]}
    p1 = {r["key"]: r["label"] for r in map(json.loads, (Path("/root") / P1_LABELS).read_text().splitlines()) if r["arm"] == "real-attack-e1"}
    counts0 = json.loads((Path("/score") / CACHE / "level0" / f"inventory.L{layer}.json").read_text())
    k0 = k_match(counts0)
    t0, out = time.time(), {"k_match": k0}
    levels, cells, Cs = (["level0"], ["all"], [1e-6, 1e-2, 100.0]) if timing else (list(levels), CELLS, C_GRID)
    for level in levels:
        d = Path("/score") / CACHE / level
        off_rows, on_rows = (json.loads((d / f"{n}_rows.L{layer}.json").read_text()) for n in ("off", "on"))
        ko = np.flatnonzero([r["kind"] == kind for r in off_rows])
        kn = np.flatnonzero([r["kind"] == kind for r in on_rows])
        Xo = torch.from_numpy(np.load(d / f"off_L{layer}.npy")[ko]).cuda()
        Xn = torch.from_numpy(np.load(d / f"on_L{layer}.npy")[kn]).cuda()
        # runs over both readouts, so a label column (and its shuffle) is the same labelling for every readout and layer
        runs, o_run = np.unique([r["key"] for r in off_rows], return_inverse=True)
        o_run = o_run[ko]
        run_arm = {r["key"]: r["arm"] for r in off_rows}
        run_main = {r["key"]: r["main_task"] for r in off_rows}
        missing = sorted({run_main[k] for k in runs} - set(fold_of)) + sorted(k for k in runs if run_arm[k] == "real-attack-e1" and k not in p1)
        if missing:
            raise ValueError(f"main tasks without a fold or attack runs without a Phase 1 label, e.g. {missing[:3]}")
        r_fold = np.array([fold_of[run_main[k]] for k in runs])
        r_y = np.array([OFF_ARMS[run_arm[k]] for k in runs], float)
        on_runs, n_run = np.unique([f"{r['world']}|{r['task']}" for r in on_rows], return_inverse=True)
        n_run = n_run[kn]
        n_fold = np.array([fold_of[main_of[u.split("|", 1)[1]]] for u in on_runs])
        rank = np.zeros(len(kn), int)  # each on-policy row's order within its run
        for u in range(len(on_runs)):
            m = np.flatnonzero(n_run == u)
            rank[sorted(m, key=lambda i: tuple(on_rows[kn[i]]["order"]))] = np.arange(len(m))
        long_enough = np.array([counts0.get(u, 0) >= k0 for u in on_runs])  # level 0's tool-output counts, both readouts
        for cell in cells:
            keep = (r_y == 0) | (np.array([p1.get(k) for k in runs]) == "attempted") if cell == "attempted" else np.ones(len(runs), bool)
            Y = np.zeros((len(runs), 1 + N_NULL))
            Y[:, 0] = r_y
            mains = np.array([run_main[k] for k in runs])
            for c in range(0 if timing else 1, 1 + N_NULL):  # --timing: column 0 is a shuffle too (no real labels)
                rng = np.random.default_rng([20261010, c + (1000 if timing and c == 0 else 0)])
                y = r_y.copy()
                for m in sorted(set(mains[keep])):
                    j = np.flatnonzero(keep & (mains == m))
                    y[j] = y[rng.permutation(j)]
                Y[:, c] = y
            K = Y.shape[1]
            for C in Cs:
                p = Path("/score") / f"retrain_{tag}{'_TIMING' if timing else ''}" / level / cell / f"L{layer}_{kind}_C{C:g}.npz"
                if p.exists():
                    continue
                r = {"val_auroc": np.full((5, K), np.nan), "grad_full": np.zeros((5, K)), "grad_val": np.zeros((5, K)),
                     "honest_mu": np.zeros((5, K)), "honest_sd": np.zeros((5, K)), "evals": np.zeros((5, 2), int),
                     "on_score": np.full((len(on_runs), K), np.nan), "on_score_k": np.full((len(on_runs), K), np.nan)}
                for f in range(5):
                    v = (f + 1) % 5
                    for name, tr_runs in (("full", keep & (r_fold != f)), ("val", keep & (r_fold != f) & (r_fold != v))):
                        a = np.flatnonzero(tr_runs[o_run])
                        b_ = np.flatnonzero(n_fold[n_run] == f) if name == "full" else np.flatnonzero((keep & (r_fold == v))[o_run])
                        Xa, Xb = tp.standardize(Xo[a].double(), (Xn if name == "full" else Xo)[b_].double())
                        Xa.mul_(C**0.5), Xb.mul_(C**0.5)
                        Ya = torch.from_numpy(Y[o_run[a]]).cuda()
                        mi, prev = MAX_ITER, None
                        while True:  # refit only fits that used all their iterations; stop when a refit repeats
                            W, b, evals = tp.fit(Xa, Ya, 1.0, mi, tol=tol_of(C))
                            Wg, bg = W.clone().requires_grad_(True), b.clone().requires_grad_(True)
                            loss = (F.binary_cross_entropy_with_logits(Xa @ Wg + bg, Ya, reduction="sum") + 0.5 * (Wg * Wg).sum()) / len(Xa)
                            loss.backward()
                            g = torch.maximum(Wg.grad.abs().amax(0), bg.grad.abs()).cpu().numpy()
                            if g.max() <= tol_of(C) or evals < mi or evals == prev or mi * 4 > MAX_ITER_CAP:
                                break
                            mi, prev = mi * 4, evals
                        r[f"grad_{name}"][f], r["evals"][f, int(name == "val")] = g, evals
                        S = tp.decision(Xb, W, b).cpu().numpy()
                        if name == "val":
                            u, inv = np.unique(o_run[b_], return_inverse=True)
                            M = np.stack([np.bincount(inv, weights=S[:, c]) for c in range(K)], 1) / np.bincount(inv)[:, None]
                            R, Yv = rankdata(M, axis=0), Y[u]
                            n1 = Yv.sum(0)
                            r["val_auroc"][f] = ((R * Yv).sum(0) - n1 * (n1 + 1) / 2) / (n1 * (len(Yv) - n1))
                        else:
                            for col, sel in (("on_score", np.ones(len(b_), bool)), ("on_score_k", (rank[b_] < k0) & long_enough[n_run[b_]])):
                                u, inv = np.unique(n_run[b_][sel], return_inverse=True)
                                M = np.stack([np.bincount(inv, weights=S[sel, c]) for c in range(K)], 1) / np.bincount(inv)[:, None]
                                full_k = np.bincount(inv) == k0 if col == "on_score_k" else np.ones(len(u), bool)
                                r[col][u[full_k]] = M[full_k]
                            Sa = tp.decision(Xa, W, b).cpu().numpy()  # its training honest runs, per column's labels
                            ua, inva = np.unique(o_run[a], return_inverse=True)
                            Ma = np.stack([np.bincount(inva, weights=Sa[:, c]) for c in range(K)], 1) / np.bincount(inva)[:, None]
                            hon = r_y[ua] == 0  # Sol's real honest training runs, for every label column (Amendment 3)
                            r["honest_mu"][f], r["honest_sd"][f] = Ma[hon].mean(0), Ma[hon].std(0)
                            if not (np.isfinite(r["honest_sd"][f]).all() and (r["honest_sd"][f] > 0).all()):
                                raise ValueError(f"L{layer} {kind} {level} {cell} C {C:g} fold {f}: a degenerate honest standardization")
                        del Xa, Xb
                p.parent.mkdir(parents=True, exist_ok=True)
                with open(p.with_suffix(".tmp"), "wb") as fh:
                    np.savez(fh, on_runs=on_runs, on_fold=n_fold, k_match=k0, long_enough=long_enough, tol=tol_of(C), **r)
                os.replace(p.with_suffix(".tmp"), p)
                score_vol.commit()
                out[f"{level} {cell} {C:g}"] = round(time.time() - t0)
                print(f"L{layer} {kind} {level} {cell} C {C:g} done at {out[f'{level} {cell} {C:g}']} s; evals max {r['evals'].max()}", flush=True)
    return out


@app.local_entrypoint()
def main(cache_only: bool = False, timing: bool = False, levels: str = "level0,level1"):
    caches = list(cache.starmap([(lv, l) for lv in (["level0"] if timing else levels.split(",")) for l in LAYERS]))
    print(json.dumps(caches), flush=True)
    if cache_only:
        return
    tag = hashlib.sha256(Path(__file__).read_bytes() + json.dumps(caches, sort_keys=True).encode()).hexdigest()[:12]
    print(f"tag {tag}", flush=True)
    lv = ("level0",) if timing else tuple(levels.split(","))
    jobs = [(tag, 51, "output_end", True, lv)] if timing else [(tag, l, k, False, lv) for l in LAYERS for k in KINDS]
    for r in grid.starmap(jobs):
        print(json.dumps(r), flush=True)
