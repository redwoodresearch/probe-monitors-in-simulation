"""The attack detector, trained by reading, applied to Qwen's own runs (exploratory; the frozen analysis and prediction
are predictions_attack_detector_onpolicy_2026-10-09.md, 007f2ae). Modal: a CPU row cache, then one A10G for the fits.

Detector (#36's recipe on Qwen3.5-27B reading Sol's runs, dump 2026-10-06_la_segment-end-v4_qwen35-27b_alllayers_v1):
per-output L2 logistic regression (probe_shift.probes.fit_logreg: sklearn's objective, unpenalized intercept) on features
standardized with the training rows' mean and population std; every output labelled with its run (real attack e1 = 1,
real honest = 0); a run's score is the mean of its outputs' logits.
- final probe: every real e1 and real honest run outside the 20 Phase 2 pilot main tasks; C from {0.001, 0.01, 0.1} by
  run-level AUROC on folds_v3 fold 0 (trained on the other folds), then refit on all of them;
- the fold-0 model (the C-choice fit at the chosen C) sets thresholds at 5% and 1% false alarms on fold 0's held-out real
  honest runs; it gives the descriptive catch and false alarms;
- off-policy reference: 5 outer folds (folds_v3), the chosen C, train on 4 folds' real runs, score the 5th fold's runs of
  all four arms (real e1, real honest, simulated attack, simulated honest);
- null (the primary cell only): 50 shuffles of the training labels among each main task's real runs, each a full refit
  with its own choice of C, applied to the on-policy runs.
On-policy rows: the extraction 2026-10-09_rerun_strict_segends_v1 (all 7 arms, replayed token-exact), the counted attempt
of each arm and task (as run_score_onpolicy.load). Every on-policy run gets a score from every probe; the analysis
(analyze_attack_detector.py) picks the worlds, drops errored runs and computes AUROCs, gaps and intervals.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.2' --with numpy modal run --detach \
      experiments/2026-10-08_q3_onpolicy_figures/run_attack_detector.py [--cache-only | --swap]
`--swap`: the saved final probes applied to the swap diagnostic's extraction (every layer and readout), per-run means.
`--per-output-scores`: the saved final probes applied to the cached on-policy rows, one score per output with its turn and
call index (for the matched-length check, analyze_attack_detector_firstk.py); no refit.
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE  # in a Modal container the script sits at /root
APP = "probe-shift_aswin-surya_sub-q3onpol_attackdet_kill-2026-11-08"
OFF = ("probe-shift_aswin-surya_sub-qwen35_acts_kill-2026-11-06", "2026-10-06_la_segment-end-v4_qwen35-27b_alllayers_v1")
ON = ("probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08", "2026-10-09_rerun_strict_segends_v1")
SWAP = "2026-10-09_rerun_strict_segends_swap_v1"  # D1: 5 tasks' attack runs with honest_b's system turn (run_swap_stage.py)
SCORE = "probe-shift_aswin-surya_sub-q3onpol_score_kill-2026-11-08"
CACHE = "attackdet_cache_v1"
PILOT = "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9.json"
FOLDS = "splits/folds_v3.json"
OFF_ARMS = {"real-attack-e1": 1, "real-honest": 0, "sim-attack": 1, "sim-honest": 0}
LAYERS, KINDS, GRID, MAX_ITER, N_NULL = [39, 51, 63], ["output_end", "call_end"], [0.001, 0.01, 0.1], 1000, 50
PRIMARY = (51, "output_end")

app = modal.App(APP)
off_vol, on_vol = modal.Volume.from_name(OFF[0], version=2), modal.Volume.from_name(ON[0], version=2)
score_vol = modal.Volume.from_name(SCORE, version=2)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy", "scikit-learn==1.7.2")
         .add_local_python_source("probe_shift")
         .add_local_file(REPO / PILOT, f"/root/{PILOT}").add_local_file(REPO / FOLDS, f"/root/{FOLDS}"))
VOLS = {"/off": off_vol.read_only(), "/on": on_vol.read_only(), "/score": score_vol}


def read_layers(root, picks, layers):
    """picks: [(chunk, global row)]; returns {layer: fp16 [len(picks), 5120]} read with 32 threads."""
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np

    groups = {}
    for i, (chunk, row) in enumerate(picks):
        g = groups.setdefault(chunk, ([], []))
        g[0].append(i)
        g[1].append(row)
    out = {layer: np.empty((len(picks), 5120), np.float16) for layer in layers}

    def one(job):
        layer, chunk, (pos, rows) = job
        out[layer][pos] = np.load(root / "by_layer" / f"L{layer}" / f"{chunk}.npy", mmap_mode="r")[rows]

    with ThreadPoolExecutor(32) as ex:
        list(ex.map(one, [(layer, c, g) for layer in layers for c, g in groups.items()]))
    for layer, x in out.items():
        if not np.isfinite(x.astype(np.float32)).all():
            raise ValueError(f"non-finite activations at layer {layer}")
    return out


@app.function(image=image, cpu=16, memory=65536, timeout=3600, retries=0, volumes=VOLS)
def cache() -> dict:
    """Off-policy and on-policy output_end and call_end rows at LAYERS, once, into /score/<CACHE>/."""
    import numpy as np

    d = Path("/score") / CACHE
    if (d / "done.json").exists():
        return json.loads((d / "done.json").read_text())
    d.mkdir(parents=True, exist_ok=True)
    off = Path("/off") / OFF[1]
    off_rows, off_picks = [], []
    for arm in OFF_ARMS:
        for f in sorted((off / "meta" / arm).glob("*.json")):
            m = json.loads(f.read_text())
            for p in m["positions"]:
                if p["kind"] in KINDS and p.get("kept") and p.get("row") is not None:
                    off_rows.append({"arm": arm, "key": m["key"], "main_task": m["main_task"], "environment": m["environment"],
                                     "kind": p["kind"], "step": p["step"], "action": p["action"]})
                    off_picks.append((m["chunk"], m["row_start"] + p["row"]))
    on = Path("/on") / ON[1]
    done = json.loads((on / "complete.json").read_text())
    on_rows, on_picks, counted = [], [], {}
    for f in sorted((on / "meta").glob("*/*.json")):
        m = json.loads(f.read_text())
        if m.get("redrawn_because") or m.get("attempt_kind") == "rerun":
            continue
        counted.setdefault((m["arm"], m["task"]), set()).add(m["run"])
        for c in m["calls"]:
            if c.get("kind") in KINDS and c.get("row") is not None:
                on_rows.append({"arm": m["arm"], "task": m["task"], "run": m["run"], "kind": c["kind"], "src_irid": c.get("src_irid"),
                                "call_index": c.get("call_index"), "label": c.get("label")})
                on_picks.append((m["chunk"], m["row_start"] + c["row"]))
    dup = {k: sorted(v) for k, v in counted.items() if len(v) > 1}
    if dup:
        raise ValueError(f"{len(dup)} arm/task pairs have several counted attempts, e.g. {next(iter(dup.items()))}")
    for name, root, picks in (("off", off, off_picks), ("on", on, on_picks)):
        X = read_layers(root, picks, LAYERS)
        for layer, x in X.items():
            np.save(d / f"{name}_L{layer}.npy", x)
    (d / "off_rows.json").write_text(json.dumps(off_rows))
    (d / "on_rows.json").write_text(json.dumps(on_rows))
    info = {"off_rows": len(off_rows), "on_rows": len(on_rows), "on_extraction_complete": done,
            "off_rows_sha256": hashlib.sha256(json.dumps(off_rows).encode()).hexdigest(),
            "on_rows_sha256": hashlib.sha256(json.dumps(on_rows).encode()).hexdigest()}
    (d / "done.json").write_text(json.dumps(info))
    score_vol.commit()
    return info


@app.function(image=image, gpu="A10G", cpu=8, memory=65536, timeout=3 * 3600, retries=0, volumes=VOLS)
def fit_all() -> dict:
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score

    from probe_shift.attack_eval import threshold
    from probe_shift.probes import fit_logreg

    score_vol.reload()
    d = Path("/score") / CACHE
    off_rows, on_rows = json.loads((d / "off_rows.json").read_text()), json.loads((d / "on_rows.json").read_text())
    pilot_main = {c["main"] for c in json.loads((Path("/root") / PILOT).read_text())["cells"]}
    folds_of = json.loads((Path("/root") / FOLDS).read_text())
    fold_task = folds_of["task_folds"]["main_task_to_fold"]  # main task -> fold (0-4)
    o_arm = np.array([r["arm"] for r in off_rows])
    o_key = np.array([r["key"] for r in off_rows])
    o_main = np.array([r["main_task"] for r in off_rows])
    o_kind = np.array([r["kind"] for r in off_rows])
    missing = sorted(set(o_main) - set(fold_task))
    if missing:
        raise ValueError(f"{len(missing)} main tasks without a fold, e.g. {missing[:3]}")
    o_fold = np.array([fold_task[t] for t in o_main])
    o_y = np.array([OFF_ARMS[a] for a in o_arm])
    real = np.isin(o_arm, ["real-attack-e1", "real-honest"])
    n_kind = np.array([r["kind"] for r in on_rows])
    on_run = np.array([f"{r['arm']}|{r['task']}" for r in on_rows])
    rng = np.random.default_rng([20261009, 36])

    def run_means(keys, s):
        u, inv = np.unique(keys, return_inverse=True)
        return u, np.bincount(inv, weights=s) / np.bincount(inv)

    def run_auc(keys, y_row, s):
        u, first = np.unique(keys, return_index=True)
        _, m = run_means(keys, s)
        y = y_row[first]
        return float(roc_auc_score(y, m)) if 0 < y.sum() < len(y) else float("nan")

    kept = {}  # the final probes' weights, saved for later scoring (e.g. the honest-prompt swap)

    def fit(X, rows, y, C, keep_as=None):
        Xt = X[rows]
        mu, sd = Xt.mean(0), Xt.std(0, unbiased=False).clamp_min(1e-6)
        w, g = fit_logreg((Xt - mu) / sd, torch.tensor(y[rows], dtype=torch.float32, device="cuda"), C, MAX_ITER)
        if keep_as:
            kept[keep_as] = {"w": w[:-1].cpu().numpy(), "b": float(w[-1]), "mu": mu.cpu().numpy(), "sd": sd.cpu().numpy(), "C": C}
        return (lambda Z: (((Z - mu) / sd) @ w[:-1] + w[-1]).cpu().numpy()), g

    out = {"cells": {}, "on_runs": None}
    for layer in LAYERS:
        Xo_all = torch.from_numpy(np.load(d / f"off_L{layer}.npy")).cuda().float()
        Xn_all = torch.from_numpy(np.load(d / f"on_L{layer}.npy")).cuda().float()
        for kind in KINDS:
            ok, nk = np.flatnonzero(o_kind == kind), np.flatnonzero(n_kind == kind)
            Xo, Xn = Xo_all[ok], Xn_all[nk]
            arm, key, main, fold, y = o_arm[ok], o_key[ok], o_main[ok], o_fold[ok], o_y[ok]
            rows_train = np.flatnonzero(real[ok] & ~np.isin(main, list(pilot_main)))
            # C on fold 0 (rows indexed within this kind)
            tr0, va0 = rows_train[fold[rows_train] != 0], rows_train[fold[rows_train] == 0]
            cres, fold0 = {}, {}
            for C in GRID:
                Xt = Xo[tr0]
                mu, sd = Xt.mean(0), Xt.std(0, unbiased=False).clamp_min(1e-6)
                w, g = fit_logreg((Xt - mu) / sd, torch.tensor(y[tr0], dtype=torch.float32, device="cuda"), C, MAX_ITER)
                sc = lambda Z, mu=mu, sd=sd, w=w: (((Z - mu) / sd) @ w[:-1] + w[-1]).cpu().numpy()
                cres[C] = run_auc(key[va0], y[va0], sc(Xo[va0]))
                fold0[C] = sc
            C = max(GRID, key=lambda c: (cres[c], -c))
            final, g = fit(Xo, rows_train, y, C, keep_as=f"L{layer}_{kind}")
            # fold-0 model: thresholds on fold 0's held-out real honest runs, at the chosen C
            hon0 = va0[y[va0] == 0]
            _, hon_scores = run_means(key[hon0], fold0[C](Xo[hon0]))
            thr = {a: float(threshold(hon_scores, a)) for a in (0.05, 0.01)}  # #36's rule: at most floor(a n) honest runs above
            # off-policy reference: 5 outer folds, all four arms scored; C chosen inside each outer training split (its
            # validation fold is f + 1), training rows as the final probe's (real runs, pilot main tasks left out)
            off_scores, outer_C = np.full(len(ok), np.nan), {}
            for f in range(5):
                v = (f + 1) % 5
                trf = rows_train[fold[rows_train] != f]
                tri, vai = trf[fold[trf] != v], trf[fold[trf] == v]
                cr = {}
                for Cf in GRID:
                    sc, _ = fit(Xo, tri, y, Cf)
                    cr[Cf] = run_auc(key[vai], y[vai], sc(Xo[vai]))
                outer_C[f] = max(GRID, key=lambda c: (cr[c], -c))
                sc, _ = fit(Xo, trf, y, outer_C[f])
                te = np.flatnonzero(fold == f)
                off_scores[te] = sc(Xo[te])
            off_runs = {}
            for a in OFF_ARMS:
                i = np.flatnonzero(arm == a)
                u, first = np.unique(key[i], return_index=True)
                _, m = run_means(key[i], off_scores[i])
                off_runs[a] = {"keys": u.tolist(), "main_task": main[i][first].tolist(), "score": m.tolist()}
            u_on, m_on = run_means(on_run[nk], final(Xn))
            _, m_on0 = run_means(on_run[nk], fold0[C](Xn))
            cell = {"layer": layer, "kind": kind, "C": C, "C_choice_val_auroc": cres, "grad_max": g, "train_rows": int(len(rows_train)),
                    "train_runs": int(len(set(key[rows_train]))), "thresholds_fold0": thr, "off_outer_C": outer_C, "fold0_honest_runs": int(len(hon_scores)),
                    "on_runs": u_on.tolist(), "on_score": m_on.tolist(), "on_score_fold0_model": m_on0.tolist(), "off_outer": off_runs}
            if (layer, kind) == PRIMARY:  # the null: labels shuffled among each main task's real runs, full refit
                null = []
                runs_tr, first = np.unique(key[rows_train], return_index=True)
                lab = dict(zip(runs_tr, y[rows_train][first]))
                task_of = dict(zip(runs_tr, main[rows_train][first]))
                for b in range(N_NULL):
                    perm = {}
                    for t in np.unique([task_of[k] for k in runs_tr]):
                        ks = [k for k in runs_tr if task_of[k] == t]
                        for k, l in zip(ks, rng.permutation([lab[k] for k in ks])):
                            perm[k] = l
                    ys = y.copy()
                    ys[rows_train] = np.array([perm[k] for k in key[rows_train]])
                    tr_b, va_b = rows_train[fold[rows_train] != 0], rows_train[fold[rows_train] == 0]
                    cr = {}
                    for Cb in GRID:
                        sc, _ = fit(Xo, tr_b, ys, Cb)
                        cr[Cb] = run_auc(key[va_b], ys[va_b], sc(Xo[va_b]))
                    Cb = max(GRID, key=lambda c: (cr[c], -c))
                    sc, _ = fit(Xo, rows_train, ys, Cb)
                    null.append({"C": Cb, "on_score": run_means(on_run[nk], sc(Xn))[1].tolist()})
                    print(f"null {b}: C {Cb}", flush=True)
                cell["null"] = null
            out["cells"][f"L{layer}_{kind}"] = cell
            print(f"L{layer} {kind}: C {C}, val {cres}", flush=True)
        del Xo_all, Xn_all
        torch.cuda.empty_cache()
    for name, k in kept.items():  # score = ((x - mu) / sd) @ w + b on the float32 activations of that layer
        np.savez(d / f"probe_{name}.npz", w=k["w"], b=k["b"], mu=k["mu"], sd=k["sd"], C=k["C"])
    score_vol.commit()
    out["probes_saved"] = [f"{CACHE}/probe_{n}.npz" for n in kept]
    return out


@app.function(image=image, cpu=8, memory=32768, timeout=1800, retries=0, volumes=VOLS)
def score_swap() -> dict:
    import numpy as np

    score_vol.reload()
    root = Path("/on") / SWAP
    rows, picks = [], []
    for f in sorted((root / "meta").glob("*/*.json")):
        m = json.loads(f.read_text())
        for c in m["calls"]:
            if c.get("kind") in KINDS and c.get("row") is not None:
                rows.append({"arm": m["arm"], "task": m["task"], "kind": c["kind"]})
                picks.append((m["chunk"], m["row_start"] + c["row"]))
    X = read_layers(root, picks, LAYERS)
    kind = np.array([r["kind"] for r in rows])
    run = np.array([f"{r['arm']}|{r['task']}" for r in rows])
    out = {}
    for layer in LAYERS:
        for k in KINDS:
            p = np.load(Path("/score") / CACHE / f"probe_L{layer}_{k}.npz")
            i = np.flatnonzero(kind == k)
            s = ((X[layer][i].astype(np.float32) - p["mu"]) / p["sd"]) @ p["w"] + float(p["b"])
            u, inv = np.unique(run[i], return_inverse=True)
            out[f"L{layer}_{k}"] = {"runs": u.tolist(), "score": (np.bincount(inv, weights=s) / np.bincount(inv)).tolist(), "rows": int(len(i))}
    return out


@app.function(image=image, cpu=4, memory=32768, timeout=1800, retries=0, volumes=VOLS)
def per_output() -> dict:
    import numpy as np

    score_vol.reload()
    d = Path("/score") / CACHE
    cached = json.loads((d / "on_rows.json").read_text())
    on = Path("/on") / ON[1]
    rows = []  # the cache's own order and filters, plus each row's turn
    for f in sorted((on / "meta").glob("*/*.json")):
        m = json.loads(f.read_text())
        if m.get("redrawn_because") or m.get("attempt_kind") == "rerun":
            continue
        for c in m["calls"]:
            if c.get("kind") in KINDS and c.get("row") is not None:
                rows.append({"arm": m["arm"], "task": m["task"], "run": m["run"], "kind": c["kind"], "src_irid": c.get("src_irid"),
                             "call_index": c.get("call_index"), "label": c.get("label"), "turn": c.get("turn")})
    if [{k: v for k, v in r.items() if k != "turn"} for r in rows] != cached:
        raise ValueError("the rebuilt on-policy rows differ from the cache's")
    out = {"rows": [{k: r[k] for k in ("arm", "task", "kind", "turn", "call_index")} for r in rows], "scores": {}}
    kind = np.array([r["kind"] for r in rows])
    for layer in LAYERS:
        X = np.load(d / f"on_L{layer}.npy")
        for k in KINDS:
            p = np.load(d / f"probe_L{layer}_{k}.npz")
            s = np.full(len(rows), np.nan)
            i = np.flatnonzero(kind == k)
            s[i] = ((X[i].astype(np.float32) - p["mu"]) / p["sd"]) @ p["w"] + float(p["b"])
            out["scores"][f"L{layer}_{k}"] = [None if np.isnan(x) else float(x) for x in s]
    return out


@app.local_entrypoint()
def main(cache_only: bool = False, swap: bool = False, per_output_scores: bool = False):
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", str(HERE)], capture_output=True, text=True, cwd=HERE).stdout.strip())
    if per_output_scores:
        res = per_output.remote() | {"meta": {"git": git, "dirty": dirty, "cache": CACHE,
                                              "script_sha256": hashlib.sha256((HERE / Path(__file__).name).read_bytes()).hexdigest()}}
        stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
        path = HERE / "results" / f"attack_detector_per_output_{stamp}.json"
        with path.open("x") as fh:
            fh.write(json.dumps(res))
        print("->", path)
        return
    if swap:
        res = {"cells": score_swap.remote(), "meta": {"git": git, "dirty": dirty, "swap_extraction": SWAP,
                                                     "script_sha256": hashlib.sha256((HERE / Path(__file__).name).read_bytes()).hexdigest()}}
        stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
        path = HERE / "results" / f"attack_detector_swap_{stamp}.json"
        with path.open("x") as fh:
            fh.write(json.dumps(res))
        print("->", path)
        return
    info = cache.remote()
    print(json.dumps(info), flush=True)
    if cache_only:
        return
    res = fit_all.remote()
    res["cache"] = info
    res["meta"] = {"git": git, "dirty": dirty, "script_sha256": hashlib.sha256((HERE / Path(__file__).name).read_bytes()).hexdigest(),
                   "off": OFF, "on": ON, "layers": LAYERS, "kinds": KINDS, "C_grid": GRID, "primary": PRIMARY, "n_null": N_NULL}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"attack_detector_onpolicy_{stamp}.json"
    with path.open("x") as fh:
        fh.write(json.dumps(res))
    print("->", path)
