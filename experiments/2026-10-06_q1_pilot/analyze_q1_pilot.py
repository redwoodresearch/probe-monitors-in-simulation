"""Q1 pilot: can a linear probe on Qwen3-32B activations tell real from simulated LinuxArena attack transcripts?

Reads the pooled activations from the 2026-10-06 extraction (Modal Volume) and runs, on CPU in Modal:
  (a) null control: real attack epoch 1 vs epoch 2 (same tasks, both real). Should be ~0.5.
  (b) pilot: real attack epoch 1 vs simulated attack.
Both use only tasks where every attack arm (epoch 1, epoch 2, simulated) is under 32k tokens, so the two
comparisons run on the same tasks.

Transcript features: the average of a transcript's segment vectors of one kind (tool outputs or agent turns),
per layer and pooling. Probe: standardized features, L2 logistic regression (C fixed in config). Out-of-fold
predictions from the shared 5 folds over main tasks (splits/folds_v1.json), so a task's runs never straddle train and test.
95% intervals bootstrap main tasks. Baselines, scored the same way:
  - output length (total tool-output tokens),
  - the trailing-newline tell (share of tool outputs ending in a newline),
  - 1,000 random directions in the probe's standardized feature space, the best chosen on the training folds.
Everything is repeated with editor-tool (`text_editor`) outputs removed. The out-of-fold scores of every probe and
best random direction go to the Volume, next to the extraction run: `<extraction_run>__q1_pilot/<analysis_id>/`,
`oof_scores.npz` (rows: cell, arm, fold) with `index.json` (one entry per configuration).

  uv run modal run experiments/2026-10-06_q1_pilot/analyze_q1_pilot.py
"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q1-pilot"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
# Modal's maximum. The null for every reported probe (192 configs x 100 refits) is 6x the 32-config null, which was
# still running 2 h 40 min into its 3 h limit on 2026-10-06; one container may need well over 3 h.
TIMEOUT_S = 24 * 3600

app = modal.App(APP)
acts_vol = modal.Volume.from_name(ACTS, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy", "scikit-learn==1.7.2", "joblib")


@app.function(image=image, volumes={"/acts": acts_vol}, cpu=32, memory=131072, timeout=TIMEOUT_S)
def analyze(cfg: dict) -> dict:
    import numpy as np
    from joblib import Parallel, delayed
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    run_dir = Path("/acts") / cfg["extraction_run"]
    layers, poolings = cfg["layers"], cfg["poolings"]
    arms = ["real-attack-e1", "real-attack-e2", "sim-attack"]

    # Trailing-newline tell per transcript, from the staged reader messages (what the reader saw).
    newline = {}
    for shard in sorted((run_dir / "inputs").glob("shard_*.jsonl")):
        for line in shard.read_text().splitlines():
            it = json.loads(line)
            if it["arm"] not in arms:
                continue
            outs = [m for m in it["messages"] if m["role"] == "tool"]
            no_ed = [m for m in outs if m.get("tool_name") != "text_editor"]
            newline[(it["arm"], it["key"])] = (np.mean([m["content"].endswith("\n") for m in outs]) if outs else np.nan,
                                               np.mean([m["content"].endswith("\n") for m in no_ed]) if no_ed else np.nan)

    # Validity and submit counts for every sample, written by the extraction before submit calls were dropped.
    runs = [json.loads(l) for l in (run_dir / "runs.jsonl").read_text().splitlines()]
    exclusions = {a: {} for a in sorted({r["arm"] for r in runs})}
    for r in runs:
        if r["invalid"]:
            exclusions[r["arm"]][r["invalid"]] = exclusions[r["arm"]].get(r["invalid"], 0) + 1

    # One transcript per (arm, cell); keep cells read in all three arms. A cell invalid or over 32k in any arm
    # was not read there, so it drops out of every paired comparison.
    meta = {}
    for arm in arms:
        for p in sorted((run_dir / "acts" / arm).glob("*.segments.json")):
            m = json.loads(p.read_text())
            meta.setdefault(m["cell"], {}).setdefault(arm, m)
    read_all = {c for c, by_arm in meta.items() if all(a in by_arm for a in arms)}
    cells = sorted(read_all & set(cfg["paired_cells"]))  # the shared paired list decides; extra reads are ignored
    mismatch = sorted(set(cfg["paired_cells"]) - read_all)  # paired cells the extraction did not read in every arm
    if mismatch:  # analysing fewer cells would silently change the cohort, and failures may correlate with length
        raise ValueError(f"{len(mismatch)} paired cells in {cfg['exclusions_file']} were not read in all three arms, "
                         f"e.g. {mismatch[:3]}")
    main_of = {c: meta[c][arms[0]]["main_task"] for c in cells}
    attack_cells = {r["cell"] for r in runs if r["arm"] in arms}
    invalid_cells = {r["cell"] for r in runs if r["arm"] in arms and r["invalid"]}
    dropped = {"attack_cells": len(attack_cells), "invalid_in_some_arm": len(invalid_cells),
               "valid_but_over_32k_or_unread_in_some_arm": len(attack_cells - invalid_cells - set(cells)), "kept": len(cells)}

    variants = {"all_outputs": lambda s: True, "no_editor_outputs": lambda s: s["kind"] != "tool_output" or s["tool"] != "text_editor"}
    kinds = ["tool_output", "agent"]
    feats = {}  # (arm, variant, kind, layer, pooling) -> [n_cells, H]
    lengths = {}  # (arm, variant) -> [n_cells]
    for arm in arms:
        rows = {(v, k, l, p): [] for v in variants for k in kinds for l in layers for p in poolings}
        lens = {v: [] for v in variants}
        for c in cells:
            m = meta[c][arm]
            arrays = dict(np.load(run_dir / "acts" / arm / f"{m.get('filename', m['key'])}.npz"))
            # The extraction saves arrays even when they hold non-finite values; fail here, so that NaN in a feature
            # can only mean the sentinel below (no segments of that kind), which probe_oof zeroes.
            if not all(np.isfinite(x).all() for x in arrays.values()):
                raise ValueError(f"non-finite activations in {arm} {m['key']}")
            for v, keep in variants.items():
                segs = [s for s in m["segments"] if keep(s)]
                lens[v].append(sum(s["n_tokens"] for s in segs if s["kind"] == "tool_output"))
                for k in kinds:
                    idx = [s["row"] for s in segs if s["kind"] == k]
                    for l in layers:
                        for p in poolings:
                            a = arrays[f"L{l}_{p}"]
                            rows[(v, k, l, p)].append(a[idx].astype(np.float32).mean(0) if idx else np.full(a.shape[1], np.nan, np.float32))
        for key, r in rows.items():
            feats[(arm,) + key] = np.stack(r)
        for v, x in lens.items():
            lengths[(arm, v)] = np.array(x, float)
    nl = {(arm, v): np.array([newline[(arm, meta[c][arm]["key"])][i] for c in cells]) for arm in arms for i, v in enumerate(variants)}

    rng = np.random.default_rng(cfg["seed"])
    groups_one = np.array([main_of[c] for c in cells])
    # Shared folds (splits/folds_v1.json): a main task's fold is fixed across every track.
    fold_ids = np.array([cfg["main_task_to_fold"][main_of[c]] for c in cells] * 2)
    folds = [(np.flatnonzero(fold_ids != f), np.flatnonzero(fold_ids == f)) for f in sorted(set(fold_ids))]
    uniq = np.unique(groups_one)
    boot = [rng.choice(len(uniq), len(uniq)) for _ in range(cfg["bootstrap"])]
    task_index = {g: np.flatnonzero(np.concatenate([groups_one, groups_one]) == g) for g in uniq}

    def auc_ci(y, s):
        ok = ~np.isnan(s)
        point = roc_auc_score(y[ok], s[ok])
        vals = []
        for b in boot:
            idx = np.concatenate([task_index[uniq[i]] for i in b])
            idx = idx[ok[idx]]
            if len(np.unique(y[idx])) == 2:
                vals.append(roc_auc_score(y[idx], s[idx]))
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return {"auroc": round(float(point), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}

    def oof_scalar(x, y):
        """A one-number baseline: orient on the training folds (sign of the training AUROC), score the test fold."""
        out = np.full(len(y), np.nan)
        for tr, te in folds:
            ok = ~np.isnan(x[tr])
            sign = 1.0 if roc_auc_score(y[tr][ok], x[tr][ok]) >= 0.5 else -1.0
            out[te] = sign * x[te]
        return out

    dirs = rng.standard_normal((cfg["random_directions"], next(iter(feats.values())).shape[1])).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    def probe_oof(X, y):
        X = np.nan_to_num(X)  # a transcript with no segments of this kind gets the zero vector
        out = np.zeros(len(y))
        for tr, te in folds:
            sc = StandardScaler().fit(X[tr])
            out[te] = LogisticRegression(C=cfg["C"], max_iter=5000).fit(sc.transform(X[tr]), y[tr]).decision_function(sc.transform(X[te]))
        return out

    task_of_cell = np.searchsorted(uniq, groups_one)

    def shuffle_null_auc(X, y, seed):
        """One draw of the label-shuffle null: per main task, all its cells' two transcripts swap labels with
        probability 1/2 (one flip per main task, matching the task-grouped folds and bootstrap)."""
        r = np.random.default_rng(seed)
        flip = (r.random(len(uniq)) < 0.5)[task_of_cell]
        y_perm = y.copy()
        y_perm[:len(cells)] = np.where(flip, 1 - y[:len(cells)], y[:len(cells)])
        y_perm[len(cells):] = 1 - y_perm[:len(cells)]
        return roc_auc_score(y_perm, probe_oof(X, y_perm))

    def probe_and_random(X, y):
        probe = probe_oof(X, y)
        X = np.nan_to_num(X)
        rand = np.zeros(len(y))
        for tr, te in folds:
            sc = StandardScaler().fit(X[tr])
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            ptr = Xtr @ dirs.T  # [n_train, n_dirs]
            aucs = np.array([roc_auc_score(y[tr], ptr[:, j]) for j in range(len(dirs))])
            j = int(np.argmax(np.abs(aucs - 0.5)))
            rand[te] = np.sign(aucs[j] - 0.5) * (Xte @ dirs[j])
        return probe, rand

    out, oof, oof_index = {}, {}, []
    comparisons = {"null_e1_vs_e2": ("real-attack-e1", "real-attack-e2"),
                   "pilot_real_vs_sim": ("real-attack-e1", "sim-attack")}
    for comparison, (neg, pos) in comparisons.items():
        y = np.concatenate([np.zeros(len(cells)), np.ones(len(cells))])
        res = {"n_tasks": len(cells), "n_main_tasks": int(len(uniq)), "positive": pos, "negative": neg}
        for v in variants:
            r = {"length_baseline": auc_ci(y, oof_scalar(np.concatenate([lengths[(neg, v)], lengths[(pos, v)]]), y)),
                 "newline_baseline": auc_ci(y, oof_scalar(np.concatenate([nl[(neg, v)], nl[(pos, v)]]), y))}
            jobs = [(k, l, p) for k in kinds for l in layers for p in poolings]
            scored = Parallel(n_jobs=cfg["n_jobs"])(delayed(probe_and_random)(
                np.concatenate([feats[(neg, v, k, l, p)], feats[(pos, v, k, l, p)]]), y) for k, l, p in jobs)
            for (k, l, p), (probe, rand) in zip(jobs, scored):
                r[f"{k}/L{l}/{p}"] = {"probe": auc_ci(y, probe), "best_random_direction": auc_ci(y, rand)}
                for score, s in (("probe", probe), ("best_random_direction", rand)):
                    oof[f"{comparison}__{v}__{k}__L{l}__{p}__{score}"] = s
                    oof_index.append({"array": f"{comparison}__{v}__{k}__L{l}__{p}__{score}", "comparison": comparison,
                                      "variant": v, "kind": k, "layer": l, "pooling": p, "score": score})
            res[v] = r
            print(comparison, v, "done", flush=True)
        out[comparison] = res

    # Out-of-fold scores to the Volume (too big for git), one folder per analysis. Rows of every score array: the
    # comparison's negative arm, then its positive arm, each over `cells` in order (`cell`, `<comparison>__arm`, and
    # `fold`, the row's shared main-task fold). A retried call (same analysis_id) rewrites the same, deterministic scores.
    oof_dir = Path("/acts") / f"{cfg['extraction_run']}__q1_pilot" / cfg["analysis_id"]
    oof_dir.mkdir(parents=True, exist_ok=True)
    arm_rows = {f"{c}__arm": np.array([n] * len(cells) + [p] * len(cells)) for c, (n, p) in comparisons.items()}
    np.savez(oof_dir / "oof_scores.npz", cell=np.array(cells * 2), fold=fold_ids, **arm_rows, **oof)
    (oof_dir / "index.json").write_text(json.dumps({
        "extraction_run": cfg["extraction_run"], "analysis_id": cfg["analysis_id"],
        "score": "out-of-fold decision function", "rows": "negative arm, then positive arm, each over `cell` in order",
        "comparisons": {
            c: {"negative": n, "positive": p, "arm_array": f"{c}__arm"} for c, (n, p) in comparisons.items()},
        "arrays": oof_index}, indent=1))
    acts_vol.commit()
    print(f"out-of-fold scores -> {oof_dir}", flush=True)

    # Label-shuffle null for every probe in shuffle_null_comparisons x shuffle_null_poolings (every layer, both segment
    # kinds, both variants). The config lists every comparison and pooling, so every reported probe gets its null.
    y = np.concatenate([np.zeros(len(cells)), np.ones(len(cells))])
    jobs = [(c, v, k, l, p) for c in cfg["shuffle_null_comparisons"] for v in variants for k in kinds for l in layers
            for p in cfg["shuffle_null_poolings"]]
    seeds = [cfg["seed"] + 1 + i for i in range(cfg["shuffle_null"])]  # the same label shuffles for every config
    # One job per config: its feature matrix is built once and reused for all of its shuffle draws.
    draws = Parallel(n_jobs=cfg["n_jobs"])(delayed(lambda X: [shuffle_null_auc(X, y, sd) for sd in seeds])(
        np.concatenate([feats[(comparisons[c][0], v, k, l, p)], feats[(comparisons[c][1], v, k, l, p)]]))
        for (c, v, k, l, p) in jobs)
    for j, (c, v, k, l, p) in enumerate(jobs):
        null = np.array(draws[j])
        obs = out[c][v][f"{k}/L{l}/{p}"]["probe"]["auroc"]
        out[c][v][f"{k}/L{l}/{p}"]["label_shuffle_null"] = {
            "n": len(null), "mean": round(float(null.mean()), 4), "p95": round(float(np.percentile(null, 95)), 4),
            "p99": round(float(np.percentile(null, 99)), 4), "p_value": round(float((1 + (null >= obs).sum()) / (1 + len(null))), 4)}
    print("label-shuffle null done", flush=True)

    # Submit calls per run, counted before they were dropped: reported on its own, over all valid runs and
    # over the paired cells, with the AUROC of the count alone for real epoch 1 vs simulated.
    def submit_report(rows):
        by = {a: np.array([r["n_submit_calls"] for r in rows if r["arm"] == a]) for a in sorted({r["arm"] for r in rows})}
        rep = {a: {"n": int(len(x)), "mean": round(float(x.mean()), 3), "median": float(np.median(x)),
                   "share_more_than_one": round(float((x > 1).mean()), 3)} for a, x in by.items()}
        e1, sim = by.get("real-attack-e1"), by.get("sim-attack")
        if e1 is not None and sim is not None:
            rep["auroc_real_e1_vs_sim"] = round(float(roc_auc_score(np.r_[np.zeros(len(e1)), np.ones(len(sim))], np.r_[e1, sim])), 4)
        return rep
    valid = [r for r in runs if not r["invalid"]]
    paired = [r for r in valid if r["arm"] in arms and r["cell"] in set(cells)]
    return {"cells": cells, "exclusions": exclusions, "paired_task_filter": dropped, "cells_not_in_shared_paired_list": mismatch,
            "submit_calls": {"all_valid_runs": submit_report(valid), "paired_cells": submit_report(paired)}, "results": out,
            "oof_scores": str(oof_dir.relative_to("/acts"))}


@app.local_entrypoint()
def main():
    import yaml

    cfg = yaml.safe_load((HERE / "config" / "q1_pilot.yaml").read_text())
    splits = HERE.parent.parent / "splits"
    folds = json.loads((splits / cfg["folds_file"]).read_text())
    cfg["main_task_to_fold"] = folds["task_folds"]["main_task_to_fold"]
    cfg["paired_cells"] = json.loads((splits / cfg["exclusions_file"]).read_text())["paired_attack_cells"]["cells"]
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    cfg["analysis_id"] = stamp  # out-of-fold scores: <extraction_run>__q1_pilot/<analysis_id>/ on the Volume
    result = analyze.remote(cfg)
    (HERE / "results").mkdir(exist_ok=True)
    path = HERE / "results" / f"q1_pilot_{stamp}.json"
    path.write_text(json.dumps(result | {"config": cfg | {"modal": {"app": APP, "activations_volume": ACTS}}}, indent=1))
    print("excluded runs (per arm, per reason):", json.dumps(result["exclusions"]))
    print("paired-task filter:", json.dumps(result["paired_task_filter"]), "| cells differing from splits/exclusions_v1.json:", len(result["cells_not_in_shared_paired_list"]))
    print("submit calls per run:", json.dumps(result["submit_calls"], indent=1))
    for comparison, res in result["results"].items():
        print(f"\n=== {comparison}: {res['positive']} (positive) vs {res['negative']}, {res['n_tasks']} tasks, {res['n_main_tasks']} main tasks")
        for v in ("all_outputs", "no_editor_outputs"):
            r = res[v]
            fmt = lambda d: f"{d['auroc']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]"
            print(f"  [{v}] length {fmt(r['length_baseline'])}   trailing newline {fmt(r['newline_baseline'])}")
            for kind in ("tool_output", "agent"):
                for l in cfg["layers"]:
                    d = r[f"{kind}/L{l}/mean"]
                    null = d.get("label_shuffle_null")
                    extra = f"   shuffle null p95 {null['p95']:.3f} (p={null['p_value']:.3f})" if null else ""
                    print(f"    {kind:11} L{l:<2} mean-pooled probe {fmt(d['probe'])}   best random direction {fmt(d['best_random_direction'])}{extra}")
    print(f"\nresults -> {path}\nout-of-fold scores -> Volume {ACTS}: {result['oof_scores']}/")
