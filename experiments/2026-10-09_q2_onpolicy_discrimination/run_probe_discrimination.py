"""Real vs simulated discrimination on the on-policy rerun with a probe on Qwen3.5-27B's own activations (Modal CPU).

Features: q3's cache of the strict rerun's segment-end rows (Volume probe-shift_aswin-surya_sub-q3onpol_score_kill-
2026-11-08, attackdet_cache_v1/: on_L{39,51,63}.npy, fp16 [row, 5120], with on_rows.json; read-only here). Errored
runs (the sample's error set; 2 runs, the same ones the TF-IDF test drops) are left out. Comparisons and folds as
run_tfidf_discrimination.py. Per layer and readout (output_end, call_end): each row is labelled by its run's world; a
logistic regression on standardized features (C = 0.01, fit on four folds, scored on the fifth); run-level AUROC from
each run's mean row score. Intervals: 2,000 task resamples of the fixed scores (seed 0). Null: 20 label swaps within
task pairs, refitting (p = (1 + draws >= AUROC) / 21). Planted check (the positive control): on the real-vs-real
reference, the second world's rows are moved by 0.5 SD (of all rows' projections) along one fixed random direction
(seed 0) before fitting, so one row alone separates the worlds at about 0.64 AUROC; the method should flag it. The
inserted-line plant of the TF-IDF tests needs activations of edited transcripts (a GPU re-extraction) and is not run.
The 36 fits run in parallel, one thread each (the first launch, 2026-10-09 16:52 PT, left Modal's BLAS thread
settings in place under 30 workers and was stopped at 120% of its estimate with no output); each opens only its own
layer's rows from the Volume, prints each finished fold, and the driver prints each finished fit's result in full.
With 20 null draws the smallest p is 1/21, and the 30 comparison cells are not corrected for multiplicity.

  uv run modal run experiments/2026-10-09_q2_onpolicy_discrimination/run_probe_discrimination.py --folds-json <path> --errored-json <path>
"""
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-onpol-disc_probe_kill-2026-11-08"
PAIRS = {"real honest_b vs sim_honest@september": ("honest_b", "sim_honest@september"),
         "real honest_b vs sim_honest@today": ("honest_b", "sim_honest@today"),
         "real attack vs sim_attack@september": ("attack", "sim_attack@september"),
         "real attack vs sim_attack@today": ("attack", "sim_attack@today"),
         "real honest_a vs real honest_b (reference)": ("honest_a", "honest_b")}
app = modal.App(APP)
vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-q3onpol_score_kill-2026-11-08")
image = modal.Image.debian_slim(python_version="3.12").apt_install("procps").uv_pip_install("numpy", "scikit-learn==1.7.2", "joblib")


@app.function(image=image, volumes={"/q3": vol}, cpu=32, memory=32768, timeout=3600, retries=0)
def discriminate(folds: dict, errored: list) -> dict:
    import time

    import numpy as np
    from joblib import Parallel, delayed, parallel_config
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    rows = json.loads(Path("/q3/attackdet_cache_v1/on_rows.json").read_text())
    keep = np.array([r["run"] not in set(errored) for r in rows])
    R = {k: np.array([r[k] for r in rows]) for k in ("kind", "arm", "task", "run")}

    def select(a, b, kind):  # small arrays only: each job is sent its rows' indices and labels, not the features
        idx = np.flatnonzero(keep & (R["kind"] == kind) & np.isin(R["arm"], [a, b]))
        y = (R["arm"][idx] == b).astype(int); task = R["task"][idx]
        both = {t for t in np.unique(task) if len(set(y[task == t])) == 2}  # tasks with a usable run in each world
        m = np.isin(task, list(both))
        return idx[m], y[m], task[m], R["run"][idx][m]

    def one(name, L, kind, idx, y, task, run, plant=0.0):
        t0 = time.time()
        fold = np.array([folds[t] for t in task])
        F = np.asarray(np.load(f"/q3/attackdet_cache_v1/on_L{L}.npy", mmap_mode="r")[idx], dtype=np.float32)
        if plant:  # move the second world along a fixed random direction by plant x the SD of the rows' projections on it
            u = np.random.default_rng(0).standard_normal(F.shape[1]).astype(np.float32); u /= np.linalg.norm(u)
            F[y == 1] += plant * float((F @ u).std()) * u
        rng = np.random.default_rng(0)
        tasks = np.unique(task)
        perms = [y] + [np.array([1 - v if s[t] else v for v, t in zip(y, task)]) for s in ({t: rng.random() < 0.5 for t in tasks} for _ in range(20))]
        scores = [np.zeros(len(y)) for _ in perms]
        for k in range(5):
            tr, te = fold != k, fold == k
            sc = StandardScaler().fit(F[tr]); Ftr, Fte = sc.transform(F[tr]), sc.transform(F[te])
            for s_, lab in zip(scores, perms):
                s_[te] = LogisticRegression(C=0.01, max_iter=2000).fit(Ftr, lab[tr]).decision_function(Fte)
            print(f"{name} L{L} {kind}: fold {k} done at {time.time() - t0:.0f} s ({len(y)} rows)", flush=True)
        runs = np.unique(run)
        def run_auc(s_, lab):
            return roc_auc_score([lab[run == r][0] for r in runs], [s_[run == r].mean() for r in runs])
        auc = run_auc(scores[0], y)
        null = [run_auc(s_, lab) for s_, lab in zip(scores[1:], perms[1:])]
        rs = {r: scores[0][run == r].mean() for r in runs}; ry = {r: y[run == r][0] for r in runs}; rt = {r: task[run == r][0] for r in runs}
        by = {t: [r for r in runs if rt[r] == t] for t in tasks}
        boot = []
        for _ in range(2000):
            rr = [r for t in rng.choice(tasks, len(tasks)) for r in by[t]]
            if len({ry[r] for r in rr}) == 2:
                boot.append(roc_auc_score([ry[r] for r in rr], [rs[r] for r in rr]))
        return name, L, kind, {"run-level AUROC": round(float(auc), 4), "95% (task resamples)": [round(float(x), 4) for x in np.percentile(boot, [2.5, 97.5])],
                               "null mean / max": [round(float(np.mean(null)), 4), round(float(np.max(null)), 4)],
                               "p": round((1 + sum(n >= auc for n in null)) / (1 + len(null)), 4), "tasks": len(tasks), "rows": len(y),
                               "row-level AUROC": round(float(roc_auc_score(y, scores[0])), 4), "seconds": round(time.time() - t0)}

    jobs = [(n, L, k, *select(a, b, k)) for n, (a, b) in PAIRS.items() for L in (39, 51, 63) for k in ("output_end", "call_end")]
    ref = "real honest_a vs real honest_b (reference)"
    jobs += [(f"{ref}, planted 0.5 SD shift", L, k, *select(*PAIRS[ref], k), 0.5) for L in (39, 51, 63) for k in ("output_end", "call_end")]
    print(len(jobs), "fits; rows per fit:", sorted({len(j[3]) for j in jobs}), flush=True)
    out = {}
    with parallel_config(backend="loky", inner_max_num_threads=1):
        for name, L, kind, r in Parallel(n_jobs=len(jobs), return_as="generator_unordered")(delayed(one)(*j) for j in jobs):
            out.setdefault(name, {})[f"L{L} {kind}"] = r
            print(f"done {sum(len(v) for v in out.values())}/{len(jobs)}: {name} L{L} {kind} {json.dumps(r)}", flush=True)
    return {"rows": len(rows), "errored runs left out": errored, "comparisons": out}


@app.local_entrypoint()
def main(folds_json: str, errored_json: str):
    folds, errored = json.loads(Path(folds_json).read_text()), json.loads(Path(errored_json).read_text())
    res = discriminate.remote(folds, errored)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = HERE / "results" / f"probe_discrimination_{stamp}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n")
    for name, d in res["comparisons"].items():
        for k, r in d.items():
            print(f"{name} | {k} | AUROC {r['run-level AUROC']} {r['95% (task resamples)']} | null {r['null mean / max']} p {r['p']} | {r['tasks']} tasks, {r['rows']} rows")
    print(out)
