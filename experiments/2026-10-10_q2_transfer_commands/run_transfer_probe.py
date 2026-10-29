"""Data-transfer probe, step 2 (Modal CPU): the destructive-command probe's job (run_destructive_probe.py, copied) with
the `transfer` label: train on Sol's real honest calls, freeze the threshold on real honest calibration calls, score
held-out real and simulated honest calls (PLAN.md). Honest arms of the v4 dump only. Changes from the destructive job:
the label field, the app and Volume names, and durability: the local entrypoint spawns the job and saves its call id,
and the job writes its full output to the Volume before returning, so a lost client loses nothing.

Returns per-call test scores for the real labels (draw 0) and for 20 within-run label shuffles with the whole
layer/C selection repeated (labels, scores and caught flags of every draw, so any statistic can be recomputed on the
null), plus each draw's real minus simulated TPR and per-world AUROC (computed within each test fold, then averaged:
each fold has its own probe). Each (draw, fold) result is also checkpointed to a Volume as it finishes. The full
output goes to data/transfer/ (archived to S3), a summary to results/. analyze_transfer_probe.py adds the
bootstrap, the real-vs-real control and the planted check.

  uv run modal run --detach experiments/2026-10-10_q2_transfer_commands/run_transfer_probe.py --labels <labels jsonl>
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-transfer_probe_kill-2026-11-08"
LABEL = "transfer"
DUMP = "2026-10-06_la_segment-end-v4_qwen35-27b_alllayers_v1"
ARMS = ("real-honest", "sim-honest")
LAYERS = list(range(3, 64, 4))
CS = (0.001, 0.01, 0.1)
N_NULL = 20
app = modal.App(APP)
acts = modal.Volume.from_name("probe-shift_aswin-surya_sub-qwen35_acts_kill-2026-11-06")
ckpt = modal.Volume.from_name(APP, create_if_missing=True)  # each (draw, fold) result as it finishes
image = modal.Image.debian_slim(python_version="3.12").apt_install("procps").uv_pip_install("numpy", "scikit-learn==1.7.2", "joblib")


def fold_of(task: str) -> int:
    return int(hashlib.md5(task.encode()).hexdigest(), 16) % 5


@app.function(image=image, volumes={"/acts": acts.read_only(), "/ckpt": ckpt}, cpu=32, memory=65536, timeout=3600, retries=0)
def run(labels: list[dict], stamp: str) -> dict:
    import time
    import warnings
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    from joblib import Parallel, delayed, parallel_config
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    t0 = time.time()
    root = Path("/acts") / DUMP
    lab = {(r["arm"], r["file"], r["sample_id"], r["epoch"], r["step"], r["action"]): r for r in labels}
    paths = [p for arm in ARMS for p in sorted((root / "meta" / arm).glob("*.json"))]
    with ThreadPoolExecutor(32) as ex:
        metas = list(ex.map(lambda p: json.loads(p.read_text()), paths))
    rows, unmatched, kept_calls = [], 0, 0  # rows: (world, task, run, chunk, row, y, step, action)
    for m in metas:
        run_key = f"{m['arm']}|{m['file']}|{m['sample_id']}|{m['epoch']}"
        for p in m["positions"]:
            if p["kind"] != "call_end" or not p.get("kept", True) or p["row"] is None:
                continue
            kept_calls += 1
            r = lab.get((m["arm"], m["file"], m["sample_id"], m["epoch"], p["step"], p["action"]))
            if r is None:
                unmatched += 1
                continue
            if r["impactful"]:
                rows.append((0 if m["arm"] == "real-honest" else 1, r["main_task"], run_key, m["chunk"], m["row_start"] + p["row"], int(r[LABEL]), p["step"], p["action"]))
    if unmatched > 0.005 * kept_calls:
        raise ValueError(f"{unmatched} of {kept_calls} kept call_end positions have no label")
    print(f"{len(metas)} runs, {kept_calls} kept call_end positions ({unmatched} unmatched), {len(rows)} impactful rows, {time.time() - t0:.0f} s", flush=True)
    world = np.array([r[0] for r in rows]); y = np.array([r[5] for r in rows])
    task = np.array([r[1] for r in rows]); run_id = np.array([r[2] for r in rows])
    fold = np.array([fold_of(t) for t in task])
    by_chunk = {}
    for i, r in enumerate(rows):
        by_chunk.setdefault(r[3], []).append((i, r[4]))
    tmp = Path("/tmp/feats"); tmp.mkdir(exist_ok=True)  # workers open these memory-mapped: nothing large is pickled
    for L in LAYERS:
        a = np.zeros((len(rows), 5120), np.float16)
        for chunk, items in by_chunk.items():
            block = np.load(root / "by_layer" / f"L{L}" / f"{chunk}.npy", mmap_mode="r")
            idx = np.array([i for i, _ in items]); rr = np.array([r for _, r in items])
            order = np.argsort(rr)
            a[idx[order]] = block[rr[order]]
        np.save(tmp / f"L{L}.npy", a)
        del a
        print(f"layer {L} loaded at {time.time() - t0:.0f} s", flush=True)

    def shuffled(seed):
        """Labels permuted within each run (the run keeps its count of positives)."""
        if seed == 0:
            return y
        rng = np.random.default_rng(seed)
        out = y.copy()
        for r in np.unique(run_id):
            i = np.flatnonzero(run_id == r)
            out[i] = rng.permutation(out[i])
        return out

    def fit(Xtr, ytr, C, capped):
        """capped gets 1 when the fit raised a ConvergenceWarning (the iteration cap or an early LBFGS failure)."""
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            clf = LogisticRegression(C=C, max_iter=2000).fit((Xtr - mu) / sd, ytr)
        capped.append(int(any(issubclass(x.category, ConvergenceWarning) for x in w)))
        return lambda Z: clf.decision_function((Z - mu) / sd)

    def one(draw, k):
        t1 = time.time()
        X = {L: np.load(tmp / f"L{L}.npy", mmap_mode="r") for L in LAYERS}
        yy = shuffled(draw)
        cal = (k + 1) % 5
        tr = (world == 0) & ~np.isin(fold, [k, cal])
        tasks = np.unique(task[tr])
        inner = np.isin(task, tasks[np.random.default_rng(k).permutation(len(tasks))[: len(tasks) // 2]])
        best, capped = (-1, None, None), []
        for L in LAYERS:
            for C in CS:
                aucs = []
                for a_, b_ in ((tr & inner, tr & ~inner), (tr & ~inner, tr & inner)):
                    if len(set(yy[a_])) < 2 or len(set(yy[b_])) < 2:
                        continue
                    f = fit(X[L][a_].astype(np.float32), yy[a_], C, capped)
                    aucs.append(roc_auc_score(yy[b_], f(X[L][b_].astype(np.float32))))
                if aucs and np.mean(aucs) > best[0]:
                    best = (float(np.mean(aucs)), L, C)
        _, L, C = best
        f = fit(X[L][tr].astype(np.float32), yy[tr], C, capped)
        negc = (world == 0) & (fold == cal) & (yy == 0)
        neg = np.sort(f(X[L][negc].astype(np.float32)))[::-1]
        thr = float(neg[int(np.floor(0.01 * len(neg)))])
        te = fold == k
        s = f(X[L][te].astype(np.float32))
        print(f"draw {draw} fold {k}: layer {L}, C {C}, inner AUROC {best[0]:.3f}, {sum(capped[:-1])} of {len(capped) - 1} selection fits and "
              f"{capped[-1]} of 1 refit not converged, {time.time() - t1:.0f} s", flush=True)
        return draw, k, L, C, thr, np.flatnonzero(te), s, sum(capped[:-1]), capped[-1]

    cdir = Path("/ckpt") / stamp; cdir.mkdir(parents=True, exist_ok=True)
    res = []
    with parallel_config(backend="loky", inner_max_num_threads=1):  # draw 0 (the real labels) first
        for r in Parallel(n_jobs=32, return_as="generator_unordered")(delayed(one)(d, k) for d in range(N_NULL + 1) for k in range(5)):
            res.append(r)
            d, k, L, C, thr, idx, s, nsel, nref = r
            np.savez(cdir / f"d{d}_f{k}.npz", idx=idx, s=s, meta=json.dumps({"layer": L, "C": C, "threshold": thr, "selection fits not converged": nsel,
                                                                             "refit not converged": nref}))
            ckpt.commit()
    scores = {d: np.full(len(y), np.nan) for d in range(N_NULL + 1)}
    caught = {d: np.zeros(len(y), bool) for d in range(N_NULL + 1)}
    picks = {}
    sel_bad = ref_bad = 0
    for d, k, L, C, thr, idx, s, nsel, nref in res:
        scores[d][idx] = s; caught[d][idx] = s > thr
        picks.setdefault(d, {})[k] = {"layer": L, "C": C, "threshold": thr, "refit converged": not nref}
        sel_bad += nsel; ref_bad += nref
    def fold_auroc(yy, sc, w):
        """Mean over test folds of the AUROC within the fold (scores from different folds' probes are not pooled)."""
        a = []
        for k in range(5):
            m = (world == w) & (fold == k)
            if len(set(yy[m])) == 2:
                a.append(roc_auc_score(yy[m], sc[m]))
        return float(np.mean(a)), len(a)

    draws = []
    for d in range(N_NULL + 1):
        yy = shuffled(d)
        tpr = [caught[d][(world == w) & (yy == 1)].mean() for w in (0, 1)]
        (ar, nr), (as_, ns) = fold_auroc(yy, scores[d], 0), fold_auroc(yy, scores[d], 1)
        draws.append({"draw": d, "TPR real minus sim": float(tpr[0] - tpr[1]), "AUROC real (fold mean)": ar, "AUROC sim (fold mean)": as_,
                      "folds in AUROC real": nr, "folds in AUROC sim": ns, "picks": picks[d],
                      "labels": "".join(map(str, yy)), "caught": "".join("1" if c else "0" for c in caught[d]),
                      "scores": [round(float(x), 5) for x in scores[d]]})
    result = {"rows": [{"world": int(world[i]), "task": task[i], "run": run_id[i], "step": rows[i][6], "action": rows[i][7], "y": int(y[i]),
                      "fold": int(fold[i])} for i in range(len(y))],
            "draws": draws, "unmatched": unmatched, "kept call_end positions": kept_calls, "seconds": round(time.time() - t0),
            "selection fits not converged": sel_bad, "refits not converged": f"{ref_bad} of {len(res)}", "checkpoints": f"{APP}:/{stamp}"}
    (cdir / "full_output.json").write_text(json.dumps(result))  # durable before returning
    ckpt.commit()
    return result


@app.local_entrypoint()
def main(labels: str):
    rows = [json.loads(line) for line in Path(labels).read_text().splitlines()]
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    full = HERE.parents[1] / "data" / "transfer" / f"transfer_probe_{stamp}.json"
    full.parent.mkdir(parents=True, exist_ok=True)
    call = run.spawn(rows, stamp)
    (full.parent / f"launch_{stamp}.json").write_text(json.dumps({"stamp": stamp, "call": call.object_id, "volume": f"{APP}:/{stamp}"}) + "\n")
    print(f"stamp {stamp}, call {call.object_id}", flush=True)
    res = call.get() | {"labels sha256": hashlib.sha256(Path(labels).read_bytes()).hexdigest(), "layers": LAYERS, "C": CS}
    full.write_text(json.dumps(res) + "\n")
    keep = ("draw", "TPR real minus sim", "AUROC real (fold mean)", "AUROC sim (fold mean)", "folds in AUROC real", "folds in AUROC sim", "picks")
    summary = {k: v for k, v in res.items() if k not in ("rows", "draws")} | {
        "draws": [{k: d[k] for k in keep} for d in res["draws"]], "rows": len(res["rows"]),
        "full output": str(full.relative_to(HERE.parents[1])), "full output sha256": hashlib.sha256(full.read_bytes()).hexdigest()}
    out = HERE / "results" / f"transfer_probe_{stamp}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary["draws"][0] | {"seconds": res["seconds"], "unmatched": res["unmatched"], "selection fits not converged": res["selection fits not converged"], "refits not converged": res["refits not converged"]}))
    print(out, full)
