"""Selection-aware shuffled-label null for the corrected destructive probe (Modal CPU; Aswin 2026-10-10, "Spend it, cap
$6.50"). run_separate_calibration.py's pipeline (copied) on 20 within-run shuffles of the `destructive_raw` labels
(seeds 1-20; each run keeps its count of destructive calls), with the layer and C search redone in every draw. Per draw
and fold it keeps the test scores and the calibration fold's real-negative scores (negatives under that draw's labels),
so analyze_selection_null.py can compute the same metrics as the real-label refit. Each (draw, fold) is checkpointed;
the full output is written to the Volume before returning.

  uv run modal run --detach experiments/2026-10-10_q2_metric_table/run_selection_null.py --labels <labels jsonl>
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-recal-null_kill-2026-11-08"
DUMP = "2026-10-06_la_segment-end-v4_qwen35-27b_alllayers_v1"
ARMS = ("real-honest", "sim-honest")
LAYERS = list(range(3, 64, 4))
CS = (0.001, 0.01, 0.1)
LABEL = "destructive_raw"
DRAWS = range(1, 21)
app = modal.App(APP)
acts = modal.Volume.from_name("probe-shift_aswin-surya_sub-qwen35_acts_kill-2026-11-06")
ckpt = modal.Volume.from_name(APP, create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").apt_install("procps").uv_pip_install("numpy", "scikit-learn==1.7.2", "joblib")


def fold_of(task: str) -> int:
    return int(hashlib.md5(task.encode()).hexdigest(), 16) % 5


@app.function(image=image, volumes={"/acts": acts.read_only(), "/ckpt": ckpt}, cpu=32, memory=65536, timeout=5400, retries=0)
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
    rows, unmatched, kept_calls = [], 0, 0  # (world, task, run, chunk, row, labels, step, action)
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
                rows.append((0 if m["arm"] == "real-honest" else 1, r["main_task"], run_key, m["chunk"], m["row_start"] + p["row"],
                             int(r[LABEL]), p["step"], p["action"]))
    if unmatched > 0.005 * kept_calls:
        raise ValueError(f"{unmatched} of {kept_calls} kept call_end positions have no label")
    print(f"{len(metas)} runs, {kept_calls} kept call_end positions ({unmatched} unmatched), {len(rows)} impactful rows, {time.time() - t0:.0f} s", flush=True)
    world = np.array([r[0] for r in rows]); task = np.array([r[1] for r in rows])
    y = np.array([r[5] for r in rows]); run_id = np.array([r[2] for r in rows])

    def shuffled(seed):
        """Labels permuted within each run (the run keeps its count of destructive calls)."""
        rng = np.random.default_rng(seed)
        out = y.copy()
        for r in np.unique(run_id):
            i = np.flatnonzero(run_id == r)
            out[i] = rng.permutation(out[i])
        return out
    fold = np.array([fold_of(t) for t in task])
    by_chunk = {}
    for i, r in enumerate(rows):
        by_chunk.setdefault(r[3], []).append((i, r[4]))
    tmp = Path("/tmp/feats"); tmp.mkdir(exist_ok=True)
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

    def fit(Xtr, ytr, C, capped):
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
        te, negc = np.flatnonzero(fold == k), np.flatnonzero((world == 0) & (fold == cal) & (yy == 0))
        s_te, s_cal = f(X[L][te].astype(np.float32)), f(X[L][negc].astype(np.float32))
        print(f"draw {draw} fold {k}: layer {L}, C {C}, inner AUROC {best[0]:.3f}, {sum(capped[:-1])} of {len(capped) - 1} selection fits and "
              f"{capped[-1]} of 1 refit not converged, {time.time() - t1:.0f} s", flush=True)
        return draw, k, L, C, te, s_te, negc, s_cal, sum(capped[:-1]), capped[-1]

    cdir = Path("/ckpt") / stamp; cdir.mkdir(parents=True, exist_ok=True)
    res = []
    with parallel_config(backend="loky", inner_max_num_threads=1):
        for r in Parallel(n_jobs=32, return_as="generator_unordered")(delayed(one)(d, k) for d in DRAWS for k in range(5)):
            res.append(r)
            d, k, L, C, te, s_te, negc, s_cal, nsel, nref = r
            np.savez(cdir / f"draw{d}_f{k}.npz", te=te, s_te=s_te, negc=negc, s_cal=s_cal,
                     meta=json.dumps({"layer": L, "C": C, "selection fits not converged": nsel, "refit not converged": nref}))
            ckpt.commit()
    out = {d: {"labels": "".join(map(str, shuffled(d))), "test scores": [None] * len(rows), "calibration negatives": {}, "picks": {}} for d in DRAWS}
    for d, k, L, C, te, s_te, negc, s_cal, nsel, nref in res:
        for i, v in zip(te, s_te):
            out[d]["test scores"][int(i)] = round(float(v), 5)
        out[d]["calibration negatives"][k] = {"rows": [int(i) for i in negc], "scores": [round(float(v), 5) for v in s_cal]}
        out[d]["picks"][k] = {"layer": L, "C": C, "selection fits not converged": nsel, "refit converged": not nref}
    result = {"rows": [{"world": int(world[i]), "task": task[i], "run": rows[i][2], "step": rows[i][6], "action": rows[i][7], "fold": int(fold[i]),
                        LABEL: int(y[i])} for i in range(len(rows))],
              "draws": out, "unmatched": unmatched, "kept call_end positions": kept_calls, "seconds": round(time.time() - t0)}
    (cdir / "full_output.json").write_text(json.dumps(result))
    ckpt.commit()
    return result


@app.local_entrypoint()
def main(labels: str):
    rows = [json.loads(line) for line in Path(labels).read_text().splitlines()]
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    data = HERE.parents[1] / "data" / "recal"
    data.mkdir(parents=True, exist_ok=True)
    call = run.spawn(rows, stamp)
    (data / f"launch_null_{stamp}.json").write_text(json.dumps({"stamp": stamp, "call": call.object_id, "volume": f"{APP}:/{stamp}"}) + "\n")
    print(f"stamp {stamp}, call {call.object_id}", flush=True)
    res = call.get() | {"labels sha256": hashlib.sha256(Path(labels).read_bytes()).hexdigest(), "layers": LAYERS, "C": CS, "stamp": stamp}
    with open(data / f"recal_null_{stamp}.json", "x") as fh:
        fh.write(json.dumps(res) + "\n")
    print({d: v["picks"] for d, v in res["draws"].items()}, res["seconds"])
    print(data / f"recal_null_{stamp}.json")
