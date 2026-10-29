"""Date decodability (PLAN.md): T1, an output-level probe aimed at the real e1/e2 outputs that carry a time or date
(found from metadata: the output's token span changes when level 1 masks times), at level 0 and level 1, with a
40-draw within-main-task label-swap null; the positive control, planted check 1's site outputs (planted vs twin); and
T2, the run-label probe on raw real e2 vs e1 with C = 100. Modal CPU; claim B step B's packs read-only, whole arrays.
Each unit (one nested fit) is checkpointed; the full output is written to the Volume before returning.

  PYTHONPATH=src uv run --no-project --with 'modal>=1.4.3' modal run --detach experiments/2026-10-11_q2_date_decodability/run_date_decode.py
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-date-decode_kill-2026-11-10"
PACKS = {"level0": "claimB/level0/packedB_a6762f5005ae", "level1": "claimB/level1/packedB_2cfd1aec393c"}
LAYERS = [19, 32, 41, 44, 47, 52, 58, 63]
T1_READOUTS, T1_CS = ["output_end", "output_end_mean", "output_end_ema0.95"], [0.01, 1.0, 100.0]
T2_READOUTS, T2_C = ["output_end_mean", "output_end_ema0.95"], 100.0
N_NULL = 40
TEST_ENVS = ["clinical_trial", "media_processing", "shared_dotfiles", "tiktok", "user_tracking"]
REAL = ("real-attack-e1", "real-attack-e2")
app = modal.App(APP)
acts = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
ckpt = modal.Volume.from_name(APP, create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").apt_install("procps").uv_pip_install("numpy", "scikit-learn==1.7.2", "joblib")


def md5_fold(s: str, k: int = 5) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % k


@app.function(image=image, volumes={"/acts": acts.read_only(), "/ckpt": ckpt}, cpu=32, memory=65536, timeout=5400, retries=0)
def run(stamp: str) -> dict:
    import time
    import warnings
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    from joblib import Parallel, delayed, parallel_config
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    t0 = time.time(); root = Path("/acts"); tmp = Path("/tmp/x"); tmp.mkdir(exist_ok=True)
    cdir = Path("/ckpt") / stamp; cdir.mkdir(parents=True, exist_ok=True)
    runs = {lv: json.loads((root / p / "runs.L19.json").read_text()) for lv, p in PACKS.items()}
    for lv, p in PACKS.items():  # every layer's runs file must match the L19 one
        for L in LAYERS[1:]:
            if json.loads((root / p / f"runs.L{L}.json").read_text()) != runs[lv]:
                raise ValueError(f"{lv}: runs.L{L}.json differs from runs.L19.json")

    def table(rs):
        """Array row of each kept output_end, in q1_eval.step_table order, keyed (run key, step, action); and spans."""
        row, span, j = {}, {}, 0
        for m in rs:
            kept = [p for p in m["positions"] if p["kind"] == "output_end" and p["kept"]]
            for p in kept:
                row[(m["key"], p["step"], p["action"])] = j; j += 1
            prev = None
            for p in sorted(kept, key=lambda q: q["token"]):
                if prev is not None:
                    span[(m["key"], p["step"], p["action"])] = p["token"] - prev["token"]
                prev = p
        return row, span, j
    tab = {lv: table(rs) for lv, rs in runs.items()}
    meta = {lv: {m["key"]: m for m in rs} for lv, rs in runs.items()}

    # T1 rows: real e1/e2 outputs whose span changes under level-1 masking
    clock = [(k, s, a) for (k, s, a), v in tab["level0"][1].items() if meta["level0"][k]["arm"] in REAL
             and (k, s, a) in tab["level1"][1] and tab["level1"][1][(k, s, a)] != v]
    c_rows = {lv: np.array([tab[lv][0][x] for x in clock]) for lv in PACKS}
    c_y = np.array([int(meta["level0"][k]["arm"] == REAL[1]) for k, _, _ in clock])
    c_task = np.array([meta["level0"][k]["main_task"] for k, _, _ in clock])
    # Positive control rows: planted check 1's site output and its twin's (level 1)
    twin = {m["cell"]: m for m in runs["level1"] if m["arm"] == "sim-attack-fixed"}
    p_rows, p_y, p_task = [], [], []
    for m in runs["level1"]:
        if m["arm"] != "sim-attack-fixed-plant1":
            continue
        t = twin[m["cell"]]; sp = tab["level1"][1]
        diff = [(s, a) for (k, s, a) in sp if k == m["key"] and (t["key"], s, a) in sp and sp[(k, s, a)] != sp[(t["key"], s, a)]]
        if len(diff) == 1:
            s, a = diff[0]
            p_rows += [tab["level1"][0][(m["key"], s, a)], tab["level1"][0][(t["key"], s, a)]]; p_y += [1, 0]; p_task += [m["main_task"]] * 2
    p_rows, p_y, p_task = np.array(p_rows), np.array(p_y), np.array(p_task)
    # T2 rows: every kept output of the real e1/e2 runs at level 0
    r2 = [(k, s, a) for (k, s, a) in tab["level0"][0] if meta["level0"][k]["arm"] in REAL]
    t2_rows = np.array([tab["level0"][0][x] for x in r2]); t2_run = np.array([k for k, _, _ in r2])
    if (np.diff(t2_rows) <= 0).any():
        raise ValueError("T2 rows are not in increasing step_table order")
    t2_keys, t2_start = np.unique(t2_run, return_index=True)  # np.unique sorts keys; reorder by first row
    order = np.argsort(t2_start); t2_keys, t2_start = t2_keys[order], t2_start[order]
    if any(len(set(t2_run[a:b])) != 1 for a, b in zip(t2_start, np.append(t2_start[1:], len(t2_run)))):
        raise ValueError("a run's T2 rows are not contiguous")
    print(f"clock outputs {len(clock)} (e2 {int(c_y.sum())}); plant pairs {len(p_y) // 2}; T2 outputs {len(r2)}; {time.time() - t0:.0f} s", flush=True)

    def load(lv, rd, L):
        a = np.load(root / PACKS[lv] / f"L{L}_{rd}.npy")
        if a.shape[0] != tab[lv][2]:
            raise ValueError(f"{lv} L{L} {rd}: {a.shape[0]} rows, expected {tab[lv][2]}")
        np.save(tmp / f"c_{lv}_{rd}_{L}.npy", a[c_rows[lv]].astype(np.float32))
        if lv == "level1":
            np.save(tmp / f"p_{rd}_{L}.npy", a[p_rows].astype(np.float32))
        if lv == "level0" and rd in T2_READOUTS:  # each run's mean over its kept outputs (rows contiguous per run)
            sub = a[t2_rows].astype(np.float32)
            np.save(tmp / f"t2_{rd}_{L}.npy", np.add.reduceat(sub, t2_start, axis=0) / np.diff(np.append(t2_start, len(t2_rows)))[:, None])
        return f"{lv} {rd} L{L}"
    jobs = [(lv, rd, L) for lv in PACKS for rd in T1_READOUTS for L in LAYERS]
    with ThreadPoolExecutor(6) as ex:
        for i, msg in enumerate(ex.map(lambda j: load(*j), jobs)):
            if i % 8 == 7:
                print(f"loaded {i + 1}/{len(jobs)} ({msg}) at {time.time() - t0:.0f} s", flush=True)

    def fit(X, y, C, capped):
        mu, sd = X.mean(0), X.std(0) + 1e-6
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            clf = LogisticRegression(C=C, max_iter=2000).fit((X - mu) / sd, y)
        capped.append(int(any(issubclass(x.category, ConvergenceWarning) for x in w)))
        return lambda Z: clf.decision_function((Z - mu) / sd)

    def nested(prefix, y, task, Cs, readouts):
        """Output-level nested selection: outer folds md5(main task) mod 5; inner halves of the training tasks."""
        X = {(rd, L): np.load(tmp / f"{prefix}_{rd}_{L}.npy", mmap_mode="r") for rd in readouts for L in LAYERS}
        fold = np.array([md5_fold(t) for t in task]); oof = np.full(len(y), np.nan); picks, capped = {}, []
        for k in range(5):
            tr, te = fold != k, fold == k
            ts = np.unique(task[tr]); half = np.isin(task, ts[np.random.default_rng(1000 + k).permutation(len(ts))[: len(ts) // 2]])
            best = (-1.0, None)
            for (rd, L), XX in X.items():
                for C in Cs:
                    aucs = []
                    for a_, b_ in ((tr & half, tr & ~half), (tr & ~half, tr & half)):
                        if len(set(y[a_])) == 2 and len(set(y[b_])) == 2:
                            aucs.append(roc_auc_score(y[b_], fit(np.asarray(XX[a_]), y[a_], C, capped)(np.asarray(XX[b_]))))
                    if aucs and np.mean(aucs) > best[0]:
                        best = (float(np.mean(aucs)), (rd, L, C))
            rd, L, C = best[1]
            oof[te] = fit(np.asarray(X[(rd, L)][tr]), y[tr], C, capped)(np.asarray(X[(rd, L)][te]))
            picks[k] = {"readout": rd, "layer": L, "C": C, "inner AUROC": best[0]}
        fa = [float(roc_auc_score(y[fold == k], oof[fold == k])) for k in range(5) if len(set(y[fold == k])) == 2]
        return {"AUROC": float(np.mean(fa)), "AUROC per fold": fa, "AUROC pooled (mixes fold scales)": float(roc_auc_score(y, oof)),
                "picks": picks, "fits not converged": f"{sum(capped)} of {len(capped)}"}, oof

    def swap(seed):
        rng = np.random.default_rng(seed); tasks = sorted(set(c_task)); flip = dict(zip(tasks, rng.random(len(tasks)) < 0.5))
        return np.array([1 - v if flip[t] else v for v, t in zip(c_y, c_task)])

    def t2_fold(env):
        """Run-label probe on run-mean features, C = 100, outer fold = one held-out environment; inner halves of the
        training main tasks. For a linear probe, scoring the run mean equals the mean of output scores."""
        y = np.array([int(meta["level0"][k]["arm"] == REAL[1]) for k in t2_keys])
        envs = np.array([meta["level0"][k]["environment"] for k in t2_keys]); task = np.array([meta["level0"][k]["main_task"] for k in t2_keys])
        tr, te = envs != env, envs == env
        ts = np.unique(task[tr]); half = np.isin(task, ts[np.random.default_rng(2000).permutation(len(ts))[: len(ts) // 2]])
        X = {(rd, L): np.load(tmp / f"t2_{rd}_{L}.npy") for rd in T2_READOUTS for L in LAYERS}
        capped, best = [], (-1.0, None)
        for key, XX in X.items():
            aucs = [roc_auc_score(y[b_], fit(XX[a_], y[a_], T2_C, capped)(XX[b_])) for a_, b_ in ((tr & half, tr & ~half), (tr & ~half, tr & half))]
            if np.mean(aucs) > best[0]:
                best = (float(np.mean(aucs)), key)
        rd, L = best[1]
        s = fit(X[(rd, L)][tr], y[tr], T2_C, capped)(X[(rd, L)][te])
        return {"environment": env, "readout": rd, "layer": L, "inner AUROC": best[0], "fits not converged": f"{sum(capped)} of {len(capped)}",
                "run scores": {k: float(v) for k, v in zip(t2_keys[te], s)}, "labels": {k: int(v) for k, v in zip(t2_keys[te], y[te])}}

    def unit(name):
        if name == "T1 level0":
            r, oof = nested("c_level0", c_y, c_task, T1_CS, T1_READOUTS); r["oof"] = [round(float(v), 4) for v in oof]
        elif name == "T1 level1":
            r, oof = nested("c_level1", c_y, c_task, T1_CS, T1_READOUTS); r["oof"] = [round(float(v), 4) for v in oof]
        elif name == "plant":
            r, oof = nested("p", p_y, p_task, T1_CS, T1_READOUTS)
            r["planted above twin"] = f"{int((oof[p_y == 1] > oof[p_y == 0]).sum())} of {len(p_y) // 2}"
        elif name.startswith("null "):
            r, _ = nested("c_level0", swap(int(name.split()[1])), c_task, T1_CS, T1_READOUTS)
        else:
            r = t2_fold(name.removeprefix("T2 "))
        (cdir / f"{name.replace(' ', '_')}.json").write_text(json.dumps(r))
        return name, r
    envs = sorted({m["environment"] for m in runs["level0"]})
    names = ["T1 level0", "T1 level1", "plant"] + [f"T2 {e}" for e in envs] + [f"null {d}" for d in range(1, N_NULL + 1)]  # priority order
    res = {}
    head = {"stamp": stamp, "clock outputs": len(clock), "clock outputs e2": int(c_y.sum()), "plant pairs": int(len(p_y) // 2), "T2 outputs": len(r2),
            "clock keys sha256": hashlib.sha256(json.dumps(sorted(clock)).encode()).hexdigest(), "packs": PACKS, "layers": LAYERS, "T1 C": T1_CS, "T2 C": T2_C}
    with parallel_config(backend="loky", inner_max_num_threads=1):
        for name, r in Parallel(n_jobs=32, return_as="generator_unordered")(delayed(unit)(n) for n in names):
            res[name] = r
            (cdir / "partial_output.json").write_text(json.dumps(head | {"units": res, "units done": len(res), "units planned": len(names)}))
            ckpt.commit()
            print(f"{name} done at {time.time() - t0:.0f} s: {r.get('AUROC', r.get('readout'))}", flush=True)
    out = head | {"units": res, "units done": len(res), "units planned": len(names), "seconds": round(time.time() - t0)}
    (cdir / "full_output.json").write_text(json.dumps(out)); ckpt.commit()
    return out


@app.local_entrypoint()
def main():
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    data = HERE.parents[1] / "data" / "date_decode"; data.mkdir(parents=True, exist_ok=True)
    call = run.spawn(stamp)
    (data / f"launch_{stamp}.json").write_text(json.dumps({"stamp": stamp, "call": call.object_id, "volume": f"{APP}:/{stamp}"}) + "\n")
    print(f"stamp {stamp}, call {call.object_id}", flush=True)
    res = call.get()
    with open(data / f"date_decode_{stamp}.json", "x") as fh:
        fh.write(json.dumps(res) + "\n")
    print(data / f"date_decode_{stamp}.json")
