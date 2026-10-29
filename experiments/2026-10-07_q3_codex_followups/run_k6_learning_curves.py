"""K6: mean difference vs logistic regression for a contrast-pair probe, by number of training pairs (Modal CPU).

Data: the saved eval-awareness contrast pairs (#41, run_eval_awareness_probe.py): 240 pairs over 60 core tasks, two
families (explicit: the system prompt says it is a test; implicit: the scenario only implies it), Qwen3-32B block
output at layers 16-48 on the tokens inside the final tool call's JSON, which are identical within a pair. One row
per dialogue: the mean over those tokens. Before anything else the job checks the activations against the committed
fit: the pairs file must equal pairs(0) from the generator blob the fit record names (the extraction's dialogue
order), and the saved probes, scored as that fit scored them, must give its held-out AUROC at every layer.

Estimators, all fitted on dialogue rows of the training pairs; a 2x2 of estimator x per-feature scaling (Codex's
check: does mean difference fail because of the estimator or because of the scale of raw coordinates?):
  mean_difference                  w = mean over pairs of (eval dialogue - deploy dialogue), raw activations; score x @ w;
  mean_difference_standardized     the same in standardized coordinates (training mean and sd): raw weight w / sd^2;
  logistic_regression              standardized on the training rows, L2, no intercept, C = 0.1 (the #41 recipe's C);
  logistic_regression_raw          centered but not scaled, otherwise the same.
The 2026-10-07_172023 run had the first and third; their numbers must come out the same here.
Folds: the 60 tasks in 5 groups (seed 0); every pair of a held-out task is scored. Per training family (combined,
explicit, implicit), size (25, 50, 100 pairs, all available) and draw (20 seeded draws of pairs; one when all are
used): held-out AUROC and pair accuracy on each family (the other family is the transfer test). Stability: cosine
between two draws' directions and Spearman between their held-out scores, same fold and size. Offsets: on held-out
pairs, the sd of pair midpoints over the mean pair gap (high = items differ more than pairs do, so pair accuracy can be
high while AUROC is low). Null: at full size, 10 refits with each training pair's labels swapped with probability 1/2.
Held-out scores of every full-size fit are saved for task-resample intervals.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.2' --with numpy modal run --detach \
      run_k6_learning_curves.py --pairs <eval_awareness_pairs_*.jsonl> --record <eval_awareness_probe_*.json> \
      --code-record <eval_awareness_probe_*_record.json> --generator <src/probe_shift/eval_awareness.py>
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
APP = "probe-shift_aswin-surya_sub-q3followups_k6_kill-2026-11-06"
OUTPUT = "probe-shift_aswin-surya_sub-q3_outputs_kill-2026-11-06"
ACTS, PROBES = "acts/eval_awareness_v1.npz", "probes/eval_awareness_v1"
LAYERS = (16, 24, 32, 40, 48)
SIZES, DRAWS, FOLDS, NULLS, C, MAX_ITER = (25, 50, 100, None), 20, 5, 10, 0.1, 2000
GATE_S = 1500  # under the 30 min timeout; about $0.4 at ~$1/h
ESTIMATORS = ("mean_difference", "mean_difference_standardized", "logistic_regression", "logistic_regression_raw")
FAMILIES = {"combined": ("explicit", "implicit"), "explicit": ("explicit",), "implicit": ("implicit",)}
CPU = 16

app = modal.App(APP)
out_vol = modal.Volume.from_name(OUTPUT)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "scikit-learn==1.7.2", "scipy", "joblib")


def seed_of(*parts) -> int:
    return int(hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:8], 16)


def layer_job(layer, D, label, family, task, pair, fold_of_task):
    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    pos = {p: i for i, (p, y) in enumerate(zip(pair, label)) if y == 1}
    neg = {p: i for i, (p, y) in enumerate(zip(pair, label)) if y == 0}
    pair_task = {p: task[pos[p]] for p in pos}
    pair_fam = {p: family[pos[p]] for p in pos}

    def fit(ps, flip=None):
        i1, i0 = np.array([pos[p] for p in ps]), np.array([neg[p] for p in ps])
        if flip is not None:
            i1, i0 = np.where(flip, i0, i1), np.where(flip, i1, i0)
        md = (D[i1] - D[i0]).mean(0)
        X = D[np.r_[i1, i0]]
        y = np.r_[np.ones(len(i1)), np.zeros(len(i0))]
        mu, sd = X.mean(0), X.std(0)
        sd = np.where(sd > 0, sd, 1.0)
        lr = LogisticRegression(C=C, fit_intercept=False, max_iter=MAX_ITER).fit((X - mu) / sd, y)
        lr_raw = LogisticRegression(C=C, fit_intercept=False, max_iter=MAX_ITER).fit(X - mu, y)
        ws = {"mean_difference": md, "mean_difference_standardized": md / sd**2,  # all as raw-space directions
              "logistic_regression": lr.coef_[0] / sd, "logistic_regression_raw": lr_raw.coef_[0]}
        conv = {"logistic_regression": int(lr.n_iter_[0] < MAX_ITER), "logistic_regression_raw": int(lr_raw.n_iter_[0] < MAX_ITER)}
        return {e: ws[e] for e in ESTIMATORS}, conv

    def metrics(w, te):
        s = D @ w
        r = {}
        for f in ("explicit", "implicit"):
            m = te & (family == f)
            ps = [p for p in pos if m[pos[p]]]
            r[f"auroc_{f}"] = float(roc_auc_score(label[m], s[m]))
            r[f"pair_acc_{f}"] = float(np.mean([s[pos[p]] > s[neg[p]] for p in ps]))
            mid = np.array([(s[pos[p]] + s[neg[p]]) / 2 for p in ps])
            gap = np.array([s[pos[p]] - s[neg[p]] for p in ps])
            r[f"midpoint_sd_over_gap_{f}"] = float(mid.std() / abs(gap.mean())) if gap.mean() else None
        return r, s

    rows, stab, nulls, heldout = [], [], [], []
    for f in range(FOLDS):
        te = np.array([fold_of_task[t] == f for t in task])
        for fam, fams in FAMILIES.items():
            avail = sorted(p for p in pos if fold_of_task[pair_task[p]] != f and pair_fam[p] in fams)
            for n in SIZES:
                if n is not None and n >= len(avail):
                    continue
                dirs = {e: [] for e in ESTIMATORS}
                scores = {e: [] for e in ESTIMATORS}
                for d in range(DRAWS if n is not None else 1):
                    ps = avail if n is None else sorted(np.random.default_rng(seed_of("k6", layer, f, fam, n, d)).choice(avail, n, replace=False).tolist())
                    ws, conv = fit(ps)
                    for est, w in ws.items():
                        r, s = metrics(w, te)
                        rows.append({"layer": layer, "fold": f, "train_family": fam, "pairs": len(ps), "size": n or "all", "draw": d,
                                     "estimator": est, "lr_converged": conv.get(est)} | r)
                        if n is None:
                            heldout.append({"layer": layer, "fold": f, "train_family": fam, "estimator": est,
                                            "dialogues": np.flatnonzero(te).tolist(), "scores": s[te].round(5).tolist()})
                        dirs[est].append(w / np.linalg.norm(w))
                        scores[est].append(s[te])
                if n is not None:
                    for est in dirs:
                        V = np.array(dirs[est])
                        cos = V @ V.T
                        iu = np.triu_indices(len(V), 1)
                        rho = [spearmanr(scores[est][a], scores[est][b])[0] for a, b in zip(*iu)]
                        stab.append({"layer": layer, "fold": f, "train_family": fam, "size": n, "estimator": est,
                                     "cosine_mean": float(cos[iu].mean()), "spearman_mean": float(np.mean(rho))})
            for k in range(NULLS):
                flip = np.random.default_rng(seed_of("k6-null", layer, f, fam, k)).random(len(avail)) < 0.5
                ws, _ = fit(avail, flip)
                for est, w in ws.items():
                    r, _ = metrics(w, te)
                    nulls.append({"layer": layer, "fold": f, "train_family": fam, "null_draw": k, "estimator": est} | r)
    return rows, stab, nulls, heldout


@app.function(image=image, volumes={"/out": out_vol}, cpu=CPU, memory=32768, timeout=1800)
def run(dialogues: list[dict], record: dict) -> dict:
    import time

    import numpy as np
    from joblib import Parallel, delayed
    from scipy.sparse import csr_matrix
    from sklearn.metrics import roc_auc_score

    t0 = time.time()
    z = np.load(Path("/out") / ACTS)
    owner = z["dialogue"]
    n = len(dialogues)
    if len(owner) != record["n_tokens"] or owner.max() + 1 != n or record["n_dialogues"] != n:
        raise ValueError(f"{len(owner)} tokens over {owner.max() + 1} dialogues; record says {record['n_tokens']} over {record['n_dialogues']}")
    cnt = np.bincount(owner, minlength=n)
    pair = np.array([d["pair"] for d in dialogues])
    if not all(cnt[pair == p].min() == cnt[pair == p].max() for p in set(pair)):  # both dialogues end in the same tool call
        raise ValueError("token counts differ within a pair: dialogue order does not match the activations")
    label = np.array([d["label"] for d in dialogues]); family = np.array([d["family"] for d in dialogues])
    task = np.array([d["task"] for d in dialogues])
    is_val = np.isin(task, record["val_tasks"])
    if cnt.min() == 0:
        raise ValueError("a dialogue has no tokens")
    M = csr_matrix((np.ones(len(owner), np.float32), (owner, np.arange(len(owner)))), shape=(n, len(owner)))  # dialogue mean
    D, repro = {}, {}
    for l in LAYERS:
        X = z[f"L{l}"].astype(np.float32)
        D[l] = np.asarray(M @ X) / cnt[:, None]
        p = np.load(Path("/out") / PROBES / f"probe_L{l}.npz")
        s = ((X - p["scaler_mean"].astype(np.float32)) / p["scaler_scale"].astype(np.float32)) @ p["coef"].astype(np.float32).reshape(-1)
        s = np.bincount(owner, weights=s, minlength=n) / cnt
        repro[l] = {"auroc": round(float(roc_auc_score(label[is_val], s[is_val])), 4), "committed": record["by_layer"][str(l)]["combined"]["val_auroc_all"]}
    if any(abs(v["auroc"] - v["committed"]) > 1e-4 for v in repro.values()):
        raise ValueError(f"saved probes do not reproduce the committed held-out AUROC: {repro}")
    print("reproduced committed held-out AUROC at every layer:", repro, flush=True)
    tasks = sorted(set(task))
    order = np.random.default_rng(0).permutation(len(tasks))
    fold_of_task = {tasks[i]: k % FOLDS for k, i in enumerate(order)}
    # timing gate for the unscaled logistic regression (it can converge slowly): one full-size fit at layer 48, fold 0;
    # ~865 such fits per layer; projected as if the layers ran one after another (the gate fit had all cores), plus the
    # ~300 s the other estimators took in the 2026-10-07_172023 run; stop here if that passes the budget
    from sklearn.linear_model import LogisticRegression

    tr = np.array([fold_of_task[t] != 0 for t in task])
    X = D[48][tr]
    t1 = time.time()
    g = LogisticRegression(C=C, fit_intercept=False, max_iter=MAX_ITER).fit(X - X.mean(0), label[tr])
    t_fit = time.time() - t1
    projected = 300 + len(LAYERS) * 865 * t_fit
    print(f"timing gate: unscaled logistic regression {t_fit:.2f} s, {g.n_iter_[0]} iterations; projected {projected / 60:.0f} min", flush=True)
    if projected > GATE_S:
        raise RuntimeError(f"projected {projected:.0f} s > {GATE_S} s budget")
    outs = Parallel(n_jobs=len(LAYERS))(delayed(layer_job)(l, D[l], label, family, task, pair, fold_of_task) for l in LAYERS)
    rows, stab, nulls, heldout = ([r for o in outs for r in o[i]] for i in range(4))
    print(f"{len(rows)} fits, {len(stab)} stability cells, {len(nulls)} null fits, {time.time() - t0:.0f} s", flush=True)
    return {"rows": rows, "stability": stab, "nulls": nulls, "heldout": heldout, "reproduced": repro, "fold_of_task": {str(k): v for k, v in fold_of_task.items()},
            "seconds": round(time.time() - t0)}


@app.local_entrypoint()
def main(pairs: str, record: str, code_record: str, generator: str):
    import importlib.util

    dialogues = [json.loads(line) for line in Path(pairs).read_text().splitlines()]
    rec = json.loads(Path(record).read_text())
    # the dialogue order the activations were extracted in: pairs(0) from the generator blob the fit record names
    blob = json.loads(Path(code_record).read_text())["code"]["src/probe_shift/eval_awareness.py"].split()[2]
    got = subprocess.run(["git", "hash-object", generator], capture_output=True, text=True, check=True).stdout.strip()
    if got != blob:
        raise ValueError(f"{generator} is blob {got}, the extraction used {blob}")
    spec = importlib.util.spec_from_file_location("eval_awareness_gen", generator)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    if gen.pairs(rec["config"]["pairs_seed"]) != dialogues:
        raise ValueError(f"{pairs} is not pairs({rec['config']['pairs_seed']}) in its order")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=REPO).stdout.strip())
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = run.remote(dialogues, rec)
    res = HERE / "results"
    meta = {"stamp": stamp, "git": head, "git_dirty": dirty, "app": APP, "cpu": CPU, "acts": f"{OUTPUT}/{ACTS}", "probes": f"{OUTPUT}/{PROBES}",
            "inputs": {k: {"file": p, "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()}
                       for k, p in (("pairs", pairs), ("record", record), ("code_record", code_record), ("generator", generator))},
            "generator_blob": blob,
            "layers": LAYERS, "sizes": [s or "all" for s in SIZES], "draws": DRAWS, "folds": FOLDS, "nulls": NULLS, "C": C,
            "estimators": ESTIMATORS, "max_iter": MAX_ITER,
            "reproduced": out["reproduced"], "fold_of_task": out["fold_of_task"], "seconds": out["seconds"]}
    with (res / f"k6_learning_curves_{stamp}.json").open("x") as f:
        f.write(json.dumps(meta, indent=1))
    for name in ("rows", "stability", "nulls", "heldout"):
        with (res / f"k6_learning_curves_{stamp}_{name}.jsonl").open("x") as f:
            f.writelines(json.dumps(r) + "\n" for r in out[name])
    print(f"-> results/k6_learning_curves_{stamp}*.json[l] ({out['seconds']} s)")
