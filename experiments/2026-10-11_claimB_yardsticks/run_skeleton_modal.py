"""I2, the punctuation/whitespace-only ("skeleton") text baseline for claim B (PRICES.md item 2; 0d's GO relayed after
22:33 PT 10-11: char_wb primary, 100 null draws, stop $25). One Modal CPU container (16 cores), retries 0.

Data: gate (a)'s staged examples of the fixed run (`2026-10-09_la_fixed-sim_v2/examples`, kept outputs), loaded with
q1's analyze_tfidf.load unchanged. Per run, Jeffrey's document (sce claude/discriminator-signal @692040b8,
code/skeleton_predicts_judge.py:16-17): each kept output's text (scrubbed to the level), cut to 4,000 characters, joined
by newlines, then every ASCII [A-Za-z0-9] deleted.
Models: primary TfidfVectorizer(analyzer="char_wb", ngram_range=(2,5), min_df=2, max_features=60000) + LogisticRegression
(max_iter=2000), his settings; secondary the same with q1's whitespace-preserving character n-grams (analyze_tfidf
.char_ngrams). Split: gate (a)'s loeo design, each of the 5 test environments held out once with the other 9 as
training; run-level AUROC pooled over the 5 test environments.
Rows: {fixed_vs_e1, e2_vs_e1} x {level1 (primary), level0 (clock-affected)} x {char_wb (primary), ws_char}.
Null (primary analyzer, level 1, both comparisons): 100 draws; per main task one flip with probability 1/2 swaps the
arms of all its cells (gate (a)'s null); the vectorizer is label-independent, so it is fitted once per fold and only the
classifier is refitted per draw; p = (1 + draws >= observed) / 101.
Planted checks (level 1, both analyzers): plant-1 and plant-2 copies and their twins from
`2026-10-10_la_fixed-sim-v2_probe-inputs-l1_v1/inputs` (tool-message text, already level 1), scored by the
fixed-vs-e1 model of their environment's fold; AUROC copy vs twin, and the number of pairs whose analyzer features
differ at all.
Output: the result JSON (rows, nulls, planted, per-run scores of every row, timings) written to the Volume
(claimB/skeleton_v1/) and returned. Numbers, keys and hashes only; no text leaves the container.

  uv run --no-project --python 3.13 --with 'modal>=1.4.3' --with pyyaml modal deploy experiments/2026-10-11_claimB_yardsticks/run_skeleton_modal.py
  uv run --no-project --python 3.13 --with 'modal>=1.4.3' --with pyyaml python experiments/2026-10-11_claimB_yardsticks/run_skeleton_modal.py spawn
  uv run --no-project --python 3.13 --with 'modal>=1.4.3' --with pyyaml python experiments/2026-10-11_claimB_yardsticks/run_skeleton_modal.py fetch <stamp>
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import modal

NAME = "probe-shift_aswin-surya_sub-claimb-skeleton_kill-2026-11-11"
RUN = "2026-10-09_la_fixed-sim_v2"
RUN_L1 = "2026-10-10_la_fixed-sim-v2_probe-inputs-l1_v1"
OUT = "claimB/skeleton_v1"
COMPS = {"fixed_vs_e1": ("real-attack-e1", "sim-attack"), "e2_vs_e1": ("real-attack-e1", "real-attack-e2")}
LEVELS, ANALYZERS = [1, 0], ["char_wb", "ws_char"]
N_NULL, SEED = 100, 20261011

app = modal.App(NAME)
acts = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations")
image = (modal.Image.debian_slim(python_version="3.13")
         .uv_pip_install("numpy", "scipy", "scikit-learn==1.9.1", "joblib==1.6.0", "pyyaml")
         .env({"JOBLIB_TEMP_FOLDER": "/tmp"}))
# Repo paths only where the repo exists: is_local() is also true in joblib worker processes inside the container,
# where this file sits at /root (parent.parent.parent is then "/" and the source file does not exist).
_SRC = Path(__file__).resolve().parent.parent.parent / "experiments/2026-10-06_q1_text_baselines/analyze_tfidf.py"
if modal.is_local() and _SRC.exists():
    image = image.add_local_python_source("probe_shift").add_local_file(_SRC, "/root/analyze_tfidf.py")


def vectorizer(kind: str):
    from functools import partial

    from sklearn.feature_extraction.text import TfidfVectorizer

    import analyze_tfidf as A

    if kind == "char_wb":
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=60000)
    return TfidfVectorizer(analyzer=partial(A.char_ngrams, lo=2, hi=5), min_df=2, max_features=60000)


def skeleton(texts) -> str:
    import re

    return re.sub(r"[A-Za-z0-9]", "", "\n".join(t[:4000] for t in texts))


def run_docs(level: int, env_fold: dict) -> dict:
    """{key: {arm, environment, main_task, cell, doc}} from the kept outputs, in file order."""
    import analyze_tfidf as A

    d = A.load(Path("/acts") / RUN / "examples", env_fold, fold_key="environment")
    runs = {}
    for r in d["rows"]:
        x = runs.setdefault(r["key"], {"arm": r["arm"], "environment": r["environment"], "main_task": r["main_task"],
                                       "cell": r["cell"], "texts": []})
        x["texts"].append(A.scrub(r["readout_text"], level))
    return {k: {**{f: v[f] for f in ("arm", "environment", "main_task", "cell")}, "doc": skeleton(v["texts"])}
            for k, v in runs.items()}


def planted_docs() -> dict:
    """{plant: [(copy doc, twin doc, environment, key)]} from the level-1 reader inputs (tool messages)."""
    recs = {}
    for f in sorted((Path("/acts") / RUN_L1 / "inputs").glob("shard_*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r["arm"] in ("sim-attack-fixed", "sim-attack-fixed-plant1", "sim-attack-fixed-plant2"):
                outs = []
                for m in r["messages"]:
                    if m["role"] == "tool":
                        c = m["content"]
                        outs.append(c if isinstance(c, str) else "".join(p.get("text", "") for p in c if isinstance(p, dict)))
                recs[(r["arm"], r["key"])] = (skeleton(outs), r["environment"])
    out = {}
    for n in (1, 2):
        pairs = []
        for (arm, key), (doc, env) in sorted(recs.items()):
            if arm == f"sim-attack-fixed-plant{n}":
                twin = recs[("sim-attack-fixed", key.removesuffix(f"__plant{n}"))]
                pairs.append((doc, twin[0], env, key))
        if len(pairs) != 150:
            raise ValueError(f"plant {n}: {len(pairs)} pairs, expected 150")
        out[n] = pairs
    return out


def fold_fit(docs: dict, comp: str, test_env: str, kind: str):
    """Fit the vectorizer on the comparison's runs outside test_env; return (vectorizer, train keys, test keys, Xtr, Xte)."""
    neg, pos = COMPS[comp]
    keys = sorted(k for k, v in docs.items() if v["arm"] in (neg, pos))
    tr = [k for k in keys if docs[k]["environment"] != test_env]
    te = [k for k in keys if docs[k]["environment"] == test_env]
    vec = vectorizer(kind)
    Xtr = vec.fit_transform([docs[k]["doc"] for k in tr])
    vec.stop_words_ = None  # the pruned-term set can be huge; not needed to transform
    return vec, tr, te, Xtr, vec.transform([docs[k]["doc"] for k in te])


def lr(Xtr, ytr):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=2000).fit(Xtr, ytr)


@app.function(image=image, volumes={"/acts": acts}, cpu=16, memory=65536, timeout=4 * 3600, retries=0)
def job(split: dict, folds: dict, stamp: str) -> dict:
    import time

    import numpy as np
    from joblib import Parallel, delayed
    from sklearn.metrics import roc_auc_score
    from threadpoolctl import threadpool_limits

    t0, log = time.time(), lambda s: print(f"[{time.time() - t0:7.1f}s] {s}", flush=True)
    env_fold, test = folds["environment_folds"]["environment_to_fold"], split["test"]
    docs = {lv: run_docs(lv, env_fold) for lv in LEVELS}
    log(f"docs: { {lv: len(v) for lv, v in docs.items()} }")
    jobs = [(comp, lv, kind, e) for comp in COMPS for lv in LEVELS for kind in ANALYZERS for e in test]

    def one(comp, lv, kind, e):
        vec, tr, te, Xtr, Xte = fold_fit(docs[lv], comp, e, kind)
        pos = COMPS[comp][1]
        ytr = np.array([docs[lv][k]["arm"] == pos for k in tr])
        clf = lr(Xtr, ytr)
        return {"comp": comp, "level": lv, "kind": kind, "env": e, "te": te, "score": clf.decision_function(Xte).tolist(),
                "vec": vec, "clf": clf, "Xtr": Xtr, "Xte": Xte, "tr": tr, "n_features": len(vec.vocabulary_)}

    fits = Parallel(n_jobs=16, backend="loky")(delayed(one)(*j) for j in jobs)
    log(f"main fits: {len(fits)}")
    rows = {}
    for comp in COMPS:
        neg, pos = COMPS[comp]
        for lv in LEVELS:
            for kind in ANALYZERS:
                fs = [f for f in fits if (f["comp"], f["level"], f["kind"]) == (comp, lv, kind)]
                keys = [k for f in fs for k in f["te"]]
                score = np.array([s for f in fs for s in f["score"]])
                y = np.array([docs[lv][k]["arm"] == pos for k in keys])
                rows[f"{comp}|level{lv}|{kind}"] = {
                    "auroc": float(roc_auc_score(y, score)), "runs": {"neg": int((~y).sum()), "pos": int(y.sum())},
                    "per_env": {f["env"]: float(roc_auc_score([docs[lv][k]["arm"] == pos for k in f["te"]], f["score"]))
                                for f in fs if len({docs[lv][k]["arm"] for k in f["te"]}) == 2},
                    "n_features": {f["env"]: f["n_features"] for f in fs},
                    "run_scores": [{"key": k, "arm": docs[lv][k]["arm"], "environment": docs[lv][k]["environment"],
                                    "main_task": docs[lv][k]["main_task"], "score": float(s)} for k, s in zip(keys, score)]}
    log("rows done")

    # Null: primary analyzer, level 1; vectorizers reused, classifier refitted per draw.
    nulls = {}
    for comp in COMPS:
        pos = COMPS[comp][1]
        fs = [f for f in fits if (f["comp"], f["level"], f["kind"]) == (comp, 1, "char_wb")]
        d1 = docs[1]
        tasks = sorted({d1[k]["main_task"] for f in fs for k in f["tr"] + f["te"]})

        def draw(i, fs=fs, d1=d1, pos=pos, tasks=tasks):
            flip = dict(zip(tasks, np.random.default_rng([SEED, i]).random(len(tasks)) < 0.5))
            lab = lambda k: (d1[k]["arm"] == pos) != flip[d1[k]["main_task"]]  # noqa: E731
            ys, ss = [], []
            for f in fs:
                ytr = np.array([lab(k) for k in f["tr"]])
                ss += lr(f["Xtr"], ytr).decision_function(f["Xte"]).tolist()
                ys += [lab(k) for k in f["te"]]
            return float(roc_auc_score(ys, ss))

        with threadpool_limits(limits=1):  # 16 threads, one native thread each
            vals = Parallel(n_jobs=16, backend="threading")(delayed(draw)(i) for i in range(N_NULL))
        obs = rows[f"{comp}|level1|char_wb"]["auroc"]
        nulls[comp] = {"draws": vals, "p95": float(np.percentile(vals, 95)), "median": float(np.median(vals)),
                       "p_upper": (1 + sum(v >= obs for v in vals)) / (N_NULL + 1)}
        log(f"null {comp} done")

    # Planted checks: fixed-vs-e1, level 1, each analyzer; copies and twins scored by their environment's fold model.
    planted = {}
    pd = planted_docs()
    for kind in ANALYZERS:
        for n, pairs in pd.items():
            sc, yy, changed = [], [], 0
            for copy, twin, env, key in pairs:
                f = next(f for f in fits if (f["comp"], f["level"], f["kind"], f["env"]) == ("fixed_vs_e1", 1, kind, env))
                X = f["vec"].transform([copy, twin])
                changed += (X[0] != X[1]).nnz > 0  # the vectors the classifier sees
                s = f["clf"].decision_function(X)
                sc += [float(s[0]), float(s[1])]
                yy += [1, 0]
            planted[f"plant{n}|{kind}"] = {"pairs": len(pairs), "pairs_with_changed_features": int(changed),
                                           "auroc_copy_vs_twin": float(roc_auc_score(yy, sc)),
                                           "mean_shift": float(np.mean(np.array(sc[0::2]) - np.array(sc[1::2])))}
    log("planted done")
    out = {"rows": rows, "nulls": nulls, "planted": planted, "seconds": round(time.time() - t0, 1),
           "fits": {"main": len(fits), "null": N_NULL * len(COMPS) * len(test)}}
    dst = Path("/acts") / OUT / f"result_{stamp}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "x") as fh:
        fh.write(json.dumps(out))
    acts.commit()
    log(f"written {dst}")
    return {"written": str(dst), "seconds": out["seconds"]}


def local(mode: str, arg: str = ""):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    here = Path(__file__).resolve().parent
    repo = here.parents[1]
    if mode == "spawn":
        split = json.loads((repo / "experiments/2026-10-09_fixed_sim/split.json").read_text())
        folds = json.loads((repo / "splits/folds_v3.json").read_text())
        stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
        call = modal.Function.from_name(NAME, "job").spawn(split, folds, stamp)
        print(json.dumps({"stamp": stamp, "call_id": call.object_id}))
    elif mode == "fetch":
        raw = b"".join(acts.read_file(f"{OUT}/result_{arg}.json"))
        out = json.loads(raw)
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=here).stdout.strip()
        out |= {"label": "I2 skeleton baseline (EXPLORATORY: reads the test environments); numbers only",
                "volume_file": f"{OUT}/result_{arg}.json", "volume_sha256": hashlib.sha256(raw).hexdigest(), "git": git,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        dst = here / "data" / f"skeleton_{arg}.json"
        dst.parent.mkdir(exist_ok=True)
        with open(dst, "x") as fh:
            fh.write(json.dumps(out))
        print({k: round(v["auroc"], 4) for k, v in out["rows"].items()})
        print({k: {kk: round(vv, 4) for kk, vv in v.items() if kk != "draws"} for k, v in out["nulls"].items()})
        print(out["planted"])
        print("->", dst)


if __name__ == "__main__":
    local(*sys.argv[1:])
