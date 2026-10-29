"""TF-IDF frozen on the cohorts, scored on the matched pairs (Modal CPU): the text counterpart of the frozen probe.

Training data: baselines-q3's staged Q1 examples (~/probe-shift-baselines/experiments/2026-10-06_q1_text_baselines,
.run_cache/examples_2026-10-06_142532/outputs.jsonl.gz, read only): every kept tool output of real attack epoch 1 (0)
and simulated attack (1) in the 939 paired cells, as the reader reads it up to output_end (`readout_text`). Same
featurizer and classifier as baselines-q3 and analyze_pairs.py (word 1-2-grams + character 2-5-grams hashed, TF-IDF,
logistic regression C = 1), at scrub level 0 and level 5. One model per held-out main-task fold (folds_v3); a pair is
scored by the model whose held-out fold holds its main task. Pair texts come from analyze_pairs.py (data/readout_*.jsonl).

  PYTHONPATH=src python -m modal run experiments/2026-10-06_matched_step_pairs/analyze_text_frozen.py \\
    --readout data/readout_<stamp>.jsonl --pairs data/pairs_<stamp>.jsonl
"""

import gzip
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
ROOT = HERE.parents[1] if len(HERE.parents) > 1 else HERE  # the container mounts this file at /root
APP = "probe-shift_aswin-surya_sub-matched_text_kill-2026-11-06"
VOLUME = "probe-shift_aswin-surya_sub-matched_pairs_kill-2026-11-06"
EXAMPLES = Path.home() / "probe-shift-baselines/experiments/2026-10-06_q1_text_baselines/.run_cache/examples_2026-10-06_142532"
PT = ZoneInfo("America/Los_Angeles")

app = modal.App(APP)
vol = modal.Volume.from_name(VOLUME, version=2)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy", "scipy", "scikit-learn==1.7.2", "joblib")
         .add_local_file(HERE / "analyze_pairs.py", "/root/analyze_pairs.py"))


@app.function(image=image, volumes={"/vol": vol}, cpu=16, memory=65536, timeout=3 * 3600)
def fit_fold(level: int, fold: int, folds: dict, pair_rows: list[dict]) -> dict:
    import numpy as np
    from joblib import Parallel, delayed
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.linear_model import LogisticRegression

    import scipy.sparse as sp
    from analyze_pairs import featurize, scrub

    texts, y = [], []
    for line in gzip.open("/vol/text_frozen/outputs.jsonl.gz", "rt"):
        r = json.loads(line)
        if r["kept"] and r["arm"] in ("real-attack-e1", "sim-attack") and folds[r["main_task"]] != fold:
            texts.append(scrub(r["readout_text"], level))
            y.append(int(r["arm"] == "sim-attack"))
    chunks = [texts[i:i + 2000] for i in range(0, len(texts), 2000)]
    X = sp.vstack(Parallel(n_jobs=16)(delayed(featurize)(c) for c in chunks)).tocsr()
    tf = TfidfTransformer(sublinear_tf=True).fit(X)
    clf = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000).fit(tf.transform(X), np.array(y))
    mine = [r for r in pair_rows if r["fold"] == fold]
    s = clf.decision_function(tf.transform(featurize([scrub(r["text"], level) for r in mine]))) if mine else []
    return {"level": level, "fold": fold, "n_train": len(y), "scores": [[r["pair"], r["version"], float(v)] for r, v in zip(mine, s)]}


@app.local_entrypoint()
def main(readout: str, pairs: str):
    folds = json.loads((ROOT / "splits" / "folds_v3.json").read_text())["task_folds"]["main_task_to_fold"]
    fold_of = {json.loads(l)["pair"]: json.loads(l)["fold"] for l in Path(pairs).read_text().splitlines()}
    rows = [json.loads(l) for l in Path(readout).read_text().splitlines()]
    for r in rows:
        r["fold"] = fold_of[r["pair"]]
    with vol.batch_upload(force=True) as batch:
        batch.put_file(str(EXAMPLES / "outputs.jsonl.gz"), "text_frozen/outputs.jsonl.gz")
    jobs = [(level, f, folds, rows) for level in (0, 5) for f in sorted(set(folds.values()))]
    out = list(fit_fold.starmap(jobs))
    stamp = datetime.now(PT).strftime("%Y-%m-%d_%H%M%S")
    import hashlib

    res = {"examples": str(EXAMPLES), "readout_file": Path(readout).name, "folds_file": "folds_v3.json",
           "pairs_file": Path(pairs).name, "pairs_sha256": hashlib.sha256(Path(pairs).read_bytes()).hexdigest(),
           "readout_sha256": hashlib.sha256(Path(readout).read_bytes()).hexdigest(),
           "fits": [{k: o[k] for k in ("level", "fold", "n_train")} for o in out],
           "scores": {f"scrub{level}": [s for o in out if o["level"] == level for s in o["scores"]] for level in (0, 5)}}
    path = HERE / "results" / f"text_frozen_scores_{stamp}.json"
    with open(path, "x") as f:
        f.write(json.dumps(res))
    print(f"wrote {path}")
