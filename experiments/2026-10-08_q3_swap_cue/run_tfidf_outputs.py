"""Control 1: a text detector on tool outputs, the same protocol as the #36 probe it is compared with. Modal CPU; no model
calls. #36's own script is not changed: its TF-IDF reads call texts only (run_q3_attack.py:389-390).

- Texts: every tool output the #36 tool-output readout reads (the output_end rows of #36's data 307cbfee925c, within the
  32k cut), taken from the same staged reader messages; for each swap arm, the tool outputs its extraction kept.
- Model: #36's TF-IDF recipe (config/q3_attack.yaml: word 1-2-grams, min_df 3, at most 200,000 features, sublinear tf,
  logistic regression C = 1, max_iter 2000), one row per tool output labelled with its run (real attack e1 = 1, real
  honest = 0), as #36 does for calls.
- Folds: #36's main-task folds. Outer model f scores fold f; nested model (f, v) scores folds f and v (its validation
  fold v's honest runs set the frozen thresholds, analyze_tfidf_outputs.py).
- Scored: every run of #36's four arms, and the swap arms (single fixes and both fixes on the 100 runs, both fixes on
  the 117-run extension) on the runs of the folds each model scores.
Texts go to the work Volume (run folder), never to git; results/ gets per-run scores only.
- `--mask-digits` replaces every digit with "0" in every text before fitting and scoring (timestamps and numbers keep
  their shape, not their values); `--reuse <work run>` fits on that run's saved texts instead of preparing new ones.
No label-shuffle null: the models are fit on real labels only, so attack-vs-honest AUROC and catch are read against
chance (0.5 and the false-alarm rate), not against a main-task-grouped shuffle null as the project rule asks for
detector scores. Any detector-quality claim from these scores carries that caveat.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.2' --with numpy modal run --detach \
      experiments/2026-10-08_q3_swap_cue/run_tfidf_outputs.py [--reuse q3_tfidf_outputs_<stamp> --mask-digits]
"""

import gzip
import json
import re
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q3swapcue_tfidf_kill-2026-11-06"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
WORK = "probe-shift_aswin-surya_sub-q3attack_work_kill-2026-11-06"
DATA_ID = "307cbfee925c"  # #36's data for run 2026-10-06_181548 (results/run_2026-10-06_181548.json)
BASE_INPUTS = "2026-10-06_la_segment-end_qwen3-32b_v1"
SWAP = {  # label: (extraction run, staged inputs, arm)
    "V2": ("2026-10-07_q2_swap_qwen3-32b_alllayers_v1", "2026-10-07_q2_swap_inputs_v1", "sim-attack-realstyle"),
    "V4": ("2026-10-08_q2_swap_q3_qwen3-32b_l48-63_v1", "2026-10-08_q2_swap_q3_inputs_v1", "sim-attack-diff-note"),
    "SR": ("2026-10-08_q2_swap_q3_qwen3-32b_l48-63_v1", "2026-10-08_q2_swap_q3_inputs_v1", "sim-attack-sr-both"),
    "COMB": ("2026-10-08_q2_swap_combined_qwen3-32b_l48-63_v1", "2026-10-08_q2_swap_combined_inputs_v1", "sim-attack-note-sr-both"),
    "COMB_ext": ("2026-10-08_q2_swap_ext_qwen3-32b_l48-63_v1", "2026-10-08_q2_swap_ext_inputs_v1", "sim-attack-note-sr-both"),
}
TFIDF = {"ngram_range": [1, 2], "min_df": 3, "max_features": 200000, "sublinear_tf": True, "C": 1.0, "max_iter": 2000}
DIGIT = re.compile(r"\d")

app = modal.App(APP)
acts_vol = modal.Volume.from_name(ACTS, version=2)
work_vol = modal.Volume.from_name(WORK, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "scikit-learn==1.7.2")


def outputs_of(messages):
    """(step, action) -> tool output text: step counts assistant messages, action the tool messages after each"""
    out, s, a = {}, -1, 0
    for m in messages:
        if m["role"] == "assistant":
            s, a = s + 1, 0
        elif m["role"] == "tool":
            c = m["content"]
            out[(s, a)] = c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
            a += 1
    return out


def staged(inputs, want):
    got = {}
    for shard in sorted((Path("/acts") / inputs / "inputs").glob("shard_*.jsonl")):
        for line in shard.read_text().splitlines():
            it = json.loads(line)
            if (it["arm"], it["key"]) in want:
                got[(it["arm"], it["key"])] = it["messages"]
    return got


@app.function(image=image, cpu=4, memory=65536, timeout=3600, volumes={"/acts": acts_vol.read_only(), "/work": work_vol})
def prepare(out_name: str) -> dict:
    import numpy as np

    d = Path("/work/data") / DATA_ID
    runs = json.loads((d / "runs.json").read_text())
    t = np.load(d / "tables.npz")
    items = staged(BASE_INPUTS, {(r["arm"], r["key"]) for r in runs})
    if len(items) != len(runs):
        raise ValueError(f"{len(runs) - len(items)} runs missing from {BASE_INPUTS}")
    outs = {}
    base = []
    for i, s, a in zip(t["output_end__run"], t["output_end__step"], t["output_end__action"]):
        r = runs[int(i)]
        k = (r["arm"], r["key"])
        if k not in outs:
            outs[k] = outputs_of(items[k])
        base.append(outs[k][(int(s), int(a))])
    extra = {}
    for label, (run, inputs, arm) in SWAP.items():
        metas = [json.loads(p.read_text()) for p in sorted((Path("/acts") / run / "meta" / arm).glob("*.json"))]
        msgs = staged(inputs, {(arm, m["key"]) for m in metas})
        rows = []
        for m in metas:
            o = outputs_of(msgs[(arm, m["key"])])
            rows += [[m["key"], p["step"], p["action"], o[(p["step"], p["action"])]]
                     for p in m["positions"] if p["kind"] == "output_end" and p["kept"]]
        extra[label] = rows
    out = Path("/work/runs") / out_name
    out.mkdir(parents=True, exist_ok=False)
    with gzip.open(out / "texts.json.gz", "wt") as f:  # transcript text: on the Volume only
        json.dump({"base": base, "extra": extra}, f)
    (out / "config.json").write_text(json.dumps({"data_id": DATA_ID, "base_inputs": BASE_INPUTS, "swap": SWAP, "tfidf": TFIDF}))
    work_vol.commit()
    return {"base_rows": len(base), "extra_rows": {k: len(v) for k, v in extra.items()},
            "extra_runs": {k: len({r[0] for r in v}) for k, v in extra.items()}}


@app.function(image=image, cpu=4, memory=49152, timeout=3600, volumes={"/work": work_vol.read_only()})
def fit(out_name: str, f: int, v: int | None, mask_digits: bool = False) -> dict:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    t0 = time.time()
    d = Path("/work/data") / DATA_ID
    runs = json.loads((d / "runs.json").read_text())
    t = np.load(d / "tables.npz")
    with gzip.open(Path("/work/runs") / out_name / "texts.json.gz", "rt") as fh:
        texts = json.load(fh)
    if mask_digits:
        texts = {"base": [DIGIT.sub("0", x) for x in texts["base"]],
                 "extra": {k: [[*r[:3], DIGIT.sub("0", r[3])] for r in rows] for k, rows in texts["extra"].items()}}
    base = np.array(texts["base"], object)
    r_run = t["output_end__run"]
    arm = np.array([r["arm"] for r in runs])
    fold = np.array([r["fold"] for r in runs])
    y = np.where(arm == "real-attack-e1", 1.0, np.where(arm == "real-honest", 0.0, np.nan))
    score_runs = (fold == f) | (fold == v) if v is not None else fold == f
    tr = np.flatnonzero((~np.isnan(y) & ~score_runs)[r_run])
    te = np.flatnonzero(score_runs[r_run])
    vec = TfidfVectorizer(ngram_range=tuple(TFIDF["ngram_range"]), min_df=TFIDF["min_df"], max_features=TFIDF["max_features"],
                          sublinear_tf=TFIDF["sublinear_tf"], dtype=np.float32)
    clf = LogisticRegression(C=TFIDF["C"], max_iter=TFIDF["max_iter"]).fit(vec.fit_transform(base[tr]), y[r_run[tr]])
    s = clf.decision_function(vec.transform(base[te]))
    fold_of_key = {r["key"]: r["fold"] for r in runs if r["arm"] == "sim-attack"}
    keep = {f, v} if v is not None else {f}
    extra = {}
    for label, rows in texts["extra"].items():
        rows = [r for r in rows if fold_of_key[r[0]] in keep]
        sc = clf.decision_function(vec.transform([r[3] for r in rows])) if rows else []
        extra[label] = [[r[0], r[1], r[2], float(x)] for r, x in zip(rows, sc)]
    return {"f": f, "v": v, "base_rows": te.tolist(), "base_scores": [float(x) for x in s], "extra": extra,
            "info": {"train_rows": int(len(tr)), "scored_rows": int(len(te)), "n_iter": int(clf.n_iter_[0]),
                     "hit_max_iter": bool(clf.n_iter_[0] >= TFIDF["max_iter"]), "vocabulary": len(vec.vocabulary_),
                     "seconds": round(time.time() - t0)}}


@app.local_entrypoint()
def main(reuse: str = "", mask_digits: bool = False):
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    print("start", datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(), flush=True)
    out_name = reuse or f"q3_tfidf_outputs_{stamp}"
    prep = None if reuse else prepare.remote(out_name)
    print("prepared", prep or f"reusing {reuse}", flush=True)
    jobs = [(out_name, f, None, mask_digits) for f in range(5)] + [(out_name, f, v, mask_digits) for f in range(5) for v in range(5) if v != f]
    res = list(fit.starmap(jobs))
    bad = [r["info"] for r in res if r["info"]["hit_max_iter"]]
    if bad:
        raise ValueError(f"{len(bad)} fits hit max_iter: {bad[:2]}")
    path = HERE / "results" / f"tfidf_outputs{'_masked' if mask_digits else ''}_{stamp}.json.gz"
    with gzip.open(path, "xt") as fh:  # per-row scores and row indices only; no text
        json.dump({"stamp": stamp, "work_run": out_name, "data_id": DATA_ID, "prepared": prep, "tfidf": TFIDF, "swap": SWAP,
                   "mask_digits": mask_digits, "fits": res}, fh)
    print("->", path, {f"{r['f']}_{r['v']}": r["info"]["seconds"] for r in res})
