"""Top TF-IDF features of the honest real-vs-real model (first real honest run by key vs second), level 7, no editor
outputs, on the 258 tasks with two real honest runs: one fit on all of them (for reading only, as the baselines' own
top features), on Modal CPU. Same featurizer and model as the baselines (analyze_top_features.py). Each column is shown
with the share of outputs containing it in three groups: first runs from the base200 batch, first runs from the
overlap800 batch, second runs (all overlap800).

  uv run modal run --detach experiments/2026-10-07_what_each_method_sees/followups/run_honest_floor_features_modal.py::main
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
EXP = HERE.parent
ROOT = EXP.parent.parent
NAME = "probe-shift_aswin-surya_sub-what-sees_floor_kill-2026-11-06"
STAGING = Path("~/probe-shift-baselines/experiments/2026-10-06_q1_text_baselines/.run_cache/examples_2026-10-06_213745").expanduser()
app = modal.App(NAME)
vol = modal.Volume.from_name(NAME, create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.13")
         .uv_pip_install("numpy", "scipy", "scikit-learn==1.9.1", "joblib==1.6.0", "pyyaml")
         .env({"JOBLIB_TEMP_FOLDER": "/tmp"})
         .add_local_python_source("probe_shift")
         .add_local_file(EXP / "analyze_top_features.py", "/root/analyze_top_features.py"))
R1, R2 = "real-honest", "real-honest-2"


@app.function(image=image, volumes={"/data": vol}, cpu=16, memory=32768, timeout=1800)
def fit(stamp: str, level: int = 7, top: int = 25) -> dict:
    import gzip

    import numpy as np
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.linear_model import LogisticRegression

    sys.path.insert(0, "/root")
    import analyze_top_features as A
    from probe_shift.scrub import scrub

    vol.reload()
    runs = [json.loads(l) for l in Path(f"/data/inputs/{stamp}/runs.jsonl").read_text().splitlines()]
    pair = {}
    for r in runs:
        if r["arm"] in (R1, R2):
            pair.setdefault(r["main_task"], {})[r["arm"]] = r["key"]
    two = {t for t, d in pair.items() if len(d) == 2}
    rows = [r for r in map(json.loads, gzip.open(f"/data/inputs/{stamp}/outputs.jsonl.gz", "rt"))
            if r["kept"] and r["arm"] in (R1, R2) and r["main_task"] in two and r["tool"] != "text_editor"]
    texts = [scrub(r["readout_text"], level) for r in rows]
    y = np.array([r["arm"] == R2 for r in rows])
    group = np.array(["second" if r["arm"] == R2 else f"first {r['key'].split('_')[0]}" for r in rows])
    X = A.featurize(texts, -1)
    clf = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000).fit(TfidfTransformer(sublinear_tf=True).fit_transform(X), y.astype(int))
    coef = clf.coef_[0]
    order = np.argsort(coef)
    sides = {"toward_second_run": order[::-1][:top], "toward_first_run": order[:top]}
    cols = np.concatenate(list(sides.values()))
    names = A.name_columns(X, texts, cols)
    present = (X[:, cols] > 0).toarray()
    groups = ("first base200", "first overlap800", "second")
    share = {int(c): {g: round(float(present[group == g, j].mean()), 4) for g in groups} for j, c in enumerate(cols)}
    prov = json.loads(Path(f"/data/inputs/{stamp}/provenance.json").read_text())  # saved at launch, before the fit
    out = {"stamp": stamp, "provenance": prov, "level": level, "outputs": len(rows), "outputs_by_group": {g: int((group == g).sum()) for g in groups},
           "tasks": len(two), "top": {s: [{"ngrams": names[int(c)], "weight": round(float(coef[c]), 4),
                                           "kind": "word" if c < A.N_FEATURES else "char", "share": share[int(c)]} for c in cs]
                                      for s, cs in sides.items()}}
    Path(f"/data/results/{stamp}").mkdir(parents=True, exist_ok=True)
    Path(f"/data/results/{stamp}/honest_floor_features_{stamp}.json").write_text(json.dumps(out, indent=1))
    vol.commit()
    return out


@app.local_entrypoint()
def main():
    sys.path.insert(0, str(EXP))
    from analyze_top_features import show
    from probe_shift.provenance import capture

    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    prov = capture(ROOT, [Path(__file__), EXP / "analyze_top_features.py"]) | {"modal_app": NAME, "staging": str(STAGING)}
    (HERE / "results").mkdir(exist_ok=True)
    launch = HERE / "results" / f"honest_floor_features_launch_{stamp}.json"
    launch.write_text(json.dumps(prov, indent=1))
    with vol.batch_upload() as batch:
        batch.put_file(str(STAGING / "outputs.jsonl.gz"), f"/inputs/{stamp}/outputs.jsonl.gz")
        batch.put_file(str(STAGING / "runs.jsonl"), f"/inputs/{stamp}/runs.jsonl")
        batch.put_file(str(launch), f"/inputs/{stamp}/provenance.json")
    print(f"stamp {stamp}", flush=True)
    out = fit.remote(stamp)
    (HERE / "results" / f"honest_floor_features_{stamp}.json").write_text(json.dumps(out, indent=1))
    md = [f"# Honest real-vs-real model, level {out['level']}, no editor outputs: top features ({stamp})", "",
          f"{out['outputs']} outputs on {out['tasks']} tasks; by group {out['outputs_by_group']}. Shares: fraction of each "
          "group's outputs containing the column.", ""]
    for side, feats in out["top"].items():
        md += [f"**{side.replace('_', ' ')}**", "", "| # | n-grams | kind | weight | first base200 | first overlap800 | second |", "|---|---|---|---|---|---|---|"]
        md += [f"| {i + 1} | {show(f['ngrams'])} | {f['kind']} | {f['weight']:+.3f} | {f['share']['first base200']:.3f} | "
               f"{f['share']['first overlap800']:.3f} | {f['share']['second']:.3f} |" for i, f in enumerate(feats)]
        md.append("")
    (HERE / "results" / f"honest_floor_features_{stamp}.md").write_text("\n".join(md))
    print("\n".join(md))
