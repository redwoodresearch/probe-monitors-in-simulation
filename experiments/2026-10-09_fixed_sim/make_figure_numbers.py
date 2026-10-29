"""Small numbers files for the morning figures (0d, 10-11): trims of results that live on S3, each with its source's
sha256. Writes results/figure_numbers/: catch_rows_<src stamp>.json (the 1%-target table's rows: catch at 1% and 5% with
intervals and achieved false alarms, AUROC), threshold_stability_trim_<stamp>.json (amendment 8 without ROC arrays),
qwen_vs_llama_POSTHOC_trim_<stamp>.json and
roc_points_2026-10-11.json (log-FPR ROC curves, at most 201 points each: TPR at 200 log-spaced false-alarm rates from
1e-3 to 1, plus FPR 0). Local, $0. Its four inputs are not in git: results/figure_numbers/inputs_manifest.json
gives each one's S3 key, sha256 and bytes (fetch them to those paths first).

  uv run --no-project --python 3.13 --with numpy --with scikit-learn python experiments/2026-10-09_fixed_sim/make_figure_numbers.py
"""

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve

HERE = Path(__file__).parent
OUT = HERE / "results" / "figure_numbers"
TPR = HERE / "results" / "tpr_at_fpr_2026-10-10_235259.json"
STAB = HERE / "results" / "threshold_stability_2026-10-11_002451.json"
DUMP = HERE.parent / "2026-10-10_claimB_qwen35_reader/data/grid_run_scores_2026-10-10_235051.json.gz"
QVL = HERE.parent / "2026-10-10_claimB_small_lm/results/qwen_vs_llama_POSTHOC_2026-10-11_003609.json"
GRID = np.concatenate([[0.0], np.logspace(-3, 0, 200)])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731


def down(fpr, tpr):
    """TPR at each grid false-alarm rate: the best TPR reachable without exceeding it (the ROC step function)."""
    fpr, tpr = np.asarray(fpr), np.asarray(tpr)
    return [round(float(tpr[fpr <= g].max()), 5) for g in GRID]


def pct(x):
    return {k: x[k] for k in ("caught", "positives", "caught_share", "caught_ci95", "false_alarms", "negatives", "fpr", "fpr_ci95")}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = json.loads(TPR.read_text())
    rows = [{"id": r["id"], **{f"at_{p}pct": pct(r[f"at_{p}pct"]) for p in (1, 5)}, "auroc": r["auroc"], "auroc_ci95": r["auroc_ci95"],
             "positive_runs": r["positive_runs"], "negative_runs": r["negative_runs"]} for r in t["rows"]]
    with open(OUT / "catch_rows_2026-10-10_235259.json", "w") as fh:
        fh.write("{\n" + f' "source": "{TPR.name}", "source_sha256": "{sha(TPR)}", "change": {json.dumps(t["change"])},\n "rows": [\n')
        fh.write(",\n".join("  " + json.dumps(r) for r in rows) + "\n ]\n}\n")
    s = json.loads(STAB.read_text())
    keep = [{k: v for k, v in r.items() if k != "roc"} for r in s["rows"]]
    with open(OUT / "threshold_stability_trim_2026-10-11_002451.json", "w") as fh:
        fh.write("{\n" + f' "source": "{STAB.name}", "source_sha256": "{sha(STAB)}", "caveat": {json.dumps(s["caveat"])},\n "rows": [\n')
        fh.write(",\n".join("  " + json.dumps(r) for r in keep) + "\n ]\n}\n")
    curves = [{"name": r["id"], "source": STAB.name, "n_pos": r["positive_runs"], "n_neg": r["negative_runs"], "auroc": r["auroc"],
               "tpr": down(r["roc"]["fpr"], r["roc"]["tpr"])} for r in s["rows"]]
    d = json.loads(gzip.decompress(DUMP.read_bytes()))
    test = {"clinical_trial", "media_processing", "shared_dotfiles", "tiktok", "user_tracking"}
    for model, key in (("Qwen3-32B (step B output_end)", "stepB"), ("Llama-3.2-1B", "llama")):
        runs = next(g for g in d["results"] if g["name"] == f"{key}|level1|fixed_vs_e1|by_readout.output_end.best")["runs"]
        for scope in ("test 5", "all 10"):
            rr = [r for r in runs if scope == "all 10" or r["environment"] in test]
            y, sc = np.array([r["label"] for r in rr]), np.array([r["score"] for r in rr])
            fpr, tpr, _ = roc_curve(y, sc)
            curves.append({"name": f"{model}|fixed_vs_e1|{scope}", "source": DUMP.name, "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()), "tpr": down(fpr, tpr)})
    with open(OUT / "roc_points_2026-10-11.json", "w") as fh:
        fh.write("{\n" + f' "fpr_grid": {json.dumps([round(float(g), 6) for g in GRID])},\n "sources_sha256": {json.dumps({STAB.name: sha(STAB), DUMP.name: sha(DUMP)})},\n "curves": [\n')
        fh.write(",\n".join("  " + json.dumps(c) for c in curves) + "\n ]\n}\n")
    q = json.loads(QVL.read_text())
    with open(OUT / "qwen_vs_llama_POSTHOC_trim_2026-10-11_003609.json", "w") as fh:
        fh.write("{\n" + f' "source": "{QVL.name}", "source_sha256": "{sha(QVL)}", "label": {json.dumps(q["label"])},\n')
        fh.write(",\n".join(f' "{s}": {json.dumps(q[s])}' for s in ("test 5", "all 10")) + "\n}\n")
    for f in sorted(OUT.iterdir()):
        print(f, sum(1 for _ in open(f)), "lines")


if __name__ == "__main__":
    main()
