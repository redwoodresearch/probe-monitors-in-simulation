"""Per-bar intervals for the probe-vs-TF-IDF figure (Monday daily, 0d's request 2026-10-11 ~14:10 PT; $0, local CPU).

b7's paired bootstrap (experiments/2026-10-11_claimB_q11_posthoc/analyze_POSTHOC_probe_vs_tfidf.py, d194b192 / aa4e57be) gives only
the probe − TF-IDF difference's interval. This repeats its exact task draws (same seed, same order, same skip rule) and also
keeps each AUROC per draw, so the two bars and their difference share one set of resampled tasks. Gates, or it stops:
all five input files (three score files, two published results) match the sha256 values b7's committed result records; both AUROCs reproduce the published values; and
the difference's interval reproduces b7's committed one to 4 decimals.

  uv run --no-project --with numpy --with scikit-learn python \\
      experiments/2026-10-11_daily_figures/analyze_probe_vs_tfidf_bars.py --fixed-sim-dir <a checkout's experiments/2026-10-09_fixed_sim>
"""

import argparse
import gzip
import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
REF = "aa4e57be"  # b7's paired result (and its input hashes) this must reproduce: the same blob as d194b192, on #614's branch
                  # (aswin/claimb-q11-tfidf), which main reaches when #614 merges; d194b192 is on no PR branch (Codex P1 on #729)
B7_RESULT = "experiments/2026-10-11_claimB_q11_posthoc/results/probe_vs_tfidf_POSTHOC_2026-10-11_040340.json"
PUBLISHED_TFIDF = "results/tfidf_per_env_2026-10-10_112443.json"
PUBLISHED_PROBE = "results/tpr_at_fpr_2026-10-10_235259.json"
PROBE = {"fixed_vs_e1": "data/claimB_output_scores_A_2026-10-10_022646.json.gz",
         "e2_vs_e1": "data/claimB_output_scores_A_e2_vs_e1_2026-10-10_024204.json.gz"}
TFIDF = "data/tfidf_run_scores_2026-10-10_112443.json.gz"
TFIDF_KEY = {"fixed_vs_e1": "fixed_sim_vs_e1|level1", "e2_vs_e1": "e2_vs_e1|level1"}
POS = {"fixed_vs_e1": "sim-attack-fixed", "e2_vs_e1": "real-attack-e2"}
TEST = {"clinical_trial", "media_processing", "shared_dotfiles", "tiktok", "user_tracking"}
ALIAS = {"sim-attack": "sim-attack-fixed"}


def git_json(path):
    root = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    return json.loads(subprocess.run(["git", "-C", root, "show", f"{REF}:{path}"], capture_output=True, text=True, check=True).stdout)


def runs_from_outputs(rows):
    """Output rows -> {run key: score (mean of its outputs), arm, env, task}, as analyze_tpr_at_fpr.runs_from_outputs with np.mean."""
    by, meta = defaultdict(list), {}
    for r in rows:
        if not np.isfinite(r["score"]):
            raise SystemExit(f"non-finite score in {r['key']}")
        by[r["key"]].append(r["score"])
        m = (r["arm"], r["environment"], r["main_task"], r.get("cell"))
        if meta.setdefault(r["key"], m) != m:
            raise SystemExit(f"run {r['key']} has inconsistent metadata")
    return {k: {"score": float(np.mean(v)), "arm": meta[k][0], "env": meta[k][1], "task": meta[k][2]} for k, v in by.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed-sim-dir", type=Path, required=True)  # the S3-archived files, not in git; hashes are checked
    args = ap.parse_args()
    b7 = git_json(B7_RESULT)
    want = {k.removeprefix("experiments/2026-10-09_fixed_sim/"): v for k, v in b7["inputs"].items()}
    data = {}
    for name in [*PROBE.values(), TFIDF, PUBLISHED_TFIDF, PUBLISHED_PROBE]:
        raw = (args.fixed_sim_dir / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != want[name]:
            raise SystemExit(f"{name}: sha256 differs from b7's recorded input")
        data[name] = json.loads(gzip.decompress(raw) if name.endswith(".gz") else raw)
    pub_tf, pub_pr = data[PUBLISHED_TFIDF]["rows"], {r["id"]: r for r in data[PUBLISHED_PROBE]["rows"]}
    out = {"label": "Per-bar AUROC intervals on b7's paired task draws (probe vs TF-IDF, level 1, 5 test environments)",
           "reproduces": f"{B7_RESULT}@{REF}", "inputs_sha256": want, "seed": b7["seed"], "draws": b7["draws"], "rows": {}}
    for comp, name in PROBE.items():
        pr = {k: v for k, v in runs_from_outputs(data[name]["levels"]["level1"]["rows"]).items() if v["env"] in TEST}
        tf = {r["key"]: r for r in data[TFIDF]["runs"][TFIDF_KEY[comp]] if r["environment"] in TEST}
        if set(pr) != set(tf):
            raise SystemExit(f"{comp}: run sets differ")
        keys = sorted(pr)
        if any(pr[k]["arm"] != ALIAS.get(tf[k]["arm"], tf[k]["arm"]) or pr[k]["task"] != tf[k]["main_task"] for k in keys):
            raise SystemExit(f"{comp}: arm or main task disagrees between the two score files")
        y = np.array([pr[k]["arm"] == POS[comp] for k in keys])
        sp, st = np.array([pr[k]["score"] for k in keys]), np.array([tf[k]["score"] for k in keys])
        a_p, a_t = roc_auc_score(y, sp), roc_auc_score(y, st)
        te = pub_tf[TFIDF_KEY[comp]]["test_envs"]
        ref_t, tol_t = (te["gate_a"], 1e-9) if "gate_a" in te else (te["auroc"], 1e-4)
        if abs(a_p - pub_pr[f"step A|level1 · run mean|{comp}|test 5"]["auroc"]) > 1e-9 or abs(a_t - ref_t) > tol_t:
            raise SystemExit(f"{comp}: AUROCs {a_p}, {a_t} do not reproduce the published values")
        tasks, inv = np.unique([pr[k]["task"] for k in keys], return_inverse=True)
        members = [np.flatnonzero(inv == j) for j in range(len(tasks))]
        rng = np.random.default_rng(b7["seed"])
        ap_, at_ = [], []
        for _ in range(b7["draws"]):
            idx = np.concatenate([members[j] for j in rng.choice(len(tasks), len(tasks))])
            if y[idx].all() or not y[idx].any():
                continue
            ap_.append(roc_auc_score(y[idx], sp[idx]))
            at_.append(roc_auc_score(y[idx], st[idx]))
        ap_, at_ = np.array(ap_), np.array(at_)
        ci = lambda x: [float(v) for v in np.percentile(x, [2.5, 97.5])]  # noqa: E731
        diff_ci = ci(ap_ - at_)
        if any(abs(a - b) > 5e-5 for a, b in zip(diff_ci, b7["rows"][comp]["ci95"])):
            raise SystemExit(f"{comp}: difference interval {diff_ci} does not reproduce b7's {b7['rows'][comp]['ci95']}")
        out["rows"][comp] = {"runs": len(keys), "positives": int(y.sum()), "main_tasks": len(tasks), "draws_used": len(ap_),
                             "auroc_probe": a_p, "auroc_probe_ci95": ci(ap_), "auroc_tfidf": a_t, "auroc_tfidf_ci95": ci(at_),
                             "difference": a_p - a_t, "difference_ci95": diff_ci}
        r = out["rows"][comp]
        print(f"{comp}: probe {a_p:.4f} {r['auroc_probe_ci95']}, TF-IDF {a_t:.4f} {r['auroc_tfidf_ci95']}, diff {a_p - a_t:+.4f} {diff_ci}")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    out |= {"git": git, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    dst = HERE / "results" / f"probe_vs_tfidf_bars_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print("->", dst)


if __name__ == "__main__":
    main()
