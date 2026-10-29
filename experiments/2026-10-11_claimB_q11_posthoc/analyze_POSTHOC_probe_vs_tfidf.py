"""POST HOC (PLAN.md check 2; 0d's skeptic-panel request, 2026-10-11): paired, task-resampled interval for step A's probe
AUROC minus TF-IDF's (gate (a)) on the same held-out runs. Level 1, 5 test environments; fixed sim vs real e1, and
real e2 vs real e1 as the control. Local and free: stored per-run scores only (no text). Gate: both AUROCs reproduce
their published values exactly and the two scores cover the same runs, or the script stops.

  uv run --no-project --python 3.13 --with numpy --with scikit-learn \\
      python experiments/2026-10-11_claimB_q11_posthoc/analyze_POSTHOC_probe_vs_tfidf.py
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
FIXED = HERE.parent / "2026-10-09_fixed_sim"
sys.path.insert(0, str(FIXED))
import analyze_tpr_at_fpr as T  # noqa: E402  (load_gz, runs_from_outputs)

PROBE = {"fixed_vs_e1": FIXED / "data/claimB_output_scores_A_2026-10-10_022646.json.gz",
         "e2_vs_e1": FIXED / "data/claimB_output_scores_A_e2_vs_e1_2026-10-10_024204.json.gz"}
TFIDF = FIXED / "data/tfidf_run_scores_2026-10-10_112443.json.gz"
TFIDF_KEY = {"fixed_vs_e1": "fixed_sim_vs_e1|level1", "e2_vs_e1": "e2_vs_e1|level1"}
PUBLISHED_TFIDF = FIXED / "results/tfidf_per_env_2026-10-10_112443.json"
PUBLISHED_PROBE = FIXED / "results/tpr_at_fpr_2026-10-10_235259.json"
POS = {"fixed_vs_e1": "sim-attack-fixed", "e2_vs_e1": "real-attack-e2"}
TEST = {"clinical_trial", "media_processing", "shared_dotfiles", "tiktok", "user_tracking"}
SEED, DRAWS = 20261011, 2000
ALIAS = {"sim-attack": "sim-attack-fixed"}  # the TF-IDF run file names the fixed-sim arm by its run alias


def main():
    tf_all = T.load_gz(TFIDF)["runs"]
    pub_tf = json.loads(PUBLISHED_TFIDF.read_text())["rows"]
    pub_pr = {r["id"]: r for r in json.loads(PUBLISHED_PROBE.read_text())["rows"]}
    out = {"label": "POST HOC (0d skeptic-panel check 2, 2026-10-11); paired task bootstrap; preliminary until reviewed",
           "inputs": {str(p.relative_to(HERE.parent.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [*PROBE.values(), TFIDF, PUBLISHED_TFIDF, PUBLISHED_PROBE]},
           "seed": SEED, "draws": DRAWS, "rows": {}}
    for comp, path in PROBE.items():
        probe = T.runs_from_outputs(T.load_gz(path)["levels"]["level1"]["rows"], np.mean)
        tf = {r["key"]: r for r in tf_all[TFIDF_KEY[comp]] if r["environment"] in TEST}
        pr = {k: v for k, v in probe.items() if v["env"] in TEST}
        if set(pr) != set(tf):
            raise SystemExit(f"{comp}: run sets differ ({len(set(pr) - set(tf))} probe-only, {len(set(tf) - set(pr))} TF-IDF-only)")
        keys = sorted(pr)
        if any(pr[k]["arm"] != ALIAS.get(tf[k]["arm"], tf[k]["arm"]) or pr[k]["task"] != tf[k]["main_task"] for k in keys):
            raise SystemExit(f"{comp}: arm or main task disagrees between the two score files")
        y = np.array([pr[k]["arm"] == POS[comp] for k in keys])
        sp, st = np.array([pr[k]["score"] for k in keys]), np.array([tf[k]["score"] for k in keys])
        a_p, a_t = roc_auc_score(y, sp), roc_auc_score(y, st)
        ref_p = pub_pr[f"step A|level1 · run mean|{comp}|test 5"]["auroc"]
        te = pub_tf[TFIDF_KEY[comp]]["test_envs"]
        ref_t, tol_t = (te["gate_a"], 1e-9) if "gate_a" in te else (te["auroc"], 1e-4)  # "auroc" is rounded to 4 decimals
        if abs(a_p - ref_p) > 1e-9 or abs(a_t - ref_t) > tol_t:
            raise SystemExit(f"{comp}: AUROCs {a_p}, {a_t} do not reproduce the published {ref_p}, {ref_t}")
        tasks, inv = np.unique([pr[k]["task"] for k in keys], return_inverse=True)
        members = [np.flatnonzero(inv == j) for j in range(len(tasks))]
        rng = np.random.default_rng(SEED)
        diffs, skipped = [], 0
        for _ in range(DRAWS):
            idx = np.concatenate([members[j] for j in rng.choice(len(tasks), len(tasks))])
            if y[idx].all() or not y[idx].any():
                skipped += 1
                continue
            diffs.append(roc_auc_score(y[idx], sp[idx]) - roc_auc_score(y[idx], st[idx]))
        diffs = np.array(diffs)
        row = {"runs": len(keys), "positives": int(y.sum()), "main_tasks": len(tasks), "auroc_probe": a_p, "auroc_tfidf": a_t,
               "difference": a_p - a_t, "ci95": [float(x) for x in np.percentile(diffs, [2.5, 97.5])],
               "share_of_draws_at_or_below_0": float((diffs <= 0).mean()), "draws_used": len(diffs), "draws_skipped": skipped}
        out["rows"][comp] = row
        print(f"{comp} [level 1, test 5, {row['runs']} runs, {row['main_tasks']} main tasks]: probe {a_p:.4f} - TF-IDF {a_t:.4f} = "
              f"{row['difference']:+.4f} [{row['ci95'][0]:+.4f}, {row['ci95'][1]:+.4f}]; draws <= 0: {row['share_of_draws_at_or_below_0']:.4f}")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    out |= {"git": git, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    dst = HERE / "results" / f"probe_vs_tfidf_POSTHOC_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print("->", dst)


if __name__ == "__main__":
    main()
