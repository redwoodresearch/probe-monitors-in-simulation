"""AI-judge planning ($0): the margin X for "the judge's run-level AUROC is within X of the probe's", from stored per-run
scores, with TF-IDF (gate (a), level 1) standing in for the judge. Also X for the judge's e2-vs-e1 pass criterion, and
TF-IDF's catch at 1% and 5% by the 1% table's two-way cross-fit (a planning number; the table has no TF-IDF row).

Joins and gates as `2026-10-11_claimB_q11_posthoc/analyze_POSTHOC_probe_vs_tfidf.py`: the same 946 test runs per
comparison, and both AUROCs must reproduce their published values. Intervals: 2,000 main-task draws
(default_rng(20261011)). X80 = (1.645 + 1.2816) x SE: two one-sided 5% tests reach 80% power when the true difference
is 0. SE is taken two ways: paired (the stand-in's draws, which share runs with the probe's) and a zero-covariance
scenario (sqrt(SE_probe^2 + SE_judge^2)). Neither bounds a real judge: a negative judge-probe covariance, or a judge
noisier than TF-IDF, would give a larger SE. Reduced N:
20 seeded subsamples of whole main tasks sized to about 200 and 100 cells per arm (the score files carry no cell id;
cells per task = runs per arm / main tasks), 500 draws each.

  uv run --no-project --python 3.12 --with numpy --with scikit-learn \\
      python experiments/2026-10-11_claimB_judge/analyze_judge_power.py
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
FIXED = Path.home() / "probe-shift-sims/experiments/2026-10-09_fixed_sim"  # the git-ignored data live in the main checkout
sys.path.insert(0, str(FIXED))
import analyze_tpr_at_fpr as T  # noqa: E402  (load_gz, runs_from_outputs, crossfit, SEED, TARGETS)

PROBE = {"fixed_vs_e1": FIXED / "data/claimB_output_scores_A_2026-10-10_022646.json.gz",
         "e2_vs_e1": FIXED / "data/claimB_output_scores_A_e2_vs_e1_2026-10-10_024204.json.gz"}
TFIDF = FIXED / "data/tfidf_run_scores_2026-10-10_112443.json.gz"
TFIDF_KEY = {"fixed_vs_e1": "fixed_sim_vs_e1|level1", "e2_vs_e1": "e2_vs_e1|level1"}
PUBLISHED_PROBE = FIXED / "results/tpr_at_fpr_2026-10-10_235259.json"
PUBLISHED_TFIDF = FIXED / "results/tfidf_per_env_2026-10-10_112443.json"
POS = {"fixed_vs_e1": "sim-attack-fixed", "e2_vs_e1": "real-attack-e2"}
TEST = {"clinical_trial", "media_processing", "shared_dotfiles", "tiktok", "user_tracking"}
ALIAS = {"sim-attack": "sim-attack-fixed"}
SEED, DRAWS, Z = 20261011, 2000, 1.645 + 1.2816


def boot(y, sp, st, task, draws, rng):
    tasks, inv = np.unique(task, return_inverse=True)
    members = [np.flatnonzero(inv == j) for j in range(len(tasks))]
    a_p, a_t = [], []
    for _ in range(draws):
        idx = np.concatenate([members[j] for j in rng.choice(len(tasks), len(tasks))])
        if y[idx].all() or not y[idx].any():
            continue
        a_p.append(roc_auc_score(y[idx], sp[idx])), a_t.append(roc_auc_score(y[idx], st[idx]))
    a_p, a_t = np.array(a_p), np.array(a_t)
    se_p, se_t, se_d = float(a_p.std()), float(a_t.std()), float((a_p - a_t).std())
    return {"se_probe": se_p, "se_standin": se_t, "se_paired_diff": se_d, "se_independent_diff": float(np.hypot(se_p, se_t)),
            "corr": float(np.corrcoef(a_p, a_t)[0, 1]), "draws_used": len(a_p)}


def main():
    tf_all = T.load_gz(TFIDF)["runs"]
    pub_pr = {r["id"]: r for r in json.loads(PUBLISHED_PROBE.read_text())["rows"]}
    pub_tf = json.loads(PUBLISHED_TFIDF.read_text())["rows"]
    out = {"label": "AI-judge planning: margin X from stored scores, TF-IDF as the stand-in judge (no text; preliminary)", "z_tost80": Z, "rows": {}}
    for comp, path in PROBE.items():
        probe = T.runs_from_outputs(T.load_gz(path)["levels"]["level1"]["rows"], np.mean)
        pr = {k: v for k, v in probe.items() if v["env"] in TEST}
        tf = {r["key"]: r for r in tf_all[TFIDF_KEY[comp]] if r["environment"] in TEST}
        if set(pr) != set(tf) or any(pr[k]["arm"] != ALIAS.get(tf[k]["arm"], tf[k]["arm"]) for k in pr):
            raise SystemExit(f"{comp}: the two score files cover different runs or arms")
        keys = sorted(pr)
        y = np.array([pr[k]["arm"] == POS[comp] for k in keys])
        sp, st = np.array([pr[k]["score"] for k in keys]), np.array([tf[k]["score"] for k in keys])
        task = np.array([pr[k]["task"] for k in keys])
        a_p, a_t = roc_auc_score(y, sp), roc_auc_score(y, st)
        te = pub_tf[TFIDF_KEY[comp]]["test_envs"]
        ref_t, tol_t = (te["gate_a"], 1e-9) if "gate_a" in te else (te["auroc"], 1e-4)
        if abs(a_p - pub_pr[f"step A|level1 · run mean|{comp}|test 5"]["auroc"]) > 1e-9 or abs(a_t - ref_t) > tol_t:
            raise SystemExit(f"{comp}: AUROCs do not reproduce the published values")
        row = {"runs": len(keys), "main_tasks": len(np.unique(task)), "auroc_probe": a_p, "auroc_standin": a_t}
        row["full"] = boot(y, sp, st, task, DRAWS, np.random.default_rng(SEED))
        for k in ("paired", "independent"):
            row["full"][f"X80_{k}"] = Z * row["full"][f"se_{k}_diff"]
        row["full"]["X80_auroc_vs_half"] = Z * row["full"]["se_standin"]  # e2-vs-e1 pass: within X of 0.5
        # catch at 1% and 5% for the stand-in, by the 1% table's two-way cross-fit (default_rng(20261010) task order)
        tasks, inv = np.unique(task, return_inverse=True)
        members = [np.flatnonzero(inv == j) for j in range(len(tasks))]
        cf = T.crossfit(y, st, members, np.ones(len(tasks), int), np.random.default_rng(T.SEED).permutation(len(tasks)))
        row["standin_catch"] = {f"at_{p}pct": {"caught": cf[p]["caught"], "positives": cf[p]["positives"], "false_alarms": cf[p]["false_alarms"],
                                               "negatives": cf[p]["negatives"]} for p in T.TARGETS}
        per_task = (len(keys) / 2) / len(tasks)  # cells per main task
        for n in (200, 100):
            ses, k_tasks = [], int(round(n / per_task))
            for s in range(20):
                keep = np.isin(task, np.random.default_rng(SEED + 100 + s).choice(tasks, k_tasks, replace=False))
                ses.append(boot(y[keep], sp[keep], st[keep], task[keep], 500, np.random.default_rng(SEED + 200 + s)))
            m = {k: float(np.mean([x[k] for x in ses])) for k in ("se_probe", "se_standin", "se_paired_diff", "se_independent_diff")}
            m["main_tasks"] = k_tasks
            m |= {"X80_paired": Z * m["se_paired_diff"], "X80_independent": Z * m["se_independent_diff"], "X80_auroc_vs_half": Z * m["se_standin"]}
            row[f"{n}_cells"] = m
        out["rows"][comp] = row
        print(comp, {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items() if not isinstance(v, dict)})
        for k in ("full", "200_cells", "100_cells"):
            print("  ", k, {x: round(v, 4) for x, v in row[k].items() if isinstance(v, float)})
        print("   standin catch", row["standin_catch"])
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    out |= {"git": git, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "inputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [*PROBE.values(), TFIDF, PUBLISHED_PROBE, PUBLISHED_TFIDF]}}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    dst = HERE / "results" / f"judge_power_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print("->", dst)


if __name__ == "__main__":
    main()
