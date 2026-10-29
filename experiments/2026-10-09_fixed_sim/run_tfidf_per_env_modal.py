"""Gate (a)'s TF-IDF per environment (Aswin asked for per-environment results for the probe and the text classifier; gate
(a) kept only pooled AUROCs). One Modal CPU container reruns gate (a)'s leave-one-environment-out fits over all 10
environments with its own functions and config (run_gate_a_modal.py prepare / design_jobs / score_jobs, config/gate_a.yaml,
unchanged) for fixed sim vs real e1 and real e2 vs e1, raw (variant all) and times masked (level1), and returns run
scores (key, arm, environment, cell, main task; no text). Locally, before anything else: the pooled run-level AUROCs are
compared with gate (a)'s archived rows (results/gate_a_main_2026-10-09_185451.json); liblinear shuffles with an unseeded
state, so small differences are recorded, and a difference over 0.005 stops the script. Then per environment: run-level
AUROC with a classifier-fixed null (1,000 draws swapping the two runs' labels within each cell with probability 1/2; the
fits are not rerun). Writes data/tfidf_run_scores_<stamp>.json.gz and results/tfidf_per_env_<stamp>.json.

  cd <worktree> && PYTHONPATH=src uv run --no-project --python 3.13 --with 'modal>=1.4.3' --with pyyaml --with numpy --with scikit-learn \
      modal run experiments/2026-10-09_fixed_sim/run_tfidf_per_env_modal.py
"""

import gzip
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REPO = HERE.parents[1] if modal.is_local() else Path("/root")
sys.path.insert(0, str(HERE) if modal.is_local() else "/root")
import run_gate_a_modal as G  # noqa: E402  (gate (a)'s code, unchanged)

app = modal.App("probe-shift_aswin-surya_sub-fixed-sim_tfidf-per-env_kill-2026-11-10")
image = G.image.add_local_file(HERE / "run_gate_a_modal.py", "/root/run_gate_a_modal.py")
VARIANTS, ARCHIVED = ["all", "level1"], HERE / "results" / "gate_a_main_2026-10-09_185451.json"


@app.function(image=image, volumes={"/acts": G.acts}, cpu=16, memory=65536, timeout=1500, retries=0)
def job(cfg: dict, split: dict, folds: dict) -> dict:
    import numpy as np

    import analyze_tfidf as A

    out = {}
    for variant in VARIANTS:
        d, X, env, _, _ = G.prepare(cfg, variant, split, folds)
        for comp, (neg, pos) in cfg["comparisons"].items():
            R = np.isin(d["arm"], [neg, pos])
            y = (d["arm"] == pos).astype(int)
            s = G.score_jobs(X, y, G.design_jobs("loeo", R, env, d, split, all_envs=True), cfg)
            run_score, first = A.run_means(s, d["key"], R)
            out[f"{comp}|{variant}"] = [{"key": str(d["key"][i]), "arm": str(d["arm"][i]), "environment": str(env[i]), "cell": str(d["cell"][i]),
                                         "main_task": str(d["main_task"][i]), "score": float(v)} for i, v in zip(first, run_score)]
            print(comp, variant, len(first), "runs", flush=True)
        del X
    return out


def per_env(runs: list[dict], pos: str, test: list[str], rng) -> dict:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    out = {}
    for e in sorted({r["environment"] for r in runs}):
        rr = [r for r in runs if r["environment"] == e]
        y = np.array([r["arm"] == pos for r in rr])
        s = np.array([r["score"] for r in rr])
        cells = {}
        for i, r in enumerate(rr):
            cells.setdefault(r["cell"], []).append(i)
        pairs = [g for g in cells.values() if len(g) == 2 and y[g].sum() == 1]
        obs, null = roc_auc_score(y, s), []
        for _ in range(1000):
            yy = y.copy()
            for g in pairs:
                if rng.random() < 0.5:
                    yy[g] = yy[g[::-1]]
            null.append(roc_auc_score(yy, s))
        out[e] = {"split": "test" if e in test else "dev", "auroc": round(float(obs), 4), "runs": {"positive": int(y.sum()), "negative": int((~y).sum())},
                  "null_p95": round(float(np.percentile(null, 95)), 4), "p": round(float((1 + (np.array(null) >= obs).sum()) / 1001), 4)}
    return out


@app.local_entrypoint()
def main():
    import numpy as np
    import yaml
    from sklearn.metrics import roc_auc_score

    if subprocess.run(["git", "status", "--porcelain", str(Path(__file__))], capture_output=True, text=True, check=True).stdout:
        raise SystemExit(f"commit {Path(__file__).name} first: the result records the commit that ran")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    cfg = yaml.safe_load((HERE / "config" / "gate_a.yaml").read_text())
    split = json.loads((HERE / cfg["split_file"]).read_text())
    folds = json.loads((REPO / "splits" / cfg["folds_file"]).read_text())
    res = job.remote(cfg, split, folds)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(HERE / "data" / f"tfidf_run_scores_{stamp}.json.gz", "xb") as fh:  # keys and scores, no text
        fh.write(gzip.compress(json.dumps({"git": git, "runs": res}).encode(), mtime=0))
    archived = json.loads(ARCHIVED.read_text())["rows"]
    out, rng = {"git": git, "config": "config/gate_a.yaml", "archived": str(ARCHIVED.name), "rows": {}}, np.random.default_rng(0)
    for k, runs in res.items():
        comp, variant = k.split("|")
        pos = cfg["comparisons"][comp][1]
        for scope, envs in (("test_envs", split["test"]), ("all_envs", split["test"] + split["dev"])):
            rr = [r for r in runs if r["environment"] in envs]
            got = roc_auc_score([r["arm"] == pos for r in rr], [r["score"] for r in rr])
            want = archived[f"{comp}|loeo|{variant}"].get(scope, {}).get("auroc")
            if want is not None and abs(got - want) > 0.005:  # same code, config and data: only solver noise may differ
                raise SystemExit(f"{k} {scope}: {got} does not reproduce gate (a)'s {want}")
            out["rows"].setdefault(k, {})[scope] = {"auroc": round(float(got), 4), "gate_a": want, "diff_from_gate_a": None if want is None else got - want}
        out["rows"][k]["per_environment"] = pe = per_env(runs, pos, split["test"], rng)
        print(k, {s: out["rows"][k][s]["auroc"] for s in ("test_envs", "all_envs")}, {e: (v["auroc"], v["null_p95"]) for e, v in pe.items()})
    with open(HERE / "results" / f"tfidf_per_env_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print("->", HERE / "results" / f"tfidf_per_env_{stamp}.json")
