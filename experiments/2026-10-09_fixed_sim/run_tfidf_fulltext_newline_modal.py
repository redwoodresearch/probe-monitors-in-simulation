"""Exploratory diagnostic (not a re-test of a fix; gate (a) stays as it was): does the create outputs' trailing newline carry
TF-IDF's separation when the classifier reads each output's FULL text? Gate (a) reads `readout_text`, which ends at the
last content token, so it never saw that newline (0 of 1,322 simulated create outputs end with one there, 977 in `text`).
Aswin, about 14:42 PT 10-10, relayed by 0d: "run it if it's cheap, u dont' need to ask below $5 remember".

One Modal CPU container: gate (a)'s leave-one-environment-out TF-IDF over all 10 environments (its code and settings,
config/gate_a.yaml) with `text` in place of `readout_text`, at levels 0 and 1:
  before   the full text as it is;
  after    every text_editor create output loses its trailing newlines, in both arms (run_tfidf_slips_evened_modal's
           even_create; real create outputs have none to lose).
Fixed sim vs real e1 before and after; real e2 vs e1 before only (the evening changes no real output). Locally: run-level
AUROC on the 5 test environments and all 10, per environment, a classifier-fixed null (one coin per main task), and the
number of outputs the evening changed per arm.

  PYTHONPATH=src uv run --no-project --python 3.13 --with 'modal>=1.4.3' --with pyyaml --with numpy --with scikit-learn \\
      modal run experiments/2026-10-09_fixed_sim/run_tfidf_fulltext_newline_modal.py
"""

import gzip
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REPO = HERE.parents[1] if modal.is_local() else Path("/root")
sys.path.insert(0, str(HERE) if modal.is_local() else "/root")
import run_gate_a_modal as G  # noqa: E402  (gate (a)'s code, unchanged)
import run_tfidf_slips_evened_modal as SL  # noqa: E402  (even_create, summarize)

app = modal.App("probe-shift_aswin-surya_sub-fixed-sim_tfidf-fulltext_kill-2026-11-10")
image = G.image.add_local_file(HERE / "run_gate_a_modal.py", "/root/run_gate_a_modal.py").add_local_file(
    HERE / "run_tfidf_slips_evened_modal.py", "/root/run_tfidf_slips_evened_modal.py")
LEVELS = {"level0": 0, "level1": 1}
ROWS = [("fixed_sim_vs_e1", "before"), ("fixed_sim_vs_e1", "after"), ("e2_vs_e1", "before")]


@app.function(image=image, volumes={"/acts": G.acts}, cpu=16, memory=65536, timeout=1500, retries=0)
def job(cfg: dict, split: dict, folds: dict, git: str) -> dict:
    import numpy as np

    import analyze_tfidf as A

    res = {"runs": {}, "changed": {}}
    d = A.load(Path("/acts") / cfg["run"] / "examples", folds["environment_folds"]["environment_to_fold"], fold_key="environment")
    env = np.array([r["environment"] for r in d["rows"]])
    raw = [r["text"] for r in d["rows"]]
    after, changed = [], Counter()
    for r, t, c in zip(d["rows"], raw, d["editor_command"]):
        e = SL.even_create(t) if c == "create" else t
        changed[r["arm"]] += e != t
        after.append(e)
    res["changed"] = dict(changed)
    out = Path("/acts/claimB") / f"tfidf_fulltext_{git[:8]}"  # each row written as it finishes
    out.mkdir(parents=True, exist_ok=True)
    (out / "changed.json").write_text(json.dumps(res["changed"]))
    G.acts.commit()
    for level, scrub in LEVELS.items():
        for cond, texts in (("before", raw), ("after", after)):
            if not any(c == cond for _, c in ROWS):
                continue
            X = A.featurize([A.scrub(t, scrub) for t in texts], cfg)
            for comp, c in ROWS:
                if c != cond:
                    continue
                neg, pos = cfg["comparisons"][comp]
                R = np.isin(d["arm"], [neg, pos])
                y = (d["arm"] == pos).astype(int)
                s = G.score_jobs(X, y, G.design_jobs("loeo", R, env, d, split, all_envs=True), cfg)
                run_score, first = A.run_means(s, d["key"], R)
                res["runs"][f"{comp}|{cond}|{level}"] = [{"key": str(d["key"][i]), "arm": str(d["arm"][i]), "environment": str(env[i]),
                                                          "cell": str(d["cell"][i]), "main_task": str(d["main_task"][i]), "score": float(v)}
                                                         for i, v in zip(first, run_score)]
                (out / f"{comp}|{cond}|{level}.json").write_text(json.dumps(res["runs"][f"{comp}|{cond}|{level}"]))
                G.acts.commit()
                print(comp, cond, level, len(first), "runs", flush=True)
            del X
    return res


@app.local_entrypoint()
def main(no_wait: bool = False, collect: str = ""):
    """`--no-wait` (with --detach): spawn and return. `--collect <commit[:8]>`: read the rows the worker wrote to the
    Volume and summarize them (no launch)."""
    import numpy as np
    import yaml

    if subprocess.run(["git", "status", "--porcelain", str(Path(__file__))], capture_output=True, text=True, check=True).stdout:
        raise SystemExit(f"commit {Path(__file__).name} first: the result records the commit that ran")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    cfg = yaml.safe_load((HERE / "config" / "gate_a.yaml").read_text())
    split = json.loads((HERE / cfg["split_file"]).read_text())
    folds = json.loads((REPO / "splits" / cfg["folds_file"]).read_text())
    if collect:
        base = f"claimB/tfidf_fulltext_{collect}"
        names = [e.path.rsplit("/", 1)[-1] for e in G.acts.listdir(base)]
        res = {"changed": json.loads(b"".join(G.acts.read_file(f"{base}/changed.json"))),
               "runs": {n.removesuffix(".json"): json.loads(b"".join(G.acts.read_file(f"{base}/{n}"))) for n in names if n != "changed.json"}}
        git = collect
    elif no_wait:
        print(f"spawned {job.spawn(cfg, split, folds, git).object_id}; rows go to claimB/tfidf_fulltext_{git[:8]}/", flush=True)
        return
    else:
        res = job.remote(cfg, split, folds, git)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(HERE / "data" / f"tfidf_fulltext_run_scores_{stamp}.json.gz", "xb") as fh:  # keys and scores, no text
        fh.write(gzip.compress(json.dumps({"git": git, "runs": res["runs"]}).encode(), mtime=0))
    rng, out = np.random.default_rng(0), {"git": git, "text_field": "text", "outputs_changed_by_evening": res["changed"], "rows": {}}
    for k, runs in res["runs"].items():
        out["rows"][k] = SL.summarize(runs, cfg["comparisons"][k.split("|")[0]][1], split["test"], rng)
        print(k, out["rows"][k]["test_envs"], "| all 10:", out["rows"][k]["all_envs"]["auroc"])
    print("outputs changed by the newline evening:", res["changed"])
    with open(HERE / "results" / f"tfidf_fulltext_newline_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print("->", HERE / "results" / f"tfidf_fulltext_newline_{stamp}.json")
