"""Follow-up (c), the text classifier: gate (a)'s TF-IDF (run_gate_a_modal.py's prepare / design_jobs / score_jobs /
run_auroc and analyze_tfidf.py, unchanged; config/gate_a.yaml's settings) on the September cohort's examples
(stage_probe_inputs.py's RUN_EX, arms renamed to gate (a)'s: real-attack-e1 = real run 1, real-attack-e2 = real run 2,
sim-attack = the fixed simulated run). Leave one environment out over all 10 environments; headline on the 5 test
environments (as claim B), all 10 and per environment alongside; variants level1 (times masked, primary), all (raw) and
planted (gate (a)'s line in one seeded output of every simulated run; must reach 0.90). Null: 200 refit draws per row,
one coin per main task (gate (a)'s null_job, copied below with its file layout), for fixed sim vs run 1 and run 2 vs run
1 at both levels. Everything is written to the Volume as it finishes; launches spawn and return.

  PYTHONPATH=src uv run --no-project --python 3.13 --with 'modal>=1.4.3' --with pyyaml --with numpy --with scikit-learn \
      modal run --detach experiments/2026-10-10_claimB_onpolicy/run_tfidf_onpolicy_modal.py --part main|null|collect
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
FS = HERE.parent / "2026-10-09_fixed_sim" if modal.is_local() else Path("/root")
REPO = HERE.parents[1] if modal.is_local() else Path("/root")
sys.path.insert(0, str(FS))
import run_gate_a_modal as G  # noqa: E402  (gate (a)'s code, unchanged)

RUN = "2026-10-10_la_september-onpolicy-examples_v1"
VARIANTS = ["level1", "all", "planted"]
NULL_ROWS = [("fixed_sim_vs_e1", "level1"), ("e2_vs_e1", "level1"), ("fixed_sim_vs_e1", "all"), ("e2_vs_e1", "all")]
DRAWS, PER, NULL_TIMEOUT = 200, 25, 1200
app = modal.App("probe-shift_aswin-surya_sub-claimb_onpol-tfidf_kill-2026-11-10")
image = G.image.add_local_file(FS / "run_gate_a_modal.py", "/root/run_gate_a_modal.py")


def config() -> tuple[dict, dict, dict]:
    import yaml

    cfg = yaml.safe_load((FS / "config" / "gate_a.yaml").read_text()) | {"run": RUN}
    cfg["variants"] = {k: v for k, v in cfg["variants"].items() if k in VARIANTS}
    return cfg, json.loads((FS / cfg["split_file"]).read_text()), json.loads((REPO / "splits" / cfg["folds_file"]).read_text())


@app.function(image=image, volumes={"/acts": G.acts}, cpu=16, memory=65536, timeout=2400, retries=0)
def main_rows(cfg: dict, split: dict, folds: dict, tag: str) -> None:
    import numpy as np

    import analyze_tfidf as A

    out = Path("/acts") / RUN / "onpolicy_tfidf" / tag / "rows"
    out.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        d, X, env, _, counts = G.prepare(cfg, variant, split, folds)
        test = np.isin(env, split["test"])
        for comp, (neg, pos) in cfg["comparisons"].items():
            if variant == "planted" and pos != "sim-attack":
                continue
            R, y = np.isin(d["arm"], [neg, pos]), (d["arm"] == pos).astype(int)
            s = G.score_jobs(X, y, G.design_jobs("loeo", R, env, d, split, all_envs=True), cfg)
            run_score, first = A.run_means(s, d["key"], R)
            row = {"test_envs": G.run_auroc(s, y, d, env, R & test, split["test"], cfg), "all_envs": G.run_auroc(s, y, d, env, R, sorted(set(env)), cfg),
                   "per_environment": {e: G.run_auroc(s, y, d, env, R & (env == e), [e], cfg, boot=False) for e in sorted(set(env))},
                   "counts": counts, "runs": [{"key": str(d["key"][i]), "arm": str(d["arm"][i]), "score": float(v)} for i, v in zip(first, run_score)]}
            with open(out / f"{comp}__{variant}.json", "x") as fh:
                fh.write(json.dumps(row))
            G.acts.commit()
            print(comp, variant, json.dumps(row["test_envs"]), flush=True)


@app.function(image=image, volumes={"/acts": G.acts}, cpu=8, memory=65536, timeout=NULL_TIMEOUT, retries=0)
def null_draws(cfg: dict, split: dict, folds: dict, comp: str, variant: str, draws: list[int], tag: str) -> None:
    """Gate (a)'s null_job (design loeo): each draw flips one coin per main task (seed + 1 + k), refits, scores the test
    environments; one file per container, so draws survive a lost client."""
    import numpy as np
    from joblib import Parallel, delayed

    import analyze_tfidf as A

    d, X, env, _, _ = G.prepare(cfg, variant, split, folds)
    neg, pos = cfg["comparisons"][comp]
    R, y = np.isin(d["arm"], [neg, pos]), (d["arm"] == pos).astype(int)
    test = np.isin(env, split["test"])
    tasks_u, task_idx = np.unique(d["main_task"], return_inverse=True)
    perms = {k: np.where((np.random.default_rng(cfg["seed"] + 1 + k).random(len(tasks_u)) < 0.5)[task_idx], 1 - y, y) for k in draws}
    work = [(k, tr, te) for k in draws for tr, te in G.design_jobs("loeo", R, env, d, split)]
    parts = Parallel(n_jobs=cfg["n_jobs"])(delayed(A._fold_scores)(X, perms[k], tr, te, cfg, cfg["C"]) for k, tr, te in work)
    scores = {k: np.full(len(y), np.nan) for k in draws}
    for (k, _, te), p in zip(work, parts):
        scores[k][te] = p
    out = {k: G.run_auroc(scores[k], perms[k], d, env, R & test, split["test"], cfg, boot=False)["auroc"] for k in draws}
    path = Path("/acts") / RUN / "gate_a_null" / tag / f"{comp}__loeo__{variant}" / f"draws_{min(draws)}-{max(draws)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x") as fh:
        fh.write(json.dumps({str(k): v for k, v in out.items()}))
    G.acts.commit()


def stored(prefix: str) -> dict:
    """{name: content} of every JSON file under RUN/prefix on the Volume."""
    out = {}
    try:
        for e in G.acts.listdir(f"{RUN}/{prefix}", recursive=True):
            if e.path.endswith(".json"):
                out[e.path[len(f"{RUN}/{prefix}/"):]] = json.loads(b"".join(G.acts.read_file(e.path)))
    except (FileNotFoundError, modal.exception.NotFoundError):
        pass
    return out


@app.local_entrypoint()
def main(part: str, git_tag: str = ""):
    import numpy as np

    if subprocess.run(["git", "status", "--porcelain", str(Path(__file__))], capture_output=True, text=True, check=True).stdout:
        raise SystemExit(f"commit {Path(__file__).name} first: results record the commit that ran")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    cfg, split, folds = config()
    if part == "collect" and not git_tag:
        raise SystemExit("--git-tag (the commit that launched main and null) is needed to collect")
    if part != "collect" and git_tag:
        raise SystemExit("--git-tag is for collect only: main and null always write under the launch commit")
    tag = git_tag or git[:8]  # main and null write under the launch commit's tag; collect reads one tag only
    draws = {}
    for name, v in stored(f"gate_a_null/{tag}").items():
        row = name.split("/")[0]
        for k, a in v.items():
            if int(k) in draws.setdefault(row, {}):
                raise ValueError(f"{row}: draw {k} is in two files")
            draws[row][int(k)] = a
    if part == "main":
        main_rows.spawn(cfg, split, folds, tag)
        print(f"spawned the main rows (rows under {RUN}/onpolicy_tfidf/{tag}/rows/); not waiting")
    elif part == "null":
        todo = [(c, v, ks[a:a + PER]) for c, v in NULL_ROWS
                for ks in [[k for k in range(DRAWS) if k not in draws.get(f"{c}__loeo__{v}", {})]] for a in range(0, len(ks), PER)]
        for c, v, ks in todo:
            null_draws.spawn(cfg, split, folds, c, v, ks, tag)
        print(f"spawned {len(todo)} null containers; not waiting")
    elif part == "collect":
        rows = {k.split("/")[-1].removesuffix(".json"): v for k, v in stored(f"onpolicy_tfidf/{tag}/rows").items()}
        out = {"run": RUN, "code": git_tag, "collected_at_git": git, "rows": {}}
        for name, row in sorted(rows.items()):
            comp, variant = name.split("__")
            vals = list(draws.get(f"{comp}__loeo__{variant}", {}).values())
            obs = row["test_envs"]["auroc"]
            null = {"draws": len(vals), "complete": len(vals) == DRAWS}
            if vals:
                null |= {"range": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))], "p95": float(np.quantile(vals, 0.95)),
                         "p": float((1 + sum(v >= obs for v in vals)) / (1 + len(vals)))}
            out["rows"][name] = {k: v for k, v in row.items() if k != "runs"} | {"null": null}
            print(name, round(obs, 4), row["test_envs"].get("ci95_env_resample"), "| all 10", round(row["all_envs"]["auroc"], 4), "| null", null.get("p95"), null["draws"])
        if "fixed_sim_vs_e1__planted" in rows:
            p = rows["fixed_sim_vs_e1__planted"]["test_envs"]["auroc"]
            out["planted_check"] = {"auroc": p, "minimum": cfg["planted_min"], "passes": p >= cfg["planted_min"]}
            print("planted check", out["planted_check"])
        stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
        with open(HERE / "results" / f"tfidf_onpolicy_{stamp}.json", "x") as fh:
            fh.write(json.dumps(out, indent=1))
        print("->", HERE / "results" / f"tfidf_onpolicy_{stamp}.json")
