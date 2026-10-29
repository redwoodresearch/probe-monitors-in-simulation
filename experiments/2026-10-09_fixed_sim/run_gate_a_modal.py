"""Gate (a): TF-IDF on fixed-sim (PLAN.md "Gate (a)"; config/gate_a.yaml), Modal CPU, on the fixed run's staged examples
(`<run>/examples/` on the la-activations Volume). Uses experiments/2026-10-06_q1_text_baselines/analyze_tfidf.py
unchanged (load, scrub, featurize, fit, run_means, name_columns) with q1's TF-IDF settings.

Every row is run-level AUROC (a run's score = the mean of its kept outputs' out-of-fold scores) on the 5 test
environments, with a 95% interval that resamples whole test environments:
  comparisons  fixed sim vs real e1 (primary); real e2 vs real e1 (the required control row)
  designs      loeo: leave one environment out over all 10, scored on the 5 test environments (pre-registered primary);
               dev_to_test: trained on the 5 dev environments; main_task: q1's main-task folds over all 10 (secondary)
  variants     all (primary); no_view_range (outputs with the invalid view_range error dropped from both arms; the
               secondary); level1 (times masked, diagnostic); planted (`INFO: cache warm` as an extra last line of one
               seeded kept output of every simulated run; must reach planted_min)
Nulls (config `shuffle_null.rows`, 200 draws each, for loeo and dev_to_test): per main task, one flip with probability 1/2
swaps the arms of all its cells; same folds; refit. Fanned out over containers, `draws_per_container` draws each.
Every decision (bar, planted check) uses full-precision AUROCs; only the printout rounds. The all-10-environment loeo
row (variant all) is supplementary; the primary scores the 5 test environments. Rendering/content labels for the top
features are added by hand in the write-up.
Top features: mean coefficient over the 5 test-environment loeo models (fixed sim vs e1, all), top N each way, with
their n-grams, outputs containing them per arm, and date/time tokens flagged.

Two launches, so the nulls are sized from measured fit times: `--part main` (one container: every row but the nulls,
with timings), then `--part null --main-file <its result>` (the null containers, then every bar verdict, combined
into gate_a_<stamp>.json). The environment variable GATE_A_NULL_TIMEOUT caps each null container (seconds).

  cd <worktree> && uv run modal run experiments/2026-10-09_fixed_sim/run_gate_a_modal.py --part main
  cd <worktree> && uv run modal run experiments/2026-10-09_fixed_sim/run_gate_a_modal.py --part null --main-file gate_a_main_<stamp>.json
"""

import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal
import yaml

HERE = Path(__file__).parent
REPO = HERE.parents[1] if modal.is_local() else Path("/root")
Q1 = REPO / "experiments" / "2026-10-06_q1_text_baselines"
NAME = "probe-shift_aswin-surya_sub-fixed-sim_kill-2026-11-09"

app = modal.App(NAME)
acts = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations")
image = (modal.Image.debian_slim(python_version="3.13")
         .uv_pip_install("numpy", "scipy", "scikit-learn==1.9.1", "joblib==1.6.0", "pyyaml")
         .env({"JOBLIB_TEMP_FOLDER": "/tmp"})  # joblib's shared-memory default (/dev/shm) is small in containers
         .add_local_python_source("probe_shift")
         .add_local_file(Q1 / "analyze_tfidf.py", "/root/analyze_tfidf.py"))


def prepare(cfg: dict, variant: str, split: dict, folds: dict):
    """The kept outputs of the three arms as the variant's texts, featurized: (d, X, env, texts, counts)."""
    import numpy as np

    import analyze_tfidf as A

    spec = cfg["variants"][variant]
    d = A.load(Path("/acts") / cfg["run"] / "examples", folds["environment_folds"]["environment_to_fold"], fold_key="environment")
    rows = d["rows"]
    counts = {"kept_outputs": dict(d["counts"]["kept_outputs"])}
    keep = np.ones(len(rows), bool)
    if spec.get("drop_outputs_matching"):
        keep = np.array([spec["drop_outputs_matching"] not in r[cfg["text"]] for r in rows])
        counts["dropped_outputs"] = {a: int((~keep & (d["arm"] == a)).sum()) for a in sorted(set(d["arm"]))}
    texts = [A.scrub(r[cfg["text"]], spec["scrub"]) for r in rows]
    if spec.get("plant"):
        sim = [i for i, r in enumerate(rows) if r["arm"] == "sim-attack"]
        by_run = {}
        for i in sim:
            by_run.setdefault(rows[i]["key"], []).append(i)
        for key, idx in sorted(by_run.items()):
            i = random.Random(f"{cfg['plant_seed']}:{key}").choice(idx)
            texts[i] = texts[i] + "\n" + spec["plant"]
        counts["planted_runs"] = len(by_run)
    for k in ("arm", "key", "cell", "main_task"):
        d[k] = d[k][keep]
    rows = [r for r, k in zip(rows, keep) if k]
    texts = [t for t, k in zip(texts, keep) if k]
    env = np.array([r["environment"] for r in rows])
    task_fold = folds["task_folds"]["main_task_to_fold"]
    d["task_fold"] = np.array([task_fold[r["main_task"]] for r in rows])
    assert set(env) == set(split["dev"]) | set(split["test"]), "environments differ from the split"
    return d, A.featurize(texts, cfg), env, texts, counts


def design_jobs(design: str, R, env, d, split: dict, all_envs: bool = False):
    """[(train rows, scored rows)] for a design, within comparison rows R. loeo holds out each test environment (and,
    with all_envs, each dev environment too, for the all-10 supplementary row)."""
    import numpy as np

    test, dev = np.isin(env, split["test"]), np.isin(env, split["dev"])
    if design == "loeo":
        return [(R & (env != e), R & (env == e)) for e in split["test"] + (split["dev"] if all_envs else [])]
    if design == "dev_to_test":
        return [(R & dev, R & test)]
    return [(R & (d["task_fold"] != f), R & (d["task_fold"] == f)) for f in sorted(set(d["task_fold"]))]


def score_jobs(X, y, jobs, cfg):
    import numpy as np
    from joblib import Parallel, delayed

    import analyze_tfidf as A

    parts = Parallel(n_jobs=min(len(jobs), cfg["n_jobs"]))(delayed(A._fold_scores)(X, y, tr, te, cfg, cfg["C"]) for tr, te in jobs)
    s = np.full(len(y), np.nan)
    for (_, te), p in zip(jobs, parts):
        s[te] = p
    return s


def run_auroc(score, y, d, env, rows, split_envs, cfg, boot=True):
    """Run-level AUROC over `rows`, and a 95% interval resampling whole environments of `split_envs`."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    import analyze_tfidf as A

    assert np.isfinite(score[rows]).all(), "a scored row has no finite score"
    run_score, first = A.run_means(score, d["key"], rows)
    run_y, run_env = y[first], env[first]
    out = {"auroc": float(roc_auc_score(run_y, run_score)),  # full precision; decisions use it, display rounds
           "runs": {"negative": int((run_y == 0).sum()), "positive": int((run_y == 1).sum())}}
    if boot:
        rng, vals = np.random.default_rng(cfg["seed"]), []
        envs = sorted(split_envs)
        for _ in range(cfg["env_bootstrap"]):
            pick = rng.choice(envs, len(envs))
            idx = np.concatenate([np.flatnonzero(run_env == e) for e in pick])
            if len(set(run_y[idx])) == 2:
                vals.append(roc_auc_score(run_y[idx], run_score[idx]))
        out["ci95_env_resample"] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
    return out


@app.function(image=image, volumes={"/acts": acts}, cpu=16, memory=65536, timeout=5400, retries=0)
def main_job(cfg: dict, split: dict, folds: dict) -> dict:
    import numpy as np

    import analyze_tfidf as A

    res = {"rows": {}, "counts": {}, "timing": {}}
    for variant in cfg["variants"]:
        t0 = time.time()
        d, X, env, texts, counts = prepare(cfg, variant, split, folds)
        res["timing"][f"prepare|{variant}"] = round(time.time() - t0, 1)
        res["counts"][variant] = counts
        print(variant, "prepared in", res["timing"][f"prepare|{variant}"], "s; X", X.shape, "nnz", X.nnz, flush=True)
        test = np.isin(env, split["test"])
        for comp, (neg, pos) in cfg["comparisons"].items():
            if variant == "planted" and pos != "sim-attack":
                continue  # the plant is in simulated runs only
            R = np.isin(d["arm"], [neg, pos])
            y = (d["arm"] == pos).astype(int)
            designs = cfg["designs"] if variant == "all" else ["loeo", "dev_to_test"]
            for design in designs:
                all10 = design == "main_task" or (design == "loeo" and variant == "all")
                jobs = design_jobs(design, R, env, d, split, all_envs=all10)
                t0 = time.time()
                s = score_jobs(X, y, jobs, cfg)
                res["timing"][f"{comp}|{design}|{variant}"] = {"seconds": round(time.time() - t0, 1), "fits": len(jobs),
                                                                "parallel": min(len(jobs), cfg["n_jobs"])}
                row = {"test_envs": run_auroc(s, y, d, env, R & test, split["test"], cfg)}
                if all10:
                    row["all_envs"] = run_auroc(s, y, d, env, R, split["dev"] + split["test"], cfg)
                res["rows"][f"{comp}|{design}|{variant}"] = row
                print(comp, design, variant, json.dumps(row), flush=True)
            if comp == "fixed_sim_vs_e1" and variant == "all":  # top features from the 5 test-environment loeo models
                coefs = np.mean([A.fit(X[tr], y[tr], cfg, cfg["C"])[1].coef_[0] for tr, _ in design_jobs("loeo", R, env, d, split)], axis=0)
                order = np.argsort(coefs)
                top = {"toward_sim": order[::-1][:cfg["top_features"]].tolist(), "toward_real": order[:cfg["top_features"]].tolist()}
                cols = sorted({c for v in top.values() for c in v})
                names = A.name_columns(X, texts, cols, cfg)
                Xc = X[:, cols].tocsc()
                per_arm = {c: {a: int((Xc[:, j].toarray().ravel()[d["arm"] == a] > 0).sum()) for a in (neg, pos)} for j, c in enumerate(cols)}
                date_rx = re.compile(cfg["date_token_pattern"])
                res["top_features"] = {way: [{"column": int(c), "kind": "word" if c < cfg["n_features"] else "char",
                                              "weight": round(float(coefs[c]), 4), "ngrams": names[int(c)],
                                              "outputs_with_it": per_arm[c],
                                              "date_or_time_token": any(date_rx.search(g) for g in names[int(c)])}
                                             for c in cs] for way, cs in top.items()}
        del X
    return res


# each null container's cap in seconds, set from the main launch's fit times (env var read where the app is defined)
NULL_TIMEOUT = int(os.environ.get("GATE_A_NULL_TIMEOUT", "2400"))


@app.function(image=image, volumes={"/acts": acts}, cpu=8, memory=65536, timeout=NULL_TIMEOUT, retries=0)
def null_job(cfg: dict, split: dict, folds: dict, comp: str, design: str, variant: str, draws: list[int]) -> list[float]:
    """Run-level AUROC on the test environments for each shuffled-label draw in `draws` (seeds seed + 1 + k, as q1)."""
    import numpy as np

    d, X, env, _, _ = prepare(cfg, variant, split, folds)
    neg, pos = cfg["comparisons"][comp]
    R = np.isin(d["arm"], [neg, pos])
    y = (d["arm"] == pos).astype(int)
    test = np.isin(env, split["test"])
    from joblib import Parallel, delayed

    import analyze_tfidf as A

    tasks_u, task_idx = np.unique(d["main_task"], return_inverse=True)
    perms = {k: np.where((np.random.default_rng(cfg["seed"] + 1 + k).random(len(tasks_u)) < 0.5)[task_idx], 1 - y, y) for k in draws}
    jobs = design_jobs(design, R, env, d, split)
    # every (draw, fold) fit in one pool, so all cores stay busy
    work = [(k, tr, te) for k in draws for tr, te in jobs]
    parts = Parallel(n_jobs=cfg["n_jobs"])(delayed(A._fold_scores)(X, perms[k], tr, te, cfg, cfg["C"]) for k, tr, te in work)
    scores = {k: np.full(len(y), np.nan) for k in draws}
    for (k, _, te), p in zip(work, parts):
        scores[k][te] = p
    out = {}
    for k in draws:
        out[k] = run_auroc(scores[k], perms[k], d, env, R & test, split["test"], cfg, boot=False)["auroc"]
        print(comp, design, variant, k, out[k], flush=True)
    # each container writes its own file, so draws survive a lost client
    path = Path("/acts") / cfg["run"] / "gate_a_null" / f"{comp}__{design}__{variant}" / f"draws_{min(draws)}-{max(draws)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x") as f:
        f.write(json.dumps({str(k): v for k, v in out.items()}))
    acts.commit()
    return [out[k] for k in draws]


def stored_draws(cfg: dict) -> dict:
    """Null draws already on the Volume, by row: {"comp|design|variant": {k: auroc}}."""
    out = {}
    try:
        rows = acts.listdir(f"{cfg['run']}/gate_a_null")
    except (FileNotFoundError, modal.exception.NotFoundError):
        return out
    for row in rows:
        name = row.path.rsplit("/", 1)[-1].replace("__", "|")
        for f in acts.listdir(row.path):
            for k, v in json.loads(b"".join(acts.read_file(f.path))).items():
                assert int(k) not in out.get(name, {}), f"{name}: draw {k} is in two files"  # draws are seeded by index
                out.setdefault(name, {})[int(k)] = v
    return out


@app.local_entrypoint()
def main(config: str = "gate_a.yaml", part: str = "main", main_file: str = "", done_file: str = ""):
    import numpy as np

    cfg = yaml.safe_load((HERE / "config" / config).read_text())
    split = json.loads((HERE / cfg["split_file"]).read_text())
    folds = json.loads((REPO / "splits" / cfg["folds_file"]).read_text())
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    if part == "main":
        res = main_job.remote(cfg, split, folds)
        path = HERE / "results" / f"gate_a_main_{stamp}.json"
        with open(path, "x") as f:
            f.write(json.dumps({"config": cfg, "git_head": head, "split": split, **res}, indent=1))
        print(json.dumps(res["timing"], indent=1))
        print(path)
        return
    assert part == "null" and main_file, "--part null needs --main-file"
    res = json.loads((HERE / "results" / main_file).read_text())
    n, per = cfg["shuffle_null"]["draws"], cfg["shuffle_null"]["draws_per_container"]
    have = stored_draws(cfg)  # earlier launches' draws: on the Volume, and a captured log (results/partial/) if given
    if done_file:
        for row, d in json.loads((HERE / "results" / done_file).read_text())["draws"].items():
            for k, v in d.items():
                assert have.setdefault(row, {}).get(int(k), v) == v, f"{row} draw {k} disagrees between sources"
                have[row][int(k)] = v
    missing = {"|".join(row): [k for k in range(n) if k not in have.get("|".join(row), {})] for row in cfg["shuffle_null"]["rows"]}
    nulls = [(row, ks[a:a + per]) for row, ks in missing.items() for a in range(0, len(ks), per)]
    print(f"draws already done: { {r: len(have.get(r, {})) for r in missing} }; {len(nulls)} null containers, timeout {NULL_TIMEOUT} s each", flush=True)
    null_calls = [null_job.spawn(cfg, split, folds, *row.split("|"), draws) for row, draws in nulls]
    failed = []
    for (row, draws), c in zip(nulls, null_calls):
        try:
            for k, v in zip(draws, c.get()):
                have.setdefault(row, {})[k] = v
        except Exception as e:  # a container that failed or timed out: its draws stay missing, reported below
            failed.append({"row": row, "draws": draws, "error": repr(e)[:300]})
    print(f"failed containers: {len(failed)}", flush=True)
    null = {row: [have[row][k] for k in range(n) if k in have.get(row, {})] for row in missing}
    res["null_draw_indices"] = {row: sorted(have.get(row, {})) for row in missing}  # one value per draw index (dict keys)
    assert all(len(set(v)) == len(v) for v in res["null_draw_indices"].values())
    lo, hi = cfg["bar"]["null_quantiles"]
    res["null_failed_containers"] = failed
    for row, vals in null.items():
        if len(vals) != n:  # incomplete: no verdict
            res["rows"][row]["null"] = {"draws": len(vals), "expected": n, "complete": False}
            res["rows"][row].pop("bar", None)
            continue
        observed = res["rows"][row]["test_envs"]["auroc"]
        q = [float(np.quantile(vals, lo)), float(np.quantile(vals, hi))]
        res["rows"][row]["null"] = {"draws": n, "range": q, "mean": round(float(np.mean(vals)), 4), "values": vals}
        res["rows"][row]["bar"] = {"auroc_at_most": cfg["bar"]["auroc_max"], "inside_null_range": q[0] <= observed <= q[1],
                                   "passes": observed <= cfg["bar"]["auroc_max"] and q[0] <= observed <= q[1]}
    planted = res["rows"]["fixed_sim_vs_e1|loeo|planted"]["test_envs"]["auroc"]
    res["planted_check"] = {"auroc": planted, "minimum": cfg["planted_min"], "passes": planted >= cfg["planted_min"]}
    out = res | {"main_file": main_file, "null_git_head": head}
    path = HERE / "results" / f"gate_a_{stamp}.json"
    with open(path, "x") as f:
        f.write(json.dumps(out, indent=1))
    r4 = lambda x: round(x, 4) if isinstance(x, float) else [r4(v) for v in x] if isinstance(x, list) else {k: r4(v) for k, v in x.items()} if isinstance(x, dict) else x
    print(json.dumps(r4({k: {kk: vv for kk, vv in v.items() if kk != "null"} | ({"null_range": v["null"].get("range"), "null_draws": v["null"]["draws"]} if "null" in v else {})
                         for k, v in res["rows"].items()}), indent=1))
    print(path)
