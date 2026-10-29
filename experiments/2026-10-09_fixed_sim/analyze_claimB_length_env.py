"""Length, position and per-environment checks of probe step A (coordinator item B, defined 2026-10-10 10:48 PDT before
computing). Local, free: step A's saved out-of-fold per-output scores (no new fits) for fixed sim vs real e1 and the real
e2 vs real e1 control, level 1 (times masked) first, level 0 alongside. Run score = mean of the run's output scores.

  reproduce    run-level AUROC on the 5 test environments must match step A's summary (0.822 at level 1) within 0.002
  (a) length   strata = quintiles of the run's number of scored outputs, pooled over both arms (test environments); AUROC
               per stratum with runs per arm; stratified AUROC = share of (simulated or e2, real e1) run pairs within the
               same stratum ordered correctly (strata weighted by positive x negative runs). The same with the
               transcript's token count (level-0 text before the 32k cut, the `tokens` field of the v2 inputs)
  (b) count    Spearman correlation of run score with output count and with token count, per arm; output count and
               token count alone as run scores (a length-only baseline)
  (c) position output-level AUROC within thirds of the output's place among its run's scored outputs (by step, action),
               and within step bins 0-4, 5-9, 10-19, 20+
  per env      run-level AUROC in each of the 10 environments (A's outer folds hold out one environment each; dev/test
               marked), the 5 test environments pooled and all 10 pooled
Intervals: 95%, 1,000 main-task resamples. Nulls hold the probes fixed (selection not rerun): 1,000 draws, each flipping
one coin per main task that swaps the two runs' labels in every cell of that task (PLAN.md's unit; the first runs, in
results/partial/, flipped each cell on its own); p95 and p reported for the pooled test rows, the stratified rows, each
quintile, the position bins and each environment (none for the length-only scores or the all-10 pool). Writes
results/claimB_length_env_<stamp>.json (no text).

  PYTHONPATH=src uv run --no-project --with numpy --with scipy --with scikit-learn --with matplotlib python \
      experiments/2026-10-09_fixed_sim/analyze_claimB_length_env.py <fixed scores> <e2 scores> <local v2 run>
"""

import gzip
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from scipy.stats import rankdata, spearmanr

from probe_shift.scoring import auroc_ci

HERE = Path(__file__).parent
SPLIT = json.loads((HERE / "split.json").read_text())
TEST = set(SPLIT["test"])
SUMMARY = HERE / "results" / "claimB_probe_summary_2026-10-10_021700.json"
POS = {"fixed_vs_e1": "sim-attack-fixed", "e2_vs_e1": "real-attack-e2"}
ARM_IN = {"sim-attack": "sim-attack-fixed", "real-attack-e1": "real-attack-e1", "real-attack-e2": "real-attack-e2"}
DRAWS, BOOT, SEED, TOL = 1000, 1000, 0, 0.002
STEP_BINS = [(0, 4), (5, 9), (10, 19), (20, 10**6)]


def cell_of(key):
    return ":".join(key.rsplit("__", 2)[-2].split(":")[:2])


def strat_auc(Y, s, strata):
    """AUROC per column of labels Y [n, k] for fixed scores s, counting only pairs within a stratum (ties half)."""
    U, P = np.zeros(Y.shape[1]), np.zeros(Y.shape[1])
    for g in np.unique(strata):
        m = strata == g
        r, Yg = rankdata(s[m]), Y[m]
        n1 = Yg.sum(0)
        U += (r[:, None] * Yg).sum(0) - n1 * (n1 + 1) / 2
        P += n1 * (m.sum() - n1)
    return np.where(P > 0, U / np.where(P > 0, P, 1), np.nan)


def swaps(y, cells, tasks, rng):
    """[n, DRAWS] labels; each draw flips one coin per main task, and a flip swaps the labels of the two runs (one per arm)
    in every cell of that task (PLAN.md's per-main-task null; runs without a partner in their cell keep their labels)."""
    by = defaultdict(list)
    for i, c in enumerate(cells):
        by[c].append(i)
    pairs = np.array([g for g in by.values() if len(g) == 2 and y[g].sum() == 1])
    Y = np.repeat(y[:, None].astype(float), DRAWS, 1)
    t_of, t_idx = np.unique([tasks[g[0]] for g in pairs], return_inverse=True)
    flip = (rng.random((len(t_of), DRAWS)) < 0.5)[t_idx]
    a, b = Y[pairs[:, 0]].copy(), Y[pairs[:, 1]].copy()
    Y[pairs[:, 0]], Y[pairs[:, 1]] = np.where(flip, b, a), np.where(flip, a, b)
    return Y


def scored(y, s, groups, strata, Yn, rng):
    """Stratified AUROC, its main-task interval and its fixed-probe null."""
    obs = float(strat_auc(y[:, None].astype(float), s, strata)[0])
    tasks = np.unique(groups)
    rows = {t: np.flatnonzero(groups == t) for t in tasks}
    boot = []
    for _ in range(BOOT):
        i = np.concatenate([rows[t] for t in rng.choice(tasks, len(tasks))])
        boot.append(strat_auc(y[i, None].astype(float), s[i], strata[i])[0])
    null = strat_auc(Yn, s, strata)
    null = null[np.isfinite(null)]
    return {"auroc": round(obs, 4), "ci95": [round(float(x), 4) for x in np.nanpercentile(boot, [2.5, 97.5])],
            "null_p95": round(float(np.percentile(null, 95)), 4), "p": round(float((1 + (null >= obs).sum()) / (1 + len(null))), 4)}


def out_auc(y, s):
    return float(strat_auc(y[:, None].astype(float), s, np.zeros(len(y), int))[0])


def p_of(null, obs):
    null = null[np.isfinite(null)]
    return round(float((1 + (null >= obs).sum()) / (1 + len(null))), 4)


def runs_of(rows, pos, tokens):
    by, meta = defaultdict(list), {}
    for r in rows:
        by[(r["arm"], r["key"])].append((r["step"], r["action"], r["score"]))
        meta[(r["arm"], r["key"])] = (r["arm"] == pos, r["environment"], r["main_task"], cell_of(r["key"]))
    keys = sorted(by)
    R = {"key": keys, "y": np.array([meta[k][0] for k in keys]), "env": np.array([meta[k][1] for k in keys]),
         "task": np.array([meta[k][2] for k in keys]), "cell": np.array([meta[k][3] for k in keys]),
         "score": np.array([np.mean([x[2] for x in by[k]]) for k in keys]), "n_out": np.array([len(by[k]) for k in keys]),
         "tokens": np.array([tokens[k] for k in keys]), "outputs": [sorted(by[k]) for k in keys]}
    return R


def quintiles(x):
    edges = np.quantile(x, [0.2, 0.4, 0.6, 0.8])
    return np.searchsorted(edges, x, side="right")


def analyze(R, rng):
    out, t = {}, np.isin(R["env"], list(TEST))
    y, s, task, cell = R["y"][t], R["score"][t], R["task"][t], R["cell"][t]
    Yn = swaps(y, cell, task, rng)
    one = np.zeros(len(y), int)
    out["test_envs"] = auroc_ci(y, s, task, BOOT, SEED) | {"runs": {"positive": int(y.sum()), "negative": int((~y).sum())}}
    out["test_envs"]["null"] = {k: v for k, v in scored(y, s, task, one, Yn, rng).items() if k in ("null_p95", "p")}
    for name in ("n_out", "tokens"):  # (a) and (b)
        x = R[name][t]
        st = quintiles(x)
        out[f"strata_{name}"] = {"stratified": scored(y, s, task, st, Yn, rng), "edges": np.quantile(x, [0, .2, .4, .6, .8, 1]).tolist(),
                                 "by_stratum": [auroc_ci(y[st == g], s[st == g], task[st == g], BOOT, SEED) | {
                                     "positive": int(y[st == g].sum()), "negative": int((~y[st == g]).sum()),
                                     "null_p95": round(float(np.nanpercentile(strat_auc(Yn[st == g], s[st == g], one[st == g]), 95)), 4),
                                     "p": p_of(strat_auc(Yn[st == g], s[st == g], one[st == g]), out_auc(y[st == g], s[st == g]))}
                                    for g in range(5)]}
        out[f"spearman_score_vs_{name}"] = {arm: {"rho": round(float(spearmanr(s[m], x[m])[0]), 4), "runs": int(m.sum())}
                                            for arm, m in (("positive", y), ("negative", ~y))}
        out[f"{name}_alone"] = auroc_ci(y, x, task, BOOT, SEED) | {
            "median": {"positive": float(np.median(x[y])), "negative": float(np.median(x[~y]))}}
    ys, ss, gs, rel, step, run = [], [], [], [], [], []  # (c): one row per scored output
    for j in np.flatnonzero(t):
        o = R["outputs"][j]
        for i, (st_, _, sc) in enumerate(o):
            ys.append(R["y"][j]), ss.append(sc), gs.append(R["task"][j]), step.append(st_), run.append(j)
            rel.append(min(int(3 * i / len(o)), 2))
    ys, ss, gs, rel, step = map(np.array, (ys, ss, gs, rel, step))
    Yo = swaps(R["y"], R["cell"], R["task"], rng)[np.array(run)]  # the same per-task swaps, carried to each run's outputs
    bins = {f"third_{n}": rel == i for i, n in enumerate(("first", "middle", "last"))}
    bins |= {f"steps_{a}-{b if b < 10**6 else ''}": (step >= a) & (step <= b) for a, b in STEP_BINS}
    out["position"] = {}
    for n, m in bins.items():
        null = strat_auc(Yo[m], ss[m], np.zeros(m.sum(), int))
        out["position"][n] = auroc_ci(ys[m], ss[m], gs[m], BOOT, SEED) | {"positive": int(ys[m].sum()), "negative": int((~ys[m]).sum()),
                                                                         "null_p95": round(float(np.nanpercentile(null, 95)), 4),
                                                                         "p": p_of(null, out_auc(ys[m], ss[m]))}
    out["by_env"] = {}
    for e in sorted(set(R["env"])):  # every environment, dev ones included
        m = R["env"] == e
        null = strat_auc(swaps(R["y"][m], R["cell"][m], R["task"][m], rng), R["score"][m], np.zeros(m.sum(), int))
        out["by_env"][e] = auroc_ci(R["y"][m], R["score"][m], R["task"][m], BOOT, SEED) | {
            "split": "test" if e in TEST else "dev", "positive": int(R["y"][m].sum()), "negative": int((~R["y"][m]).sum()),
            "null_p95": round(float(np.percentile(null, 95)), 4), "p": p_of(null, out_auc(R["y"][m], R["score"][m]))}
    out["all_envs"] = auroc_ci(R["y"], R["score"], R["task"], BOOT, SEED)
    return out


def figure(res, path):
    """One question: is the masked 0.82 a run-length effect? Overall, matched within output-count quintiles, count alone."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [("All runs", lambda v: v["test_envs"]), ("Runs matched on\nnumber of outputs", lambda v: v["strata_n_out"]["stratified"]),
              ("Number of outputs\nalone (no probe)", lambda v: v["n_out_alone"])]
    series = [("Simulated vs real", "level1|fixed_vs_e1", "#2a78d6"), ("Real vs real", "level1|e2_vs_e1", "#eb6834")]
    fig, ax = plt.subplots(figsize=(7, 4.4), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    w = 0.36
    for k, (name, row, col) in enumerate(series):
        for i, (_, get) in enumerate(groups):
            x, v = i + (k - 0.5) * (w + 0.02), get(res["rows"][row])
            ax.bar(x, v["auroc"], w, color=col, label=name if i == 0 else None)
            ax.errorbar(x, v["auroc"], yerr=[[v["auroc"] - v["ci95"][0]], [v["ci95"][1] - v["auroc"]]], color="#0b0b0b", capsize=4, lw=1.2)
            ax.text(x, v["ci95"][1] + 0.02, f"{v['auroc']:.2f}", ha="center", fontsize=12, fontweight="bold", color="#0b0b0b")
    ax.axhline(0.5, color="#52514e", lw=1, ls="--")
    ax.text(2.62, 0.505, "chance", fontsize=9, color="#52514e", va="bottom", ha="right")
    ax.set_xticks(range(len(groups)), [g for g, _ in groups], fontsize=10, color="#0b0b0b")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Run-level AUROC\n(higher = the two batches are easier to tell apart)", fontsize=10, color="#0b0b0b")
    f = res["rows"]["level1|fixed_vs_e1"]
    ax.set_title(f"Matching runs on length leaves the probe at {f['strata_n_out']['stratified']['auroc']:.2f}; "
                 f"length alone scores {f['n_out_alone']['auroc']:.2f}", fontsize=11, color="#0b0b0b")
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(0.01, 0.01, "Times masked; 5 held-out test environments; 95% intervals over main tasks.", fontsize=8, color="#52514e")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())


def main(fixed, e2, run_dir):
    tokens = {}
    for f in sorted((Path(run_dir) / "inputs").glob("shard_*.jsonl")):
        for line in f.open():
            r = json.loads(line)
            tokens[(ARM_IN[r["arm"]], r["key"])] = r["tokens"]
    summ = json.loads(SUMMARY.read_text())["levels"]
    rng = np.random.default_rng(SEED)
    import hashlib
    import subprocess

    res = {"files": {"fixed_vs_e1": Path(fixed).name, "e2_vs_e1": Path(e2).name, "summary": SUMMARY.name}, "rows": {},
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "text_classifier": {"per_environment": "results/tfidf_per_env_2026-10-10_112443.json (run_tfidf_per_env_modal.py)"}}
    for comp, path in (("fixed_vs_e1", fixed), ("e2_vs_e1", e2)):
        for level, d in json.loads(gzip.decompress(Path(path).read_bytes()))["levels"].items():
            out = analyze(runs_of(d["rows"], POS[comp], tokens), rng)
            want = summ[level][comp]["best_layer_and_C"]["test_envs"]["test"]["auroc"]
            if abs(out["test_envs"]["auroc"] - want) > TOL:  # the saved scores must reproduce step A's headline
                raise ValueError(f"{level} {comp}: {out['test_envs']['auroc']} vs step A's {want}")
            out["test_envs"]["step_A_summary"] = want
            res["rows"][f"{level}|{comp}"] = out
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"claimB_length_env_{stamp}.json"
    with open(path, "x") as f:
        f.write(json.dumps(res, indent=1))
    figure(res, path.with_suffix(".png"))
    for k in sorted(res["rows"], key=lambda k: (k.split("|")[0] != "level1", k)):
        v = res["rows"][k]
        f = lambda x: f"{x['auroc']:.3f} [{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}]"
        print(f"\n{k}: test envs {f(v['test_envs'])} (A {v['test_envs']['step_A_summary']}), null p95 {v['test_envs']['null']['null_p95']}; "
              f"all 10 {f(v['all_envs'])}")
        for name in ("n_out", "tokens"):
            st = v[f"strata_{name}"]
            print(f"  {name}: stratified {f(st['stratified'])} null p95 {st['stratified']['null_p95']} | per quintile "
                  + " ".join(f"{b['auroc']:.3f}({b['positive']}/{b['negative']})" for b in st["by_stratum"])
                  + f" | alone {f(v[f'{name}_alone'])} medians {v[f'{name}_alone']['median']}"
                  + f" | spearman {v[f'spearman_score_vs_{name}']}")
        print("  position: " + " ".join(f"{n} {f(x)} p95 {x['null_p95']}" for n, x in v["position"].items()))
        print("  by env: " + " ".join(f"{e}[{x['split']}] {x['auroc']:.3f} p95 {x['null_p95']}" for e, x in v["by_env"].items()))
    print(path)


if __name__ == "__main__":
    main(*sys.argv[1:4])
