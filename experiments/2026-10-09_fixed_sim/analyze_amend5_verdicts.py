"""PLAN.md amendment 5, verdicts as revised after the runs (PR review of #383/#385; local, free, from saved outputs only).
Detected = run-level AUROC of the max >= 0.95 (Aswin's "about the planted amount") AND its 95% main-task interval's
unrounded lower end > 0.5 (the unplanted value) AND the AUROC above the 95th percentile of a fixed-choice null (1,000
draws swapping a pair's two labels with probability 1/2; the probes and their readout, layer and C choices are held fixed,
so this is not a selection-aware null). The runs' own verdicts used only the first two parts, with a rounded endpoint.
  oracle: per-output scores of the chosen candidates (analyze_plant_oracle.load and its result's choices), run max
  pilot:  its per-pair rows (planted run max = max of the line and its other tokens; twin run max)
Also, for the pilot: how many of each fold's 20,000 sampled negatives are tokens after the line in a planted copy (they
have read the line), by replaying its seeded draw from the run's meta.json (row counts only). Writes
results/amend5_verdicts_<stamp>.json.

  PYTHONPATH=src:experiments/2026-10-09_fixed_sim uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy \
      --with scikit-learn --with scipy python experiments/2026-10-09_fixed_sim/analyze_amend5_verdicts.py \
      <oracle scores dir> <oracle launch record> <oracle result .json> <pilot result .json> <pilot meta.json>
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
DRAWS, BOOT, SEED, BAR = 1000, 1000, 0, 0.95


def verdict(y, s, task, pair, rng):
    """y, s, task, pair: one entry per run; pair links a planted copy to its twin."""
    auc = roc_auc_score(y, s)
    tasks = np.unique(task)
    rows = {t: np.flatnonzero(task == t) for t in tasks}
    boot = []
    for _ in range(BOOT):  # main-task resamples, unrounded
        i = np.concatenate([rows[t] for t in rng.choice(tasks, len(tasks))])
        if len(set(y[i])) == 2:
            boot.append(roc_auc_score(y[i], s[i]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    groups = [np.flatnonzero(pair == p) for p in np.unique(pair)]
    null = []
    for _ in range(DRAWS):
        yy = y.copy()
        for g, f in zip(groups, rng.random(len(groups)) < 0.5):
            if f:
                yy[g] = yy[g[::-1]]
        null.append(roc_auc_score(yy, s))
    p95 = float(np.percentile(null, 95))
    return {"auroc": auc, "ci95": [float(lo), float(hi)], "null_p95": p95, "p": float((1 + (np.array(null) >= auc).sum()) / (1 + DRAWS)),
            "parts": {"auroc_at_least_0.95": bool(auc >= BAR), "ci_lower_above_0.5": bool(lo > 0.5), "above_null_p95": bool(auc > p95)},
            "detected": bool(auc >= BAR and lo > 0.5 and auc > p95)}


def main(oracle_dir, launches, oracle_result, pilot_result, pilot_meta):
    import analyze_plant_oracle as A
    from run_plant_token_pilot_modal import N_NEG, PLANTED, SEED as PILOT_SEED, TWIN, seed_of

    rng, out = np.random.default_rng(SEED), {"rule": __doc__.split("\n\n")[0], "oracle": {}, "pilot": {}}
    d, res = A.load(oracle_dir, launches), json.loads(Path(oracle_result).read_text())
    L = {o["layer"]: o for o in d["layers_out"]}
    base = L[d["layers"][0]]
    keys = sorted(base["runs"])
    pos = {k: n for n, k in enumerate(keys)}
    inv = np.array([pos[i] for i in base["run"]])
    y = np.array([base["runs"][i]["arm"].endswith("plant1") for i in keys])
    task = np.array([base["runs"][i]["main_task"] for i in keys])
    pair = np.array([base["runs"][i]["key"].removesuffix("__plant1") for i in keys])
    for variant, v in res["variants"].items():
        s = np.full(len(inv), np.nan)
        for e, (rd, layer, C) in v["choices"].items():
            t = L[layer]["results"][f"{rd}|{variant}|{C:g}"]["test"][e]
            s[t["rows"]] = t["score"]
        mx = np.full(len(keys), -np.inf)
        np.maximum.at(mx, inv, s)
        out["oracle"][variant] = verdict(y, mx, task, pair, rng)
        if abs(out["oracle"][variant]["auroc"] - v["run_max"]["auroc"]) > 1e-4:
            raise ValueError(f"oracle {variant}: does not reproduce the analysis's run-max AUROC")
    pr = json.loads(Path(pilot_result).read_text())
    for C, r in pr["by_C"].items():
        pp = r["per_pair"]
        s = np.r_[[max(q["line_max"], q["planted_other_max"]) for q in pp], [q["twin_max"] for q in pp]]
        yy = np.r_[np.ones(len(pp)), np.zeros(len(pp))].astype(bool)
        out["pilot"][C] = verdict(yy, s, np.array([q["main_task"] for q in pp] * 2), np.array([q["pair"] for q in pp] * 2), rng)
        margin = [q["line_max"] - max(q["planted_other_max"], q["twin_max"]) for q in pp]  # the line vs the best other token
        if not r["all_converged"]:  # the pilot's own rule: an unconverged fit gives no verdict
            out["pilot"][C]["detected"] = "incomplete: a fit failed the gradient check"
        out["pilot"][C] |= {"primary": r["primary"], "all_converged": r["all_converged"], "margin_min": float(min(margin)), "margin_median": float(np.median(margin)),
                            "tokens_per_run": [int(min(q[k] for q in pp for k in ("planted_tokens", "twin_tokens"))),
                                               int(max(q[k] for q in pp for k in ("planted_tokens", "twin_tokens")))]}
    meta = json.loads(Path(pilot_meta).read_text())
    env_of = {q["pair"]: q["env"] for q in next(iter(pr["by_C"].values()))["per_pair"]}
    neg = {}
    for e in sorted(set(env_of.values())):  # replay of the pilot's seeded draw (analyze() in run_plant_token_pilot_modal.py)
        train = [k for k in meta["pairs"] if env_of[k] != e]
        pool, after = [], []
        for k in train:
            for a in (TWIN, PLANTED):
                t = meta["transcripts"][f"{a}__{k}"]
                line = set(t["line_rows"])
                last = max(line) if line else None
                for i in range(t["rows"]):
                    if i not in line:
                        pool.append((a, k, i))
                        after.append(a == PLANTED and i > last)
        pick = np.random.default_rng(seed_of(f"{PILOT_SEED}:neg:{e}")).choice(len(pool), N_NEG, replace=False)
        neg[e] = {"pool": len(pool), "pool_after_line": int(sum(after)), "sampled_after_line": int(np.array(after)[pick].sum()), "sampled": N_NEG}
    out["pilot_negatives_after_line"] = neg
    for name, v in [(f"oracle {k}", x) for k, x in out["oracle"].items()] + [(f"pilot C {k}", x) for k, x in out["pilot"].items()]:
        print(f"{name}: AUROC {v['auroc']:.4f} [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}], null p95 {v['null_p95']:.4f}, p {v['p']:.4f}; "
              f"{v['parts']} -> detected {v['detected']}")
    print("pilot margins (min, median) and tokens per run:", {C: (round(v["margin_min"], 3), round(v["margin_median"], 3), v["tokens_per_run"])
                                                               for C, v in out["pilot"].items()})
    print("pilot negatives after the line, per fold:", {e: f"{x['sampled_after_line']}/{x['sampled']}" for e, x in neg.items()})
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(HERE / "results" / f"amend5_verdicts_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out | {"inputs": [str(oracle_dir), str(launches), str(oracle_result), str(pilot_result), str(pilot_meta)]}, indent=1))
    print("->", HERE / "results" / f"amend5_verdicts_{stamp}.json")


if __name__ == "__main__":
    main(*sys.argv[1:6])
