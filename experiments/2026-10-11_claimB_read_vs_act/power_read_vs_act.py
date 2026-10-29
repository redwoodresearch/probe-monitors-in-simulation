"""$0 power check for the reading-vs-acting comparison (plan only; numbers only, laptop).

G = D_read - D_act, where D = AUROC(sim vs real 1) - AUROC(real 2 vs real 1) on one side, level 1, test environments.
Reading side: follow-up (b), Qwen3.5-27B on the fixed off-policy data (per-run scores of its chosen probes,
`experiments/2026-10-10_claimB_qwen35_reader/data/grid_run_scores_2026-10-10_235051.json.gz`, manifest v2).
Acting side: follow-up (c), Qwen3.5-27B on its own September runs (per-output scores of the grid's chosen probes,
`experiments/2026-10-10_claimB_onpolicy/data/onpolicy_output_scores_2026-10-10_170651.json.gz`; run score = mean over
outputs, as the grid). Both files must reproduce the published AUROCs (gate) before anything else is computed.

SE(G) from 2,000 main-task draws (seed 20261011): within a side, sim-vs-1 and 2-vs-1 share the draw (shared coins);
across sides, two modes: independent draws, and shared draws from the union of main tasks (the acting side's test tasks
also occur on the reading side). Also a composition-matched variant: the reading side restricted to the acting side's
main tasks. Clear-answer chances use a normal approximation with each SE, for stated true G values and the margin X;
they are scenario probabilities (frozen probes; the equalised reading side's SE assumed equal to today's).

  uv run --no-project --python 3.12 --with numpy --with scikit-learn \\
      python experiments/2026-10-11_claimB_read_vs_act/power_read_vs_act.py [DATA_ROOT]
"""

import gzip
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
DATA_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT  # the checkout holding the git-ignored data/ files
READ = DATA_ROOT / "experiments/2026-10-10_claimB_qwen35_reader/data/grid_run_scores_2026-10-10_235051.json.gz"
ACT = DATA_ROOT / "experiments/2026-10-10_claimB_onpolicy/data/onpolicy_output_scores_2026-10-10_170651.json.gz"
SPLIT = ROOT / "experiments/2026-10-09_fixed_sim/split.json"
PUBLISHED = {("read", "fixed_vs_e1"): 0.6718, ("read", "e2_vs_e1"): 0.498, ("act", "fixed_vs_e1"): 0.561, ("act", "e2_vs_e1"): 0.465}
TOL = 0.002
N_DRAWS, SEED, MARGIN = 2000, 20261011, 0.06
TRUE_G = [0.0, 0.039, 0.078, 0.12]


def read_side(test_envs):
    d = json.loads(gzip.open(READ).read())
    out = {}
    for comp in ("fixed_vs_e1", "e2_vs_e1"):
        rows = [r for r in d["results"] if r["grid"] == "b" and r["level"] == "level1" and r["comparison"] == comp
                and r["selector"] == "by_readout.output_end.best"]
        if len(rows) != 1:
            raise ValueError(f"reading side {comp}: {len(rows)} rows")
        out[comp] = [(r["main_task"], r["label"], r["score"]) for r in rows[0]["runs"] if r["environment"] in test_envs]
    return out


def act_side(test_envs):
    d = json.loads(gzip.open(ACT).read())
    out = {}
    for comp in ("fixed_vs_e1", "e2_vs_e1"):
        runs = defaultdict(list)
        meta = {}
        for r in d["scores"][f"level1|{comp}"]["rows"]:
            if r["environment"] in test_envs:
                runs[r["key"]].append(r["score"])
                meta[r["key"]] = (r["main_task"], 0 if r["arm"] == "real-attack-e1" else 1)
        out[comp] = [(meta[k][0], meta[k][1], float(np.mean(v))) for k, v in runs.items()]
    return out


def auroc(rows):
    y = np.array([r[1] for r in rows])
    s = np.array([r[2] for r in rows])
    if y.min() == y.max():
        raise ValueError("one class only")
    return float(roc_auc_score(y, s))


def by_task(rows):
    t = defaultdict(list)
    for r in rows:
        t[r[0]].append(r)
    return t


def d_of(side, a, b, pick):
    ra = [r for t in pick for r in a.get(t, [])]
    rb = [r for t in pick for r in b.get(t, [])]
    return auroc(ra) - auroc(rb)


def draws(sides, rng, mode):
    """G per draw. 'independent': each side resamples its own main tasks; 'shared': one draw of main tasks from the
    union applies to both sides (the acting side's tasks are a subset of the reading side's), so shared task effects
    are kept. Within a side, sim-vs-1 and 2-vs-1 always share the draw. A draw is kept only if every AUROC is defined."""
    idx = {s: (by_task(sides[s]["fixed_vs_e1"]), by_task(sides[s]["e2_vs_e1"])) for s in sides}
    own = {s: sorted(set(idx[s][0]) | set(idx[s][1])) for s in sides}
    union = sorted(set(own["read"]) | set(own["act"]))
    g, dr, da = [], [], []
    for _ in range(N_DRAWS):
        if mode == "shared":
            pick = rng.choice(union, size=len(union), replace=True)
            picks = {"read": pick, "act": pick}
        else:
            picks = {s: rng.choice(own[s], size=len(own[s]), replace=True) for s in sides}
        try:
            x = d_of(sides["read"], *idx["read"], picks["read"])
            y = d_of(sides["act"], *idx["act"], picks["act"])
        except ValueError:
            continue
        g.append(x - y)
        dr.append(x)
        da.append(y)
    if len(g) < 0.99 * N_DRAWS:
        raise ValueError(f"{mode}: too many one-class draws ({N_DRAWS - len(g)})")
    return np.array(g), np.array(dr), np.array(da), {s: len(own[s]) for s in sides}, len(union)


def clear(g, se):
    """Normal-approximation chances for the estimate ghat ~ N(g, se): beyond 0 (lower bound > 0: ghat > h); within X
    (both bounds inside +-X: -X + h < ghat < X - h); and either (the union of the two sets, no double counting)."""
    from math import erf, sqrt

    cdf = lambda z: 0.5 * (1 + erf(z / sqrt(2)))  # noqa: E731
    h = 1.96 * se
    prob = lambda lo, hi: max(0.0, cdf((hi - g) / se) - cdf((lo - g) / se))  # noqa: E731
    beyond = prob(h, float("inf"))
    within = prob(-MARGIN + h, MARGIN - h) if h < MARGIN else 0.0
    union = prob(min(h, -MARGIN + h), float("inf")) if (h < MARGIN and MARGIN - h >= h) else beyond + within
    return {"beyond_0": beyond, "within_X": within, "either": union}


def main():
    split = json.loads(SPLIT.read_text())
    test_envs = set(split["test"])
    sides = {"read": read_side(test_envs), "act": act_side(test_envs)}
    gate = {}
    for (side, comp), pub in PUBLISHED.items():
        got = auroc(sides[side][comp])
        gate[f"{side}|{comp}"] = {"published": pub, "recomputed": got, "runs": len(sides[side][comp])}
        if abs(got - pub) > TOL:
            raise SystemExit(f"gate failed: {side} {comp} {got:.4f} vs published {pub}")
    rng = np.random.default_rng(SEED)
    obs = {s: auroc(sides[s]["fixed_vs_e1"]) - auroc(sides[s]["e2_vs_e1"]) for s in sides}
    acting_tasks = {r[0] for c in sides["act"].values() for r in c}
    matched = {"act": sides["act"], "read": {c: [r for r in rows if r[0] in acting_tasks] for c, rows in sides["read"].items()}}
    obs_m = {s: auroc(matched[s]["fixed_vs_e1"]) - auroc(matched[s]["e2_vs_e1"]) for s in matched}
    modes = {}
    for name, data in (("all_reading_tasks", sides), ("reading_restricted_to_acting_tasks", matched)):
        for mode in ("independent", "shared"):
            g, dr, da, n_tasks, n_union = draws(data, rng, mode)
            se = float(g.std(ddof=1))
            modes[f"{name}|{mode}"] = {
                "draws_kept": int(len(g)), "main_tasks": n_tasks, "union_tasks": n_union,
                "se": {"D_read": float(dr.std(ddof=1)), "D_act": float(da.std(ddof=1)), "G": se},
                "g_interval_today": [float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))],
                "half_width_95": 1.96 * se, "power": {str(x): clear(x, se) for x in TRUE_G}}
    out = {
        "label": "$0 power check for reading vs acting (plan only; normal approximation with bootstrap SEs)",
        "assumptions": ["scenario probabilities, not validated power: the selected probes are frozen (no refit or "
                        "selection uncertainty), and the equalised reading side's SE is assumed equal to today's",
                        "G compares two cohorts (different simulators, agents, run counts and task weights); subtracting "
                        "each side's real-vs-real row does not isolate reading vs acting",
                        "'shared' draws keep the task effects common to both sides; 'independent' draws ignore them"],
        "gate": gate, "test_environments": sorted(test_envs),
        "acting_tasks_in_reading": f"{len(acting_tasks & {r[0] for c in sides['read'].values() for r in c})} of {len(acting_tasks)}",
        "observed": {"all_reading_tasks": {"D_read": obs["read"], "D_act": obs["act"], "G": obs["read"] - obs["act"]},
                     "reading_restricted_to_acting_tasks": {"D_read": obs_m["read"], "D_act": obs_m["act"], "G": obs_m["read"] - obs_m["act"]}},
        "margin_X": MARGIN, "modes": modes, "draws": N_DRAWS, "seed": SEED,
        "inputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (READ, ACT, SPLIT)},
        "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    dst = HERE / "results" / f"power_read_vs_act_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("gate", "acting_tasks_in_reading", "observed", "modes")}, indent=1))
    print("->", dst)


if __name__ == "__main__":
    main()
