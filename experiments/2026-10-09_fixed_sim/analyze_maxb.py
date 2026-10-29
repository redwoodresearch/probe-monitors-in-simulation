"""Max-over-tokens version (b) (PLAN.md amendment 7), the analysis (local; scores only). Inputs, fetched from the Volume one
exact file at a time: `claimB/level1/maxb_<tag>/` runs.json, scan_merged.npz (per comparison, run and probe column:
run max, top-5 mean, mean of per-output maxima; run_maxb_read_modal.py --mode merge), bank/choices.json, scan_diag.json.
Per comparison (labels and null columns as in training, maxb_core.labels), on the 5 test environments and all 10:
  - run-level AUROC of each run score; the headline is the run max;
  - the selection-aware null: each shuffle column scored by its own chosen probes against its own labels;
  - main-task 95% interval; per-environment AUROC (the real column);
  - the length baseline d9 asked for: the run's mean log token count per output as the score (a max grows with n);
  - planted checks: "detected" by Aswin's rule as revised in amendment 5 (maxb_core.verdict; no verdict if a fold's real
    column has no converged choice); for check 1, how often the run max of a planted copy falls on the line's tokens.
Writes results/maxb_<stamp>.json.

  PYTHONPATH=src:experiments/2026-10-09_fixed_sim uv run --no-project --python 3.12 --with numpy --with scipy --with scikit-learn \\
      python experiments/2026-10-09_fixed_sim/analyze_maxb.py <local copy of claimB/level1/maxb_<tag>/>
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.metrics import roc_auc_score

import maxb_core as K

HERE = Path(__file__).parent
TEST = set(json.loads((HERE / "split.json").read_text())["test"])
STATS = ["run_max", "top5_mean", "output_max_mean"]


def analyze(runs, merged, choices, diag, rng):
    out = {}
    env = np.array([m["environment"] for m in runs])
    task = np.array([m["main_task"] for m in runs])
    length = np.array([np.mean([np.log(b - a) for _, _, a, b in m["outputs"]]) if m["outputs"] else np.nan for m in runs])
    for comp in K.COMPS:
        use, Y = K.labels(runs, comp)
        S = merged[comp]  # [n_runs, 3, 1 + N_NULL]
        converged = all(c is not None for c in [row[0] for row in choices[comp]["layer"]])
        missing = int((use & ~np.isfinite(S[:, 0, 0])).sum())
        res = {"runs": {"positive": int(Y[use, 0].sum()), "negative": int((1 - Y[use, 0]).sum())},
               "missing_scores": missing, "real_column_converged_every_fold": converged}
        for scope, m in (("test_envs", use & np.isin(env, list(TEST))), ("all_envs", use)):
            m &= np.isfinite(S[:, 0, 0])
            r = {}
            for j, stat in enumerate(STATS):
                null = [roc_auc_score(Y[m, c], S[m, j, c]) for c in range(1, Y.shape[1]) if np.isfinite(S[m, j, c]).all()]
                bar = K.BARS.get(comp) if stat == "run_max" else None
                # a verdict needs every run of the comparison scored and at most 5 of the 100 shuffles lost (B's rule)
                complete = converged and missing == 0 and len(null) >= K.N_NULL - 5
                r[stat] = K.verdict(Y[m, 0].astype(bool), S[m, j, 0], task[m], null, bar, complete, rng)
            r["length_baseline"] = round(float(roc_auc_score(Y[m, 0], length[m])), 4)
            r["per_environment"] = {e: round(float(roc_auc_score(Y[m & (env == e), 0], S[m & (env == e), 0, 0])), 4)
                                    for e in sorted(set(env[m])) if len(set(Y[m & (env == e), 0])) == 2}
            res[scope] = r
        if comp == "plant1":  # where the planted copies' run max sits (the real column's per-output maxima and tokens)
            hits = []
            for m in runs:
                d = diag.get(comp, {}).get(m["key"])
                if m["arm"].endswith("plant1") and d and m["line_tokens"]:
                    j = int(np.argmax(d["out_max"]))
                    hits.append(d["out_arg_token"][j] in set(m["line_tokens"]))
            res["planted_run_max_on_line"] = {"copies": len(hits), "on_line": int(sum(hits))}
        out[comp] = res
    return out


def main(folder):
    f = Path(folder)
    runs = json.loads((f / "runs.json").read_text())
    merged = dict(np.load(f / "scan_merged.npz"))
    res = analyze(runs, merged, json.loads((f / "bank" / "choices.json").read_text()), json.loads((f / "scan_diag.json").read_text()), np.random.default_rng(K.SEED))
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(HERE / "results" / f"maxb_{stamp}.json", "x") as fh:
        fh.write(json.dumps({"source": str(f), "results": res}, indent=1))
    for comp, r in res.items():
        t = r["test_envs"]["run_max"]
        print(f"{comp}: run max {t['auroc']:.4f} [{t['ci95'][0]:.4f}, {t['ci95'][1]:.4f}], null p95 {t['null_p95']:.4f} "
              f"({t['null_draws']} draws), detected {t['detected']}; length alone {r['test_envs']['length_baseline']}")
    print("->", HERE / "results" / f"maxb_{stamp}.json")


if __name__ == "__main__":
    main(sys.argv[1])
