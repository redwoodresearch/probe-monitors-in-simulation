"""Does the Needham eval-awareness probe score simulated LinuxArena attack runs differently from real ones? And does it
agree with the templated probe? Reads run_score_la.py's table (and the templated probe's stored call_end scores).

Populations: the 939 paired attack cells of splits/exclusions_v3.json (real attack epoch 1 "e1", epoch 2 "e2",
simulated "sim"), plus real honest runs on the same main tasks. A run's score is the mean of its call_end scores.
Per probe column (readout x layer): run means per arm; shifts sim - e1 (the question), e2 - e1 (the real-vs-real
yardstick) and honest - e1 (reference), with 95% intervals resampling main tasks (1,000 draws); run AUROCs sim vs e1,
e2 vs e1, e1 vs honest; shifts in units of the e1 run-score sd. The templated probe's stored scores (layers 16-48)
go through the same code, and run scores of the two probes are correlated per layer.

  uv run --with pandas --with pyarrow python experiments/2026-10-06_q3_eval_awareness_needham/analyze_la.py --scores data/scores_needham_....parquet \
      --templated data/scores_eval_awareness_segment_end_2026-10-06_145310.parquet --fit results/fit_....json
"""

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
E1, E2, SIM, HON = "real-attack-e1", "real-attack-e2", "sim-attack", "real-honest"


class Boot:
    """Main-task bootstrap: B draws of T tasks with replacement, as a [B, T] count matrix shared by every statistic."""

    def __init__(self, tasks, n, seed):
        self.tid = {t: i for i, t in enumerate(sorted(tasks))}
        draws = np.random.default_rng(seed).integers(0, len(self.tid), size=(n, len(self.tid)))
        self.W = np.stack([np.bincount(d, minlength=len(self.tid)) for d in draws]).astype(float)

    def mean(self, tasks, x):
        t = np.array([self.tid[k] for k in tasks])
        s, c = np.bincount(t, weights=x, minlength=len(self.tid)), np.bincount(t, minlength=len(self.tid)).astype(float)
        return s.sum() / c.sum(), (self.W @ s) / (self.W @ c)

    def auroc(self, y, s, tasks, n=None):
        t = np.array([self.tid[k] for k in tasks])
        W = self.W[:n] if n else self.W
        vals = [roc_auc_score(y, s, sample_weight=w[t]) for w in W if len(np.unique(y[w[t] > 0])) == 2]
        return est(roc_auc_score(y, s), np.array(vals))


def est(point, draws):
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return {"est": round(float(point), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}


def populations(scores: pd.DataFrame, paired: set) -> pd.DataFrame:
    attack = scores.arm.isin([E1, E2, SIM])
    tasks = set(scores.loc[attack & scores.cell.isin(paired), "main_task"])
    keep = (attack & scores.cell.isin(paired)) | ((scores.arm == HON) & scores.main_task.isin(tasks))
    df = scores[keep]
    per_cell = df[df.arm != HON].groupby("cell").arm.nunique()
    runs = df[df.arm != HON].groupby(["arm", "cell"]).key.nunique()
    if len(per_cell) != len(paired) or (per_cell != 3).any() or (runs != 1).any():
        raise ValueError(f"paired cells: {len(per_cell)} of {len(paired)} scored, or an arm missing / duplicated")
    return df


def analyze(boot, runs: pd.DataFrame, col: str, full: bool) -> dict:
    """Run-level statistics for one score column. `full` adds AUROC intervals (slow)."""
    out = {"mean": {a: round(float(runs.loc[runs.arm == a, col].mean()), 4) for a in (E1, E2, SIM, HON)},
           "sd_e1": round(float(runs.loc[runs.arm == E1, col].std()), 4)}
    for name, (a, b) in {"sim_minus_e1": (E1, SIM), "e2_minus_e1": (E1, E2), "honest_minus_e1": (E1, HON)}.items():
        pa, da = boot.mean(runs.loc[runs.arm == a, "main_task"], runs.loc[runs.arm == a, col].to_numpy())
        pb, db = boot.mean(runs.loc[runs.arm == b, "main_task"], runs.loc[runs.arm == b, col].to_numpy())
        out[name] = est(pb - pa, db - da) | {"in_sd_e1": round(float((pb - pa) / out["sd_e1"]), 4)}
        r = runs[runs.arm.isin([a, b])]
        y = (r.arm == (a if name == "honest_minus_e1" else b)).to_numpy().astype(int)  # e1 positive against honest
        auc_name = {"sim_minus_e1": "auroc_sim_vs_e1", "e2_minus_e1": "auroc_e2_vs_e1", "honest_minus_e1": "auroc_e1_vs_honest"}[name]
        out[auc_name] = boot.auroc(y, r[col].to_numpy(), r.main_task) if full else {"est": round(float(roc_auc_score(y, r[col])), 4)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--templated", required=True)
    ap.add_argument("--fit", required=True)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ex = json.loads((HERE.parent.parent / "splits" / "exclusions_v3.json").read_text())
    paired = set(ex["paired_attack_cells"]["cells"])
    fit = json.loads((HERE / args.fit).read_text())
    primary = (fit["primary"]["readout"], fit["primary"]["layer"])
    best = fit["best_layer_per_readout"]  # the best layer of each assistant readout, also analyzed in full

    df = populations(pd.read_parquet(HERE / args.scores), paired)
    cols = [c for c in df.columns if c.startswith(("asst_", "all_"))]
    if f"{primary[0]}_L{primary[1]}" not in cols:
        raise ValueError(f"the score table has no column for the primary probe {primary}; score that layer")
    runs = df.groupby(["arm", "key", "main_task"], as_index=False)[cols].mean()
    boot = Boot(set(runs.main_task), args.bootstrap, args.seed)
    full_layers = {16, 24, 32, 40, 48, 56, 63, primary[1], *best.values()}
    res = {c: analyze(boot, runs, c, int(c.rsplit("_L", 1)[1]) in full_layers) for c in cols}

    tdf = populations(pd.read_parquet(HERE / args.templated).query("kind == 'call_end'"), paired)
    tcols = [c for c in tdf.columns if c.startswith("L")]
    truns = tdf.groupby(["arm", "key", "main_task"], as_index=False)[tcols].mean()
    if set(truns.key) != set(runs.key):
        raise ValueError("the templated and Needham score tables cover different runs")
    templated = {c: analyze(boot, truns, c, True) for c in tcols}
    both = runs.merge(truns, on=["arm", "key", "main_task"], validate="one_to_one")
    corr = {c: {r: {"pearson_all_runs": round(float(np.corrcoef(both[f"{r}_{c}"], both[c])[0, 1]), 4),
                    "pearson_e1_runs": round(float(np.corrcoef(both.loc[both.arm == E1, f"{r}_{c}"], both.loc[both.arm == E1, c])[0, 1]), 4)}
                for r in ("asst_last", "asst_mean")} for c in tcols if f"asst_last_{c}" in both}

    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    out = HERE / "results" / f"la_shift_{stamp}.json"
    out.write_text(json.dumps({"inputs": vars(args), "git_commit": commit, "primary": {"readout": primary[0], "layer": primary[1]}, "best_layer_per_readout": best,
                               "runs_per_arm": runs.groupby("arm").size().to_dict(), "n_main_tasks": len(boot.tid),
                               "needham": res, "templated": templated, "run_score_correlation": corr}, indent=1))
    f = lambda d: f"{d['est']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"
    a = lambda d: f"{d['est']:.3f}" + (f" [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]" if "ci95" in d else "")
    print("runs per arm:", runs.groupby("arm").size().to_dict(), "| primary:", primary)
    for name, table in (("needham asst_last", {c: v for c, v in res.items() if c.startswith("asst_last_")}),
                        ("needham asst_mean", {c: v for c, v in res.items() if c.startswith("asst_mean_")}),
                        ("templated", templated)):
        print(f"\n== {name}")
        for c, r in table.items():
            if int(c.rsplit("L", 1)[1]) not in full_layers:
                continue
            print(f"  {c}: sim-e1 {f(r['sim_minus_e1'])} ({r['sim_minus_e1']['in_sd_e1']:+.3f} sd) | e2-e1 {f(r['e2_minus_e1'])} | "
                  f"honest-e1 {f(r['honest_minus_e1'])} | AUROC sim/e1 {a(r['auroc_sim_vs_e1'])} e2/e1 {a(r['auroc_e2_vs_e1'])} e1/hon {a(r['auroc_e1_vs_honest'])}")
    print("\nrun-score correlation with the templated probe:", json.dumps(corr))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
