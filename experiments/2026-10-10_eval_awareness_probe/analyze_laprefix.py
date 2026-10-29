"""Analyze reduced content test A as the README's "Reduced content test A" sets out ($0, saved rows only).

Caveat (Codex): this shows the probe's sensitivity to the substituted prefix, not whether retraining would fix the gap.
Refuses unless the run is valid (its unpadded pairs reproduce the stored held-out scores within 0.1), its prefixes sit
in the 8,000-8,400 token band from 40 distinct sources and LinuxArena tasks, and it used the padded sweep's probes and
pair file. For each pair: d_LA = implied minus ordinary with the real prefix, d_syn = the same at 8,000 synthetic filler
tokens (the padded sweep). R = mean d_LA / mean d_syn, with a 95% range from 2,000 resamples of the 40 held-out tasks
(seed 20261010; a task's pairs, and so its prefix source, move together). Outcome, as set before the run: suppression
if R's range is entirely below 0.5, equivalence if entirely within [0.8, 1.25], otherwise inconclusive.
Post hoc, added at review after the first run started and before any result was read: a condition-swap null.
For each of 2,000 draws, one coin per held-out task swaps that task's d_LA and d_syn for all its pairs; R under no
context difference sits near 1. It is reported beside the pre-set outcome, not used to classify it.

  uv run --no-project --with pandas python experiments/2026-10-10_eval_awareness_probe/analyze_laprefix.py --laprefix results/diag_laprefix_<stamp>.json --padded results/diag_padded_2026-10-10_143844.json --fit results/fit_2026-10-10_022437.json
"""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
B, SEED, TOL, BAND = 2000, 20261010, 0.1, (8000, 8400)

ap = argparse.ArgumentParser()
for a_ in ("laprefix", "padded", "fit"):
    ap.add_argument(f"--{a_}", required=True)
a = ap.parse_args()
lap, pad, fit = (json.loads((HERE / getattr(a, x)).read_text()) for x in ("laprefix", "padded", "fit"))


def rows_of(summary):
    raw = (ROOT / ".run_cache" / "eval_awareness_probe" / summary["rows"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != summary["rows_sha256"]:
        raise ValueError(f"{summary['rows']} does not match its recorded sha256")
    return pd.DataFrame(json.loads(raw))


if lap["mode"] != "laprefix" or pad["mode"] != "padded" or not (lap["valid"] and pad["valid"]):
    raise ValueError("expected a valid laprefix run and a valid padded run")
if len({(x["fit_id"], x["probes_sha256"], x["sets_sha256"]) for x in (lap, pad)} | {(fit["fit_id"], fit["probes_sha256"], fit["sets_sha256"])}) != 1:
    raise ValueError("the runs used different probes or pair files")
df, dp = rows_of(lap), rows_of(pad)
if not (np.isfinite(df[["score_per_token", "score_last_token"]].to_numpy()).all()
        and np.isfinite(dp[["score_per_token", "score_last_token"]].to_numpy()).all()):
    raise ValueError("non-finite scores in the laprefix or padded rows")
base = df[(df["version"] == "unchanged") & (df["length"] == 0)]
repro = {}
for r in ("per_token", "last_token"):
    stored = pd.Series({x["id"]: x["score"] for x in fit["heldout_scores_at_chosen_layer"][r]})
    got = base.set_index("id")[f"score_{r}"]
    if not got.index.is_unique or set(got.index) != set(stored.index):
        raise ValueError(f"{r}: the unpadded dialogues are not exactly the 240 held-out dialogues")
    gap = (got - stored.loc[got.index]).abs()
    repro[r] = {"dialogues": len(gap), "max abs diff": round(float(gap.max()), 5), "tolerance": TOL}
    if not (np.isfinite(gap.to_numpy()).all() and float(gap.max()) <= TOL):
        raise ValueError(f"{r}: the unpadded dialogues do not reproduce the stored scores ({repro[r]})")
la8 = df[(df["version"] == "unchanged") & (df["length"] == BAND[0])]
src = la8.groupby("task")["prefix_source"].agg(lambda s: s.unique().tolist())
if any(len(v) != 1 for v in src) or len(src) != 40 or len({v[0] for v in src}) != 40 \
        or len({v[0].split("__")[1].split(":")[0] for v in src}) != 40:
    raise ValueError("expected one prefix source per held-out task, 40 distinct sources from 40 LinuxArena tasks")
if not la8["prefix_tokens"].between(*BAND).all():
    raise ValueError("a prefix is outside the 8,000-8,400 token band")
out = {"caveat": "this shows the probe's sensitivity to the substituted prefix, not whether retraining would fix the gap",
       "laprefix": a.laprefix, "padded": a.padded, "fit": a.fit, "reproduction": repro,
       "prefix tokens (min/median/max)": [int(la8["prefix_tokens"].min()), int(la8["prefix_tokens"].median()), int(la8["prefix_tokens"].max())],
       "measured prefix in place (min/median/max)": [int(la8["filler_tokens"].min()), int(la8["filler_tokens"].median()), int(la8["filler_tokens"].max())],
       "bootstrap": {"resamples": B, "seed": SEED, "unit": "held-out task (one prefix source each)"},
       "outcome rule": "suppression: R 95% range < 0.5; equivalence: within [0.8, 1.25]; else inconclusive",
       "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=True).stdout.strip()}


def by_pair(d, lab, L, col):
    s = d[(d["version"] == "unchanged") & (d["label"] == lab) & (d["length"] == L)].set_index("pair")[col]
    if not s.index.is_unique or len(s) != 120:
        raise ValueError(f"expected 120 pairs for label {lab} at length {L}")
    return s


for r in ("per_token", "last_token"):
    col = f"score_{r}"
    d_la = by_pair(df, 1, BAND[0], col) - by_pair(df, 0, BAND[0], col)
    d_syn = (by_pair(dp, 1, 8000, col) - by_pair(dp, 0, 8000, col)).loc[d_la.index]
    task = by_pair(df, 1, BAND[0], "task").loc[d_la.index].to_numpy()
    tasks = sorted(set(task))
    groups = [np.flatnonzero(task == t) for t in tasks]
    rng = np.random.default_rng(SEED)
    ratios = []
    for _ in range(B):
        pick = np.concatenate([groups[k] for k in rng.integers(0, len(tasks), len(tasks))])
        ratios.append(d_la.to_numpy()[pick].mean() / d_syn.to_numpy()[pick].mean())
    nulls = []  # post hoc condition-swap null: per task, swap the two conditions' changes
    t_idx = np.searchsorted(tasks, task)
    for coins in np.random.default_rng(SEED + 1).integers(0, 2, size=(B, len(tasks))):
        sw = coins[t_idx].astype(bool)
        a_, b_ = np.where(sw, d_syn.to_numpy(), d_la.to_numpy()), np.where(sw, d_la.to_numpy(), d_syn.to_numpy())
        nulls.append(a_.mean() / b_.mean())
    R = float(d_la.mean() / d_syn.mean())
    if not (np.isfinite(R) and np.isfinite(ratios).all() and float(d_syn.mean()) > 0):
        raise ValueError(f"{r}: the ratio is not finite or the synthetic comparator is not positive")
    lo, hi = (float(q) for q in np.percentile(ratios, [2.5, 97.5]))
    outcome = "suppression" if hi < 0.5 else "equivalence" if lo >= 0.8 and hi <= 1.25 else "inconclusive"
    o0, o8 = by_pair(df, 0, 0, col), by_pair(df, 0, BAND[0], col).loc[by_pair(df, 0, 0, col).index]
    out[r] = {"pairs": len(d_la), "d_LA median": round(float(d_la.median()), 3), "d_LA mean": round(float(d_la.mean()), 3),
              "d_syn median": round(float(d_syn.median()), 3), "d_syn mean": round(float(d_syn.mean()), 3),
              "R (mean d_LA / mean d_syn)": round(R, 4), "R 95% range": [round(lo, 4), round(hi, 4)], "outcome": outcome,
              "ordinary drift from unpadded with the real prefix (mean)": round(float((o8 - o0).mean()), 3),
              "share of pairs with d_LA > 0": round(float((d_la > 0).mean()), 4),
              "condition-swap null for R (post hoc), 2.5/97.5": [round(float(q), 4) for q in np.percentile(nulls, [2.5, 97.5])],
              "R outside the swap null's 95% band (post hoc)": bool(not (np.percentile(nulls, 2.5) <= R <= np.percentile(nulls, 97.5)))}
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
path = HERE / "results" / f"laprefix_analysis_{stamp}.json"
path.write_text("{\n" + ",\n".join(f" {json.dumps(k)}: {json.dumps(v)}" for k, v in out.items()) + "\n}\n")
print(json.dumps(out, indent=1))
print(f"-> {path}")
