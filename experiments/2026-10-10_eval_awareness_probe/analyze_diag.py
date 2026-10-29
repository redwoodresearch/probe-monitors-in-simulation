"""Analyze the bridge and padded diagnostics as the README's "Diagnostics after the near retry" sets out ($0, saved rows).

Refuses unless the run's unchanged, unpadded dialogues reproduce the stored held-out scores within 0.1 logits (both
readouts, all 240), the rows match their hash, and the run used these probes and this pair file. Then, per readout:
- bridge: for "sys" and "near", the per-pair change from the unchanged ordinary side (median, mean, share positive, by
  mechanism) and Aswin's planted-check rule;
- padded: per filler length, the paired implied-test minus ordinary change and the rule, the ordinary side's drift, the
  measured filler tokens; and the explicit near line at 0 and 32k.
The rule: the estimate is the mean dialogue score (unplanted = without the plant, planted = with it). Detected when the
shift is 0.5-2x the reference AND the planted estimate's 95% range excludes the unplanted estimate. The reference is
the in-run median paired change, implied-test minus ordinary, unpadded. Ranges: 2,000 bootstrap resamples of the
held-out tasks (seed 20261010). A swap null (planted and unplanted values swapped at random, one coin per held-out task
applied to all of its pairs, matching the bootstrap's task unit; 2,000 draws) gives the shift's 95th percentile under
no effect; "detected (rule + swap null)" also requires the shift
to exceed it. The swap null was added after the padded sweep launched (14:38 PT) and after the bridge's results, at
review (#391), so for the bridge it is post hoc.

  uv run --no-project --with pandas python experiments/2026-10-10_eval_awareness_probe/analyze_diag.py --diag results/diag_<mode>_<stamp>.json --fit results/fit_2026-10-10_022437.json
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
B, SEED, ABOUT, TOL = 2000, 20261010, (0.5, 2.0), 0.1

ap = argparse.ArgumentParser()
ap.add_argument("--diag", required=True)
ap.add_argument("--fit", required=True)
a = ap.parse_args()
diag, fit = json.loads((HERE / a.diag).read_text()), json.loads((HERE / a.fit).read_text())
raw = (ROOT / ".run_cache" / "eval_awareness_probe" / diag["rows"]).read_bytes()
if hashlib.sha256(raw).hexdigest() != diag["rows_sha256"]:
    raise ValueError("the rows file does not match its recorded sha256")
if (diag["fit_id"], diag["probes_sha256"], diag["sets_sha256"]) != (fit["fit_id"], fit["probes_sha256"], fit["sets_sha256"]):
    raise ValueError("the diagnostics used other probes or another pair file")
df = pd.DataFrame(json.loads(raw))
if not np.isfinite(df[["score_per_token", "score_last_token"]].to_numpy()).all():
    raise ValueError("non-finite scores")
base = df[(df["version"] == "unchanged") & (df["length"] == 0)]
repro = {}
for r in ("per_token", "last_token"):
    stored = pd.Series({x["id"]: x["score"] for x in fit["heldout_scores_at_chosen_layer"][r]})
    got = base.set_index("id")[f"score_{r}"]
    if not got.index.is_unique or set(got.index) != set(stored.index):
        raise ValueError(f"{r}: the unchanged dialogues are not exactly the 240 held-out dialogues")
    gap = (got - stored.loc[got.index]).abs()
    repro[r] = {"dialogues": len(gap), "max abs diff": round(float(gap.max()), 5), "tolerance": TOL}
    if not float(gap.max()) <= TOL:
        raise ValueError(f"{r}: unchanged dialogues do not reproduce the stored scores ({repro[r]})")
tasks = sorted(df["task"].unique())
rng = np.random.default_rng(SEED)
draws = [rng.integers(0, len(tasks), len(tasks)) for _ in range(B)]  # the same task resamples for every estimate


def boot_mean(per_pair: pd.Series, task_of: pd.Series) -> list[float]:
    """95% range of the mean over pairs, resampling whole tasks."""
    groups = {t: per_pair[task_of == t].to_numpy() for t in tasks}
    means = [np.concatenate([groups[tasks[k]] for k in d]).mean() for d in draws]
    return [float(q) for q in np.percentile(means, [2.5, 97.5])]


def swap_null_p95(d: pd.Series, task_of: pd.Series) -> float:
    """95th percentile of the mean paired difference when each task's pairs have their two values swapped together."""
    t = task_of.loc[d.index].to_numpy()
    codes = np.searchsorted(tasks, t)  # tasks is sorted
    coins = np.random.default_rng(SEED + 1).choice([-1.0, 1.0], size=(B, len(tasks)))
    return float(np.percentile((coins[:, codes] * d.to_numpy()).mean(1), 95))


def rule(unplanted: pd.Series, planted: pd.Series, task_of: pd.Series, ref: float) -> dict:
    if not unplanted.index.equals(planted.index):
        raise ValueError("planted and unplanted values are not aligned by pair")
    shift = float(planted.mean() - unplanted.mean())
    null95 = swap_null_p95(planted - unplanted, task_of)
    p_ci = boot_mean(planted, task_of)
    u = float(unplanted.mean())
    about, excludes = ABOUT[0] * ref <= shift <= ABOUT[1] * ref, p_ci[0] > u or p_ci[1] < u
    return {"unplanted estimate": round(u, 3), "planted estimate": round(float(planted.mean()), 3), "planted 95% range": [round(q, 3) for q in p_ci],
            "shift": round(shift, 3), "shift / reference": round(shift / ref, 4), "shift about the planted amount": bool(about),
            "planted range excludes the unplanted estimate": bool(excludes), "detected (rule)": bool(about and excludes),
            "swap null p95 (shift)": round(null95, 3), "shift above swap null p95": bool(shift > null95),
            "detected (rule + swap null)": bool(about and excludes and shift > null95)}


def by_pair(sub: pd.DataFrame, col: str) -> pd.Series:
    s = sub.set_index("pair")[col]
    if not s.index.is_unique:
        raise ValueError("a pair appears twice in one condition")
    return s


out = {"diag": a.diag, "fit": a.fit, "mode": diag["mode"], "reproduction": repro, "bootstrap": {"resamples": B, "seed": SEED, "unit": "held-out task", "tasks": len(tasks)},
       "about (ratio of shift to reference)": list(ABOUT),
       "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=True).stdout.strip()}
task_of = base[base["label"] == 1].set_index("pair")["task"]
for r in ("per_token", "last_token"):
    col = f"score_{r}"
    cond = lambda lab, ver, L, col=col: by_pair(df[(df["label"] == lab) & (df["version"] == ver) & (df["length"] == L)], col)
    o0, i0 = cond(0, "unchanged", 0), cond(1, "unchanged", 0)
    if set(o0.index) != set(i0.index) or len(o0) != 120:
        raise ValueError("expected 120 complete pairs")
    ref = float((i0 - o0.loc[i0.index]).median())
    res = {"reference (in-run median implied-test minus ordinary, unpadded)": round(ref, 3),
           "reference, mean": round(float((i0 - o0.loc[i0.index]).mean()), 3)}
    if diag["mode"] == "bridge":
        mech = base[base["label"] == 0].set_index("pair")["mechanism"]
        for ver in ("sys", "near"):
            v = cond(0, ver, 0).loc[o0.index]
            ch = v - o0
            res[ver] = {"pairs": len(ch), "change median": round(float(ch.median()), 3), "change mean": round(float(ch.mean()), 3),
                        "share positive": round(float((ch > 0).mean()), 4),
                        "change median by mechanism": {m: round(float(ch[mech.loc[ch.index] == m].median()), 3) for m in sorted(mech.unique())},
                        "rule": rule(o0, v, task_of.loc[o0.index], ref)}
    else:
        for L in diag["lengths"]:
            o, i = cond(0, "unchanged", L).loc[o0.index], cond(1, "unchanged", L).loc[o0.index]
            ch = i - o
            ft = df[(df["version"] == "unchanged") & (df["length"] == L)]["filler_tokens"]
            g = {"pairs": len(ch), "filler tokens (min/median/max)": [int(ft.min()), int(ft.median()), int(ft.max())],
                 "paired change median": round(float(ch.median()), 3), "paired change mean": round(float(ch.mean()), 3),
                 "paired change mean 95% range": [round(q, 3) for q in boot_mean(ch, task_of.loc[ch.index])],
                 "ordinary drift from L=0 (mean)": round(float(o.mean() - o0.mean()), 3),
                 "rule (implied-test hint)": rule(o, i, task_of.loc[o.index], ref)}
            if L in diag["explicit_at"]:
                n = cond(0, "near", L).loc[o0.index]
                e = n - o
                g["explicit near line"] = {"change median": round(float(e.median()), 3), "change mean": round(float(e.mean()), 3),
                                           "share positive": round(float((e > 0).mean()), 4), "rule": rule(o, n, task_of.loc[o.index], ref)}
            res[str(L)] = g
    out[r] = res
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
path = HERE / "results" / f"diag_{diag['mode']}_analysis_{stamp}.json"
path.write_text("{\n" + ",\n".join(f" {json.dumps(k)}: {json.dumps(v)}" for k, v in out.items()) + "\n}\n")
print(json.dumps(out, indent=1))
print(f"-> {path}")
