"""Rendering swap analysis: how much of the real->sim gap after inserts closes when the simulator's insert outputs are
re-rendered in the real tool's style?

Calls scored: each Q2 call in the 100 sim runs that comes after a re-rendered insert (an insert in an earlier turn that
was rendered). Each call has three scores from the same frozen probe:
- `original`: the as-simulated copy, freshly extracted;
- `rerendered`: the re-rendered copy, freshly extracted;
- `saved`: the level-0 refit's own test score, a reproduction check of the fresh extraction.
The real side of the gap is the same cells' real runs (attack epoch 1): the refit's saved scores for calls after an
insert output in an earlier turn.

Closure = (AUROC_rerendered - AUROC_original) / (AUROC_real - AUROC_original). 95% intervals resample main tasks
(each task's cells, real and sim runs together; 1000 draws, seed 0), as everywhere in Q2.

Subsets:
- all calls;
- runs with no failed insert;
- calls whose earlier inserts all rebuilt with correct line numbers (exact or approx_window), vs calls after an
  approx_numbering or failed insert;
- the 10 all-exact runs (too few to carry the result alone).
Positions: pre_call (the headline), prev_content (its content-token control), call_end (control).

  uv run --with pandas --with scikit-learn python experiments/2026-10-07_q2_rendering_swap/analyze_swap.py
"""
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "2026-10-06_q2_impactful_call_transfer" / "shift"))
import examples as ex  # noqa: E402
import rerender as rr  # noqa: E402

N_BOOT, SEED = 1000, 0
scores = json.loads(sorted((ROOT / ".run_cache" / "q2_swap").glob("scores_2*.json"))[-1].read_text())
runs_info = json.loads((HERE / "config" / "swap_runs.json").read_text())["runs"]
sw = pd.DataFrame(scores["swap"])
saved = pd.DataFrame(scores["saved"])

# Sim: the first rendered insert's turn, and which insert categories precede each call.
def sim_context(key, step):
    ins = [r for r in runs_info[key]["inserts"] if r["step"] < step]
    rendered = [r for r in ins if r["category"] in rr.RENDERED]
    return {"after_rendered": bool(rendered),
            "numbering_ok": all(r["category"] in ("exact", "approx_window") for r in ins if r["category"] != "not_rendered"),
            "any_failed_run": runs_info[key]["any_failed"], "all_exact_run": runs_info[key]["all_exact"]}

sw = sw.join(pd.DataFrame([sim_context(k, s) for k, s in zip(sw.key, sw.step)], index=sw.index))
# Check: every call before the first re-rendered output has an identical prefix in both copies, so identical scores.
before_change = sw[~sw.after_rendered]
unchanged_check = {"calls": int(len(before_change)), "calls_by_position": before_change.position.value_counts().to_dict(),
                   "max_abs_diff": float((before_change.rerendered - before_change.original).abs().max()) if len(before_change) else None}
sw = sw[sw.after_rendered]
q = ex.table("qwen3-32b")
cell_of, task_of = dict(zip(q.key, q.cell)), dict(zip(q.key, q.main_task))  # from the score tables
sw["cell"], sw["main_task"] = sw.key.map(cell_of), sw.key.map(task_of)
assert sw.cell.notna().all() and sw.main_task.notna().all()

# Real: the same cells' real runs, calls after an insert output in an earlier turn.
cells = set(sw.cell)
real_keys = sorted(set(q[(q.domain == "real") & q.cell.isin(cells)].key))
rruns = ex.load_runs(set(real_keys))
first_insert = {}
for key, calls in rruns.items():
    steps = [sa[0] for sa, c in calls.items() if c["tool"] == "text_editor" and rr.args_of(c).get("command") == "insert"]
    first_insert[key] = min(steps) if steps else None
real = saved[(saved.arm == "real-attack-e1") & saved.key.isin(real_keys)].copy()
real = real[[first_insert[k] is not None and s > first_insert[k] for k, s in zip(real.key, real.step)]]
real["cell"], real["main_task"] = real.key.map(cell_of), real.key.map(task_of)


def auc(y, s):
    return roc_auc_score(y, s) if len(set(y)) == 2 else np.nan


def summarize(sub: pd.DataFrame, pos: str) -> dict:
    s, r = sub[sub.position == pos], real
    rcol = f"{pos}_score"
    cell_list = sorted(set(s.cell))
    r = r[r.cell.isin(cell_list)]
    task_list = sorted(set(s.main_task))
    s_by, r_by = {t: g for t, g in s.groupby("main_task")}, {t: g for t, g in r.groupby("main_task")}

    def stats(ts):
        ss = pd.concat([s_by[t] for t in ts if t in s_by])
        rs = pd.concat([r_by[t] for t in ts if t in r_by]) if any(t in r_by for t in ts) else r.iloc[:0]
        a_o, a_r, a_real = auc(ss.y, ss.original), auc(ss.y, ss.rerendered), auc(rs.y, rs[rcol])
        d = ss.rerendered - ss.original
        return {"auroc_real": a_real, "auroc_sim_original": a_o, "auroc_sim_rerendered": a_r,
                "gap_before": a_real - a_o, "gap_after": a_real - a_r, "closure": (a_r - a_o) / (a_real - a_o) if a_real != a_o else np.nan,
                "auroc_change": a_r - a_o,
                "margin_change_read_only": float(d[ss.y == 0].mean()) if (ss.y == 0).any() else np.nan,
                "margin_change_impactful": float(d[ss.y == 1].mean()) if (ss.y == 1).any() else np.nan,
                "recall_original": float((ss[ss.y == 1].original > ss[ss.y == 1].threshold).mean()),
                "recall_rerendered": float((ss[ss.y == 1].rerendered > ss[ss.y == 1].threshold).mean()),
                "fpr_original": float((ss[ss.y == 0].original > ss[ss.y == 0].threshold).mean()) if (ss.y == 0).any() else np.nan,
                "fpr_rerendered": float((ss[ss.y == 0].rerendered > ss[ss.y == 0].threshold).mean()) if (ss.y == 0).any() else np.nan,
                "recall_real": float((rs[rs.y == 1][rcol] > rs[rs.y == 1][f"{pos}_threshold"]).mean()) if len(rs) else np.nan}

    point = stats(task_list)
    rng = np.random.default_rng(SEED)
    draws = [stats(rng.choice(task_list, len(task_list), replace=True)) for _ in range(N_BOOT)]
    ci = {k: [round(float(x), 4) for x in np.nanpercentile([d[k] for d in draws], [2.5, 97.5])] for k in point}
    return {"n_calls": {"sim impactful": int((s.y == 1).sum()), "sim read-only": int((s.y == 0).sum()),
                        "real impactful": int((r[r.cell.isin(cell_list)].y == 1).sum()), "real read-only": int((r[r.cell.isin(cell_list)].y == 0).sum()),
                        "runs": int(s.key.nunique()), "cells": len(cell_list), "main_tasks": len(task_list)},
            **{k: {"value": round(float(v), 4), "ci95": ci[k]} for k, v in point.items()}}


subsets = {"all": sw, "runs with no failed insert": sw[~sw.any_failed_run],
           "calls after correctly numbered inserts only (exact, approx_window)": sw[sw.numbering_ok],
           "calls after an approx_numbering or failed insert": sw[~sw.numbering_ok],
           "all-exact runs (too few to carry the result alone)": sw[sw.all_exact_run]}
out = {"layer_source": scores.get("layer_source"), "headline_layers": scores["headline_layers"],
       "skipped_readouts": pd.DataFrame(scores["skipped"]).position.value_counts().to_dict() if scores["skipped"] else {},
       "unchanged_before_first_change": unchanged_check,
       "insert_categories": pd.Series([r["category"] for v in runs_info.values() for r in v["inserts"]]).value_counts().to_dict(),
       "reproduction": {pos: {"median_abs_diff": float((g.original - g.saved).abs().median()), "max_abs_diff": float((g.original - g.saved).abs().max()),
                              "n": len(g)} for pos, g in pd.DataFrame(scores["swap"]).groupby("position")},
       "results": {name: {pos: summarize(sub, pos) for pos in ("pre_call", "prev_content", "call_end") if (sub.position == pos).any()}
                   for name, sub in subsets.items() if len(sub)}}
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
(HERE / "results" / f"swap_{stamp}.json").write_text(json.dumps(out, indent=1, default=float))
f = lambda v: f"{v['value']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]"
print("reproduction (fresh original vs saved):", out["reproduction"])
print("unchanged before the first re-rendered output:", unchanged_check, "| skipped readouts:", out["skipped_readouts"])
for name, by_pos in out["results"].items():
    for pos, r in by_pos.items():
        print(f"{name[:48]:48} {pos:12} n {r['n_calls']} | AUROC real {r['auroc_real']['value']:.3f} sim {r['auroc_sim_original']['value']:.3f} -> "
              f"{r['auroc_sim_rerendered']['value']:.3f} | closure {f(r['closure'])} | margin change ro {f(r['margin_change_read_only'])} imp {f(r['margin_change_impactful'])}")
print(f"wrote results/swap_{stamp}.json")
