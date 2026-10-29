"""Pick the simulated attack runs for the rendering swap (local, from the shift analysis's saved tables and transcripts).

Eligible: a sim-attack run in the Q2 attack view with at least one Q2 call after an insert output that can be re-rendered
(an insert in an earlier turn whose category is exact, approx_window or approx_numbering).
The sample, stratified on rebuild quality so that the subset where every rebuild is exact is not left to chance:
- every eligible run whose inserts all rebuild exactly;
- the rest drawn at random (seed from config) up to n_runs.
Writes config/swap_runs.json with each run's insert categories, as computed here from the local transcripts. The
staging step recomputes them from the staged messages and must agree.
With --relabel: keep the runs already in config/swap_runs.json and their selection stratum, and only recompute each
run's insert categories (2026-10-08: hunks with an empty old range, e.g. -233,0, had been labeled approx_numbering).

  uv run --with pandas python experiments/2026-10-07_q2_rendering_swap/select_runs.py [--relabel]
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "2026-10-06_q2_impactful_call_transfer" / "shift"))
import examples as ex  # noqa: E402
import rerender as rr  # noqa: E402

cfg = yaml.safe_load((HERE / "config" / "swap.yaml").read_text())
q = ex.table("qwen3-32b")
sim_keys = sorted(set(q[q.domain == "sim"].key))
runs = ex.load_runs(set(sim_keys))
q2_calls = {k: set(zip(g.step, g.action)) for k, g in q[q.domain == "sim"].groupby("key")}
info = {}
for key in sim_keys:
    calls = runs[key]
    order = sorted(calls)
    records = []
    for sa in order:
        c = calls[sa]
        a = rr.args_of(c)
        if c["tool"] != "text_editor" or a.get("command") != "insert":
            continue
        history = [(rr.args_of(calls[sb]), calls[sb]["output"], calls[sb]["tool"]) for sb in order if sb < sa]  # every earlier call
        records.append({"step": sa[0], "action": sa[1], "category": rr.rerender(a, c["output"] or "", history)[1]})
    first = min((r["step"] for r in records if r["category"] in rr.RENDERED), default=None)
    after = sum(1 for s, _ in q2_calls[key] if first is not None and s > first)
    cats = Counter(r["category"] for r in records)
    info[key] = {"inserts": records, "q2_calls_after_rendered_insert": after,
                 "all_exact": cats.get("exact", 0) > 0 and set(cats) <= {"exact", "not_rendered"}, "any_failed": cats.get("failed", 0) > 0}
eligible = [k for k, v in info.items() if v["q2_calls_after_rendered_insert"] > 0]
if "--relabel" in sys.argv:
    out = json.loads((HERE / "config" / "swap_runs.json").read_text())
    old_runs = out["runs"]
    stratum = {k: v.get("stratum", "all_exact" if v["all_exact"] else "random") for k, v in old_runs.items()}
    pick = sorted(old_runs)
    changed = {k: [(a["step"], a["action"], a["category"], b["category"]) for a, b in zip(old_runs[k]["inserts"], info[k]["inserts"])
                   if a["category"] != b["category"]] for k in pick}
    out["relabeled"] = {"date": "2026-10-08", "why": "hunks with an empty old range (no context lines, e.g. -233,0) were labeled approx_numbering; "
                        "the selection and its strata are unchanged (made with the old labels)",
                        "changed (step, action, old, new)": {k: v for k, v in changed.items() if v}}
    out["runs"] = {k: info[k] | {"stratum": stratum[k]} for k in pick}
else:
    exact = [k for k in eligible if info[k]["all_exact"]]
    rest = [k for k in eligible if not info[k]["all_exact"]]
    rng = np.random.default_rng(cfg["seed"])
    pick = sorted(exact) + sorted(rng.choice(rest, cfg["n_runs"] - len(exact), replace=False).tolist())
    out = {"seed": cfg["seed"], "n_eligible": len(eligible), "n_all_exact_eligible": len(exact),
           "strata": {"all_exact (all included)": len(exact), "random from the rest": len(pick) - len(exact)},
           "runs": {k: info[k] | {"stratum": "all_exact" if k in exact else "random"} for k in pick}}
(HERE / "config" / "swap_runs.json").write_text(json.dumps(out, indent=1))
cats = Counter(r["category"] for k in pick for r in info[k]["inserts"])
print(f"eligible {len(eligible)}; picked {len(pick)} {dict(Counter(v['stratum'] for v in out['runs'].values()))}; insert categories in the sample {dict(cats)}; "
      f"Q2 calls after a rendered insert {sum(info[k]['q2_calls_after_rendered_insert'] for k in pick)}; "
      f"runs with a failed insert {sum(info[k]['any_failed'] for k in pick)}")
