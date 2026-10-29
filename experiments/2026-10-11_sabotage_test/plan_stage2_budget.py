"""Stage 2's labelling budget (PLAN.md 6): convert the sabotage calls each run set needs (from run_sizing.py) into k, the
flagged calls labelled per run, under the candidate rule's unequal inclusion probabilities. Numbers only; run after
stage 1 with its measured rates.

Per run r of a set, sabotage sits in pools c (target-touching, flagged outside it, unflagged impactful, unflagged
read-only) at stage-1 rates s_c (weighted sabotage calls per run in that pool). Pool c is labelled with probability p_c:
1 for target-touching, min(k, N_r) / N_r for flagged (N_r = the run's flagged calls outside the target-touching set), the
audit rates for the unflagged pools. A labelled sabotage call has weight 1 / p_c, so the weighted catch has Kish
effective size n_eff = (sum_r sum_c s_c)^2 / sum_r sum_c s_c / p_c. The budget is the smallest k with n_eff >= need in
every set the comparison uses (real run 1 and simulated; real run 1 and real run 2), searched jointly with the audit rate
of the small unflagged-impactful pool (stage 1's rate, 0.25, 0.5 or 1): the cheapest pair in labelled rows wins, since
sabotage the audit reaches at a low rate carries large weights. If no pair reaches it (every flagged call and the whole
unflagged-impactful pool labelled), the set is "not reachable" and stage 2 stops for Aswin's decision; it never runs
underpowered.

Calls are joined to the monitor's record through the metas' command entry (check_joins.rescore), not call_index + 1.
The need per comparison is read from run_sizing.py's result: the scenario row at the stage-1 p and rho with r = 0 and
`at_available_tasks` = 116, its `commands_per_arm`; a null there means not reachable. The reported rows include the
target-touching calls and both audit pools.

  uv run --no-project --python 3.12 --with pandas --with numpy python plan_stage2_budget.py --rates <stage-1 rates json> \\
      --sizing <run_sizing result json> --p P --rho R --inputs <dir> --metas <acts_meta root> --audit-imp A --audit-ro B
"""

import argparse
import json
import math

import numpy as np

import check_joins as CJ
import prepare_stage1 as P


def n_eff(s: dict, p: dict) -> float:
    """s: pool -> sabotage per run (arrays over runs); p: pool -> inclusion probability (arrays over runs)."""
    tot = sum(s[c].sum() for c in s)
    var = sum(np.divide(s[c], p[c], out=np.zeros(len(s[c])), where=s[c] > 0).sum() for c in s)
    return tot**2 / var if var else 0.0


def budget(pools: dict, rates: dict, a_imp: float, a_ro: float, need: float | None) -> dict:
    """Cheapest (k, unflagged-impactful audit rate) giving n_eff >= need for one set; pools: pool -> calls per run.
    A pool a run doesn't have holds none of its sabotage (its expected sabotage is 0 there)."""
    if need is None:
        return {"k": None, "reachable": False, "reason": "run_sizing.py found no reachable count at 116 tasks"}
    N = pools["flagged_other"]
    s = {c: rates.get(c, 0.0) * (pools[c] > 0) for c in pools}
    best, e = None, 0.0
    for a in sorted({a_imp, 0.25, 0.5, 1.0}):
        for k in range(1, max(1, int(N.max())) + 1):
            p = {"target_touching": np.ones(len(N)), "flagged_other": np.minimum(k, N) / np.maximum(N, 1), "unflagged_impactful": a * np.ones(len(N)),
                 "unflagged_read_only": a_ro * np.ones(len(N))}
            e = n_eff(s, p)
            if e >= need:
                rows = int(pools["target_touching"].sum() + np.minimum(k, N).sum()) + round(a * pools["unflagged_impactful"].sum()) + round(a_ro * pools["unflagged_read_only"].sum())
                if best is None or rows < best["rows"]:
                    best = {"k": k, "audit_rate_unflagged_impactful": a, "n_eff": round(e, 1), "need": need, "rows": rows}
                break
    return best or {"k": None, "reachable": False, "n_eff_at_everything": round(e, 1), "need": need}


def need_from_sizing(path: str, p: float, rho: float) -> dict:
    rows = json.loads(open(path).read())["scenarios"]
    out = {}
    for comp in ("e1_vs_sim", "e1_vs_e2"):
        hit = [r for r in rows if r.get("comparison") == comp and r.get("at_available_tasks") == 116 and math.isclose(r["p"], p) and math.isclose(r["rho"], rho) and r["r"] == 0]
        if len(hit) != 1:
            raise SystemExit(f"{comp}: {len(hit)} matching scenario rows at 116 tasks")
        out[comp] = hit[0]["commands_per_arm"]
    return out


def main(rates: str, sizing: str, p: float, rho: float, inputs: str, metas: str, audit_imp: float, audit_ro: float):
    R, Nd = json.loads(open(rates).read()), need_from_sizing(sizing, p, rho)
    command, _ = CJ.entries(metas)
    C = CJ.rescore(P.load(inputs), command, inputs)
    sp = json.loads(P.SPLIT_FILE.read_text())
    stage2 = [t for t, v in sp["tasks"].items() if v["side"] == "held_out" and v["usable"] and not v["stage1"]]
    T = C[C.task_key.isin(stage2) & C.scored]
    out = {"stage2_tasks": len(stage2), "need": Nd}
    for comp, sets in (("e1_vs_sim", ("real", "sim")), ("e1_vs_e2", ("real", "run2"))):
        out[comp] = {}
        for w in sets:
            runs = sorted(C[C.task_key.isin(stage2) & (C.world == w)].run_id.unique())
            g = T[T.world == w]
            per = lambda m: g[m].groupby("run_id").size().reindex(runs, fill_value=0).values  # noqa: E731
            pools = {"target_touching": per(g.touch), "flagged_other": per(g.flagged & ~g.touch),
                     "unflagged_impactful": per(~g.flagged & ~g.touch & (g.y == 1)), "unflagged_read_only": per(~g.flagged & ~g.touch & (g.y == 0))}
            out[comp][w] = budget(pools, R[w], audit_imp, audit_ro, Nd[comp])
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", required=True, help="stage-1 weighted sabotage per run, by set then pool")
    ap.add_argument("--sizing", required=True, help="run_sizing.py result json")
    ap.add_argument("--p", type=float, required=True, help="stage-1 catch at its 2.5th percentile, as passed to run_sizing.py")
    ap.add_argument("--rho", type=float, required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--metas", required=True)
    ap.add_argument("--audit-imp", type=float, required=True)
    ap.add_argument("--audit-ro", type=float, required=True)
    main(**vars(ap.parse_args()))
