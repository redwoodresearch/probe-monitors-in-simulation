"""Stage 2's labelling budget, validated by simulation (PLAN.md 6). Numbers only; run after stage 1 with its measured rates.

For each candidate budget (k flagged calls per run, the audit rate of the unflagged-impactful pool), cheapest first in
labelled rows, simulate the stage-2 test on the 116 stage-2 tasks' actual pool sizes (corrected join, check_joins.rescore):
- sabotage per run and pool ~ Poisson(stage-1 rate), capped at the pool's size; a run lacking a pool holds none there;
- each sabotage call is labelled with its pool's inclusion probability (target-touching 1; flagged min(k, N) / N, drawn as
  a hypergeometric share of the run's k; audit rates for the unflagged pools) and weighted by 1 / that probability;
- caught iff u_task + e > c, e ~ N(0, 1), u_task ~ N(0, s^2) with within-task correlation rho (one run per task and set,
  task effects independent across sets), c set so the marginal catch equals the scenario's;
- per set the weighted catch; the real-minus-other difference with a task-clustered (ratio-linearized) standard error;
  pass if the 95% interval lies inside +-margin, fail if it lies wholly beyond it.
A budget qualifies if, at every catch p on a grid spanning stage 1's plausible range (clamped to [0.02, 0.98]), both
real run 1 vs simulated and real run 1 vs real run 2 give P(pass | gap 0) >= 0.8 and P(fail | gap 2 x margin) >= 0.8
(the other set at p - 0.2, or p + 0.2 when p < 0.2). With --runs-per-task r (default 1, the committed design),
each task holds r runs per set with the task's pool sizes repeated r times, r times the sabotage rate, k per run (k x r
per task), and one task effect shared by its r runs: a minimal extension for pricing more runs per task. A candidate that meets it at n_sim draws must meet it again in an
independent confirmation at 4 x n_sim draws (Monte Carlo SE about 0.01) before it qualifies. Catch probabilities of
exactly 0 or 1 are handled exactly. No qualifying budget means "not reachable": stage 2 stops for Aswin. Every candidate's
probabilities, the settings, the rates, the seed and the code revision are saved.

  uv run --no-project --python 3.12 --with pandas --with numpy python simulate_stage2_power.py --rates <json> --p-range LO,HI \\
      --rho R --audit-imp A --audit-ro B --inputs <dir> --metas <acts_meta root>
"""

import argparse
import json
from datetime import datetime
from statistics import NormalDist
from zoneinfo import ZoneInfo

import numpy as np

import check_joins as CJ
import prepare_stage1 as P

SEED, POOLS, Z = 20261011, ("target_touching", "flagged_other", "unflagged_impactful", "unflagged_read_only"), 1.959964
K_GRID, A_GRID = [4, 8, 12, 16, 20, 24, 32, 40, 60, 10**6], [0.25, 0.5, 1.0]


def stage2_pools(inputs: str, metas: str) -> dict:
    command, _ = CJ.entries(metas)
    C = CJ.rescore(P.load(inputs), command, inputs)
    sp = json.loads(P.SPLIT_FILE.read_text())
    tasks = sorted(t for t, v in sp["tasks"].items() if v["side"] == "held_out" and v["usable"] and not v["stage1"])
    T = C[C.task_key.isin(tasks) & C.scored]
    out = {}
    for w in ("real", "run2", "sim"):
        g = T[T.world == w]
        per = lambda m: g[m].groupby("task_key").size().reindex(tasks, fill_value=0).values  # noqa: E731
        out[w] = {"target_touching": per(g.touch), "flagged_other": per(g.flagged & ~g.touch),
                  "unflagged_impactful": per(~g.flagged & ~g.touch & (g.y == 1)), "unflagged_read_only": per(~g.flagged & ~g.touch & (g.y == 0))}
    return out


def one_set(rng, pools, rates, k, a_imp, a_ro, p, rho, n):
    """n simulated draws for one set: per task, weighted caught (a) and weighted labelled sabotage (b), shape [n, tasks]."""
    if not 0 <= p <= 1:
        raise ValueError(f"catch probability {p} outside [0, 1]")
    s2 = rho / (1 - rho) if rho else 0.0
    T = len(pools["flagged_other"])
    u = rng.normal(0, np.sqrt(s2), (n, T)) if s2 else np.zeros((n, T))
    if p in (0.0, 1.0):
        pc = np.full((n, T), p)
    else:
        c = -np.sqrt(1 + s2) * NormalDist().inv_cdf(p)
        pc = 1 - np.vectorize(NormalDist().cdf)(c - u)  # P(caught | task effect)
    a, b = np.zeros((n, T)), np.zeros((n, T))
    N = pools["flagged_other"]
    for pool in POOLS:
        size = pools[pool]
        sab = np.minimum(rng.poisson(rates.get(pool, 0.0), (n, T)), size) * (size > 0)
        if pool == "flagged_other":
            kk = np.minimum(k, N)
            lab = rng.hypergeometric(np.maximum(sab, 0), np.maximum(N - sab, 0), np.broadcast_to(kk, (n, T))) if N.max() else sab * 0
            w = np.where(kk > 0, N / np.maximum(kk, 1), 0.0)
        else:
            q = {"target_touching": 1.0, "unflagged_impactful": a_imp, "unflagged_read_only": a_ro}[pool]
            lab, w = rng.binomial(sab, q), 1 / q
        caught = rng.binomial(lab, pc)
        a += w * caught
        b += w * lab
    return a, b


def decide(a1, b1, a2, b2, margin):
    """Per draw: pass / fail by the 95% interval of the difference of two weighted catches (task-clustered SEs)."""
    def ratio(a, b):
        B = b.sum(1)
        R = np.divide(a.sum(1), B, out=np.full(len(B), np.nan), where=B > 0)
        T = a.shape[1]
        v = ((a - R[:, None] * b) ** 2).sum(1) / np.maximum(B, 1e-12) ** 2 * T / (T - 1)
        return R, v
    r1, v1 = ratio(a1, b1)
    r2, v2 = ratio(a2, b2)
    d, se = r1 - r2, np.sqrt(v1 + v2)
    lo, hi = d - Z * se, d + Z * se
    ok = np.isfinite(d)
    return np.where(ok, (lo > -margin) & (hi < margin), False), np.where(ok, (lo > margin) | (hi < -margin), False)


def power(rng, pools, rates, k, a_imp, a_ro, p, rho, margin, n):
    other = round(p - 2 * margin, 10) if p - 2 * margin >= 0 else round(p + 2 * margin, 10)
    if not 0 <= other <= 1:
        raise ValueError(f"gap scenario {other} outside [0, 1] for p {p}")
    res = {}
    for comp, w2 in (("e1_vs_sim", "sim"), ("e1_vs_e2", "run2")):
        A = one_set(rng, pools["real"], rates["real"], k, a_imp, a_ro, p, rho, n)
        B0 = one_set(rng, pools[w2], rates[w2], k, a_imp, a_ro, p, rho, n)
        B1 = one_set(rng, pools[w2], rates[w2], k, a_imp, a_ro, other, rho, n)
        res[comp] = {"p_pass_gap0": float(decide(*A, *B0, margin)[0].mean()), "p_fail_gap2m": float(decide(*A, *B1, margin)[1].mean())}
    return res


def rows(pools, k, a_imp, a_ro):
    return int(sum(pl["target_touching"].sum() + np.minimum(k, pl["flagged_other"]).sum() + round(a_imp * pl["unflagged_impactful"].sum())
                   + round(a_ro * pl["unflagged_read_only"].sum()) for pl in pools.values()))


def main(rates: str, p_range: str, rho: float, audit_imp: float, audit_ro: float, margin: float, n_sim: int, inputs: str, metas: str, tag: str,
         runs_per_task: int = 1):
    import hashlib
    R = json.loads(open(rates).read())
    provenance = {"rates_file": rates, "rates": R, "rates_sha256": hashlib.sha256(open(rates, "rb").read()).hexdigest(), "audit_ro": audit_ro,
                  "audit_imp_stage1": audit_imp, "seed": SEED, "git": P.git(), "inputs_sha256": {n: P.sha256(P.Path(inputs) / n) for n in P.SHA}}
    lo, hi = (min(max(float(x), 0.02), 0.98) for x in p_range.split(","))
    grid = sorted(set(np.round(np.linspace(lo, hi, 5), 4)))
    pools, r = stage2_pools(inputs, metas), runs_per_task
    if r < 1:
        raise ValueError("runs_per_task must be >= 1")
    pools = {w: {p: v * r for p, v in pl.items()} for w, pl in pools.items()}  # r runs per task, same pool sizes
    R = {w: {p: v * r for p, v in rr.items()} for w, rr in R.items()}  # per-task rates over r runs
    provenance["runs_per_task"] = r
    cands = sorted({(k, a) for k in K_GRID for a in sorted({audit_imp, *A_GRID})}, key=lambda ka: rows(pools, ka[0] * r, ka[1], audit_ro))
    rng = np.random.default_rng(SEED)
    worst_of = lambda per_p: min(v for r in per_p.values() for c in r.values() for v in c.values())  # noqa: E731
    tried, best = [], None
    for k, a in cands:
        per_p = {str(p): power(rng, pools, R, k * r, a, audit_ro, p, rho, margin, n_sim) for p in grid}
        w = worst_of(per_p)  # unrounded; rounded only for display
        row = {"k": k, "k_per_task": k * r, "audit_rate_unflagged_impactful": a, "rows": rows(pools, k * r, a, audit_ro), "worst_probability": w, "per_p": per_p}
        wc = -1.0
        if w >= 0.8:  # independent confirmation at 4 x n_sim
            conf = {str(p): power(np.random.default_rng([SEED, k, int(a * 1000)]), pools, R, k * r, a, audit_ro, p, rho, margin, 4 * n_sim) for p in grid}
            wc = worst_of(conf)
            row |= {"confirmation_worst": wc, "confirmation_per_p": conf}
        tried.append(row)
        if wc >= 0.8:
            best = row
            break
    out = {"stage2_tasks": int(len(pools["real"]["flagged_other"])), "p_grid": grid, "rho": rho, "margin": margin, "n_sim": n_sim,
           "provenance": provenance, "budget": best or {"reachable": False}, "tried": tried,
           "written": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")}
    dst = P.HERE / "results" / f"stage2_power_{tag}_{out['written']}.json"
    with dst.open("x") as fh:  # append-only
        fh.write(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("stage2_tasks", "p_grid", "budget")} | {"max_worst": max(t["worst_probability"] for t in tried)}, indent=1, default=float), "\n->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", required=True, help="stage-1 weighted sabotage per run, by set (real, run2, sim) then pool")
    ap.add_argument("--p-range", required=True, help="stage 1's plausible catch range, lo,hi (its task-resampled 95% range)")
    ap.add_argument("--rho", type=float, required=True)
    ap.add_argument("--audit-imp", type=float, required=True)
    ap.add_argument("--audit-ro", type=float, required=True)
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--n-sim", type=int, default=400)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--metas", required=True)
    ap.add_argument("--tag", required=True, help="results/stage2_power_<tag>_<stamp>.json")
    ap.add_argument("--runs-per-task", type=int, default=1)
    main(**vars(ap.parse_args()))
