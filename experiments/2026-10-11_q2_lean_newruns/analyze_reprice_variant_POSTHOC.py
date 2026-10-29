"""Re-pricing variants (post hoc, for Codex's two P1s on #795; no verdict, and the decision is made). The "unpaired"
scenario of analyze_reprice_POSTHOC.py had two simplifications:
1. Its real rate pooled all of an unpaired task's overlap800 runs, so tasks with more runs weighed more. A projected
   task adds exactly two runs, so an equal-task mean is the matching rate.
2. It thinned all k pseudo-tasks at the cohort rate, which replaced the 50 already-observed stage-1 pairs with the
   cohort's higher rate. Keeping them means pseudo-tasks 1–50 take stage 1's rates (the "stage 1" scenario's factors)
   and only the rest take the cohort's.
This reruns the same machinery at k = 230 and 274:
- V0: the original (must reproduce 0.641 / 0.696 exactly);
- V1: equal-task real rate;
- V2: observed 50 kept;
- V3: both changes.
Inputs are those of analyze_reprice_POSTHOC.py, and its own result supplies the stage-1 factors. No sealed runs are
read. Local, $0.

Revised after Codex on #798: with the observed 50 kept, the continuation factor comes from the remaining task positions
(51-274 of the seeded order), not all 274. The re-pricing result it reads is pinned by sha256 and guarded.

Revised after Codex's P2 on #798: a factor above 1 (only V2's real transfer continuation, 1.059) was clamped to 1,
which understated those real positives. Now a task with f > 1 keeps all its positives and adds Binomial(p, f - 1)
extra ones, each caught at that task's caught share, so E[positives] = f p. Factors at or below 1 take the same draws
as before, so V0, V1 and V3 must reproduce the previous result (reprice_variant_POSTHOC_2026-10-11_210204.json) exactly.

  uv run --no-project --with numpy python experiments/2026-10-11_q2_lean_newruns/analyze_reprice_variant_POSTHOC.py \\
      <per_task_catch_196.json> <recal_2026-10-10_225753.json> <batch_spec_2026-10-11_151207.json>
"""
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("lean", HERE / "analyze_lean_sizing.py"); L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
REPRICE = HERE / "results/reprice_POSTHOC_2026-10-11_191730.json"
REPRICE_SHA = "34c29b72eb60518a2acb3f3225442ca10162160d3f37ffb1276e5de8218f6975"
PREV, PREV_SHA = HERE / "results/reprice_variant_POSTHOC_2026-10-11_210204.json", "ef852bd79e23c74ef501c1f2c99bd70b789b5e52ef6be1c18cf282592ccfd6a2"
SPEC_SHA, ORDER_SHA = "ce22799937e35d7db7634c618d0e1ab2a0d72963982797761b958cf7eacc6c4a", "b804d4d44f1ea8d716ab30b0bb099e7d9e59ef908d427de013cf3d36b0d00c8b"
KS, TARGET, OBSERVED = (230, 274), "1%", 50
MONS = ("transfer", "destructive_raw")


def thin(c, p, f, g):
    """analyze_reprice_POSTHOC.thin with a per-task f (the same draws when f is constant and below 1). Where f > 1, all
    positives are kept and Binomial(p, f - 1) extra ones are added, each caught at the task's caught share."""
    kept = g.binomial(p.astype(int), np.minimum(f, 1.0))
    cc = np.array([g.hypergeometric(int(ci), int(pi - ci), int(ki)) if ki else 0 for ci, pi, ki in zip(c, p, kept)], float)
    if (f > 2).any():
        raise SystemExit("a factor above 2 can't be realized this way")
    if (f > 1).any():
        extra = g.binomial(p.astype(int), np.where(f > 1, f - 1, 0.0))
        cc = cc + g.binomial(extra, np.divide(c, p, out=np.zeros(len(c)), where=p > 0))
        kept = kept + extra
    return cc, kept.astype(float)


def main(m1_p, recal_p, spec_p):
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    me = Path(__file__).resolve()
    for f in (me, HERE / "analyze_lean_sizing.py", REPRICE):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    got = {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (m1_p, recal_p)}
    if got != L.SHA:
        raise SystemExit("inputs differ from the sizing's pins")
    if hashlib.sha256(REPRICE.read_bytes()).hexdigest() != REPRICE_SHA or hashlib.sha256(Path(spec_p).read_bytes()).hexdigest() != SPEC_SHA:
        raise SystemExit("the re-pricing result or the batch spec differs from its pinned sha256")
    order = json.loads(Path(spec_p).read_text())["task order"]
    if hashlib.sha256("\n".join(order).encode()).hexdigest() != ORDER_SHA:
        raise SystemExit("task order differs")
    rp = json.loads(REPRICE.read_text())
    full = json.loads(Path(recal_p).read_text()); rows = full["rows"]
    sim_tasks = {r["task"] for r in rows if r["world"] == 1}
    runs = defaultdict(set)
    for r in rows:
        if r["world"] == 0:
            runs[r["task"]].add(r["run"])
    first2 = {t: sorted(v)[:2] for t, v in runs.items() if len(v) >= 2}
    same = sorted(t for t, (a, b) in first2.items() if "|overlap800_" in a and "|overlap800_" in b)
    unpaired = sorted(sim_tasks - set(same))
    o800 = {t: sorted(x for x in runs[t] if "|overlap800_" in x) for t in unpaired}
    pos_run = {m: Counter() for m in MONS}
    for r in rows:
        for m in MONS:
            if r[m] == 1 and r["world"] == 0:
                pos_run[m][r["run"]] += 1
    rate163 = {m: rp["task mix"][m]["real positives a run"]["163 first two"] for m in MONS}  # two runs each: pooled = equal-task
    eq = {m: float(np.mean([np.mean([pos_run[m][x] for x in o800[t]]) for t in unpaired])) for m in MONS}
    f_un, f_s1 = rp["thinning f"]["unpaired"], rp["thinning f"]["stage 1"]
    f_eq = {m: {"real": eq[m] / rate163[m], "simulated": f_un[m]["simulated"]} for m in MONS}
    # the continuation (positions 51-274 of the seeded order), for the scenarios that keep the observed 50
    rest = [t for t in order[OBSERVED:] if t in o800]
    pos_sim = {m: Counter() for m in MONS}
    for r in rows:
        for m in MONS:
            if r[m] == 1 and r["world"] == 1:
                pos_sim[m][r["task"]] += 1
    sim163 = {m: rp["task mix"][m]["simulated positives a task"]["163"] for m in MONS}
    sim_rest = {m: sum(pos_sim[m][t] for t in rest) / len(rest) for m in MONS}
    pooled_rest = {m: sum(pos_run[m][x] for t in rest for x in o800[t]) / sum(len(o800[t]) for t in rest) for m in MONS}
    eq_rest = {m: float(np.mean([np.mean([pos_run[m][x] for x in o800[t]]) for t in rest])) for m in MONS}
    f_rest_pooled = {m: {"real": pooled_rest[m] / rate163[m], "simulated": sim_rest[m] / sim163[m]} for m in MONS}
    f_rest_eq = {m: {"real": eq_rest[m] / rate163[m], "simulated": sim_rest[m] / sim163[m]} for m in MONS}
    variants = {"V0 original (pooled rate, all k)": (f_un, False), "V1 equal-task real rate, all k": (f_eq, False),
                "V2 pooled rate, observed 50 kept (continuation from positions 51-274)": (f_rest_pooled, True),
                "V3 equal-task real rate, observed 50 kept (continuation from positions 51-274)": (f_rest_eq, True)}
    q = L.TARGETS[TARGET]
    m1 = L.m1_counts(m1_p, f"caught_{TARGET[:-1]}pct"); T1 = len(m1["real_run1"][0])
    W1 = np.vstack([np.ones(T1), np.random.default_rng([L.SEED, 1]).poisson(1.0, (L.DRAWS, T1))])
    g1_rs, g1_rr = L.gap(*m1["real_run1"], *m1["simulated"], W1), L.gap(*m1["real_run1"], *m1["real_run2"], W1)
    sols = [L.sol_counts(full, n, q) for n in MONS]; T = len(sols[0][0])
    W = np.vstack([np.ones(T + L.MAXK), np.random.default_rng([L.SEED, 23]).poisson(1.0, (L.DRAWS, T + L.MAXK))])
    out = {"label": "POST HOC (Codex P1s on #795; no verdict; the decision is made)", "git": head, "inputs sha256": got,
           "re-pricing result": REPRICE.name, "unpaired real rate, pooled vs equal-task": {m: {"pooled": rp["task mix"][m]["real positives a run"]["274 unpaired, overlap800 runs"], "equal-task": eq[m]} for m in MONS},
           "factors": {"pooled (as run)": f_un, "equal-task": f_eq, "stage 1 (for the observed 50)": f_s1,
                       "continuation, positions 51-274: pooled": f_rest_pooled, "continuation, positions 51-274: equal-task": f_rest_eq},
           "continuation tasks": len(rest), "re-pricing result sha256": REPRICE_SHA, "batch spec sha256": SPEC_SHA, "P(clear) at 1%": {}}
    for vname, (fa, keep) in variants.items():
        res = {}
        for k in KS:
            se_rs, se_rr, rhos, sets0 = [], [], [], None
            for j in range(L.SETS):
                don = np.random.default_rng([L.SEED, 7, k, j]).integers(0, T, k)
                gt = np.random.default_rng([L.SEED, 9, k, j])
                aug = []
                for name, (_, c) in zip(MONS, sols):
                    a_ = {}
                    for arm in L.ARMS:
                        side = "simulated" if arm == "simulated" else "real"
                        fv = np.full(k, fa[name][side])
                        if keep:
                            fv[:min(OBSERVED, k)] = f_s1[name][side]
                        cc, pp = thin(c[arm][0][don], c[arm][1][don], fv, gt)
                        a_[arm] = (np.concatenate([c[arm][0], cc]), np.concatenate([c[arm][1], pp]))
                    aug.append(a_)
                Wk = W[:, :T + k]
                G_rs = (g1_rs + sum(L.gap(*c["real_run1"], *c["simulated"], Wk) for c in aug)) / 3
                G_rr = (g1_rr + sum(L.gap(*c["real_run1"], *c["real_run2"], Wk) for c in aug)) / 3
                se_rs.append(G_rs[1:].var()); se_rr.append(G_rr[1:].var()); rhos.append(np.corrcoef(G_rs[1:], G_rr[1:])[0, 1])
                if j == 0:
                    sets0 = (aug, Wk, G_rs, G_rr)
            S_rs, S_rr, rho = float(np.sqrt(np.mean(se_rs))), float(np.sqrt(np.mean(se_rr))), float(np.mean(rhos))
            gz = np.random.default_rng(L.SEED); z1, z2 = gz.standard_normal(L.MC), gz.standard_normal(L.MC)
            x, yv = S_rs * z1, S_rr * (rho * z1 + np.sqrt(1 - rho ** 2) * z2)
            p_both = float(((np.abs(x) < L.M - 1.96 * S_rs) & (np.abs(yv) < L.M - 1.96 * S_rr)).mean())
            aug, Wk, G_rs, G_rr = sets0
            pd = {}
            for arm, unpl in (("simulated", G_rs), ("real_run2", G_rr)):
                hits = [L.detected(float(Gp[0]), tuple(L.ci(Gp[1:])), float(unpl[0]), L.PLANT) for Gp in
                        (sum(L.gap(*c["real_run1"], *c[arm], W_) for c, W_ in mons) / 3 for mons in L.plant([(m1, W1)] + [(c, Wk) for c in aug], arm, L.COPIES))]
                pd[arm] = float(np.mean(hits))
            res[str(k)] = {"SE G_RS": S_rs, "SE G_RR": S_rr, "P(both pass)": p_both, "P(clear)": p_both * L.p_testable(pd["simulated"]) * L.p_testable(pd["real_run2"])}
            print(vname, k, round(res[str(k)]["P(clear)"], 3), flush=True)
        out["P(clear) at 1%"][vname] = res
    for k in KS:
        if abs(out["P(clear) at 1%"]["V0 original (pooled rate, all k)"][str(k)]["P(clear)"] - rp["pooled lean, re-priced"]["unpaired"][str(k)]["P(clear)"]) > 1e-12:
            raise SystemExit(f"V0 doesn't reproduce the as-run unpaired P(clear) at k = {k}")
    out["V0 reproduces the as-run unpaired row"] = True
    if hashlib.sha256(PREV.read_bytes()).hexdigest() != PREV_SHA:
        raise SystemExit("the previous variant result differs from its pinned sha256")
    prev = json.loads(PREV.read_text())["P(clear) at 1%"]
    for vname in out["P(clear) at 1%"]:
        if not vname.startswith("V2") and out["P(clear) at 1%"][vname] != prev[vname]:
            raise SystemExit(f"{vname} doesn't reproduce the previous result (only V2 has a factor above 1)")
    out["V0, V1 and V3 reproduce the previous result exactly"] = True
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"reprice_variant_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("inputs sha256",)}, indent=1)); print(dest)


if __name__ == "__main__":
    main(*sys.argv[1:4])
