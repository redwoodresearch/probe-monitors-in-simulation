"""Price and power for the cheaper history stage (PLAN.md amendment 2, committed before this script), $0, numbers only.
Inputs: the k = 1 stage's stored per-site scores (claimB/k1_v1/scores_<tag>.json), its simulator ledger (cost per site;
<cache>/sim_results.jsonl, cost fields only) and the records manifest (each site's history length k).

Power: H = (mean Δ8 − mean Δ1) / mean G at the option's sites, "grows" if the 95% lower bound > 0, "doesn't grow" if
the upper bound < 0.25. Per replicate, the option's real sites (subsampled to N; above the pool, the whole pool plus pool sites drawn again in their own main tasks, since the test set's tasks are fixed) keep
their stored G; Δ8 − Δ1 is drawn around H_true x mean G with SD sqrt(s8² + s1² − 2ρ s8 s1), s8 = c x σ1
(σ1 = per-site SD of Δ1 on the non-identical sites, s1 = SD of Δ1 over all 400; on option (b) Δ1 = 0 and the SD is s8), plus a main-task effect at the ICC
estimated from Δ1. Intervals: the main-task bootstrap over the replicate's own sites (numerator and G together; 1,000 draws
per replicate here, 2,000 in the real analysis). 500 replicates per cell (Monte Carlo SE about 1.8 points at 80%). c in {1, 1.5, 2}, ρ in {0, 0.5}, H_true in
{0 ("doesn't grow" power), 0.25, 0.5, 0.75 ("grows" power)}.

Price: simulator calls from the k = 1 ledger's cost vs history length (least squares), the k = 8 chain's 8 calls at
history lengths k − 7 … k; reads at the k = 1 stage's measured Modal cost per read ($13.65 / 920, start-up and scorer
included); 120 planted and gate reads; sizes beyond the pool add new k = 1 sites (1 call and 2 reads each; on option (b)
one identical site per 1 / 0.495 new sites). Raw and +30%.

  uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy python \\
      experiments/2026-10-11_claimB_k_replies/analyze_POWER_k8.py --tag 4be7958c7157 --cache <dir> --records results/k1_records_<stamp>.json
"""

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal
import numpy as np

HERE = Path(__file__).parent
OUT, M_MARGIN, REPS, BOOT, SEED = "claimB/k1_v1", 0.25, 500, 1000, 20261011
MODAL_PER_READ, FIXED_READS, ELIGIBLE_RUNS = 13.65 / 920, 120, 452  # eligible e1 test runs (k1_sites result)


def cell_power(key, args):
    n, pool, G, task, sd, gbar, icc = args
    rng = np.random.default_rng([SEED, int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)])
    cell = {}
    for h in (0.0, 0.25, 0.5, 0.75):
        grows = nogrow = 0
        for _ in range(REPS):
            if n <= len(pool):
                chosen = list(rng.choice(pool, n, replace=False))
            else:  # beyond the pool: the test set's main tasks are fixed, so extra sites are more sites in the
                # same tasks: the whole pool plus pool sites drawn again (same task, own site noise, that site's G)
                chosen = list(pool) + list(rng.choice(pool, n - len(pool), replace=True))
            tk = sorted({task[s] for s in chosen})
            ti = {t: i for i, t in enumerate(tk)}
            idx = np.array([ti[task[s]] for s in chosen])
            u = rng.standard_normal(len(tk))[idx]
            noise = sd * (np.sqrt(icc) * u + np.sqrt(1 - icc) * rng.standard_normal(len(chosen)))
            dd = h * gbar + noise  # Δ8 − Δ1 per site (on (b): Δ8, since Δ1 = 0 there)
            gg = np.array([G[s] for s in chosen])
            tsum_d = np.bincount(idx, dd, len(tk))
            tsum_g = np.bincount(idx, gg, len(tk))
            W = rng.multinomial(len(tk), np.full(len(tk), 1 / len(tk)), size=BOOT)  # task bootstrap counts
            num, den = W @ tsum_d, W @ tsum_g
            if (den <= 0).mean() > 0.025:  # G not clearly positive: too few, as in the k = 1 rule
                continue
            lo, hi = np.percentile(num[den > 0] / den[den > 0], [2.5, 97.5])
            grows += lo > 0
            nogrow += hi < M_MARGIN
        cell[f"H={h}"] = {"grows": grows / REPS, "doesnt_grow": nogrow / REPS}
    return cell


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--records", type=Path, required=True)
    a = p.parse_args()
    vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
    raw = b"".join(vol.read_file(f"{OUT}/scores_{a.tag}.json"))
    sites = json.loads(raw)["sites"]
    ledger_raw = (a.cache / "sim_results.jsonl").read_bytes()
    cost = {json.loads(l)["site"]: json.loads(l)["cost_usd"] for l in ledger_raw.decode().splitlines()}
    man = {m["site"]: m for m in json.loads((HERE / a.records).read_text())["manifest"]}
    # price model: cost per call vs history length k
    ks = np.array([man[s]["k"] for s in sorted(cost)], float)
    cs = np.array([cost[s] for s in sorted(cost)], float)
    b1, b0 = np.polyfit(ks, cs, 1)
    chain = lambda k: float(sum(b0 + b1 * max(k - 7 + j, 0) for j in range(8)))  # noqa: E731

    d1 = {r["site"]: r["63"]["sim"] - r["63"]["self"] for r in sites}
    G = {r["site"]: r["63"]["stored_fixed_sim"] - r["63"]["stored_e1"] for r in sites}
    task = {r["site"]: r["main_task"] for r in sites}
    ident = [r["site"] for r in sites if r["sim_text_equals_self_text"]]
    nonid = [r["site"] for r in sites if not r["sim_text_equals_self_text"]]
    sigma1 = float(np.std([d1[s] for s in nonid], ddof=1))
    s1_all = float(np.std(list(d1.values()), ddof=1))
    # ICC of Δ1 by main task (one-way ANOVA estimator on the non-identical sites)
    by = defaultdict(list)
    for s in nonid:
        by[task[s]].append(d1[s])
    grand = np.mean([d1[s] for s in nonid])
    k_ = len(by)
    n_ = len(nonid)
    msb = sum(len(v) * (np.mean(v) - grand) ** 2 for v in by.values()) / (k_ - 1)
    msw = sum(sum((x - np.mean(v)) ** 2 for x in v) for v in by.values()) / (n_ - k_)
    n0 = (n_ - sum(len(v) ** 2 for v in by.values()) / n_) / (k_ - 1)
    icc = float(max(0.0, (msb - msw) / (msb + (n0 - 1) * msw)))
    jobs = {}
    out = {"label": "price and power for the history stage (amendment 2; $0)", "margin_m": M_MARGIN, "sigma1_nonidentical": sigma1, "sd_delta1_all_sites": s1_all, "icc_delta1": icc,
           "price_model": {"usd_per_call_intercept": float(b0), "usd_per_call_per_history_action": float(b1),
                           "k1_mean_usd_per_call": float(cs.mean()), "modal_usd_per_read": MODAL_PER_READ},
           "power": {}, "price": {}}
    pools = {"a": sorted(d1), "b": ident}
    sizes = {"a": [100, 200, 300, 400, 600, 800, 1200], "b": [100, 150, 198, 300, 400, 600, 800]}
    for opt, pool in pools.items():
        gbar = float(np.mean([G[s] for s in pool]))
        for n in sizes[opt]:
            for c in (1.0, 1.5, 2.0):
                for rho in ((0.0, 0.5) if opt == "a" else (0.0,)):
                    s8 = c * sigma1  # SD(Δ8) = c x the non-identical sites' SD of Δ1; Δ1 on (a) keeps its all-site SD
                    sd = s8 if opt == "b" else np.sqrt(s8 * s8 + s1_all * s1_all - 2 * rho * s8 * s1_all)
                    jobs[f"{opt}|N={n}|c={c}|rho={rho}"] = (n, pool, G, task, float(sd), gbar, icc)
            # price for this option and size
            base = pool
            kmean = float(np.mean([man[s]["k"] for s in d1]))  # new k = 1 sites are screened from all candidates
            per_site_k8 = float(np.mean([chain(man[s]["k"]) for s in base]))
            extra = max(0, n - len(pool))
            new_sites = extra if opt == "a" else int(np.ceil(extra / 0.495))
            sim = n * per_site_k8 + new_sites * (b0 + b1 * kmean)  # k = 8 chains at n sites + one k = 1 call per new site
            reads = (n + FIXED_READS + 2 * new_sites) * MODAL_PER_READ
            raw_ = sim + reads
            out["price"][f"{opt}|N={n}"] = {"k8_chain_usd_per_site": per_site_k8, "new_k1_sites": new_sites, "simulator_usd": sim,
                                             "modal_usd": reads, "raw_usd": raw_, "with_30pct_usd": raw_ * 1.3,
                                             "needs_second_site_per_run": n > ELIGIBLE_RUNS if opt == "a" else len(pool) + 0.495 * (ELIGIBLE_RUNS - 400) < n}
    with ProcessPoolExecutor() as ex:  # every core; each cell seeded from its own key
        for key, cell in zip(jobs, ex.map(cell_power, jobs, jobs.values())):
            out["power"][key] = cell
    # (c): the smallest size per option, noise setting and "grows" effect where both powers reach 80%
    out["smallest_80"] = {"_note": "smallest TESTED size; the two labels can both fire on an interval inside (0, 0.25), as amendment 2 reads"}
    for key in {k.split("|N=")[0] + "|" + k.split("|", 2)[2] for k in out["power"]}:
        opt, rest = key.split("|", 1)
        for h in ("H=0.25", "H=0.5", "H=0.75"):
            ok = [n for n in sizes[opt] if out["power"][f"{opt}|N={n}|{rest}"]["H=0.0"]["doesnt_grow"] >= 0.8
                  and out["power"][f"{opt}|N={n}|{rest}"][h]["grows"] >= 0.8]
            out["smallest_80"][f"{key}|grows at {h}"] = min(ok) if ok else None  # smallest TESTED size; None = none tested
    out |= {"inputs": {"scores_sha256": hashlib.sha256(raw).hexdigest(), "ledger_sha256": hashlib.sha256(ledger_raw).hexdigest()},
            "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"power_k8_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("sigma1_nonidentical", "sd_delta1_all_sites", "icc_delta1", "price_model")}))
    for k, v in out["power"].items():
        print(k, {h: (x["grows"], x["doesnt_grow"]) for h, x in v.items()})
    for k, v in sorted(out["smallest_80"].items()):
        print("smallest_80", k, v)
    for k, v in out["price"].items():
        print(k, {kk: round(vv, 1) if isinstance(vv, float) else vv for kk, vv in v.items()})
    print("->", dst)


if __name__ == "__main__":
    main()
