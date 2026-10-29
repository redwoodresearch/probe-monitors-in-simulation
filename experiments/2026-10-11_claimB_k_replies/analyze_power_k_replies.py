"""Power and price for the k-simulated-replies test (Aswin's design, before 15:27 PT 10-11), $0, before any compute.
Inputs (keys and numbers only): step A's stored per-output scores at level 1 (`claimB_output_scores_A_2026-10-10_022646.json.gz`
in the main checkout's git-ignored data/, sha256 recorded), the pack's run metadata (`runs.L63.json`, Volume read), and the
H1 swap pilot's paired-to-natural noise ratio (h1_swap_pilot_2026-10-11_113444.json at tag archive/claimb-h1-swap-2026-10-11).

- Sites: real e1 test runs' outputs with at least 7 earlier kept outputs (so the last 8 replies can be simulated),
  matched by (step, action) to the same cell's fixed-sim output. d = s(fixed sim) - s(e1) per matched site is the "full
  gap" quantity at the same steps; G = mean d over sites (late sites, index >= 7), with main-task-clustered SE.
- Swap noise: a simulated-vs-real reply difference on the same history is assumed to have SD = ratio x SD(d), ratio from
  the pilot (0.84; 1.0 as a sensitivity); task clustering by the design effect estimated from d.
- Rule A ("the reply itself", Aswin): R1 = Delta1 / G; reads "the reply itself" if R1's 95% lower bound >= 0.5, "not the
  reply itself" if its upper bound < 0.5, else too few. Rule B ("history"): H = (Delta8 - Delta1) / G; "history" if A
  reads "not the reply itself" and H's lower bound > 0; "not history" if H's upper bound < 0.25; else too few.
- P(clear) by simulation (20,000 draws) over N sites, one site per run, for true R1 in {0.1, 0.3, 0.5, 0.7, 0.9} and H in
  {0.3, 0.5, 0.7}; the per-site Delta8 and Delta1 errors correlated rho in {0, 0.5}.
- Price per site: 15 simulator calls (k = 1, 2, 4, 8 as separate chains ending at the same step) at the Oct 7 test's
  measured $0.452 a call (2026-10-06_matched_step_pairs README: $90.48 for 200 calls), 1.5x for late steps as the upper
  case; 5 reads per site (four chains and the self-swap) at B-dev's $0.0257; fixed parts: 100 planted reads, 20 gate
  reads, scoring and start-up about $10.

  uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy python experiments/2026-10-11_claimB_k_replies/analyze_power_k_replies.py
"""

import gzip
import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal
import numpy as np

HERE = Path(__file__).parent
SCORES = Path.home() / "probe-shift-sims/experiments/2026-10-09_fixed_sim/data/claimB_output_scores_A_2026-10-10_022646.json.gz"
PACK = "claimB/level1/packed_39f842231a46"
PILOT = "refs/tags/archive/claimb-h1-swap-2026-10-11:experiments/2026-10-11_claimB_h1_text_vs_layout/results/h1_swap_pilot_2026-10-11_113444.json"
TEST = {"clinical_trial", "media_processing", "shared_dotfiles", "tiktok", "user_tracking"}
SIM, E1 = "sim-attack-fixed", "real-attack-e1"
PER_CALL, CALLS, READS, PER_READ, FIXED = 0.452, 15, 5, 0.0257, 120 * 0.0257 + 10.0


def main():
    raw = SCORES.read_bytes()
    rows = json.loads(gzip.decompress(raw))["levels"]["level1"]["rows"]
    vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
    runs_raw = b"".join(vol.read_file(f"{PACK}/runs.L63.json"))
    cell_of = {(r["arm"], r["key"]): r["cell"] for r in json.loads(runs_raw)}
    pilot_raw = subprocess.run(["git", "show", PILOT], capture_output=True, check=True, cwd=HERE).stdout
    ratio = json.loads(pilot_raw)["noise_ratio_sd_u_over_sd_d"]
    by = defaultdict(dict)
    for r in rows:
        if r["environment"] in TEST:
            by[(cell_of[(r["arm"], r["key"])], r["arm"])][(r["step"], r["action"])] = (r["score"], r["main_task"])
    sites, site_run = [], []  # (main_task, d) per e1 run's eligible matched site, and its run
    eligible_runs = 0
    for (cell, arm), outs in by.items():
        if arm != E1 or (cell, SIM) not in by:
            continue
        order = sorted(outs)
        late = [sa for i, sa in enumerate(order) if i >= 7 and sa in by[(cell, SIM)]]
        if late:
            eligible_runs += 1
        for sa in late:
            sites.append((outs[sa][1], by[(cell, SIM)][sa][0] - outs[sa][0]))
            site_run.append(cell)
    tasks = sorted({t for t, _ in sites})
    d = np.array([x for _, x in sites])
    G = float(d.mean())

    def clustered_se(pairs, mean):  # task-level sums of site residuals (the estimator weights every site equally)
        tot = defaultdict(float)
        for t, x in pairs:
            tot[t] += x - mean
        return float(np.sqrt(sum(v * v for v in tot.values())) / len(pairs))
    se_G = clustered_se(sites, G)
    # the planned sample: one seeded site per eligible e1 run; its SD and design effect drive the noise
    rs = np.random.default_rng(20261011)
    by_run = defaultdict(list)
    for (t, x), run in zip(sites, site_run):
        by_run[run].append((t, x))
    one = [v[rs.integers(len(v))] for _, v in sorted(by_run.items())]
    d1 = np.array([x for _, x in one])
    sd = float(d1.std(ddof=1))
    deff = max(1.0, (clustered_se(one, float(d1.mean())) / (sd / np.sqrt(len(one)))) ** 2)
    rng = np.random.default_rng(20261011)
    D = 20000
    out = {"label": "power and price for the k-simulated-replies test (plan input; $0)", "G_full_gap": G, "G_se_clustered": se_G,
           "sites_all_late": len(d), "main_tasks": len(tasks), "eligible_e1_runs": eligible_runs, "one_site_per_run_sd": sd,
           "one_site_per_run_mean": float(d1.mean()), "design_effect_one_site_per_run": deff, "pilot_noise_ratio": ratio, "power": {}, "price": {}}
    for rat in (ratio, 1.0):
        s_swap = rat * sd
        for n in (100, 200, 300, 400, eligible_runs):
            se = s_swap * np.sqrt(deff / n)
            g = G + se_G * rng.standard_normal(D)
            row = {}

            def r1_interval(delta1):
                r = delta1 / g
                rse = np.sqrt((se / g) ** 2 + (delta1 * se_G / g ** 2) ** 2)
                return r - 1.96 * rse, r + 1.96 * rse
            for r1 in (0.1, 0.3, 0.5, 0.7, 0.9):
                lo, hi = r1_interval(r1 * G + se * rng.standard_normal(D))
                row[f"R1={r1}"] = {"reply_itself": float((lo >= 0.5).mean()), "not_reply": float((hi < 0.5).mean()), "too_few": float(((lo < 0.5) & (hi >= 0.5)).mean())}
            for h in (0.3, 0.5, 0.7):
                for rho in (0.0, 0.5):
                    e1 = se * rng.standard_normal(D)
                    e8 = rho * e1 + np.sqrt(1 - rho ** 2) * se * rng.standard_normal(D)
                    dl1, dl8 = 0.3 * G + e1, (0.3 + h) * G + e8
                    _, hi1 = r1_interval(dl1)
                    hh = (dl8 - dl1) / g
                    hse = np.sqrt((se * np.sqrt(2 * (1 - rho)) / g) ** 2 + ((dl8 - dl1) * se_G / g ** 2) ** 2)
                    row[f"R1=0.3,H={h},rho={rho}"] = {"history": float(((hi1 < 0.5) & (hh - 1.96 * hse > 0)).mean()),
                                                      "not_history_if_H0": None}
                    if h == 0.3:
                        e8b = rho * e1 + np.sqrt(1 - rho ** 2) * se * rng.standard_normal(D)
                        h0 = ((0.3 * G + e8b) - dl1) / g
                        h0se = np.sqrt((se * np.sqrt(2 * (1 - rho)) / g) ** 2 + ((e8b - e1) * se_G / g ** 2) ** 2)
                        row[f"R1=0.3,H={h},rho={rho}"]["not_history_if_H0"] = float((h0 + 1.96 * h0se < 0.25).mean())
            out["power"][f"ratio={rat:.2f}|N={n}"] = row
    for n in (100, 200, 300, 400, eligible_runs):
        base = n * (CALLS * PER_CALL + READS * PER_READ) + FIXED
        late = n * (CALLS * PER_CALL * 1.5 + READS * PER_READ) + FIXED
        k1_only = n * (1 * PER_CALL + 2 * PER_READ) + FIXED
        out["price"][f"N={n}"] = {"all_k_raw": base, "all_k_30pct": base * 1.3, "all_k_late_upper_30pct": late * 1.3,
                                  "k1_only_raw": k1_only, "k1_only_30pct": k1_only * 1.3}
    out |= {"inputs": {"scores_sha256": hashlib.sha256(raw).hexdigest(), "runs_json_sha256": hashlib.sha256(runs_raw).hexdigest(),
                       "pilot_sha256": hashlib.sha256(pilot_raw).hexdigest()},
            "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"power_k_replies_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("G_full_gap", "G_se_clustered", "sites_all_late", "main_tasks", "eligible_e1_runs", "one_site_per_run_sd",
                                         "one_site_per_run_mean", "design_effect_one_site_per_run", "pilot_noise_ratio")}, indent=1))
    for k, v in out["power"].items():
        print(k, {kk: (round(vv["reply_itself"] + vv["not_reply"], 2) if "H=" not in kk else (round(vv["history"], 2), vv["not_history_if_H0"] and round(vv["not_history_if_H0"], 2))) for kk, vv in v.items()})
    for k, v in out["price"].items():
        print(k, {kk: round(vv) for kk, vv in v.items()})
    print("->", dst)


if __name__ == "__main__":
    main()
