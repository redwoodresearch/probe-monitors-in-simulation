"""Addendum 1 (two-direction swap on cached replies): scenario power, price and the step-fork mapping, $0, numbers only.
Committed before it runs; PLAN.md's addendum 1 states the design, rules and assumptions.

1. Mapping to Jeffrey's step-level 2AFC: on k = 1's 400 teacher-forced sites, the probe's catch accuracy (s(SIM) above
   s(SELF) scores 1, ties, i.e. identical text, score 1/2), all sites and differing sites (L63). A related measure, not
   numerically the judge's.
2. Scenario power per direction (F or N), not an established probability: the rule tests the contrast mean(D - 0.5 G),
   normal intervals, at z 1.96 and at Holm's first step for two directions (two-sided 0.025, z 2.24). Per-pair variance
   c^2 var(u) + (r - 0.5)^2 var(g) + 2 c (r - 0.5) cov(u, g), from k = 1's differing sites in score units (u = d - R g);
   c = 1 or 1.5 (the cached, audited bash pairs may be noisier); the contrast's main-task correlation assumed equal to
   u's. Mean G at 0.686 (H1's unaudited 257 test pairs, @ f2ba9ac3), 0.5 or 0.35; design effect 1 + icc (m_eff - 1),
   m_eff = pairs per main task (1.99) or that times the shape factor sum(m^2)/N / (N/tasks) of H1's 102 triples.
   Audited pair counts are scenarios (no audit has run).
3. Price: reads only (cached replies), target and yardstick survivors as separate scenarios, at k = 1's billed rate and
   at H1's planning rate.

  uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy python \\
      experiments/2026-10-11_claimB_swap_changed/analyze_POWER_stepfork.py

With --audit <stepfork_audit_*.json> (added after the audit ran, before this mode's result): the audited target and
yardstick counts and the survivors' sum(m^2)/N replace the scenarios, and the chance that the yardstick gate passes
(its 95% interval inside [-0.25, 0.25] at a true real-vs-real shift of 0, the C noise taken as u's) is added.
"""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime
from math import erf, sqrt
from pathlib import Path
from zoneinfo import ZoneInfo

import modal
import numpy as np

HERE = Path(__file__).parent
VOL = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
SCORES = ("claimB/k1_v1/scores_4be7958c7157.json", "b9e4eeac0d82ffef762fdf83d4dfbf7ff6242607b983abbfd11da03fd207a8af")
CELLS_VOL = ("claimB/h1_test_swap_v1/cells.json", "a2e8336724a7ffe9f6dc093c5acd988afdac0516a7384f2f9f5f0f1859a1316a")
H1CELLS = ("f2ba9ac3", "experiments/2026-10-11_claimB_h1_text_vs_layout/results/h1_test_swap_cells_2026-10-11_143409.json")
RATES = {"k1_billed": 12.2068 / 920, "h1_planning": 0.0257}
N_PAIRS, TRUE, G_SC, NOISE = [43, 64, 86, 129, 257], [0.2, 0.3, 0.5, 0.7, 0.8], [None, 0.5, 0.35], [1.0, 1.5]
FIXED_READS, STEPS_USD, AUDIT_USD = 100 + 2 * 20, 2.0, 0.1  # planted + gate per arm; H100 matching/scoring; CPU audit
PHI = lambda x: 0.5 * (1 + erf(x / sqrt(2)))  # noqa: E731


def read(path, sha):
    raw = b"".join(VOL.read_file(path))
    assert hashlib.sha256(raw).hexdigest() == sha, path
    return json.loads(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", type=Path, default=None)
    args = ap.parse_args()
    audit = json.loads(args.audit.read_text())["survivors"] if args.audit else None
    sites = read(*SCORES)["sites"]
    triples = read(*CELLS_VOL)["triples"]
    cells = json.loads(subprocess.run(["git", "show", f"{H1CELLS[0]}:{H1CELLS[1]}"], capture_output=True, check=True, cwd=HERE).stdout)["cells"]
    acc = lambda rows: float(np.mean([1.0 if r["63"]["sim"] > r["63"]["self"] else 0.5 if r["63"]["sim"] == r["63"]["self"] else 0.0 for r in rows]))  # noqa: E731
    diff = [r for r in sites if not r["sim_text_equals_self_text"]]
    out = {"label": "addendum 1: two-direction cached swap, scenario power, price, step-fork mapping ($0, numbers only)",
           "probe_catch_accuracy_k1": {"all_sites": acc(sites), "n_all": len(sites), "differing_sites": acc(diff), "n_differing": len(diff)}}
    t = np.array([r["main_task"] for r in diff])
    d = np.array([r["63"]["sim"] - r["63"]["self"] for r in diff])
    g = np.array([r["63"]["stored_fixed_sim"] - r["63"]["stored_e1"] for r in diff])
    R = d.sum() / g.sum()
    u = d - R * g
    tk = np.unique(t, return_inverse=True)[1]
    k, n = tk.max() + 1, len(u)
    m_, s_ = np.bincount(tk), np.bincount(tk, u)
    msb = (m_ * (s_ / m_ - u.mean()) ** 2).sum() / (k - 1)
    msw = ((u - (s_ / m_)[tk]) ** 2).sum() / (n - k)
    n0 = (n - (m_ ** 2).sum() / n) / (k - 1)
    icc, var_u, var_g = max(0.0, (msb - msw) / (msb + (n0 - 1) * msw)), float(u.var(ddof=1)), float(g.var(ddof=1))
    cov_ug = float(np.cov(u, g, ddof=1)[0, 1])
    pairs, tasks, G0 = cells["all_sim_e1_pairs"]["pairs"], cells["all_sim_e1_pairs"]["main_tasks"], cells["all_sim_e1_pairs"]["mean"]
    tm = Counter(x["main_task"] for x in triples)
    shape = (sum(v * v for v in tm.values()) / len(triples)) / (len(triples) / len(tm))
    out["noise"] = {"k1_differing_sites": n, "resid_sd_score_units": sqrt(var_u), "g_sd": sqrt(var_g), "cov_u_g": cov_ug, "icc_main_task": icc,
                    "G_identical_call_pairs": G0, "pairs_on_test": pairs, "main_tasks": tasks, "triples_shape_factor": shape}
    power = {}
    n_list = [audit["sim|e1"]["pairs"]] if audit else N_PAIRS
    for npair in n_list:
        for G in G_SC:
            G = G or G0
            for c in NOISE:
                for m_eff in ((audit["sim|e1"]["pairs_per_task_sum_m2_over_n"],) if audit else (pairs / tasks, shape * pairs / tasks)):
                    row = {}
                    for r in TRUE:
                        v = c * c * var_u + (r - 0.5) ** 2 * var_g + 2 * c * (r - 0.5) * cov_ug
                        se = sqrt(v * (1 + icc * (m_eff - 1)) / npair)
                        z = abs(r - 0.5) * G / se
                        for zc, zname in ((1.96, "z1.96"), (2.24, "holm_first_z2.24")):
                            row[f"true={r}|{zname}"] = PHI(z - zc) + PHI(-z - zc)
                    power[f"pairs={npair}|G={G:.3f}|noise_x={c}|m_eff={m_eff:.2f}"] = row
    out["scenario_power_per_direction"] = power
    e2 = cells["pair_counts"]["e2|e1"]
    price = {}
    if audit:
        ny, my = audit["e2|e1"]["pairs"], audit["e2|e1"]["pairs_per_task_sum_m2_over_n"] or 1.0
        gate = {}
        for G in (G0, 0.5, 0.35):
            se_y = sqrt(var_u * (1 + icc * (my - 1)) / max(ny, 1)) / G
            half = 0.25 - 1.96 * se_y
            gate[f"G={G:.3f}"] = {"se_Y": se_y, "p_inside_band_at_true_0": (2 * PHI(half / se_y) - 1) if half > 0 else 0.0}
        out["yardstick_gate"] = {"pairs": ny, "m_eff": my, **gate}
    for npair in n_list:
        for y_name, y in ((("audited", audit["e2|e1"]["pairs"]),) if audit else (("same_share", round(e2 * npair / pairs)), ("all", e2))):
            reads = 2 * npair + 2 * y + FIXED_READS
            for name, rate in RATES.items():
                rawp = reads * rate + STEPS_USD + AUDIT_USD
                price[f"pairs={npair}|yardstick={y_name}|{name}"] = {"reads": reads, "raw_usd": rawp, "with_30pct_usd": 1.3 * rawp}
    out["price"] = price
    out |= {"inputs": {SCORES[0]: SCORES[1], CELLS_VOL[0]: CELLS_VOL[1], "h1_cells_result": f"{H1CELLS[0]}:{H1CELLS[1]}",
                       "audit": (str(args.audit), hashlib.sha256(args.audit.read_bytes()).hexdigest()) if args.audit else None},
            "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"power_stepfork_{'audited_' if audit else ''}{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out))
    print(json.dumps({k: out[k] for k in ("probe_catch_accuracy_k1", "noise", "yardstick_gate") if k in out}, indent=1))
    for key, row in power.items():
        if audit or "noise_x=1.0" in key:
            print(key, {a: round(b, 2) for a, b in row.items()})
    for key, row in price.items():
        print(key, {a: round(b, 1) for a, b in row.items()})
    print("->", dst)


if __name__ == "__main__":
    main()
