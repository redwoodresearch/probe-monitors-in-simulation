"""Rough normal-approximation power for measure 2 of PLAN.md (this folder). Local, $0.
  uv run --no-project --with numpy python experiments/2026-10-11_q2_behaviour_redesign/analyze_power_rough.py
The tasks-for-80% line uses P(|Z| < 1.2816) = 0.80 (central 80%), checked by simulation: 368 tasks at 95% and 434 at
97.5% each give 0.801.
"""
import numpy as np
from math import sqrt
g = np.random.default_rng(20261011); n = 1_000_000
for N in (190,):
    for rho_task in (0.3,):
        se = sqrt(0.25 * 2 * (1 - rho_task) / N)
        for M in (0.10, 0.15):
            for z, lab in ((1.96, "95%"), (2.24, "Holm first step, 97.5%")):
                z1, z2 = g.standard_normal(n), g.standard_normal(n); r = 0.5  # the two gaps share real run 1
                x, y = se * z1, se * (r * z1 + sqrt(1 - r * r) * z2)
                p_rs = np.mean(np.abs(x) < M - z * se); p_both = np.mean((np.abs(x) < M - z * se) & (np.abs(y) < M - z * se))
                print(f"N={N} task corr={rho_task} SE={se:.4f} M=±{M:.2f} {lab}: P(real-vs-sim inside)={p_rs:.2f} P(both inside)={p_both:.2f}")
        need = lambda M, z: (sqrt(0.25 * 2 * (1 - rho_task)) * (z + 1.2816) / M) ** 2
        print(f"tasks for 80% on the real-vs-sim interval alone at ±0.10: {need(0.10,1.96):.0f} (95%), {need(0.10,2.24):.0f} (97.5%)")
