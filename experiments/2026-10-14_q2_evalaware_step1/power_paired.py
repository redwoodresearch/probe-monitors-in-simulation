"""Power of the registered control-gated reading (RULE.md rev 5) to call "sim higher" when the simulated arm's marker
rate is twice the real arms' (e.g. 2% -> 4%), by Monte Carlo on Modal CPU. Model (an approximation, stated in the
output): each task's three runs (sim, run1, run2) are independent Bernoulli draws, the sim run at rate ratio*p and both
real runs at rate p, with the same rates on every task. Each simulated dataset is read with the RULE's rule: >= 10
informative tasks (d != 0), p_E < 0.05 with E > 0, and p_C >= 0.05. p_E and p_C are computed EXACTLY (the full
within-task permutation distribution, by convolution; the registered 20,000-draw test approximates it). Null rows
(ratio 1) check the rule's false-positive rate. A two-proportion z-test (sim n vs pooled real 2n, two-sided 5%) is
reported beside it as an analytic cross-check.
  uv run --no-project --python 3.12 --with modal==1.6.1 modal run --detach <this file>
"""
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-q2-evalaware_power_kill-2026-11-14"
SEED, R, ALPHA, MIN_INFORMATIVE = 20261014, 20000, 0.05, 10
# (n tasks, real rate p, ratio): the asked rows (n = 190; 1% and 2%), nulls, the honest pass's n, and n for 80% power
SCENARIOS = ([(190, 0.02, 2.0), (190, 0.01, 2.0), (190, 0.02, 1.0), (190, 0.01, 1.0), (149, 0.02, 2.0)]
             + [(n, 0.02, 2.0) for n in (400, 800, 1600, 3200)])
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.3.3")
app = modal.App(APP)


@app.function(image=image, cpu=(2, 2), memory=(2048, 2048), timeout=600, retries=0, startup_timeout=180,
              scaledown_window=2)
def power() -> list:
    import functools
    import math

    import numpy as np
    rng = np.random.default_rng(SEED)
    # per informative task, 2d* = 2*sim - run1 - run2 over the 6 label permutations, in half-units shifted by +2:
    one_pos = np.array([0, 2, 0, 0, 1]) / 3  # one positive: -1 w.p. 2/3, +2 w.p. 1/3
    two_pos = np.array([1, 0, 0, 2, 0]) / 3  # two positives: -2 w.p. 1/3, +1 w.p. 2/3
    cache = {}

    def p_exact(n1, n2, two_e):
        if (n1, n2) not in cache:
            dist = np.array([1.0])
            for _ in range(n1):
                dist = np.convolve(dist, one_pos)
            for _ in range(n2):
                dist = np.convolve(dist, two_pos)
            cache[(n1, n2)] = dist  # support: -2(n1+n2) .. +2(n1+n2) in half-units
        dist = cache[(n1, n2)]
        support = np.arange(len(dist)) - 2 * (n1 + n2)
        return float(dist[np.abs(support) >= abs(two_e) - 1e-9].sum())

    @functools.lru_cache(maxsize=None)
    def p_sign(m, c):  # exact sign-flip test on m nonzero c's
        k = np.arange(m + 1)
        probs = np.array([math.comb(m, int(i)) for i in k], float) / 2.0 ** m
        return float(probs[np.abs(2 * k - m) >= abs(c)].sum())

    def z_power(n, p1, p2):  # two-proportion z-test, sim (n) vs pooled real (2n), two-sided ALPHA
        pb = (n * p1 + 2 * n * p2) / (3 * n)
        se0, se1 = math.sqrt(pb * (1 - pb) * (1 / n + 1 / (2 * n))), math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / (2 * n))
        phi = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))  # noqa: E731
        delta = abs(p1 - p2)  # both rejection tails
        return phi((delta - 1.959964 * se0) / se1) + phi((-delta - 1.959964 * se0) / se1) if p1 != p2 else ALPHA

    rows = []
    for n, p, ratio in SCENARIOS:
        parts = []
        for _ in range(R // 1000):  # 1,000 datasets at a time keeps memory under the 2 GiB limit
            V = (rng.random((1000, n, 3)) < np.array([ratio * p, p, p])).astype(np.int64)
            two_d, npos, c = 2 * V[:, :, 0] - V[:, :, 1] - V[:, :, 2], V.sum(axis=2), V[:, :, 2] - V[:, :, 1]
            parts.append(np.stack([(npos == 1).sum(1), (npos == 2).sum(1), (two_d != 0).sum(1), (c != 0).sum(1),
                                   c.sum(1), two_d.sum(1)], axis=1))
        n1, n2, informative, m, C, E2 = np.concatenate(parts).T
        readings = {"sim higher": 0, "sim lower": 0, "difference detected, control failed": 0,
                    "no detectable difference": 0, "too few informative tasks": 0}
        for r in range(R):
            if informative[r] < MIN_INFORMATIVE:
                readings["too few informative tasks"] += 1
                continue
            if p_exact(int(n1[r]), int(n2[r]), int(E2[r])) >= ALPHA:
                readings["no detectable difference"] += 1
            elif p_sign(int(m[r]), int(C[r])) < ALPHA:
                readings["difference detected, control failed"] += 1
            else:
                readings["sim higher" if E2[r] > 0 else "sim lower"] += 1
        pw = readings["sim higher"] / R
        rows.append({"n tasks": n, "real rate": p, "sim rate": ratio * p, "datasets": R,
                     "power: P(reading = 'sim higher')": round(pw, 4), "Monte Carlo SE": round(math.sqrt(pw * (1 - pw) / R), 4),
                     "reading shares": {k: round(v / R, 4) for k, v in readings.items()},
                     "mean informative tasks": round(float(informative.mean()), 2),
                     "two-proportion z-test power (sim n vs real 2n, two-sided 5%)": round(z_power(n, ratio * p, p), 4)})
    return rows


@app.local_entrypoint()
def main():
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rows = power.remote()
    out = {"status": "PRELIMINARY until Aswin reviews", "what": "power of the registered control-gated reading",
           "model": "independent Bernoulli runs per task, rates equal across tasks (no task heterogeneity); exact "
                    "within-task permutation p-values; the RULE's 10-informative-task floor and control gate",
           "seed": SEED, "launched (PT)": stamp, "rows": rows}
    p = HERE / "results" / f"power_paired_{stamp}.json"
    p.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(rows, indent=1))
