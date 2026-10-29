"""POST HOC (chosen after project_third_real.py's output): a double bootstrap for the power column.

project_third_real.py used the mean within-subset half-width h for both the interval and the gap's spread (s = h / 1.96).
Its acting fixed-fit widths fell more slowly than 1/sqrt(n) (beta 0.31), while drawing n tasks with replacement gave a
wider spread at 50 tasks (half-width 4.68 against 3.63). So this check simulates each subset's gap and its reported
interval together: outer draws take k main tasks with replacement from the full set (acting k = n; reading
k = round(n x tasks / cells), so about n cells), the inner 1,000-draw task bootstrap gives that subset's half-width,
and kappa (from project_third_real.py) scales both the gap's deviation from the full-data gap and the half-width.
P(within +-X) = share of outer draws with |delta + kappa d| + kappa h <= X.
    uv run python experiments/2026-10-12_q3_third_real_price/posthoc_double_boot.py --draws0 <...> --scores36 <...> \
        --projection results/third_real_power_2026-10-12_142247.json
"""
import argparse
import importlib.util
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("ptr", HERE / "project_third_real.py")
spec.loader.exec_module(P := importlib.util.module_from_spec(spec))
OUTER, SEED = 1000, 20261013


def main():
    ap = argparse.ArgumentParser()
    for a in ("--draws0", "--scores36", "--projection"):
        ap.add_argument(a, required=True)
    args = ap.parse_args()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=P.REPO).stdout.strip()
    if dirty:
        raise SystemExit(f"refusing to run from a dirty tree:\n{dirty}")
    if P.sha(args.draws0) != P.SHA["draws0"] or P.sha(args.scores36) != P.SHA["scores36"]:
        raise SystemExit("input hashes differ")
    proj = json.loads((HERE / args.projection).read_text())
    Ua, ta = P.acting_units(args.draws0)
    Ur, tr = P.reading_units(args.scores36)
    specs = {"acting catch at 1% (run 2 - run 1)": ("acting", Ua, ta, P.acting_gap),
             "#36 catch at 5% (e2 - e1)": ("reading", Ur, tr, P.reading_gap(0)),
             "#36 catch at 1% (e2 - e1)": ("reading", Ur, tr, P.reading_gap(2))}
    out = {"label": "POST HOC double bootstrap; price only, not launched; Aswin decides", "projection": args.projection,
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=P.REPO).stdout.strip(),
           "settings": {"outer": OUTER, "inner": P.B, "seed": SEED, "X": P.XS}, "yardsticks": {}}
    for j, (name, (kind, U, ut, gap)) in enumerate(specs.items()):
        y = proj["yardsticks"][name]
        kappa, obs = y["kappa"], y["stored"]["observed"]
        g_full = float(gap(U, np.ones((1, len(U))))[0])
        tasks = np.unique(ut)
        members = [np.flatnonzero(ut == t) for t in tasks]
        rng = np.random.default_rng(SEED + 100 * j)
        rows = {}
        for n in P.SIZES[kind]:
            k = n if kind == "acting" else int(round(n * len(tasks) / len(U)))
            d, h, cells = [], [], []
            for _ in range(OUTER):
                pick = rng.integers(0, len(tasks), k)
                idx = np.concatenate([members[t] for t in pick])
                lab = np.concatenate([np.full(len(members[t]), i) for i, t in enumerate(pick)])  # each draw its own cluster
                d.append(float(gap(U[idx], np.ones((1, len(idx))))[0]) - g_full)
                h.append(P.boot(U[idx], lab, gap, rng, P.B)[0])
                cells.append(len(idx))
            d, h = kappa * np.array(d), kappa * np.array(h)
            rows[n] = {"tasks": k, "cells_mean": float(np.mean(cells)), "half_width_mean": float(h.mean()),
                       "half_width_p10_p90": [float(np.percentile(h, 10)), float(np.percentile(h, 90))], "gap_sd": float(d.std(ddof=1)),
                       "gap_sd_x1.96": float(1.96 * d.std(ddof=1)),
                       "P_within": {f"±{X:g}": {"true_0": float((np.abs(d) + h <= X).mean()), "true_observed": float((np.abs(obs + d) + h <= X).mean())}
                                    for X in P.XS}}
        out["yardsticks"][name] = {"kappa": kappa, "observed_full": obs, "by_n": rows}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    p = HERE / f"results/third_real_power_doubleboot_POSTHOC_{stamp}.json"
    p.write_text(json.dumps(out, indent=1))
    print(p.relative_to(P.REPO))
    print(json.dumps({k: {n: [round(r["half_width_mean"], 2), round(r["gap_sd_x1.96"], 2), {x: round(q["true_0"], 3) for x, q in r["P_within"].items()}]
                          for n, r in y["by_n"].items()} for k, y in out["yardsticks"].items()}, indent=1))


if __name__ == "__main__":
    main()
