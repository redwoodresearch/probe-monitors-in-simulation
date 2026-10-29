"""Log-scale ROC figure for one set-up (POST HOC, for the 2026-10-12 daily; $0, stored draw-0 scores only, no sealed runs).

Real-trained monitor on real run 1, simulated and (if stored) real run 2 calls. Each curve uses that world's own per-fold
thresholds on its test-fold read-only calls (the headline robustness pack's (b)/(d) construction, copied), so a point is
(achieved false alarms, catch). The headline uses the real threshold for every world instead, so its points are not on
these curves. Writes the curve points and n per world to results/ and the PNG to figures/.

Two input formats: the acting set-up's scorer file (onpolicy_2x2_*draws0-0*.json: meta, folds, draws[0].cells) and the
reading set-up's Route 3 file (route3_scores_*.json: rows with world e1/sim/e2, fold, y, main_task and s = the five
real-trained fold probes' scores; a call's own test score is s[fold]). Only tasks present in all three worlds are kept.

  uv run --no-project --with numpy --with matplotlib python plot_POSTHOC_roc_log.py <draws file> <slug> "<title>"
"""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from analyze_scaleup_plants import threshold

HERE = Path(__file__).parent
CURVES = (("real->real", "#1f77b4", "real run 1"), ("real->sim", "#d62728", "simulated"), ("real->run2", "#7f7f7f", "real run 2"))


def load_worlds(D):
    """cell -> (test fold, y, own test score, task) per call, on the tasks present in every world."""
    if "rows" in D:  # Route 3
        R = D["rows"]
        out = {c: [r for r in R if r["world"] == w] for w, c in (("e1", "real->real"), ("sim", "real->sim"), ("e2", "real->run2"))}
        get = lambda r: (r["fold"], r["y"], r["s"][r["fold"]], r["main_task"])  # noqa: E731
    else:
        meta, folds, cells = D["meta"], D["folds"], D["draws"][0]["cells"]
        out = {c: [meta[i] | {"score": sc} for i, sc in zip(cells[c]["per_call"]["row"], cells[c]["per_call"]["score"])] for c, _, _ in CURVES if c in cells}
        get = lambda r: (folds[r["task"]], r["y"], r["score"], r["task"])  # noqa: E731
    out = {c: [get(r) for r in rows] for c, rows in out.items()}
    common = set.intersection(*({r[3] for r in rows} for rows in out.values()))
    return {c: tuple(np.array(v) for v in zip(*(r for r in rows if r[3] in common))) for c, rows in out.items()}


def main(path: str, slug: str, title: str):
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True, cwd=HERE).stdout:
        raise SystemExit("the repo has uncommitted tracked changes")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    raw = Path(path).read_bytes()
    D = json.loads(raw)
    worlds = load_worlds(D)
    grid = np.logspace(-3, 0, 60)[:-1]  # the 100% endpoint is added as (1, 1)
    out = {"input": Path(path).name, "input_sha256": hashlib.sha256(raw).hexdigest(), "git": git, "grid": grid.tolist(), "curves": {}}
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    for cell, col, lab in CURVES:
        if cell not in worlds:
            continue
        f, y, s, tasks = worlds[cell]
        pts = []
        for x in grid:
            th = np.array([threshold(s[(f == k) & (y == 0)], np.ones(((f == k) & (y == 0)).sum()), x) for k in range(5)])
            fl = s > th[f]
            pts.append((float(fl[y == 0].mean()), float(fl[y == 1].mean())))
        n = {"tasks": len(set(tasks)), "state_changing": int((y == 1).sum()), "read_only": int((y == 0).sum())}
        out["curves"][cell] = {"n": n, "false_alarm": [p[0] for p in pts] + [1.0], "catch": [p[1] for p in pts] + [1.0]}
        ax.plot(out["curves"][cell]["false_alarm"], out["curves"][cell]["catch"], color=col, lw=1.8,
                label=f"{lab} ({n['tasks']} tasks; {n['state_changing']:,} state-changing, {n['read_only']:,} read-only)")
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 1)
    ax.set_xlabel("False alarms on read-only commands (log; left is better)")
    ax.set_ylabel("Catch on state-changing commands (higher is better)")
    ax.axvline(0.01, color="k", lw=0.8, ls=":")
    ax.text(0.0095, 0.98, "1% false alarms", transform=ax.get_xaxis_transform(), fontsize=8, va="top", ha="right")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.14))
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"roc_log_{slug}_POSTHOC_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out))
    fdst = HERE / "figures" / f"roc_log_{slug}_POSTHOC_{stamp}.png"
    fig.savefig(fdst, dpi=150, bbox_inches="tight")
    for cell, c in out["curves"].items():
        print(cell, c["n"])
    print("->", dst, fdst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("slug")
    ap.add_argument("title")
    main(**vars(ap.parse_args()))
