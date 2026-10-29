"""Write-up figure f8d95273, redrawn from raw data (POST HOC; $0, local): catch at the headline's frozen 1% false-alarm
threshold on real runs against the number of training tasks per fold model, for the Q2.6 learning curve.

Subsamples of n = 20, 50 and 100 tasks (10 random draws each) train each fold model on 3n/5 tasks; the full data trains
it on the mean over the five fold models of the tasks in the three training folds (from the draw-0 file's folds). Per
size: the median over draws, with the range over draws; the full data is one fit, so it has no interval here.

Every drawn value is read by key and printed. The draw-0 file (for the fold sizes) is checked by sha256 against
results/s3_manifest_scaleup196_v1.json and the learning-curve result's recorded input. The numbers on the doc's image
are parsed from 09's FIGURES.md (row for asset f8d95273) and checked: MATCH if |value - doc| <= half a unit in the doc's
last shown digit ("~118" is checked as 118).

  uv run --no-project --with matplotlib python plot_POSTHOC_writeup_learning_curve.py \\
      results/learning_curve_POSTHOC_q26_2026-10-11_101250.json <onpolicy_2x2_scaleup196_draws0-0_2026-10-10_141824.json> \\
      [--figures-ref af47bc10]
"""

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import median
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
ASSET, FIGURES = "f8d95273", "experiments/2026-10-12_writeup_numbers/FIGURES.md"
MANIFEST = HERE / "results" / "s3_manifest_scaleup196_v1.json"


def git(*a: str) -> str:
    return subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=HERE).stdout


def doc_numbers(ref: str) -> list[str]:
    row = next(r for r in git("show", f"{ref}:{FIGURES}").splitlines() if f"| {ASSET} |" in r)
    return re.findall(r"[-+]?\d+(?:\.\d+)?", row.split("|")[4].replace("−", "-"))


def main(path: str, percall: str, figures_ref: str):
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("the repo has uncommitted tracked changes")
    head = git("rev-parse", "HEAD").strip()
    raw = Path(path).read_bytes()
    R = json.loads(raw)
    if not R["gate"]["passed"]:
        raise SystemExit("the learning curve's gate failed; nothing to plot")
    pc_raw = Path(percall).read_bytes()
    pc_sha = hashlib.sha256(pc_raw).hexdigest()
    listed = {Path(f["key"]).name: f["sha256"] for f in json.loads(MANIFEST.read_text())["files"]}
    if listed.get(Path(percall).name) != pc_sha or R["observed_file"]["sha256"] != pc_sha:
        raise SystemExit("the draw-0 file's sha256 does not match the S3 manifest or the learning curve's recorded input")
    fold_n = Counter(json.loads(pc_raw)["folds"].values())
    full_x = sum(sum(fold_n[f] for f in range(5) if f not in (k, (k + 1) % 5)) for k in range(5)) / 5
    by = {}
    for r in R["configs"]:
        by.setdefault(r["n"], []).append(100 * r["metrics"]["real->real"]["catch_frozen"])
    sizes = sorted(n for n in by if n is not None) + [None]
    pts = []  # (n, x, median, lo, hi, draws)
    for n in sizes:
        v = by[n]
        pts.append((n, full_x if n is None else 3 * n / 5, median(v), min(v), max(v), len(v)))
    for n, x, m, lo, hi, k in pts:
        print(f"n={n if n else 'all'}: training tasks per fold model {x:.1f}; catch median {m:.4f}% range [{lo:.4f}, {hi:.4f}] over {k} draw(s)")
    drawn = [(f"catch, {x:.1f} training tasks", m) for n, x, m, *_ in pts] + [(f"training tasks, n={n or 'all'}", x) for n, x, *_ in pts]
    doc = doc_numbers(figures_ref)
    if len(doc) != len(drawn):
        raise SystemExit(f"FIGURES.md lists {len(doc)} numbers for {ASSET}; this script draws {len(drawn)} of them")
    checks = []
    for (label, v), d in zip(drawn, doc):
        tol = 0.5 * 10 ** -(len(d.split(".")[1]) if "." in d else 0)
        ok = abs(v - float(d)) <= tol + 1e-9
        checks.append({"value": label, "drawn": v, "doc": d, "tolerance": tol, "match": ok})
        print(f"{'MATCH' if ok else 'MISMATCH'}: {label}: drawn {v:.4f}, doc {d} (tolerance {tol})")

    fig, ax = plt.subplots(figsize=(7, 5.2))
    sub = [p for p in pts if p[0] is not None]
    ax.errorbar([p[1] for p in sub], [p[2] for p in sub], yerr=[[p[2] - p[3] for p in sub], [p[4] - p[2] for p in sub]],
                fmt="-o", color="#4C72B0", capsize=4, lw=1.8,
                label=f"random task subsets: median, bars = range over {sub[0][5]} draws")
    f = pts[-1]
    ax.plot([f[1]], [f[2]], "s", color="#C44E52", ms=8, label=f"all tasks ({len(json.loads(pc_raw)['folds'])} in the folds; one fit, no interval)")
    for n, x, m, lo, hi, k in pts:
        ax.annotate(f"{m:.1f}%" + ("" if n is None else f"\n[{lo:.0f}–{hi:.0f}]"), (x, m), textcoords="offset points",
                    xytext=(9, -6) if n is not None else (0, 10), ha="left" if n is not None else "center", fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks([p[1] for p in pts])
    ax.set_xticklabels([f"{p[1]:.0f}" for p in pts])
    ax.minorticks_off()
    ax.set_ylim(30, 100)
    ax.set_xlabel("Training tasks per fold model (log; n subsampled tasks x 3/5)")
    ax.set_ylabel("Catch on state-changing commands at 1% false alarms (%)\n(higher is better)")
    ax.set_title("Catch rises with training tasks (real-trained monitor, real runs)", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    fdst = HERE / "figures" / f"writeup_{ASSET}_learning_curve_POSTHOC_{stamp}.png"
    fig.text(0.01, 0.01, f"Threshold: the headline's frozen real 1%.\nData: {Path(path).name} (sha256 {hashlib.sha256(raw).hexdigest()[:8]})\n"
             f"Script: {Path(__file__).name} @ {head[:8]}", fontsize=7, family="monospace")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(fdst, dpi=150)
    out = {"asset": ASSET, "git": head, "figures_ref": figures_ref, "input": Path(path).name,
           "input_sha256": hashlib.sha256(raw).hexdigest(), "percall": Path(percall).name, "percall_sha256": pc_sha,
           "points": [{"n": n, "training_tasks": x, "catch_median": m, "catch_min": lo, "catch_max": hi, "draws": k}
                      for n, x, m, lo, hi, k in pts],
           "doc_checks": checks, "all_match": all(c["match"] for c in checks), "figure": fdst.name}
    dst = HERE / "results" / f"writeup_{ASSET}_learning_curve_POSTHOC_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out))
    print("all match:", out["all_match"])
    print("->", dst, fdst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("percall")
    ap.add_argument("--figures-ref", default="af47bc10")
    main(**{k.replace("-", "_"): v for k, v in vars(ap.parse_args()).items()})
