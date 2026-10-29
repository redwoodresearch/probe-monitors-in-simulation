"""Write-up figure e8be9e7c, redrawn from raw data (POST HOC, exploratory data; $0, local): false alarms per command
(pooled) beside per task (each task once), at the headline's frozen real 1% threshold, for real run 1, simulated and
real run 2, with the paired real - simulated and real run 1 - run 2 differences and their 95% intervals.

Every drawn value is read by key from per_task_false_alarms_POSTHOC_2026-10-11_122057.json and printed. The numbers on
the doc's image are parsed from 09's FIGURES.md (git show <ref>:<path>, the row for asset e8be9e7c) and checked one by
one: MATCH if |value - doc| <= half a unit in the doc's last shown digit.

  uv run --no-project --with matplotlib python plot_POSTHOC_writeup_false_alarms_per_task.py \\
      results/per_task_false_alarms_POSTHOC_2026-10-11_122057.json [--figures-ref af47bc10]
"""

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
ASSET, FIGURES = "e8be9e7c", "experiments/2026-10-12_writeup_numbers/FIGURES.md"
NAME = {"real": "real run 1", "sim": "simulated", "run2": "real run 2"}


def git(*a: str) -> str:
    return subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=HERE).stdout


def doc_numbers(ref: str) -> list[str]:
    row = next(r for r in git("show", f"{ref}:{FIGURES}").splitlines() if f"| {ASSET} |" in r)
    cell = row.split("|")[4]  # "numbers on the image"
    return re.findall(r"[-+]?\d+(?:\.\d+)?", cell.replace("−", "-"))


def main(path: str, figures_ref: str):
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("the repo has uncommitted tracked changes")
    head = git("rev-parse", "HEAD").strip()
    raw = Path(path).read_bytes()
    D = json.loads(raw)
    W, C = D["worlds"], D["comparisons"]
    pct = lambda x: 100 * x  # noqa: E731
    drawn = []  # (label, value in %, as drawn)
    for w in ("real", "sim", "run2"):
        drawn += [(f"{NAME[w]} per command", pct(W[w]["pooled"])), (f"{NAME[w]} per task", pct(W[w]["per_task_mean"]))]
    diffs = {}
    for comp, label in (("real - sim", "real - simulated"), ("real - run2", "real run 1 - run 2")):
        for kind, kl in (("per_task", "per task"), ("pooled", "per command")):
            c = C[comp][kind]
            diffs[(label, kl)] = (pct(c["point"]), pct(c["ci95"][0]), pct(c["ci95"][1]))
    drawn += [(f"real - simulated per task {s}", v) for s, v in zip(("point", "lo", "hi"), diffs[("real - simulated", "per task")])]
    drawn += [(f"real - simulated per command {s}", v) for s, v in zip(("point", "lo", "hi"), diffs[("real - simulated", "per command")])]
    for label, v in drawn:
        print(f"{label}: {v:.4f}%")
    for (label, kl), (p, lo, hi) in diffs.items():
        print(f"{label} {kl}: {p:+.4f} [{lo:+.4f}, {hi:+.4f}] points")
    doc = doc_numbers(figures_ref)
    if len(doc) != len(drawn):
        raise SystemExit(f"FIGURES.md lists {len(doc)} numbers for {ASSET}; this script draws {len(drawn)} of them")
    checks = []
    for (label, v), d in zip(drawn, doc):
        tol = 0.5 * 10 ** -(len(d.split(".")[1]) if "." in d else 0)
        ok = abs(v - float(d)) <= tol + 1e-9
        checks.append({"value": label, "drawn": v, "doc": d, "tolerance": tol, "match": ok})
        print(f"{'MATCH' if ok else 'MISMATCH'}: {label}: drawn {v:.4f}, doc {d} (tolerance {tol})")

    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    xs = range(3)
    for j, (kind, color, lab) in enumerate((("pooled", "#4C72B0", "per command (all read-only calls pooled)"),
                                           ("per_task_mean", "#DD8452", "per task (each task's rate, averaged over tasks)"))):
        vals = [pct(W[w][kind]) for w in ("real", "sim", "run2")]
        bars = ax.bar([x + (j - 0.5) * 0.38 for x in xs], vals, 0.36, color=color, label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.1f}%", ha="center", fontsize=10)
    ax.axhline(1.0, color="k", lw=0.7, ls=":")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{NAME[w]}\n{W[w]['tasks_with_read_only_calls']} tasks\n{W[w]['read_only_calls']:,} read-only calls"
                        for w in ("real", "sim", "run2")], fontsize=9)
    ax.set_ylabel("False alarms on read-only commands (%)\n(lower is better)")
    ax.set_title("False alarms per command vs per task\n(the extra simulated false alarms come from a few tasks)", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_ylim(0, max(pct(W[w]["pooled"]) for w in W) * 1.35)
    lines = [f"{label}, {kl}: {p:+.1f} points [{lo:+.1f}, {hi:+.1f}]" for (label, kl), (p, lo, hi) in diffs.items()]
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    fdst = HERE / "figures" / f"writeup_{ASSET}_false_alarms_per_task_POSTHOC_{stamp}.png"
    note = ("Paired differences (95% intervals, task resamples):\n" + "\n".join(lines) +
            "\nThreshold: the headline's frozen real 1% (per fold k, real run 1's read-only calls in fold k+1).\n"
            "Dotted line: the 1% target. No intervals on the per-world rates in the data file.\n"
            f"Data: {Path(path).name} (sha256 {hashlib.sha256(raw).hexdigest()[:8]})\n"
            f"Script: {Path(__file__).name} @ {head[:8]}")
    fig.text(0.01, 0.01, note, fontsize=7.5, va="bottom", family="monospace")
    fig.tight_layout(rect=(0, 0.27, 1, 1))
    fig.savefig(fdst, dpi=150)
    out = {"asset": ASSET, "git": head, "figures_ref": figures_ref, "input": Path(path).name,
           "input_sha256": hashlib.sha256(raw).hexdigest(), "drawn": dict(drawn),
           "differences_points": {f"{a} {b}": v for (a, b), v in diffs.items()}, "doc_checks": checks,
           "all_match": all(c["match"] for c in checks), "figure": fdst.name}
    dst = HERE / "results" / f"writeup_{ASSET}_false_alarms_per_task_POSTHOC_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out))
    print("all match:", out["all_match"])
    print("->", dst, fdst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--figures-ref", default="af47bc10")
    main(**{k.replace("-", "_"): v for k, v in vars(ap.parse_args()).items()})
