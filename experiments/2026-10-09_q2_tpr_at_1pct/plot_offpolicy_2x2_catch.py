"""The write-up's off-policy 2x2 as catch rate (outline image 9e9734be, "Off-policy 2x2 as catch rate (Qwen3.5 reads
Sol's runs, 530 tasks)"), redrawn from raw data (Aswin's write-up list: "Final figures regenerated from raw data"; no
script produced the outline's image). Local, $0. Committed before it rendered anything.

Reads by key, after checking each file's sha256 against its S3 manifest in this repo:
- results/tpr_frozen_2026-10-10_110209.json (S3; results/manifest_v2.json): the off-policy block's four cells (catch
  rate = TPR at the training world's frozen 1%-FPR threshold, refit 95% intervals, realized FPR) and real minus
  simulated for the real-trained probe;
- ../2026-10-10_route3_phase1_e2/results/route3_2026-10-11_013050.json (S3; that folder's results/s3_manifest_v1.json):
  two real runs (Sol epoch 2 − epoch 1) at 1% false alarms, on route 3's 505 matched tasks.
Prints every value, compares them with the numbers on the outline's image (91 / 91 / 93 / 93%, two real runs differ by
0.4), and writes the PNG and a JSON of the values and any mismatch to results/.

  uv run --no-project --with matplotlib --with numpy python experiments/2026-10-09_q2_tpr_at_1pct/plot_offpolicy_2x2_catch.py \\
      <tpr_frozen_2026-10-10_110209.json> <route3_2026-10-11_013050.json>
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
SETUP = "off-policy (Qwen3.5-27B reading Sol's runs; refit)"
CELLS = ["real->real", "real->sim", "sim->real", "sim->sim"]  # training world -> scored world
IMAGE = {"real->real": 91, "real->sim": 91, "sim->real": 93, "sim->sim": 93, "two real runs (points)": 0.4, "tasks": 530}
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"  # categorical slots 1-2 of the dataviz reference palette


def f1(x: float) -> str:
    """One decimal, half-up (as the write-up's numbers sheet rounds; 93.35 -> 93.4, which float formatting gives as 93.3)."""
    from decimal import ROUND_HALF_UP, Decimal
    return str(Decimal(repr(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(tpr_p: Path, r3_p: Path):
    me = Path(__file__).resolve()
    if subprocess.run(["git", "ls-files", "--error-unmatch", str(me)], cwd=HERE, capture_output=True).returncode \
            or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(me)], cwd=HERE).returncode:
        raise SystemExit("this script differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    m_tpr = json.loads((HERE / "results" / "manifest_v2.json").read_text())["uploaded here (key under prefix experiments/2026-10-09_q2_tpr_at_1pct/: sha256)"]
    want_tpr = m_tpr[f"results/{tpr_p.name}"]
    m_r3 = json.loads((HERE.parent / "2026-10-10_route3_phase1_e2" / "results" / "s3_manifest_v1.json").read_text())
    want_r3 = next(f["sha256"] for f in m_r3["files"] if f["key"].endswith(f"/results/{r3_p.name}"))
    if sha(tpr_p) != want_tpr or sha(r3_p) != want_r3:
        raise SystemExit("an input's sha256 differs from its S3 manifest")
    T = json.loads(tpr_p.read_text())[SETUP]
    R3 = json.loads(r3_p.read_text())

    vals = {"tasks": T["tasks"], "tasks per world": T["tasks per world"], "cells": {}}
    for c in CELLS:
        x = T["cells"][c]
        vals["cells"][c] = {"catch %": round(100 * x["TPR"], 2), "catch 95% (refit), %": [round(100 * v, 2) for v in x["TPR 95% (refit)"]],
                            "false alarms %": round(100 * x["FPR"], 2), "positives": x["positives"], "negatives": x["negatives"]}
    rm = T["real minus sim, real-trained (refit 95%)"]["catch, frozen training-world threshold"]
    vals["real minus simulated, real-trained probe (points)"] = [round(100 * rm["value"], 2), [round(100 * v, 2) for v in rm["ci95"]]]
    g = R3["primary"]["point"]["gap e2-e1"]["catch@0.01"]; gi = R3["primary"]["intervals"]["gap e2-e1"]["catch@0.01"]["ci95"]
    vals["two real runs, e2 - e1 at 1% (points)"] = [round(100 * g, 2), [round(100 * v, 2) for v in gi]]
    vals["two real runs: tasks"] = R3["evaluation_set"]["tasks"]

    mismatch = []
    for c in CELLS:
        got = round(vals["cells"][c]["catch %"])
        if got != IMAGE[c]:
            mismatch.append(f"{c}: image says {IMAGE[c]}%, data says {vals['cells'][c]['catch %']}%")
    if round(100 * g, 1) != IMAGE["two real runs (points)"]:
        mismatch.append(f"two real runs: image says {IMAGE['two real runs (points)']}, data says {round(100 * g, 2)}")
    if vals["tasks"] != IMAGE["tasks"]:
        mismatch.append(f"tasks: image says {IMAGE['tasks']}, data says {vals['tasks']}")
    if vals["two real runs: tasks"] != vals["tasks"]:
        mismatch.append(f"scope, not a number: the two-real-runs difference is on route 3's {vals['two real runs: tasks']} matched tasks, "
                        f"not the 2x2's {vals['tasks']}")

    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    for gi_, train in enumerate(["real", "sim"]):
        for k, (scored, color) in enumerate([("real", BLUE), ("sim", ORANGE)]):
            v = vals["cells"][f"{train}->{scored}"]; x = gi_ + (k - 0.5) * 0.36
            lo, hi = v["catch 95% (refit), %"]
            ax.bar(x, v["catch %"], width=0.32, color=color, edgecolor="white", linewidth=2)
            ax.errorbar(x, v["catch %"], yerr=[[v["catch %"] - lo], [hi - v["catch %"]]], color=INK, capsize=4, lw=1.2)
            ax.text(x, hi + 5.0, f"{f1(v['catch %'])}%", ha="center", va="bottom", fontsize=13, fontweight="bold", color=INK)
            ax.text(x, hi + 0.8, f"[{f1(lo)}, {f1(hi)}]", ha="center", va="bottom", fontsize=8.5, color=MUTED)
            ax.text(x, 4, f"false alarms\n{v['false alarms %']:.2f}%", ha="center", va="bottom", fontsize=8, color="white")
    ax.set_xticks([0, 1], ["probe trained on real runs", "probe trained on simulated runs"], fontsize=11)
    ax.set_ylim(0, 114); ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Catch rate at the 1%-false-alarm threshold (%), higher is better", fontsize=10.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e6e6e3", lw=0.8); ax.set_axisbelow(True)
    ax.bar(0, 0, color=BLUE, label="scored on real runs"); ax.bar(0, 0, color=ORANGE, label="scored on simulated runs")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=10)
    rr, (rl, rh) = vals["real minus simulated, real-trained probe (points)"]
    tr, (tl, th) = vals["two real runs, e2 - e1 at 1% (points)"]
    ax.set_title(f"Qwen3.5 reads Sol's runs ({vals['tasks']} tasks): catch rate at a frozen 1% false-alarm threshold\n"
                 f"real − simulated, real-trained probe: {rr:+.2f} points [{rl:+.2f}, {rh:+.2f}]\n"
                 f"two real runs (route 3, {vals['two real runs: tasks']} tasks): {tr:+.2f} points [{tl:+.2f}, {th:+.2f}]", fontsize=10.5, loc="left")
    fig.text(0.01, 0.005, "\n".join([
        "Impactful-command probe, per call. Threshold set at 1% false alarms on the training world's calibration calls,",
        "applied unchanged to both worlds (realized false alarms in each bar).",
        "Bars' 95% intervals: 500 task resamples, each refitting layer, C and threshold.",
        "Two real runs (Sol epoch 2 − epoch 1, route 3): 2,000 task resamples, thresholds re-estimated, fitted probes fixed.",
        f"Data: tpr_frozen_2026-10-10_110209.json (S3, sha256 {want_tpr[:8]}…), route3_2026-10-11_013050.json (S3, {want_r3[:8]}…);",
        f"script plot_offpolicy_2x2_catch.py @ {head[:8]}."]), fontsize=6.8, color=MUTED, va="bottom")
    fig.tight_layout(rect=(0, 0.13, 1, 1))

    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    png = HERE / "results" / f"offpolicy_2x2_catch_{stamp}.png"
    fig.savefig(png, dpi=200)
    out = {"label": "redrawn from raw data for the write-up (outline image 9e9734be)", "git": head,
           "inputs sha256": {tpr_p.name: want_tpr, r3_p.name: want_r3}, "values": vals, "image numbers": IMAGE,
           "mismatches with the image": mismatch, "png": png.name}
    with open(HERE / "results" / f"offpolicy_2x2_catch_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1)); print(png)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
