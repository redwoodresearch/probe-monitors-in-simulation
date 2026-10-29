"""Figure for the 32k vs full-length appendix record (PREPARED ONLY: render it only if the author asks; 2e via the coordinator, 10-16).

Draws the table JSON written by make_fl_vs_32k_table.py (read by sha256; no other input, no recomputation): one horizontal-bar panel per
readout (command first: the lead; reply beside). Per probe, one bar per window and probe weights: first 32k tokens (registered first read)
and full length (no 32k cut) (lead), with the registered weights scored on full transcripts beside the retrain where the table has them.
Below a rule: S, Y and the excess for both windows. Interval lines are the records' 95% joint intervals, drawn between their endpoints;
values are printed; no difference test. Every bar's tick label states its probe weights' training length from the FIT-spec file.

  python plot_fl_vs_32k.py --table fl_vs_32k_table<stamp>_<ts>.json <sha256> --stamp _final [--res-dir results] [--banner "..."]
"""

import argparse
import datetime
import io
import json
import subprocess
import textwrap
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flio import LABEL, NO_TEST, SENS_ROLE, load, stop, write_new

HERE = Path(__file__).resolve().parent
C32, CFULL, CREG = "#0072b2", "#e69f00", "#7570b3"  # three colours, no red/green
WSHORT = {"cut32k": "first 32k tokens", "full": "full length (no 32k cut)"}
WTICK = {"cut32k": "first 32k", "full": "full length"}  # tick labels; the title, legend and footer spell out "full length (no 32k cut)"
CLASS_SHORT = {"i": "trained on whole transcripts", "ii": "trained within the 32k cut", "iii": "trained on short external text",
               "unverified": "training length unverified"}
READOUT = {"C": "command readout", "R": "reply readout"}


def short_length(tl):
    if "value" in tl:
        v = tl["value"]
        return "trained on full transcripts" if v is None else f"trained within {v:,} tokens"
    if tl.get("class_key") in CLASS_SHORT:
        return CLASS_SHORT[tl["class_key"]]
    stop(f"training length {tl.get('entry')!r} has neither a cut value nor a known class key")


def colour(window, role):
    if window == "cut32k":
        return C32, C32
    if role == SENS_ROLE:
        return "white", CREG  # hollow: the registered weights read past their training window
    return CFULL, CFULL


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--table", nargs=2, metavar=("FILE", "SHA256"), required=True)
    ap.add_argument("--stamp", required=True)
    ap.add_argument("--res-dir", default=str(HERE / "results"))
    ap.add_argument("--banner", default="", help="optional first title line (e.g. a draft stamp the plan asks for)")
    ap.add_argument("--git", default="")
    a = ap.parse_args()
    res = Path(a.res_dir)
    t = load(res / Path(a.table[0]).name, a.table[1])
    if t.get("label") != LABEL or t.get("no_difference_test") != NO_TEST:
        stop("the table is not a 32k vs full-length appendix record (label or no-test field differs)")
    git = a.git or subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    readouts = t["settings"]["readouts"]
    nrows = max(sum(len(r["bars"]) for r in t["probe_rows"] if r["readout"] == ro) + 2 * sum(1 for g in t["group_rows"] if g["readout"] == ro)
                for ro in readouts)
    fig, axes = plt.subplots(1, len(readouts), figsize=(8.5 * len(readouts), 0.3 * nrows + 3.4), squeeze=False)
    numbers = {"table": {"file": a.table[0], "sha256": a.table[1]}, "git": git, "label": LABEL, "no_difference_test": NO_TEST, "bars": []}
    withheld = set()
    for ax, ro in zip(axes[0], readouts):
        y, yt, yl, items = 0.0, [], [], []  # items: (y, value node, face, edge, tick label)
        for r in [r for r in t["probe_rows"] if r["readout"] == ro]:
            for b in r["bars"]:
                face, edge = colour(b["window"], b["role"])
                tl = b["training_length"]
                reg = " (registered)" if b["role"] == SENS_ROLE else ""
                items.append((y, b["value"], face, edge, f"{b['record_label'][:-2]}{reg} · {WTICK[b['window']]} · {short_length(tl)}"))
                numbers["bars"].append({"readout": ro, "probe": r["probe"], "record_label": b["record_label"], "window": b["window"], "role": b["role"],
                                        "value_key": b["value"]["key"], "training_length": tl["text"], "training_length_source": tl.get("source")})
                y += 1
            y += 0.6
        y += 0.6
        rule_y = y - 0.9
        for g in [g for g in t["group_rows"] if g["readout"] == ro]:
            for wv in g["windows"]:
                face = C32 if wv["window"] == "cut32k" else CFULL
                stat = "excess (secondary check; off-policy five; comparator (A))" if g["stat"] == "excess" else g["stat"]
                items.append((y, wv["value"], face, face, f"{stat} · {WTICK[wv['window']]}"))
                numbers["bars"].append({"readout": ro, "stat": g["stat"], "window": wv["window"], "value_key": wv["value"]["key"], "members": wv.get("members")})
                y += 1
            y += 0.6
        his = []
        for yy, v, face, edge, lab in items:
            yt.append(yy)
            yl.append(lab)
            if "unavailable" in v or "not_readable" in v:
                ax.text(0, yy, f"  {'unavailable: ' + v['unavailable'] if 'unavailable' in v else 'not readable'}", va="center", fontsize=6.5, color="#777777")
                continue
            p = 100 * v["point"]
            ax.barh(yy, p, 0.8, color=face, edgecolor=edge, lw=1.2)
            if v.get("ci95"):
                lo, hi = (100 * x for x in v["ci95"])  # percentile endpoints: drawn as given (they need not bracket the point)
                ax.hlines(yy, lo, hi, color="#333333", lw=0.9)
                ax.vlines([lo, hi], yy - 0.18, yy + 0.18, color="#333333", lw=0.9)
                txt, xr = f"{p:+.1f} [{lo:+.1f}, {hi:+.1f}]", max(hi, p, 0)
            else:
                withheld.add(str(v.get("withheld")))
                txt, xr = f"{p:+.1f} ‡", max(p, 0)
            ax.text(xr + 0.8, yy, txt, va="center", fontsize=7)
            his.append(xr)
        ax.axhline(rule_y, color="#999999", lw=0.6)
        ax.axvline(0, color="#555555", lw=0.8)
        ax.set_yticks(yt)
        ax.set_yticklabels(yl, fontsize=6.5)
        ax.set_ylim(y - 0.4, -0.8)  # top to bottom; room under the last row
        if his:
            x0, x1 = ax.get_xlim()
            ax.set_xlim(x0, max(x1, max(his)) + 0.3 * (max(x1, max(his)) - x0))  # room for the printed values
        ax.set_xlabel("percentage points (0 = no gap; footer defines each row)")
        ax.set_title(f"{READOUT[ro]}{' (lead)' if ro == readouts[0] else ''}")
    # No count of intervals excluding 0 (09 RULE §4b: no counts of separate tests); the intervals and the group rows speak for themselves.
    title = f"{LABEL}\nSim − real catch gap at 1% false alarms per action, per probe, with S, Y and the excess"
    fig.suptitle(f"{a.banner}\n{title}" if a.banner else title, fontsize=11)
    members = " + ".join(t["probe_rows"][0]["bars"][0]["cohort"]["members"])
    foot = (f"Probe rows: run-mean Δ_full = sim catch − mean of real runs 1 and 2, each side at its own per-action 1% threshold, pooled over {members} "
            f"({t['settings']['combo']}); lines: 95% joint task-bootstrap intervals; the same no-filter cohort and draws in both windows (gated by key "
            f"in the table); {NO_TEST}. Blue: first 32k tokens; orange: full length (no 32k cut); hollow purple: {SENS_ROLE}. "
            "Group rows: S = signed mean of Δ_full over the six headline probes; Y = mean over the six of run 2 − run 1, both at the real "
            "threshold τ_B (signed; S − Y is not shown); excess = the secondary check: the registered PANEL noise-matched excess over the off-policy five on "
            "comparator (A) (mean |sim − real| minus mean |rerun yardstick|, its own definition in the record; lower = closer to rerun noise). Each tick names the probe weights, the window and the "
            "weights' training length, read from each probe's fit spec (table field training_length)."
            + (f" ‡ joint interval withheld ({'; '.join(sorted(withheld))}): point only." if withheld else "")
            + " Builds: " + "; ".join(f"{WSHORT[w]} = {v['build']}, {v['calibration']}, {v['probe_set']}" for w, v in t["build_labels"].items()) + "."
            + f"\nTable {a.table[0]} ({a.table[1][:8]}…) from {t['inputs']['records']['cut32k']['file']} ({t['inputs']['records']['cut32k']['sha256'][:8]}…) "
            f"and {t['inputs']['records']['full']['file']} ({t['inputs']['records']['full']['sha256'][:8]}…). Script @ {git[:9]}.")
    foot = "\n".join(textwrap.fill(p, int(20 * fig.get_figwidth())) for p in foot.split("\n"))
    fig.text(0.01, 0.005, foot, fontsize=6.5, va="bottom")
    fig.tight_layout(rect=(0, 0.012 * (foot.count("\n") + 1) + 0.02, 1, 0.93))
    now = datetime.datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    buf = io.BytesIO()
    fig.savefig(buf, dpi=150)
    write_new(res / f"fl_vs_32k_fig{a.stamp}_{now}.png", buf.getvalue())
    numbers |= {"caption": foot, "title": title}
    write_new(res / f"fl_vs_32k_fig{a.stamp}_{now}_figure_numbers.json", json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
