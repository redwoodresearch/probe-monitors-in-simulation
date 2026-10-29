"""B3: one table from the result files, by key (no number typed by hand). Per reader: the reader's command last token and mean
at its primary layer, Qwen3.5 on the same calls (layer 30, command), and Qwen3.5 full-call (part A's grid_result_2026-10-13_163659,
sha256 46064923…, command readout). Rows: S, R44, M, D at 5% and 1%, on the 132 held-out tasks; each catch with its flagged count;
each probe's simulated − first real run and S − R44 on the first real run with 95% intervals and readings. A 1% row where some
world has fewer than 10 flags is marked "lead with 5%".

  uv run --no-project --python 3.12 python report_readers.py --partA <grid_result_2026-10-13_163659.json> --results <reader and
      qwen35_on_reader grid_result files...> --out results
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PART_A = "46064923706b8404304b6f485797d0d349faacff6675ef5cd9b46eae0feb8c86"
SET, W = "held_out_132", ("real", "run2", "sim")


def pct(x):
    return "–" if x is None else f"{100 * x:.1f}"


def ci(c):
    if not c or c.get("ci95") is None:
        return c.get("reading", "–") if c else "–"
    lo, hi = c["ci95"]
    return f"{100 * c['estimate']:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] {c['reading']}"


def rows_of(block, label, lvl, s_r44=None):
    out = []
    for g in ("S", "R44", "M", "D"):
        p = block["probes"].get(g)
        if p is None:
            continue
        sb = p["sets"][SET]
        fl = sb.get("flagged_sabotage") or {w: None if sb["catch"][w] is None else sb["catch"][w] * sb["counts"][w]["sabotage"] for w in W}  # part A: catch x positives
        catch = " / ".join(pct(sb["catch"][w]) + ("" if fl[w] is None else f" ({fl[w]:.1f})") for w in W)
        lead = "lead with 5%" if lvl == "1pct" and any(fl[w] is not None and fl[w] < 10 for w in W) else ""
        gap = "not readable"
        if p["passes"]:
            final = sb.get("sim_minus_real_reading", sb["sim_minus_real"].get("reading"))
            gap = f"{ci(sb['sim_minus_real'])}; yardstick {ci(sb['run2_minus_real_yardstick'])}" + (f"; READ AS: {final}" if final != sb["sim_minus_real"].get("reading") else "")
        sr = ""
        if g == "S" and s_r44 is not None:
            v = s_r44.get(SET, {})
            sr = ci(v["real"]) if "real" in v else v.get("reading", "–")
        out.append(f"| {label} | {lvl} | {g} | {'pass' if p['passes'] else 'FAIL'} | {catch} | {gap} | {sr} | {lead} |")
    return out


def main(partA, results, out):
    if hashlib.sha256(Path(partA).read_bytes()).hexdigest() != PART_A:
        raise SystemExit("part A's result differs from its pinned sha256")
    A = json.loads(Path(partA).read_text())
    R = {json.loads(Path(f).read_text())["reader"]: (f, json.loads(Path(f).read_text())) for f in results}
    lines = ["| variant | level | probe | validity | catch % first real / second real / simulated (flags) | simulated − first real; yardstick (second − first real) | S − R44, first real | note |",
             "|---|---|---|---|---|---|---|---|"]
    for reader in ("seed", "olmo", "llama", "qwen3"):
        if reader not in R:
            continue
        f, D = R[reader]
        L = D["primary_layer"]
        for lvl in ("5pct", "1pct"):
            for ro in ("last", "mean"):
                b = D["results"][f"L{L}/{ro}"][lvl]
                lines += rows_of(b, f"{reader} L{L} {ro}", lvl, b["S_minus_R44"])
            q = R.get(f"qwen35_on_{reader}")
            if q:
                b = q[1]["results"]["L30/command"][lvl]
                lines += rows_of(b, f"Qwen3.5 on {reader}'s calls, L30 command", lvl, b["S_minus_R44"])
            lines += rows_of(A["readouts"]["command"][lvl], "Qwen3.5 full-call, L30 command (part A)", lvl)
        if f"{reader}_paired" in R:  # Amendment 2: the same spec read as-is and repaired (simulated rows only differ)
            P = R[f"{reader}_paired"][1]
            for lvl in ("5pct", "1pct"):
                for ro in ("last", "mean"):
                    for vn in ("asis", "repaired"):
                        b = P["results"][f"{vn}:L{L}/{ro}"][lvl]
                        lines += rows_of(b, f"{reader} L{L} {ro}, {vn} (paired spec)", lvl, b["S_minus_R44"])
        lines.append("| | | | | | | | |")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = Path(out) / f"report_readers_{stamp}.md"
    srcs = "\n".join(f"- {k}: {Path(f).name} (sha256 {hashlib.sha256(Path(f).read_bytes()).hexdigest()[:12]}…)" for k, (f, _) in sorted(R.items()))
    with dst.open("x") as fh:
        fh.write(f"# B3 table (written {stamp} PT by report_readers.py; PRELIMINARY until Codex recomputes)\n\nSources:\n{srcs}\n- part A: {Path(partA).name} (sha256 {PART_A[:12]}…)\n\n"
                 "Held-out 132 tasks. Catches pool read eligible calls over tasks; multi-draw probes are means over 5 draws.\n\n" + "\n".join(lines) + "\n")
    print(dst.read_text())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--partA", required=True)
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    main(**vars(ap.parse_args()))
