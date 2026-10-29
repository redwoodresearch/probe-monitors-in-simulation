"""Rule derivation on the DEV environments only (split.json): the evidence each candidate rule needs before freezing.

Reads the q1 staging's per-output rows (`readout_text`, as the probe reads each output) for real epoch 1 and the
simulated arm, keeps only dev-environment rows (asserted), and reports per rule:
  R2  str_replace diffs: share whose hunk lines all carry the real gutter `{n:>4}|{sign} {code}`; the sim's forms
      (plain, gutter, the `{sign} {n}|code` form that the old swap code numbered twice).
  R3  hunk-header forms: canonical `@@ -a,b +c,d @@` vs short `@@ -N +N @@` (or any other).
  R4  positive control: each real hunk's old side (context + removed) and new side (context + added) re-diffed with
      difflib.unified_diff; the share reproducing the real hunk's sign sequence exactly, by context size. R4': the
      same for H2, block rendering (shared leading/trailing lines as context, then all removed, then all added).
  R2b the real tool's empty context line form (`N|`); R7b empty lines in `cat -n` views and snippets (number only vs
      number + tab); trailing newline on editor diffs and snippets.
  R5  dpkg/apt progress: `(Reading database ...` fragments joined by `\\r` vs by newlines, per arm.
  R6  date and time formats present per arm (counts), and real epoch-1 time-of-day / date distribution.
  R7  `text_editor view` of a file: the `cat -n` line form (`%6d\\t`) vs anything else, per arm.
Writes results/dev_rules_<stamp>.json and prints a summary.

  uv run python experiments/2026-10-09_fixed_sim/analyze_dev_rules.py --staging <dir with outputs.jsonl.gz, calls.jsonl.gz>
"""

import argparse
import difflib
import gzip
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
E1, SIM = "real-attack-e1", "sim-attack"
DIFF = "Here's a diff showing the changes made:"
GUTTER = re.compile(r"^\s*\d+\|([ +\-]) ?(.*)$")
GUTTER_EMPTY = re.compile(r"^\s*\d+\|$")  # the real tool's empty context line: number and bar only
VIEW_EMPTY = re.compile(r"^ *\d+$")  # a real `cat -n` empty line: the number, no tab
VIEW_EMPTY_TAB = re.compile(r"^ *\d+\t$")
SNIPPET = "Here's the result of running `cat -n` on a snippet of"
SIGN_FIRST = re.compile(r"^([ +\-])\s*\d+\|(.*)$")  # `+  313|code`, the form the old swap code numbered twice
HUNK = re.compile(r"^@@ .* @@")
CANON = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@$")
SHORT = re.compile(r"^@@ -\d+ \+\d+ @@$")
DPKG = re.compile(r"\(Reading database \.\.\.")
DATE_FORMATS = {
    "iso_date": re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    "iso_datetime": re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2})?"),
    "ls_month_day_time": re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) +\d{1,2} \d{2}:\d{2}\b"),
    "http_date": re.compile(r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun), \d{2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{4} \d{2}:\d{2}:\d{2}"),
    "clock_hms": re.compile(r"\b\d{2}:\d{2}:\d{2}\b"),
    "epoch_s": re.compile(r"\b1[67]\d{8}\b"),
    "year_2025": re.compile(r"\b2025\b"),
    "year_2026": re.compile(r"\b2026\b"),
}
CAT_N_VIEW = re.compile(r"^Here's the result of running `cat -n` on \S+:\n")
CAT_N_LINE = re.compile(r"^ {0,5}\d{1,6}\t")


def block_render(old: list[str], new: list[str]) -> str:
    """Hypothesis H2 for the real tool's hunk: lines shared at the start and end of the old and new sides are context;
    everything between is all removed lines, then all added lines (no alignment inside the replaced block)."""
    i = 0
    while i < min(len(old), len(new)) and old[i] == new[i]:
        i += 1
    j = 0
    while j < min(len(old), len(new)) - i and old[-1 - j] == new[-1 - j]:
        j += 1
    return " " * i + "-" * (len(old) - i - j) + "+" * (len(new) - i - j) + " " * j


def parse_gutter(body: list[str]) -> list[tuple[str, str]] | None:
    """(sign, code) per hunk line of a real-format diff, the empty-context form included; None if any line is neither."""
    out = []
    for l in body:
        if GUTTER_EMPTY.match(l):
            out.append((" ", ""))
        elif m := GUTTER.match(l):
            out.append((m.group(1), m.group(2)))
        elif l.strip():
            return None
    return out


def hunks(text: str) -> list[tuple[str, list[str]]]:
    """(header, body lines) of each hunk after the diff preamble."""
    out, cur = [], None
    for line in text.split(DIFF, 1)[1].splitlines():
        if HUNK.match(line):
            cur = (line, [])
            out.append(cur)
        elif cur is not None:
            cur[1].append(line)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--staging", type=Path, required=True)
    args = p.parse_args()
    split = json.loads((HERE / "split.json").read_text())
    dev = set(split["dev"])
    command = {}
    for line in gzip.open(args.staging / "calls.jsonl.gz", "rt"):
        c = json.loads(line)
        if c["tool"] == "text_editor" and (m := re.search(r'"command": "(\w+)"', c["text"])):
            command[(c["key"], c["step"], c["action"])] = m.group(1)
    rows = [r for r in map(json.loads, gzip.open(args.staging / "outputs.jsonl.gz", "rt"))
            if r["kept"] and r["arm"] in (E1, SIM) and r["environment"] in dev]
    assert rows and {r["environment"] for r in rows} <= dev, "a test-environment row was read"
    out = {"dev_environments": sorted(dev), "outputs": dict(Counter(r["arm"] for r in rows))}
    c = Counter()
    rediff = Counter()
    dates = {E1: Counter(), SIM: Counter()}
    real_times = Counter()
    for r in rows:
        a, t = r["arm"], r["readout_text"]
        cmd = command.get((r["key"], r["step"], r["action"]), "") if r["tool"] == "text_editor" else ""
        c[(a, "outputs")] += 1
        if r["tool"] == "text_editor" and DIFF in t:
            c[(a, f"diff:{cmd or '?'}")] += 1
            hs = hunks(t)
            body = [l for _, b in hs for l in b if l.strip()]
            c[(a, "gutter_empty_context_lines")] += sum(bool(GUTTER_EMPTY.match(l)) for _, b in hs for l in b)
            form = ("gutter" if body and all(GUTTER.match(l) or GUTTER_EMPTY.match(l) for l in body) else
                    "sign_first" if body and any(SIGN_FIRST.match(l) for l in body) else
                    "plain" if body else "no_hunk_lines")
            c[(a, f"diff_form:{form}")] += 1
            for h, _ in hs:
                c[(a, "hunk:" + ("canonical" if CANON.match(h) else "short" if SHORT.match(h) else "other"))] += 1
            if a == E1 and form == "gutter":
                for h, b in hs:  # R4' positive control (H2, block rendering) on real hunks, empty-context lines parsed
                    parsed = parse_gutter(b)
                    if parsed is None:
                        rediff[("h2", "unparsed")] += 1
                        continue
                    old = [x for sg, x in parsed if sg in " -"]
                    new = [x for sg, x in parsed if sg in " +"]
                    rediff[("h2", block_render(old, new) == "".join(sg for sg, _ in parsed))] += 1
                for h, b in hs:  # R4 positive control on real hunks
                    lines = [GUTTER.match(l) for l in b if l.strip() or l == ""]
                    lines = [m for m in lines if m]
                    old = [m.group(2) for m in lines if m.group(1) in " -"]
                    new = [m.group(2) for m in lines if m.group(1) in " +"]
                    signs = "".join(m.group(1) for m in lines)
                    for n in (3, 4, 5):
                        d = list(difflib.unified_diff(old, new, n=n, lineterm=""))[2:]
                        got = "".join(x[0] for x in d if not x.startswith("@@"))
                        rediff[(n, got == signs)] += 1
        if DPKG.search(t):
            c[(a, "dpkg_progress")] += 1
            c[(a, "dpkg_with_cr")] += "\r(Reading database" in t
        if r["tool"] == "text_editor" and cmd == "view" and CAT_N_VIEW.match(t):
            body = t.split("\n", 1)[1].splitlines()
            c[(a, "view_files")] += 1
            c[(a, "view_cat_n_form")] += bool(body) and all(CAT_N_LINE.match(l) or VIEW_EMPTY.match(l) for l in body if l)
            c[(a, "view_empty_lines_number_only")] += sum(bool(VIEW_EMPTY.match(l)) for l in body)
            c[(a, "view_empty_lines_number_tab")] += sum(bool(VIEW_EMPTY_TAB.match(l)) for l in body)
        if r["tool"] == "text_editor" and SNIPPET in t:  # real insert / large str_replace snippets
            body = t.split(":\n", 1)[1].splitlines() if ":\n" in t else []
            c[(a, f"snippet:{cmd or '?'}")] += 1
            c[(a, "snippet_empty_lines_number_only")] += sum(bool(VIEW_EMPTY.match(l)) for l in body)
            c[(a, "snippet_empty_lines_number_tab")] += sum(bool(VIEW_EMPTY_TAB.match(l)) for l in body)
            c[(a, "snippet_ends_with_newline")] += t.endswith("\n")
        if r["tool"] == "text_editor" and DIFF in t:
            c[(a, "diff_ends_with_newline")] += t.endswith("\n")
        for name, rx in DATE_FORMATS.items():
            if rx.search(t):
                dates[a][name] += 1
        if a == E1:
            for m in DATE_FORMATS["iso_datetime"].finditer(t):
                real_times[m.group(0)[:13]] += 1
    out["counts"] = {f"{a}|{k}": v for (a, k), v in sorted(c.items())}
    out["r4_rediff_real_hunks"] = {f"context_{n}": {"reproduced": rediff[(n, True)], "differs": rediff[(n, False)]}
                                   for n in (3, 4, 5)}
    out["r4_block_render_real_hunks"] = {"reproduced": rediff[("h2", True)], "differs": rediff[("h2", False)],
                                         "unparsed": rediff[("h2", "unparsed")]}
    out["date_formats_outputs_containing"] = {a: dict(v) for a, v in dates.items()}
    out["real_e1_iso_hours_top"] = real_times.most_common(15)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    path = HERE / "results" / f"dev_rules_{stamp}.json"
    with open(path, "x") as f:
        f.write(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)[:6000])
    print(path)


if __name__ == "__main__":
    main()
