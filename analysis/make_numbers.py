"""Recompute every number listed in numbers.csv from the analysis results in analysis/results/, and check each one against the post.

Run it with:  uv run analysis/make_numbers.py   (from the repository root)

Each row of numbers.csv is one number printed in the post: the line of the post's source text it sits on (post_line), its section,
the words around it, the number as printed, where it comes from (source) and how the post formats it (format). This script looks the
value up, formats it the way the post does, and checks that the result equals the printed text exactly. Where the post's words state
a direction ("lower", "more"), the `direction` column records it and the value's sign is checked too. The script exits with an error
if any number differs.

A source is either a path into results/main_results.json, with "/" between keys (for example
`per_probe/on_policy_probe/commands/0.01/joint/simulated_caught/value`, or `.../ci95/0` for the low end of the 95% interval), or one
of these calculations:
  tasks:<probe>:<scores>            the number of tasks in the probe's comparison
  main_tasks:<probe>:<scores>       the number of distinct main tasks among those tasks (from results/per_task_values.csv)
  half_width:<path>                 half the width of the quantity's 95% interval
  max_half_width:<path>;<path>;...  the widest of those half-widths; min_half_width: the narrowest
  min:<path>;...  max:<path>;...    the smallest or largest of the values
  min_ratio:<a>|<b>;...             the smallest of the ratios a / b (max_ratio: the largest), e.g. scored units per honest run
  same:<path>;...                   a value that must be the same for every path (for example, a count shared by all probes)
  per_main_task:<path>;<probe>:<scores>  the value divided by the number of distinct main tasks (as main_tasks)
  lost_share:<a>;...|<b>;...        (a - b) / a, where every path before "|" must give the same a and every path after it the
                                    same b; e.g. the share of the published probes' commands that the trained probes skip
  near_100:<path>;...               the claim "near 100%": every value is at least 0.99; prints 100%
Values in main_results.json are fractions; the formats turn them into the post's percent and percentage points.
"""

import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).parent
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
RESULTS = HERE / "results"
MINUS = "−"  # the post prints a true minus sign, not a hyphen


def lookup(results, path):
    node = results
    for key in path.split("/"):
        node = node[int(key)] if isinstance(node, list) else node[key]
    if not isinstance(node, (int, float)) or isinstance(node, bool) or not math.isfinite(node):
        raise ValueError(f"{path}: {node!r} is not a finite number")
    return node


def same(results, paths):
    """The one value that every path gives; an error if any two differ."""
    vals = {lookup(results, p) for p in paths}
    if len(vals) != 1:
        raise ValueError(f"{';'.join(paths)}: the values differ: {sorted(vals)}")
    return vals.pop()


def half_width(results, path):
    low, high = lookup(results, path + "/ci95/0"), lookup(results, path + "/ci95/1")
    return (high - low) / 2


def main_tasks(probe, scores):
    with open(RESULTS / "per_task_values.csv", newline="", encoding="utf-8") as f:
        return len({r["main_task"] for r in csv.DictReader(f) if r["probe"] == probe and r["scores"] == scores.replace("_", " ")})


def value(results, source):
    """The number behind a source (see the module docstring)."""
    kind, _, rest = source.partition(":")
    items = rest.split(";")
    if kind in ("tasks", "main_tasks"):
        probe, scores = rest.split(":")
        return results["tasks"][probe][scores] if kind == "tasks" else main_tasks(probe, scores)
    if kind in ("min_ratio", "max_ratio"):
        ratios = [lookup(results, a) / lookup(results, b) for a, b in (item.split("|") for item in items)]
        return min(ratios) if kind == "min_ratio" else max(ratios)
    if kind == "same":
        return same(results, items)
    if kind == "per_main_task":
        probe, scores = items[1].split(":")
        return lookup(results, items[0]) / main_tasks(probe, scores)
    if kind == "lost_share":
        a, b = (same(results, side.split(";")) for side in rest.split("|"))
        return (a - b) / a
    if kind == "half_width":
        return half_width(results, rest)
    if kind in ("max_half_width", "min_half_width"):
        return (max if kind.startswith("max") else min)(half_width(results, p) for p in items)
    if kind in ("max", "min"):
        return (max if kind == "max" else min)(lookup(results, p) for p in items)
    if kind == "near_100":
        return min(lookup(results, p) for p in items)
    return lookup(results, source)


def show(x, form):
    """Format a value the way the post prints it."""
    if form == "count":  # a whole number with thousands separators: 1,313
        if x != int(x):
            raise ValueError(f"{x} is not a whole number")
        return f"{int(x):,}"
    if form == "points_1":  # percentage points, one decimal, no sign: "1.0 points lower"
        return f"{abs(100 * x):.1f}"
    if form == "signed_points_1":  # percentage points with their sign: "−3.0 to +1.0"
        return f"{100 * x:+.1f}".replace("-", MINUS)
    if form == "signed_points_2":
        return f"{100 * x:+.2f}".replace("-", MINUS)
    if form == "percent_1":  # a table cell: 66.1
        return f"{100 * x:.1f}"
    if form == "percent_1_sign":
        return f"{100 * x:.1f}%"
    if form == "percent_0_sign":
        return f"{100 * x:.0f}%"
    if form == "plain_2":  # an R^2: 0.21
        return f"{x:.2f}"
    if form == "signed_2":
        return f"{x:+.2f}".replace("-", MINUS)
    if form == "nearest_5":  # "about 65 to 75": rounded to the nearest 5, halves up
        return str(int(math.floor(x / 5 + 0.5) * 5))
    if form == "word":  # "about two per main task"
        return WORDS[round(x)]
    if form == "one_in_words":  # a share as "one in seven"
        return f"one in {WORDS[round(1 / x)]}"
    if form == "near_100":
        return "100%" if x >= 0.99 else f"not near 100% (lowest {100 * x:.1f}%)"
    raise ValueError(f"unknown format {form!r}")


def main():
    with open(HERE / "numbers.csv", newline="", encoding="utf-8") as f:
        numbers = list(csv.DictReader(f))
    if [n["id"] for n in numbers] != [str(i) for i in range(1, len(numbers) + 1)]:
        raise ValueError("numbers.csv ids must run 1, 2, 3, ... with no gaps or repeats")
    results = json.loads((RESULTS / "main_results.json").read_text())
    wrong = 0
    print(f"{'id':>3}  {'line':>7}  {'printed':>8}  {'rebuilt':>8}  words around the number")
    for n in numbers:
        x = value(results, n["source"])
        got = show(x, n["format"])
        ok = got == n["printed"] and {"negative": x < 0, "positive": x > 0, "": True}[n["direction"]]
        wrong += not ok
        print(f"{n['id']:>3}  {n['post_line']:>7}  {n['printed']:>8}  {got:>8}  {n['words around the number']}"
              + ("" if ok else "   <-- DIFFERENT"))
    print(f"\n{len(numbers) - wrong} of {len(numbers)} numbers match the post.")
    sys.exit(1 if wrong else 0)


if __name__ == "__main__":
    main()
