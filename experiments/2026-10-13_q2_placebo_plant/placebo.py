"""G1's placebo (PLAN.md): at exactly the tool outputs P4's `plant_both()` changes, a same-place edit in a format that
neither the simulator nor the real tool writes.

P4 (`../2026-10-12_plantslip_transfer/plant_both.py`) picks its sites with `plant_output()`; this module calls that same
function to pick the sites, so the placebo changes the same outputs in the same runs, and then edits them differently:
1. str_replace (P4: each gutter line `{pad}{n}|{sign} {code}` becomes `{sign}{code}`, and one trailing newline is
   appended). Placebo: on the same lines, the prefix `{pad}{n}|` becomes the line number left-aligned with a colon,
   `{n}:` padded with spaces to the same width (`    12|` becomes `12:    `), so each line keeps its length and
   information; and one end character is appended where P4 appends the newline: the first of END_CANDIDATES that
   ends no str_replace output in any arm, chosen by the staging's count (PLAN.md, Addendum 2; the trailing space of
   the first staging was a real-side form).
2. insert (P4: the trailing review note is removed). Placebo: the note is replaced, in place, by NOTE, a different
   note of exactly the same length (135 characters).
"""

import re

from plant_both import GUTTER, plant_output  # P4's plant, unchanged: it picks the sites

PREFIX = re.compile(r"^( *)(\d+)\|")
END_CANDIDATES = ("\t", " ", "\r")  # PLAN.md Addendum 2's fixed list, in order, as relayed (01:48 PT 10-13)
NOTE = ("The edit above has been written to disk. Line numbers in any earlier view of this file may now be out of date "
        "after this new insertion.")


def restyle_gutter(out: str) -> str:
    """Each gutter line's `{pad}{n}|` becomes `{n}:` left-aligned to the same width; every other character unchanged."""
    head, sep, body = out.partition("+++ edited_version\n")
    lines = []
    for line in body.split("\n"):
        if line.startswith("@@") or not GUTTER.match(line):
            lines.append(line)
            continue
        p = PREFIX.match(line)
        lines.append(f"{p.group(2)}:".ljust(p.end()) + line[p.end():])
    return head + sep + "\n".join(lines)


def placebo_output(o: str, cmd: str, fixes, rerender, end: str):
    """The placebo form of one text_editor output, or None where P4's plant leaves it alone."""
    p4 = plant_output(o, cmd, fixes, rerender)
    if p4 is None or p4 == o:
        return None
    if cmd == "str_replace":
        if end not in END_CANDIDATES:
            raise ValueError("the end character must come from END_CANDIDATES")
        return restyle_gutter(o).rstrip("\n") + end
    if len(NOTE) != len(rerender.REVIEW):
        raise ValueError("NOTE must have the review note's length")
    i = len(o.rstrip()) - len(rerender.REVIEW)
    if o[i:i + len(rerender.REVIEW)] != rerender.REVIEW:
        raise ValueError("the review note is not where P4 removes it")
    return o[:i] + NOTE + o[i + len(rerender.REVIEW):]


def plant_placebo(messages: list[dict], fixes, rerender, end: str):
    """(placebo messages, [(message index, step, action, slip)] changed, in order), as `plant_both()` returns."""
    out, changed = [dict(m) for m in messages], []
    for c in fixes.calls_of(messages):
        if c["tool"] != "text_editor":
            continue
        cmd, o = c["args"].get("command"), messages[c["msg"]]["content"]
        new = placebo_output(o, cmd, fixes, rerender, end)
        if new is not None and new != o:
            out[c["msg"]]["content"] = new
            changed.append((c["msg"], c["step"], c["action"], "gutter-restyle+end" if cmd == "str_replace" else "note-text"))
    return out, changed
