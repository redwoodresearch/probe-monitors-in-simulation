"""The planted-slip transfer plant (RULE.md): both editor slips planted together into a real run's own tool outputs.

1. str_replace: an output whose edited section shows the numbered gutter becomes the simulator's form, a plain diff
   with exactly one trailing newline. The real format is the gutter with no trailing newline, and the simulator's is
   the reverse (str_replace_tells.py@origin/main: "sr_gutter_no_newline: both (the real format)").
2. insert: a trailing review note is removed.

`strip_gutter`, GUTTER and the output selection (text_editor only, string content, truncated outputs skipped, the
`rerender.DIFF` header, `fixes.numbered_form`, the `rerender.REVIEW` ending) are copied from q2's
`experiments/2026-10-12_q2_template_slips/run_plant_stage.py@8353bcb5` lines 31 and 50-79 (verbatim, except that the str_replace condition is joined onto one line). The one addition is the
trailing newline. Callers pass q2's `fixes` and `rerender` modules (`experiments/2026-10-09_fixed_sim/fixes.py`,
`experiments/2026-10-07_q2_rendering_swap/rerender.py`), loaded as in that script's `_mods()`. Off-policy owners
re-render the planted messages with their reader's template; on-policy owners splice `plant_output` per output.
"""

import re

GUTTER = re.compile(r"^ *\d+\|(?:([ +\-])(?: (.*))?)?$")  # fixes.REAL_GUTTER_LINE, with the sign and code captured


def strip_gutter(out: str) -> str:
    """Step 1's (b) pair, as in run_template_slips.py: each gutter line `{n}|{sign} {code}` becomes `{sign}{code}`."""
    head, sep, body = out.partition("+++ edited_version\n")
    lines = []
    for line in body.split("\n"):
        m = GUTTER.match(line)
        lines.append(line if line.startswith("@@") or not m else (m.group(1) or " ") + (m.group(2) or ""))
    return head + sep + "\n".join(lines)


def plant_output(o: str, cmd: str, fixes, rerender):
    """The planted form of one text_editor output, or None if it carries no plant site."""
    if not isinstance(o, str) or fixes.TRUNCATED in o:
        return None
    if cmd == "str_replace" and rerender.DIFF in o[:300] and fixes.numbered_form(o.partition("+++ edited_version\n")[2]) == "numbered":
        return strip_gutter(o).rstrip("\n") + "\n"  # gutter stripped, one trailing newline: the simulator's form
    if cmd == "insert" and o.rstrip().endswith(rerender.REVIEW):
        return o.rstrip().rstrip("\n")[: -len(rerender.REVIEW)].rstrip("\n")
    return None


def plant_both(messages: list[dict], fixes, rerender):
    """(planted messages, [(message index, step, action, slip)] changed, in order). Both slips in one pass."""
    out, changed = [dict(m) for m in messages], []
    for c in fixes.calls_of(messages):
        if c["tool"] != "text_editor":
            continue
        cmd, o = c["args"].get("command"), messages[c["msg"]]["content"]
        new = plant_output(o, cmd, fixes, rerender)
        if new is not None and new != o:
            out[c["msg"]]["content"] = new
            changed.append((c["msg"], c["step"], c["action"], "gutter+newline" if cmd == "str_replace" else "note"))
    return out, changed
