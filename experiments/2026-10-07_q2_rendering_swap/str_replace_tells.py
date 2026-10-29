"""The two str_replace format tells of the September simulator, applied one at a time to its str_replace outputs.

Real and today's-simulator (e561f936) str_replace diffs number every hunk line: `{n:>4}|{sign} {code}`, where context
and removed lines carry the old file's line number and added lines the new file's, counted from the hunk header. The
September simulator writes plain unified-diff lines (`{sign}{code}`). Real outputs end without a newline; both
simulators end with one. Variants (only successful str_replace diffs change; errors read the same in both tools):
- sr_gutter: add the numbered gutter, keep the trailing newline (what today's simulator writes);
- sr_no_newline: drop the trailing newline only;
- sr_gutter_no_newline: both (the real format).
Each str_replace output gets one category:
- plain: a diff whose hunk lines all parse, numbered from the hunk headers;
- numbered: already carries the gutter (the gutter variants leave it as is);
- failed: a diff with a line that is not a hunk line, or no hunk; the gutter variants leave it as is;
- not_rendered: not a diff (an error message).
"""
import re

from rerender import DIFF, args_of

HUNK = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@.*$")
NUMBERED = re.compile(r"^\s*\d+\|[ +\-] ")
VARIANTS = ("sr_gutter", "sr_no_newline", "sr_gutter_no_newline")


def add_gutter(out: str) -> tuple[str, str]:
    """(numbered output, category) for one str_replace output."""
    if DIFF not in out[:300]:
        return out, "not_rendered"
    head, sep, body = out.partition("+++ edited_version\n")
    lines = body.split("\n")
    end = len(lines) - 1 if lines[-1] == "" else len(lines)  # a trailing newline leaves one empty piece
    if not sep or not lines or not HUNK.match(lines[0]):
        return out, "failed"
    hunk = [l for l in lines[:end] if not l.startswith("@@")]
    if hunk and all(NUMBERED.match(l) for l in hunk):
        return out, "numbered"
    new, old_n, new_n = [], 0, 0
    for line in lines[:end]:
        m = HUNK.match(line)
        if m:
            old_n, new_n = int(m.group(1)), int(m.group(2))
            new.append(line)
            continue
        sign, code = (line[:1], line[1:]) if line else (" ", "")  # a blank context line may have lost its space
        if sign == " ":
            new.append(f"{old_n:>4}|  {code}")
            old_n, new_n = old_n + 1, new_n + 1
        elif sign == "-":
            new.append(f"{old_n:>4}|- {code}")
            old_n += 1
        elif sign == "+":
            new.append(f"{new_n:>4}|+ {code}")
            new_n += 1
        else:
            return out, "failed"
    return head + sep + "\n".join(new + lines[end:]), "plain"


def variant_output(out: str, variant: str) -> tuple[str, str]:
    """(output under `variant`, category)."""
    numbered, category = add_gutter(out)
    if category == "not_rendered":
        return out, category
    if variant == "sr_gutter":
        return numbered, category
    if variant == "sr_no_newline":
        return out.rstrip("\n"), category
    if variant == "sr_gutter_no_newline":
        return numbered.rstrip("\n"), category
    raise ValueError(variant)


def swap_messages(messages: list[dict], variant: str) -> tuple[list[dict], list[dict]]:
    """Reader messages with every text_editor str_replace output replaced under `variant`, and one record per
    str_replace: (step, action), its message index, category and whether its text changed. Steps and actions are
    counted as in rerender.rerender_messages."""
    calls, step, pending = [], -1, []
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            step += 1
            pending = [(step, j, c["function"]["name"], args_of(c["function"])) for j, c in enumerate(m.get("tool_calls") or [])]
        elif m["role"] == "tool" and pending:
            s, j, name, a = pending.pop(0)
            calls.append({"step": s, "action": j, "tool": name, "args": a, "msg": i})
    out, records = [dict(m) for m in messages], []
    for c in calls:
        if c["tool"] != "text_editor" or c["args"].get("command") != "str_replace":
            continue
        new, category = variant_output(messages[c["msg"]]["content"], variant)
        out[c["msg"]]["content"] = new
        records.append({"step": c["step"], "action": c["action"], "msg": c["msg"], "category": category,
                        "changed": new != messages[c["msg"]]["content"]})
    return out, records
