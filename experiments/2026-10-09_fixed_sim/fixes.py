"""The fixed-sim rules: general corrections of the simulator's systematic rendering mistakes, applied to SIMULATED
reader messages only (real runs are never passed in). Each rule takes a transcript's reader messages and returns
(new messages, records), one record per tool output it changed. Derived on the dev environments only
(analyze_dev_rules.py, results/dev_rules_2026-10-09_170728.json); see PLAN.md for the evidence and the dropped rules.

R1  text_editor insert: the simulator's diff -> the real tool's `cat -n` snippet + review note, no trailing newline
    (experiments/2026-10-07_q2_rendering_swap/rerender.py, variant `full`, imported unchanged). R1 and R2 leave an
    output the harness truncated (`<START_TOOL_OUTPUT>` wrapper) unchanged.
R2  text_editor str_replace: the simulator's plain diff -> the real gutter `{n:>4}|{sign} {code}`, no trailing newline
    (str_replace_tells.py, variant `sr_gutter_no_newline`, imported unchanged). A diff already numbered (the real form,
    its empty forms included, or the `{sign}{n}|code` form the swap code numbered twice) keeps its numbering and only
    loses its trailing newline.
R5  dpkg/apt output (outputs that contain `Preparing to unpack` and no `\\r`): dpkg's own lines end in `\\r\\n`, and
    each run of database-read lines (frames and/or the final count) becomes one real 20-frame read:
    `(Reading database ... \\r(Reading database ... 5%` ... `100%\\r(Reading database ... N files and directories
    currently installed.)`.
R6  clock (Aswin, 2026-10-09: offsets from the 5 dev environments only): clock.py, applied by the apply step.
"""

import re
import sys
from pathlib import Path

SWAP = Path(__file__).resolve().parents[1] / "2026-10-07_q2_rendering_swap"
sys.path.insert(0, str(SWAP))
import rerender  # noqa: E402  (the rendering swap's code, unchanged)
import str_replace_tells  # noqa: E402

# A hunk line in the `{sign}{n}|code` form the swap code would number twice: a sign BEFORE the number. A bare sign is
# an empty line whose number the simulator dropped (2 of the 3 dev sign-first str_replace diffs have one).
SIGN_FIRST_LINE = re.compile(r"^[ +\-](?: *\d+\|.*)?$")
# A hunk line already in the real gutter form, the empty forms included (`   2|`, `   2| `, `   2|+ `).
REAL_GUTTER_LINE = re.compile(r"^ *\d+\|(?:[ +\-](?: .*)?)?$")
READ_FRAME = re.compile(r"^\(Reading database \.\.\.(?: \d+%)?\s*$")
# The harness's wrapper for an output too long to show; it always ends in a newline, so R1 and R2 leave such outputs
# alone (dev real dry run: the only real outputs R2 changed were 5 of these).
TRUNCATED = "<START_TOOL_OUTPUT>"
DPKG_MARKER = "Preparing to unpack"
# dpkg's own lines, the prefixes that end in \r\n in >= 90% of dev real outputs with >= 10 occurrences (dev counts in
# results/dev_rules_*.json and the PLAN's R5 row); `Processing` (67%) and `debconf:`/`Starting` (always \n) excluded.
CRLF_PREFIXES = ("Selecting previously unselected package", "(Reading database ...", "Preparing to unpack",
                 "Unpacking ", "Setting up ", "Adding ", "update-alternatives", "Creating ", "selecting ",
                 "invoke-rc.d")
READ_FINAL = re.compile(r"^\(Reading database \.\.\. (\d+) files and directories currently installed\.\)\s*$")


def read_line(line: str) -> bool:
    """A database-read line dpkg prints: a progress frame or the final count (other lines are never consumed)."""
    return bool(READ_FRAME.match(line) or READ_FINAL.match(line))


def numbered_form(body: str) -> str | None:
    """`numbered` when every hunk line is already in the real gutter form, `sign_first` when every hunk line is in the
    sign-first form with at least one added or removed line, else None (a plain diff)."""
    hunk = [l for l in body.split("\n") if l and not l.startswith("@@")]
    if not hunk:
        return None
    if all(REAL_GUTTER_LINE.match(l) for l in hunk):
        return "numbered"
    if all(SIGN_FIRST_LINE.match(l) for l in hunk) and any(l[0] in "+-" and len(l) > 1 for l in hunk):
        return "sign_first"
    return None


def calls_of(messages: list[dict]) -> list[dict]:
    """Tool outputs with their calls, paired by order as the swap code pairs them (tool messages carry no call id)."""
    out, step, pending = [], -1, []
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            step += 1
            pending = [(step, j, c["function"]["name"], rerender.args_of(c["function"])) for j, c in enumerate(m.get("tool_calls") or [])]
        elif m["role"] == "tool" and pending:
            s, j, name, a = pending.pop(0)
            out.append({"step": s, "action": j, "tool": name, "args": a, "msg": i})
    return out


def r1_insert(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    if any(m["role"] == "tool" and not isinstance(m.get("content"), str) for m in messages):
        # the renderer reads earlier outputs as text; a transcript with any non-text tool output is left alone
        return [dict(m) for m in messages], [{"step": None, "action": None, "msg": None, "rule": "R1",
                                              "category": "skipped_non_str_transcript", "changed": False}]
    new, recs = rerender.rerender_messages(messages, "full")
    out = []
    for r in recs:
        if TRUNCATED in str(messages[r["msg"]]["content"]):
            new[r["msg"]]["content"] = messages[r["msg"]]["content"]
            r = dict(r, category="skipped_truncated")
        out.append(dict(r, rule="R1", changed=new[r["msg"]]["content"] != messages[r["msg"]]["content"]))
    return new, out


def r2_str_replace(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    out, recs = [dict(m) for m in messages], []
    for c in calls_of(messages):
        if c["tool"] != "text_editor" or c["args"].get("command") != "str_replace":
            continue
        old = messages[c["msg"]]["content"]
        if not isinstance(old, str):
            recs.append({"step": c["step"], "action": c["action"], "msg": c["msg"], "rule": "R2", "category": "skipped_non_str", "changed": False})
            continue
        if TRUNCATED in old:
            recs.append({"step": c["step"], "action": c["action"], "msg": c["msg"], "rule": "R2", "category": "skipped_truncated", "changed": False})
            continue
        body = old.partition("+++ edited_version\n")[2] if rerender.DIFF in old[:300] else ""
        if body and (category := numbered_form(body)):
            # numbered already (real form, or the sign-first form the swap code would number twice): only the
            # independent trailing-newline correction applies
            new = old.rstrip("\n")
            out[c["msg"]]["content"] = new
            recs.append({"step": c["step"], "action": c["action"], "msg": c["msg"], "rule": "R2", "category": category, "changed": new != old})
            continue
        new, category = str_replace_tells.variant_output(old, "sr_gutter_no_newline")
        out[c["msg"]]["content"] = new
        recs.append({"step": c["step"], "action": c["action"], "msg": c["msg"], "rule": "R2", "category": category, "changed": new != old})
    return out, recs


FRAMES = "(Reading database ... " + "".join(f"\r(Reading database ... {p}%" for p in range(5, 101, 5))


def dpkg_render(text: str) -> str:
    if DPKG_MARKER not in text or "\r" in text:
        return text
    lines, out, k = text.split("\n"), [], 0
    while k < len(lines):
        if read_line(lines[k]):  # a run of database-read lines (frames and/or the final count): one real read
            j = k
            while j < len(lines) and read_line(lines[j]):
                j += 1
            final = next((m for m in map(READ_FINAL.match, lines[k:j]) if m), None)
            out.append(FRAMES + (f"\r(Reading database ... {final.group(1)} files and directories currently installed.)" if final else ""))
            k = j
        else:
            out.append(lines[k])
            k += 1
    for i, line in enumerate(out):
        if line.startswith(CRLF_PREFIXES) and i < len(out) - 1:  # followed by a newline: \r\n as the real terminal writes it
            out[i] = line + "\r"
    return "\n".join(out)


def r5_dpkg(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    out, recs = [dict(m) for m in messages], []
    for c in calls_of(messages):
        if c["tool"] == "text_editor":
            continue
        old = messages[c["msg"]]["content"]
        new = dpkg_render(old) if isinstance(old, str) else old
        if new != old:
            out[c["msg"]]["content"] = new
            recs.append({"step": c["step"], "action": c["action"], "msg": c["msg"], "rule": "R5", "category": "rendered", "changed": True})
    return out, recs


RULES = {"R1": r1_insert, "R2": r2_str_replace, "R5": r5_dpkg}


def apply(messages: list[dict], rules: tuple[str, ...]) -> tuple[list[dict], list[dict]]:
    """All `rules` in order on one simulated transcript; the records name the rule that changed each output."""
    records = []
    for name in rules:
        messages, recs = RULES[name](messages)
        records += recs
    return messages, records
