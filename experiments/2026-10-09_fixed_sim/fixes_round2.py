"""Round-two fixed-sim rules (PLAN.md amendment 6, "Round two"; Aswin, 2026-10-10: one round, derived on the 5 dev
environments only, frozen before use, applied with round one to Qwen3.5's September runs in follow-up (c)). Each rule
takes a transcript's reader messages and returns (new messages, records), one record per tool output it looked at, as
fixes.py's rules do. Applied to SIMULATED runs only; analyze_round2_rules.py dry-runs them on real dev runs and decides
which are kept by the bars written in PLAN.md before any count was seen.

R8   text_editor create: the output loses its trailing newlines (real create outputs end without one).
R9   full-file text_editor view (no `view_range`), in the `cat -n` form: an output that does not end with an empty
     numbered line gains one, numbered one past its last line, in the real form `{n:6d}\\t` plus a final newline.
R10  non-editor output holding an error-style line (a whole line matching scrub.STDERR_LINE, compiled as the viewer's
     verifier did): those lines move to the front, each group keeping its order; trailing newlines stay at the end.
Outputs the harness truncated (fixes.TRUNCATED) and non-text outputs are left alone by all three.
"""

import re

import fixes
from probe_shift.scrub import STDERR_LINE

CAT_N = "Here's the result of running `cat -n` on "
EMPTY_NUMBERED = re.compile(r" *(\d+)\t")  # an empty numbered line; last_is_empty_numbered checks the 6-wide field
NUMBERED = re.compile(r"^ *(\d+)\t")
ERROR_LINE = re.compile(STDERR_LINE.pattern.replace(r")\n?", ")"), re.I)


def _record(c, rule, category, changed):
    return {"step": c["step"], "action": c["action"], "msg": c["msg"], "rule": rule, "category": category, "changed": changed}


def _apply(messages, rule, applies, edit):
    """Run `edit(text) -> (new text, category)` over the outputs `applies(call)` selects."""
    out, recs = [dict(m) for m in messages], []
    for c in fixes.calls_of(messages):
        if not applies(c):
            continue
        old = messages[c["msg"]]["content"]
        if not isinstance(old, str):
            recs.append(_record(c, rule, "skipped_non_str", False))
            continue
        if fixes.TRUNCATED in old:
            recs.append(_record(c, rule, "skipped_truncated", False))
            continue
        new, category = edit(old)
        out[c["msg"]]["content"] = new
        recs.append(_record(c, rule, category, new != old))
    return out, recs


def _r8_edit(text):
    return text.rstrip("\n"), "create"


def r8_create(messages):
    return _apply(messages, "R8", lambda c: c["tool"] == "text_editor" and c["args"].get("command") == "create", _r8_edit)


def last_is_empty_numbered(text):
    """The real form: the output ends with an empty numbered line in `%6d\\t` form, numbered one past the line before it,
    then exactly one newline."""
    if not text.endswith("\t\n") or text.endswith("\n\n"):
        return False
    lines = text[:-1].split("\n")
    m = EMPTY_NUMBERED.fullmatch(lines[-1])
    if not m or lines[-1] != f"{int(m.group(1)):6d}\t":
        return False
    prev = NUMBERED.match(lines[-2]) if len(lines) > 1 else None
    return prev is None or int(prev.group(1)) + 1 == int(m.group(1))


def _r9_edit(text):
    if not text.startswith(CAT_N):
        return text, "not_cat_n"
    body = text.rstrip("\n")
    if EMPTY_NUMBERED.fullmatch(body.split("\n")[-1].rstrip(" ")):  # already ends with an empty numbered line: R9 only adds one
        return text, "has_empty_last"
    m = NUMBERED.match(body.split("\n")[-1])
    if not m:
        return text, "last_line_unnumbered"
    return f"{body}\n{int(m.group(1)) + 1:6d}\t\n", "added_empty_last"


def r9_view(messages):
    return _apply(messages, "R9", lambda c: c["tool"] == "text_editor" and c["args"].get("command") == "view"
                  and "view_range" not in c["args"], _r9_edit)


def error_flags(text):
    """Per line of the text without its trailing newlines: is it an error-style line."""
    return [ERROR_LINE.fullmatch(l) is not None for l in text.rstrip("\n").split("\n")]


def _r10_edit(text):
    body = text.rstrip("\n")
    lines, err = body.split("\n"), error_flags(text)
    if not any(err) or all(err):
        return text, "no_error_line" if not any(err) else "all_error_lines"
    return "\n".join([l for l, e in zip(lines, err) if e] + [l for l, e in zip(lines, err) if not e]) + text[len(body):], "reordered"


def r10_error_order(messages):
    return _apply(messages, "R10", lambda c: c["tool"] != "text_editor", _r10_edit)


RULES = {"R8": r8_create, "R9": r9_view, "R10": r10_error_order}


def apply(messages, kept):
    """The kept round-two rules, in order, on a simulated transcript that round one already fixed."""
    recs = []
    for name in kept:
        messages, r = RULES[name](messages)
        recs += r
    return messages, recs
