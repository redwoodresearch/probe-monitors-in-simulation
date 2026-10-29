"""Re-render a simulated run's text_editor insert outputs in the real tool's style.

The real tool answers an insert with a `cat -n` snippet: the 4 lines before the insertion point, the inserted text, the
4 lines after (fewer at either end of the file), numbered from max(1, insert_line - 3) as `%6d<TAB>line`. It ends with a
fixed review note and no trailing newline. The September simulator answers with a unified diff:
- 3 context lines on each side;
- a trailing newline.

The snippet is rebuilt from:
- the diff's context lines;
- the inserted text in the call's arguments (`insert_text`, or `new_str` in older tool versions);
- an earlier view of the file, for the 4th line on each side. The view must be a text_editor `view` in an earlier turn,
  with no edit to that path after it.

Each insert gets one category:
- exact: the real tool's full window is known (lines from the diff and an earlier view, or the file's start or end).
  The diff's hunk header agrees with insert_line.
- approx_window: real template and numbering, but one or both sides lack their 4th line, so the window is narrower.
- approx_numbering: the hunk header's line numbers disagree with insert_line. Lines are numbered from insert_line, as
  the real tool does.
- failed: not a single-hunk diff whose added lines are the inserted text. The output is left unchanged.
- not_rendered: not an insert diff (e.g. an error message, which reads the same in both tools). Left unchanged.
"""
import json
import re

PREAMBLE = "Here's the result of running `cat -n` on a snippet of the edited file:\n"
REVIEW = "Review the changes and make sure they are as expected (correct indentation, no duplicate lines, etc). Edit the file again if necessary."
DIFF = "Here's a diff showing the changes made:"
VIEW = re.compile(r"^Here's the result of running `cat -n` on \S+:\n")
CAT_N = re.compile(r"^\s*(\d+)\t(.*)$")
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@")
EDITS = ("str_replace", "insert", "create", "undo_edit")
SNIPPET = 4


def args_of(call: dict) -> dict:
    a = call.get("arguments") if "arguments" in call else call.get("args")
    return a if isinstance(a, dict) else json.loads(a or "{}")


def inserted_text(a: dict) -> str | None:
    return a.get("insert_text", a.get("new_str"))


def real_style(path: str, insert_line: int, before: list[str], text: str, after: list[str]) -> str:
    """The real tool's insert output for these lines: `before` ends at insert_line, `after` starts right after it."""
    lines = [line.expandtabs() for line in before + text.split("\n") + after]  # the real tool expands tabs
    start = insert_line - len(before) + 1
    body = "\n".join(f"{n:6}\t{line}" for n, line in enumerate(lines, start))
    shown = path if path.startswith("/") else f"/app/{path}"  # the real tool prints the resolved path (cwd /app)
    return f"The file {shown} has been edited. {PREAMBLE}{body}\n{REVIEW}"


def read_only(tool: str, a: dict) -> bool:
    """Whether the Q2 labeler calls this call read-only; False for a tool it has no rule for."""
    from probe_shift.calls import label_call

    try:
        return label_call(tool, a)["label"] == "read_only"
    except ValueError:
        return False


def names_file(a: dict, base: str) -> bool:
    """Whether some whole token of the call's arguments is the file's basename or a path ending in it (`a.py` is not
    named by `data.py`; a backslash before a quote is a separator too)."""
    text = " ".join(str(v) for v in a.values())
    return any(t == base or t.endswith("/" + base) for t in re.split(r"[\s'\"`;|&<>(){}=,:\\]+", text))  # \\: escaped quotes


def file_view(history: list[tuple], path: str) -> tuple[dict[int, str] | None, bool]:
    """(line number -> text, whether that view showed the whole file) from the latest text_editor view of `path` in
    `history`; (None, False) if there is none or the file may have changed after it. `history`: (args, output) or
    (args, output, tool) for every call before the insert, in order, whatever the tool and whether or not its output
    was visible (earlier calls of the insert's own turn included). A view is dropped after a text_editor edit of the
    path, or after any other tool call whose arguments name the file (a whole token that is its basename or a path
    ending in it) unless the Q2 labeler calls it read-only (`stat`, `grep`, `cat` keep the view; a tool it has no rule for drops it). Only a view without a
    view_range, or one running to the end (-1), shows where the file ends."""
    known, whole = None, False
    base = path.rstrip("/").rsplit("/", 1)[-1] if path else None
    for h in history:
        a, out, tool = h[0], h[1], (h[2] if len(h) > 2 else "text_editor")
        if tool != "text_editor":
            if base and names_file(a, base) and not read_only(tool, a):
                known, whole = None, False
            continue
        if a.get("path") != path:
            continue
        if a.get("command") == "view" and VIEW.match(out or ""):
            known = {int(m.group(1)): m.group(2) for m in map(CAT_N.match, (out or "").split("\n")[1:]) if m}
            vr = a.get("view_range")
            whole = vr is None or (isinstance(vr, list) and len(vr) == 2 and vr[1] == -1)  # a suffix view also reaches the end
        elif a.get("command") in EDITS:
            known, whole = None, False
    return known, whole


def rerender(a: dict, out: str, history: list[tuple[dict, str]]) -> tuple[str, str, dict]:
    """(new output, category, details) for one insert call. `history`: every call before it, as file_view takes it."""
    if a.get("command") != "insert" or DIFF not in (out or "")[:300]:
        return out, "not_rendered", {}
    path, il, text = a.get("path"), a.get("insert_line"), inserted_text(a)
    if not isinstance(il, int) or text is None:
        return out, "failed", {"why": "no insert_line or text"}
    body = out.split("+++ edited_version\n", 1)[-1].rstrip("\n").split("\n")
    hunks = [i for i, line in enumerate(body) if line.startswith("@@")]
    if len(hunks) != 1:
        return out, "failed", {"why": f"{len(hunks)} hunks"}
    header, lines = body[hunks[0]], body[hunks[0] + 1:]
    # Every line must be a hunk line (context, +, -, or a blank context line), so text after the diff (e.g. a review
    # note) is never numbered as code. The line shape decides, not the header counts: the simulator's counts are
    # sometimes wrong for well-formed lines (30 of 1,126 sim insert diffs, 1 in the 100 swap runs).
    if any(line and line[0] not in " +-" for line in lines):
        return out, "failed", {"why": "a line that is not a hunk line (text after the diff?)"}
    plus = [i for i, line in enumerate(lines) if line.startswith("+")]
    if not plus or plus != list(range(plus[0], plus[-1] + 1)) or any(line.startswith("-") for line in lines):
        return out, "failed", {"why": "added lines not one contiguous block, or removals"}
    added = [line[1:] for line in lines[plus[0]:plus[-1] + 1]]
    wants = [w for t in (text, text.expandtabs()) for w in (t.split("\n"), t.split("\n")[:-1] if t.endswith("\n") else None) if w is not None]
    if added not in wants:
        return out, "failed", {"why": "added lines differ from the inserted text"}
    ctx = lambda ls: [line[1:] if line[:1] == " " else line for line in ls]  # a context line starts with one space
    before, after = ctx(lines[:plus[0]]), ctx(lines[plus[-1] + 1:])
    m = HUNK.match(header)
    # The old range starts at its first line. An empty old range (a hunk with no context, e.g. -233,0, or -0,0 for an empty
    # file) starts at the line it follows: insert_line itself.
    numbering_ok = bool(m) and int(m.group(1)) == (il if m.group(2) == "0" else il - len(before) + 1)
    view, whole = file_view(history, path)
    # The view must agree with every line the diff shows (old-file numbering), else it is stale (e.g. a shell edit).
    shown = {il - len(before) + 1 + i: line for i, line in enumerate(before)} | {il + 1 + i: line for i, line in enumerate(after)}
    if view and any(n in view and view[n] != line for n, line in shown.items()):
        view = None
    need_before = min(SNIPPET, il)
    missing = range(il - need_before + 1, il - len(before) + 1)
    if len(missing) and view and all(n in view for n in missing):
        before = [view[n] for n in missing] + before
    full_before = len(before) >= need_before
    if len(after) >= SNIPPET or len(after) < 3:  # fewer than 3 after: the file ends there, as in the real snippet
        full_after = True
    elif view and il + len(after) + 1 in view:  # the 4th line after, from the view
        after, full_after = after + [view[il + len(after) + 1]], True
    else:  # unknown unless a view of the whole file shows it ending here
        full_after = bool(view) and whole and max(view) == il + len(after)
    before, after = before[-SNIPPET:], after[:SNIPPET]
    new = real_style(path, il, before, text, after)
    category = "approx_numbering" if not numbering_ok else "exact" if (full_before and full_after) else "approx_window"
    return new, category, {"before": len(before), "after": len(after), "numbering_ok": numbering_ok}


RENDERED = ("exact", "approx_window", "approx_numbering")


VARIANTS = ("full", "snippet_no_note", "diff_with_note", "diff_no_newline")


def variant_output(original: str, rerendered: str, variant: str) -> str:
    """One insert output under a variant, for an insert that `rerender` rendered:
    - full: the real tool's output (numbered snippet plus the review note);
    - snippet_no_note: the numbered snippet without the review note (and no trailing newline);
    - diff_with_note: the simulator's diff with the review note appended after it, as the real tool appends it;
    - diff_no_newline: the simulator's diff without its trailing newline (the control for diff_with_note)."""
    if variant == "full":
        return rerendered
    if variant == "snippet_no_note":
        if not rerendered.endswith("\n" + REVIEW):
            raise ValueError("re-rendered output does not end with the review note")
        return rerendered[: -len("\n" + REVIEW)]
    if variant == "diff_with_note":
        return original.rstrip("\n") + "\n" + REVIEW
    if variant == "diff_no_newline":
        return original.rstrip("\n")
    raise ValueError(variant)


def rerender_messages(messages: list[dict], variant: str = "full") -> tuple[list[dict], list[dict]]:
    """Reader messages with every text_editor insert output that can be re-rendered replaced under `variant` (see
    variant_output), and one record per insert: its (step, action), category and details. Step = agent turn index;
    action = the call's index in its turn, which is also its output's index among the tool messages after the turn."""
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
        if c["tool"] != "text_editor" or c["args"].get("command") != "insert":
            continue
        history = [(h["args"], messages[h["msg"]]["content"], h["tool"]) for h in calls if (h["step"], h["action"]) < (c["step"], c["action"])]
        new, category, details = rerender(c["args"], messages[c["msg"]]["content"], history)
        if category in RENDERED:
            out[c["msg"]]["content"] = variant_output(messages[c["msg"]]["content"], new, variant)
        records.append({"step": c["step"], "action": c["action"], "msg": c["msg"], "category": category, **details})
    return out, records
