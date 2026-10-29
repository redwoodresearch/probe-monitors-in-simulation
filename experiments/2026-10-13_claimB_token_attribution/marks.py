"""Token marks for the token attribution (RULE.md): planted tokens and known-slip tokens, as token positions of the
rendered transcript. Shared by count_marks.py (counts, before any score exists) and analyze_token_attr.py.

- Planted tokens: the tokens holding a planted character (stage_inputs.py records the characters).
- Known-slip tokens in simulated replies, found by the fixed-sim repairs (experiments/2026-10-09_fixed_sim/fixes.py, R1
  and R2, unchanged; it is applied to simulated messages only, as its docstring requires):
    gutter   an output R2 rewrites from the simulator's plain diff into the real gutter: the tokens holding the first
             character of every diff body line (not the @@ hunk headers);
    newline  an output R2 changes: the token holding its final newline (R2 strips it, in both its categories);
    insert   an output R1 re-renders: every token of its window (the whole insert format is the slip), and the note
             position, the window's last token (the note belongs right after it).
An output's window (layout) is its content tokens plus a token that holds its last character and runs on into the
template.
"""

import bisect

import fixes
from probe_shift.reader import _layout, segment_end_positions

SEP = "+++ edited_version\n"


def layout(tok, msgs):
    """(token ids, token offsets, per tool message index: (content char start, window start, window end)). The window is
    the output's content tokens as reader.segment_end_positions reports them (uncut), plus the token holding the
    content's last character when that token runs on into the template (a final newline merges with the template's
    newline: "\n\n"), so every content character is inside the window."""
    _, found, enc = _layout(tok, msgs)
    seg = segment_end_positions(tok, msgs, None)
    tools = [i for i, m in enumerate(msgs) if m["role"] == "tool"]
    outs = [f for f in found if f[0] == "tool_output"]
    if not (len(tools) == len(outs) == len(seg["outputs"])) or seg["ids"] != enc["input_ids"]:
        raise ValueError("layout and segment positions disagree")
    off, spans = enc["offset_mapping"], {}
    for i, f, o in zip(tools, outs, seg["outputs"]):
        a, b, end = f[4], f[5], o["end"]
        if b > a and end < len(off) and off[end][0] < b < off[end][1]:
            end += 1
        spans[i] = (a, o["start"], end)
    return enc["input_ids"], off, spans


def tokens_at(offsets, chars):
    """Sorted token positions holding any of the absolute characters."""
    starts = [s for s, _ in offsets]
    out = set()
    for c in chars:
        t = bisect.bisect_right(starts, c) - 1
        if t < 0 or not offsets[t][0] <= c < offsets[t][1]:
            raise ValueError(f"character {c} is in no token")
        out.add(t)
    return sorted(out)


def planted_tokens(tok, item):
    ids, off, spans = layout(tok, item["messages"])
    a, t0, t1 = spans[len(item["messages"]) - 1]
    got = tokens_at(off, [a + c for c in item["planted_chars"]])
    if not all(t0 <= t < t1 for t in got):
        raise ValueError("a planted token is outside the reply's window")
    return got


def gutter_line_chars(text):
    head, sep, body = text.partition(SEP)
    at, chars = len(head) + len(sep), []
    for line in body.split("\n"):
        if line and not line.startswith("@@"):
            chars.append(at)
        at += len(line) + 1
    return chars


def known_slips(tok, msgs, which):
    """Known-slip token positions of a simulated transcript's tool outputs in `which` (message indices):
    {"gutter": [...], "newline": [...], "insert": [...], "note": [...]}, plus per-output counts."""
    _, r1 = fixes.r1_insert(msgs)
    _, r2 = fixes.r2_str_replace(msgs)
    ids, off, spans = layout(tok, msgs)
    marks, outs = {"gutter": set(), "newline": set(), "insert": set(), "note": set()}, {"r1": 0, "r2_plain": 0, "r2_newline_only": 0}
    for r in r2:
        if not r["changed"] or r["msg"] not in which:
            continue
        text, (a, t0, t1) = msgs[r["msg"]]["content"], spans[r["msg"]]
        if r["category"] in ("numbered", "sign_first"):
            outs["r2_newline_only"] += 1
        else:
            outs["r2_plain"] += 1
            marks["gutter"] |= set(tokens_at(off, [a + c for c in gutter_line_chars(text)]))
        if text.endswith("\n"):
            marks["newline"] |= set(tokens_at(off, [a + len(text) - 1]))
    for r in r1:
        if not r["changed"] or r["msg"] not in which:
            continue
        a, t0, t1 = spans[r["msg"]]
        outs["r1"] += 1
        marks["insert"] |= set(range(t0, t1))
        marks["note"].add(t1 - 1)
    return {k: sorted(v) for k, v in marks.items()}, outs
