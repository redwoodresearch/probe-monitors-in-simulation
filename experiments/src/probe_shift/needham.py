"""Needham et al. (2025) eval-awareness transcripts (HF `jjpn2/eval_awareness`, gated): Inspect samples labeled
evaluation vs deployment, from 61 source datasets, chat and agentic. Converted and rendered as the LinuxArena reader
renders (`probe_shift.reader`): Qwen chat template, no thinking, no tool definitions, tool outputs as the agent saw
them. The probe reads assistant content tokens: tokens lying entirely inside an assistant turn's text, its rendered
reasoning, or a tool call's JSON. Template tokens (`<think>`, `<tool_call>`, `<|im_end|>`, ...) are never read.
"""

import bisect
from collections import Counter

from probe_shift.reader import _layout

CALL_OPEN, CALL_CLOSE = "<tool_call>\n", "\n</tool_call>"
THINK_OPEN, THINK_CLOSE = "<think>\n", "\n</think>\n\n"


def _text(content, dropped: Counter) -> str:
    """Visible text of an Inspect content field (str or parts); non-text parts are dropped and counted by type."""
    if isinstance(content, str):
        return content
    for c in content or []:
        if c.get("type") != "text":
            dropped[c.get("type")] += 1
    return "\n".join(c.get("text") or "" for c in content or [] if c.get("type") == "text")


def reader_messages(sample: dict) -> tuple[list[dict], Counter]:
    """Chat messages for the reader, as `probe_shift.transcripts.reader_messages` builds them for LinuxArena, except
    that the sample keeps its own system prompt (if any) and no call is dropped (no runner `submit` to remove).
    Tool outputs are the error message when the call errored (what the agent saw). An assistant turn with neither
    text nor calls is dropped. Returns the messages and the count of dropped content parts by type."""
    out, dropped = [], Counter()
    for m in sample["input"]:
        role = m["role"]
        if role in ("system", "user"):
            out.append({"role": role, "content": _text(m["content"], dropped)})
        elif role == "assistant":
            calls = [{"type": "function", "function": {"name": tc["function"], "arguments": tc.get("arguments") or {}}}
                     for tc in m.get("tool_calls") or []]
            text = _text(m["content"], dropped)
            if calls or text.strip():
                out.append({"role": "assistant", "content": text, "tool_calls": calls})
            else:
                dropped["empty_assistant_turn"] += 1
        elif role == "tool":
            error = m.get("error")
            text = _text(m["content"], dropped)
            out.append({"role": "tool", "content": (error.get("message") or text) if error else text,
                        "tool_name": m.get("function")})
        else:
            raise ValueError(f"{sample['id']}: unknown role {role}")
    return out, dropped


def content_tokens(tokenizer, messages: list[dict], max_tokens: int) -> dict:
    """Token ids cut to max_tokens, the uncut length, and the content tokens within the cut: `assistant` (the
    probe's primary readouts) and `all` (every message's content, for transcripts with no assistant turn).

    Per agent turn (text `<|im_start|>assistant\\n` ... `<|im_end|>`, as `reader._layout` finds it), the content
    spans are: the reasoning inside a rendered think block (Qwen's template renders one, empty, on the last turn after
    the last user message), the turn's text, and each tool call's `{"name": ..., "arguments": ...}`. System and user
    messages and tool outputs add their text. A token is read only if it lies entirely inside a span, so template
    tokens and tokens merging content with template text are not. Raises if the rendering does not match a message."""
    text, found, enc = _layout(tokenizer, messages)
    starts = [s for s, _ in enc["offset_mapping"]]
    ends = [e for _, e in enc["offset_mapping"]]
    ids = enc["input_ids"]
    cut = min(len(ids), max_tokens)

    def inside(s, e):
        return [t for t in range(bisect.bisect_left(starts, s), bisect.bisect_right(ends, e)) if t < cut]

    assistant, other, think_turns, pos, segs = [], [], 0, 0, iter(found)
    for m in messages:
        if m["role"] in ("system", "user"):
            a = text.index(f"<|im_start|>{m['role']}\n", pos) + len(f"<|im_start|>{m['role']}\n")
            if text[a:a + len(m["content"])] != m["content"]:
                raise ValueError(f"{m['role']} message is not verbatim in the rendered text")
            other += inside(a, a + len(m["content"]))
            pos = a + len(m["content"])
            continue
        kind, step, _, _, a, b = next(segs)
        pos = b
        if kind == "tool_output":
            other += inside(a, b)  # reader._layout checked that the output is verbatim
            continue
        spans, content = [], m["content"]
        if "</think>" in content:  # the template moves the reasoning out of the text
            content = content.split("</think>")[-1].lstrip("\n")
        if text.startswith(THINK_OPEN, a):
            close = text.index(THINK_CLOSE, a)
            spans.append((a + len(THINK_OPEN), close))
            think_turns += close > a + len(THINK_OPEN)  # a non-empty reasoning block was rendered
            a = close + len(THINK_CLOSE)
            content = content.lstrip("\n")
        calls, j = [], a
        while (k := text.find(CALL_OPEN, j, b)) >= 0:
            c0 = k + len(CALL_OPEN)
            c1 = text.index(CALL_CLOSE, c0)
            calls.append((c0, c1))
            j = c1 + len(CALL_CLOSE)
        names = [c["function"]["name"] for c in m["tool_calls"]]
        if len(calls) != len(names) or any(not text[c0:c1].startswith(f'{{"name": "{n}"') for (c0, c1), n in zip(calls, names)):
            raise ValueError(f"turn {step}: rendered tool calls do not match the message")
        if text[a:a + len(content)] != content:
            raise ValueError(f"turn {step}: rendered text does not match the message")
        for s, e in [*spans, (a, a + len(content)), *calls]:
            assistant += inside(s, e)
    assistant, every = sorted(set(assistant)), sorted(set(assistant) | set(other))
    return {"ids": ids[:cut], "n_tokens": len(ids), "truncated": len(ids) > cut, "text_cut": text[:ends[cut - 1]] if cut else "",
            "think_turns": think_turns, "assistant": assistant, "all": every}
