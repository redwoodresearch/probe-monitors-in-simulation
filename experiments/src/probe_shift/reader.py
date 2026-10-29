"""Render reader messages with the model's chat template, locate segments, and pool activations per segment.

Runs inside the GPU container. torch is imported lazily so this module imports anywhere.
Written against Qwen3's chat template: an agent turn is `<|im_start|>assistant\\n ... <|im_end|>` and each
tool output is wrapped in `<tool_response>\\n ... \\n</tool_response>`.
"""

import bisect
import json

AGENT_OPEN = "<|im_start|>assistant\n"
IM_END = "<|im_end|>"
TOOL_OPEN = "<tool_response>\n"


def render(tokenizer, messages: list[dict]) -> tuple[list[int], list[dict]]:
    """Token ids and one segment per agent turn and per tool output.

    Each segment: kind ("agent" or "tool_output"), step (index of the agent turn, 0-based), action (which
    of that turn's calls a tool output answers; None for agent turns), tool (tool name, or the turn's
    called tools joined by ","), start/end token indices [start, end), and n_tokens.
    """
    _, found, enc = _layout(tokenizer, messages)
    return enc["input_ids"], _segments(found, enc)


def template_of(tokenizer) -> str:
    """"qwen3.5" when the chat template writes XML tool calls (`<function=NAME>`), else "qwen3" (JSON tool calls)."""
    return "qwen3.5" if "<function=" in (tokenizer.chat_template or "") else "qwen3"


def prepare(tokenizer, messages: list[dict]) -> list[dict]:
    """The messages as the template will write them. Qwen3.5's template trims every message's content (so a tool
    output's trailing newline never reaches the model); trimming up front keeps tool outputs verbatim in the rendering."""
    if template_of(tokenizer) != "qwen3.5":
        return messages
    return [m | {"content": m["content"].strip()} if isinstance(m.get("content"), str) else m for m in messages]


def _layout(tokenizer, messages: list[dict]):
    """The rendered text, character spans of agent turns and tool outputs, and the tokenization with offsets."""
    messages = prepare(tokenizer, messages)
    text = tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False)
    # Segments are found by searching for template delimiters, so a message that quotes one would be misread: reject.
    n_agent, n_tool = sum(m["role"] == "assistant" for m in messages), sum(m["role"] == "tool" for m in messages)
    # An agent turn ends at its first <|im_end|>, so its text and its tool calls (names, arguments) must not quote one.
    quoted_end = any(m["role"] == "assistant" and IM_END in json.dumps(m) for m in messages)
    if text.count(AGENT_OPEN) != n_agent or text.count(TOOL_OPEN) != n_tool or quoted_end:
        raise ValueError("message content contains a chat-template delimiter; segments would be ambiguous")
    found, pos, step, action = [], 0, -1, 0
    for m in messages:
        if m["role"] == "assistant":
            step, action = step + 1, 0
            a = text.index(AGENT_OPEN, pos) + len(AGENT_OPEN)
            b = text.index(IM_END, a)
            tools = ",".join(c["function"]["name"] for c in m.get("tool_calls") or [])
            found.append(("agent", step, None, tools, a, b))
        elif m["role"] == "tool":
            a = text.index(TOOL_OPEN, pos) + len(TOOL_OPEN)
            b = a + len(m["content"])
            if text[a:b] != m["content"]:
                raise ValueError(f"tool output (step {step}, action {action}) is not verbatim in the rendered text")
            found.append(("tool_output", step, action, m.get("tool_name"), a, b))
            action += 1
        else:
            continue
        pos = b
    return text, found, tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)


def _segments(found: list[tuple], enc) -> list[dict]:
    starts = [s for s, _ in enc["offset_mapping"]]
    ends = [e for _, e in enc["offset_mapping"]]
    segments = []
    for kind, step_i, action_i, tool, a, b in found:
        start = bisect.bisect_right(ends, a)  # first token that ends after the segment starts
        end = bisect.bisect_left(starts, b)  # first token that starts at or after the segment ends
        if a == b:  # an empty output: the template's newlines merge into one token around it, so map to no tokens
            end = start
        segments.append({"kind": kind, "step": step_i, "action": action_i, "tool": tool,
                         "start": start, "end": end, "n_tokens": max(end - start, 0)})
    return segments


CALL_OPEN, CALL_CLOSE, OUTPUT_CLOSE = "<tool_call>", "</tool_call>", "</tool_response>"


def _calls(text: str, a: int, b: int, template: str) -> list[tuple]:
    """Tool calls written in the agent span [a, b): (opening `<tool_call>` char, content spans, where the closing
    `</tool_call>` search starts, function name, number of arguments written).
    Content spans, in the order call_end tries them: Qwen3, the call's JSON `{"name": ..., "arguments": ...}`; Qwen3.5
    (XML), each parameter's value from the last to the first, then the function name, so a trailing value with no token
    of its own (`new_str: ""`, or `"\n"`, which merges into the template's newlines) falls back to an earlier one."""
    out, j = [], a
    if template == "qwen3.5":
        head = CALL_OPEN + "\n<function="
        while (k := text.find(head, j, b)) >= 0:
            n0 = k + len(head)
            n1 = text.index(">\n", n0)
            close = text.index("</function>\n" + CALL_CLOSE, n1)
            spans, v_end = [], close
            while (v_end := text.rfind("\n</parameter>\n", n1, v_end)) >= 0:  # parameter values, last first
                spans.append((text.index(">\n", text.rindex("<parameter=", n1, v_end)) + 2, v_end))
            out.append((k, spans + [(n0, n1)], close, text[n0:n1], text.count("<parameter=", n1, close)))
            j = close + len("</function>\n" + CALL_CLOSE)
        return out
    while (k := text.find(CALL_OPEN + "\n", j, b)) >= 0:
        c0 = k + len(CALL_OPEN) + 1
        c1 = text.index("\n" + CALL_CLOSE, c0)
        name = text[c0:].split('"name": "', 1)[1].split('"', 1)[0] if text.startswith('{"name": "', c0) else None
        out.append((k, [(c0, c1)], c1, name, None))
        j = c1 + 1 + len(CALL_CLOSE)
    return out


def segment_end_positions(tokenizer, messages: list[dict], max_tokens: int | None = None) -> dict:
    """Per action, the token positions of its segment ends; the transcript is cut to its first max_tokens tokens.

    Not the end-of-sequence readout: every position sits at the end of one tool call or tool output, mid-transcript.
    For action k of a step (k-th call of the agent turn; the k-th tool output after it):
      pre_call      the `<tool_call>` token opening the call (sees only context before the call)
      call_end      the call's last content token: the last token lying entirely inside the call's content, before any
                    template token. Qwen3 (JSON calls): `{"name": ..., "arguments": ...}`. Qwen3.5 (XML calls,
                    `<function=NAME><parameter=ARG>VALUE</parameter>`): the last parameter value with a token lying
                    entirely inside it, else the name; span is that value's
      call_marker   the `</tool_call>` token closing it
      output_end    the output's last content token: the last token lying entirely inside the output text
                    (None if no token does, e.g. an empty output)
      output_marker the `</tool_response>` token closing it
    A boundary token that merges content with template text (e.g. `}}` plus the template's newline) is not
    "entirely inside", so content ends read before it. Each position: {kind, step, action, tool, token, kept,
    token_text}, plus for call_end and output_end the content's token span [start, end); kept means token < the cut. step and action index agent turns and their calls in `messages`
    (reader_messages: submit calls and empty turns already dropped), as `render` does.
    Also returns ids cut to max_tokens, the uncut length, and per tool output its content tokens within the cut,
    [start, end). Its n_tokens is the output's full length before the cut, so it exceeds end - start for any output
    that reaches past the cut; use start and end for anything aligned with the kept tokens.
    """
    text, found, enc = _layout(tokenizer, messages)
    template = template_of(tokenizer)
    starts = [s for s, _ in enc["offset_mapping"]]
    ends = [e for _, e in enc["offset_mapping"]]
    ids = enc["input_ids"]
    cut = len(ids) if max_tokens is None else min(len(ids), max_tokens)
    names = tokenizer.convert_ids_to_tokens(ids)

    def inside(a, b):
        """Token range [i0, i1) of the tokens lying entirely inside characters [a, b)."""
        return bisect.bisect_left(starts, a), bisect.bisect_right(ends, b)

    def marker(literal, frm):
        c = text.index(literal, frm)
        i = bisect.bisect_left(starts, c)
        if names[i] != literal:
            raise ValueError(f"expected {literal} at character {c}, found {names[i]!r}")
        return i

    def pos(kind, step, action, tool, t, span=None):
        p = {"kind": kind, "step": step, "action": action, "tool": tool, "token": t,
             "kept": t is not None and t < cut, "token_text": None if t is None else names[t]}
        return p | ({"span": span} if span is not None else {})

    tool_calls = [m.get("tool_calls") or [] for m in messages if m["role"] == "assistant"]
    if template == "qwen3.5":  # XML calls write string values raw, so a value quoting the call's own tags is ambiguous
        tags = ("<tool_call>", "</tool_call>", "<function=", "</function>", "<parameter=", "</parameter>")
        if any(t in json.dumps((c["function"].get("arguments") or {})) for calls in tool_calls for c in calls for t in tags):
            raise ValueError("a tool-call argument quotes the XML call tags; call boundaries would be ambiguous")
    positions, outputs = [], []
    for kind, step, action, tool, a, b in found:
        if kind == "agent":
            calls = _calls(text, a, b, template)
            expected = tool_calls[step]
            # Each written call must be the message's call: same count and names, and (XML) every argument written, so a
            # value quoting the template's own tags cannot shift the positions.
            if len(calls) != len(expected) or any(
                    name != c["function"]["name"] or (n_args is not None and n_args != len(c["function"].get("arguments") or {}))
                    for (_, _, _, name, n_args), c in zip(calls, expected)):
                raise ValueError(f"step {step}: tool calls in the rendered text do not match the message")
            for i, ((k, spans, close_from, name, _), c) in enumerate(zip(calls, expected)):
                opener = marker(CALL_OPEN, k)
                ranges = [inside(c0, c1) for c0, c1 in spans]
                i0, i1 = next((r for r in ranges if r[1] > r[0]), ranges[0])  # no content token anywhere: the first span
                positions += [pos("pre_call", step, i, name, opener),
                              pos("call_end", step, i, name, i1 - 1 if i1 > i0 else None, [i0, i1]),
                              pos("call_marker", step, i, name, marker(CALL_CLOSE, close_from))]
        else:
            i0, i1 = inside(a, b)
            positions += [pos("output_end", step, action, tool, i1 - 1 if i1 > i0 else None, [i0, i1]),
                          pos("output_marker", step, action, tool, marker(OUTPUT_CLOSE, b))]
            outputs.append({"step": step, "action": action, "tool": tool, "start": min(i0, cut),
                            "end": min(max(i1, i0), cut), "n_tokens": max(i1 - i0, 0)})
    return {"ids": ids[:cut], "n_tokens": len(ids), "truncated": len(ids) > cut, "positions": positions, "outputs": outputs}


TEXT_EDITOR_PARAMS = ("file_text", "new_str", "insert_text", "path")
BASH_PARAMS = ("cmd", "command")


def command_param(name: str, arguments) -> str | None:
    """The parameter holding a call's command or code: the readout pre-registered for Phase 2 probes (rule 1).
    text_editor by its command: create -> file_text, str_replace -> new_str, insert -> new_str or insert_text (the order
    Inspect's text_editor applies them: insert_text only when new_str is None), view and
    undo_edit -> path (any other command: the first of those present). bash and `*_bash` tools: cmd, then command. Any
    other tool: its last string-valued parameter. Never a numeric parameter. None when the chosen value is missing or
    empty (e.g. a str_replace deleting text): the call has no readout and callers drop and count it."""
    if not isinstance(arguments, dict):
        return None
    if name == "text_editor":
        by_command = {"create": ("file_text",), "str_replace": ("new_str",), "insert": ("new_str", "insert_text"),
                      "view": ("path",), "undo_edit": ("path",)}
        order = by_command.get(arguments.get("command"), TEXT_EDITOR_PARAMS)
        key = next((k for k in order if arguments.get(k) is not None), None)  # a None value is skipped, as the tool does
    elif name == "bash" or name.endswith("_bash"):
        key = next((k for k in BASH_PARAMS if k in arguments), None)
    else:
        strs = [k for k, v in arguments.items() if isinstance(v, str)]
        key = strs[-1] if strs else None
    if key is None or not isinstance(arguments[key], str) or not arguments[key].strip():
        return None
    return key


def command_end_positions(tokenizer, messages: list[dict], max_tokens: int | None = None) -> dict:
    """Per tool call, the last token lying entirely inside the value of its command parameter (`command_param`), the
    readout pre-registered for Phase 2 probes. Qwen3.5 (XML calls) only, where `<parameter=NAME>\\nVALUE\\n</parameter>`
    writes string values raw. Each position: {step, action, tool, param, token, kept, token_text}; a call with no command
    value has param and token None (callers drop and count it). step and action index agent turns and their calls in
    `messages`, as `segment_end_positions` does. Also returns ids cut to max_tokens and the uncut length."""
    if template_of(tokenizer) != "qwen3.5":
        raise ValueError("command_end_positions reads Qwen3.5's XML tool calls")
    text, found, enc = _layout(tokenizer, messages)
    starts = [s for s, _ in enc["offset_mapping"]]
    ends = [e for _, e in enc["offset_mapping"]]
    ids = enc["input_ids"]
    cut = len(ids) if max_tokens is None else min(len(ids), max_tokens)
    names = tokenizer.convert_ids_to_tokens(ids)
    tool_calls = [m.get("tool_calls") or [] for m in prepare(tokenizer, messages) if m["role"] == "assistant"]
    tags = ("<tool_call>", "</tool_call>", "<function=", "</function>", "<parameter=", "</parameter>")
    if any(t in json.dumps((c["function"].get("arguments") or {})) for calls in tool_calls for c in calls for t in tags):
        raise ValueError("a tool-call argument quotes the XML call tags; call boundaries would be ambiguous")
    positions = []
    for kind, step, _, _, a, b in found:
        if kind != "agent":
            continue
        calls, expected = _calls(text, a, b, "qwen3.5"), tool_calls[step]
        if len(calls) != len(expected) or any(name != c["function"]["name"] or n_args != len(c["function"].get("arguments") or {})
                                              for (*_, name, n_args), c in zip(calls, expected)):
            raise ValueError(f"step {step}: tool calls in the rendered text do not match the message")
        for i, ((k, _spans, close, name, _), c) in enumerate(zip(calls, expected)):  # _calls: (k, spans, close, name, n_args)
            args = c["function"].get("arguments") or {}
            param, t = command_param(name, args), None
            if param is not None:
                head = f"<parameter={param}>\n"
                v0 = text.index(head, k, close) + len(head)
                # the value as written (the template writes it raw), else stripped; checked up to the closing tag
                value = next((v for v in (args[param], args[param].strip()) if text.startswith(v + "\n</parameter>", v0)), None)
                if value is None:
                    raise ValueError(f"step {step} call {i}: {param} is not verbatim in the rendered text")
                i0, i1 = bisect.bisect_left(starts, v0), bisect.bisect_right(ends, v0 + len(value))
                t = i1 - 1 if i1 > i0 else None
            positions.append({"step": step, "action": i, "tool": name, "param": param, "token": t,
                              "kept": t is not None and t < cut, "token_text": None if t is None else names[t]})
    return {"ids": ids[:cut], "n_tokens": len(ids), "truncated": len(ids) > cut, "positions": positions}


def ema_max(x, decay: float, block: int = 2048):
    """Elementwise max over t of the moving average ema_t = decay*ema_{t-1} + (1-decay)*x_t, ema_0 = x_0.

    x is [T, H] float32. Computed block by block with a lower-triangular weight matrix and a carried state,
    so memory stays at block^2 floats.
    """
    import torch

    best, carry = None, None
    for b0 in range(0, x.shape[0], block):
        xb = x[b0 : b0 + block]
        idx = torch.arange(xb.shape[0], device=x.device, dtype=torch.float32)
        lag = idx[:, None] - idx[None, :]
        w = torch.where(lag >= 0, (1 - decay) * decay ** lag.clamp(min=0), torch.zeros_like(lag))
        if carry is None:
            w[:, 0] = decay**idx  # ema_0 = x_0
            e = w @ xb
        else:
            e = w @ xb + (decay ** (idx + 1))[:, None] * carry
        top = e.max(0).values
        best = top if best is None else torch.maximum(best, top)
        carry = e[-1]
    return best


def pool(hidden, segments: list[dict], decays: list[float]) -> dict:
    """{"mean": [n, H], "emamax{d}": [n, H]} as fp16 numpy, one row per non-empty segment, in order."""
    import torch

    keep = [s for s in segments if s["n_tokens"] > 0]
    rows = {"mean": []} | {f"emamax{d}": [] for d in decays}
    for s in keep:
        x = hidden[s["start"] : s["end"]].float()
        rows["mean"].append(x.mean(0))
        for d in decays:
            rows[f"emamax{d}"].append(ema_max(x, d))
    return {k: torch.stack(v).half().cpu().numpy() for k, v in rows.items()}


def read(base_model, input_ids: list[int], segments: list[dict], layers: list[int], decays: list[float]) -> dict:
    """One forward pass of the decoder stack (no LM head). Returns {layer: {pooling: [n_nonempty, H]}}."""
    import torch

    results = {}

    def hook_for(layer):
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            results[layer] = pool(h[0], segments, decays)

        return hook

    handles = [base_model.layers[layer].register_forward_hook(hook_for(layer)) for layer in layers]
    try:
        with torch.inference_mode():
            base_model(input_ids=torch.tensor([input_ids], device="cuda"), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return results
