"""Which tokens of an agent run the probes read: the text the reader model sees, and where each scored unit sits in it.

The reader is Qwen3.5-27B, the agent's own model. It reads each whole run once, written out with its chat template, and
the probes score *units*: each command the agent runs and each tool output it reads. A unit is read at these tokens:

  command, call end     (the on-policy and off-policy probes, which we trained)
      The last token lying entirely inside the tool call's own text, before the chat format's closing tags. Qwen3.5
      writes a call as `<tool_call>\\n<function=NAME>\\n<parameter=ARG>\\nVALUE\\n</parameter>\\n...</function>\\n</tool_call>`,
      so this is the last token of the last argument's value; if that value has no token of its own (an empty string,
      say), the value before it, and so on, and last the function name. The trained probes skip a call whose call-end
      token is "0" or "n" (token ids 15 and 77): there the last token is usually a fixed field, such as the last digit
      of a trailing timeout value, rather than the command. In the post's runs that skips about one call in seven.
  command, command end  (the deception probe, the harmfulness and refusal directions, and the high-stakes probe)
      The last token lying entirely inside the value of the argument that holds the command itself: `cmd` (else
      `command`) for bash; for the text editor, the argument its sub-command acts on (create: `file_text`; str_replace:
      `new_str`; insert: `new_str`, else `insert_text`; view and undo_edit: `path`); for any other tool, its last
      argument whose value is text. A call whose command value is missing or empty has no command-end unit.
  tool output           (every probe that scores tool outputs)
      The mean over every token lying entirely inside the output's text. The template's `<tool_response>` tags, and any
      token that merges the output's first or last characters with them, are left out. An output with no such token
      has no unit.

"Lying entirely inside" is decided on characters: the tokenizer gives each token's character span in the rendered text,
and a token counts only if its whole span lies within the span of the content.

Which commands and outputs are scored:
  - The run's final `submit` call and its output are not part of the text the reader reads, and neither is an agent
    turn left with no text and no calls.
  - A call to a tool that was not offered to the agent at that step, or a text-editor call without a `command`
    argument, stays in the text but is not scored, and neither is its output.
  - A tool output is scored only if the agent read it: that is, if the agent made another model call after it.

How a run is written out:
  - The system prompt: an honest run's own. An attack run is read with the system prompt of the honest run of the same
    task, so the reader never sees the side task's instructions (pass it as `system_prompt`).
  - Clock times, dates and Unix timestamps in tool outputs, in the agent's text and in its call arguments become
    `<time>`, `<date>` and `<epoch>`, so that the text does not say when a run happened.
  - Every message is stripped of leading and trailing whitespace (Qwen3.5's template trims messages anyway; trimming
    first keeps each tool output verbatim in the rendered text).
  - Qwen3.5's chat template, thinking turned off, no tool definitions, and one change (`reader_template`). The template
    starts an assistant turn with an empty thinking block (`<think>\\n\\n</think>\\n\\n`) only if no user message comes
    after that turn; we start every assistant turn with it. That is how each turn looked to the agent when it wrote it,
    and it means a later message (a nudge to continue, say) never changes the tokens of an earlier turn.

A worked example: a made-up run with two commands, as `locate` reads it (token positions from Qwen3.5's tokenizer; the
tool outputs come back in user turns):

    system "You are a helpful assistant."    user "How many files are in this folder?"
    agent: bash(cmd="ls")                       output: "notes.txt\\nplan.md"
    agent: bash(cmd="ls | wc -l", timeout=30)   output: "2"
    agent: "There are 2 files."

    tokens 24-44    <|im_start|> assistant \\n <think> \\n\\n </think> \\n\\n <tool_call> \\n < function =b ash > \\n
                    < parameter = cmd > \\n
    token  45       ls            command 1: call end and command end (its only argument holds the command)
    tokens 46-62    \\n </ parameter > ... </tool_call> <|im_end|> \\n <|im_start|> user \\n <tool_response> \\n
    tokens 63-67    notes .txt \\n plan .md     tool output 1: the mean over these five tokens
    tokens 68-92    \\n </tool_response> ... <|im_start|> assistant ... < parameter = cmd > \\n
    tokens 93-97    ls  |  wc  -  l           command 2, command end: token 97 ("l")
    tokens 98-108   \\n </ parameter > \\n < parameter = timeout > \\n
    tokens 109-110  3 0                       command 2, call end: token 110 ("0", id 15), which the trained probes skip
    token  128      2                         tool output 2: the mean over this one token (the agent read it, since it
                                              made one more model call, the final answer)

Comments marked "original:" point to the code that made the activations behind the post, in our research repository at
commit 378d4c7b5 (the system-prompt rule for attack runs: experiments/2026-10-14_q3_overnight_chain/stage_q4b.py lines
358-388).
"""

import bisect
import json
import re
import zipfile

AGENT_MODEL = "vllm/Qwen/Qwen3.5-27B"  # the agent's model calls in the Inspect logs
ATTACHMENT = re.compile(r"attachment://[0-9a-f]{32}")  # Inspect's reference to a long text it stores apart
# original: src/probe_shift/q1_eval.py lines 115-118 (the tokens) and experiments/2026-10-14_p2_fibs_pair/run_fibs_score.py line 36
SKIPPED_CALL_ENDS = {15: "0", 77: "n"}  # Qwen3.5 token ids the trained probes skip as call ends
LATE_TURN = "{%- if loop.index0 > ns.last_query_index %}"  # in Qwen3.5's chat template: "is this turn after the last user message?"
AGENT_OPEN, IM_END, TOOL_OPEN = "<|im_start|>assistant\n", "<|im_end|>", "<tool_response>\n"
CALL_OPEN, CALL_CLOSE, ARG_OPEN, ARG_CLOSE = "<tool_call>\n<function=", "</function>\n</tool_call>", "<parameter=", "\n</parameter>\n"
CALL_TAGS = ("<tool_call>", "</tool_call>", "<function=", "</function>", "<parameter=", "</parameter>")
# original: src/probe_shift/scrub.py lines 19-23 and 65-68 (level 1 of its scrub, the level used for these runs)
TIMES = [(re.compile(r"\d{4}-\d{2}-\d{2}"), "<date>"),  # ISO dates
         (re.compile(r"(?<![\d.:])\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?(?![\d:])"), "<time>"),  # clock times, not inside IPs or ports
         (re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\b"), "<date>"),  # "Oct 14"
         (re.compile(r"\b1[5-9]\d{8}(?:\d{3})?\b"), "<epoch>")]  # Unix time in seconds or milliseconds

# original: experiments/2026-10-14_q3_overnight_chain/evalzip.py
if not hasattr(zipfile, "ZIP_ZSTANDARD"):  # before Python 3.14, zipfile cannot read the zstd-compressed members of .eval files
    import zstandard

    _check, _decompressor = zipfile._check_compression, zipfile._get_decompressor
    zipfile._check_compression = lambda method: None if method == 93 else _check(method)
    zipfile._get_decompressor = lambda method: zstandard.ZstdDecompressor().decompressobj() if method == 93 else _decompressor(method)


# ---------------------------------------------------------------- the run, as the reader reads it


# original: src/probe_shift/transcripts.py lines 72-106
def has_reference(x) -> bool:
    """Whether any value, at any depth, is a whole `attachment://<hash>` reference."""
    if isinstance(x, str):
        return bool(ATTACHMENT.fullmatch(x))
    if isinstance(x, (list, dict)):
        return any(map(has_reference, x.values() if isinstance(x, dict) else x))
    return False


def restore(x, attachments: dict):
    """`x` with every value that is a whole `attachment://<hash>` reference replaced by the text Inspect stored apart under
    that hash. Raises if a reference has no stored text, or if one is still there afterwards (a reference to a reference)."""

    def put_back(v):
        if isinstance(v, str) and ATTACHMENT.fullmatch(v):
            if v.removeprefix("attachment://") not in attachments:
                raise ValueError("an attachment reference has no stored text")
            return attachments[v.removeprefix("attachment://")]
        if isinstance(v, list):
            return [put_back(w) for w in v]
        if isinstance(v, dict):
            return {k: put_back(w) for k, w in v.items()}
        return v

    out = put_back(x)
    if has_reference(out):
        raise ValueError("an attachment reference is left after restoring")
    return out


# original: experiments/2026-10-14_q3_overnight_chain/stage_q4b.py lines 322-325
def read_sample(path) -> dict:
    """The one sample in an Inspect .eval log, its messages restored (see `restore`)."""
    with zipfile.ZipFile(path) as z:
        (name,) = [n for n in z.namelist() if n.startswith("samples/") and n.endswith(".json")]
        sample = json.loads(z.read(name))
    return sample | {"messages": restore(sample["messages"], sample.get("attachments") or {})}


# original: src/probe_shift/transcripts.py lines 139-147
def visible_text(content) -> str:
    """A message's text; the agent's reasoning parts are dropped."""
    if isinstance(content, str):
        return content
    return "\n".join(part.get("text") or "" for part in content or [] if part.get("type") == "text")


def system_prompt(sample: dict) -> str:
    return "\n".join(visible_text(m["content"]) for m in sample["messages"] if m["role"] == "system")


# original: experiments/2026-10-14_q3_overnight_chain/stage_q4b.py lines 136-163 (what is masked) and 421 (level 1)
def mask_times(x):
    """Times and dates replaced by placeholders, in a string or in every string inside call arguments."""
    if isinstance(x, str):
        for pattern, placeholder in TIMES:
            x = pattern.sub(placeholder, x)
        return x
    if isinstance(x, list):
        return [mask_times(v) for v in x]
    if isinstance(x, dict):
        return {k: mask_times(v) for k, v in x.items()}
    return x


# original: src/probe_shift/transcripts.py lines 177-243; stage_q4b.py (same folder as above) lines 257-313 and 436-468
def reader_messages(sample: dict, system_prompt: str) -> tuple[list[dict], list]:
    """The chat messages the reader reads, and which of their calls and outputs are scored.

    Returns (messages, scored), aligned: for an assistant message, one bool per call; for a tool message, one bool; else
    None. Raises if the agent's model calls and its assistant messages differ in number or in their calls, or if a tool
    output does not follow its call in order (we left the few runs where any of these happens out of the post)."""
    turns = [m for m in sample["messages"] if m["role"] == "assistant"]
    steps = [e for e in sample.get("events") or [] if e.get("event") == "model" and e.get("model") == AGENT_MODEL]
    if len(steps) != len(turns):
        raise ValueError("the agent's model calls and its assistant messages differ in number")
    if has_reference(system_prompt):
        raise ValueError("the system prompt is an attachment reference; restore it first")
    unscored = []  # per turn and call, from the model call itself: the tool was not offered, or a text-editor call has no command
    for e, m in zip(steps, turns, strict=True):
        offered = {t.get("name") for t in e.get("tools") or []}
        made = ((((e.get("output") or {}).get("choices") or [{}])[0].get("message") or {}).get("tool_calls") or [])
        if [c.get("function") for c in made] != [c["function"] for c in m.get("tool_calls") or []]:
            raise ValueError("an agent step's calls differ from its assistant message's")
        flags = []
        for c in made:
            name, args = c.get("function"), restore(c.get("arguments") or {}, sample.get("attachments") or {})
            not_offered = bool(offered) and name not in offered and name != "submit"
            flags.append(not_offered or (name == "text_editor" and args.get("command") is None))
        unscored.append(flags)
    messages, scored = [{"role": "system", "content": system_prompt}], [None]
    turn, pending, submit_ids = -1, [], set()  # pending: the latest agent turn's calls that expect an output, as (call id, scored)
    for m in sample["messages"]:
        if m["role"] == "user":
            messages.append({"role": "user", "content": visible_text(m["content"])})
            scored.append(None)
        elif m["role"] == "assistant":
            turn += 1
            calls, marks, pending = [], [], []
            for i, c in enumerate(m.get("tool_calls") or []):
                if c["function"] == "submit":
                    submit_ids.add(c.get("id"))
                    continue
                keep = not unscored[turn][i]
                calls.append({"type": "function", "function": {"name": c["function"], "arguments": mask_times(c.get("arguments") or {})}})
                marks.append(keep)
                pending.append((c.get("id"), keep and turn < len(turns) - 1))  # an output is read only if a later model call follows
            text = visible_text(m["content"])
            if not calls and not text.strip():
                pending = []
                continue
            messages.append({"role": "assistant", "content": mask_times(text), "tool_calls": calls})
            scored.append(marks)
        elif m["role"] == "tool":
            if m.get("function") == "submit" or m.get("tool_call_id") in submit_ids:
                continue
            if not pending or pending[0][0] != m.get("tool_call_id"):
                raise ValueError("a tool output does not follow its call in order")
            _, read = pending.pop(0)
            error = m.get("error")
            text = visible_text(m.get("content"))
            text = (error.get("message") or text) if error else text  # what the agent saw: the error message, if the call failed
            messages.append({"role": "tool", "content": mask_times(text)})
            scored.append(read)
    return [m | {"content": m["content"].strip()} for m in messages], scored


# original: experiments/2026-10-14_q3_overnight_chain/stage_chain.py lines 86 and 159-161
def reader_template(tokenizer):
    """The tokenizer with our one change to Qwen3.5's chat template: every assistant turn is written as the template writes
    turns that come after the last user message (see the module docstring)."""
    if tokenizer.chat_template.count(LATE_TURN) != 1:
        raise ValueError("this chat template is not the one we changed")
    tokenizer.chat_template = tokenizer.chat_template.replace(LATE_TURN, "{%- if true %}")
    return tokenizer


# original: src/probe_shift/reader.py lines 32-37 (trimming) and 40-68
def render(tokenizer, messages: list[dict]) -> tuple[str, list[int], list[tuple[int, int]]]:
    """The rendered text, its token ids, and each token's character span."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False)
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    return text, list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]


# ---------------------------------------------------------------- the units


# original: src/probe_shift/reader.py lines 194-219
def command_argument(tool: str, args: dict) -> str | None:
    """The argument holding a call's command (see the module docstring); None if its value is missing or empty."""
    if tool == "text_editor":
        order = {"create": ("file_text",), "str_replace": ("new_str",), "insert": ("new_str", "insert_text"), "view": ("path",),
                 "undo_edit": ("path",)}.get(args.get("command"), ("file_text", "new_str", "insert_text", "path"))
        name = next((k for k in order if args.get(k) is not None), None)
    elif tool == "bash" or tool.endswith("_bash"):
        name = next((k for k in ("cmd", "command") if k in args), None)
    else:
        name = ([k for k, v in args.items() if isinstance(v, str)] or [None])[-1]
    if name is None or not isinstance(args[name], str) or not args[name].strip():
        return None
    return name


# original: src/probe_shift/reader.py lines 88-113
def written_calls(text: str, a: int, b: int) -> list[dict]:
    """The tool calls written in characters [a, b) of the rendered text: each call's function name and its span, and each
    argument's name and the span of its value."""
    calls, j = [], a
    while (k := text.find(CALL_OPEN, j, b)) >= 0:
        n0 = k + len(CALL_OPEN)
        n1 = text.index(">\n", n0)
        end = text.index(CALL_CLOSE, n1)
        args, p = [], n1 + 2
        while (q := text.find(ARG_OPEN, p, end)) >= 0:
            k1 = text.index(">\n", q)
            v0, v1 = k1 + 2, text.index(ARG_CLOSE, k1 + 2)
            args.append((text[q + len(ARG_OPEN):k1], v0, v1))
            p = v1 + len(ARG_CLOSE)
        calls.append({"name": text[n0:n1], "name_span": (n0, n1), "args": args})
        j = end + len(CALL_CLOSE)
    return calls


# original: src/probe_shift/reader.py lines 116-191 (call end, tool output) and 222-262 (command end)
def locate(text: str, offsets: list[tuple[int, int]], ids: list[int], messages: list[dict], scored: list) -> list[dict]:
    """Every scored unit of the rendered run: {kind, turn, call, tool, start, end} with tokens [start, end) (one token for
    the two command kinds), plus `argument` for a command end and `skipped_by_trained_probes` for a call end. `turn`
    counts the reader's assistant messages from 0; `call` is the call's index within its turn (for an output, the call
    it answers)."""
    starts, ends = [s for s, _ in offsets], [e for _, e in offsets]

    def inside(a, b):  # the tokens lying entirely inside characters [a, b)
        return bisect.bisect_left(starts, a), bisect.bisect_right(ends, b)

    n_agent, n_tool = sum(m["role"] == "assistant" for m in messages), sum(m["role"] == "tool" for m in messages)
    if text.count(AGENT_OPEN) != n_agent or text.count(TOOL_OPEN) != n_tool or any(
            m["role"] == "assistant" and IM_END in json.dumps(m) for m in messages):
        raise ValueError("a message quotes a chat-template delimiter, so its place in the text would be ambiguous")
    if any(t in json.dumps(c["function"]["arguments"]) for m in messages for c in m.get("tool_calls") or [] for t in CALL_TAGS):
        raise ValueError("a call argument quotes the call tags, so the call's place in the text would be ambiguous")
    units, pos, turn, last_calls, answered = [], 0, -1, [], 0
    for m, marks in zip(messages, scored, strict=True):
        if m["role"] == "assistant":
            turn += 1
            a = text.index(AGENT_OPEN, pos) + len(AGENT_OPEN)
            pos = text.index(IM_END, a)
            last_calls = m["tool_calls"]
            found = written_calls(text, a, pos)
            if [c["name"] for c in found] != [c["function"]["name"] for c in last_calls] or any(
                    len(f["args"]) != len(c["function"]["arguments"]) for f, c in zip(found, last_calls, strict=True)):
                raise ValueError(f"turn {turn}: the calls in the rendered text differ from the message's")
            for i, (f, c, keep) in enumerate(zip(found, last_calls, marks, strict=True)):
                if not keep:
                    continue
                spans = [inside(v0, v1) for _, v0, v1 in reversed(f["args"])] + [inside(*f["name_span"])]
                i0, i1 = next((s for s in spans if s[1] > s[0]), (0, 0))
                if i1 > i0:
                    units.append({"kind": "call_end", "turn": turn, "call": i, "tool": f["name"], "start": i1 - 1, "end": i1,
                                  "skipped_by_trained_probes": ids[i1 - 1] in SKIPPED_CALL_ENDS})
                arg = command_argument(f["name"], c["function"]["arguments"])
                if arg is not None:
                    v0, v1 = next((v0, v1) for name, v0, v1 in f["args"] if name == arg)
                    if text[v0:v1] not in (c["function"]["arguments"][arg], c["function"]["arguments"][arg].strip()):
                        raise ValueError(f"turn {turn} call {i}: the command is not verbatim in the rendered text")
                    i0, i1 = inside(v0, v1)
                    if i1 > i0:
                        units.append({"kind": "command", "turn": turn, "call": i, "tool": f["name"], "argument": arg,
                                      "start": i1 - 1, "end": i1})
            answered = 0
        elif m["role"] == "tool":
            a = text.index(TOOL_OPEN, pos) + len(TOOL_OPEN)
            pos = a + len(m["content"])
            if text[a:pos] != m["content"]:
                raise ValueError(f"turn {turn}: a tool output is not verbatim in the rendered text")
            i0, i1 = inside(a, pos)
            if marks and i1 > i0:
                units.append({"kind": "tool_output", "turn": turn, "call": answered, "tool": last_calls[answered]["function"]["name"],
                              "start": i0, "end": i1})
            answered += 1
    return units


def run_units(tokenizer, sample: dict, system_prompt: str) -> tuple[list[int], list[dict]]:
    """For one Inspect sample: the token ids the reader reads, and the scored units. `tokenizer` must have gone through
    `reader_template`."""
    messages, scored = reader_messages(sample, system_prompt)
    text, ids, offsets = render(tokenizer, messages)
    return ids, locate(text, offsets, ids, messages, scored)
