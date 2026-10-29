"""Can we reproduce exactly the token sequence Qwen3.5 saw at each model call? Laptop, no spend.

  uv run --group viewer python experiments/2026-10-07_phase2_attack_check/analyze_reader_fidelity.py <step-1 run dir>

Inspect logged every model call's raw request (messages and tool definitions) and vLLM's `usage.prompt_tokens`, but
not the prompt text. Each request is rendered through Qwen3.5's chat template at the pinned revision, and the token
count must equal `usage.prompt_tokens` on every call. Equal counts are not proof of equal tokens, so the script also
checks the prefix property (each call's prompt is the start of the next call's). Prints and saves numbers only.

Inspect keeps the raw request only for each run's first 5 model calls, but every call's input messages and token usage.
So the requests are also rebuilt from the input messages (`request_from_input`); the rebuild must render to exactly
the same text as the raw request wherever both exist, and is then count-checked on every call.
"""

import argparse
import glob
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from inspect_ai.event import ModelEvent
from inspect_ai.log import read_eval_log
from transformers import AutoTokenizer

from probe_shift.models import QWEN35_27B

HERE = Path(__file__).parent


def calls(sample) -> list[dict]:
    """Completed model calls, in order: the raw request and raw response (with vLLM's usage). The log stores request
    messages in a shared pool; read_eval_log puts them back."""
    return [e.call.model_dump() for e in sample.events
            if isinstance(e, ModelEvent) and e.error is None and e.call is not None
            and (e.call.response or {}).get("usage")]


def request_from_input(event, tools: list) -> dict:
    """A model call's request rebuilt from Inspect's record of its input, in the raw request's shape: assistant
    reasoning under `reasoning`, tool-call arguments as JSON strings. `tools` comes from the run's raw requests."""
    messages = []
    for m in event.input:
        out = {"role": m.role, "content": m.text}
        if m.role == "assistant":
            reasoning = [c.reasoning for c in m.content if c.type == "reasoning"] if isinstance(m.content, list) else []
            if reasoning:
                out["reasoning"] = "".join(reasoning)
            if m.tool_calls:
                out["tool_calls"] = [{"function": {"name": tc.function, "arguments": json.dumps(tc.arguments)}}
                                     for tc in m.tool_calls]
        elif m.role == "tool" and m.error is not None:
            out["content"] = f"Error: {m.error.message}"  # as Inspect's chat-completions conversion sends it
        messages.append(out)
    return {"messages": messages, "tools": tools}


def template_input(message: dict, reasoning_key: str | None) -> dict:
    """One request message as the chat template sees it: content as text, tool-call arguments parsed from their JSON
    string, and the assistant's `reasoning` passed under `reasoning_key` (None: dropped)."""
    out = {"role": message["role"]}
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    out["content"] = content
    if message.get("tool_calls"):
        out["tool_calls"] = [{"type": "function", "function": {
            "name": tc["function"]["name"], "arguments": json.loads(tc["function"]["arguments"] or "{}")}}
            for tc in message["tool_calls"]]
    if reasoning_key and message.get("reasoning"):
        out[reasoning_key] = message["reasoning"]
    return out


def render(tokenizer, request: dict, reasoning_key: str | None) -> str:
    messages = [template_input(m, reasoning_key) for m in request["messages"]]
    return tokenizer.apply_chat_template(messages, tools=request.get("tools"), tokenize=False,
                                         add_generation_prompt=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(QWEN35_27B["name"], revision=QWEN35_27B["revision"])
    variants = {"reasoning_as_reasoning_content": "reasoning_content", "reasoning_dropped": None}
    rows = []
    for path in sorted(glob.glob(str(args.run_dir / "*" / "*" / "*.eval"))):
        arm, env = Path(path).parts[-3], Path(path).parts[-2]
        for sample in read_eval_log(path, resolve_attachments=True).samples or []:
            cs = calls(sample)
            row = {"arm": arm, "env": env, "raw_calls": len(cs)}
            if not cs:  # no raw request kept (a run that died early): no tool block to rebuild calls with
                rows.append(row | {"skipped": "no raw requests kept"})
                print(json.dumps(rows[-1]), flush=True)
                continue
            # 1. The raw requests Inspect kept: which way of passing reasoning reproduces vLLM's prompt length?
            for name, key in variants.items():
                texts = [render(tok, c["request"], key) for c in cs]
                counts = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
                logged = [c["response"]["usage"]["prompt_tokens"] for c in cs]
                row[f"raw_{name}"] = {"count_equal": sum(a == b for a, b in zip(counts, logged)),
                                      "max_abs_diff": max((abs(a - b) for a, b in zip(counts, logged)), default=0)}
            # 2. Every call, rebuilt from its input record: identical text where the raw request exists, then counts.
            events = [e for e in sample.events if isinstance(e, ModelEvent) and e.error is None
                      and e.output is not None and e.output.usage is not None]
            tools = cs[0]["request"]["tools"]
            texts = [render(tok, request_from_input(e, tools), "reasoning_content") for e in events]
            raw_texts = [render(tok, c["request"], "reasoning_content") for c in cs]
            counts = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
            logged = [e.output.usage.input_tokens for e in events]
            mismatched = [i for i, (a, b) in enumerate(zip(counts, logged)) if a != b]
            row["all_calls"] = {"calls": len(events),
                                "rebuild_equals_raw_text": sum(a == b for a, b in zip(texts, raw_texts)),
                                "rebuild_checked_against_raw": len(raw_texts),
                                "count_equal": len(events) - len(mismatched), "mismatched_calls": mismatched,
                                "max_abs_diff": max((abs(a - b) for a, b in zip(counts, logged)), default=0),
                                "prefix_ok": sum(texts[i + 1].startswith(texts[i]) for i in range(len(texts) - 1)),
                                "prefix_pairs": max(len(texts) - 1, 0),
                                "prefix_broken_at": [i for i in range(len(texts) - 1)
                                                     if not texts[i + 1].startswith(texts[i])]}
            rows.append(row)
            print(json.dumps(row), flush=True)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = HERE / "results" / f"reader_fidelity_counts_{stamp}.json"
    with open(out, "x") as f:
        json.dump({"run_dir": str(args.run_dir), "tokenizer": QWEN35_27B, "rows": rows}, f, indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
