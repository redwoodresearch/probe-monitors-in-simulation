"""Checks (a) and (a') for the repetition guard, Inspect's side, around check_repetition_cut.py (vLLM's side, on
Modal). Laptop, no spend:

  uv run python experiments/2026-10-07_phase2_attack_check/analyze_repetition_cut.py prepare <run dir> <calls.json>
  ... modal run check_repetition_cut.py --calls <calls.json> [--setting ...]
      ->  data/phase2_attack_check/repetition_cut_parse_<stamp>.json (the full parse; results/ gets it without text)
  uv run python experiments/2026-10-07_phase2_attack_check/analyze_repetition_cut.py compare <run dir> <full parse> \\
      [<the run's token log .jsonl.gz>]   (with --with transformers==5.19.0 --with jinja2, for check (a'))

prepare: every policy call in the run whose completion reached the 32,768-token cap, with the tools and tool_choice
its raw request sent (kept by --log-model-api) and the next policy call's request ID, keyed by Inspect's request ID.
compare: both of vLLM's parses (full and cut at the guard's stop) taken through Inspect's own conversion (the vLLM
provider's chat_choices_from_completion and model_output_from_openai, with the call's recorded tools). The full one
must reproduce the message the eval log recorded, or the check proves nothing. Then either the cut gives the same
assistant message as the full output (check (a), the whitespace tails), or it gives the same action and text with its
reasoning a prefix of the full one's, and the next turn's prompt is the same token for token whether the turn was
cut or not (check (a'), a reasoning loop). That last part re-renders the next call's raw request (kept by
--log-model-api) with the policy's own chat template: as sent, it must reproduce the prompt token IDs the proxy logged,
and with the turn's reasoning replaced by the cut one's it must give the same IDs. Writes
results/repetition_cut_<stamp>.json; exits nonzero unless every call passes.
"""

import copy
import gzip
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from inspect_ai.event import ModelEvent
from inspect_ai.log import read_eval_log

HERE = Path(__file__).parent
CAP = 32768
POLICY = "vllm/Qwen/Qwen3.5-27B"  # config/run.yaml `policy`; a simulated sample also holds the simulator's calls


def capped_calls(run_dir: Path) -> dict:
    """Request ID -> (model event, its raw request, the next policy call's raw request or None) for every policy call
    that ran to the cap."""
    found = {}
    for path in sorted(run_dir.glob("*/*/*.eval")):
        for sample in read_eval_log(str(path), resolve_attachments=True).samples or []:
            policy = [e for e in sample.events if isinstance(e, ModelEvent) and e.model == POLICY and e.call]
            for k, e in enumerate(policy):
                if e.output and e.output.usage and e.output.usage.output_tokens >= CAP:
                    nxt = policy[k + 1].call.request if k + 1 < len(policy) else None
                    found[(e.call.request.get("extra_headers") or {})["x-irid"]] = (e, e.call.request, nxt)
    return found


def render(request: dict, tokenizer) -> list[int]:
    """A raw chat request as vLLM 0.30 renders it with the model's template: the tools as sent, each tool call's
    arguments as a dict (vLLM parses the JSON string before templating). Checked against the logged prompt below."""
    messages = copy.deepcopy(request["messages"])
    for m in messages:
        for tc in m.get("tool_calls") or []:
            if isinstance(tc["function"]["arguments"], str):
                tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
    return list(tokenizer.apply_chat_template(messages, tools=request.get("tools"), add_generation_prompt=True,
                                              tokenize=True)["input_ids"])


def next_prompt_check(next_request: dict, full_reasoning: str, cut_reasoning: str, logged: list[int]) -> dict:
    """The next turn's prompt as sent and with the capped turn's reasoning replaced by the cut's: the capped turn is
    the assistant message whose reasoning is the full output's."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B", revision="fc05daec18b0a78c049392ed2e771dde82bdf654")
    turns = [i for i, m in enumerate(next_request["messages"])
             if m["role"] == "assistant" and (m.get("reasoning") or "").strip() == full_reasoning.strip()]
    if len(turns) != 1:
        return {"capped_turn_found": len(turns), "full_reproduces_logged": False, "cut_same_as_full": False}
    full = render(next_request, tokenizer)
    cut_request = copy.deepcopy(next_request)
    cut_request["messages"][turns[0]]["reasoning"] = cut_reasoning
    return {"capped_turn_found": 1, "logged_tokens": len(logged), "full_reproduces_logged": full == logged,
            "cut_same_as_full": render(cut_request, tokenizer) == full}


def described(key: dict) -> dict:
    """A message key with its text replaced by lengths and sha256s (results/ holds no model text)."""
    sha = lambda t: hashlib.sha256(t.encode()).hexdigest()  # noqa: E731
    return {"tool_calls": [{"name": n, "arguments_sha256": sha(json.dumps(a, sort_keys=True))} for n, a in key["tool_calls"]],
            "text_chars": len(key["text"]), "text_sha256": sha(key["text"]),
            "reasoning_chars": [len(r) for r in key["reasoning"]]}


def message_key(message) -> dict:
    """What decides the episode: the tool calls (name and arguments), the text and the reasoning."""
    reasoning = [c.reasoning for c in message.content if c.type == "reasoning"] if isinstance(message.content, list) \
        else []
    return {"tool_calls": [(tc.function, tc.arguments) for tc in message.tool_calls or []],
            "text": message.text, "reasoning": reasoning}


def main() -> None:
    mode, run_dir, path = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    calls = capped_calls(run_dir)
    if mode == "prepare":
        with open(path, "x") as f:
            json.dump({i: {"tools": req.get("tools"), "tool_choice": req.get("tool_choice"),
                           "next_request_id": nxt and (nxt.get("extra_headers") or {})["x-irid"]}
                       for i, (_, req, nxt) in calls.items()}, f)
        print(f"{len(calls)} capped calls -> {path}")
        return
    from inspect_ai.model import get_model
    from inspect_ai.model._openai import model_output_from_openai
    from openai.types.chat import ChatCompletion

    os.environ.setdefault("VLLM_BASE_URL", "http://unused/v1")  # never contacted: only the response conversion runs
    api = get_model("vllm/Qwen/Qwen3.5-27B").api
    parsed = json.loads(path.read_text())
    token_log = {}
    if len(sys.argv) > 4:  # the next turns' logged prompts, for check (a')
        with gzip.open(sys.argv[4], "rt") as f:
            token_log = {r["request_id"]: r for r in map(json.loads, f) if r.get("request_id")}
    rows, ok = {}, True
    for irid, row in parsed["calls"].items():
        event, _, next_request = calls[irid]

        def inspect_message(version: dict):
            message = {"role": "assistant", "content": version["content"], "reasoning": version["reasoning"],
                       "tool_calls": [{"id": f"call_{k}", "type": "function", "function": tc}
                                      for k, tc in enumerate(version["tool_calls"])]}
            completion = ChatCompletion.model_validate({
                "id": "x", "object": "chat.completion", "created": 0, "model": "Qwen/Qwen3.5-27B",
                "choices": [{"index": 0, "message": message,
                             "finish_reason": "tool_calls" if version["tool_calls"] else "stop"}]})
            return model_output_from_openai(completion, api.chat_choices_from_completion(completion, event.tools))

        if row["cut"] is None:  # the guard never fires on this call: nothing to compare, and the check fails
            rows[irid] = {"stop_at": None, "tokens": row["tokens"], "pass": False}
            ok = False
            continue
        full, cut = inspect_message(row["full"]), inspect_message(row["cut"])
        recorded, f, c = message_key(event.output.message), message_key(full.message), message_key(cut.message)
        nxt = row.get("next_call") or {}
        rendered = None
        if next_request and token_log:
            rendered = next_prompt_check(next_request, (row["full"]["reasoning"] or ""), (row["cut"]["reasoning"] or ""),
                                         token_log[(next_request.get("extra_headers") or {})["x-irid"]]["prompt_token_ids"])
        result = {"stop_at": row["stop_at"], "tokens": row["tokens"], "cut_equals_full": c == f,
                  "same_action_and_text": (c["tool_calls"], c["text"]) == (f["tool_calls"], f["text"]),
                  "cut_reasoning_is_prefix": len(c["reasoning"]) == len(f["reasoning"]) and all(
                      fr.startswith(cr.rstrip()) for cr, fr in zip(c["reasoning"], f["reasoning"])),
                  "full_reproduces_recorded": f == recorded,
                  "next_prompt_holds_none_of_the_turn": bool(nxt) and not nxt["chunks_found_before_cut"]
                  and not nxt["chunks_found_after_cut"], "next_prompt_rendered": rendered,
                  "stop_reasons": [full.stop_reason, cut.stop_reason],
                  "next_call": {k: v for k, v in nxt.items() if k != "tail"},
                  "recorded": described(recorded), "cut": described(c)}
        result["pass"] = result["full_reproduces_recorded"] and (result["cut_equals_full"] or (
            result["same_action_and_text"] and result["cut_reasoning_is_prefix"] and rendered is not None
            and rendered["full_reproduces_logged"] and rendered["cut_same_as_full"]))
        ok &= result["pass"]
        rows[irid] = result
        print(irid, {k: result[k] for k in ("stop_at", "cut_equals_full", "same_action_and_text",
                                             "cut_reasoning_is_prefix", "full_reproduces_recorded",
                                             "next_prompt_holds_none_of_the_turn", "next_prompt_rendered", "pass")},
              "tool calls:", [t["name"] for t in result["cut"]["tool_calls"]])
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = HERE / "results" / f"repetition_cut_{stamp}.json"
    with open(out, "x") as f:
        json.dump({"parse": str(path), "calls": rows, "pass": ok}, f, indent=1)
    print(f"-> {out}")
    if not ok:
        raise SystemExit("check (a) failed")


if __name__ == "__main__":
    main()
