"""Smoke-test the tunnel endpoint before the attack check launches: auth, tool calls, long context, decode speed,
and the exact Inspect path the policy takes (`vllm/<model>` with reasoning_effort=medium, via build_policy_model).
Copied from experiments/2026-10-06_phase2_feasibility/run_smoke.py (PR #50); only the provenance lines differ (the
devbox copy has no git, so the commit is passed in and the script's sha256 recorded).

Runs ON THE DEVBOX with the September snapshot's Python (it needs inspect_ai and runner), the key from rr-env:
  cd ~/sims/simulated-control-evals-sept && RR_OWNER_ID=aswin-surya rr-env run \
    --env-file ~/phase2/attack_check/config/env.aws.tpl -- env VLLM_BASE_URL=https://<tunnel>/v1 \
    .venv/bin/python ~/phase2/attack_check/run_smoke.py --model Qwen/Qwen3.5-27B --commit <probe-shift commit>
"""

import argparse
import asyncio
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
from openai import AsyncOpenAI

HERE = Path(__file__).parent
BASH_TOOL = {"type": "function", "function": {
    "name": "bash", "description": "Run a bash command in the sandbox and return its output.",
    "parameters": {"type": "object", "properties": {"cmd": {"type": "string", "description": "The command."}},
                   "required": ["cmd"]}}}


async def auth_checks(base: str, key: str) -> dict:
    """Requests without the key, or with a wrong one, must be refused."""
    # Modal answers a request still pending after 150 s with a 303 to its result URL; following it waits on the same request.
    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
        t0 = time.time()  # a cold start (image, weights from the volume, CUDA graphs) holds this request
        health = await client.get(base.removesuffix("/v1") + "/health", timeout=1800)
        health.raise_for_status()
        print(f"server ready after {time.time() - t0:.0f} s", flush=True)
        no_key = await client.get(f"{base}/models")
        wrong = await client.get(f"{base}/models", headers={"Authorization": "Bearer wrong"})
        right = await client.get(f"{base}/models", headers={"Authorization": f"Bearer {key}"})
        version = await client.get(base.removesuffix("/v1") + "/version")
    assert (no_key.status_code, wrong.status_code, right.status_code) == (401, 401, 200), (
        f"auth is not enforced as expected: no key {no_key.status_code}, wrong key {wrong.status_code}, "
        f"right key {right.status_code}")
    return {"no_key_status": no_key.status_code, "wrong_key_status": wrong.status_code,
            "right_key_status": right.status_code, "models": [m["id"] for m in right.json()["data"]],
            "vllm_version": version.json() if version.status_code == 200 else version.status_code}


def parses_as_object(arguments: str) -> bool:
    """False (recorded, then failed after saving) rather than an exception that would lose the evidence."""
    try:
        return isinstance(json.loads(arguments), dict)
    except json.JSONDecodeError:
        return False


async def tool_round_trip(client: AsyncOpenAI, model: str) -> dict:
    """One tool call, the tool's answer, then the model's next turn."""
    messages = [{"role": "system", "content": "You are an agent in a Linux sandbox. Use the bash tool to act."},
                {"role": "user", "content": "How many lines does /etc/passwd have? Use the bash tool."}]
    t0 = time.time()
    first = await client.chat.completions.create(model=model, messages=messages, tools=[BASH_TOOL])
    t1 = time.time()
    msg = first.choices[0].message
    calls = msg.tool_calls or []
    out = {"first_request": list(messages), "first_response": first.model_dump(), "first_seconds": round(t1 - t0, 2),
           "tool_calls": [{"name": c.function.name, "arguments": c.function.arguments} for c in calls],
           "args_parse": [parses_as_object(c.function.arguments) for c in calls]}
    if calls:
        messages += [msg.model_dump(exclude_none=True),
                     {"role": "tool", "tool_call_id": calls[0].id, "content": "41 /etc/passwd\n"}]
        second = await client.chat.completions.create(model=model, messages=messages, tools=[BASH_TOOL])
        out |= {"second_request": messages, "second_response": second.model_dump()}
    return out


async def long_context(client: AsyncOpenAI, model: str, target_tokens: int) -> dict:
    """A prompt past the native 32k window, to show the server's max length is real, with a fact to recall.
    The filler sentence is 10 Qwen tokens (no digits: Qwen splits every digit into its own token)."""
    filler = "The quick brown fox jumps over the lazy dog. " * (target_tokens // 10)
    prompt = f"The secret word is PELICAN.\n{filler}\nWhat is the secret word? Answer with one word."
    t0 = time.time()
    r = await client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], max_tokens=2048)
    return {"target_tokens": target_tokens, "prompt_tokens": r.usage.prompt_tokens,
            "answer": r.choices[0].message.content, "seconds": round(time.time() - t0, 2), "response": r.model_dump()}


async def decode_speed(client: AsyncOpenAI, model: str) -> dict:
    t0 = time.time()
    r = await client.chat.completions.create(model=model, max_tokens=1024, messages=[
        {"role": "user", "content": "Write a 600-word essay on why unit tests matter. /no_think"}])
    dt = time.time() - t0
    return {"completion_tokens": r.usage.completion_tokens, "seconds": round(dt, 2),
            "tokens_per_s": round(r.usage.completion_tokens / dt, 1), "response": r.model_dump()}


async def inspect_policy_path(model: str) -> dict:
    """The model exactly as sce's real arena policy builds it, with a bash tool."""
    from inspect_ai.model import ChatMessageSystem, ChatMessageUser
    from inspect_ai.tool import ToolInfo, ToolParam, ToolParams
    from runner.policy.agent import build_policy_model
    from runner.simulate.model import resolve_model_spec

    name = resolve_model_spec(f"vllm/{model}").model_name
    policy = build_policy_model(name)
    tool = ToolInfo(name="bash", description="Run a bash command.",
                    parameters=ToolParams(properties={"cmd": ToolParam(type="string", description="The command.")},
                                          required=["cmd"]))
    out = await policy.generate([ChatMessageSystem(content="You are an agent in a Linux sandbox."),
                                 ChatMessageUser(content="List the files in /tmp with the bash tool.")], tools=[tool])
    msg = out.message
    return {"resolved_name": name, "stop_reason": out.stop_reason,
            "reasoning_blocks": sum(1 for c in (msg.content if isinstance(msg.content, list) else []) if c.type == "reasoning"),
            "tool_calls": [{"function": c.function, "arguments": c.arguments, "parse_error": c.parse_error}
                           for c in msg.tool_calls or []],
            "config": {"reasoning_effort": policy.config.reasoning_effort, "attempt_timeout": policy.config.attempt_timeout},
            "output": out.model_dump()}


def failed_checks(result: dict) -> list[str]:
    """The names of the functional checks a completed smoke run failed (empty when it passed)."""
    tool, long, policy = result["tool_round_trip"], result["long_context"], result["inspect_policy_path"]
    return [name for name, ok in {
        "tool call parsed": bool(tool["tool_calls"]) and all(tool["args_parse"]),
        "second turn after the tool output": "second_response" in tool,
        "long-context recall": "PELICAN" in (long["answer"] or ""),
        "long prompt past 32k": long["prompt_tokens"] > 32768,
        "inspect path: tool call without parse error": bool(policy["tool_calls"])
            and all(c["parse_error"] is None for c in policy["tool_calls"]),
    }.items() if not ok]


async def main(model: str, long_tokens: int, result: dict) -> None:
    """Fills `result` stage by stage, so a failure still leaves every finished stage to save."""
    base, key = os.environ["VLLM_BASE_URL"].rstrip("/"), os.environ["VLLM_API_KEY"]
    result |= {"model": model, "base_url": base}
    client = AsyncOpenAI(base_url=base, api_key=key, timeout=1800)
    result["auth"] = await auth_checks(base, key)
    result["decode_speed"] = await decode_speed(client, model)  # alone, so it measures one stream
    stages = {"tool_round_trip": tool_round_trip(client, model),
              "long_context": long_context(client, model, long_tokens),
              "inspect_policy_path": inspect_policy_path(model)}
    outs = await asyncio.gather(*stages.values(), return_exceptions=True)
    result |= {k: {"error": repr(o)} if isinstance(o, BaseException) else o for k, o in zip(stages, outs)}
    errors = [o for o in outs if isinstance(o, BaseException)]
    if errors:
        raise errors[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--long-tokens", type=int, default=60000, help="filler length; the server allows 131,072")
    ap.add_argument("--commit", required=True, help="the probe-shift commit this folder was copied from")
    args = ap.parse_args()
    (HERE / "results").mkdir(exist_ok=True)
    result = {"git_commit": args.commit,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    error = None
    try:
        asyncio.run(main(args.model, args.long_tokens, result))
    except Exception as e:  # noqa: BLE001 — saved below with every finished stage, then re-raised
        error, result["error"] = e, repr(e)
    if error is None:
        # Judged before saving, so the saved file says which checks failed, not only the README.
        result["failed_checks"] = failed_checks(result)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"smoke_{args.model.split('/')[-1]}_{stamp}.json"
    with open(path, "x") as f:
        f.write(json.dumps(result, indent=1, default=str))
    if error:
        raise error
    summary = {"auth": result["auth"], "decode_speed": {k: v for k, v in result["decode_speed"].items() if k != "response"},
               "tool_calls": result["tool_round_trip"]["tool_calls"], "args_parse": result["tool_round_trip"]["args_parse"],
               "long_context": {k: v for k, v in result["long_context"].items() if k != "response"},
               "inspect": {k: v for k, v in result["inspect_policy_path"].items() if k != "output"}}
    print(json.dumps(summary, indent=1, default=str))
    print(f"-> {path}")
    if result["failed_checks"]:  # a smoke test that records a failure and exits 0 is not a check
        raise SystemExit(f"smoke checks failed: {result['failed_checks']}")
