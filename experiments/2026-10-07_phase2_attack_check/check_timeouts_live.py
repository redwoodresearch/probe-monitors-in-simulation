"""Check the policy's call limits on the real server before the reruns: Aswin's condition for launching them.

  repetition: (decision D) one short call that sets repetition_detection with min_count 2, so any token repeated once
        stops it. The proxy's /proxy/stop_reasons must return stop_reason "repetition_detected" for that call's
        request ID: the path run_attack_check.py counts D's stops by.
  tiny: one call with a 15 s attempt timeout and a prompt that takes minutes to answer. It must fail once with
        AttemptTimeoutError, with exactly one HTTP request sent, and the proxy must record its request ID as cancelled
        (run_vllm_tunnel's app log prints the line).
  newline: (the pilot's server-start check) a tool output with and without a trailing newline must tokenize to the
        same IDs through the server's own chat template (/tokenize): Qwen3.5's template trims every message, tool
        outputs included, so today's simulator's leftover str_replace newline never reaches the policy. A tool output
        with one character changed must tokenize differently (the control).
  long: one call that runs past the OpenAI SDK's 600 s read timeout, under the runs' 1,800 s attempt timeout, with a
        45,000-token cap for this call only. It must finish as one HTTP request. This is the resend the patch fixes; the
        tiny call can't show it, because the SDK's own timeout only cuts in after 600 s.

Runs ON THE DEVBOX, with the runs' own code and environment (VLLM_API_KEY from rr-env, never argv):

  cd ~/phase2/sims-sept-policy-limits && PYTHONPATH=$PWD/src VLLM_BASE_URL=https://<tunnel>/v1 \\
      ~/sims/simulated-control-evals-sept/.venv/bin/python ~/phase2/attack_check/check_timeouts_live.py tiny

Takes one or more cases, run in order (the pilot's server-start set: `tiny repetition newline`; run_pilot.py runs it on
each server before launching anything). Writes results/timeout_check_<case>_<stamp>.json per case, and --summary
<path> if given; exits nonzero unless every case passes (INCONCLUSIVE included).
"""

import argparse
import asyncio
import json
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import runner.policy.agent as agent
import yaml
from inspect_ai.model import ChatMessageUser
from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI

HERE = Path(__file__).parent
CASES = {
    "tiny": {"attempt_timeout": 15, "max_tokens": 32768,
             "prompt": "Write a detailed 5,000-word history of the Roman road network."},
    "long": {"attempt_timeout": 1800, "max_tokens": 45000,
             "prompt": "Write out every integer from 1 to 30000 in English words, one per line, with no omissions, "
                       "abbreviations or commentary."},
    # The first live `long` call stopped on its own after 3,987 tokens (77 s), too soon to test anything; this asks for
    # ~45k tokens of plain digits instead.
    "long_digits": {"attempt_timeout": 1800, "max_tokens": 45000,
                    "prompt": "Print every integer from 1 to 15000, one per line, as digits only. Print nothing else: "
                              "no code, no ranges, no ellipses, no commentary, and do not stop before 15000."},
}
SENT: Counter = Counter()  # HTTP requests the client actually sent, per Inspect request ID (x-irid)

_create_http_client = OpenAICompatibleAPI._create_http_client


def _counting_http_client(self):
    """Inspect's own client, with a hook that counts every request it sends (including the OpenAI SDK's resends)."""
    client = _create_http_client(self)

    async def count(request: httpx.Request) -> None:
        SENT[request.headers.get("x-irid")] += 1

    client.event_hooks = {"request": [*client.event_hooks["request"], count], "response": client.event_hooks["response"]}
    return client


OpenAICompatibleAPI._create_http_client = _counting_http_client


async def repetition_check(policy: str) -> dict:
    """One short call with a hair-trigger repetition_detection, then its stop reason from the proxy."""
    import urllib.request

    from inspect_ai.model import GenerateConfig

    os.environ |= {"PT_POLICY_ATTEMPT_TIMEOUT": "1800", "PT_POLICY_MAX_RETRIES": "2", "PT_POLICY_MAX_TOKENS": "200"}
    model = agent.build_policy_model(policy)
    out = await model.generate([ChatMessageUser(content="Reply with exactly: yes yes yes yes yes yes yes yes")],
                               config=GenerateConfig(extra_body={"repetition_detection": {
                                   "min_pattern_size": 1, "max_pattern_size": 1, "min_count": 2}}))
    ids = sorted(SENT)
    request = urllib.request.Request(f"{os.environ['VLLM_BASE_URL'].removesuffix('/v1')}/proxy/stop_reasons",
                                     data=json.dumps({"request_ids": ids}).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {os.environ['VLLM_API_KEY']}",
                                              "Content-Type": "application/json"})
    reasons = json.loads(urllib.request.urlopen(request, timeout=60).read())
    stopped = [i for i in ids if any("repetition_detected" in r["stop_reasons"] for r in reasons.get(i, []))]
    return {"inspect_stop_reason": out.stop_reason, "output_tokens": out.usage.output_tokens if out.usage else None,
            "http_requests_by_request_id": dict(SENT), "proxy_stop_reasons": reasons,
            "verdict": "PASS" if ids and stopped == ids else "FAIL"}


def newline_check(policy: str) -> dict:
    """The same short exchange tokenized by the server with three tool outputs: two that differ only in a trailing
    newline (must match) and one with a character changed (must not: /tokenize really reads the tool output)."""
    import urllib.request

    def tokens(output: str) -> list[int]:
        messages = [{"role": "user", "content": "List the files."},
                    {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {
                        "name": "bash", "arguments": json.dumps({"cmd": "ls"})}}]},
                    {"role": "tool", "tool_call_id": "call_1", "content": output}]
        body = {"model": policy.removeprefix("vllm/"), "messages": messages, "add_generation_prompt": True}
        request = urllib.request.Request(f"{os.environ['VLLM_BASE_URL'].removesuffix('/v1')}/tokenize",
                                         data=json.dumps(body).encode(), method="POST",
                                         headers={"Authorization": f"Bearer {os.environ['VLLM_API_KEY']}",
                                                  "Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(request, timeout=60).read())["tokens"]

    with_newline, without, changed = tokens("a.txt\nb.txt\n"), tokens("a.txt\nb.txt"), tokens("a.txt\nc.txt")
    return {"tokens": {"with_newline": len(with_newline), "without": len(without), "changed": len(changed)},
            "newline_identical": with_newline == without, "control_differs": changed != without,
            "verdict": "PASS" if with_newline == without and changed != without else "FAIL"}


async def run_case(name: str, policy: str) -> dict:
    """One check against the server; writes results/timeout_check_<case>_<stamp>.json and returns its record."""
    SENT.clear()  # each case counts only its own HTTP requests
    if name in ("repetition", "newline"):
        check = await repetition_check(policy) if name == "repetition" else newline_check(policy)
        record = {"case": name, "started": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(
            timespec="seconds")} | check
    else:
        case = CASES[name]
        os.environ |= {"PT_POLICY_ATTEMPT_TIMEOUT": str(case["attempt_timeout"]), "PT_POLICY_MAX_RETRIES": "2",
                       "PT_POLICY_MAX_TOKENS": str(case["max_tokens"])}
        model = agent.build_policy_model(policy)
        record = {"case": name, "started": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds"),
                  "policy": policy, "policy_imported_from": agent.__file__, "limits": case | {"prompt": None},
                  "prompt": case["prompt"], "client_timeout": getattr(model.api, "_client_timeout", None)}
        t0 = time.time()
        try:
            out = await model.generate([ChatMessageUser(content=case["prompt"])])
            record["result"] = {"completed": True, "stop_reason": out.stop_reason,
                                "output_tokens": out.usage.output_tokens if out.usage else None}
        except Exception as e:  # noqa: BLE001 — the tiny case's expected outcome; recorded and judged below
            record["result"] = {"completed": False, "error": f"{type(e).__name__}: {e}"[:500]}
        record |= {"seconds": round(time.time() - t0, 1), "http_requests_by_request_id": dict(SENT)}
        sent = sum(SENT.values())
        if name == "tiny":
            passed = not record["result"]["completed"] and "AttemptTimeoutError" in record["result"]["error"] \
                and sent == 1
            record["verdict"] = "PASS" if passed else "FAIL"
        else:
            long_enough = record["seconds"] > agent.OPENAI_SDK_READ_TIMEOUT
            passed = record["result"]["completed"] and sent == 1
            record["verdict"] = ("PASS" if passed and long_enough else "INCONCLUSIVE (finished before 600 s)"
                                 if passed else "FAIL")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out_path = HERE / "results" / f"timeout_check_{name}_{stamp}.json"
    with open(out_path, "x") as f:
        json.dump(record, f, indent=1)
    print(json.dumps({k: v for k, v in record.items() if k != "prompt"}), flush=True)
    print(f"-> {out_path}", flush=True)
    return record | {"file": str(out_path)}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="+", choices=sorted(CASES) + ["repetition", "newline"],
                    help="run in order; the pilot's server-start set is: tiny repetition newline")
    ap.add_argument("--summary", type=Path, help="also write {case: verdict, file} here (run_pilot.py reads it)")
    ap.add_argument("--policy", help="the served policy model (default: config/run.yaml's)")
    args = ap.parse_args()
    policy = args.policy or yaml.safe_load((HERE / "config" / "run.yaml").read_text())["policy"]
    records = [await run_case(name, policy) for name in args.cases]
    if args.summary:
        args.summary.write_text(json.dumps({r["case"]: {"verdict": r["verdict"], "file": r["file"]} for r in records},
                                           indent=1))
    failed = [f"{r['case']}: {r['verdict']}" for r in records if r["verdict"] != "PASS"]
    if failed:
        raise SystemExit("; ".join(failed))


if __name__ == "__main__":
    asyncio.run(main())
