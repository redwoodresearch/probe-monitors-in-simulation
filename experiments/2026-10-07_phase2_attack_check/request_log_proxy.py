"""Log the tokens of every chat request the policy sends to vLLM. A reverse proxy inside the vLLM container, between
the tunnel and vLLM; run_vllm_tunnel.py starts it:

  python request_log_proxy.py --port 8000 --upstream http://127.0.0.1:8001 --log /logs/<name>.jsonl.gz --header '{...}'

The eval log keeps each call's messages and parsed output, not its tokens. For each POST /v1/chat/completions this asks
vLLM for the prompt and completion token IDs (`return_token_ids`) and each sampled token's logprob (`logprobs`, no
alternatives), appends one record to the log, then clears those fields so Inspect gets the response it would have got
without the proxy. Records are keyed by Inspect's request ID (the x-irid header). The OpenAI SDK's own resends reuse
an ID, so one ID can have several records; a request the client abandons (a timeout) is recorded as `cancelled`, and
vLLM aborts it when the connection closes. Every other path passes through untouched.

`--inject` adds request fields to every chat request that doesn't set them itself (since 2026-10-07, vLLM's
`repetition_detection`, Aswin's decision D); each record lists what was injected, and vLLM's own `stop_reason` per
choice, which is the only place a repetition stop shows ("repetition_detected": vLLM still reports finish_reason
`tool_calls` when it parsed a call). `POST /proxy/stop_reasons` with {"request_ids": [...]} returns, per ID, each
record's status, finish and stop reasons and whether its token record is complete, so the launcher can count those
stops per arm and tool and check that every policy call has its token record; it needs the same bearer key as vLLM.

The log is gzip members appended one per record (`gzip.open(path, "rt")` reads it as one JSONL stream). Its first record
is `--header`. Prompt token IDs are stored in full on every record, not once per context epoch: at ~30k prompt tokens a
call that is ~60 KB compressed, ~6 MB for a 100-call run.
"""

import argparse
import gzip
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import aiohttp
from aiohttp import web

CHAT = "/v1/chat/completions"
REQUEST_ID = "x-irid"  # inspect_ai.model._providers.util.hooks.HttpxHooks.REQUEST_ID_HEADER
HOP_BY_HOP = {"host", "content-length", "transfer-encoding", "connection", "keep-alive", "accept-encoding"}


def now() -> str:
    return datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="milliseconds")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def with_token_fields(body: dict) -> tuple[dict, set[str]]:
    """The request with token IDs and sampled-token logprobs turned on, and which of the two the client hadn't asked for
    itself (only those are cleared from the response)."""
    added = {k for k in ("return_token_ids", "logprobs") if not body.get(k)}
    sent = body | dict.fromkeys(added, True)
    if "logprobs" in added:
        sent["top_logprobs"] = 0  # the sampled token's own logprob, no alternatives
    return sent, added


def logged_and_cleared(response: dict, added: set[str]) -> tuple[dict, dict]:
    """What the record keeps from vLLM's response, and the response with the fields the client didn't ask for cleared
    (set to null, as vLLM sends them when not asked)."""
    prompt_ids = response.get("prompt_token_ids")
    choices = [{"finish_reason": c.get("finish_reason"), "stop_reason": c.get("stop_reason"),
                "token_ids": c.get("token_ids"),
                "logprobs": [x["logprob"] for x in ((c.get("logprobs") or {}).get("content") or [])]}
               for c in response["choices"]]
    record = {"response_id": response.get("id"), "usage": response.get("usage"), "prompt_token_ids": prompt_ids,
              "choices": choices,
              # Every completion token has its logprob, and the prompt IDs came back; anything else is a logging gap.
              "complete": prompt_ids is not None and all(
                  c["token_ids"] is not None and len(c["token_ids"]) == len(c["logprobs"]) for c in choices)}
    if "return_token_ids" in added:
        response["prompt_token_ids"] = None
        for c in response["choices"]:
            c["token_ids"] = None
    if "logprobs" in added:
        for c in response["choices"]:
            c["logprobs"] = None
    return record, response


def make_app(upstream: str, log: Path, inject: dict | None = None,
             connector=lambda: aiohttp.TCPConnector(limit=0)) -> web.Application:
    # aiohttp refuses request bodies over 1 MB by default; a 100k-token conversation is several MB of JSON.
    app = web.Application(client_max_size=1 << 30)

    reasons: dict[str, list[dict]] = {}  # request ID -> each of its records' outcome, for /proxy/stop_reasons

    def write(record: dict) -> None:
        # Synchronous on purpose: one event loop, so records never interleave inside the file.
        with gzip.open(log, "at") as f:
            f.write(json.dumps(record) + "\n")
        reasons.setdefault(record.get("request_id"), []).append(
            {"status": record.get("status"), "received": record.get("received"),
             "finish_reasons": [c.get("finish_reason") for c in record.get("choices") or []],
             "stop_reasons": [c.get("stop_reason") for c in record.get("choices") or []],
             "complete": record.get("complete")})

    def forwarded(request: web.Request) -> dict:
        return {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}

    async def session_ctx(app: web.Application):
        # No client-side timeout: Inspect's attempt timeout decides when a call ends, by closing its connection.
        app["session"] = aiohttp.ClientSession(connector=connector(), timeout=aiohttp.ClientTimeout(total=None))
        yield
        await app["session"].close()

    async def chat(request: web.Request) -> web.Response:
        raw = await request.read()
        body = json.loads(raw)
        record = {"request_id": request.headers.get(REQUEST_ID), "received": now(), "request_sha256": sha256(raw),
                  "params": {k: v for k, v in body.items() if k not in ("messages", "tools")},
                  "messages": len(body.get("messages") or []),
                  "tools_sha256": sha256(json.dumps(body.get("tools"), sort_keys=True).encode())}
        sent, added = (body, set()) if body.get("stream") else with_token_fields(body)
        injected = {k: v for k, v in (inject or {}).items() if k not in body}
        sent = sent | injected
        record["injected"] = sorted(injected)
        if body.get("stream"):  # Inspect never streams to vLLM; passed through so a run can't break, and recorded
            record["status_note"] = "streamed: passed through unchanged, tokens not logged"
        t0 = time.monotonic()
        try:
            headers = forwarded(request) | {"Content-Type": "application/json"}
            async with request.app["session"].post(f"{upstream}{CHAT}", data=json.dumps(sent).encode(),
                                                   headers=headers) as up:
                data, status = await up.read(), up.status
        except BaseException as e:  # the client went away (CancelledError) or vLLM did: record it, then re-raise
            status = "cancelled" if not isinstance(e, Exception) else f"upstream error: {e!r}"[:500]
            write(record | {"status": status, "seconds": round(time.monotonic() - t0, 3), "finished": now()})
            print(f"{now()} {record['request_id']} {status} after {time.monotonic() - t0:.1f}s", flush=True)
            raise
        record |= {"status": status, "seconds": round(time.monotonic() - t0, 3), "finished": now()}
        if status == 200 and not body.get("stream"):
            logged, response = logged_and_cleared(json.loads(data), added)
            record |= logged
            data = json.dumps(response).encode()
        elif status != 200:
            record["error_body"] = data[:2000].decode(errors="replace")
        write(record)
        n_prompt = len(record.get("prompt_token_ids") or [])
        n_out = sum(len(c["token_ids"] or []) for c in record.get("choices", []))
        print(f"{record['finished']} {record['request_id']} {status} prompt={n_prompt} completion={n_out} "
              f"{record['seconds']:.1f}s{'' if record.get('complete', True) else ' INCOMPLETE'}", flush=True)
        return web.Response(body=data, status=status, headers={"Content-Type": "application/json"})

    async def stop_reasons(request: web.Request) -> web.Response:
        if request.headers.get("Authorization") != f"Bearer {os.environ['VLLM_API_KEY']}":
            return web.json_response({"error": "unauthorized"}, status=401)
        ids = (await request.json())["request_ids"]
        return web.json_response({i: reasons.get(i, []) for i in ids})

    async def passthrough(request: web.Request) -> web.Response:
        async with request.app["session"].request(request.method, f"{upstream}{request.rel_url.path_qs}",
                                                  data=await request.read(), headers=forwarded(request)) as up:
            keep = {k: v for k, v in up.headers.items() if k.lower() == "content-type"}  # the body arrives decoded
            return web.Response(body=await up.read(), status=up.status, headers=keep)

    app.cleanup_ctx.append(session_ctx)
    app.router.add_post(CHAT, chat)
    app.router.add_post("/proxy/stop_reasons", stop_reasons)
    app.router.add_route("*", "/{tail:.*}", passthrough)
    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--upstream", required=True, help="vLLM's base URL, without /v1")
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--header", default="{}", help="JSON object written as the log's first record")
    ap.add_argument("--inject", default="{}", help="JSON object of fields added to each chat request lacking them")
    args = ap.parse_args()
    assert not args.log.exists(), f"{args.log} exists; logs are append-only, one per server start"
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.log, "at") as f:
        f.write(json.dumps({"kind": "header", "written": now(), **json.loads(args.header)}) + "\n")
    # handler_cancellation: when Inspect closes a connection (an attempt timeout), cancel its handler and with it the
    # upstream request, so vLLM aborts the generation instead of finishing it for nobody.
    # A 5 s shutdown grace: a call still running at shutdown is recorded as cancelled, within the server's 60 s wait.
    web.run_app(make_app(args.upstream, args.log, json.loads(args.inject)), host="0.0.0.0", port=args.port,
                handler_cancellation=True,
                shutdown_timeout=5, print=lambda *_: None)


if __name__ == "__main__":
    main()
