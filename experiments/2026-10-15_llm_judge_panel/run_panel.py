"""Ask a panel of LLM judges (OpenRouter) one question with one context file, all judges in parallel.

    uv run --no-project --with httpx python -I experiments/2026-10-15_llm_judge_panel/run_panel.py \
        --question-file Q.md --context-file C.md [--out experiments/2026-10-15_llm_judge_panel/results/panel_<stamp>.jsonl]
    ... run_panel.py --summarize results/panel_<stamp>.jsonl      # only rewrite the verbatim-answers markdown

Judges come from config/judges.json and are checked against OpenRouter's live model list (name match, supported
reasoning effort) before anything is sent; a missing model stops the run. The key is OPENROUTER_API_KEY from the
environment, else from the main checkout's git-ignored .env; it is never printed or stored. Worst case = one billed
attempt per judge: prompt at one token per UTF-8 byte plus 1,000, max_tokens of completion, list price (routing is
capped at list price). Only unbilled errors are retried (429, connection failures); a timeout, 5xx or broken
response is a failed row, never an empty answer. Above --cap-usd nothing is sent. A call whose
sha256(model, params, messages) has an ok row in results/*.jsonl is reused, not re-sent. Writes one JSONL row per
judge as each finishes, a <out>.md with every answer verbatim and no synthesis, and a SPEND.md row.
"""

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

HERE = Path(__file__).resolve().parent
API = "https://openrouter.ai/api/v1"
PT = ZoneInfo("America/Los_Angeles")
SYSTEM = (
    "You are one of several independent expert reviewers (AI safety and AI control, ML evaluation methodology, "
    "statistics) advising a research team. You do not see the other reviewers' answers. Be concrete and skeptical, "
    "disagree with the team's framing where warranted, and say how confident you are. If something in the context "
    "looks inconsistent or under-specified, say so instead of guessing. All results are preliminary unless marked "
    "otherwise."
)


def now() -> str:
    return datetime.now(PT).isoformat(timespec="seconds")


def sha(x) -> str:
    return hashlib.sha256((x if isinstance(x, str) else json.dumps(x, sort_keys=True)).encode()).hexdigest()


def num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=HERE, capture_output=True, text=True, check=True).stdout.strip()


def api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        env = Path(git("rev-parse", "--path-format=absolute", "--git-common-dir")).parent / ".env"
        for line in env.read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip().removeprefix("export ").strip() == "OPENROUTER_API_KEY":
                key = v.strip().strip("'\"")
    if not key or any(c.isspace() or not c.isprintable() for c in key):
        sys.exit("OPENROUTER_API_KEY missing or malformed (value not shown)")
    return key


def price(model: dict, n_prompt: int) -> tuple[float, float, float]:
    """List price: $/prompt token, $/completion token (incl. reasoning), $/request; long-prompt overrides applied."""
    p = dict(model["pricing"])
    for o in p.get("overrides", []):
        if n_prompt >= o["min_prompt_tokens"]:
            p.update(o)
    out = max(float(p["completion"]), float(p.get("internal_reasoning") or 0))
    return float(p["prompt"]), out, float(p.get("request") or 0)


def resolve(cfg: dict) -> list[dict]:
    models = {m["id"]: m for m in httpx.get(f"{API}/models", timeout=60).raise_for_status().json()["data"]}
    out, missing = [], []
    for i, j in enumerate(cfg["judges"]):
        m = models.get(j["slug"])
        efforts = ((m or {}).get("reasoning") or {}).get("supported_efforts", [])
        if m is None or j["name"].lower() not in m["name"].lower() or j["effort"] not in efforts:
            missing.append(
                f"{j['name']} -> {j['slug']}: "
                + ("not listed" if m is None else f"name {m['name']!r}, efforts {efforts}")
            )
            continue
        max_tokens = min(cfg["max_tokens"], m["top_provider"]["max_completion_tokens"] or cfg["max_tokens"])
        out.append({**j, "index": i, "model": m, "max_tokens": max_tokens})
    if missing:
        sys.exit("Judges not resolved on OpenRouter (nothing sent):\n  " + "\n  ".join(missing))
    return out


def cache(results_dir: Path) -> dict:
    hits = {}
    for f in sorted(results_dir.glob("*.jsonl")):
        for n, line in enumerate(f.read_text().splitlines(), 1):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                print(f"cache: skipping unreadable line {f.name}:{n}")
                continue
            ok = isinstance(r, dict) and r.get("status") == "ok" and r.get("finish_reason") == "stop"
            if ok and isinstance(r.get("call_sha256"), str) and str(r.get("response_text") or "").strip():
                hits[r["call_sha256"]] = {**r, "cached_from": r.get("cached_from") or f.name}
    return hits


async def call(client: httpx.AsyncClient, body: dict, timeout_s: float, retries: int, scrub) -> dict:
    """One judge call. Retries only errors that cannot have been billed (429, connection failures)."""
    errors = []
    for attempt in range(retries + 1):
        if attempt:
            await asyncio.sleep(15 * 2 ** (attempt - 1))
        try:
            r = await asyncio.wait_for(client.post(f"{API}/chat/completions", json=body), timeout_s)
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            errors.append(f"attempt {attempt + 1}: {type(e).__name__}")
            continue
        except (TimeoutError, httpx.TimeoutException) as e:
            return {
                "status": "failed",
                "error": f"timeout after {timeout_s:.0f} s ({type(e).__name__}); may be billed",
                "errors": errors,
            }
        except httpx.TransportError as e:
            return {
                "status": "failed",
                "error": f"{type(e).__name__} during the request; may be billed",
                "errors": errors,
            }
        if r.status_code == 429:
            errors.append(f"attempt {attempt + 1}: http 429: {scrub(r.text)[:300]}")
            continue
        if r.status_code != 200:
            return {
                "status": "failed",
                "error": f"http {r.status_code}: {scrub(r.text)[:1000]}"
                + ("; may be billed" if r.status_code >= 500 else ""),
                "errors": errors,
            }
        data = None
        try:
            data = json.loads(r.text)
            choice = data["choices"][0]
            msg = choice.get("message") or {}
            text, finish = msg.get("content") or "", choice.get("finish_reason")
        except (json.JSONDecodeError, KeyError, IndexError, TypeError, AttributeError) as e:
            d = data if isinstance(data, dict) else {}
            return {
                "status": "failed",
                "error": f"unusable 200 body ({type(e).__name__}): {scrub(json.dumps(d['error']) if 'error' in d else r.text)[:1000]}; may be billed",
                "usage": d.get("usage") if isinstance(d.get("usage"), dict) else None,
                "errors": errors,
            }
        if not isinstance(text, str):
            text, finish = json.dumps(text), f"{finish} (content was not a string)"
        ok = bool(text.strip()) and finish == "stop" and "error" not in data
        return {
            "status": "ok" if ok else "failed",
            "error": None
            if ok
            else f"finish_reason={finish}, {len(text)} chars of content"
            + (f", error {scrub(json.dumps(data['error']))}" if "error" in data else ""),
            "response_text": text,
            "reasoning_text": msg.get("reasoning"),
            "finish_reason": finish,
            "native_finish_reason": choice.get("native_finish_reason"),
            "usage": data.get("usage") if isinstance(data.get("usage"), dict) else None,
            "provider": data.get("provider"),
            "served_model": data.get("model"),
            "generation_id": data.get("id"),
            "errors": errors,
        }
    return {"status": "failed", "error": f"gave up after {retries + 1} unbilled attempts", "errors": errors}


async def run_panel(a) -> Path:
    question, context = Path(a.question_file).read_text(), Path(a.context_file).read_text()
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"# Context\n\n{context}\n\n# Questions\n\n{question}"},
    ]
    n_prompt = len((SYSTEM + messages[1]["content"]).encode()) + 1000  # upper bound: a token is at least one byte
    judges, out = resolve(json.loads(Path(a.config).read_text())), Path(a.out)
    hits = cache(out.parent)
    for j in judges:
        p_in, p_out, p_req = price(j["model"], n_prompt)
        j["params"] = {
            "max_tokens": j["max_tokens"],
            "reasoning": {"effort": j["effort"]},
            "usage": {"include": True},
            "provider": {"only": j["provider"], "max_price": {"prompt": p_in * 1e6, "completion": p_out * 1e6}},
        }
        j["body"] = {"model": j["slug"], "messages": messages, **j["params"]}
        j["call_sha256"], j["price"] = sha(j["body"]), (p_in, p_out, p_req)
        j["worst_usd"] = 0.0 if j["call_sha256"] in hits else n_prompt * p_in + j["max_tokens"] * p_out + p_req
    worst = sum(j["worst_usd"] for j in judges)
    print(f"prompt at most {n_prompt:,} tokens; worst case one billed attempt per judge at max_tokens:")
    for j in judges:
        print(
            f"  {j['name']:<18} {j['slug']:<28} effort={j['effort']:<6} max_tokens={j['max_tokens']:>7,} "
            f"${j['price'][0] * 1e6:g}/${j['price'][1] * 1e6:g} per M  worst ${j['worst_usd']:.2f}"
            + ("  (cached)" if j["call_sha256"] in hits else "")
        )
    print(f"panel worst case ${worst:.2f} (cap ${a.cap_usd:.2f})")
    if worst > a.cap_usd:
        sys.exit("worst case above the cap; nothing sent")
    if a.dry_run:
        sys.exit(0)

    key = api_key()
    scrub = lambda s: s.replace(key, "<redacted>")
    git_head, dirty = git("rev-parse", "HEAD"), bool(git("status", "--porcelain", "--", str(HERE)))
    base = {
        "panel": out.stem,
        "question_file": a.question_file,
        "question_sha256": sha(question),
        "context_file": a.context_file,
        "context_sha256": sha(context),
        "prompt_sha256": sha(messages),
        "messages": messages,
        "git_head": git_head,
        "git_dirty": dirty,
    }
    for j in judges:  # built before any call, so nothing after a paid call can fail on metadata
        j["meta"] = {
            **base,
            "judge_index": j["index"],
            "judge": j["name"],
            "slug": j["slug"],
            "canonical_slug": j["model"].get("canonical_slug"),
            "list_pricing": j["model"].get("pricing"),
            "params_sent": j["params"],
            "call_sha256": j["call_sha256"],
            "worst_usd": j["worst_usd"],
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    rows, n_send = [], sum(j["call_sha256"] not in hits for j in judges)
    with out.open("x") as f:  # noqa: ASYNC230 -- reserved before any call; results are append-only, never overwritten
        with (HERE / "SPEND.md").open("a") as s:  # launch row first, so an interrupted run is still in the ledger
            s.write(
                f"| {now()} | `{out.name}` | launched: {n_send} to send, {len(judges) - n_send} cached | ${worst:.2f} | pending | pending |\n"
            )

        def save(row):
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(a.timeout_s, connect=30), headers={"Authorization": f"Bearer {key}"}
        ) as client:

            async def one(j):
                if j["call_sha256"] in hits:
                    h = hits[j["call_sha256"]]
                    keep = (
                        "status",
                        "error",
                        "response_text",
                        "reasoning_text",
                        "finish_reason",
                        "native_finish_reason",
                        "usage",
                        "provider",
                        "served_model",
                        "generation_id",
                        "started_at",
                        "finished_at",
                        "latency_s",
                        "cost_usd",
                        "cost_source",
                    )
                    return save(
                        {
                            **j["meta"],
                            **{k: h.get(k) for k in keep},
                            "cached_from": h["cached_from"],
                            "cost_usd_this_run": 0.0,
                        }
                    )
                t0, started = time.time(), now()
                r = await call(client, j["body"], a.timeout_s, a.retries, scrub)
                u = r.get("usage") if isinstance(r.get("usage"), dict) else {}
                pt, ct = num(u.get("prompt_tokens")), num(u.get("completion_tokens"))
                if num(u.get("cost")) is not None:
                    cost, src = float(u["cost"]), "openrouter usage.cost"
                elif pt is not None and ct is not None:
                    cost, src = pt * j["price"][0] + ct * j["price"][1] + j["price"][2], "estimated from list price"
                else:
                    cost, src = None, f"unknown (no usage returned; at most {j['worst_usd']:.2f})"
                latency = round(time.time() - t0, 1)
                save(  # saved before printing, so a paid answer is on disk first
                    {
                        **j["meta"],
                        **r,
                        "started_at": started,
                        "finished_at": now(),
                        "latency_s": latency,
                        "cost_usd": cost,
                        "cost_source": src,
                        "cost_usd_this_run": cost,
                        "cached_from": None,
                    }
                )
                print(
                    f"  {j['name']}: {r['status']} in {latency:.0f} s"
                    + (f" ({r['error']})" if r["error"] else "")
                    + (f", ${cost:.4f}" if cost is not None else ", cost unknown"),
                    flush=True,
                )

            results = await asyncio.gather(*(one(j) for j in judges), return_exceptions=True)
        for j, res in zip(judges, results):
            if isinstance(
                res, BaseException
            ):  # a bug in this script: recorded as a failed row, and the run exits non-zero
                save(
                    {
                        **j["meta"],
                        "status": "failed",
                        "error": f"script error {type(res).__name__}: {scrub(str(res))}",
                        "cost_usd": None,
                        "cost_source": f"unknown (at most {j['worst_usd']:.2f})",
                        "cost_usd_this_run": None,
                        "cached_from": None,
                    }
                )
    failed = [r for r in rows if r["status"] != "ok"]
    spent = sum(r["cost_usd_this_run"] or 0 for r in rows)
    unknown = [r for r in rows if r["cost_usd_this_run"] is None]
    n_cached = sum(bool(r["cached_from"]) for r in rows)
    with (HERE / "SPEND.md").open("a") as f:
        f.write(
            f"| {now()} | `{out.name}` | {len(rows) - n_cached} sent, {n_cached} cached | ${worst:.2f} | ${spent:.2f}"
            + "".join(f" + unknown for {r['judge']} (at most ${r['worst_usd']:.2f})" for r in unknown)
            + f" | {', '.join(r['judge'] for r in failed) or 'none'} |\n"
        )
    print(
        f"wrote {out}; actual ${spent:.2f}"
        + (f" + {len(unknown)} unknown" if unknown else "")
        + f"; {len(rows) - len(failed)} ok, {len(failed)} failed"
        + (f": {[r['judge'] for r in failed]}" if failed else "")
    )
    return out


def summarize(path: Path) -> Path:
    rows = sorted((json.loads(line) for line in path.read_text().splitlines()), key=lambda r: r["judge_index"])
    r0 = rows[0]
    lines = [
        f"# LLM judge panel `{path.stem}`",
        "",
        "Each judge's answer verbatim, unedited. This file contains no synthesis.",
        "",
        f"Question `{r0['question_file']}` (sha256 {r0['question_sha256'][:12]}), context `{r0['context_file']}` "
        f"(sha256 {r0['context_sha256'][:12]}), prompt sha256 {r0['prompt_sha256'][:12]}, git {r0['git_head'][:10]}"
        + (" (experiment folder dirty)" if r0["git_dirty"] else "")
        + ".",
        "",
        "| judge | slug (served) | effort sent | status | prompt / completion / reasoning tokens | cost | provider |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        u = r["usage"] if isinstance(r.get("usage"), dict) else {}
        d = u["completion_tokens_details"] if isinstance(u.get("completion_tokens_details"), dict) else {}
        rt = d.get("reasoning_tokens")
        cost = (
            "?"
            if r.get("cost_usd") is None
            else f"${r['cost_usd']:.4f}" + (" (cached)" if r.get("cached_from") else "")
        )
        lines.append(
            f"| {r['judge']} | `{r['slug']}` ({r.get('served_model') or r.get('canonical_slug')}) | {r['params_sent']['reasoning']['effort']} | "
            f"{r['status']} | {u.get('prompt_tokens', '?')} / {u.get('completion_tokens', '?')} / {rt if rt is not None else '?'} | {cost} | {r.get('provider') or '?'} |"
        )
    for r in rows:
        lines += [
            "",
            "---",
            "",
            f"## {r['judge']} (`{r['slug']}`, effort {r['params_sent']['reasoning']['effort']})",
            "",
        ]
        if r["status"] != "ok":
            lines += [f"**FAILED: {r['error']}**", ""] + (
                ["Partial content as returned:", ""] if r.get("response_text") else []
            )
        lines.append(str(r.get("response_text") or ""))
    md = path.with_suffix(".md")
    md.write_text("\n".join(lines) + "\n")
    return md


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--question-file")
    ap.add_argument("--context-file")
    ap.add_argument("--out", help="results/panel_<stamp>.jsonl (default: stamped now, in this folder's results/)")
    ap.add_argument("--config", default=str(HERE / "config" / "judges.json"))
    ap.add_argument(
        "--cap-usd", type=float, default=40.0, help="refuse to send if the panel's worst case is above this"
    )
    ap.add_argument("--timeout-s", type=float, default=900.0, help="per-call deadline")
    ap.add_argument(
        "--retries", type=int, default=2, help="extra attempts after unbilled errors (429, connection failures)"
    )
    ap.add_argument("--dry-run", action="store_true", help="resolve judges and print the cost estimate only")
    ap.add_argument("--summarize", metavar="JSONL", help="only (re)write the markdown for an existing panel file")
    a = ap.parse_args()
    if a.summarize:
        print(f"wrote {summarize(Path(a.summarize))}")
        return
    if not (a.question_file and a.context_file):
        ap.error("--question-file and --context-file are required")
    a.out = a.out or str(HERE / "results" / f"panel_{datetime.now(PT):%Y-%m-%d_%H%M%S}.jsonl")
    out = asyncio.run(run_panel(a))
    print(f"wrote {summarize(out)}")
    if any(json.loads(line)["status"] != "ok" for line in out.read_text().splitlines()):
        sys.exit(1)


if __name__ == "__main__":
    main()
