"""Row 11 pilot, step 3: the two judges (PLAN.md; judge_prompt.md fixed before any call). gpt-5.6-sol at medium reasoning
effort, on the laptop only: the key is read from the repo's git-ignored .env into this process, and is never printed,
logged or passed as an argument (Aswin, about 20:56 PT, relayed by the orchestrator). Async, cached by a content hash
(data/hide_pilot/judge_cache/). A parse failure gets one retry, then the sample is excluded and counted. No new request
starts once the estimated spend reaches the cap ($10; $5 / $30 per million input / output tokens). A refusal or a
provider error stops the run. Raw answers are model text, so they go to data/; results/ keeps verdicts, token counts and
cost. The model accepts only its default temperature (1): temperature 0 returned 400 unsupported_value at 22:35 PT,
before any judge answer existed. So each judge answers once at temperature 1, and the cache freezes that answer.

  uv run --no-project --with openai python experiments/2026-10-11_q2_hide_same_instruction/run_judges.py <judge inputs jsonl> [--dry-run]
"""
import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
CACHE = ROOT / "data" / "hide_pilot" / "judge_cache"
ENV = Path.home() / "probe-distribution-shift-from-sims" / ".env"
MODEL, EFFORT, PRICE_IN, PRICE_OUT, CAP, CONCURRENCY = "gpt-5.6-sol", "medium", 5.0, 30.0, 10.0, 8
MAX_OUT = 4000  # output and reasoning tokens a call: bounds each call's cost, which is reserved before the call starts


def prompts():
    t = (HERE / "judge_prompt.md").read_text()
    system = t.split("# System\n", 1)[1].split("# User J1", 1)[0].strip()
    j1 = t.split("# User J1\n", 1)[1].split("# User J2", 1)[0].strip()
    j2 = t.split("# User J2\n", 1)[1].strip()
    return system, {"J1": j1, "J2": j2}


def parse(text):
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    body = body[body.find("{"): body.rfind("}") + 1] if "{" in body else body
    try:
        d = json.loads(body)
    except json.JSONDecodeError:
        return None
    p = d.get("probability") if isinstance(d, dict) else None
    return int(p) if isinstance(p, (int, float)) and 0 <= p <= 100 and float(p).is_integer() and isinstance(d.get("reason"), str) else None


def load_key():
    for line in ENV.read_text().splitlines():
        if line.startswith("OPENAI_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("OPENAI_API_KEY not found in the repo's .env")


async def run(items, dry):
    system, users = prompts()
    CACHE.mkdir(parents=True, exist_ok=True)
    jobs = [(it, j) for it in items for j in ("J1", "J2")]
    msgs = {(it["key"], it["seq"], j): users[j].replace("{text}", it["J1"]).replace("{reasoning}", it["J2 reasoning"]) for it, j in jobs}
    if dry:
        n_in = sum(len(system) + len(m) for m in msgs.values()) / 4
        return {"requests": len(msgs), "estimated input tokens": round(n_in), "estimated cost, inputs only": round(n_in * PRICE_IN / 1e6, 2)}, []
    from openai import AsyncOpenAI
    client, sem, spent, reserved, stop = AsyncOpenAI(api_key=load_key()), asyncio.Semaphore(CONCURRENCY), [0.0], [0.0], [None]

    async def ask(k, user, attempt):
        h = hashlib.sha256(json.dumps([MODEL, EFFORT, 1, MAX_OUT, system, user, attempt]).encode()).hexdigest()  # temperature 1 (the API default)
        f = CACHE / f"{h}.json"
        if f.exists():
            rec = json.loads(f.read_text())
            if rec.get("refusal"):
                stop[0] = f"refusal on {k} (cached)"
            return rec
        async with sem:  # the cap and the stop are checked once a slot is free, so queued calls can't pass them
            worst = (len(system.encode()) + len(user.encode()) + 200) * PRICE_IN / 1e6 + MAX_OUT * PRICE_OUT / 1e6  # bytes bound the tokens
            if stop[0] or spent[0] + reserved[0] + worst > CAP:
                return None
            reserved[0] += worst
            try:
                r = await client.chat.completions.create(model=MODEL, reasoning_effort=EFFORT, max_completion_tokens=MAX_OUT,
                                                         messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            finally:
                reserved[0] -= worst
        rec = {"text": r.choices[0].message.content or "", "refusal": getattr(r.choices[0].message, "refusal", None),
               "input_tokens": r.usage.prompt_tokens, "output_tokens": r.usage.completion_tokens}
        spent[0] += rec["input_tokens"] * PRICE_IN / 1e6 + rec["output_tokens"] * PRICE_OUT / 1e6
        if rec["refusal"]:
            stop[0] = f"refusal on {k}"
        f.write_text(json.dumps(rec))
        return rec

    async def one(k):
        for attempt in (0, 1):
            try:
                rec = await ask(k, msgs[k], attempt)
            except Exception as e:  # a provider error stops the run; counted, never scored
                stop[0] = f"{type(e).__name__} on {k}"
                return {"key": k, "error": stop[0]}
            if rec is None:
                return {"key": k, "skipped": "cap or stop"}
            if rec.get("refusal"):
                return {"key": k, "refused": True, "raw": rec["text"]}
            v = parse(rec["text"])
            if v is not None:
                return {"key": k, "probability": v, "attempt": attempt, "tokens": [rec["input_tokens"], rec["output_tokens"]], "raw": rec["text"]}
        return {"key": k, "excluded": "unparseable twice", "raw": rec["text"]}
    res = await asyncio.gather(*(one(k) for k in msgs))
    return {"requests": len(msgs), "estimated spend (USD)": round(spent[0], 4), "stopped": stop[0]}, res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("inputs"); ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args()
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    for f in (Path(__file__).resolve(), HERE / "judge_prompt.md"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    inp = Path(a.inputs); items = [json.loads(line) for line in inp.open()]
    summary, res = asyncio.run(run(items, a.dry_run))
    if a.dry_run:
        print(json.dumps(summary, indent=1)); return
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(raw := ROOT / "data" / "hide_pilot" / f"judgments_raw_{stamp}.jsonl", "x") as fh:
        fh.writelines(json.dumps({"key": list(r["key"]), "raw": r.get("raw")}) + "\n" for r in res)
    rows = [{"key": r["key"][0], "seq": r["key"][1], "judge": r["key"][2]} | {k: v for k, v in r.items() if k not in ("key", "raw")}
            | ({"reason sha256": hashlib.sha256((r.get("raw") or "").encode()).hexdigest()} if r.get("raw") else {}) for r in res]
    out = {"git": head, "model": MODEL, "effort": EFFORT, "inputs": inp.name, "inputs sha256": hashlib.sha256(inp.read_bytes()).hexdigest(),
           "prompt sha256": hashlib.sha256((HERE / "judge_prompt.md").read_bytes()).hexdigest(), "summary": summary,
           "excluded or skipped": {j: sum(1 for r in rows if r["judge"] == j and "probability" not in r) for j in ("J1", "J2")},
           "raw answers (data/)": raw.name, "rows": rows}
    with open(dest := HERE / "results" / f"judgments_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, separators=(",", ":")) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1)); print(dest)
    if summary["stopped"]:
        raise SystemExit(f"stopped: {summary['stopped']}")


if __name__ == "__main__":
    main()
