"""Deception black-box, step 3: the judge, one call per run (PLAN.md; judge_prompt.md fixed before any call).
gpt-5.6-sol at medium effort and the API's only temperature (1), on the laptop only: the key is read from the repo's
git-ignored .env into this process and is never printed, logged or passed as an argument (Aswin, relayed by the
orchestrator: "The judge runs on the laptop only, with OPENAI_API_KEY read from .env in-process: never printed, never
copied, never on Modal or a devbox"). Async, cached by a content hash (data/deception_blackbox/judge_cache/).
- Runs over the token check's input limit are not sent (dropped for both scorers and counted).
- No new request starts once the spend so far, plus every in-flight call's worst case, plus this call's worst case would
  pass the $90 stop (Aswin about 23:33 PT: "Raise stop to $90 (Recommended)"). A call's worst case is its counted input
  tokens plus 4,000 output and reasoning tokens, at $5 / $30 per million.
- Any refusal (structured, a content-filter finish, or a refusal in plain text), safety-filter stop or provider error
  stops the whole run: no new request starts, and the at most 8 already in flight finish. Nothing is rerouted. The
  SDK's own retries are off.
- Spend ledger (data/deception_blackbox/spend_ledger.jsonl, append-only): each call books its worst case before it is
  sent and settles to the billed cost after its answer is processed, so a crash, an error or an unprocessable answer
  leaves the worst case booked. A run starts from the ledger's total, so a restart can't add another $90; a lock file
  stops two runs at once. Reaching the limit stops the run like any other stop.
- It refuses to start unless analyze_blackbox.py is committed (the readings are fixed before any answer) and the token
  check was made with this prompt.
- A parse failure gets one retry; then the run is excluded and counted.
Raw answers are model text, so they go to data/; results/ keeps the probability, token counts and cost per run.

  uv run --no-project --with openai python experiments/2026-10-11_q2_deception_blackbox/run_judges.py <judge inputs jsonl> <token check json>
"""
import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from judge_common import HERE, fill, prompts

ROOT = HERE.parents[1]
DATA = ROOT / "data" / "deception_blackbox"
CACHE = DATA / "judge_cache"
LEDGER = DATA / "spend_ledger.jsonl"
ENV = Path.home() / "probe-distribution-shift-from-sims" / ".env"
MODEL, EFFORT, PRICE_IN, PRICE_OUT, STOP, CONCURRENCY, MAX_OUT = "gpt-5.6-sol", "medium", 5.0, 30.0, 90.0, 8, 4000
REFUSAL = re.compile(r"\b(?:can(?:not|'t)|unable to|won't|will not)\s+(?:help|assist|comply|provide|do that|evaluate)|\bI(?:'m| am) sorry\b", re.I)
cost = lambda rec: rec["input_tokens"] * PRICE_IN / 1e6 + rec["output_tokens"] * PRICE_OUT / 1e6
key_of = lambda system, msg, attempt: hashlib.sha256(json.dumps([MODEL, EFFORT, 1, MAX_OUT, system, msg, attempt]).encode()).hexdigest()  # temperature 1


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


async def run(runs, ntok, over):
    from openai import AsyncOpenAI
    system, user = prompts(); CACHE.mkdir(parents=True, exist_ok=True)
    client, sem = AsyncOpenAI(api_key=load_key(), max_retries=0), asyncio.Semaphore(CONCURRENCY)
    paths = {CACHE / f"{key_of(system, fill(user, r), a)}.json" for r in runs for a in (0, 1)}  # unique cache files
    prior = [json.loads(f.read_text()) for f in sorted(paths) if f.exists()]
    if any(x.get("refusal") or x.get("finish") == "content_filter" or REFUSAL.search(x["text"]) for x in prior):
        raise SystemExit("a cached answer is a refusal or a safety-filter stop: nothing is sent")
    led = [json.loads(line) for line in LEDGER.open()] if LEDGER.exists() else []
    spent, stop = [sum(x["cost"] for x in led)], [None]  # every earlier call: billed, or its worst case if unsettled
    if {f.stem for f in paths if f.exists()} - {x["key"] for x in led}:
        raise SystemExit("cached answers with no ledger entry: reconcile their cost before any new call")

    def book(k, c, kind):
        spent[0] += c
        with open(LEDGER, "a") as fh:
            fh.write(json.dumps({"key": k, "cost": c, "kind": kind}) + "\n"); fh.flush(); os.fsync(fh.fileno())
    locks = {}  # one paid call per cache key: identical prompts wait and reuse it

    async def ask(r, attempt):
        msg = fill(user, r); k = key_of(system, msg, attempt)
        async with locks.setdefault(k, asyncio.Lock()):
            return await ask_locked(r, msg, CACHE / f"{k}.json")

    async def ask_locked(r, msg, f):
        if f.exists():
            rec = json.loads(f.read_text())
            if rec.get("refusal") or rec.get("finish") == "content_filter":
                stop[0] = f"refusal or safety filter on {r['key']} (cached)"
            return rec
        async with sem:  # the stop is checked once a slot is free, so queued calls can't pass it
            worst = ntok[r["key"]] * PRICE_IN / 1e6 + MAX_OUT * PRICE_OUT / 1e6
            if stop[0]:
                return None
            if spent[0] + worst > STOP:
                stop[0] = stop[0] or f"spend limit: ${spent[0]:.2f} booked"
                return None
            book(f.stem, worst, "worst case, booked before sending")
            resp = await client.chat.completions.create(model=MODEL, reasoning_effort=EFFORT, max_completion_tokens=MAX_OUT,
                                                        messages=[{"role": "system", "content": system}, {"role": "user", "content": msg}])
        ch = resp.choices[0]
        rec = {"text": ch.message.content or "", "refusal": getattr(ch.message, "refusal", None), "finish": ch.finish_reason,
               "input_tokens": resp.usage.prompt_tokens, "output_tokens": resp.usage.completion_tokens}
        book(f.stem, cost(rec) - worst, "settled to the billed cost")
        if rec["refusal"] or rec["finish"] == "content_filter":
            stop[0] = f"refusal or safety filter on {r['key']}"
        f.write_text(json.dumps(rec))
        return rec

    async def one(r):
        if r["key"] in over:
            return {"key": r["key"], "dropped": "over the input limit"}
        rec = None
        for attempt in (0, 1):
            try:
                rec = await ask(r, attempt)
            except Exception as e:  # a provider error stops the run; counted, never scored
                stop[0] = f"{type(e).__name__} on {r['key']}"
                return {"key": r["key"], "error": stop[0]}
            if rec is None:
                return {"key": r["key"], "skipped": "stop or spend limit"}
            if rec.get("refusal") or rec.get("finish") == "content_filter":
                return {"key": r["key"], "refused": True, "raw": rec["text"]}
            v = parse(rec["text"])
            if REFUSAL.search(rec["text"]):  # a refusal in plain text, even inside a parsed answer, halts everything too
                stop[0] = f"refusal in text on {r['key']}"
                return {"key": r["key"], "refused": True, "raw": rec["text"]}
            if v is not None:
                return {"key": r["key"], "probability": v, "attempt": attempt, "tokens": [rec["input_tokens"], rec["output_tokens"]], "raw": rec["text"]}
        return {"key": r["key"], "excluded": "unparseable twice", "raw": rec["text"]}
    res = await asyncio.gather(*(one(r) for r in runs))
    return {"runs": len(runs), "estimated spend (USD)": round(spent[0], 4), "stopped": stop[0]}, res


def main(inp: Path, tok_p: Path):
    for f in (Path(__file__).resolve(), HERE / "judge_common.py", HERE / "judge_prompt.md", HERE / "analyze_blackbox.py"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    tok = json.loads(tok_p.read_text())
    if tok["inputs sha256"] != hashlib.sha256(inp.read_bytes()).hexdigest():
        raise SystemExit("the token check was made from other inputs")
    for name, k in (("judge_prompt.md", "prompt sha256"), ("judge_common.py", "judge_common sha256")):
        if tok[k] != hashlib.sha256((HERE / name).read_bytes()).hexdigest():
            raise SystemExit(f"the token check was made with another {name}")
    runs = [json.loads(line) for line in inp.open()]
    if set(tok["tokens per run"]) != {r["key"] for r in runs}:
        raise SystemExit("the token check doesn't cover exactly these runs")
    DATA.mkdir(parents=True, exist_ok=True); lock = DATA / "judge_run.lock"
    try:
        lock.touch(exist_ok=False)
    except FileExistsError:
        raise SystemExit(f"another judge run holds {lock} (remove it only if no run is alive)")
    try:
        summary, res = asyncio.run(run(runs, tok["tokens per run"], set(tok["runs over the limit (dropped for both scorers)"])))
    finally:
        lock.unlink()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(raw := DATA / f"judgments_raw_{stamp}.jsonl", "x") as fh:
        fh.writelines(json.dumps({"key": r["key"], "raw": r.get("raw")}) + "\n" for r in res)
    rows = [{k: v for k, v in r.items() if k != "raw"} | ({"reason sha256": hashlib.sha256(r["raw"].encode()).hexdigest()} if r.get("raw") else {}) for r in res]
    out = {"plan": "PLAN.md (this folder)", "git": head, "model": MODEL, "effort": EFFORT, "inputs": inp.name, "inputs sha256": tok["inputs sha256"],
           "token check": tok_p.name, "planted.py blob": subprocess.run(["git", "rev-parse", "HEAD:src/probe_shift/planted.py"], capture_output=True, text=True, check=True, cwd=ROOT).stdout.strip(),
           "analysis blob": subprocess.run(["git", "rev-parse", "HEAD:./analyze_blackbox.py"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
           "prompt sha256": hashlib.sha256((HERE / "judge_prompt.md").read_bytes()).hexdigest(), "summary": summary,
           "not scored": {k: sum(1 for r in rows if k in r) for k in ("dropped", "excluded", "skipped", "refused", "error")},
           "raw answers (data/)": raw.name, "rows": rows}
    with open(dest := HERE / "results" / f"judgments_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, separators=(",", ":")) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1)); print(dest)
    if summary["stopped"]:
        raise SystemExit(f"stopped: {summary['stopped']}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
