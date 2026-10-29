"""Skeptic panel (PLAN.md): three models on OpenRouter, each pinned to its first-party provider, judge each question's
plain answer line against the numbers and checks given. One call per (model, question), async, cached by the request's
sha256, a running-cost stop at $9. The key is read from the git-ignored .env into this process only; never printed.

  uv run --no-project --with httpx python experiments/2026-10-11_q2_skeptic_panel/run_panel.py --input <panel_input json> [--dry-run]
"""
import argparse
import asyncio
import hashlib
import json
import math
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
ENV = Path.home() / "probe-distribution-shift-from-sims" / ".env"
URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS = {"openai/gpt-6-astra": "openai", "google/gemini-3.1-pro-preview": "google-ai-studio", "anthropic/claude-opus-5.5": "anthropic"}
PRICE = {"openai/gpt-6-astra": (10e-6, 50e-6), "google/gemini-3.1-pro-preview": (2e-6, 12e-6), "anthropic/claude-opus-5.5": (4e-6, 20e-6)}  # $/token, list
SETTINGS = {"reasoning": {"effort": "medium"}, "max_tokens": 8000}
CAP = 10.0  # hard: worst-case cost is reserved before each attempt; a ledger file carries spend across restarts
SYSTEM = ("You are a skeptical reviewer of empirical AI-safety research. Judge only from the numbers and checks given for each question. "
          "Be concrete and brief.")
PREAMBLE = ("Project: probes (logistic regressions on a language model's internal activations) and text classifiers are compared between real "
            "agent runs in a software environment (LinuxArena) and simulated runs, where an LLM simulator stands in for the environment. "
            "Terms: AUROC (0.5 is chance); catch rate at target X% false alarms (the share of positives above a threshold set to flag X% of "
            "negatives); real-vs-real control (the same statistic between two real runs of the same tasks); planted check (a known difference "
            "inserted into copies, to test whether the method finds it); label-shuffle null (the statistic with labels shuffled); Holm (a "
            "multiple-comparison correction); margin (a pre-stated range within which a gap counts as equivalent).")
REQUEST = ("Answer in JSON with keys: supported (\"yes\", \"partly\" or \"no\": is the plain answer line supported by the numbers and checks "
           "given?), most_likely_overclaim (one sentence), cheapest_check (one sentence: the cheapest check that could change the "
           "conclusion), confidence (0 to 1).")
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["supported", "most_likely_overclaim", "cheapest_check", "confidence"],
          "properties": {"supported": {"type": "string", "enum": ["yes", "partly", "no"]}, "most_likely_overclaim": {"type": "string"},
                         "cheapest_check": {"type": "string"}, "confidence": {"type": "number"}}}


def load_key():
    """Only OPENROUTER_API_KEY, into this process's environment; the value is never printed or written."""
    for line in ENV.read_text().splitlines():
        s = line.strip().removeprefix("export ").strip()
        if s.startswith("OPENROUTER_API_KEY="):
            os.environ["OPENROUTER_API_KEY"] = s.split("=", 1)[1].strip().strip("'\"")
            return
    raise SystemExit("OPENROUTER_API_KEY not found in the .env file")


def block(q: dict) -> str:
    lines = [f"Question {q['id']}: {q['question']}", f"Status claimed: {q['status']}", f"Plain answer line: \"{q['line']}\"", "Key numbers:"]
    lines += [f"- {x}" for x in q["numbers"]] + ["Checks:"] + [f"- {x}" for x in q["checks"]]
    return "\n".join(lines)


def request_body(model: str, q: dict) -> dict:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"{PREAMBLE}\n\n{block(q)}\n\n{REQUEST}"}]
    return {"model": model, "messages": messages, "provider": {"order": [MODELS[model]], "allow_fallbacks": False, "require_parameters": True}, "usage": {"include": True},
            "response_format": {"type": "json_schema", "json_schema": {"name": "verdict", "strict": True, "schema": SCHEMA}}, **SETTINGS}


def parse(content):
    try:
        v = json.loads(content)
    except (TypeError, json.JSONDecodeError) as e:
        return None, f"not JSON: {type(e).__name__}"
    ok = (isinstance(v, dict) and set(v) == set(SCHEMA["required"]) and v["supported"] in ("yes", "partly", "no")
          and all(isinstance(v[k], str) and v[k].strip() for k in ("most_likely_overclaim", "cheapest_check"))
          and isinstance(v["confidence"], (int, float)) and not isinstance(v["confidence"], bool) and 0 <= v["confidence"] <= 1)
    return (v, None) if ok else (None, "schema mismatch")


def worst(model: str, body: dict) -> float:
    """Worst-case cost of one attempt: input at one token per character of the whole request (schema included), output at
    max_tokens, list prices."""
    return len(json.dumps(body)) * PRICE[model][0] + SETTINGS["max_tokens"] * PRICE[model][1]


def save_ledger(led):
    """Atomic: spent plus everything reserved, so a crash leaves outstanding attempts counted."""
    f = Path(led["file"]); tmp = f.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"spent": led["spent"], "reserved": led["reserved"]})); tmp.replace(f)


def billed_cost(data, w):
    c = (data.get("usage") or {}).get("cost") if isinstance(data, dict) else None
    try:
        c = float(c)
    except (TypeError, ValueError):
        return None
    return c if math.isfinite(c) and c >= 0 else w


async def one(client, sem, lock, led, model, q, cache):
    """One (model, question) call. Never raises: every failure becomes a fixed status code, counted and reported."""
    out = {"model": model, "qid": q.get("id"), "cached": False, "status": "not attempted", "attempt cost": 0.0}
    t0 = time.time()
    try:
        body = request_body(model, q); key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        path, w = cache / f"{key}.json", worst(model, body); out["body_sha256"] = key
        if path.exists():
            return json.loads(path.read_text()) | {"cached": True, "attempt cost": 0.0}
        data = None
        async with sem:
            for attempt in range(4):
                async with lock:  # reserve the worst case before each attempt, on disk too
                    if led["spent"] + led["reserved"] + w > CAP:
                        out["status"] = "skipped: cost cap"
                        break
                    led["reserved"] += w; save_ledger(led)
                try:
                    r = await client.post(URL, json=body, headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}, timeout=600)
                    code = r.status_code
                    data = r.json() if code == 200 else None
                except (httpx.HTTPError, ValueError) as e:
                    code, data = f"transport_{type(e).__name__}", None
                c = billed_cost(data, w)
                c = c if c is not None else (0.0 if isinstance(code, int) and 400 <= code < 500 else w)
                async with lock:
                    led["reserved"] -= w; led["spent"] += c; save_ledger(led)
                out["attempt cost"] += c
                if code == 200 and isinstance(data, dict) and "error" not in data and data.get("choices"):
                    out["status"] = "ok"
                    break
                out["status"] = "error_in_body" if code == 200 else (f"http_{code}" if isinstance(code, int) else code)
                if isinstance(code, int) and code not in (200, 429, 500, 502, 503, 504):
                    break
                if attempt < 3:
                    await asyncio.sleep(5 * 2 ** attempt)
        if out["status"] == "ok":
            u = data.get("usage") or {}; content = (data["choices"][0].get("message") or {}).get("content")
            verdict, err = parse(content)
            out |= {"provider": data.get("provider"), "usage": u, "cost": out["attempt cost"], "content": content, "verdict": verdict, "parse_error": err}
            tmp = path.with_suffix(f".{os.getpid()}.tmp"); tmp.write_text(json.dumps(out)); tmp.replace(path)
    except Exception as e:  # recorded as a failure, never as agreement
        out["status"] = f"exception_{type(e).__name__}"
    out["latency_s"] = round(time.time() - t0, 1)
    return out


async def run(questions, out_dir):
    cache = out_dir.parent / "cache"; cache.mkdir(parents=True, exist_ok=True)
    ledger = out_dir.parent / "spent.json"
    prior = json.loads(ledger.read_text()) if ledger.exists() else {"spent": 0.0, "reserved": 0.0}
    led = {"spent": prior["spent"] + prior.get("reserved", 0.0), "reserved": 0.0, "file": str(ledger)}  # leftover reservations count as spent
    print(f"spent before this run: ${led['spent']:.2f} of the ${CAP:.0f} cap", flush=True)
    sem, lock = asyncio.Semaphore(6), asyncio.Lock()
    async with httpx.AsyncClient() as client:
        return await asyncio.gather(*(one(client, sem, lock, led, m, q, cache) for q in questions for m in MODELS))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--input", required=True); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    inp = Path(a.input); spec = json.loads(inp.read_text()); qs = spec["questions"]
    if len({q["id"] for q in qs}) != len(qs):
        raise SystemExit("duplicate question IDs")
    for q in qs:
        missing = {"id", "question", "status", "line", "numbers", "checks"} - set(q)
        if missing:
            raise SystemExit(f"{q.get('id')}: missing {missing}")
    if a.dry_run:
        n_chars = sum(len(json.dumps(request_body(m, q))) for q in qs for m in MODELS)
        print(f"{len(qs)} questions x {len(MODELS)} models = {len(qs) * len(MODELS)} calls; about {n_chars // 4:,} input tokens in all")
        print(request_body(next(iter(MODELS)), qs[0])["messages"][1]["content"])
        return
    load_key()  # before the lock, so a missing key leaves no stale lock behind
    lockf = ROOT / "data" / "skeptic_panel" / "run.lock"; lockf.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lockf, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f"another panel run holds {lockf}; remove it only if no run is active")
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    try:
        stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
        out_dir = ROOT / "data" / "skeptic_panel" / stamp; out_dir.mkdir(parents=True, exist_ok=True)
        rows = asyncio.run(run(qs, out_dir))
    finally:
        lockf.unlink(missing_ok=True)
    env = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
    git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, env=env).stdout.strip()
    with open(out_dir / f"panel_{stamp}.jsonl", "x") as fh:
        for r in rows:
            fh.write(json.dumps(r | {"git": git, "settings": SETTINGS, "input": str(inp), "input_sha256": hashlib.sha256(inp.read_bytes()).hexdigest()}) + "\n")
    ok = [r for r in rows if r["status"] == "ok" and r.get("verdict")]
    summary = {"stamp": stamp, "input": str(inp), "input sha256": hashlib.sha256(inp.read_bytes()).hexdigest(), "git": git, "models": MODELS, "settings": SETTINGS,
               "calls": len(rows), "ok": len(ok), "failed or unparsed": [[r["model"], r["qid"], r["status"], r.get("parse_error")] for r in rows if r not in ok],
               "cost (new calls, every attempt)": round(sum(r.get("attempt cost", 0.0) for r in rows), 4),
               "ledger after run": json.loads((out_dir.parent / "spent.json").read_text()) if (out_dir.parent / "spent.json").exists() else None,
               "verdicts": {q["id"]: {r["model"]: (r["verdict"]["supported"] if r in ok else None) for r in rows if r["qid"] == q["id"]} for q in qs},
               "responses": [[r["model"], r["qid"], hashlib.sha256((r.get("content") or "").encode()).hexdigest(), len(r.get("content") or "")] for r in rows]}
    dest = HERE / "results" / f"panel_summary_{stamp}.json"; dest.parent.mkdir(exist_ok=True)
    with open(dest, "x") as fh:
        fh.write("{\n" + ",\n".join(f"{json.dumps(k)}: {json.dumps(v)}" for k, v in summary.items()) + "\n}\n")
    with open(out_dir / f"table_{stamp}.md", "x") as fh:
        fh.write("| question | " + " | ".join(MODELS) + " |\n|---|" + "---|" * len(MODELS) + "\n")
        for q in qs:
            cells = []
            for m in MODELS:
                r = next(x for x in rows if x["qid"] == q["id"] and x["model"] == m)
                cells.append(f"{r['verdict']['supported']}: {r['verdict']['most_likely_overclaim']}" if r in ok else f"({r['status']})")
            fh.write(f"| {q['id']} | " + " | ".join(c.replace("|", "/") for c in cells) + " |\n")
    print(f"{len(ok)} of {len(rows)} calls ok; new cost ${summary['cost (new calls, every attempt)']:.2f}; failures {len(summary['failed or unparsed'])}")
    print(dest); print(out_dir)


if __name__ == "__main__":
    main()
