"""Single-judge code review (GPT-6 Astra via OpenRouter; substitute while Codex is out of credits, approved by Aswin 10-16 ~12:35).
A one-judge copy of ~/probe-shift-writing-handoff-2026-10-15/w/panel/run.py. usage: run_review.py <round dir> (context.md + question.md in it).
The key is read in-process from the git-ignored repo .env and never printed."""
import asyncio, hashlib, json, sys, time
from pathlib import Path
import httpx

W = Path(sys.argv[1])
SLUG, EFFORT, PROV, MAXTOK = "openai/gpt-6-astra", "xhigh", ["openai"], 48000
SYSTEM = ("You are a careful senior code reviewer for research analysis code (statistics, ML evaluation). Review ONLY what you are shown. "
          "Report findings as REQUIRED (must fix before the code runs: wrong results, crashes, silent mislabeling) or OPTIONAL (nits, "
          "clarity), each with the line number and a concrete fix. Do not invent problems; if unsure, say what would settle it. Do not "
          "reproduce file contents beyond short line references. End with exactly one line: 'VERDICT: PASS' (no REQUIRED findings) or "
          "'VERDICT: FIX' (one or more REQUIRED findings).")
key = ""
for line in (Path.home() / "probe-distribution-shift-from-sims" / ".env").read_text().splitlines():
    if line.startswith("OPENROUTER_API_KEY="):
        key = line.split("=", 1)[1].strip().strip('"').strip("'")
if not key:
    sys.exit("OPENROUTER_API_KEY missing (value not shown)")
msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": (W / "context.md").read_text() + "\n\n# Question\n" + (W / "question.md").read_text()}]
m = {x["id"]: x for x in httpx.get("https://openrouter.ai/api/v1/models", timeout=60).json()["data"]}[SLUG]
assert EFFORT in m["reasoning"]["supported_efforts"]
worst = (len(json.dumps(msgs)) + 1000) * float(m["pricing"]["prompt"]) + MAXTOK * float(m["pricing"]["completion"])
print(f"worst case ${worst:.2f}")
assert worst < 5.0


async def main():
    body = {"model": SLUG, "messages": msgs, "max_tokens": MAXTOK, "reasoning": {"effort": EFFORT}, "usage": {"include": True},
            "provider": {"order": PROV, "allow_fallbacks": False}}
    t = time.time()
    async with httpx.AsyncClient(headers={"Authorization": f"Bearer {key}"}) as c:
        r = await c.post("https://openrouter.ai/api/v1/chat/completions", json=body, timeout=1800)
    d = r.json()
    if r.status_code != 200:
        sys.exit(f"http {r.status_code}: {str(d)[:500]}")
    ch = d["choices"][0]
    rec = {"judge": "GPT-6 Astra", "slug": SLUG, "effort": EFFORT, "provider": PROV, "finish_reason": ch.get("finish_reason"),
           "cost": (d.get("usage") or {}).get("cost"), "secs": round(time.time() - t), "prompt_sha256": hashlib.sha256(json.dumps(msgs).encode()).hexdigest(),
           "text": ch["message"].get("content")}
    (W / "review.jsonl").open("x").write(json.dumps(rec) + "\n")
    (W / "review.md").open("x").write(f"## GPT-6 Astra via OpenRouter (finish {rec['finish_reason']}, cost {rec['cost']}, {rec['secs']} s)\n\n{rec['text']}\n")
    print(rec["finish_reason"], rec["cost"], rec["secs"])
    print((rec["text"] or "").strip().splitlines()[-1] if rec["text"] else "NO TEXT")


asyncio.run(main())
