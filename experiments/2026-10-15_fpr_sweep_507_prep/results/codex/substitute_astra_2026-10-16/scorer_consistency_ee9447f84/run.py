import asyncio, json, os, sys, time, hashlib
import httpx
# Substitute code reviewer while Codex is out of credits (Aswin 10-16 ~12:35, relayed by 2e): GPT-6 Astra via OpenRouter only,
# provider pinned to openai, no fallbacks. Copied from ~/probe-shift-writing-handoff-2026-10-15/w/panel/run.py (JUDGES and SYSTEM changed,
# the returned model id recorded).
W = sys.argv[1]
JUDGES = [("GPT-6 Astra", "openai/gpt-6-astra", "xhigh", ["openai"])]
MAXTOK = 36000
SYSTEM = ("You are a strict code reviewer. Review ONLY the diff you are given, using the surrounding code for context. Report every "
          "finding as REQUIRED (a wrong result, a check that can pass when it should fail, a crash on realistic inputs) or OPTIONAL "
          "(style, robustness on inputs that cannot occur). Each REQUIRED item: file:line, the concrete input that triggers it, and the "
          "fix. Do not suggest new features. If something is ambiguous say so rather than guess. End your answer with exactly one line: "
          "'VERDICT: PASS' if there is no REQUIRED item, else 'VERDICT: REQUIRED CHANGES'.")
key = os.environ.get("OPENROUTER_API_KEY", "")
if not key: sys.exit("OPENROUTER_API_KEY missing (value not shown)")
ctx = open(f"{W}/context.md").read(); q = open(f"{W}/question.md").read()
msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ctx + "\n\n# Question\n" + q}]
models = {m["id"]: m for m in httpx.get("https://openrouter.ai/api/v1/models", timeout=60).json()["data"]}
worst = 0
for n, s, e, _ in JUDGES:
    m = models[s]; assert e in m["reasoning"]["supported_efforts"], (n, e)
    p = m["pricing"]; worst += (len(json.dumps(msgs)) + 1000) * float(p["prompt"]) + MAXTOK * float(p["completion"])
print(f"worst case ${worst:.2f}"); assert worst < 5.0
async def one(c, n, s, e, prov):
    body = {"model": s, "messages": msgs, "max_tokens": MAXTOK, "reasoning": {"effort": e}, "usage": {"include": True},
            "provider": {"order": prov, "allow_fallbacks": False}}
    t = time.time()
    try:
        r = await c.post("https://openrouter.ai/api/v1/chat/completions", json=body, timeout=1800)
        d = r.json()
        if r.status_code != 200: return {"judge": n, "status": "failed", "error": f"http {r.status_code}: {str(d)[:500]}"}
        ch = d["choices"][0]
        return {"judge": n, "slug": s, "model_returned": d.get("model"), "provider_returned": d.get("provider"), "effort": e, "status": "ok",
                "finish_reason": ch.get("finish_reason"), "text": ch["message"].get("content"), "cost": (d.get("usage") or {}).get("cost"),
                "secs": round(time.time() - t)}
    except Exception as ex:
        return {"judge": n, "status": "failed", "error": f"{type(ex).__name__}: {ex}"[:500]}
async def main():
    async with httpx.AsyncClient(headers={"Authorization": f"Bearer {key}"}) as c:
        res = await asyncio.gather(*[one(c, *j) for j in JUDGES])
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    with open(f"{W}/review_{stamp}.jsonl", "w") as f:
        for r in res: f.write(json.dumps({**r, "prompt_sha256": hashlib.sha256(json.dumps(msgs).encode()).hexdigest()}) + "\n")
    with open(f"{W}/review_{stamp}.md", "w") as f:
        for r in res: f.write(f"## {r['judge']} (model {r.get('model_returned')}, provider {r.get('provider_returned')}, {r['status']}, finish {r.get('finish_reason')}, cost {r.get('cost')})\n\n{r.get('text') or r.get('error')}\n\n")
    print(stamp, [(r["judge"], r.get("model_returned"), r.get("provider_returned"), r["status"], r.get("finish_reason"), r.get("cost")) for r in res])
asyncio.run(main())
