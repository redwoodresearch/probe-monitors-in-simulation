import asyncio, json, os, sys, time, hashlib
import httpx
W = sys.argv[1]
JUDGES = [("GPT-6 Astra","openai/gpt-6-astra","xhigh",["openai"])]  # d9 10-16: single substitute code reviewer (Codex out of credits; Aswin ~12:35)
MAXTOK = 48000
SYSTEM = ("You are a senior code reviewer for research code (Python, Modal, Inspect AI). Review the code and claims you are given "
          "for correctness bugs, silent failures, crash paths, and leaks of sensitive text. Be concrete: cite line numbers. Classify each "
          "finding as REQUIRED (must fix before running) or OPTIONAL. If something cannot be checked from what you are given, say so "
          "instead of guessing. End with one line: VERDICT: AGREE (no REQUIRED findings) or VERDICT: DISAGREE (n REQUIRED).")
def _env_key(path=os.path.expanduser("~/probe-distribution-shift-from-sims/.env")):  # the git-ignored repo .env (the panel tool's source)  # in-process only: never printed, logged or passed on argv
    for line in open(path):
        k, _, v = line.strip().partition("=")
        if k.strip().removeprefix("export ").strip() == "OPENROUTER_API_KEY":
            return v.strip().strip('"').strip("'")
    return ""
key = os.environ.get("OPENROUTER_API_KEY", "") or _env_key()
if not key: sys.exit("OPENROUTER_API_KEY missing (value not shown)")
ctx = open(f"{W}/context.md").read(); q = open(f"{W}/question.md").read()
msgs = [{"role":"system","content":SYSTEM},{"role":"user","content":ctx+"\n\n# Question\n"+q}]
models = {m["id"]: m for m in httpx.get("https://openrouter.ai/api/v1/models", timeout=60).json()["data"]}
worst = 0
for n, s, e, _ in JUDGES:
    m = models[s]; assert e in m["reasoning"]["supported_efforts"], (n, e)
    p = m["pricing"]; worst += (len(json.dumps(msgs))+1000)*float(p["prompt"]) + MAXTOK*float(p["completion"])
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
        return {"judge": n, "slug": s, "effort": e, "status": "ok", "finish_reason": ch.get("finish_reason"),
                "text": ch["message"].get("content"), "cost": (d.get("usage") or {}).get("cost"), "secs": round(time.time()-t)}
    except Exception as ex:
        return {"judge": n, "status": "failed", "error": f"{type(ex).__name__}: {ex}"[:500]}
async def main():
    async with httpx.AsyncClient(headers={"Authorization": f"Bearer {key}"}) as c:
        res = await asyncio.gather(*[one(c, *j) for j in JUDGES])
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    with open(f"{W}/panel_{stamp}.jsonl", "w") as f:
        for r in res: f.write(json.dumps({**r, "prompt_sha256": hashlib.sha256(json.dumps(msgs).encode()).hexdigest()})+"\n")
    with open(f"{W}/panel_{stamp}.md", "w") as f:
        for r in res: f.write(f"## {r['judge']} ({r['status']}, finish {r.get('finish_reason')}, cost {r.get('cost')})\n\n{r.get('text') or r.get('error')}\n\n")
    print(stamp, [(r["judge"], r["status"], r.get("finish_reason"), r.get("cost")) for r in res])
asyncio.run(main())
