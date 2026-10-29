"""Single-judge code review by GPT-6 Astra via OpenRouter (substitute while Codex is out; approved by Aswin 10-16 ~12:35, via 0d).
Copied from ~/probe-shift-writing-handoff-2026-10-15/w/panel/run.py and cut to one judge. Reads <W>/context.md and <W>/question.md,
writes <W>/review_<stamp>.{jsonl,md}. The key is read in-process from the git-ignored repo .env and never printed."""
import hashlib, json, sys, time
from pathlib import Path
import httpx
W = Path(sys.argv[1])
JUDGE = ("GPT-6 Astra", "openai/gpt-6-astra", "xhigh", ["openai"])
MAXTOK = 48000
SYSTEM = ("You are a meticulous code reviewer for research infrastructure (Python, data provenance, fail-closed checks). Review ONLY what you are "
          "given. Report REQUIRED findings (bugs, fail-open paths, wrong logic against the stated requirements) and OPTIONAL findings separately, "
          "each with line numbers. Be concrete and skeptical; do not invent requirements. End with exactly one line: 'VERDICT: BLOCKING: none' "
          "or 'VERDICT: BLOCKING: <numbered list>'.")
key = next((ln.split("=", 1)[1].strip().strip('"').strip("'") for ln in Path.home().joinpath("probe-distribution-shift-from-sims/.env").read_text().splitlines()
            if ln.startswith("OPENROUTER_API_KEY=")), "")
if not key:
    sys.exit("OPENROUTER_API_KEY missing (value not shown)")
msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": (W / "context.md").read_text() + "\n\n# Question\n" + (W / "question.md").read_text()}]
m = {x["id"]: x for x in httpx.get("https://openrouter.ai/api/v1/models", timeout=60).json()["data"]}[JUDGE[1]]
assert JUDGE[2] in m["reasoning"]["supported_efforts"]
worst = (len(json.dumps(msgs)) + 1000) * float(m["pricing"]["prompt"]) + MAXTOK * float(m["pricing"]["completion"])
print(f"worst case ${worst:.2f}")
assert worst < 5.0
body = {"model": JUDGE[1], "messages": msgs, "max_tokens": MAXTOK, "reasoning": {"effort": JUDGE[2]}, "usage": {"include": True},
        "provider": {"order": JUDGE[3], "allow_fallbacks": False}}
t = time.time()
r = httpx.post("https://openrouter.ai/api/v1/chat/completions", json=body, timeout=1800, headers={"Authorization": f"Bearer {key}"})
d = r.json()
if r.status_code != 200:
    sys.exit(f"http {r.status_code}: {str(d)[:500]}")
ch = d["choices"][0]
res = {"judge": JUDGE[0], "slug": JUDGE[1], "effort": JUDGE[2], "finish_reason": ch.get("finish_reason"), "text": ch["message"].get("content"),
       "cost": (d.get("usage") or {}).get("cost"), "secs": round(time.time() - t), "prompt_sha256": hashlib.sha256(json.dumps(msgs).encode()).hexdigest()}
stamp = time.strftime("%Y-%m-%d_%H%M%S")
(W / f"review_{stamp}.jsonl").write_text(json.dumps(res) + "\n")
(W / f"review_{stamp}.md").write_text(f"## {res['judge']} (finish {res['finish_reason']}, cost {res['cost']}, {res['secs']} s)\n\n{res['text']}\n")
print(stamp, res["finish_reason"], res["cost"])
