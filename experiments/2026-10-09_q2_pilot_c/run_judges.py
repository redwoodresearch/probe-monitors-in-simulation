"""Pilot C, step 2 (API calls, Aswin's keys through `op run`): two judges from different model families, neither Sol,
answer "did the marked call advance the side task?" for every drawn call (judge_prompt.md). Async, every response
cached by a content hash (data/pilot_c/cache/), so a rerun or resume pays only for missing calls. Raw responses are
model text: data/pilot_c/judgments_<stamp>.jsonl; results/judgments_<stamp>.json keeps each verdict, the reason's
length and sha256, token counts and cost. Unparseable answers are counted and listed, never scored as "no".
--dry-run prints the requests and the estimated cost without calling anything.

  uv run --no-project --with anthropic --with openai python experiments/2026-10-09_q2_pilot_c/run_judges.py \
      data/pilot_c/judge_inputs_<stamp>.jsonl [--env-file <repo>/.env] [--dry-run]   (or under `op run --env-file=.env.tpl`)
"""
import argparse
import asyncio
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
CACHE = ROOT / "data" / "pilot_c" / "cache"
# (name, provider, model, $ per million input tokens, $ per million output tokens): OpenRouter's list, 2026-10-09
# The Anthropic SDK (1.13) takes no sampling settings, so Claude runs at its default.
SETTINGS = {"anthropic": {"max_tokens": 1500}, "openrouter": {"temperature": 0, "max_tokens": 8000}}  # Claude reasons before its JSON
JUDGES = [("primary", "anthropic", "claude-sonnet-5-5", 2.0, 10.0),
          ("second", "openrouter", "google/gemini-3.1-pro-preview", 2.0, 12.0),
          # Aswin, 2026-10-09: Claude refused most calls (cyber policy), so a GPT judge (not Sol) is added as a sanity check
          ("gpt", "openrouter", "openai/gpt-5.4", 2.5, 15.0),
          # Aswin, 2026-10-09: a third family for a majority vote in place of his hand-check of all 20
          ("deepseek", "openrouter", "deepseek/deepseek-v4-pro", 0.21, 0.42),
          ("kimi", "openrouter", "moonshotai/kimi-k3", 0.90, 14.0),
          # Aswin, 2026-10-09: a second GPT model (not Sol; Sol's generation sibling) to see whether GPT judges lean "yes"
          ("gpt56terra", "openrouter", "openai/gpt-5.6-terra", 2.0, 12.0)]
CONCURRENCY = 8


def prompts() -> tuple[str, str]:
    text = (HERE / "judge_prompt.md").read_text()
    system = text.split("# System\n", 1)[1].split("# User", 1)[0].strip()
    user = text.split("# User (after the rendered tasks and transcript)\n", 1)[1].strip()
    return system, user


def parse(text: str) -> str | None:
    """The verdict from a one-line JSON answer (an optional code fence allowed); None if it is not valid."""
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    body = body[body.find("{"): body.rfind("}") + 1] if "{" in body else body
    try:
        d = json.loads(body)
    except json.JSONDecodeError:
        return None
    v = str(d.get("advanced", "")).strip().lower() if isinstance(d, dict) else ""
    return v if v in ("yes", "no") and isinstance(d.get("reason"), str) and d["reason"].strip() else None


async def ask(provider: str, model: str, system: str, user: str, clients: dict) -> dict:
    if provider == "anthropic":
        r = await clients["anthropic"].messages.create(model=model, **SETTINGS["anthropic"], system=system,
                                                       messages=[{"role": "user", "content": user}])
        return {"text": "".join(b.text for b in r.content if b.type == "text"), "input_tokens": r.usage.input_tokens,
                "output_tokens": r.usage.output_tokens}
    r = await clients["openrouter"].chat.completions.create(model=model, **SETTINGS["openrouter"],
                                                            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    if not r.choices:  # OpenRouter returned an error body (e.g. a provider refusal): keep its reason, not a TypeError
        raise RuntimeError(f"no choices: {json.dumps(getattr(r, 'error', None) or r.model_dump().get('error'))[:300]}")
    return {"text": r.choices[0].message.content or "", "input_tokens": r.usage.prompt_tokens, "output_tokens": r.usage.completion_tokens}


async def judge_all(items: list[dict], dry_run: bool, budget: float, judges: list[tuple]) -> list[dict]:
    """budget: no new request starts once the new spend reaches it (requests in flight still finish and are kept)."""
    system, question = prompts()
    sem, clients, spent = asyncio.Semaphore(CONCURRENCY), {}, [0.0]
    if not dry_run:
        from anthropic import AsyncAnthropic
        from openai import AsyncOpenAI
        clients = {"anthropic": AsyncAnthropic(), "openrouter": AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])}
    CACHE.mkdir(parents=True, exist_ok=True)

    async def one(item: dict, judge: tuple) -> dict:
        name, provider, model, p_in, p_out = judge
        user = f"{item['prompt']}\n\n{question}"
        key = hashlib.sha256(f"{provider}\0{model}\0{json.dumps(SETTINGS[provider], sort_keys=True)}\0{system}\0{user}".encode()).hexdigest()
        path = CACHE / f"{key}.json"
        row = {"id": item["id"], "judge": name, "model": model, "cache_key": key}
        cached = path.exists()
        if cached:
            out = json.loads(path.read_text())
        elif dry_run:
            return row | {"dry_run": True, "approx_input_tokens": item["prompt_tokens_o200k"] + 300}
        else:
            async with sem:
                if spent[0] >= budget:
                    return row | {"error": f"skipped: new spend reached the ${budget} budget"}
                out = await ask(provider, model, system, user, clients)
            path.write_text(json.dumps(out))
        cost = out["input_tokens"] * p_in / 1e6 + out["output_tokens"] * p_out / 1e6
        spent[0] += 0 if cached else cost
        return row | out | {"verdict": parse(out["text"]), "cached": cached, "cost_usd": cost}

    # every call runs to the end (paid answers are cached as they arrive); failures are collected, not raised mid-run
    out = await asyncio.gather(*(one(it, j) for it in items for j in judges), return_exceptions=True)
    return [r if not isinstance(r, BaseException) else {"id": it["id"], "judge": j[0], "model": j[2], "error": repr(r)[:300]}
            for r, (it, j) in zip(out, [(it, j) for it in items for j in judges])]


def load_env(path: Path):
    """Keys from a .env file into the environment, without printing them or putting them in arguments."""
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip().strip('"').strip("'")


def main(inputs: Path, dry_run: bool, budget: float, names: list[str] | None):
    items = [json.loads(line) for line in inputs.read_text().splitlines()]
    judges = [j for j in JUDGES if not names or j[0] in names]
    res = asyncio.run(judge_all(items, dry_run, budget, judges))
    if dry_run:
        for name, provider, model, p_in, p_out in judges:
            todo = [r for r in res if r["judge"] == name and r.get("dry_run")]
            toks, cap = sum(r["approx_input_tokens"] for r in todo), SETTINGS[provider]["max_tokens"]
            print(f"{name} {model}: {len(todo)} uncached requests, ~{toks:,} input tokens: ~${toks * p_in / 1e6:.2f} input; output at most "
                  f"{cap} tokens each, so at most ${len(todo) * cap * p_out / 1e6:.2f}")
        return
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(ROOT / "data" / "pilot_c" / f"judgments_{stamp}.jsonl", "w") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in res)
    kept = [{k: v for k, v in r.items() if k != "text"} | ({"text_chars": len(r["text"]), "text_sha256": hashlib.sha256(r["text"].encode()).hexdigest()}
                                                         if "text" in r else {}) for r in res]
    errors = [f"{r['id']}/{r['judge']}: {r['error']}" for r in res if "error" in r]
    failed = [r["id"] + "/" + r["judge"] for r in res if "error" not in r and r["verdict"] is None]
    summary = {"inputs": inputs.name, "judges": [j[:3] for j in judges], "settings": SETTINGS, "requests": len(res), "errors": errors,
               "unparseable": failed, "cost_usd (all answers)": round(sum(r.get("cost_usd", 0) for r in res), 4),
               "cost_usd (new this run)": round(sum(r.get("cost_usd", 0) for r in res if not r.get("cached")), 4), "judgments": kept}
    (HERE / "results").mkdir(exist_ok=True)
    out = HERE / "results" / f"judgments_{stamp}.json"
    out.write_text(json.dumps(summary, indent=1) + "\n")
    print({k: v for k, v in summary.items() if k != "judgments"})
    print(out)
    if errors:  # after everything is saved: a rerun pays only for these
        raise SystemExit(f"{len(errors)} requests failed; rerun to retry them (answers so far are cached)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--env-file", type=Path, help="read API keys from this .env instead of the environment")
    ap.add_argument("--budget", type=float, default=9.0, help="stop starting new requests once this much new spend is reached ($)")
    ap.add_argument("--judges", help="comma-separated judge names to run (default: all)")
    a = ap.parse_args()
    if a.env_file:
        load_env(a.env_file)
    main(a.inputs, a.dry_run, a.budget, a.judges.split(",") if a.judges else None)
