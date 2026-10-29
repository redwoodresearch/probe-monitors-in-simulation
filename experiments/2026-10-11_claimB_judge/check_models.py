"""AI-judge planning ($0): which candidate judge models our keys can reach, and OpenRouter's listed prices. Reads the
repo's .env into this process's environment only (never printed or logged); lists models; prints matching ids and
prices only. No model is called.

  uv run --no-project --python 3.12 --with httpx python experiments/2026-10-11_claimB_judge/check_models.py
"""

import json
import os
import re
from pathlib import Path

import httpx

ENV = Path.home() / "probe-distribution-shift-from-sims/.env"  # the main checkout's .env (git-ignored); keys go into this process only
WANT = re.compile(r"gpt-6-astra|gpt-5\.6-sol|claude-fable|claude-opus-5|gemini-3\.\d-pro|gemini-3\.1", re.I)


def get(url: str, key: str):
    """GET with a bearer key; on any failure return a fixed code, never exception text (it can contain the header)."""
    try:
        r = httpx.get(url, headers={"Authorization": f"Bearer {key.strip()}"}, timeout=30)
        return r.status_code, (r.json() if r.status_code == 200 else None)
    except Exception:  # noqa: BLE001  (deliberately opaque: no message, no chaining, so no key can leak)
        return "request_failed", None


def main():
    for line in ENV.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    out = {}
    st, js = get("https://api.openai.com/v1/models", os.environ["OPENAI_API_KEY"])
    out["openai_status"], out["openai"] = st, sorted(m["id"] for m in (js or {}).get("data", []) if WANT.search(m["id"]))
    # /models/user applies this key's provider preferences and guardrails, so it shows what the key can actually reach
    st, js = get("https://openrouter.ai/api/v1/models/user", os.environ["OPENROUTER_API_KEY"])
    out["openrouter_status"] = st
    out["openrouter"] = {m["id"]: {"prompt_per_M": round(float(m["pricing"]["prompt"]) * 1e6, 4), "completion_per_M": round(float(m["pricing"]["completion"]) * 1e6, 4),
                         "context": m.get("context_length")} for m in (js or {}).get("data", []) if WANT.search(m["id"])}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
