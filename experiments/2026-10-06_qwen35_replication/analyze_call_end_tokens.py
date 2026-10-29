"""Which tokens does Qwen3.5-27B's call_end read, per text level (2026-10-06)?

From scrub level 2 on, numbers are masked as `<n>`, so a bash call's last content token can be the placeholder rather
than command text; Qwen3-32B's call_end then reads `>` (main's Amendment 3). Qwen3.5 writes tool calls as XML with raw
values and its own tokenizer, so its fixed slots may differ. Counts call_end token texts over every transcript of each
Qwen3.5 run, overall and for the last parameter being `timeout`-like (a bare `<n>`), on Modal CPU.

  uv run modal run experiments/2026-10-06_qwen35_replication/analyze_call_end_tokens.py
"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-qwen35_callend-tokens_kill-2026-11-06"
OUTPUT = "probe-shift_aswin-surya_sub-qwen35_acts_kill-2026-11-06"
RUNS = {"original": "2026-10-06_la_segment-end_qwen35-27b_alllayers_v1",
        **{f"scrub{l}": f"2026-10-06_la_segment-end-scrub{l}_qwen35-27b_alllayers_v1" for l in range(1, 8)}}

app = modal.App(APP)
vol = modal.Volume.from_name(OUTPUT, version=2).read_only()
image = modal.Image.debian_slim(python_version="3.12")


@app.function(image=image, volumes={"/out": vol}, cpu=16, memory=16384, timeout=1800)
def count(run: str) -> dict:
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor

    paths = sorted((Path("/out") / run / "meta").glob("*/*.json"))
    with ThreadPoolExecutor(32) as pool:
        metas = list(pool.map(lambda p: json.loads(p.read_text()), paths))
    tokens, by_tool, n = Counter(), Counter(), 0
    for m in metas:
        if m["arm"] not in ("real-attack-e1", "real-attack-e2", "sim-attack"):
            continue
        for p in m["positions"]:
            if p["kind"] == "call_end" and p["token"] is not None:
                n += 1
                tokens[p["token_text"]] += 1
                by_tool[(p["tool"], p["token_text"])] += 1
    return {"run": run, "transcripts": len(metas), "call_end": n, "top_tokens": tokens.most_common(25),
            "top_tool_tokens": [[list(k), v] for k, v in by_tool.most_common(25)]}


@app.local_entrypoint()
def main():
    out = dict(zip(RUNS, count.map(list(RUNS.values()))))
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"call_end_tokens_{stamp}.json"
    path.write_text(json.dumps(out, indent=1))
    for text, r in out.items():
        top = ", ".join(f"{t!r}:{c / r['call_end']:.1%}" for t, c in r["top_tokens"][:8])
        print(f"{text:9} {r['call_end']:>6} call_end (attack arms) | {top}")
    print(f"-> {path}")
