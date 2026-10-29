"""Qwen3.5-27B as the reader: token counts and segment-end positions for every staged transcript of the given texts.

The qwen35 track's copy of experiments/2026-10-06_qwen35_reader/run_token_counts.py (a0a4cda), with `--texts` (name=staged
run, comma-separated), this track's app name and a read-only mount: the parse check before each new text's GPU
extraction (one failing transcript would crash its whole shard there). The original docstring follows.


Reads the staged reader messages of the Qwen3-32B segment-end run (original text) and of its level-5 scrubbed run, renders
each with Qwen3.5-27B's chat template (its tokenizer at the pinned revision, from the HF cache Volume) and runs
`probe_shift.reader.segment_end_positions` on the whole transcript (no 32k cut). On Modal CPU; no model weights.
Writes one row per (text, arm, key): tokens, positions found, empty content ends, and any failure.

  uv run modal run experiments/2026-10-06_qwen35_replication/run_token_counts_qwen35.py --texts scrub7=2026-10-06_la_segment-end-scrub7_qwen3-32b_v1
"""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-qwen35_token-counts_kill-2026-11-06"

app = modal.App(APP)
acts_vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2).read_only()
hf_cache = modal.Volume.from_name("probe-shift_aswin-surya_hf-cache")
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("transformers==4.56.2", "jinja2")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}).add_local_python_source("probe_shift"))


def count_one(line: str) -> dict:
    from transformers import AutoTokenizer

    from probe_shift.models import QWEN35_27B
    from probe_shift.reader import segment_end_positions

    global _tok
    if "_tok" not in globals():
        _tok = AutoTokenizer.from_pretrained(QWEN35_27B["name"], revision=QWEN35_27B["revision"])
    it = json.loads(line)
    row = {"arm": it["arm"], "key": it["key"], "tokens_qwen3": it["tokens"]}
    try:
        r = segment_end_positions(_tok, it["messages"], None)
        kinds = Counter(p["kind"] for p in r["positions"])
        empty = Counter(p["kind"] for p in r["positions"] if p["token"] is None)
        return row | {"tokens": r["n_tokens"], "positions": len(r["positions"]), "by_kind": dict(kinds), "empty": dict(empty)}
    except Exception as e:
        return row | {"error": f"{type(e).__name__}: {str(e)[:200]}"}


@app.function(image=image, volumes={"/acts": acts_vol, "/hf": hf_cache}, cpu=32, memory=65536, timeout=3600)
def count(run: str) -> list[dict]:
    from multiprocessing import Pool

    lines = [l for p in sorted((Path("/acts") / run / "inputs").glob("shard_*.jsonl")) for l in p.read_text().splitlines()]
    ids = [(json.loads(l)["arm"], json.loads(l)["key"]) for l in lines]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{len(ids) - len(set(ids))} duplicate transcripts in {run}/inputs")
    with Pool(32) as pool:
        return pool.map(count_one, lines, chunksize=8)


@app.local_entrypoint()
def main(texts: str):
    from probe_shift.models import QWEN35_27B

    TEXTS = dict(t.split("=", 1) for t in texts.split(","))

    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    rows = []
    path = HERE / "results" / f"qwen35_token_counts_{stamp}.jsonl"
    for text, rows_t in zip(TEXTS, count.map(list(TEXTS.values()))):
        rows += [r | {"text": text} for r in rows_t]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))  # saved before any summary, failures included
        ok = [r for r in rows_t if "error" not in r]
        if not ok:
            print(f"{text}: all {len(rows_t)} transcripts failed, e.g. {rows_t[0].get('error')}")
            continue
        toks = sorted(r["tokens"] for r in ok)
        print(f"{text}: {len(rows_t)} transcripts, {len(rows_t) - len(ok)} failed; tokens total {sum(toks) / 1e6:.1f}M, mean "
              f"{sum(toks) / len(toks):,.0f}, max {toks[-1]:,}; over 32k {sum(t > 32768 for t in toks)}, over 80k "
              f"{sum(t > 80000 for t in toks)}, over 131k {sum(t > 131072 for t in toks)}; positions {sum(r['positions'] for r in ok):,}; "
              f"Qwen3.5/Qwen3 tokens {sum(r['tokens'] for r in ok) / sum(r['tokens_qwen3'] for r in ok):.3f}")
        for r in rows_t:
            if "error" in r:
                print("  FAILED", text, r["arm"], r["key"][:70], r["error"][:120])
    (HERE / "results" / f"qwen35_token_counts_{stamp}.meta.json").write_text(json.dumps(
        {"model": QWEN35_27B, "texts": TEXTS, "max_tokens": None, "rows": len(rows)}, indent=1))
    print(f"-> {path}")
