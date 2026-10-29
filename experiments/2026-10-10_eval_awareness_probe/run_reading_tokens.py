"""What token the probe reads at: real LinuxArena calls vs synthetic final calls (Modal CPU, aggregates only).

Aswin's item 2 (21:40 PT): tabulate the content argument's last token by kind and the argument's length, for the far
run's scored calls on the original (unedited) real honest runs, and for the synthetic pairs' final calls (all 1,080
dialogues). The calls are chosen as run_score_la.py does: the content argument comes last and the reader's span text
matches it. Runs in the container, where the staged inputs live, and returns only counts and length statistics: no
argument or transcript text leaves it.

Token kinds, from the last span token's decoded text: whitespace (only spaces or newlines), quote, bracket (ends in
a closing bracket), path (contains "/" or ends in a dot-extension), number, word (letters, digits or underscore),
punctuation (anything else). Also: whether the token ends in a newline, and the tool (bash, text_editor or other: no
free-text labels). CPU only (2 cores, 4 GiB), timeout 240 s, retries 0, which caps the bill near $0.01.

  PYTHONPATH=src uv run modal run --detach experiments/2026-10-10_eval_awareness_probe/run_reading_tokens.py
"""

import hashlib
import json
import random
import re
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
APP = "probe-shift_aswin-surya_sub-evalaware_readtok_kill-2026-11-09"
INPUTS, STAGED = "probe-shift_aswin-surya_sub-la-activations", "2026-10-06_la_segment-end-v4_qwen3-32b_v1/inputs"
SETS = ".run_cache/eval_awareness_probe/sets_2026-10-10_004845.jsonl"
SEED = 20261010

app = modal.App(APP)
in_vol = modal.Volume.from_name(INPUTS, version=2).read_only()
hf_cache = modal.Volume.from_name("probe-shift_aswin-surya_hf-cache")
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("transformers==5.19.0", "huggingface_hub", "numpy", "jinja2")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}).add_local_python_source("probe_shift"))


def kind(t: str) -> str:
    if t.strip() == "":
        return "whitespace"
    s = t.strip()
    if re.fullmatch(r"[\"'`]+", s) or s[-1] in "\"'`":
        return "quote"
    if s[-1] in ")]}>":
        return "bracket"
    if "/" in s or re.search(r"\.[A-Za-z0-9]{1,6}$", s):
        return "path"
    if re.fullmatch(r"[0-9]+", s):
        return "number"
    if re.fullmatch(r"\w+", s):
        return "word"
    return "punctuation"


def tool_kind(name: str) -> str:
    return name if name in ("bash", "text_editor") else "other"


def stats(xs):
    xs = sorted(xs)
    q = lambda p: xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]
    return {"n": len(xs), "min": xs[0], "median": q(0.5), "p90": q(0.9), "max": xs[-1], "mean": round(sum(xs) / len(xs), 1)}


@app.function(image=image, cpu=2, memory=4096, volumes={"/hf": hf_cache, "/in": in_vol}, timeout=240, retries=0)
def tabulate(synthetic: list[dict], exclude: list[str]) -> dict:
    from transformers import AutoTokenizer

    from probe_shift.models import QWEN35_27B
    from probe_shift.reader import command_param, segment_end_positions

    tok = AutoTokenizer.from_pretrained(QWEN35_27B["name"], revision=QWEN35_27B["revision"])
    runs = []
    for p in sorted((Path("/in") / STAGED).glob("shard_*.jsonl")):  # run_score_la.py's seeded sample
        for line in p.read_text().splitlines():
            it = json.loads(line)
            if it["arm"] == "real-honest" and it["tokens"] <= 40_000:
                runs.append(it)
    runs.sort(key=lambda it: it["key"])
    sample = [it for it in random.Random(SEED).sample(runs, 100) if it["key"] not in set(exclude)]

    def summarize(groups):
        out = {}
        for name, calls in groups.items():
            out[name] = {"calls": len(calls), "last-token kind": dict(Counter(c["kind"] for c in calls).most_common()),
                         "last token ends in a newline (share)": round(sum(c["nl"] for c in calls) / max(len(calls), 1), 4),
                         "argument length in tokens": stats([c["len"] for c in calls]) if calls else None,
                         "tool": dict(Counter(c["tool"] for c in calls).most_common())}
        return out

    real = []
    for it in sample:  # scored calls on the original runs, chosen as run_score_la.py does
        msgs = it["messages"]
        r = segment_end_positions(tok, msgs, None)
        ok, step = {}, -1
        for m in msgs:
            if m["role"] == "assistant":
                step += 1
                for a_, c in enumerate(m.get("tool_calls") or []):
                    args = c["function"].get("arguments") or {}
                    args = json.loads(args) if isinstance(args, str) else args
                    key = command_param(c["function"]["name"], args)
                    if key is not None and list(args)[-1] == key:
                        ok[(step, a_)] = (str(args[key]), c["function"]["name"])
        for p in r["positions"]:
            if p["kind"] != "call_end" or not p["kept"] or not p.get("span") or (p["step"], p["action"]) not in ok:
                continue
            i0, i1 = p["span"]
            text = tok.decode(r["ids"][i0:i1]).strip()
            if not text or text[-40:] not in ok[(p["step"], p["action"])][0]:
                continue
            last = tok.decode(r["ids"][i1 - 1:i1])
            real.append({"kind": kind(last), "nl": last.endswith("\n"), "len": i1 - i0, "tool": tool_kind(ok[(p["step"], p["action"])][1])})
    syn = []
    for it in synthetic:  # the final call, as the extraction reads it
        r = segment_end_positions(tok, it["messages"], None)
        ends = [p for p in r["positions"] if p["kind"] == "call_end" and p["kept"] and p.get("span")]
        last = max(ends, key=lambda p: (p["step"], p["action"]))
        i0, i1 = last["span"]
        t = tok.decode(r["ids"][i1 - 1:i1])
        syn.append({"kind": kind(t), "nl": t.endswith("\n"), "len": i1 - i0, "tool": tool_kind(it["messages"][-2]["tool_calls"][0]["function"]["name"])})
    return {"runs": len(sample), "real (original runs, scored calls)": summarize({"all": real}),
            "synthetic (final calls)": summarize({"all": syn, "heldout": [s for s, it in zip(syn, synthetic) if it["split"] == "heldout"]})}


@app.local_entrypoint()
def main():
    raw = (ROOT / SETS).read_bytes()
    items = [json.loads(x) for x in raw.decode().splitlines()]
    gate = json.loads((HERE / "results" / "step1_gate_2026-10-10_025308.json").read_text())
    exclude = gate["per_token"]["(b) runs without a scored call in both versions (excluded)"]
    out = tabulate.remote(items, exclude)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=True).stdout.strip()
    out = {"sets_sha256": hashlib.sha256(raw).hexdigest(), "excluded runs": exclude, "git": git, "aggregates only": True} | out
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"reading_tokens_{stamp}.json"
    path.write_text("{\n" + ",\n".join(f" {json.dumps(k)}: {json.dumps(v)}" for k, v in out.items()) + "\n}\n")
    print(json.dumps(out, indent=1))
    print(f"-> {path}")
