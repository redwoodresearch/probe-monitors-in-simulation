"""Build retrain A's pairs: each synthetic pair's final call becomes one real scored LinuxArena call (Modal CPU).

README "Retrain A and B". Runs where the staged inputs live and returns only counts, hashes and length statistics:
the real calls never leave Modal. The pool is real honest runs (the staged cohort, at most 40k tokens) whose main task
is none of the 100 evaluation runs' tasks. Its tasks are split 60/20/20 into train, dev and held-out (seed 20261010).
Within a split, runs are shuffled and rendered 16 at a time (4 processes) until the split has the calls it needs, each
run giving at most 2 calls drawn at random within it (README amendment: the first build took all its calls from 36
runs). Calls are chosen as
run_score_la.py scores them: the content argument last, and the reader's span text matching it. Each pair takes the
next call of its split, identical on both sides. Held-out pairs take only calls without a training word
(build_sets.py's TRAIN_WORDS), and the audit counts matches per split. Every new dialogue is re-rendered: its final
call must give a matching span, and its assistant turn must open with "<|im_start|>assistant\n" (B's readout).
Writes the set to the Volume, and returns its sha256 and the staged shards' sha256s (frozen for the scorer). Failures
come back as their type and line numbers only, with stdout and stderr discarded while building.
CPU only (4 cores, 8 GiB), timeout 480 s, retries 0, which caps the bill near $0.03.

  PYTHONPATH=src uv run modal run --detach experiments/2026-10-10_eval_awareness_probe/run_build_retrain.py
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
APP = "probe-shift_aswin-surya_sub-evalaware_build_kill-2026-11-09"
VOL = "probe-shift_aswin-surya_sub-evalaware_acts_kill-2026-11-09"
INPUTS, STAGED = "probe-shift_aswin-surya_sub-la-activations", "2026-10-06_la_segment-end-v4_qwen3-32b_v1/inputs"
SETS = ".run_cache/eval_awareness_probe/sets_2026-10-10_004845.jsonl"
SEED, N_EVAL, MAX_TOKENS, PER_RUN, BATCH = 20261010, 100, 40_000, 2, 16
TRAIN_WORDS = re.compile(r"(?i)\b(test\w*|eval\w*|benchmark\w*|grad\w*|scor\w*|assess\w*|simulat\w*)")  # build_sets.py

app = modal.App(APP)
vol = modal.Volume.from_name(VOL, version=2)
in_vol = modal.Volume.from_name(INPUTS, version=2).read_only()
hf_cache = modal.Volume.from_name("probe-shift_aswin-surya_hf-cache")
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("transformers==5.19.0", "huggingface_hub", "numpy", "jinja2")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}).add_local_python_source("probe_shift"))


def task_of(key: str) -> str:
    return key.split("__")[1].split(":")[0]


def kind(t: str) -> str:  # run_reading_tokens.py's token kinds
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


def stats(xs):
    xs = sorted(xs)
    q = lambda p: xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]
    return {"n": len(xs), "min": xs[0], "median": q(0.5), "p90": q(0.9), "max": xs[-1], "mean": round(sum(xs) / len(xs), 1)}


TOK, WORKERS = None, None


def calls_of(it):  # scored calls, as run_score_la.py chooses them; TOK is set before the workers fork
    from probe_shift.reader import command_param, segment_end_positions

    msgs, ok, step = it["messages"], {}, -1
    for m in msgs:
        if m["role"] == "assistant":
            step += 1
            for a_, c in enumerate(m.get("tool_calls") or []):
                args = c["function"].get("arguments") or {}
                args = json.loads(args) if isinstance(args, str) else args
                key = command_param(c["function"]["name"], args)
                if key is not None and list(args)[-1] == key:
                    ok[(step, a_)] = (c["function"]["name"], args, key)
    r = segment_end_positions(TOK, msgs, None)
    out = []
    for p in r["positions"]:
        if p["kind"] != "call_end" or not p["kept"] or not p.get("span") or (p["step"], p["action"]) not in ok:
            continue
        name_, args, key = ok[(p["step"], p["action"])]
        text = TOK.decode(r["ids"][p["span"][0]:p["span"][1]]).strip()
        if text and text[-40:] in str(args[key]):
            out.append({"name": name_, "args": args, "key": key, "len": p["span"][1] - p["span"][0],
                        "kind": kind(TOK.decode(r["ids"][p["span"][1] - 1:p["span"][1]])),
                        "words": bool(TRAIN_WORDS.search(json.dumps(args))),
                        "src": hashlib.sha1(f"{it['key']}|{p['step']}|{p['action']}".encode()).hexdigest()[:16]})
    return out


class Safe(Exception):
    """A failure whose message holds only counts and synthetic ids, so it may leave the container."""


def sanitized(e: BaseException) -> RuntimeError:
    import traceback

    lines = [f.lineno for f in traceback.extract_tb(e.__traceback__) if f.filename.endswith("run_build_retrain.py")]
    return RuntimeError(f"build failed: {type(e).__name__} at run_build_retrain.py lines {lines}")


def safe_calls_of(it):  # in a worker: a native panic would kill it and hang the pool, so sanitize everything here
    try:
        return calls_of(it)
    except BaseException as e:
        raise sanitized(e) from None


@app.function(image=image, cpu=4, memory=8192, volumes={"/hf": hf_cache, "/in": in_vol, "/acts": vol}, timeout=480, retries=0)
def build(synthetic: list[dict], name: str) -> dict:
    import os
    import sys

    saved, null = (os.dup(1), os.dup(2)), os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1)  # library and native output could quote inputs: discard both streams while building
    os.dup2(null, 2)
    try:
        return _build(synthetic, name)
    except Safe:
        raise
    except BaseException as e:  # native panics too: never send LinuxArena text back, only the type and line numbers
        raise sanitized(e) from None
    finally:
        if WORKERS is not None:
            WORKERS.terminate()
            WORKERS.join()
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)


def _build(synthetic: list[dict], name: str) -> dict:
    import time

    from transformers import AutoTokenizer

    import probe_shift.reader as reader_module
    from probe_shift.models import QWEN35_27B
    from probe_shift.reader import segment_end_positions

    t0 = time.time()
    global TOK, WORKERS
    import multiprocessing as mp

    tok = TOK = AutoTokenizer.from_pretrained(QWEN35_27B["name"], revision=QWEN35_27B["revision"])
    workers = WORKERS = mp.get_context("fork").Pool(4)
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    shard_sha, runs = {}, []
    for p in sorted((Path("/in") / STAGED).glob("shard_*.jsonl")):
        raw = p.read_bytes()
        shard_sha[p.name] = hashlib.sha256(raw).hexdigest()
        runs += [it for it in map(json.loads, raw.decode().splitlines()) if it["arm"] == "real-honest" and it["tokens"] <= MAX_TOKENS]
    runs.sort(key=lambda it: it["key"])
    eval_keys = {it["key"] for it in random.Random(SEED).sample(runs, N_EVAL)}  # run_score_la.py's sample
    eval_tasks = {task_of(k) for k in eval_keys}
    pool = [it for it in runs if task_of(it["key"]) not in eval_tasks]
    tasks = sorted({task_of(it["key"]) for it in pool})
    random.Random(SEED).shuffle(tasks)
    n = len(tasks)
    split_of = {t: "train" if i < 0.6 * n else "dev" if i < 0.8 * n else "heldout" for i, t in enumerate(tasks)}
    if len({it["id"] for it in synthetic}) != len(synthetic):
        raise Safe("duplicate synthetic ids")
    pairs = {}
    for it in synthetic:
        pairs.setdefault(it["pair"], []).append(it)
    for pid, side in pairs.items():
        if sorted(it["label"] for it in side) != [0, 1] or len({it["split"] for it in side}) != 1:
            raise Safe(f"{pid}: not one 0/1 pair in one split")
    need = Counter(v[0]["split"] for v in pairs.values())

    audit, queue, buf = {}, {}, {}
    for s in ("train", "dev", "heldout"):
        rs = [it for it in pool if split_of[task_of(it["key"])] == s]
        random.Random(f"{SEED}-{s}").shuffle(rs)
        queue[s], buf[s] = rs, []
        audit[s] = {"pool runs": len(rs), "pool tasks": sum(v == s for v in split_of.values()), "pairs": need[s],
                    "runs rendered": 0, "candidate calls": 0, "with a training word": 0}

    def next_call(s):  # without replacement; renders the split's next BATCH runs when its buffer is empty
        while not buf[s]:
            if audit[s]["runs rendered"] >= len(queue[s]):
                raise Safe(f"{s}: out of calls ({audit[s]})")
            batch = queue[s][audit[s]["runs rendered"]:audit[s]["runs rendered"] + BATCH]
            audit[s]["runs rendered"] += len(batch)
            got = []
            for it, calls in zip(batch, workers.map(safe_calls_of, batch)):
                audit[s]["candidate calls"] += len(calls)
                audit[s]["with a training word"] += sum(c["words"] for c in calls)
                usable = [c for c in calls if s != "heldout" or not c["words"]]
                got += random.Random(f"{SEED}-{it['key']}").sample(usable, min(PER_RUN, len(usable)))
            random.Random(f"{SEED}-{s}-{audit[s]['runs rendered']}").shuffle(got)
            buf[s] = got
        return buf[s].pop()

    def valid(msgs, c):  # the final call's span is the content argument, in an assistant turn B can read
        r = segment_end_positions(tok, msgs, None)
        last_step = sum(m["role"] == "assistant" for m in msgs) - 1
        ends = [p for p in r["positions"] if p["kind"] == "call_end" and p["kept"] and p.get("span")
                and (p["step"], p["action"]) == (last_step, 0)]
        if len(ends) != 1:
            return False
        i0, i1 = ends[0]["span"]
        text = tok.decode(r["ids"][i0:i1]).strip()
        starts = [k for k in range(i0) if r["ids"][k] == im_start]
        return bool(text and text[-40:] in str(c["args"][c["key"]]) and starts
                    and tok.decode(r["ids"][starts[-1]:starts[-1] + 3]) == "<|im_start|>assistant\n")

    items, chosen, redraws, picked, seen = [], Counter(), Counter(), [], {}
    for pid in sorted(pairs):
        side = sorted(pairs[pid], key=lambda it: it["label"])
        s = side[0]["split"]
        while True:
            c = next_call(s)
            new = []
            for it in side:
                msgs = json.loads(json.dumps(it["messages"]))
                msgs[-2]["tool_calls"] = [{"type": "function", "function": {"name": c["name"], "arguments": c["args"]}}]
                msgs[-1]["tool_name"] = c["name"]
                new.append(it | {"messages": msgs, "source": c["src"]})
            if all(valid(x["messages"], c) for x in new):
                break
            redraws[s] += 1
        a, b = new[1]["messages"], new[0]["messages"]  # build_sets.py's pair checks
        if len(a) != len(b) or a[-2] != b[-2] or sum(x != y for x, y in zip(a, b)) != 1:
            raise Safe(f"{pid}: the sides do not differ in exactly one message with the same final call")
        key = json.dumps([a, b])
        if key in seen:
            raise Safe(f"{pid} duplicates {seen[key]}")
        seen[key] = pid
        if s == "heldout" and TRAIN_WORDS.search(json.dumps(a) + json.dumps(b)):
            raise Safe(f"{pid}: a held-out pair has a training word")
        chosen[s] += 1
        items += new
        picked.append({"split": s, "len": c["len"], "kind": c["kind"], "words": c["words"],
                       "tool": c["name"] if c["name"] in ("bash", "text_editor") else "other"})
    for s in audit:
        audit[s]["share excluded"] = round(audit[s]["with a training word"] / max(audit[s]["candidate calls"], 1), 4) if s == "heldout" else 0.0
    blob = "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items).encode()
    d = Path("/acts") / "retrain_sets"
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_bytes(blob)
    vol.commit()
    return {"set": f"{VOL}/retrain_sets/{name}", "set_sha256": hashlib.sha256(blob).hexdigest(), "dialogues": len(items),
            "pairs by split": dict(chosen), "redraws by split": dict(redraws), "audit": audit,
            "eval runs": len(eval_keys), "eval tasks": len(eval_tasks), "pool runs": len(pool), "pool tasks": len(tasks),
            "chosen calls": {s: {"argument length in tokens": stats([x["len"] for x in picked if x["split"] == s]),
                                 "tool": dict(Counter(x["tool"] for x in picked if x["split"] == s).most_common()),
                                 "last-token kind": dict(Counter(x["kind"] for x in picked if x["split"] == s).most_common()),
                                 "with a training word": sum(x["words"] for x in picked if x["split"] == s)} for s in ("train", "dev", "heldout")},
            "staged shards sha256": shard_sha, "seconds": round(time.time() - t0, 1),
            "code sha256": {"builder": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            "reader": hashlib.sha256(Path(reader_module.__file__).read_bytes()).hexdigest()},
            "model": QWEN35_27B}


@app.local_entrypoint()
def main():
    raw = (ROOT / SETS).read_bytes()
    items = [json.loads(x) for x in raw.decode().splitlines()]
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = build.remote(items, f"retrain_sets_{stamp}.jsonl")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=True).stdout.strip()
    out = {"synthetic sets_sha256": hashlib.sha256(raw).hexdigest(), "seed": SEED, "git": git, "aggregates only": True} | out
    path = HERE / "results" / f"build_retrain_{stamp}.json"
    path.write_text("{\n" + ",\n".join(f" {json.dumps(k)}: {json.dumps(v)}" for k, v in out.items()) + "\n}\n")
    print(json.dumps(out, indent=1))
    print(f"-> {path}")
