"""Offline check of the Qwen3.5 call_end fix (2026-10-07): which readout positions the fixed reader moves.

Codex's #47 finding: a Qwen3.5 call whose last XML argument is empty (e.g. `new_str: ""`) had no call_end. The fix
reads the last non-empty argument instead (the function name if none has content). For every transcript of each
Qwen3.5 text this compares three things:
  - stored: the positions in the stored extraction's metadata (this track's Volume);
  - old: the reader that made those extractions (`src/probe_shift/reader.py` at a0a4cda, passed in as source);
  - new: the fixed reader (this checkout's `probe_shift.reader`).
It checks that old reproduces stored exactly (same tokens and positions, so the texts and tokenizer are the ones the
extraction read), that old and new tokenize identically, and that new differs from old only at call_end positions
old left empty. For each changed call it keeps the new token, its decoded text and the argument value it ends. On the
original texts it also runs the Qwen3-32B reader on the same messages and counts its call_end positions with no token.
Also records each staged input shard's sha256. Modal CPU, tokenizers only (no weights, no GPU).

  uv run modal run experiments/2026-10-06_qwen35_replication/analyze_callend_fix_positions.py
"""

import gzip
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
APP = "probe-shift_aswin-surya_sub-qwen35_callend-fix-check_kill-2026-11-06"
OLD_READER = "a0a4cda"  # the reader of every stored Qwen3.5 extraction (reader.py's last change before them)
PIP = ["transformers==5.19.0", "jinja2", "numpy", "huggingface_hub"]  # the extraction's transformers

app = modal.App(APP)
in_vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2).read_only()
acts_vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-qwen35_acts_kill-2026-11-06", version=2).read_only()
hf_cache = modal.Volume.from_name("probe-shift_aswin-surya_hf-cache")
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PIP).env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})
         .add_local_python_source("probe_shift"))
FIELDS = ("kind", "step", "action", "tool", "token", "token_text", "kept", "span")


def filename(key: str) -> str:
    """The extraction's file stems (run_extract_qwen35.py)."""
    import hashlib

    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


def _init(old_src: str, qwen3: bool):
    import types

    from transformers import AutoTokenizer

    from probe_shift.models import QWEN3_32B, QWEN35_27B

    global _old, _tok, _tok3
    _old = types.ModuleType("reader_old")
    exec(compile(old_src, f"reader.py@{OLD_READER}", "exec"), _old.__dict__)
    _tok = AutoTokenizer.from_pretrained(QWEN35_27B["name"], revision=QWEN35_27B["revision"])
    _tok3 = AutoTokenizer.from_pretrained(QWEN3_32B["name"], revision=QWEN3_32B["revision"]) if qwen3 else None


def _one(args) -> dict:
    import bisect
    import hashlib

    from probe_shift import reader as new_reader

    line, meta_path = args
    it = json.loads(line)
    row = {"arm": it["arm"], "key": it["key"]}
    stored = json.loads(Path(meta_path).read_text())
    old = _old.segment_end_positions(_tok, it["messages"], None)
    new = new_reader.segment_end_positions(_tok, it["messages"], None)
    pick = lambda p: tuple(json.dumps(p.get(f)) for f in FIELDS)
    row["stored_matches_old"] = (stored["n_tokens"] == old["n_tokens"]
                                 and [pick(p) for p in stored["positions"]] == [pick(p) for p in old["positions"]])
    row["same_ids"] = hashlib.sha256(json.dumps(old["ids"]).encode()).digest() == hashlib.sha256(json.dumps(new["ids"]).encode()).digest()
    if len(old["positions"]) != len(new["positions"]):
        row["changed_other"] = ["position count differs"]
        return row
    changed, other = [], []
    agent = [m for m in it["messages"] if m["role"] == "assistant"]
    text, found, enc = new_reader._layout(_tok, it["messages"])
    offsets = enc["offset_mapping"]
    starts, ends = [o[0] for o in offsets], [o[1] for o in offsets]
    whole = lambda c0, c1: list(range(bisect.bisect_left(starts, c0), bisect.bisect_right(ends, c1)))  # tokens inside
    calls_at = {}  # (step, action) -> (call start, candidate content spans: values last first, then the name, close)
    for kind, step, _, _, a, b in found:
        if kind == "agent":
            for i, (k, cands, close, _, _) in enumerate(new_reader._calls(text, a, b, "qwen3.5")):
                calls_at[(step, i)] = (k, cands, close)
    as_text = lambda v: v if isinstance(v, str) else json.dumps(v)

    def call_parts(p):
        call = agent[p["step"]]["tool_calls"][p["action"]]["function"]
        args_ = call.get("arguments") or {}
        args_ = json.loads(args_) if isinstance(args_, str) else args_
        labels = [k for k in reversed(list(args_))] + ["<function name>"]  # the candidates' order
        values = [as_text(v) for v in reversed(list(args_.values()))] + [call["name"]]
        return call, labels, values

    for po, pn in zip(old["positions"], new["positions"]):
        if pick(po) == pick(pn):
            continue
        if po["kind"] == "call_end" and po["token"] is None and pn["kind"] == "call_end" and pn["token"] is not None:
            _, labels, values = call_parts(pn)
            _, cands, _ = calls_at[(pn["step"], pn["action"])]
            t = pn["token"]
            j = next((j for j, (c0, c1) in enumerate(cands) if c0 <= offsets[t][0] and offsets[t][1] <= c1), None)
            c0, c1 = cands[j] if j is not None else (0, 0)
            changed.append({"step": pn["step"], "action": pn["action"], "tool": pn["tool"], "token": t,
                            "token_text": pn["token_text"], "decoded": _tok.decode([new["ids"][t]]),
                            "argument": labels[j] if j is not None else None, "value_tail": values[j][-60:] if j is not None else None,
                            "skipped": [[labels[x], values[x][-20:]] for x in range(j or 0)],
                            "token_inside": j is not None,  # the token lies entirely inside one candidate's text
                            "value_matches": j is not None and text[c0:c1] == values[j],  # which is that argument's value
                            "token_is_last": j is not None and whole(c0, c1)[-1:] == [t],  # its last whole token
                            "skipped_have_no_token": j is not None and not any(whole(*cands[x]) for x in range(j)),
                            "after_token": text[offsets[t][1]:c1]})
        else:
            other.append({"old": po, "new": pn})
    none_left = []
    for pn in new["positions"]:
        if pn["kind"] == "call_end" and pn["token"] is None:
            call, labels, values = call_parts(pn)
            k, cands, close = calls_at[(pn["step"], pn["action"])]
            none_left.append({"step": pn["step"], "action": pn["action"], "tool": pn["tool"],
                              "call_text_tail": text[max(k, close - 300):close], "arguments": json.dumps(call.get("arguments"))[-300:]})
    row |= {"changed": changed, "changed_other": other, "call_end_none_new": len(none_left), "none_left": none_left}
    if _tok3 is not None:
        q3 = new_reader.segment_end_positions(_tok3, it["messages"], None)
        row["qwen3_call_end"] = sum(p["kind"] == "call_end" for p in q3["positions"])
        row["qwen3_call_end_none"] = sum(p["kind"] == "call_end" and p["token"] is None for p in q3["positions"])
    return row


@app.function(image=image, volumes={"/in": in_vol, "/out": acts_vol, "/hf": hf_cache}, cpu=32, memory=65536, timeout=3600)
def check(text: str, inputs_run: str, acts_run: str, old_src: str, qwen3: bool) -> dict:
    import hashlib
    from multiprocessing import Pool

    shards = sorted((Path("/in") / inputs_run / "inputs").glob("shard_*.jsonl"))
    sha = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in shards}
    lines = [l for p in shards for l in p.read_text().splitlines()]
    jobs = []
    for l in lines:
        it = json.loads(l)
        jobs.append((l, str(Path("/out") / acts_run / "meta" / it["arm"] / f"{filename(it['key'])}.json")))
    with Pool(32, initializer=_init, initargs=(old_src, qwen3)) as pool:
        rows = pool.map(_one, jobs, chunksize=4)
    return {"text": text, "inputs_run": inputs_run, "acts_run": acts_run, "input_sha256": sha, "rows": rows}


@app.local_entrypoint()
def main(texts: str = ""):
    import yaml

    old_src = subprocess.run(["git", "show", f"{OLD_READER}:src/probe_shift/reader.py"], capture_output=True, text=True,
                             check=True, cwd=ROOT).stdout
    import hashlib

    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=ROOT).stdout.strip()
    tested = [ROOT / "src" / "probe_shift" / "reader.py", Path(__file__).resolve()]
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *map(str, tested)], capture_output=True, text=True,
                           check=True, cwd=ROOT).stdout.strip()
    if dirty:  # the audit must name the exact reader it tested: commit it first
        raise SystemExit(f"commit the reader and this checker before auditing them:\n{dirty}")
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in tested}
    runs = {}
    for f in sorted((HERE / "config").glob("qwen35_alllayers*.yaml")):
        if "nopad" in f.name:
            continue
        c = yaml.safe_load(f.read_text())
        runs[f.stem.removeprefix("qwen35_alllayers").lstrip("_") or "original"] = (c["inputs_from"], c["run"])
    if texts:
        runs = {k: v for k, v in runs.items() if k in texts.split(",")}
    jobs = [(t, i, r, old_src, t in ("original", "v4")) for t, (i, r) in runs.items()]
    out = list(check.starmap(jobs))
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"callend_fix_positions_{stamp}.json.gz"  # every changed call per text: tens of MB as text
    path.write_bytes(gzip.compress(json.dumps({"git": git, "sources_sha256": sources, "old_reader": OLD_READER, "pip": PIP,
                                               "texts": out}).encode()))
    for t in out:
        rows = t["rows"]
        ch = sum(len(r.get("changed", [])) for r in rows)
        other = sum(len(r.get("changed_other", [])) for r in rows)
        line = (f"{t['text']:10} {len(rows):5} transcripts | stored==old {sum(r['stored_matches_old'] for r in rows)} | "
                f"same ids {sum(r['same_ids'] for r in rows)} | call_end None->token {ch} in "
                f"{sum(bool(r.get('changed')) for r in rows)} transcripts | other changes {other} | "
                f"call_end None left {sum(r.get('call_end_none_new', 0) for r in rows)}")
        if "qwen3_call_end" in rows[0]:
            line += (f" | Qwen3-32B call_end None {sum(r['qwen3_call_end_none'] for r in rows)} of "
                     f"{sum(r['qwen3_call_end'] for r in rows)}")
        print(line)
    print(f"-> {path}")
