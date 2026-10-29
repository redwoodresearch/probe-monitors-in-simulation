"""A fourth reader (PLAN.md here): Llama-3.3-70B-Instruct on #36's staged transcripts and fix939's re-rendered ones,
at the R2J readout of every tool output that Qwen3-32B kept (C2). Built on ../2026-10-12_q2_olmo_reader/run_extract_olmo.py
(same layout, chunk markers, resume, orchestrator and check), with Llama's template and the JSON-escaped readout. The
weights come read-only from our Llama Volume. The shared code is untouched, and transformers lives only in this job's
images. One position function, positions(), serves every mode.

  --mode render    (CPU, counts only): per arm, newline membership of R0, R1 and R2J at C2, R2J's coverage, slip
                   survival per component, the JSON round trip, forward tokens.
  --mode diff      (CPU): one real-honest and one simulated reply each of str_replace and insert, as token lists with
                   every content token masked; a leak check fails the job if any other token text would leave.
  --mode extract   (2 H100s per shard, through one remote orchestrator): the residual stream after every decoder
                   block, all 80 layers, float16, at the kept R2J readouts. --test: the 4 longest transcripts on one
                   shard under a separate run name (the smoke test).
  --mode check     (CPU, counts only): an extraction's files against its index and meta; forward timing.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run [--detach] \\
      experiments/2026-10-12_q2_llama_reader/run_extract_llama.py --mode <render|diff|extract|check> [--test] \\
      --runs <#36 runs.json> --fix <fix939 per-run jsonl>
"""
import hashlib
import json
import os
import re
import socket
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-llama_extract_kill-2026-11-12"
ACTS = "probe-shift_aswin-surya_sub-la-activations"  # read-only: #36's dump (staged inputs, Qwen3-32B meta) and fix939's
OUT = "probe-shift_aswin-surya_sub-q2-llama_acts_kill-2026-11-12"
WEIGHTS = "probe-shift_aswin-surya_sub-apollo_llama-cache"  # read-only: the Llama snapshot (models--… at its root)
DUMP = "2026-10-06_la_segment-end_qwen3-32b_v1"
FIX_DUMP, FIX_INPUTS, FIX_ARM, FIX_LABEL = ("2026-10-11_q3_fix939_qwen3-32b_l48-63_v1", "2026-10-11_q3_fix939_inputs_v1",
                                            "sim-attack-note-sr-both", "sim-attack-fix939")
RUN = "2026-10-12_la_output-end-r2j_llama33-70b_alllayers_v1"
MODEL, REVISION, CTX, CUT, H, N_LAYERS = "meta-llama/Llama-3.3-70B-Instruct", "6f6073b423013f6a7d4d9f39144961bfbfbc386b", 131072, 32768, 8192, 80
DATE = "26 Jul 2024"  # the template's default date_string, passed explicitly so every run gets the same line
HDR, END = "<|start_header_id|>ipython<|end_header_id|>\n\n", "<|eot_id|>"
GPU, SHARDS, CHUNK, TIMEOUT_S = "H100:2", 24, 25, 4 * 3600
SWAP = HERE.parent / "2026-10-07_q2_rendering_swap"
PIP_CPU = ["transformers==4.57.1", "jinja2==3.1.6", "huggingface_hub==0.35.3", "numpy"]
PIP_GPU = ["torch==2.8.0", "accelerate==1.10.1", *PIP_CPU]
PIN = {"runs.json": "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96",
       "fix939_frozen_runs_POSTHOC_2026-10-11_020458.jsonl": "5f6ecbc42b1c6bb9586cb58e474d74c5c1d914bedb95d80a5a07dd078b0e5ebd"}
ENV = {"HF_HUB_CACHE": "/llama", "HF_HUB_OFFLINE": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2).read_only()
weights = modal.Volume.from_name(WEIGHTS).read_only()
out_vol = modal.Volume.from_name(OUT, version=2, create_if_missing=True)
cpu_image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PIP_CPU).env(ENV)
             .add_local_file(SWAP / "rerender.py", "/root/rerender.py"))
gpu_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PIP_GPU).env(ENV)


class Fail(Exception):
    def __init__(self, stage):
        super().__init__(stage); self.stage = stage


def prep(msgs):
    """Llama's template, per PLAN.md: string content; tool-call arguments as a dict; each assistant turn split into its
    text (if any, after trimming) and one message per tool call. Tool outputs and everything else in their order."""
    out = []
    for m in msgs:
        m = dict(m); c = m.get("content")
        m["content"] = "".join(p.get("text", "") for p in c if p.get("type") == "text") if isinstance(c, list) else (c or "")
        if m["role"] == "assistant" and m.get("tool_calls"):
            calls = [{**tc, "function": {**(tc.get("function") or {}), "arguments": (lambda a: json.loads(a) if isinstance(a, str) else (a or {}))(
                (tc.get("function") or {}).get("arguments"))}} for tc in m["tool_calls"]]
            if m["content"].strip():
                out.append({"role": "assistant", "content": m["content"]})
            out += [{"role": "assistant", "content": "", "tool_calls": [tc]} for tc in calls]
        else:
            m.pop("tool_calls", None)
            out.append(m)
    return out


def classes(e: str) -> tuple[list[str], list[int]]:
    """Per character of a JSON string body (no quotes): the class of the character it encodes ("nl" for a newline,
    "ws" for other whitespace, "c" for content; an escape sequence's characters all take its decoded character's class),
    and the decoded index each escaped character belongs to."""
    esc = {"n": "\n", "r": "\r", "t": "\t", '"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f"}
    cls, dec, i, k = [], [], 0, 0
    while i < len(e):
        if e[i] == "\\":
            ch, L = (chr(int(e[i + 2:i + 6], 16)), 6) if e[i + 1] == "u" else (esc[e[i + 1]], 2)
        else:
            ch, L = e[i], 1
        c = "nl" if ch == "\n" else "ws" if ch.strip() == "" else "c"
        cls += [c] * L; dec += [k] * L; i += L; k += 1
    return cls, dec


def positions(tok, msgs, q):
    """One staged transcript under Llama's template. q: Qwen3-32B's output_end positions, {"step,action": [kept, tool]}.
    Returns ids, the meta positions (every call's call_end, not kept; every output's output_end with R2J as its token,
    kept under C2 and below CTX), and per output (qwen kept, R0, R1, R2J as (index, classes), raw content, the
    rendered span (a, b) of the escaped body, its classes) for the render and diff modes. Raises Fail(stage)."""
    import bisect
    staged = [dict(m) for m in msgs]
    tools = [m for m in staged if m["role"] == "tool"]
    for m in tools:
        c = m.get("content")
        m["content"] = "".join(p.get("text", "") for p in c if p.get("type") == "text") if isinstance(c, list) else (c or "")
    text = tok.apply_chat_template(prep(msgs), tokenize=False, date_string=DATE)
    if text.count(HDR) != len(tools) or any(HDR in m["content"] or END in m["content"] for m in tools):
        raise Fail("a delimiter quoted in content")
    spans, calls, pos, step, action = [], [], 0, -1, 0
    for m in staged:
        if m["role"] == "assistant":
            step, action = step + 1, 0
            calls += [(step, i, (tc.get("function") or {}).get("name")) for i, tc in enumerate(m.get("tool_calls") or [])]
        elif m["role"] == "tool":
            a = text.index(HDR, pos) + len(HDR); enc = json.dumps(m["content"], ensure_ascii=False); b = a + len(enc)
            if text[a:b] != enc or not text.startswith(END, b):
                raise Fail("a tool output not rendered as its JSON string")
            spans.append((step, action, a + 1, b - 1, m["content"])); action += 1; pos = b
    enc_ = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc_["input_ids"]; starts, ends = [s for s, _ in enc_["offset_mapping"]], [e for _, e in enc_["offset_mapping"]]

    def inside(a, b):  # the tokens lying entirely inside [a, b)
        return bisect.bisect_left(starts, a), bisect.bisect_right(ends, b)
    out = [{"kind": "call_end", "step": s, "action": i, "tool": name, "token": None, "kept": False, "token_text": None} for s, i, name in calls]
    per = []
    for s, a_, a, b, content in spans:
        cls, dec = classes(text[a:b])
        tc = lambda t: cls[starts[t] - a:ends[t] - a]  # noqa: E731
        i0, i1 = inside(a, b)
        r0 = i1 - 1 if i1 > i0 else None
        last_c = max((j for j, c in enumerate(cls) if c == "c"), default=None)
        j0, j1 = inside(a, a + last_c + 1) if last_c is not None else (0, 0)
        r1 = j1 - 1 if j1 > j0 else None
        r2 = next((t for t in range(i1 - 1, i0 - 1, -1) if "nl" not in tc(t) and any(c != "ws" for c in tc(t))), None)
        qk = q.get(f"{s},{a_}")
        kept = bool(qk and qk[0]) and r2 is not None and r2 < CTX
        out.append({"kind": "output_end", "step": s, "action": a_, "tool": qk[1] if qk else None, "token": r2, "kept": kept, "token_text": None})
        per.append({"qk": qk, "rules": {r: (t, None if t is None else tc(t)) for r, t in (("R0", r0), ("R1", r1), ("R2J", r2))},
                    "content": content, "span": (a, b), "cls": cls, "dec": dec})
    return {"ids": ids, "n_tokens": len(ids), "positions": out, "outputs": per, "text": text, "starts": starts, "ends": ends}


def staged(run_dir, by_key):
    out = {}
    for shard in sorted((Path("/acts") / run_dir / "inputs").glob("shard_*.jsonl")):
        for line in shard.read_text().splitlines():
            it = json.loads(line); out[it["key"] if by_key else (it["arm"], it["key"])] = it
    return out


def qsum(meta):  # Qwen3-32B's output_end positions: [kept, tool] by "step,action"
    return {f"{p['step']},{p['action']}": [bool(p["kept"]), p.get("tool")] for p in meta["positions"] if p["kind"] == "output_end"}


def work_items(cohort, fix_keys):
    """(label, filename, staged item, Qwen meta path) for #36's runs and fix939's re-rendered ones."""
    base, fixed = staged(DUMP, False), staged(FIX_INPUTS, True)
    items = [(r["arm"], r["filename"], base[(r["arm"], r["key"])], Path("/acts") / DUMP / "acts" / r["arm"] / f"{r['filename']}.json") for r in cohort]
    fname = {r["key"]: r["filename"] for r in cohort if r["arm"] == "sim-attack"}
    items += [(FIX_LABEL, fname[k], fixed[k], Path("/acts") / FIX_DUMP / "meta" / FIX_ARM / f"{fname[k]}.json") for k in fix_keys]
    return items


GUTTER = re.compile(r"(?m)^ *\d+\t")
HUNK = re.compile(r"(?m)^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@")


# ---------- render (CPU, counts only) ----------
def _init():
    import sys

    from transformers import AutoTokenizer
    sys.path.insert(0, "/root")
    from rerender import REVIEW
    global TOK, REV
    TOK, REV = AutoTokenizer.from_pretrained(MODEL, revision=REVISION), REVIEW.strip()


def count_one(item):
    """Counts for one run. A failure is counted by its stage and type only (an exception may quote content), never raised."""
    from collections import Counter
    label, _, it, qpath = item
    c = Counter(); c[(label, "runs")] += 1
    try:
        r = positions(TOK, it["messages"], qsum(json.loads(qpath.read_text())))
    except Exception as e:
        c[(label, f"render failure: {getattr(e, 'stage', 'render')} ({type(e).__name__})")] += 1
        return c
    kept_idx = [p["token"] for p in r["positions"] if p["kept"]]
    c[(label, "Llama tokens")] += r["n_tokens"]; c[(label, "forward tokens (to the last kept readout)")] += max(kept_idx) + 1 if kept_idx else 0
    c[(label, "runs with no kept readout")] += not kept_idx
    for o in r["outputs"]:
        qk, content = o["qk"], o["content"]; a, b = o["span"]
        c[(label, "tool outputs")] += 1
        c[(label, "tool outputs with no Qwen3-32B position")] += qk is None
        rt = json.loads(r["text"][a - 1:b + 1]) == content
        c[(label, "tool outputs whose rendered span JSON-decodes exactly to the raw content")] += rt
        for comp, has in (("(a) trailing newline", content.endswith("\n")), ("(b) str_replace gutter", bool(GUTTER.search(content))),
                          ("(c) insert review note", REV in content), ("(d) hunk header", bool(HUNK.search(content)))):
            c[(label, f"slip {comp}: in the raw output")] += has
            c[(label, f"slip {comp}: in the raw output and in the rendered span")] += has and rt
        if not (qk and qk[0]):
            continue
        r2 = o["rules"]["R2J"][0]
        c[(label, "outputs Qwen3-32B kept")] += 1
        c[(label, "of those, kept under C2 (R2J readout below the context length)")] += r2 is not None and r2 < CTX
        c[(label, "of those, no R2J readout")] += r2 is None
        c[(label, "of those, R2J at index 32,768 or beyond")] += r2 is not None and r2 >= CUT
        for rule, (t, cl) in o["rules"].items():
            c[(label, f"{rule} readouts at C2")] += t is not None
            c[(label, f"{rule} readouts at C2: encodes only whitespace")] += t is not None and all(x in ("ws", "nl") for x in cl)
            c[(label, f"{rule} readouts at C2: encodes a newline")] += t is not None and "nl" in cl
    return c


@app.function(image=cpu_image, volumes={"/acts": acts, "/llama": weights}, cpu=32, memory=65536, timeout=3600, retries=0)
def render(cohort: list[dict], fix_keys: list[str]) -> dict:
    try:
        return _render(cohort, fix_keys)
    except Exception as e:  # nothing from the exception (which may quote staged text) crosses the boundary
        raise RuntimeError(f"render job failed: {type(e).__name__}") from None


def _render(cohort, fix_keys):
    from collections import Counter
    from multiprocessing import Pool
    items = work_items(cohort, fix_keys)
    total = Counter()
    with Pool(32, initializer=_init) as pool:
        for cnt in pool.imap(count_one, items, chunksize=4):
            total.update(cnt)
    return {"rows": [{"arm": a, "count": k, "n": int(v)} for (a, k), v in sorted(total.items())]}


# ---------- diff (CPU, masked tokens only) ----------
@app.function(image=cpu_image, volumes={"/acts": acts, "/llama": weights}, cpu=4, memory=16384, timeout=1800, retries=0)
def diff(cohort: list[dict], fix_keys: list[str]) -> dict:
    try:
        return _diff(cohort, fix_keys)
    except Exception as e:
        raise RuntimeError(f"diff job failed: {type(e).__name__}") from None


def _diff(cohort, fix_keys):
    import random
    _init()
    items = [it for it in work_items(cohort, []) if it[0] in ("real-honest", "sim-attack")]
    random.Random(20261012).shuffle(items)
    found, out = set(), []
    for label, _, it, qpath in items:
        cmds = [(tc.get("function") or {}).get("arguments") for m in it["messages"] if m["role"] == "assistant" for tc in (m.get("tool_calls") or [])]
        cmds = [(json.loads(x) if isinstance(x, str) else (x or {})).get("command") for x in cmds]
        r = positions(TOK, it["messages"], qsum(json.loads(qpath.read_text())))
        for cmd, o in zip(cmds, r["outputs"]):
            if cmd not in ("str_replace", "insert") or (label, cmd) in found:
                continue
            a, b = o["span"]; cls, dec = o["cls"], o["dec"]
            decoded = o["content"]; gut = set()
            for m in GUTTER.finditer(decoded):
                gut |= set(range(m.start(), m.end()))
            keep_char = [cls[j] != "c" or dec[j] in gut for j in range(len(cls))]  # whitespace, newline or gutter characters
            blk0, blk1 = r["text"].rfind("<|start_header_id|>", 0, a), b + 1 + len(END)  # this output's whole block
            frame = HDR + '"' + '"' + END  # the block's template text: header, the two quotes, end of turn
            toks = []
            for t in range(len(r["ids"])):
                s_, e_ = r["starts"][t], r["ends"][t]
                if e_ <= blk0 or s_ >= blk1:
                    continue
                txt = r["text"][s_:e_]
                if e_ <= a or s_ >= b:  # wholly template: the header, a quote or the end of turn
                    if not all(ch in frame for ch in txt):
                        raise ValueError("leak check failed: a token outside the body is not template text")
                    toks.append(txt)
                elif a <= s_ and e_ <= b and all(keep_char[j - a] for j in range(s_, e_)):  # escaped whitespace, newlines, gutter
                    if not re.fullmatch(r"[\s\\0-9ntrfu]+", txt):  # whitespace, backslashes, escape letters (split escapes too), digits
                        raise ValueError("leak check failed: a kept body token is not whitespace, an escape or a digit")
                    toks.append(txt)
                else:  # content, or a token straddling content and a quote
                    toks.append("▮")
            out.append({"arm": label, "command": cmd, "tokens": toks, "n_tokens": len(toks), "masked": toks.count("▮"),
                        "raw ends with a newline": decoded.endswith("\n"), "rendered body ends with an escaped newline": r["text"][b - 2:b] == "\\n"})
            found.add((label, cmd))
        if len(found) == 4:
            break
    return {"replies": out}


# ---------- extract (GPU) ----------
# No automatic retry (a retry once started a second container on a running shard). After a crash, relaunch: the
# resume skips every transcript already done.
@app.function(image=gpu_image, gpu=GPU, memory=131072, volumes={"/llama": weights, "/out": out_vol, "/acts": acts}, timeout=TIMEOUT_S, retries=0)
def extract_shard(cfg: dict, shard: int, cohort: list[dict], fix_keys: list[str]) -> list[dict]:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    items = work_items(cohort, fix_keys)
    if cfg.get("scope") == "fix939":  # part 2's launch: only the 823 re-rendered runs
        items = [it for it in items if it[0] == FIX_LABEL]
    items.sort(key=lambda it: (-(it[2].get("tokens") or 0), it[0], it[2]["key"]))  # longest first (Qwen's count), dealt round-robin
    mine = items[shard::cfg["shards"]]
    run_dir = Path("/out") / cfg["run"]
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    if cfg.get("limit_items"):  # the smoke test: the longest Llama forward prefixes among the 40 longest by Qwen's count
        def fwd(it):
            try:
                r = positions(tok, it[2]["messages"], qsum(json.loads(it[3].read_text())))
            except Exception:  # a render failure can't be the smoke test's pick
                return -1
            return max((p["token"] for p in r["positions"] if p["kept"]), default=-1) + 1
        mine = sorted(mine[:40], key=lambda it: -fwd(it))[: cfg["limit_items"]]
    layers = list(range(N_LAYERS))
    for l in layers:
        (run_dir / "by_layer" / f"L{l}").mkdir(parents=True, exist_ok=True)
    (run_dir / "chunks").mkdir(parents=True, exist_ok=True)
    (run_dir / "chunk_markers").touch()
    names = {p.stem for l in layers for p in (run_dir / "by_layer" / f"L{l}").glob(f"shard{shard:03d}_chunk*.npy")}
    names |= {p.stem for p in (run_dir / "chunks").glob(f"shard{shard:03d}_chunk*.json")}
    for name in sorted(n for n in names if not (run_dir / "chunks" / f"{n}.done").exists()):  # drop incomplete chunks
        idx = run_dir / "chunks" / f"{name}.json"
        for t in (json.loads(idx.read_text())["transcripts"] if idx.exists() else []):
            meta = run_dir / "meta" / t["arm"] / f"{t['filename']}.json"
            if meta.exists() and json.loads(meta.read_text()).get("chunk") == name:
                meta.unlink()
        for l in layers:
            (run_dir / "by_layer" / f"L{l}" / f"{name}.npy").unlink(missing_ok=True)
        idx.unlink(missing_ok=True)
        print(f"shard {shard}: dropped incomplete chunk {name} before resuming", flush=True)
    out_vol.commit()
    chunk = len(list((run_dir / "by_layer" / "L0").glob(f"shard{shard:03d}_chunk*.npy")))
    buf, buf_meta, timing, failures = {l: [] for l in layers}, [], [], []

    def shard_stats(extra):  # over every attempt's chunks, counting a transcript only in the chunk its meta points to
        stats = []
        for p in sorted((run_dir / "chunks").glob(f"shard{shard:03d}_chunk*.json")):
            c = json.loads(p.read_text())
            live = [t for t in c["transcripts"] if (f := run_dir / "meta" / t["arm"] / f"{t['filename']}.json").exists()
                    and json.loads(f.read_text())["chunk"] == p.stem]
            stats.append({"chunk": p.stem, "transcripts": len(live), "rows": sum(t["n_rows"] for t in live), "nonfinite": c["nonfinite"]})
        summary = {"shard": shard, "transcripts": len(mine), "chunks": stats} | extra
        (run_dir / "stats").mkdir(exist_ok=True)
        (run_dir / "stats" / f"shard_{shard}.json").write_text(json.dumps(summary))
        out_vol.commit()
        return [summary]
    if all((run_dir / "meta" / label / f"{stem}.json").exists() for label, stem, _, _ in mine):  # resume, after the cleanup
        return shard_stats({"skipped": "every transcript already done; the model was not loaded"})
    t_load = time.time()
    # An even split by hand (accelerate's "auto" with 67 GiB caps offloaded to disk; with 76 GiB it filled GPU 0): layers
    # 0-39 and the embeddings on GPU 0, layers 40-79, the final norm and the head on GPU 1, about 66 GiB each, 13 GiB free.
    dmap = {"model.embed_tokens": 0, "model.rotary_emb": 0, "model.norm": 1, "lm_head": 1} | {f"model.layers.{i}": int(i >= N_LAYERS // 2) for i in range(N_LAYERS)}
    lm = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16, device_map=dmap, attn_implementation="sdpa").eval()
    if any(str(v) in ("cpu", "disk") for v in lm.hf_device_map.values()):  # offloading would be very slow: fail loudly
        raise ValueError("the model did not fit on the two GPUs")
    load_s = round(time.time() - t_load)
    base = lm.model
    if lm.config.num_hidden_layers != N_LAYERS or lm.config.hidden_size != H:
        raise ValueError(f"unexpected shape: {lm.config.num_hidden_layers} layers x {lm.config.hidden_size}")

    def flush():
        nonlocal chunk
        if not buf_meta:
            return
        name = f"shard{shard:03d}_chunk{chunk:03d}"
        nonfinite = 0
        for l in layers:
            x = np.concatenate(buf[l]) if buf[l] else np.zeros((0, H), np.float16)
            nonfinite += int((~np.isfinite(x)).sum())
            tmp = run_dir / "by_layer" / f"L{l}" / f".{name}.npy.tmp"
            with open(tmp, "wb") as fh:
                np.save(fh, x)
            os.replace(tmp, run_dir / "by_layer" / f"L{l}" / f"{name}.npy")
            buf[l].clear()
        index, start = [], 0
        for m in buf_meta:
            index.append({"arm": m["arm"], "key": m["key"], "filename": m["filename"], "row_start": start, "n_rows": m["n_rows"]})
            m |= {"chunk": name, "row_start": start}
            start += m["n_rows"]
        tmp = run_dir / "chunks" / f".{name}.json.tmp"
        tmp.write_text(json.dumps({"layers": layers, "rows": start, "nonfinite": nonfinite, "transcripts": index}))
        os.replace(tmp, run_dir / "chunks" / f"{name}.json")
        for m in buf_meta:
            d = run_dir / "meta" / m["arm"]
            d.mkdir(parents=True, exist_ok=True)
            tmp = d / f".{m['filename']}.json.tmp"
            tmp.write_text(json.dumps(m))
            os.replace(tmp, d / f"{m['filename']}.json")
        (run_dir / "chunks" / f"{name}.done").touch()
        buf_meta.clear()
        chunk += 1
        out_vol.commit()

    t_all = time.time()
    first = base.embed_tokens.weight.device
    for n, (label, stem, it, qpath) in enumerate(mine, 1):
        if (run_dir / "meta" / label / f"{stem}.json").exists():
            continue
        try:
            r = positions(tok, it["messages"], qsum(json.loads(qpath.read_text())))
        except Exception as e:  # counted by stage and type only (the gate allows at most 0.5%); the run gets no meta
            failures.append({"arm": label, "failure": f"{getattr(e, 'stage', 'render')} ({type(e).__name__})"})
            continue
        rows = [p for p in r["positions"] if p["kept"]]
        t0 = time.time()
        if rows:
            last = max(p["token"] for p in rows)
            idx_cpu = torch.tensor([p["token"] for p in rows], dtype=torch.long)
            got = {}

            def hook_for(layer):
                def hook(module, args, output):
                    h = output[0] if isinstance(output, tuple) else output
                    got[layer] = h[0].index_select(0, idx_cpu.to(h.device)).to(torch.float16).cpu().numpy()
                return hook
            handles = [base.layers[l].register_forward_hook(hook_for(l)) for l in layers]
            try:
                with torch.inference_mode():
                    base(input_ids=torch.tensor([r["ids"][: last + 1]], device=first), use_cache=False)
            finally:
                for h_ in handles:
                    h_.remove()
            for l in layers:
                buf[l].append(got[l])
            timing.append([last + 1, round(time.time() - t0, 3)])
        row = iter(range(len(rows)))
        positions_ = [p | {"row": next(row) if p["kept"] else None} for p in r["positions"]]
        buf_meta.append({"key": it["key"], "arm": label, "filename": stem,
                         **{k: it.get(k) for k in ("cohort", "file", "sample_id", "epoch", "environment", "main_task", "side_task", "cell")},
                         "n_tokens": r["n_tokens"], "truncated": r["n_tokens"] > CUT, "forward_tokens": (max(p["token"] for p in rows) + 1) if rows else 0,
                         "n_rows": len(rows), "positions": positions_})
        if len(buf_meta) == cfg["chunk_size"]:
            flush()
            print(f"shard {shard}: {n}/{len(mine)}", flush=True)
    flush()
    return shard_stats({"render failures": failures, "load seconds": load_s, "seconds": round(time.time() - t_all), "forward [tokens, seconds]": timing})


@app.function(image=cpu_image, cpu=2, memory=8192, volumes={"/out": out_vol}, timeout=24 * 3600)
def orchestrate(cfg: dict, n: int, cohort: list[dict], fix_keys: list[str]) -> dict:
    """Run every shard from inside Modal (so the laptop client can go away) and write stats/summary.json."""
    started = time.time()
    calls = [extract_shard.spawn(cfg, i, cohort, fix_keys) for i in range(n)]
    stats = []
    for shard, call in enumerate(calls):
        for _ in range(120):
            try:
                stats.extend(call.get())
                break
            except (socket.gaierror, ConnectionError, TimeoutError, modal.exception.ConnectionError, modal.exception.ServiceError):
                time.sleep(30)
            except Exception as e:  # recorded by type only (a message may quote staged text)
                stats.append({"shard": shard, "crashed": type(e).__name__})
                break
        else:
            stats.append({"shard": shard, "unreachable": True})
    ok = [x for x in stats if "chunks" in x]
    summary = {"wall_seconds": round(time.time() - started), "shards": stats, "n_ok": len(ok), "n_crashed": len(stats) - len(ok),
               "transcripts": sum(c["transcripts"] for x in ok for c in x["chunks"]), "rows": sum(c["rows"] for x in ok for c in x["chunks"]),
               "nonfinite": sum(c["nonfinite"] for x in ok for c in x["chunks"])}
    out_vol.reload()
    (Path("/out") / cfg["run"] / "stats").mkdir(parents=True, exist_ok=True)
    (Path("/out") / cfg["run"] / "stats" / "summary.json").write_text(json.dumps(summary))
    out_vol.commit()
    return summary


@app.function(image=cpu_image, cpu=8, memory=32768, volumes={"/out": out_vol}, timeout=3600, retries=0)
def check(run: str, expected: dict) -> dict:
    """Counts only. As the Olmo check, but it opens only layers 0 and 79 per chunk (shape and non-finite recount) and
    checks the other layers' files exist by listing (opening every file was too slow on the Volume at full size)."""
    import numpy as np
    from collections import Counter
    out_vol.reload()
    d = Path("/out") / run
    metas = Counter(); kept = 0; problems = Counter(); rows_total = 0; nonfinite_index = 0; nonfinite_recount = 0
    index = {}
    listed = {l: {p.stem for p in (d / "by_layer" / f"L{l}").glob("*.npy")} for l in range(N_LAYERS)}
    for idx in sorted((d / "chunks").glob("shard*_chunk*.json")):
        c = json.loads(idx.read_text())
        problems["chunk without its done marker"] += not (d / "chunks" / f"{idx.stem}.done").exists()
        problems["chunk index rows != its transcripts' rows"] += c["rows"] != sum(t["n_rows"] for t in c["transcripts"])
        problems["chunk index offsets not cumulative from 0"] += [t["row_start"] for t in c["transcripts"]] != [sum(u["n_rows"] for u in c["transcripts"][:j]) for j in range(len(c["transcripts"]))]
        for t in c["transcripts"]:
            problems["duplicate index entry"] += (t["arm"], t["filename"], idx.stem) in index
            index[(t["arm"], t["filename"], idx.stem)] = (t["row_start"], t["n_rows"])
        nonfinite_index += c["nonfinite"]
        problems["chunk missing a layer file"] += sum(idx.stem not in listed[l] for l in range(N_LAYERS))
        for l in (0, N_LAYERS - 1):
            x = np.load(d / "by_layer" / f"L{l}" / f"{idx.stem}.npy")
            problems["layer file with the wrong shape (layers 0 and 79)"] += x.shape != (c["rows"], H)
            nonfinite_recount += int((~np.isfinite(x)).sum())
        rows_total += c["rows"]
    for arm_dir in sorted((d / "meta").iterdir()):
        for f in arm_dir.glob("*.json"):
            m = json.loads(f.read_text()); metas[arm_dir.name] += 1
            kr = [p["row"] for p in m["positions"] if p["kind"] == "output_end" and p["kept"]]; kept += len(kr)
            problems["meta rows != kept readouts"] += m["n_rows"] != len(kr)
            problems["kept rows not 0..n-1 in order"] += kr != list(range(len(kr)))
            problems["unkept position with a row"] += any(p.get("row") is not None for p in m["positions"] if not p["kept"])
            problems["meta not in its chunk's index, or row_start/n_rows differ"] += index.pop((arm_dir.name, m["filename"], m["chunk"]), None) != (m["row_start"], m["n_rows"])
    problems["index entries with no live meta"] += len(index)
    problems["kept readouts (meta) != rows (chunks)"] += kept != rows_total
    problems["transcripts by arm != expected"] += (sum(metas.values()) != expected["total"]) if "total" in expected else (dict(metas) != expected)
    st = [json.loads(f.read_text()) for f in sorted((d / "stats").glob("shard_*.json"))]
    timing = [t for s in st for t in s.get("forward [tokens, seconds]", [])]
    toks, secs = sum(t[0] for t in timing), sum(t[1] for t in timing)
    return {"run": run, "meta files by arm": dict(metas), "kept readouts (meta)": kept, "rows (chunks)": rows_total,
            "non-finite values: every layer (chunk indexes)": nonfinite_index, "non-finite values: layers 0 and 79 recounted": nonfinite_recount,
            "expected transcripts by arm": expected, "problems": {k: v for k, v in problems.items()},
            "forward passes": len(timing), "forward tokens": toks, "forward seconds": round(secs, 1),
            "seconds per 1,000 forward tokens": round(1000 * secs / toks, 4) if toks else None,
            "longest forward pass [tokens, seconds]": max(timing) if timing else None,
            "load seconds by shard": [s.get("load seconds") for s in st], "shard seconds after load": [s.get("seconds") for s in st]}


@app.local_entrypoint()
def main(mode: str, runs: str = "", fix: str = "", test: bool = False, scope: str = "all"):
    """--scope (extract and check): "part1" (#36's 3,807 runs; Aswin's 12:53 option B), "fix939" (the 823 re-rendered runs
    alone, under their own run name; only if part 1 replicates), or "all"."""
    if scope not in ("all", "part1", "fix939"):
        raise SystemExit(f"unknown scope {scope}")
    import subprocess
    for f in (Path(__file__).resolve(), HERE / "PLAN.md", SWAP / "rerender.py"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    for p_, role in ((Path(runs), "runs.json"), (Path(fix), "fix939_frozen_runs_POSTHOC_2026-10-11_020458.jsonl")):
        if hashlib.sha256(p_.read_bytes()).hexdigest() != PIN[role]:
            raise SystemExit(f"{p_.name} is not the pinned {role}")
    rs = json.loads(Path(runs).read_text())
    fix_keys = sorted({json.loads(l)["key"] for l in open(fix) if json.loads(l)["label"] == "FIX"})
    if len(rs) != 3807 or len(fix_keys) != 823 or not set(fix_keys) <= {r["key"] for r in rs if r["arm"] == "sim-attack"}:
        raise SystemExit("not #36's 3,807 runs and fix939's 823 re-rendered ones")
    cohort = [{"arm": r["arm"], "key": r["key"], "filename": r["filename"]} for r in rs]
    common = {"plan": "PLAN.md (this folder)", "git": head, "reader": {"model": MODEL, "revision": REVISION, "context": CTX, "date_string": DATE},
              "inputs sha256": {"#36 runs.json": PIN["runs.json"], "fix939 per-run file": PIN["fix939_frozen_runs_POSTHOC_2026-10-11_020458.jsonl"]}}

    def write(name, out):
        with open(dest := HERE / "results" / f"{name}_{stamp}.json", "x") as fh:
            fh.write(json.dumps(out, indent=1) + "\n")
        print(dest)
    if mode == "render":
        return write("llama_render_r2j", common | render.remote(cohort, fix_keys))
    if mode == "diff":
        return write("llama_token_diff", common | diff.remote(cohort, fix_keys))
    if mode == "check":
        from collections import Counter
        expected = ({"total": 4} if test else {FIX_LABEL: len(fix_keys)} if scope == "fix939" else dict(Counter(r["arm"] for r in cohort))
                    | ({FIX_LABEL: len(fix_keys)} if scope == "all" else {}))
        run = RUN + (f"_test_{hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:8]}" if test else "_fix939" if scope == "fix939" else "")  # a smoke run is named by its code
        return write(f"extract_check_{run}", {"git": head} | check.remote(run, expected))
    if mode != "extract":
        raise SystemExit(f"unknown mode {mode}")
    src = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if not test and scope == "part1":
        fix_keys = []  # part 1 only: no re-rendered run is extracted
    cfg = {"run": RUN + (f"_test_{src[:8]}" if test else "_fix939" if scope == "fix939" else ""), "scope": scope, "shards": 1 if test else 8 if scope == "fix939" else SHARDS, "chunk_size": 2 if test else CHUNK,
           "limit_items": 4 if test else None, "git": head, "source sha256": src,
           "modal": {"app": APP, "gpu": GPU, "pip": PIP_GPU}} | common
    try:
        stored = json.loads(b"".join(out_vol.read_file(f"{cfg['run']}/config.json")))
    except (FileNotFoundError, modal.exception.NotFoundError):
        stored = None
    if stored is not None and {k: v for k, v in stored.items() if k not in ("git", "plan")} != {k: v for k, v in cfg.items() if k not in ("git", "plan")}:
        raise SystemExit(f"{cfg['run']} exists with different settings or code (source sha256); use a new run name")
    import io
    with out_vol.batch_upload(force=True) as batch:
        batch.put_file(io.BytesIO(json.dumps(cfg, indent=1).encode()), f"{cfg['run']}/config.json")
    call = orchestrate.spawn(cfg, cfg["shards"], cohort, fix_keys)
    rec = {"launched": stamp, "function call": call.object_id} | cfg
    (HERE / "results" / f"extract_launch_{cfg['run']}_{stamp}.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(f"{cfg['run']}: orchestrating {cfg['shards']} shards (function call {call.object_id}); summary at {cfg['run']}/stats/summary.json")
