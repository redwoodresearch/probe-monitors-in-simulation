"""A third reader, step 2 (PLAN.md, step 2): Olmo-3.1-32B-Instruct on #36's staged transcripts and fix939's re-rendered
ones, at the R2 readout of every tool output that Qwen3-32B kept (C2). A copy of
../2026-10-06_all_layers_extraction/run_extract_all_layers.py with Olmo's chat template and the R2 rule in place of
probe_shift.reader; the shared code is untouched, and transformers 4.57.1 lives only in this job's images. One position
function, positions(), serves both modes, so the re-render counts exactly the readouts the GPU job reads.

  --mode render    (CPU, counts only; no token or output text leaves the container): per arm, newline membership of
                   the R0, R1 and R2 readouts at C2, R2's coverage of Qwen3-32B's kept outputs, and forward tokens.
  --mode download  (CPU): the pinned weights into the shared HF cache Volume, once.
  --mode check     (CPU, counts only): an extraction's files against its index and meta; forward timing.
  --mode extract   (one H100 per shard, through one remote orchestrator): the residual stream after every decoder
                   block, all 64 layers, float16, at the kept R2 readouts; #36's layout (by_layer/L<l>/<chunk>.npy,
                   meta/<arm>/<filename>.json, chunk markers, resume). --test: 4 transcripts on one shard (the longest
                   of shard 0) under a separate run name, the smoke test.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with pyyaml modal run [--detach] \\
      experiments/2026-10-12_q2_olmo_reader/run_extract_olmo.py --mode <render|download|extract> [--test] \\
      --runs <#36 runs.json> --fix <fix939 per-run jsonl>
"""
import hashlib
import json
import os
import socket
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-olmo_extract_kill-2026-11-12"
ACTS = "probe-shift_aswin-surya_sub-la-activations"  # read-only: #36's dump (staged inputs, Qwen3-32B meta) and fix939's
OUT = "probe-shift_aswin-surya_sub-q2-olmo_acts_kill-2026-11-12"
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
DUMP = "2026-10-06_la_segment-end_qwen3-32b_v1"
FIX_DUMP, FIX_INPUTS, FIX_ARM, FIX_LABEL = ("2026-10-11_q3_fix939_qwen3-32b_l48-63_v1", "2026-10-11_q3_fix939_inputs_v1",
                                            "sim-attack-note-sr-both", "sim-attack-fix939")
RUN = "2026-10-12_la_output-end-r2_olmo31-32b_alllayers_v1"
MODEL, REVISION, CTX, CUT, H, N_LAYERS = "allenai/Olmo-3.1-32B-Instruct", "ac0587e4a7744a551c059d8cd17ba220bc940dae", 65536, 32768, 5120, 64
ENV_OPEN, END = "<|im_start|>environment\n", "<|im_end|>"
GPU, SHARDS, CHUNK, TIMEOUT_S = "H100!", 48, 25, 3 * 3600
PIP_CPU = ["transformers==4.57.1", "jinja2==3.1.6", "huggingface_hub[hf_transfer]==0.35.3", "numpy"]
PIP_GPU = ["torch==2.8.0", "accelerate==1.10.1", *PIP_CPU]
PIN = {"runs.json": "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96",
       "fix939_frozen_runs_POSTHOC_2026-10-11_020458.jsonl": "5f6ecbc42b1c6bb9586cb58e474d74c5c1d914bedb95d80a5a07dd078b0e5ebd"}

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2).read_only()
hf_cache = modal.Volume.from_name(HF_CACHE)
out_vol = modal.Volume.from_name(OUT, version=2, create_if_missing=True)
cpu_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PIP_CPU).env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
gpu_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PIP_GPU).env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})


class Fail(Exception):
    def __init__(self, stage):
        super().__init__(stage); self.stage = stage


def prep(msgs):
    """Olmo's template takes string content and tool-call arguments as a dict; the staged messages may hold either form."""
    out = []
    for m in msgs:
        m = dict(m); c = m.get("content")
        m["content"] = "".join(p.get("text", "") for p in c if p.get("type") == "text") if isinstance(c, list) else (c or "")
        if m.get("tool_calls"):
            m["tool_calls"] = [{**tc, "function": {**(tc.get("function") or {}), "arguments": (lambda a: json.loads(a) if isinstance(a, str) else (a or {}))(
                (tc.get("function") or {}).get("arguments"))}} for tc in m["tool_calls"]]
        out.append(m)
    return out


def bad(t: str) -> bool:
    """R2 skips a token that is whitespace-only or has a newline anywhere in it."""
    return t.strip() == "" or "\n" in t


def positions(tok, msgs, q):
    """One staged transcript under Olmo's template. q: Qwen3-32B's output_end positions, {"step,action": [kept, tool]}.
    Returns ids, the meta positions (every call's call_end, not kept; every output's output_end with R2 as its token,
    kept under C2 and below CTX), and per output (qwen kept, R0, R1, R2 tokens as (index, text)) for the render.
    Raises Fail(stage) on a template mismatch."""
    import bisect
    msgs = prep(msgs)
    text = tok.apply_chat_template(msgs, tokenize=False)
    tools = [m for m in msgs if m["role"] == "tool"]
    if text.count(ENV_OPEN) != len(tools) or any(ENV_OPEN in m["content"] or END in m["content"] for m in tools):
        raise Fail("a delimiter quoted in content")
    spans, calls, pos, step, action = [], [], 0, -1, 0
    for m in msgs:
        if m["role"] == "assistant":
            step, action = step + 1, 0
            calls += [(step, i, (tc.get("function") or {}).get("name")) for i, tc in enumerate(m.get("tool_calls") or [])]
        elif m["role"] == "tool":
            a = text.index(ENV_OPEN, pos) + len(ENV_OPEN); b = a + len(m["content"])
            if text[a:b] != m["content"] or not text.startswith(END, b):
                raise Fail("a tool output not verbatim")
            spans.append((step, action, a, b)); action += 1; pos = b
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc["input_ids"]; starts, ends = [s for s, _ in enc["offset_mapping"]], [e for _, e in enc["offset_mapping"]]
    tt = lambda i: text[starts[i]:ends[i]]  # noqa: E731

    def inside(a, b):  # the tokens lying entirely inside [a, b)
        return bisect.bisect_left(starts, a), bisect.bisect_right(ends, b)
    out = [{"kind": "call_end", "step": s, "action": i, "tool": name, "token": None, "kept": False, "token_text": None} for s, i, name in calls]
    per = []
    for s, a_, a, b in spans:
        i0, i1 = inside(a, b)
        r0 = i1 - 1 if i1 > i0 else None
        j0, j1 = inside(a, a + len(text[a:b].rstrip()))
        r1 = j1 - 1 if j1 > j0 else None
        r2 = next((i for i in range(i1 - 1, i0 - 1, -1) if not bad(tt(i))), None)
        qk = q.get(f"{s},{a_}")
        kept = bool(qk and qk[0]) and r2 is not None and r2 < CTX
        out.append({"kind": "output_end", "step": s, "action": a_, "tool": qk[1] if qk else None, "token": r2, "kept": kept,
                    "token_text": None})
        per.append((qk, *[(r, None if r is None else tt(r)) for r in (r0, r1, r2)]))
    return {"ids": ids, "n_tokens": len(ids), "positions": out, "outputs": per}


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


# ---------- render (CPU, counts only) ----------
def _init():
    from transformers import AutoTokenizer
    global TOK
    TOK = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)


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
    c[(label, "Olmo tokens")] += r["n_tokens"]; c[(label, "forward tokens (to the last kept readout)")] += max(kept_idx) + 1 if kept_idx else 0
    c[(label, "runs with no kept readout")] += not kept_idx
    for qk, (r0, t0), (r1, t1), (r2, t2) in r["outputs"]:
        c[(label, "tool outputs")] += 1
        c[(label, "tool outputs with no Qwen3-32B position")] += qk is None
        if not (qk and qk[0]):
            continue
        c[(label, "outputs Qwen3-32B kept")] += 1
        c[(label, "of those, kept under C2 (R2 readout below the context length)")] += r2 is not None and r2 < CTX
        c[(label, "of those, no R2 readout")] += r2 is None
        c[(label, "of those, R2 at index 32,768 or beyond (kept if below 65,536)")] += r2 is not None and r2 >= CUT
        c[(label, "R2 differs from R1")] += r2 != r1; c[(label, "R2 differs from R0")] += r2 != r0
        for rule, t in (("R0", t0), ("R1", t1), ("R2", t2)):
            c[(label, f"{rule} readouts at C2")] += t is not None
            c[(label, f"{rule} readouts at C2: whitespace-only")] += t is not None and t.strip() == ""
            c[(label, f"{rule} readouts at C2: contains a newline")] += t is not None and "\n" in t
    return c


@app.function(image=cpu_image, volumes={"/acts": acts}, cpu=32, memory=65536, timeout=3600, retries=0)
def render(cohort: list[dict], fix_keys: list[str]) -> dict:
    try:
        return _render(cohort, fix_keys)
    except Exception as e:  # nothing from the exception (which may quote staged text) crosses the boundary
        raise RuntimeError(f"render job failed: {type(e).__name__}") from None


def _render(cohort, fix_keys):
    from collections import Counter
    from multiprocessing import Pool

    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(MODEL, revision=REVISION)  # download once; the pool's processes load from the cache
    items = work_items(cohort, fix_keys)
    total = Counter()
    with Pool(32, initializer=_init) as pool:
        for cnt in pool.imap(count_one, items, chunksize=4):
            total.update(cnt)
    return {"rows": [{"arm": a, "count": k, "n": int(v)} for (a, k), v in sorted(total.items())]}


# ---------- download (CPU) ----------
@app.function(image=cpu_image, volumes={"/hf": hf_cache}, cpu=8, memory=16384, timeout=3600, retries=0)
def download() -> dict:
    from huggingface_hub import snapshot_download
    path = snapshot_download(MODEL, revision=REVISION, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
    hf_cache.commit()
    return {"path": path, "bytes": sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())}


# ---------- extract (GPU) ----------
# No automatic retry (a retry once started a second container on a running shard: two writers, double cost). After a
# crash, relaunch: the resume skips every transcript already done.
@app.function(image=gpu_image, gpu=GPU, memory=65536, volumes={"/hf": hf_cache, "/out": out_vol, "/acts": acts}, timeout=TIMEOUT_S, retries=0)
def extract_shard(cfg: dict, shard: int, cohort: list[dict], fix_keys: list[str]) -> list[dict]:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    items = work_items(cohort, fix_keys)
    items.sort(key=lambda it: (-(it[2].get("tokens") or 0), it[0], it[2]["key"]))  # longest first (Qwen's count), dealt round-robin
    mine = items[shard::cfg["shards"]][: cfg.get("limit_items") or None]
    run_dir = Path("/out") / cfg["run"]
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    lm = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
    base = lm.model
    layers = list(range(lm.config.num_hidden_layers))
    if len(layers) != N_LAYERS or lm.config.hidden_size != H:
        raise ValueError(f"unexpected shape: {len(layers)} layers x {lm.config.hidden_size}")
    for l in layers:
        (run_dir / "by_layer" / f"L{l}").mkdir(parents=True, exist_ok=True)
    (run_dir / "chunks").mkdir(parents=True, exist_ok=True)
    (run_dir / "chunk_markers").touch()  # always a fresh, marker-mode run (no legacy chunks)
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
    buf, buf_meta, timing = {l: [] for l in layers}, [], []

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
    for n, (label, stem, it, qpath) in enumerate(mine, 1):
        if (run_dir / "meta" / label / f"{stem}.json").exists():
            continue
        r = positions(tok, it["messages"], qsum(json.loads(qpath.read_text())))
        rows = [p for p in r["positions"] if p["kept"]]
        t0 = time.time()
        if rows:
            last = max(p["token"] for p in rows)
            idx = torch.tensor([p["token"] for p in rows], device="cuda", dtype=torch.long)
            got = {}

            def hook_for(layer):
                def hook(module, args, output):
                    h = output[0] if isinstance(output, tuple) else output
                    got[layer] = h[0].index_select(0, idx).to(torch.float16).cpu().numpy()
                return hook
            handles = [base.layers[l].register_forward_hook(hook_for(l)) for l in layers]
            try:
                with torch.inference_mode():
                    base(input_ids=torch.tensor([r["ids"][: last + 1]], device="cuda"), use_cache=False)
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
    stats = []
    for p in sorted((run_dir / "chunks").glob(f"shard{shard:03d}_chunk*.json")):
        c = json.loads(p.read_text())
        live = [t for t in c["transcripts"] if (f := run_dir / "meta" / t["arm"] / f"{t['filename']}.json").exists()
                and json.loads(f.read_text())["chunk"] == p.stem]
        stats.append({"chunk": p.stem, "transcripts": len(live), "rows": sum(t["n_rows"] for t in live), "nonfinite": c["nonfinite"]})
    summary = {"shard": shard, "transcripts": len(mine), "chunks": stats, "seconds": round(time.time() - t_all), "forward [tokens, seconds]": timing}
    (run_dir / "stats").mkdir(exist_ok=True)
    (run_dir / "stats" / f"shard_{shard}.json").write_text(json.dumps(summary))
    out_vol.commit()
    return [summary]


@app.function(image=cpu_image, cpu=2, memory=8192, volumes={"/out": out_vol}, timeout=24 * 3600)
def orchestrate(cfg: dict, n: int, cohort: list[dict], fix_keys: list[str]) -> dict:
    """Run every shard from inside Modal (so the laptop client can go away) and write stats/summary.json."""
    started = time.time()
    calls = [extract_shard.spawn(cfg, i, cohort, fix_keys) for i in range(n)]
    stats = []
    for shard, call in enumerate(calls):
        for _ in range(120):  # transport errors: poll again (the shard may still be running or done)
            try:
                stats.extend(call.get())
                break
            except (socket.gaierror, ConnectionError, TimeoutError, modal.exception.ConnectionError, modal.exception.ServiceError):
                time.sleep(30)
            except Exception as e:  # the shard itself failed: recorded by type only (a message may quote staged text)
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
    """Counts only. Meta files by arm against `expected` (arm -> transcripts); each meta's kept readouts, rows 0..n-1 in
    order, and its chunk and row_start against that chunk's index; per chunk, its done marker, 64 layer files of shape
    (rows, 5120), and non-finite values (every layer, from the index written with the arrays; layers 0 and 63 recounted
    here); forward timing from the shard stats."""
    import numpy as np
    from collections import Counter
    out_vol.reload()
    d = Path("/out") / run
    metas = Counter(); kept = 0; problems = Counter(); rows_total = 0; nonfinite_index = 0; nonfinite_recount = 0
    index = {}
    for idx in sorted((d / "chunks").glob("shard*_chunk*.json")):
        c = json.loads(idx.read_text())
        problems["chunk without its done marker"] += not (d / "chunks" / f"{idx.stem}.done").exists()
        problems["chunk index rows != its transcripts' rows"] += c["rows"] != sum(t["n_rows"] for t in c["transcripts"])
        problems["chunk index offsets not cumulative from 0"] += [t["row_start"] for t in c["transcripts"]] != [sum(u["n_rows"] for u in c["transcripts"][:j]) for j in range(len(c["transcripts"]))]
        for t in c["transcripts"]:
            problems["duplicate index entry"] += (t["arm"], t["filename"], idx.stem) in index
            index[(t["arm"], t["filename"], idx.stem)] = (t["row_start"], t["n_rows"])
        nonfinite_index += c["nonfinite"]
        for l in range(N_LAYERS):
            x = np.load(d / "by_layer" / f"L{l}" / f"{idx.stem}.npy", mmap_mode="r")
            problems["layer file with the wrong shape"] += x.shape != (c["rows"], H)
            if l in (0, N_LAYERS - 1):
                nonfinite_recount += int((~np.isfinite(np.asarray(x))).sum())
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
    timing = [t for f in sorted((d / "stats").glob("shard_*.json")) for t in json.loads(f.read_text()).get("forward [tokens, seconds]", [])]
    toks, secs = sum(t[0] for t in timing), sum(t[1] for t in timing)
    return {"run": run, "meta files by arm": dict(metas), "kept readouts (meta)": kept, "rows (chunks)": rows_total,
            "non-finite values: every layer (chunk indexes)": nonfinite_index, "non-finite values: layers 0 and 63 recounted": nonfinite_recount,
            "expected transcripts by arm": expected, "problems": {k: v for k, v in problems.items()},
            "forward passes": len(timing), "forward tokens": toks, "forward seconds": round(secs, 1),
            "seconds per 1,000 forward tokens": round(1000 * secs / toks, 4) if toks else None,
            "longest forward pass [tokens, seconds]": max(timing) if timing else None}


@app.local_entrypoint()
def main(mode: str, runs: str = "", fix: str = "", test: bool = False):
    import subprocess
    for f in (Path(__file__).resolve(), HERE / "PLAN.md"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    if mode == "download":
        print(json.dumps(download.remote()))
        return
    for p_, role in ((Path(runs), "runs.json"), (Path(fix), "fix939_frozen_runs_POSTHOC_2026-10-11_020458.jsonl")):
        if hashlib.sha256(p_.read_bytes()).hexdigest() != PIN[role]:
            raise SystemExit(f"{p_.name} is not the pinned {role}")
    rs = json.loads(Path(runs).read_text())
    fix_keys = sorted({json.loads(l)["key"] for l in open(fix) if json.loads(l)["label"] == "FIX"})
    if len(rs) != 3807 or len(fix_keys) != 823 or not set(fix_keys) <= {r["key"] for r in rs if r["arm"] == "sim-attack"}:
        raise SystemExit("not #36's 3,807 runs and fix939's 823 re-rendered ones")
    cohort = [{"arm": r["arm"], "key": r["key"], "filename": r["filename"]} for r in rs]
    common = {"plan": "PLAN.md step 2 (this folder)", "git": head, "reader": {"model": MODEL, "revision": REVISION, "context": CTX},
              "inputs sha256": {"#36 runs.json": PIN["runs.json"], "fix939 per-run file": PIN["fix939_frozen_runs_POSTHOC_2026-10-11_020458.jsonl"]}}
    if mode == "check":  # --test checks the smoke run (4 transcripts, arms not known in advance)
        from collections import Counter
        expected = {"total": 4} if test else dict(Counter(r["arm"] for r in cohort)) | {FIX_LABEL: len(fix_keys)}
        out = {"git": head} | check.remote(RUN + ("_test" if test else ""), expected)
        with open(dest := HERE / "results" / f"extract_check_{RUN}{'_test' if test else ''}_{stamp}.json", "x") as fh:
            fh.write(json.dumps(out, indent=1) + "\n")
        print(json.dumps(out, indent=1)); print(dest)
        return
    if mode == "render":
        out = common | render.remote(cohort, fix_keys)
        with open(dest := HERE / "results" / f"olmo_render_r2_{stamp}.json", "x") as fh:
            fh.write(json.dumps(out, indent=1) + "\n")
        print(dest)
        return
    if mode != "extract":
        raise SystemExit(f"unknown mode {mode}")
    cfg = {"run": RUN + ("_test" if test else ""), "shards": 1 if test else SHARDS, "chunk_size": 2 if test else CHUNK,
           "limit_items": 4 if test else None, "git": head, "source sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
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
