"""Planted-slip transfer, q2's pair P5 (RULE.md, Amendment 1): a copy of
../2026-10-12_q2_olmo_reader/run_extract_olmo.py (at c70b5993) that reads the runs staged by q2_reply_stage.py instead of
#36's cohort and fix939's runs. Only its app, its inputs (`work_items`) and its entry point changed. Each staged run's C2
keep flags come from its ORIGINAL run's Qwen3-32B meta (#36's dump, by step and action). Olmo, the template, `positions()`
(R2 at C2), the forward pass and the layout are unchanged; all 64 layers are kept, as there.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run --detach \\
      experiments/2026-10-12_plantslip_transfer/q2_reply_extract_olmo.py --which <controls|planted>
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
APP = "probe-shift_aswin-surya_sub-q2-plantq2_olmo_kill-2026-11-12"
ACTS = "probe-shift_aswin-surya_sub-la-activations"  # read-only: #36's dump (staged inputs, Qwen3-32B meta) and fix939's
OUT = "probe-shift_aswin-surya_sub-q2-olmo_acts_kill-2026-11-12"
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
DUMP = "2026-10-06_la_segment-end_qwen3-32b_v1"
FIX_DUMP, FIX_INPUTS, FIX_ARM, FIX_LABEL = ("2026-10-11_q3_fix939_qwen3-32b_l48-63_v1", "2026-10-11_q3_fix939_inputs_v1",
                                            "sim-attack-note-sr-both", "sim-attack-fix939")
RUN = "2026-10-12_q2_plantslip_reply_{which}_olmo31-32b_v1"
STAGED = "2026-10-12_q2_plantslip_reply_{which}_inputs_v1"  # on ACTS, staged by q2_reply_stage.py
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


def filename(key: str) -> str:  # the extractors' file stems
    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


def work_items(which):
    """(arm, filename, staged item, the ORIGINAL run's Qwen3-32B meta path) for one staged run (controls or planted)."""
    it = staged(STAGED.format(which=which), False)
    return [(arm, filename(k), x, Path("/acts") / DUMP / "acts" / x["orig_arm"] / f"{filename(k)}.json") for (arm, k), x in sorted(it.items())]


# ---------- extract (GPU) ----------
# No automatic retry (a retry once started a second container on a running shard: two writers, double cost). After a
# crash, relaunch: the resume skips every transcript already done.
@app.function(image=gpu_image, gpu=GPU, memory=65536, volumes={"/hf": hf_cache, "/out": out_vol, "/acts": acts}, timeout=TIMEOUT_S, retries=0)
def extract_shard(cfg: dict, shard: int) -> list[dict]:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    items = work_items(cfg["which"])
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
def orchestrate(cfg: dict, n: int) -> dict:
    """Run every shard from inside Modal (so the laptop client can go away) and write stats/summary.json."""
    started = time.time()
    calls = [extract_shard.spawn(cfg, i) for i in range(n)]
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


@app.local_entrypoint()
def main(which: str):
    import subprocess
    if which not in ("controls", "planted"):
        raise SystemExit("--which controls|planted")
    for f in (Path(__file__).resolve(), HERE / "RULE.md"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    cfg = {"run": RUN.format(which=which), "which": which, "staged": STAGED.format(which=which), "shards": 1 if which == "controls" else 12,
           "chunk_size": CHUNK, "limit_items": None, "git": head, "source sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "reader": {"model": MODEL, "revision": REVISION, "context": CTX}, "modal": {"app": APP, "gpu": GPU, "pip": PIP_GPU}}
    try:
        stored = json.loads(b"".join(out_vol.read_file(f"{cfg['run']}/config.json")))
    except (FileNotFoundError, modal.exception.NotFoundError):
        stored = None
    if stored is not None and {k: v for k, v in stored.items() if k != "git"} != {k: v for k, v in cfg.items() if k != "git"}:
        raise SystemExit(f"{cfg['run']} exists with different settings or code (source sha256); use a new run name")
    import io
    with out_vol.batch_upload(force=True) as batch:
        batch.put_file(io.BytesIO(json.dumps(cfg, indent=1).encode()), f"{cfg['run']}/config.json")
    call = orchestrate.spawn(cfg, cfg["shards"])
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"q2_reply_extract_launch_P5_{which}_{stamp}.json").write_text(json.dumps({"launched": stamp, "function call": call.object_id} | cfg, indent=1) + "\n")
    print(f"{cfg['run']}: orchestrating {cfg['shards']} shards (function call {call.object_id})")
