"""All-layers extraction with Qwen3.5-27B as the reader: the segment-end positions at every one of its 64 layers.

Prepared for the Qwen3.5 redo of Phase 1 (2026-10-06); launch only after Aswin confirms whole transcripts. Differences
from the Qwen3-32B all-layers extraction (experiments/2026-10-06_all_layers_extraction): the model (Qwen3.5-27B loaded
text-only as Qwen3_5ForCausalLM, transformers 5.19, flash-linear-attention kernels), whole transcripts (no 32k cut),
H200s (a 153k-token transcript projects to about 76 GB), and each sequence right-padded to a multiple of `pad_to`
tokens so the linear-attention kernels autotune once per length bucket instead of once per transcript; padding sits
after every readout position, so in this causal model it changes none of them.

Same transcripts and text as an existing segment-end run (its staged inputs on the Volume are read directly, so
nothing is restaged), same per-action positions (`probe_shift.reader.segment_end_positions`: pre_call, call_end,
call_marker, output_end, output_marker), the residual stream after every decoder block, fp16; no log-probs or EMA.

Layout, so an analysis can load one layer at a time from a few large files (GPU containers read many small files very
slowly): each shard buffers `chunk_size` transcripts, then writes `by_layer/L<l>/shard<s>_chunk<c>.npy` for every layer
and `chunks/shard<s>_chunk<c>.json` (each transcript's arm, key, filename and row range), and last each transcript's
`meta/<arm>/<filename>.json` (positions with rows relative to its block, plus `chunk` and `row_start`), which marks it
done: a retried or resumed shard skips finished transcripts and continues its chunk numbering.

  uv run modal run --detach experiments/2026-10-06_qwen35_reader/run_extract_all_layers_qwen35.py --config qwen35_alllayers.yaml
"""

import json
import os
import socket
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-qwen35_all-layers_kill-2026-11-06"
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
OUTPUT = "probe-shift_aswin-surya_sub-la-activations"
GPU = "H200"
TIMEOUT_S = 3 * 3600
PIP = ["torch==2.8.0", "transformers==5.19.0", "accelerate>=1.1.0", "numpy", "huggingface_hub", "flash-linear-attention==0.5.2"]

app = modal.App(APP)
hf_cache = modal.Volume.from_name(HF_CACHE)
out_vol = modal.Volume.from_name(OUTPUT, version=2)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PIP).env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})
         .add_local_python_source("probe_shift"))


def filename(key: str) -> str:
    """Same file stems as the segment-end extraction."""
    import hashlib

    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


@app.function(image=image, gpu=GPU, memory=65536, volumes={"/hf": hf_cache, "/out": out_vol}, timeout=TIMEOUT_S,
              retries=modal.Retries(max_retries=2, initial_delay=10.0))
def extract_shard(cfg: dict, shard: int) -> list[dict]:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from probe_shift.reader import segment_end_positions

    src = Path("/out") / cfg["inputs_from"] / "inputs"
    items = [json.loads(l) for p in sorted(src.glob("shard_*.jsonl")) for l in p.read_text().splitlines()]
    ids = [(it["arm"], it["key"]) for it in items]
    if len(set(ids)) != len(ids):  # e.g. stale shard files left by a resume with fewer shards
        raise ValueError(f"{len(ids) - len(set(ids))} duplicate transcripts in {src}")
    items.sort(key=lambda it: (-it["tokens"], it["arm"], it["key"]))  # longest first, dealt round-robin: balanced
    mine = items[shard::cfg["shards"]][: cfg.get("limit_items") or None]
    run_dir = Path("/out") / cfg["run"]
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    lm = AutoModelForCausalLM.from_pretrained(cfg["model"], revision=cfg["revision"], dtype=getattr(torch, cfg["dtype"]),
                                              device_map="cuda", attn_implementation=cfg["attn_implementation"]).eval()
    base = lm.model
    layers = list(range(lm.config.num_hidden_layers))
    for l in layers:
        (run_dir / "by_layer" / f"L{l}").mkdir(parents=True, exist_ok=True)
    (run_dir / "chunks").mkdir(parents=True, exist_ok=True)
    chunk = len(list((run_dir / "chunks").glob(f"shard{shard:03d}_chunk*.json")))  # continue numbering after a retry
    buf, buf_meta, stats = {l: [] for l in layers}, [], []

    def flush():
        nonlocal chunk
        if not buf_meta:
            return
        name = f"shard{shard:03d}_chunk{chunk:03d}"
        nonfinite = 0
        for l in layers:
            x = np.concatenate(buf[l]) if buf[l] else np.zeros((0, 5120), np.float16)
            nonfinite += int((~np.isfinite(x)).sum())
            np.save(run_dir / "by_layer" / f"L{l}" / f"{name}.npy", x)
            buf[l].clear()
        index, start = [], 0
        for m in buf_meta:
            index.append({"arm": m["arm"], "key": m["key"], "filename": m["filename"], "row_start": start, "n_rows": m["n_rows"]})
            m |= {"chunk": name, "row_start": start}
            start += m["n_rows"]
        (run_dir / "chunks" / f"{name}.json").write_text(json.dumps({"layers": layers, "rows": start, "transcripts": index}))
        for m in buf_meta:  # written last: each transcript's metadata marks it done
            d = run_dir / "meta" / m["arm"]
            d.mkdir(parents=True, exist_ok=True)
            tmp = d / f".{m['filename']}.json.tmp"
            tmp.write_text(json.dumps(m))
            os.replace(tmp, d / f"{m['filename']}.json")
        stats.append({"chunk": name, "transcripts": len(buf_meta), "rows": start, "nonfinite": nonfinite})
        buf_meta.clear()
        chunk += 1
        out_vol.commit()

    t_all = time.time()
    for n, it in enumerate(mine, 1):
        stem = filename(it["key"])
        if (run_dir / "meta" / it["arm"] / f"{stem}.json").exists():
            continue
        r = segment_end_positions(tok, it["messages"], cfg["max_tokens"])
        kept = [p for p in r["positions"] if p["kept"]]
        idx = torch.tensor([p["token"] for p in kept], device="cuda", dtype=torch.long)
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        ids = r["ids"] + [pad] * (-len(r["ids"]) % cfg["pad_to"])  # after every position read: no effect on them
        acts = {}

        def hook_for(layer):
            def hook(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                acts[layer] = h[0].index_select(0, idx).to(torch.float16).cpu().numpy()
            return hook

        handles = [base.layers[l].register_forward_hook(hook_for(l)) for l in layers]
        try:
            with torch.inference_mode():
                base(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)
        finally:
            for h in handles:
                h.remove()
        for l in layers:
            buf[l].append(acts[l])
        row = iter(range(len(kept)))
        buf_meta.append({**{k: it[k] for k in ("key", "arm", "cohort", "file", "sample_id", "epoch", "environment", "main_task",
                                               "side_task", "cell", "n_submit_calls", "sim_reference")},
                         "filename": stem, "n_tokens": r["n_tokens"], "truncated": r["truncated"], "kept_tokens": len(r["ids"]),
                         "n_rows": len(kept), "positions": [p | {"row": next(row) if p["kept"] else None} for p in r["positions"]]})
        if len(buf_meta) == cfg["chunk_size"]:
            flush()
            print(f"shard {shard}: {n}/{len(mine)}", flush=True)
    flush()
    summary = {"shard": shard, "transcripts": len(mine), "chunks": stats, "seconds": round(time.time() - t_all)}
    (run_dir / "stats").mkdir(exist_ok=True)
    (run_dir / "stats" / f"shard_{shard}.json").write_text(json.dumps(summary))
    out_vol.commit()
    return [summary]


@app.local_entrypoint()
def main(config: str = "qwen35_alllayers.yaml", test: bool = False):
    """`--test`: one shard, 4 transcripts, chunks of 2, under a separate run name."""
    import yaml

    from probe_shift.models import QWEN35_27B

    cfg = yaml.safe_load((HERE / "config" / config).read_text())
    if cfg["model"] != QWEN35_27B["name"]:
        raise ValueError(f"{cfg['model']} has no pinned revision in probe_shift.models")
    cfg["revision"] = QWEN35_27B["revision"]
    if test:
        cfg |= {"run": cfg["run"] + "_test", "limit_items": 4, "chunk_size": 2}
    try:
        stored = json.loads(b"".join(out_vol.read_file(f"{cfg['run']}/config.json")))
    except (FileNotFoundError, modal.exception.NotFoundError):
        stored = None
    if stored is not None and stored != cfg:
        raise ValueError(f"{cfg['run']} exists with different settings; use a new run name")
    with out_vol.batch_upload(force=True) as batch:
        import io

        batch.put_file(io.BytesIO(json.dumps(cfg, indent=1).encode()), f"{cfg['run']}/config.json")
    n = 1 if test else cfg["shards"]
    started = time.time()
    calls = [extract_shard.spawn(cfg, i) for i in range(n)]
    print(f"{cfg['run']}: spawned {n} shards reading inputs of {cfg['inputs_from']}", flush=True)
    stats = []
    for shard, call in enumerate(calls):
        for attempt in range(120):
            try:
                stats.extend(call.get())
                break
            except (socket.gaierror, ConnectionError, TimeoutError, modal.exception.ConnectionError, modal.exception.ServiceError) as e:
                print(f"lost the connection while waiting on shard {shard} ({e!r}); retry {attempt + 1} in 30 s", flush=True)
                time.sleep(30)
            except Exception as e:
                print(f"shard {shard} crashed: {e!r}")
                stats.append({"shard": shard, "crashed": repr(e)[:500]})
                break
        else:  # an hour of network errors: unknown outcome; its stats are on the Volume if it finished
            stats.append({"shard": shard, "unreachable": True})
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    out = {"wall_seconds": round(time.time() - started), "config": cfg | {"modal": {"app": APP, "gpu": GPU, "pip": PIP}}, "shards": stats}
    (HERE / "results" / f"extract_{cfg['run']}_{stamp}.json").write_text(json.dumps(out, indent=1))
    ok = [s for s in stats if "chunks" in s]
    print(f"done in {out['wall_seconds'] / 3600:.2f} h: {len(ok)} shards ok, {len(stats) - len(ok)} crashed or unreachable; "
          f"transcripts {sum(c['transcripts'] for s in ok for c in s['chunks'])}, rows {sum(c['rows'] for s in ok for c in s['chunks'])}, "
          f"non-finite {sum(c['nonfinite'] for s in ok for c in s['chunks'])}")
