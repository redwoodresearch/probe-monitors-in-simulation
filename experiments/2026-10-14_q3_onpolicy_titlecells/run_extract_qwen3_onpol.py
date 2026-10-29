"""On-policy title cells (RULE.md here), POST HOC: q2's run_extract_qwen3_fresh.py (aswin/q2-fresh-honest @ f55ccfea) with its own
app name only; config config/onpol_qwen3.yaml (all 64 layers, mean_final, the 32k cut, on run_onpol_stage.py's staged inputs).
Its docstring follows.

Fresh-task honest runs (PLAN.md there): run_extract_qwen3_fxavg.py (aswin/q2-f10-tabs @ 51846989) with its own app name only;
configs config/fresh_*_qwen3.yaml. Its docstring follows.

Fix-full averaged readout + F10 (PLAN_F10.md Amendment 8b, POST HOC; Aswin "yes go avg" and "only as its own arm", via 0d):
a copy of ../2026-10-06_all_layers_extraction/run_extract_all_layers.py at 82c87660 (src/probe_shift/reader.py identical at
82c87660 and this branch) with its own app name, the GPU list ["H100", "B200", "H200"] (0d's speed rules v2; the reproduction
gate decides which GPU types' rows are used), and each shard's GPU name in its stats. Everything else unchanged; its docstring
follows.

All-layers extraction: the segment-end positions at every one of Qwen3-32B's 64 layers, written per layer.

Same transcripts and text as an existing segment-end run (its staged inputs on the Volume are read directly, so
nothing is restaged), same per-action positions (`probe_shift.reader.segment_end_positions`: pre_call, call_end,
call_marker, output_end, output_marker), the residual stream after every decoder block, fp16; no log-probs. Optionally
(`ema_final_decays`, `ema_only`) the final value of a moving average over each call's and output's content tokens,
and (`mean_final`) the plain mean over the same tokens.

Layout, so an analysis can load one layer at a time from a few large files (GPU containers read many small files very
slowly): each shard buffers `chunk_size` transcripts, then writes `by_layer/L<l>/shard<s>_chunk<c>.npy` for every layer
and `chunks/shard<s>_chunk<c>.json` (each transcript's arm, key, filename and row range), and last each transcript's
`meta/<arm>/<filename>.json` (positions with rows relative to its block, plus `chunk` and `row_start`), which marks it
done: a retried or resumed shard skips finished transcripts and continues its chunk numbering.

  uv run modal run --detach experiments/2026-10-06_all_layers_extraction/run_extract_all_layers.py --config alllayers.yaml
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
APP = "probe-shift_aswin-surya_sub-q3onpol_qwen3_kill-2026-11-13"
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
OUTPUT = "probe-shift_aswin-surya_sub-la-activations"
GPU = ["H100", "B200", "H200"]
TIMEOUT_S = 3 * 3600
PIP = ["torch==2.8.0", "transformers==4.56.2", "accelerate==1.10.1", "numpy", "huggingface_hub[hf_transfer]"]

app = modal.App(APP)
hf_cache = modal.Volume.from_name(HF_CACHE)
out_vol = modal.Volume.from_name(OUTPUT, version=2)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PIP).env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})
         .add_local_python_source("probe_shift"))


def filename(key: str) -> str:
    """Same file stems as the segment-end extraction."""
    import hashlib

    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


# No automatic retry: on 2026-10-08 one started a second container on a shard whose first was still running (two
# writers, double cost). After a crash, relaunch: the resume skips every transcript already done.
@app.function(image=image, gpu=GPU, memory=65536, volumes={"/hf": hf_cache, "/out": out_vol}, timeout=TIMEOUT_S,
              retries=0)
def extract_shard(cfg: dict, shard: int) -> list[dict]:
    try:
        return _extract_shard(cfg, shard)
    except Exception as e:  # nothing from the exception (which may quote staged text) crosses the boundary (row 5, Codex)
        import traceback
        tb = traceback.extract_tb(e.__traceback__)[-1]
        raise RuntimeError(f"shard {shard} failed: {type(e).__name__} at {Path(tb.filename).name}:{tb.lineno}") from None


def _extract_shard(cfg: dict, shard: int) -> list[dict]:
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
    # Optional (config `layers`): save only these layers, and stop the forward pass after the last of them. The layers
    # kept are computed exactly as in a full pass; the later ones are skipped.
    layers = sorted(cfg.get("layers") or range(lm.config.num_hidden_layers))
    stop_after = layers[-1] if cfg.get("layers") else None

    class Stop(Exception):
        pass
    for l in layers:
        (run_dir / "by_layer" / f"L{l}").mkdir(parents=True, exist_ok=True)
    (run_dir / "chunks").mkdir(parents=True, exist_ok=True)
    # A chunk is complete once its done marker exists (written after its arrays, index and metadata). Runs started with
    # this marker scheme (run_dir/chunk_markers) drop every incomplete chunk of this shard before resuming, so a crash
    # mid-commit never leaves an array with rows no metadata points to; older runs keep the earlier behavior.
    first_layer = run_dir / "by_layer" / f"L{layers[0]}"
    # Marker mode is a property of the whole run, set only when the run directory holds no chunk from any shard (a
    # fresh run). A run that predates markers never gets them, so its legacy chunks are never dropped.
    fresh = not any(first_layer.glob("shard*_chunk*.npy")) and not any((run_dir / "chunks").glob("shard*_chunk*.json"))
    if fresh:
        (run_dir / "chunk_markers").touch()
    if (run_dir / "chunk_markers").exists():
        names = {p.stem for l in layers for p in (run_dir / "by_layer" / f"L{l}").glob(f"shard{shard:03d}_chunk*.npy")}
        names |= {p.stem for p in (run_dir / "chunks").glob(f"shard{shard:03d}_chunk*.json")}
        for name in sorted(n for n in names if not (run_dir / "chunks" / f"{n}.done").exists()):
            idx = run_dir / "chunks" / f"{name}.json"
            for t in (json.loads(idx.read_text())["transcripts"] if idx.exists() else []):
                meta = run_dir / "meta" / t["arm"] / f"{t['filename']}.json"
                if meta.exists() and json.loads(meta.read_text()).get("chunk") == name:
                    meta.unlink()  # the transcript is redone
            for l in layers:
                (run_dir / "by_layer" / f"L{l}" / f"{name}.npy").unlink(missing_ok=True)
            idx.unlink(missing_ok=True)
            print(f"shard {shard}: dropped incomplete chunk {name} before resuming", flush=True)
        out_vol.commit()
    # Continue numbering after a retry from the chunk files written first, so a retry never reuses a name holding data.
    chunk = len(list(first_layer.glob(f"shard{shard:03d}_chunk*.npy")))
    buf, buf_meta = {l: [] for l in layers}, []

    def flush():
        nonlocal chunk
        if not buf_meta:
            return
        name = f"shard{shard:03d}_chunk{chunk:03d}"
        nonfinite = 0
        for l in layers:
            x = np.concatenate(buf[l]) if buf[l] else np.zeros((0, 5120), np.float16)
            nonfinite += int((~np.isfinite(x)).sum())
            tmp = run_dir / "by_layer" / f"L{l}" / f".{name}.npy.tmp"  # atomic: a crash never leaves a partial chunk file
            with open(tmp, "wb") as fh:
                np.save(fh, x)
            os.replace(tmp, run_dir / "by_layer" / f"L{l}" / f"{name}.npy")
            buf[l].clear()
        index, start = [], 0
        for m in buf_meta:
            index.append({"arm": m["arm"], "key": m["key"], "filename": m["filename"], "row_start": start, "n_rows": m["n_rows"]})
            m |= {"chunk": name, "row_start": start}
            start += m["n_rows"]
        tmp = run_dir / "chunks" / f".{name}.json.tmp"  # the index, atomically, before any transcript is marked done
        tmp.write_text(json.dumps({"layers": layers, "rows": start, "nonfinite": nonfinite, "transcripts": index}))
        os.replace(tmp, run_dir / "chunks" / f"{name}.json")
        for m in buf_meta:  # last: each transcript's metadata marks it done, and is what analyses read
            d = run_dir / "meta" / m["arm"]
            d.mkdir(parents=True, exist_ok=True)
            tmp = d / f".{m['filename']}.json.tmp"
            tmp.write_text(json.dumps(m))
            os.replace(tmp, d / f"{m['filename']}.json")
        (run_dir / "chunks" / f"{name}.done").touch()  # last: the whole chunk is committed
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
        # Optional (config ema_final_decays): the final value of the running average over each call's and output's own
        # content tokens, ema_t = d*ema_{t-1} + (1-d)*x_t with ema_0 = x_0, read at its last content token, as kinds
        # "<kind>_ema<d>" (rows after the plain readouts; ema_only: only these). As in the segment-end extraction.
        # Optional (config mean_final): the plain mean over the same content tokens, as kinds "<kind>_mean" (rows after
        # the moving averages). Optional (config max_final): the element-wise max over the same content tokens, as kinds
        # "<kind>_max" (rows after the means). ema_only: only the pooled rows (moving averages, means and maxima).
        decays = cfg.get("ema_final_decays") or []
        pooled = [*decays, *(["mean"] if cfg.get("mean_final") else []), *(["max"] if cfg.get("max_final") else [])]
        plain = [] if cfg.get("ema_only") else kept
        ema_src = [p for p in kept if p["kind"] in ("call_end", "output_end") and p.get("span")]
        ema_w = {}
        for d in pooled:
            ws = []
            for p in ema_src:
                a, b = p["span"]
                if d == "max":
                    ws.append((a, b, None))  # not a weighted sum: the hook takes the maximum
                    continue
                if d == "mean":
                    w = torch.full((b - a,), 1.0 / (b - a), dtype=torch.float32)
                else:
                    w = (1 - d) * d ** torch.arange(b - a - 1, -1, -1, dtype=torch.float32)
                    w[0] = d ** (b - a - 1)  # ema_0 = x_0
                ws.append((a, b, w.cuda()))
            ema_w[d] = ws
        idx = torch.tensor([p["token"] for p in plain], device="cuda", dtype=torch.long)
        acts = {}

        def hook_for(layer):
            def hook(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                parts = [h[0].index_select(0, idx).float()]
                for d, ws in ema_w.items():
                    pool = (lambda a, b, w: h[0, a:b].float().amax(0)) if d == "max" else (lambda a, b, w: w @ h[0, a:b].float())
                    parts.append(torch.stack([pool(a, b, w) for a, b, w in ws]) if ws else h.new_zeros((0, h.shape[-1])).float())
                acts[layer] = torch.cat(parts).to(torch.float16).cpu().numpy()
                if layer == stop_after:  # the last saved layer: skip the rest of the forward pass
                    raise Stop
            return hook

        handles = [base.layers[l].register_forward_hook(hook_for(l)) for l in layers]
        try:
            with torch.inference_mode():
                base(input_ids=torch.tensor([r["ids"]], device="cuda"), use_cache=False)
        except Stop:  # every saved layer is in hand
            pass
        finally:
            for h in handles:
                h.remove()
        for l in layers:
            buf[l].append(acts[l])
        row = iter(range(len(plain) + len(pooled) * len(ema_src)))
        positions = [] if cfg.get("ema_only") else [p | {"row": next(row) if p["kept"] else None} for p in r["positions"]]
        positions += [p | ({"kind": f"{p['kind']}_{d}", "row": next(row)} if d in ("mean", "max") else
                           {"kind": f"{p['kind']}_ema{d}", "row": next(row), "ema_decay": d}) for d in pooled for p in ema_src]
        buf_meta.append({**{k: it[k] for k in ("key", "arm", "cohort", "file", "sample_id", "epoch", "environment", "main_task",
                                               "side_task", "cell", "n_submit_calls", "sim_reference")},
                         "filename": stem, "n_tokens": r["n_tokens"], "truncated": r["truncated"], "kept_tokens": len(r["ids"]),
                         "n_rows": len(plain) + len(pooled) * len(ema_src), "positions": positions})
        if len(buf_meta) == cfg["chunk_size"]:
            flush()
            print(f"shard {shard}: {n}/{len(mine)}", flush=True)
    flush()
    stats = []  # over every attempt's chunks, counting a transcript only in the chunk its metadata points to
    for p in sorted((run_dir / "chunks").glob(f"shard{shard:03d}_chunk*.json")):
        c = json.loads(p.read_text())
        live = [t for t in c["transcripts"] if (f := run_dir / "meta" / t["arm"] / f"{t['filename']}.json").exists()
                and json.loads(f.read_text())["chunk"] == p.stem]
        stats.append({"chunk": p.stem, "transcripts": len(live), "rows": sum(t["n_rows"] for t in live), "nonfinite": c["nonfinite"]})
    summary = {"shard": shard, "transcripts": len(mine), "chunks": stats, "seconds": round(time.time() - t_all), "gpu": torch.cuda.get_device_name(0)}
    (run_dir / "stats").mkdir(exist_ok=True)
    (run_dir / "stats" / f"shard_{shard}.json").write_text(json.dumps(summary))
    out_vol.commit()
    return [summary]


@app.function(image=image, cpu=2, memory=8192, volumes={"/out": out_vol}, timeout=24 * 3600)
def orchestrate(cfg: dict, n: int) -> dict:
    """Run every shard from inside Modal (so the laptop client can go away) and write stats/summary.json."""
    started = time.time()
    calls = [extract_shard.spawn(cfg, i) for i in range(n)]
    stats = []
    for shard, call in enumerate(calls):
        for attempt in range(120):  # transport errors: poll again (the shard may still be running or done)
            try:
                stats.extend(call.get())
                break
            except (socket.gaierror, ConnectionError, TimeoutError, modal.exception.ConnectionError, modal.exception.ServiceError):
                time.sleep(30)
            except Exception as e:  # the shard itself failed: recorded, so the summary says which (and the chain stops)
                stats.append({"shard": shard, "crashed": repr(e)[:500]})
                break
        else:
            stats.append({"shard": shard, "unreachable": True})
    ok = [x for x in stats if "chunks" in x]
    summary = {"provenance": cfg.get("provenance"), "wall_seconds": round(time.time() - started), "shards": stats, "n_ok": len(ok), "n_crashed": len(stats) - len(ok),
               "n_unreachable": sum("unreachable" in x for x in stats),
               "transcripts": sum(c["transcripts"] for x in ok for c in x["chunks"]), "rows": sum(c["rows"] for x in ok for c in x["chunks"]),
               "nonfinite": sum(c["nonfinite"] for x in ok for c in x["chunks"])}
    out_vol.reload()
    (Path("/out") / cfg["run"] / "stats").mkdir(parents=True, exist_ok=True)
    (Path("/out") / cfg["run"] / "stats" / "summary.json").write_text(json.dumps(summary))
    out_vol.commit()
    return summary


@app.local_entrypoint()
def main(config: str = "alllayers.yaml", test: bool = False, remote: bool = False, commit: str = ""):
    """`--test`: one shard, 4 transcripts, chunks of 2, under a separate run name. `--remote` (with --detach): the shards
    run from one Modal function, which writes <run>/stats/summary.json; the laptop client exits after launching.
    `--commit`: the commit holding this code (checked byte for byte), recorded in the run's config.json."""
    import yaml

    from probe_shift.models import QWEN3_32B
    from probe_shift.provenance import capture

    cfg = yaml.safe_load((HERE / "config" / config).read_text())
    unknown = set(cfg) - {"run", "inputs_from", "model", "dtype", "attn_implementation", "max_tokens", "shards", "chunk_size",
                          "ema_final_decays", "ema_only", "mean_final", "max_final", "layers"}
    if unknown:  # a setting the code does not read must not pass silently
        raise ValueError(f"config/{config}: settings this extraction does not implement: {sorted(unknown)}")
    if cfg["model"] != QWEN3_32B["name"]:
        raise ValueError(f"{cfg['model']} has no pinned revision in probe_shift.models")
    cfg["revision"] = QWEN3_32B["revision"]
    root = HERE.parent.parent
    cfg["provenance"] = capture(root, [Path(__file__), HERE / "config" / config, *sorted((root / "src" / "probe_shift").glob("*.py"))],
                                commit)
    if test:
        cfg |= {"run": cfg["run"] + "_test", "limit_items": 4, "chunk_size": 2}
    try:
        stored = json.loads(b"".join(out_vol.read_file(f"{cfg['run']}/config.json")))
    except (FileNotFoundError, modal.exception.NotFoundError):
        stored = None
    # A resume needs the same settings and code; the commit and dirty flag may differ (an unrelated commit in between).
    same = lambda c: {k: v for k, v in c.items() if k != "provenance"} | {"source_sha": (c.get("provenance") or {}).get("source_sha")}
    if stored is not None and same(stored) != same(cfg):
        raise ValueError(f"{cfg['run']} exists with different settings or code; use a new run name")
    with out_vol.batch_upload(force=True) as batch:
        import io

        batch.put_file(io.BytesIO(json.dumps(cfg, indent=1).encode()), f"{cfg['run']}/config.json")
    staged = [e.path for e in out_vol.listdir(f"{cfg['inputs_from']}/inputs") if e.path.endswith(".jsonl")]
    if not staged:
        raise FileNotFoundError(f"no staged inputs under {cfg['inputs_from']}/inputs")
    n = 1 if test else cfg["shards"]
    if remote:
        call = orchestrate.spawn(cfg, n)
        print(f"{cfg['run']}: orchestrating {n} shards on Modal (function call {call.object_id}); summary at "
              f"{cfg['run']}/stats/summary.json", flush=True)
        return
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
