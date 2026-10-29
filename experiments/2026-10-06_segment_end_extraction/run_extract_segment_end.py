"""Segment-end extraction: Qwen3-32B reads every valid LinuxArena transcript (cut to its first 32k tokens) and saves
the residual stream at the end of each tool call and tool output, plus tool-output token log-probs.

Not the end-of-sequence readout (AGENTS.md): every position sits at one segment's end, mid-transcript. Per action,
positions from `probe_shift.reader.segment_end_positions`: pre_call (`<tool_call>`), call_end and output_end (the
last content tokens), call_marker and output_marker (the closing template tokens; sensitivity check only).

The local entrypoint streams the samples from S3, keeps the valid ones per splits/exclusions_v3.json, normalizes
them, packs them into shards balanced by projected run time, and uploads the shards to the output Volume
(containers have no AWS credentials). Each container skips transcripts already written, so a retried shard resumes.
Shards are spawned, the wait retries through network errors, and each shard writes its stats to the Volume: in the
2026-10-06 pooled run a laptop DNS error made the entrypoint raise, and Modal cancelled every shard. Rerunning resumes.

  uv run modal run --detach experiments/2026-10-06_segment_end_extraction/run_extract_segment_end.py [--limit 2]
"""

import io
import json
import socket
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-la-activations_segment-end"
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
OUTPUT = "probe-shift_aswin-surya_sub-la-activations"
GPU = "H100!"
TIMEOUT_S = 3 * 3600
PIP = ["torch==2.8.0", "transformers==4.56.2", "accelerate==1.10.1", "numpy", "huggingface_hub[hf_transfer]"]
TIME_FIT = (1.51e-4, 1.55e-8)  # seconds ≈ a·T + b·T², from the 2026-10-05 smoke test (with pooling; an upper bound)
LOGPROB_CHUNK = 2048

app = modal.App(APP)
hf_cache = modal.Volume.from_name(HF_CACHE, create_if_missing=True)
out_vol = modal.Volume.from_name(OUTPUT, create_if_missing=True, version=2)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(*PIP)
    .env({"HF_HOME": "/hf"})
    .add_local_python_source("probe_shift")
)


@app.function(image=image, gpu=GPU, volumes={"/hf": hf_cache, "/out": out_vol}, timeout=TIMEOUT_S,
              retries=modal.Retries(max_retries=2, initial_delay=10.0))
def extract_shard(cfg: dict, shard: int) -> list[dict]:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from probe_shift.reader import segment_end_positions

    run_dir = Path("/out") / cfg["run"]
    items = [json.loads(l) for l in (run_dir / "inputs" / f"shard_{shard}.jsonl").read_text().splitlines()]
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    lm = AutoModelForCausalLM.from_pretrained(cfg["model"], revision=cfg["revision"], dtype=getattr(torch, cfg["dtype"]),
                                              device_map="cuda", attn_implementation=cfg["attn_implementation"]).eval()
    base, head, layers = lm.model, lm.lm_head, cfg["layers"]

    import os

    def read_one(it, stem):
        r = segment_end_positions(tok, it["messages"], cfg["max_tokens"])
        kept = [p for p in r["positions"] if p["kept"]]
        idx = torch.tensor([p["token"] for p in kept], device="cuda", dtype=torch.long)
        # Optional (config ema_final_decays): the final value of the running average over each content segment's own
        # tokens, ema_t = d*ema_{t-1} + (1-d)*x_t with ema_0 = x_0, read at the segment's last content token. The probe is
        # linear, so this equals the running average of per-token probe scores, final value (James, 2026-10-06).
        decays = cfg.get("ema_final_decays") or []
        ema_rows = [j for j, p in enumerate(kept) if p["kind"] in ("call_end", "output_end") and p.get("span")]
        ema_w = {}
        for d in decays:
            ws = []
            for j in ema_rows:
                a, b = kept[j]["span"]
                n = b - a
                w = (1 - d) * d ** torch.arange(n - 1, -1, -1, dtype=torch.float32)
                w[0] = d ** (n - 1)  # ema_0 = x_0
                ws.append((a, b, w.cuda()))
            ema_w[d] = ws
        ids = torch.tensor([r["ids"]], device="cuda")
        acts = {}

        def hook_for(layer):
            def hook(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                acts[layer] = h[0].index_select(0, idx).to(torch.float16).cpu().numpy()
                for d, ws in ema_w.items():
                    rows = [w @ h[0, a:b].float() for a, b, w in ws]
                    acts[f"{layer}_ema{d}"] = (torch.stack(rows) if rows else h.new_zeros((0, h.shape[-1]))).to(torch.float16).cpu().numpy()
            return hook

        handles = [base.layers[l].register_forward_hook(hook_for(l)) for l in layers]
        torch.cuda.synchronize()
        t = time.time()
        try:
            with torch.inference_mode():
                hidden = base(input_ids=ids, use_cache=False).last_hidden_state[0]  # after the final norm
                # log p(token t | tokens < t) for every tool-output content token, from the logits at t - 1.
                targets = torch.tensor([t for o in r["outputs"] for t in range(o["start"], o["end"]) if t >= 1],
                                       device="cuda", dtype=torch.long)
                lps = []
                for c in targets.split(LOGPROB_CHUNK):
                    logits = head(hidden.index_select(0, c - 1)).float()
                    lps.append(logits.log_softmax(-1).gather(1, ids[0, c][:, None])[:, 0])
                logprobs = (torch.cat(lps) if lps else torch.zeros(0, device="cuda")).cpu().numpy()
        finally:
            for h in handles:
                h.remove()
        torch.cuda.synchronize()
        seconds = time.time() - t
        arrays = {f"L{l}": acts[l] for l in layers} | {"logprobs": logprobs.astype(np.float32)}
        arrays |= {f"L{l}_ema{d}": acts[f"{l}_ema{d}"] for l in layers for d in decays}
        if decays:
            arrays["ema_rows"] = np.array(ema_rows, np.int64)  # which rows of L<layer> the ema arrays describe
        np.savez(stem.parent / f"{stem.name}.npz", **arrays)
        row = iter(range(len(kept)))
        positions = [p | {"row": next(row) if p["kept"] else None} for p in r["positions"]]
        lp_at = 0
        outputs = []
        for o in r["outputs"]:
            n = sum(1 for t in range(o["start"], o["end"]) if t >= 1)
            outputs.append(o | {"lp_start": lp_at, "lp_end": lp_at + n})
            lp_at += n
        meta = {**{k: it[k] for k in ("key", "arm", "cohort", "file", "sample_id", "epoch", "environment", "main_task",
                                      "side_task", "cell", "n_submit_calls", "sim_reference")},
                "filename": stem.name, "n_tokens": r["n_tokens"], "truncated": r["truncated"], "kept_tokens": len(r["ids"]),
                "positions": positions, "outputs": outputs}
        tmp = stem.parent / f".{stem.name}.json.tmp"  # written last, then renamed: the JSON marks the transcript done
        tmp.write_text(json.dumps(meta))
        os.replace(tmp, stem.parent / f"{stem.name}.json")
        nonfinite = int(sum((~np.isfinite(v)).sum() for v in arrays.values()))
        return {"key": it["key"], "arm": it["arm"], "status": "ok", "n_tokens": r["n_tokens"], "truncated": r["truncated"],
                "seconds": round(seconds, 2), "positions_kept": len(kept), "positions_dropped": len(positions) - len(kept),
                "nonfinite": nonfinite}

    stats = []
    for n, it in enumerate(items, 1):
        stem = run_dir / "acts" / it["arm"] / filename(it["key"])
        if (stem.parent / f"{stem.name}.json").exists():  # done earlier: rebuild its stats from what was saved
            meta, saved = json.loads((stem.parent / f"{stem.name}.json").read_text()), np.load(stem.parent / f"{stem.name}.npz")
            n_kept = sum(p["kept"] for p in meta["positions"])
            stats.append({"key": it["key"], "arm": it["arm"], "status": "already_done", "n_tokens": meta["n_tokens"],
                          "truncated": meta["truncated"], "positions_kept": n_kept,
                          "positions_dropped": len(meta["positions"]) - n_kept,  # no timing: it is not saved
                          "nonfinite": int(sum((~np.isfinite(saved[k])).sum() for k in saved.files))})
            continue
        stem.parent.mkdir(parents=True, exist_ok=True)
        try:
            stats.append(read_one(it, stem))
        except Exception as e:  # one bad transcript must not take down the shard, unless the GPU itself is broken
            if "CUDA error" in str(e) or isinstance(e, torch.OutOfMemoryError):
                raise
            stats.append({"key": it["key"], "arm": it["arm"], "status": "failed", "error": f"{type(e).__name__}: {str(e)[:300]}"})
            print(f"shard {shard}: FAILED {it['key'][:80]}: {type(e).__name__}", flush=True)
        if n % 20 == 0:
            out_vol.commit()
            print(f"shard {shard}: {n}/{len(items)}", flush=True)
    (run_dir / "stats").mkdir(exist_ok=True)
    (run_dir / "stats" / f"shard_{shard}.json").write_text(json.dumps(stats))  # the summary survives a lost client
    out_vol.commit()
    return stats


def filename(key: str) -> str:
    """Output file stem: the key itself, or a prefix plus a hash when the name would pass the 255-byte limit."""
    import hashlib

    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


@app.local_entrypoint()
def main(limit: int = 0, config: str = "segment_end.yaml", configs: str = ""):
    """`--limit N` runs a test on N transcripts per cohort (one of them truncated) in one container, under a
    separate run name. `--configs a.yaml,b.yaml,...`: stage several text levels from one download (stage_only configs
    that differ only in run name and scrub level); each run gets exactly what staging it alone would write."""
    import yaml

    from probe_shift.models import QWEN3_32B
    from probe_shift.transcripts import eval_files, file_samples, main_task_id, reader_messages, submit_calls, system_prompt

    level_cfgs = [yaml.safe_load((HERE / "config" / c).read_text()) for c in configs.split(",")] if configs else []
    if level_cfgs:
        base = {k: v for k, v in level_cfgs[0].items() if k not in ("run", "scrub_level")}
        bad = [c for c, x in zip(configs.split(","), level_cfgs)
               if {k: v for k, v in x.items() if k not in ("run", "scrub_level")} != base or not x.get("stage_only")]
        if bad or limit:
            raise ValueError(f"--configs must be stage_only and differ only in run and scrub_level (and no --limit): {bad}")
        for x in level_cfgs:
            x["revision"] = QWEN3_32B["revision"]
    cfg = level_cfgs[0] if level_cfgs else yaml.safe_load((HERE / "config" / config).read_text())  # segment_end_ema.yaml adds running-average finals
    if cfg["model"] != QWEN3_32B["name"]:
        raise ValueError(f"{cfg['model']} has no pinned revision in probe_shift.models")
    cfg["revision"] = QWEN3_32B["revision"]  # passed to both loaders; part of the settings a resume must match
    from probe_shift.scrub import SCRUB_VERSION

    for x in level_cfgs or [cfg]:  # from level 2 on, the scrub's version is too (and is saved with the run's results)
        if x.get("scrub_level", 0) >= 2:
            x["scrub_version"] = SCRUB_VERSION
    n_shards = 1 if limit else cfg["shards"]
    if limit:
        cfg["run"] += f"_test{limit}"
    shared = json.loads((HERE.parent.parent / "splits" / cfg["exclusions_file"]).read_text())
    table = {(r["arm"], r["key"]): r for r in shared["samples"]}
    per_arm = Counter()

    def stream(arm):
        alias = cfg["cohorts"][arm]
        with ThreadPoolExecutor(32) as pool:
            for samples in pool.map(lambda f: [(f, s) for s in file_samples(alias, f)], eval_files(alias)):
                yield from samples

    items, seen = [], Counter()

    def scrubbed(messages, level=None):
        """Optional (config scrub_level): the text baselines' harness-normalizing scrub (probe_shift.scrub), in both
        arms, on every tool output (exactly TF-IDF's text at that level), and on the agent's tool calls: every string
        argument is scrubbed and every numeric argument becomes "<n>", so numbers and IDs the simulator invented and the
        agent echoed back (kill <pid>, cd <dir>) are masked there too. TF-IDF scrubs a call's rendered JSON instead,
        so call text can differ slightly (escaped newlines, number quoting); outputs are identical."""
        level = cfg.get("scrub_level", 0) if level is None else level
        if not level:
            return messages
        from probe_shift.scrub import scrub

        def args(v):
            if isinstance(v, str):
                return scrub(v, level)
            if isinstance(v, bool) or v is None:
                return v
            if isinstance(v, (int, float)):  # numbers are masked from level 2 on, as in the scrub itself
                return "<n>" if level >= 2 else v
            if isinstance(v, list):
                return [args(x) for x in v]
            if isinstance(v, dict):
                return {k: args(x) for k, x in v.items()}
            return v

        out = []
        for m in messages:
            if m["role"] == "tool":
                m = m | {"content": scrub(m["content"], level)}
            elif m["role"] == "assistant":
                m = m | {"content": scrub(m["content"], level) if m["content"] else m["content"],
                         "tool_calls": [c | {"function": c["function"] | {"arguments": args(c["function"]["arguments"])}}
                                        for c in m.get("tool_calls") or []]}
            out.append(m)
        return out

    def take(arm, f, s, prompt):
        key = f"{f.removesuffix('.eval')}__{s['id'].replace('/', '_')}__{s['epoch']}"
        row = table.get((arm, key))
        if row is None:
            raise ValueError(f"{arm} {key} is not in {cfg['exclusions_file']}")
        seen[arm] += 1
        if row["reason"]:
            return
        if limit:  # test: per cohort, one truncated transcript and limit - 1 short ones
            want_long = not any(it["arm"] == arm and it["truncated"] for it in items)
            if per_arm[arm] >= limit or (row["truncated"] and not want_long) or (not row["truncated"] and per_arm[arm] >= limit - want_long):
                return
        items.append({k: row[k] for k in ("key", "arm", "cohort", "file", "sample_id", "epoch", "environment", "main_task",
                                          "side_task", "cell", "sim_reference", "tokens", "truncated")}
                     | {"n_submit_calls": submit_calls(s),
                        "messages": reader_messages(s, prompt) if level_cfgs else scrubbed(reader_messages(s, prompt))})
        per_arm[arm] += 1

    # Honest runs first: they supply the system prompt for every attack run of the same main task.
    honest = {}
    for f, s in stream("real-honest"):
        honest.setdefault(main_task_id(s), set()).add(system_prompt(s))
        take("real-honest", f, s, system_prompt(s))
    if any(len(p) != 1 for p in honest.values()):
        raise ValueError("a main task has more than one honest system prompt")
    honest = {mt: p.pop() for mt, p in honest.items()}
    for arm in [a for a in cfg["cohorts"] if a != "real-honest"]:
        for f, s in stream(arm):
            take(arm, f, s, honest[main_task_id(s)])
    expected = Counter(r["arm"] for r in shared["samples"])
    if seen != expected:
        raise ValueError(f"streamed {dict(seen)}, expected {dict(expected)} samples per arm")
    print("staged per arm:", dict(Counter(it["arm"] for it in items)), "| truncated:",
          dict(Counter(it["arm"] for it in items if it["truncated"])))
    if level_cfgs:  # one download, every level: each run's inputs scrubbed at its own level, then uploaded
        raw = items
        for x in level_cfgs:
            upload(x, [it | {"messages": scrubbed(it["messages"], x.get("scrub_level", 0))} for it in raw], x["shards"])
        return
    upload(cfg, items, n_shards)
    if cfg.get("stage_only"):  # inputs for another extraction (e.g. all layers), read through its inputs_from
        return
    run_shards(cfg, n_shards, items, limit)


def upload(cfg: dict, items: list[dict], n_shards: int):
    """Deal the staged items to shards (longest first, balanced by projected time) and upload them with the config."""
    from probe_shift.models import QWEN3_32B

    a, b = TIME_FIT
    cost = lambda it: a * min(it["tokens"], cfg["max_tokens"]) + b * min(it["tokens"], cfg["max_tokens"]) ** 2
    shards, load_s = [[] for _ in range(n_shards)], [0.0] * n_shards
    for it in sorted(items, key=cost, reverse=True):
        i = min(range(n_shards), key=load_s.__getitem__)
        shards[i].append(it)
        load_s[i] += cost(it)
    print("projected hours per shard (upper bound):", [round(x / 3600, 2) for x in load_s])
    # Resuming skips transcripts already written, so a resumed run must have the same settings as the one it resumes,
    # including the scrub's version from level 2 on (set in main; runs staged before the version existed used version 1).
    try:
        stored = json.loads(b"".join(out_vol.read_file(f"{cfg['run']}/config.json")))
    except (FileNotFoundError, modal.exception.NotFoundError):
        stored = None
    if stored is not None:  # runs from before the pin loaded this revision, the only one on the HF cache Volume
        stored.setdefault("revision", QWEN3_32B["revision"])
        if stored.get("scrub_level", 0) >= 2:
            stored.setdefault("scrub_version", 1)
    if stored is not None and {k: v for k, v in stored.items() if k != "shards"} != {k: v for k, v in cfg.items() if k != "shards"}:
        raise ValueError(f"{cfg['run']} exists with different settings; use a new run name")
    with out_vol.batch_upload(force=True) as batch:
        for i, sh in enumerate(shards):
            batch.put_file(io.BytesIO("".join(json.dumps(x) + "\n" for x in sh).encode()), f"{cfg['run']}/inputs/shard_{i}.jsonl")
        batch.put_file(io.BytesIO(json.dumps(cfg, indent=1).encode()), f"{cfg['run']}/config.json")
    print(f"uploaded {len(items)} transcripts in {n_shards} shards to {OUTPUT}/{cfg['run']}", flush=True)


def run_shards(cfg: dict, n_shards: int, items: list[dict], limit: int):
    started = time.time()
    calls = [extract_shard.spawn(cfg, i) for i in range(n_shards)]
    print("spawned:", [c.object_id for c in calls], flush=True)
    stats = []
    for shard, call in enumerate(calls):
        for attempt in range(120):  # up to an hour of network trouble per shard, then record it and move on
            try:
                stats.extend(call.get())
                break
            except (socket.gaierror, ConnectionError, TimeoutError, modal.exception.ConnectionError, modal.exception.ServiceError) as e:
                # Usually our network (the pooled run died on a DNS error), so retry; a shard that itself ended on
                # a ConnectionError re-raises it every time, hence the bound. Its stats are on the Volume either way.
                print(f"lost the connection while waiting on shard {shard} ({e!r}); retry {attempt + 1} in 30 s", flush=True)
                time.sleep(30)
            except Exception as e:
                print(f"shard {shard} crashed: {e!r}")
                stats.append({"key": None, "arm": None, "status": "shard_crashed", "shard": shard, "error": repr(e)[:500]})
                break
        else:
            stats.append({"key": None, "arm": None, "status": "shard_unreachable", "shard": shard})
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = {"wall_seconds": round(time.time() - started), "staged": len(items),
           "config": cfg | {"modal": {"app": APP, "gpu": GPU, "shards": n_shards, "timeout_s": TIMEOUT_S, "pip": PIP, "output_volume": OUTPUT}},
           "transcripts": stats}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"extract_{stamp}{'_test' if limit else ''}.json").write_text(json.dumps(out, indent=1))
    ok = [s for s in stats if s["status"] == "ok"]
    done = ok + [s for s in stats if s["status"] == "already_done"]  # resumed transcripts carry their saved stats
    bad = [s for s in stats if s["status"] in ("failed", "shard_crashed", "shard_unreachable")]
    print(f"done in {out['wall_seconds'] / 3600:.2f} h: {len(ok)} read ({sum(s['truncated'] for s in ok)} truncated), "
          f"{len(bad)} failed, {sum(s['status'] == 'already_done' for s in stats)} already done; over read and already done: "
          f"positions kept {sum(s['positions_kept'] for s in done)}, dropped past the cut {sum(s['positions_dropped'] for s in done)}; "
          f"non-finite values {sum(s['nonfinite'] for s in done)}; GPU seconds this launch {sum(s['seconds'] for s in ok):.0f}")
