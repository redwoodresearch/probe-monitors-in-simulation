"""Full activation extraction: Qwen3-32B reads every LinuxArena transcript under 32k tokens, 8 H100 containers.

Same settings as the 2026-10-05 smoke test (render, layers, pooling, fp16). The local entrypoint streams the
selected samples from S3, normalizes them, packs them into 8 shards balanced by projected run time, and
uploads the shards to the output Volume (containers have no AWS credentials). Each container reads its shard,
skips transcripts already written (so a retried or preempted shard resumes), and commits every 20 transcripts.

  uv run modal run --detach experiments/2026-10-06_activation_extraction/run_extract.py
"""

import io
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-la-activations_extract"
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
OUTPUT = "probe-shift_aswin-surya_sub-la-activations"  # v2 volume: 8 concurrent writers; regenerable, so beta is fine
GPU = "H100!"
N_SHARDS = 8
TIMEOUT_S = 3 * 3600
PIP = ["torch==2.8.0", "transformers==4.56.2", "accelerate==1.10.1", "numpy", "huggingface_hub[hf_transfer]"]
TIME_FIT = (1.51e-4, 1.55e-8)  # seconds ≈ a·T + b·T², from the smoke test

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

    from probe_shift.reader import read, render

    run_dir = Path("/out") / cfg["run"]
    items = [json.loads(l) for l in (run_dir / "inputs" / f"shard_{shard}.jsonl").read_text().splitlines()]
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    base = AutoModelForCausalLM.from_pretrained(cfg["model"], revision=cfg["revision"], dtype=getattr(torch, cfg["dtype"]),
                                                device_map="cuda", attn_implementation=cfg["attn_implementation"]).model.eval()
    layers, decays = cfg["layers"], cfg["pooling"]["ema_max_decays"]
    def read_one(it, stem):
        ids, segments = render(tok, it["messages"])
        if len(ids) > cfg["max_tokens"]:  # the 32k cut is applied here, on the exact rendered length
            return {"key": it["key"], "arm": it["arm"], "status": "over_max_tokens", "tokens": len(ids)}
        torch.cuda.synchronize()
        t = time.time()
        res = read(base, ids, segments, layers, decays)
        torch.cuda.synchronize()
        seconds = time.time() - t
        kept = [s for s in segments if s["n_tokens"] > 0]
        arrays = {f"L{l}_{k}": v for l, d in res.items() for k, v in d.items()}
        np.savez(stem.parent / f"{stem.name}.npz", **arrays)
        meta = {**{k: it[k] for k in ("key", "arm", "cohort", "file", "sample_id", "epoch", "environment", "main_task", "side_task", "cell", "n_submit_calls")},
                "filename": stem.name, "tokens": len(ids),
                "segments": [{"row": r, **{k: s[k] for k in ("kind", "step", "action", "tool", "n_tokens")}} for r, s in enumerate(kept)]}
        (stem.parent / f"{stem.name}.segments.json").write_text(json.dumps(meta))  # written last: marks the transcript done
        return {"key": it["key"], "arm": it["arm"], "status": "ok", "tokens": len(ids), "seconds": round(seconds, 2),
                "segments": len(kept), "nonfinite": int(sum((~np.isfinite(v)).sum() for v in arrays.values()))}

    stats = []
    for n, it in enumerate(items, 1):
        stem = run_dir / "acts" / it["arm"] / filename(it["key"])
        if (stem.parent / f"{stem.name}.segments.json").exists():
            stats.append({"key": it["key"], "arm": it["arm"], "status": "already_done"})
            continue
        stem.parent.mkdir(parents=True, exist_ok=True)
        try:
            stats.append(read_one(it, stem))
        except Exception as e:  # one bad transcript must not take down the shard
            stats.append({"key": it["key"], "arm": it["arm"], "status": "failed", "error": f"{type(e).__name__}: {str(e)[:300]}"})
            print(f"shard {shard}: FAILED {it['key'][:80]}: {type(e).__name__}", flush=True)
        if n % 20 == 0:
            out_vol.commit()
            print(f"shard {shard}: {n}/{len(items)}", flush=True)
    out_vol.commit()
    return stats


def filename(key: str) -> str:
    """Output file stem: the key itself, or a prefix plus a hash when "<key>.segments.json" would pass the 255-byte
    filename limit. Keys that fit keep their name, so outputs written before this rule are still found."""
    import hashlib

    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


@app.local_entrypoint()
def main(limit: int = 0):
    """`--limit N` runs a test on N transcripts per cohort in one container, under a separate run name."""
    import yaml

    from probe_shift.models import QWEN3_32B
    from probe_shift.transcripts import file_samples, invalidity, main_task_id, reader_messages, submit_calls, system_prompt

    cfg = yaml.safe_load((HERE / "config" / "extract.yaml").read_text())
    if cfg["model"] != QWEN3_32B["name"]:
        raise ValueError(f"{cfg['model']} has no pinned revision in probe_shift.models")
    cfg["revision"] = QWEN3_32B["revision"]  # passed to both loaders and saved with the results config
    n_shards = 1 if limit else N_SHARDS
    if limit:
        cfg["run"] += f"_test{limit}"
    # Token counts from the earlier rendering (submit calls kept) are used only to balance shards; the 32k cut
    # is applied in the container on the exact rendered length.
    counts = {(r["arm"], r["file"], r["sample_id"], r["epoch"]): r["tokens"]
              for r in map(json.loads, (HERE.parent / cfg["token_counts"]).read_text().splitlines())}
    # What to read comes from the shared exclusion list (validity and the exact 32k cut); the earlier token counts
    # are only used to balance shards.
    shared = json.loads((HERE.parent.parent / "splits" / cfg["exclusions_file"]).read_text())
    shared_excluded = {(r["arm"], r["key"]): r["reason"] for r in shared["excluded"]}
    files = {arm: sorted({f for (a, f, _, _) in counts if a == arm}) for arm in cfg["cohorts"]}
    honest_prompts, items, runs = {}, [], []
    per_arm = {arm: 0 for arm in cfg["cohorts"]}

    def stream(arm):
        with ThreadPoolExecutor(32) as pool:
            for samples in pool.map(lambda f: [(f, s) for s in file_samples(cfg["cohorts"][arm], f)], files[arm]):
                yield from samples

    def describe(arm, f, s):
        md = s.get("metadata") or {}
        mt = main_task_id(s)
        side = md.get("side_task")
        side = side.get("id") if isinstance(side, dict) else side
        env = md.get("environment") or md.get("env")
        env = env.get("id") if isinstance(env, dict) else env
        return {"key": f"{f.removesuffix('.eval')}__{s['id'].replace('/', '_')}__{s['epoch']}", "arm": arm,
                "cohort": cfg["cohorts"][arm], "file": f, "sample_id": s["id"], "epoch": s["epoch"], "environment": env,
                "main_task": mt, "side_task": side, "cell": f"{mt}:{side}" if side else mt,
                "n_submit_calls": submit_calls(s), "invalid": invalidity(s), "old_tokens": counts[(arm, f, s["id"], s["epoch"])]}

    def take(arm, f, s, prompt):
        d = describe(arm, f, s)
        runs.append(d)  # every sample, valid or not: exclusions and submit counts are reported over all of them
        # In test mode take only transcripts comfortably under the cut (by the earlier count), so every cohort
        # gets read; the exact cut is applied in the container.
        test_ok = not limit or (per_arm[arm] < limit and d["old_tokens"] <= 0.9 * cfg["max_tokens"])
        reason = shared_excluded.get((arm, d["key"]))
        if (reason in ("errored", "stalled", "no_actions")) != (d["invalid"] is not None):
            raise ValueError(f"validity disagrees with {cfg['exclusions_file']} for {arm} {d['key']}: {d['invalid']} vs {reason}")
        if reason is None and test_ok:
            items.append(d | {"messages": reader_messages(s, prompt)})
            per_arm[arm] += 1

    # Honest runs first: they supply the system prompt for every attack run of the same main task, so each main
    # task must have exactly one honest prompt.
    for f, s in stream("real-honest"):
        honest_prompts.setdefault(main_task_id(s), set()).add(system_prompt(s))
        take("real-honest", f, s, system_prompt(s))
    ambiguous = [mt for mt, ps in honest_prompts.items() if len(ps) != 1]
    if ambiguous:
        raise ValueError(f"{len(ambiguous)} main tasks have more than one honest system prompt")
    honest_prompts = {mt: ps.pop() for mt, ps in honest_prompts.items()}
    for arm in [a for a in cfg["cohorts"] if a != "real-honest"]:
        for f, s in stream(arm):
            take(arm, f, s, honest_prompts[main_task_id(s)])
    if len(runs) != len(counts):
        raise ValueError(f"streamed {len(runs)} samples, expected {len(counts)}")

    excluded = {arm: dict(Counter(r["invalid"] for r in runs if r["arm"] == arm and r["invalid"])) for arm in cfg["cohorts"]}
    print("excluded (per arm, per reason):", json.dumps(excluded))
    print(f"{len(items)} transcripts staged: valid and under {cfg['max_tokens']} tokens per {cfg['exclusions_file']} ({len(runs)} streamed)")

    # Balance shards by projected seconds (longest first, into the lightest shard).
    a, b = TIME_FIT
    cost = lambda it: a * it["old_tokens"] + b * it["old_tokens"] ** 2
    shards, load_s = [[] for _ in range(n_shards)], [0.0] * n_shards
    for it in sorted(items, key=cost, reverse=True):
        i = min(range(n_shards), key=load_s.__getitem__)
        shards[i].append(it)
        load_s[i] += cost(it)
    print("projected hours per shard:", [round(x / 3600, 2) for x in load_s])
    run_table = "".join(json.dumps({k: v for k, v in r.items() if k != "old_tokens"}) + "\n" for r in sorted(runs, key=lambda r: (r["arm"], r["key"])))
    with out_vol.batch_upload(force=True) as batch:
        for i, sh in enumerate(shards):
            batch.put_file(io.BytesIO("".join(json.dumps(x) + "\n" for x in sh).encode()), f"{cfg['run']}/inputs/shard_{i}.jsonl")
        batch.put_file(io.BytesIO(run_table.encode()), f"{cfg['run']}/runs.jsonl")  # validity and submit counts, all samples
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    tag = "_test" if limit else ""
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"runs_{stamp}{tag}.jsonl").write_text(run_table)
    print(f"uploaded {len(items)} transcripts in {n_shards} shards to {OUTPUT}/{cfg['run']}")

    started = time.time()
    stats = []
    for shard, shard_stats in enumerate(extract_shard.map([cfg] * n_shards, range(n_shards), return_exceptions=True)):
        if isinstance(shard_stats, BaseException):
            print(f"shard {shard} crashed: {shard_stats!r}")
            stats.append({"key": None, "arm": None, "status": "shard_crashed", "shard": shard, "error": repr(shard_stats)[:500]})
        else:
            stats.extend(shard_stats)
    out = {"wall_seconds": round(time.time() - started), "excluded": excluded,
           "config": cfg | {"modal": {"app": APP, "gpu": GPU, "shards": n_shards, "timeout_s": TIMEOUT_S, "pip": PIP, "output_volume": OUTPUT}},
           "transcripts": stats}
    (HERE / "results" / f"extract_{stamp}{tag}.json").write_text(json.dumps(out, indent=1))
    ok = [s for s in stats if s["status"] == "ok"]
    over = [s for s in stats if s["status"] == "over_max_tokens"]
    bad = [s for s in stats if s["status"] in ("failed", "shard_crashed")]
    print(f"done in {out['wall_seconds'] / 3600:.2f} h: {len(ok)} read, {len(over)} over {cfg['max_tokens']} tokens, {len(bad)} failed, "
          f"{sum(s['status'] == 'already_done' for s in stats)} already done; non-finite values {sum(s['nonfinite'] for s in ok)}; "
          f"GPU seconds {sum(s['seconds'] for s in ok):.0f}")
