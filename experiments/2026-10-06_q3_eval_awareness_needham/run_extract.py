"""Qwen3-32B reads every Needham transcript (Modal H100s) and saves four readouts per transcript at all 64 layers
(residual stream after each decoder layer, fp16), from the token indices run_fetch.py's prepare stage wrote:
  asst_mean / asst_last   mean over the assistant content tokens / the last one (primary; only if the transcript
                          has an assistant content token within the cut)
  all_mean / all_last     the same over every message's content tokens (most chat samples are a prompt only)
Never the end-of-sequence template token: every index lies entirely inside message content.

Shards are balanced by projected time; each container skips transcripts already written (the .json marker is
written last), so a retry resumes. Each shard writes its stats to the Volume, so they survive a lost client.

  uv run modal run --detach experiments/2026-10-06_q3_eval_awareness_needham/run_extract.py [--limit 4]
"""

import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q3evalaw_extract_kill-2026-11-06"
VOLUME = "probe-shift_aswin-surya_sub-q3evalaw_data_kill-2026-11-06"
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
GPU = "H100!"
PIP = ["torch==2.8.0", "transformers==4.56.2", "accelerate==1.10.1", "numpy"]
RUN = "2026-10-06_needham_qwen3-32b_v1"

app = modal.App(APP)
vol = modal.Volume.from_name(VOLUME)
hf_cache = modal.Volume.from_name(HF_CACHE).read_only()
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PIP)
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}).add_local_python_source("probe_shift"))


@app.function(image=image, volumes={"/data": vol}, cpu=4, memory=16384, timeout=1800)
def plan(cfg: dict, n_shards: int, limit: int) -> list[list[str]]:
    """uids per shard, balanced by projected seconds (longest first, greedy)."""
    a, b = cfg["time_fit"]
    items = [json.loads(l) for l in Path("/data/inputs/items.jsonl").read_text().splitlines()]
    items = [it for it in items if it["all"]]
    if limit:  # a test: the longest and limit - 1 short transcripts
        items = sorted(items, key=lambda it: -len(it["ids"]))[:1] + sorted(items, key=lambda it: len(it["ids"]))[:limit - 1]
    secs = {it["uid"]: a * len(it["ids"]) + b * len(it["ids"]) ** 2 for it in items}
    shards, load = [[] for _ in range(n_shards)], [0.0] * n_shards
    for uid in sorted(secs, key=secs.get, reverse=True):
        i = load.index(min(load))
        shards[i].append(uid)
        load[i] += secs[uid]
    return shards


@app.function(image=image, gpu=GPU, volumes={"/hf": hf_cache, "/data": vol}, timeout=3 * 3600,
              retries=modal.Retries(max_retries=1, initial_delay=10.0))
def extract_shard(cfg: dict, run: str, uids: list[str], shard: int) -> list[dict]:
    import os

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    want = set(uids)
    items = [it for it in map(json.loads, Path("/data/inputs/items.jsonl").read_text().splitlines()) if it["uid"] in want]
    out = Path("/data/acts") / run
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(cfg["model"], revision=cfg["revision"], dtype=getattr(torch, cfg["dtype"]),
                                                device_map="cuda", attn_implementation=cfg["attn_implementation"]).model.eval()
    load_s = time.time() - t0
    layers = list(range(len(base.layers)))
    stats = []
    for n, it in enumerate(items, 1):
        fp = fingerprint(cfg, it)
        if (out / f"{it['uid']}.json").exists():  # resume only over outputs of exactly these inputs and settings
            done = json.loads((out / f"{it['uid']}.json").read_text())
            if done.get("fingerprint") != fp:
                raise FileExistsError(f"{run}/{it['uid']} was extracted from other inputs or settings; change RUN")
            stats.append({"uid": it["uid"], "status": "already_done", "nonfinite": done["nonfinite"]})
            continue
        idx = {k: torch.tensor(it[k], device="cuda", dtype=torch.long) for k in ("assistant", "all") if it[k]}
        rows = {f"{k}_{r}": [] for k in idx for r in ("mean", "last")}

        def hook(module, args, output):
            x = (output[0] if isinstance(output, tuple) else output)[0]
            for k, i in idx.items():
                rows[f"{k}_mean"].append(x.index_select(0, i).float().mean(0))
                rows[f"{k}_last"].append(x[i[-1]].float())

        handles = [base.layers[l].register_forward_hook(hook) for l in layers]  # hooks fire in layer order
        torch.cuda.synchronize()
        t = time.time()
        try:
            with torch.inference_mode():
                base(input_ids=torch.tensor([it["ids"]], device="cuda"), use_cache=False)
        finally:
            for h in handles:
                h.remove()
        arrays = {k.replace("assistant", "asst"): torch.stack(v).half().cpu().numpy() for k, v in rows.items()}
        seconds = time.time() - t
        nonfinite = int(sum((~np.isfinite(v.astype(np.float32))).sum() for v in arrays.values()))
        np.savez(out / f"{it['uid']}.npz", **arrays)
        meta = {k: it[k] for k in ("uid", "id", "source", "label", "agentic", "n_tokens", "truncated")} | {
            "kept_tokens": len(it["ids"]), "n_assistant": len(it["assistant"]), "n_all": len(it["all"]),
            "arrays": sorted(arrays), "layers": layers, "seconds": round(seconds, 2), "nonfinite": nonfinite, "fingerprint": fp}
        tmp = out / f".{it['uid']}.json.tmp"
        tmp.write_text(json.dumps(meta))
        os.replace(tmp, out / f"{it['uid']}.json")
        stats.append({"uid": it["uid"], "status": "ok", "tokens": len(it["ids"]), "seconds": round(seconds, 2), "nonfinite": nonfinite})
        if n % 25 == 0:
            vol.commit()
            print(f"shard {shard}: {n}/{len(items)}", flush=True)
    (out / "stats").mkdir(exist_ok=True)
    (out / "stats" / f"shard_{shard}.json").write_text(json.dumps({"load_seconds": round(load_s), "device": torch.cuda.get_device_name(),
                                                                   "items": stats}))
    vol.commit()
    return stats


def fingerprint(cfg: dict, it: dict) -> str:
    """sha256 of what a transcript's activations depend on: the model and its settings, the token ids and the
    readout indices (which also fix max_tokens and the rendering)."""
    import hashlib

    keys = ("model", "revision", "dtype", "attn_implementation", "max_tokens")
    return hashlib.sha256(json.dumps([{k: cfg[k] for k in keys}, it["ids"], it["assistant"], it["all"]]).encode()).hexdigest()


@app.local_entrypoint()
def main(limit: int = 0):
    import yaml

    from probe_shift.models import QWEN3_32B
    from probe_shift.offpolicy import provenance

    cfg = yaml.safe_load((HERE / "config" / "needham.yaml").read_text()) | {"revision": QWEN3_32B["revision"]}
    if cfg["model"] != QWEN3_32B["name"] or cfg["volume"] != VOLUME:
        raise ValueError("config/needham.yaml disagrees with this script")
    run = RUN + (f"_test{limit}" if limit else "")
    n_shards = 1 if limit else cfg["shards"]
    shards = plan.remote(cfg, n_shards, limit)
    print(f"{run}: {sum(map(len, shards))} transcripts in {n_shards} shards", flush=True)
    stats = [s for part in extract_shard.starmap([(cfg, run, u, i) for i, u in enumerate(shards)]) for s in part]
    ok = [s for s in stats if s["status"] == "ok"]
    bad = int(sum(s.get("nonfinite", 0) for s in stats))
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S") + ("_test" if limit else "")
    summary = {"run": run, "volume_dir": f"{VOLUME}/acts/{run}", "transcripts": len(stats), "written": len(ok),
               "already_done": sum(s["status"] == "already_done" for s in stats), "nonfinite": bad,
               "gpu_seconds_forward": round(sum(s["seconds"] for s in ok), 1), "config": cfg | {"app": APP, "gpu": GPU, "pip": PIP},
               "provenance": provenance(__file__, HERE.parent.parent / "src/probe_shift/needham.py")}
    (HERE / "results" / f"extract_{stamp}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("config", "provenance")}, indent=1))
    if bad:
        raise ValueError(f"{bad} non-finite activations")
