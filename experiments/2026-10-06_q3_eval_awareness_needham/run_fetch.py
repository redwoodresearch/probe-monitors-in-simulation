"""Fetch and check the Needham et al. (2025) transcripts on Modal CPU (no GPU). Raw data stays on the private Volume.

Stages (run in order; each is idempotent):
  download  with the HF secret: the gated dataset repo at a pinned revision -> /data/raw. Prints decrypt.sh, which
            the next stage runs: read it first.
  decrypt   no secret: runs the dataset's scripts/decrypt.sh in a scratch copy -> /data/dataset.json
  prepare   no secret: converts each sample to reader messages, renders it with Qwen3-32B's chat template as the
            LinuxArena reader does, cuts it to max_tokens, and finds the assistant content tokens
            (probe_shift.needham). Writes /data/inputs/items.jsonl (ids and token indices) and texts.jsonl (the
            rendered text within the cut, for TF-IDF), and returns counts only: per source, label, chat/agentic,
            lengths, truncation, render failures, transcripts with no assistant token, and the projected GPU cost.
            Most chat samples are a prompt with no response, so `all` (every message's content tokens) is read too.
Writes results/fetch_<stamp>.json (counts, no transcript text: GAIA rows must not leave a gated or private store).

  uv run modal run experiments/2026-10-06_q3_eval_awareness_needham/run_fetch.py --stage download
"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q3evalaw_fetch_kill-2026-11-06"
VOLUME = "probe-shift_aswin-surya_sub-q3evalaw_data_kill-2026-11-06"  # config/needham.yaml names the same (checked)
HF_CACHE, HF_SECRET = "probe-shift_aswin-surya_hf-cache", "huggingface-secret-aswin"

app = modal.App(APP)
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE).read_only()
image = (modal.Image.debian_slim(python_version="3.12").apt_install("unzip", "zip", "p7zip-full", "openssl", "gnupg")
         .uv_pip_install("huggingface_hub>=0.34", "transformers==4.56.2", "jinja2", "numpy")
         .env({"HF_HOME": "/hf"}).add_local_python_source("probe_shift"))


@app.function(image=image, volumes={"/data": vol}, secrets=[modal.Secret.from_name(HF_SECRET)], timeout=1800)
def download(cfg: dict) -> dict:
    from huggingface_hub import snapshot_download

    d = cfg["dataset"]
    snapshot_download(d["repo"], repo_type="dataset", revision=d["revision"], local_dir="/data/raw")
    vol.commit()
    files = sorted(str(p.relative_to("/data/raw")) for p in Path("/data/raw").rglob("*") if p.is_file() and ".cache" not in p.parts)
    return {"files": files, "decrypt_sh": Path("/data/raw/scripts/decrypt.sh").read_text()}


@app.function(image=image, volumes={"/data": vol}, timeout=1800)
def decrypt(cfg: dict) -> dict:
    import shutil
    import subprocess

    work = Path("/tmp/work")
    shutil.copytree("/data/raw", work, ignore=shutil.ignore_patterns(".cache"))
    r = subprocess.run(["bash", "scripts/decrypt.sh"], cwd=work, capture_output=True, text=True, timeout=1200, env={"PATH": "/usr/bin:/bin"})
    if r.returncode:
        raise RuntimeError(f"decrypt.sh failed ({r.returncode}): {r.stderr[-2000:]}")
    found = sorted(work.rglob(cfg["dataset"]["file"]))
    if len(found) != 1:
        raise FileNotFoundError(f"expected one {cfg['dataset']['file']}, found {found}")
    shutil.copy(found[0], "/data/dataset.json")
    vol.commit()
    return {"from": str(found[0].relative_to(work)), "bytes": found[0].stat().st_size, "stdout_tail": r.stdout[-500:]}


def projected_seconds(cfg, n_tokens: int) -> float:
    a, b = cfg["time_fit"]
    t = min(n_tokens, cfg["max_tokens"])
    return a * t + b * t * t


@app.function(image=image, volumes={"/data": vol, "/hf": hf_cache}, cpu=8, memory=32768, timeout=3600)
def prepare(cfg: dict) -> dict:
    import hashlib
    from collections import Counter, defaultdict

    import numpy as np
    from transformers import AutoTokenizer

    from probe_shift.needham import content_tokens, reader_messages

    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"], local_files_only=True)
    samples = json.loads(Path("/data/dataset.json").read_text())
    items, texts, failed, dropped_all = [], [], [], Counter()
    for s in samples:
        md = s.get("metadata") or {}
        if isinstance(s["input"], str):
            s = s | {"input": [{"role": "user", "content": s["input"]}]}
        row = {"id": s["id"], "uid": hashlib.sha1(str(s["id"]).encode()).hexdigest()[:16], "source": md["category"],
               "label": int(bool(md["eval_category"])), "agentic": md.get("scaffold") is not None,
               "scaffold": md.get("scaffold"), "eval_type": md.get("eval_type", s.get("eval_type")),
               "human": [md.get(k, s.get(k)) for k in ("eval_author1", "eval_author2")],
               "has_system": bool(s["input"]) and s["input"][0]["role"] == "system"}
        try:
            msgs, dropped = reader_messages(s)
            r = content_tokens(tok, msgs, cfg["max_tokens"])
        except ValueError as e:  # a render mismatch: counted and reported, never silently dropped; anything else raises
            failed.append(row | {"error": f"{type(e).__name__}: {str(e)[:200]}"})
            continue
        dropped_all.update(dropped)
        items.append(row | {"n_tokens": r["n_tokens"], "truncated": r["truncated"], "think_turns": r["think_turns"],
                            "dropped": dict(dropped), "n_assistant_turns": sum(m["role"] == "assistant" for m in msgs),
                            "ids": r["ids"], "assistant": r["assistant"], "all": r["all"]})
        texts.append({"uid": row["uid"], "text": r["text_cut"]})
    if len({it["uid"] for it in items}) != len(items):
        raise ValueError("duplicate sample ids")
    out = Path("/data/inputs")
    out.mkdir(exist_ok=True)
    (out / "items.jsonl").write_text("".join(json.dumps(it) + "\n" for it in items))
    (out / "texts.jsonl").write_text("".join(json.dumps(t) + "\n" for t in texts))
    vol.commit()

    by = defaultdict(list)
    for it in items:
        by[it["source"]].append(it)
    per_source = {src: {"n": len(v), "label": v[0]["label"], "agentic": sorted({x["agentic"] for x in v}),
                        "labels": sorted({x["label"] for x in v}),
                        "tokens_median": int(np.median([x["n_tokens"] for x in v])), "tokens_max": max(x["n_tokens"] for x in v),
                        "truncated": sum(x["truncated"] for x in v), "no_assistant_token": sum(not x["assistant"] for x in v),
                        "gpu_seconds": round(sum(projected_seconds(cfg, x["n_tokens"]) for x in v), 1)}
                  for src, v in sorted(by.items())}
    cell = Counter((("agentic" if it["agentic"] else "chat"), ("eval" if it["label"] else "deploy")) for it in items)
    usable = [it for it in items if it["all"]]  # read on the GPU; the assistant readouts need it["assistant"] too
    gpu_s = sum(projected_seconds(cfg, it["n_tokens"]) for it in usable)
    return {"samples": len(samples), "rendered": len(items), "failed": [{k: f[k] for k in ("id", "source", "label", "error")} for f in failed],
            "usable": len(usable), "cells": {f"{a}/{b}": n for (a, b), n in sorted(cell.items())},
            "with_assistant_token": {f"{a}/{b}": n for (a, b), n in sorted(Counter((("agentic" if it["agentic"] else "chat"), ("eval" if it["label"] else "deploy")) for it in items if it["assistant"]).items())},
            "sources": len(by), "sources_with_mixed_labels": [s for s, v in per_source.items() if len(v["labels"]) > 1],
            "has_system_by_label": {str(l): round(float(np.mean([it["has_system"] for it in items if it["label"] == l])), 3) for l in (0, 1)},
            "think_transcripts": sum(it["think_turns"] > 0 for it in items), "dropped_parts": dict(dropped_all),
            "tokens_total_cut": int(sum(min(it["n_tokens"], cfg["max_tokens"]) for it in usable)),
            "truncated": sum(it["truncated"] for it in items),
            "projected": {"gpu_hours": round(gpu_s / 3600, 3),
                          "dollars": round((gpu_s + cfg["shards"] * cfg["load_seconds"]) / 3600 * cfg["gpu_dollars_per_hour"], 2)},
            "per_source": per_source}


@app.local_entrypoint()
def main(stage: str):
    import yaml

    from probe_shift.models import QWEN3_32B

    cfg = yaml.safe_load((HERE / "config" / "needham.yaml").read_text()) | {"revision": QWEN3_32B["revision"]}
    if cfg["model"] != QWEN3_32B["name"] or (cfg["volume"], cfg["hf_cache"], cfg["hf_secret"]) != (VOLUME, HF_CACHE, HF_SECRET):
        raise ValueError("config/needham.yaml disagrees with this script (model revision pin or Modal names)")
    fn = {"download": download, "decrypt": decrypt, "prepare": prepare}[stage]
    res = fn.remote(cfg)
    if stage == "download":
        print("files:", res["files"], "\n--- scripts/decrypt.sh ---\n" + res["decrypt_sh"])
        return
    print(json.dumps({k: v for k, v in res.items() if k != "per_source"}, indent=1))
    if stage == "prepare":
        for src, v in res["per_source"].items():
            print(f"  {src:28s} n={v['n']:3d} eval={v['label']} agentic={v['agentic']} median={v['tokens_median']:6d} "
                  f"max={v['tokens_max']:7d} trunc={v['truncated']:2d} no_asst={v['no_assistant_token']:2d} gpu_s={v['gpu_seconds']}")
        stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
        from probe_shift.offpolicy import provenance

        path = HERE / "results" / f"fetch_{stamp}.json"
        path.write_text(json.dumps({"config": cfg, "app": APP, "provenance": provenance(__file__, HERE.parent.parent / "src/probe_shift/needham.py",
                                                                                      HERE.parent.parent / "src/probe_shift/reader.py")} | res, indent=1))
        print("->", path)
