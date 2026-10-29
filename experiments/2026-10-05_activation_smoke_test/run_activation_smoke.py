"""Smoke test: Qwen3-32B reads 20 LinuxArena attack transcripts (10 tasks x real/simulated) on Modal.

Per transcript: render with the Qwen chat template (thinking off), skip it if over `max_tokens`, run one
forward pass on one H100, pool the residual stream at 8 layers per segment (each agent turn and each tool
output: mean, and EMA-max at two decays), and save fp16 vectors plus per-segment metadata to a Volume.
Reports seconds per transcript, tokens per second, peak memory, skipped counts, NaN/inf counts, and the
segment boundaries of two transcripts.

  uv run modal run experiments/2026-10-05_activation_smoke_test/run_activation_smoke.py
  uv run modal run experiments/2026-10-05_activation_smoke_test/run_activation_smoke.py::download   # weights only
"""

import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-smoke_activations"
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
OUTPUT = "probe-shift_aswin-surya_sub-smoke_activations"
GPU = "H100!"  # "!" pins H100 (Modal may otherwise upgrade to H200), so timings project to H100s
TIMEOUT_S = 3600
PIP = ["torch==2.8.0", "transformers==4.56.2", "accelerate==1.10.1", "numpy", "huggingface_hub[hf_transfer]"]

app = modal.App(APP)
hf_cache = modal.Volume.from_name(HF_CACHE, create_if_missing=True)
out_vol = modal.Volume.from_name(OUTPUT, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(*PIP)
    .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("probe_shift")
)


@app.function(image=image, volumes={"/hf": hf_cache}, cpu=8, memory=16384, timeout=TIMEOUT_S)
def download(model: str = "Qwen/Qwen3-32B") -> str:
    from huggingface_hub import snapshot_download

    path = snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*"])
    hf_cache.commit()
    return path


def boundaries(tok, ids, segments, limit=12) -> list[str]:
    lines = []
    for s in segments[:limit]:
        text = tok.decode(ids[s["start"] : s["end"]])
        lines.append(f"{s['kind']:11} step {s['step']:>3} action {str(s['action']):>4} tool {str(s['tool']):24} "
                     f"tokens [{s['start']}, {s['end']}) n={s['n_tokens']:<5} | {text[:70]!r} ... {text[-30:]!r}")
    return lines


@app.function(image=image, gpu=GPU, volumes={"/hf": hf_cache, "/out": out_vol}, timeout=TIMEOUT_S)
def read_all(cfg: dict, items: list[dict], show_boundaries_for: list[str]) -> dict:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from probe_shift.reader import read, render

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    lm = AutoModelForCausalLM.from_pretrained(
        cfg["model"], revision=cfg["revision"], torch_dtype=getattr(torch, cfg["dtype"]), device_map="cuda",
        attn_implementation=cfg["attn_implementation"]
    )
    base = lm.model.eval()
    load_s = time.time() - t0
    layers, decays = cfg["layers"], cfg["pooling"]["ema_max_decays"]
    out_dir = Path("/out") / cfg["run"]
    out_dir.mkdir(parents=True, exist_ok=True)

    per_item, skipped, shown = [], [], {}
    for it in items:
        ids, segments = render(tok, it["messages"])
        if it["id"] in show_boundaries_for:
            shown[it["id"]] = boundaries(tok, ids, segments)
        if len(ids) > cfg["max_tokens"]:
            skipped.append({"id": it["id"], "tokens": len(ids)})
            continue
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t = time.time()
        res = read(base, ids, segments, layers, decays)
        torch.cuda.synchronize()
        seconds = time.time() - t
        kept = [s for s in segments if s["n_tokens"] > 0]
        arrays = {f"L{l}_{k}": v for l, d in res.items() for k, v in d.items()}
        nonfinite = int(sum((~np.isfinite(v)).sum() for v in arrays.values()))
        np.savez(out_dir / f"{it['id']}.npz", **arrays)
        meta = [{"task": it["cell"], "arm": it["origin"], "environment": it["environment"], "segment": s["kind"],
                 "step": s["step"], "action": s["action"], "tool": s["tool"], "n_tokens": s["n_tokens"], "row": r}
                for r, s in enumerate(kept)]
        (out_dir / f"{it['id']}.segments.json").write_text(json.dumps(meta))
        per_item.append({
            "id": it["id"], "arm": it["origin"], "environment": it["environment"], "tokens": len(ids),
            "segments": len(kept), "empty_segments": len(segments) - len(kept),
            "seconds": round(seconds, 2), "tokens_per_s": round(len(ids) / seconds),
            "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
            "nonfinite_values": nonfinite, "max_abs": float(max(np.nanmax(np.abs(v.astype(np.float32))) for v in arrays.values())),
        })
        print(per_item[-1], flush=True)
    out_vol.commit()
    return {"load_seconds": round(load_s, 1), "device": torch.cuda.get_device_name(), "torch": str(torch.__version__),
            "items": per_item, "skipped_over_max_tokens": skipped, "boundaries": shown}


@app.local_entrypoint()
def main():
    import yaml

    from probe_shift.models import QWEN3_32B
    from probe_shift.transcripts import load_sample, reader_messages, system_prompt

    cfg = yaml.safe_load((HERE / "config" / "smoke.yaml").read_text())
    if cfg["model"] != QWEN3_32B["name"]:
        raise ValueError(f"{cfg['model']} has no pinned revision in probe_shift.models")
    cfg["revision"] = QWEN3_32B["revision"]  # passed to both loaders and saved with the results config
    items = []
    for c in cfg["cells"]:
        honest = load_sample(cfg["cohorts"]["honest"], **c["honest_prompt_from"])
        for origin in ("real", "sim"):
            sample = load_sample(cfg["cohorts"][origin], **{k: c[origin][k] for k in ("file", "sample_id", "epoch")})
            items.append({"id": f"{origin}__{c['environment']}", "origin": origin, "environment": c["environment"],
                          "cell": c["cell"], "messages": reader_messages(sample, system_prompt(honest))})
    show = [items[2]["id"], items[3]["id"]]  # one real/sim pair
    print(f"{len(items)} transcripts loaded; showing boundaries for {show}")

    result = read_all.remote(cfg, items, show)
    result["config"] = cfg | {"modal": {"app": APP, "gpu": GPU, "timeout_s": TIMEOUT_S, "pip": PIP,
                                        "hf_cache": HF_CACHE, "output_volume": OUTPUT}}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    path = HERE / "results" / f"smoke_{stamp}.json"
    path.write_text(json.dumps(result, indent=1))

    print(f"\nmodel load {result['load_seconds']}s on {result['device']}")
    for r in result["items"]:
        print(f"  {r['id']:26} {r['tokens']:>6} tok {r['seconds']:>6}s {r['tokens_per_s']:>6} tok/s peak {r['peak_gb']} GB "
              f"segments {r['segments']} (empty {r['empty_segments']}) nonfinite {r['nonfinite_values']}")
    print(f"skipped (over {cfg['max_tokens']} tokens): {result['skipped_over_max_tokens']}")
    for iid, lines in result["boundaries"].items():
        print(f"\nsegment boundaries, {iid}:")
        print("\n".join(lines))
    print(f"\nresults -> {path}")
