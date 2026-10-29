"""Read step 1's pairs with Qwen3.5-27B and keep the final call's content-argument tokens (one H100, 2026-10-10).

For every dialogue (build_sets.py's file, passed in by sha256), the reader's segment-end positions give the final
call's content-argument span. One forward pass (pinned revision, bf16, sdpa, batch 1) stores that
span's residual stream at layers 4, 8, ..., 60 and 63 in fp16. Dialogues are padded at the end to a multiple of
1,024 tokens: Qwen3.5's linear-attention kernels re-tune for every new length (a 20-dialogue pilot unpadded ran at 103
tok/s), and end padding cannot change earlier positions in a causal model. The last row of a span is Stage 1's readout; all rows
feed Apollo's per-token probe. Writes per layer `L<l>.npy` (span rows) plus `meta.json` (one entry per dialogue: id,
labels, split, family, mechanism, its rows) to the Volume under the run name, committing as it goes. Retries 0: a
crashed run is rerun, not resumed (it is a few minutes).

  uv run modal run experiments/2026-10-10_eval_awareness_probe/run_extract.py --sets .run_cache/eval_awareness_probe/sets_<stamp>.jsonl
"""

import hashlib
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
APP = "probe-shift_aswin-surya_sub-evalaware_extract_kill-2026-11-09"
VOL = "probe-shift_aswin-surya_sub-evalaware_acts_kill-2026-11-09"
BUCKET = 1024  # dialogues are padded at the end to a multiple of this (no effect on earlier positions: causal)
LAYERS = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 63]
PIP = ["torch==2.8.0", "transformers==5.19.0", "accelerate>=1.1.0", "numpy", "huggingface_hub", "flash-linear-attention==0.5.2"]

app = modal.App(APP)
vol = modal.Volume.from_name(VOL, create_if_missing=True, version=2)
hf_cache = modal.Volume.from_name("probe-shift_aswin-surya_hf-cache")
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PIP).env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})
         .add_local_python_source("probe_shift"))


@app.function(image=image, gpu="H100", volumes={"/hf": hf_cache, "/acts": vol}, memory=65536, timeout=1800, retries=0)
def extract(items: list[dict], run: str, provenance: dict) -> dict:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from probe_shift.models import QWEN35_27B
    from probe_shift.reader import segment_end_positions

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(QWEN35_27B["name"], revision=QWEN35_27B["revision"])
    lm = AutoModelForCausalLM.from_pretrained(QWEN35_27B["name"], revision=QWEN35_27B["revision"], dtype=torch.bfloat16,
                                              device_map="cuda", attn_implementation="sdpa").eval()
    load_s = time.time() - t0
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    rows = {l: [] for l in LAYERS}
    meta, n_rows, fwd_s, n_tok = [], 0, 0.0, 0
    for k, it in enumerate(items):
        r = segment_end_positions(tok, it["messages"], None)
        ends = [p for p in r["positions"] if p["kind"] == "call_end" and p["kept"] and p.get("span")]
        last = max(ends, key=lambda p: (p["step"], p["action"]))  # the final call
        i0, i1 = last["span"]
        acts = {}

        def hook_for(layer):
            def hook(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                acts[layer] = h[0, i0:i1].to(torch.float16).cpu().numpy()
            return hook

        handles = [lm.model.layers[l].register_forward_hook(hook_for(l)) for l in LAYERS]
        t1 = time.time()
        try:
            with torch.inference_mode():
                ids = r["ids"] + [pad] * (-len(r["ids"]) % BUCKET)  # one padded shape: Qwen3.5's linear-attention kernels
                lm.model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)  # re-tune for every new length
        finally:
            for h in handles:
                h.remove()
        fwd_s += time.time() - t1
        n_tok += len(r["ids"])
        for l in LAYERS:
            rows[l].append(acts[l])
        meta.append({k_: it[k_] for k_ in ("id", "family", "split", "pair", "label", "task", "mechanism")} |
                    {"row_start": n_rows, "n_rows": i1 - i0, "span": [i0, i1], "last_token": last["token"]})
        n_rows += i1 - i0
        if k % 200 == 0:
            print(f"{k}/{len(items)} dialogues, {n_tok / max(fwd_s, 1e-9):.0f} tok/s in forward passes", flush=True)
    d = Path("/acts") / run
    d.mkdir(parents=True, exist_ok=True)
    for l in LAYERS:
        np.save(d / f"L{l}.npy", np.concatenate(rows[l]))
    (d / "meta.json").write_text(json.dumps(meta))
    (d / "manifest.json").write_text(json.dumps(provenance | {"model": QWEN35_27B["name"], "revision": QWEN35_27B["revision"],
                                                              "dtype": "bfloat16", "attn": "sdpa", "batch": 1, "padding": f"end, to a multiple of {BUCKET}",
                                                              "layers": LAYERS, "readout": "final call_end span"}))
    vol.commit()
    return {"dialogues": len(items), "span_rows": n_rows, "tokens": n_tok, "forward_s": round(fwd_s, 1), "load_s": round(load_s, 1),
            "tokens_per_s": round(n_tok / max(fwd_s, 1e-9)), "gpu": torch.cuda.get_device_properties(0).name}


@app.local_entrypoint()
def main(sets: str):
    raw = (ROOT / sets).read_bytes()
    items = [json.loads(l) for l in raw.decode().splitlines()]
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    run = f"step1_{stamp}"
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=ROOT).stdout.strip())
    prov = {"git": git, "dirty": dirty, "sets": Path(sets).name, "sets_sha256": hashlib.sha256(raw).hexdigest()}
    out = extract.remote(items, run, prov)
    out |= {"run": run, "volume": VOL, "layers": LAYERS} | prov
    path = HERE / "results" / f"extract_{stamp}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
