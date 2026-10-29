"""Read retrain A's pairs with Qwen3.5-27B: A's capped content rows and B's turn-start row (one H100, 2026-10-10).

README "Retrain A and B". The set (run_build_retrain.py's, on the Volume) is checked against the build result's
sha256. For each dialogue, one forward pass (pinned revision, bf16, sdpa, batch 1, end-padded to a multiple of 1,024
tokens, as run_extract.py) stores in fp16 at layers 8, 16, 24, 32, 40, 48, 56 and 63:
- A: the last 32 tokens of the final call's content span (all of it when shorter);
- B: the last token of "<|im_start|>assistant\\n" opening the final call's turn (one row).
They go to two runs on the Volume, `retrainA_<stamp>` and `retrainB_<stamp>`, laid out as run_extract.py's (per layer
`L<l>.npy`, `meta.json`, `manifest.json`), so run_fit.py fits both unchanged: per token is A's capped mean (or B's one
row), last token is the span's last token. Real calls never leave Modal: the result holds counts and timings, failures
only their type and line numbers, and stdout and stderr stay discarded. Timeout 600 s, retries 0, and a list-rate
watcher stops the app at the stage's $0.94 stop less $0.10.

  PYTHONPATH=src uv run modal run --detach experiments/2026-10-10_eval_awareness_probe/run_extract_retrain.py --build results/build_retrain_<stamp>.json
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
APP = "probe-shift_aswin-surya_sub-evalaware_extract-retrain_kill-2026-11-09"
VOL = "probe-shift_aswin-surya_sub-evalaware_acts_kill-2026-11-09"
BUCKET, CAP = 1024, 32
LAYERS = [8, 16, 24, 32, 40, 48, 56, 63]
PIP = ["torch==2.8.0", "transformers==5.19.0", "accelerate>=1.1.0", "numpy", "huggingface_hub", "flash-linear-attention==0.5.2"]

app = modal.App(APP)
vol = modal.Volume.from_name(VOL, version=2)
hf_cache = modal.Volume.from_name("probe-shift_aswin-surya_hf-cache")
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PIP).env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})
         .add_local_python_source("probe_shift"))


LOG_FD = 1


def log(msg: str):  # counts only, to the real stdout (the Modal log) while the streams are discarded
    import os
    os.write(LOG_FD, (msg + "\n").encode())


class Safe(Exception):
    """A failure whose message holds only counts and synthetic ids, so it may leave the container."""


@app.function(image=image, gpu="H100", volumes={"/hf": hf_cache, "/acts": vol}, memory=65536, timeout=600, retries=0)
def extract(set_path: str, set_sha: str, stamp: str, provenance: dict) -> dict:
    global LOG_FD
    import os
    import sys
    import traceback

    null = os.open(os.devnull, os.O_WRONLY)
    LOG_FD = os.dup(1)
    os.dup2(null, 1)
    os.dup2(null, 2)
    try:
        return _extract(set_path, set_sha, stamp, provenance)
    except Safe:
        raise
    except BaseException as e:  # never send LinuxArena text back: only the type and line numbers
        lines = [f.lineno for f in traceback.extract_tb(e.__traceback__) if f.filename.endswith("run_extract_retrain.py")]
        raise RuntimeError(f"extract failed: {type(e).__name__} at run_extract_retrain.py lines {lines}") from None
    finally:  # the streams stay discarded for the container's life: native buffers may flush late
        sys.stdout.flush()
        sys.stderr.flush()


def _extract(set_path: str, set_sha: str, stamp: str, provenance: dict) -> dict:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from probe_shift.models import QWEN35_27B
    from probe_shift.reader import segment_end_positions

    raw = (Path("/acts") / set_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != set_sha:
        raise Safe("the set on the Volume does not match the build's sha256")
    items = [json.loads(x) for x in raw.decode().splitlines()]
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(QWEN35_27B["name"], revision=QWEN35_27B["revision"])
    lm = AutoModelForCausalLM.from_pretrained(QWEN35_27B["name"], revision=QWEN35_27B["revision"], dtype=torch.bfloat16,
                                              device_map="cuda", attn_implementation="sdpa").eval()
    load_s = time.time() - t0
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    rows = {"A": {l: [] for l in LAYERS}, "B": {l: [] for l in LAYERS}}
    meta = {"A": [], "B": []}
    n_a, fwd_s, n_tok, span_lens = 0, 0.0, 0, []
    for k, it in enumerate(items):
        r = segment_end_positions(tok, it["messages"], None)
        last_step = sum(m["role"] == "assistant" for m in it["messages"]) - 1
        ends = [p for p in r["positions"] if p["kind"] == "call_end" and p["kept"] and p.get("span")
                and (p["step"], p["action"]) == (last_step, 0)]
        if len(ends) != 1:
            raise Safe(f"{it['id']}: {len(ends)} final-call spans")
        i0, i1 = ends[0]["span"]
        a0 = max(i0, i1 - CAP)
        starts = [j for j in range(i0) if r["ids"][j] == im_start]
        if not starts or tok.decode(r["ids"][starts[-1]:starts[-1] + 3]) != "<|im_start|>assistant\n":
            raise Safe(f"{it['id']}: the final call's turn does not open with the assistant header")
        b = starts[-1] + 2
        acts = {}

        def hook_for(layer):
            def hook(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                acts[layer] = (h[0, a0:i1].to(torch.float16).cpu().numpy(), h[0, b:b + 1].to(torch.float16).cpu().numpy())
            return hook

        handles = [lm.model.layers[l].register_forward_hook(hook_for(l)) for l in LAYERS]
        t1 = time.time()
        try:
            with torch.inference_mode():
                ids = r["ids"] + [pad] * (-len(r["ids"]) % BUCKET)
                lm.model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)
        finally:
            for h in handles:
                h.remove()
        fwd_s += time.time() - t1
        n_tok += len(r["ids"])
        for l in LAYERS:
            rows["A"][l].append(acts[l][0])
            rows["B"][l].append(acts[l][1])
        base = {k_: it[k_] for k_ in ("id", "family", "split", "pair", "label", "task", "mechanism", "source")}
        meta["A"].append(base | {"row_start": n_a, "n_rows": i1 - a0, "span": [a0, i1], "full_span": [i0, i1]})
        meta["B"].append(base | {"row_start": k, "n_rows": 1, "span": [b, b + 1]})
        n_a += i1 - a0
        span_lens.append(i1 - i0)
        if k % 200 == 0:
            log(f"{k}/{len(items)} dialogues, {n_tok / max(fwd_s, 1e-9):.0f} tok/s, {time.time() - t0:.0f} s")
    runs = {}
    for v in ("A", "B"):
        d = Path("/acts") / f"retrain{v}_{stamp}"
        d.mkdir(parents=True, exist_ok=False)
        for l in LAYERS:
            np.save(d / f"L{l}.npy", np.concatenate(rows[v][l]))
        (d / "meta.json").write_text(json.dumps(meta[v]))
        (d / "manifest.json").write_text(json.dumps(provenance | {
            "sets_sha256": set_sha, "set": set_path, "model": QWEN35_27B["name"], "revision": QWEN35_27B["revision"],
            "dtype": "bfloat16", "attn": "sdpa", "batch": 1, "padding": f"end, to a multiple of {BUCKET}", "layers": LAYERS,
            "readout": {"A": f"last {CAP} tokens of the final call_end span", "B": "turn start (last token of <|im_start|>assistant\\n)"}[v]}))
        runs[v] = d.name
    vol.commit()
    return {"runs": runs, "dialogues": len(items), "A rows": n_a, "B rows": len(items), "tokens": n_tok,
            "full span length (median)": sorted(span_lens)[len(span_lens) // 2], "spans capped": sum(x > CAP for x in span_lens),
            "forward_s": round(fwd_s, 1), "load_s": round(load_s, 1), "tokens_per_s": round(n_tok / max(fwd_s, 1e-9)),
            "gpu": torch.cuda.get_device_properties(0).name}


@app.local_entrypoint()
def main(build: str):
    b = json.loads((HERE / build).read_text())
    if b["dialogues"] != 1080 or b["pairs by split"] != {"train": 320, "dev": 100, "heldout": 120}:
        raise ValueError("the build result is not a full 540-pair set")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=ROOT).stdout.strip())
    prov = {"git": git, "dirty": dirty, "build": build}
    rel = b["set"].split("/", 1)[1]  # the path inside the Volume
    out = extract.remote(rel, b["set_sha256"], stamp, prov)
    out = {"volume": VOL, "layers": LAYERS, "cap": CAP, "set_sha256": b["set_sha256"]} | prov | out
    path = HERE / "results" / f"extract_retrain_{stamp}.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    print(f"-> {path}")
