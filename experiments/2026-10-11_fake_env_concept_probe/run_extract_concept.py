"""Stage 2: read the concept pairs with Qwen3-32B at the content and turn-start readouts (one H100; README.md).

For each dialogue (build_concept_sets.py's file, passed in by sha256), one forward pass (Qwen3-32B @ 9216db57, bf16,
sdpa, batch 1, end-padded to a multiple of 1,024 tokens, as run_extract.py) stores in fp16, at layers 8, 16, ..., 56, 63:
- content: the last 32 tokens of the final call's call_end span (for Qwen3, the JSON call);
- turn start: the last token of "<|im_start|>assistant\\n" opening the final call's turn.
They go to two runs on the Volume, `conceptC_<stamp>` and `conceptT_<stamp>`, laid out as run_extract.py's, so
experiments/2026-10-10_eval_awareness_probe/run_fit.py fits both unchanged. The dialogues are synthetic; the result
holds counts and timings. Timeout 600 s (about $0.79 at $4.72/h, the H100 + 96 GiB list rate) as a backstop, retries 0,
and an external list-rate watcher stops the app at $0.70.

  PYTHONPATH=src uv run modal run --detach experiments/2026-10-11_fake_env_concept_probe/run_extract_concept.py --sets .run_cache/fake_env_concept_probe/concept_sets_<stamp>.jsonl
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
APP = "probe-shift_aswin-surya_sub-fakeenv_extract_kill-2026-11-09"
VOL = "probe-shift_aswin-surya_sub-evalaware_acts_kill-2026-11-09"
BUCKET, CAP = 1024, 32
LAYERS = [8, 16, 24, 32, 40, 48, 56, 63]
PIP = ["torch==2.8.0", "transformers==5.19.0", "accelerate>=1.1.0", "numpy", "huggingface_hub"]

app = modal.App(APP)
vol = modal.Volume.from_name(VOL, version=2)
hf_cache = modal.Volume.from_name("probe-shift_aswin-surya_hf-cache")
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PIP).env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})
         .add_local_python_source("probe_shift"))


@app.function(image=image, gpu="H100", volumes={"/hf": hf_cache, "/acts": vol}, memory=98304, timeout=600, retries=0)
def extract(items: list[dict], stamp: str, provenance: dict) -> dict:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from probe_shift.models import QWEN3_32B
    from probe_shift.reader import segment_end_positions

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(QWEN3_32B["name"], revision=QWEN3_32B["revision"])
    lm = AutoModelForCausalLM.from_pretrained(QWEN3_32B["name"], revision=QWEN3_32B["revision"], dtype=torch.bfloat16,
                                              device_map="cuda", attn_implementation="sdpa").eval()
    load_s = time.time() - t0
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    rows = {"C": {l: [] for l in LAYERS}, "T": {l: [] for l in LAYERS}}
    meta = {"C": [], "T": []}
    n_c, fwd_s, n_tok, n_pad, spans = 0, 0.0, 0, 0, []
    for k, it in enumerate(items):
        r = segment_end_positions(tok, it["messages"], None)
        last_step = sum(m["role"] == "assistant" for m in it["messages"]) - 1
        ends = [p for p in r["positions"] if p["kind"] == "call_end" and p["kept"] and p.get("span") and (p["step"], p["action"]) == (last_step, 0)]
        if len(ends) != 1:
            raise ValueError(f"{it['id']}: {len(ends)} final-call spans")
        i0, i1 = ends[0]["span"]
        a0 = max(i0, i1 - CAP)
        starts = [j for j in range(i0) if r["ids"][j] == im_start]
        if not starts or tok.decode(r["ids"][starts[-1]:starts[-1] + 3]) != "<|im_start|>assistant\n":
            raise ValueError(f"{it['id']}: the final call's turn does not open with the assistant header")
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
        n_pad += len(ids)
        for l in LAYERS:
            rows["C"][l].append(acts[l][0])
            rows["T"][l].append(acts[l][1])
        base = {k_: it[k_] for k_ in ("id", "family", "split", "pair", "label", "task", "mechanism")}
        meta["C"].append(base | {"row_start": n_c, "n_rows": i1 - a0, "span": [a0, i1], "full_span": [i0, i1]})
        meta["T"].append(base | {"row_start": k, "n_rows": 1, "span": [b, b + 1]})
        n_c += i1 - a0
        spans.append(i1 - i0)
        if k % 200 == 0:
            print(f"{k}/{len(items)} dialogues, {n_tok / max(fwd_s, 1e-9):.0f} tok/s, {time.time() - t0:.0f} s", flush=True)
    runs = {}
    for v in ("C", "T"):
        d = Path("/acts") / f"concept{v}_{stamp}"
        d.mkdir(parents=True, exist_ok=False)
        for l in LAYERS:
            np.save(d / f"L{l}.npy", np.concatenate(rows[v][l]))
        (d / "meta.json").write_text(json.dumps(meta[v]))
        (d / "manifest.json").write_text(json.dumps(provenance | {
            "model": QWEN3_32B["name"], "revision": QWEN3_32B["revision"], "dtype": "bfloat16", "attn": "sdpa", "batch": 1,
            "padding": f"end, to a multiple of {BUCKET}", "layers": LAYERS,
            "readout": {"C": f"last {CAP} tokens of the final call_end span (Qwen3: the JSON call)",
                        "T": "turn start (last token of <|im_start|>assistant\\n)"}[v]}))
        runs[v] = d.name
    vol.commit()
    return {"runs": runs, "dialogues": len(items), "content rows": n_c, "turn-start rows": len(items), "tokens": n_tok,
            "call span length (median)": sorted(spans)[len(spans) // 2], "spans capped": sum(x > CAP for x in spans),
            "padded tokens": n_pad, "forward_s": round(fwd_s, 1), "load_s": round(load_s, 1), "tokens_per_s": round(n_tok / max(fwd_s, 1e-9)),
            "padded tokens_per_s (short contexts; not the long-run rate)": round(n_pad / max(fwd_s, 1e-9)),
            "gpu": torch.cuda.get_device_properties(0).name}


@app.local_entrypoint()
def main(sets: str):
    raw = (ROOT / sets).read_bytes()
    built = sorted((HERE / "results").glob("concept_sets_*.json"))
    if not any(json.loads(p.read_text())["sha256"] == hashlib.sha256(raw).hexdigest() for p in built):
        raise ValueError("the set file matches no recorded build")
    items = [json.loads(x) for x in raw.decode().splitlines()]
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=ROOT).stdout.strip())
    prov = {"git": git, "dirty": dirty, "sets": Path(sets).name, "sets_sha256": hashlib.sha256(raw).hexdigest()}
    out = extract.remote(items, stamp, prov)
    out = {"volume": VOL, "layers": LAYERS, "cap": CAP} | prov | out
    path = HERE / "results" / f"extract_concept_{stamp}.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    print(f"-> {path}")
