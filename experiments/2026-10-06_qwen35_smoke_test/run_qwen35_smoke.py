"""Smoke test: can Qwen3.5-27B, loaded text-only, replace Qwen3-32B as the reader? One H100 on Modal.

Same 20 LinuxArena attack transcripts as the 2026-10-05 smoke test, read whole (no 32k cut). Per transcript:
render with Qwen3.5's chat template (thinking off), one forward pass of the language model, forward hooks on 8
decoder layers. Reports seconds and tokens/s, peak GPU memory, hook output shapes, non-finite values over the full
hidden states, and whether `probe_shift.reader` (written for Qwen3's template) still finds the segments. Qwen3.5's
template trims every message (`|trim`, so tool outputs lose leading and trailing whitespace, trailing newlines
included) and writes tool calls as XML (`<function=...><parameter=...>`), so the reader is expected to fail; a local
Qwen3.5 variant here reads, on trimmed tool outputs, the last content token of each output and of each call's last
parameter value.
Also reports whether the fast Gated DeltaNet kernel (flash-linear-attention) was used.

  uv run modal run experiments/2026-10-06_qwen35_smoke_test/run_qwen35_smoke.py::download   # weights only
  uv run modal run experiments/2026-10-06_qwen35_smoke_test/run_qwen35_smoke.py
"""

import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-qwen35-smoke_test"
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
GPU = "H100!"  # "!" pins H100, so timings project to H100s
TIMEOUT_S = 3600
PIP = ["torch==2.8.0", "transformers==5.19.0", "accelerate>=1.1.0", "numpy", "huggingface_hub",
       "flash-linear-attention==0.5.2"]

app = modal.App(APP)
hf_cache = modal.Volume.from_name(HF_CACHE, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(*PIP)
    .env({"HF_HOME": "/hf"})
    .add_local_python_source("probe_shift")
)


@app.function(image=image, volumes={"/hf": hf_cache}, cpu=8, memory=16384, timeout=TIMEOUT_S)
def download(model: str = "Qwen/Qwen3.5-27B") -> str:
    from huggingface_hub import snapshot_download

    path = snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "tokenizer*"])
    hf_cache.commit()
    return path


def inside(starts, ends, a, b):
    """Index of the last token lying entirely inside characters [a, b), or None."""
    import bisect

    i0, i1 = bisect.bisect_left(starts, a), bisect.bisect_right(ends, b)
    return i1 - 1 if i1 > i0 else None


def qwen35_positions(tok, messages) -> dict:
    """Segment-end positions under Qwen3.5's template: output_end (last token inside each tool output), call_end
    (last token inside each call's last parameter value; the function name if it has none), and the closing
    markers. Mirrors probe_shift.reader.segment_end_positions, which assumes Qwen3's JSON tool calls."""
    from probe_shift.reader import _layout

    text, found, enc = _layout(tok, messages)
    starts = [s for s, _ in enc["offset_mapping"]]
    ends = [e for _, e in enc["offset_mapping"]]
    names = tok.convert_ids_to_tokens(enc["input_ids"])
    calls_by_step = [m.get("tool_calls") or [] for m in messages if m["role"] == "assistant"]
    out = {"call_end": [], "call_marker": [], "output_end": [], "output_marker": [], "call_mismatch": 0}
    for kind, step, action, tool, a, b in found:
        if kind == "agent":
            j, n = a, 0
            while (k := text.find("<tool_call>\n<function=", j, b)) >= 0:
                close = text.index("\n</tool_call>", k)
                block = text[k:close]
                head = len("<tool_call>\n<function=")
                name = block[head:block.index(">\n", head)]
                expected = calls_by_step[step][n]["function"]["name"] if n < len(calls_by_step[step]) else None
                out["call_mismatch"] += name != expected
                v_end = block.rfind("\n</parameter>")
                if v_end >= 0:
                    v_start = block.index(">\n", block.rindex("<parameter=", 0, v_end)) + 2
                    t = inside(starts, ends, k + v_start, k + v_end)
                else:
                    t = inside(starts, ends, k + head, k + head + len(name))
                out["call_end"].append(t)
                m = bisect_marker(starts, names, text, "</tool_call>", close)
                out["call_marker"].append(m)
                j, n = close + len("\n</tool_call>"), n + 1
            out["call_mismatch"] += n != len(calls_by_step[step])
        else:
            out["output_end"].append(inside(starts, ends, a, b))
            out["output_marker"].append(bisect_marker(starts, names, text, "</tool_response>", b))
    out["token_texts"] = {k: [names[t] if t is not None else None for t in v][:3] for k, v in out.items() if isinstance(v, list)}
    return out


def bisect_marker(starts, names, text, literal, frm):
    import bisect

    i = bisect.bisect_left(starts, text.index(literal, frm))
    return i if names[i] == literal else None


@app.function(image=image, gpu=GPU, volumes={"/hf": hf_cache}, timeout=TIMEOUT_S)
def read_all(cfg: dict, items: list[dict]) -> dict:
    import importlib.metadata as md
    import logging

    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from probe_shift.reader import render, segment_end_positions

    class Capture(logging.Handler):  # transformers warns once per kernel that falls back to the PyTorch reference path
        def __init__(self):
            super().__init__()
            self.messages = []

        def emit(self, record):
            self.messages.append(record.getMessage())

    capture = Capture()
    logging.getLogger("transformers").addHandler(capture)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    lm = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=getattr(torch, cfg["dtype"]), device_map="cuda",
                                              attn_implementation=cfg["attn_implementation"]).eval()
    base = lm.model
    load_s = time.time() - t0
    layers = cfg["layers"]
    model_info = {"class": type(lm).__name__, "base_class": type(base).__name__, "n_layers": len(base.layers),
                  "hidden_size": lm.config.hidden_size, "layer_types": sorted(set(lm.config.layer_types)),
                  "has_visual": any("visual" in n for n, _ in lm.named_parameters()),
                  "params_b": round(sum(p.numel() for p in lm.parameters()) / 1e9, 2),
                  "weights_gb": round(torch.cuda.memory_allocated() / 2**30, 1)}
    versions = {p: md.version(p) for p in ("torch", "transformers", "flash-linear-attention", "fla-core", "triton")
                if _installed(md, p)}

    per_item = []
    for it in sorted(items, key=lambda x: x["qwen3_tokens"]):  # the long transcript last, so an OOM loses nothing else
        rec = {"id": it["id"], "arm": it["origin"], "environment": it["environment"]}
        text = tok.apply_chat_template(it["messages"], tokenize=False, enable_thinking=False)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        rec["tokens"], rec["qwen3_tokens"] = len(ids), it["qwen3_tokens"]
        # The template trims each message; with tool outputs trimmed up front the rendering must be the same.
        trimmed = [m | {"content": m["content"].strip()} if m["role"] == "tool" else m for m in it["messages"]]
        outs = [m["content"] for m in it["messages"] if m["role"] == "tool"]
        rec["tool_outputs_changed_by_trim"] = sum(o != o.strip() for o in outs)
        rec["tool_outputs_with_trailing_newline"] = sum(o.endswith("\n") for o in outs)
        rec["tool_outputs"] = len(outs)
        rec["trimmed_renders_identically"] = tok.apply_chat_template(trimmed, tokenize=False, enable_thinking=False) == text
        for name, msgs in (("as_read", it["messages"]), ("trimmed", trimmed)):
            try:
                _, segments = render(tok, msgs)
                rec[f"render_{name}"] = {"ok": True, "segments": len(segments)}
            except Exception as e:
                rec[f"render_{name}"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
            try:
                segment_end_positions(tok, msgs)
                rec[f"segment_end_positions_{name}"] = {"ok": True}
            except Exception as e:
                rec[f"segment_end_positions_{name}"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
        pos = qwen35_positions(tok, trimmed)
        rec["qwen35_positions"] = {k: (sum(t is not None for t in v), len(v)) for k, v in pos.items() if isinstance(v, list)}
        rec["qwen35_positions"]["call_mismatch"] = pos["call_mismatch"]
        rec["qwen35_token_texts"] = pos["token_texts"]
        rows = [t for k in ("call_end", "output_end") for t in pos[k] if t is not None]
        idx = torch.tensor(rows, device="cuda", dtype=torch.long)
        hooks_seen = {}

        def hook_for(layer):
            def hook(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                hooks_seen[layer] = {"shape": list(h.shape), "nonfinite": int((~torch.isfinite(h)).sum()),
                                     "max_abs": float(h.abs().max()), "rows": h[0].index_select(0, idx).float().cpu().numpy()}
            return hook

        handles = [base.layers[l].register_forward_hook(hook_for(l)) for l in layers]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t = time.time()
        try:
            with torch.inference_mode():
                base(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)
            torch.cuda.synchronize()
            seconds = time.time() - t
            rec |= {"seconds": round(seconds, 2), "tokens_per_s": round(len(ids) / seconds),
                    "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
                    "hooks": {l: {k: v for k, v in d.items() if k != "rows"} for l, d in hooks_seen.items()},
                    "nonfinite_hidden": sum(d["nonfinite"] for d in hooks_seen.values()),
                    "nonfinite_readout_rows": int(sum((~np.isfinite(d["rows"])).sum() for d in hooks_seen.values())),
                    "readout_rows": len(rows)}
        except torch.OutOfMemoryError as e:
            rec |= {"oom": True, "error": str(e)[:300], "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
        finally:
            for h in handles:
                h.remove()
        print({k: v for k, v in rec.items() if k not in ("hooks", "qwen35_token_texts")}, flush=True)
        per_item.append(rec)
    return {"load_seconds": round(load_s, 1), "device": torch.cuda.get_device_name(), "versions": versions,
            "model": model_info, "kernel_fallback_warnings": [m for m in capture.messages if "falling back" in m],
            "items": per_item}


def _installed(md, package):
    try:
        md.version(package)
        return True
    except md.PackageNotFoundError:
        return False


@app.local_entrypoint()
def main():
    import yaml

    from probe_shift.transcripts import load_sample, reader_messages, system_prompt

    cfg = yaml.safe_load((HERE / "config" / "qwen35_smoke.yaml").read_text())
    smoke = yaml.safe_load((HERE.parent.parent / cfg["cells_from"]).read_text())
    shared = json.loads((HERE.parent.parent / "splits" / "exclusions_v3.json").read_text())
    qwen3_tokens = {(r["file"], r["sample_id"]): r["tokens"] for r in shared["samples"]}
    items = []
    for c in smoke["cells"]:
        honest = load_sample(smoke["cohorts"]["honest"], **c["honest_prompt_from"])
        for origin in ("real", "sim"):
            ref = {k: c[origin][k] for k in ("file", "sample_id", "epoch")}
            sample = load_sample(smoke["cohorts"][origin], **ref)
            items.append({"id": f"{origin}__{c['environment']}", "origin": origin, "environment": c["environment"],
                          "cell": c["cell"], "qwen3_tokens": qwen3_tokens[(ref["file"], ref["sample_id"])],
                          "messages": reader_messages(sample, system_prompt(honest))})
    print(f"{len(items)} transcripts loaded; longest {max(i['qwen3_tokens'] for i in items)} Qwen3 tokens")

    result = read_all.remote(cfg, items)
    result["config"] = cfg | {"modal": {"app": APP, "gpu": GPU, "timeout_s": TIMEOUT_S, "pip": PIP, "hf_cache": HF_CACHE}}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    path = HERE / "results" / f"qwen35_smoke_{stamp}.json"
    path.write_text(json.dumps(result, indent=1))

    print(f"\nmodel load {result['load_seconds']}s on {result['device']}; {result['model']}; versions {result['versions']}")
    print("kernel fallbacks:", result["kernel_fallback_warnings"] or "none")
    for r in result["items"]:
        if r.get("oom"):
            print(f"  {r['id']:26} {r['tokens']:>6} tok  OOM at peak {r['peak_gb']} GB")
            continue
        shapes = sorted({tuple(h["shape"]) for h in r["hooks"].values()})
        print(f"  {r['id']:26} {r['tokens']:>6} tok (qwen3 {r['qwen3_tokens']:>6}) {r['seconds']:>6}s {r['tokens_per_s']:>6} tok/s "
              f"peak {r['peak_gb']} GB hooks {len(r['hooks'])} {shapes} nonfinite {r['nonfinite_hidden']} | render as read/trimmed "
              f"{r['render_as_read']['ok']}/{r['render_trimmed']['ok']} segment_end_positions {r['segment_end_positions_as_read']['ok']}/"
              f"{r['segment_end_positions_trimmed']['ok']} | trim changes {r['tool_outputs_changed_by_trim']}/{r['tool_outputs']} outputs "
              f"| qwen35 {r['qwen35_positions']}")
    print(f"\nresults -> {path}")
