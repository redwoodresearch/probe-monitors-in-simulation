# /// script
# requires-python = "==3.12.*"
# dependencies = ["torch==2.8.0", "transformers==5.19.0", "accelerate>=1.1.0", "flash-linear-attention==0.5.2", "numpy",
#                 "zstandard"]
# ///
"""The activation rows the probes read: Qwen3.5-27B reads one agent run, and we keep its residual stream at each unit.

  uv run pipeline/extract_activations.py --eval RUN.eval --layers 48 63 --out rows.npz
  uv run pipeline/extract_activations.py --eval ATTACK_RUN.eval --system-prompt-from HONEST_RUN_OF_THE_SAME_TASK.eval ...

Needs one GPU with 80 GB: the model is about 54 GB in bf16, and on an H100 a run of 38,508 tokens peaked at 58 GiB and
took 14 seconds (longer the first time, while the linear-attention kernels tune). The longest runs need a larger GPU
(we used H200s).

What it does, for one run (an Inspect .eval log):
  1. Writes the run out as the reader reads it and finds its units (units.py: which units, and at which tokens).
  2. Runs Qwen3.5-27B (bf16, PyTorch's scaled-dot-product attention) once over the whole run, with no cut.
  3. At each requested layer L, takes the residual stream after decoder block L: the output of model.model.layers[L], read
     with a forward hook. Layers count from 0 to 63. For L up to 62 this equals entry L + 1 of Hugging Face's
     `hidden_states`; for L = 63 (the off-policy probe's layer for commands) it does not: the last `hidden_states` entry
     is taken after the model's final norm, and our rows, like the stored ones, are taken before it.
  4. Keeps one row of 5,120 numbers per unit: the activation at the unit's token for the two command kinds, the mean over
     the output's tokens for a tool output (taken in float32, then rounded to bf16, the model's own precision). Rows are
     stored as fp16.

This is a short rewrite of the code that made the activations behind the post (comments marked "original:" point to it,
in our research repository at commit 378d4c7b5). It differs in arithmetic only: the original took each tool output's
mean as a difference of float32 running sums over the whole run, and ran on H200 GPUs. On five honest runs, this script
reproduced the stored rows of all 143 commands bit for bit at every layer the post's probes read, and the 124 tool-output
means to within 0.1% (the largest difference relative to the row's length; cosine similarity at least 0.9999995).
"""

import argparse

import numpy as np

import units

MODEL, REVISION = "Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654"
PAD_TO = 4096  # right padding to a multiple of this: the linear-attention kernels then tune once per length bucket, and in this
# causal model tokens after the run change none of its positions


def load_reader():
    """Qwen3.5-27B on the GPU, and its tokenizer with our one template change."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # original: experiments/2026-10-14_q3_overnight_chain/extract_chain.py lines 157-161
    tokenizer = units.reader_template(AutoTokenizer.from_pretrained(MODEL, revision=REVISION))
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="sdpa").eval()
    return tokenizer, model


def read_rows(model, ids: list[int], spans: list[tuple[int, int]], layers: list[int], pad_id: int) -> dict[int, np.ndarray]:
    """One forward pass over the whole run. For each layer, one fp16 row per span [start, end): the mean of the residual
    stream over the span's tokens (for a one-token span, that token's activation)."""
    import torch

    # original: experiments/2026-10-14_q3_overnight_chain/extract_chain.py lines 164-188 (hooks, span means) and 231 (padding)
    rows = {}

    def keep(layer):
        def hook(module, inputs, output):
            h = (output[0] if isinstance(output, tuple) else output)[0]  # [tokens, 5120], bf16
            means = [h[a:b].float().mean(0) for a, b in spans]
            rows[layer] = (torch.stack(means).to(torch.bfloat16).to(torch.float16).cpu().numpy() if means
                           else np.zeros((0, h.shape[1]), np.float16))
        return hook

    handles = [model.model.layers[layer].register_forward_hook(keep(layer)) for layer in layers]
    try:
        with torch.inference_mode():
            model.model(input_ids=torch.tensor([ids + [pad_id] * (-len(ids) % PAD_TO)], device=model.device), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    bad = sum(int((~np.isfinite(x)).sum()) for x in rows.values())
    if bad:
        raise ValueError(f"{bad} activations are not finite in fp16")
    return rows


def extract(tokenizer, model, sample: dict, system_prompt: str, layers: list[int]) -> tuple[list[dict], dict[int, np.ndarray]]:
    """The units of one run and, per layer, their rows (row i belongs to unit i)."""
    ids, found = units.run_units(tokenizer, sample, system_prompt)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    return found, read_rows(model, ids, [(u["start"], u["end"]) for u in found], layers, pad_id)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--eval", required=True, help="the run's Inspect .eval log")
    ap.add_argument("--system-prompt-from", help="for an attack run: the .eval log of the honest run of the same task")
    ap.add_argument("--layers", type=int, nargs="+", default=[48, 63])
    ap.add_argument("--out", required=True, help="output .npz: rows_L<layer> [units, 5120] fp16, and the units' fields")
    args = ap.parse_args()
    sample = units.read_sample(args.eval)
    prompt = units.system_prompt(units.read_sample(args.system_prompt_from) if args.system_prompt_from else sample)
    tokenizer, model = load_reader()
    found, rows = extract(tokenizer, model, sample, prompt, args.layers)
    fields = {"kind": np.array([u["kind"] for u in found], dtype=str), "tool": np.array([u["tool"] for u in found], dtype=str),
              "argument": np.array([u.get("argument") or "" for u in found], dtype=str),
              **{k: np.array([u[k] for u in found], dtype=np.int64) for k in ("turn", "call", "start", "end")},
              "skipped_by_trained_probes": np.array([bool(u.get("skipped_by_trained_probes")) for u in found])}
    np.savez(args.out, **{f"rows_L{layer}": x for layer, x in rows.items()}, **fields)
    counts = ", ".join(f"{k} {int((fields['kind'] == k).sum())}" for k in ("call_end", "command", "tool_output"))
    print(f"{len(found)} units ({counts}) at layers {args.layers} -> {args.out}")


if __name__ == "__main__":
    main()
