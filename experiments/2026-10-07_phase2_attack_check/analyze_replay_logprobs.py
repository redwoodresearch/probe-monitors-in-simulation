"""Does the replay reader see what the acting model saw? A cheap screen (D6(1), approved 2026-10-07): on the exact
tokens of real policy calls, compare the reader's teacher-forced bf16 logprobs with vLLM's decode-time logprobs.

The inputs come from the honest reruns' request log (request_log_proxy.py): each call's prompt and completion token
IDs, plus vLLM's raw logprob of every sampled token (log-softmax of the logits before temperature and top-k/top-p).
The reader is the Qwen3.5 reader's own setup (experiments/2026-10-06_qwen35_reader): the same pinned weights,
transformers 5.19 with the flash-linear-attention kernels for the Gated DeltaNet layers, bf16, sdpa attention. One
forward pass over prompt + completion per call; the log-softmax (fp32) of its logits at each completion token is
compared with vLLM's.

Calls: 6 ordinary completions (status 200, every token logged, under the 32,768 cap, so not the 3 degenerate submit
tails, and not the timeout checks), chosen nearest to prompt lengths of 5k, 15k, 30k, 50k and 70k tokens, plus the
longest. On Modal, one H100, reading both volumes directly (no laptop download):

  uv run --no-project --python 3.12 --with modal==1.2.6 modal run \\
      experiments/2026-10-07_phase2_attack_check/analyze_replay_logprobs.py --log requests_2026-10-07_211507.jsonl.gz

Reports per call, and for the long-context calls (prompt >= 50k) together: |delta| = |reader - vLLM| (mean, median,
p90, p99, max), the signed mean, and |delta| by position within the completion. Writes
results/replay_logprobs_<stamp>.json.
"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-phase2_replay-lp_kill-2026-11-06"  # Modal names: at most 64 characters
REQUEST_LOG = "probe-shift_aswin-surya_sub-phase2_request-log_kill-2026-11-06"
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
MODEL, REVISION = "Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654"  # src/probe_shift/models.py QWEN35_27B
CAP = 32768
TARGET_PROMPT_TOKENS = [5_000, 15_000, 30_000, 50_000, 70_000]
LONG = 50_000
BINS = [(0, 64), (64, 256), (256, 1024), (1024, None)]
PIP = ["torch==2.8.0", "transformers==5.19.0", "accelerate>=1.1.0", "numpy", "huggingface_hub",
       "flash-linear-attention==0.5.2"]  # the Qwen3.5 reader's image

image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PIP)
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}))
app = modal.App(APP)


def pick(records: list[dict]) -> list[dict]:
    """The calls to compare: ordinary, fully logged completions nearest to each target prompt length, plus the
    longest prompt, each call once."""
    ok = [r for r in records if r.get("status") == 200 and r.get("complete") and len(r.get("choices") or []) == 1
          and 0 < len(r["choices"][0]["token_ids"]) < CAP and len(r.get("prompt_token_ids") or []) >= 1000]
    chosen = []
    for target in TARGET_PROMPT_TOKENS:
        left = [r for r in ok if r not in chosen]
        chosen.append(min(left, key=lambda r: abs(len(r["prompt_token_ids"]) - target)))
    longest = max(ok, key=lambda r: len(r["prompt_token_ids"]))
    if longest not in chosen:
        chosen.append(longest)
    return sorted(chosen, key=lambda r: len(r["prompt_token_ids"]))


def summary(delta) -> dict:
    import numpy as np

    a = np.abs(delta)
    return {"n": int(a.size), "mean_abs": float(a.mean()), "median_abs": float(np.median(a)),
            "p90_abs": float(np.percentile(a, 90)), "p99_abs": float(np.percentile(a, 99)), "max_abs": float(a.max()),
            "mean_signed": float(delta.mean())}


@app.function(image=image, gpu="H100", memory=65536, timeout=3600,
              volumes={"/logs": modal.Volume.from_name(REQUEST_LOG), "/hf": modal.Volume.from_name(HF_CACHE)})
def compare(log_name: str) -> str:
    import gzip
    import time

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    with gzip.open(f"/logs/{log_name}", "rt") as f:
        records = [json.loads(line) for line in f]
    calls = pick(records)
    t0 = time.time()
    lm = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16, device_map="cuda",
                                              attn_implementation="sdpa").eval()
    out = {"model": MODEL, "revision": REVISION, "log": log_name, "reader": {"dtype": "bfloat16", "attn": "sdpa",
           "transformers": str(__import__("transformers").__version__), "torch": str(torch.__version__)},
           "load_seconds": round(time.time() - t0, 1), "calls": []}
    raw = {}  # unrounded deltas per call, for the combined summaries (the JSON keeps them rounded)
    for r in calls:
        prompt, completion = r["prompt_token_ids"], r["choices"][0]["token_ids"]
        vllm = np.array(r["choices"][0]["logprobs"], dtype=np.float64)
        row = {"request_id": r["request_id"], "prompt_tokens": len(prompt), "completion_tokens": len(completion)}
        try:
            torch.cuda.reset_peak_memory_stats()
            t1 = time.time()
            with torch.inference_mode():
                hidden = lm.model(input_ids=torch.tensor([prompt + completion], device="cuda"),
                                  use_cache=False).last_hidden_state[0]
                # The logits at position i predict token i + 1: the completion's tokens are predicted from
                # positions len(prompt) - 1 ... len(prompt) + len(completion) - 2.
                at = hidden[len(prompt) - 1: len(prompt) + len(completion) - 1]
                target = torch.tensor(completion, device="cuda")
                reader = torch.cat([
                    torch.log_softmax(lm.lm_head(at[i:i + 512]).float(), dim=-1)
                    .gather(1, target[i:i + 512, None])[:, 0] for i in range(0, len(completion), 512)]).cpu().numpy()
            delta = reader.astype(np.float64) - vllm
            raw[r["request_id"]] = delta
            row |= {"seconds": round(time.time() - t1, 1),
                    "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1), **summary(delta),
                    "by_completion_position": {f"{a}-{b or 'end'}": summary(delta[a:b]) for a, b in BINS
                                               if len(delta[a:b])},
                    "top_abs": sorted(((float(abs(d)), int(i)) for i, d in enumerate(delta)), reverse=True)[:5],
                    "delta": [round(float(d), 5) for d in delta]}
        except torch.OutOfMemoryError as e:  # recorded per call, so one long context can't hide the others
            row["error"] = f"OutOfMemoryError: {e}"[:300]
            torch.cuda.empty_cache()
        out["calls"].append(row)
        print({k: v for k, v in row.items() if k not in ("delta", "by_completion_position")}, flush=True)
    long = [raw[c["request_id"]] for c in out["calls"] if "delta" in c and c["prompt_tokens"] >= LONG]
    short = [raw[c["request_id"]] for c in out["calls"] if "delta" in c and c["prompt_tokens"] < LONG]
    out["long_context"] = summary(np.concatenate(long)) if long else None
    out["short_context"] = summary(np.concatenate(short)) if short else None
    out["failed_calls"] = sum("error" in c for c in out["calls"])
    # JSON text, not a dict: the local side has no torch or numpy, so nothing from them may cross back (a first run
    # returned torch.__version__, a torch object, and lost its result at deserialization).
    return json.dumps(out)


@app.local_entrypoint()
def main(log: str):
    result = json.loads(compare.remote(log))
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"replay_logprobs_{stamp}.json"
    with open(path, "x") as f:
        json.dump(result, f)
    print(json.dumps({"short_context": result["short_context"], "long_context": result["long_context"],
                      "failed_calls": result["failed_calls"], "load_seconds": result["load_seconds"]}, indent=1))
    print(f"-> {path}")
