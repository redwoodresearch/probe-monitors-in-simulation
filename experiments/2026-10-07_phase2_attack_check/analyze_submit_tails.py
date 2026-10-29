"""What did Qwen3.5 generate after its final `submit` call? In all 3 honest reruns the submit turn ran to the
32,768-token cap, while the eval log kept only submit({}) and a short reasoning block: vLLM's parsers dropped the rest.
The vLLM proxy's request log holds the real completion tokens. On Modal (CPU; the request log and the HF cache are
volumes), no laptop download:

  uv run --no-project --python 3.12 --with modal==1.2.6 modal run \\
      experiments/2026-10-07_phase2_attack_check/analyze_submit_tails.py --log requests_2026-10-07_211507.jsonl.gz

For every record whose completion reached the cap: the special tokens in it and where they sit; the text around the
first </tool_call> and at the end; the shortest period the last 4,000 tokens repeat with; and the mean logprob of the
tail. Also what the pinned chat template renders after an assistant tool call, and which tokens end generation
(generation_config.json's eos ids). Writes results/submit_tails_<stamp>.json.
"""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-phase2_submit-tails_kill-2026-11-06"
REQUEST_LOG = "probe-shift_aswin-surya_sub-phase2_request-log_kill-2026-11-06"
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
MODEL, REVISION = "Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654"
CAP = 32768
SPECIAL = ["<tool_call>", "</tool_call>", "<|im_end|>", "<|im_start|>", "<|endoftext|>", "<think>", "</think>",
           "<function=submit>", "<tool_response>"]

image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("tokenizers==0.22.1")
app = modal.App(APP)


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None  # None, never 0.0, for an empty slice: 0 would mean probability 1


def period(ids: list[int], max_period: int = 2000) -> int | None:
    """The smallest p with ids[i] == ids[i - p] for every i in the window, or None."""
    for p in range(1, min(max_period, len(ids) // 2) + 1):
        if all(ids[i] == ids[i - p] for i in range(p, len(ids))):
            return p
    return None


@app.function(image=image, cpu=1, memory=8192, timeout=900,  # single-threaded
              volumes={"/logs": modal.Volume.from_name(REQUEST_LOG), "/hf": modal.Volume.from_name(HF_CACHE)})
def tails(log_name: str) -> dict:
    import gzip

    from tokenizers import Tokenizer

    snap = Path(f"/hf/hub/models--{MODEL.replace('/', '--')}/snapshots/{REVISION}")
    tok = Tokenizer.from_file(str(snap / "tokenizer.json"))
    special = {s: tok.token_to_id(s) for s in SPECIAL}
    gen_cfg = json.loads((snap / "generation_config.json").read_text())
    template_file = snap / "chat_template.jinja"
    template = (template_file.read_text() if template_file.exists()
                else json.loads((snap / "tokenizer_config.json").read_text()).get("chat_template", ""))
    at = template.find("tool_call")
    eos = gen_cfg.get("eos_token_id")
    eos = eos if isinstance(eos, list) else [eos] if eos is not None else []
    out = {"model": MODEL, "revision": REVISION, "log": log_name, "special_token_ids": special,
           "generation_config_eos_token_id": eos, "eos_tokens": [tok.id_to_token(i) for i in eos],
           "template_around_first_tool_call": template[max(0, at - 300):at + 900], "tails": []}
    with gzip.open(f"/logs/{log_name}", "rt") as f:
        records = [json.loads(line) for line in f]
    for r in records:
        for c in r.get("choices") or []:
            ids, lps = c.get("token_ids") or [], c.get("logprobs") or []
            if len(ids) < CAP:
                continue
            first_close = next((i for i, t in enumerate(ids) if t == special["</tool_call>"]), None)
            positions = {s: [i for i, t in enumerate(ids) if t == tid][:20] for s, tid in special.items()
                         if tid is not None}
            counts = {s: sum(t == tid for t in ids) for s, tid in special.items() if tid is not None}
            after = first_close + 1 if first_close is not None else 0
            last = ids[-4000:]
            out["tails"].append({
                "request_id": r.get("request_id"), "finish_reason": c.get("finish_reason"), "tokens": len(ids),
                "prompt_tokens": len(r.get("prompt_token_ids") or []), "special_counts": counts,
                "special_first_positions": positions, "first_tool_call_close": first_close,
                "head_through_first_close": tok.decode(ids[:after], skip_special_tokens=False)[-1500:],
                "after_first_close_400_tokens": tok.decode(ids[after:after + 400], skip_special_tokens=False),
                "last_300_tokens": tok.decode(ids[-300:], skip_special_tokens=False),
                "distinct_tokens_after_close": len(set(ids[after:])),
                "top_tokens_after_close": [(n, tok.id_to_token(t)) for t, n in Counter(ids[after:]).most_common(15)],
                "period_of_last_4000": period(last),
                "logprobs_aligned": len(lps) == len(ids),
                "mean_logprob_after_close": mean(lps[after:]) if len(lps) == len(ids) else None,
                "mean_logprob_before_close": mean(lps[:after]) if len(lps) == len(ids) else None,
            })
    # For a repetition guard (vLLM's repetition_detection): the longest run of one token repeated back to back, in every
    # completion, so a threshold can be set above anything a normal turn does.
    runs = []
    for r in records:
        for c in r.get("choices") or []:
            ids = c.get("token_ids")
            if not ids:  # a logging gap (the proxy marks it complete=False): unknown, never a run of 0
                runs.append({"request_id": r.get("request_id"), "tokens": None, "longest_single_token_run": None})
                continue
            best = cur = 0
            for i, t in enumerate(ids):
                cur = cur + 1 if i and t == ids[i - 1] else 1
                best = max(best, cur)
            runs.append({"request_id": r.get("request_id"), "tokens": len(ids), "longest_single_token_run": best})
    out["longest_runs"] = sorted(runs, key=lambda x: -(x["longest_single_token_run"] or 0))
    out["completions_without_token_ids"] = sum(x["tokens"] is None for x in runs)
    return out


@app.local_entrypoint()
def main(log: str):
    result = tails.remote(log)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"submit_tails_{stamp}.json"
    with open(path, "x") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({"eos": result["eos_tokens"], "tails": [
        {k: t[k] for k in ("request_id", "tokens", "special_counts", "first_tool_call_close", "period_of_last_4000",
                           "distinct_tokens_after_close", "mean_logprob_after_close")} for t in result["tails"]]},
        indent=1))
    top = [(x["longest_single_token_run"], x["tokens"]) for x in result["longest_runs"][:8]]
    print("longest single-token runs:", top, "of", len(result["longest_runs"]), "completions;",
          result["completions_without_token_ids"], "without token IDs")
    print(f"-> {path}")
