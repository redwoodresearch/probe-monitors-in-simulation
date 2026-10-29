"""Is the bf16 replay precise enough for a frozen probe? D6(2): on the exact tokens of real policy calls, compare a
frozen Qwen3.5 probe's score read from the bf16 replay with the same probe read from an fp32 replay (the reference).

The probe is Stage 1's impactful-command control (experiments/2026-10-07_stage1_awareness on aswin/q3-stage1-awareness,
results/freeze_2026-10-07_234608.json): layer 48, C 0.001, `score = x @ impact__w + impact__b` with x the residual stream
after decoder block 48 at the call's last command-content token (reader.command_param: bash -> cmd, text_editor create ->
file_text, str_replace -> new_str, ...), + = impactful. Its weights are read on Modal from the Stage 1 volume.

The bf16 replay is Stage 1's extraction setup (run_extract_stage1.py): the pinned weights, transformers 5.19 with the
flash-linear-attention kernels, bf16, sdpa, right-padded to a power-of-two bucket, the hidden state stored as fp16 and
scored in fp32. The reference is the same weights cast to fp32 (layers 49-63 and the LM head dropped: block 48's output
doesn't depend on them), no padding, PyTorch's TF32 off, and sdpa limited to its memory-efficient kernel (never the
math kernel, which would materialize the full attention matrix). The flash-linear-attention Triton kernels' fp32 dots
default to TF32, which PyTorch's flags don't reach, and Triton's in-memory kernel cache ignores a precision change, so
the reference runs in its own container with Triton's default set to IEEE before anything compiles, and with the
one dot precision flash-linear-attention 0.5.2 hard-codes to TF32 on H200 (the Gated DeltaNet forward's triangular
solve, SOLVE_TRIL_DOT_PRECISION) set to IEEE too. The bf16 container
also reads the shortest calls in fp32 at Triton's default (TF32): a difference from the IEEE reads shows the setting
takes effect. The reference shares the replay's tokens and implementation, so it checks numerics only; D6(1)
(analyze_replay_logprobs.py) checked the implementation against vLLM's own logprobs.

Calls, from each request log given: ordinary completions (status 200, every token logged, under the 32,768 cap, ended
on a tool call) whose bash or text_editor call has a command value, 16 at evenly spaced quantiles of prompt length plus
the longest. Every such call in a completion is one readout, at the last token whose decoded span lies inside the
parameter's value. The honest reruns' log gives honest contexts, step 2's gives attack contexts (simulated
environment). On Modal, one H200, reading the request logs, the HF cache and the Stage 1 volume directly (no laptop
download):

  uv run --no-project --python 3.12 --with modal==1.2.6 --with numpy modal run \\
      experiments/2026-10-07_phase2_attack_check/analyze_replay_probe.py \\
      --logs requests_2026-10-07_211507.jsonl.gz,requests_2026-10-07_235557.jsonl.gz

Reports per readout the two scores and delta = bf16 - fp32, the hidden states' relative error and cosine, and |delta|
summaries for short (< 50k prompt tokens) and long contexts and per log. Investigate if any |delta| > 0.04 logit (the
coordinator's threshold). Two H200 containers run at once and stream one message per call as it finishes
(results/partial/replay_probe_<stamp>.jsonl, without the hidden vectors); no call starts unless it should end before its
container's timeout, so a slow kernel costs calls, not the run.
Writes results/replay_probe_<stamp>.json.
"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-phase2_replay-probe_kill-2026-11-06"
REQUEST_LOG = "probe-shift_aswin-surya_sub-phase2_request-log_kill-2026-11-06"
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
STAGE1 = "probe-shift_aswin-surya_sub-stage1_acts_kill-2026-11-06"
PROBE = "2026-10-07_stage1_qwen35-27b_v1/probes/L48.npz"  # keys impact__w, impact__b
LAYER, KEY = 48, "impact"
MODEL, REVISION = "Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654"  # src/probe_shift/models.py QWEN35_27B
CAP, N_CALLS, LONG, THRESHOLD = 32768, 16, 50_000, 0.04  # N_CALLS per log, plus its longest
TIMEOUT_S = {"bf16": 1200, "ieee": 3000}  # per container; no call starts unless it should end RESERVE_S before
RESERVE_S, PRIOR_S_PER_TOKEN, TF32_CALLS = 120, 0.003, 2
PIP = ["torch==2.8.0", "transformers==5.19.0", "accelerate>=1.1.0", "numpy", "huggingface_hub",
       "flash-linear-attention==0.5.2"]  # the Qwen3.5 reader's image

image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PIP)
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}))
app = modal.App(APP)


def command_param(name: str, arguments: dict) -> str | None:
    """src/probe_shift/reader.py command_param (Stage 1 branch), for bash and text_editor only."""
    if name == "text_editor":
        by_command = {"create": ("file_text",), "str_replace": ("new_str",), "insert": ("insert_text", "new_str"),
                      "view": ("path",), "undo_edit": ("path",)}
        order = by_command.get(arguments.get("command"), ("file_text", "new_str", "insert_text", "path"))
    elif name == "bash":
        order = ("cmd", "command")
    else:
        return None
    key = next((k for k in order if k in arguments), None)
    return key if key is not None and arguments[key].strip() else None


def readouts(tokenizer, completion: list[int]) -> list[dict]:
    """Each bash or text_editor call's readout token in the completion: the last token whose decoded span ends inside
    the command value (qwen3_coder's `<parameter=NAME>\\nVALUE\\n</parameter>`, the value written raw)."""
    import re

    text = tokenizer.decode(completion, skip_special_tokens=False)

    def end(i: int) -> int:  # token i's end offset in the decoded text (a split UTF-8 character ends at the next one)
        return len(tokenizer.decode(completion[: i + 1], skip_special_tokens=False).rstrip("\ufffd"))

    found = []
    for call in re.finditer(r"<tool_call>\n<function=([^>\n]+)>\n(.*?)</function>\n</tool_call>", text, re.S):
        params = {m.group(1): (m.start(2) + call.start(2), m.group(2)) for m in
                  re.finditer(r"<parameter=([^>\n]+)>\n(.*?)\n</parameter>", call.group(2), re.S)}
        name = call.group(1)
        param = command_param(name, {k: v for k, (_, v) in params.items()})
        if param is None:
            continue
        start, value = params[param]
        stop = start + len(value)
        lo, hi = -1, len(completion) - 1  # the last token ending at or before the value's end (end() is monotone)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            lo, hi = (mid, hi) if end(mid) <= stop else (lo, mid - 1)
        if lo < 0 or end(lo) <= start:
            continue
        found.append({"tool": name, "param": param, "token": lo, "value_chars": len(value),
                      "token_text": tokenizer.decode([completion[lo]])})
    return found


def pick(records: list[dict], tokenizer) -> list[dict]:
    """Ordinary calls with at least one readout: the longest, then N_CALLS more at evenly spaced prompt-length
    quantiles of the other calls (so a log gives N_CALLS + 1 whenever it has that many with readouts)."""
    ok = [r for r in records if r.get("status") == 200 and r.get("complete") and len(r.get("choices") or []) == 1
          and r["choices"][0].get("finish_reason") == "tool_calls" and 0 < len(r["choices"][0]["token_ids"]) < CAP
          and len(r.get("prompt_token_ids") or []) >= 1000]
    ok.sort(key=lambda r: len(r["prompt_token_ids"]))
    longest = None
    for r in reversed(ok):  # reserved first, so a quantile pick can't take its place
        if found := readouts(tokenizer, r["choices"][0]["token_ids"]):
            longest = r | {"readouts": found}
            break
    rest = [r for r in ok if not longest or r["request_id"] != longest["request_id"]]  # quantiles over these
    chosen, seen = [], set()
    for q in ((k + 0.5) / N_CALLS for k in range(N_CALLS)):
        for r in rest[min(int(q * len(rest)), len(rest) - 1):]:  # the first call at or after the quantile with a readout
            if r["request_id"] in seen:
                continue
            found = readouts(tokenizer, r["choices"][0]["token_ids"])
            if found:
                chosen.append(r | {"readouts": found})
                seen.add(r["request_id"])
                break
    chosen += [longest] if longest else []
    return chosen, len(ok)


def summary(delta) -> dict:
    import numpy as np

    a = np.abs(np.asarray(delta, dtype=np.float64))
    return {"n": int(a.size), "mean_abs": float(a.mean()), "median_abs": float(np.median(a)),
            "p90_abs": float(np.percentile(a, 90)), "max_abs": float(a.max()),
            "mean_signed": float(np.mean(delta)), "over_threshold": int((a > THRESHOLD).sum())}


@app.function(image=image, gpu="H200", memory=32768, timeout=max(TIMEOUT_S.values()),
              volumes={"/logs": modal.Volume.from_name(REQUEST_LOG), "/hf": modal.Volume.from_name(HF_CACHE),
                       "/stage1": modal.Volume.from_name(STAGE1)})
def replay(logs: list[str], mode: str, only: list[str] | None = None):
    """One container's reads, streamed as JSON text: a header, then one message per call as it finishes (so a timeout
    keeps what's done). mode "bf16": Stage 1's setup on every call, then the same weights in fp32 with Triton's default
    fp32 dot precision (TF32) on the TF32_CALLS shortest. mode "ieee": the fp32 reference on every call, with Triton's
    fp32 dot default set to IEEE before anything compiles (Triton's in-memory kernel cache ignores the setting, so it is
    never changed inside a process; the TF32 reads above, from a separate process, show whether it takes effect).
    `only`: just these request IDs of the calls picked (a rerun of calls an earlier run skipped)."""
    import os

    if mode == "ieee":
        os.environ["TRITON_F32_DEFAULT"] = "ieee"
    import base64
    import gzip
    import time

    import numpy as np
    import torch
    import triton
    from torch.nn.attention import SDPBackend, sdpa_kernel
    import triton.language as tl
    import importlib

    chunk_fwd = importlib.import_module("fla.ops.gated_delta_rule.chunk_fwd")  # the Gated DeltaNet forward kernels
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if mode == "ieee":  # fla hard-codes TF32 for this solve's dots on GPUs that have it, whatever Triton's default
        chunk_fwd.SOLVE_TRIL_DOT_PRECISION = tl.constexpr("ieee")
    deadline = time.time() + TIMEOUT_S[mode] - RESERVE_S
    z = np.load(f"/stage1/{PROBE}")
    w, b = z[f"{KEY}__w"].astype(np.float32), float(z[f"{KEY}__b"])
    assert np.isfinite(w).all() and np.isfinite(b), "non-finite probe weights"
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    calls, n_ordinary = [], {}
    for log in logs:
        with gzip.open(f"/logs/{log}", "rt") as f:
            records = [json.loads(line) for line in f]
        chosen, n_ordinary[log] = pick(records, tok)
        assert len(chosen) >= N_CALLS // 2, f"{log}: only {len(chosen)} calls with a readout of {n_ordinary[log]}"
        calls += [c | {"log": log} for c in chosen]
    if only:
        assert set(only) <= {c["request_id"] for c in calls}, f"not among the calls picked: {set(only)}"
        calls = [c for c in calls if c["request_id"] in only]
    t0 = time.time()
    lm = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16, device_map="cuda",
                                              attn_implementation="sdpa").eval()
    base = lm.model
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    knobs = getattr(getattr(triton, "knobs", None), "language", None)

    class Stop(Exception):
        pass

    def layer48(ids: list[int], positions: list[int], padded: bool) -> np.ndarray:
        """Block 48's output at `positions`, as float32 [len(positions), 5120]; the forward stops after block 48."""
        n = max(256, 1 << (len(ids) - 1).bit_length()) if padded else len(ids)
        got = {}

        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            got["h"] = h[0, positions].float().cpu().numpy() if not padded else \
                h[0, positions].to(torch.float16).float().cpu().numpy()  # Stage 1 stores fp16, scores in fp32
            raise Stop

        handle = base.layers[LAYER].register_forward_hook(hook)
        try:
            with torch.inference_mode():
                base(input_ids=torch.tensor([ids + [pad] * (n - len(ids))], device="cuda"), use_cache=False)
        except Stop:
            pass
        finally:
            handle.remove()
        assert np.isfinite(got["h"]).all(), "non-finite hidden state"
        return got["h"]

    def to_fp32() -> None:
        """Layers 49-63 and the LM head dropped (the forward stops at block 48), the rest cast to fp32, PyTorch's TF32
        off. Grouped-query attention reaches sdpa as enable_gqa, which its memory-efficient kernel may not take;
        expanding the key and value heads first (repeat_kv, a copy) is the same arithmetic."""
        import transformers.integrations.sdpa_attention as sdpa_attention

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        del base.layers[LAYER + 1:]
        lm.lm_head = None
        torch.cuda.empty_cache()
        base.float()
        sdpa_attention.use_gqa_in_sdpa = lambda *args, **kwargs: False

    def read_fp32(ids, positions) -> tuple[np.ndarray, str]:
        try:
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                return layer48(ids, positions, padded=False), "efficient"
        except RuntimeError as e:  # no fp32 kernel for this shape: the math kernel only where it fits
            if "No available kernel" not in str(e) or len(ids) > 8192:
                raise
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                return layer48(ids, positions, padded=False), "efficient or math"

    yield json.dumps({"header": {
        "mode": mode, "model": MODEL, "revision": REVISION, "logs": logs, "ordinary_calls": n_ordinary,
        "calls": [c["request_id"] for c in calls], "triton_f32_default": getattr(knobs, "fp32_default", None),
        "fla_solve_tril_precision": str(chunk_fwd.SOLVE_TRIL_DOT_PRECISION),
        "probe": {"volume": STAGE1, "path": PROBE, "key": KEY, "layer": LAYER, "w_norm": float(np.linalg.norm(w)),
                  "b": b},
        "versions": {"transformers": str(__import__("transformers").__version__), "torch": str(torch.__version__),
                     "triton": str(triton.__version__)},
        "load_seconds": round(time.time() - t0, 1)}})

    def run(read, kind: str, todo: list[dict]):
        """Read `todo` in order, streaming one message per call; a call starts only if 1.5 times the median seconds
        per token so far says it ends before the deadline. The first call is left out of that median (it pays the
        kernel compiles: 2.7k tokens took longer than 6.6k in the first run), and a prior stands in until there is one."""
        rates = []
        for c in todo:
            prompt = c["prompt_token_ids"]
            ids = prompt + c["choices"][0]["token_ids"]
            positions = [len(prompt) + x["token"] for x in c["readouts"]]
            out = {"kind": kind, "log": c["log"], "request_id": c["request_id"], "prompt_tokens": len(prompt),
                   "completion_tokens": len(c["choices"][0]["token_ids"]), "readouts": c["readouts"]}
            if time.time() + 1.5 * (float(np.median(rates[1:])) if rates[1:] else PRIOR_S_PER_TOKEN) * len(ids) \
                    > deadline:
                out["error"] = "skipped: would pass the deadline"
            else:
                try:
                    torch.cuda.reset_peak_memory_stats()
                    t1 = time.time()
                    h, kernel = read(ids, positions)
                    rates.append((time.time() - t1) / len(ids))
                    score = h @ w + b
                    assert np.isfinite(score).all(), "non-finite score"
                    out |= {"seconds": round(time.time() - t1, 1), "attention": kernel,
                            "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1),
                            "score": [float(x) for x in score],
                            "h_b64": [base64.b64encode(x.astype(np.float32).tobytes()).decode() for x in h]}
                except (torch.OutOfMemoryError, RuntimeError, AssertionError) as e:  # per call: one can't hide the rest
                    out["error"] = f"{type(e).__name__}: {e}"[:400]
                    torch.cuda.empty_cache()
            print(kind, {k: v for k, v in out.items() if k not in ("readouts", "h_b64")}, flush=True)
            yield json.dumps({"call": out}, allow_nan=False)  # JSON text: nothing from torch or numpy crosses back

    by_length = sorted(calls, key=lambda c: len(c["prompt_token_ids"]))
    if mode == "bf16":
        yield from run(lambda ids, pos: (layer48(ids, pos, padded=True), "sdpa default"), "bf16", calls)
        to_fp32()
        yield from run(read_fp32, "fp32_tf32", by_length[:TF32_CALLS])
    else:
        to_fp32()
        # Shortest and longest alternately, so a deadline cut still leaves both short and long contexts.
        n = len(by_length)
        order = [by_length[i] for k in range((n + 1) // 2) for i in dict.fromkeys((k, n - 1 - k))]
        yield from run(read_fp32, "fp32_ieee", order)


def merge(messages: list[dict]) -> dict:
    """Per call: the bf16 and fp32 (IEEE) scores, delta = bf16 - fp32, the hidden states' relative error and cosine;
    the TF32 check on the shortest calls; |delta| summaries by context length and log."""
    import base64

    import numpy as np

    reads = {}
    for m in messages:
        if "call" in m:
            reads.setdefault(m["call"]["request_id"], {})[m["call"]["kind"]] = m["call"]
    vec = lambda r: [np.frombuffer(base64.b64decode(x), dtype=np.float32) for x in r["h_b64"]]  # noqa: E731
    calls, tf32 = [], []
    for rid, r in reads.items():
        row = {k: r["bf16"][k] for k in ("log", "request_id", "prompt_tokens", "completion_tokens", "readouts")} \
            if "bf16" in r else {"request_id": rid}
        ok = {k for k, v in r.items() if "error" not in v}
        row["errors"] = {k: v["error"] for k, v in r.items() if "error" in v}
        if {"bf16", "fp32_ieee"} <= ok:
            hb, h32 = vec(r["bf16"]), vec(r["fp32_ieee"])
            row |= {"score_bf16": r["bf16"]["score"], "score_fp32": r["fp32_ieee"]["score"],
                    "delta": [x - y for x, y in zip(r["bf16"]["score"], r["fp32_ieee"]["score"])],
                    "rel_err": [float(np.linalg.norm(x - y) / np.linalg.norm(y)) for x, y in zip(hb, h32)],
                    "cosine": [float(x @ y / np.linalg.norm(x) / np.linalg.norm(y)) for x, y in zip(hb, h32)],
                    "seconds": {k: r[k]["seconds"] for k in ok}, "peak_gib": {k: r[k]["peak_gib"] for k in ok}}
        if {"fp32_tf32", "fp32_ieee"} <= ok:
            ht, hi = vec(r["fp32_tf32"]), vec(r["fp32_ieee"])
            tf32.append({"request_id": rid, "tokens": r["fp32_ieee"]["prompt_tokens"],
                         "max_abs_hidden_diff": float(max(np.abs(x - y).max() for x, y in zip(ht, hi))),
                         "score_tf32_minus_ieee": [x - y for x, y in zip(r["fp32_tf32"]["score"],
                                                                          r["fp32_ieee"]["score"])]})
        calls.append(row)
    done = [c for c in calls if "delta" in c]
    parts = {"short_context": lambda c: c["prompt_tokens"] < LONG, "long_context": lambda c: c["prompt_tokens"] >= LONG,
             "all": lambda c: True} | {f"log {log}": (lambda c, log=log: c["log"] == log)
                                       for log in dict.fromkeys(c["log"] for c in done)}
    out = {name: summary([d for c in done if keep(c) for d in c["delta"]]) if any(keep(c) for c in done) else None
           for name, keep in parts.items()}
    scores = np.array([s for c in done for s in c["score_fp32"]])
    out |= {"fp32_score_sd": float(scores.std()) if scores.size > 1 else None,
            "compared_calls": len(done), "calls_not_compared": len(calls) - len(done),
            "tf32_check": tf32, "ieee_took_effect": any(t["max_abs_hidden_diff"] > 0 for t in tf32) if tf32 else None}
    return {"summary": out, "calls": sorted(calls, key=lambda c: c.get("prompt_tokens", 0))}


@app.local_entrypoint()
def main(logs: str, only: str = ""):
    import threading

    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    partial = HERE / "results" / "partial" / f"replay_probe_{stamp}.jsonl"  # every message as it arrives, no vectors
    partial.parent.mkdir(exist_ok=True)
    messages, lock, failures = [], threading.Lock(), []

    def consume(mode: str) -> None:
        try:
            for text in replay.remote_gen(logs.split(","), mode, only.split(",") if only else None):
                message = json.loads(text)
                with lock:
                    messages.append(message)
                    slim = {k: ({kk: vv for kk, vv in v.items() if kk != "h_b64"} if k == "call" else v)
                            for k, v in message.items()}
                    f.write(json.dumps({"mode": mode} | slim) + "\n")
                    f.flush()
                print(mode, text[:300], flush=True)
        except Exception as e:  # noqa: BLE001 — recorded, then the run fails loudly after both containers end
            failures.append(f"{mode}: {e!r}")

    with open(partial, "x") as f:
        threads = [threading.Thread(target=consume, args=(m,)) for m in TIMEOUT_S]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    headers = {m["header"]["mode"]: m["header"] for m in messages if "header" in m}
    assert len({tuple(h["calls"]) for h in headers.values()}) <= 1, "the containers picked different calls"
    result = {"headers": headers, "container_failures": failures} | merge(messages)
    path = HERE / "results" / f"replay_probe_{stamp}.json"
    with open(path, "x") as f:
        json.dump(result, f, indent=1, allow_nan=False)
    print(json.dumps(result["summary"], indent=1))
    print(f"-> {path} (messages as received: {partial})")
    if failures:
        raise SystemExit(f"container failures: {failures}")
