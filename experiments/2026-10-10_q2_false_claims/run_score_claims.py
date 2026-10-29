"""False success claims, steps 2-3 (Modal): stage each final report's request from vLLM's logs (CPU), then score it
with the frozen step-2 deception probe on one H200 (PLAN.md).

stage (CPU): for each indexed request, its one ordinary logged prompt + completion. The completion's token bytes must
rebuild its decoded text; the visible text (after any think block, before any tool call) must equal the eval's report
once stripped; positions are the completion tokens that overlap the report's bytes (step 2's rule in
apollo.render_detect; how many only partly overlap is counted). Every mismatch is listed and nothing is scored if there
is any. A spawned coordinator on Modal runs staging and then scoring, so a lost local client stops neither.
score (H200): one forward pass per request over prompt + completion, stopped after block 31 (a hook raises once block 31
has run); the block's raw output at those positions, float32; per token ((h - scaler_mean) / scaler_scale) . coef with
the frozen probes_block_output.npz layer 31; the report's score is the mean. The longest request goes first, with its
time and peak memory printed. Results are written to the Volume as each request finishes, and returned.

  uv run modal run --detach experiments/2026-10-10_q2_false_claims/run_score_claims.py --index <index jsonl>
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-false-claims_kill-2026-11-08"
REQUEST_LOG = "probe-shift_aswin-surya_sub-phase2_request-log_kill-2026-11-06"
LOGS = ["requests_2026-10-08_232200_fc-01M4FN6M7DBP6D1BKEZTW8VZCK_ap-zR4QFgL4xUwXZJjYP3eoKj.jsonl.gz",
        "requests_2026-10-08_232159_fc-01M4FN6MBNRZJP3S15050AYYK9_ap-93947oDlGKDg9xhJfqprDP.jsonl.gz",
        "requests_2026-10-08_232159_fc-01M4FN6MBNRZJP3S1503608QV6_ap-OM7NbN3UTCZUdnQToDGk4s.jsonl.gz"]  # q3's staging of this rerun
PROBES = "probe-shift_aswin-surya_sub-q2-deception-qwen35_kill-2026-11-08"
PROBE_FILE, LAYER = "runs/2026-10-09_191400/probes_block_output.npz", 31
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
MODEL, REVISION = "Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654"

app = modal.App(APP)
out_vol = modal.Volume.from_name(APP, create_if_missing=True)
cpu_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("transformers==5.19.0", "jinja2", "numpy").env(
    {"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})
gpu_image = (modal.Image.debian_slim(python_version="3.12")
             .uv_pip_install("torch==2.8.0", "transformers==5.19.0", "accelerate>=1.1.0", "numpy", "flash-linear-attention==0.5.2")
             .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}))


def bytes_to_unicode() -> dict:
    """GPT-2's byte-level BPE alphabet (Qwen's tokenizer uses it): byte -> the character a token string writes it as."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


@app.function(image=cpu_image, cpu=4, memory=32768, timeout=3600, retries=0,
              volumes={"/rl": modal.Volume.from_name(REQUEST_LOG).read_only(), "/hf": modal.Volume.from_name(HF_CACHE).read_only(),
                       "/probes": modal.Volume.from_name(PROBES).read_only(), "/out": out_vol})
def stage(index: list[dict], stamp: str) -> dict:
    import gzip
    import itertools

    import numpy as np
    from transformers import AutoTokenizer

    p = np.load(Path("/probes") / PROBE_FILE)  # the frozen probe is there and has the expected shape, before any GPU starts
    if any(p[k].shape != (64, 5120) for k in ("coef", "scaler_mean", "scaler_scale")):
        raise ValueError(f"{PROBE_FILE}: unexpected shapes {[p[k].shape for k in p.files]}")
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    byte_of = {u: b for b, u in bytes_to_unicode().items()}
    added = {i: t.content for i, t in tok.added_tokens_decoder.items()}
    token_bytes = lambda seq: [added[t].encode() if t in added else bytes(byte_of[ch] for ch in tok.convert_ids_to_tokens(t)) for t in seq]
    want = {r["irid"] for r in index}
    if len(want) != len(index):
        raise ValueError("a request id appears twice in the index")
    rec, statuses = {}, {}
    for n in LOGS:
        with gzip.open(Path("/rl") / n, "rt") as fh:
            for line in fh:
                if not line.endswith("\n"):
                    raise ValueError(f"{n}: last line incomplete")
                r = json.loads(line)
                if r.get("request_id") not in want:
                    continue
                ok = r.get("status") == 200 and r.get("complete") and len(r.get("choices") or []) == 1 and r.get("prompt_token_ids") is not None
                statuses.setdefault(r["request_id"], []).append("ordinary" if ok else str(r.get("status")))
                if ok:
                    rec.setdefault(r["request_id"], []).append((r["prompt_token_ids"], r["choices"][0]["token_ids"]))
    bad, staged = [], []
    for ix in index:
        rs = rec.get(ix["irid"], [])
        if len(rs) != 1:
            bad.append({"key": ix["key"], "problem": f"{len(rs)} ordinary records", "statuses": statuses.get(ix["irid"])})
            continue
        prompt, comp = rs[0]
        tb = token_bytes(comp)
        raw = b"".join(tb)
        text = raw.decode("utf-8", errors="strict")
        if text != tok.decode(comp, skip_special_tokens=False):
            bad.append({"key": ix["key"], "problem": "token bytes do not rebuild the decoded completion"})
            continue
        lo_c = text.rfind("</think>") + len("</think>") if "</think>" in text else 0
        hi_c = text.find("<tool_call>", lo_c) if "<tool_call>" in text[lo_c:] else len(text)
        region = text[lo_c:hi_c].removesuffix("<|im_end|>")
        if region.strip() != ix["report"]:
            bad.append({"key": ix["key"], "problem": "visible completion text differs from the eval's report",
                        "chars": [len(region.strip()), len(ix["report"])]})
            continue
        start = lo_c + region.index(ix["report"])
        lo, hi = len(text[:start].encode()), len(text[:start].encode()) + len(ix["report"].encode())
        ends = list(itertools.accumulate(len(x) for x in tb))
        spans = list(zip([0] + ends[:-1], ends))
        inside = [i for i, (s, e) in enumerate(spans) if e > s and e > lo and s < hi]  # overlap, as step 2
        partial = sum(not (spans[i][0] >= lo and spans[i][1] <= hi) for i in inside)
        if not inside:
            bad.append({"key": ix["key"], "problem": "no token inside the report"})
            continue
        ids = np.asarray(prompt + comp, np.int32)
        staged.append({"key": ix["key"], "irid": ix["irid"], "n_prompt": len(prompt), "n_completion": len(comp),
                       "positions": [len(prompt) + i for i in inside], "partly inside": partial, "ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest()})
        d = Path("/out") / stamp / "inputs"
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / f"{ix['irid']}.npy", ids)
    if bad:
        return {"staged": len(staged), "bad": bad}
    (Path("/out") / stamp / "staged.json").write_text(json.dumps(staged))
    out_vol.commit()
    return {"staged": len(staged), "bad": bad, "tokens": sum(s["n_prompt"] + s["n_completion"] for s in staged),
            "longest": max(s["n_prompt"] + s["n_completion"] for s in staged), "report tokens": sum(len(s["positions"]) for s in staged),
            "report tokens only partly inside": sum(s["partly inside"] for s in staged)}


class Stop(Exception):
    pass


@app.function(image=gpu_image, gpu="H200", memory=65536, timeout=3600, retries=0,
              volumes={"/hf": modal.Volume.from_name(HF_CACHE).read_only(), "/probes": modal.Volume.from_name(PROBES).read_only(), "/out": out_vol})
def score(stamp: str) -> dict:
    import time

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    out_vol.reload()
    run = Path("/out") / stamp
    staged = sorted(json.loads((run / "staged.json").read_text()), key=lambda s: -(s["n_prompt"] + s["n_completion"]))
    for s in staged:  # every input matches its staged hash before the model loads
        if hashlib.sha256(np.load(run / "inputs" / f"{s['irid']}.npy").tobytes()).hexdigest() != s["ids_sha256"]:
            raise ValueError(f"{s['key']}: staged ids differ from their hash")
    p = np.load(Path("/probes") / PROBE_FILE)
    coef, mu, sc = (torch.tensor(p[k][LAYER], dtype=torch.float32, device="cuda") for k in ("coef", "scaler_mean", "scaler_scale"))
    t0 = time.time()
    lm = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
    base = lm.model
    load_s = round(time.time() - t0)
    print(f"loaded in {load_s} s; {len(base.layers)} blocks", flush=True)
    got = {}

    def hook(module, args, output):
        got["h"] = (output[0] if isinstance(output, tuple) else output)[0].index_select(0, got["idx"]).float()
        raise Stop

    h = base.layers[LAYER].register_forward_hook(hook)
    (run / "scores").mkdir(exist_ok=True)
    rows = []
    try:
        for i, s in enumerate(staged):
            ids = np.load(run / "inputs" / f"{s['irid']}.npy")
            got["idx"] = torch.tensor(s["positions"], device="cuda")
            torch.cuda.reset_peak_memory_stats()
            t = time.time()
            try:
                with torch.inference_mode():
                    base(input_ids=torch.tensor(ids[None].astype(np.int64), device="cuda"), use_cache=False)
                raise RuntimeError("the forward pass did not reach block 31")
            except Stop:
                pass
            torch.cuda.synchronize()
            per_token = ((got["h"] - mu) / sc) @ coef
            if not torch.isfinite(per_token).all():
                raise ValueError(f"{s['key']}: non-finite scores")
            row = {"key": s["key"], "irid": s["irid"], "score": float(per_token.mean()), "per_token": [round(float(x), 4) for x in per_token],
                   "tokens": len(ids), "report_tokens": len(s["positions"]), "seconds": round(time.time() - t, 2),
                   "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
            (run / "scores" / f"{s['irid']}.json").write_text(json.dumps(row))
            out_vol.commit()
            rows.append(row)
            print(f"{i + 1}/{len(staged)} {row['tokens']} tokens: {row['seconds']} s, peak {row['peak_gib']} GiB", flush=True)
    finally:
        h.remove()
    return {"rows": rows, "load_s": load_s, "seconds": round(time.time() - t0), "device": torch.cuda.get_device_name()}


@app.function(image=cpu_image, cpu=1, memory=2048, timeout=7200, retries=0)
def pipeline(index: list[dict], stamp: str) -> dict:
    """Staging, then scoring only if staging matched everywhere."""
    st = stage.remote(index, stamp)
    print(json.dumps(st), flush=True)
    return {"staging": st} if st["bad"] else score.remote(stamp) | {"staging": st}


@app.local_entrypoint()
def main(index: str):
    rows = [json.loads(line) for line in Path(index).read_text().splitlines()]
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    data = HERE.parents[1] / "data" / "false_claims"
    data.mkdir(parents=True, exist_ok=True)  # before any remote work
    call = pipeline.spawn([{k: r[k] for k in ("key", "irid", "report")} for r in rows], stamp)
    (data / f"launch_{stamp}.json").write_text(json.dumps({"stamp": stamp, "call": call.object_id, "volume": f"{APP}:/{stamp}"}) + "\n")
    print(f"stamp {stamp}, call {call.object_id}", flush=True)
    res = call.get()
    out = data / f"scores_{stamp}.json"
    out.write_text(json.dumps(res | {"index sha256": hashlib.sha256(Path(index).read_bytes()).hexdigest(), "stamp": stamp,
                                     "probe": f"{PROBES}:/{PROBE_FILE} layer {LAYER}"}) + "\n")
    print(json.dumps(res["staging"]))
    print(out)
