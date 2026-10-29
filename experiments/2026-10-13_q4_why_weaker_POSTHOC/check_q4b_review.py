"""Q4 (B) review checks (Codex threads on the Q4 PRs), counts and differences only, no text:
- unpaired: in the 4 epochs with an unpaired think tag, the non-whitespace tokens inside every paired block and inside the
  unclosed final block (Qwen3.5-27B's vocabulary decides whitespace);
- means: for a fixed sample of staged epochs (the first 2 by key in run 1, honest A and simulated attack), re-read Qwen3.5-27B
  and compare the as-run output_end_mean rows (float32 mean cast to bfloat16, then fp16) with float32 means cast straight to fp16,
  at all 64 layers: max abs difference, max per-row relative L2 difference; and the last-token rows (should be identical).

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 --with numpy modal run --detach \\
      experiments/2026-10-13_q4_why_weaker_POSTHOC/check_q4b_review.py
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q4own_review_kill-2026-11-13"
VOL, HF = "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13", "probe-shift_aswin-surya_hf-cache"
QWEN35 = ("Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654")
PIP = ["torch==2.8.0", "transformers==5.19.0", "accelerate>=1.1.0", "numpy", "huggingface_hub", "flash-linear-attention==0.5.2", "jinja2"]
app = modal.App(APP)
vol = modal.Volume.from_name(VOL, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PIP).env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})
MOUNTS = {"/v": vol.read_only(), "/hf": modal.Volume.from_name(HF).read_only()}


def filename(key: str) -> str:  # extract_q4b.filename
    return hashlib.sha1(key.encode()).hexdigest()[:20]


@app.function(image=image, cpu=4, memory=16384, timeout=900, retries=0, volumes=MOUNTS)
def unpaired() -> list:
    import numpy as np
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(*QWEN35[:1], revision=QWEN35[1])
    T, Tc = tok.convert_tokens_to_ids("<think>"), tok.convert_tokens_to_ids("</think>")
    ws = np.array([tok.decode([i]).strip() == "" for i in range(len(tok))])
    out = []
    for staged in ("q4b_main_v1", "q4b_run2_v1"):
        for f in sorted(Path(f"/v/{staged}/inputs").glob("*.jsonl")):
            for line in f.open():
                ids = np.asarray(json.loads(line)["ids"])
                o, c = np.flatnonzero(ids == T), np.flatnonzero(ids == Tc)
                if len(o) == len(c) and not np.any(c < o):
                    continue
                pairs = list(zip(o[: len(c)], c))
                out.append({"file": f"{staged}/{f.name}", "paired_blocks": len(pairs),
                            "paired_blocks_with_text": int(sum(bool((~ws[ids[a + 1:b]]).any()) for a, b in pairs)),
                            "unclosed_block_tokens": int(len(ids) - 1 - o[-1]), "unclosed_block_nonwhitespace_tokens": int((~ws[ids[o[-1] + 1:]]).sum())})
    return out


@app.function(image=image, gpu="H200", memory=65536, timeout=1800, retries=0, volumes=MOUNTS)
def means() -> dict:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    sample = []
    for arm in ("attack", "honest_a", "sim_attack@september"):
        eps = sorted((json.loads(l) for l in Path(f"/v/q4b_main_v1/inputs/{arm}.jsonl").open()), key=lambda e: e["key"])[:2]
        sample += eps
    lm = AutoModelForCausalLM.from_pretrained(QWEN35[0], revision=QWEN35[1], dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
    base, layers = lm.model, list(range(lm.config.num_hidden_layers))
    worst = {"mean_max_abs": 0.0, "mean_max_rel_l2": 0.0, "last_max_abs": 0.0, "rows": 0}
    per_epoch = []
    for e in sample:
        m = json.loads(Path(f"/v/q4b_extract_main_v1/meta/{e['arm']}/{filename(e['key'])}.json").read_text())
        calls = [c for c in m["calls"] if c["kind"] == "output_end" and c.get("row_mean") is not None]
        idx = torch.tensor([c["readout"] for c in calls], device="cuda")
        sp = torch.tensor([c["span"] for c in calls], device="cuda")
        got = {}

        def grab(l, out):
            h = (out[0] if isinstance(out, tuple) else out)[0]
            cs = torch.cumsum(h.float(), 0)
            a, b = sp[:, 0], sp[:, 1]
            lo = torch.where((a > 0).unsqueeze(1), cs.index_select(0, (a - 1).clamp(min=0)), torch.zeros_like(cs[:1]))
            mean32 = (cs.index_select(0, b - 1) - lo) / (b - a).unsqueeze(1).float()
            got[l] = (mean32.to(torch.float16).cpu().numpy(), h.index_select(0, idx).to(torch.float16).cpu().numpy())

        hooks = [base.layers[l].register_forward_hook(lambda mod, a, out, l=l: grab(l, out)) for l in layers]
        with torch.inference_mode():
            base(input_ids=torch.tensor([e["ids"]], device="cuda"), use_cache=False)
        for h in hooks:
            h.remove()
        rec = {"key": e["key"], "outputs": len(calls), "mean_max_abs": 0.0, "mean_max_rel_l2": 0.0, "last_max_abs": 0.0}
        for l in layers:
            X = np.load(f"/v/q4b_extract_main_v1/by_layer/L{l}/{m['chunk']}.npy", mmap_mode="r")
            asrun_mean = np.asarray(X[[m["row_start"] + c["row_mean"] for c in calls]], np.float32)
            asrun_last = np.asarray(X[[m["row_start"] + c["row"] for c in calls]], np.float32)
            f32, last = got[l][0].astype(np.float32), got[l][1].astype(np.float32)
            d = np.abs(asrun_mean - f32)
            rec["mean_max_abs"] = max(rec["mean_max_abs"], float(d.max(initial=0)))
            rec["mean_max_rel_l2"] = max(rec["mean_max_rel_l2"], float((np.linalg.norm(asrun_mean - f32, axis=1) / np.maximum(np.linalg.norm(f32, axis=1), 1e-12)).max(initial=0)))
            rec["last_max_abs"] = max(rec["last_max_abs"], float(np.abs(asrun_last - last).max(initial=0)))
        per_epoch.append(rec)
        for k in ("mean_max_abs", "mean_max_rel_l2", "last_max_abs"):
            worst[k] = max(worst[k], rec[k])
        worst["rows"] += len(calls) * len(layers)
    return {"worst": worst, "per_epoch": per_epoch}


@app.local_entrypoint()
def main():
    u = unpaired.spawn()
    res = {"means": means.remote(), "unpaired": u.get()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dest = HERE / "results" / f"q4b_review_checks_{stamp}.json"
    dest.open("x").write(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print(dest.name)
