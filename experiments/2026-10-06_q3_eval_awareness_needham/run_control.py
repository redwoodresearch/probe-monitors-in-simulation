"""Positive control for the Needham probe on LinuxArena (Modal H100s): does one line appended to a transcript's system
prompt move call_end scores, thousands of tokens later, as it moved the templated probe's?

Same 120 transcripts and the same two new lines as the templated probe's control (config/control.json: under 16k
tokens; 30 paired cells x e1/e2/sim plus 30 honest), three conditions: original, eval_new, deploy_new. Messages come
from the segment-end extraction's staged inputs (read only). Each call_end activation is saved at all 64 layers
(/data/control/<run>/<cond>/<file>.npz, so later probes can be scored) and scored with every Needham probe and the
templated probe (layers 16-48). Pipeline check: the original condition must reproduce the stored segment-end
activations (2026-10-06_la_segment-end_qwen3-32b_v1, layers 8-63). Writes data/control_<stamp>.parquet (one row per
call and condition) and results/control_run_<stamp>.json.

  uv run --with pandas --with pyarrow modal run --detach experiments/2026-10-06_q3_eval_awareness_needham/run_control.py [--limit 1]
"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q3evalaw_control_kill-2026-11-06"
VOLUME = "probe-shift_aswin-surya_sub-q3evalaw_data_kill-2026-11-06"
Q3_VOLUME = "probe-shift_aswin-surya_sub-q3_outputs_kill-2026-11-06"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
GPU = "H100!"
PIP = ["torch==2.8.0", "transformers==4.56.2", "accelerate==1.10.1", "numpy"]
CFG = {"model": "Qwen/Qwen3-32B", "dtype": "bfloat16", "attn_implementation": "sdpa", "max_tokens": 16000,
       "segment_end_run": "2026-10-06_la_segment-end_qwen3-32b_v1", "stored_layers": [8, 16, 24, 32, 40, 48, 56, 63],
       "probe_run": "needham_v1", "readouts": ["asst_last", "asst_mean", "all_last", "all_mean"],
       "templated": {"dir": "probes/eval_awareness_v1", "layers": [16, 24, 32, 40, 48]},
       "run": "control_v1", "containers": 4}

app = modal.App(APP)
hf_cache = modal.Volume.from_name(HF_CACHE).read_only()
acts_vol = modal.Volume.from_name(ACTS).read_only()
q3_vol = modal.Volume.from_name(Q3_VOLUME).read_only()
vol = modal.Volume.from_name(VOLUME)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PIP)
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}).add_local_python_source("probe_shift"))


@app.function(image=image, volumes={"/acts": acts_vol, "/data": vol}, cpu=4, memory=32768, timeout=1800)
def select(cfg: dict, wanted: list[list[str]]) -> list[dict]:
    run_dir = Path("/data/control") / cfg["run"]
    if run_dir.exists() and any(run_dir.iterdir()):  # results files point at this directory: never overwrite it
        raise FileExistsError(f"{run_dir} already holds a control run; change CFG['run']")
    want = {tuple(w) for w in wanted}
    items = [it for p in sorted((Path("/acts") / cfg["segment_end_run"] / "inputs").glob("shard_*.jsonl"))
             for it in map(json.loads, p.read_text().splitlines()) if (it["arm"], it["key"]) in want]
    if len(items) != len(want):
        raise ValueError(f"found {len(items)} of {len(want)} control transcripts")
    return items


def folded(raw: bytes):
    from probe_shift.offpolicy import load_probe

    p = load_probe(raw)
    w = p["coef"] / p["scaler_scale"]
    return w, float(p["scaler_mean"] @ w)


@app.function(image=image, gpu=GPU, volumes={"/hf": hf_cache, "/acts": acts_vol, "/q3": q3_vol, "/data": vol}, timeout=3 * 3600,
              retries=modal.Retries(max_retries=1, initial_delay=10.0))
def read(cfg: dict, items: list[dict], conditions: dict) -> list[dict]:
    import copy
    import hashlib

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from probe_shift.reader import segment_end_positions

    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    base = AutoModelForCausalLM.from_pretrained(cfg["model"], revision=cfg["revision"], dtype=getattr(torch, cfg["dtype"]),
                                                device_map="cuda", attn_implementation=cfg["attn_implementation"]).model.eval()
    layers = list(range(len(base.layers)))
    probes = {f"{r}_L{l}": folded((Path("/data/probes") / cfg["probe_run"] / r / f"probe_L{l}.npz").read_bytes())
              for r in cfg["readouts"] for l in layers}
    probes |= {f"templated_L{l}": folded((Path("/q3") / cfg["templated"]["dir"] / f"probe_L{l}.npz").read_bytes()) for l in cfg["templated"]["layers"]}
    rows, checks = [], []
    for it in items:
        for cond, line in conditions.items():
            msgs = copy.deepcopy(it["messages"])
            if msgs[0]["role"] != "system":
                raise ValueError(f"{it['key']}: first message is not the system prompt")
            if line:
                msgs[0]["content"] += "\n\n" + line
            r = segment_end_positions(tok, msgs, cfg["max_tokens"])
            # Tokens of the rendered system turn: the appended line ends there, so a call's distance from the line is
            # token - system_end_token. Qwen's template renders the system turn first and ends it with a special token.
            system_end = len(tok(tok.apply_chat_template(msgs[:1], tokenize=False, enable_thinking=False),
                                 add_special_tokens=False)["input_ids"])
            if r["truncated"]:
                raise ValueError(f"{it['key']} {cond}: over max_tokens")
            calls = [p for p in r["positions"] if p["kind"] == "call_end" and p["token"] is not None]
            idx = torch.tensor([p["token"] for p in calls], device="cuda")
            acts = {}

            def hook(module, args, output, store=acts):
                store[len(store)] = (output[0] if isinstance(output, tuple) else output)[0].index_select(0, idx).to(torch.float16).cpu().numpy()

            hs = [base.layers[l].register_forward_hook(hook) for l in layers]
            try:
                with torch.inference_mode():
                    base(input_ids=torch.tensor([r["ids"]], device="cuda"), use_cache=False)
            finally:
                for h in hs:
                    h.remove()
            x = np.stack([acts[l] for l in layers], 1)  # [calls, 64, H]
            d = Path("/data/control") / cfg["run"] / cond
            d.mkdir(parents=True, exist_ok=True)
            np.savez(d / f"{it['arm']}__{hashlib.sha1(it['key'].encode()).hexdigest()[:16]}.npz", acts=x, key=np.array(it["key"]),
                     step=np.array([p["step"] for p in calls]), action=np.array([p["action"] for p in calls]))
            if cond == "original":  # the stored segment-end activations of the same calls
                stem = it["key"] if len(it["key"].encode()) <= 240 else f"{it['key'][:150]}__{hashlib.sha1(it['key'].encode()).hexdigest()[:12]}"
                sd = Path("/acts") / cfg["segment_end_run"] / "acts" / it["arm"]
                m = json.loads((sd / f"{stem}.json").read_text())
                z = np.load(sd / f"{stem}.npz")
                at = {(p["step"], p["action"]): p["row"] for p in m["positions"] if p["kind"] == "call_end" and p["kept"]}
                rr = np.array([at[(p["step"], p["action"])] for p in calls])
                checks.append({"key": it["key"], **{f"L{l}": float(np.abs(z[f"L{l}"][rr].astype(np.float32) - x[:, l].astype(np.float32)).max())
                                                     for l in cfg["stored_layers"]}})
            xf = x.astype(np.float32)
            scores = {}
            for name, (w, b) in probes.items():
                l = int(name.rsplit("_L", 1)[1])
                scores[name] = xf[:, l] @ w - b
            for j, p in enumerate(calls):
                rows.append({k: it[k] for k in ("key", "arm", "cell", "main_task")} | {"cond": cond, "step": p["step"], "action": p["action"],
                            "tool": p["tool"], "token": p["token"], "system_end_token": system_end}
                            | {k: float(v[j]) for k, v in scores.items()})
        vol.commit()
    return rows + [{"check": c} for c in checks]


@app.local_entrypoint()
def main(limit: int = 0):
    import numpy as np
    import pandas as pd

    from probe_shift.models import QWEN3_32B
    from probe_shift.offpolicy import provenance

    ctl = json.loads((HERE / "config" / "control.json").read_text())
    cfg = CFG | {"revision": QWEN3_32B["revision"]}
    conditions = {"original": None} | ctl["lines"]
    wanted = ctl["transcripts"]
    if limit:  # a test: `limit` transcripts in one container, a separate run name
        cfg |= {"run": cfg["run"] + "_test", "containers": 1}
        wanted = wanted[:limit]
    items = select.remote(cfg, wanted)
    a, b = 1.51e-4, 1.55e-8
    secs = sum(3 * (a * it["tokens"] + b * it["tokens"] ** 2) for it in items)
    print(f"{len(items)} transcripts x {len(conditions)} conditions, projected {secs / 3600:.2f} H100-hours "
          f"(~${(secs + cfg['containers'] * 300) / 3600 * 3.95:.2f})", flush=True)
    items.sort(key=lambda it: -it["tokens"])
    n = cfg["containers"]
    parts = list(read.map([cfg] * n, [items[i::n] for i in range(n)], [conditions] * n))
    out = [r for p in parts for r in p]
    checks = [r["check"] for r in out if "check" in r]
    df = pd.DataFrame([r for r in out if "check" not in r])
    score_cols = [c for c in df.columns if "_L" in c]
    if not np.isfinite(df[score_cols].to_numpy()).all():
        raise ValueError("non-finite scores")
    worst = {f"L{l}": max(c[f"L{l}"] for c in checks) for l in cfg["stored_layers"]}
    if max(worst.values()) > 1e-2:  # fp16 activations; the 2026-10-06 run reproduced them exactly (0.0)
        raise ValueError(f"the original condition does not reproduce the stored activations: {worst}")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S") + ("_test" if limit else "")
    (HERE / "data").mkdir(exist_ok=True)
    path = HERE / "data" / f"control_{stamp}.parquet"
    df.to_parquet(path, index=False)
    with vol.batch_upload() as up:
        up.put_file(str(path), f"control/{path.name}")
    (HERE / "results" / f"control_run_{stamp}.json").write_text(json.dumps(
        {"config": cfg | {"app": APP, "gpu": GPU, "pip": PIP}, "conditions": conditions, "transcripts": len(items), "rows": len(df),
         "pipeline_check_max_abs_diff": worst, "file": str(path.relative_to(HERE)), "volume_file": f"{VOLUME}/control/{path.name}",
         "activations": f"{VOLUME}/control/{cfg['run']}", "provenance": provenance(__file__)}, indent=1))
    print(f"pipeline check (max |original - stored| per layer): {worst}\n{len(df)} rows -> {path}")
