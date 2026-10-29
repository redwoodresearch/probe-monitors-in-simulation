"""E1 confirmation, the long-context equivalence check (PLAN.md here, committed with this script before it runs): one H200
container scores the selected confirmation sequences in one uncached pass, exactly `run_confirm.score()`'s own check
branch (same model, revision, layer, frozen probe and staged positions), reading the confirmation's stage folder
read-only. Only the one-pass per-token scores (numbers) leave the container; the comparison is analyze_equiv_check.py.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run \\
      experiments/2026-10-12_q2_e1_equiv_check/run_equiv_check.py --staged <archived staged.json>
"""
import hashlib
import json
import random
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-e1-equiv_kill-2026-11-12"
CONFIRM = "probe-shift_aswin-surya_sub-q2-e1-confirm_kill-2026-11-08"  # the confirmation's Volume, mounted read-only
STAMP = "2026-10-11_164551_confirm"
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
PROBES = "probe-shift_aswin-surya_sub-q2-deception-qwen35_kill-2026-11-08"
PROBE_FILE, PROBE_SHA, LAYER = "runs/2026-10-09_191400/probes_block_output.npz", "c4cba16e0eb74925a946a46d17271cb8e8ccc3df2a28e652b9bd8a2d3bce74bd", 31
MODEL, REVISION = "Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654"
STAGED_SHA = "2fa0706be1e8664f"  # prefix of staged.json's sha256, as confirm_masks_2026-10-11_185020.json records
N_LONGEST, N_RANDOM, SEED = 3, 5, 20261012
app = modal.App(APP)
gpu_image = (modal.Image.debian_slim(python_version="3.12")
             .uv_pip_install("torch==2.8.0", "transformers==5.19.0", "accelerate>=1.1.0", "numpy", "flash-linear-attention==0.5.2")
             .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}))


def select(staged: list[dict]) -> list[str]:
    """The 3 runs with the longest sequences, then 5 others drawn by seed (PLAN.md)."""
    by = {}
    for s in staged:
        by.setdefault(s["key"], []).append(s["n"])
    longest = sorted(by, key=lambda k: (-max(by[k]), k))[:N_LONGEST]
    rest = sorted(k for k in by if k not in longest)
    return longest + random.Random(SEED).sample(rest, N_RANDOM)


@app.function(image=gpu_image, gpu="H200", memory=65536, timeout=3600, startup_timeout=600, scaledown_window=60, retries=0,
              volumes={"/hf": modal.Volume.from_name(HF_CACHE).read_only(), "/out": modal.Volume.from_name(CONFIRM).read_only(),
                       "/probes": modal.Volume.from_name(PROBES).read_only()})
def one_pass(keys: list[str]) -> dict:
    import time

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    run = Path("/out") / STAMP
    if hashlib.sha256((run / "staged.json").read_bytes()).hexdigest()[:16] != STAGED_SHA:
        raise ValueError("staged.json differs from the one the masks were made on")
    if hashlib.sha256((Path("/probes") / PROBE_FILE).read_bytes()).hexdigest() != PROBE_SHA:  # in the scoring container itself
        raise ValueError("probe file differs from its recorded sha256")
    staged = [s for s in json.loads((run / "staged.json").read_text()) if s["key"] in set(keys)]
    p = np.load(Path("/probes") / PROBE_FILE)
    coef, mu, sc = (torch.tensor(p[k][LAYER], dtype=torch.float32, device="cuda") for k in ("coef", "scaler_mean", "scaler_scale"))
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval().model
    base.config.num_hidden_layers = LAYER + 1
    load_s = round(time.time() - t0)
    got = {}
    h = base.layers[LAYER].register_forward_hook(
        lambda m, a, out: got.__setitem__("h", (out[0] if isinstance(out, tuple) else out)[0].index_select(0, got["idx"]).float()))
    rows = []
    try:
        for s in sorted(staged, key=lambda s: (-s["n"], s["key"], s["seq"])):
            ids = np.load(run / "inputs" / f"{s['key']}__{s['seq']}.npy")
            if hashlib.sha256(ids.tobytes()).hexdigest() != s["ids_sha256"]:
                raise ValueError(f"{s['key']} {s['seq']}: staged ids differ from their hash")
            torch.cuda.reset_peak_memory_stats(); t = time.time()
            got["idx"] = torch.tensor(s["positions"], device="cuda")
            failed = False
            try:
                with torch.inference_mode():
                    base(input_ids=torch.tensor(ids[None].astype(np.int64), device="cuda"), use_cache=False)
            except torch.OutOfMemoryError:
                failed = True
            if failed:  # cleaned up outside the handler; recorded as unchecked, never scored as 0
                got.pop("h", None); torch.cuda.empty_cache()
                rows.append({"key": s["key"], "seq": s["seq"], "tokens": s["n"], "oom": True})
                continue
            full = ((got["h"] - mu) / sc) @ coef
            if not torch.isfinite(full).all():
                raise ValueError(f"{s['key']} {s['seq']}: non-finite one-pass scores")
            rows.append({"key": s["key"], "seq": s["seq"], "ids_sha256": s["ids_sha256"], "tokens": s["n"], "one_pass_score": float(full.mean()),
                         "one_pass_per_token": [float(x) for x in full], "seconds": round(time.time() - t, 2),
                         "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1)})
            print(f"{len(rows)}/{len(staged)} {s['key']} {s['seq']}: {s['n']} tokens in {rows[-1]['seconds']} s", flush=True)
    finally:
        h.remove()
    return {"load_s": load_s, "rows": rows}


@app.local_entrypoint()
def main(staged: str):
    for f in (Path(__file__).resolve(), HERE / "PLAN.md"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    sp = Path(staged)
    if hashlib.sha256(sp.read_bytes()).hexdigest()[:16] != STAGED_SHA:
        raise SystemExit("the archived staged.json is not the one the masks were made on")
    keys = select(json.loads(sp.read_text()))
    res = one_pass.remote(keys)
    out = {"plan": "PLAN.md (this folder)", "git": head, "stage folder": f"{CONFIRM}:{STAMP}", "staged.json sha256 prefix": STAGED_SHA,
           "selected runs (3 longest, then 5 by seed)": keys} | res
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    d = HERE.parents[1] / "data" / "e1_equiv_check"; d.mkdir(parents=True, exist_ok=True)
    with open(dest := d / f"one_pass_{stamp}.json", "x") as fh:  # numbers only, but per token: data/, then S3 with a manifest
        fh.write(json.dumps(out) + "\n")
    print(dest)
