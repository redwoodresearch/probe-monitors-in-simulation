"""The honest-prompt swap diagnostic (D1 of the attack-detector plan): 5 tasks' attack runs replayed with the honest
system prompt swapped in. Modal CPU, tokenizer only; about $0.05.

From the staged 7-arm inputs (rerun_strict_segends_v2): 5 of the 20 tasks, drawn with numpy's default_rng(20261009); for
every staged epoch of the attack arm on those tasks, the system turn (ids from the start through the first
<|im_end|>) is replaced by honest_b's system turn for the same task (from its first staged epoch). Every row's readout
and n move by the length difference and its sha256 is recomputed on the swapped ids, so run_extract_replay.py's gate
checks the swapped inputs for consistency. The policy's own tokens after the system turn are unchanged; the context is
no longer the one Qwen had (the point of the diagnostic). Writes <stage>/inputs/attack.jsonl on the activations Volume.

  uv run --no-project --python 3.12 --with 'modal>=1.2' modal run experiments/2026-10-08_q3_onpolicy_figures/run_swap_stage.py
"""

import hashlib
import json
from pathlib import Path

import modal

APP = "probe-shift_aswin-surya_sub-q3onpol_readouts_kill-2026-11-08"
ACTS = "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08"
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
MODEL, REVISION = "Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654"
SRC, DST, N_TASKS, SEED = "rerun_strict_segends_v2", "rerun_strict_segends_swap_v1", 5, 20261009

app = modal.App(APP)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("transformers==5.19.0", "jinja2", "numpy")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}))


@app.function(image=image, cpu=4, memory=32768, timeout=1800, retries=0,
              volumes={"/acts": modal.Volume.from_name(ACTS, version=2), "/hf": modal.Volume.from_name(HF_CACHE).read_only()})
def swap() -> dict:
    import numpy as np
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    src = Path("/acts") / SRC / "inputs"
    att = [json.loads(x) for x in (src / "attack.jsonl").read_text().splitlines()]
    hon = [json.loads(x) for x in (src / "honest_b.jsonl").read_text().splitlines()]
    tasks = sorted({e["task"] for e in att})
    pick = sorted(np.random.default_rng(SEED).choice(tasks, N_TASKS, replace=False).tolist())
    first_h = {}
    for e in sorted(hon, key=lambda e: e["epoch"]):
        first_h.setdefault(e["task"], e)
    out, info = [], []
    for e in att:
        if e["task"] not in pick:
            continue
        ids, hids = e["ids"], first_h[e["task"]]["ids"]
        s_old, s_new = ids.index(im_end) + 1, hids.index(im_end) + 1  # the system turn, through its <|im_end|>
        new = hids[:s_new] + ids[s_old:]
        delta = s_new - s_old
        calls = []
        for c in e["calls"]:
            n = c["n"] + delta
            calls.append(c | {"n": n, "readout": None if c["readout"] is None else c["readout"] + delta,
                              "sha256": hashlib.sha256(np.asarray(new[:n], np.int32).tobytes()).hexdigest()})
            if c["readout"] is not None and not s_new <= c["readout"] + delta < n:
                raise ValueError(f"{e['key']}: a readout falls inside the swapped system turn")
        out.append(e | {"ids": new, "calls": calls, "swap": {"system_tokens_attack": s_old, "system_tokens_honest_b": s_new,
                                                              "honest_b_key": first_h[e["task"]]["key"]}})
        info.append({"key": e["key"], "task": e["task"], "tokens": len(new), "delta": delta})
    d = Path("/acts") / DST / "inputs"
    if d.exists():
        raise FileExistsError(f"{d} exists; stage under a new name")
    d.mkdir(parents=True)
    b = "".join(json.dumps(e) + "\n" for e in out).encode()
    (d / "attack.jsonl").write_bytes(b)
    modal.Volume.from_name(ACTS, version=2).commit()
    return {"tasks": pick, "epochs": len(out), "tokens": sum(x["tokens"] for x in info), "epochs_detail": info,
            "staged_inputs_sha256": {"attack.jsonl": hashlib.sha256(b).hexdigest()}}


@app.local_entrypoint()
def main():
    print(json.dumps(swap.remote(), indent=1))
