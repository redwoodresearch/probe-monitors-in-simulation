"""Q4 (B): the epochs with an unpaired think tag (results/q4b_think_empty_check.json): is the cause the 32,768-token cut? Per such
epoch: its length, whether the last <think> comes after the last </think>, and how many tokens follow it. Counts only.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 modal run experiments/2026-10-13_q4_why_weaker_POSTHOC/check_q4b_unpaired.py
"""

import json
from pathlib import Path

import modal

APP = "probe-shift_aswin-surya_sub-q4own_unpaired_kill-2026-11-13"
VOL, HF = "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13", "probe-shift_aswin-surya_hf-cache"
app = modal.App(APP)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("transformers==5.19.0", "numpy", "jinja2")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}))


@app.function(image=image, cpu=4, memory=16384, timeout=900, retries=0,
              volumes={"/v": modal.Volume.from_name(VOL).read_only(), "/hf": modal.Volume.from_name(HF).read_only()})
def check() -> list:
    import numpy as np
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B", revision="fc05daec18b0a78c049392ed2e771dde82bdf654")
    T, Tc = tok.convert_tokens_to_ids("<think>"), tok.convert_tokens_to_ids("</think>")
    out = []
    for staged in ("q4b_main_v1", "q4b_run2_v1"):
        for f in sorted(Path(f"/v/{staged}/inputs").glob("*.jsonl")):
            for line in f.open():
                e = json.loads(line)
                ids = np.asarray(e["ids"])
                o, c = np.flatnonzero(ids == T), np.flatnonzero(ids == Tc)
                if len(o) == len(c) and not np.any(c < o):
                    continue
                out.append({"file": f"{staged}/{f.name}", "n_ids": int(len(ids)), "at_cut": bool(len(ids) == 32768), "opens": int(len(o)), "closes": int(len(c)),
                            "last_open_after_last_close": bool(len(o) and (not len(c) or o[-1] > c[-1])), "tokens_after_last_open": int(len(ids) - 1 - o[-1]) if len(o) else None})
    return out


@app.local_entrypoint()
def main():
    print(json.dumps(check.remote(), indent=1))
