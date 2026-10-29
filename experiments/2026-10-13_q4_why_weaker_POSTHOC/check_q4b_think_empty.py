"""Q4 (B), Codex's required fix to the staging gate: over EVERY staged epoch of B (q4b_main_v1, q4b_run2_v1), count think blocks
holding any non-whitespace token (by Qwen3.5-27B's vocabulary: a token is whitespace if it decodes to only whitespace), unpaired
think tags, <tools> blocks in the system turn and submit rows. Counts only, no text. One Modal CPU container.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 modal run experiments/2026-10-13_q4_why_weaker_POSTHOC/check_q4b_think_empty.py
"""

import json
from pathlib import Path

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q4own_thinkcheck_kill-2026-11-13"
VOL, HF = "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13", "probe-shift_aswin-surya_hf-cache"
QWEN35 = ("Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654")
app = modal.App(APP)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("transformers==5.19.0", "numpy", "jinja2")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}))


@app.function(image=image, cpu=8, memory=32768, timeout=1800, retries=0,
              volumes={"/v": modal.Volume.from_name(VOL).read_only(), "/hf": modal.Volume.from_name(HF).read_only()})
def check() -> dict:
    import numpy as np
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(QWEN35[0], revision=QWEN35[1])
    T, Tc, S = (tok.convert_tokens_to_ids(s) for s in ("<think>", "</think>", "<|im_start|>"))
    ws = np.array([tok.decode([i]).strip() == "" for i in range(len(tok))])
    tools_seq = tok.encode("<tools>", add_special_tokens=False)
    out = {}
    for staged in ("q4b_main_v1", "q4b_run2_v1"):
        for f in sorted(Path(f"/v/{staged}/inputs").glob("*.jsonl")):
            rec = {"epochs": 0, "think_blocks": 0, "think_blocks_with_text": 0, "epochs_with_unpaired_tags": 0, "epochs_with_tools_block": 0, "submit_rows": 0}
            for line in f.open():
                e = json.loads(line)
                ids = np.asarray(e["ids"])
                o, c = np.flatnonzero(ids == T), np.flatnonzero(ids == Tc)
                rec["epochs"] += 1
                if len(o) != len(c) or np.any(c < o):
                    rec["epochs_with_unpaired_tags"] += 1
                else:
                    rec["think_blocks"] += len(o)
                    rec["think_blocks_with_text"] += int(sum(bool((~ws[ids[a + 1:b]]).any()) for a, b in zip(o, c)))
                st = np.flatnonzero(ids == S)
                sys_end = st[1] if len(st) > 1 else min(len(ids), 6000)
                rec["epochs_with_tools_block"] += int(any(np.array_equal(ids[i:i + len(tools_seq)], tools_seq) for i in range(sys_end)))
                rec["submit_rows"] += sum(1 for k in e["calls"] if k.get("name") == "submit")
            out[f"{staged}/{f.name}"] = rec
    return out


@app.local_entrypoint()
def main():
    res = check.remote()
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "q4b_think_empty_check.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
