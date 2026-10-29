"""Q4 (b) input-variant check (RULE.md Amendment 3): does d9's Qwen3-32B cross-read of Qwen3.5's own runs carry the
agent's reasoning inline? One Modal CPU container on the staged token ids (xread_main_v1/inputs), counts only, no text:
per run and assistant turn, think blocks and the tokens inside them, and the non-whitespace tokens outside think blocks and
tool calls (the agent's chat); plus the system turn's <tools> block.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 modal run --detach \\
      experiments/2026-10-13_q4_why_weaker_POSTHOC/check_q4_xread_reasoning.py
"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q4-xread-think_kill-2026-11-13"
XREAD_ACTS, HF = "probe-shift_aswin-surya_sub-q3xread_acts_kill-2026-11-13", "probe-shift_aswin-surya_hf-cache"
ARMS = ("attack", "sim_attack@september", "honest_a")
QWEN3 = ("Qwen/Qwen3-32B", "9216db5781bf21249d130ec9da846c4624c16137")
PER_ARM = 3

app = modal.App(APP)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("transformers==5.19.0", "numpy", "jinja2")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}))


@app.function(image=image, cpu=4, memory=16384, timeout=900, retries=0,
              volumes={"/x": modal.Volume.from_name(XREAD_ACTS).read_only(), "/hf": modal.Volume.from_name(HF).read_only()})
def check() -> dict:
    import numpy as np
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(QWEN3[0], revision=QWEN3[1])
    sid = {s: tok.convert_tokens_to_ids(s) for s in ("<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>", "</tool_call>")}
    asst = tok.encode("assistant", add_special_tokens=False)
    if len(asst) != 1:
        raise ValueError("assistant is not one token")
    ws = np.array([tok.decode([i]).strip() == "" for i in range(len(tok))])
    tools_seq = tok.encode("<tools>", add_special_tokens=False)
    out = {}
    for arm in ARMS:
        n = 0
        for line in Path("/x/xread_main_v1/inputs", f"{arm}.jsonl").open():
            ep = json.loads(line)
            ids = np.asarray(ep["ids"])
            starts = np.flatnonzero(ids == sid["<|im_start|>"])
            e0 = np.flatnonzero(ids == sid["<|im_end|>"])
            sys_ids = ids[starts[0]: e0[e0 > starts[0]][0]] if len(starts) and len(e0) else ids[:0]
            rec = {"assistant_turns": 0, "think_blocks": 0, "think_nonempty_turns": 0, "max_think_tokens": 0, "turns_with_chat": 0,
                   "tools_block_in_system_turn": any(np.array_equal(sys_ids[i:i + len(tools_seq)], tools_seq) for i in range(len(sys_ids)))}
            for s in starts:
                if s + 1 >= len(ids) or ids[s + 1] != asst[0]:
                    continue
                ends = np.flatnonzero(ids[s + 2:] == sid["<|im_end|>"])
                t = ids[s + 2: s + 2 + (ends[0] if len(ends) else len(ids) - s - 2)]
                rec["assistant_turns"] += 1
                inside, state, think_len, chat = np.zeros(len(t), bool), None, 0, 0
                for i, v in enumerate(t):
                    if v == sid["<think>"]:
                        state, think_len = "think", 0
                        rec["think_blocks"] += 1
                    elif v == sid["</think>"]:
                        state = None
                        rec["max_think_tokens"] = max(rec["max_think_tokens"], think_len)
                        rec["think_nonempty_turns"] += int(think_len > 0)
                    elif v == sid["<tool_call>"]:
                        state = "call"
                    elif v == sid["</tool_call>"]:
                        state = None
                    elif state == "think":
                        think_len += int(not ws[v])
                    elif state is None:
                        chat += int(not ws[v])
                    inside[i] = state is not None
                rec["turns_with_chat"] += int(chat > 0)
            out.setdefault(arm, []).append(rec)
            n += 1
            if n >= PER_ARM:
                break
    return out


@app.local_entrypoint()
def main():
    res = check.remote()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    dest = HERE / "results" / f"q4_xread_reasoning_POSTHOC_{stamp}.json"
    dest.open("x").write(json.dumps({"label": "POST HOC (RULE.md Amendment 3); counts only", "stamp": stamp, "runs": res}, indent=1))
    print(json.dumps(res, indent=1))
    print(dest.name)
