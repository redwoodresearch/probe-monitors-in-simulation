"""Q4 item 3 (RULE.md Amendment 2): how Qwen3.5-27B's input is built for Sol's runs (the reader format of #36's v4 dumps) vs
for its own runs (claim C's swapped replay, full and reasoning-stripped). One Modal CPU container; counts only, no
transcript text is printed or returned. Per format, on a few runs per arm: token counts of the template markers in the
token ids the extractor used (<think>, </think>, <tool_call>, <tool_response>, the role after each <|im_start|>), whether a
<tools> block is in the system turn, the system turn's length, and for Sol the number of assistant turns with visible text
and the number of assistant turns after the last user query (the turns Qwen3.5's template gives an empty think block).

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 modal run --detach \\
      experiments/2026-10-13_q4_why_weaker_POSTHOC/check_q4_formats.py
"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q4-formats_kill-2026-11-13"
LA_ACTS, ONPOL_ACTS, HF = ("probe-shift_aswin-surya_sub-la-activations", "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08",
                           "probe-shift_aswin-surya_hf-cache")
SOL_INPUTS = "2026-10-06_la_segment-end-v4_qwen3-32b_v1/inputs"
ONPOL = {"full (swapsys)": "rerun_strict_segends_v2_scrub1_swapsys/inputs/attack.jsonl",
         "stripped (swapsys_noagent)": "rerun_strict_segends_v2_scrub1_swapsys_noagent/inputs/attack.jsonl"}
QWEN35 = ("Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654")
PER_ARM = 3

app = modal.App(APP)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("transformers==5.19.0", "numpy", "jinja2")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}).add_local_python_source("probe_shift"))


@app.function(image=image, cpu=4, memory=16384, timeout=900, retries=0,
              volumes={"/la": modal.Volume.from_name(LA_ACTS).read_only(), "/onpol": modal.Volume.from_name(ONPOL_ACTS).read_only(),
                       "/hf": modal.Volume.from_name(HF).read_only()})
def check() -> dict:
    import numpy as np
    from transformers import AutoTokenizer

    from probe_shift import reader

    tok = AutoTokenizer.from_pretrained(QWEN35[0], revision=QWEN35[1])
    sid = {s: tok.convert_tokens_to_ids(s) for s in ("<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>", "</tool_call>",
                                                     "<tool_response>", "</tool_response>")}
    roles = {r: tok.encode(r, add_special_tokens=False) for r in ("system", "user", "assistant")}
    if any(len(v) != 1 for v in roles.values()):
        raise ValueError("a role is not one token")
    tools_seq = tok.encode("<tools>", add_special_tokens=False)

    def stats(ids) -> dict:
        ids = np.asarray(ids)
        starts = np.flatnonzero(ids == sid["<|im_start|>"])
        role_of = {v[0]: k for k, v in roles.items()}
        turn_roles = [role_of.get(int(ids[s + 1]), "other") for s in starts if s + 1 < len(ids)]
        sys_len = None
        if turn_roles and turn_roles[0] == "system":
            e = np.flatnonzero(ids[starts[0]:] == sid["<|im_end|>"])
            sys_len = int(e[0]) if len(e) else None
            sys_ids = ids[starts[0]: starts[0] + (sys_len or 0)]
        else:
            sys_ids = ids[:0]
        has_tools = any(np.array_equal(sys_ids[i:i + len(tools_seq)], tools_seq) for i in range(max(0, len(sys_ids) - len(tools_seq) + 1)))
        out = {k: int((ids == v).sum()) for k, v in sid.items() if k not in ("<|im_start|>", "<|im_end|>")}
        return out | {"tokens": int(len(ids)), "turns_by_role": {r: turn_roles.count(r) for r in ("system", "user", "assistant", "other")},
                      "system_turn_tokens": sys_len, "tools_block_in_system_turn": bool(has_tools)}

    res = {"sol_reader_format": {}, "onpolicy": {}}
    seen = {}
    for p in sorted(Path("/la", SOL_INPUTS).glob("shard_*.jsonl")):
        for line in p.open():
            r = json.loads(line)
            a = r.get("arm")
            if a not in ("real-attack-e1", "sim-attack", "real-honest") or seen.get(a, 0) >= PER_ARM:
                continue
            msgs = r["messages"]
            ids, _ = reader.render(tok, msgs)
            last_q = max([i for i, m in enumerate(msgs) if m["role"] == "user"], default=-1)
            n_after = sum(1 for i, m in enumerate(msgs) if m["role"] == "assistant" and i > last_q)
            text_turns = sum(1 for m in msgs if m["role"] == "assistant" and (m.get("content") or "").strip())
            res["sol_reader_format"].setdefault(a, []).append(
                stats(ids) | {"assistant_turns": sum(m["role"] == "assistant" for m in msgs), "assistant_turns_with_visible_text": text_turns,
                              "assistant_turns_after_last_user_query": n_after, "first_user_message_chars": len(next((m["content"] for m in msgs if m["role"] == "user"), ""))})
            seen[a] = seen.get(a, 0) + 1
        if all(seen.get(a, 0) >= PER_ARM for a in ("real-attack-e1", "sim-attack", "real-honest")):
            break
    for name, path in ONPOL.items():
        by_key = {}
        for line in Path("/onpol", path).open():
            ep = json.loads(line)
            by_key.setdefault(ep["key"], []).append(ep)
            if len(by_key) > PER_ARM:
                break
        for k in list(by_key)[:PER_ARM]:
            ep = max(by_key[k], key=lambda e: len(e["ids"]))  # the longest context epoch of the run
            res["onpolicy"].setdefault(name, []).append(stats(ep["ids"]) | {"epochs_seen": len(by_key[k])})
    return res


@app.local_entrypoint()
def main():
    res = check.remote()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    dest = HERE / "results" / f"q4_formats_POSTHOC_{stamp}.json"
    dest.open("x").write(json.dumps({"label": "POST HOC (RULE.md Amendment 2); counts only", "stamp": stamp} | res, indent=1))
    print(json.dumps(res, indent=1))
    print(dest.name)
