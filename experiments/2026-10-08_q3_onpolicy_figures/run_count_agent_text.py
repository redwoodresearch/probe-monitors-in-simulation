"""POST-HOC step 0 of plan Amendment 11: how much of each staged transcript is the agent's own reasoning and chat text.
Modal CPU, tokenizer only; keys and numbers only (no transcript text is decoded or printed: the tokenizer decodes only
vocabulary entries, to find the whitespace-only token ids).

For every staged epoch the claim C readings use (swapped run 1, swapped run 2, swapped simulated attack; honest_a and
honest_b from the masked base stagings), each assistant turn's interior (after <|im_start|>, up to <|im_end|> or the end)
is split by special-token ids into:
  - reasoning: non-whitespace tokens strictly inside <think> ... </think> (the template writes an empty block as
    whitespace only);
  - chat: tokens outside the think block (tags included) and outside every <tool_call> ... </tool_call> span (tags
    included), excluding the role token and whitespace-only tokens;
  - commands: tokens inside tool-call spans, tags included.
Epochs the cache skips (redrawn, or attempt_kind "rerun") are counted separately. A turn whose tags do not alternate
open/close (nesting, overlap, or left open) is not split: it is counted as malformed, with its tokens and its tag pattern
(symbols only: T/t think open/close, C/c tool call open/close, E for a final turn without <|im_end|>).

  uv run --no-project --python 3.12 --with 'modal>=1.4.3' modal run experiments/2026-10-08_q3_onpolicy_figures/run_count_agent_text.py
"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
ACTS, HF_CACHE = "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08", "probe-shift_aswin-surya_hf-cache"
MODEL, REVISION = "Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654"
BASES = ["rerun_strict_segends_v2_scrub1", "scaleup_canary_segends_v1_scrub1", "scaleup_l2_segends_v1_scrub1", "scaleup_topup_segends_v1_scrub1"]
SOURCES = ([("run1", f"{b}_swapsys", "attack") for b in BASES] + [("run2", "second_attack_segends_v1_scrub1_swapsys", "attack")]
           + [("sim", f"{b}_swapsys2", "sim_attack@september") for b in BASES]
           + [(h, b, h) for b in BASES for h in ("honest_a", "honest_b")])

app = modal.App("probe-shift_aswin-surya_sub-q3onpol_agenttext_kill-2026-11-10")
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("transformers==5.19.0", "jinja2", "numpy")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}))


@app.function(image=image, cpu=4, memory=32768, timeout=1800, retries=0,
              volumes={"/acts": modal.Volume.from_name(ACTS, version=2).read_only(), "/hf": modal.Volume.from_name(HF_CACHE).read_only()})
def count() -> dict:
    import numpy as np
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    sid = {s: tok.convert_tokens_to_ids(s) for s in ("<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>", "</tool_call>")}
    if any(v is None or v == tok.unk_token_id for v in sid.values()):
        raise ValueError(f"a template marker is not a single token: {sid}")
    role = tok.encode("assistant", add_special_tokens=False)
    if len(role) != 1:
        raise ValueError("the assistant role is not one token")
    ws = np.zeros(len(tok), bool)
    for i in range(len(tok)):
        ws[i] = tok.decode([i]).strip() == ""
    per = {}
    for world, staging, arm in SOURCES:
        p = Path("/acts") / staging / "inputs" / f"{arm}.jsonl"
        for line in p.read_text().splitlines():
            ep = json.loads(line)
            ids = np.asarray(ep["ids"])
            skipped = bool(ep.get("redrawn_because") or ep.get("attempt_kind") == "rerun")
            r = {"total": len(ids), "reasoning": 0, "chat": 0, "commands": 0, "assistant_turns": 0, "turns_with_reasoning": 0, "turns_with_chat": 0,
                 "malformed_turns": 0, "malformed_tokens": 0, "malformed_last_turn": 0, "patterns": []}
            starts = np.flatnonzero(ids == sid["<|im_start|>"])
            for s in starts:
                ends = np.flatnonzero(ids[s + 1:] == sid["<|im_end|>"])
                e = s + 1 + (ends[0] if len(ends) else len(ids) - s - 1)
                if s + 1 >= e or ids[s + 1] != role[0]:
                    continue
                t = ids[s + 2:e]  # the interior after the role token
                tags = [(i, int(t[i])) for i in np.flatnonzero(np.isin(t, [sid[x] for x in ("<think>", "</think>", "<tool_call>", "</tool_call>")]))]
                state, bad = None, False  # well formed: tags alternate open/close, never nest, and close before the turn ends
                for i, v in tags:
                    if v in (sid["<think>"], sid["<tool_call>"]) and state is None:
                        state = v
                    elif (state, v) in ((sid["<think>"], sid["</think>"]), (sid["<tool_call>"], sid["</tool_call>"])):
                        state = None
                    else:
                        bad = True
                bad = bad or state is not None
                r["assistant_turns"] += 1
                if bad:  # counted and described by its tag symbols only (T/t think open/close, C/c call open/close; E: turn has no <|im_end|>)
                    pat = "".join({sid["<think>"]: "T", sid["</think>"]: "t", sid["<tool_call>"]: "C", sid["</tool_call>"]: "c"}[v] for _, v in tags)
                    r["malformed_turns"] += 1
                    r["malformed_tokens"] += len(t)
                    r["patterns"].append(pat if len(pat) <= 40 else pat[:40] + "...") 
                    r["patterns"][-1] += "" if len(ends) else "E"
                    r["malformed_last_turn"] += int(not len(ends))
                    continue
                in_think = np.cumsum((t == sid["<think>"]).astype(int) - np.r_[0, (t == sid["</think>"])[:-1]].astype(int)) > 0  # tags included
                in_call = np.cumsum((t == sid["<tool_call>"]).astype(int) - np.r_[0, (t == sid["</tool_call>"])[:-1]].astype(int)) > 0
                inner_think = in_think & (t != sid["<think>"]) & (t != sid["</think>"]) & ~ws[t]  # non-whitespace reasoning
                chat = ~in_think & ~in_call & ~ws[t]
                r["reasoning"] += int(inner_think.sum())
                r["chat"] += int(chat.sum())
                r["commands"] += int(in_call.sum())
                r["turns_with_reasoning"] += int(inner_think.any())
                r["turns_with_chat"] += int(chat.any())
            key = f"{world}|{'skipped' if skipped else 'counted'}"
            per.setdefault(key, []).append(r | {"staging": staging})
    out = {}
    for key, rs in sorted(per.items()):
        a = {k: np.array([r[k] for r in rs]) for k in ("total", "reasoning", "chat", "commands", "assistant_turns", "turns_with_reasoning", "turns_with_chat",
                                                       "malformed_turns", "malformed_tokens", "malformed_last_turn")}
        from collections import Counter
        pats = Counter(p for r in rs for p in r["patterns"])
        free = a["reasoning"] + a["chat"]
        q = lambda x: {"median": float(np.median(x)), "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90)), "mean": float(x.mean())}
        out[key] = {"epochs": len(rs), "by_staging": {s: sum(r["staging"] == s for r in rs) for s in sorted({r["staging"] for r in rs})},
                    "tokens_total": int(a["total"].sum()), "share_reasoning": float(a["reasoning"].sum() / a["total"].sum()),
                    "share_chat": float(a["chat"].sum() / a["total"].sum()), "share_commands": float(a["commands"].sum() / a["total"].sum()),
                    "free_text_per_epoch": q(free), "reasoning_per_epoch": q(a["reasoning"]), "chat_per_epoch": q(a["chat"]),
                    "epochs_with_zero_free_text": int((free == 0).sum()), "epochs_with_zero_reasoning": int((a["reasoning"] == 0).sum()),
                    "assistant_turns": int(a["assistant_turns"].sum()), "turns_with_reasoning": int(a["turns_with_reasoning"].sum()),
                    "turns_with_chat": int(a["turns_with_chat"].sum()), "malformed_turns": int(a["malformed_turns"].sum()),
                    "malformed_last_turns": int(a["malformed_last_turn"].sum()), "epochs_with_malformed_turns": int((a["malformed_turns"] > 0).sum()),
                    "share_tokens_in_malformed_turns": float(a["malformed_tokens"].sum() / a["total"].sum()), "malformed_patterns_top": pats.most_common(12)}
    return out


@app.local_entrypoint()
def main():
    res = count.remote()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"agent_text_count_POSTHOC_{stamp}.json"
    with path.open("x") as fh:
        fh.write(json.dumps({"amendment": "11 (post hoc), step 0", "model": MODEL, "revision": REVISION, "counts": res}, indent=1))
    for k, v in res.items():
        print(k, json.dumps({x: v[x] for x in ("epochs", "share_reasoning", "share_chat", "free_text_per_epoch", "epochs_with_zero_free_text",
                                               "assistant_turns", "turns_with_reasoning", "turns_with_chat", "malformed_turns", "malformed_last_turns",
                                               "share_tokens_in_malformed_turns", "malformed_patterns_top")}))
    print(path.name)
