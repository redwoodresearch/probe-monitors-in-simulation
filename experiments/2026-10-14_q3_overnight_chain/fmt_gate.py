"""The staging's assemble format gate, in ONE file so a shared fix replaces this file only (stage_chain.py records its sha256 in
every staging record). Now the 0d-approved, Codex-reviewed shared patch aswin/q3-stage-cut-think-fix @ ef3e9f7f3a30e2eeee6579cc6f365e250c9916e1
(run_q4f_stage_c.py file sha256 11635cf5...): empty_think_block, cut_think, gate_row and gate_passes copied verbatim from it, and
gate() is its assemble loop verbatim. Rows carry `think_open_truncated_at_cut`; each arm `think_open_truncated_at_cut_runs`.
3 staged runs per arm, counts on the staged ids only. (The format check, 597e97b7c, is the gate that catches a think-block strip.)"""
import numpy as np


def empty_think_block(tok) -> list:
    """The template's empty think block exactly as it emits it for a reasoning-free assistant turn, tokenizer-encoded
    (AMENDMENT_cut_think_2026-10-14.md; never a hand-typed list)."""
    probe = tok.apply_chat_template([{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}], tokenize=False)
    i = probe.index("<think>")
    j = probe.index("</think>", i) + len("</think>")
    if probe[i + len("<think>"):j - len("</think>")].strip():
        raise ValueError("the template's think block for a reasoning-free turn is not empty")
    return tok.encode(probe[i:j], add_special_tokens=False)


def cut_think(ids, opens, closes, empty, cut) -> bool:
    """AMENDMENT_cut_think_2026-10-14.md: True only when the epoch is exactly `cut` tokens, its final <think> comes after
    its last </think>, and the tail from that <think> is a STRICT prefix of `empty`."""
    if len(ids) != cut or not len(opens) or (len(closes) and closes[-1] > opens[-1]):
        return False
    tail = [int(x) for x in ids[int(opens[-1]):]]
    return len(tail) < len(empty) and tail == [int(x) for x in empty[:len(tail)]]


def gate_row(ids, calls, sid, asst, tools_seq, empty, cut, submit) -> dict:
    """One staged epoch's format-gate counts; the final <think> leaves the counts only under cut_think (amendment)."""
    import numpy as np

    ids = np.asarray(ids)
    st = np.flatnonzero(ids == sid["<|im_start|>"])
    n_asst = int(sum(1 for s in st if s + 1 < len(ids) and ids[s + 1] == asst))
    has_tools = any(np.array_equal(ids[i:i + len(tools_seq)], tools_seq) for i in range(min(len(ids), 4000)))
    n_submit = sum(1 for c in calls if c.get("name") == submit)
    opens, closes = np.flatnonzero(ids == sid["<think>"]), np.flatnonzero(ids == sid["</think>"])
    flag = cut_think(ids, opens, closes, empty, cut)
    if flag:
        opens = opens[:-1]
    nonempty = sum(1 for o, c in zip(opens, closes) if c - o > 3) if len(opens) == len(closes) else -1
    return {"assistant_turns": n_asst, "think_open": len(opens), "think_close": len(closes), "think_blocks_with_content": nonempty,
            "tools_block": bool(has_tools), "submit_rows": n_submit, "think_open_truncated_at_cut": flag}


def gate_passes(r) -> bool:
    return (r["tools_block"] is False and r["submit_rows"] == 0 and r["think_open"] == r["think_close"]
            and r["think_blocks_with_content"] == 0 and r["think_open"] <= r["assistant_turns"])


def gate(eps: dict, tok, S) -> dict:  # run_q4f_stage_c.py @ ef3e9f7f, assemble's gate loop, verbatim
    sid = {s: tok.convert_tokens_to_ids(s) for s in ("<|im_start|>", "<think>", "</think>", "<tool_call>", "<tool_response>")}
    asst = tok.encode("assistant", add_special_tokens=False)[0]
    tools_seq = tok.encode("<tools>", add_special_tokens=False)
    empty = empty_think_block(tok)  # AMENDMENT_cut_think_2026-10-14.md
    gate = {}
    for arm, lst in eps.items():
        rows = [gate_row(e["ids"], e["calls"], sid, asst, tools_seq, empty, S.CUT, S.transcripts.RUNNER_SUBMIT)
                for e in sorted(lst, key=lambda e: e["key"])[:3]]
        ok = all(gate_passes(r) for r in rows)
        flagged = sum(cut_think(e["ids"], np.flatnonzero(np.asarray(e["ids"]) == sid["<think>"]),
                                np.flatnonzero(np.asarray(e["ids"]) == sid["</think>"]), empty, S.CUT) for e in lst)
        gate[arm] = {"runs": rows, "passes": ok, "think_open_truncated_at_cut_runs": int(flagged), "epochs": len(lst)}
    return gate
