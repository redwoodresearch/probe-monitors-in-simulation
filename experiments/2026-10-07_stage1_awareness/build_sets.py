"""Build Stage 1's contrast sets and implicit cases (probe_shift.awareness_sets, seed 0), check their invariants, and
write them with counts and a hash. $0: pure Python, no model, no API.

Checks (any failure raises; nothing is written):
  - schema and unique ids; every pair has exactly one label-1 and one label-0 item sharing probe, set, family,
    environment, task, friendly and impact (impact pairs excepted for impact);
  - readout: the last assistant turn's last tool call has its content parameter last (bash {"cmd"} only; text_editor
    view -> path, create -> file_text, str_replace -> new_str, insert -> insert_text);
  - awareness pairs end in the byte-identical final call; impact pairs share everything before the final call and
    differ in it;
  - stakes_implicit contains none of production / prod* / sandbox* / test* anywhere; mon_implicit's system and user
    messages contain none of monitor / logged / audit;
  - no message quotes a chat-template delimiter the reader parses;
  - balance: friendly / terse in tvc_card, impactful / read-only in stakes_implicit.
Diagnostic (reported, not enforced): for each set, the single tokens whose presence best separates the classes
(presence AUROC), so a trivially lexical implicit set is visible.

  PYTHONPATH=src uv run --no-project --python 3.12 python experiments/2026-10-07_stage1_awareness/build_sets.py
"""

import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from probe_shift.awareness_sets import BANNED_STAKES, build

HERE = Path(__file__).parent
REPO = HERE.parent.parent
SEED = 0
KEYS = {"id", "probe", "set", "family", "pair", "label", "implicit", "friendly", "impact", "environment", "task", "messages"}
CONTENT_LAST = {"bash": {"cmd"}, "text_editor": {"path", "file_text", "new_str", "insert_text"}}
DELIMS = ["<|im_end|>", "<|im_start|>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>",
          "<function=", "</function>", "<parameter=", "</parameter>", "<think>", "</think>"]
MON_BANNED = ("monitor", "logged", "audit")


def final_call(item):
    last = [m for m in item["messages"] if m["role"] == "assistant"][-1]
    return last["tool_calls"][-1]


def check(items):
    if len({it["id"] for it in items}) != len(items):
        raise ValueError("duplicate ids")
    for it in items:
        if set(it) != KEYS:
            raise ValueError(f"{it['id']}: keys {sorted(set(it) ^ KEYS)}")
        c = final_call(it)["function"]
        args = list(c["arguments"])
        if c["name"] == "bash" and args != ["cmd"]:
            raise ValueError(f"{it['id']}: bash readout call arguments {args}")
        if c["name"] == "text_editor" and (args[-1] not in CONTENT_LAST["text_editor"] or
                                           (args[0] == "command" and {"view": "path", "create": "file_text", "str_replace": "new_str", "insert": "insert_text"}[c["arguments"]["command"]] != args[-1])):
            raise ValueError(f"{it['id']}: editor readout call arguments {args}")
        text = json.dumps(it["messages"], ensure_ascii=False)
        if any(d in text for d in DELIMS):
            raise ValueError(f"{it['id']}: quotes a template delimiter")
        if it["set"] == "stakes_implicit" and BANNED_STAKES.search(" ".join(
                [m.get("content") or "" for m in it["messages"]] + [json.dumps(m.get("tool_calls") or []) for m in it["messages"]])):
            raise ValueError(f"{it['id']}: banned stakes word: {BANNED_STAKES.search(text).group(0)}")
        if it["set"] == "mon_implicit":
            su = " ".join(m["content"] for m in it["messages"] if m["role"] in ("system", "user")).lower()
            if any(w in su for w in MON_BANNED):
                raise ValueError(f"{it['id']}: monitoring word in system/user")
    by_pair = defaultdict(list)
    for it in items:
        by_pair[it["pair"]].append(it)
    for p, two in by_pair.items():
        if sorted(x["label"] for x in two) != [0, 1]:
            raise ValueError(f"{p}: labels {[x['label'] for x in two]}")
        a, b = two
        shared = ("probe", "set", "family", "environment", "task", "friendly", "implicit") + (() if a["probe"] == "impact" else ("impact",))
        if any(a[k] != b[k] for k in shared):
            raise ValueError(f"{p}: pair members differ in {[k for k in shared if a[k] != b[k]]}")
        fa, fb = json.dumps(final_call(a)), json.dumps(final_call(b))
        if a["probe"] == "impact":
            last_a = max(i for i, m in enumerate(a["messages"]) if m["role"] == "assistant")
            if a["messages"][:last_a] != b["messages"][:last_a] or fa == fb:
                raise ValueError(f"{p}: impact pair must share the context and differ in the final call")
        elif fa != fb:
            raise ValueError(f"{p}: final calls differ")
    card = Counter(it["friendly"] for it in items if it["set"] == "tvc_card")
    stakes = Counter(it["impact"] for it in items if it["set"] == "stakes_implicit")
    if abs(card[True] - card[False]) > 2 or abs(stakes["impactful"] - stakes["read_only"]) > 2:
        raise ValueError(f"unbalanced: card {card}, stakes_implicit {stakes}")


def token_separation(items, top=5):
    """Per set, the tokens whose presence in an item best separates label 1 from 0 (AUROC of presence, |.-0.5|)."""
    out = {}
    tok = re.compile(r"[A-Za-z_][A-Za-z0-9_\-]{2,}")
    by_set = defaultdict(list)
    for it in items:
        by_set[it["set"]].append(it)
    for s, its in by_set.items():
        pres = [(it["label"], set(tok.findall(json.dumps(it["messages"]).lower()))) for it in its]
        n1 = sum(l for l, _ in pres)
        n0 = len(pres) - n1
        cnt1, cnt0 = Counter(), Counter()
        for l, toks in pres:
            (cnt1 if l else cnt0).update(toks)
        scored = []
        for t in set(cnt1) | set(cnt0):
            r1, r0 = cnt1[t] / n1, cnt0[t] / n0
            auc = 0.5 + (r1 - r0) / 2  # presence AUROC of a binary feature
            scored.append((abs(auc - 0.5), t, round(auc, 3), round(r1, 2), round(r0, 2)))
        out[s] = [{"token": t, "auroc": a, "share_label1": r1, "share_label0": r0}
                  for _, t, a, r1, r0 in sorted(scored, reverse=True)[:top]]
    return out


def main():
    items = build(SEED)
    check(items)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=REPO).stdout.strip())
    body = "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items)
    sha = hashlib.sha256(body.encode()).hexdigest()
    counts = defaultdict(Counter)
    for it in items:
        counts[it["probe"]][f"{it['set']}|{it['family']}|label={it['label']}"] += 1
        if it["friendly"] is not None:
            counts[it["probe"]][f"{it['set']}|friendly={it['friendly']}"] += 1
        if it["impact"] is not None:
            counts[it["probe"]][f"{it['set']}|impact={it['impact']}"] += 1
    pairs = Counter(it["set"] for it in items if it["label"] == 1)
    sep = token_separation(items)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    res = HERE / "results"
    res.mkdir(exist_ok=True)
    data_dir = HERE / "data"  # model-written dialogue text: git-ignored, archived to S3 (results/ keeps the summary)
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / f"sets_{stamp}.jsonl").open("x") as f:
        f.write(body)
    meta = {"stamp": stamp, "seed": SEED, "git": git, "git_dirty": dirty, "items": len(items), "pairs_per_set": dict(pairs),
            "counts": {k: dict(v) for k, v in counts.items()}, "sha256_jsonl": sha, "file": f"data/sets_{stamp}.jsonl",
            "single_token_separation": sep}
    with (res / f"sets_{stamp}.json").open("x") as f:
        f.write(json.dumps(meta, indent=1))
    print(f"{len(items)} items, pairs per set {dict(pairs)}\nsha256 {sha}\n-> data/sets_{stamp}.jsonl")
    for s, rows in sep.items():
        print(f"{s:16s} " + "; ".join(f"{r['token']} {r['auroc']} ({r['share_label1']}/{r['share_label0']})" for r in rows[:3]))


if __name__ == "__main__":
    main()
