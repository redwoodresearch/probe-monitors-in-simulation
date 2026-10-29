"""B2's repair, "fix-full minus the clock (F7)", for the readers' messages. Copied from stage_b2.py (aswin/q3-sabotage-avg-repair
@ 31e4551c): norm_first, trail, the F8 template learning, f5, f8 and repair() are unchanged. B2 applied it to the policy's view
of each run to rebuild Qwen3.5's logged tokens; here it is applied to transcripts.reader_messages, the messages q2's fix-full
(stage_fixfull.py) was written for, and each reader renders the result in its own template. Only tool outputs may change.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

BT = re.compile(r"`[^`]*`")
BASE = "2026-10-06_la_segment-end_qwen3-32b_v1"  # q2's F8 source: #36's real-attack-e1/e2


def norm_first(text: str) -> str:  # stage_fixfull.norm_first, unchanged
    first = BT.sub("<v>", text.split("\n", 1)[0])
    return " ".join("<v>" if any(ch.isdigit() or ch == "/" for ch in t) else t for t in first.split(" "))


def trail(text: str) -> str:  # stage_fixfull.trail, unchanged
    n = len(text) - len(text.rstrip("\n"))
    return "none" if n == 0 else "one" if n == 1 else "more"


def learn_templates(la_root: Path, errors_file: Path, fixes) -> dict:
    """F8's templates, exactly as stage_b2.run_job learns them (REAL errored outputs of #36's real attack runs)."""
    errored = [[r["arm"], r["key"], r["step"], r["action"], r["tool"]] for r in map(json.loads, errors_file.read_text().splitlines())
               if r["error"] is not None and r["arm"] in ("real-attack-e1", "real-attack-e2", "sim-attack")]
    base = {}
    for p in sorted((la_root / BASE / "inputs").glob("shard_*.jsonl")):
        for line in p.read_text().splitlines():
            it = json.loads(line)
            if it["arm"] in ("real-attack-e1", "real-attack-e2"):
                base[(it["arm"], it["key"])] = it
    err = {(a, k, s, j): t for a, k, s, j, t in errored}
    real_err = defaultdict(Counter)
    for (a, k), it in base.items():
        for x in fixes.calls_of(it["messages"]):
            if err.get((a, k, x["step"], x["action"])) is None:
                continue
            text = it["messages"][x["msg"]]["content"]
            if isinstance(text, str):
                real_err[norm_first(text)][trail(text)] += 1
    templates = {}
    for key, v in real_err.items():
        n = sum(v.values())
        if n >= 5:
            templates[key] = "strip" if v["none"] >= 0.9 * n else "one" if v["one"] >= 0.9 * n else "keep"
    if len(templates) != 14 or set(templates.values()) != {"strip"}:
        raise ValueError(f"F8: {len(templates)} templates {Counter(templates.values())}, not q2's 14 all-strip table")
    return templates


def make_repair(templates: dict, fixes, rr, st):
    def is_err(text):
        return isinstance(text, str) and norm_first(text) in templates

    def f5(msgs):  # stage_fixfull.f5, unchanged
        out = [dict(m) for m in msgs]
        for x in fixes.calls_of(msgs):
            o = msgs[x["msg"]]["content"]
            if x["tool"] == "text_editor" and x["args"].get("command") == "create" and isinstance(o, str) and not is_err(o):
                out[x["msg"]]["content"] = o.rstrip("\n")
        return out

    def f8(msgs):  # stage_fixfull.f8, unchanged
        out = [dict(m) for m in msgs]
        for x in fixes.calls_of(msgs):
            o = msgs[x["msg"]]["content"]
            if is_err(o):
                rule = templates[norm_first(o)]
                if rule == "strip":
                    out[x["msg"]]["content"] = o.rstrip("\n")
                elif rule == "one":
                    out[x["msg"]]["content"] = o.rstrip("\n") + "\n"
        return out

    def repair(msgs):  # stage_fixfull's variant "full" with F7 removed
        m = rr.rerender_messages(msgs, "diff_with_note")[0]
        m = st.swap_messages(m, "sr_gutter_no_newline")[0]
        m = fixes.r5_dpkg(f5(m))[0]
        new = f8(m)
        if len(new) != len(msgs):
            raise ValueError("the repair changed the number of messages")
        changed = [i for i, (x, y) in enumerate(zip(msgs, new)) if x != y]
        if any(msgs[i]["role"] != "tool" or set(msgs[i]) != set(new[i]) or any(msgs[i][f] != new[i][f] for f in msgs[i] if f != "content")
               for i in changed):
            raise ValueError("the repair changed a non-tool message or a field other than content")
        return new, len(changed)
    return repair
