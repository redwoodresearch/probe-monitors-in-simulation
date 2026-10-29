"""fix939 follow-up (D), readout token identity (PLAN.md, committed with this script before it runs): one Modal CPU
container counts, for #36's output_end readout on text_editor str_replace and insert outputs, which CATEGORY of token is
read, in real e1, real e2, the original simulated runs and the re-rendered (fix939) ones. Token and output text stay in
the container: it returns counts only, and nothing else is printed, logged or saved.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run \\
      experiments/2026-10-12_q2_fix939_checks/run_readout_tokens.py --runs <#36 runs.json> --fix <fix939 per-run jsonl>
"""
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-fix939-tokens_kill-2026-11-12"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
DUMP, DUMP_INPUTS = "2026-10-06_la_segment-end_qwen3-32b_v1", "2026-10-06_la_segment-end_qwen3-32b_v1"
FIX_DUMP, FIX_INPUTS, FIX_ARM = "2026-10-11_q3_fix939_qwen3-32b_l48-63_v1", "2026-10-11_q3_fix939_inputs_v1", "sim-attack-note-sr-both"
SWAP = HERE.parent / "2026-10-07_q2_rendering_swap"
app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2).read_only()
image = modal.Image.debian_slim(python_version="3.12").add_local_file(SWAP / "rerender.py", "/root/rerender.py")
COMMANDS = ("str_replace", "insert")
PIN = {"runs.json": "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96",
       "fix939_frozen_runs_POSTHOC_2026-10-11_020458.jsonl": "5f6ecbc42b1c6bb9586cb58e474d74c5c1d914bedb95d80a5a07dd078b0e5ebd"}


def bytes_to_unicode() -> dict:
    """GPT-2 byte-level BPE's byte -> printable-character map (Qwen's tokenizer uses it): vocabulary strings like 'Ċ' (newline)
    and 'Ġ' (space) are these characters."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, map(chr, cs)))


BYTE_OF = {c: b for b, c in bytes_to_unicode().items()}


def decode(tok: str) -> str:
    """A vocabulary string to its text; an already-decoded or special token (a character outside the map) is kept as is."""
    return bytes(BYTE_OF[ch] for ch in tok).decode("utf-8", "replace") if tok and all(ch in BYTE_OF for ch in tok) else tok


def category(tok: str, ends_with_note: bool, review: str) -> str:
    t = decode(tok).strip()
    if t == "":
        return "newline or whitespace"
    if ends_with_note and t in review:
        return "review-note template"
    if set(t) <= set("0123456789|"):
        return "gutter or digit"
    return "other content"


@app.function(image=image, volumes={"/acts": acts}, cpu=4, memory=16384, timeout=1800, retries=0)
def count(cohort: list[dict], fix_keys: list[str]) -> dict:
    import sys
    sys.path.insert(0, "/root")
    from rerender import REVIEW
    review = REVIEW.strip()

    def items(run_dir, by_key=False):
        out = {}
        for shard in sorted((Path("/acts") / run_dir / "inputs").glob("shard_*.jsonl")):
            for line in shard.read_text().splitlines():
                it = json.loads(line)
                out[it["key"] if by_key else (it["arm"], it["key"])] = it["messages"]
        return out

    def text_of(c):
        return "".join(p.get("text", "") for p in c if p.get("type") == "text") if isinstance(c, list) else (c or "")

    def outputs(msgs):
        """(step, action) -> (command, output text): assistant step s's action a and the a-th tool message after it."""
        res, asst = {}, [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
        for s, i in enumerate(asst):
            nxt = asst[s + 1] if s + 1 < len(asst) else len(msgs)
            tools = [m for m in msgs[i + 1: nxt] if m["role"] == "tool"]
            for a, tc in enumerate(msgs[i].get("tool_calls") or []):  # staged calls: {"function": {"name", "arguments"}}
                fn = tc.get("function") or {}
                args = fn.get("arguments") or {}
                args = json.loads(args) if isinstance(args, str) else args
                cmd = args.get("command") if fn.get("name") == "text_editor" else None
                if a < len(tools):
                    res[(s, a)] = (cmd, text_of(tools[a].get("content")))
        return res
    base, fixed = items(DUMP_INPUTS), items(FIX_INPUTS, by_key=True)  # the re-rendered inputs, by the original key
    counts, missing = Counter(), Counter()

    def tally(label, msgs, meta):
        outs = outputs(msgs)
        for p in meta["positions"]:
            if p["kind"] != "output_end" or not p["kept"]:
                continue
            cmd, text = outs.get((p["step"], p["action"]), (None, None))
            if text is None:  # fail loudly: every kept readout must map to a tool output
                raise ValueError(f"{label}: a kept output_end position has no tool output at its (step, action)")
            if cmd not in COMMANDS:
                continue
            note = text.rstrip().endswith(review)
            counts[(label, cmd, category(p["token_text"] or "", note, review), note)] += 1
    fix_set = set(fix_keys)
    sim_rows = {r["key"]: r for r in cohort if r["arm"] == "sim-attack"}
    bad = [k for k in fix_keys if k not in sim_rows or k not in fixed or ("sim-attack", k) not in base
           or not (Path("/acts") / FIX_DUMP / "meta" / FIX_ARM / f"{sim_rows[k]['filename']}.json").exists()]
    if bad:  # every re-rendered run must have its original, its inputs and its meta before either side is counted
        raise ValueError(f"{len(bad)} of the 823 re-rendered runs lack an original, inputs or meta")
    for r in cohort:
        meta = json.loads((Path("/acts") / DUMP / "acts" / r["arm"] / f"{r['filename']}.json").read_text())
        msgs = base.get((r["arm"], r["key"]))
        if msgs is None:
            raise ValueError(f"{r['arm']}: a cohort run has no staged inputs")
        label = {"real-attack-e1": "real e1", "real-attack-e2": "real e2", "sim-attack": "simulated (as simulated)"}[r["arm"]]
        tally(label, msgs, meta)
        if r["arm"] == "sim-attack" and r["key"] in fix_set:
            tally("simulated, the 823 re-rendered: before", msgs, meta)
            fmsgs = fixed[r["key"]]
            fmeta = json.loads((Path("/acts") / FIX_DUMP / "meta" / FIX_ARM / f"{r['filename']}.json").read_text())
            tally("simulated, the 823 re-rendered: after", fmsgs, fmeta)
    per_label = Counter(k[0] for k in counts.elements())
    for lab in ("real e1", "real e2", "simulated (as simulated)", "simulated, the 823 re-rendered: before", "simulated, the 823 re-rendered: after"):
        if per_label[lab] == 0:  # fail loudly: an empty arm means the mapping broke, not that there are no editor outputs
            raise ValueError(f"no str_replace or insert readouts counted for {lab}")
    rows = [{"arm": k[0], "command": k[1], "readout token": k[2], "output ends with the review note": k[3], "n": n} for k, n in sorted(counts.items())]
    return {"rows": rows, "missing": dict(missing), "review note sha256": hashlib.sha256(REVIEW.encode()).hexdigest()}


@app.local_entrypoint()
def main(runs: str, fix: str):
    for p_, role in ((Path(runs), "runs.json"), (Path(fix), "fix939_frozen_runs_POSTHOC_2026-10-11_020458.jsonl")):
        if hashlib.sha256(p_.read_bytes()).hexdigest() != PIN[role]:
            raise SystemExit(f"{p_.name} is not the pinned {role}")
    rs = json.loads(Path(runs).read_text())
    attack = [r for r in rs if r["arm"] in ("real-attack-e1", "real-attack-e2", "sim-attack")]
    cells = {a: {r["cell"] for r in attack if r["arm"] == a} for a in ("real-attack-e1", "real-attack-e2", "sim-attack")}
    keep = cells["real-attack-e1"] & cells["real-attack-e2"] & cells["sim-attack"]
    cohort = [{"arm": r["arm"], "key": r["key"], "filename": r["filename"]} for r in attack if r["cell"] in keep]
    fix_keys = sorted({json.loads(l)["key"] for l in open(fix) if json.loads(l)["label"] == "FIX"})
    sim_keys = {r["key"] for r in attack if r["arm"] == "sim-attack" and r["cell"] in keep}
    if len(keep) != 939 or len(fix_keys) != 823 or not set(fix_keys) <= sim_keys:
        raise SystemExit(f"cohort {len(keep)} cells, {len(fix_keys)} re-rendered: not fix939's 939 / 823")
    res = count.remote(cohort, fix_keys)
    out = {"plan": "PLAN.md (this folder), part (D)", "inputs sha256": {Path(runs).name: hashlib.sha256(Path(runs).read_bytes()).hexdigest(),
           Path(fix).name: hashlib.sha256(Path(fix).read_bytes()).hexdigest()}, "cells": len(keep), "re-rendered runs": len(fix_keys)} | res
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"readout_tokens_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(dest)
