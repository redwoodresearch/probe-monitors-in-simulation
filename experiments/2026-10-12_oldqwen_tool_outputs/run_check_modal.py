"""Did tool outputs reach Qwen3-32B ("old Qwen") as a reader? A setup check for James (0d's request, 10-12; logs only,
one Modal CPU container, honest runs only).

Reads the v4 Qwen3-32B extraction's own records on the Volume:
- staged inputs `2026-10-06_la_segment-end-v4_qwen3-32b_v1/inputs/shard_*.jsonl`: the exact messages the all-layers dump
  `2026-10-06_la_segment-end-v4_qwen3-32b_alllayers_v1` rendered (its config: inputs_from that run);
- that dump's per-run meta, `meta/real-honest/*.json` (positions the extraction read, n_tokens, truncated).

(1) Counts over all real-honest runs, from the meta: tool outputs (output_end positions), how many lie inside the
    32,768-token cut, how many have no content token (empty output), and runs truncated.
(2) Real-honest runs in the order random.Random(20261012).shuffle gives the sorted meta keys; the first of the first 20
    with a kept, non-empty bash `ls` output is used. Its staged messages are rendered
    again with the extraction's own code (probe_shift.reader.segment_end_positions, Qwen3-32B tokenizer at the pinned
    revision, max_tokens 32,768), and the check confirms n_tokens and every output_end token equal the meta's. The excerpt
    is the first kept, non-empty output of a bash `ls` call (file and directory names only; 0d, 10-12): the decoded
    tokens from `<tool_response>` to `</tool_response>`, content cut to 8 lines. No other kind of output is used. The
    call's command text is not returned. Only numbers and that excerpt leave the container.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' modal run \\
      experiments/2026-10-12_oldqwen_tool_outputs/run_check_modal.py
"""

import hashlib
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

import modal

APP = "probe-shift_aswin-surya_sub-claimb-oldqwen-check_kill-2026-11-12"
VOL = "probe-shift_aswin-surya_sub-la-activations"
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
STAGED = "2026-10-06_la_segment-end-v4_qwen3-32b_v1"
DUMP = "2026-10-06_la_segment-end-v4_qwen3-32b_alllayers_v1"
ARM, SEED, MAX_TOKENS, LINES = "real-honest", 20261012, 32768, 8

app = modal.App(APP)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("transformers==4.56.2", "jinja2", "numpy")
         .env({"HF_HOME": "/hf"}).add_local_python_source("probe_shift"))

def command_of(call):
    a = call["function"].get("arguments") or {}
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except json.JSONDecodeError:
            return "", ""
    return str(a.get("cmd") or a.get("command") or ""), str(a.get("path") or "")


@app.function(image=image, volumes={"/out": modal.Volume.from_name(VOL, version=2), "/hf": modal.Volume.from_name(HF_CACHE)},
              cpu=4, memory=16384, timeout=1800)
def check():
    from transformers import AutoTokenizer

    from probe_shift.models import QWEN3_32B
    from probe_shift.reader import segment_end_positions

    metas = {}
    for f in sorted((Path("/out") / DUMP / "meta" / ARM).glob("*.json")):
        m = json.loads(f.read_text())
        metas[m["key"]] = m
    outs = [p for m in metas.values() for p in m["positions"] if p["kind"] == "output_end"]
    counts = {"runs": len(metas), "runs_truncated": sum(bool(m["truncated"]) for m in metas.values()),
              "tool_outputs": len(outs), "inside_cut": sum(bool(p["kept"]) for p in outs),
              "empty_no_content_token": sum(p["token"] is None for p in outs),
              "beyond_cut": sum(p["token"] is not None and not p["kept"] for p in outs),
              "runs_with_any_tool_output_inside_cut": sum(any(p["kind"] == "output_end" and p["kept"] for p in m["positions"]) for m in metas.values())}

    order = sorted(metas)
    random.Random(SEED).shuffle(order)
    want = {k: i for i, k in enumerate(order[:20])}
    names_of = {metas[k]["filename"]: k for k in want}
    items = {}
    for f in sorted((Path("/out") / STAGED / "inputs").glob("shard_*.jsonl")):
        for line in f.open():
            if any(k in line for k in want) or any(n in line for n in names_of):
                it = json.loads(line)
                k = it.get("key") if it.get("key") in want else names_of.get(it.get("filename"))
                if k is not None:
                    items[k] = (it, f.name)
        if len(items) == len(want):
            break
    tok = AutoTokenizer.from_pretrained(QWEN3_32B["name"], revision=QWEN3_32B["revision"])
    tried = []
    for key in order[:20]:
        if key not in items:
            raise SystemExit(f"{key}: not found in the staged inputs")
        item, shard = items[key]
        r = segment_end_positions(tok, item["messages"], MAX_TOKENS)
        meta = metas[key]
        mine = [(p["step"], p["action"], p["token"]) for p in r["positions"] if p["kind"] == "output_end"]
        theirs = [(p["step"], p["action"], p["token"]) for p in meta["positions"] if p["kind"] == "output_end"]
        agree = {"n_tokens": r["n_tokens"] == meta["n_tokens"], "truncated": r["truncated"] == meta["truncated"],
                 "output_end_tokens": mine == theirs}
        if not all(agree.values()):
            raise SystemExit(f"{key}: re-render disagrees with the extraction's meta: {agree}")
        calls = [m.get("tool_calls") or [] for m in item["messages"] if m["role"] == "assistant"]
        closed = {(q["step"], q["action"]) for q in r["positions"] if q["kind"] == "output_marker" and q["kept"]}
        cands = [p for p in r["positions"] if p["kind"] == "output_end" and p["kept"] and p["tool"] == "bash"
                 and (p["step"], p["action"]) in closed  # its closing </tool_response> is inside the cut too
                 and re.match(r"\s*ls\b", command_of(calls[p["step"]][p["action"]])[0])]
        tried.append({"key": key, "rerender_matches_meta": agree, "ls_outputs_kept": len(cands)})
        if cands:
            p = cands[0]  # the run's first
            break
    else:
        raise SystemExit("none of the first 20 seeded honest runs has a kept, non-empty ls output")
    ids, names = r["ids"], tok.convert_ids_to_tokens(r["ids"])
    i0, i1 = p["span"]
    opener = max(i for i in range(i0) if names[i] == "<tool_response>")
    closer = next(i for i in range(i1, len(ids)) if names[i] == "</tool_response>")
    content = tok.decode(ids[i0:i1])
    lines = content.split("\n")
    shown = "\n".join(lines[:LINES])
    tail = tok.decode(ids[i1:closer + 1]) if len(lines) <= LINES else f"\n… ({len(lines) - LINES} more lines)\n" + tok.decode(ids[closer:closer + 1])
    excerpt = tok.decode(ids[opener:i0]) + shown + tail
    return {"counts": counts, "runs_tried": tried, "run": {"key": key, "environment": meta["environment"], "main_task": meta["main_task"],
                                      "staged_shard": shard, "n_tokens": r["n_tokens"], "truncated": r["truncated"]},
            "rerender_matches_meta": agree,
            "excerpt_output": {"step": p["step"], "action": p["action"], "tool": p["tool"],
                               "selected_as": "the run's first kept, non-empty bash ls output",
                               "tool_response_token": opener, "content_tokens": [i0, i1], "n_content_tokens": i1 - i0,
                               "content_lines": len(lines), "readout_output_end_token": p["token"],
                               "readout_token_text": p["token_text"], "close_token": closer, "inside_cut": p["kept"]},
            "excerpt": excerpt}


@app.local_entrypoint()
def main():
    here = Path(__file__).parent
    res = check.remote()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%SZ")
    excerpt = res.pop("excerpt")
    (here / "data").mkdir(exist_ok=True)
    (here / "data" / f"excerpt_{stamp}.txt").write_text(excerpt)  # transcript text: git-ignored data/, then S3 only
    res |= {"excerpt_sha256": hashlib.sha256(excerpt.encode()).hexdigest(), "excerpt_file": f"data/excerpt_{stamp}.txt",
            "seed": SEED, "arm": ARM, "staged_inputs": STAGED, "dump": DUMP, "max_tokens": MAX_TOKENS,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with open(here / "results" / f"oldqwen_tool_outputs_{stamp}.json", "x") as fh:
        fh.write(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    print("----- excerpt (not saved in git) -----")
    print(excerpt)
