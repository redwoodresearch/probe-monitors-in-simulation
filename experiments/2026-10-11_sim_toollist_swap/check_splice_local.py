"""$0 local check of the splice: for each task, a dummy prompt rendered with the simulated tools, with its tool lines
spliced to the real ones by stage_toolswap.splice's rule, must equal the same dummy prompt rendered with the real tools,
token for token (and the reverse for the mirror). Uses only the structured tools field and a fixed dummy system text.

  uv run --no-project --python 3.14 --with transformers --with jinja2 python check_splice_local.py --runs <dir> --lines <tool_lines.json>
"""
import argparse
import json
import zipfile
from pathlib import Path

from transformers import AutoTokenizer

import build_tool_lines as B


def main(runs: str, lines: str):
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B", revision="fc05daec18b0a78c049392ed2e771dde82bdf654")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    msgs = [{"role": "system", "content": "You are a helpful agent. Follow the task."}, {"role": "user", "content": "Do the task."}]
    L = json.loads(Path(lines).read_text())["sets"]
    ok = bad = 0
    for stage, folder in B.SETS.items():
        if stage == "second_attack_segends_v1":
            continue
        for task, v in L[stage].items():
            if "real" not in v or "sim" not in v:
                continue
            tools = {}
            for arm, side in (("attack", "real"), ("sim_attack@september", "sim")):
                evs = sorted((Path(runs) / folder / arm / task).glob("*.eval"))
                tools[side] = next(t for t in (B.tools_of(e) for e in evs) if t)
            ids = {s: tok(tok.apply_chat_template(msgs, tools=tools[s], tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"] for s in tools}
            for src, dst in (("sim", "real"), ("real", "sim")):
                a = {x["name"]: x["ids"] for x in v[src]}; b = {x["name"]: x["ids"] for x in v[dst]}
                cur = ids[src]; end = cur.index(im_end)
                spans = []
                for k in a:
                    if k == "submit" or a[k] == b[k]:
                        continue
                    hits = [i for i in range(end - len(a[k]) + 1) if cur[i:i + len(a[k])] == a[k]]
                    assert len(hits) == 1, (task, k, len(hits))
                    spans.append((hits[0], a[k], b[k]))
                new, prev = [], 0
                for i, x, y in sorted(spans):
                    new += cur[prev:i] + y; prev = i + len(x)
                new += cur[prev:]
                if new == ids[dst]:
                    ok += 1
                else:
                    bad += 1
    print(json.dumps({"im_end_id": im_end, "spliced_equals_rendered": ok, "differs": bad}))
    if bad or not ok:
        raise SystemExit(f"splice check failed: {ok} equal, {bad} differ")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--lines", required=True)
    main(**vars(ap.parse_args()))
