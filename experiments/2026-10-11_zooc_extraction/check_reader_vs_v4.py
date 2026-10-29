"""Offline check: does today's reader give the v4 dump's readout positions on the staged zoo C texts?

The v4 Qwen3.5 dump was read with `src/probe_shift/reader.py` at a0a4cda (unchanged through 8e4d6d6, the run's commit).
Today's reader has the call_end fix (6c6fc56): a call whose last argument value has no token of its own now gets a
call_end (the last argument that has one, else the function name), where v4's reader read none. This runs both readers
with the Qwen3.5 tokenizer (pinned revision) on every staged transcript and checks: the token ids are identical; every
position other than those call_ends is identical (kind, step, action, tool, token, kept); and lists, by IDs only, each
call_end v4's reader would not have read, so an analysis can drop them to match v4 exactly. Laptop CPU, tokenizer only.

  uv run --no-project --python 3.14 --with 'transformers==5.19.0' --with jinja2 python \
      experiments/2026-10-11_zooc_extraction/check_reader_vs_v4.py --inputs <local staged inputs dir>
"""

import argparse
import importlib.util
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from probe_shift import reader as new_reader  # noqa: E402
from probe_shift.models import QWEN35_27B  # noqa: E402

V4_READER = "a0a4cda"
FIELDS = ("kind", "step", "action", "tool", "token", "kept")


def old_reader():
    src = subprocess.run(["git", "show", f"{V4_READER}:src/probe_shift/reader.py"], capture_output=True, text=True, check=True, cwd=ROOT).stdout
    spec = importlib.util.spec_from_loader("reader_v4", loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(src, f"reader.py@{V4_READER}", "exec"), mod.__dict__)
    return mod


def main(inputs: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(QWEN35_27B["name"], revision=QWEN35_27B["revision"])
    old = old_reader()
    items = [json.loads(line) for f in sorted(Path(inputs).glob("shard_*.jsonl")) for line in f.read_text().splitlines()]
    c, moved, mismatched = Counter(), [], []
    for it in items:
        a = old.segment_end_positions(tok, it["messages"], None)
        b = new_reader.segment_end_positions(tok, it["messages"], None)
        if a["ids"] != b["ids"]:
            mismatched.append({"key": it["key"], "why": "token ids differ"})
            continue
        pa, pb = [tuple(p.get(k) for k in FIELDS) for p in a["positions"]], [tuple(p.get(k) for k in FIELDS) for p in b["positions"]]
        if len(pa) != len(pb):
            mismatched.append({"key": it["key"], "why": f"{len(pa)} vs {len(pb)} positions"})
            continue
        c["transcripts"] += 1
        for x, y in zip(pa, pb):
            c[f"positions {y[0]}"] += 1
            if x == y:
                continue
            if x[:4] == y[:4] and y[0] == "call_end" and x[4] is None and y[4] is not None:  # v4 read no call_end here; today's reader does
                c["call_end read now, not by v4"] += 1
                moved.append({"key": it["key"], "step": y[1], "action": y[2], "tool": y[3], "token": y[4]})
            else:
                mismatched.append({"key": it["key"], "why": f"{y[0]} differs", "step": y[1], "action": y[2]})
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    out = {"written": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S"), "git": git, "v4_reader": V4_READER,
           "tokenizer": QWEN35_27B, "staged": len(items), "counts": dict(c), "mismatched": mismatched, "call_end_not_in_v4": moved}
    path = HERE / "results" / f"reader_vs_v4_{out['written']}.json"
    path.parent.mkdir(exist_ok=True)
    with path.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("call_end_not_in_v4", "mismatched")} | {"mismatched": len(mismatched)}, indent=1), "\n->", path)
    if mismatched or not items:  # a launch gate: anything other than the known call_end fix fails loudly
        raise SystemExit(f"{len(mismatched)} mismatches over {len(items)} staged transcripts")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    main(**vars(ap.parse_args()))
