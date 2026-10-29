"""Cell (a) transfer, Amendment 2's full marker counts (POST HOC; one Modal CPU container, counts only, no text): in the
first 4 counted epochs of the same staged replay inputs run_xfer_cpu.py sampled (scaleup_l2 levels 0 and 1, honest A and
attack), per epoch: tokens; "tools>" ([15449, 29]) before the first assistant turn; <think> (248068) and </think> (248069)
counts; think blocks whose inner span holds more than the empty "\\n\\n" (271); <tool_response> (248066) and
</tool_response> (248067) counts; <|im_start|>assistant (248045, 74455) turns.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run experiments/2026-10-13_cell_a_transfer_POSTHOC/run_markers.py
"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-onpol-reply_xfermarkers_kill-2026-11-13"
ACTS = "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08"
MARK = [(b, a) for b in ("scaleup_l2_segends_v1", "scaleup_l2_segends_v1_scrub1") for a in ("honest_a", "attack")]
app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2).read_only()


@app.function(image=modal.Image.debian_slim(python_version="3.12"), cpu=2, memory=8192, timeout=900, retries=0, volumes={"/acts": acts})
def count() -> list:
    out = []
    for s, a in MARK:
        n = 0
        with (Path("/acts") / s / "inputs" / f"{a}.jsonl").open() as fh:
            for line in fh:
                ep = json.loads(line)
                if ep.get("redrawn_because") or ep.get("attempt_kind") == "rerun":
                    continue
                ids = ep["ids"]
                turns = [i for i in range(len(ids) - 1) if ids[i] == 248045 and ids[i + 1] == 74455]
                first = turns[0] if turns else len(ids)
                opens = [i for i, x in enumerate(ids) if x == 248068]
                closes = [i for i, x in enumerate(ids) if x == 248069]
                pairs = list(zip(opens, closes))
                out.append({"staging": s, "arm": a, "tokens": len(ids), "tools_markers_before_first_assistant": sum(ids[i:i + 2] == [15449, 29] for i in range(first)),
                            "assistant_turns": len(turns), "think_open": len(opens), "think_close": len(closes),
                            "think_blocks_nonempty": sum(c - o - 1 > 1 for o, c in pairs), "think_blocks_empty": sum(c - o - 1 == 1 and ids[o + 1] == 271 for o, c in pairs),
                            "tool_response_open": ids.count(248066), "tool_response_close": ids.count(248067)})
                n += 1
                if n == 4:
                    break
    return out


@app.local_entrypoint()
def main():
    res = count.remote()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dest = HERE / "results" / f"markers_POSTHOC_{stamp}.json"
    dest.open("x").write(json.dumps({"label": "POST HOC (RULE.md Amendment 2): marker counts only"} | {"epochs": res}, indent=1))
    for r in res:
        print(r)
    print(dest.name)
