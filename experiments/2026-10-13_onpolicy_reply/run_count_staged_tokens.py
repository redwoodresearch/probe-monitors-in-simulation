"""Staged token counts for pricing the averaged-token extraction (RULE.md's last section): per staging and arm, epochs
and tokens (sum of len(ids)) of the replay inputs, counted and all. One Modal CPU container, counts only. $0-pennies.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run experiments/2026-10-13_onpolicy_reply/run_count_staged_tokens.py
"""

import json
from pathlib import Path

import modal

APP = "probe-shift_aswin-surya_sub-onpol-reply_tokens_kill-2026-11-13"
ACTS = "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08"
BASES = ["rerun_strict_segends_v2", "scaleup_canary_segends_v1", "scaleup_l2_segends_v1", "scaleup_topup_segends_v1"]
STAGINGS = ([(b, a) for b in BASES for a in ("attack", "honest_a", "honest_b", "sim_attack@september")] + [("second_attack_segends_v1", "attack")]
            + [(f"{b}_scrub1", a) for b in BASES for a in ("attack", "honest_a", "honest_b", "sim_attack@september")] + [("second_attack_segends_v1_scrub1", "attack")]
            + [(f"{b}_scrub1_swapsys", "attack") for b in BASES] + [("second_attack_segends_v1_scrub1_swapsys", "attack")]
            + [(f"{b}_scrub1_swapsys2", "sim_attack@september") for b in BASES])
app = modal.App(APP)
vol = modal.Volume.from_name(ACTS, version=2).read_only()


@app.function(image=modal.Image.debian_slim(python_version="3.12"), cpu=4, memory=16384, timeout=1800, retries=0, volumes={"/acts": vol})
def count() -> list:
    out = []
    for s, a in STAGINGS:
        p = Path("/acts") / s / "inputs" / f"{a}.jsonl"
        if not p.exists():
            out.append({"staging": s, "arm": a, "missing": True})
            continue
        n = t = nc = tc = 0
        with p.open() as fh:
            for line in fh:
                ep = json.loads(line)
                k = len(ep["ids"])
                counted = not (ep.get("redrawn_because") or ep.get("attempt_kind") == "rerun")
                n, t, nc, tc = n + 1, t + k, nc + counted, tc + k * counted
        out.append({"staging": s, "arm": a, "epochs": n, "tokens": t, "counted_epochs": nc, "counted_tokens": tc})
    return out


@app.local_entrypoint()
def main():
    res = count.remote()
    print(json.dumps(res))
