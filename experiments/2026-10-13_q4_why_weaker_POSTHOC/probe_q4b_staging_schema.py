"""Q4 (B) staging probe (RULE.md Amendment 4, step 0): the schema of claim C's full replay stagings, counts only. Per
(staging, arm): epochs, distinct runs, epochs per run, whether each run's epochs are prefixes of its longest, and the field
names of an epoch and of a call (names only, never values that hold text).

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 modal run experiments/2026-10-13_q4_why_weaker_POSTHOC/probe_q4b_staging_schema.py
"""

import json
from collections import Counter
from pathlib import Path

import modal

APP = "probe-shift_aswin-surya_sub-q4-own-schema_kill-2026-11-13"
ACTS = "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08"
STAGINGS = ([(f"{s}_swapsys", "attack") for s in ("rerun_strict_segends_v2_scrub1", "scaleup_canary_segends_v1_scrub1", "scaleup_l2_segends_v1_scrub1",
                                                  "scaleup_topup_segends_v1_scrub1")] + [("second_attack_segends_v1_scrub1_swapsys", "attack")]
            + [(f"{s}_swapsys2", "sim_attack@september") for s in ("rerun_strict_segends_v2_scrub1", "scaleup_canary_segends_v1_scrub1",
                                                                    "scaleup_l2_segends_v1_scrub1", "scaleup_topup_segends_v1_scrub1")]
            + [(s, h) for s in ("rerun_strict_segends_v2_scrub1", "scaleup_canary_segends_v1_scrub1", "scaleup_l2_segends_v1_scrub1",
                                "scaleup_topup_segends_v1_scrub1") for h in ("honest_a", "honest_b")])

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")


@app.function(image=image, cpu=4, memory=16384, timeout=900, retries=0, volumes={"/acts": modal.Volume.from_name(ACTS).read_only()})
def probe() -> dict:
    import numpy as np

    out = {}
    for staging, arm in STAGINGS:
        p = Path("/acts") / staging / "inputs" / f"{arm}.jsonl"
        if not p.exists():
            out[f"{staging}|{arm}"] = {"missing": True}
            continue
        eps, ep_keys, call_keys, scalar = [], Counter(), Counter(), {}
        for line in p.open():
            ep = json.loads(line)
            ep_keys.update(ep.keys())
            for c in ep.get("calls", [])[:1]:
                call_keys.update(c.keys())
            for k, v in ep.items():
                if k not in ("ids", "calls") and isinstance(v, (str, int, float, bool)) and k not in scalar and len(str(v)) < 80:
                    scalar[k] = type(v).__name__
            eps.append((ep.get("run") or ep.get("key"), ep.get("task"), np.asarray(ep["ids"], np.int32)))
        by = {}
        for r, t, ids in eps:
            by.setdefault((r, t), []).append(ids)
        multi = {k: v for k, v in by.items() if len(v) > 1}
        nonprefix = sum(1 for v in multi.values() if not all(np.array_equal(max(v, key=len)[: len(x)], x) for x in v))
        out[f"{staging}|{arm}"] = {"epochs": len(eps), "runs": len(by), "epochs_per_run": dict(Counter(len(v) for v in by.values())),
                                   "multi_epoch_runs": len(multi), "multi_epoch_runs_not_prefix": nonprefix,
                                   "epoch_fields": sorted(ep_keys), "call_fields": sorted(call_keys), "scalar_field_types": scalar}
    return out


@app.local_entrypoint()
def main():
    print(json.dumps(probe.remote(), indent=1))
