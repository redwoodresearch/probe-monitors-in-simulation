"""Anchor follow-up (09's request, 10-20): a keys-only record, written on Modal, of which yardstick-halfwidth reading the addendum 24 run
identified (half_range and / or max_side) and whether it fit every rc1b pair. Read from the run's values file on the output Volume
(sha-pinned). Labels, booleans and counts only; no value leaves the container.

  uv run --no-project --python 3.12 --with modal==1.6.1 modal run anchor_followup24.py
"""

import json
import time
from pathlib import Path

import modal

OUT = "probe-shift_aswin-surya_sub-repro_cleanroom_kill-2026-11-16"
VALUES = ("cleanroom_anchor_values_2026-10-20_135936.json", "8d501fd89a2a3faa860cf378e13a59973030b49fec37602173fa55ecffaf872d")
IDENT_KEY = "halfwidth reading identified (one reading fits every pair)"
PAIR_PREFIX = "24.r2.yardstick_halfwidth."
READINGS = {"half_range", "max_side"}

app = modal.App(OUT)
OUT_VOL = modal.Volume.from_name(OUT, version=2)
image = modal.Image.debian_slim(python_version="3.12")


@app.function(image=image, cpu=1, memory=4096, timeout=300, retries=0, volumes={"/o": OUT_VOL.read_only()})  # 09: read-only
def follow() -> dict:
    import hashlib

    def labels(r):  # only the two pre-stated reading names ever leave the container
        rd = r.get("readings")
        return sorted(x for x in rd if x in READINGS) if isinstance(rd, list) else None

    out, err = None, None
    try:  # R1: the whole body is inside the boundary; nothing but a type name leaves on failure, raised after the except block
        raw = Path("/o", VALUES[0]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != VALUES[1]:
            err = "values sha"
        else:
            rows = json.loads(raw)["rows"]
            rc = [r for r in rows if r.get("family") == "rc1b_24"]
            ident = [r for r in rc if r.get("key") == IDENT_KEY]
            pairs = [r for r in rc if str(r.get("key", "")).startswith(PAIR_PREFIX)]
            per_pair = {r["key"][len(PAIR_PREFIX):]: {"readings": labels(r), "match": r.get("match") is True} for r in pairs}
            out = {"n_ident_rows": len(ident), "identified_readings": labels(ident[0]) if len(ident) == 1 else None,
                   "ident_match": ident[0].get("match") is True if len(ident) == 1 else None,
                   "n_pairs": len(per_pair), "per_pair": per_pair,
                   "n_pairs_hit_both": sum(v["readings"] == sorted(READINGS) for v in per_pair.values()),
                   "all_pairs_match": bool(per_pair) and all(v["match"] for v in per_pair.values())}
    except Exception as e:
        err = type(e).__name__
    if err:
        raise RuntimeError(f"STOP: {err}")
    return out  # Codex P1: provenance constants are attached locally, not sent from the container


@app.local_entrypoint()
def main():
    out = follow.remote()
    out["values_file"] = {"volume": OUT, "path": VALUES[0], "sha256": VALUES[1]}  # the pinned constants, attached on the laptop
    path = Path(__file__).parent / "results" / f"anchor_followup24_{time.strftime('%Y-%m-%d_%H%M%S')}.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps({k: out[k] for k in ("n_ident_rows", "identified_readings", "ident_match", "n_pairs", "n_pairs_hit_both", "all_pairs_match")}))
    print("wrote", path)
