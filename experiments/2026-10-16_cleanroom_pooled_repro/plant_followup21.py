"""PLANT follow-up (09's request, 10-20): a keys-only record, written on Modal, of what the 21e run identified for the excess:
per aggregate excess row, the bootstrap-mean-move reading(s) that reproduced plant_rows (r1 / r2 / r3 labels), and per row whether the
r1 dual verdict equals the record's verdict. Read from the run's values file on the output Volume (sha-pinned). Labels and booleans
only; no value leaves the container. Replaces my disclosed laptop pipe of 12:55.

  uv run --no-project --python 3.12 --with modal==1.6.1 modal run plant_followup21.py
"""

import json
import time
from pathlib import Path

import modal

OUT = "probe-shift_aswin-surya_sub-repro_cleanroom_kill-2026-11-16"
VALUES = ("cleanroom_plant_values_2026-10-20_125308.json", "6e4cda26220c56feea4be6a72f0baef438fd6adfa141d4f6fe063af18398e91a")
PRINTED_EXCESS = {  # 09's map (i): rows 48 / 52 / 56 / 60 = one delta at both levels; 83 / 84 / 85 as listed
    48: ["0.01.C.-3", "0.05.C.-3"], 52: ["0.01.C.-5", "0.05.C.-5"], 56: ["0.01.C.-8", "0.05.C.-8"], 60: ["0.01.C.-12", "0.05.C.-12"],
    83: ["0.01.C.-5", "0.01.C.-8", "0.01.C.-12"], 84: ["0.01.C.-3"], 85: ["0.05.C.-3"]}

app = modal.App(OUT)
OUT_VOL = modal.Volume.from_name(OUT, version=2)
image = modal.Image.debian_slim(python_version="3.12")


@app.function(image=image, cpu=1, memory=2048, timeout=300, retries=0, volumes={"/o": OUT_VOL})
def follow() -> dict:
    import hashlib

    raw = Path("/o", VALUES[0]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == VALUES[1], "values sha"
    rows = json.loads(raw)["rows"]
    move = {r["key"].removesuffix(".excess.move_reading"): sorted(r["mine"]) for r in rows if r["family"] == "aggregate_excess_move"}
    r1 = {r["key"].removesuffix(".excess.verdict|r1"): bool(r["match"]) for r in rows if r["family"] == "aggregate_excess_r1"}
    by_row = {}
    for pr, specs in PRINTED_EXCESS.items():
        keys = [f"aggregate.by_level.{x}" for x in specs]
        by_row[str(pr)] = {"keys": keys, "identified_reading": {k: move.get(k) for k in keys},
                           "r1_verdict_equals_record": {k: r1.get(k) for k in keys}}
    return {"values_file": {"volume": OUT, "path": VALUES[0], "sha256": VALUES[1]}, "n_move_rows": len(move), "n_r1_rows": len(r1),
            "identified_reading": move, "r1_verdict_equals_record": r1, "printed_excess_rows": by_row,
            "all_printed_r1_equal": all(v for b in by_row.values() for v in b["r1_verdict_equals_record"].values())}


@app.local_entrypoint()
def main():
    out = follow.remote()
    path = Path(__file__).parent / "results" / f"plant_followup21_{time.strftime('%Y-%m-%d_%H%M%S')}.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps({k: out[k] for k in ("n_move_rows", "n_r1_rows", "all_printed_r1_equal")}))
    print("wrote", path)
