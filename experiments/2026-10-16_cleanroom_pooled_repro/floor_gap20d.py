"""Floor-gap number for Aswin's "check the formula" (0d's ask, 10-20; under $5): from my 20d values file (Volume), over the 12 plotted
Figs 1/6 floor points (pw_shares B at 0.01, the six lead probes x C / R), the max and min of |B|record_convention - B|l3002| in
percentage points (my two readings of the same share: the structural no-row run counted on honest_b, RULE l.3642, vs left out, l.3002),
and the label where the max occurs. One number pair and labels to the laptop; nothing else.

  uv run --no-project --python 3.12 --with modal==1.6.1 modal run floor_gap20d.py
"""

import json
import time
from pathlib import Path

import modal

OUT = "probe-shift_aswin-surya_sub-repro_cleanroom_kill-2026-11-16"
VALUES = ("cleanroom_downstream_values_2026-10-20_122951.json", "21b7b95d7b11a88e7302e9d21a8e6bf8e1666b9560a8ff02f7b1514c46f16422")
SIX = ["sol_a_uncut", "FIBS|uncut", "harmfulness|uncut", "refusal|uncut", "high-stakes|uncut", "own_all_rs_uncut_L48"]

app = modal.App(OUT)
OUT_VOL = modal.Volume.from_name(OUT, version=2)
image = modal.Image.debian_slim(python_version="3.12")


@app.function(image=image, cpu=1, memory=1024, timeout=300, retries=0, volumes={"/o": OUT_VOL})
def gap() -> dict:
    import hashlib

    raw = Path("/o", VALUES[0]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == VALUES[1], "values sha"
    rows = {r["key"]: r for r in json.loads(raw)["rows"] if r.get("family") == "pw_shares"}
    d = {}
    for n in SIX:
        for ro in ("C", "R"):
            k0 = f"shares.{n} {ro}.0.01"
            a, b = rows[f"{k0}.B|record_convention"]["mine"], rows[f"{k0}.B|l3002"]["mine"]
            d[f"{n} {ro}"] = 100.0 * abs(a - b)
    assert len(d) == 12
    k = max(d, key=d.get)
    return {"n": len(d), "max_points": d[k], "max_at": k, "min_points": min(d.values()),
            "n_zero": sum(1 for v in d.values() if v == 0.0)}


@app.local_entrypoint()
def main():
    out = gap.remote()
    path = Path(__file__).parent / "results" / f"floor_gap20d_{time.strftime('%Y-%m-%d_%H%M%S')}.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps(out))
    print("wrote", path)
