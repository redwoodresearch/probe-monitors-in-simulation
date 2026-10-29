"""POST HOC (RULE.md), step 3: the pipeline check. The 4 unedited simulated honest runs re-extracted now
(2026-10-12_q3_edithonest_smoke_qwen3-32b_l48-63_v1) against the stored v4 all-layers activations of the same runs, row
for row at layers 48 and 63: same positions, and the largest absolute difference per layer. A second read of the same
text, so it is also the determinism check. Modal CPU; numbers only. Gate (RULE.md): identical positions and token
counts, finite values, and at both layers max |difference| / max |stored value| <= 1e-3 (fp16's relative precision; the
prefix checks of 10-08 and 10-11 found differences of exactly 0).

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 modal run experiments/2026-10-12_q3_edit_honest_fa/check_smoke.py \
      --stage results/stage_<stamp>.json
"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q3edithonest_smoke_kill-2026-11-12"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
V4, SMOKE, ARM, LAYERS, TOL = "2026-10-06_la_segment-end-v4_qwen3-32b_alllayers_v1", "2026-10-12_q3_edithonest_smoke_qwen3-32b_l48-63_v1", "sim-honest", (48, 63), 1e-3

app = modal.App(APP)
acts_vol = modal.Volume.from_name(ACTS, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")


@app.function(image=image, volumes={"/acts": acts_vol.read_only()}, cpu=2, memory=16384, timeout=900, retries=0)
def check(keys: list[str]) -> dict:
    import numpy as np

    new_metas = [json.loads(p.read_text()) for p in sorted((Path("/acts") / SMOKE / "meta" / ARM).glob("*.json"))]
    out = {"runs": len(new_metas), "keys_equal_stage_smoke_keys": sorted(m["key"] for m in new_metas) == sorted(keys), "per_run": []}
    for m in new_metas:
        old = json.loads((Path("/acts") / V4 / "meta" / ARM / f"{m['filename']}.json").read_text())
        pos = lambda mm: [(p["kind"], p["step"], p["action"], p["kept"], p.get("token"), p.get("row")) for p in mm["positions"]]
        r = {"key_sha256": __import__("hashlib").sha256(m["key"].encode()).hexdigest()[:12], "n_tokens_equal": m["n_tokens"] == old["n_tokens"],
             "positions_equal": pos(m) == pos(old), "rows": m["n_rows"], "max_abs_diff": {}}
        if r["positions_equal"]:
            for l in LAYERS:
                a = np.load(Path("/acts") / SMOKE / "by_layer" / f"L{l}" / f"{m['chunk']}.npy", mmap_mode="r")[m["row_start"]:m["row_start"] + m["n_rows"]]
                b = np.load(Path("/acts") / V4 / "by_layer" / f"L{l}" / f"{old['chunk']}.npy", mmap_mode="r")[old["row_start"]:old["row_start"] + old["n_rows"]]
                if not a.shape == b.shape == (m["n_rows"], 5120):  # shapes first: a failed record, not a crash
                    r["finite"], r["max_abs_diff"][str(l)], r.setdefault("rel_diff", {})[str(l)] = False, float("inf"), float("inf")
                    continue
                r["max_abs_diff"][str(l)] = float(np.max(np.abs(a.astype(np.float32) - b.astype(np.float32)))) if len(a) else 0.0
                r.setdefault("rel_diff", {})[str(l)] = r["max_abs_diff"][str(l)] / max(float(np.max(np.abs(b.astype(np.float32)))), 1e-12) if len(a) else 0.0
                r.setdefault("finite", True)
                r["finite"] &= bool(np.isfinite(a).all() and np.isfinite(b).all() and a.shape == b.shape == (m["n_rows"], 5120))
        out["per_run"].append(r)
    out["pass"] = bool(len(new_metas) == 4 and out["keys_equal_stage_smoke_keys"] and all(r["positions_equal"] and r["n_tokens_equal"] and r.get("finite") and
                                                  max(r.get("rel_diff", {"none": float("inf")}).values()) <= TOL for r in out["per_run"]))
    out["max_abs_diff_overall"] = {str(l): max((r["max_abs_diff"].get(str(l), float("nan")) for r in out["per_run"]), default=None) for l in LAYERS}
    return out


@app.local_entrypoint()
def main(stage: str):
    res = check.remote(json.loads((HERE / stage).read_text())["smoke_keys"])
    res["stage_record"] = Path(stage).name
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with (HERE / "results" / f"smoke_check_{stamp}.json").open("x") as fh:
        fh.write(json.dumps(res | {"stamp": stamp, "tolerance": TOL}, indent=1))
    print(json.dumps(res, indent=1))
    if not res["pass"]:
        raise SystemExit("smoke check FAILED: stop and report")
