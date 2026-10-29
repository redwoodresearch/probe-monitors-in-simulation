"""Probe step A's reproduction check (PLAN.md amendment 2), on Modal CPU: the 20 real e1 runs re-extracted in each new
dump (arm real-attack-e1-repro) against the same runs in Q1's dump of that level (arm real-attack-e1). Per layer: the
minimum cosine over every output_end row of the 20 runs, and the largest absolute difference. Pass: the same positions
in both, and cosine >= 0.9999 at every layer; otherwise no cross-run comparison is run.

  cd <worktree> && uv run modal run experiments/2026-10-09_fixed_sim/run_repro_check_modal.py
"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
NAME = "probe-shift_aswin-surya_sub-fixed-sim_kill-2026-11-09"
PAIRS = {  # level -> (new dump, Q1's dump)
    0: ("2026-10-10_la_fixed-sim-v2_qwen3-32b_alllayers_v1", "2026-10-06_la_segment-end_qwen3-32b_alllayers_v1"),
    1: ("2026-10-10_la_fixed-sim-v2-scrub1_qwen3-32b_alllayers_v1", "2026-10-06_la_segment-end-scrub1_qwen3-32b_alllayers_v1"),
}
MIN_COSINE = 0.9999

app = modal.App(NAME)
acts = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
image = modal.Image.debian_slim(python_version="3.13").uv_pip_install("numpy")


def rows(run: Path, arm: str, key_to_file: dict, layer: int, cache: dict):
    """{key: {(step, action): vector}} of output_end rows at one layer."""
    import numpy as np

    out = {}
    for key, stem in key_to_file.items():
        m = json.loads((run / "meta" / arm / f"{stem}.json").read_text())
        arr = cache.setdefault((str(run), m["chunk"], layer), np.load(run / "by_layer" / f"L{layer}" / f"{m['chunk']}.npy", mmap_mode="r"))
        out[key] = {(p["step"], p["action"]): np.asarray(arr[m["row_start"] + p["row"]], dtype=np.float64)
                    for p in m["positions"] if p["kind"] == "output_end" and p.get("row") is not None}
    return out


@app.function(image=image, volumes={"/acts": acts}, cpu=4, memory=16384, timeout=1800, retries=0)
def check(level: int) -> dict:
    import numpy as np

    new, old = (Path("/acts") / r for r in PAIRS[level])
    stems = {json.loads(f.read_text())["key"]: f.stem for f in (new / "meta" / "real-attack-e1-repro").glob("*.json")}
    assert len(stems) == 20, f"{len(stems)} reproduction runs in {new.name}"
    layers = sorted(int(p.name[1:]) for p in (new / "by_layer").iterdir())
    res, cache_new, cache_old = {}, {}, {}
    for layer in layers:
        a = rows(new, "real-attack-e1-repro", stems, layer, cache_new)
        b = rows(old, "real-attack-e1", stems, layer, cache_old)
        cos, diff, n = [], 0.0, 0
        for key in stems:
            assert a[key].keys() == b[key].keys(), f"{key}: different output_end positions"
            for pos, x in a[key].items():
                y = b[key][pos]
                cos.append(float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))))
                diff = max(diff, float(np.abs(x - y).max()))
                n += 1
        res[layer] = {"rows": n, "min_cosine": min(cos), "max_abs_diff": diff}
    return {"level": level, "new": new.name, "old": old.name, "layers": res,
            "passes": all(v["min_cosine"] >= MIN_COSINE for v in res.values())}


@app.local_entrypoint()
def main():
    out = [check.remote(level) for level in PAIRS]
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"repro_check_{stamp}.json"
    with open(path, "x") as f:
        f.write(json.dumps(out, indent=1))
    for r in out:
        worst = min(r["layers"].items(), key=lambda kv: kv[1]["min_cosine"])
        print(f"level {r['level']}: passes {r['passes']}; worst layer {worst[0]}: min cosine {worst[1]['min_cosine']:.6f}, "
              f"max abs diff {worst[1]['max_abs_diff']:.4g}, {worst[1]['rows']} rows")
    print(path)
