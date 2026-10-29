"""Q4 (B): the epoch map for B's extractions, from the staged inputs on the Volume (keys, arms and tasks only): `map` {epoch key ->
[world, task]} for worlds run1, run2, sim, honest_a, honest_b (task = the 196-task file's cells[].key), `arm_of` {epoch key ->
meta arm folder}, `run_of` {epoch key -> extraction run}. Written to the Volume and returned. One Modal CPU container.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 modal run experiments/2026-10-13_q4_why_weaker_POSTHOC/make_q4b_map.py
"""

import json
from pathlib import Path

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q4own_map_kill-2026-11-13"
VOL = "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13"
SETS = {"q4b_main_v1": ("q4b_extract_main_v1", {"attack": "run1", "sim_attack@september": "sim", "honest_a": "honest_a", "honest_b": "honest_b"}),
        "q4b_run2_v1": ("q4b_extract_run2_v1", {"attack": "run2"})}
app = modal.App(APP)
vol = modal.Volume.from_name(VOL, version=2)


@app.function(image=modal.Image.debian_slim(python_version="3.12"), cpu=2, memory=16384, timeout=900, retries=0, volumes={"/v": vol})
def build() -> dict:
    out = {"map": {}, "arm_of": {}, "run_of": {}}
    for staged, (run, worlds) in SETS.items():
        for arm, world in worlds.items():
            for line in Path(f"/v/{staged}/inputs/{arm}.jsonl").open():
                e = json.loads(line)
                if e["key"] in out["map"]:
                    raise ValueError("duplicate epoch key")
                out["map"][e["key"]] = [world, e["task"]]
                out["arm_of"][e["key"]] = e["arm"]
                out["run_of"][e["key"]] = run
    Path("/v/q4b_epoch_map_v1.json").write_text(json.dumps(out, indent=1))
    vol.commit()
    return out


@app.local_entrypoint()
def main():
    out = build.remote()
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "q4b_epoch_map_v1.json").write_text(json.dumps(out, indent=1) + "\n")
    from collections import Counter
    print(Counter(w for w, _ in out["map"].values()))
