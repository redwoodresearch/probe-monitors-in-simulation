"""Probe step B (PLAN.md amendment 2; conditional): stage one extraction's inputs per level holding every arm B reads,
so all of B's pooled readouts come from one forward-pass run. Modal CPU. Prepared, run only if B's trigger fires.

  level 0: real-attack-e1 and real-attack-e2 from Q1's level-0 inputs (`2026-10-06_la_segment-end_qwen3-32b_v1`),
           sim-attack-fixed from probe step A's staged level-0 inputs;
  level 1: the same from Q1's level-1 inputs (`2026-10-06_la_segment-end-scrub1_qwen3-32b_v1`) and A's level-1 inputs,
           plus A's planted copies (sim-attack-fixed-plant1, -plant2; 150 each), added because B was triggered by
           planted check 1, so B shows whether every-token readouts catch the plant.
Only the 939 paired cells; never overwrites; manifest with each file's sha256.

  cd <worktree> && uv run modal run experiments/2026-10-09_fixed_sim/run_stage_probe_inputs_b_modal.py
"""

import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REPO = HERE.parents[1] if modal.is_local() else Path("/root")
NAME = "probe-shift_aswin-surya_sub-fixed-sim_kill-2026-11-09"
SOURCES = {  # level -> (Q1's inputs for the real arms, A's staged inputs for the fixed sim, B's staged run)
    0: ("2026-10-06_la_segment-end_qwen3-32b_v1", "2026-10-10_la_fixed-sim-v2_probe-inputs-l0_v1", "2026-10-10_la_fixed-sim-v2_probe-inputs-b-l0_v1"),
    1: ("2026-10-06_la_segment-end-scrub1_qwen3-32b_v1", "2026-10-10_la_fixed-sim-v2_probe-inputs-l1_v1", "2026-10-10_la_fixed-sim-v2_probe-inputs-b-l1_v1"),
}

app = modal.App(NAME)
acts = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations")


def rows_of(run: str) -> list[dict]:
    return [json.loads(l) for f in sorted(Path(f"/acts/{run}/inputs").glob("shard_*.jsonl")) for l in f.read_text().splitlines()]


@app.function(image=modal.Image.debian_slim(python_version="3.13"), volumes={"/acts": acts}, cpu=4, memory=32768, timeout=1800, retries=0)
def stage(cells: list[str]) -> dict:
    cells, out = set(cells), {}
    for level, (q1, a, run) in SOURCES.items():
        assert not Path(f"/acts/{run}").exists(), f"{run} exists; staging is done once"
        real = [r for r in rows_of(q1) if r["arm"] in ("real-attack-e1", "real-attack-e2") and r["cell"] in cells]
        keep = {"sim-attack-fixed"} | ({"sim-attack-fixed-plant1", "sim-attack-fixed-plant2"} if level == 1 else set())
        sim = [r for r in rows_of(a) if r["arm"] in keep]
        items = real + sim
        arms = Counter(r["arm"] for r in items)
        want = {"real-attack-e1": 939, "real-attack-e2": 939, "sim-attack-fixed": 939}
        assert arms == (want | ({"sim-attack-fixed-plant1": 150, "sim-attack-fixed-plant2": 150} if level == 1 else {})), arms
        d = Path(f"/acts/{run}/inputs")
        d.mkdir(parents=True)
        files = {}
        for s in range(8):
            data = "".join(json.dumps(x) + "\n" for x in items[s::8]).encode()
            with open(d / f"shard_{s}.jsonl", "xb") as fh:
                fh.write(data)
            files[f"inputs/shard_{s}.jsonl"] = hashlib.sha256(data).hexdigest()
        with open(f"/acts/{run}/manifest.json", "x") as fh:
            fh.write(json.dumps({"run": run, "level": level, "from": {"real": q1, "fixed_sim": a}, "arms": dict(arms), "files": files}, indent=1))
        out[run] = {"arms": dict(arms), "files": files}
    acts.commit()
    return out


@app.local_entrypoint()
def main():
    cells = json.loads((REPO / "splits" / "exclusions_v4.json").read_text())["paired_attack_cells"]["cells"]
    res = stage.remote(cells)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"stage_probe_inputs_b_{stamp}.json"
    with open(path, "x") as f:
        f.write(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print(path)
