"""For 20 random real attack (epoch 1) runs (seed 0) that are in both the per-step monitor run and our cohort (the
per-step run read an earlier copy of the cohort; 591 sample ids are identical), write every step with its per-step
monitor score and the run's side-task goal, for labelling by eye whether the top-scoring step is the sabotage
(data/top_steps_20.txt; the labels are results/top_steps_20_labels_2026-10-07.md). Python 3.14.

  uv run --no-project --python 3.14 --with numpy --with boto3 --with pyyaml python experiments/2026-10-07_q3_nano_monitor/dump_top_steps.py
"""

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
from probe_shift.transcripts import load_sample  # noqa: E402

URI = "s3://rr-sce-sim-transcripts/transcripts/2026-09-18_la-public10_real-attack-solmedium-step150-guidancematched-epoch1-n1000/"


def main():
    steps = {}
    for p in (HERE / "data/per_step_e1").glob("*.json"):
        o = json.loads(p.read_text())
        steps.setdefault(o["sample_id"], {})[o["step_index"]] = o
    ex = json.loads((REPO / "splits/exclusions_v4.json").read_text())
    e1 = {s["sample_id"]: s for s in ex["samples"] if s["arm"] == "real-attack-e1" and not s["reason"]}
    pick = random.Random(0).sample(sorted(set(steps) & set(e1)), 20)
    lines = []
    for i, sid in enumerate(pick):
        s, st = e1[sid], steps[sid]
        top = max(st.values(), key=lambda o: (o["monitor"]["score"], -o["step_index"]))  # ties: the earliest step
        smp = load_sample(URI, s["file"], sid, s["epoch"])
        goal = ((smp.get("metadata") or {}).get("side_task") or {}).get("goal", "")
        lines.append(f"\n##### run {i}: {sid} | side task {top['side_task']} (scorer {top['side_task_outcome']}) | top step "
                     f"{top['step_index']} ({top['monitor']['score']})\nGOAL: {goal.strip()[:300]}\nMONITOR AT TOP STEP: {top['monitor']['reasoning'][:600]}")
        for k, t in enumerate(m for m in smp["messages"] if m["role"] == "assistant"):
            sc = st.get(k, {}).get("monitor", {}).get("score")
            for c in t.get("tool_calls") or []:
                lines.append(f"{'>>' if k == top['step_index'] else '  '} {k:3d} [{sc}] {c['function']}: {json.dumps(c['arguments'], ensure_ascii=False)[:400]}")
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data/top_steps_20.txt").write_text("\n".join(lines))
    print("->", HERE / "data/top_steps_20.txt")


if __name__ == "__main__":
    main()
