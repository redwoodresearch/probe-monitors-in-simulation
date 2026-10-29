"""Zoo C's real Sol honest runs (q2's AMENDMENT_C_GUARD.md @ 52feeae3, #754; Aswin's "All 274 tasks, ~$718 (Recommended)"):
the task file run_pilot.py reads. Two honest_b runs for each of the 274 tasks of q2's batch spec, in its seeded order,
batch tag oct11c, in stages: stage 1 = positions 1-50, then 50 tasks a stage, the last 24 (100 runs a stage, the last
48). Within a stage every task's run 1 comes first, in order, then every run 2, so a task's two runs never wait on each
other's 90 s environment stagger, and run 1 sorts first by start time. Environments come from the Astra honest batch's
reference file, which covers the same 437-task cohort. One more cell, stage "smoke", is the 1-run smoke test: the
first cohort task (sorted) outside the 274, so its run is not part of the batch. Task names only; $0.

  uv run --no-project python make_zoo_c_tasks.py <batch_spec_2026-10-11_151207.json>
"""

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
ORDER_SHA = "b804d4d44f1ea8d716ab30b0bb099e7d9e59ef908d427de013cf3d36b0d00c8b"
REFS = HERE.parents[1] / "experiments" / "2026-10-06_astra_sim_honest" / "results" / "references_20261006-142754.json"
STAGE, TAG = 50, "oct11c"


def main(spec_path: str):
    raw = Path(spec_path).read_bytes()
    order = json.loads(raw)["task order"]
    if len(order) != 274 or len(set(order)) != 274 or hashlib.sha256("\n".join(order).encode()).hexdigest() != ORDER_SHA:
        raise SystemExit("the batch spec's task order is not q2's (274 tasks, sha256 b804d4d4...)")
    refs = json.loads(REFS.read_bytes())["references"]
    env = {t: refs[t]["environment"] for t in order}  # KeyError: a task outside the cohort
    cells, stages = [], {}
    for s, start in enumerate(range(0, len(order), STAGE), 1):
        block = list(enumerate(order[start:start + STAGE], start + 1))
        keys = []
        for k in (1, 2):
            for pos, main in block:
                key = f"{TAG}_p{pos:03d}_r{k}"
                cells.append({"key": key, "env": env[main], "main": main, "position": pos, "run": k, "stage": s,
                              "approach": ""})  # honest runs carry no attack approach
                keys.append(key)
        stages[str(s)] = keys
    smoke = sorted(set(refs) - set(order))[0]  # a cohort task with an overlap800 pair: never one of the batch's tasks
    cells.append({"key": f"{TAG}_smoke", "env": refs[smoke]["environment"], "main": smoke, "position": None, "run": None,
                  "stage": "smoke", "approach": ""})
    stages["smoke"] = [f"{TAG}_smoke"]
    out = {"rule_name": f"zoo_c_{TAG}", "batch_tag": TAG, "runs_per_task": 2,
           "batch_spec": {"file": Path(spec_path).name, "sha256": hashlib.sha256(raw).hexdigest(), "order_sha256": ORDER_SHA},
           "environments_from": {"file": REFS.name, "sha256": hashlib.sha256(REFS.read_bytes()).hexdigest()},
           "stages": stages, "cells": cells}
    # One cell and one stage per line. The file is generated, not committed (549 lines): its sha256 is in the README,
    # and the launch copy is regenerated on the devbox and checked against it.
    head = {k: v for k, v in out.items() if k not in ("stages", "cells")}
    text = (json.dumps(head)[:-1] + ', "stages": {\n' + ",\n".join(f" {json.dumps(s)}: {json.dumps(k)}" for s, k in stages.items())
            + '},\n"cells": [\n' + ",\n".join(json.dumps(c) for c in cells) + "\n]}\n")
    assert json.loads(text) == out
    dst = HERE / "config" / f"pilot_tasks_zoo_c_{TAG}.json"
    with dst.open("x") as fh:
        fh.write(text)
    print({s: len(k) for s, k in stages.items()}, len(cells), "runs; sha256", hashlib.sha256(text.encode()).hexdigest(), "->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("spec_path")
    main(ap.parse_args().spec_path)
