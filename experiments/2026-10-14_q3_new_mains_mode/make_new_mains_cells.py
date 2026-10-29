"""The new_mains_v1 cells file and its manifest, from q2's committed record (README_new_mains.md @ 8a3af7b1): it
re-derives q2's seeded pairing (each environment's new mains, in seeded order, dealt round-robin over its side tasks in
seeded order) from the record's own mains and side tasks, and stops unless it equals q2's `proposed_cells` exactly.
Keys, names and hashes only (no approach or prompt text exists here). Laptop, $0; reads git only.

  python make_new_mains_cells.py   (writes config/new_mains_cells_v1.json and results/new_mains_manifest_v1.json)
"""

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
Q2 = ("8a3af7b1e", "experiments/2026-10-14_q2_scaleup_cohort/results/new_mains_counts_2026-10-14_163152.json")
# The 196's masked attacker-template hashes (attacker_2026_08_28.md), as every scale-up build checks them.
TEMPLATE = ("bd321a81a", "experiments/2026-10-14_q3_scaleup_311/config/tasks_311_q1_built.json")
FIXTURE_SALT, FIXTURES_PER_ENV = "new-mains-fixtures-v1:", 2
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731


def show(rev, path):
    return subprocess.run(["git", "show", f"{rev}:{path}"], capture_output=True, check=True, cwd=HERE).stdout


def main():
    raw = show(*Q2)
    q2 = json.loads(raw)
    seed, salt = q2["seed"], q2["salt"]
    proposed = q2["proposed_cells"]
    derived = []
    for env in sorted({c["env"] for c in proposed}):
        mains = sorted((c["main"] for c in proposed if c["env"] == env), key=lambda m: sha(f"{salt}{seed}:{env}:{m}".encode()))
        sides = sorted((k.split("|", 1)[1] for k in q2["approach_by_side_task"] if k.split("|", 1)[0] == env),
                       key=lambda s: sha(f"{salt}{seed}:{env}:{s}".encode()))
        derived += [(env, m, sides[i % len(sides)]) for i, m in enumerate(mains)]
    assert derived == [(c["env"], c["main"], c["side"]) for c in proposed], "q2's pairing doesn't reproduce"
    template = json.loads(show(*TEMPLATE))["template_sha256"]
    cells = []
    for i, c in enumerate(proposed):
        ap = q2["approach_by_side_task"][f"{c['env']}|{c['side']}"]
        assert (ap["approach_sha256"], ap["approach_chars"]) == (c["approach_sha256"], c["approach_chars"]), c
        cells.append({"cell": f"{c['main']}:{c['side']}", "key": f"nm{i:03d}_{c['env']}", "env": c["env"],
                      "main": c["main"], "side": c["side"], "construction": "new_mains_v1",
                      "q2_approach_sha256": ap["approach_sha256"], "q2_approach_chars": ap["approach_chars"],
                      "sol_system_sha256": None, "sol_input_sha256": None,
                      "flags": {k: c[k] for k in ("in_fresh_327", "fresh_honest_ok", "fresh_honest_broken")}})
    assert len({c["cell"] for c in cells}) == len({c["main"] for c in cells}) == len(cells) == 360
    by_env = {}
    for c in cells:
        by_env.setdefault(c["env"], []).append(c["cell"])
    fixtures = sorted(x for cs in by_env.values()
                      for x in sorted(cs, key=lambda x: sha(f"{FIXTURE_SALT}{x}".encode()))[:FIXTURES_PER_ENV])
    head = {"construction": "new_mains_v1", "rule": "experiments/2026-10-14_q3_new_mains_mode/RULE.md; pairing = q2's "
            "seeded round-robin, reproduced; approach = the fleet rule per side task (q2's sha256, re-recorded on the "
            "devbox by record_fleet_approaches.py, which must match it)", "seed": seed, "salt": salt,
            "source": {"q2_record": f"{Q2[1]} @ {Q2[0]}", "q2_record_sha256": sha(raw),
                       "template": f"{TEMPLATE[1]} @ {TEMPLATE[0]}"},
            "template_sha256": template, "fixture_sample": fixtures}
    out = HERE / "config" / "new_mains_cells_v1.json"
    body = json.dumps(head, indent=1)[:-2] + ',\n "cells": [\n' + ",\n".join(json.dumps(c) for c in cells) + "\n ]\n}\n"
    assert json.loads(body)["cells"] == cells
    out.write_text(body)
    manifest = {"what": "new_mains_v1 cells: PREPARED, not launched; provenance, not equivalence validation",
                "cells_file": str(out.relative_to(HERE.parents[1])), "cells_file_sha256": sha(body.encode()),
                "n_cells": len(cells), "pairing_reproduced": f"{len(derived)}/{len(proposed)}",
                "per_env": {e: len(v) for e, v in sorted(by_env.items())},
                "strategy_bearing": sum(c["q2_approach_chars"] > 0 for c in cells),
                "fresh_honest_ok": sum(c["flags"]["fresh_honest_ok"] for c in cells),
                "side_tasks_per_env": dict(sorted(Counter(k.split("|")[0] for k in q2["approach_by_side_task"]).items())),
                **{k: head[k] for k in ("seed", "salt", "source", "template_sha256", "fixture_sample")}}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "new_mains_manifest_v1.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{len(cells)} cells, pairing reproduced {manifest['pairing_reproduced']}, cells file {manifest['cells_file_sha256'][:16]}")


if __name__ == "__main__":
    main()
