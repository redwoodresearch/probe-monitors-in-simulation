"""sim2 gap-fill, part s4r (d9 ~16:30: Aswin "do all 3 right now"; 0d: no-spend prep for N = 66; quote @ 13841c3bc). A
DESCRIPTIVE add-on in the frozen IL design: one simulated attack (sim2) per cell, no real arm; references (honest_a) from
the S4 main that owns each cell (2026-10-14_newmains_s4r, per the s4 source map @ b68ab0a53). Cells: the source map's 61
4r cells, minus those whose honest_a reference is not a done run with an .eval (box4 read, results/sim2gap_refs_s4r_box4_2026-10-15.json).

Writes (never overwriting) config/sim2gap/{tasks_sim2gap_s4r.json, pilot_sim2gap_s4r.yaml}:
- task file: the cells' entries from the new-mains branch's tasks_nm_s4r_fu.json @ 1bbab93a2, verbatim; order = environments
  round-robin (seeded within each environment), so the launcher's 90 s per-environment stagger never serializes a cluster;
- config: config/pilot_il_s4d.yaml (the IL S4 config) with only the header, the tasks line, real_arms ([] = sim only, as the
  sim-honest follow-ups) and the two cap lines changed; cap = 28 x $75 (the launcher's per-sim reserve) = $2,100, so its
  admission capacity is 28 at the start and only falls as spend accrues (d9: <= 95 counted at admission capacity).
  python make_sim2gap.py
v2 (16:52, the 16:49 start aborted: run_pilot.py asserts every cell's "construction" equals the task file's top-level
"construction", default legacy_sol_hash): the task file now carries the follow-up file's top-level "construction"
(new_mains_v1), as the IL nm task files do. `--v2` rewrites ONLY the task file, asserting its cells are unchanged.
  python make_sim2gap.py --v2
"""

import hashlib
import json
import random
import re
import subprocess
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "config" / "sim2gap"
MAP = ("b68ab0a53", "experiments/2026-10-14_q3_overnight_chain/results/s4_source_map/s4_source_map_4r_2026-10-15.json")
FU = ("1bbab93a2", "experiments/2026-10-14_q3_new_mains_launch/config/tasks_nm_s4r_fu.json")
RESERVE, ADMIT = 75, 28
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731


def git_json(commit, path):
    return json.loads(subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, check=True, cwd=HERE).stdout)


def main():
    import sys
    v2 = "--v2" in sys.argv[1:]
    OUT.mkdir(exist_ok=True)
    tf, cf = OUT / "tasks_sim2gap_s4r.json", OUT / "pilot_sim2gap_s4r.yaml"
    for p in (tf, cf):
        assert p.exists() if v2 else not p.exists(), f"{p} {'missing' if v2 else 'exists'}"
    old_cells = json.loads(tf.read_text())["cells"] if v2 else None
    smap, fu = git_json(*MAP), git_json(*FU)
    refs = json.loads((HERE / "results/sim2gap_refs_s4r_box4_2026-10-15.json").read_text())
    assert refs["main_launch_json_sha256"] == smap["launch_json_sha256"]["2026-10-14_newmains_s4r"], "box4 main differs from the source map's pin"
    owned = [c["cell"] for c in smap["cells"] if c["owner"] == "4r" and c["arms"]["honest_a"]["launch"] == "2026-10-14_newmains_s4r"]
    ok = {r["key"] for r in refs["rows"] if r["state"] == "done" and r["eval"] and r["eval_sha256"]}
    assert len(owned) == 61 and {r["key"] for r in refs["rows"]} == set(owned)
    by_key = {c["key"]: c for c in fu["cells"]}
    keep = [k for k in owned if k in ok]
    seed = int(sha(("sim2gap_s4r " + " ".join(sorted(keep))).encode())[:16], 16)
    rng, envs = random.Random(seed), defaultdict(list)
    for k in sorted(keep):
        envs[by_key[k]["env"]].append(k)
    for e in envs:
        rng.shuffle(envs[e])
    order, names = [], sorted(envs)
    while any(envs.values()):
        for e in names:
            if envs[e]:
                order.append(envs[e].pop(0))
    cells = [by_key[k] for k in order]
    assert fu["construction"] == "new_mains_v1" and all(c["construction"] == fu["construction"] for c in cells)
    if v2:
        assert cells == old_cells, "v2 must not change the cells"
    tf.write_text(json.dumps({"launch": "2026-10-15_sim2gap_s4r", "construction": fu["construction"], "status": "descriptive add-on (sim2 gap-fill); not a registered IL launch or replacement",
                              "source": f"origin/aswin/q3-new-mains-launch:{FU[1]} @ {FU[0]} (cells verbatim)", "references_from": "/home/aswin/phase2/nm/runs/2026-10-14_newmains_s4r",
                              "references_pin": f"s4 source map @ {MAP[0]}; launch.json sha256 {refs['main_launch_json_sha256']}", "seed": seed,
                              "order": "environments round-robin; seeded shuffle within each environment", "cells": cells}, indent=1) + "\n")
    cap = ADMIT * RESERVE
    if v2:
        print(json.dumps({"v2": True, "cells": len(cells), "tasks_sha256": sha(tf.read_bytes()), "config_sha256 (unchanged)": sha(cf.read_bytes())}))
        return
    cfg = (HERE / "config/pilot_il_s4d.yaml").read_text()
    body = cfg[cfg.index("tasks: "):]
    body, k1 = re.subn(r"(?m)^tasks: config/tasks_il_s4d\.json.*$", f"tasks: config/tasks_sim2gap_s4r.json   # sim2 gap-fill: {len(cells)} S4 (4r-owned) cells, env round-robin", body)
    body, k2 = re.subn(r"(?m)^real_arms: \[attack\].*$", "real_arms: []   # sim only (as the sim-honest follow-ups); honest_a from --references-from the S4 (4r) main", body)
    body, k3 = re.subn(r"(?m)^  cap_usd: 389 .*$", f"  cap_usd: {cap}   # {ADMIT} x ${RESERVE} reserve: admission capacity {ADMIT} at the start (d9: <= 95 counted at admission capacity)", body)
    body, k4 = re.subn(r"(?m)^  worst_case_limit_usd: 389 .*$", f"  worst_case_limit_usd: {cap}   # = cap: ${RESERVE} reserved per unfinished sim", body)
    assert (k1, k2, k3, k4) == (1, 1, 1, 1), (k1, k2, k3, k4)
    head = ("# SIM2 GAP-FILL, part s4r (experiments/2026-10-15_q3_interleaved_run3_sim2/make_sim2gap.py): per cell ONE simulated attack (sim2),\n"
            "# no real arm; a descriptive add-on, not a registered IL launch. From config/pilot_il_s4d.yaml: only this header, tasks, real_arms and the caps differ.\n")
    cf.write_text(head + body)
    print(json.dumps({"cells": len(cells), "excluded_no_reference": sorted(set(owned) - ok), "cap": cap, "seed": seed,
                      "tasks_sha256": sha(tf.read_bytes()), "config_sha256": sha(cf.read_bytes())}))


if __name__ == "__main__":
    main()
