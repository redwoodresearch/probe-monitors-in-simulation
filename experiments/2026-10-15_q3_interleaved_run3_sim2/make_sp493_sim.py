"""S_paper493 sim-attack follow-ups, launches L1 (box3) and L2 (box4) (d9 ~17:00-17:45; 09 Append 39 + amendment 1; 09's pin
ruling ~17:30: the FORK PIN is the main's pinned honest_a reference in its origin's registered launch). Sims only, frozen
IL design, +311 harness. Keys only; no outcome or approach text read.

Cells: the B fork's per-box task entries (tasks_sp493_box{A,B}.json @ 98364da05, aswin/q3-sp493), VERBATIM, for the 455 cells
of results/sp493_sim_refpins_check_2026-10-15.json with status "pinned" (t699_model_registry goes to L3: 09). Split:
- L1 (box3): the +311 cells whose origin main is scaleup311_q2 or _q3 (their reference copies are on box3) + half the T196 cells;
- L2 (box4): the +311 cells of scaleup311_q1 or _q4 (copies on box4) + the other T196 half (environment-balanced: within each
  environment, cells sorted by sha256(key), alternately L1 / L2).
T196 copies (il_refindex_old's, under /home/aswin/phase2/refs_old/) reach box3/box4 by relay_allowlist.sh.
Writes (never overwriting) config/sp493sim/:
- tasks_sp493_sim_L<k>_in.json: the cells (on-box prep_sp493_sim.sh adds approach_sha256/template_sha256 and drops cells that
  fail the no-model-call preflight, as the B fork's prep);
- refindex_L<k>.json: the launch's reference index ({"finished", "references": {cell key: {"copy", "sha256", ...}}}), every copy
  a path on that box whose sha256 = the pin;
- pilot_sp493_sim_L<k>.yaml: config/completion/pilot_ilc_old.yaml with only the header, tasks, real_arms ([]), max_runs_per_server
  (5: 09's OK, concurrency bounded independently of the cap) and the caps (cells x $13.48 x 1.3 + $75, rounded up) changed;
- freeze_exempt_box3.tsv / freeze_exempt_box4.tsv: the exact-path exemptions 09 accepted (pinned sha256).
  python make_sp493_sim.py
"""

import hashlib
import json
import math
import re
import subprocess
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "config" / "sp493sim"
BTASKS = ("98364da05", "experiments/2026-10-15_q3_sp493/config/tasks_sp493_box{}.json")
PINS = HERE / "results/sp493_sim_refpins_check_2026-10-15.json"
MEAN, BUF, RESERVE = 13.48, 1.3, 75
L1_MAINS = {"2026-10-14_154401_scaleup311_q2", "2026-10-14_155838_scaleup311_q3"}
L2_MAINS = {"2026-10-14_154401_scaleup311_q1", "2026-10-14_160246_scaleup311_q4"}
EXEMPT = {  # 09: box3 via d9 ~17:30 (the 4 staged sim-honest configs); box4 option (b) @ e1849823d with box4's values (~17:50)
    "box3": None,  # filled from the box read below (results/sp493_box3_exempt_read_2026-10-15.json)
    "box4": [("./LAUNCH_COMMIT", None), ("./config/pilot_311_q4.yaml", None), ("./config/tasks_311_q4_launch.json", None),
             ("./config/tasks_311_q4c_built.json", None), ("./config/tasks_311_q4c_launch.json", None)],
}
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731


def git_json(commit, path):
    return json.loads(subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, check=True, cwd=HERE).stdout)


def main():
    OUT.mkdir(exist_ok=True)
    pins = json.loads(PINS.read_text())
    rows = {r["key"]: r for r in pins["rows"] if r["status"] == "pinned"}
    cells = {}
    for b in ("A", "B"):
        for c in git_json(BTASKS[0], BTASKS[1].format(b))["cells"]:
            assert c["key"] not in cells
            cells[c["key"]] = c
    assert set(rows) <= set(cells), sorted(set(rows) - set(cells))[:5]
    split = {"L1": [], "L2": []}
    t196 = defaultdict(list)
    for k, r in rows.items():
        if r["main_origin"] == "p311":
            split["L1" if r["source_main"] in L1_MAINS else "L2"].append(k)
            assert r["source_main"] in L1_MAINS | L2_MAINS, r
        else:
            t196[cells[k]["env"]].append(k)
    for env in sorted(t196):
        for i, k in enumerate(sorted(t196[env], key=lambda k: sha(k.encode()))):
            split["L1" if i % 2 == 0 else "L2"].append(k)
    box = {"L1": "box3", "L2": "box4"}
    summary = {}
    for L, keys in split.items():
        keys = sorted(keys, key=lambda k: (cells[k]["env"], sha(k.encode())))
        tf, rf, cf = OUT / f"tasks_sp493_sim_{L}_in.json", OUT / f"refindex_{L}.json", OUT / f"pilot_sp493_sim_{L}.yaml"
        for p in (tf, rf, cf):
            assert not p.exists(), f"{p} exists"
        tf.write_text(json.dumps({"launch": f"2026-10-15_sp493_sim_{L}", "status": "S_paper493 sim-attack follow-up (09 Append 39 + amendment 1); sims only",
                                  "source": f"aswin/q3-sp493:{BTASKS[1].format('A|B')} @ {BTASKS[0]} (entries verbatim)", "box": box[L],
                                  "order": "environment, then sha256(key)", "cells": [cells[k] for k in keys]}, indent=1) + "\n")
        refs = {k: {"copy": rows[k]["copy"], "sha256": rows[k]["pin_sha256"], "origin_key": rows[k]["origin_key"], "source_main": rows[k]["source_main"]} for k in keys}
        rf.write_text(json.dumps({"launch": f"sp493_sim_refindex_{L}", "finished": pins["read_pt"],
                                  "what": "reference index for --references-from (fork pins by key: results/sp493_sim_refpins_check_2026-10-15.json)",
                                  "references": refs}, indent=1) + "\n")
        n = len(keys)
        cap = math.ceil(n * MEAN * BUF + RESERVE)
        cfg = (HERE / "config/completion/pilot_ilc_old.yaml").read_text()
        body = cfg[cfg.index("tasks: "):]
        subs = [(r"(?m)^tasks: config/tasks_ilc_old\.json.*$", f"tasks: config/tasks_sp493_sim_{L}.json   # S_paper493 sim attack {L}: the on-box prep's built cells (<= {n})"),
                (r"(?m)^real_arms: \[attack\].*$", "real_arms: []   # sims only; honest_a from --references-from this launch's reference index (fork pins by key)"),
                (r"(?m)^max_runs_per_server: 10 .*$", "max_runs_per_server: 5   # 09: concurrency bounded independently of the cap; 6 servers x 5 = 30 per launch"),
                (r"(?m)^  cap_usd: 1656 .*$", f"  cap_usd: {cap}   # {n} cells x ${MEAN} x 1.3 + ${RESERVE}"),
                (r"(?m)^  worst_case_limit_usd: 1656 .*$", f"  worst_case_limit_usd: {cap}   # = cap")]
        for pat, rep in subs:
            body, k = re.subn(pat, rep, body)
            assert k == 1, pat
        cf.write_text(f"# S_PAPER493 SIM ATTACK {L} ({box[L]}; make_sp493_sim.py): sims only, frozen IL design. From config/completion/pilot_ilc_old.yaml: only\n"
                      "# this header, tasks, real_arms, max_runs_per_server and the caps differ.\n" + body)
        summary[L] = {"cells": n, "p311": sum(rows[k]["main_origin"] == "p311" for k in keys), "t196": sum(rows[k]["main_origin"] == "t196" for k in keys),
                      "cap": cap, "tasks_in_sha256": sha(tf.read_bytes()), "refindex_sha256": sha(rf.read_bytes()), "config_sha256": sha(cf.read_bytes())}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
