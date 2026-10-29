"""Write each +311 quarter's simulated-honest pilot config from its follow-up's frozen config (RULE.md, "Configs").

Only these lines change from pilot_311_fu_q<k>_iii.yaml: `sim_arms: [sim_honest]` (was [sim_attack]) and the four spend
lines (cap = worst-case limit = usable cells x $21.13 x 1.3, floored; worst_per_sim_run_usd 106 and prior 21.13, the
simulated-honest continuation's values). `tasks` stays the follow-up's frozen task file (the quarter's r1 entries; a cell
without a usable honest_a reference is recorded "not run" by the launcher, as in the follow-up). Run from the repo root:
python3 experiments/2026-10-15_q3_simhonest_311/make_configs.py
"""

from __future__ import annotations

import math
import subprocess
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
SRC = "origin/aswin/q3-scaleup-311:experiments/2026-10-14_q3_scaleup_311/config/pilot_311_fu_q{k}_iii.yaml"
# usable honest_a references per quarter = the follow-up's simulated runs that ran (scale-up fork, launch.json, 23:12 PT):
# Q1 77, Q2 75 (20 first follow-up + 55 rerun), Q3 72 (t404 has none), Q4 85
USABLE = {"1": 77, "2": 75, "3": 72, "4": 85}
RATE, BUFFER = 21.13, 1.3  # $ per called simulated-honest run (gate12_cont_2026-10-14_160932_simh_cont.json) and +30%
APPROVAL = 'Aswin ~00:05 via 0d: "go for 3"; 0d ~00:1x: +311 first'


def write(k: str) -> int:
    rev, path = SRC.format(k=k).split(":", 1)
    commit = subprocess.check_output(["git", "rev-parse", f"{rev}^{{commit}}"], text=True).strip()
    text = subprocess.check_output(["git", "show", f"{commit}:{path}"], text=True)
    cap, n, out = math.floor(USABLE[k] * RATE * BUFFER), Counter(), []
    for line in text.splitlines():
        key = line.split("#")[0].strip().split(":")[0]
        if key == "sim_arms":
            line, n["arms"] = "sim_arms: [sim_honest]   # simulated honest only (was [sim_attack])", n["arms"] + 1
        elif key == "cap_usd":
            line, n["cap"] = f"  cap_usd: {cap}   # {USABLE[k]} usable cells x ${RATE} x {BUFFER} ({APPROVAL})", n["cap"] + 1
        elif key == "worst_case_limit_usd":
            line, n["wc"] = f"  worst_case_limit_usd: {cap}   # = cap: the box-side admission bound", n["wc"] + 1
        elif key == "worst_per_sim_run_usd":
            line, n["worst"] = "  worst_per_sim_run_usd: 106   # the simulated-honest continuation's value", n["worst"] + 1
        elif key == "prior_per_sim_run_usd":
            line, n["prior"] = f"  prior_per_sim_run_usd: {RATE}   # tonight's continuation mean", n["prior"] + 1
        out.append(line)
    assert dict(n) == {"arms": 1, "cap": 1, "wc": 1, "worst": 1, "prior": 1}, (k, n)
    head = [
        f"# SIMULATED-HONEST launch simh_q{k} (experiments/2026-10-15_q3_simhonest_311/RULE.md): per cell one",
        f"# sim_honest@september forked at 0 from the main's honest_a. From {path} @ {commit[:12]}.",
    ]
    (HERE / "config" / f"pilot_simh_q{k}.yaml").write_text("\n".join(head + out) + "\n")
    return cap


caps = {k: write(k) for k in "1234"}
print(caps, "total", sum(caps.values()))
