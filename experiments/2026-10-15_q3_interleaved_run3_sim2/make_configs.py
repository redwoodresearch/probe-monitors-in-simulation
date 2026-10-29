"""Write each interleaved launch's pilot config from its follow-up's frozen config (RULE.md, "Configs").

Only three things change from the follow-up config: `tasks` (this launch's task file), `real_arms: [attack]` (run3; the
honest_a come from --references-from), and the two spend lines (cap = worst-case limit = cells x rate). Run from the repo
root: python3 experiments/2026-10-15_q3_interleaved_run3_sim2/make_configs.py
"""

from __future__ import annotations

import json
import math
import subprocess
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
SCALEUP = "origin/aswin/q3-scaleup-311:experiments/2026-10-14_q3_scaleup_311/config/pilot_311_fu_q{k}_iii.yaml"
NEWMAINS = "origin/aswin/q3-new-mains-launch:experiments/2026-10-14_q3_new_mains_launch/config/pilot_nm_fu_s{k}.yaml"
RATE = {"pilot": 28.0, "est": 20.5, "full": 20.5}  # $ per cell (RULE.md, Spend)
APPROVAL = 'Aswin ~21:12 via 0d: "(b) 507 + new mains, ~$20k"'


def write(name: str, src: str, tasks: str, cells: int, rate: float) -> int:
    rev = src.split(":", 1)[0]
    commit = subprocess.check_output(
        ["git", "rev-parse", f"{rev}^{{commit}}"], text=True
    ).strip()
    text = subprocess.check_output(
        ["git", "show", f"{commit}:{src.split(':', 1)[1]}"], text=True
    )
    cap, n = math.floor(cells * rate), Counter()
    out = []
    for line in text.splitlines():
        key = line.split("#")[0].strip().split(":")[0]
        if key == "tasks" and not line.startswith(" "):
            line, n["tasks"] = (
                f"tasks: config/{tasks}   # this launch's {cells} cells (tasks_il_manifest*.json: sha256)",
                n["tasks"] + 1,
            )
        elif key == "real_arms":
            line, n["real"] = (
                "real_arms: [attack]   # run3; honest_a from --references-from <the main's run dir>",
                n["real"] + 1,
            )
        elif key == "cap_usd":
            line, n["cap"] = (
                f"  cap_usd: {cap}   # {cells} cells x ${rate} ({APPROVAL})",
                n["cap"] + 1,
            )
        elif key == "worst_case_limit_usd":
            line, n["wc"] = (
                f"  worst_case_limit_usd: {cap}   # = cap: $75 reserved per unfinished sim",
                n["wc"] + 1,
            )
        out.append(line)
    assert dict(n) == {"tasks": 1, "real": 1, "cap": 1, "wc": 1}, (name, n)
    head = [
        f"# INTERLEAVED launch {name} (experiments/2026-10-15_q3_interleaved_run3_sim2/RULE.md): per cell one real attack (run3)",
        f"# and one simulated attack (sim2), one launch, same servers. From {src.split(':', 1)[1]} @ {commit[:12]}.",
    ]
    (HERE / "config" / f"pilot_il_{name}.yaml").write_text("\n".join(head + out) + "\n")
    return cap


stages = json.loads((HERE / "config" / "tasks_il_stages_manifest.json").read_text())[
    "stages"
]
launches = json.loads((HERE / "config" / "tasks_il_manifest.json").read_text())[
    "launches"
]
caps = {}
for k in "1234":
    for st in ("pilot", "est"):
        caps[f"q{k}_{st}"] = write(
            f"q{k}_{st}",
            SCALEUP.format(k=k),
            f"tasks_il_q{k}_{st}.json",
            stages[f"q{k}_{st}"]["cells"],
            RATE[st],
        )
for k in "12345":
    caps[f"s{k}"] = write(
        f"s{k}",
        NEWMAINS.format(k=k),
        f"tasks_il_s{k}.json",
        launches[f"s{k}"]["cells"],
        RATE["full"],
    )
print(json.dumps(caps), "total", sum(caps.values()))
