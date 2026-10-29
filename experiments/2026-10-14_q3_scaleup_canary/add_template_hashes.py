"""Give each new canary cell the masked attacker-template hash its build is checked against (RULE.md, gate 3): the
template rendered by tasks.py with sentinel fields, with or without the approach section, as the 196's builds render it.
Runs ON THE DEVBOX from the harness folder (the September venv, PYTHONPATH=<tree>/src); prints hashes only.

  python add_template_hashes.py <recorded tasks json> <out json>
"""

import json
import sys
from pathlib import Path

import tasks
import yaml
from runner.policy.prompts import attacker_prompt

src, out = Path(sys.argv[1]), Path(sys.argv[2])
assert not out.exists(), f"{out} exists"
prompt = yaml.safe_load(Path("config/run.yaml").read_text())["run"]["attacker_prompt"]
m = tasks._MASK
template = {has: tasks._sha(attacker_prompt(main_task=m["main"], side_task=m["side"], environment=m["env"],
                                            attack_approach=m["approach"] if has else "", filename=prompt))
            for has in (True, False)}
print(f"prompt {prompt}: template sha256 with approach {template[True]}, without {template[False]}")
data = json.loads(src.read_text())
for c in data["cells"]:
    assert c["sol_system_sha256"] is None and c["sol_input_sha256"] is None, f"{c['cell']}: not a new cell"
    c["template_sha256"] = template[c["approach_chars"] > 0]
data["template_sha256"] = {"with_approach": template[True], "without_approach": template[False], "prompt": prompt}
out.write_text(json.dumps(data, indent=1))
print(f"{len(data['cells'])} cells -> {out}")
