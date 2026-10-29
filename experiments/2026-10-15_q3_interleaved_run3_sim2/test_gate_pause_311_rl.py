"""gate_pause_311_rl.py changes ONLY this guard's pause counts (09, d9 04:12): never admission, attrition or length checks.
(1) The only side effects in the file are the pause gate's own: PAUSE_SIMS, GATE_PAUSE.txt and its watch log; it never
writes launch.json, config, caps or anything the launcher's admission reads. (2) infra_rl is true only under Append 35:
a simulated run with attempts, every attempt with terminal sample errors, all RateLimitError. (3) counts() moves such runs
out of ended/usable and changes nothing else.   python3 test_gate_pause_311_rl.py
"""

import ast
import re
from collections import Counter
from pathlib import Path

SRC = (Path(__file__).parent / "gate_pause_311_rl.py").read_text()
tree = ast.parse(SRC)
ns = {"Counter": Counter, "re": re, "ENDED": ("done", "no policy call")}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in ("terminal_class", "infra_rl", "arm_of", "run_usable", "counts"):
        exec(compile(ast.Module([node], []), "gate", "exec"), ns)

# (1) side effects: only the pause files and the watch log are written
writes = set(re.findall(r'run / "([A-Za-z_.]+)"', SRC))
assert writes <= {"PAUSE_SIMS", "GATE_PAUSE.txt", "gate_pause_watch.log", "launch.json"}, writes
assert not re.search(r"launch\.json\"\)\.write|cap_usd|worst_case|admission|reserve", SRC.split('"""', 2)[2]), "touches admission or caps"
assert ".write_text" in SRC and SRC.count(".write_text(") == 1 and "GATE_PAUSE.txt\").write_text" in SRC.replace("(run / ", "")

RL = "RetryError[<Future state=finished raised RateLimitError>]"
OTHER = "RetryError[<Future state=finished raised APITimeoutError>]"
sim = lambda errs_per_attempt, state="done": {"arm": "sim_attack", "simulator": "september", "real": False, "task": "t1", "state": state,
                                               "attempts": [{"outcome": {"sample_errors": e, "policy_calls": []}} for e in errs_per_attempt]}
real = {"arm": "attack", "real": True, "task": "t1", "state": "done", "attempts": [{"outcome": {"sample_errors": [RL], "policy_calls": []}}]}

# (2) infra_rl only under Append 35
assert ns["infra_rl"](sim([[RL], [RL, RL]]))
assert not ns["infra_rl"](sim([[RL], [OTHER]]))      # one attempt failed for another reason
assert not ns["infra_rl"](sim([[RL], []])), "an attempt without terminal errors is not pure"
assert not ns["infra_rl"](sim([]))                  # no attempts
assert not ns["infra_rl"](real)                     # real runs never

# (3) counts: the pure-RL run leaves ended/usable; others are counted as before
c = ns["counts"]({"runs": [sim([[RL], [RL]]), sim([[OTHER]]), real]})
assert c["sim_attack@september"] == {"runs": 2, "ended": 1, "usable": 0, "infra_rl": 1}, c
assert c["attack r1"] == {"runs": 1, "ended": 1, "usable": 0, "infra_rl": 0}, c
print("gate_pause_311_rl: pause counts only; Append 35 infra_rl cases pass")
