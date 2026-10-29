"""The new-mains splits (experiments/2026-10-14_q3_new_mains_launch/RULE.md, Splits), ON THE DEVBOX after prep_nm.sh: the
cells that passed the preflight (config/nm_built.json) split into S1..S<N> by int(sha256(SALT + cell), 16) % N; within a
split each environment's cells are spread evenly over the file (as the scale-up's make_quarters_311.py). For each split:
config/tasks_nm_s<k>_built.json (cells only), add_r2.py -> tasks_nm_s<k>_launch.json (the interleaved `<key>_r2` second
attack, same launch), the follow-up file tasks_nm_s<k>_fu.json (the r1 entries only, simulated attacks only), and the two
pilot configs (main: real arms, simulated attacks paused by results/PAUSE_SIMS; follow-up: simulated attacks only) with
the split's simulator cap = cells x $16.50 + $300 (d9/0d), $75 worst per run. Keys, hashes and counts only.

  <sept venv python> make_splits_nm.py <n splits> [--small-split <k> --small-max-real 40]
"""

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

SALT = "q3-newmains-v1:"
HERE = Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("n", type=int)
ap.add_argument("--small-split", type=int, default=0, help="a split on a smaller box (sims-astra)")
ap.add_argument("--small-max-real", type=int, default=40)
a = ap.parse_args()
h = lambda s: hashlib.sha256((SALT + s).encode()).hexdigest()  # noqa: E731
src = HERE / "config" / "nm_built.json"
data = json.loads(src.read_text())
cells = data["cells"]
assert data["construction"] == "new_mains_v1" and len(cells) == len({c["cell"] for c in cells}) == len({c["key"] for c in cells})
top = {k: v for k, v in data.items() if k not in ("cells", "dropped_at_preflight", "fixture_sample")}
summary = {"source": src.name, "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(), "salt": SALT, "splits": {}}

MAIN = """# New-mains split S{k} (experiments/2026-10-14_q3_new_mains_launch/RULE.md; Aswin via 0d: "run the 360 new main tasks now,
# it's okay, just run it"). The scale-up's quarter config (pilot_311_q1.yaml) on this split's cells: per cell honest_a (real;
# the fork source), attack and honest_b (real) and sim_attack@september (PAUSED for good by results/PAUSE_SIMS: the
# simulated attacks run only in the follow-up); plus each cell's `<key>_r2` entry (real_arms [attack]): its second real
# attack, in this same launch. construction new_mains_v1.
tasks: config/tasks_nm_s{k}_launch.json   # make_splits_nm.py -> add_r2.py
real_simulator: september
real_arms: [honest_a, attack, honest_b]
simulators: [september]
sim_arms: [sim_attack]
simulator_confirmed: true
max_runs_per_server: 10
max_real_runs: {max_real}
reference_tries: 3
digest_sets: config/digest_sets_tagform_2026-10-08.json
spend:
  cap_usd: {cap}
  worst_case_limit_usd: {cap}
  worst_per_sim_run_usd: 75
  prior_per_sim_run_usd: 16.48
"""
FU = """# FOLLOW-UP (simulated-only) launch of new-mains split S{k} (experiments/2026-10-14_q3_new_mains_launch/RULE.md), as
# the scale-up's variant (iii): references from the main launch (--references-from); the launcher's own box-side admission
# guard bounds simulator spend (cap = cells x $16.50 + $300; $75 worst per run); sim_cap_stop.py pauses at cap + $100.
tasks: config/tasks_nm_s{k}_fu.json   # make_splits_nm.py: the split's r1 entries, simulated attacks only
real_simulator: september
real_arms: []
simulators: [september]
sim_arms: [sim_attack]
simulator_confirmed: true
max_runs_per_server: 10
max_real_runs: {max_real}
reference_tries: 3
digest_sets: config/digest_sets_tagform_2026-10-08.json
spend:
  cap_usd: {cap}
  worst_case_limit_usd: {cap}
  worst_per_sim_run_usd: 75
  prior_per_sim_run_usd: 16.48
"""
for k in range(1, a.n + 1):
    mine = [c for c in cells if int(h(c["cell"]), 16) % a.n == k - 1]
    n_env, rank, keyed = Counter(c["env"] for c in mine), Counter(), []
    for c in sorted(mine, key=lambda c: h(c["cell"])):
        keyed.append(((rank[c["env"]] + 0.5) / n_env[c["env"]], h(c["env"]), c))
        rank[c["env"]] += 1
    ordered = [c for _, _, c in sorted(keyed, key=lambda x: (x[0], x[1]))]
    cfg = HERE / "config"
    built, launch, fu = cfg / f"tasks_nm_s{k}_built.json", cfg / f"tasks_nm_s{k}_launch.json", cfg / f"tasks_nm_s{k}_fu.json"
    for f in (built, launch, fu, cfg / f"pilot_nm_s{k}.yaml", cfg / f"pilot_nm_fu_s{k}.yaml"):
        assert not f.exists(), f"{f} exists"
    built.write_text(json.dumps(dict(top, rule=__doc__.split("\n\n")[0], salt=SALT, split=k, cells=ordered), indent=1))
    subprocess.run([sys.executable, str(HERE / "add_r2.py"), str(built), str(launch)], check=True, stdout=subprocess.DEVNULL)
    d = json.loads(launch.read_text())
    assert d.get("construction") == "new_mains_v1", "add_r2.py dropped the top-level construction"
    r1 = [c for c in d["cells"] if "real_arms" not in c]
    assert len(r1) * 2 == len(d["cells"]) and all(not c["key"].endswith("_r2") for c in r1)
    fu.write_text(json.dumps(dict(d, cells=r1, followup={"from": launch.name, "from_sha256": hashlib.sha256(launch.read_bytes()).hexdigest(),
                                                       "rule": "the split's r1 entries only, unchanged; simulated attacks only"}), indent=1))
    cap = round(len(ordered) * 16.5 + 300)
    max_real = a.small_max_real if k == a.small_split else 60
    (cfg / f"pilot_nm_s{k}.yaml").write_text(MAIN.format(k=k, cap=cap, max_real=max_real))
    (cfg / f"pilot_nm_fu_s{k}.yaml").write_text(FU.format(k=k, cap=cap, max_real=max_real))
    summary["splits"][k] = {"cells": len(ordered), "empty_approach": sum(c["approach_chars"] == 0 for c in ordered),
                            "by_env": dict(sorted(n_env.items())), "sim_cap_usd": cap, "max_real_runs": max_real,
                            "launch_sha256": hashlib.sha256(launch.read_bytes()).hexdigest(),
                            "fu_sha256": hashlib.sha256(fu.read_bytes()).hexdigest()}
assert sum(v["cells"] for v in summary["splits"].values()) == len(cells)
summary["sim_caps_total_usd"] = sum(v["sim_cap_usd"] for v in summary["splits"].values())
(HERE / "results" / "make_splits_nm.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
