"""Pricing input ($0, metadata only, no transcript read): how many of the 198 independent simulated rerolls
(sce experiments/2026-09-29_monitor_score_variance/staged_rerolls.json) fall on claim B's paired attack cells
(splits/exclusions_v3.json: valid in real e1, real e2 and the September Astra sim), split by claim B's dev/test
environments (experiments/2026-10-09_fixed_sim/split.json) and by the first sim's reference tag (base200 / overlap800).
The reroll's environment is read from its staged file name; its cell is the staged row's `cell` (main:side).

  uv run --no-project --python 3.12 python experiments/2026-10-11_claimB_yardsticks/count_reroll_overlap.py \
      ../simulated-control-evals
"""

import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
REPO = HERE.parents[1]
SCE = Path(sys.argv[1]).resolve()
STAGED = SCE / "experiments/2026-09-29_monitor_score_variance/staged_rerolls.json"
EXCL = REPO / "splits/exclusions_v3.json"
SPLIT = REPO / "experiments/2026-10-09_fixed_sim/split.json"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(cwd: Path) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=cwd).stdout.strip()


def main():
    staged = json.loads(STAGED.read_text())
    excl = json.loads(EXCL.read_text())
    split = json.loads(SPLIT.read_text())
    test = set(split["test"])
    reroll = {}
    for row in staged:
        m = re.search(r"_la-([a-z-]+?)-sim-reroll-\d+_", Path(row["source"]).name)
        if not m:
            raise ValueError(f"no environment in {row['source']}")
        reroll[row["cell"]] = m.group(1).replace("-", "_")
    if len(reroll) != len(staged) or len(staged) != 198:
        raise ValueError(f"expected 198 distinct cells, got {len(reroll)} of {len(staged)}")
    paired = set(excl["paired_attack_cells"]["cells"])
    under32k = set(excl["paired_attack_cells_under_32k"]["cells"])
    sims = [s for s in excl["samples"] if s["arm"] == "sim-attack"]
    sim_env = {s["cell"]: s["environment"] for s in sims}
    sim_ref = {s["cell"]: s.get("sim_reference") for s in sims}
    if len(sim_env) != len(sims):
        raise ValueError("duplicate sim cells")
    env_mismatch = sum(1 for c, e in reroll.items() if c in sim_env and sim_env[c] != e)
    inp = [c for c in reroll if c in paired]
    out = {
        "label": "pricing input: reroll cells vs claim B's paired cells (metadata only, counts)",
        "reroll_cells": len(reroll),
        "reroll_by_env": dict(sorted(Counter(reroll.values()).items())),
        "reroll_cells_in_sim_n1000": sum(1 for c in reroll if c in sim_env),
        "env_mismatch_vs_sim_n1000": env_mismatch,
        "in_claimB_paired_939": len(inp),
        "in_paired_under_32k": sum(1 for c in inp if c in under32k),
        "in_paired_test_envs": sum(1 for c in inp if reroll[c] in test),
        "in_paired_dev_envs": sum(1 for c in inp if reroll[c] not in test),
        "in_paired_by_env": dict(sorted(Counter(reroll[c] for c in inp).items())),
        "in_paired_test_main_tasks": len({c.split(":")[0] for c in inp if reroll[c] in test}),
        "first_sim_reference_in_paired": dict(Counter(str(sim_ref.get(c)) for c in inp)),
        "first_sim_reference_in_paired_test": dict(Counter(str(sim_ref.get(c)) for c in inp if reroll[c] in test)),
        "test_envs": sorted(test),
        "inputs": {"staged_rerolls": {"path": str(STAGED.relative_to(SCE)), "sce_git": git(SCE), "sha256": sha(STAGED)},
                   "exclusions_v3": sha(EXCL), "split": sha(SPLIT)},
        "git": git(REPO),
        "script_sha256": sha(Path(__file__)),
    }
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"reroll_overlap_{stamp}.json"
    dst.parent.mkdir(exist_ok=True)
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("inputs",)}, indent=1))
    print("->", dst)


if __name__ == "__main__":
    main()
