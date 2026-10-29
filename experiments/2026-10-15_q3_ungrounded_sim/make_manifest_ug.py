"""The ungrounded arm's FROZEN CONFIG MANIFEST (09's Append 34 requirement), written ON q3-xbox1 after build_ug.sh and
build_index_ug.sh and BEFORE any generation; committed before the launch. Read-only: it only reads files and git HEADs.

  PYTHONDONTWRITEBYTECODE=1 ~/sims/simulated-control-evals-sept/.venv/bin/python make_manifest_ug.py --repo-commit <sha> --out <new json>
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # before any other import: nothing below may write __pycache__ (only --out is written)

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import yaml

HOME = Path("/home/aswin/phase2")
HARNESS, TREE, FROZEN_TREE = HOME / "attack_check_ug", HOME / "sims-sept-ungrounded", HOME / "sims-sept-strict-off"
INDEX = HOME / "il_refindex_ug" / "launch.json"
SIMS_PATCH_COMMIT = "e2ece50c3bcc39aec6bf936cdea4b17fd9a45a9e"  # simulated-control-evals, aswin/ungrounded-sim-sept


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-commit", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(HARNESS))
    from run_attack_check import src_sha256  # the launcher's own tree hash

    run_yaml = yaml.safe_load((HARNESS / "config" / "run.yaml").read_text())
    sim_yaml = yaml.safe_load((HARNESS / "config" / "sim.yaml").read_text())
    pilot = yaml.safe_load((HARNESS / "config" / "pilot_ug.yaml").read_text())
    simulator = run_yaml["simulators"]["september_ungrounded"]
    assert pilot["simulators"] == ["september_ungrounded"] and pilot["real_arms"] == [] and pilot["sim_arms"] == ["sim_attack"]
    tree_sha = src_sha256(TREE)
    assert tree_sha == simulator["src_sha256"], (tree_sha, simulator["src_sha256"])
    settings_dir = Path(run_yaml["settings_dir"]).expanduser()
    repos = {}
    for repo in sorted((settings_dir / "linuxarena").iterdir()):
        if (repo / ".git").exists():
            git = ["git", "--no-optional-locks", "-C", str(repo)]  # read-only: no index refresh
            head = subprocess.run([*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
            dirty = subprocess.run([*git, "status", "--porcelain", "--untracked-files=no"], check=True,
                                   capture_output=True, text=True).stdout
            repos[repo.name] = {"head": head, "tracked_clean": not dirty.strip()}
    settings_digest = hashlib.sha256(json.dumps(repos, sort_keys=True).encode()).hexdigest()
    tasks_path = HARNESS / pilot["tasks"]
    cells = json.loads(tasks_path.read_text())["cells"]
    refs = json.loads(INDEX.read_text())["references"]
    per_cell = []
    for c in cells:
        r = refs[c["key"]]
        got = sha(Path(r["copy"]))
        assert got == r["sha256"], (c["key"], got, r["sha256"])
        per_cell.append({"key": c["key"], "env": c["env"], "main": c["main"], "side": c.get("side"),
                         "fork_source": "the cell's real honest_a (BA_REAL_LOG; forked at 0, no action seeded)",
                         "reference_record": r, "reference_sha256_verified": got})
    out = {
        "arm": "sim_attack@september_ungrounded", "world": "sim_ug",
        "spec": "09 Append 34 @ eabb98afa (aswin/convergence-audit-09): G = delta_ug - delta_g on identical paired cells",
        "approval": "Aswin ~00:10 PT 2026-10-15 via 0d: ungrounded simulator, cap $3k raw / $3.9k with 30%",
        "repo_commit": args.repo_commit,
        "sims_patch": {"commit": SIMS_PATCH_COMMIT, "file": "patches/ungrounded_sept/ungrounded_no_honest_reference.patch",
                       "sha256": sha(HARNESS / "config" / "ungrounded_no_honest_reference.patch")},
        "flag": simulator["env"],
        "tree": {"path": str(TREE), "src_sha256": tree_sha, "frozen_base": str(FROZEN_TREE),
                 "frozen_base_src_sha256": src_sha256(FROZEN_TREE)},
        "harness_copy": {"path": str(HARNESS), "run_pilot_sha256": sha(HARNESS / "run_pilot.py"),
                         "check_sim_samples_sha256": sha(HARNESS / "check_sim_samples.py"),
                         "run_yaml_sha256": sha(HARNESS / "config" / "run.yaml"),
                         "manifest_sha256": sha(HARNESS / "config" / "harness_manifest_ug.tsv")},
        "resolved_simulator_env": sim_yaml["env"] | simulator["env"],
        "per_run_env_set_by_launcher": ["LA_ENV", "BA_REAL_LOG = BA_HONEST_LOG = the cell's reference copy",
                                        "BA_MODE=attack, BA_CELLS, PHASE2_CELLS", "server/common variables"],
        "inspect_args": sim_yaml["inspect_args"], "policy_limits": run_yaml["policy_limits"],
        "attacker_prompt": run_yaml["run"]["attacker_prompt"],
        "settings": {"dir": str(settings_dir), "repos": repos, "digest_sha256": settings_digest},
        "config": {"path": str(HARNESS / "config" / "pilot_ug.yaml"), "sha256": sha(HARNESS / "config" / "pilot_ug.yaml"),
                   "spend": pilot["spend"], "max_runs_per_server": pilot["max_runs_per_server"]},
        "tasks_file": {"path": str(tasks_path), "sha256": sha(tasks_path), "n_cells": len(cells)},
        "reference_index": {"path": str(INDEX), "sha256": sha(INDEX)},
        "cells": per_cell,
    }
    with open(args.out, "x") as f:
        json.dump(out, f, indent=1)
    print(f"manifest: {len(per_cell)} cells, tree {tree_sha[:16]}, settings digest {settings_digest[:16]} -> {args.out}")


if __name__ == "__main__":
    main()
