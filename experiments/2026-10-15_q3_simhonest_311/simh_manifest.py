"""simh_manifest.py (RULE.md, "Freeze before generation"; 09's Append 25f): ON THE BOX, before quarter k's simulated-honest
launch, write its immutable manifest: what will generate the runs, fixed before any run exists. Read-only: it reads files
and git state, runs no task or simulator code and writes only --out. Run with the September venv's python (PyYAML):

  <sept venv python> simh_manifest.py --k <k> [--host <h>] --config <staged pilot_simh_q<k>.yaml> --out <file>

The laptop then commits --out, and the start script's checks and the launch record are verified against it afterwards. It
exits nonzero (writing nothing) if any reference fails its provenance checks.

Rendered simulator-reference hash. The frozen preflight (check_sim_samples.py) builds each task's simulator prompt but
records only its length (`simulator_prompt_chars`), not a hash, and running it here would rewrite the environment's
generated compose file. So the manifest pins the prompt's INPUTS instead, as `rendered_input_sha256`: the sha256 of a
canonical JSON of the reference copy's sha256, the simulator tree's src_sha256, sim.yaml's and run.yaml's sha256, the
settings digest, the task's digest set and environment, and the honest mode. These are everything the prompt is built
from (check_sim_samples.simulator_prompt). After the launch, the admission evidence adds the launcher's own preflight per
task (built, prompt length, the reasoning marker absent).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

MAINS = {
    "1": "/home/aswin/phase2/runs/2026-10-14_154401_scaleup311_q1",
    "2": "/home/aswin/phase2/runs/2026-10-14_154401_scaleup311_q2",
    "3": "/home/aswin/phase2/runs/2026-10-14_155838_scaleup311_q3",
    "4": "/home/aswin/phase2/runs/2026-10-14_160246_scaleup311_q4",
}
HARNESS = Path("/home/aswin/phase2/attack_check_311")
STAGE = Path("/home/aswin/phase2/simh_stage")
# the three committed frozen 311 harness manifests (origin/aswin/q3-scaleup-311 @ cfdd656c96cd, config/), staged in STAGE
FROZEN = {"harness_manifest_fu_q123.tsv": "71fd27724a045d3df5d8dba9b6e63d5f86514377c025a11a57fd78f6bf1aa287",
          "harness_manifest_fu_q2r.tsv": "b8f813c9a1f4ff4371a981b3aea8a9406b7fa0143eda3212a8b053e8123a4992",
          "harness_manifest_fu_q4.tsv": "32ed72b3e3aa95b21a41ae0dfcf47cacb9efb6e760e9577536e49b57e898bd3d"}
# the Q<k> mains' own task files (scale-up branch = each main box's), staged in STAGE
LAUNCH_TASKS = {"1": "ee413a10eb2c448575fa34a2eafad25428f0947e8ac3450dfbc7e2c7152e4124",
                "2": "eb72f9532e42953fb813d6b226b2eb815c9d161dc3f3e47c5062694de97eaf4d",
                "3": "186633516c694d6087ad2454ee5f8f7cb854dd8bf319c1717682330673a916a9",
                "4": "033d4bebf7bb93c78cada607f698e9ce32e157a6d808adb735d24574d6fc8a2e"}
# 09's Append 25f H_A notes (aswin/convergence-audit-09, priority1_spec_09/RULE.md): the +311 cells whose selected honest_a
# (the fork source) is a later attempt than the stager's counted one, with the selected reference's pinned sha256
SELECTED_PINS = {"t389_model_registry": "d160ffdc956ffc89180043b3575b002d4b0f1cc3f7600082f38decc81f28dae0",
                 "t325_model_registry": "e1cdf8d1594bac78b6e64377557bf63d008f2af0248912f21c7bd60c523b7247",
                 "t309_slack_clone": "69c964700620933d0fbfa771fc83068d208a3e20c48dfad47d443a2ec88f5028"}
RULE = ("each cell's simulated-honest run forks at step 0 from that cell's REAL honest_a run of the quarter's main "
        "launch: the reference the main launch pinned for the task (launch.json `references`), copied verbatim by the "
        "launcher with its sha256 re-checked; a cell with no pinned reference is recorded 'not run'")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def settings_digest(settings: Path) -> dict:
    """Each environment repo's HEAD and clean state, and a sha256 over every non-git file except control-tower's
    generated compose files (.compose-nobuild-*, rewritten by every launch)."""
    repos = {}
    for git in sorted(settings.glob("linuxarena/*/.git")):
        repo = git.parent
        env = {"GIT_OPTIONAL_LOCKS": "0", "PATH": "/usr/bin:/bin"}  # read-only: never refresh the index of a live box
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              env=env)
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True,
                               check=True, env=env)
        repos[repo.name] = {"head": head.stdout.strip(), "dirty_lines": len(dirty.stdout.splitlines())}
    assert repos, f"no environment repos under {settings}"
    rows = []
    for p in sorted(settings.rglob("*")):
        if p.is_file() and ".git" not in p.relative_to(settings).parts and not p.name.startswith(".compose-nobuild-"):
            rows.append(f"{p.relative_to(settings)}\t{sha(p)}")
    return {"repos": repos, "files": len(rows), "sha256": hashlib.sha256("\n".join(rows).encode()).hexdigest()}


def harness_now() -> list[str]:
    """The harness's file hashes exactly as the start script lists them (same exclusions; byte order of './path')."""
    rows = []
    for root, _dirs, files in os.walk(HARNESS, followlinks=False):  # as `find . -type f`: regular files, no symlinks
        for name in files:
            p = Path(root) / name
            if p.is_symlink() or not p.is_file():
                continue
            rel = "./" + str(p.relative_to(HARNESS))
            if (rel.startswith("./results/") or "/__pycache__/" in rel or rel.startswith("./attempt/results/")
                    or rel.startswith("./config/harness_manifest") or rel.startswith("./config/pilot_il")
                    or rel.startswith("./config/tasks_il") or rel.startswith("./config/ilrr")
                    or rel.startswith("./config/pilot_simh_")):
                continue
            rows.append((rel.encode(), f"{sha(p)}  {rel}"))
    return [r for _, r in sorted(rows)]


def eval_task_name(p: Path) -> str:
    from inspect_ai.log import read_eval_log  # .eval members are zstd-compressed: read the header the launcher's way

    return read_eval_log(str(p), header_only=True).eval.task


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", required=True, choices=list(MAINS))
    ap.add_argument("--host", required=True, help="the host's label (box1..box4, xbox1, astra)")
    ap.add_argument("--config", type=Path, required=True)

    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    k, host = args.k, args.host
    main_dir = Path(MAINS[k])
    cfg = yaml.safe_load(args.config.read_text())
    run_yaml, sim_yaml = HARNESS / "config" / "run.yaml", HARNESS / "config" / "sim.yaml"
    run, sim = yaml.safe_load(run_yaml.read_text()), yaml.safe_load(sim_yaml.read_text())
    simulator = run["simulators"]["september"]
    tree = Path(simulator["tree"]).expanduser()
    sys.path.insert(0, str(HARNESS))
    from run_attack_check import src_sha256  # the launcher's own tree hash

    tree_hash = src_sha256(tree)
    assert tree_hash == simulator["src_sha256"], (tree_hash, simulator["src_sha256"])
    timeout_lines = {}
    for rel in ("src/sim/config.py", "src/sim/reconstruction.py"):
        text = (tree / rel).read_text().splitlines()
        timeout_lines[rel] = {"sha256": sha(tree / rel),
                              "lines": {str(i + 1): t.strip() for i, t in enumerate(text) if "attempt_timeout" in t}}
    now = harness_now()
    match = None
    for name, want in FROZEN.items():
        assert sha(STAGE / name) == want, f"staged {name} is not the committed manifest"
        frozen = [line for line in (STAGE / name).read_text().splitlines() if "  ./config/harness_manifest" not in line]
        if frozen == now:
            match = name
    assert match, "the harness equals none of the three committed frozen manifests"
    manifest_file = STAGE / match
    settings = Path(run["settings_dir"]).expanduser()
    sdig = settings_digest(settings)
    tasks_fu = HARNESS / cfg["tasks"]
    tasks_launch = STAGE / f"tasks_311_q{k}_launch.json"
    assert sha(tasks_launch) == LAUNCH_TASKS[k], "the staged main task file is not the pinned one"
    cells = json.loads(tasks_fu.read_text())["cells"]
    main_cells = [c for c in json.loads(tasks_launch.read_text())["cells"] if "real_arms" not in c]
    canon = lambda c: json.dumps(c, sort_keys=True)  # noqa: E731
    assert sorted(map(canon, cells)) == sorted(map(canon, main_cells)), "cells are not the main's r1 entries"
    digest_sets = json.loads((HARNESS / cfg["digest_sets"]).read_text())["sets"] if cfg.get("digest_sets") else {}
    launch = json.loads((main_dir / "launch.json").read_text())
    runs = launch["runs"]
    runs = list(runs.values()) if isinstance(runs, dict) else runs
    honest_a = {r["key"].split("/", 1)[1]: r for r in runs if r.get("arm") == "honest_a"}
    common = {"simulator_src_sha256": tree_hash, "sim_yaml_sha256": sha(sim_yaml), "run_yaml_sha256": sha(run_yaml),
              "settings_sha256": sdig["sha256"], "mode": "honest"}
    rows, bad = [], []
    for c in sorted(cells, key=lambda c: c["key"]):
        key, ref = c["key"], launch["references"].get(c["key"])
        row = {"key": f"simh_311|{key}", "cell": c["cell"], "task": key, "env": c["env"], "main": c.get("main"),
               "strategy": bool(c.get("approach_chars"))}
        if ref is None:
            row["status"] = "no pinned reference (recorded 'not run' by the launcher)"
            rows.append(row)
            continue
        src, copy, ha = Path(ref["source"]), Path(ref["copy"]), honest_a.get(key)
        problems = []
        if ha is None:
            problems.append("no honest_a run record for the task")
        else:  # the source must be in the folder of one of THIS task's honest_a attempts, and that attempt usable
            hits = [a for a in ha["attempts"] if a.get("log_dir")
                    and Path(a["log_dir"]).resolve() == src.resolve().parent]
            if len(hits) != 1:
                problems.append("the source is not in exactly one of the task's honest_a attempt folders")
            elif not (hits[0].get("outcome") or {}).get("usable_reference"):
                problems.append("the source's attempt is not a usable reference")
        if not src.resolve().is_relative_to((main_dir / "honest_a").resolve()):  # a rerun's folder is <task>_rerun<n>
            problems.append("the source is not under <main>/honest_a/")
        if "honest_b" in str(src) or "honest_b" in str(copy):
            problems.append("an honest_b path")
        copy_sha, src_sha = sha(copy), sha(src)
        if not copy_sha == src_sha == ref["sha256"]:
            problems.append("sha256 mismatch among the source, the copy and the recorded value")
        task_name = eval_task_name(copy)
        if "honest" not in task_name or "attack" in task_name:
            problems.append(f"eval task name {task_name!r} is not an honest run")
        row |= {"status": "pinned", "honest_a_run_key": f"honest_a/{key}", "reference_source": str(src),
                "reference_copy": str(copy), "eval_sha256": copy_sha, "eval_task": task_name, "try": ref.get("try"),
                "digest_set": digest_sets.get(key, [])}
        row["rendered_input_sha256"] = hashlib.sha256(json.dumps(
            common | {"eval_sha256": copy_sha, "env": c["env"], "digest_set": row["digest_set"]},
            sort_keys=True).encode()).hexdigest()
        if problems:
            row["provenance_problems"] = problems
            bad.append(key)
        rows.append(row)
    # 25f selected-run identity: every listed cell of this quarter matches its pin, and the cells whose selected reference
    # came from a RERUN attempt (the stager counts redraws, skips reruns: box_files.py:74 @ 9be2fa9b6) are exactly those listed
    rerun_selected = []
    for r in rows:
        if r.get("status") != "pinned":
            continue
        ha = honest_a[r["task"]]
        hit = [a for a in ha["attempts"] if a.get("log_dir") and Path(a["log_dir"]).resolve() == Path(r["reference_source"]).resolve().parent]
        r["selected_attempt_kind"] = hit[0].get("kind") if hit else None
        if r["selected_attempt_kind"] == "rerun":
            rerun_selected.append(r["task"])
        if r["task"] in SELECTED_PINS:
            r["ha25f_pin"] = SELECTED_PINS[r["task"]]
            if r["eval_sha256"] != SELECTED_PINS[r["task"]]:
                r.setdefault("provenance_problems", []).append("selected reference sha256 != Append 25f pin")
                bad.append(r["task"])
    listed_here = sorted(t for t in SELECTED_PINS if t in {r["task"] for r in rows})
    census = {"rerun_selected": sorted(rerun_selected), "ha25f_listed_in_quarter": listed_here,
              "equal": sorted(rerun_selected) == listed_here}
    if not census["equal"]:
        bad.append("census")
        rows.append({"task": "census", "provenance_problems": [f"rerun-selected {sorted(rerun_selected)} != 25f-listed {listed_here}"]})
    if bad:
        detail = {r["task"]: r["provenance_problems"] for r in rows if r.get("provenance_problems")}
        raise SystemExit(f"provenance check failed, nothing written: {json.dumps(detail)}")
    out = {
        "quarter": f"q{k}", "host_box": host, "launch_dir": f"/home/aswin/phase2/runs/2026-10-15_simh_q{k}",
        "written": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds"),
        "world": "simh_311", "key_format": "simh_311|<task>", "arm": "sim_honest@september", "reference_rule": RULE, "references_from": str(main_dir),
        "config": {"path": str(args.config), "sha256": sha(args.config), "resolved": cfg},
        "actor": {"policy": launch.get("policy") or run.get("policy"), "policy_limits": run.get("policy_limits")},
        "simulator": {"name": "september", "tree": str(tree), "src_sha256": tree_hash, "run_yaml_entry": simulator,
                      "sim_yaml_sha256": sha(sim_yaml), "sim_yaml": sim,
                      "attempt_timeout_and_retries": timeout_lines},
        "harness": {"path": str(HARNESS), "frozen_manifest": str(manifest_file), "frozen_manifest_sha256":
                    sha(manifest_file), "run_yaml_sha256": sha(run_yaml), "run_pilot_sha256": sha(HARNESS / "run_pilot.py"),
                    "check_sim_samples_sha256": sha(HARNESS / "check_sim_samples.py")},
        "settings": {"path": str(settings), **sdig},
        "task_files": {"tasks": {"path": str(tasks_fu), "sha256": sha(tasks_fu)},
                       "main_launch_tasks": {"path": str(tasks_launch), "sha256": sha(tasks_launch)}},
        "main_launch_json_sha256": sha(main_dir / "launch.json"),
        "counts": {"cells": len(rows), "pinned": sum(r["status"] == "pinned" for r in rows)},
        "selected_run_census": census,
        "rendered_input_definition": "sha256 of the canonical JSON of " + json.dumps(sorted(common) + ["eval_sha256", "env",
                                                                                                    "digest_set"]),
        "cells": rows,
    }
    with open(args.out, "x") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(f"-> {args.out}: {out['counts']}")


if __name__ == "__main__":
    main()
