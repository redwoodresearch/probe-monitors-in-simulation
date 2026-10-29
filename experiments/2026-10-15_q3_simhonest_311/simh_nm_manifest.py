"""simh_nm_manifest.py (nm360 sim-honest; derived from simh_manifest.py, Codex r5/r6 AGREE, changed where marked "nm:"):
RULE.md, "Freeze before generation"; 09's Append 25f, and 25g for rerun cells. ON THE BOX, before group g's simulated-honest
launch on the nm harness, write its immutable manifest: what will generate the runs, fixed before any run exists. Read-only: it reads files
and git state, runs no task or simulator code and writes only --out. Run with the September venv's python (PyYAML):

  <sept venv python> simh_nm_manifest.py --g <g> --host <h> --config <staged pilot_simh_nm_<g>.yaml> --out <file>

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

# nm: the nm mains (s4d's 19 cells from newmains_s4, 4r's from newmains_s4r, per the S4 source maps @ 30f547dbd/b68ab0a53)
NM = "/home/aswin/phase2/nm/runs/2026-10-14_newmains_"
MAINS = {"s1": NM + "s1", "s2": NM + "s2", "s3": NM + "s3", "s4d": NM + "s4", "s4r": NM + "s4r", "s5": NM + "s5"}
HARNESS = Path("/home/aswin/phase2/nm/attack_check_nm")  # nm: the nm harness (Codex nm-delta review: SAFE)
STAGE = Path("/home/aswin/phase2/simh_nm_stage")
# nm: the three committed frozen nm harness manifests (origin/aswin/q3-new-mains-launch, experiments/2026-10-14_q3_new_mains_launch/config/), staged in STAGE
FROZEN = {"harness_manifest_fu_nm.tsv": "18fbbaa225bbb0f70890f1436c17d254d676495136f8bd2aa616cb626281b7ed",
          "harness_manifest_fu_box4.tsv": "363d5b65a03900495372b5160f28bdd96673e37b563a42dd46148a67e4eda45d",
          "harness_manifest_fu_simsastra.tsv": "deb5f193780be039ca55b948b2083ac458a0cdaa7231440bbdb8358db6247b83"}
# nm: the mains' own task files (new-mains branch config/), staged in STAGE as tasks_nm_<main>_launch.json
LAUNCH_TASKS = {"s1": ("tasks_nm_s1_launch.json", "8249c208d3dc44a033f8949bd1f323caf7cbe2f96f1d1a0ff3447678dab280ee"),
                "s2": ("tasks_nm_s2_launch.json", "4e83a59f99c4d03872f904fb7c18e09a69be6a72b6590ad208821ae7cdab4d9f"),
                "s3": ("tasks_nm_s3_launch.json", "83ec89869204591d5390eb95ba4c4f6663f9a9d8fc591c97939cec2a3cbac196"),
                "s4d": ("tasks_nm_s4_launch.json", "88ddad22c047ab2258010947d01cc40af769ac45106e57e15e5d5ba62cc5e9ab"),
                "s4r": ("tasks_nm_s4r_launch.json", "9b7e3f4d1a5b7735fd0ca1a6aee428d420c41f26f21b82cfd012b3e49127fe7f"),
                "s5": ("tasks_nm_s5_launch.json", "f5589918a368d3509709206b5eefbcd4b3da40f435a9c1e3af37fc0192d984f7")}
# nm: the follow-up task files the simh task files were built from (make_simh_nm_tasks.py; sha256), in the harness config/
FU_TASKS = {"s1": ("tasks_nm_s1_fu.json", "9c39726a2f1464f2c86fa8866691c6cecc47f7b907abd64590e94b970205fde8"),
            "s2": ("tasks_nm_s2_fu.json", "3db7f864fa574feec01e0830830c567d256ecaeb4fb8f17a0fa20b448c80caa9"),
            "s3": ("tasks_nm_s3_fu.json", "1dd8280de5fc511823da64a6ae716320d27b0bf6ee732343dcab91c8b006cdd5"),
            "s4d": ("tasks_nm_s4d_fu.json", "a16c73db952dd77a415bea0ad686860b7e5ffd039eec9d80408d3e5ab9d112d9"),
            "s4r": ("tasks_nm_s4r_fu.json", "b921721819f00b673389bc9e8a7568d8f861a5a8b052ef76c16ecbf08ded4e96"),
            "s5": ("tasks_nm_s5_fu.json", "1591fcce8bea840fb24a58260c6d8e20a7c811ddd057958994ea3feccb6cbe3a")}
EXCLUDED = ["nm208", "nm213", "nm241", "nm222", "nm106", "nm219", "nm126", "nm109", "nm203",  # broken setup (v4 @ 396da80fd)
            "nm142", "nm209", "nm280", "nm145", "nm235"]  # 09 Append 25g: counted attempt != pinned rerun -> not run
# 09 (S2 ack, ~12:40): every removal from the group's U cells is listed explicitly by key with its reason, so the
# generation subset is derivable by key. Per-key notes for rows whose reason is more specific than the default.
NOT_RUN_NOTES = {
    "nm105_media_processing": "not run: counted honest_a unusable (submit-only; unusable_honest_nm360 @ 72e1fac21); harness placeholder "
                              "cell dummy_mt_media_processing:delete_media_no_trace, not in U_nm as a generated cell, never a counted run",
}
RULE = ("each cell's simulated-honest run forks at step 0 from that cell's REAL honest_a run of the group's nm main "
        "launch: the reference the main launch pinned for the task (launch.json `references`), copied verbatim by the "
        "launcher with its sha256 re-checked, and it must be the stager's COUNTED attempt (no rerun, no redraw; 09 Append "
        "25g); a cell with no pinned reference is recorded 'not run'; broken-setup and 25g rerun cells are not in the task file")


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
                    or rel.startswith("./config/pilot_simh_") or rel.startswith("./config/tasks_simh_nm_")):  # nm: + our task files
                continue
            rows.append((rel.encode(), f"{sha(p)}  {rel}"))
    return [r for _, r in sorted(rows)]


def eval_task_name(p: Path) -> str:
    from inspect_ai.log import read_eval_log  # .eval members are zstd-compressed: read the header the launcher's way

    return read_eval_log(str(p), header_only=True).eval.task


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g", required=True, choices=list(MAINS))
    ap.add_argument("--host", required=True, help="the host's label (box1, box3, box4, xbox2, xbox3, astra)")
    ap.add_argument("--config", type=Path, required=True)

    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--s4-ruling", default=None, help="S4 groups on box4: 09's ruling commit accepting the s4r-armed harness")
    args = ap.parse_args()
    # nm: the render step may rewrite generated compose files in the shared settings dir, so no launcher may run on the box
    live = subprocess.run(["pgrep", "-f", r"run_pilot\.py"], capture_output=True, text=True)
    assert live.returncode == 1 and not live.stdout.strip(), f"a launcher is running on this box (pgrep rc {live.returncode})"
    k, host = args.g, args.host  # nm: k is the group
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
    frozen_allowed = dict(FROZEN)
    if k in ("s4d", "s4r"):
        assert host == "box4" and args.s4_ruling and len(args.s4_ruling) >= 9, "S4 groups run on box4 with 09's ruling cited"
    if host == "box4" and k in ("s4d", "s4r"):  # 09 @ df46b9be1: box4's nm harness as s4r_arm.sh @ 6dde150b1 left it
        frozen_allowed["harness_manifest_fu_box4_s4r.tsv"] = "543394b2f6c0d590bea597c83feac7d7f69550dfb5b0c040d83cbd78a4143cc1"
    for name, want in frozen_allowed.items():
        assert sha(STAGE / name) == want, f"staged {name} is not the committed manifest"
        frozen = [line for line in (STAGE / name).read_text().splitlines() if "  ./config/harness_manifest" not in line]
        if frozen == now:
            match = name
    assert match, "the harness equals none of the committed frozen manifests allowed on this host"
    manifest_file = STAGE / match
    settings = Path(run["settings_dir"]).expanduser()
    sdig = settings_digest(settings)
    # nm: our task file is staged (STAGE/<basename>); its cells must be the follow-up file's cells minus EXCLUDED, each one
    # canonically equal to the main launch file's r1 entry of the same key
    tasks_fu = STAGE / Path(cfg["tasks"]).name
    assert cfg["tasks"] == f"config/tasks_simh_nm_{k}.json", cfg["tasks"]
    lt_name, lt_sha = LAUNCH_TASKS[k]
    tasks_launch = STAGE / lt_name
    assert sha(tasks_launch) == lt_sha, "the staged main task file is not the pinned one"
    fu_name, fu_sha = FU_TASKS[k]
    assert sha(HARNESS / "config" / fu_name) == fu_sha, "the harness follow-up task file is not the pinned one"
    cells = json.loads(tasks_fu.read_text())["cells"]
    fu_cells = json.loads((HARNESS / "config" / fu_name).read_text())["cells"]
    main_cells = {c["key"]: c for c in json.loads(tasks_launch.read_text())["cells"] if "real_arms" not in c}
    canon = lambda c: json.dumps(c, sort_keys=True)  # noqa: E731
    want = [c for c in fu_cells if c["key"].split("_")[0] not in EXCLUDED]
    assert sorted(map(canon, cells)) == sorted(map(canon, want)), "cells are not the follow-up file's minus EXCLUDED"
    assert all(c["key"] in main_cells and canon(main_cells[c["key"]]) == canon(c) for c in cells), "not the main's r1 entries"
    assert not any(c["key"].split("_")[0] in EXCLUDED for c in cells)
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
        row = {"key": f"simh_nm360|{key}", "cell": c["cell"], "task": key, "env": c["env"], "main": c.get("main"),
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
    # nm: 09 Append 25g: every pinned cell's selected attempt must be the stager's COUNTED one (not kind "rerun", not a
    # pre-start redraw); the 25g rerun cells were removed from the task file, so any hit here is a provenance failure
    rerun_selected = []
    for r in rows:
        if r.get("status") != "pinned":
            continue
        ha = honest_a[r["task"]]
        hit = [a for a in ha["attempts"] if a.get("log_dir") and Path(a["log_dir"]).resolve() == Path(r["reference_source"]).resolve().parent]
        r["selected_attempt_kind"] = hit[0].get("kind") if hit else None
        counted = [i for i, a in enumerate(ha["attempts"]) if not a.get("redrawn_because") and a.get("kind") != "rerun"]
        sel = [i for i, a in enumerate(ha["attempts"]) if a.get("log_dir") and Path(a["log_dir"]).resolve() == Path(r["reference_source"]).resolve().parent]
        r["counted_honest_a"] = {"run_key": f"honest_a/{r['task']}", "attempt": counted[0] if len(counted) == 1 else None,
                                 "n_counted": len(counted)}
        # 25g item 2: the fork source (what --references-from copies) must be the UNIQUE counted attempt, source == copy sha256
        if len(counted) != 1 or sel != counted or not hit or hit[0].get("kind") == "rerun" or hit[0].get("redrawn_because"):
            rerun_selected.append(r["task"])
            r.setdefault("provenance_problems", []).append("fork source is not the unique counted attempt (09 Append 25g)")
            bad.append(r["task"])
    census = {"rule": "25g item 2: fork source == the unique counted honest_a attempt, source sha256 == copy sha256, by key",
              "not_counted": sorted(rerun_selected), "excluded_upfront": EXCLUDED, "equal": not rerun_selected}
    # 25g item 4: the RENDERED simulator-reference hash per pinned cell: the simulator's system prompt exactly as the
    # harness's own preflight builds it (check_sim_samples.simulator_prompt, honest mode, the cell's environment, the
    # reference as BA_REAL_LOG/BA_HONEST_LOG), rendered twice (must be identical) and sha256-hashed. Needs the simulator
    # tree on PYTHONPATH (the start path's own setting) and no launcher on the box (rendering may rewrite generated
    # compose files, which the launcher rewrites anyway). A render failure is a provenance failure (nothing written).
    sys.path.insert(0, str(tree / "src"))  # the pinned simulator tree, as the launcher's preflight imports it
    import check_sim_samples as css  # imports the harness config, settings dir and the tree (asserted below)

    assert Path(css.runner.__file__).resolve().parents[1] == tree / "src", "render: sims code is not the pinned tree"
    for r in rows:
        if r.get("status") != "pinned" or r.get("provenance_problems"):
            continue
        env = sim["env"] | simulator.get("env", {}) | {"LA_ENV": r["env"], "BA_MODE": "honest",
                                                         "BA_REAL_LOG": r["reference_copy"], "BA_HONEST_LOG": r["reference_copy"]}
        try:
            p1, p2 = css.simulator_prompt(env), css.simulator_prompt(env)
            assert p1 == p2, "two renders differ"
            r["rendered_reference_sha256"] = hashlib.sha256(p1.encode()).hexdigest()
            r["rendered_reference_chars"] = len(p1)
            r["rendered_reference_path_in_prompt"] = r["reference_copy"] in p1
        except Exception as e:  # noqa: BLE001 -- recorded per cell; any failure stops the manifest below
            r.setdefault("provenance_problems", []).append(f"render failed: {e!r}"[:300])
            bad.append(r["task"])
    if bad:
        detail = {r["task"]: r["provenance_problems"] for r in rows if r.get("provenance_problems")}
        raise SystemExit(f"provenance check failed, nothing written: {json.dumps(detail)}")
    out = {
        "group": k, "host_box": host, "launch_dir": f"/home/aswin/phase2/runs/2026-10-15_simh_nm_{k}",
        "written": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds"),
        "world": "simh_nm360", "key_format": "simh_nm360|<task>", "arm": "sim_honest@september", "reference_rule": RULE, "references_from": str(main_dir),
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
        # 09 (~12:40): every removal from this group's U cells, by key with its reason; generation subset = U cells - these
        "not_run_rows": sorted(
            [{"task": c["key"], "cell": c["cell"], "reason": "not run: no counted usable reference (counted attempt differs "
              "from the pinned rerun; 09 Append 25g)"} for c in fu_cells if c["key"].split("_")[0] in EXCLUDED[9:]]
            + [{"task": r["task"], "cell": r["cell"], "reason": NOT_RUN_NOTES.get(r["task"], "not run: no reference (no pinned "
                "honest_a reference in the main launch; a real task)")} for r in rows if r["status"] != "pinned"],
            key=lambda x: x["task"]),
        # 25g item 1: U_nm (setup-only ceiling) and the generation subset, kept separate
        "universe": {"U_nm_ceiling_total": 351, "group_cells_in_U": len([c for c in fu_cells if c["key"].split("_")[0] not in EXCLUDED[:9]]),
                     "group_excluded_25g": sorted(c["key"] for c in fu_cells if c["key"].split("_")[0] in EXCLUDED[9:]),
                     "group_no_pinned_reference": sorted(r["task"] for r in rows if r["status"] != "pinned"),
                     "generation_subset": sum(r["status"] == "pinned" for r in rows)},
        "spec": "09 Append 25g @ e6ded1811 (aswin/convergence-audit-09, experiments/2026-10-14_priority1_spec_09/RULE.md)",
        **({"harness_exception": {
            "ruling": f"09 accepts box4's s4r-armed nm harness for S4d/S4r, box4 only (@ {args.s4_ruling})",
            "frozen_manifest": "harness_manifest_fu_box4_s4r.tsv", "frozen_manifest_sha256": frozen_allowed.get("harness_manifest_fu_box4_s4r.tsv"),
            "one_file_difference": {"path": "./prelaunch_gates.sh", "committed_fu_box4": "8951059eadfe2b90f7691ba911224c8fb15871c5ea4ee469ce5ae539b9f0fb36",
                                    "now": "b08e671d4fe9f8827f311e9f9a166b6b61ece1ba52b8bdf2c59b79746aa0812e",
                                    "made_by": "s4r_arm.sh @ 6dde150b1 (Amendment 9 @ ef6f05eec; ~00:11 PT)", "backed_up_original_sha256": "8951059eadfe2b90f7691ba911224c8fb15871c5ea4ee469ce5ae539b9f0fb36"},
            "disclosure": ["S4r's sim-honest runs on the same harness as S4r's real and sim-attack arms.",
                           "S4d's sim-attack (s4d_fu, ended 23:59, before the 00:11 edit) ran pre-edit, so S4d's sim-honest differs from it in prelaunch_gates.sh only, which is off the follow-up path."]}}
           if match == "harness_manifest_fu_box4_s4r.tsv" else {}),
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
