"""simh_rlr_manifest.py (sim-honest REPLACEMENT launches: 09 @ 13be06982, 91485ce80, c5420f2f5, c10a14948; derived from
simh_nm_manifest.py, changed where marked "rlr:"): ON THE BOX, before a replacement launch, write its immutable generation
manifest: the replacement cells (a subset of the parent batch's task file, each the main's r1 entry), each one's pinned
reference (the SAME fork source as the original: --references-from the parent's main), the per-cell reference rule of the
parent family (+311: 25f, selected rerun attempts only for 25f-listed cells, pins checked; nm360: 25g, the unique counted
attempt), the rendered simulator-reference hash, and per cell the replacement class (infrastructure for the 7
RateLimitError keys, unknown for the 9 terminal-error keys) and the original it supersedes. Writes only --out.
  <sept venv python> simh_rlr_manifest.py --launch <name> --host <h> --config <staged pilot_simh_<name>.yaml> --out <file> [--s4-ruling <sha>]
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

STAGE = Path("/home/aswin/phase2/simh_rlr_stage")  # rlr: one stage dir for both families
H311, HNM = Path("/home/aswin/phase2/attack_check_311"), Path("/home/aswin/phase2/nm/attack_check_nm")
FROZEN_311 = {"harness_manifest_fu_q123.tsv": "71fd27724a045d3df5d8dba9b6e63d5f86514377c025a11a57fd78f6bf1aa287",
              "harness_manifest_fu_q2r.tsv": "b8f813c9a1f4ff4371a981b3aea8a9406b7fa0143eda3212a8b053e8123a4992",
              "harness_manifest_fu_q4.tsv": "32ed72b3e3aa95b21a41ae0dfcf47cacb9efb6e760e9577536e49b57e898bd3d"}
FROZEN_NM = {"harness_manifest_fu_nm.tsv": "18fbbaa225bbb0f70890f1436c17d254d676495136f8bd2aa616cb626281b7ed",
             "harness_manifest_fu_box4.tsv": "363d5b65a03900495372b5160f28bdd96673e37b563a42dd46148a67e4eda45d",
             "harness_manifest_fu_simsastra.tsv": "deb5f193780be039ca55b948b2083ac458a0cdaa7231440bbdb8358db6247b83"}
R311 = "/home/aswin/phase2/runs/2026-10-14_"
# rlr: launch -> (family, main, main's own task file (staged name, sha256), parent task file (staged name, sha256), world)
LAUNCHES = {
    "q1_rlr": ("311", R311 + "154401_scaleup311_q1", ("tasks_311_q1_launch.json", "ee413a10eb2c448575fa34a2eafad25428f0947e8ac3450dfbc7e2c7152e4124"),
               ("tasks_311_q1_fu.json", "b5b02a57e6e4f02eeed7b7dc38ec7f48a1e8ab7669a4e9a97b98c6f23f860818"), "simh_311"),
    "q2_rlr": ("311", R311 + "154401_scaleup311_q2", ("tasks_311_q2_launch.json", "eb72f9532e42953fb813d6b226b2eb815c9d161dc3f3e47c5062694de97eaf4d"),
               ("tasks_311_q2_fu.json", "31ee96143aa74b35eb898edb64d9c7b33b79ff779d6b8b84b360b754207bd1cf"), "simh_311"),
    "q3_rlr": ("311", R311 + "155838_scaleup311_q3", ("tasks_311_q3_launch.json", "186633516c694d6087ad2454ee5f8f7cb854dd8bf319c1717682330673a916a9"),
               ("tasks_311_q3_fu.json", "a477bb8ee83d0e29943caae41f8d43c5c7892aa620240f6095076461d0631305"), "simh_311"),
    "q4_rlr": ("311", R311 + "160246_scaleup311_q4", ("tasks_311_q4_launch.json", "033d4bebf7bb93c78cada607f698e9ce32e157a6d808adb735d24574d6fc8a2e"),
               ("tasks_311_q4_fu.json", "dab4bf2d00b890b2add9e7f9746f66dbf12457f11d66b5004f00599dd344122a"), "simh_311"),
    "nm_s3_rlr": ("nm", "/home/aswin/phase2/nm/runs/2026-10-14_newmains_s3", ("tasks_nm_s3_launch.json", "83ec89869204591d5390eb95ba4c4f6663f9a9d8fc591c97939cec2a3cbac196"),
                  ("tasks_simh_nm_s3.json", "53856cef12148a1d0bc279b399f460e8f258480fe4bc34de0dcffb10026565e3"), "simh_nm360"),
    "nm_s4d_rlr": ("nm", "/home/aswin/phase2/nm/runs/2026-10-14_newmains_s4", ("tasks_nm_s4_launch.json", "88ddad22c047ab2258010947d01cc40af769ac45106e57e15e5d5ba62cc5e9ab"),
                   ("tasks_simh_nm_s4d.json", "bf54b635bfadca2addccb0b10092181b9ad0c28222228f85106a5ec37599eba3"), "simh_nm360"),
    "nm_s2_rlr": ("nm", "/home/aswin/phase2/nm/runs/2026-10-14_newmains_s2", ("tasks_nm_s2_launch.json", "4e83a59f99c4d03872f904fb7c18e09a69be6a72b6590ad208821ae7cdab4d9f"),
                  ("tasks_simh_nm_s2.json", "06ac7c1ef5080c302b28ab5db22379dc73653a309cb01cb924ab031573411eb5"), "simh_nm360"),  # 09 @ dd93a41cd §2
    "q3_rlr2": ("311", R311 + "155838_scaleup311_q3", ("tasks_311_q3_launch.json", "186633516c694d6087ad2454ee5f8f7cb854dd8bf319c1717682330673a916a9"),
                ("tasks_311_q3_fu.json", "a477bb8ee83d0e29943caae41f8d43c5c7892aa620240f6095076461d0631305"), "simh_311"),  # Q3_rlr cut by astra's EC2 stop
    "q2_rlr2": ("311", R311 + "154401_scaleup311_q2", ("tasks_311_q2_launch.json", "eb72f9532e42953fb813d6b226b2eb815c9d161dc3f3e47c5062694de97eaf4d"),
                ("tasks_311_q2_fu.json", "31ee96143aa74b35eb898edb64d9c7b33b79ff779d6b8b84b360b754207bd1cf"), "simh_311"),  # t293: second Q2-parent launch
    "nm_s4d_rlr2": ("nm", "/home/aswin/phase2/nm/runs/2026-10-14_newmains_s4", ("tasks_nm_s4_launch.json", "88ddad22c047ab2258010947d01cc40af769ac45106e57e15e5d5ba62cc5e9ab"),
                    ("tasks_simh_nm_s4d.json", "bf54b635bfadca2addccb0b10092181b9ad0c28222228f85106a5ec37599eba3"), "simh_nm360"),  # nm107: second S4d-parent launch
    "nm_s4r_rlr": ("nm", "/home/aswin/phase2/nm/runs/2026-10-14_newmains_s4r", ("tasks_nm_s4r_launch.json", "9b7e3f4d1a5b7735fd0ca1a6aee428d420c41f26f21b82cfd012b3e49127fe7f"),
                   ("tasks_simh_nm_s4r.json", "92aaff8bc144f9d34da9a10a568c7ef9c481ec39b4d62ba7b4a287a6117da0d1"), "simh_nm360"),  # nm353: step_limit.py:72, 0d 15:4x on census @ 391fa659b
    "q1_rlr2": ("311", R311 + "154401_scaleup311_q1", ("tasks_311_q1_launch.json", "ee413a10eb2c448575fa34a2eafad25428f0947e8ac3450dfbc7e2c7152e4124"),
                ("tasks_311_q1_fu.json", "b5b02a57e6e4f02eeed7b7dc38ec7f48a1e8ab7669a4e9a97b98c6f23f860818"), "simh_311"),  # t320: second Q1-parent launch
}
# rlr: 09's Append 25f H_A notes for +311 (simh_manifest.py): the cells whose selected honest_a is a later attempt than the
# counted one, with the selected reference's pinned sha256 (t389 is a Q1 replacement cell)
SELECTED_PINS = {"t389_model_registry": "d160ffdc956ffc89180043b3575b002d4b0f1cc3f7600082f38decc81f28dae0",
                 "t325_model_registry": "e1cdf8d1594bac78b6e64377557bf63d008f2af0248912f21c7bd60c523b7247",
                 "t309_slack_clone": "69c964700620933d0fbfa771fc83068d208a3e20c48dfad47d443a2ec88f5028"}
# 09 via d9 16:3x: nm_s4d_rlr2 (nm107) runs on box1 under box1's OWN frozen nm harness (fu_nm), never box4's; the manifest records
# its S4d parent's box4 harness and the difference, both settings fingerprints, and asserts the simulator tree, run.yaml and
# sim.yaml hashes equal the parent's (results/manifest_simh_nm_s4d_2026-10-15_114650.json, written on box4)
PARENT_S4D = {"source": "results/manifest_simh_nm_s4d_2026-10-15_114650.json", "host": "box4",
              "frozen_manifest": "harness_manifest_fu_box4_s4r.tsv",
              "frozen_manifest_sha256": "543394b2f6c0d590bea597c83feac7d7f69550dfb5b0c040d83cbd78a4143cc1",
              "settings": {"sha256": "38ddafe42041e0e6053b7e8df9de72535c969c699408f125c8e6e5601f8e89b8", "files": 5587,
                           "label": "host-local digest; not an identity (25h §13)"},
              "simulator_src_sha256": "1ea3772209663ef8a848effaf6677d23d5bab06c248a37c1fb61f0957efd7b8c",
              "sim_yaml_sha256": "03340860a688a04ea6cb017b46067e037748f5864108b79a4ff84d02245f4612",
              "run_yaml_sha256": "bf4678a0abed2241ce95642619f58db456d1506aef088a70881139087daffa4e"}
# 09 @ 502c1b05e: the routed roots of the second replacement launches (run dir 2026-10-15_simh_<launch> <-> root)
ROUTED_ROOTS = {"q3_rlr2": "simh311_q3_sim_rlr2_v1__sim_honest", "q1_rlr2": "simh311_q1_sim_rlr2_v1__sim_honest", "q2_rlr2": "simh311_q2_sim_rlr2_v1__sim_honest",
                "nm_s4d_rlr2": "simh_nm360_s4d_sim_rlr2_v1__sim_honest"}
RL_KEYS = {"t462", "t242", "t271", "t410", "t298", "t310", "nm103", "t240", "t275", "t457"}  # 09 @ 13be06982, c5420f2f5, 81f75b763: infrastructure
RULE = ("each replacement cell's simulated-honest run forks at step 0 from the SAME pinned reference as the run it replaces "
        "(its parent batch's main launch, launch.json `references`, copied verbatim with its sha256 re-checked), under the parent "
        "family's reference rule (+311: 25f; nm360: 25g); one designation per cell, its first counted attempt, no fallback")
HARNESS = H311  # set per launch in main() before harness_now() runs


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
                    or rel.startswith("./config/pilot_simh_") or rel.startswith("./config/tasks_simh_")):  # rlr: + all simh task files
                continue
            rows.append((rel.encode(), f"{sha(p)}  {rel}"))
    return [r for _, r in sorted(rows)]


def eval_task_name(p: Path) -> str:
    from inspect_ai.log import read_eval_log  # .eval members are zstd-compressed: read the header the launcher's way

    return read_eval_log(str(p), header_only=True).eval.task


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch", required=True, choices=list(LAUNCHES))
    ap.add_argument("--host", required=True, help="the host's label (box1, box3, box4, xbox2, xbox3, astra)")
    ap.add_argument("--config", type=Path, required=True)

    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--s4-ruling", default=None, help="S4 groups on box4: 09's ruling commit accepting the s4r-armed harness")
    args = ap.parse_args()
    # nm: the render step may rewrite generated compose files in the shared settings dir, so no launcher may run on the box
    live = subprocess.run(["pgrep", "-f", r"run_pilot\.py"], capture_output=True, text=True)
    assert live.returncode == 1 and not live.stdout.strip(), f"a launcher is running on this box (pgrep rc {live.returncode})"
    global HARNESS
    k, host = args.launch, args.host
    fam, main_s, (lt_name, lt_sha), (pt_name, pt_sha), world = LAUNCHES[k]
    HARNESS = H311 if fam == "311" else HNM
    main_dir = Path(main_s)
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
    frozen_allowed = dict(FROZEN_311 if fam == "311" else FROZEN_NM)
    if k == "nm_s4d_rlr2":
        assert host == "box1", "nm_s4d_rlr2 runs on box1 with its own fu_nm harness (09 via d9 16:3x)"
    if k in ("nm_s4d_rlr", "nm_s4r_rlr"):
        assert host == "box4" and args.s4_ruling and len(args.s4_ruling) >= 9, "S4d/S4r replacements run on box4 with 09's ruling cited"
    if host == "box4" and k in ("nm_s4d_rlr", "nm_s4r_rlr"):  # 09 @ df46b9be1: box4's nm harness as s4r_arm.sh @ 6dde150b1 left it
        frozen_allowed["harness_manifest_fu_box4_s4r.tsv"] = "543394b2f6c0d590bea597c83feac7d7f69550dfb5b0c040d83cbd78a4143cc1"
    for name, want in frozen_allowed.items():
        assert sha(STAGE / name) == want, f"staged {name} is not the committed manifest"
        frozen = [line for line in (STAGE / name).read_text().splitlines() if "  ./config/harness_manifest" not in line]
        if frozen == now:
            match = name
    assert match, "the harness equals none of the committed frozen manifests allowed on this host"
    manifest_file = STAGE / match
    parent_s4d = None
    if k == "nm_s4d_rlr2":
        assert match == "harness_manifest_fu_nm.tsv", match
        p4 = STAGE / PARENT_S4D["frozen_manifest"]
        assert sha(p4) == PARENT_S4D["frozen_manifest_sha256"], "staged box4 parent harness manifest is not the committed one"
        rd = lambda f: dict(reversed(x.split("  ", 1)) for x in f.read_text().splitlines() if "  ./config/harness_manifest" not in x)  # noqa: E731
        par, own = rd(p4), rd(manifest_file)
        differ = sorted(q for q in set(par) & set(own) if par[q] != own[q])
        assert differ == ["./prelaunch_gates.sh"] and not set(own) - set(par), (differ, sorted(set(own) - set(par)))
        ident = {"simulator_src_sha256": tree_hash, "sim_yaml_sha256": sha(sim_yaml), "run_yaml_sha256": sha(run_yaml)}
        assert all(ident[x] == PARENT_S4D[x] for x in ident), ("identity differs from the S4d parent", ident)
        parent_s4d = dict(PARENT_S4D, harness_difference={"differing_files": differ, "parent_only_config_files": sorted(set(par) - set(own))},
                          identity_equal=True, this_identity=ident)
    settings = Path(run["settings_dir"]).expanduser()
    sdig = settings_digest(settings)
    # rlr: our task file is staged; its cells must be the parent task file's entries for the planned keys (byte-identical as
    # JSON objects), each canonically equal to the main launch file's r1 entry of the same key
    tasks_fu = STAGE / Path(cfg["tasks"]).name
    assert cfg["tasks"] == f"config/tasks_simh_{k}.json", cfg["tasks"]
    tasks_launch = STAGE / lt_name
    assert sha(tasks_launch) == lt_sha, "the staged main task file is not the pinned one"
    assert sha(STAGE / pt_name) == pt_sha, "the staged parent task file is not the pinned one"
    cells = json.loads(tasks_fu.read_text())["cells"]
    parent = {c["key"]: c for c in json.loads((STAGE / pt_name).read_text())["cells"]}
    main_cells = {c["key"]: c for c in json.loads(tasks_launch.read_text())["cells"] if "real_arms" not in c}
    canon = lambda c: json.dumps(c, sort_keys=True)  # noqa: E731
    assert cells and len({c["key"] for c in cells}) == len(cells), "empty or duplicate cells"
    assert all(c["key"] in parent and canon(parent[c["key"]]) == canon(c) for c in cells), "cells are not the parent's entries"
    assert all(c["key"] in main_cells and canon(main_cells[c["key"]]) == canon(c) for c in cells), "not the main's r1 entries"
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
        row = {"key": f"{world}|{key}", "cell": c["cell"], "task": key, "env": c["env"], "main": c.get("main"),
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
    # rlr: the parent family's reference rule. nm360 (25g item 2): the fork source is the UNIQUE counted attempt. +311 (25f):
    # the fork source is the counted attempt, except 25f-listed cells, whose selected (rerun) reference must equal the pin.
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
                                 "n_counted": len(counted), "selected_attempt": sel[0] if len(sel) == 1 else None}
        listed = fam == "311" and r["task"] in SELECTED_PINS
        ok = len(sel) == 1 and (sel == counted if not listed else (r["eval_sha256"] == SELECTED_PINS[r["task"]] and hit[0].get("kind") == "rerun"))
        if listed:
            r["ha25f_pin"] = SELECTED_PINS[r["task"]]
        if not ok or len(counted) != 1:
            rerun_selected.append(r["task"])
            r.setdefault("provenance_problems", []).append("fork source breaks the parent family's reference rule (25f/25g)")
            bad.append(r["task"])
    census = {"rule": ("25f: fork source == the counted attempt, except 25f-listed cells == their pin" if fam == "311" else
                       "25g item 2: fork source == the unique counted honest_a attempt"),
              "violations": sorted(rerun_selected), "equal": not rerun_selected,
              "listed_25f_in_launch": sorted(t for t in SELECTED_PINS if fam == "311" and t in {r["task"] for r in rows})}
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
        "launch": k, "family": fam, "host_box": host, "launch_dir": f"/home/aswin/phase2/runs/2026-10-15_simh_{k}",
        "written": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds"),
        "world": world, "key_format": f"{world}|<task>", "arm": "sim_honest@september", "reference_rule": RULE, "references_from": str(main_dir),
        "config": {"path": str(args.config), "sha256": sha(args.config), "resolved": cfg},
        "actor": {"policy": launch.get("policy") or run.get("policy"), "policy_limits": run.get("policy_limits")},
        "simulator": {"name": "september", "tree": str(tree), "src_sha256": tree_hash, "run_yaml_entry": simulator,
                      "sim_yaml_sha256": sha(sim_yaml), "sim_yaml": sim,
                      "attempt_timeout_and_retries": timeout_lines},
        "harness": {"path": str(HARNESS), "frozen_manifest": str(manifest_file), "frozen_manifest_sha256":
                    sha(manifest_file), "run_yaml_sha256": sha(run_yaml), "run_pilot_sha256": sha(HARNESS / "run_pilot.py"),
                    "check_sim_samples_sha256": sha(HARNESS / "check_sim_samples.py")},
        "settings": {"path": str(settings), **sdig, "label": "host-local digest; not an identity (25h §13)"},
        "task_files": {"tasks": {"path": str(tasks_fu), "sha256": sha(tasks_fu)},
                       "main_launch_tasks": {"path": str(tasks_launch), "sha256": sha(tasks_launch)}},
        "main_launch_json_sha256": sha(main_dir / "launch.json"),
        "counts": {"cells": len(rows), "pinned": sum(r["status"] == "pinned" for r in rows)},
        "parent_s4d": parent_s4d,
        "routed_root": ({"root": ROUTED_ROOTS[k], "run_dir": "/home/aswin/phase2/runs/2026-10-15_simh_" + k, "ruling": "09 @ 502c1b05e"} if k in ROUTED_ROOTS else None),
        # rlr: per cell, the replacement designation (09's rulings), its class and the original run it supersedes
        "replacements": [{"task": r["task"], "class": "infrastructure" if r["task"].split("_")[0] in RL_KEYS else "unknown",
                          "supersedes": {"parent_launch_dir": ("/home/aswin/phase2/runs/2026-10-15_simh_" + k.split("_rlr")[0]), "key": r["task"]},
                          "release": ("released from the base-rate hold: 0d rule 15:4x; t237 replacement completed OK 16:19:39 (hold record v5)" if r["task"].split("_")[0] in ("t293", "t411", "t413", "nm107") else None),
                          "ruling": ("09 via d9 ~16:4x: the Q3_rlr run was cut by astra's EC2 stop (d9 describe-instances read 16:31:12 PT; results/simh_q3_rlr_cut_2026-10-15.json), class infrastructure; root simh311_q3_sim_rlr2_v1__sim_honest" if k == "q3_rlr2" else
                                     "09 @ 81f75b763 (RateLimitError, 14:20 bin; root check by the chain fork)" if r["task"].split("_")[0] in ("t240", "t275", "t457")
                                     else "09 @ 13be06982 / c5420f2f5 (RateLimitError; root check @ 98ffa9e40)" if r["task"].split("_")[0] in RL_KEYS
                                     else "09 @ dd93a41cd (terminal error, class unknown; root check by the chain fork)" if r["task"].split("_")[0] in ("t293", "nm150")
                                     else "0d 15:4x on census @ 391fa659b (terminal error at step_limit.py:72; standing-rule admission, one approved replacement)" if r["task"].split("_")[0] in ("t320", "nm353")
                                     else "09 @ c10a14948 (terminal error, class unknown; root check @ 2b66c9dbe)")} for r in rows],
        "not_run_rows": [{"task": r["task"], "cell": r["cell"], "reason": "not run: no pinned reference"} for r in rows if r["status"] != "pinned"],
        "spec": "09 @ 13be06982, 91485ce80, c5420f2f5, c10a14948 (aswin/convergence-audit-09); parent family rules 25f (+311) / 25g (nm360)",
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
