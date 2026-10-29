"""simh_sp493_manifest.py (S_paper493 sim-honest, Phase A (g4): 14 T196 mains; 09 Append 39 + rulings via d9 ~17:30) AND the A25 fill-in (8 T196 tasks, 09 @ 286b2f518 + addendum 6): ON THE
BOX, write the immutable generation manifest of launch sp493_t196 or t196fill BEFORE it starts (one routed root per launch). REUSES simh_rlr_manifest.py's reviewed helpers
(settings digest, harness hash, eval task name) from the same stage dir; new here is the REFINDEX reference rule:
  - every cell forks at step 0 from il_refindex_old's pinned reference for its origin key (launch.json "references"[key],
    the SAME reference as the S_paper493 sim-attack arm); a missing key or a copy whose sha256 differs is a launch error;
  - for keys that also have a primary_v3 pin (merged_refs_launch_2026-10-14.json, the merged refs index simh_full forked
    from; staged, sha-pinned) the two pins are asserted EQUAL;
  - t43_auto_workflow is labelled "regenerated; primary_v3 excluded this main (25e)" (09).
The +311 harness must equal one committed frozen manifest; its identity to simh_full is EQUAL (@ d1cf9bbc8). Writes only --out.
  <sept venv python> simh_sp493_manifest.py --launch sp493_t196 --host <h> --config <staged pilot_simh_sp493_t196.yaml> --out <file> [--routed-root <root>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

STAGE = Path("/home/aswin/phase2/simh_rlr_stage")
sys.path.insert(0, str(STAGE))
import simh_rlr_manifest as rm  # noqa: E402  (reviewed helpers: sha, settings_digest, harness_now, eval_task_name, FROZEN_311)

REFOLD = (Path("/home/aswin/phase2/il_refindex_old"), "79acbfc1ef4523f9a57cc19dc876252159c6c1d62521001f1ac71e2644036940")
# g5 (d9 ~19:35): the design fork's shared L3 index (built from the B fork's new-main honest_a; committed as config/sp493sim/refindex_L3.json);
# its launch.json sha is filled in once committed, and the writer refuses until then
REFL3 = (Path("/home/aswin/phase2/sp493sim/refindex_L3"), "7f98a7de46d2748c31333427191bd3cc3a55a3cdcb07670c0a6a3638f1de064a")  # @ 627400b22 (design fork)
T196_SRC = ("tasks_il_old_all.json", "69c709d91da2f8952c0a3f4ceddf3e05cac5044f73a596eb253bb5763f70ee4e")
G5_SRC = ("sp493_v2_r1_entries.json", "2f005d1b51d295fa547d4fcda4a0efd0a5cabba35e6890743be075580565d1f7")  # make_sp493_g5.py: the B fork's launched v2 r1 entries
MERGED = ("merged_refs_launch_2026-10-14.json", "26487d40031765d91f195f6086247522c1b2e4a3235902d2a8036a7a4d31409b")
KEYS_G4 = ["t01_media_processing", "t02_clinical_trial", "t03_auto_workflow", "t05_user_tracking", "t06_file_indexing", "t09_web_scraping", "t115_model_registry", "t14_tiktok", "t15_user_tracking", "t16_file_indexing", "t18_model_registry", "t19_web_scraping", "t20_shared_dotfiles", "t43_auto_workflow"]  # 14 keys (09 addendum 6: 13 of the 27 admitted from the A25 pilot)
KEYS_FILL = ["t04_tiktok", "t07_slack_clone", "t08_model_registry", "t11_media_processing", "t12_clinical_trial", "t13_auto_workflow", "t17_slack_clone", "t48_model_registry"]  # 8 keys (09 addendum 6); the 15-key list is superseded
KEYS_G5 = ["t675_shared_dotfiles", "t699_model_registry", "t750_media_processing", "t762_auto_workflow", "t763_media_processing", "t765_media_processing", "t776_media_processing", "t789_clinical_trial", "t805_auto_workflow", "t809_auto_workflow", "t815_media_processing", "t822_tiktok", "t861_user_tracking", "t876_clinical_trial", "t886_clinical_trial", "t892_clinical_trial", "t903_file_indexing", "t921_user_tracking", "t925_model_registry", "t927_tiktok", "t937_slack_clone", "t939_user_tracking", "t947_auto_workflow", "t948_slack_clone", "t950_web_scraping", "t951_auto_workflow", "t953_tiktok", "t958_model_registry", "t969_tiktok", "t971_web_scraping", "t972_web_scraping", "t975_model_registry", "t977_file_indexing", "t989_user_tracking", "t996_web_scraping"]  # 35: t987 dropped (no_usable_reference), t699 in (09 via d9 ~21:5x)
T43 = "t43_auto_workflow"
SPECS = {  # launch -> (planned keys, registration tag, routed-root regex)
    "sp493_t196": (KEYS_G4, "S_paper493 g4 (09 Append 39 + amendment 1)",
                   re.compile(r"^sp493_g[0-9]+_simh(?:_s[0-9]+|_rlr(?:[2-9]|[1-9][0-9]+)?)?_v1(?:_try[0-9]+)?__sim_honest$")),  # 09 @ 9cb3d0291
    "t196fill": (KEYS_FILL, "A25 fill-in (09 @ 286b2f518; not Append 39; second refresh only)",
                 re.compile(r"^simh_t196fill(?:_s[0-9]+)?_v1(?:_try[0-9]+)?__sim_honest$")),  # 09's proposed regex (d9 ~17:50)
    "sp493_new": (KEYS_G5, "S_paper493 g5 (09 Append 39 + amendment 1; as-run entries, d9 ~19:35)",
                  re.compile(r"^sp493_g[0-9]+_simh(?:_s[0-9]+|_rlr(?:[2-9]|[1-9][0-9]+)?)?_v1(?:_try[0-9]+)?__sim_honest$")),
}
WHERE = {"sp493_t196": (REFOLD, T196_SRC), "t196fill": (REFOLD, T196_SRC), "sp493_new": (REFL3, G5_SRC)}  # launch -> (index, cell source)
RULE = ("S_paper493 sim-honest (09 Append 39; rulings via d9 ~17:30): each cell forks at step 0 from il_refindex_old's pinned "
        "reference for its origin key (the same reference as the S_paper493 sim-attack arm), copied verbatim by "
        "--references-from with its sha256 re-checked; for keys with a primary_v3 pin the two pins are equal")


RULE_G5 = ("S_paper493 g5 sim-honest (09 Append 39; rulings via d9 ~19:25-19:35): each cell forks at step 0 from refindex_L3's pinned "
           "reference for its key (the B fork's new-main honest_a, shared byte for byte with the S_paper493 sim-attack L3 cells on the "
           "same main), copied verbatim by --references-from with its sha256 re-checked")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch", required=True, choices=list(SPECS))
    ap.add_argument("--host", required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--routed-root", required=True, help="the 09-confirmed root for this launch (SPECS regex)")
    args = ap.parse_args()
    live = subprocess.run(["pgrep", "-f", r"run_pilot\.py"], capture_output=True, text=True)
    assert live.returncode == 1 and not live.stdout.strip(), f"a launcher is running on this box (pgrep rc {live.returncode})"
    k, sha = args.launch, rm.sha
    KEYS, TAG, ROOT_RE = SPECS[k]
    (INDEX, INDEX_LAUNCH_SHA), TASKS = WHERE[k]
    assert re.fullmatch(r"[0-9a-f]{64}", INDEX_LAUNCH_SHA), f"{INDEX} launch.json sha not pinned yet"
    assert ROOT_RE.match(args.routed_root), args.routed_root
    rm.HARNESS = HARNESS = rm.H311
    cfg = yaml.safe_load(args.config.read_text())
    run_yaml, sim_yaml = HARNESS / "config" / "run.yaml", HARNESS / "config" / "sim.yaml"
    run, sim = yaml.safe_load(run_yaml.read_text()), yaml.safe_load(sim_yaml.read_text())
    simulator = run["simulators"]["september"]
    tree = Path(simulator["tree"]).expanduser()
    sys.path.insert(0, str(HARNESS))
    from run_attack_check import src_sha256  # the launcher's own tree hash

    tree_hash = src_sha256(tree)
    assert tree_hash == simulator["src_sha256"], (tree_hash, simulator["src_sha256"])
    now = rm.harness_now()
    match = None
    for name, want in rm.FROZEN_311.items():
        assert sha(STAGE / name) == want, f"staged {name} is not the committed manifest"
        frozen = [line for line in (STAGE / name).read_text().splitlines() if "  ./config/harness_manifest" not in line]
        if frozen == now:
            match = name
    assert match, "the harness equals none of the committed frozen +311 manifests"
    manifest_file = STAGE / match
    settings = Path(run["settings_dir"]).expanduser()
    sdig = rm.settings_digest(settings)
    tasks_fu = STAGE / Path(cfg["tasks"]).name
    assert cfg["tasks"] == f"config/tasks_simh_{k}.json", cfg["tasks"]
    assert sha(STAGE / TASKS[0]) == TASKS[1], f"the staged cell source {TASKS[0]} is not the pinned one"
    assert sha(STAGE / MERGED[0]) == MERGED[1], "the staged primary_v3 merged refs index is not the pinned one"
    cells = json.loads(tasks_fu.read_text())["cells"]
    src = {c["key"]: c for c in json.loads((STAGE / TASKS[0]).read_text())["cells"]}
    canon = lambda c: json.dumps(c, sort_keys=True)  # noqa: E731
    assert sorted(c["key"] for c in cells) == sorted(KEYS) and len(cells) == len(KEYS), "cells != the planned keys"
    assert all(canon(src[c["key"]]) == canon(c) for c in cells), f"cells are not {TASKS[0]}'s entries"
    assert sha(INDEX / "launch.json") == INDEX_LAUNCH_SHA, f"{INDEX}/launch.json is not the pinned one"
    refs = json.loads((INDEX / "launch.json").read_text())["references"]
    merged = json.loads((STAGE / MERGED[0]).read_text())["references"]
    digest_sets = json.loads((HARNESS / cfg["digest_sets"]).read_text())["sets"] if cfg.get("digest_sets") else {}
    common = {"simulator_src_sha256": tree_hash, "sim_yaml_sha256": sha(sim_yaml), "run_yaml_sha256": sha(run_yaml),
              "settings_sha256": sdig["sha256"], "mode": "honest"}
    rows, bad = [], []
    for c in sorted(cells, key=lambda c: c["key"]):
        key, ref = c["key"], refs.get(c["key"])
        row = {"key": f"{k}|{key}", "cell": c["cell"], "task": key, "env": c["env"], "main": c.get("main"),
               "strategy": bool(c.get("approach_chars")), "registration": TAG}
        problems = []
        if ref is None:
            row["status"] = "missing"; problems.append("no reference for the key in il_refindex_old (a launch error)")
        else:
            copy = Path(ref["copy"])
            copy_sha = sha(copy) if copy.is_file() else None
            if copy_sha != ref["sha256"]:
                problems.append("the copy is missing or its sha256 differs from il_refindex_old's pin (a launch error)")
            task_name = rm.eval_task_name(copy) if copy_sha else ""
            if "honest" not in task_name or "attack" in task_name:
                problems.append(f"eval task name {task_name!r} is not an honest run")
            pv3 = merged.get(key)
            row |= {"status": "pinned", "reference_copy": str(copy), "eval_sha256": copy_sha, "eval_task": task_name,
                    "index_pin": ref["sha256"], "index_source": ref.get("index_source"), "relayed_from": ref.get("relayed_from"),
                    "primary_v3_pin": (pv3 or {}).get("sha256"),
                    "primary_v3_equal": (None if pv3 is None else pv3.get("sha256") == ref["sha256"]),
                    "digest_set": digest_sets.get(key, [])}
            if pv3 is not None and pv3.get("sha256") != ref["sha256"]:
                problems.append("il_refindex_old's pin differs from primary_v3's pin")
            if key == T43:
                row["label"] = "regenerated; primary_v3 excluded this main (25e)"
            row["rendered_input_sha256"] = hashlib.sha256(json.dumps(
                common | {"eval_sha256": copy_sha, "env": c["env"], "digest_set": row["digest_set"]}, sort_keys=True).encode()).hexdigest()
        if problems:
            row["provenance_problems"] = problems; bad.append(key)
        rows.append(row)
    sys.path.insert(0, str(tree / "src"))
    import check_sim_samples as css

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
            r.setdefault("provenance_problems", []).append(f"render failed: {e!r}"[:300]); bad.append(r["task"])
    if bad:
        detail = {r["task"]: r["provenance_problems"] for r in rows if r.get("provenance_problems")}
        raise SystemExit(f"provenance check failed, nothing written: {json.dumps(detail)}")
    out = {
        "launch": k, "family": "311", "host_box": args.host, "launch_dir": f"/home/aswin/phase2/runs/2026-10-15_simh_{k}",
        "written": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds"),
        "registration": TAG, "world": k, "key_format": f"{k}|<task>", "arm": "sim_honest@september", "reference_rule": RULE_G5 if k == "sp493_new" else RULE, "references_from": str(INDEX),
        "config": {"path": str(args.config), "sha256": sha(args.config), "resolved": cfg},
        "actor": {"policy": run.get("policy"), "policy_limits": run.get("policy_limits")},
        "simulator": {"name": "september", "tree": str(tree), "src_sha256": tree_hash, "run_yaml_entry": simulator,
                      "sim_yaml_sha256": sha(sim_yaml), "sim_yaml": sim},
        "harness": {"path": str(HARNESS), "frozen_manifest": str(manifest_file), "frozen_manifest_sha256": sha(manifest_file),
                    "run_yaml_sha256": sha(run_yaml), "run_pilot_sha256": sha(HARNESS / "run_pilot.py"),
                    "check_sim_samples_sha256": sha(HARNESS / "check_sim_samples.py"),
                    "identity_to_simh_full": "EQUAL (results/sp493_simh_identity_2026-10-15_172827.json @ d1cf9bbc8; 09 accepted)"},
        "settings": {"path": str(settings), **sdig, "label": "host-local digest; not an identity (25h §13)"},
        "task_files": {"tasks": {"path": str(tasks_fu), "sha256": sha(tasks_fu)}, "source": {"path": str(STAGE / TASKS[0]), "sha256": TASKS[1]}},
        "index": {"path": str(INDEX), "launch_json_sha256": INDEX_LAUNCH_SHA, "primary_v3_merged_refs": {"path": str(STAGE / MERGED[0]), "sha256": MERGED[1]}},
        "counts": {"cells": len(rows), "pinned": sum(r["status"] == "pinned" for r in rows),
                   "primary_v3_equal": sum(r.get("primary_v3_equal") is True for r in rows),
                   "no_primary_v3_pin": sum(r.get("primary_v3_equal") is None for r in rows)},
        "routed_root": {"root": args.routed_root, "run_dir": f"/home/aswin/phase2/runs/2026-10-15_simh_{k}"},
        "g5_rules": ({"cell": "each cell = the B fork's as-run honest entry for its key (launched v2 files @ 548c75d30), with 09's gates asserted in make_sp493_g5.py: input fields equal (@ 27c327b03), only prep fields added " + json.dumps(["approach_chars", "approach_s3", "approach_sha256", "attack_pair_launch", "template_sha256"]) + ", approach_sha256 == facts v2 (@ d800edc6a), 98364da05 -> 27c327b03 equal by key",
                      "t699_model_registry": "main = model_registry|locked_stage_filter_inverted (T196 origin key t98); pinned to refindex_L3's usable rerun2 reference (the launcher's selected one, not the counted attempt; 09 via d9 ~21:5x)",
                      "dropped": {"t987_model_registry": "no_usable_reference (refindex_L3 @ 627400b22): no sim-honest cell (09)"},
                      "reference": "refindex_L3, shared byte for byte with the S_paper493 sim-attack L3 cells on each main"} if k == "sp493_new" else {}),
        "deferred": ({"t98_model_registry": "no reference in il_refindex_old (09 Append 35 @ 221bb94ea); fork pin from the B fork's new honest_a; Phase B (g5)"} if k == "sp493_t196" else {}),
        "excluded": ({"t88": "failed the September preflight in il_old", "t139": "no pinned reference", "t151": "no pinned reference", "t169": "failed the September preflight in il_old"} if k == "t196fill" else {}),
        "admitted_from_pilot_not_generated": (["t21", "t23", "t28", "t30", "t31", "t32", "t33"] if k == "t196fill" else ["t22_clinical_trial", "t24_tiktok", "t25_user_tracking", "t26_file_indexing", "t27_slack_clone", "t29_web_scraping", "t34_tiktok", "t35_user_tracking", "t36_file_indexing", "t37_slack_clone", "t38_model_registry", "t39_web_scraping", "t40_shared_dotfiles"]),
        "reference_census": {"rule": "refindex: fork source == il_refindex_old's pin per key; primary_v3 pins equal where present", "violations": [], "equal": True},
        "rendered_input_definition": "sha256 of the canonical JSON of " + json.dumps(sorted(common) + ["eval_sha256", "env", "digest_set"]),
        "cells": rows,
    }
    with open(args.out, "x") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(f"-> {args.out}: {out['counts']}")


if __name__ == "__main__":
    main()
