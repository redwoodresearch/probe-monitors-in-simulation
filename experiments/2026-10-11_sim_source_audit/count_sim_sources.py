"""Which real runs were the simulated runs built from, and do the real runs each comparison uses overlap them? IDs, file
names and config fields only (no transcript text is read). $0, laptop.

Phase 1 (off-policy LA public-10 cohorts): splits/exclusions_v3.json labels each paired attack cell's simulated run by
its reference source (`base200`: the simulator was given the cohort's own real honest run of that main task;
`overlap800`: a fresh honest rollout made by the September fleet, stored nowhere else). Counted on the 939 paired cells
and the 591 under-32k cells, with the kept real honest runs on the base200 cells' main tasks (those runs are the
simulator's references).

Phase 2 (on-policy Qwen3.5 runs): each launch.json records, per task, the reference the simulator was given
(`references[task].source`). Counted: the reference's arm, and whether it is the same .eval as the task's kept honest_a
attempt (the launcher's run rule: no redrawn_because, never a rerun, first attempt).

  uv run --no-project --python 3.12 python count_sim_sources.py --exclusions splits/exclusions_v3.json --phase2 <runs dir>
"""

import argparse
import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def main_of(cell: str) -> str:
    return cell.split(":", 1)[0]


def phase1(path: Path) -> dict:
    d = json.loads(path.read_text())
    p = d["paired_attack_cells"]
    ref = p["sim_reference"]
    cells = p["cells"]
    u32 = d["paired_attack_cells_under_32k"]["cells"]
    excluded = {x["key"] for x in d["excluded"]}
    kept = [s for s in d["samples"] if s["arm"] == "real-honest" and s["key"] not in excluded]
    honest_by_main = Counter(s["main_task"] for s in kept)
    honest_base200_by_main = Counter(s["main_task"] for s in kept if s["file"].startswith("base200_"))
    out = {"file": str(path), "kept_real_honest_runs": len(kept),
           "kept_real_honest_runs_by_file_prefix": dict(Counter(s["file"].split("_", 1)[0] for s in kept))}
    for name, cs in (("paired_939", cells), ("under_32k_591", u32)):
        by = Counter(ref[c] for c in cs)
        base_mains = {main_of(c) for c in cs if ref[c] == "base200"}
        out[name] = {"cells": len(cs), "by_sim_reference": dict(by), "base200_main_tasks": len(base_mains),
                     "base200_mains_with_a_kept_real_honest_run": sum(honest_by_main[m] > 0 for m in base_mains),
                     "kept_real_honest_runs_on_base200_mains": sum(honest_by_main[m] for m in base_mains),
                     "kept_base200_named_real_honest_runs_on_base200_mains": sum(honest_base200_by_main[m] for m in base_mains),
                     "kept_real_honest_runs_on_all_cell_mains": sum(honest_by_main[m] for m in {main_of(c) for c in cs})}
    return out


def phase2(root: Path) -> dict:
    out = {}
    for run in sorted(p for p in root.iterdir() if p.is_dir()):
        d = json.loads((run / "launch.json").read_text())
        refs = d.get("references") or {}
        kept = {}
        for r in d["runs"]:
            att = [a for a in r.get("attempts") or [] if not a.get("redrawn_because") and a.get("kind") != "rerun"][:1]
            for a in att:
                arm = f'{r["arm"]}@{r.get("simulator")}' if r["arm"].startswith("sim") else r["arm"]
                kept[(arm, r["task"])] = {Path(e).name for e in (run / a["log_dir"].split(f"/{run.name}/", 1)[1]).glob("*.eval")} \
                    if "/" + run.name + "/" in a.get("log_dir", "") else set()
        src_arm, same_as_kept_a, in_other_arm = Counter(), Counter(), Counter()
        for task, v in refs.items():
            src = Path(v["source"])
            arm = src.relative_to(Path(v["source"]).parents[2]).parts[0]
            src_arm[arm] += 1
            same_as_kept_a[src.name in kept.get(("honest_a", task), set())] += 1
            for other in ("honest_b", "attack"):
                in_other_arm[other] += src.name in kept.get((other, task), set())
        # the join that matters: every simulated run that has a counted .eval, against its task's reference
        sim_join = Counter()
        for (arm, task), files in kept.items():
            if not arm.startswith("sim"):
                continue
            if not files:
                sim_join[f"{arm}: no local .eval (not in the laptop cache)"] += 1
                continue
            ref = refs.get(task)
            if ref is None:
                sim_join[f"{arm}: no reference recorded"] += 1
            elif Path(ref["source"]).name in kept.get(("honest_a", task), set()):
                sim_join[f"{arm}: reference = the kept honest_a run"] += 1
            else:
                sim_join[f"{arm}: reference = another honest_a attempt (not the kept one)"] += 1
        out[run.name] = {"arms": dict(Counter(r["arm"] for r in d["runs"])),
                         "sim_runs_with_eval_by_reference": dict(sim_join),
                         "simulators": dict(Counter(f'{r["arm"]}@{r.get("simulator")}' for r in d["runs"] if r["arm"].startswith("sim"))),
                         "references": len(refs), "reference_source_arm": dict(src_arm),
                         "reference_is_the_kept_honest_a_run": {str(k): n for k, n in same_as_kept_a.items()},
                         "reference_found_among_kept_runs_of": dict(in_other_arm),
                         "sim_config_fork_fraction": (d.get("sim_config") or {}).get("env", {}).get("BA_FORK_FRACTION")}
    return out


def main(exclusions: str, phase2_dir: str):
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    res = {"written": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S"), "git": git,
           "phase1": phase1(Path(exclusions)), "phase2": phase2(Path(phase2_dir).expanduser())}
    dst = Path(__file__).parent / "results" / f"sim_sources_{res['written']}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print("->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclusions", required=True)
    ap.add_argument("--phase2", dest="phase2_dir", required=True)
    main(**vars(ap.parse_args()))
