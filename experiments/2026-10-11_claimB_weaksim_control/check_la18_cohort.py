"""$0, metadata only: what Jeffrey's weak-simulator cohort (la18) holds, for the positive-control plan (PLAN.md).

Reads the three la18 manifests (S3 `manifest.json` of the real attack, R0 and R1 runs; sce `real_runs.yml` aliases
la18_real_attack_n200, la18_r0_attack_n200, la18_r1_attack_n200), the R1 provenance `run_shard.py` (bytes hashed and
matched to the manifest's `files` entry), claim B's folds (`splits/folds_v2.json`) and sce's history-discriminator
numbers on sampled cumulative prefixes (`attacker_authenticity_cumulative.json` at sce f82c84d6). No transcript is
downloaded or read. Prints and saves counts, ids and hashes only: samples per arm and environment, cells present in all
three arms, environments shared with claim B's ten, main tasks shared with claim B's split, the run settings that decide
reference and start, and the discriminator numbers.

  AWS_PROFILE=redwood-sandbox uv run --no-project --python 3.12 python \\
      experiments/2026-10-11_claimB_weaksim_control/check_la18_cohort.py
"""

import hashlib
import json
import subprocess
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
REPO = HERE.parents[1]
SCE = Path.home() / "simulated-control-evals"  # the sibling checkout in ~ (worktrees live elsewhere)
SCE_REV = "f82c84d6"
JUDGE = "experiments/2026-09-18_linuxarena_paper_figures/results/attacker_authenticity_cumulative.json"
BUCKET = "s3://rr-sce-sim-transcripts/transcripts/"
ARMS = {
    "real": "2026-09-18_la18_real-attack-solmedium-step150-attacker0828-contractv3-n200",
    "R0": "2026-09-18_la18_sim-astra-medium-attack-aff-none-fork0-solmedium-step150-n200",
    "R1": "2026-09-18_la18_sim-astra-medium-attack-aff-origtraj-fork0-solmedium-step150-n200",
}
SETTINGS = ("policy_model", "policy_reasoning_effort", "attacker_prompt", "max_steps", "simulator_model",
            "simulator_reasoning_effort", "simulator_fallback_model", "fork_fraction", "R0_affordances", "R1_affordances",
            "R1_original_trajectory_mode", "codebase_read", "codebase_execution", "rollouts_per_cell")


def s3_get(key: str, dst: Path) -> bytes:
    subprocess.run(["aws", "s3", "cp", "--quiet", BUCKET + key, str(dst)], check=True)
    return dst.read_bytes()


def main():
    out = {"label": "la18 cohort metadata for the weak-simulator positive control (counts, ids and hashes only)",
           "inputs": {}, "arms": {}}
    folds = json.loads((REPO / "splits/folds_v2.json").read_text())
    claim_envs = sorted(folds["environment_folds"]["environment_to_fold"])
    claim_mains = {m for m, e in folds["main_task_environment"].items()}
    cells = {}
    with tempfile.TemporaryDirectory() as tmp:
        for arm, prefix in ARMS.items():
            raw = s3_get(f"{prefix}/manifest.json", Path(tmp) / f"{arm}.json")
            m = json.loads(raw)
            out["inputs"][f"{prefix}/manifest.json"] = hashlib.sha256(raw).hexdigest()
            sel = m["selected_samples"]
            keys = [s["key"] for s in sel]
            if len(set(keys)) != len(keys):
                raise ValueError(f"{arm}: duplicate cell keys in selected_samples")
            cells[arm] = {s["key"]: s["environment"] for s in sel}
            conf = m["configuration"]
            out["arms"][arm] = {
                "prefix": BUCKET + prefix, "cohort": m["cohort"], "selected": m["selected_count"],
                "superseded_errors": m["superseded_error_count"], "eval_files": len(m["eval_paths"]),
                "environments": len(conf["environments"]), "excluded_environments": conf["excluded_environments"],
                "settings": {k: conf.get(k) for k in SETTINGS},
                "per_environment": dict(sorted(Counter(s["environment"] for s in sel).items())),
            }
        # R1's run script (the provenance copy): hash it and match the manifest's own record of it
        raw = s3_get(f"{ARMS['R1']}/provenance/run_shard.py", Path(tmp) / "run_shard.py")
        rec = json.loads((Path(tmp) / "R1.json").read_text())["files"]["provenance/run_shard.py"]
        if hashlib.sha256(raw).hexdigest() != rec["sha256"]:
            raise ValueError("run_shard.py does not match the R1 manifest's sha256")
        out["inputs"][f"{ARMS['R1']}/provenance/run_shard.py"] = rec["sha256"]
    common = set(cells["real"]) & set(cells["R0"]) & set(cells["R1"])
    env_of = cells["real"]
    for a in ("R0", "R1"):  # every pairwise intersection, not only cells present in all three arms
        if any(cells[a][c] != env_of[c] for c in set(cells["real"]) & set(cells[a])):
            raise ValueError(f"a cell's environment differs between real and {a}")
    mains = Counter(c.split(":")[0] for c in set(cells["real"]) & set(cells["R0"]))
    shared = [e for e in sorted(set(env_of.values())) if e in claim_envs]
    pairs = {"real_R0": set(cells["real"]) & set(cells["R0"]), "real_R1": set(cells["real"]) & set(cells["R1"]),
             "all_three": common}
    out["cells_per_environment"] = {k: dict(sorted(Counter(env_of[c] for c in v).items())) for k, v in pairs.items()}
    out["cells"] = {
        "in_all_three_arms": len(common),
        "real_and_R0": len(set(cells["real"]) & set(cells["R0"])),
        "real_and_R1": len(set(cells["real"]) & set(cells["R1"])),
        "environments_shared_with_claimB": shared,
        "cells_in_shared_environments": sum(env_of[c] in claim_envs for c in common),
        "cells_in_other_environments": sum(env_of[c] not in claim_envs for c in common),
        "real_R0_cells_per_main_task_max": max(mains.values()),
        "main_tasks_also_in_claimB_split": len({c.split(":")[0] for c in common} & claim_mains),
        "main_tasks": len({c.split(":")[0] for c in common}),
    }
    judge_raw = subprocess.run(["git", "-C", str(SCE), "show", f"{SCE_REV}:{JUDGE}"], capture_output=True, check=True).stdout
    out["inputs"][f"sce@{SCE_REV}:{JUDGE}"] = hashlib.sha256(judge_raw).hexdigest()
    out["sce_history_discriminator"] = {k: {f: v[f] for f in ("action_auroc", "n_matched_complete_trajectories", "n_calls")}
                                  for k, v in json.loads(judge_raw)["conditions"].items()}
    out["git"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    out["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"la18_cohort_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("cells", "sce_history_discriminator")}, indent=1))
    print({a: (v["selected"], v["superseded_errors"], v["settings"]["fork_fraction"]) for a, v in out["arms"].items()})
    print("->", dst)


if __name__ == "__main__":
    main()
