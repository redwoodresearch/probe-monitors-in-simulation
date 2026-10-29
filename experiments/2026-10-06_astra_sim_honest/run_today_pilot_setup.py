"""Set up the small today's-simulator pilot: 20 tasks x (Astra simulated honest + Astra simulated attack).

The pilot differs from the September cohorts in one intended way: the simulator code (sims main at a pinned commit, which
includes the numbered text_editor diff format of 4b31bd30). Everything else is held fixed:
  - the same Astra settings (config/today_pilot_*.json; narrow router, no briefing, codebase read, 32,768 cap, fork 0);
  - the same references: honest runs use the rebuilt reference of the 437-task batch; attack runs use the cell's own
    September reference, which is that same reference, since every chosen reference comes from a paired attack cell;
  - the same attacker guidance: today's code takes `attack_approach` from the fork source's metadata when it is
    non-empty, so each attack reference is a copy of the honest one carrying the September cell's `attack_approach`.
    An empty value would fall back to the side task's first strategy, so tasks are drawn only from cells where
    September's approach was non-empty.
Tasks: from the 437, excluding media_processing and model_registry (their environment digests were stamped locally),
paired cells with a non-empty September approach; 3 tasks for the 4 envs with the most eligible cells, 2 for the other
4, drawn with a fixed seed.

Checks with today's code (no model calls): the attack reference plans the September approach, side task and fork.

Run inside today's sims tree, with the September snapshot's env checkouts:
  cd ~/simulated-control-evals-today && CONTROL_TOWER_SETTINGS_DIR=~/simulated-control-evals-sept/.settings uv run \\
    python ~/probe-shift-sims/experiments/2026-10-06_astra_sim_honest/run_today_pilot_setup.py \\
    --references <results/references_*.json> --honest-refs <built references dir> --out <attack references dir>
"""

import argparse
import io
import json
import random
import subprocess
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import boto3
from botocore.config import Config
from inspect_ai.log import read_eval_log, write_eval_log

import runner
from runner.build.run_plan import planned_samples

HERE = Path(__file__).parent
EXCLUDED_ENVS = ("media_processing", "model_registry")
SEED = 0


LOCAL: Path | None = None  # --sim-logs: a local `aws s3 sync` copy of the cohort's logs/, read instead of S3


def stored_sample(uri: str, sample_id: str) -> dict:
    bucket, key = uri.removeprefix("s3://").split("/", 1)
    local = LOCAL / key.rsplit("/", 1)[1] if LOCAL else None
    if local and local.exists():
        body = local.read_bytes()
    else:  # only now does the setup need AWS
        s3 = boto3.Session(profile_name="redwood-sandbox").client(
            "s3", config=Config(retries={"max_attempts": 10, "mode": "adaptive"}, read_timeout=120))
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    with zipfile.ZipFile(io.BytesIO(body)) as z:
        for name in z.namelist():
            if name.startswith("samples/") and name.endswith(".json"):
                sample = json.loads(z.read(name))
                if sample["id"] == sample_id:
                    return sample
    raise KeyError(f"{sample_id} not in {uri}")


def reference_path(ref: dict, refs_dir: Path) -> Path:
    return refs_dir / ref["environment"] / ref["fork_run"] / f"{ref['source_sample'].replace(':', '_')}.eval"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--honest-refs", type=Path, required=True, help="the batch's built references (verified)")
    p.add_argument("--out", type=Path, required=True, help="where the attack references go")
    p.add_argument("--sim-logs", type=Path, help="local copy of the September attack cohort's logs/ (else S3)")
    args = p.parse_args()
    global LOCAL
    LOCAL = args.sim_logs
    refs = json.loads(args.references.read_text())["references"]

    approaches, eligible = {}, defaultdict(list)
    for main, ref in sorted(refs.items()):
        if ref["environment"] in EXCLUDED_ENVS or not ref["paired"]:
            continue
        approach = (stored_sample(ref["sim_uri"], ref["sim_sample_id"])["metadata"].get("attack_approach") or "").strip()
        if approach:
            approaches[main] = approach
            eligible[ref["environment"]].append(main)
    by_size = sorted(eligible, key=lambda e: (-len(eligible[e]), e))
    quota = {env: 3 if i < 4 else 2 for i, env in enumerate(by_size)}
    rng = random.Random(SEED)
    chosen = sorted(m for env in sorted(eligible) for m in rng.sample(sorted(eligible[env]), quota[env]))
    assert len(chosen) == 20, len(chosen)

    rows = []
    for main in chosen:
        ref = refs[main]
        log = read_eval_log(str(reference_path(ref, args.honest_refs)))
        [sample] = log.samples
        sample.metadata["attack_approach"] = approaches[main]
        sample.metadata["reference_provenance"]["attack_approach_from"] = ref["sim_uri"]
        path = reference_path(ref, args.out.resolve())
        path.parent.mkdir(parents=True, exist_ok=True)
        write_eval_log(log, str(path))
        [row] = planned_samples([path], one_per_cell=True, fork_fraction=0.0)
        assert row.approach == approaches[main], f"{main}: today's code does not plan the September approach"
        sept = sample.metadata["reference_provenance"]["sept_fork"]
        got = {"run": row.fork.run, "source_sample": row.fork.source_sample, "of_actions": row.fork.of_actions,
               "source_step_sizes": list(row.fork.source_step_sizes), "after": row.fork.after}
        assert row.side is None and got == {k: sept[k] for k in got}, f"{main}: fork differs: {got}"
        rows.append({"main": main, "environment": ref["environment"], "cell": ref["cell"],
                     "honest_reference": str(reference_path(ref, args.honest_refs)), "attack_reference": str(path),
                     "attack_approach_chars": len(approaches[main])})
        print(f"{ref['environment']:16s} {ref['cell']}")

    sims_tree = Path(runner.__file__).resolve().parents[2]
    git = lambda *a: subprocess.run(["git", "-C", str(sims_tree), *a], capture_output=True, text=True).stdout.strip()
    out = {"created": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds"),
           "sims_commit": git("rev-parse", "HEAD"), "sims_dirty": bool(git("status", "--porcelain", "src")),
           "seed": SEED, "quota": quota, "eligible_per_env": {e: len(v) for e, v in sorted(eligible.items())},
           "tasks": rows}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S")
    with open(HERE / "results" / f"today_pilot_tasks_{stamp}.json", "x") as f:
        f.write(json.dumps(out, indent=1))
    print(f"20 tasks; wrote results/today_pilot_tasks_{stamp}.json")


if __name__ == "__main__":
    main()
