"""Author-split classifier on Modal CPU (RULE.md Amendment 1): both splits in parallel, one 16-CPU container each, with
the analysis code of analyze_POSTHOC_author_split.py unchanged. The rows come from the Volume: q2's fix-full shard
(sha-checked) and the v4 staging's original simulated honest and real honest rows (the same staging code the local
rebuild used). Each result JSON (numbers and n-grams of at most 5 characters only) goes to the Volume and is committed;
nothing else is written. No AWS credentials are used.

  uv run --no-project --python 3.12 --with modal==1.6.1 modal run --detach \\
      experiments/2026-10-13_claimB_author_split/run_author_split_modal.py
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REPO = HERE.parents[1] if modal.is_local() else Path("/root")
APP = "probe-shift_aswin-surya_sub-claimb-author-split_kill-2026-11-12"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
V4 = "2026-10-06_la_segment-end-v4_qwen3-32b_v1"  # staged inputs, scrub level 0: real-honest and sim-honest arms
FX = "2026-10-13_q2_fixfull_inputs_v2/inputs/shard_12.jsonl"  # q2's sim-honest-fx-full (stage record @ 82909df0)
OUT = "2026-10-13_claimB_author_split_v1/results"  # this job's results on the Volume (new folder)
FOLDER = "experiments/2026-10-13_claimB_author_split"
if modal.is_local():  # add_local_python_source needs probe_shift importable locally (uv --no-project has no src path)
    sys.path.insert(0, str(REPO / "src"))

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==2.5.4",
        "scipy==1.18.1",
        "scikit-learn==1.9.1",
        "joblib==1.6.0",
        "boto3==1.43.113",
        "pyyaml==6.0.3",
    )
    .env({"AUTHOR_SPLIT_STAGE": "/tmp/author_split_stage"})
    .add_local_python_source("probe_shift")
    .add_local_file(
        REPO / "splits" / "exclusions_v4.json", "/root/splits/exclusions_v4.json"
    )
    .add_local_file(
        HERE / "analyze_POSTHOC_author_split.py",
        f"/root/{FOLDER}/analyze_POSTHOC_author_split.py",
    )
)


def say(*a):
    print(
        datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%H:%M:%S"),
        *a,
        flush=True,
    )


@app.function(
    image=image,
    volumes={"/acts": acts},
    cpu=16,
    memory=65536,
    timeout=3 * 3600,
    retries=0,
)
def run_split(split: str, git: str) -> str:
    sys.path.insert(0, f"/root/{FOLDER}")
    import hashlib

    import analyze_POSTHOC_author_split as A

    say(split, "start")
    raw = (Path("/acts") / FX).read_bytes()
    if hashlib.sha256(raw).hexdigest() != A.FX_SHA:
        raise SystemExit("the fix-full shard's sha256 does not match the stage record")
    fx = {r["key"]: r for r in map(json.loads, raw.decode().splitlines())}
    sim, first, second, ex_sha = A.cohort_rows()
    if set(fx) != set(sim) or any(
        r["arm"] != "sim-honest-fx-full" for r in fx.values()
    ):
        raise SystemExit("the fix-full rows are not the 437 simulated honest runs")
    want = {("sim-honest", k) for k in sim} | {
        ("real-honest", r["key"]) for r in [*first.values(), *second.values()]
    }
    rows = {}
    for p in sorted((Path("/acts") / V4 / "inputs").glob("shard_*.jsonl")):
        for line in p.read_text().splitlines():
            it = json.loads(line)
            if (it["arm"], it["key"]) in want:
                if (it["arm"], it["key"]) in rows:
                    raise SystemExit("a run appears twice in the v4 inputs")
                rows[(it["arm"], it["key"])] = it["messages"]
    if set(rows) != want:
        raise SystemExit(
            f"{len(want - set(rows))} wanted runs missing from the v4 inputs"
        )
    st = {
        "real": {k: m for (a, k), m in rows.items() if a == "real-honest"},
        "orig": {k: m for (a, k), m in rows.items() if a == "sim-honest"},
    }
    say(
        split,
        "rows",
        {"fix_full": len(fx), "orig": len(st["orig"]), "real": len(st["real"])},
    )
    out, name = A.analyze(
        split, sim, first, second, ex_sha, st, fx, git, f"Volume: {FX} and {V4} inputs"
    )
    out["modal"] = {"app": APP, "cpu": 16, "memory_mib": 65536}
    dst = Path("/acts") / OUT / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    acts.commit()
    say(split, "wrote", str(dst))
    return str(dst)


@app.function(image=image, cpu=0.25, memory=512, timeout=3 * 3600 + 600, retries=0)
def orchestrate(git: str) -> list[str]:
    """Runs remotely so the two splits survive a lost local client under --detach."""
    return list(run_split.map(["replies", "commands"], kwargs={"git": git}))


@app.local_entrypoint()
def main():
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=HERE,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", FOLDER],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO,
    ).stdout
    if dirty.strip():
        raise SystemExit("commit the experiment folder before launching")
    print(orchestrate.remote(git))
