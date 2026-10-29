"""Fix-full F9 (PLAN.md Amendment 4), POST HOC: the re-read subset of F9's staged run. Modal CPU; counts and hashes only.

F9 changes 66 view outputs in 36 attack runs, so full + F9 equals fix-full's `full` arm byte for byte in every other run
(0d's cheaper design, 10-13 about 15:00). This copies, from `2026-10-13_q2_fixfull_inputs_f9_v1`, only the rows to re-read:
F9 alone (all its 36 runs) and the full + F9 attack and honest runs whose messages differ from fix-full's `full` rows in
`2026-10-13_q2_fixfull_inputs_v2` (sha256 of each run's messages, both stage records, checked on the rows themselves),
into `2026-10-13_q2_fixfull_inputs_f9r_v1`, one shard per arm, never overwriting. Every other full + F9 run is fix-full's,
asserted identical here (its message sha256 equals the full row's); copy_runs.py later copies those runs' activations.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' modal run experiments/2026-10-13_q2_fixfull/subset_f9.py
"""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-fixfull_stage_kill-2026-11-13"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
SRC, FULL, DST = "2026-10-13_q2_fixfull_inputs_f9_v1", "2026-10-13_q2_fixfull_inputs_v2", "2026-10-13_q2_fixfull_inputs_f9r_v1"
PAIRS = {"sim-attack-fx-fullF9": "sim-attack-fx-full", "sim-honest-fx-fullF9": "sim-honest-fx-full"}

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2)
image = modal.Image.debian_slim(python_version="3.12")


def msha(msgs):
    return hashlib.sha256(json.dumps(msgs, sort_keys=True).encode()).hexdigest()


@app.function(image=image, volumes={"/acts": acts}, cpu=4, memory=32768, timeout=1800, retries=0)
def subset() -> dict:
    def rows(run):
        out = {}
        for p in sorted((Path("/acts") / run / "inputs").glob("shard_*.jsonl")):
            for line in p.read_text().splitlines():
                it = json.loads(line)
                if (it["arm"], it["key"]) in out:
                    raise ValueError("a run twice in a staged run")
                out[(it["arm"], it["key"])] = it
        return out
    src, full = rows(SRC), rows(FULL)
    keep, same = {}, {}
    for (arm, key), it in src.items():
        if arm == "sim-attack-fx-F9":
            keep.setdefault(arm, []).append(it)
            continue
        f = full[(PAIRS[arm], key)]
        if msha(it["messages"]) != msha(f["messages"]):
            keep.setdefault(arm, []).append(it)
        else:
            same.setdefault(arm, []).append(key)
    dst = Path("/acts") / DST
    if dst.exists():
        raise FileExistsError(f"{DST} exists: a staged run is never overwritten")
    (dst / "inputs").mkdir(parents=True)
    shards = {}
    for n, (arm, rs) in enumerate(sorted(keep.items())):
        body = "".join(json.dumps(x) + "\n" for x in sorted(rs, key=lambda x: x["key"])).encode()
        (dst / "inputs" / f"shard_{n}.jsonl").write_bytes(body)
        shards[f"inputs/shard_{n}.jsonl"] = {"arm": arm, "runs": len(rs), "sha256": hashlib.sha256(body).hexdigest()}
    rec = {"runs": [{"arm": a, "key": x["key"], "messages_sha256": msha(x["messages"])} for a, rs in sorted(keep.items()) for x in rs],
           "identical to fix-full's full rows (copied later, not re-read)": {a: sorted(v) for a, v in same.items()}}
    (dst / "stage_record.json").write_text(json.dumps(rec))
    acts.commit()
    return {"staged": DST, "shards": shards, "runs by arm": {a: sorted(x["key"] for x in rs) for a, rs in sorted(keep.items())},
            "identical to fix-full's full rows (copied later, not re-read)": {a: sorted(v) for a, v in same.items()}}


@app.local_entrypoint()
def main():
    for f in (Path(__file__), HERE / "PLAN.md"):
        if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode \
                or subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    out = subset.remote()
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    p = HERE / "results" / f"stage_f9r_{stamp}.json"
    with open(p, "x") as fh:
        fh.write(json.dumps({"POST HOC": True, "git": head, "mode": "stage"} | out, indent=1) + "\n")
    print(p)
    print({a: len(v) for a, v in out["runs by arm"].items()}, {a: len(v) for a, v in out["identical to fix-full's full rows (copied later, not re-read)"].items()})
