"""Step 2 pilot: the September simulator's inputs for EVERY tool call of 30 real honest runs (simulator content alone).

For each drawn run and each non-submit call k, the record holds the real prefix (every earlier call with the REAL output
the agent saw), call k without its output, and the real output (kept for comparison only). The simulator inputs are the
main task's overlap800 reference cell, exactly as Oct 7's honest pairs used them (`2026-10-06_matched_step_pairs/
run_select_pairs.py`, helpers imported unchanged): the stored September Astra sample's metadata and system prompt.

Draw: real honest runs from splits/exclusions_v3.json that #36 scored, whose main task has an overlap800 reference
(references_20261006-142754.json), outside media_processing and model_registry (not reproduced, as Oct 7); shuffled by
key with seed 20261013, at most one run per main task, taken in order until 30 runs pass every check. A run is skipped
(and listed) if a call has no real output or a tool has no September description to copy.

Checks per run: 0 shared tool-call ids with its reference (the reference is never the compared run); the reference is a
fresh overlap800 honest run (fork.run sce-paired-honest-*); no `attachment://` string in any record.
$0 (S3 reads from the laptop). Records stay outside the repo (--cache): <cache>/runs.jsonl (per-run simulator inputs)
and <cache>/calls.jsonl (per-call prefix and step). The repo gets a keys-only manifest.

  PYTHONPATH=src uv run --no-project --python 3.14 --with boto3 --with pyyaml python \\
      experiments/2026-10-13_step2_sim_rewrite/build_rewrite_records.py --cache <dir>
"""

import argparse
import gzip
import hashlib
import json
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "2026-10-06_matched_step_pairs"))
import run_select_pairs as RSP  # noqa: E402  (actions_of, SIM_KEYS, DESCRIBED)
from probe_shift.transcripts import file_samples  # noqa: E402

SEED, N_RUNS = 20261013, 30
HONEST, SIM = "la_public10_real_honest_n1000", "la_public10_sim_astra_attack_n1000"
REFS = ROOT / "experiments/2026-10-06_astra_sim_honest/results/references_20261006-142754.json"
RUNS36 = ROOT / "experiments/2026-10-06_q3_attack_detector/results/run_scores_2026-10-06_181548/runs.json"
EXCLUDED_ENVS = {"media_processing", "model_registry"}


def cached(cache: Path, alias: str, file: str) -> list[dict]:
    path = cache / "samples" / alias / f"{file}.json.gz"
    if path.exists():
        return json.loads(gzip.decompress(path.read_bytes()))
    for attempt in range(5):
        try:
            samples = file_samples(alias, file)
            break
        except Exception as e:  # noqa: BLE001  (network errors; re-raised after the last attempt)
            if attempt == 4:
                raise
            print(f"retrying {file} after {type(e).__name__}", flush=True)
            time.sleep(2 + 4 * attempt)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(json.dumps(samples).encode()))
    tmp.replace(path)
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    a = p.parse_args()
    if ROOT in a.cache.resolve().parents:
        raise SystemExit("the cache must lie outside the repo")
    for name in ("runs.jsonl", "calls.jsonl"):
        if (a.cache / name).exists():
            raise SystemExit(f"{a.cache / name} exists; never overwritten")
    refs = json.loads(REFS.read_text())["references"]
    scored = {r["key"] for r in json.loads(RUNS36.read_text()) if r["arm"] == "real-honest"}
    rows = [r for r in json.loads((ROOT / "splits/exclusions_v3.json").read_text())["samples"]
            if r["arm"] == "real-honest" and not r["reason"] and r["key"] in scored and r["main_task"] in refs
            and r["environment"] not in EXCLUDED_ENVS]
    rows.sort(key=lambda r: r["key"])
    random.Random(SEED).shuffle(rows)
    print(f"{len(rows)} eligible runs", flush=True)

    runs, calls, manifest, skipped, used_mains = [], [], [], [], set()
    for r in rows:
        if len(runs) == N_RUNS:
            break
        if r["main_task"] in used_mains:
            continue
        s = next(x for x in cached(a.cache, HONEST, r["file"]) if x["id"] == r["sample_id"] and x["epoch"] == r["epoch"])
        ref_info = refs[r["main_task"]]
        ref_file = ref_info["sim_uri"].rsplit("/", 1)[-1]
        ref = next(x for x in cached(a.cache, SIM, ref_file) if x["id"] == ref_info["sim_sample_id"] and x["epoch"] == ref_info["sim_epoch"])
        md = ref["metadata"]
        traj = md["orig_trajectory"]
        assert md["fork"]["run"].startswith("sce-paired-honest-"), md["fork"]
        descriptions = {}
        for m in ref["store"]["SimulatorState:sim_messages"]:
            for name, desc in RSP.DESCRIBED.findall(m["content"] if isinstance(m["content"], str) else ""):
                descriptions.setdefault(name, desc)
        acts = RSP.actions_of(s)
        shared_ids = len({x["id"] for x in acts} & {x.get("id") for x in traj})
        problem = ("shares call ids with its reference" if shared_ids else "no actions" if not acts else
                   "a call has no real output" if any(x["output"] is None for x in acts) else
                   "a tool has no September description" if any(x["function"] not in ("bash", "text_editor") and x["function"] not in descriptions for x in acts)
                   else None)
        if problem:
            skipped.append({"key": r["key"], "problem": problem})
            continue
        stored = ref["store"]["SimulatorState:sim_messages"][0]
        assert stored["role"] == "system"
        run_i = len(runs)
        run = {"run": run_i, "key": r["key"], "file": r["file"], "sample_id": r["sample_id"], "epoch": r["epoch"],
               "environment": r["environment"], "main_task": r["main_task"], "n_actions": len(acts),
               "reference": {"file": ref_file, "sample_id": ref_info["sim_sample_id"], "epoch": ref_info["sim_epoch"],
                             "cell": ref_info["cell"], "fork_run": md["fork"]["run"], "n_actions": len(traj), "shared_call_ids": shared_ids},
               "sim_metadata": {key: md.get(key) for key in RSP.SIM_KEYS}, "stored_system_prompt": stored["content"]}
        run_calls = []
        for k in range(len(acts)):
            prefix = [{key: x[key] for key in ("function", "arguments", "output", "id")} for x in acts[:k]]
            for x in prefix:
                if x["function"] not in ("bash", "text_editor"):
                    x["description"] = descriptions[x["function"]]
            step = {key: acts[k][key] for key in ("function", "arguments", "id")}
            if step["function"] not in ("bash", "text_editor"):
                step["description"] = descriptions[step["function"]]
            run_calls.append({"run": run_i, "k": k, "prefix": prefix, "step": step, "real_output": acts[k]["output"],
                              "real_error": acts[k]["error"]})
        lines = [json.dumps(run)] + [json.dumps(c) for c in run_calls]
        if any("attachment://" in line for line in lines):
            raise SystemExit(f"{r['key']}: an unresolved attachment:// string in a built record")
        runs.append(run)
        calls += run_calls
        used_mains.add(r["main_task"])
        manifest.append({k: run[k] for k in ("run", "key", "environment", "main_task", "n_actions")} | {"reference": run["reference"]})
        print(f"run {run_i}: {r['environment']} {len(acts)} calls", flush=True)
    if len(runs) < N_RUNS:
        raise SystemExit(f"only {len(runs)} runs passed the checks")

    a.cache.mkdir(parents=True, exist_ok=True)
    for name, items in (("runs.jsonl", runs), ("calls.jsonl", calls)):
        with open(a.cache / name, "x") as fh:
            fh.writelines(json.dumps(x) + "\n" for x in items)
    res = {"label": "Step 2 pilot: simulator inputs for every call of 30 real honest runs (keys only; records stay outside the repo)",
           "seed": SEED, "n_runs": len(runs), "n_calls": len(calls), "excluded_environments": sorted(EXCLUDED_ENVS),
           "eligible_runs": len(rows), "skipped": skipped, "runs": manifest,
           "runs_sha256": hashlib.sha256((a.cache / "runs.jsonl").read_bytes()).hexdigest(),
           "calls_sha256": hashlib.sha256((a.cache / "calls.jsonl").read_bytes()).hexdigest(),
           "references_file": REFS.name, "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    dst = HERE / "results" / f"records_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(res, indent=1))
    print({"runs": len(runs), "calls": len(calls), "skipped": len(skipped)}, "->", dst)


if __name__ == "__main__":
    main()
