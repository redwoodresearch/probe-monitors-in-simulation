"""k = 1 stage: the simulator's inputs for each site, as the Oct 7 test built them (`2026-10-06_matched_step_pairs/
run_select_pairs.py`, helpers imported unchanged): the real e1 run's September sample (its own epoch-1 log) gives the
prefix (every earlier call with the real output the agent saw) and step t's call; the cell's September Astra sample gives
the simulator's inputs (stored metadata and stored system prompt). The site's (step, action) is located with the reader's
counting (`probe_shift.transcripts.tool_errors`: submit calls and empty turns dropped). Checks: the call exists, its real
output is not None, and the run shares no tool-call id with the simulator's reference. $0 (S3 reads from the laptop).

Transcripts stay outside the repo (`--cache`, a scratch folder); the repo gets a keys-only manifest
(results/k1_records_<stamp>.json). Records: <cache>/records.jsonl.

  PYTHONPATH=src:$HOME/simulated-control-evals/src uv run --no-project --python 3.14 --with boto3 --with pyyaml python \\
      experiments/2026-10-11_claimB_k_replies/build_k1_records.py --sites results/k1_sites_<stamp>.json --cache <dir>
"""

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "2026-10-06_matched_step_pairs"))
import run_select_pairs as RSP  # noqa: E402  (actions_of, SIM_KEYS, DESCRIBED)
from probe_shift.transcripts import RUNNER_SUBMIT, file_samples, visible_text  # noqa: E402

COHORTS = {"attack": "la_public10_real_attack_n1000", "sim": "la_public10_sim_astra_attack_n1000"}  # as pairs.yaml


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


def call_at(sample: dict, step: int, action: int) -> str | None:
    """The tool-call id at the reader's (step, action), counting as tool_errors / reader_messages do."""
    st, ac, submit_ids, pending = -1, 0, set(), []
    for m in sample["messages"]:
        if m["role"] == "assistant":
            calls = [tc for tc in m.get("tool_calls") or [] if tc["function"] != RUNNER_SUBMIT]
            submit_ids |= {tc.get("id") for tc in m.get("tool_calls") or [] if tc["function"] == RUNNER_SUBMIT}
            if calls or visible_text(m["content"]).strip():
                st, ac = st + 1, 0
        elif m["role"] == "tool":
            if m.get("tool_call_id") in submit_ids or m.get("function") == RUNNER_SUBMIT:
                continue
            if (st, ac) == (step, action):
                return m.get("tool_call_id")
            ac += 1
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sites", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    a = p.parse_args()
    if ROOT in a.cache.resolve().parents:
        raise SystemExit("the cache must lie outside the repo")
    sites_raw = (HERE / a.sites).read_bytes()
    sites = json.loads(sites_raw)["sites"]
    shared = json.loads((ROOT / "splits" / "exclusions_v3.json").read_text())
    e1_rows = {r["key"]: r for r in shared["samples"] if r["arm"] == "real-attack-e1"}
    sim_rows = {r["cell"]: r for r in shared["samples"] if r["arm"] == "sim-attack" and not r["reason"]}
    need_e1 = sorted({e1_rows[s["key_e1"]]["file"] for s in sites})
    need_sim = sorted({sim_rows[s["cell"]]["file"] for s in sites})
    with ThreadPoolExecutor(16) as pool:
        e1 = {(f, x["id"], x["epoch"]): x for f, xs in zip(need_e1, pool.map(lambda f: cached(a.cache, COHORTS["attack"], f), need_e1)) for x in xs}
        sim = {(f, x["id"], x["epoch"]): x for f, xs in zip(need_sim, pool.map(lambda f: cached(a.cache, COHORTS["sim"], f), need_sim)) for x in xs}
    descriptions = {}
    for x in sim.values():
        for m in x["store"]["SimulatorState:sim_messages"]:
            for name, desc in RSP.DESCRIBED.findall(m["content"] if isinstance(m["content"], str) else ""):
                if descriptions.setdefault(name, desc) != desc:
                    raise SystemExit(f"two descriptions for {name}")
    out_path = a.cache / "records.jsonl"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; never overwritten")
    tmp_path = out_path.with_suffix(".tmp")  # renamed only after every record passed its checks
    tmp_path.unlink(missing_ok=True)
    manifest, problems = [], []
    with open(tmp_path, "x") as fh:
        for s in sites:
            r = e1_rows[s["key_e1"]]
            sample = e1[(r["file"], r["sample_id"], r["epoch"])]
            cid = call_at(sample, s["step"], s["action"])
            acts = RSP.actions_of(sample)
            ids = [x["id"] for x in acts]
            if cid is None or cid not in ids:
                problems.append({"site": s["site"], "problem": "call not found"})
                continue
            k = ids.index(cid)
            if acts[k]["output"] is None:
                problems.append({"site": s["site"], "problem": "no real output"})
                continue
            sr = sim_rows[s["cell"]]
            ref = sim[(sr["file"], sr["sample_id"], sr["epoch"])]
            md = ref["metadata"]
            traj = md["orig_trajectory"]
            shared_ids = len(set(ids) & {x.get("id") for x in traj})
            if shared_ids:
                problems.append({"site": s["site"], "problem": f"shares {shared_ids} call ids with its reference"})
                continue
            prefix = [{key: x[key] for key in ("function", "arguments", "output", "id")} for x in acts[:k]]
            for x in prefix:
                if x["function"] not in ("bash", "text_editor"):
                    x["description"] = descriptions[x["function"]]  # KeyError = no September rendering to copy
            step = {key: acts[k][key] for key in ("function", "arguments", "id")}
            stored = ref["store"]["SimulatorState:sim_messages"][0]
            if stored["role"] != "system":
                raise SystemExit(f"site {s['site']}: the stored simulator prompt is not a system message")
            rec = {"site": s["site"], "cell": s["cell"], "environment": s["environment"], "main_task": s["main_task"],
                   "key": s["key_e1"], "step_action": [s["step"], s["action"]], "k": k, "n_actions": len(acts),
                   "prefix": prefix, "step": step, "real_output": acts[k]["output"],
                   "sim_metadata": {key: md.get(key) for key in RSP.SIM_KEYS}, "stored_system_prompt": stored["content"]}
            line = json.dumps(rec)
            if "attachment://" in line:  # Inspect placeholders must never reach the simulator (attachment check, 16:5x PT)
                raise SystemExit(f"site {s['site']}: an unresolved attachment:// string in the built record")
            fh.write(line + "\n")
            canonical = json.dumps(rec, sort_keys=True)  # the hash the simulator runner caches by
            sa = json.dumps(step["arguments"], sort_keys=True)
            in_ref = [x for x in traj if x["function"] == step["function"] and json.dumps(x.get("arguments"), sort_keys=True) == sa]
            manifest.append({"site": s["site"], "cell": s["cell"], "environment": s["environment"], "k": k, "n_actions": len(acts),
                             "tool": step["function"] + (f":{step['arguments'].get('command')}" if step["function"] == "text_editor" else ""),
                             "shared_call_ids": shared_ids, "step_command_in_reference": len(in_ref),
                             "reference_output_equals_real": any(x.get("output") == acts[k]["output"] for x in in_ref),
                             "record_sha256": hashlib.sha256(canonical.encode()).hexdigest()})
    tmp_path.replace(out_path)
    res = {"label": "k = 1 stage: simulator inputs per site (keys only; the records stay outside the repo)", "sites_file": a.sites.name,
           "sites_sha256": hashlib.sha256(sites_raw).hexdigest(), "records": len(manifest), "problems": problems,
           "records_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(), "manifest": manifest,
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"k1_records_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(res, indent=1))
    from collections import Counter
    print({"records": len(manifest), "problems": len(problems)}, Counter(m["tool"] for m in manifest).most_common(8), "->", dst)


if __name__ == "__main__":
    main()
