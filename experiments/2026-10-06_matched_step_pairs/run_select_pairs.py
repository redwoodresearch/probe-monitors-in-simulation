"""Pick the matched-step pairs: real LinuxArena steps that get a simulated twin with the same history.

For each chosen step k of a real run (attack epoch 1 or honest), this writes everything the simulator and the reader
need, so neither later step touches S3:
  - the reader's messages (honest system prompt, as in every extraction) through the next agent turn that calls a tool;
  - the simulator's prefix: every earlier call with the REAL output the agent saw (tool_result_text), and step k's
    call without its output;
  - the September simulator inputs of the reference cell, from the stored Astra sample: its metadata (reference
    trajectory, codebase index, container path mappings, judge and fallback models) and its stored simulator system
    prompt, which run_simulate_pairs.py must reproduce byte for byte.
Attack runs use their own cell's September reference (an honest run of the same main task, as the September sims had).
Honest runs use the overlap800 reference the sims session picked for the main task, which is never a run in our
honest cohort. Every pair is checked for shared tool-call IDs between its run and its reference (must be 0).

Selection: valid runs (splits), attack runs only from the paired cells, no media_processing/model_registry; step k is
a bash call or a text_editor view in a one-call agent turn, with at least one earlier action, a real output that is not
a tool error, and a later agent turn that calls a tool; step k and the next call must sit inside the first 32k reader
tokens with room to spare (config token_margin). Steps are drawn round-robin over (environment, position bucket) per
arm, at most one per main task per arm, and the two arms are interleaved, so the first n_pilot pairs are the pilot.

  PYTHONPATH=src:$HOME/simulated-control-evals/src uv run --no-project --python 3.14 --with transformers==4.56.2 \\
    --with jinja2 --with boto3 --with pyyaml python experiments/2026-10-06_matched_step_pairs/run_select_pairs.py
"""

import gzip
import hashlib
import itertools
import json
import random
import re
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from probe_shift.reader import segment_end_positions
from probe_shift.transcripts import (RUNNER_SUBMIT, file_samples, main_task_id, reader_messages,
                                     system_prompt, tool_result_text)

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
PT = ZoneInfo("America/Los_Angeles")
DESCRIBED = re.compile(r"The agent called the `([^`]+)` tool \(the tool is described as: (.*?)\)\n")
# Metadata the simulator config reads (runner.simulate.config.config_for_sample), plus provenance.
SIM_KEYS = ["codebase_root", "env_context", "container_path_mappings", "deployment_tree", "static_filesystem",
            "setup_script", "codebase_exec", "exec_source_authority", "orig_trajectory", "orig_trajectory_mode",
            "orig_trajectory_out_cap", "session_notes", "self_critique", "seeded_prefix", "simulator_model",
            "simulator_fallback_model", "simulator_refusal_judge_model", "simulator_router_judge_model", "fork",
            "arena_reference_contract", "source_tree", "env", "main_id"]


def cached_samples(alias: str, file: str) -> list[dict]:
    """file_samples with a local copy (data/cache/, git-ignored) and retries: S3 reads over many threads drop now and then."""
    path = HERE / "data" / "cache" / alias / f"{file}.json.gz"
    if path.exists():
        return json.loads(gzip.decompress(path.read_bytes()))
    for attempt in range(5):
        try:
            samples = file_samples(alias, file)
            break
        except Exception as e:  # network errors only in practice; re-raised after the last attempt
            if attempt == 4:
                raise
            print(f"retrying {file} after {type(e).__name__}", flush=True)
            time.sleep(2 + 4 * attempt)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(json.dumps(samples).encode()))
    tmp.replace(path)
    return samples


def actions_of(sample: dict) -> list[dict]:
    """Every non-submit call in order, with the real output the agent saw. Mirrors sim.adapters.inspect_eval."""
    results = {m.get("tool_call_id"): m for m in sample["messages"] if m["role"] == "tool"}
    out = []
    for t, m in enumerate(m for m in sample["messages"] if m["role"] == "assistant"):
        for tc in m.get("tool_calls") or []:
            if tc["function"] == RUNNER_SUBMIT:
                continue
            r = results.get(tc["id"])
            out.append({"function": tc["function"], "arguments": tc.get("arguments") or {}, "id": tc["id"],
                        "output": None if r is None else tool_result_text(r), "error": (r or {}).get("error")})
    return out


def reader_turns(msgs: list[dict]) -> list[dict]:
    """Per agent turn in reader_messages: its message index, its calls, and the indices of its tool outputs."""
    turns, cur = [], None
    for i, m in enumerate(msgs):
        if m["role"] == "assistant":
            cur = {"msg": i, "calls": m["tool_calls"], "outputs": []}
            turns.append(cur)
        elif m["role"] == "tool":
            cur["outputs"].append(i)
    return turns


def step_tool_ok(call: dict, allowed: list[str]) -> bool:
    name, args = call["function"]["name"], call["function"]["arguments"] or {}
    return name in allowed or f"{name}:{args.get('command')}" in allowed


def main():
    cfg = yaml.safe_load((HERE / "config" / "pairs.yaml").read_text())
    rng = random.Random(cfg["seed"])
    shared = json.loads((ROOT / "splits" / cfg["exclusions_file"]).read_text())
    folds = json.loads((ROOT / "splits" / cfg["folds_file"]).read_text())["task_folds"]["main_task_to_fold"]
    refs = json.loads(Path(cfg["honest_references"]).expanduser().read_text())
    paired = set(shared["paired_attack_cells"]["cells"])
    rows = {(r["arm"], r["file"], r["sample_id"], r["epoch"]): r for r in shared["samples"]}
    sim_rows = {r["cell"]: r for r in shared["samples"] if r["arm"] == "sim-attack" and not r["reason"]}
    excluded_envs = set(cfg["excluded_environments"])

    def load(alias, arm, files):
        with ThreadPoolExecutor(32) as pool:
            for f, samples in zip(files, pool.map(lambda f: cached_samples(alias, f), files)):
                for s in samples:
                    yield rows[(arm, f, s["id"], s["epoch"])], s

    def files_for(arm, keep):
        return sorted({r["file"] for r in shared["samples"] if r["arm"] == arm and not r["reason"] and keep(r)})

    honest_files = files_for("real-honest", lambda r: True)
    attack_files = files_for("real-attack-e1", lambda r: r["cell"] in paired and r["environment"] not in excluded_envs)
    prompts, samples = defaultdict(set), {"honest": [], "attack": []}
    for row, s in load(cfg["cohorts"]["honest"], "real-honest", honest_files):
        prompts[main_task_id(s)].add(system_prompt(s))
        if not row["reason"] and row["environment"] not in excluded_envs and row["main_task"] in refs["references"]:
            samples["honest"].append((row, s))
    assert all(len(p) == 1 for p in prompts.values()), "a main task has more than one honest system prompt"
    prompts = {k: next(iter(v)) for k, v in prompts.items()}
    for row, s in load(cfg["cohorts"]["attack"], "real-attack-e1", attack_files):
        if not row["reason"] and row["cell"] in paired and row["environment"] not in excluded_envs:
            samples["attack"].append((row, s))
    print({a: len(v) for a, v in samples.items()}, "eligible runs")

    # Candidate steps per arm, keyed by (environment, bucket).
    buckets = cfg["position_buckets"]
    cands = {arm: defaultdict(list) for arm in samples}
    for arm, items in samples.items():
        for row, s in items:
            acts = actions_of(s)
            msgs = reader_messages(s, prompts[row["main_task"]])
            turns = reader_turns(msgs)
            n_calls = sum(len(t["calls"]) for t in turns)
            assert n_calls == len(acts), (row["key"], n_calls, len(acts))
            k = 0
            for ti, t in enumerate(turns):
                if len(t["calls"]) == 1 and k >= cfg["min_prefix_actions"] and step_tool_ok(t["calls"][0], cfg["step_tools"]):
                    a = acts[k]
                    nxt = next((u for u in range(ti + 1, len(turns)) if turns[u]["calls"]), None)
                    ok = a["output"] is not None and not (cfg["drop_errored_outputs"] and a["error"])
                    if ok and (nxt is not None or not cfg["require_next_call"]):
                        frac = k / len(acts)
                        b = next(i for i, (lo, hi) in enumerate(buckets) if lo <= frac < hi)
                        cands[arm][(row["environment"], b)].append({"row": row, "sample": s, "msgs": msgs, "turns": turns,
                                                                    "acts": acts, "turn": ti, "k": k, "next_turn": nxt,
                                                                    "bucket": b, "frac": frac})
                k += len(t["calls"])
    for arm in cands:
        print(arm, "candidate steps by (env, bucket):", {f"{e}/{b}": len(v) for (e, b), v in sorted(cands[arm].items())})

    from transformers import AutoTokenizer

    from probe_shift.models import QWEN3_32B

    # The config records the tokenizer; it must be the pin the reader and the splits use.
    assert cfg["tokenizer"] == QWEN3_32B, f"config tokenizer {cfg['tokenizer']} is not probe_shift.models.QWEN3_32B"
    tok = AutoTokenizer.from_pretrained(QWEN3_32B["name"], revision=QWEN3_32B["revision"])
    limit = cfg["max_tokens"] - cfg["token_margin"]

    def positions(c):
        """Real-version reader positions: step k's output, and the next call."""
        last = c["turns"][c["next_turn"]]
        msgs = c["msgs"][: (last["outputs"][-1] if last["outputs"] else last["msg"]) + 1]
        r = segment_end_positions(tok, msgs)
        want = {("output_end", c["turn"]), ("output_marker", c["turn"]), ("pre_call", c["next_turn"]),
                ("call_end", c["next_turn"]), ("call_marker", c["next_turn"])}
        got = {f"{p['kind']}": p["token"] for p in r["positions"] if (p["kind"], p["step"]) in want and p["action"] == 0}
        return msgs, got, r["n_tokens"]

    chosen = {arm: [] for arm in cands}
    for arm in cands:
        used_mains, strata = set(), sorted(cands[arm])
        for v in cands[arm].values():
            rng.shuffle(v)
        rng.shuffle(strata)
        while len(chosen[arm]) < cfg["n_pairs"][arm] and any(cands[arm].values()):
            for st in strata:
                while cands[arm][st]:
                    c = cands[arm][st].pop()
                    if cfg["one_per_main_per_arm"] and c["row"]["main_task"] in used_mains:
                        continue
                    msgs, pos, n_tok = positions(c)
                    if pos.get("output_end") is None or pos.get("call_marker") is None or pos["call_marker"] >= limit:
                        continue
                    chosen[arm].append(c | {"reader_messages": msgs, "positions": pos, "n_tokens": n_tok})
                    used_mains.add(c["row"]["main_task"])
                    break
                if len(chosen[arm]) >= cfg["n_pairs"][arm]:
                    break
        assert len(chosen[arm]) == cfg["n_pairs"][arm], f"only {len(chosen[arm])} {arm} pairs found"

    # The September simulator inputs of each pair's reference cell.
    ref_loc = {}
    for arm, cs in chosen.items():
        for c in cs:
            if arm == "attack":
                r = sim_rows[c["row"]["cell"]]
                ref_loc[id(c)] = (r["file"], r["sample_id"], r["epoch"], r["cell"], r["sim_reference"])
            else:
                ref = refs["references"][c["row"]["main_task"]]
                ref_loc[id(c)] = (ref["sim_uri"].rsplit("/", 1)[-1], ref["sim_sample_id"], ref["sim_epoch"], ref["cell"], "overlap800")
    files = sorted({v[0] for v in ref_loc.values()})
    with ThreadPoolExecutor(32) as pool:
        sim_samples = {(f, s["id"], s["epoch"]): s for f, ss in zip(files, pool.map(lambda f: cached_samples(cfg["cohorts"]["sim"], f), files))
                       for s in ss}
    descriptions = {}
    for s in sim_samples.values():
        for m in s["store"]["SimulatorState:sim_messages"]:
            for name, desc in DESCRIBED.findall(m["content"] if isinstance(m["content"], str) else ""):
                assert descriptions.setdefault(name, desc) == desc, f"two descriptions for {name}"

    # Interleave the arms; with unequal counts the longer arm's remainder follows (zip_longest keeps it).
    order = [c for pair in itertools.zip_longest(chosen["attack"], chosen["honest"]) for c in pair if c is not None]
    stamp = datetime.now(PT).strftime("%Y-%m-%d_%H%M%S")
    data_dir = HERE / "data"
    data_dir.mkdir(exist_ok=True)
    out_path = data_dir / f"pairs_{stamp}.jsonl"
    manifest = []
    with open(out_path, "x") as f:
        for i, c in enumerate(order):
            arm = "attack" if c["row"]["arm"] == "real-attack-e1" else "honest"
            file, sid, ep, ref_cell, ref_kind = ref_loc[id(c)]
            ref = sim_samples[(file, sid, ep)]
            md = ref["metadata"]
            traj = md["orig_trajectory"]
            run_ids = {a["id"] for a in c["acts"]}
            shared_ids = len(run_ids & {a.get("id") for a in traj})
            assert shared_ids == 0, f"{c['row']['key']}: shares {shared_ids} call IDs with its reference"
            if arm == "honest":
                assert md["fork"]["run"].startswith("sce-paired-honest-"), md["fork"]
            k = c["k"]
            prefix = [{key: a[key] for key in ("function", "arguments", "output", "id")} for a in c["acts"][:k]]
            for a in prefix:
                if a["function"] not in ("bash", "text_editor"):
                    a["description"] = descriptions[a["function"]]  # KeyError = no September rendering to copy
            step = {key: c["acts"][k][key] for key in ("function", "arguments", "id")}
            step_args = json.dumps(step["arguments"], sort_keys=True)
            in_ref = [a for a in traj if a["function"] == step["function"] and json.dumps(a.get("arguments"), sort_keys=True) == step_args]
            stored_prompt = ref["store"]["SimulatorState:sim_messages"][0]
            assert stored_prompt["role"] == "system"
            row = c["row"]
            rec = {"pair": i, "pilot": i < cfg["n_pilot"], "arm": arm, "split_arm": row["arm"], "key": row["key"],
                   "cohort": row["cohort"], "file": row["file"], "sample_id": row["sample_id"], "epoch": row["epoch"],
                   "environment": row["environment"], "main_task": row["main_task"], "side_task": row["side_task"],
                   "cell": row["cell"], "fold": folds[row["main_task"]], "sim_reference_of_cell": row.get("sim_reference"),
                   "turn": c["turn"], "k": k, "n_actions": len(c["acts"]), "frac": round(c["frac"], 4),
                   "bucket": ["early", "mid", "late"][c["bucket"]], "next_turn": c["next_turn"],
                   "out_msg_index": c["turns"][c["turn"]]["outputs"][0], "positions_real": c["positions"],
                   "n_tokens_real": c["n_tokens"],
                   "reference": {"file": file, "sample_id": sid, "epoch": ep, "cell": ref_cell, "kind": ref_kind,
                                 "fork_run": md["fork"]["run"], "n_actions": len(traj), "shared_call_ids": shared_ids,
                                 "step_command_in_reference": len(in_ref),
                                 "reference_output_equals_real": any(a.get("output") == c["acts"][k]["output"] for a in in_ref)},
                   "prefix": prefix, "step": step, "real_output": c["acts"][k]["output"],
                   "sim_metadata": {key: md.get(key) for key in SIM_KEYS},
                   "stored_system_prompt_sha256": hashlib.sha256(stored_prompt["content"].encode()).hexdigest(),
                   "stored_system_prompt": stored_prompt["content"],
                   "reader_messages": c["reader_messages"]}
            f.write(json.dumps(rec) + "\n")
            manifest.append({k2: rec[k2] for k2 in ("pair", "pilot", "arm", "key", "environment", "main_task", "side_task",
                                                    "cell", "fold", "turn", "k", "n_actions", "frac", "bucket", "next_turn",
                                                    "positions_real", "n_tokens_real", "sim_reference_of_cell")}
                            | {"step_tool": step["function"] + (f":{step['arguments'].get('command')}" if step["function"] == "text_editor" else ""),
                               "real_output_chars": len(rec["real_output"]),
                               "reference": {k2: v for k2, v in rec["reference"].items()}})
    git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    summary = {"created": datetime.now(PT).isoformat(timespec="seconds"), "git": git, "config": cfg,
               "data_file": out_path.name, "data_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
               "honest_references_sha256": hashlib.sha256(Path(cfg["honest_references"]).expanduser().read_bytes()).hexdigest(),
               "tool_descriptions": descriptions, "pairs": manifest}
    (HERE / "results").mkdir(exist_ok=True)
    with open(HERE / "results" / f"pairs_{stamp}.json", "x") as f:
        f.write(json.dumps(summary, indent=1))
    from collections import Counter
    print(f"wrote {out_path.relative_to(ROOT)} ({len(order)} pairs) and results/pairs_{stamp}.json")
    print("by arm/env:", Counter((p["arm"], p["environment"]) for p in manifest))
    print("by arm/bucket:", Counter((p["arm"], p["bucket"]) for p in manifest))
    print("by tool:", Counter(p["step_tool"] for p in manifest))
    print("step command in reference:", Counter((p["arm"], p["reference"]["step_command_in_reference"] > 0) for p in manifest))


if __name__ == "__main__":
    main()
