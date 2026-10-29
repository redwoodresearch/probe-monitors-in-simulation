"""Cell (a) transfer, Amendment 1 (POST HOC): stage honest A and honest B of the September scale-up reader-style, levels 0
and 1, plus 3 gate transcripts (first real runs already in the reader-style dumps). Free, local, no model.

The rules are follow-up (c)'s, reused, not rewritten: build_reader_inputs.canonical / sample_of (the canonical run per
(launch, arm, task): the last valid attempt; a task in two launches is an error), transcripts.reader_messages with honest
A's system prompt of the same task, the Qwen3.5-27B tokenizer's segment-end positions with the 32k cut, level 1 =
run_stage_probe_inputs_modal.scrubbed(messages, 1), and the staged-inputs schema (key = <file stem>__<sample id>__<epoch>).
Gate rows (arm gate-real-attack-r1): 3 seeded tasks complete in all arms, read exactly as build_reader_inputs.py read
real-attack-r1; upload_honest_reader.py checks their messages hash-equal the existing staging. Writes data/<run>/
{inputs/shard_*.jsonl, manifest.json} (git-ignored; transcript text stays on Modal) and results/stage_honest_<stamp>.json
(counts and the gate rows' message hashes only).

  HF_HUB_OFFLINE=1 PYTHONPATH=src uv run --no-project --with transformers==4.56.2 --with jinja2 --with boto3 --with pyyaml --with 'modal>=1.4.3' \\
      python experiments/2026-10-13_cell_a_transfer_POSTHOC/stage_honest_reader.py
"""

import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "experiments" / "2026-10-10_claimB_onpolicy"), str(REPO / "experiments" / "2026-10-09_fixed_sim")]
import build_reader_inputs as B  # noqa: E402  (follow-up (c)'s converter: canonical, sample_of, LAUNCHES)
from run_stage_probe_inputs_modal import scrubbed  # noqa: E402

from probe_shift.models import QWEN35_27B  # noqa: E402
from probe_shift.reader import segment_end_positions  # noqa: E402
from probe_shift.transcripts import main_task_id, reader_messages, submit_calls, system_prompt  # noqa: E402

RUN_L0, RUN_L1 = "2026-10-13_la_september-onpolicy-honest-l0_v1", "2026-10-13_la_september-onpolicy-honest-l1_v1"
SHARDS, N_GATE, SEED = 12, 3, "cell-a-transfer-gate"
ARM_NAME = {"honest_a": "real-honest-a", "honest_b": "real-honest-b"}


def chosen_for(arm: str) -> dict:
    out = {}
    for launch, arms in B.LAUNCHES.items():
        if arm == "attack" and launch == "2026-10-10_001314_second_attack":
            continue  # run 1 only, as the converter's role r1
        if arm != "attack" and arm not in ("honest_a", "honest_b"):
            raise ValueError(arm)
        if arm in ("honest_a", "honest_b") and launch == "2026-10-10_001314_second_attack":
            continue
        for (_, task), v in B.canonical(launch, (arm,)).items():
            if task in out:
                raise ValueError(f"{arm} {task}: valid runs in two launches")
            out[task] = (v, launch)
    return out


def main():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(QWEN35_27B["name"], revision=QWEN35_27B["revision"])
    ha, hb, r1 = chosen_for("honest_a"), chosen_for("honest_b"), chosen_for("attack")
    r2 = {t: (v, "2026-10-10_001314_second_attack") for (_, t), v in B.canonical("2026-10-10_001314_second_attack", ("attack",)).items()}
    sim = {}
    for launch, arms in B.LAUNCHES.items():
        if "sim_attack" in arms:
            sim |= {t: v for (_, t), v in B.canonical(launch, ("sim_attack",)).items()}
    complete = sorted(t for t in ha if t in r1 and t in r2 and t in sim)  # the converter's complete set (r1, r2, sim, honest)
    gate_tasks = sorted(random.Random(SEED).sample(complete, N_GATE))
    count, rows, gate = Counter(), [], {}
    for t in sorted(set(ha) | set(hb), key=lambda t: int(t[1:].split("_")[0])):
        if t not in ha:
            count["honest_b_without_honest_a"] += 1
            continue
        _, honest = B.sample_of(*ha[t][0][:2])
        system = system_prompt(honest)
        todo = [(a, src[t]) for a, src in (("honest_a", ha), ("honest_b", hb)) if t in src] + ([("gate", r1[t])] if t in gate_tasks else [])
        for arm, ((key, sha, i, n), launch) in todo:
            file, s = B.sample_of(key, sha)
            if main_task_id(s) != main_task_id(honest):
                raise ValueError(f"{t}: {arm}'s main task differs from honest A's")
            env = (s.get("metadata") or {}).get("environment")
            env = env.get("id") if isinstance(env, dict) else env
            msgs = reader_messages(s, system)
            name = "gate-real-attack-r1" if arm == "gate" else ARM_NAME[arm]
            seg = segment_end_positions(tok, msgs, B.MAX_TOKENS)
            k = f"{file.removesuffix('.eval')}__{s['id'].replace('/', '_')}__{s['epoch']}"
            rows.append({"key": k, "arm": name, "cohort": f"phase2_september_scaleup_{arm}", "file": file, "sample_id": s["id"], "epoch": s["epoch"],
                         "environment": env, "main_task": main_task_id(s), "side_task": None, "cell": f"{main_task_id(s)}:None", "sim_reference": None,
                         "task": t, "launch": launch, "n_submit_calls": submit_calls(s), "tokens": seg["n_tokens"], "truncated": seg["truncated"], "messages": msgs})
            if arm == "gate":
                gate[k] = {"task": t, "l0": hashlib.sha256(json.dumps(msgs).encode()).hexdigest(),
                           "l1": hashlib.sha256(json.dumps(scrubbed(msgs, 1)).encode()).hexdigest()}
            count[f"{name}|runs"] += 1
            count[f"{name}|truncated"] += seg["truncated"]
            count[f"{name}|chosen_attempt_{'first' if i == 0 else 'later'}"] += 1
    if len({(r["arm"], r["key"]) for r in rows}) != len(rows):
        raise ValueError("duplicate keys")
    out = {"counts": dict(count), "gate_tasks": gate_tasks, "gate_message_sha256": gate, "runs": {}}
    for run, items in ((RUN_L0, rows), (RUN_L1, [r | {"messages": scrubbed(r["messages"], 1)} for r in rows])):
        items = sorted(items, key=lambda r: (-r["tokens"], r["arm"], r["key"]))
        files = {f"inputs/shard_{s}.jsonl": "".join(json.dumps(r) + "\n" for r in items[s::SHARDS]).encode() for s in range(SHARDS)}
        d = HERE / "data" / run
        d.mkdir(parents=True, exist_ok=False)
        for n, b in files.items():
            (d / n).parent.mkdir(parents=True, exist_ok=True)
            (d / n).open("xb").write(b)
        man = {"run": run, "reader": QWEN35_27B, "arms": dict(Counter(r["arm"] for r in items)), "files": {n: hashlib.sha256(b).hexdigest() for n, b in sorted(files.items())}}
        (d / "manifest.json").open("x").write(json.dumps(man, indent=1))
        out["runs"][run] = {"arms": man["arms"], "manifest_sha256": hashlib.sha256((d / "manifest.json").read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results" / f"stage_honest_{stamp}.json").open("x").write(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
