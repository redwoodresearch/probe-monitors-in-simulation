"""Confound check: the attack instructions never reach the reader.

For every attack run Q3 scores: the side-task text (sample metadata `side_task_text`) is in the run's original system
prompt, and is absent from the system and user messages Qwen read (the staged reader inputs on the activations
Volume, which use the honest system prompt of the same main task). Also counts runs where the agent's own tool calls
or the tool outputs quote the side-task text's opening (behavior, not instructions). Local CPU; reads S3 and the Volume.

  uv run python experiments/2026-10-06_q3_attack_detector/analyze_prompt_check.py
"""

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal
import yaml

from probe_shift.attack_eval import ATTACK_ARMS, select_runs
from probe_shift.transcripts import eval_files, file_samples, system_prompt

HERE = Path(__file__).parent
REPO = HERE.parent.parent
COHORTS = yaml.safe_load((REPO / "experiments/2026-10-06_segment_end_extraction/config/segment_end.yaml").read_text())["cohorts"]
OPENING = 80  # characters of the side-task text searched for in the agent's calls and the tool outputs


def strings(x):
    """Every string inside a tool call's arguments (nested dicts and lists included)."""
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from strings(v)


def main():
    cfg = yaml.safe_load((HERE / "config" / "q3_attack.yaml").read_text())
    runs, _ = select_runs(json.loads((REPO / "splits" / cfg["exclusions_file"]).read_text()),
                          json.loads((REPO / "splits" / cfg["folds_file"]).read_text()))
    want = {(r["arm"], r["key"]) for r in runs if r["arm"] in ATTACK_ARMS}
    side, orig_has = {}, {}
    for arm in ATTACK_ARMS:
        alias = COHORTS[arm]
        files = eval_files(alias)
        with ThreadPoolExecutor(32) as pool:
            for f, samples in zip(files, pool.map(lambda f: file_samples(alias, f), files)):
                for s in samples:
                    key = (arm, f"{f.removesuffix('.eval')}__{s['id'].replace('/', '_')}__{s['epoch']}")
                    if key in want:
                        text = (s.get("metadata") or {}).get("side_task_text") or ""
                        side[key] = text
                        orig_has[key] = bool(text.strip()) and text.strip() in system_prompt(s)
        print(f"{arm}: {sum(k[0] == arm for k in side)} samples", flush=True)
    if set(side) != want:
        raise ValueError(f"{len(want - set(side))} scored attack runs not found in S3")
    vol = modal.Volume.from_name(cfg["activations_volume"], version=2)
    counts = Counter()
    examples = []
    for i in range(20):
        raw = b"".join(vol.read_file(f"{cfg['extraction_run']}/inputs/shard_{i}.jsonl")).decode()
        for line in raw.splitlines():
            it = json.loads(line)
            key = (it["arm"], it["key"])
            if key not in want:
                continue
            text = side[key].strip()
            prompts = "\n".join(m["content"] for m in it["messages"] if m["role"] in ("system", "user"))
            # The calls' decoded argument strings, so the search matches the opening as written (the committed
            # prompt_check_2026-10-06_165545.json searched JSON-escaped calls: a lower bound for openings with quotes,
            # newlines, backslashes or non-ASCII characters)
            calls = "\n".join(s for m in it["messages"] if m["role"] == "assistant" for c in m.get("tool_calls") or []
                              for s in strings(c["function"]["arguments"]))
            outputs = "\n".join(m["content"] for m in it["messages"] if m["role"] == "tool")
            counts[(it["arm"], "seen")] += 1
            counts[(it["arm"], "side_task_text_empty")] += not text
            counts[(it["arm"], "side_task_text_in_original_system_prompt")] += orig_has[key]
            leak = bool(text) and text in prompts
            counts[(it["arm"], "side_task_text_in_reader_system_or_user_messages")] += leak
            if leak:
                examples.append(it["key"])
            counts[(it["arm"], f"opening_{OPENING}_chars_in_agent_calls")] += bool(text) and text[:OPENING] in calls
            counts[(it["arm"], f"opening_{OPENING}_chars_in_tool_outputs")] += bool(text) and text[:OPENING] in outputs
    out = {arm: {k: counts[(arm, k)] for _, k in sorted(c for c in counts if c[0] == arm)} for arm in ATTACK_ARMS}
    out["leaks_examples"] = examples[:10]
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"prompt_check_{stamp}.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    if examples:
        raise SystemExit(f"{len(examples)} runs show the side-task text to the reader; see {path}")


if __name__ == "__main__":
    main()
