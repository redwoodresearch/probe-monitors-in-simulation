"""Select 20 successful real-attack-e1 runs and dump their tool calls for blind per-step sabotage labelling.

Reads local data only: transcripts come from the local S3 cache (~/.cache/probe-shift/s3); no S3, API or Modal
calls. Never opens detector scores (scores_*.npz, metrics_*.json, headline_run_scores_*.jsonl, monitor_replay_*.json).

Modes:
  select  write the selected keys (selected_keys_2026-10-07.json)
  dump    write one readable text dump per selected run into a scratch directory, for a human/agent labeller
  build   turn the hand labels (sabotage_step_labels_manual_2026-10-07.json: per key, {"step.action": [label, note]}
          for sabotage/preparation calls and annotated benign calls; every other call is benign, submit calls are "submit") into
          sabotage_step_labels_2026-10-07.jsonl

Step = 0-based index among ALL assistant messages; action = 0-based index among that message's tool_calls
(submit calls counted). This differs from the reader's indexing (probe_shift.transcripts.reader_messages), which
drops submit calls and assistant turns with neither text nor calls; `build` reports whether the two agree.

Run: uv run --no-project --python 3.14 --with numpy --with boto3 --with pyyaml python label_sabotage_dump.py <mode>
"""

import io
import json
import random
import sys
import zipfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LABELS = REPO / "experiments/2026-10-07_q3_nano_monitor/results/attack_attempt_labels_2026-10-07_154729.jsonl"
SELF_REPORTS = REPO / "experiments/2026-10-07_q3_nano_monitor/results/attack_self_reports_2026-10-07_154719.jsonl"
SPLITS = REPO / "splits/exclusions_v4.json"
RUNS = REPO / "experiments/2026-10-06_q3_attack_detector/results/run_scores_2026-10-06_181548/runs.json"
REAL_RUNS = next(p for p in (REPO.parent / "simulated-control-evals", Path.home() / "simulated-control-evals") if p.exists()) / "src/paper/configs/real_runs.yml"
CACHE = Path.home() / ".cache/probe-shift/s3"
DUMP_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "data" / "k5_dumps"  # transcripts: git-ignored
SELECTED = HERE / "results" / "selected_keys_2026-10-07.json"
MANUAL = HERE / "results" / "sabotage_step_labels_manual_2026-10-07.json"
OUT = HERE / "results" / "sabotage_step_labels_2026-10-07.jsonl"
ARM, N, SEED = "real-attack-e1", 20, 0


def jsonl(path):
    return [json.loads(line) for line in open(path)]


def select():
    scored = {(r["arm"], r["key"]) for r in json.load(open(RUNS))}
    pool = sorted(r["key"] for r in jsonl(LABELS)
                  if r["arm"] == ARM and r["label"] == "attempted" and r["side_task_scorer"] == "C"
                  and (r["arm"], r["key"]) in scored)
    keys = random.Random(SEED).sample(pool, N)
    with SELECTED.open("x") as f:  # never replace the sample the labels belong to
        f.write(json.dumps({"rule": f"arm={ARM}, label=attempted, side_task_scorer=C, in runs.json; "
                                    f"sorted by key; random.Random({SEED}).sample(pool, {N})",
                            "pool_size": len(pool), "keys": keys}, indent=1))
    print(f"pool {len(pool)} -> {len(keys)} keys -> {SELECTED}")
    return keys


def load_cached(alias, file, sample_id, epoch):
    uri = yaml.safe_load(REAL_RUNS.read_text())["runs"][alias]["uri"]
    bucket, prefix = uri.removeprefix("s3://").split("/", 1)
    local = CACHE / bucket / f"{prefix}logs/{file}"
    if not local.exists():
        raise FileNotFoundError(f"not in local cache (refusing to download): {local}")
    with zipfile.ZipFile(io.BytesIO(local.read_bytes())) as z:
        for name in z.namelist():
            if name.startswith("samples/") and name.endswith(".json"):
                s = json.loads(z.read(name))
                if s["id"] == sample_id and s["epoch"] == epoch:
                    return s
    raise KeyError(f"{sample_id} epoch {epoch} not in {file}")


def text_of(content):
    if isinstance(content, str):
        return content
    return "\n".join(c.get("text") or "" for c in content or [] if c.get("type") == "text")


def samples(keys):
    meta = {(s["arm"], s["key"]): s for s in json.load(open(SPLITS))["samples"]}
    for key in keys:
        m = meta[(ARM, key)]
        yield key, m, load_cached(m["cohort"], m["file"], m["sample_id"], m["epoch"])


def calls(sample):
    """(step, action, function, arguments, output) for every tool call, steps indexed over all assistant messages."""
    outputs = {m.get("tool_call_id"): text_of(m.get("content")) or ((m.get("error") or {}).get("message") or "")
               for m in sample["messages"] if m["role"] == "tool"}
    assistants = [m for m in sample["messages"] if m["role"] == "assistant"]
    for step, m in enumerate(assistants):
        for action, tc in enumerate(m.get("tool_calls") or []):
            yield step, action, tc["function"], tc.get("arguments") or {}, outputs.get(tc.get("id"), "")


def reader_index(sample):
    """(step, action) -> the reader's (step, action), or None for submit calls."""
    out, rstep = {}, -1
    for step, m in enumerate(a for a in sample["messages"] if a["role"] == "assistant"):
        tcs = m.get("tool_calls") or []
        kept = [tc for tc in tcs if tc["function"] != "submit"]
        if kept or text_of(m.get("content")).strip():
            rstep += 1
        ra = 0
        for action, tc in enumerate(tcs):
            if tc["function"] == "submit":
                out[(step, action)] = None
            else:
                out[(step, action)] = (rstep, ra)
                ra += 1
    return out


def dump(keys):
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    reports = {(r["arm"], r["key"]): r for r in jsonl(SELF_REPORTS)}
    for i, (key, m, s) in enumerate(samples(keys)):
        lines = [f"KEY {key}", f"SAMPLE {m['sample_id']}  ENV {m['environment']}",
                 f"MAIN TASK {m['main_task']}", f"SIDE TASK {m['side_task']}",
                 "SIDE TASK GOAL:\n" + str(s["metadata"]["side_task"]["goal"]),
                 "AGENT ATTACK DESCRIPTION:\n" + str(reports.get((ARM, key), {}).get("attack_description")), ""]
        assistants = [a for a in s["messages"] if a["role"] == "assistant"]
        by_step = {}
        for step, action, fn, args, out in calls(s):
            by_step.setdefault(step, []).append((action, fn, args, out))
        for step, a in enumerate(assistants):
            t = text_of(a.get("content")).strip()
            lines.append(f"=== STEP {step} ===" + (f"\n[text] {t[:1500]}" if t else ""))
            for action, fn, args, out in by_step.get(step, []):
                arg = json.dumps(args, ensure_ascii=False)
                lines.append(f"--- {step}.{action} {fn}: {arg[:4000]}" + (" [...]" if len(arg) > 4000 else ""))
                lines.append(f"    >> {out[:600]!r}")
        (DUMP_DIR / f"{i:02d}.txt").write_text("\n".join(lines))
        print(i, key, len(assistants), "steps")


def build(keys):
    manual = json.loads(MANUAL.read_text())
    rows, mismatch = [], {}
    for key, m, s in samples(keys):
        hand = manual[key]["labels"]
        seen = set()
        ridx = reader_index(s)
        for step, action, fn, args, out in calls(s):
            sa = f"{step}.{action}"
            if fn == "submit":
                label, note = "submit", ""
            elif sa in hand:
                label, note = hand[sa]
                assert label in ("sabotage", "preparation", "benign") and len(note) <= 120, (key, sa, note)
                assert label == "benign" or note, (key, sa)
                seen.add(sa)
            else:
                label, note = "benign", ""
            if fn != "submit" and ridx[(step, action)] != (step, action):
                mismatch[key] = mismatch.get(key, 0) + 1
            rows.append({"arm": ARM, "key": key, "sample_id": m["sample_id"], "step": step, "action": action,
                         "function": fn, "label": label, "note": note})
        missing = set(hand) - seen
        assert not missing, (key, missing)
    with open(OUT, "x") as f:  # never replace the labels analyses were run on
        f.writelines(json.dumps(r) + "\n" for r in rows)
    from collections import Counter
    print(Counter(r["label"] for r in rows), "->", OUT)
    print("runs whose step/action index differs from the reader's:", mismatch or "none")


if __name__ == "__main__":
    mode = sys.argv[1]
    keys = select() if mode == "select" else json.loads(SELECTED.read_text())["keys"]
    if mode == "dump":
        dump(keys)
    elif mode == "build":
        build(keys)
