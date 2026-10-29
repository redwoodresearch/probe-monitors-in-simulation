"""Destructive-command probe, step 1 (local, free): one row per tool call of Sol's real and simulated honest runs, keyed
the way the v4 activation dump keys its positions (file, sample id, epoch, step, action; step = agent turn and action =
call within the turn of probe_shift.transcripts.reader_messages, as the reader numbers them). Each row carries the
tool name, the impactful/read-only rule label (probe_shift.calls.label_call) and the destructive rule of
analyze_destructive_counts.py. No command text is written. Honest runs only.

  uv run --no-project --python 3.14 python experiments/2026-10-10_q2_destructive_commands/build_labels.py <transcript cache> <out jsonl>
"""
import glob
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from analyze_destructive_counts import COHORTS, DESTRUCTIVE

from probe_shift.calls import label_call
from probe_shift.transcripts import reader_messages

ARM = {"real honest": "real-honest", "sim honest": "sim-honest"}


def calls(sample: dict):
    """(step, action, tool, arguments) in the reader's numbering."""
    step = -1
    for m in reader_messages(sample, "system"):
        if m["role"] != "assistant":
            continue
        step += 1
        for action, tc in enumerate(m.get("tool_calls") or []):
            fn = tc.get("function")
            name = fn.get("name") if isinstance(fn, dict) else fn
            args = fn.get("arguments") if isinstance(fn, dict) else tc.get("arguments")
            if isinstance(args, str):
                args = json.loads(args) if args.strip().startswith("{") else {"cmd": args}
            yield step, action, name, args or {}


def main(cache: Path, out: Path):
    n = Counter()
    with open(out, "w") as fh:
        for world, d in COHORTS.items():
            for ev in sorted(glob.glob(str(cache / d / "logs" / "*.eval"))):
                with zipfile.ZipFile(ev) as z:
                    for name in sorted(z.namelist()):
                        if not (name.startswith("samples/") and name.endswith(".json")):
                            continue
                        s = json.loads(z.read(name))
                        mt = s.get("metadata", {}).get("main_task")
                        task = (mt.get("id") if isinstance(mt, dict) else mt) or "?"
                        for step, action, tool, args in calls(s):
                            lab = label_call(tool, args)
                            row = {"arm": ARM[world], "file": Path(ev).name, "sample_id": s["id"], "epoch": s.get("epoch", 1), "main_task": task,
                                   "step": step, "action": action, "tool": tool, "impactful": lab["label"] == "impactful",
                                   "destructive": bool(DESTRUCTIVE.search(json.dumps(args)))}
                            fh.write(json.dumps(row) + "\n")
                            n[(world, "calls")] += 1; n[(world, "impactful")] += row["impactful"]; n[(world, "destructive")] += row["destructive"]
                            n[(world, "destructive and impactful")] += row["destructive"] and row["impactful"]
    print(json.dumps({f"{w} | {k}": v for (w, k), v in sorted(n.items())}, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
