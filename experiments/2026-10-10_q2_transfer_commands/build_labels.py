"""Data-transfer probe, step 1 (local, free): one row per tool call of Sol's real and simulated honest runs, keyed the
way the v4 activation dump keys its positions (file, sample id, epoch, step, action; the destructive probe's
`build_labels.calls` numbering). Each row carries the tool name, the impactful/read-only rule label and `transfer`
(this folder's rule, executed-command tools only). No command text is written. Honest runs only.

  uv run --no-project --python 3.14 --with boto3 --with pyyaml python experiments/2026-10-10_q2_transfer_commands/build_labels.py <transcript cache> <out jsonl>
"""
import glob
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analyze_transfer_counts import COHORTS, calls, executed, is_transfer, label_call, planted_ok

ARM = {"real honest": "real-honest", "sim honest": "sim-honest"}


def main(cache: Path, out: Path):
    if not planted_ok():
        raise ValueError("the planted check failed")
    n = Counter()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x") as fh:
        for world, d in COHORTS.items():
            for ev in sorted(glob.glob(str(cache / d / "logs" / "*.eval"))):
                with zipfile.ZipFile(ev) as z:
                    for name in sorted(x for x in z.namelist() if x.startswith("samples/") and x.endswith(".json")):
                        s = json.loads(z.read(name))
                        mt = s.get("metadata", {}).get("main_task")
                        task = (mt.get("id") if isinstance(mt, dict) else mt) or "?"
                        for step, action, tool, args in calls(s):
                            imp = label_call(tool, args)["label"] == "impactful"
                            tr = executed(tool or "") and is_transfer(args)
                            fh.write(json.dumps({"arm": ARM[world], "file": Path(ev).name, "sample_id": s["id"], "epoch": s.get("epoch", 1), "main_task": task,
                                                 "step": step, "action": action, "tool": tool, "impactful": imp, "transfer": tr}) + "\n")
                            n[(world, "calls")] += 1; n[(world, "impactful")] += imp; n[(world, "transfer and impactful")] += tr and imp
    print(json.dumps({f"{w} | {k}": v for (w, k), v in sorted(n.items())}, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]).expanduser(), Path(sys.argv[2]))
