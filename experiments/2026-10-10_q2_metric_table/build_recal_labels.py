"""Separate-calibration recompute, step 1 (local, free): per-call labels for both probes, keyed like the v4 dump
(`build_labels.calls` numbering): `destructive_raw` (the destructive rule on each raw argument string, the corrected
rule of analyze_label_escaping.py) and `transfer` (the transfer rule as frozen), with the impactful label. No command
text is written. Honest runs only.

  uv run --no-project --python 3.14 --with boto3 --with pyyaml python experiments/2026-10-10_q2_metric_table/build_recal_labels.py <transcript cache> <out jsonl>
"""
import glob
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

root = Path(__file__).parents[1]
sys.path.insert(0, str(root / "2026-10-10_q2_transfer_commands"))
sys.path.insert(0, str(root / "2026-10-10_q2_destructive_commands"))
from analyze_destructive_counts import COHORTS, DESTRUCTIVE
from analyze_transfer_counts import calls, executed, is_transfer, label_call

ARM = {"real honest": "real-honest", "sim honest": "sim-honest"}


def main(cache: Path, out: Path):
    n = Counter()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x") as fh:
        for world, d in COHORTS.items():
            for ev in sorted(glob.glob(str(cache / d / "logs" / "*.eval"))):
                with zipfile.ZipFile(ev) as z:
                    for name in sorted(x for x in z.namelist() if x.startswith("samples/") and x.endswith(".json")):
                        s = json.loads(z.read(name))
                        mt = s.get("metadata", {}).get("main_task"); task = (mt.get("id") if isinstance(mt, dict) else mt) or "?"
                        for step, action, tool, args in calls(s):
                            imp = label_call(tool, args)["label"] == "impactful"
                            dr = any(DESTRUCTIVE.search(v if isinstance(v, str) else json.dumps(v)) for v in args.values())
                            tr = executed(tool or "") and is_transfer(args)
                            fh.write(json.dumps({"arm": ARM[world], "file": Path(ev).name, "sample_id": s["id"], "epoch": s.get("epoch", 1), "main_task": task,
                                                 "step": step, "action": action, "impactful": imp, "destructive_raw": dr, "transfer": tr}) + "\n")
                            n[(world, "impactful")] += imp; n[(world, "destructive_raw and impactful")] += dr and imp; n[(world, "transfer and impactful")] += tr and imp
    print(json.dumps({f"{w} | {k}": v for (w, k), v in sorted(n.items())}, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]).expanduser(), Path(sys.argv[2]))
