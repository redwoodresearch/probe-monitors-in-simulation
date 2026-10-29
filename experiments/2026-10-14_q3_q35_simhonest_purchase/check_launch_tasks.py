"""Exactly the intended tasks, with exactly their original cells (RULE.md, amendment 7).

Pre-launch (no run dir): the keys file must equal the intended set (--expect-from: task files with "tasks" lists,
minus --minus keys files), and each key's cell in the launch's task file (--launch-tasks) must equal, as canonical JSON,
its cell in the source files (--cells: the 176 and amendment-9 files). With --run, the same for the tasks the launcher
recorded (launch.json "tasks"). Exit 1 on any difference. Keys and hashes only.
ON THE DEVBOX (system python3):
  python3 check_launch_tasks.py --keys <keys.txt> --launch-tasks <task file> --cells <176> <amendment9>
      [--expect-from tasks_simhonest_170.json --minus pilot_keys.txt] [--run ~/phase2/runs/<run>]
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def canon(x) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", type=Path, required=True)
    ap.add_argument("--launch-tasks", type=Path, required=True)
    ap.add_argument("--cells", nargs="+", type=Path, required=True)
    ap.add_argument("--expect-from", nargs="*", type=Path, default=[])
    ap.add_argument("--minus", nargs="*", type=Path, default=[])
    ap.add_argument("--run", type=Path)
    args = ap.parse_args()
    keys = [k for k in args.keys.read_text().replace("\n", ",").split(",") if k]
    bad = []
    if len(keys) != len(set(keys)):
        bad.append("duplicate keys")
    if args.expect_from:
        want = {t["key"] for f in args.expect_from for t in json.loads(f.read_text())["tasks"]}
        for m in args.minus:
            want -= {k for k in m.read_text().replace("\n", ",").split(",") if k}
        if set(keys) != want:
            bad.append(f"keys != intended: {len(set(keys) - want)} extra, {len(want - set(keys))} missing")
    src = {}
    for f in args.cells:
        for c in json.loads(f.read_text())["cells"]:
            assert src.setdefault(c["key"], canon(c)) == canon(c), f"{c['key']}: two different source cells"
    launch = {c["key"]: canon(c) for c in json.loads(args.launch_tasks.read_text())["cells"]}
    diff = [k for k in keys if k not in src or launch.get(k) != src[k]]
    if diff:
        bad.append(f"{len(diff)} cells differ from their source, e.g. {diff[:3]}")
    if args.run:
        rec = json.loads((args.run.expanduser() / "launch.json").read_text())
        ran = {t["key"]: canon(t) for t in rec["tasks"]}
        if set(ran) != set(keys):
            bad.append(f"the launcher's tasks != keys: {len(set(ran) - set(keys))} extra, "
                       f"{len(set(keys) - set(ran))} missing")
        rdiff = [k for k, v in ran.items() if src.get(k) != v]
        if rdiff:
            bad.append(f"{len(rdiff)} recorded cells differ from their source, e.g. {rdiff[:3]}")
    digest = hashlib.sha256(",".join(sorted(keys)).encode()).hexdigest()[:16]
    print(f"{len(keys)} keys (sha256 of the sorted list {digest}); "
          f"cells checked against {len(src)} source cells{'; launch.json checked' if args.run else ''}: "
          f"{'PASS' if not bad else 'FAIL ' + '; '.join(bad)}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
