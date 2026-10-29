"""One --references-from folder for every launch of this job (RULE.md, amendment 7: one launch per phase).

run_pilot.py reads only the prior launch.json's "references" (run_pilot.py:661) and sha256-checks each copied reference
(:676). This writes <out>/launch.json holding, for every task of the given task files, its reference entry copied
VERBATIM from the launch the task file names (simulated honest: honest_a_launch; A: first_sim_attack_launch). It stops
if a task file names two launches for one task, if the named launch has no entry, if any SOURCE launch holds a
different entry for the task (canonical JSON), or if a reference file's sha256 differs from its entry. It records each
task's source launch and the canonical-JSON sha256 of each entry. Keys, paths and hashes only; no transcript is opened.
ON THE DEVBOX (system python3):
  python3 make_merged_refs.py --out ~/phase2/runs/<name> --runs ~/phase2/runs --sources <launch> ... -- <task files>
"""

import argparse
import hashlib
import json
from pathlib import Path


def canon(x) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--sources", nargs="+", required=True, help="every launch that holds references")
    ap.add_argument("task_files", nargs="+", type=Path)
    args = ap.parse_args()
    out = args.out.expanduser()
    assert not out.exists(), f"{out} exists"
    named = {}
    for f in args.task_files:
        for t in json.loads(f.read_text())["tasks"]:
            launch = t.get("honest_a_launch") or t["first_sim_attack_launch"]
            assert named.setdefault(t["key"], launch) == launch, f"{t['key']}: two reference launches named"
    src = {s: json.loads((args.runs.expanduser() / s / "launch.json").read_text())["references"] for s in args.sources}
    assert set(named.values()) <= set(src), f"a named launch is not among --sources: {set(named.values()) - set(src)}"
    refs, per_task = {}, {}
    for key, launch in sorted(named.items()):
        assert key in src[launch], f"{key}: no entry in {launch}"
        entry = src[launch][key]
        others = {s for s, r in src.items() if key in r and canon(r[key]) != canon(entry)}
        assert not others, f"{key}: a different entry in {sorted(others)}"
        got = hashlib.sha256(Path(entry["copy"]).read_bytes()).hexdigest()
        assert got == entry["sha256"], f"{key}: {entry['copy']} differs from its recorded sha256"
        refs[key] = entry
        per_task[key] = {"source": launch, "entry_sha256": hashlib.sha256(canon(entry).encode()).hexdigest(),
                         "also_in": sorted(s for s, r in src.items() if key in r and s != launch)}
    out.mkdir(parents=True)
    rec = {"references": refs, "per_task": per_task,
           "task_files": {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in args.task_files},
           "sources": args.sources,
           "note": "each entry verbatim from its source launch.json; read by run_pilot.py --references-from"}
    b = json.dumps(rec, indent=1).encode()
    (out / "launch.json").write_bytes(b)
    by = {}
    for v in per_task.values():
        by[v["source"]] = by.get(v["source"], 0) + 1
    print(f"{out}/launch.json sha256 {hashlib.sha256(b).hexdigest()}: {len(refs)} references, each file's sha256 "
          f"checked, no conflicting entry; from {by}")


if __name__ == "__main__":
    main()
