"""The upload manifest of one phase's usable runs (RULE.md amendment 6; the stager's input list, run_simh_stage.py).

From the launch's launch.json and its gate12 result: for each usable run, its counted (last) attempt's single .eval,
with bytes and sha256 (hashed on the devbox). Schema as stage_q4b.counted_attempts reads it: files[] with launch, rel,
sha256, bytes, arm_dir, task, attempt_index, attempt_kind, policy_called, path; counted_attempts_without_eval per set
(must be empty). Keys, paths and hashes only; no .eval is opened.
ON THE DEVBOX (system python3):  python3 make_upload_manifest.py --gate12 <json> --out <manifest.json> <run dir>
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate12", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("launch", type=Path)
    args = ap.parse_args()
    assert not args.out.exists(), f"{args.out} exists"
    launch = args.launch.expanduser()
    usable = {r["run"] for r in json.loads(args.gate12.read_text())["runs"] if r["usable"]}
    rec = json.loads((launch / "launch.json").read_text())
    files, missing = [], []
    for r in rec["runs"]:
        if r["key"] not in usable:
            continue
        i = len(r["attempts"]) - 1
        d = Path(r["attempts"][i]["log_dir"])
        evals = sorted((launch / d.parent.name / d.name).glob("*.eval"))
        if len(evals) != 1:
            missing.append(r["key"])
            continue
        rel = f"{d.parent.name}/{d.name}/{evals[0].name}"
        b = evals[0].read_bytes()
        files.append({"launch": launch.name, "rel": rel, "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b),
                      "arm_dir": d.parent.name, "task": r["task"], "attempt_index": i,
                      "attempt_kind": r["attempts"][i].get("kind", "first"),
                      "policy_called": r["attempts"][i].get("policy_called"), "path": f"{launch.name}/{rel}"})
    out = {"source": str(launch), "gate12": str(args.gate12), "counts": {"usable": len(usable), "files": len(files)},
           "files": files, "counted_attempts_without_eval": {"simh": missing}}
    args.out.write_text(json.dumps(out, indent=1))
    print(f"{len(files)} files of {len(usable)} usable runs, {sum(f['bytes'] for f in files)} bytes; "
          f"without one .eval: {len(missing)}")
    sys.exit(1 if missing or len(files) != len(usable) else 0)


if __name__ == "__main__":
    main()
