"""Counted-key manifest of one launch (RULE.md amendments 10 and 11): every run under the counted-attempt rule
(run_pilot.py: the last attempt of each run), its counted .eval FILE's sha256 and bytes, and whether gate12 counts it
usable; a non-usable run stays in, as a counted failure, with its state, how it ended and, if given, its recorded
cause. Keys, hashes and counts only; no .eval is opened. ON THE DEVBOX (system python3):
  python3 make_counted_keys.py --gate12 <json> --out <json> --world <w> [--stratum S_old] [--cause <task>=<text> ...] <run dir>
Each key also carries world (09's names: simh_pilot, simh_full for the continuation, sim_e2 for A), stratum and launch_id.
"""

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate12", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cause", action="append", default=[], help="<task>=<recorded cause of its failure>")
    ap.add_argument("--world", required=True)
    ap.add_argument("--stratum", default="S_old")
    ap.add_argument("launch", type=Path)
    args = ap.parse_args()
    assert not args.out.exists(), f"{args.out} exists"
    launch = args.launch.expanduser()
    g = {r["run"]: r for r in json.loads(args.gate12.read_text())["runs"]}
    causes = dict(c.split("=", 1) for c in args.cause)
    rec = json.loads((launch / "launch.json").read_text())
    keys = []
    for r in rec["runs"]:
        if r["real"]:
            continue
        i = len(r["attempts"]) - 1
        a = r["attempts"][i] if r["attempts"] else {}
        d = Path(a["log_dir"]) if a else None
        evals = sorted((launch / d.parent.name / d.name).glob("*.eval")) if d else []
        e = evals[0].read_bytes() if len(evals) == 1 else None
        row = {"run_key": r["key"], "task": r["task"], "launch": launch.name, "attempt_index": i,
               "attempt_kind": a.get("kind"), "eval_file": f"{d.parent.name}/{d.name}/{evals[0].name}" if e else None,
               "eval_sha256": hashlib.sha256(e).hexdigest() if e else None, "eval_bytes": len(e) if e else None,
               "usable": g[r["key"]]["usable"], "world": args.world, "stratum": args.stratum, "launch_id": launch.name}
        if not row["usable"]:
            row["counted_failure"] = {"state": g[r["key"]]["state"], "ended": g[r["key"]]["ended"],
                                      "sample_errors": g[r["key"]]["sample_errors"], "cause": causes.get(r["task"])}
        keys.append(row)
    out = {"rule": "counted-attempt rule: the last attempt of each run; sha256 of its .eval file; usable per gate12; "
                   "non-usable runs kept as counted failures", "launch": launch.name, "gate12": str(args.gate12),
           "n": len(keys), "usable": sum(k["usable"] for k in keys), "keys": sorted(keys, key=lambda k: k["run_key"])}
    args.out.write_text(json.dumps(out, indent=1))
    print(f"{len(keys)} counted keys, {out['usable']} usable, failures: "
          f"{[k['task'] for k in keys if not k['usable']]}")


if __name__ == "__main__":
    main()
