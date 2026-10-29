"""Append-only NOTE (RULE F.13): the rh1e4 (honest_b_relaunch) cutfull pack's runs vs 09's frozen members file, by key. Keys and counts only
(git and JSON; no score is read). The pack's staged epochs minus the 5 rows outside membership must equal the 958 members (eval_path#e0),
and those 5 must be exactly the pack's runs without rows (structural zeros). Writes results/cutfull_rh1e4_members_note_<stamp>.json.

  python3 -I check_relaunch_members.py
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MEMBERS = ("051899844", "experiments/2026-10-15_fpr_sweep_507_prep/results/honest_b_relaunch_members_2026-10-16_032731.json")
git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, check=True).stdout  # noqa: E731


def main():
    mb = git("show", f"{MEMBERS[0]}:{MEMBERS[1]}")
    mem = json.loads(mb)
    members = {m["eval_path"] + "#e0" for m in mem["members"]}
    outside = {x["staged_key"]: x for x in mem["staged_outside_membership"]}
    msg = json.loads((HERE / "results" / "cutfull_b_msgs" / "msg_rh1e4_stage_cutfull_r2.json").read_text())
    staged = {e["key"] for e in msg["epochs"]}
    rv = sorted((HERE / "results").glob("cutfull_rh1e4_stage_cutfull_r2_*_reverify_*.json"))
    rowless = {k for p in rv for x in json.loads(p.read_text())["result"].values() for k in x["rowless_epochs"]}
    checks = {"members_958": len(members) == 958 == mem["counts"]["members"], "outside_5": len(outside) == 5,
              "staged_minus_outside_eq_members": staged - set(outside) == members, "outside_subset_staged": set(outside) <= staged,
              "outside_eq_rowless": set(outside) == rowless and len(rv) == 1}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = {"what": "append-only NOTE: the rh1e4 cutfull pack (world honest_b_relaunch) vs 09's frozen members file, by key; keys and counts only",
           "git": git("rev-parse", "HEAD").decode().strip(), "stamp (PT)": stamp,
           "members_file": {"commit": MEMBERS[0], "path": MEMBERS[1], "sha256": hashlib.sha256(mb).hexdigest()},
           "pack": "/cutfull/sold/step3/rh1e4_stage_cutfull_v1__honest_b/ (record cutfull_rh1e4_stage_cutfull_r2_2026-10-16_024922.json)",
           "counts": {"staged": len(staged), "members": len(members), "excluded_by_key": len(outside), "rowless": len(rowless)},
           "excluded_by_key": [{"key": k, "member_key": x.get("key"), "reason": x.get("cause") or x.get("reason")} for k, x in sorted(outside.items())],
           "checks": checks, "pass": all(checks.values())}
    q = HERE / "results" / f"cutfull_rh1e4_members_note_{stamp}.json"
    q.open("x").write(json.dumps(out, indent=1))
    print(q.name, out["counts"], checks)
    if not out["pass"]:
        raise SystemExit("STOP: the rh1e4 pack's keys differ from the members file")


if __name__ == "__main__":
    main()
