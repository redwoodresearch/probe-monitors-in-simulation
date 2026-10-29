"""Evidence row for 09's class ruling (via d9 10-15 ~07:10): il-s5's run3|nm203_slack_clone, the one row staging excluded, in the
corrected-record format of setup_failures_nm360 v3 (586ab8784), with its cross-checks: the S5 main's run1 / run2 / honest records
of the cell in setup_failures v3, the S5 sim follow-up's state of the cell (s5_sim_status_manifests @ 08f0fe3c4), il-s5's own
sim2 run of the cell (its run table), and the v1 provenance row (no policy call in run1, run2 and run3). Keys and hashes only.

  python3 make_il_s5_nm203_evidence.py
"""
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
CELL = "nm203_slack_clone"


def show(commit: str, path: str) -> dict:
    return json.loads(subprocess.run(["git", "show", f"{commit}:experiments/2026-10-14_q3_overnight_chain/{path}"], capture_output=True, check=True, cwd=HERE).stdout)


def main():
    (row,) = [f for f in json.loads((HERE / "config" / "manifest_ilnm_s5_run3.json").read_text())["files"] if f["arm_dir"] == "attack" and f["task"] == CELL]
    rec = json.loads((HERE / "results" / "chain_stage_ilnm_s5_run3_2026-10-15_063859.json").read_text())
    k = next(x for x in rec if x not in ("git", "stamp"))
    (why,) = [v for key, v in rec[k]["c_gate"]["excluded"].items() if key.split("#", 1)[0] == row["path"]]
    sf3 = show("586ab8784", "results/hygiene/setup_failures_nm360_v3_2026-10-15.json")
    s5main = [{x: r[x] for x in ("world", "arm", "launch_task", "state", "policy_called", "record", "sha256", "staged", "staging_record_reason")}
              for r in sf3["records"] if r["new_main"] == "S5" and r["cell"] == CELL]
    sim = show("08f0fe3c4", "results/hygiene/s5_sim_status_manifests_2026-10-15.json")
    sim_rows = {bt: keys[f"sim|{CELL}"] for bt, bd in sim["batches"].items() for keys in [bd["rows"]] if f"sim|{CELL}" in keys}
    rt = json.loads((HERE / "results" / "hygiene" / "run_table_ilnm_s5.json").read_text())
    il_rows = [{x: c.get(x) for x in ("key", "world", "status")} for c in rt["cells"] if c.get("cell") == CELL]
    pv = json.loads((HERE / "results" / "hygiene" / "approach_provenance_ilnm_s5.json").read_text())
    (prow,) = [r for r in pv["rows"] if r["key"] == f"run3|{CELL}"]
    doc = {"for": "09's class ruling (09 @ 65ef3a9ff; via d9 10-15): il-s5 run3|nm203_slack_clone, staged out; corrected-record format of setup_failures_nm360 v3 @ 586ab8784",
           "record": {"new_main": "S5", "batch": "il-s5", "world": "run3", "arm": "attack", "cell": CELL, "launch_task": row["launch_task"], "launch": "2026-10-14_il_s5",
                      "state": row["state"], "policy_called": row["policy_called"], "attempt_kind": row["attempt_kind"], "record": row["path"], "sha256": row["sha256"],
                      "staged": False, "staging_record_reason": why, "staging_exclusion": True, "reason": "run_failed (no policy call)", "class": "unknown"},
           "cross_checks": {"setup_failures_nm360_v3_S5_records": s5main, "setup_failures_nm360_v3_reason": sf3["reason"],
                            "s5_sim_status_08f0fe3c4": sim_rows, "il_s5_run_table": il_rows,
                            "provenance_v1_row": {"no_policy_call": prow.get("no_policy_call"), "pairs": len(prow["pairs"]), "pass": prow["pass"],
                                                  "worlds": {w: {x: v.get(x) for x in ("launch", "launch_task", "policy_calls", "eval", "eval_sha256")} for w, v in prow["worlds"].items()}}}}
    out = HERE / "results" / "hygiene" / "evidence_il_s5_nm203_2026-10-15.json"
    with out.open("x") as f:
        f.write(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"record": doc["record"], "s5_main_records": len(s5main), "s5_main_states": sorted({r["state"] for r in s5main}),
                      "sim_follow_up": sim_rows, "il_s5": il_rows, "no_policy_call": prow.get("no_policy_call")}, indent=1))


if __name__ == "__main__":
    main()
