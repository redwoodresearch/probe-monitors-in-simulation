"""Gates 1 and 2 of one launch (RULE.md, Gates; frozen before any result), from its final launch.json. Exit 0 only if
both pass and the selected tasks are exactly the expected ones; 1 otherwise.

Selected: every simulated run the launch lists (one per task); its task keys must equal --expect-keys. Usable: state
"done", its last attempt's outcome clean (eval status success, one sample, no sample error, nothing unreadable), no
sample limit, and the policy's call record usable (a tool result other than submit, no sample error;
run_attack_check.py:174-178).
Gate 1: usable / selected >= 0.90; every non-usable run is listed with its state and reason.
Gate 2: cost per policy-called run <= $25. A run's cost = the sum of its attempts' priced simulator spend (sim_usd). If
any attempt ended without a price and did not certainly skip the policy (policy_called True or unknown), the run counts
at $60 in total instead, replacing its partial charges. An attempt that certainly skipped the policy (policy_called
False) and has no price adds its live reading (the harness's own, normally 0). Denominator: the selected runs with an
attempt whose policy was called or may have been (True or unknown). The substitution is the rule: unpriced runs do not
fail the gate by themselves.
Tallies (the compose-race record): attempts by kind, pre-start failures (attempt "prestart" set), redraws, and runs
started more than once.
ON THE DEVBOX (system python3):  python3 gate12.py --out <json> --expect-keys <keys.txt> <run dir>
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

UNPRICED_USD = 60.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--expect-keys", type=Path, required=True)
    ap.add_argument("launch", type=Path)
    args = ap.parse_args()
    assert not args.out.exists(), f"{args.out} exists"
    expect = {k for k in args.expect_keys.read_text().replace("\n", ",").split(",") if k}
    rec = json.loads((args.launch.expanduser() / "launch.json").read_text())
    finished = bool(rec["finished"])
    runs, kinds, prestart = [], Counter(), 0
    for r in rec["runs"]:
        if r["real"]:
            continue
        last = r["attempts"][-1] if r["attempts"] else {}
        out = last.get("outcome") or {}
        calls = out.get("policy_calls") or []
        usable = (r["state"] == "done" and bool(out.get("clean")) and not out.get("sample_limits")
                  and len(calls) == 1 and calls[0].get("usable") == 1)
        called = any(a.get("policy_called") is not False for a in r["attempts"])
        unpriced = any("sim_usd" not in a and a.get("policy_called") is not False for a in r["attempts"])
        cost = UNPRICED_USD if unpriced else sum(a["sim_usd"] if "sim_usd" in a else a.get("live_usd", 0.0)
                                                 for a in r["attempts"])
        kinds.update(a.get("kind", "?") for a in r["attempts"])
        prestart += sum(bool(a.get("prestart")) for a in r["attempts"])
        runs.append({"run": r["key"], "task": r["task"], "state": r["state"], "why": r.get("why"),
                     "attempts": len(r["attempts"]), "usable": usable, "policy_called": called, "unpriced": unpriced,
                     "cost_usd": round(cost, 4), "ended": calls[0].get("ended") if calls else None,
                     "sample_errors": len(out.get("sample_errors") or []), "sample_limits": out.get("sample_limits")})
    keys = [r["task"] for r in runs]
    keys_ok = len(keys) == len(set(keys)) and set(keys) == expect
    selected, usable = len(runs), sum(r["usable"] for r in runs)
    called = [r for r in runs if r["policy_called"]]
    mean = sum(r["cost_usd"] for r in called) / len(called) if called else None
    res = {"launch": str(args.launch), "launcher_finished": finished, "selected": selected,
           "expected_keys": len(expect), "keys_ok": keys_ok, "usable": usable,
           "gate1_completion": usable / selected if selected else None,
           "gate1_pass": bool(selected) and usable / selected >= 0.90,
           "policy_called": len(called), "unpriced_runs": sum(r["unpriced"] for r in runs),
           "gate2_cost_per_called_run_usd": round(mean, 2) if mean is not None else None,
           "gate2_pass": mean is not None and mean <= 25.0,
           "tallies": {"attempts_by_kind": dict(kinds), "prestart_failures": prestart,
                       "runs_started_more_than_once": sum(r["attempts"] > 1 for r in runs)},
           "runs": runs}
    ok = finished and keys_ok and res["gate1_pass"] and res["gate2_pass"]
    res["pass"] = ok
    args.out.write_text(json.dumps(res, indent=1))
    print(f"launcher finished: {finished}; tasks: {selected} selected, {len(expect)} expected, match {keys_ok}")
    print(f"gate 1: {usable}/{selected} usable ({res['gate1_completion']}) -> "
          f"{'PASS' if res['gate1_pass'] else 'FAIL'}")
    print(f"gate 2: ${res['gate2_cost_per_called_run_usd']} per policy-called run over {len(called)} runs "
          f"({res['unpriced_runs']} unpriced at ${UNPRICED_USD:.0f}) -> {'PASS' if res['gate2_pass'] else 'FAIL'}")
    print(f"tallies: {res['tallies']}")
    for r in runs:
        if not r["usable"]:
            print(f"  not usable: {r['run']} state={r['state']} why={r['why']} ended={r['ended']} "
                  f"errors={r['sample_errors']} limits={r['sample_limits']}")
    print("GATES 1-2", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
