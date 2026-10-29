"""Addendum to the compose-race check (d9, 10-15 ~00:40 PT): joins, per REAL attempt of a new-mains main launch, the
box-level label (compose_race_check.py: the attempt's sandbox project found by time order, then clean / mismatch / mixed /
bad_tag; "unmatched" = no project found for it within 900 s after its start in that order, i.e. the time-order mapping
lost alignment; simulated attempts are not in that mapping at all) with the per-attempt verdict of
compose_race_check_attempts.py (recorded compose inputs vs every overlapping attempt of another launcher in the same
environment and settings folder: unaffected / affected-clean / mismatched / failed_loud). Keys and labels only.
  python3 compose_race_join.py <box-level json> <per-attempt json>"""
import collections, json, sys
box, att = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
verdict = {}
for name, lst in (("affected-clean", att["affected"]), ("mismatched", att["mismatched"]), ("failed_loud", att["failed_loud"])):
    for r in lst:
        k = (r["launch"], r["key"], r["attempt"])
        verdict[k] = name if name != "affected-clean" or k not in verdict else verdict[k]
for r in att["mismatched"] + att["failed_loud"]:
    verdict[(r["launch"], r["key"], r["attempt"])] = "mismatched" if r in att["mismatched"] else "failed_loud"
out = {"box": att["box"], "launches": {}}
for launch, v in box["per_run"].items():
    cross, rows = collections.Counter(), []
    for kk, lab in v["labels_by_key"].items():
        key, i = kk.rsplit("#", 1)
        ver = verdict.get((launch, key, int(i) + 1), "unaffected")
        cross[(lab, ver)] += 1
        if lab == "unmatched":
            rows.append({"key": key, "attempt": int(i) + 1, "per_attempt_verdict": ver})
    bad = [r for r in rows if r["per_attempt_verdict"] in ("mismatched", "failed_loud")]
    out["launches"][launch] = {"real_attempts": len(v["labels_by_key"]), "crosstab": {f"{a} x {b}": n for (a, b), n in sorted(cross.items())},
                               "unmatched": rows, "unmatched_not_clean": bad}
print(json.dumps(out))
