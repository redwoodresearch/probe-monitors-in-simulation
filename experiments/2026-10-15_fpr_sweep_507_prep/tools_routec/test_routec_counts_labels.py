"""Synthetic checks (10-18): preflight_job_count.selection (09's R-5 (a) rule) and first_route_c_label.awk (a8953dad's R-4, with the
two-string control). No Modal call, no real spec, no READ_LOG line; awk runs on fixture strings only."""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import preflight_job_count as P  # noqa: E402


def label(line):
    return subprocess.run(["awk", "-f", str(HERE / "first_route_c_label.awk")], input=line + "\n", capture_output=True, text=True, check=True).stdout


def main():
    t = {"simh_old": {"L C": [{"s_old_only": True}, {"x": 1}]}}
    spec = {"jobs": [("L C", {"origin": "S_old"} | t, {}), ("L R", {"origin": "S_old"} | t, {}), ("L C", {"origin": "S_old", "cohort": "registered"} | t, {}),
                     ("L C", {"origin": "S_new", "simh_old": {"L C": [{"x": 1}]}}, {}), ("L C", {"origin": "S_newmains"}, {})]}
    bad = spec | {"jobs": spec["jobs"] + [("L C", {"origin": "S_new"} | t, {})]}
    cases = {
        "selection: tagged S_old jobs counted (registered-cohort duplicates too), untagged and other origins not": P.selection(spec)
        == {"n_tagged": 3, "n_tagged_s_old": 3, "n_tagged_not_s_old": 0, "n_jobs_all_audit": 5},
        "selection: a tagged non-S_old job is counted apart": P.selection(bad)["n_tagged_not_s_old"] == 1 and P.selection(bad)["n_tagged_s_old"] == 3,
        "label: one string": label('x "route C pooled a b" y') == "route C pooled a b\n",
        "label, the two-string control: the FIRST string, not the last": label('"route C pooled first" and "route C pooled second"') == "route C pooled first\n",
        "label: a quoted string before it is skipped": label('"other" "route C pooled z"') == "route C pooled z\n",
        "label: none on the line prints nothing": label('no label "route C" here') == "",
    }
    for name, ok in cases.items():
        print(("PASS " if ok else "FAIL ") + name)
    assert all(cases.values()), "tests failed"
    print(f"{len(cases)} checks pass")


if __name__ == "__main__":
    main()
