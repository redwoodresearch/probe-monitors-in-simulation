"""Negative tests for --levels rule37 (09 RULE 37 @ 6977f80f1), synthetic only (no inputs, no Modal): the level set is exactly 1e-4, 2e-4, 5e-4
(no other GRID point below 1e-3) and sweep15's 15 levels are unchanged; the readability gate passes only when every contributing origin's S
(tau_sim) and B (tau_B) pools pass at draw 0 (a missing flag fails closed); the smallest shown level ignores unreadable levels. Exit 1 on any failure.

  <modal env python> experiments/2026-10-15_fpr_sweep_507_prep/test_rule37.py
"""

import sys

import run_peraction311 as R
from run_fprsweep_507 import GRID

bad = []


def check(name, cond):
    if not cond:
        bad.append(name)


lv = R.rule37_levels()
check("exact levels", lv == [1e-4, 2e-4, 5e-4])
check("all below 1e-3", all(f < 1e-3 for f in lv))
below = [f for f in GRID if f < 1e-3 * (1 - 1e-9)]
check("GRID has other points below 1e-3 (the test means something)", len(below) > 1)
check("no other GRID point below 1e-3", [f for f in lv if any(abs(f - g) <= 1e-12 * g for g in below)] == [1e-4])
s15 = sorted({f for f in GRID if 1e-3 * (1 - 1e-9) <= f <= 0.05 * (1 + 1e-9)} | {0.05})  # sweep15's expression, unchanged in run_peraction311
check("sweep15 still 15 levels with 1e-2", len(s15) == 15 and 0.01 in s15)
check("sets disjoint", not set(lv) & set(s15))
check("labels", R.RULE37_META["label"].startswith("added after registration, before values") and R.RULE37_META["levels"] == lv)

ok = {"S": True, "B": True, "A": True}
check("gate passes", R.rule37_gate({"O1": ok, "O2": ok}, ["O1", "O2"]))
check("gate ignores A", R.rule37_gate({"O1": ok | {"A": False}}, ["O1"]))
check("gate fails on one origin's S", not R.rule37_gate({"O1": ok, "O2": ok | {"S": False}}, ["O1", "O2"]))
check("gate fails on B", not R.rule37_gate({"O1": ok | {"B": False}}, ["O1"]))
check("gate fails closed on a missing flag", not R.rule37_gate({"O1": {"S": True}}, ["O1"]))
check("gate fails closed on a non-bool flag", not R.rule37_gate({"O1": {"S": 1, "B": True}}, ["O1"]))
check("gate fails on no members", not R.rule37_gate({}, []))
check("gate counts only members", R.rule37_gate({"O1": ok, "O2": ok | {"S": False}}, ["O1"]))

fx = {"0.0001": {"readable_rule37": False}, "0.0002": {"readable_rule37": True}, "0.0005": {"readable_rule37": True}}
check("smallest shown", R.rule37_smallest(fx) == 2e-4)
check("smallest: non-bool ignored", R.rule37_smallest({"0.0001": {"readable_rule37": 1}, "0.0005": {"readable_rule37": True}}) == 5e-4)
none_ = R.rule37_smallest({"0.0001": {"readable_rule37": False}, "0.0002": {}})
check("smallest: none shown", isinstance(none_, str) and none_.endswith(R.RULE37_UNREAD))

for lv_, pl_, stop_ in (("rule37", True, True), ("rule37", False, False), ("sweep15", True, False), ("", True, False)):
    try:
        R.rule37_args(lv_, pl_)
        got_ = False
    except SystemExit:
        got_ = True
    check(f"rule37_args({lv_!r}, plant={pl_}) stops={stop_}", got_ is stop_)

print({"failures": bad, "n_checks_failed": len(bad)})
sys.exit(1 if bad else 0)
