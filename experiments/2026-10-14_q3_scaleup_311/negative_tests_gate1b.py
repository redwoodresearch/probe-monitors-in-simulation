"""Negative tests of gate1_311b.py's run-level counting (RULE.md Amendment 4), $0, local: the counting functions are
loaded from the script itself (everything before its watch loop) and run on synthetic launch records.
  python3 negative_tests_gate1b.py"""
import sys
from pathlib import Path

src = (Path(__file__).parent / "gate1_311b.py").read_text()
ns = {"__name__": "gate1b_test", "__file__": str(Path(__file__).parent / "gate1_311b.py")}
sys.argv = ["gate1_311b.py", "/nonexistent", "s311qX", "0"]
exec(src.split("stopped = paused = False")[0], ns)
counts = ns["counts"]


def run(arm, task, state, attempts, real=True):
    return {"arm": arm, "task": task, "state": state, "attempts": attempts, "real": real, "simulator": "september"}


def att(usable_calls, redrawn=False):
    a = {"outcome": {"policy_calls": [{"usable": u} for u in usable_calls]}}
    if redrawn:
        a["redrawn_because"] = "prestart"
    return a


ok = True
# 1. many usable calls, few usable runs: 30 ended honest_a runs, each with 5 usable reference-rerun attempts but a LAST
#    attempt that is not usable, except 3 -> 3 of 30 usable: the gate must FAIL (call counting would have passed).
recs = [run("honest_a", f"t{i}", "done", [att([True])] * 5 + [att([i < 3])]) for i in range(30)]
c = counts({"runs": recs})["honest_a"]
ok &= c["ended"] == 30 and c["usable"] == 3 and c["usable"] < 0.9 * c["ended"]
print("1 many calls, few usable runs:", c, "-> FAIL as required" if c["usable"] < 0.9 * c["ended"] else "-> WRONGLY PASSES")
# 2. running and not-yet-ended runs are not counted; a redrawn last attempt is skipped for the earlier counted one.
recs = [run("attack", "t1", "running", [att([True])]), run("attack", "t2_r2", "done", [att([True]), att([False], True)]),
        run("attack", "t3", "done", [])]
c = counts({"runs": recs})
ok &= c["attack r1"] == {"runs": 2, "ended": 1, "usable": 0} and c["attack r2"] == {"runs": 1, "ended": 1, "usable": 1}
print("2 running not counted; redrawn skipped; no outcome = not usable:", c)
# 3. an arm with no ended run has explicit zeros (no KeyError).
c = counts({"runs": [run("honest_b", "t9", "ready", [])]})["honest_b"]
ok &= c == {"runs": 1, "ended": 0, "usable": 0}
print("3 zeros:", c)
# 4. exactly 10% unusable (18 of 20) must stop: the gate's own comparison, read from the script.
src_low = [l for l in src.splitlines() if "low = [k for k, v in c.items()" in l][0]
c = {"attack r1": {"runs": 20, "ended": 20, "usable": 18}}
low = eval(src_low.split("=", 1)[1].strip(), {"c": c})
ok &= low == ["attack r1"]
print("4 exactly 10% unusable (18 of 20) stops:", low)
# 5. the deadline check comes before reading launch.json and before the pair check in the loop.
loop = src.split("stopped = paused = False")[1]
ok &= loop.index("DEADLINE_STOP.txt") < loop.index('launch.json').__index__() if "launch.json" in loop else False
ok &= "timeout=max(1.0, min(900.0, left))" in src
print("5 deadline first; pair check bounded by the time left:", loop.index("DEADLINE_STOP.txt") < loop.index("launch.json"))
print("ALL PASS" if ok else "SOME FAIL")
sys.exit(0 if ok else 1)
