"""Negative tests for the own_all slot (INPUTS_FINAL extensions.own_all_L48): each tampered manifest must fail with its own diagnostic.
Usage: <modal env python> test_own_all_neg.py <INPUTS_FINAL with the own_all slot 'in'>. Reads git and local manifest copies only; no Modal call."""
import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import inputs_final as IF  # noqa: E402
import run_peraction311 as R  # noqa: E402

base = json.loads(Path(sys.argv[1]).read_text())
assert base["extensions"]["own_all_L48"]["status"] == "in" and not IF.verify(base), "the base manifest must have the slot on and pass"
fails = []


def expect(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    fails.extend([] if ok else [name])


def tamper(name, f, must):
    m = copy.deepcopy(base)
    f(m)
    probs = IF.own_all_problems(m, m["extensions"]["own_all_L48"]) if m["extensions"]["own_all_L48"].get("status") == "in" else IF.verify(m)
    expect(f"{name} -> {must!r}", any(must in x for x in probs))


oa = lambda m: m["extensions"]["own_all_L48"]  # noqa: E731
tamper("score record sha differs", lambda m: oa(m)["pointer"]["score"].update(sha256="0" * 64), "pointer.score: sha256 differs")
tamper("09's verification pin missing", lambda m: oa(m)["pointer"].pop("verification_09"), "pointer.verification_09: missing")
tamper("routing-57 addendum dropped (Q2 _rlr root uncovered)", lambda m: oa(m)["pointer"].pop("score_addendum"),
       "simh311_q2_sim_rlr_v1__sim_honest has no entry in the score records")
tamper("a mapped root without own_all", lambda m: m["sets"]["S_new"]["roots"]["simh311_q1"].append("simh311_q1_sim_s9_v1__sim_honest"),
       "simh311_q1_sim_s9_v1__sim_honest has no entry")
tamper("unknown slot status", lambda m: oa(m).update(status="on"), "extensions.own_all_L48: status must be")
m_ = copy.deepcopy(base)
m_["extensions"]["own_all_L48"] = {"status": "not in this build; second refresh", "pointer": None}
expect("slot OFF ('not in this build; second refresh') verifies", not [x for x in IF.verify(m_) if "own_all" in x])
# local copy check: a manifest copy whose sha differs from the score record stops own_all_files
shas = {"su311_q1_main_v1__attack": "0" * 64}
try:
    R.own_all_files("su311_q1_main_v1__attack", "scaleup/su311_q1_main_v1__attack/own_all", shas)
    expect("own_all_files: copy sha differs from the score record stops", False)
except SystemExit as e:
    expect("own_all_files: copy sha differs from the score record stops", "sha256 differs" in str(e))
print(json.dumps({"manifest": sys.argv[1], "failed": fails, "pass": not fails}))
sys.exit(1 if fails else 0)
