"""Tests of derive_xc.derive / compare (09's XC fill rule, 10-18) on synthetic AUTH fixtures with fake git functions: no repository, no
git call, no real record. Cases, each with its exact outcome:
- positive control: an AUTH citing READ_LOG @ c1:10 (c1 not an ancestor of RH, on a pushed branch, the file 100 lines long) gives
  XC = [c1], and --expect [c1] passes;
- 09's two positive controls: an AUTH citing a commit missing from the expected XC STOPs, and an extra XC entry no rule cites STOPs;
- a cited commit that is an ancestor of RH is not in XC; a commit on no pushed branch, an absent cited file, a cited line beyond the
  file's line count, an unknown commit and a rule not in rp's form each STOP; RULE item N @ c needs RULE.md only;
- RX_RULE is rp's own pattern, by text, in run_peraction311.py at this checkout; a03f5d93 (10-18): an ambiguous or non-extending
  abbreviation STOPs, a resolution that is not 40 hex STOPs (mutant M8), and an expected XC of null STOPs (not treated as []).

  uv run --offline --no-project --python 3.12 python -I experiments/2026-10-15_fpr_sweep_507_prep/tools_routec/test_derive_xc.py
"""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
s = importlib.util.spec_from_file_location("derive_xc_under_test", HERE / "derive_xc.py")
D = importlib.util.module_from_spec(s)
s.loader.exec_module(D)

C1, C2, C0 = "1" * 40, "2" * 40, "0" * 40  # C0: an ancestor of RH
SHORT = {"1111111": C1, "2222222": C2, "0000000": C0}
LOG, RULE = D.SPEC09 + "READ_LOG.md", D.SPEC09 + "RULE.md"


def fakes(pushed=(C1, C2), files=None):
    files = files if files is not None else {(C1, LOG): 100, (C2, LOG): 100, (C1, RULE): 50, (C2, RULE): 50}
    return (lambda c: SHORT.get(c) or (c if c in (C1, C2, C0) else None), lambda f: f == C0, lambda f: f in pushed,
            lambda f, p: files.get((f, p)))


def auth(*rules):
    return {"authorized_overlaps": [{"key": f"k{i}", "roles": ["A", "B"], "rule": r, "note": "n"} for i, r in enumerate(rules)]}


def run(a, expect=None, **kw):
    xc, pr = D.derive(a, *fakes(**kw))
    return xc, pr + (D.compare(xc, expect) if expect is not None else [])


res = {}
xc, pr = run(auth("READ_LOG @ 1111111:10", "READ_LOG @ 1111111:10"), expect=[C1])
res["positive control: a cited, non-ancestor, pushed commit with its line: XC = [c1], --expect [c1] passes"] = xc == [C1] and pr == []
xc, pr = run(auth("READ_LOG @ 1111111:10"), expect=[])
res["09 control 1: an AUTH citing a commit missing from XC: STOP"] = pr == [f"a cited commit missing from XC: {C1}"]
xc, pr = run(auth("READ_LOG @ 1111111:10"), expect=[C1, C2])
res["09 control 2: an extra XC entry no rule cites: STOP"] = pr == [f"an XC entry no rule cites: {C2}"]
xc, pr = run(auth("READ_LOG @ 0000000:10"), expect=[])
res["a cited commit that is an ancestor of RH is not in XC"] = xc == [] and pr == []
xc, pr = run(auth("READ_LOG @ 1111111:10"), pushed=())
res["a cited commit on no pushed branch: STOP"] = xc == [] and pr == [f"a rule commit on no pushed branch: {C1}"]
xc, pr = run(auth("READ_LOG @ 1111111:10"), files={})
res["the cited file absent at the commit: STOP"] = xc == [] and pr == [f"the cited file is absent at {C1}: {LOG}"]
xc, pr = run(auth("READ_LOG @ 1111111:101"))
res["a cited line beyond the file's line count: STOP"] = xc == [] and pr == [f"a cited line outside the file at {C1}: line 101 of 100"]
xc, pr = run(auth("READ_LOG @ 3333333:10"))
res["an unknown commit: STOP"] = pr == ["a rule commit that does not resolve to exactly one commit: 3333333"]
xc, pr = run(auth("see READ_LOG line 10"))
res["a rule not in rp's form: STOP"] = pr == ["a rule not in rp's form: 'see READ_LOG line 10'"]
xc, pr = run(auth("RULE item 7 @ 2222222"), files={(C2, RULE): 50})
res["RULE item N @ c: needs RULE.md at c, no line check"] = xc == [C2] and pr == []
xc, pr = run(auth("READ_LOG @ 1111111:10"), expect=["A" * 40])
res["an expected entry not in 40 lowercase hex: STOP"] = pr == ["the expected XC is not a JSON list of full 40-hex commits"]
xc, pr = D.derive(auth("READ_LOG @ 4444444:10"), lambda c: None if c == "4444444" else SHORT.get(c), lambda f: False, lambda f: True,
                  lambda f, p: 100)
res["a03f5d93: an ambiguous abbreviation (no single commit): STOP"] = pr == ["a rule commit that does not resolve to exactly one commit: 4444444"]
xc, pr = D.derive(auth("READ_LOG @ 1111111:10"), lambda c: C2, lambda f: False, lambda f: True, lambda f, p: 100)
res["a03f5d93: a resolution that does not extend the abbreviation: STOP"] = pr == ["a rule commit that does not resolve to exactly one commit: 1111111"]
xc, pr = D.derive(auth("READ_LOG @ 1111111:10"), lambda c: "1" * 9, lambda f: False, lambda f: True, lambda f, p: 100)
res["a03f5d93 M8: a resolution that extends the abbreviation but is not 40 hex: STOP"] = \
    xc == [] and pr == ["a rule commit that does not resolve to exactly one commit: 1111111"]
res["a03f5d93: XC = null (expected not a list): STOP, not treated as []"] = D.compare([], None) == ["the expected XC is not a JSON list of full 40-hex commits"] \
    and D.compare([C1], None) == ["the expected XC is not a JSON list of full 40-hex commits"]
res["RX_RULE is rp's RX_RULE_ (by text)"] = f'RX_RULE_ = r"{D.RX_RULE}"' in (HERE.parent / "run_peraction311.py").read_text()
for k, v in res.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
print("ALL PASS" if all(res.values()) else "FAILED")
if not all(res.values()):
    raise SystemExit(1)
