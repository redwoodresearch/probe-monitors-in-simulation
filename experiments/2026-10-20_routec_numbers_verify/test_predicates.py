"""A positive and a negative fixture for every predicate op (synthetic values only), plus the closed-set and argument checks."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import predicates as P  # noqa: E402

FAILS = []


def check(n, ok):
    print(("PASS " if ok else "FAIL ") + n)
    if not ok:
        FAILS.append(n)


def ok(pred, vals):
    return P.check(pred, vals)[0]


plant = {"verdict": "detected", "detected": 6, "misses": {}}
plant5 = {"verdict": "not detected", "detected": 3, "misses": {"x": "not detected", "y": "not detected", "z": "unavailable"}}
CASES = [
    ("count_in_range", {"op": "count_in_range", "flatten": True, "lo": 45, "hi": 49, "min_frac": 0.6, "report": "free text"},
     [[45, 46], [47, 49], [63]], [[45, 63], [63, 63], [10]]),
    ("abs_diff_le", {"op": "abs_diff_le", "target": 0.2, "tol": 0.025}, [0.21], [0.24]),
    ("max_abs_lt", {"op": "max_abs_lt", "bound": 3.0}, [-2.9, 1.0], [3.0]),
    ("max_abs_le", {"op": "max_abs_le", "bound": 3.0}, [3.0, -1.0], [3.01]),
    ("diff_in_open_interval", {"op": "diff_in_open_interval", "expr": "z - T", "lo": 0, "hi": 0.01}, [2.004, 2.0], [2.02, 2.0]),
    ("eq", {"op": "eq", "value": 7}, [7], [8]),
    ("min_max_eq", {"op": "min_max_eq", "min": 2, "max": 9}, [2, 5, 9], [3, 5, 9]),
    ("abs_round_eq_each", {"op": "abs_round_eq_each", "dp": 1, "values": [2.2, 2.2], "sign": "-"}, [-2.24, -2.16], [-2.24, 2.16]),
    ("abs_round_eq", {"op": "abs_round_eq", "dp": 0, "value": 12, "sign": "-"}, [-12.4], [12.4]),
    ("all_round_pct_eq", {"op": "all_round_pct_eq", "dp": 0, "value": 14}, [0.141, 0.139], [0.141, 0.15]),
    ("verdicts_eq", {"op": "verdicts_eq", "levels": ["0.01", "0.05"], "values": ["detected", "not detected"]}, [plant, plant5], [plant, plant]),
    ("count_eq", {"op": "count_eq", "field": "detected", "value": 6}, [plant], [plant5]),
    ("detected_set_eq", {"op": "detected_set_eq", "value": ["a", "b", "c"]}, [plant5], [plant]),
    ("string_contains", {"op": "string_contains", "value": "abc"}, ["xxabcxx"], ["xxabxx"]),
    ("count_eq_all", {"op": "count_eq_all", "field": "detected", "value": 6}, [plant, plant], [plant, plant5]),
    ("count_lt_all", {"op": "count_lt_all", "field": "detected", "bound": 6}, [plant5, plant5], [plant5, plant]),
    ("verdicts_eq_all", {"op": "verdicts_eq_all", "value": "detected"}, [plant, plant], [plant, plant5]),
    ("slot_string_eq identity", {"op": "slot_string_eq", "transform": "identity"}, ["+1.2 (â0.3 to +2.7)", "+1.2 (â0.3 to +2.7)"], ["+1.2 (â0.3 to +2.7)", "+1.3 (â0.3 to +2.7)"]),
    ("slot_string_eq point_part_strip_percent", {"op": "slot_string_eq", "transform": "point_part_strip_percent"}, ["12.3% (10.0% to 15.0%)", "12.3"], ["12.3% (10.0% to 15.0%)", "12.4"]),
    ("slot_string_eq interval_part", {"op": "slot_string_eq", "transform": "interval_part"}, ["+1.2 (â0.3 to +2.7)", "â0.3 to +2.7"], ["+1.2 (â0.3 to +2.7)", "â0.3 to +2.8"]),
    ("slot_string_eq template", {"op": "slot_string_eq", "transform": "template 'P points (95% interval L to H)' from slot 'P (L to H)'"}, ["+3.3 (+0.4 to +7.3)", "+3.3 points (95% interval +0.4 to +7.3)"], ["+3.3 (+0.4 to +7.3)", "+3.3 points (95% interval +0.4 to +7.4)"]),
    ("round_eq", {"op": "round_eq", "dp": 2, "value": 0.21}, [0.2149], [0.2151]),
    ("diff_round_eq", {"op": "diff_round_eq", "expr": "z - T", "dp": 2, "value": 0.42, "sign": "+"}, [3.42, 3.0], [2.58, 3.0]),
    ("sum_eq", {"op": "sum_eq", "value": 1313}, [300, 313, 400, 300], [300, 313, 400, 301]),
    ("ratio_pct_round_eq", {"op": "ratio_pct_round_eq", "dp": 0, "value": 42, "num": 0, "den": 1}, [21, 50], [2127, 5000]),
    ("slot_component_eq", {"op": "slot_component_eq", "component": "interval_low", "transform": "identity", "value_string": "+1.2 (−0.3 to +2.7)"},
     ["+1.2 (−0.3 to +2.7)", "−0.3"], ["+1.2 (−0.3 to +2.7)", "+2.7"]),
]
for name, pred, pos, neg in CASES:
    check(f"{name}: positive", ok(pred, pos))
    check(f"{name}: negative", not ok(pred, neg))
check("count_in_range reports the count, n and the out-of-range values", P.check(CASES[0][1], CASES[0][2])[1] == {"count_in_range": 4, "n": 5, "outside": [63]})
check("the closed set has 25 ops", len(P.OPS) == 25)
# 09 op #25 all_detected_at_size (10-20): printed == expected == |size|; every node's field == of_field; level and size in the key path
ADS = {"op": "all_detected_at_size", "expected": 8, "field": "detected", "of_field": "cases", "level": "0.01", "size": -8,
       "qualitative_word": "reliably", "args": [{"role": "planted fall size", "text": "8", "value": 8}], "note": "x"}
KP = [["cells", "-8", "C", "0.01"]]
check("op 25: detected == cases at the keyed cell, printed == expected == |size| passes, with a 0/1 report per check",
      P.check(ADS, [[{"detected": 6, "cases": 6}], "8", KP]) == (True, {"field_eq_of_field": 1, "printed_eq_expected_eq_abs_size": 1,
                                                                    "level_and_size_in_key_path": 1}))
for nm, pr_, vals in (("detected < cases", ADS, [[{"detected": 5, "cases": 6}], "8", KP]),
                      ("printed != |size|", ADS, [[{"detected": 6, "cases": 6}], "6", KP]),
                      ("expected != |size|", ADS | {"expected": 6}, [[{"detected": 6, "cases": 6}], "8", KP]),
                      ("a key path at -5 with size -8", ADS, [[{"detected": 6, "cases": 6}], "8", [["cells", "-5", "C", "0.01"]]]),
                      ("a key path at another level", ADS, [[{"detected": 6, "cases": 6}], "8", [["cells", "-8", "C", "0.05"]]])):
    check(f"op 25 negative: {nm} fails", P.check(pr_, vals)[0] is False)
for nm, pr_, vals in (("a node without the field", ADS, [[{"cases": 6}], "8", KP]),
                      ("a missing predicate argument (level)", {k: v for k, v in ADS.items() if k != "level"}, [[{"detected": 6, "cases": 6}], "8", KP]),
                      ("no key path", ADS, [[], "8", []])):
    try:
        P.check(pr_, vals)
        r = False
    except P.PredicateStop:
        r = True
    check(f"op 25: {nm} STOPs (PredicateStop)", r)
check("op 23 (v15): component_string equal to the computed component passes; a different one raises",
      P.check({"op": "slot_component_eq", "component": "interval_low", "transform": "identity", "value_string": "+1.2 (−0.3 to +2.7)",
               "component_string": "−0.3"}, ["+1.2 (−0.3 to +2.7)", "−0.3"])[0])
try:
    P.check({"op": "slot_component_eq", "component": "interval_low", "transform": "identity", "value_string": "+1.2 (−0.3 to +2.7)",
             "component_string": "-0.3"}, ["+1.2 (−0.3 to +2.7)", "−0.3"])
    r = False
except P.PredicateError:
    r = True
check("op 23 (v15) negative: a component_string that is not byte-equal (ASCII hyphen) raises", r)
SL3 = "−1.3 (−2.0 to +0.4)"
check("09 (10-20) op 23: component_string is the component BEFORE the transform: an abs row with the signed component passes",
      P.check({"op": "slot_component_eq", "component": "point", "transform": "abs (sign stated in words)", "value_string": SL3,
               "component_string": "−1.3"}, [SL3, "1.3"])[0])
SLp = "+1.3 (+0.2 to +2.4)"
OP23 = {"op": "slot_component_eq", "component": "point", "transform": "identity"}
check("09 (10-20) op 23: a dropped '+' passes ('+1.3' printed '1.3'); the exact token still passes",
      P.check(OP23 | {"value_string": SLp, "component_string": "+1.3"}, [SLp, "1.3"])[0] and P.check(OP23 | {"value_string": SLp}, [SLp, "+1.3"])[0])
check("09 (10-20) op 23 negative: a dropped minus fails with identity ('−1.3' printed '1.3'); only the abs transform allows it",
      P.check(OP23 | {"value_string": SL3}, [SL3, "1.3"])[0] is False
      and P.check(OP23 | {"value_string": SL3, "transform": "abs (sign stated in words)"}, [SL3, "1.3"])[0])
check("scale (09, v14): round(|100 * x|, dp) == value passes with scale 100; a missing scale means 1",
      P.check({"op": "abs_round_eq", "dp": 1, "value": 2.3, "scale": 100}, [0.0231])[0] and P.check({"op": "abs_round_eq", "dp": 1, "value": 2.3}, [2.31])[0])
check("scale negative: a 100x mismatch fails (scale 100 on a value already in points)", not P.check({"op": "abs_round_eq", "dp": 1, "value": 2.3, "scale": 100},
                                                                                                   [2.31])[0])
check("scale on abs_round_eq_each applies to each value", P.check({"op": "abs_round_eq_each", "dp": 1, "values": [2.3, 1.2], "scale": 100,
                                                                    "scale_note": "fractions to points"}, [0.0231, 0.0118])[0])
for nm, sc in (("scale 10", 10), ("scale True", True), ("scale as a string", "100")):
    try:
        P.check({"op": "abs_round_eq", "dp": 1, "value": 2.3, "scale": sc}, [0.0231])
        r = False
    except P.PredicateError:
        r = True
    check(f"scale negative: {nm} raises (not in the closed set; the row STOPs)", r)
AW = {"op": "all_within", "target": 100, "tol": 0.5, "unit": "percent (record fractions x 100)"}
check("op 24 all_within: every value within tol of target passes", P.check(AW, [0.998, 1.0, 0.996])[0])
check("op 24 negative: one value outside tol fails", not P.check(AW, [0.998, 0.98, 1.0])[0])
for nm, pr_, vals in (("an empty key list (no values)", AW, []), ("an unknown unit", AW | {"unit": "points"}, [1.0])):
    try:
        P.check(pr_, vals)
        r = False
    except P.PredicateError:
        r = True
    check(f"op 24 negative: {nm} raises (the row STOPs, never vacuous)", r)
SL = "+1.2 (−0.3 to +2.7)"
check("op 23: each component of a P (L to H) slot, and of an A to B range; the abs transform drops the sign",
      [P.slot_component(SL, c, "identity") for c in ("point", "interval_low", "interval_high")] == ["+1.2", "−0.3", "+2.7"]
      and P.slot_component("3 to 9", "range_high", "identity") == "9" and P.slot_component(SL, "interval_low", "abs (sign stated in words)") == "0.3")
for nm, pr_ in (("a wrong component (the printed token is another component)", {"component": "point"}),):
    check(f"op 23 negative: {nm} fails", not P.check({"op": "slot_component_eq", "transform": "identity", "value_string": SL} | pr_, [SL, "−0.3"])[0])
for nm, pr_ in (("an unknown transform", {"component": "point", "transform": "negate", "value_string": SL}),
                ("an unknown component", {"component": "median", "transform": "identity", "value_string": SL}),
                ("a value_string that is not the verified slot string byte for byte", {"component": "point", "transform": "identity",
                                                                                        "value_string": SL.replace("−", "-")})):
    try:
        P.check({"op": "slot_component_eq"} | pr_, [SL, "+1.2"])
        r = False
    except P.PredicateError:
        r = True
    check(f"op 23 negative: {nm} raises (the row STOPs)", r)
check("ratio_pct_round_eq: 2127 / 5000 = 42.54 crosses .5 and rounds to 43 (fails 42); 85 / 200 = 42.5 ties to even 42 (the slot rule)",
      not ok({"op": "ratio_pct_round_eq", "dp": 0, "value": 42}, [2127, 5000]) and ok({"op": "ratio_pct_round_eq", "dp": 0, "value": 42}, [85, 200]))
check("ratio_pct_round_eq: num / den pick the numerator and denominator by index", ok({"op": "ratio_pct_round_eq", "dp": 0, "value": 42, "num": 1,
                                                                                         "den": 0}, [50, 21]))
for nm, vals, xa in (("a float count", [21.0, 50], {}), ("a zero denominator", [21, 0], {}), ("one value", [21], {}),
                     ("num equal to den", [21, 50], {"num": 0, "den": 0})):
    try:
        P.check({"op": "ratio_pct_round_eq", "dp": 0, "value": 42} | xa, vals)
        r = False
    except P.PredicateError:
        r = True
    check(f"ratio_pct_round_eq: {nm} raises PredicateError", r)
for nm, pred in (("an unknown op", {"op": "about_equal", "value": 1}), ("a missing argument", {"op": "abs_diff_le", "target": 1}),
                 ("an unexpected argument", {"op": "eq", "value": 1, "tol": 2}), ("a malformed argument", {"op": "max_abs_lt", "bound": "3"})):
    try:
        P.check(pred, [1.0])
        r = False
    except P.PredicateError:
        r = True
    check(f"{nm} raises PredicateError (the row STOPs)", r)

import hashlib  # noqa: E402

TXT = "Suppose a probe scores 62% on real runs and 60% on simulated ones.\n\nIf p = 90%, two raters would agree on 82% by chance."
LIT1, LIT2 = "62% on real runs and 60%", "90%"
HROWS = [{"row": 1, "anchor_sentence": "Suppose a probe scores 62% on real runs and 60% on simulated ones.", "text": LIT1},
         {"row": 2, "anchor_sentence": "If p = 90%, two raters would agree on 82% by chance.", "text": LIT2}]
EX = {"1": {"expressions": [["62 - 60", 2]], "percent_reading": "points", "text_sha256": hashlib.sha256(LIT1.encode()).hexdigest()},
      "2": {"expressions": [["(0.90) * (0.90) + (1 - (0.90)) * (1 - (0.90))", 0.82], ["0.82 > 0.5", True]],
            "percent_reading": "fraction", "text_sha256": hashlib.sha256(LIT2.encode()).hexdigest()}}


def blk(o):
    return 0 if o < TXT.index("\n\n") else 1


h = P.hypothetical_checks(HROWS, EX, TXT, {}, blk)
check("hypothetical: marker, exact arithmetic (points and fraction readings) and no shadowing all hold", all(all(v.values()) for v in h.values()))
check("hypothetical: 0.9*0.9 + 0.1*0.1 == 0.82 holds exactly (fractions, not binary floats)", P._holds(EX["2"]["expressions"][0][0], 0.82))
h = P.hypothetical_checks(HROWS, EX | {"1": EX["1"] | {"expressions": [["62 - 60", 3]]}}, TXT, {}, blk)
check("hypothetical (b): a relation that does not hold fails arith", h[1]["arith"] is False and h[2]["arith"] is True)
h = P.hypothetical_checks(HROWS, EX | {"1": EX["1"] | {"text_sha256": "0" * 64}}, TXT, {}, blk)
check("hypothetical (b): relations bound to another literal fail arith", h[1]["arith"] is False)
h = P.hypothetical_checks(HROWS, EX | {"1": EX["1"] | {"expressions": [["62 - 57", 5]]}}, TXT, {}, blk)
check("hypothetical (b): a literal number in no relation fails arith", h[1]["arith"] is False)
h = P.hypothetical_checks(HROWS, EX, TXT, {0: {60.0}}, blk)
check("hypothetical (c): a literal equal to a verified number in the same block fails no_shadow (the other block unaffected)",
      h[1]["no_shadow"] is False and h[2]["no_shadow"] is True)
h = P.hypothetical_checks([HROWS[0] | {"anchor_sentence": "A probe scores 62% on real runs and 60% on simulated ones."}], EX,
                          TXT.replace("Suppose a", "A"), {}, blk)
check("hypothetical (a): no marker of the closed list fails marker", h[1]["marker"] is False)
h = P.hypothetical_checks(HROWS, {"1": EX["1"] | {"percent_reading": "fraction"}, "2": EX["2"] | {"percent_reading": "points"}}, TXT, {}, blk)
check("09 pinned reading: each row fails (b) under the OTHER reading (points row read as fractions, fraction row as points)",
      h[1]["arith"] is False and h[2]["arith"] is False)
h = P.hypothetical_checks(HROWS, {"1": {k: v for k, v in EX["1"].items() if k != "percent_reading"}, "2": EX["2"] | {"percent_reading": "either"}}, TXT,
                          {}, blk)
check("09 pinned reading: no pin, or a value other than points / fraction, fails (b) (never 'either reading')",
      h[1]["arith"] is False and h[2]["arith"] is False)
for bad in ("__import__('os')", "len('ab')", "2 ** 3"):
    try:
        P._exact(bad)
        r = False
    except P.PredicateError:
        r = True
    check(f"hypothetical: a non-arithmetic relation ({bad[:6]}) raises", r)
T2 = "The verified share was 5% here. Suppose a probe caught 5% of the runs."
L2 = "5% of the runs"
EX2 = {"9": {"expressions": [["0.05 * 100", 5]], "percent_reading": "fraction", "text_sha256": hashlib.sha256(L2.encode()).hexdigest()}}
h = P.hypothetical_checks([{"row": 9, "anchor_sentence": "Suppose a probe caught 5% of the runs.", "text": L2}], EX2, T2, {0: {5.0}}, lambda o: 0)
check("R6: a fraction-read literal '5%' beside a verified printed '5%' in the same block fails no_shadow (compared as printed)",
      h[9]["no_shadow"] is False)
for nm, pred, vals in (("abs_round_eq_each", {"op": "abs_round_eq_each", "dp": 1, "values": []}, []),
                       ("all_round_pct_eq", {"op": "all_round_pct_eq", "dp": 0, "value": 14}, [])):
    try:
        P.check(pred, vals)
        r = False
    except P.PredicateError:
        r = True
    check(f"O5: {nm} on no values raises (never passes vacuously)", r)
print(f"{len(FAILS)} failed")
sys.exit(1 if FAILS else 0)
