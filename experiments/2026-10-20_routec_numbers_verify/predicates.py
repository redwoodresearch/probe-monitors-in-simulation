"""The closed set of provenance predicate ops (24; writing's provenance v4 through v13, 09's rulings 10-20). Pure; container-only on real data.

check(pred, values) -> (ok: bool, report: dict of counts / layer indices / booleans, never a result value). values: the record values the
row's key paths read, in key-path order (numbers, strings or nodes). An unknown op, a missing or extra argument, or an argument of the
wrong type raises PredicateError (the row STOPs).
"""

import math

MARKERS = ("Suppose", "would", "for example", "e.g.", "imagine")  # 09 (10-20): the hypothetical class's closed marker list


class PredicateError(Exception):
    pass


class PredicateStop(PredicateError):
    """Fail closed with a STOP of the pass, not only a failed row (09 op #25: a missing key, field or argument)."""


def _num(x, what):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise PredicateError(f"{what} is not a finite number")
    return x


def _flat(vs):
    out = []
    for v in vs:
        out += _flat(v) if isinstance(v, list) else [v]
    return out


def _round_half_even(x, dp):
    return float(f"{x:.{dp}f}")


def _verdict(n):
    v = n.get("verdict") if isinstance(n, dict) else n
    if not isinstance(v, str):
        raise PredicateError("a verdict is not a string")
    return v


def _field(n, f):
    if not isinstance(n, dict) or f not in n:
        raise PredicateError("a node lacks the predicate's field")
    return n[f]


# op -> (required args, optional args, function(args, values) -> (ok, report))
def _count_in_range(a, vs):
    """count of the values in [lo, hi] (values flattened when flatten) >= ceil(min_frac * n); reports the count, n and every value
    outside the range (layer indices)."""
    xs = [_num(x, "a value") for x in (_flat(vs) if a["flatten"] else vs)]
    inside = [x for x in xs if a["lo"] <= x <= a["hi"]]
    need = math.ceil(a["min_frac"] * len(xs) - 1e-12)
    return len(xs) > 0 and len(inside) >= need, {"count_in_range": len(inside), "n": len(xs), "outside": sorted(x for x in xs if not a["lo"] <= x <= a["hi"])}


def _abs_diff_le(a, vs):
    """|x - target| <= tol for the single value x."""
    (x,) = vs
    return abs(_num(x, "the value") - _num(a["target"], "target")) <= _num(a["tol"], "tol"), {}


def _max_abs_lt(a, vs):
    """max |x| over the values < bound."""
    return max(abs(_num(x, "a value")) for x in _flat(vs)) < _num(a["bound"], "bound"), {}


def _max_abs_le(a, vs):
    """max |x| over the values <= bound (inclusive; 09, 10-20: 'stay within')."""
    return max(abs(_num(x, "a value")) for x in _flat(vs)) <= _num(a["bound"], "bound"), {}




def _diff_in_open_interval(a, vs):
    """lo < z - T < hi for the two values (z, T); expr must be 'z - T'."""
    if a["expr"] != "z - T" or len(vs) != 2:
        raise PredicateError("diff_in_open_interval takes expr 'z - T' and two values")
    d = _num(vs[0], "z") - _num(vs[1], "T")
    return a["lo"] < d < a["hi"], {}


def _eq(a, vs):
    """the single value equals value (numbers exactly; strings byte-equal)."""
    (x,) = vs
    return x == a["value"] and type(x) is not bool, {}


def _min_max_eq(a, vs):
    """min and max over the values equal min and max."""
    xs = [_num(x, "a value") for x in _flat(vs)]
    return min(xs) == a["min"] and max(xs) == a["max"], {}


def _sign_ok(x, s):
    return s is None or (s == "-" and x < 0) or (s == "+" and x > 0)


SCALES = (1, 100)  # writing's provenance v14: the record value x scale before rounding (closed set; 09)


def _scale(a):
    sc = a.get("scale", 1)
    if isinstance(sc, bool) or sc not in SCALES:
        raise PredicateError("unknown scale")
    return sc


def _abs_round_eq_each(a, vs):
    """for each value i: round(|scale * x_i|, dp) == values[i], and its sign is `sign` (scale: optional, from SCALES, default 1)."""
    sc = _scale(a)
    xs = [_num(x, "a value") for x in _flat(vs)]
    if not xs or len(xs) != len(a["values"]):
        raise PredicateError("abs_round_eq_each: values and key paths differ in length")
    return all(_round_half_even(abs(sc * x), a["dp"]) == v and _sign_ok(x, a.get("sign")) for x, v in zip(xs, a["values"], strict=True)), {}


def _abs_round_eq(a, vs):
    """round(|scale * x|, dp) == value for the single value x (sign checked when given; scale: optional, from SCALES, default 1)."""
    sc = _scale(a)
    (x,) = vs
    x = _num(x, "the value")
    return _round_half_even(abs(sc * x), a["dp"]) == a["value"] and _sign_ok(x, a.get("sign")), {}


def _all_round_pct_eq(a, vs):
    """every value (at least one), as a percentage rounded to dp, equals value."""
    if not _flat(vs):
        raise PredicateError("all_round_pct_eq: no values")
    return all(_round_half_even(100 * _num(x, "a value"), a["dp"]) == a["value"] for x in _flat(vs)), {}


def _verdicts_eq(a, vs):
    """the verdict at each key path (one per level) equals values[i]."""
    if len(vs) != len(a["values"]) or len(vs) != len(a["levels"]):
        raise PredicateError("verdicts_eq: levels, values and key paths differ in length")
    return all(_verdict(n) == v for n, v in zip(vs, a["values"], strict=True)), {}


def _count_eq(a, vs):
    """the node's field equals value (one key path)."""
    (n,) = vs
    return _field(n, a["field"]) == a["value"], {}


def _detected_set_eq(a, vs):
    """the node's detected count equals len(value), and no name in value is among the node's misses."""
    (n,) = vs
    misses = _field(n, "misses")
    if not isinstance(misses, dict):
        raise PredicateError("misses is not a map")
    return _field(n, "detected") == len(a["value"]) and not set(a["value"]) & set(misses), {}


def _string_contains(a, vs):
    """the single string value contains value."""
    (x,) = vs
    if not isinstance(x, str):
        raise PredicateError("string_contains on a non-string")
    return a["value"] in x, {}


def _count_eq_all(a, vs):
    """every node's field equals value."""
    return len(vs) > 0 and all(_field(n, a["field"]) == a["value"] for n in vs), {}


def _count_lt_all(a, vs):
    """every node's field is strictly below bound."""
    return len(vs) > 0 and all(_num(_field(n, a["field"]), "the field") < _num(a["bound"], "bound") for n in vs), {}


def _verdicts_eq_all(a, vs):
    """every node's verdict equals value."""
    return len(vs) > 0 and all(_verdict(n) == a["value"] for n in vs), {}




MINUS = "−"


def _point_part(s):
    return s.split(" (", 1)[0]


def _interval_part(s):
    m = __import__("re").search(r"\(([^()]*)\)\s*$", s)
    if not m:
        raise PredicateError("the slot string has no interval part")
    return m.group(1)


def _template(t, s):
    """'P points (95% interval L to H)'-style templates: P, L and H are the slot's point and interval ends, verbatim."""
    m = __import__("re").fullmatch(r"(\S+) \((\S+) to (\S+)\)", s)
    if not m:
        raise PredicateError("the slot string is not 'P (L to H)'")
    return t.replace("P", m.group(1), 1).replace("L", m.group(2), 1).replace("H", m.group(3), 1)


TRANSFORMS = {"identity": lambda s: s, "integer with thousands comma": lambda s: s, "point_part": _point_part,
              "point_part_strip_percent": lambda s: _point_part(s).replace("%", ""), "interval_part": _interval_part}


def _slot_string_eq(a, vs):
    """printed == transform(the verified slot string). transform: identity | integer with thousands comma (the slot string itself) |
    point_part | point_part_strip_percent | interval_part | template '<...P...L...H...>'. values: [slot string, printed]."""
    s, printed = vs
    t = a["transform"]
    if t.startswith("template "):
        m = __import__("re").match(r"template '([^']*)'", t)  # writing's form: template '<P ... L ... H>' from slot 'P (L to H)'
        if not m:
            raise PredicateError("a template transform is not template '<...>'")
        want = _template(m.group(1), s)
    elif t in TRANSFORMS:
        want = TRANSFORMS[t](s)
    else:
        raise PredicateError("unknown slot_string_eq transform")
    return printed == want, {}


SLOT_COMPONENTS = ("point", "interval_low", "interval_high", "range_low", "range_high")
COMPONENT_TRANSFORMS = ("identity", "abs (sign stated in words)")


def slot_component(s, component, transform):
    """One number of a verified slot string, by its structural place (writing's provenance v12, 09 op #23 pending): point = the one
    number before " (" (or the slot's only number); interval_low / interval_high = the two numbers of "(L to H)"; range_low / range_high =
    the two numbers of a parenthesis-free "A to B". transform (closed): identity (the token as the slot prints it, sign included) |
    "abs (sign stated in words)" (the token without its sign). Anything that does not parse to exactly one number raises."""
    import re

    num = __import__("verify_core").NUM_RX
    if component not in SLOT_COMPONENTS:
        raise PredicateError("slot_component_eq: unknown component")
    if transform not in COMPONENT_TRANSFORMS:
        raise PredicateError("slot_component_eq: unknown transform")
    m = re.fullmatch(r"\s*([^()]*?)\s*\(([^()]*)\)\s*", s)
    if component == "point":
        part = m.group(1) if m else s
    elif component.startswith("interval_"):
        if not m or len(m.group(2).split(" to ")) != 2:
            raise PredicateError("slot_component_eq: the slot has no (L to H) interval")
        part = m.group(2).split(" to ")[0 if component.endswith("low") else 1]
    else:
        if m or len(s.split(" to ")) != 2:
            raise PredicateError("slot_component_eq: the slot is not a parenthesis-free A to B range")
        part = s.split(" to ")[0 if component.endswith("low") else 1]
    toks = num.findall(part)
    if len(toks) != 1:
        raise PredicateError("slot_component_eq: the component is not exactly one number")
    t = toks[0]
    return t.lstrip("+-\u2212") if transform == COMPONENT_TRANSFORMS[1] else t


def _slot_component_eq(a, vs):
    """09 op #23 (pending): value_string must be the verified slot string byte for byte, and the printed text must be exactly the one token
    transform(component(slot)). values: [the verified slot string, the printed text]. A wrong value_string, component or transform fails."""
    s, printed = vs
    if not isinstance(a["value_string"], str) or a["value_string"] != s:
        raise PredicateError("slot_component_eq: value_string is not the verified slot string")
    want = slot_component(s, a["component"], a["transform"])
    # writing's v15 (09, 10-20): optional, byte-equal to the computed component BEFORE the transform (identity)
    if "component_string" in a and a["component_string"] != slot_component(s, a["component"], COMPONENT_TRANSFORMS[0]):
        raise PredicateError("slot_component_eq: component_string is not the slot's component")
    toks = __import__("verify_core").NUM_RX.findall(printed)
    return len(toks) == 1 and component_printed(want, toks[0]), {}


def component_printed(want, t):
    """09 (10-20): the printed token is the component, or the component without a leading "+" the post drops (as slot_string_eq allows);
    never a dropped minus."""
    return t == want or (want.startswith("+") and t == want[1:])


def _all_detected_at_size(a, vs):
    """09 op #25 (10-20; row 81, RULE item 47 scope). values: [[the node at each key path], the printed text, [the key paths]].
    Every node holds field and of_field (else STOP) and field == of_field (exact); the printed text is exactly one number, equal to
    expected and to |size|; level and size (as written, e.g. "-8") are each one of every key path's keys, verbatim, so the printed size
    is tied to the cell actually checked. The qualitative word is not machine-checked (the caller lists it). -> (ok, 0/1 per check)."""
    nodes, printed, paths = vs
    if not nodes or len(nodes) != len(paths) or not all(isinstance(q, list) for q in paths):
        raise PredicateStop("all_detected_at_size: a listed key is missing")
    if any(not isinstance(n, dict) or a["field"] not in n or a["of_field"] not in n for n in nodes):
        raise PredicateStop("all_detected_at_size: a node lacks field or of_field")
    all_eq = all(_num(n[a["field"]], "field") == _num(n[a["of_field"]], "of_field") for n in nodes)
    toks = __import__("verify_core").NUM_RX.findall(printed)
    size = _num(a["size"], "size")
    printed_ok = len(toks) == 1 and float(toks[0].lstrip("+-−")) == _num(a["expected"], "expected") == abs(size)
    keys_ok = all(str(a["level"]) in [str(k) for k in q] and str(a["size"]) in [str(k) for k in q] for q in paths)
    return all_eq and printed_ok and keys_ok, {"field_eq_of_field": int(all_eq), "printed_eq_expected_eq_abs_size": int(printed_ok),
                                               "level_and_size_in_key_path": int(keys_ok)}


ALL_WITHIN_UNITS = {"percent (record fractions x 100)": 100.0}  # closed (09: x100 only): the record value is scaled by this before the test


def _all_within(a, vs):
    """09 op #24 (pending; writing's provenance v13 row 249): EVERY listed value, scaled by its unit (closed list ALL_WITHIN_UNITS), lies
    within tol of target: |scale * x - target| <= tol. No value at all, an unknown unit or a negative tol fails the row."""
    xs = [_num(x, "a value") for x in _flat(vs)]
    if not xs:
        raise PredicateError("all_within: no values")
    if a["unit"] not in ALL_WITHIN_UNITS:
        raise PredicateError("all_within: unknown unit")
    tol, target = _num(a["tol"], "tol"), _num(a["target"], "target")
    if tol < 0:
        raise PredicateError("all_within: a negative tol")
    sc = ALL_WITHIN_UNITS[a["unit"]]
    return all(abs(sc * x - target) <= tol for x in xs), {"n": len(xs), "n_outside": sum(abs(sc * x - target) > tol for x in xs)}


def _round_eq(a, vs):
    """round(x, dp) == value for the single value x (half-even on the binary float, as f-strings round)."""
    (x,) = vs
    return _round_half_even(_num(x, "the value"), a["dp"]) == a["value"], {}


def _diff_round_eq(a, vs):
    """round(|z - T|, dp) == value and the sign of z - T is `sign`; expr must be 'z - T'; values: (z, T)."""
    if a["expr"] != "z - T" or len(vs) != 2:
        raise PredicateError("diff_round_eq takes expr 'z - T' and two values")
    d = _num(vs[0], "z") - _num(vs[1], "T")
    return _round_half_even(abs(d), a["dp"]) == abs(a["value"]) and _sign_ok(d, a.get("sign")), {}


def _sum_eq(a, vs):
    """the sum of the values equals value (integers exactly)."""
    xs = [_num(x, "a value") for x in _flat(vs)]
    return sum(xs) == a["value"], {}


def _ratio_pct_round_eq(a, vs):
    """round(100 * numerator / denominator, dp) == value; values: the row's two key paths' values, and num / den (optional; default 0 / 1)
    name which of them is the numerator and which the denominator (distinct indices 0 and 1). Both must be counts (non-negative
    integers), the denominator positive (a zero denominator fails the row). Rounding: the slot rounding rule (verify_core.slots_rows),
    an f-string of the binary float, i.e. round half to even on the binary value (an exact x.5 goes to the even neighbour: 42.5 -> 42;
    42.54 -> 43). Writing's provenance v9, 09 op #22, 10-20."""
    if len(vs) != 2:
        raise PredicateError("ratio_pct_round_eq takes two values")
    i, j = a.get("num", 0), a.get("den", 1)
    if type(i) is not int or type(j) is not int or sorted((i, j)) != [0, 1]:
        raise PredicateError("ratio_pct_round_eq: num and den must be the distinct indices 0 and 1")
    n, d = _num(vs[i], "the numerator"), _num(vs[j], "the denominator")
    if type(n) is not int or type(d) is not int or n < 0 or d <= 0:
        raise PredicateError("ratio_pct_round_eq: the numerator and denominator must be counts, the denominator positive")
    return _round_half_even(100 * n / d, a["dp"]) == a["value"], {}


OPS = {
    "count_in_range": ({"flatten", "lo", "hi", "min_frac"}, {"report"}, _count_in_range),
    "abs_diff_le": ({"target", "tol"}, set(), _abs_diff_le),
    "max_abs_lt": ({"bound"}, set(), _max_abs_lt),
    "max_abs_le": ({"bound"}, set(), _max_abs_le),
    "diff_in_open_interval": ({"expr", "lo", "hi"}, set(), _diff_in_open_interval),
    "eq": ({"value"}, set(), _eq),
    "min_max_eq": ({"min", "max"}, set(), _min_max_eq),
    "abs_round_eq_each": ({"dp", "values"}, {"sign", "scale"}, _abs_round_eq_each),
    "abs_round_eq": ({"dp", "value"}, {"sign", "scale"}, _abs_round_eq),
    "all_round_pct_eq": ({"dp", "value"}, set(), _all_round_pct_eq),
    "verdicts_eq": ({"levels", "values"}, set(), _verdicts_eq),
    "count_eq": ({"field", "value"}, set(), _count_eq),
    "detected_set_eq": ({"value"}, set(), _detected_set_eq),
    "string_contains": ({"value"}, set(), _string_contains),
    "count_eq_all": ({"field", "value"}, set(), _count_eq_all),
    "count_lt_all": ({"field", "bound"}, set(), _count_lt_all),
    "verdicts_eq_all": ({"value"}, set(), _verdicts_eq_all),
    "slot_string_eq": ({"transform"}, set(), _slot_string_eq),
    "slot_component_eq": ({"component", "transform", "value_string"}, {"component_string"}, _slot_component_eq),
    "all_within": ({"target", "tol", "unit"}, set(), _all_within),
    "all_detected_at_size": ({"expected", "field", "of_field", "level", "size", "qualitative_word"}, set(), _all_detected_at_size),
    "round_eq": ({"dp", "value"}, set(), _round_eq),
    "diff_round_eq": ({"expr", "dp", "value"}, {"sign"}, _diff_round_eq),
    "sum_eq": ({"value"}, set(), _sum_eq),
    "ratio_pct_round_eq": ({"dp", "value"}, {"num", "den"}, _ratio_pct_round_eq),
}
STOP_ON_MISSING = {"all_detected_at_size"}  # 09 op #25: a missing argument STOPs the pass (fail closed), not only the row


def check(pred, values):
    if not isinstance(pred, dict) or pred.get("op") not in OPS:
        raise PredicateError("unknown predicate op")
    req, opt, fn = OPS[pred["op"]]
    args = {k: v for k, v in pred.items() if k not in ("op", "note", "args", "scale_note")}  # "args": printed-token declarations (verify_post), not inputs
    if not req <= set(args) or set(args) - req - opt:
        raise (PredicateStop if pred["op"] in STOP_ON_MISSING else PredicateError)(f"{pred['op']}: missing or unexpected arguments")
    try:
        return fn(args, values)
    except (ValueError, TypeError) as e:
        raise PredicateError(f"{pred['op']}: {type(e).__name__}") from None


def _exact(e):
    """A relation in exact arithmetic: decimal literals as fractions; + - * / and unary minus; one comparison (> < >= <=) -> bool."""
    import ast
    import operator as op
    from fractions import Fraction

    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.USub: op.neg}
    cmp = {ast.Gt: op.gt, ast.Lt: op.lt, ast.GtE: op.ge, ast.LtE: op.le}

    def ev(n):
        if isinstance(n, ast.Constant) and type(n.value) in (int, float):
            return Fraction(repr(n.value))
        if isinstance(n, ast.BinOp) and type(n.op) in ops:
            return ops[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in ops:
            return ops[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Compare) and len(n.ops) == 1 and type(n.ops[0]) in cmp:
            return cmp[type(n.ops[0])](ev(n.left), ev(n.comparators[0]))
        raise PredicateError("a hypothetical relation is not plain arithmetic")

    return ev(ast.parse(e, mode="eval").body)


def _holds(e, v):
    from fractions import Fraction

    r = _exact(e)
    if isinstance(v, bool):
        return r is v
    return type(v) in (int, float) and not isinstance(r, bool) and r == Fraction(repr(v))


def _nums(s):
    import re

    return {float(t) for t in re.findall(r"\d+(?:\.\d+)?", s.replace(",", ""))}


def _lit_numbers(s, reading):
    """A literal's numbers under the row's ONE pinned percent reading (09, 10-20): "points" reads N% as N, "fraction" as N/100.
    Any other value (or none) -> None: the row fails check (b). The container never accepts 'either reading'."""
    import re
    from fractions import Fraction

    if reading not in ("points", "fraction"):
        return None
    return [float(Fraction(t) / (100 if pc and reading == "fraction" else 1)) for t, pc in re.findall(r"(\d+(?:\.\d+)?)(%?)", s.replace(",", ""))]


def _lit_both(s):
    """A literal's numbers under both readings: N as printed, and N/100 for "N%" (R6)."""
    import re
    from fractions import Fraction

    out = set()
    for t, pc in re.findall(r"(\d+(?:\.\d+)?)(%?)", s.replace(",", "")):
        out.add(float(Fraction(t)))
        if pc:
            out.add(float(Fraction(t) / 100))
    return out


def hypothetical_checks(rows, exprs_by_row, text, verified_by_block, block_of):
    """09's hypothetical class (10-20), three container checks per writing's hypothetical constants row (row, anchor_sentence, text):
      marker     the anchor sentence occurs once in the post, holds the row's literal, and contains a marker of the closed list MARKERS;
      arith      the row's committed relations (config, bound to the literal by its sha256) each hold in exact arithmetic, and every
                 number of the literal, read under the row's pinned percent_reading ("points": N% = N; "fraction": N% = N/100; no pin
                 -> False), is an operand or result of one of them (no relation -> False);
      no_shadow  no number of the literal, AS PRINTED (N) or as a fraction (N/100 for "N%"), equals a verified result number printed in
                 the same block (R6: whatever the pinned reading).
    exprs_by_row: row (str) -> {"expressions": [[expr, value], ...], "text_sha256"}; verified_by_block: block -> {float};
    block_of(offset) -> the block of an offset. -> {row: {marker, arith, no_shadow}} (booleans only)."""
    import hashlib

    out = {}
    for r in rows:
        s, lit = r["anchor_sentence"], r["text"]
        once = text.count(s) == 1 and lit in s
        ent = exprs_by_row.get(str(r["row"])) or {}
        ex = ent.get("expressions") or []
        bound = ent.get("text_sha256") == hashlib.sha256(lit.encode()).hexdigest()
        try:
            holds = bool(ex) and all(_holds(e, v) for e, v in ex)
        except (PredicateError, ZeroDivisionError, SyntaxError):
            holds = False
        ln = _lit_numbers(lit, ent.get("percent_reading"))
        in_ex = {x for e, v in ex for x in _nums(e) | (set() if isinstance(v, bool) else {float(v)})}
        ver = set(verified_by_block.get(block_of(text.index(s)), ())) if once else set()
        out[r["row"]] = {"marker": once and any(m in s for m in MARKERS), "arith": bound and holds and ln is not None and set(ln) <= in_ex,
                         "no_shadow": once and not (_lit_both(lit) & ver)}
    return out
