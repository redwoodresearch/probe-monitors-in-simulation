"""09's hypothetical class, check (b) (10-20): each hypothetical constants row's arithmetic as a small committed expression list, derived
BY KEY from the row's own fields. A row with a `relation` field (writing's constants v4, row 21: "<formula in p> = a with p = P gives
a = A, and a > B") gets that relation, P substituted: [formula(P), A] and [A > B, true]. Any other row gets the relations of 09's list
(worked-example arithmetic, verbatim) that share a number (operand or result) with the row's literal `text`. Every relation must hold
in exact arithmetic (fractions of the decimal literals); each row pins ONE percent reading of its literal (09: row 21 "fraction", N% = N/100; the rows on 09's list "points", N% = N). No literal is printed: counts only.

  uv run --no-project --python 3.12 python -I tools/build_hypothetical_exprs.py <constants.jsonl> <out.json>
"""

import ast
import hashlib
import json
import operator as op
import re
import sys
from fractions import Fraction

RELATIONS = [["62 - 60", 2], ["62 - 57", 5], ["(2 + 5) / 2", 3.5], ["60 - 57", 3], ["3.5 - 3", 0.5]]  # 09 (10-20), "62−60=2, ..." verbatim
NUM = re.compile(r"\d+(?:\.\d+)?")
REL_RX = re.compile(r"^(?P<f>[p0-9.*+\-() ]+?) = a with p = (?P<p>\d+(?:\.\d+)?) gives a = (?P<a>\d+(?:\.\d+)?), and a > (?P<b>\d+(?:\.\d+)?)\b")
OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.USub: op.neg}
CMP = {ast.Gt: op.gt, ast.Lt: op.lt, ast.GtE: op.ge, ast.LtE: op.le}


def exact(e):
    def ev(n):
        if isinstance(n, ast.Constant) and type(n.value) in (int, float):
            return Fraction(repr(n.value))
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            return OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in OPS:
            return OPS[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Compare) and len(n.ops) == 1 and type(n.ops[0]) in CMP:
            return CMP[type(n.ops[0])](ev(n.left), ev(n.comparators[0]))
        raise SystemExit("STOP: a relation is not plain arithmetic")
    return ev(ast.parse(e, mode="eval").body)


def holds(e, v):
    r = exact(e)
    return r is v if isinstance(v, bool) else (not isinstance(r, bool) and r == Fraction(repr(v)))


def nums(s):
    return {float(t) for t in NUM.findall(s)}


def lit_nums(s, reading):
    """A literal's numbers under the row's ONE pinned percent reading (09, 10-20): "points" reads N% as N, "fraction" as N/100."""
    if reading not in ("points", "fraction"):
        raise SystemExit("STOP: a percent reading is not points or fraction")
    return [float(Fraction(t) / (100 if pc and reading == "fraction" else 1)) for t, pc in re.findall(r"(\d+(?:\.\d+)?)(%?)", s.replace(",", ""))]


def main(cpath, out):
    rows = [json.loads(x) for x in open(cpath, encoding="utf-8") if x.strip()]
    hyp = [r for r in rows if r.get("kind") == "hypothetical"]
    by_row, report = {}, []
    for r in hyp:
        reading = "fraction" if "relation" in r else "points"  # 09 (10-20): row 21 (its relation field) N/100; 09's list rows points N
        lit = lit_nums(r["text"], reading)
        if "relation" in r:
            m = REL_RX.match(r["relation"])
            if not m:
                raise SystemExit(f"STOP: row {r['row']}'s relation field is not in the expected form")
            f = m["f"].replace("p", f"({m['p']})")
            rel = [[f, float(m["a"])], [f"{m['a']} > {m['b']}", True]]
            how = "relation field"
        else:
            rel = [[e, v] for e, v in RELATIONS if (nums(e) | {float(v)}) & set(lit)]
            how = "09's list"
        if not all(holds(e, v) for e, v in rel):
            raise SystemExit(f"STOP: a relation of row {r['row']} does not hold exactly")
        in_rel = {x for e, v in rel for x in nums(e) | (set() if isinstance(v, bool) else {float(v)})}
        covered = set(lit) <= in_rel
        by_row[str(r["row"])] = {"expressions": rel, "from": how, "percent_reading": reading, "text_sha256": hashlib.sha256(r["text"].encode()).hexdigest()}
        report.append({"row": r["row"], "from": how, "percent_reading": reading, "n_expressions": len(rel), "n_literal_numbers": len(lit),
                       "all_literal_numbers_in_expressions": covered})
    src = {"path": cpath, "sha256": hashlib.sha256(open(cpath, "rb").read()).hexdigest()}
    json.dump({"what": "09's hypothetical check (b): per-row committed relations (a row's own relation field, else 09's list assigned by the "
                       "row's literal numbers); the container evaluates each in exact arithmetic and requires every number of the row's "
                       "literal to be an operand or result", "source": src, "relations_09": RELATIONS, "rows": by_row},
              open(out, "x", encoding="utf-8"), indent=1)
    print(json.dumps({"hypothetical_rows": len(hyp), "per_row": report}))


if __name__ == "__main__":
    main(*sys.argv[1:3])
