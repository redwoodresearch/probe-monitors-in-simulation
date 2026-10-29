"""Independent cross-check of 09's report_507_21j markdown (v1, v2 and v3 layouts) against the FINAL record.

v1/v2 are checked against Revision 6r (sha256 7dceb8bb...); v3 against Revision 6s (sha256 d0933874...), whose
append21j_2b block (mixtures, X_err_strat/U_strat, between-origin D with MDD_0 and power for the cross, with and
without strategy presentations) and append37_item1_POST_HOC block (Delta_S_new_std, D_std) v3 prints.

Checks (a) every printed number against the source key its label claims (1-decimal rounding, four rounding
conventions tested), with per-line accounting of every numeric token; (b) every Append 21j verdict and every
old-rule (pool-if-match, TOST ±5) line, recomputed from the stored ci90/ci95 with rules written from the RULE
text (Append 21j ACCEPTED §2 and §3; Append 18 as kept by 21j); (c) the source file's sha256.

Two verdicts are written: "overall" (numbers, verdicts, old-rule lines, record labels, structure, sha256,
token accounting) and "overall_presentation_and_completeness" (GFM rendering, Append 18 prominence, and rule
items the report leaves pending or omits). Codex round 1 (v1/v2 modes, 7 blocking) and the v3 round (9 blocking)
are addressed in this version. Numbers are compared at the printed precision, with a per-field minimum.

Blinding: stdout and the JSON output carry ONLY counts, key paths, booleans and allowlisted labels; a final
scrubber masks any decimal token (other than the alpha strings 0.01/0.05) and any float, and counts them.

Usage:
  python -I check_report_507_21j.py --format {v1,v2,v3} --source SRC.json --report REPORT.md --out OUT.json \
      [--expected-sha-prefix 7dceb8bb] [--meta key=value ...]
"""

import argparse
import hashlib
import json
import math
import re
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal

MARGIN = 5.0  # 21j §2: X = 5 percentage points
MIN_JOINT_DRAWS = 9500  # 21j §2: fewer than 9,500 jointly finite draws -> decision unavailable
REGISTERED_DRAWS = 10000  # 21j §2: "out of 10,000"

READOUT = {"reply": "R", "command": "C"}
FIXED_ROW_ORIGIN = [("mixture S_old ∪ S_new", "combined"), ("S_old", "S_old"), ("S_new", "S_new")]
CONVENTIONS = ["py_format_binary_half_even", "decimal_repr_half_up", "decimal_repr_half_even",
               "naive_floor_half_up_toward_pos_inf"]


# ---------- helpers (never print values) ----------

def get(d, path):
    """Follow a list of keys; return (found, value)."""
    cur = d
    for k in path:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        elif isinstance(cur, list) and isinstance(k, int) and 0 <= k < len(cur):
            cur = cur[k]
        else:
            return False, None
    return True, cur


def kp(path):
    return "/".join(str(k) for k in path)


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def norm_printed(s):
    s = s.strip().replace("−", "-")
    if s.startswith("+"):
        s = s[1:]
    return s


def round_strings(v, dp=1):
    q = Decimal(1).scaleb(-dp)
    out = {
        "py_format_binary_half_even": format(v, f".{dp}f"),
        "decimal_repr_half_up": str(Decimal(repr(v)).quantize(q, rounding=ROUND_HALF_UP)),
        "decimal_repr_half_even": str(Decimal(repr(v)).quantize(q, rounding=ROUND_HALF_EVEN)),
        "naive_floor_half_up_toward_pos_inf": format(math.floor(v * 10 ** dp + 0.5) / 10 ** dp, f".{dp}f"),
    }
    return out


def zero_norm(s):
    return s[1:] if s.startswith("-") and set(s[1:]) <= set("0.") else s


def safe_id(x):
    """Report-derived names used in ids and key paths must look like identifiers (never a number)."""
    x = x.replace("\\|", "|").strip()
    return x if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_|+]*", x) else "UNRECOGNIZED"


def line_of(loc):
    parts = loc.split()
    return int(parts[1]) if len(parts) > 1 and parts[0] == "line" and parts[1].isdigit() else None


def printed_dp(p):
    return len(p.split(".", 1)[1]) if "." in p else 0


class Checker:
    def __init__(self, src):
        self.src = src
        self.items = []
        self.verdicts = []
        self.old_rule = []
        self.labels = []
        self.structure = []
        self.observations = []
        self.presentation = []
        self.completeness = []
        self.bools = []
        self.record_checks = []

    # (a) numbers
    def num(self, item_id, loc, path, printed, kind="dec1", alts=(), min_dp=1):
        found, v = get(self.src, path)
        p = norm_printed(printed)
        row = {"item_id": item_id, "report_location": loc, "key_path": kp(path), "kind": kind,
               "key_found": found, "min_dp": min_dp if kind == "dec1" else 0}
        if p in ("n/a", "—", "-", "NA", "nan"):
            row["printed_na"] = True
            if found:
                row["match"] = v is None or (isinstance(v, float) and not math.isfinite(v))
            else:
                # absent key: accepted only for family-summary fields of a withheld pooled block
                # (X_err null plus a stored withholding note); never for per-probe or count fields
                row["match"] = self._withheld(path)
                row["matched_as_withheld_block"] = row["match"]
            row["conventions_matched"] = []
            row["convention_dependent"] = False
        elif kind == "int":
            row["printed_na"] = False
            try:
                pi = int(p.replace(",", ""))
            except ValueError:
                pi = None
            row["match"] = found and isinstance(v, int) and not isinstance(v, bool) and pi == v
            row["conventions_matched"] = []
            row["convention_dependent"] = False
        else:
            row["printed_na"] = False
            self._dec_compare(row, v if found else None, p)
            row["precision_ok"] = row.get("printed_dp", 0) >= min_dp
            row["match"] = row["match"] and row["precision_ok"]
        if not row["match"] and alts:
            row["alternative_keys_matching"] = [kp(a) for a in alts if self._alt_match(a, p, kind)]
        self.items.append(row)
        return row["match"]

    @staticmethod
    def _dec_compare(row, v, p):
        if not finite(v):
            row.update(match=False, conventions_matched=[], convention_dependent=False)
            return
        row["printed_dp"] = printed_dp(p)
        rs = round_strings(float(v), row["printed_dp"])
        strict = [c for c in CONVENTIONS if rs[c] == p]
        loose = [c for c in CONVENTIONS if zero_norm(rs[c]) == zero_norm(p)]
        row["conventions_matched"] = strict
        row["signed_zero_only_match"] = bool(loose) and not strict
        row["match"] = bool(loose)
        row["convention_dependent"] = bool(loose) and len(loose) < len(CONVENTIONS)

    def derived(self, item_id, loc, input_paths, value, printed, min_dp=1):
        """A printed number that the report derives from stored keys (e.g. an origin share in %)."""
        row = {"item_id": item_id, "report_location": loc, "kind": "derived_dec1",
               "key_path": "derived from: " + " ; ".join(kp(x) for x in input_paths),
               "key_found": value is not None, "printed_na": False, "min_dp": min_dp}
        self._dec_compare(row, value, norm_printed(printed))
        row["precision_ok"] = row.get("printed_dp", 0) >= min_dp
        row["match"] = row["match"] and row["precision_ok"]
        self.items.append(row)
        return row["match"]

    WITHHOLDABLE = ("U_approx_upper_95", "bias_mean", "bias_ci95")

    def _withheld(self, path):
        if "pooled" not in path:
            return False
        i = path.index("pooled")
        field = path[i + 1] if len(path) > i + 1 else None
        if field not in self.WITHHOLDABLE or (field != "bias_ci95" and len(path) != i + 2) or \
                (field == "bias_ci95" and (len(path) != i + 3 or path[-1] not in (0, 1))):
            return False
        found, d = get(self.src, path[: i + 1])
        return (found and isinstance(d, dict) and "X_err" in d and d["X_err"] is None
                and isinstance(d.get("note"), str))

    def _alt_match(self, path, p, kind):
        found, v = get(self.src, path)
        if not found or not finite(v):
            return False
        if kind == "int":
            try:
                return int(p.replace(",", "")) == v
            except ValueError:
                return False
        return any(zero_norm(s) == zero_norm(p) for s in round_strings(float(v), printed_dp(p)).values())

    def struct(self, check_id, loc, ok, detail_keys=(), consts=0):
        """consts = how many numeric tokens on that report line this check validates (rule constants,
        interval levels in headers); used by the per-line token accounting."""
        self.structure.append({"check_id": check_id, "report_location": loc, "match": bool(ok),
                               "key_paths": list(detail_keys), "consts": consts})

    def complete(self, check_id, loc, ok, why):
        """Completeness/presentation requirement from the rule text (not a number or verdict check)."""
        self.completeness.append({"check_id": check_id, "report_location": loc, "match": bool(ok), "why": why})


# ---------- 21j rules, written from RULE.md (Append 21j ACCEPTED §2, §3) ----------

def interval_ok(ci):
    return isinstance(ci, list) and len(ci) == 2 and all(finite(x) for x in ci) and ci[0] <= ci[1]


def verdict_21j(available, ci90, ci95):
    """§2: equivalent iff L90 > -5 and U90 < +5; materially different iff L95 > +5 or U95 < -5;
    otherwise inconclusive at this n. Strict inequalities. Unavailable -> decision unavailable."""
    if not available:
        return "decision unavailable"
    l90, u90 = ci90
    l95, u95 = ci95
    if l90 > -MARGIN and u90 < MARGIN:
        return "equivalent"
    if l95 > MARGIN or u95 < -MARGIN:
        return "materially different"
    return "inconclusive at this n"


def tost_pass(available, ci90):
    """§3: a member/pair passes iff -5 < L90 and U90 < +5 (strict)."""
    if not available:
        return None
    return ci90[0] > -MARGIN and ci90[1] < MARGIN


def gate_old(member_passes):
    """§3: gate passes iff every frozen eligible member passes every required pair; missing or
    undefined required tests, or an empty eligible set, give decision unavailable."""
    if not member_passes or any(v is None for v in member_passes.values()):
        return "decision unavailable"
    return "PASS" if all(member_passes.values()) else "FAIL"


def strategy_gate(src, base, stratum, members):
    """Append 18 strategy gate within one origin: TOST ±5 on each member's 90% interval of the
    with-minus-without difference; same strict bounds and unavailability rules as §3."""
    _, st = get(src, base + ["strategy_pooling_tests_tost5", stratum])
    st = st or {}
    per = st.get("per_probe") or {}
    ok = st.get("defined") is True
    return gate_old({p: (tost_pass(True, per[p]["ci90"]) if ok and p in per and interval_ok(per[p].get("ci90"))
                         and finite(per[p].get("diff")) else None) for p in members})


def extended_state(src, epath):
    """§3 extended gate: with no extended estimates in the record (withheld, e.g. missing honest_b pool),
    the extended decision is unavailable. A computable extended gate is not implemented here and fails."""
    _, ext = get(src, epath)
    ext = ext or {}
    if "sim_minus_mean_real" not in ext and isinstance(ext.get("note"), str):
        return "unavailable"
    return "computable"


def old_presentation(gate, ext="unavailable"):
    """§3: an unavailable extended decision retains 7b's hierarchy; within 7b the combined row leads only
    when its own gate passes; a defined nonpassing gate gives origin rows prominence."""
    if ext != "unavailable":
        return "NOT IMPLEMENTED (extended selector)"
    return {"PASS": "combined row leads", "FAIL": "origin rows lead",
            "decision unavailable": "decision unavailable"}[gate]


def differs_from_21j(gate, ext="unavailable"):
    """21j fixes the mixture first. With the extended decision unavailable, the old rule matches that only
    when the 7b gate passes."""
    if ext != "unavailable":
        return "NOT IMPLEMENTED (extended selector)"
    return {"PASS": "no", "FAIL": "yes", "decision unavailable": "comparison unavailable"}[gate]


META_KEYS = ("source_commit", "source_path", "report_commit", "report_path", "rule_commit", "rule_path",
             "report_code_commit", "checker_commit_parent", "branch")
INT_KEYS = {"line", "numeric_tokens", "accounted", "printed_dp", "min_dp", "consts", "sanitizer_masked_tokens"}
INT_PARENTS = {"totals", "numbers_matched_per_convention", "verdict_rows_by_kind"}
LABEL_FIELDS = ("printed_label", "expected_label", "printed_label_gate", "recomputed_tost_gate")
LABEL_ALLOW = {"equivalent", "materially different", "inconclusive at this n", "decision unavailable",
               "PASS", "FAIL", "yes", "no", "comparison unavailable", "pass", "fail", "unavailable", "missing",
               "origin rows lead", "combined row leads", "unparsed", "string equal to record", "as printed",
               "strategy rows lead", "strategy pooled", "not stated", "extended headline decision unavailable",
               "other", "NOT IMPLEMENTED (extended selector)", "UNRECOGNIZED"}
DEC_TOKEN = r"(?<![A-Za-z_0-9.])-?\d+\.\d+(?![A-Za-z_0-9])"
INTERPRETATIONS = [
    "Verdict availability (21j §2): pooling test defined, 9,500 <= complete draws <= 10,000, registered total "
    "10,000, finite D, finite ci90 and ci95, member in the frozen eligible set. The family minimum-member rule "
    "withholds family X_err/U only; it does not make an estimable per-member contrast unavailable.",
    "Old-rule TOST availability (§3): the old rule's own validity (test defined, finite D and ci90); the 21j "
    "9,500-draw threshold is not imported into the old rule.",
    "Extended gate (§3): with no extended estimates in the record, the extended decision is unavailable and "
    "7b's hierarchy is retained; a computable extended selector is not implemented and would fail the check.",
    "Append 18 (kept by 21j acceptance §1, PROPOSED §4): when strategy equivalence is not established in some "
    "origin, the with/without-strategy rows lead and cross-strategy numbers are labelled.",
    "Interleaved (Append 32) presentation: not present in this record; not checkable here.",
    "Withheld rows: 'n/a' is accepted for X_err when it is null, and for U/bias only when the pooled block "
    "has X_err null and a stored withholding note.",
    "v3: numbers are compared at the precision printed (1 dp for catch, X_err, D and intervals; 2-4 dp for "
    "companion values, weights, power and w), under the same four rounding conventions.",
    "v3: 21j §2b D rows and the Append 37 D_std rows are available iff 9,500 <= jointly_finite_draws <= draws = "
    "10,000, D finite, both intervals finite, member eligible; verdicts recomputed with 21j §2's strict rule.",
    "v3: the line-3 leaf count ('on all N leaves') refers to check_repro_rev6s (not opened); it is exempt and "
    "unverified here.",
    "Record-consistency checks (MDD_0, D_true_80, equivalence and two-sided detection power formulas; companion "
    "counts, weights and RMS; 2b cross D equal to the pooling test) test the record, not the report, and are "
    "reported outside 'overall'. The D_std composition and all D orientations are enforced inside 'overall'.",
    "Instability: every verdict input's unstable_over_1pct_nan flag is emitted (validated boolean); availability "
    "does not use it because 21j's text does not define its availability consequence.",
]


def scrub(obj):
    """Output guard: labels must come from the allowlist; any decimal token other than the alpha strings
    0.01/0.05 is masked. Returns (scrubbed, count_masked)."""
    n = [0]

    def s_str(x):
        def sub(m):
            if m.group(0) in ("0.01", "0.05"):
                return m.group(0)
            n[0] += 1
            return "<masked>"
        return re.sub(DEC_TOKEN, sub, x)

    def rec(o, key=None, parent=None):
        if isinstance(o, dict):
            return {k: rec(v, k, key) for k, v in o.items()}
        if isinstance(o, list):
            return [rec(v, key, parent) for v in o]
        if isinstance(o, str):
            if key in LABEL_FIELDS and o not in LABEL_ALLOW:
                n[0] += 1
                return "UNRECOGNIZED"
            return s_str(o)
        if isinstance(o, float):
            n[0] += 1
            return "<masked>"
        if isinstance(o, int) and not isinstance(o, bool) and key not in INT_KEYS and parent not in INT_PARENTS:
            n[0] += 1
            return "<masked>"
        return o
    return rec(obj), n[0]


def selftest():
    """Synthetic intervals only (no source values): boundary and orientation cases of §2/§3."""
    assert verdict_21j(True, [-4.9, 4.9], [-6.0, 6.0]) == "equivalent"
    assert verdict_21j(True, [-5.0, 4.0], [-6.0, 5.0]) == "inconclusive at this n"  # L90 = -5 is not > -5
    assert verdict_21j(True, [5.5, 9.0], [5.0, 9.5]) == "inconclusive at this n"  # L95 = +5 is not > +5
    assert verdict_21j(True, [5.5, 9.0], [5.1, 9.5]) == "materially different"
    assert verdict_21j(True, [-9.0, -5.5], [-9.5, -5.1]) == "materially different"
    assert verdict_21j(False, [-1.0, 1.0], [-2.0, 2.0]) == "decision unavailable"
    assert tost_pass(True, [-4.99, 4.99]) is True and tost_pass(True, [-5.0, 1.0]) is False
    assert gate_old({"a": True, "b": True}) == "PASS" and gate_old({"a": True, "b": False}) == "FAIL"
    assert gate_old({}) == "decision unavailable" and gate_old({"a": True, "b": None}) == "decision unavailable"
    assert round_strings(0.25)["decimal_repr_half_up"] == "0.3" and round_strings(0.25)["py_format_binary_half_even"] == "0.2"
    assert norm_printed("−1.5") == "-1.5"
    assert round_strings(0.12345, 3)["py_format_binary_half_even"] == "0.123" and zero_norm("-0.000") == "0.000"
    assert abs(phi(1.959963984540054) - 0.975) < 1e-9
    sc, nm = scrub({"printed_label": "1.5", "key_path": "a/R|0.05/x 2.5", "v": 0.3})
    assert sc == {"printed_label": "UNRECOGNIZED", "key_path": "a/R|0.05/x <masked>", "v": "<masked>"} and nm == 3
    assert n_tokens("Complete draws: 9 of 10,000 (minimum 9,500).") == 3 and n_tokens("Appends 22–30, §2b(iii)") == 0
    print("selftest: PASS")


# ---------- report parsing ----------

def parse_sections(lines):
    secs = []
    cur = None
    for i, line in enumerate(lines, 1):
        m = re.match(r"^## Readout (reply|command) \((mean|max)\), FPR (\d+)%\s*$", line)
        if m:
            alpha = int(m.group(3)) / 100
            cur = {"ak": f"{READOUT[m.group(1)]}|{alpha:.2f}", "start": i, "lines": [],
                   "readout_agg": (m.group(1), m.group(2))}
            secs.append(cur)
            continue
        if line.startswith("---") and cur is not None and not line.startswith("|"):
            cur = None
            continue
        if cur is not None:
            cur["lines"].append((i, line))
    return secs


NUM_TOKEN = r"(?<![A-Za-z_0-9.,])[-+−]?\d+(?:,\d{3})*(?:\.\d+)?(?![A-Za-z_0-9])"
# reference-only numbers (section/append references, dates, times, the cohort's own name), removed before
# per-line token accounting
REF_PATTERNS = [r"Append \d+ item \d+", r"\(after the 507 read",
                r"^# \d+ X_err report", r"\d{4}-\d{2}-\d{2}(?:_\d+)?", r"\b\d{2}:\d{2}:\d{2}\b",
                r"Appends? \d+[a-z]?(?:\s*[–/-]\s*\d+[a-z]?)*", r"Amendment \d+[a-z]?",
                r"§\s?\d+[a-z]?(?:\([ivx]+\))?", r"S_new \(\+?311\)", r"n < 30 mains"]


def n_tokens(line):
    if line.startswith("Record: "):  # provenance line only: the reproduction check's leaf count (other artifact)
        line = re.sub(r"on all [\d,]+ leaves", " ", line)
    for pat in REF_PATTERNS:
        line = re.sub(pat, " ", line)
    return len(re.findall(NUM_TOKEN, line))


SECTION_AGG = {"reply": "mean", "command": "max"}


def schema(C, cid, hl, header_line, expected):
    got = cells_of(header_line)
    C.struct(cid, f"line {hl}", got == expected, consts=n_tokens(header_line))
    return got == expected


def cells_of(line):
    """Split a markdown table row on ' | ' so an unescaped '|' inside a label (e.g. S_old|with_strategy)
    stays in its cell. pipe_count_gfm is the cell count a GFM renderer would see."""
    body = line.strip()
    body = body[1:] if body.startswith("|") else body
    body = body[:-1] if body.endswith("|") else body
    return [c.strip() for c in re.split(r"\s\|\s", " " + body + " ")]


def gfm_cell_count(line):
    body = line.strip().strip("|")
    return len(re.split(r"(?<!\\)\|", body))


def table_after(sec_lines, header_prefix):
    """Return (header_line_no, header_cells, [(line_no, cells), ...]) for the table whose header
    starts with header_prefix."""
    for idx, (ln, line) in enumerate(sec_lines):
        if line.startswith(header_prefix):
            header = cells_of(line)
            rows = []
            for ln2, l2 in sec_lines[idx + 2:]:
                if not l2.startswith("|"):
                    break
                rows.append((ln2, cells_of(l2)))
            return ln, header, rows
    return None, None, []


def split_ci(cell):
    """'a [b, c]' -> (a, b, c); 'n/a n/a' -> ('n/a','n/a','n/a')."""
    m = re.match(r"^(\S+) \[(\S+), (\S+)\]$", cell)
    if m:
        return m.group(1), m.group(2), m.group(3)
    if cell.strip() == "n/a n/a":
        return "n/a", "n/a", "n/a"
    return None


def bracket(cell):
    m = re.match(r"^\[(\S+), (\S+)\]$", cell)
    return (m.group(1), m.group(2)) if m else None


def level_of(text):
    m = re.search(r"(\d+)%", text)
    return int(m.group(1)) if m else None


def do_verdicts(C, src, pre, pt_path, smr, members, rows, lv_a, lv_b, draws_total, draws_total_path):
    """21j §2 verdict per printed member row; returns old-rule (§3) TOST pass per member."""
    _, pt = get(src, pt_path)
    pt = pt or {}
    compared_ok = pt.get("compared") == ["S_new", "S_old"]
    cd = pt.get("complete_draws")
    member_passes = {}
    printed_probes = []
    for ln, cells in rows:
        probe = safe_id(cells[0])
        printed_probes.append(probe)
        loc = f"line {ln} ({pre} 21j verdicts, {probe})"
        pid = f"{pre}|verdict|{probe}"
        pp = pt_path + ["per_probe", probe]
        C.num(pid + "|D", loc, pp + ["diff"], cells[1])
        ia, ib = bracket(cells[2]), bracket(cells[3])
        for lv, iv in ((lv_a, ia), (lv_b, ib)):
            if iv is None:
                C.struct(f"{pid}|ci{lv}_parsed", loc, False)
                continue
            C.num(f"{pid}|ci{lv}_lo", loc, pp + [f"ci{lv}", 0], iv[0])
            C.num(f"{pid}|ci{lv}_hi", loc, pp + [f"ci{lv}", 1], iv[1])
        _, rec = get(src, pp)
        rec = rec or {}
        ci90, ci95 = rec.get("ci90"), rec.get("ci95")
        avail = (pt.get("defined") is True and isinstance(cd, int) and MIN_JOINT_DRAWS <= cd <= REGISTERED_DRAWS
                 and draws_total == REGISTERED_DRAWS and interval_ok(ci90) and interval_ok(ci95)
                 and finite(rec.get("diff")) and probe in members)
        avail_old = (pt.get("defined") is True and interval_ok(ci90) and finite(rec.get("diff"))
                     and probe in members)
        exp = verdict_21j(avail, ci90, ci95)
        printed_label = cells[4].strip()
        _, dn = get(src, smr + ["S_new", "pooled", "per_probe", probe, "delta"])
        _, do = get(src, smr + ["S_old", "pooled", "per_probe", probe, "delta"])
        orient = ((compared_ok and abs(rec["diff"] - (dn - do)) <= 1e-9 * max(1.0, abs(dn), abs(do)))
                  if finite(dn) and finite(do) and finite(rec.get("diff")) else (None if not avail else False))
        nested = ((ci95[0] <= ci90[0] and ci90[1] <= ci95[1]) if interval_ok(ci90) and interval_ok(ci95)
                  else (None if not avail else False))
        near_keys = ([kp(pp + [f"ci{lv}", i]) for lv, ci in ((90, ci90), (95, ci95)) for i in (0, 1)
                      if abs(abs(ci[i]) - MARGIN) < 0.05] if interval_ok(ci90) and interval_ok(ci95) else [])
        near = bool(near_keys)
        # a near bound matters only on the side that decides a verdict: L90 vs -5, U90 vs +5 (equivalent),
        # L95 vs +5, U95 vs -5 (materially different)
        near_relevant = (interval_ok(ci90) and interval_ok(ci95) and any(
            abs(x - t) < 0.05 for x, t in ((ci90[0], -MARGIN), (ci90[1], MARGIN), (ci95[0], MARGIN), (ci95[1], -MARGIN))))
        C.verdicts.append({
            "verdict_id": pid, "report_location": loc,
            "rule_inputs_keys": [kp(pp + ["ci90"]), kp(pp + ["ci95"]), kp(pt_path + ["complete_draws"]),
                                 kp(pt_path + ["defined"]), kp(draws_total_path)],
            "available": avail, "expected_label": exp, "printed_label": printed_label,
            "match": exp == printed_label,
            "orientation_D_equals_S_new_minus_S_old": orient,
            "intervals_nested_95_contains_90": nested,
            "stored_within_5_agrees": rec.get("within_5") == tost_pass(avail_old, ci90),
            "stored_difference_beyond_5_agrees": rec.get("difference_beyond_5_established")
            == (exp == "materially different"),
            "any_bound_within_0.05_of_margin": near,
            "bounds_within_0.05_of_margin_key_paths": near_keys,
            "near_bound_on_decision_side": near_relevant,
        })
        member_passes[probe] = tost_pass(avail_old, ci90)
    C.struct(f"{pre}|verdict_rows_equal_members", f"line {rows[0][0] if rows else 0}", printed_probes == members)
    C.struct(f"{pre}|verdict_members_equal_stored_test_members", f"line {rows[0][0] if rows else 0}",
             sorted(printed_probes) == sorted((pt.get("per_probe") or {}).keys()))
    for p in members:  # members missing from the printed table still enter the gate
        if p not in member_passes:
            rec = (pt.get("per_probe") or {}).get(p, {})
            member_passes[p] = tost_pass(pt.get("defined") is True and interval_ok(rec.get("ci90"))
                                         and finite(rec.get("diff")), rec.get("ci90"))
    return pt, member_passes


def check_v1(C, src, lines, draws_total, draws_total_path):
    """v1 layout: per section, fixed rows (mixture first), per-probe table, 21j verdicts, old-rule block,
    strategy table last."""
    # rule-text header line: 95% for materially different, 90% for equivalent, margin 5
    hdr = next((ln for ln in lines if ln.startswith('21j "materially different"')), "")
    hm = re.search(r'the (\d+)% interval of D = Δ_S_new − Δ_S_old lies entirely outside ±(\d+) points; '
                   r'"equivalent": the (\d+)% interval strictly inside ±(\d+)', hdr)
    hdr_ln = next((i for i, ln in enumerate(lines, 1) if ln.startswith('21j "materially different"')), None)
    C.struct("header_rule_text_levels_and_margin", f"line {hdr_ln}", bool(hm) and
             (int(hm.group(1)), int(hm.group(2)), int(hm.group(3)), int(hm.group(4))) == (95, 5, 90, 5),
             consts=4)

    secs = parse_sections(lines)
    C.struct("sections_present_R005_R001_C005_C001", "section headers",
             sorted(s["ak"] for s in secs) == sorted(["R|0.05", "R|0.01", "C|0.05", "C|0.01"]))

    for s in secs:
        ak = s["ak"]
        SL = s["lines"]
        base = ["result", "results", ak, "primary"]
        smr = base + ["sim_minus_mean_real"]
        rex = base + ["reexecution_run2_minus_run1"]
        pt_path = base + ["pooling_tests_tost5", "S_new-S_old"]
        _, elig = get(src, base + ["eligible_v6_frozen"])
        elig = elig or []
        C.struct(f"{ak}|section_header_maps_to_results_key", f"line {s['start']}",
                 get(src, ["result", "results", ak])[0] and SECTION_AGG[s["readout_agg"][0]] == s["readout_agg"][1],
                 [kp(["result", "results", ak])], consts=1)
        om = next(((ln, l) for ln, l in SL if l.startswith("**Old pool-if-match rule (TOST ±")), None)
        if om:
            mm = re.search(r"TOST ±(\d+)", om[1])
            C.struct(f"{ak}|old_rule_margin_is_5", f"line {om[0]}", bool(mm) and int(mm.group(1)) == MARGIN, consts=1)
        else:
            C.struct(f"{ak}|old_rule_heading_present", ak, False)
        ext_state = extended_state(src, ["result", "results", ak, "extended"])
        vc = next(((ln, l) for ln, l in SL if l.startswith("**21j between-origin verdicts")), None)
        _, cmp_ = get(src, pt_path + ["compared"])
        C.struct(f"{ak}|verdict_caption_orientation_matches_stored_pair", f"line {vc[0] if vc else s['start']}",
                 bool(vc) and "D = Δ_S_new − Δ_S_old" in vc[1] and cmp_ == ["S_new", "S_old"],
                 [kp(pt_path + ["compared"])])

        # Frozen eligible members line
        fe = next(((ln, l) for ln, l in SL if l.startswith("Frozen eligible members:")), None)
        if fe:
            printed = [x.strip() for x in fe[1].split(":", 1)[1].strip().rstrip(".").split(",")]
            C.labels.append({"label_id": f"{ak}|eligible_members", "report_location": f"line {fe[0]}",
                             "key_path": kp(base + ["eligible_v6_frozen"]), "match": printed == elig})
        else:
            C.struct(f"{ak}|eligible_members_line_present", ak, False)

        fam_na = next(((ln, l) for ln, l in SL if l.startswith("Family X_err/U unavailable")), None)
        if fam_na:
            nmin = re.search(r"fewer than (\d+) eligible", fam_na[1])
            _, xe = get(src, smr + ["combined", "pooled", "X_err"])
            C.struct(f"{ak}|family_unavailable_line_consistent", f"line {fam_na[0]}",
                     bool(nmin) and len(elig) < int(nmin.group(1)) and xe is None,
                     [kp(base + ["eligible_v6_frozen"]), kp(smr + ["combined", "pooled", "X_err"])], consts=1)

        # --- fixed rows ---
        hl, hdr_cells, rows = table_after(SL, "| row | mains | cells | X_err")
        if hl is None:
            C.struct(f"{ak}|fixed_rows_table_present", ak, False)
        else:
            schema(C, f"{ak}|fixed_rows_header_schema", hl, next(l for n0, l in SL if n0 == hl),
                   ["row", "mains", "cells", "X_err", "U (approx 95%)", "bias (family mean Δ) [95%]",
                    "rerun yardstick run2−run1: X_err / U"])
            labels_in_order = [r[1][0] for r in rows]
            order_ok = len(rows) == 3 and all(labels_in_order[i].startswith(FIXED_ROW_ORIGIN[i][0])
                                              for i in range(3))
            C.struct(f"{ak}|fixed_rows_order_mixture_S_old_S_new", f"line {hl}", order_ok)
            for ln, cells in rows:
                origin = next((o for lab, o in FIXED_ROW_ORIGIN if cells[0].startswith(lab)), None)
                if origin is None:
                    C.struct(f"{ak}|fixed_row_label_unmapped", f"line {ln}", False)
                    continue
                loc = f"line {ln} ({ak} fixed rows, {origin})"
                pid = f"{ak}|fixed|{origin}"
                C.num(pid + "|mains", loc, smr + [origin, "mains"], cells[1], "int")
                C.num(pid + "|cells", loc, smr + [origin, "cells"], cells[2], "int")
                pooled = smr + [origin, "pooled"]
                fold = smr + [origin, "fold_stratified"]
                C.num(pid + "|X_err", loc, pooled + ["X_err"], cells[3], alts=[fold + ["X_err"]])
                C.num(pid + "|U", loc, pooled + ["U_approx_upper_95"], cells[4],
                      alts=[fold + ["U_approx_upper_95"]])
                b = split_ci(cells[5])
                if b is None:
                    C.struct(pid + "|bias_cell_parsed", loc, False)
                else:
                    C.num(pid + "|bias_mean", loc, pooled + ["bias_mean"], b[0], alts=[fold + ["bias_mean"]])
                    C.num(pid + "|bias_ci95_lo", loc, pooled + ["bias_ci95", 0], b[1])
                    C.num(pid + "|bias_ci95_hi", loc, pooled + ["bias_ci95", 1], b[2])
                y = [x.strip() for x in re.split(r"\s/\s", cells[6])]
                rpool = rex + [origin, "pooled"]
                alt_x = [base + ["S_new_same_launch_run2_minus_run1", "pooled", "X_err"],
                         base + ["S_new_all_cells_run2_minus_run1_sensitivity", "pooled", "X_err"],
                         rex + [origin, "fold_stratified", "X_err"]]
                alt_u = [base + ["S_new_same_launch_run2_minus_run1", "pooled", "U_approx_upper_95"],
                         base + ["S_new_all_cells_run2_minus_run1_sensitivity", "pooled", "U_approx_upper_95"],
                         rex + [origin, "fold_stratified", "U_approx_upper_95"]]
                if len(y) == 2:
                    C.num(pid + "|rerun_X_err", loc, rpool + ["X_err"], y[0], alts=alt_x)
                    C.num(pid + "|rerun_U", loc, rpool + ["U_approx_upper_95"], y[1], alts=alt_u)
                else:
                    C.struct(pid + "|yardstick_cell_parsed", loc, False)
                # floored-wording observation: record wording exists but report shows only the number
                _, wording = get(src, pooled + ["X_err_wording"])
                if wording is not None:
                    C.observations.append({"id": pid + "|X_err_wording_not_printed",
                                           "key_path": kp(pooled + ["X_err_wording"]),
                                           "wording_in_report": "floored" in cells[3]})

        # --- per-probe catch and delta ---
        hl, hdr_cells, rows = table_after(SL, "| origin | probe | catch run1")
        if hl is None:
            C.struct(f"{ak}|per_probe_table_present", ak, False)
        else:
            d_lvl = level_of(hdr_cells[6])
            schema(C, f"{ak}|per_probe_header_schema", hl, next(l for n0, l in SL if n0 == hl),
                   ["origin", "probe", "catch run1", "catch run2", "catch mean real", "catch sim",
                    "Δ = sim − mean real [95%]"])
            seen = []
            for ln, cells in rows:
                origin, probe = safe_id(cells[0]), safe_id(cells[1])
                seen.append((origin, probe))
                loc = f"line {ln} ({ak} per probe, {origin}, {probe})"
                pid = f"{ak}|perprobe|{origin}|{probe}"
                cp = base + ["catch_percent", origin]
                for col, key in zip(cells[2:6], ["catch_run1", "catch_run2", "catch_mean_real", "catch_sim"]):
                    C.num(pid + "|" + key, loc, cp + [key, probe], col)
                dd = split_ci(cells[6])
                pp = smr + [origin, "pooled", "per_probe", probe]
                if dd is None:
                    C.struct(pid + "|delta_cell_parsed", loc, False)
                else:
                    C.num(pid + "|delta", loc, pp + ["delta"], dd[0])
                    C.num(pid + "|delta_ci95_lo", loc, pp + [f"ci{d_lvl}", 0], dd[1])
                    C.num(pid + "|delta_ci95_hi", loc, pp + [f"ci{d_lvl}", 1], dd[2])
            expected_rows = [(o, p) for o in ["combined", "S_old", "S_new"] for p in elig]
            C.struct(f"{ak}|per_probe_rows_cover_eligible_x_origins", f"line {hl}", seen == expected_rows)

        # --- 21j verdicts ---
        hl, hdr_cells, rows = table_after(SL, "| probe | D |")
        member_passes = {}
        pt = {}
        if hl is None:
            C.struct(f"{ak}|verdict_table_present", ak, False)
        else:
            lv_a, lv_b = level_of(hdr_cells[2]), level_of(hdr_cells[3])
            schema(C, f"{ak}|verdict_header_schema", hl, next(l for n0, l in SL if n0 == hl),
                   ["probe", "D", "90%", "95%", "verdict"])
            pt, member_passes = do_verdicts(C, src, ak, pt_path, smr, elig, rows, lv_a, lv_b,
                                            draws_total, draws_total_path)

        # draws complete line
        dc = next(((ln, l) for ln, l in SL if l.startswith("Draws complete:")), None)
        if dc:
            mm = re.match(r"Draws complete: ([\d,]+) of ([\d,]+)\.", dc[1])
            loc = f"line {dc[0]} ({ak} draws)"
            if mm:
                C.num(f"{ak}|draws|complete_draws", loc, pt_path + ["complete_draws"], mm.group(1), "int")
                C.num(f"{ak}|draws|total_draws", loc, draws_total_path, mm.group(2), "int")
            else:
                C.struct(f"{ak}|draws_line_parsed", loc, False)
        else:
            C.struct(f"{ak}|draws_line_present", ak, False)

        # --- old-rule lines (§3) ---
        gate = gate_old({p: member_passes.get(p) for p in elig})
        tl = next(((ln, l) for ln, l in SL if l.startswith("Per member TOST pass:")), None)
        if tl:
            body, _, gpart = tl[1].partition("Gate:")
            marks = dict(re.findall(r"(\w+) ([✓✗])", body))
            for p in elig:
                exp = member_passes.get(p)
                exp_lab = "pass" if exp is True else "fail" if exp is False else "unavailable"
                pr = marks.get(p)
                pr_lab = "pass" if pr == "✓" else "fail" if pr == "✗" else "missing"
                C.old_rule.append({"line_id": f"{ak}|old_rule|tost|{p}", "report_location": f"line {tl[0]}",
                                   "rule_inputs_keys": [kp(pt_path + ["per_probe", p, "ci90"])],
                                   "expected_label": exp_lab, "printed_label": pr_lab, "match": exp_lab == pr_lab})
            extra = sorted(set(marks) - set(elig))
            C.struct(f"{ak}|old_rule_no_extra_members", f"line {tl[0]}", not extra)
            pg = gpart.strip().rstrip(".").strip()
            C.old_rule.append({"line_id": f"{ak}|old_rule|gate", "report_location": f"line {tl[0]}",
                               "rule_inputs_keys": [kp(pt_path + ["per_probe", p, "ci90"]) for p in elig],
                               "expected_label": gate, "printed_label": pg, "match": gate == pg})
        else:
            C.struct(f"{ak}|old_rule_tost_line_present", ak, False)

        ip = next(((ln, l) for ln, l in SL if l.startswith("Implied presentation under the old rule:")), None)
        if ip:
            txt = ip[1]
            if "stratum rows would lead" in txt and "strata differ" in txt:
                pr = "origin rows lead"
            elif "combined" in txt and "lead" in txt:
                pr = "combined row leads"
            elif "unavailable" in txt:
                pr = "decision unavailable"
            else:
                pr = "unparsed"
            exp = old_presentation(gate, ext_state)
            C.old_rule.append({"line_id": f"{ak}|old_rule|presentation", "report_location": f"line {ip[0]}",
                               "rule_inputs_keys": ["derived from old_rule|gate"],
                               "expected_label": exp, "printed_label": pr, "match": exp == pr,
                               "legacy_label_qualifier_printed": "legacy label" in txt})
        else:
            C.struct(f"{ak}|old_rule_presentation_line_present", ak, False)

        df = next(((ln, l) for ln, l in SL if l.startswith("Differs from the 21j presentation")), None)
        if df:
            pr = df[1].rsplit(":", 1)[1].strip().rstrip(".")
            exp = differs_from_21j(gate, ext_state)
            C.old_rule.append({"line_id": f"{ak}|old_rule|differs", "report_location": f"line {df[0]}",
                               "rule_inputs_keys": ["derived from old_rule|gate"],
                               "expected_label": exp, "printed_label": pr, "match": exp == pr})
        else:
            C.struct(f"{ak}|old_rule_differs_line_present", ak, False)

        rl = next(((ln, l) for ln, l in SL if l.startswith("Record's own labels")), None)
        if rl:
            mm = re.search(r'pooled_label "(.*?)"; headline "(.*)"\.?$', rl[1])
            _, plab = get(src, pt_path + ["pooled_label"])
            _, head = get(src, base + ["headline"])
            ok1 = bool(mm) and mm.group(1) == plab
            ok2 = bool(mm) and mm.group(2).rstrip(".").rstrip('"') == head
            C.old_rule.append({"line_id": f"{ak}|old_rule|record_pooled_label", "report_location": f"line {rl[0]}",
                               "rule_inputs_keys": [kp(pt_path + ["pooled_label"])],
                               "expected_label": "string equal to record", "printed_label": "as printed",
                               "match": ok1})
            C.old_rule.append({"line_id": f"{ak}|old_rule|record_headline", "report_location": f"line {rl[0]}",
                               "rule_inputs_keys": [kp(base + ["headline"])],
                               "expected_label": "string equal to record", "printed_label": "as printed",
                               "match": ok2})
            # record's own label consistent with the recomputed gate
            C.old_rule.append({"line_id": f"{ak}|old_rule|record_label_vs_recomputed_gate",
                               "report_location": f"line {rl[0]}",
                               "rule_inputs_keys": [kp(pt_path + ["pooled"]), kp(pt_path + ["defined"])],
                               "expected_label": gate,
                               "printed_label": ("PASS" if pt.get("pooled") is True else "FAIL")
                               if pt.get("defined") is True else "decision unavailable",
                               "match": gate == (("PASS" if pt.get("pooled") is True else "FAIL")
                                                 if pt.get("defined") is True else "decision unavailable")})
        else:
            C.struct(f"{ak}|old_rule_record_labels_line_present", ak, False)

        # §3 extended gate: required line ("extended headline decision unavailable" when unavailable)
        sec_text = "\n".join(l for _, l in SL)
        C.complete(f"{ak}|old_rule_extended_gate_line_printed", f"line {s['start']}",
                   ext_state != "unavailable" or "extended headline decision unavailable" in sec_text,
                   "21j §3: an unavailable extended decision is labelled 'extended headline decision unavailable'")
        _, rlab = get(src, ["result", "results", ak, "readout_label"])
        if rlab is not None:
            C.complete(f"{ak}|record_readout_label_printed", f"line {s['start']}", rlab in sec_text,
                       "record carries a readout label (caveat) for this readout")
        C.complete(f"{ak}|companion_X_err_strat_U_strat_reported", f"line {s['start']}",
                   "X_err_strat" in sec_text and "pending" not in sec_text.split("X_err_strat", 1)[1][:200],
                   "21j §2b(i): weighted-RMS companion always shown beside the mixture row")
        C.complete(f"{ak}|MDD_0_reported", f"line {s['start']}",
                   "MDD_0" in sec_text and "MDD_0: pending" not in sec_text,
                   "21j §2b(ii): per-member MDD_0 reported")

        # --- strategy strata ---
        hl, hdr_cells, rows = table_after(SL, "| row | mains | X_err | U |")
        if hl is None:
            C.struct(f"{ak}|strategy_table_present", ak, False)
        else:
            hdr_line = next(l for ln0, l in SL if ln0 == hl)
            schema(C, f"{ak}|strategy_header_schema", hl, hdr_line, ["row", "mains", "X_err", "U"])
            C.struct(f"{ak}|strategy_rows_inventory", f"line {hl}",
                     [c[0].replace("\\|", "|") for _, c in rows] ==
                     ["S_old|with_strategy", "S_old|without_strategy", "S_new|with_strategy", "S_new|without_strategy"])
            for ln, cells in rows:
                row_line = next(l for ln0, l in SL if ln0 == ln)
                C.presentation.append({"check_id": f"{ak}|strategy_row_gfm_cell_count_equals_header",
                                       "report_location": f"line {ln}",
                                       "match": gfm_cell_count(row_line) == gfm_cell_count(hdr_line),
                                       "why": "an unescaped '|' inside the row label adds a cell under GFM; "
                                              "excess cells are dropped and later values shift one header left"})
                st = safe_id(cells[0])
                loc = f"line {ln} ({ak} strategy, {st})"
                pid = f"{ak}|strategy|{st}"
                C.num(pid + "|mains", loc, smr + [st, "mains"], cells[1], "int")
                C.num(pid + "|X_err", loc, smr + [st, "pooled", "X_err"], cells[2],
                      alts=[smr + [st, "fold_stratified", "X_err"]])
                C.num(pid + "|U", loc, smr + [st, "pooled", "U_approx_upper_95"], cells[3],
                      alts=[smr + [st, "fold_stratified", "U_approx_upper_95"]])
                _, wording = get(src, smr + [st, "pooled", "X_err_wording"])
                if wording is not None:
                    C.observations.append({"id": pid + "|X_err_wording_not_printed",
                                           "key_path": kp(smr + [st, "pooled", "X_err_wording"]),
                                           "wording_in_report": "floored" in cells[2]})
        for stratum in ["S_old", "S_new"]:
            sl = next(((ln, l) for ln, l in SL if l.startswith(f"Strategy pooling test within {stratum}:")), None)
            sp = base + ["strategy_pooling_tests_tost5", stratum]
            if not sl:
                C.struct(f"{ak}|strategy_line_{stratum}_present", ak, False)
                continue
            pr = sl[1].split(":", 1)[1].strip()
            pr = pr[:-1] if pr.endswith(".") else pr
            _, lab = get(src, sp + ["pooled_label"])
            g = strategy_gate(src, base, stratum, elig)
            label_says = ("FAIL" if pr.startswith("strata differ") else
                          "PASS" if pr and not pr.startswith("strata differ") and "unavailable" not in pr
                          else "decision unavailable")
            C.labels.append({"label_id": f"{ak}|strategy_label|{stratum}", "report_location": f"line {sl[0]}",
                             "key_path": kp(sp + ["pooled_label"]), "match": pr == lab,
                             "recomputed_tost_gate": g, "printed_label_gate": label_says,
                             "gate_match": g == label_says})
        lead_expected = any(strategy_gate(src, base, st, elig) != "PASS" for st in ("S_old", "S_new"))
        hs, _, _ = table_after(SL, "| row | mains | X_err | U |")
        hf, _, _ = table_after(SL, "| row | mains | cells | X_err")
        C.presentation.append({"check_id": f"{ak}|append18_strategy_rows_lead_when_strategy_gate_fails",
                               "report_location": f"line {hs}",
                               "strategy_gate_fails_in_some_origin": lead_expected,
                               "match": (not lead_expected) or (hs is not None and hf is not None and hs < hf),
                               "why": "21j acceptance §1 keeps Append 18 strategy prominence; PROPOSED 21j §4: when "
                                      "strategy equivalence is not established the with/without-strategy rows lead"})



# ---------- v2 layout ----------

def unesc(x):
    return x.replace("\\|", "|")


def family_blocks(sec_lines):
    blocks, cur = [], None
    for ln, line in sec_lines:
        if line.startswith("### "):
            kind = ("primary" if "Primary family" in line else "expansion" if "Expansion family" in line
                    else "other")
            cur = {"kind": kind, "start": ln, "lines": []}
            blocks.append(cur)
        elif cur is not None:
            cur["lines"].append((ln, line))
    return blocks


def tables_in(block_lines):
    out, i, n = [], 0, len(block_lines)
    while i < n:
        ln, line = block_lines[i]
        if line.startswith("|") and i + 1 < n and block_lines[i + 1][1].startswith("|---"):
            rows, j = [], i + 2
            while j < n and block_lines[j][1].startswith("|"):
                rows.append((block_lines[j][0], cells_of(block_lines[j][1]), block_lines[j][1]))
                j += 1
            cap = next((t for _, t in reversed(block_lines[:i]) if t.strip() and not t.startswith("|")), "")
            out.append({"hl": ln, "header": cells_of(line), "header_line": line, "rows": rows, "caption": cap})
            i = j
        else:
            i += 1
    return out


def first_line(block_lines, prefix):
    return next(((ln, l) for ln, l in block_lines if l.startswith(prefix)), None)


def num_row(C, src, pid, loc, base_st, rex_st, cells, with_yardstick):
    """Summary-row numbers: estimation mains, X_err, U, bias [95%], complete draws (+ yardstick)."""
    pooled = base_st + ["pooled"]
    C.num(pid + "|estimation_mains", loc, base_st + ["mains"], cells[1], "int")
    C.num(pid + "|X_err", loc, pooled + ["X_err"], cells[2], alts=[base_st + ["fold_stratified", "X_err"]])
    C.num(pid + "|U", loc, pooled + ["U_approx_upper_95"], cells[3],
          alts=[base_st + ["fold_stratified", "U_approx_upper_95"]])
    b = split_ci(cells[4])
    if b is None:
        C.struct(pid + "|bias_cell_parsed", loc, False)
    else:
        C.num(pid + "|bias_mean", loc, pooled + ["bias_mean"], b[0])
        C.num(pid + "|bias_ci95_lo", loc, pooled + ["bias_ci95", 0], b[1])
        C.num(pid + "|bias_ci95_hi", loc, pooled + ["bias_ci95", 1], b[2])
    C.num(pid + "|complete_draws", loc, pooled + ["complete_draws"], cells[5], "int")
    if with_yardstick:
        y = [x.strip() for x in re.split(r"\s/\s", cells[6])]
        if len(y) == 2:
            C.num(pid + "|rerun_X_err", loc, rex_st + ["pooled", "X_err"], y[0])
            C.num(pid + "|rerun_U", loc, rex_st + ["pooled", "U_approx_upper_95"], y[1])
        else:
            C.struct(pid + "|yardstick_cell_parsed", loc, False)
    _, wording = get(src, pooled + ["X_err_wording"])
    if wording is not None:
        C.observations.append({"id": pid + "|X_err_wording_printed", "key_path": kp(pooled + ["X_err_wording"]),
                               "wording_in_report": "floored" in cells[2]})


# ---------- v3 additions: 21j §2b rows (Revision 6s append21j_2b) and Append 37 item 1 (POST HOC D_std) ----------

Z95, Z975, Z80 = 1.645, 1.960, 0.842  # constants as printed in the record's mdd_label / 21j §2b(ii)
Z95_X, Z975_X, Z80_X = 1.6448536269514722, 1.959963984540054, 0.8416212335729143
STRAT_KEY = {"with": "with_strategy", "without": "without_strategy", "cross": "cross_strategy"}
STRAT_SUF = {"with": "|with_strategy", "without": "|without_strategy", "cross": ""}


def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def close(a, b, tol=1e-9):
    return finite(a) and finite(b) and abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def record_formula_checks(C, vid, rec_path, rec):
    """Informational record-consistency checks of 21j §2b(ii) formulas (they test the record, not the report)."""
    se = rec.get("SE_diff")
    pw = rec.get("power") or {}
    det = pw.get("material_difference_detection_power") or {}
    out = {}
    if finite(se) and se > 0:
        out["MDD_0_rounded_constants"] = close(rec.get("MDD_0"), (Z95 + Z80) * se)
        out["MDD_0_exact_quantiles"] = close(rec.get("MDD_0"), (Z95_X + Z80_X) * se)
        out["D_true_80_rounded_constants"] = close(rec.get("D_true_for_about_80pct_material_detection"),
                                                   MARGIN + (Z975 + Z80) * se)
        out["D_true_80_exact_quantiles"] = close(rec.get("D_true_for_about_80pct_material_detection"),
                                                 MARGIN + (Z975_X + Z80_X) * se)
        for z, tag in ((Z95, "rounded"), (Z95_X, "exact")):
            out[f"equivalence_power_{tag}"] = close(pw.get("equivalence_power_at_true_D_0"),
                                                    max(0.0, 2 * phi(MARGIN / se - z) - 1), 1e-6)
        for z, tag in ((Z975, "rounded"), (Z975_X, "exact")):
            ok = True
            for d, key in ((6.0, "abs_D_true_6"), (10.0, "abs_D_true_10")):
                two = phi((d - MARGIN) / se - z) + phi((-d - MARGIN) / se - z)
                ok = ok and close(det.get(key), two, 1e-6)
            out[f"detection_power_two_sided_{tag}"] = ok
    ok = bool(out) and all(out.get(f"{q}_rounded_constants") or out.get(f"{q}_exact_quantiles")
                           for q in ("MDD_0", "D_true_80")) and \
        (out.get("equivalence_power_rounded") or out.get("equivalence_power_exact")) and \
        (out.get("detection_power_two_sided_rounded") or out.get("detection_power_two_sided_exact"))
    C.record_checks.append({"check_id": vid, "key_path": kp(rec_path), **out, "match": bool(ok)})


def verdict_row_ext(C, src, vid, loc, rec_path, cells, members, probe, expected_D, draws_total, layout, kind):
    """One row of a 21j §2b D table (layout d10) or the Append 37 D_std table (layout dstd)."""
    _, rec = get(src, rec_path)
    rec = rec or {}
    if layout == "d10":
        i_d, i_90, i_95, i_v, i_x = 1, 2, 3, 4, 9
    else:
        i_d, i_90, i_95, i_v, i_x = 2, 3, 4, 5, 6
    C.num(vid + "|D", loc, rec_path + ["D"], cells[i_d])
    for lv, ci_cell in ((90, cells[i_90]), (95, cells[i_95])):
        iv = bracket(ci_cell)
        if iv is None:
            C.struct(f"{vid}|ci{lv}_parsed_or_unavailable", loc,
                     ci_cell.strip() in ("n/a", "unavailable") and not interval_ok(rec.get(f"ci{lv}")))
            continue
        C.num(f"{vid}|ci{lv}_lo", loc, rec_path + [f"ci{lv}", 0], iv[0])
        C.num(f"{vid}|ci{lv}_hi", loc, rec_path + [f"ci{lv}", 1], iv[1])
    if layout == "d10":
        C.num(vid + "|MDD_0", loc, rec_path + ["MDD_0"], cells[6])
        C.num(vid + "|equivalence_power_D0", loc, rec_path + ["power", "equivalence_power_at_true_D_0"], cells[7],
              min_dp=3)
        dp = [x.strip() for x in re.split(r"\s/\s", cells[8])]
        if len(dp) == 2:
            for x, key in zip(dp, ("abs_D_true_6", "abs_D_true_10")):
                C.num(f"{vid}|detection_power_{key}", loc,
                      rec_path + ["power", "material_difference_detection_power", key], x, min_dp=3)
        else:
            C.struct(vid + "|detection_power_cell_parsed", loc, False)
        rv = cells[5].strip()
        C.labels.append({"label_id": vid + "|record_verdict", "report_location": loc,
                         "key_path": kp(rec_path + ["verdict_21j"]), "match": rv == rec.get("verdict_21j")})
        record_formula_checks(C, vid, rec_path, rec)
    excl = cells[i_x].strip()
    stored_excl = rec.get("ci95_excludes_0_direction_only")
    C.bools.append({"item_id": vid + "|ci95_excludes_0", "report_location": loc,
                    "key_path": kp(rec_path + ["ci95_excludes_0_direction_only"]),
                    "match": excl in ("True", "False") and isinstance(stored_excl, bool) and (excl == "True") == stored_excl})
    ci90, ci95 = rec.get("ci90"), rec.get("ci95")
    jfd, ndr = rec.get("jointly_finite_draws"), rec.get("draws")
    avail = (isinstance(jfd, int) and isinstance(ndr, int) and MIN_JOINT_DRAWS <= jfd <= ndr
             and ndr == REGISTERED_DRAWS and draws_total == REGISTERED_DRAWS and finite(rec.get("D"))
             and interval_ok(ci90) and interval_ok(ci95) and probe in members)
    exp = verdict_21j(avail, ci90, ci95)
    recomputed_excl = interval_ok(ci95) and (ci95[0] > 0 or ci95[1] < 0)
    orient = (close(rec.get("D"), expected_D, 1e-9) if expected_D is not None and finite(rec.get("D"))
              else (None if not avail else False))
    nested = (ci95[0] <= ci90[0] and ci90[1] <= ci95[1]) if interval_ok(ci90) and interval_ok(ci95) else (
        None if not avail else False)
    near_keys = ([kp(rec_path + [f"ci{lv}", i]) for lv, ci in ((90, ci90), (95, ci95)) for i in (0, 1)
                  if abs(abs(ci[i]) - MARGIN) < 0.05] if interval_ok(ci90) and interval_ok(ci95) else [])
    near_rel = (interval_ok(ci90) and interval_ok(ci95) and any(
        abs(x - t) < 0.05 for x, t in ((ci90[0], -MARGIN), (ci90[1], MARGIN), (ci95[0], MARGIN), (ci95[1], -MARGIN))))
    stored_ok = rec.get("verdict_21j") == exp and stored_excl == recomputed_excl
    C.verdicts.append({
        "verdict_id": vid, "kind": kind, "report_location": loc,
        "rule_inputs_keys": [kp(rec_path + ["ci90"]), kp(rec_path + ["ci95"]), kp(rec_path + ["jointly_finite_draws"]),
                             kp(rec_path + ["draws"])],
        "available": avail, "expected_label": exp, "printed_label": cells[i_v].strip(),
        "match": exp == cells[i_v].strip(),
        "orientation_D_equals_S_new_minus_S_old": orient,
        "intervals_nested_95_contains_90": nested,
        "stored_record_fields_agree": stored_ok,
        "stored_within_5_agrees": stored_ok, "stored_difference_beyond_5_agrees": stored_ok,
        "unstable_over_1pct_nan": rec.get("unstable_over_1pct_nan")
        if isinstance(rec.get("unstable_over_1pct_nan"), bool) else None,
        "unstable_flag_schema_ok": isinstance(rec.get("unstable_over_1pct_nan"), bool),
        "any_bound_within_0.05_of_margin": bool(near_keys),
        "bounds_within_0.05_of_margin_key_paths": near_keys,
        "near_bound_on_decision_side": near_rel,
    })


def d2b_table(C, src, pre, kind, t, base, smr, members, draws_total):
    bo = base + ["append21j_2b", STRAT_KEY[kind], "between_origin_D", "S_new-S_old"]
    suf = STRAT_SUF[kind]
    _, contrast = get(src, bo + ["contrast"])
    C.struct(f"{pre}|d2b_{kind}_contrast_is_registered_primary", f"line {t['hl']}",
             contrast == "sim_minus_mean_real", [kp(bo + ["contrast"])])
    printed = []
    for ln, cells, _ in t["rows"]:
        probe = safe_id(cells[0])
        printed.append(probe)
        _, dn = get(src, smr + ["S_new" + suf, "pooled", "per_probe", probe, "delta"])
        _, do = get(src, smr + ["S_old" + suf, "pooled", "per_probe", probe, "delta"])
        verdict_row_ext(C, src, f"{pre}|d2b_{kind}|{probe}", f"line {ln} ({pre} 21j D {kind})",
                        bo + ["per_probe", probe], cells, members, probe,
                        dn - do if finite(dn) and finite(do) else None, draws_total, "d10", f"21j_2b_{kind}")
    _, per = get(src, bo + ["per_probe"])
    C.struct(f"{pre}|d2b_{kind}_rows_equal_members", f"line {t['hl']}", printed == members)
    C.struct(f"{pre}|d2b_{kind}_rows_equal_stored_members", f"line {t['hl']}",
             sorted(printed) == sorted((per or {}).keys()))
    return bo


def dstd_table(C, src, pre, t, base, smr, members, draws_total):
    ap = base + ["append37_item1_POST_HOC"]
    _, w = get(src, ap + ["w_fixed_append37"])
    printed = []
    for ln, cells, _ in t["rows"]:
        probe = safe_id(cells[0])
        printed.append(probe)
        loc = f"line {ln} ({pre} POST HOC D_std)"
        vid = f"{pre}|dstd|{probe}"
        dd = split_ci(cells[1])
        sp = ap + ["Delta_S_new_std", probe]
        if dd is None:
            C.struct(vid + "|delta_std_cell_parsed", loc, False)
        else:
            C.num(vid + "|Delta_S_new_std", loc, sp + ["D"], dd[0])
            C.num(vid + "|Delta_S_new_std_ci95_lo", loc, sp + ["ci95", 0], dd[1])
            C.num(vid + "|Delta_S_new_std_ci95_hi", loc, sp + ["ci95", 1], dd[2])
        _, ds = get(src, sp + ["D"])
        _, dw = get(src, smr + ["S_new|with_strategy", "pooled", "per_probe", probe, "delta"])
        _, dwo = get(src, smr + ["S_new|without_strategy", "pooled", "per_probe", probe, "delta"])
        _, dold = get(src, smr + ["S_old", "pooled", "per_probe", probe, "delta"])
        std_ok = finite(w) and finite(dw) and finite(dwo) and close(ds, w * dw + (1 - w) * dwo, 1e-9)
        exp_D = ds - dold if std_ok and finite(dold) else None
        verdict_row_ext(C, src, vid, loc, ap + ["D_std", probe], cells, members, probe, exp_D, draws_total,
                        "dstd", "post_hoc_D_std")
    C.struct(f"{pre}|dstd_rows_equal_members", f"line {t['hl']}", printed == members)


COMPANION_RE = re.compile(r"^Weighted-RMS companion \((with-strategy|without-strategy|cross-strategy); (.*?)\): "
                          r"X_err_strat = (\S+), U_strat = (\S+) \(weights S_old (\S+), S_new (\S+); "
                          r"origin X_err S_old (\S+), S_new (\S+)\)\.\s*(.*)$")


def companion_line(C, src, pre, ln, line, base, smr):
    m = COMPANION_RE.match(line)
    if not m:
        C.struct(f"{pre}|companion_line_parsed", f"line {ln}", False)
        return None
    kind = m.group(1).split("-")[0]
    X = base + ["append21j_2b", STRAT_KEY[kind], "X_err_strat", "sim_minus_mean_real"]
    loc = f"line {ln} ({pre} companion {kind})"
    vid = f"{pre}|companion_{kind}"
    C.num(vid + "|X_err_strat", loc, X + ["X_err_strat"], m.group(3), min_dp=2)
    C.num(vid + "|U_strat", loc, X + ["U_strat"], m.group(4), min_dp=2)
    C.num(vid + "|weight_S_old", loc, X + ["weights", "S_old"], m.group(5), min_dp=3)
    C.num(vid + "|weight_S_new", loc, X + ["weights", "S_new"], m.group(6), min_dp=3)
    C.num(vid + "|X_err_S_old", loc, X + ["X_err_by_origin", "S_old"], m.group(7), min_dp=2)
    C.num(vid + "|X_err_S_new", loc, X + ["X_err_by_origin", "S_new"], m.group(8), min_dp=2)
    _, rec = get(src, X)
    rec = rec or {}
    tail = m.group(9).strip().rstrip(".")
    ulab = rec.get("U_strat_label")
    C.labels.append({"label_id": vid + "|labels", "report_location": loc, "key_path": kp(X + ["label"]) + " ; " +
                     kp(X + ["U_strat_label"]),
                     "match": m.group(2) == rec.get("label") and (tail == (ulab or "").rstrip("."))})
    # record consistency: weights are origin mains shares; origin X_err are the stratum rows' X_err; RMS formula
    em = rec.get("estimation_mains") or {}
    w = rec.get("weights") or {}
    xo = rec.get("X_err_by_origin") or {}
    tot = (em.get("S_old") or 0) + (em.get("S_new") or 0)
    suf = STRAT_SUF[kind]
    counts_ok = all(em.get(o) == get(src, smr + [o + STRAT_SUF[kind], "mains"])[1] for o in ("S_old", "S_new"))
    w_ok = counts_ok and tot > 0 and all(close(w.get(o), em.get(o) / tot) for o in ("S_old", "S_new"))
    xo_ok = all(xo.get(o) == get(src, smr + [o + suf, "pooled", "X_err"])[1] for o in ("S_old", "S_new"))
    xs = rec.get("X_err_strat")
    f_ok = (xs is None and any(xo.get(o) is None for o in ("S_old", "S_new"))) or (
        finite(xs) and all(finite(xo.get(o)) for o in ("S_old", "S_new")) and
        close(xs, math.sqrt(sum(w[o] * xo[o] ** 2 for o in ("S_old", "S_new")))))
    C.record_checks.append({"check_id": vid, "key_path": kp(X), "companion_counts_equal_stratum_rows": counts_ok,
                            "weights_are_mains_shares": w_ok,
                            "origin_X_err_equal_stratum_rows": xo_ok, "rms_formula": f_ok,
                            "match": w_ok and xo_ok and f_ok})
    return kind



def v3_extras(C, src, pre, BL, FB, base, smr, members, draws_total, d2b_tabs, dstd_tabs, pt_path):
    A = base + ["append21j_2b"]
    _, A2 = get(src, A)
    A2 = A2 or {}
    stl = first_line(BL, "21j quantities' status (record):")
    if stl:
        C.labels.append({"label_id": f"{pre}|append21j_2b_status", "report_location": f"line {stl[0]}",
                         "key_path": kp(A + ["status"]),
                         "match": stl[1].split(":", 1)[1].strip().rstrip(".") == str(A2.get("status")).rstrip(".")})
    else:
        C.struct(f"{pre}|append21j_2b_status_line_present", pre, False)
    for ln, line in BL:
        if line.startswith("Weighted-RMS companion ("):
            companion_line(C, src, pre, ln, line, base, smr)
        elif line.startswith("MDD_0 / power: "):
            txt = line.split(": ", 1)[1].replace("\\|", "|")
            labs = {str(get(src, A + [STRAT_KEY[k], "between_origin_D", "S_new-S_old", "mdd_label"])[1])
                    for k in ("with", "without", "cross")}
            C.struct(f"{pre}|mdd_label_equals_record", f"line {ln}", len(labs) == 1 and txt in labs,
                     [kp(A + ["cross_strategy", "between_origin_D", "S_new-S_old", "mdd_label"])], consts=n_tokens(line))
    by_ln = {ln: l for ln, l in BL}
    for kind, t in d2b_tabs.items():
        last = t["rows"][-1][0] if t["rows"] else t["hl"]
        nxt = next((by_ln[x] for x in sorted(by_ln) if x > last and by_ln[x].strip()), "")
        C.struct(f"{pre}|d2b_{kind}_followed_by_MDD_explanation", f"line {t['hl']}", nxt.startswith("MDD_0 / power: "))
    dc = first_line(BL, "Complete draws:")
    mm = re.match(r"Complete draws: ([\d,]+) of", dc[1]) if dc else None
    shared = int(mm.group(1).replace(",", "")) if mm else None
    counts = []
    for kind in ("with", "without", "cross"):
        _, per = get(src, A + [STRAT_KEY[kind], "between_origin_D", "S_new-S_old", "per_probe"])
        counts += [(r or {}).get("jointly_finite_draws") for r in (per or {}).values()]
    _, per = get(src, base + ["append37_item1_POST_HOC", "D_std"])
    counts += [(r or {}).get("jointly_finite_draws") for r in (per or {}).values()]
    C.complete(f"{pre}|jointly_finite_count_reported_for_every_21j_contrast", f"line {dc[0] if dc else FB['start']}",
               shared is not None and bool(counts) and all(c == shared for c in counts),
               "21j §2: report the jointly finite count out of 10,000 for each contrast; a single shared line is "
               "accepted only if it equals every 2b and D_std contrast's stored count")
    for kind, t in d2b_tabs.items():
        bo = d2b_table(C, src, pre, kind, t, base, smr, members, draws_total)
        if kind == "cross":
            _, per2b = get(src, bo + ["per_probe"])
            _, perpt = get(src, pt_path + ["per_probe"])
            def ci_close(a, b):
                return interval_ok(a) and interval_ok(b) and all(close(x, y, 1e-12) for x, y in zip(a, b))
            same = bool(per2b) and sorted(per2b) == sorted(perpt or {}) and all(
                close((per2b[p] or {}).get("D"), (perpt or {}).get(p, {}).get("diff"), 1e-12)
                and ci_close((per2b[p] or {}).get("ci90"), (perpt or {}).get(p, {}).get("ci90"))
                and ci_close((per2b[p] or {}).get("ci95"), (perpt or {}).get(p, {}).get("ci95")) for p in per2b)
            C.record_checks.append({"check_id": f"{pre}|d2b_cross_equals_pooling_test", "key_path": kp(bo),
                                    "match": same})
    ph = first_line(BL, "**POST HOC (Append ")
    if dstd_tabs:
        ap = base + ["append37_item1_POST_HOC"]
        m = re.match(r"^\*\*POST HOC \(Append (\d+) item (\d+)\): strategy-standardized gap, w = (\S+) fixed\.\*\* (.*)$",
                     ph[1]) if ph else None
        if m:
            loc = f"line {ph[0]} ({pre} POST HOC heading)"
            C.num(f"{pre}|dstd|w_fixed", loc, ap + ["w_fixed_append37"], m.group(3), min_dp=4)
            _, w = get(src, ap + ["w_fixed_append37"])
            C.struct(f"{pre}|dstd_w_is_85_over_90_and_cites_37_1", loc,
                     close(w, 85 / 90, 1e-12) and (m.group(1), m.group(2)) == ("37", "1"))
            _, lab = get(src, ap + ["label"])
            C.labels.append({"label_id": f"{pre}|dstd_label", "report_location": loc, "key_path": kp(ap + ["label"]),
                             "match": m.group(4).strip() == lab})
        else:
            C.struct(f"{pre}|dstd_heading_parsed", pre, False)
        dstd_table(C, src, pre, dstd_tabs[0], base, smr, members, draws_total)


def check_family_v2(C, src, ak, fam, FB, draws_total, draws_total_path, ver="v2"):
    if fam == "primary":
        base = ["result", "results", ak, "primary"]
        members_path = base + ["eligible_v6_frozen"]
    else:
        base = ["result", "expansion", ak]
        members_path = base + ["members"]
    smr = base + ["sim_minus_mean_real"]
    rex = base + ["reexecution_run2_minus_run1"]
    pt_path = base + ["pooling_tests_tost5", "S_new-S_old"]
    _, members = get(src, members_path)
    members = members or []
    BL = FB["lines"]
    pre = f"{ak}|{fam}"
    btext = "\n".join(l for _, l in BL)

    ml = first_line(BL, "Members:")
    if ml:
        printed = [x.strip() for x in ml[1].split(":", 1)[1].strip().rstrip(".").split(",")]
        C.labels.append({"label_id": f"{pre}|members", "report_location": f"line {ml[0]}",
                         "key_path": kp(members_path), "match": printed == members})
    else:
        C.struct(f"{pre}|members_line_present", pre, False)

    _, rlab = get(src, base + ["readout_label"])
    rl = first_line(BL, "Readout label (record):")
    C.struct(f"{pre}|readout_label_line_present_iff_record_label", f"line {rl[0] if rl else FB['start']}",
             (rl is not None) == (rlab is not None), [kp(base + ["readout_label"])])
    if rl and rlab is not None:
        C.labels.append({"label_id": f"{pre}|readout_label", "report_location": f"line {rl[0]}",
                         "key_path": kp(base + ["readout_label"]),
                         "match": rl[1].split(":", 1)[1].strip().rstrip(".") == rlab.rstrip(".")})

    _, xe = get(src, smr + ["combined", "pooled", "X_err"])
    fna = first_line(BL, "Family X_err/U unavailable")
    C.struct(f"{pre}|family_unavailable_line_iff_family_withheld", f"line {fna[0] if fna else FB['start']}",
             (fna is not None) == (xe is None), [kp(smr + ["combined", "pooled", "X_err"])])
    if fna:
        nmin = re.search(r"fewer than (\d+) members", fna[1])
        C.struct(f"{pre}|family_unavailable_min_members", f"line {fna[0]}",
                 bool(nmin) and len(members) < int(nmin.group(1)), [kp(members_path)], consts=1)

    # Append 18 status and lead statement
    gates = {st: strategy_gate(src, base, st, members) for st in ("S_old", "S_new")}
    lead_expected = any(g != "PASS" for g in gates.values())
    a18 = first_line(BL, "**Append ")
    m = re.match(r"^\*\*Append (\d+) status:\*\* within S_old: (.*); within S_new: (.*?)\.?$", a18[1]) if a18 else None
    if not m:
        C.struct(f"{pre}|append18_status_line_parsed", pre, False)
    else:
        C.struct(f"{pre}|append18_status_cites_18", f"line {a18[0]}", m.group(1) == "18")
        for st, txt in (("S_old", m.group(2)), ("S_new", m.group(3))):
            sp = base + ["strategy_pooling_tests_tost5", st]
            _, lab = get(src, sp + ["pooled_label"])
            label_gate = "FAIL" if txt.startswith("strata differ") else (
                "decision unavailable" if "unavailable" in txt else "PASS")
            C.labels.append({"label_id": f"{pre}|strategy_label|{st}", "report_location": f"line {a18[0]}",
                             "key_path": kp(sp + ["pooled_label"]), "match": txt == lab,
                             "recomputed_tost_gate": gates[st], "printed_label_gate": label_gate,
                             "gate_match": gates[st] == label_gate})
    lead = first_line(BL, "Strategy equivalence not established in every origin")
    C.struct(f"{pre}|strategy_rows_lead_statement_iff_strategy_gate_fails",
             f"line {lead[0] if lead else FB['start']}", (lead is not None) == lead_expected)

    T = tables_in(BL)
    strat_sum = [t for t in T if t["header"][:2] == ["row", "estimation mains"] and len(t["header"]) == 6]
    cross_sum = [t for t in T if t["header"][:2] == ["row", "estimation mains"] and len(t["header"]) == 7]
    pp_tabs = [t for t in T if t["header"][:3] == ["row", "probe", "catch run1"]]
    ver_tabs = [t for t in T if t["header"][:2] == ["probe", "D"]]
    D10 = ["probe", "D", "90%", "95%", "verdict", "record verdict", "MDD_0", "equivalence power at D=0",
           "detection power at abs D 6 / 10", "95% excludes 0 (direction only)"]
    d2b_tabs, dstd_tabs = {}, []
    if ver == "v3":
        pool_tabs = [t for t in ver_tabs if len(t["header"]) == 5]
        for t in ver_tabs:
            if len(t["header"]) == 10:
                cap = t["caption"]
                hits = [k for k, ok in (("with", "within the with-strategy stratum" in cap),
                                        ("without", "within the without-strategy stratum" in cap),
                                        ("cross", cap.startswith("From append21j_2b.cross_strategy"))) if ok]
                kind = hits[0] if len(hits) == 1 else None
                C.struct(f"{pre}|d2b_table_caption_mapped", f"line {t['hl']}", kind is not None and kind not in d2b_tabs)
                schema(C, f"{pre}|d2b_header_schema", t["hl"], t["header_line"], D10)
                if kind:
                    d2b_tabs[kind] = t
                    if kind != "cross":
                        C.struct(f"{pre}|d2b_{kind}_caption_orientation", f"line {t['hl']}",
                                 "(ordered pair (S_new, S_old); registered)" in cap)
        dstd_tabs = [t for t in T if t["header"][:2] == ["probe", "Δ_S_new,std [95%]"]]
        for t in dstd_tabs:
            schema(C, f"{pre}|dstd_header_schema", t["hl"], t["header_line"],
                   ["probe", "Δ_S_new,std [95%]", "D_std", "90%", "95%", "verdict wording (21j, readability only)",
                    "95% excludes 0 (direction only)"])
        ver_tabs = pool_tabs
    SUMMARY = ["row", "estimation mains", "X_err", "U (approx 95%)", "bias (family mean Δ) [95%]", "complete draws"]
    for t in strat_sum:
        schema(C, f"{pre}|strategy_summary_header_schema", t["hl"], t["header_line"], SUMMARY)
    for t in cross_sum:
        schema(C, f"{pre}|cross_summary_header_schema", t["hl"], t["header_line"],
               SUMMARY + ["rerun yardstick run2−run1: X_err / U"])
    for t in pp_tabs:
        schema(C, f"{pre}|per_probe_header_schema", t["hl"], t["header_line"],
               ["row", "probe", "catch run1", "catch run2", "catch mean real", "catch sim", "Δ = sim − mean real [95%]"])
    for t in ver_tabs:
        schema(C, f"{pre}|verdict_header_schema", t["hl"], t["header_line"],
               ["probe", "D", "90%", "95%", "verdict", "MDD_0", "equivalence power at D=0",
                "detection power at abs D 6 / 10"] if ver == "v2" else ["probe", "D", "90%", "95%", "verdict"])
    C.struct(f"{pre}|table_inventory", f"line {FB['start']}",
             (len(strat_sum), len(cross_sum), len(pp_tabs), len(ver_tabs)) == (1, 1, 2, 1)
             and (ver == "v2" or (sorted(d2b_tabs) == ["cross", "with", "without"] and len(dstd_tabs) == 1)))
    for t in T:
        for ln, _, raw in t["rows"]:
            C.presentation.append({"check_id": f"{pre}|row_gfm_cell_count_equals_header", "report_location": f"line {ln}",
                                   "match": gfm_cell_count(raw) == gfm_cell_count(t["header_line"]),
                                   "why": "GFM drops excess cells and shifts later values left"})
    pp_strat = [t for t in pp_tabs if any("|" in unesc(r[1][0]) for r in t["rows"])]
    pp_cross = [t for t in pp_tabs if t not in pp_strat]
    if strat_sum and cross_sum and pp_strat:
        C.struct(f"{pre}|append18_strategy_tables_precede_cross_strategy", f"line {strat_sum[0]['hl']}",
                 (not lead_expected) or (strat_sum[0]["hl"] < cross_sum[0]["hl"] and pp_strat[0]["hl"] < cross_sum[0]["hl"]))

    # strategy summary
    if strat_sum:
        t = strat_sum[0]
        seq = []
        for ln, cells, _ in t["rows"]:
            lab = unesc(cells[0])
            loc = f"line {ln} ({pre} strategy summary)"
            mm = re.match(r"^(with|without)-strategy mixture", lab)
            ms = re.match(r"^S_(old|new) \(.*?\), (with|without)-strategy$", lab)
            if mm and ver == "v3":
                seq.append(f"{mm.group(1)}|mixture")
                st = f"combined|{mm.group(1)}_strategy"
                C.struct(f"{pre}|strategy_mixture_{mm.group(1)}_label", loc,
                         lab == f"{mm.group(1)}-strategy mixture (S_old ∪ S_new), equal-main-weighted")
                num_row(C, src, f"{pre}|strategy|{st}", loc, smr + [st], None, cells, False)
            elif mm:
                seq.append(f"{mm.group(1)}|mixture")
                C.struct(f"{pre}|strategy_mixture_{mm.group(1)}_pending_and_no_record_key", loc,
                         cells[1].startswith("pending") and not get(src, smr + [f"combined|{mm.group(1)}_strategy"])[0])
            elif ms:
                st = f"S_{ms.group(1)}|{ms.group(2)}_strategy"
                seq.append(f"{ms.group(2)}|S_{ms.group(1)}")
                num_row(C, src, f"{pre}|strategy|{st}", loc, smr + [st], None, cells, False)
            else:
                C.struct(f"{pre}|strategy_row_label_unmapped", loc, False)
        C.struct(f"{pre}|strategy_summary_order", f"line {t['hl']}",
                 seq == ["with|mixture", "with|S_old", "with|S_new", "without|mixture", "without|S_old", "without|S_new"])

    # cross-strategy summary
    if cross_sum:
        t = cross_sum[0]
        seq = []
        _, m_old = get(src, smr + ["S_old", "mains"])
        _, m_new = get(src, smr + ["S_new", "mains"])
        _, m_all = get(src, smr + ["combined", "mains"])
        for ln, cells, _ in t["rows"]:
            lab = cells[0]
            loc = f"line {ln} ({pre} cross-strategy summary)"
            origin = ("combined" if lab.startswith("mixture:") else "S_old" if lab.startswith("S_old (")
                      else "S_new" if lab.startswith("S_new (") else None)
            if origin is None:
                C.struct(f"{pre}|cross_row_label_unmapped", loc, False)
                continue
            seq.append(origin)
            num_row(C, src, f"{pre}|cross|{origin}", loc, smr + [origin], rex + [origin], cells, True)
            if origin == "combined":
                mm = re.search(r"\[S_old, S_new; (\d+) \+ (\d+) estimation mains; effective origin contributions "
                               r"S_old (\d+)/(\d+) = (\S+)%, S_new (\d+)/(\d+) = (\S+)%\]", lab)
                if not mm:
                    C.struct(f"{pre}|mixture_label_parsed", loc, False)
                    continue
                P = smr
                if ver == "v3":
                    MR = base + ["append21j_2b", "cross_strategy", "mixture_row"]
                    _, mr = get(src, MR)
                    mr = mr or {}
                    em, eoc = mr.get("estimation_mains") or {}, mr.get("effective_origin_contributions") or {}
                    C.num(f"{pre}|mixture_label|N_S_old", loc, MR + ["estimation_mains", "S_old"], mm.group(1), "int")
                    C.num(f"{pre}|mixture_label|N_S_new", loc, MR + ["estimation_mains", "S_new"], mm.group(2), "int")
                    C.num(f"{pre}|mixture_label|S_old_share_num", loc, MR + ["estimation_mains", "S_old"], mm.group(3), "int")
                    C.num(f"{pre}|mixture_label|S_old_share_den", loc, P + ["combined", "mains"], mm.group(4), "int")
                    C.num(f"{pre}|mixture_label|S_new_share_num", loc, MR + ["estimation_mains", "S_new"], mm.group(6), "int")
                    C.num(f"{pre}|mixture_label|S_new_share_den", loc, P + ["combined", "mains"], mm.group(7), "int")
                    for o, g in (("S_old", 5), ("S_new", 8)):
                        v = eoc.get(o)
                        C.derived(f"{pre}|mixture_label|{o}_share_pct", loc,
                                  [MR + ["effective_origin_contributions", o]], 100.0 * v if finite(v) else None,
                                  mm.group(g))
                    C.struct(f"{pre}|mixture_row_keys_consistent", loc,
                             mr.get("row") == "combined" and em.get("S_old") == m_old and em.get("S_new") == m_new
                             and isinstance(m_all, int) and em.get("S_old", 0) + em.get("S_new", 0) == m_all
                             and all(close(eoc.get(o), em.get(o, 0) / m_all, 1e-12) for o in ("S_old", "S_new")),
                             [kp(MR), kp(P + ["combined", "mains"])])
                    continue
                C.num(f"{pre}|mixture_label|N_S_old", loc, P + ["S_old", "mains"], mm.group(1), "int")
                C.num(f"{pre}|mixture_label|N_S_new", loc, P + ["S_new", "mains"], mm.group(2), "int")
                C.num(f"{pre}|mixture_label|S_old_share_num", loc, P + ["S_old", "mains"], mm.group(3), "int")
                C.num(f"{pre}|mixture_label|S_old_share_den", loc, P + ["combined", "mains"], mm.group(4), "int")
                C.num(f"{pre}|mixture_label|S_new_share_num", loc, P + ["S_new", "mains"], mm.group(6), "int")
                C.num(f"{pre}|mixture_label|S_new_share_den", loc, P + ["combined", "mains"], mm.group(7), "int")
                ok = all(isinstance(x, int) for x in (m_old, m_new, m_all)) and m_all > 0
                C.derived(f"{pre}|mixture_label|S_old_share_pct", loc, [P + ["S_old", "mains"], P + ["combined", "mains"]],
                          100.0 * m_old / m_all if ok else None, mm.group(5))
                C.derived(f"{pre}|mixture_label|S_new_share_pct", loc, [P + ["S_new", "mains"], P + ["combined", "mains"]],
                          100.0 * m_new / m_all if ok else None, mm.group(8))
                C.struct(f"{pre}|mixture_origins_disjoint_counts_add_up", loc, ok and m_old + m_new == m_all,
                         [kp(P + ["S_old", "mains"]), kp(P + ["S_new", "mains"]), kp(P + ["combined", "mains"])])
        C.struct(f"{pre}|cross_summary_order_mixture_S_old_S_new", f"line {t['hl']}", seq == ["combined", "S_old", "S_new"])

    # per-probe tables
    for t, kind in ((pp_strat[0] if pp_strat else None, "strategy"), (pp_cross[0] if pp_cross else None, "cross")):
        if t is None:
            C.struct(f"{pre}|per_probe_{kind}_table_present", pre, False)
            continue
        d_lvl = level_of(t["header"][6])
        seen = []
        for ln, cells, _ in t["rows"]:
            origin, probe = safe_id(unesc(cells[0])), safe_id(cells[1])
            seen.append((origin, probe))
            loc = f"line {ln} ({pre} per probe {kind})"
            pid = f"{pre}|perprobe|{origin}|{probe}"
            cp = base + ["catch_percent", origin]
            for col, key in zip(cells[2:6], ["catch_run1", "catch_run2", "catch_mean_real", "catch_sim"]):
                C.num(pid + "|" + key, loc, cp + [key, probe], col)
            dd = split_ci(cells[6])
            pp = smr + [origin, "pooled", "per_probe", probe]
            if dd is None:
                C.struct(pid + "|delta_cell_parsed", loc, False)
            else:
                C.num(pid + "|delta", loc, pp + ["delta"], dd[0])
                C.num(pid + "|delta_ci_lo", loc, pp + [f"ci{d_lvl}", 0], dd[1])
                C.num(pid + "|delta_ci_hi", loc, pp + [f"ci{d_lvl}", 1], dd[2])
        if kind == "strategy" and ver == "v3":
            exp = [(f"{o}|{w}_strategy", p) for w in ("with", "without") for o in ("combined", "S_old", "S_new")
                   for p in members]
        elif kind == "strategy":
            exp = [(st, p) for st in ("S_old|with_strategy", "S_new|with_strategy", "S_old|without_strategy",
                                      "S_new|without_strategy") for p in members]
        else:
            exp = [(o, p) for o in ("combined", "S_old", "S_new") for p in members]
        C.struct(f"{pre}|per_probe_{kind}_rows_cover_members", f"line {t['hl']}", seen == exp)

    # verdicts
    member_passes, pt = {}, {}
    if ver_tabs:
        t = ver_tabs[0]
        h = t["header"]
        lv_a, lv_b = level_of(h[2]), level_of(h[3])
        vc = first_line(BL, "**21j between-origin verdicts, cross-strategy")
        _, cmp_ = get(src, pt_path + ["compared"])
        C.struct(f"{pre}|verdict_caption_orientation_matches_stored_pair", f"line {vc[0] if vc else FB['start']}",
                 bool(vc) and "ordered pair (S_new, S_old); D = Δ_S_new − Δ_S_old" in vc[1]
                 and cmp_ == ["S_new", "S_old"], [kp(pt_path + ["compared"])])
        rows = [(ln, cells) for ln, cells, _ in t["rows"]]
        pt, member_passes = do_verdicts(C, src, pre, pt_path, smr, members, rows, lv_a, lv_b,
                                        draws_total, draws_total_path)
        if ver == "v2":
            stored_keys = set()
            for rec in (pt.get("per_probe") or {}).values():
                stored_keys |= set(rec.keys())
            pending_ok = all(all(c.startswith("pending") for c in cells[5:8]) for _, cells in rows)
            C.struct(f"{pre}|mdd_and_power_pending_and_absent_from_record", f"line {t['hl']}",
                     pending_ok and not any(re.search(r"mdd|power|se_diff", k, re.I) for k in stored_keys))
    else:
        C.struct(f"{pre}|verdict_table_present", pre, False)
    if ver == "v3":
        v3_extras(C, src, pre, BL, FB, base, smr, members, draws_total, d2b_tabs, dstd_tabs, pt_path)

    dc = first_line(BL, "Complete draws:")
    if dc:
        mm = re.match(r"Complete draws: ([\d,]+) of ([\d,]+) \(minimum ([\d,]+)\)\.", dc[1])
        loc = f"line {dc[0]} ({pre} draws)"
        if mm:
            C.num(f"{pre}|draws|complete_draws", loc, pt_path + ["complete_draws"], mm.group(1), "int",
                  alts=[base + ["append21j_2b", "cross_strategy", "between_origin_D", "S_new-S_old", "per_probe",
                                members[0] if members else "", "jointly_finite_draws"]])
            C.num(f"{pre}|draws|total_draws", loc, draws_total_path, mm.group(2), "int")
            C.struct(f"{pre}|draws_minimum_is_9500", loc, int(mm.group(3).replace(",", "")) == MIN_JOINT_DRAWS, consts=1)
        else:
            C.struct(f"{pre}|draws_line_parsed", loc, False)
    else:
        C.struct(f"{pre}|draws_line_present", pre, False)

    # old-rule block (§3)
    om = first_line(BL, "**Registered rule before 21j (pool-if-match, TOST ±")
    if om:
        mm = re.search(r"TOST ±(\d+)", om[1])
        C.struct(f"{pre}|old_rule_margin_is_5", f"line {om[0]}", bool(mm) and int(mm.group(1)) == MARGIN, consts=1)
    else:
        C.struct(f"{pre}|old_rule_heading_present", pre, False)
    gate = gate_old({p: member_passes.get(p) for p in members})
    if fam == "expansion":
        hpath = ["result", "expansion", "headline_21i|" + ak]
        epath = ["result", "expansion", "extended|" + ak]
    else:
        hpath = ["result", "results", ak, "headline_21i"]
        epath = ["result", "results", ak, "extended"]
    ext_state = extended_state(src, epath)
    tl = first_line(BL, "Per member TOST pass:")
    if tl:
        body, _, gpart = tl[1].split(":", 1)[1].partition("Gate:")
        marks = dict(re.findall(r"(\w+) (pass|fail|unavailable)", body))
        for p in members:
            e = member_passes.get(p)
            e_lab = "pass" if e is True else "fail" if e is False else "unavailable"
            C.old_rule.append({"line_id": f"{pre}|old_rule|tost|{p}", "report_location": f"line {tl[0]}",
                               "rule_inputs_keys": [kp(pt_path + ["per_probe", p, "ci90"])],
                               "expected_label": e_lab, "printed_label": marks.get(p, "missing"),
                               "match": e_lab == marks.get(p)})
        C.struct(f"{pre}|old_rule_no_extra_members", f"line {tl[0]}", not (set(marks) - set(members)))
        pg = gpart.strip().rstrip(".").strip()
        C.old_rule.append({"line_id": f"{pre}|old_rule|gate", "report_location": f"line {tl[0]}",
                           "rule_inputs_keys": [kp(pt_path + ["per_probe", p, "ci90"]) for p in members],
                           "expected_label": gate, "printed_label": pg, "match": gate == pg})
    else:
        C.struct(f"{pre}|old_rule_tost_line_present", pre, False)
    ip = first_line(BL, "Old-rule presentation:")
    if ip:
        txt = ip[1]
        pr = ("origin rows lead" if "origin rows have prominence over the combined row" in txt else
              "combined row leads" if ("combined row leads" in txt or "combined row has prominence" in txt) else
              "decision unavailable" if "unavailable" in txt else "unparsed")
        e = old_presentation(gate, ext_state)
        C.old_rule.append({"line_id": f"{pre}|old_rule|presentation", "report_location": f"line {ip[0]}",
                           "rule_inputs_keys": ["derived from old_rule|gate"], "expected_label": e,
                           "printed_label": pr, "match": e == pr,
                           "legacy_label_qualifier_printed": "legacy label" in txt})
        s18 = "strategy rows lead in both rules" in txt
        C.old_rule.append({"line_id": f"{pre}|old_rule|append18_strategy_rows_lead",
                           "report_location": f"line {ip[0]}",
                           "rule_inputs_keys": [kp(base + ["strategy_pooling_tests_tost5", st, "per_probe"])
                                                for st in ("S_old", "S_new")],
                           "expected_label": "strategy rows lead" if lead_expected else "strategy pooled",
                           "printed_label": "strategy rows lead" if s18 else "not stated",
                           "match": s18 == lead_expected})
    else:
        C.struct(f"{pre}|old_rule_presentation_line_present", pre, False)
    df = first_line(BL, "Differs from 21j")
    if df:
        mm = re.search(r"\):\s*(yes|no|comparison unavailable)\b", df[1])
        pr = mm.group(1) if mm else "unparsed"
        e = differs_from_21j(gate, ext_state)
        C.old_rule.append({"line_id": f"{pre}|old_rule|differs", "report_location": f"line {df[0]}",
                           "rule_inputs_keys": ["derived from old_rule|gate"], "expected_label": e,
                           "printed_label": pr, "match": e == pr})
    else:
        C.struct(f"{pre}|old_rule_differs_line_present", pre, False)
    rl2 = first_line(BL, "Record's own old-rule labels:")
    if rl2:
        mm = re.search(r'pooled_label "(.*?)"; headline "(.*)"\.?$', rl2[1])
        _, plab = get(src, pt_path + ["pooled_label"])
        _, head = get(src, base + ["headline"])
        for lid, ok, kpth in (("record_pooled_label", bool(mm) and mm.group(1) == plab, pt_path + ["pooled_label"]),
                              ("record_headline", bool(mm) and mm.group(2).rstrip(".").rstrip('"') == head,
                               base + ["headline"])):
            C.old_rule.append({"line_id": f"{pre}|old_rule|{lid}", "report_location": f"line {rl2[0]}",
                               "rule_inputs_keys": [kp(kpth)], "expected_label": "string equal to record",
                               "printed_label": "as printed", "match": ok})
        rec_gate = (("PASS" if pt.get("pooled") is True else "FAIL") if pt.get("defined") is True
                    else "decision unavailable")
        C.old_rule.append({"line_id": f"{pre}|old_rule|record_label_vs_recomputed_gate",
                           "report_location": f"line {rl2[0]}",
                           "rule_inputs_keys": [kp(pt_path + ["pooled"]), kp(pt_path + ["defined"])],
                           "expected_label": gate, "printed_label": rec_gate, "match": gate == rec_gate})
    else:
        C.struct(f"{pre}|old_rule_record_labels_line_present", pre, False)

    # §3 extended gate (21i): "extended headline decision unavailable" when the extended analysis is unavailable
    _, h21 = get(src, hpath)
    _, ext = get(src, epath)
    h21, ext = h21 or {}, ext or {}
    e_lab = ("extended headline decision unavailable" if ext_state == "unavailable"
             else "NOT IMPLEMENTED (extended selector)")
    xl = first_line(BL, "**21i extended")
    if xl:
        txt = xl[1]
        _, reg = get(src, ["result", "registry"])
        reg_ok = ("S_newmains: no registry file" in txt) == ("S_newmains" not in (reg or {}))
        C.old_rule.append({"line_id": f"{pre}|old_rule|extended_21i_decision", "report_location": f"line {xl[0]}",
                           "rule_inputs_keys": [kp(epath + ["missing_pool"]), kp(hpath + ["decision"])],
                           "expected_label": e_lab,
                           "printed_label": "extended headline decision unavailable"
                           if "extended headline decision unavailable" in txt else "other",
                           "match": e_lab in txt and h21.get("decision") == e_lab})
        C.labels.append({"label_id": f"{pre}|extended_21i_reason_strings", "report_location": f"line {xl[0]}",
                         "key_path": kp(hpath + ["why"]) + " ; " + kp(epath + ["note"]) + " ; result/registry",
                         "match": bool(h21.get("why")) and h21["why"] in txt
                         and ("Extended row: " + str(ext.get("note"))) in txt and reg_ok})
    C.complete(f"{pre}|old_rule_extended_gate_line_printed", f"line {FB['start']}", xl is not None,
               "21j §3: the old-rule row reports the extended gate per family; unavailable -> "
               "'extended headline decision unavailable'")
    if ver == "v3":
        kinds = {COMPANION_RE.match(l).group(1).split("-")[0] for _, l in BL if COMPANION_RE.match(l)}
        C.complete(f"{pre}|companion_X_err_strat_U_strat_reported", f"line {FB['start']}",
                   kinds == {"with", "without", "cross"},
                   "21j §2b(i): weighted-RMS companion shown beside each mixture row (cross, with, without)")
        C.complete(f"{pre}|MDD_0_and_power_reported", f"line {FB['start']}", sorted(d2b_tabs) == ["cross", "with", "without"],
                   "21j §2b(ii) and PROPOSED §4: per member MDD_0, equivalence and detection power reported")
        C.complete(f"{pre}|within_strategy_stratum_verdicts_reported", f"line {FB['start']}",
                   "with" in d2b_tabs and "without" in d2b_tabs,
                   "21j §2 applies within each registered presentation, including each strategy stratum")
        C.complete(f"{pre}|strategy_mixture_rows_reported", f"line {FB['start']}",
                   "strategy mixture (S_old ∪ S_new) | pending" not in btext
                   and "with-strategy mixture (S_old ∪ S_new), equal-main-weighted" in btext
                   and "without-strategy mixture (S_old ∪ S_new), equal-main-weighted" in btext,
                   "21j §2 fixed rows: the cohort-mixture row within each strategy stratum")
        if dstd_tabs:
            ph = first_line(BL, "**POST HOC (Append ")
            C.complete(f"{pre}|post_hoc_label_on_D_std", f"line {ph[0] if ph else FB['start']}",
                       bool(ph) and "POST HOC (after the 507 read" in ph[1],
                       "Append 37 status: each post hoc output carries 'POST HOC (after the 507 read)'")
        return
    C.complete(f"{pre}|companion_X_err_strat_U_strat_reported", f"line {FB['start']}",
               "X_err_strat / U_strat within each stratum: pending" not in btext
               and "opposite-signed origin gaps cannot cancel): pending" not in btext,
               "21j §2b(i): weighted-RMS companion always shown beside the mixture row")
    C.complete(f"{pre}|MDD_0_and_power_reported", f"line {FB['start']}",
               "pending (job revision) | pending (job revision)" not in btext,
               "21j §2b(ii) and PROPOSED §4: per member MDD_0, equivalence and detection power reported")
    C.complete(f"{pre}|within_strategy_stratum_verdicts_reported", f"line {FB['start']}",
               "within each stratum (D_with, D_without; registered by 21j, computed after the read): pending" not in btext,
               "21j §2 applies within each registered presentation, including each strategy stratum")
    C.complete(f"{pre}|strategy_mixture_rows_reported", f"line {FB['start']}",
               "strategy mixture (S_old ∪ S_new) | pending" not in btext,
               "21j §2 fixed rows: the cohort-mixture row within each strategy stratum")


def check_v2(C, src, lines, draws_total, draws_total_path, ver="v2"):
    hdr_ln, hdr = next(((i, l) for i, l in enumerate(lines, 1) if l.startswith("21j verdicts (acceptance §")),
                       (None, ""))
    hm = re.search(r'"equivalent" iff the (\d+)% interval of D lies strictly inside ±(\d+); "materially different" iff '
                   r'the (\d+)% interval lies entirely outside ±(\d+); otherwise "inconclusive at this n"; '
                   r'"decision unavailable" if fewer than ([\d,]+) complete draws', hdr)
    C.struct("header_rule_text_levels_margin_min_draws", f"line {hdr_ln}", bool(hm) and
             (int(hm.group(1)), int(hm.group(2)), int(hm.group(3)), int(hm.group(4)),
              int(hm.group(5).replace(",", ""))) == (90, 5, 95, 5, MIN_JOINT_DRAWS), consts=5)
    secs = parse_sections(lines)
    C.struct("sections_present_R005_R001_C005_C001", "section headers",
             sorted(s["ak"] for s in secs) == sorted(["R|0.05", "R|0.01", "C|0.05", "C|0.01"]))
    for s in secs:
        ak = s["ak"]
        C.struct(f"{ak}|section_header_maps_to_results_key", f"line {s['start']}",
                 get(src, ["result", "results", ak])[0] and get(src, ["result", "expansion", ak])[0]
                 and SECTION_AGG[s["readout_agg"][0]] == s["readout_agg"][1], consts=1)
        fbs = family_blocks(s["lines"])
        C.struct(f"{ak}|family_blocks_primary_then_expansion", f"line {s['start']}",
                 [b["kind"] for b in fbs] == ["primary", "expansion"])
        for FB in fbs:
            if FB["kind"] in ("primary", "expansion"):
                check_family_v2(C, src, ak, FB["kind"], FB, draws_total, draws_total_path, ver)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expected-sha-prefix", default="7dceb8bb")
    ap.add_argument("--meta", nargs="*", default=[])
    ap.add_argument("--format", choices=["v1", "v2", "v3"], default="v1")
    a = ap.parse_args()
    selftest()

    raw = open(a.source, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    src = json.loads(raw)
    report = open(a.report, encoding="utf-8").read()
    lines = report.splitlines()
    C = Checker(src)

    # (c) sha256, plus the prefix the report quotes
    sha_ok = sha.startswith(a.expected_sha_prefix)
    m = re.search(r"sha256 ([0-9a-f]{8,})", report)
    report_sha_ok = bool(m) and sha.startswith(m.group(1))
    sha_line = next((i for i, ln in enumerate(report.splitlines(), 1) if "sha256 " in ln), None)

    C.struct("record_sha256_prefix_quoted_in_report", f"line {sha_line}", report_sha_ok)

    draws_total_path = ["result", "design", "draws"]
    _, draws_total = get(src, draws_total_path)
    if a.format == "v1":
        check_v1(C, src, lines, draws_total, draws_total_path)
    else:
        check_v2(C, src, lines, draws_total, draws_total_path, a.format)

    # --- per-line numeric-token accounting: every number on every line is a checked item, a rule constant
    # validated by a structure check, or a reference-only number removed by REF_PATTERNS ---
    per_line_items, per_line_consts = {}, {}

    for r in C.items:
        ln = line_of(r["report_location"])
        if ln is not None and not r.get("printed_na"):
            per_line_items[ln] = per_line_items.get(ln, 0) + 1
    for r in C.structure:
        ln = line_of(r["report_location"])
        if ln is not None and r.get("consts"):
            per_line_consts[ln] = per_line_consts.get(ln, 0) + r["consts"]
    token_lines_bad = []
    for i, line in enumerate(lines, 1):
        n = n_tokens(line)
        acc = per_line_items.get(i, 0) + per_line_consts.get(i, 0)
        if n != acc:
            token_lines_bad.append({"line": i, "numeric_tokens": n, "accounted": acc,
                                    "context_digits_masked": re.sub(r"\d", "#", line)[:160]})
    table_numeric_in_report = sum(
        len(re.findall(r"(?<![A-Za-z_0-9.])-?\d+\.\d+(?![A-Za-z_0-9])", l)) for l in lines if l.startswith("|"))
    table_lines = {i for i, l in enumerate(lines, 1) if l.startswith("|")}
    dec_items = sum(1 for r in C.items if r["kind"] in ("dec1", "derived_dec1") and not r.get("printed_na")
                    and line_of(r["report_location"]) in table_lines)

    tot = {
        "numbers_checked": len(C.items),
        "numbers_matched": sum(r["match"] for r in C.items),
        "numbers_printed_na": sum(bool(r.get("printed_na")) for r in C.items),
        "decimal_numbers_checked_on_table_lines": dec_items,
        "decimal_numbers_checked_all_lines": sum(1 for r in C.items if r["kind"] in ("dec1", "derived_dec1")
                                                 and not r.get("printed_na")),
        "decimal_tokens_in_report_tables": table_numeric_in_report,
        "report_lines": len(lines),
        "lines_with_unaccounted_numeric_tokens": len(token_lines_bad),
        "numbers_convention_dependent": sum(bool(r.get("convention_dependent")) for r in C.items),
        "numbers_signed_zero_only": sum(bool(r.get("signed_zero_only_match")) for r in C.items),
        "verdicts_checked": len(C.verdicts),
        "verdicts_matched": sum(r["match"] for r in C.verdicts),
        "verdict_orientation_ok": sum(r["orientation_D_equals_S_new_minus_S_old"] is not False for r in C.verdicts),
        "verdict_intervals_nested": sum(r["intervals_nested_95_contains_90"] is not False for r in C.verdicts),
        "verdict_checks_not_applicable_unavailable": sum(r["orientation_D_equals_S_new_minus_S_old"] is None
                                                         for r in C.verdicts),
        "verdict_stored_flags_agree": sum(r["stored_within_5_agrees"] and r["stored_difference_beyond_5_agrees"]
                                          for r in C.verdicts),
        "verdicts_near_margin": sum(r["any_bound_within_0.05_of_margin"] for r in C.verdicts),
        "verdicts_near_margin_on_decision_side": sum(r["near_bound_on_decision_side"] for r in C.verdicts),
        "old_rule_lines_checked": len(C.old_rule),
        "old_rule_lines_matched": sum(r["match"] for r in C.old_rule),
        "label_lines_checked": len(C.labels),
        "label_lines_matched": sum(r["match"] and r.get("gate_match", True) for r in C.labels),
        "structure_checks": len(C.structure),
        "structure_checks_passed": sum(r["match"] for r in C.structure),
        "presentation_checks": len(C.presentation),
        "presentation_checks_passed": sum(r["match"] for r in C.presentation),
        "completeness_checks": len(C.completeness),
        "completeness_checks_passed": sum(r["match"] for r in C.completeness),
        "booleans_checked": len(C.bools),
        "booleans_matched": sum(r["match"] for r in C.bools),
        "record_consistency_checks": len(C.record_checks),
        "record_consistency_checks_passed": sum(bool(r["match"]) for r in C.record_checks),
        "verdict_rows_by_kind": dict(sorted({k: sum(1 for r in C.verdicts if r.get("kind", "pooling_tests") == k)
                                             for k in {r.get("kind", "pooling_tests") for r in C.verdicts}}.items())),
        "verdict_inputs_unstable_flag_true": sum(1 for r in C.verdicts if r.get("unstable_over_1pct_nan") is True),
    }
    nv = tot["verdicts_checked"]
    per_conv = {c: sum(c in r.get("conventions_matched", []) for r in C.items) for c in CONVENTIONS}
    content_ok = (tot["numbers_checked"] == tot["numbers_matched"] and nv == tot["verdicts_matched"]
                  and nv == tot["verdict_orientation_ok"] and nv == tot["verdict_intervals_nested"]
                  and nv == tot["verdict_stored_flags_agree"]
                  and tot["old_rule_lines_checked"] == tot["old_rule_lines_matched"]
                  and tot["label_lines_checked"] == tot["label_lines_matched"]
                  and tot["booleans_checked"] == tot["booleans_matched"]
                  and tot["structure_checks"] == tot["structure_checks_passed"] and sha_ok and report_sha_ok
                  and tot["decimal_numbers_checked_on_table_lines"] == tot["decimal_tokens_in_report_tables"]
                  and not token_lines_bad)
    pres_ok = (tot["presentation_checks"] == tot["presentation_checks_passed"]
               and tot["completeness_checks"] == tot["completeness_checks_passed"])
    meta = {}
    for kv in a.meta:
        k, _, v = kv.partition("=")
        if k in META_KEYS and re.fullmatch(r"[0-9a-f]{7,40}|[\w./-]{1,200}", v):
            meta[k] = v
    meta["checker_sha256"] = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    meta["format"] = a.format
    out = {
        "meta": meta,
        "rule": "Append 21j ACCEPTED §2 (verdicts, X=5, strict, >=9500 of 10000 jointly finite draws) "
                "and §3 (old pool-if-match TOST ±5 row); Append 18 strategy strata as kept by 21j",
        "interpretations": INTERPRETATIONS,
        "source_sha256_check": {"expected_prefix": a.expected_sha_prefix, "computed_prefix": sha[:8],
                                "matches": sha_ok, "report_quoted_prefix_matches": report_sha_ok},
        "totals": tot,
        "conventions_tested": CONVENTIONS,
        "numbers_matched_per_convention": per_conv,
        "overall": "MATCH" if content_ok else "NOT MATCH",
        "overall_scope": "printed numbers vs source keys, 21j verdicts, old-rule lines, record labels, "
                         "report structure, sha256, per-line numeric-token accounting",
        "overall_presentation_and_completeness": "PASS" if pres_ok else "FAIL",
        "failing_presentation_and_completeness_ids": sorted(
            {re.sub(r"^(R|C)\|0\.0[15]\|((primary|expansion)\|)?", "", r["check_id"])
             for r in C.presentation + C.completeness if not r["match"]}),
        "items": C.items,
        "verdicts": C.verdicts,
        "old_rule_lines": C.old_rule,
        "label_lines": C.labels,
        "structure_checks": C.structure,
        "presentation_checks": C.presentation,
        "completeness_checks": C.completeness,
        "booleans": C.bools,
        "record_consistency_checks": C.record_checks,
        "observations": C.observations,
        "lines_with_unaccounted_numeric_tokens": token_lines_bad,
    }
    out, n_masked = scrub(out)
    out["sanitizer_masked_tokens"] = n_masked
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    # stdout: counts, key paths, booleans and labels only (all from the scrubbed output)
    print("source sha256 prefix matches expected:", sha_ok, "| report-quoted prefix matches:", report_sha_ok)
    for k, v in out["totals"].items():
        print(f"{k}: {v}")
    print("numbers matched per convention:", out["numbers_matched_per_convention"])
    print("sanitizer_masked_tokens:", n_masked)
    for r in out["items"]:
        if not r["match"]:
            print("MISMATCH", r["item_id"], r["key_path"], "alts_matching:", r.get("alternative_keys_matching", []))
        elif r.get("convention_dependent"):
            print("CONVENTION-DEPENDENT", r["item_id"], r["key_path"], "matches only:", r["conventions_matched"])
        elif r.get("signed_zero_only_match"):
            print("SIGNED-ZERO", r["item_id"], r["key_path"])
    for r in out["verdicts"]:
        print("VERDICT", r["verdict_id"], "available", r["available"], "expected", repr(r["expected_label"]),
              "printed", repr(r["printed_label"]), "match", r["match"],
              "orient", r["orientation_D_equals_S_new_minus_S_old"], "nested", r["intervals_nested_95_contains_90"],
              "near_margin_decision_side", r["near_bound_on_decision_side"],
              "stored_flags_agree", r["stored_within_5_agrees"] and r["stored_difference_beyond_5_agrees"])
    for r in out["old_rule_lines"]:
        print("OLD-RULE", r["line_id"], "expected", repr(r["expected_label"]), "printed", repr(r["printed_label"]),
              "match", r["match"])
    for r in out["label_lines"]:
        if not (r["match"] and r.get("gate_match", True)):
            print("LABEL MISMATCH", r["label_id"], r["key_path"])
    for r in out["booleans"]:
        if not r["match"]:
            print("BOOLEAN MISMATCH", r["item_id"], r["key_path"])
    for r in out["record_consistency_checks"]:
        if not r["match"]:
            print("RECORD-CONSISTENCY FAIL", r["check_id"], {k: v for k, v in r.items() if isinstance(v, bool)})
    for r in out["structure_checks"]:
        if not r["match"]:
            print("STRUCTURE FAIL", r["check_id"], r["report_location"])
    for r in out["presentation_checks"] + out["completeness_checks"]:
        if not r["match"]:
            print("PRESENTATION/COMPLETENESS FAIL", r["check_id"], r["report_location"])
    for o in out["observations"]:
        print("OBS", o["id"], {k: v for k, v in o.items() if k in ("wording_in_report",)})
    for u in out["lines_with_unaccounted_numeric_tokens"]:
        print("UNACCOUNTED-NUMERIC-LINE", u["line"], u["numeric_tokens"], u["accounted"], u["context_digits_masked"][:100])
    print("OVERALL (numbers and verdicts):", out["overall"])
    print("OVERALL (presentation and completeness):", out["overall_presentation_and_completeness"])

if __name__ == "__main__":
    main()
