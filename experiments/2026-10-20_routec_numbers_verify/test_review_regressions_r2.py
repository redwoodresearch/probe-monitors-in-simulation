"""Regression tests from the v84 adapter review, round 2 (reviewer's fixtures_review_r2.py, sha256 f3af4389d11a2a14..., @ e825bc9e6),
run on this folder: every case must print NOT-REPRODUCED and every CHECK must be OK (N6 updated: superscripts are now tokens, S1).
Exits 1 otherwise. Synthetic text only.

  uv run --no-project --python 3.12 --with numpy==2.3.3 python -I -B test_review_regressions_r2.py
"""
import hashlib
import json
import sys
import tempfile
from pathlib import Path

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D))
import predicates as PR  # noqa: E402
import verify_core as VC  # noqa: E402
import verify_post as VP  # noqa: E402

PATS = VP.compile_patterns(json.loads((D / "config" / "v84_nonresult_patterns.json").read_text()))
SIG = {"shaS"}
OUT = []
H3, H4 = VP.TABLE_HEADERS["Table 3"], VP.TABLE_HEADERS["Table 4"]


def doc(body, t3="", t4=""):
    return body + "\n" + H3 + "\n|---|---|---|---|---|\n" + t3 + "\n" + H4 + "\n|---|---|---|\n" + t4


def say(name, reproduced):
    OUT.append((name, reproduced))
    print(("FALSE-PASS-REPRODUCED " if reproduced else "NOT-REPRODUCED ") + name)


CHECKS = []


def check(name, ok):
    CHECKS.append(ok)
    print(("CHECK-OK " if ok else "CHECK-FAIL ") + name)


def base_ctx(text, prov, recs, **kw):
    c = {"text_sha": VC.sha(text.encode()), "provenance": prov, "deferred": [], "deferred_record_ok": {}, "constants": [], "hyp_exprs": {},
         "rule_claims": [], "released_all": {}, "readout_of_slot": {}, "verified": {}, "slot_rows": {}, "by_jpath": {}, "artifact_of_sha": {},
         "kind_of": {}, "released": {}, "slot_raw": {}, "records": recs, "signed": SIG, "not_route_C": set(), "T": None, "not_printed": [],
         "not_printed_v84_approved": {"sha256": VC.sha(b""), "count": 0, "scope": "v84"}, "command_check": lambda *a: False,
         "added_spans": [], "orphan_notes": [], "patterns": PATS, "header_labels": [], "accepted_reasons": []}
    return c | kw


def p(text, anchor, printed, kp, pred, rec="shaS"):
    return {"v84_sha256": VC.sha(text.encode()), "anchor": anchor, "printed": printed, "record": "x/r.json@abcdef1", "record_sha256": rec,
            "key_path": kp, "predicate": pred}


def verdict(text, ctx):
    ctx = dict(ctx, provenance=ctx["provenance"] + [p(text, H4, "1", ["one"], {"op": "eq", "value": 1})],
               records=dict(ctx["records"], shaS=dict(ctx["records"].get("shaS", {}), one=1)))
    rows, plans, stops, dg = VP.post_rows("post", text, ctx)
    _, s_, n_ = VC.nonresult_list(rows)
    s = VC.summarize(rows, plans, stops, pins={"accepted_reasons": [], "nonresult_approved": {"sha256": s_, "count": n_}})
    return s, rows, dg, stops


def note_of(rows, j=0):
    return next((r["note"] for r in rows if r["artifact"] == "post/map" and r["slot"] == f"map:{j}"), None)


# ---- my 14 cases, strict: each must STOP for the stated reason ----
t = doc("We saw 15 runs.\n")
s, rows, _, _ = verdict(t, base_ctx(t, [p(t, "saw 15 runs", "5", ["n"], {"op": "eq", "value": 5})], {"shaS": {"n": 5}}))
say("F1 printed '5' verifies '15'", s["verdict_base"] != "STOP" or "whole-token" not in (note_of(rows) or ""))
t = doc("The rate was 12.5 points.\n")
s, rows, _, _ = verdict(t, base_ctx(t, [p(t, "rate was 12.5 points", "2.5", ["x"], {"op": "round_eq", "dp": 1, "value": 2.5})], {"shaS": {"x": 2.5}}))
say("F1b printed '2.5' verifies '12.5'", s["verdict_base"] != "STOP" or "whole-token" not in (note_of(rows) or ""))
t = doc("The gap was +2.5 points.\n")
s, rows, _, _ = verdict(t, base_ctx(t, [p(t, "gap was +2.5 points", "+2.5", ["d"], {"op": "abs_round_eq", "dp": 1, "value": 2.5, "sign": "-"})],
                                    {"shaS": {"d": -2.5}}))
say("F2 post '+2.5' (printed '+2.5', sign '-') for record -2.5", s["verdict_base"] != "STOP" or "sign" not in (note_of(rows) or ""))
t = doc("The rate was 12.3% (99.9 to 0.1).\n")
s, rows, _, _ = verdict(t, base_ctx(t, [p(t, "rate was 12.3% (99.9 to 0.1)", "12.3% (99.9 to 0.1)", ["r"],
                                          {"op": "all_round_pct_eq", "dp": 1, "value": 12.3})], {"shaS": {"r": 0.123}}))
# 2e / 09 (10-20) refined R2 to the token level: the row claims only the value it verifies (12.3); the interval ends fall through and
# are ORPHANS (the intended reason now), not a failed row note
say("F3 unchecked interval ends", s["verdict_base"] != "STOP" or not {r["printed"] for r in rows if r["slot"].startswith("orphan@")} >= {"99.9", "0.1"}
    or any(r["slot"].startswith("map:") and r["printed"] in ("99.9", "0.1") for r in rows))
t = doc("It sits at about 0.7 here.\n")
s, rows, _, _ = verdict(t, base_ctx(t, [p(t, "about 0.7 here", "0.7", ["x"], {"op": "abs_diff_le", "target": 0.5, "tol": 0.01})], {"shaS": {"x": 0.5}}))
say("F4 abs_diff_le target not printed", s["verdict_base"] != "STOP")
t = doc("All stay within 0.01 of it.\n")
s, rows, _, _ = verdict(t, base_ctx(t, [p(t, "within 0.01 of it", "0.01", ["xs"], {"op": "max_abs_le", "bound": 0.05})], {"shaS": {"xs": [0.03, -0.04]}}))
say("F4b bound 0.05 printed as 0.01", s["verdict_base"] != "STOP")
t = doc("We use 5 folds.\n\nThe probe beat the baseline on 5 folds of the held-out data.\n")
cons = [{"row": 1, "text": "5 folds", "class": "constant", "checks_ok": True, "anchor": "We use 5 folds."}]
s, rows, dg, st = verdict(t, base_ctx(t, [], {}, constants=cons))
second = t.index("5 folds", t.index("5 folds") + 1)
say("F6 anchored constant still covers the second occurrence",
    s["verdict_base"] != "STOP" or not any(r["slot"] == f"orphan@{second}" for r in rows) or dg["tokens_by_class"].get("constant") != 1)
t = doc("Alarms rose 3x; the gap was 10pp; p<.001; 5k runs; the 2nd draw.\n")
s, rows, dg, st = verdict(t, base_ctx(t, [], {}))
body_orph = [r for r in rows if r["slot"].startswith("orphan@") and int(r["slot"].split("@")[1]) < t.index(H3)]
say(f"F7 glued / leading-dot tokens get no orphan row ({len(body_orph)} orphan rows, kinds {dg['tokens_by_kind']})", len(body_orph) != 5)
t = doc("The verified share was 5% here. Suppose a probe caught 5% of the runs.\n")
lit = "5% of the runs"
hx = {"9": {"expressions": [["0.05 * 100", 5]], "percent_reading": "fraction", "text_sha256": hashlib.sha256(lit.encode()).hexdigest()}}
chk = PR.hypothetical_checks([{"row": 9, "anchor_sentence": "Suppose a probe caught 5% of the runs.", "text": lit}], hx, t,
                             {VP.block_of(t, 0): {5.0}}, lambda o: VP.block_of(t, o))
say("F8 no_shadow under 'fraction'", chk[9]["no_shadow"] is True)
src = "l1\nl2\nl3\nl4\nl5\nl6\n"
_, _, _, hb = VP.apply_diff_map(src, "--- a\n+++ b\n@@ -3 +5 @@\n-l3\n+NEW 7\n")
say(f"F9 wrong +start not reported (header problems {hb})", hb == 0)
t = doc("Layer 43 is used.\n")
s, rows, _, _ = verdict(t, base_ctx(t, [p(t, "Layer 43 is", "43", ["layer"], {"op": "eq", "value": 43}, rec="shaFIT")], {"shaFIT": {"layer": 43}},
                                    not_route_C={"shaFIT"}))
say(f"F10 Q-b verdict line is {s['verdict']!r}", s["verdict"] != "PASS; 1 not route C, checked against unsigned records by key")
t = doc("Intro.\n", t3="| `a` | x | p | s | 43 |\n")
s, rows, dg, _ = verdict(t, base_ctx(t, [p(t, "| `a` | x | p | s | 43 |", "43", ["layer"], {"op": "eq", "value": 43}, rec="shaFIT")],
                                     {"shaFIT": {"layer": 43}}, not_route_C={"shaFIT"}))
say("F11 not-route-C in Table 3 sets the tables34 printed flag", dg["tables34_printed_numbers_verified"] is True)
for txt, tok in (("In Section 4, 12 of 20 runs failed.\n", "12"), ("See Table 7 – 0.52 on either.\n", "0.52"), ("Tables 3 - 4 here.\n", "4")):
    t = doc(txt)
    _, rows, _, _ = verdict(t, base_ctx(t, [], {}))
    say(f"F12 {txt.strip()!r}: {tok} classed reference", any(r["printed"] == tok and r.get("nr_class") == "reference" for r in rows))
for txt, tok in (("Tables 3–4 show it.\n", "4"), ("Tables 3 and 4 show it.\n", "4")):
    t = doc(txt)
    _, rows, _, _ = verdict(t, base_ctx(t, [], {}))
    check(f"F12 control {txt.strip()!r}: {tok} is a reference", any(r["printed"] == tok and r.get("nr_class") == "reference" for r in rows))

# ---- F5 / R3 on the REAL runner: a swap whose apply also edits the text OUTSIDE sub() (as apply_v84's Tables 5/8 block does) ----
APPLY = b'''import json, sys
din, rin, dout, rep = sys.argv[1:5]
text = open(din, encoding="utf-8").read()
S = json.load(open(rin, encoding="utf-8"))["slots"]


def sub(rule, old, new, keys):
    global text
    text = text.replace(old, new(S))


sub("R1", "OLD", lambda S: S["k"], ["k"])
text = text.replace("P.", "P. Q 6")  # a module-level rewrite before the inserted span (no sub() call)
exec(open(__file__.replace("apply_v84.py", "appendix_rules.py"), encoding="utf-8").read())
open(dout, "w", encoding="utf-8").write(text)
open(rep, "w", encoding="utf-8").write("ok")
'''
with tempfile.TemporaryDirectory() as w:
    out, log, spans, info = VP.rerun_apply(APPLY, b"", b"P.\nQ OLD.\n", json.dumps({"slots": {"k": "6"}}).encode(), w)
body = out.decode()
real = body.index("Q 6.") + 2  # where sub() actually wrote "6"
t = doc(body)
claims = [{"rule": log[x]["rule"], "keys": log[x]["keys"], "inserted": log[x]["inserted"], "start": a, "end": b} for a, b, x in spans]
ver = {("rc1a", "k#0"): VC.row("rc1a", "k#0", "-", "-", "6", "6", "count", True, ["shaS"], SIG)}
prov = [p(t, "Q 6.", "6", ["n"], {"op": "eq", "value": 6})]  # the real swapped number is also covered legitimately by a map row
s, rows, dg, st = verdict(t, base_ctx(t, prov, {"shaS": {"n": 6}}, rule_claims=claims, released_all={"k": "6"}, readout_of_slot={"k": ["rc1a"]},
                                      verified=ver))
misattr = [r for r in rows if r["slot"].startswith("rule:") and int(r["slot"].split("@")[1]) != real]
say(f"N1 R3: an out-of-sub edit leaves a stale span on another '6' (spans {spans}, real at {real}, unattributable {info['unattributable']}, "
    f"lost {info['spans_lost']}): verdict {s['verdict']!r}, misattributed rule rows {len(misattr)}", bool(misattr) and s["verdict_base"] == "PASS")

# ---- N2: sign-free ops (no `sign` argument): an explicit printed sign is never checked against the record ----
t = doc("The gap was +2.5 points.\n")
s, rows, _, _ = verdict(t, base_ctx(t, [p(t, "gap was +2.5 points", "+2.5", ["d"], {"op": "abs_round_eq", "dp": 1, "value": 2.5})], {"shaS": {"d": -2.5}}))
say(f"N2a abs_round_eq without sign: post '+2.5', record -2.5 -> {s['verdict']!r}", s["verdict_base"] == "PASS")
t = doc("z - T was −0.42 there.\n")
s, rows, _, _ = verdict(t, base_ctx(t, [p(t, "was −0.42 there", "−0.42", [["z"], ["T"]], {"op": "diff_round_eq", "expr": "z - T", "dp": 2, "value": -0.42})],
                                    {"shaS": {"z": 2.42, "T": 2.0}}))
say(f"N2b diff_round_eq without sign: post '−0.42', z - T = +0.42 -> {s['verdict']!r}", s["verdict_base"] == "PASS")

# ---- N3: the x100 reading applies to bound arguments with no percent sign printed ----
t = doc("All stay within 5 of it.\n")
s, rows, _, _ = verdict(t, base_ctx(t, [p(t, "within 5 of it", "5", ["xs"], {"op": "max_abs_le", "bound": 0.05})], {"shaS": {"xs": [0.03, -0.04]}}))
say(f"N3 bound 0.05 accounts a bare printed '5' (no %) -> {s['verdict']!r}", s["verdict_base"] == "PASS")

# ---- N4: the non-result list (09 approves {sha, count}) carries no position ----
t = doc("See Table 3 here.\n\nAnd again Table 3 there.\n")
_, rows, _, _ = verdict(t, base_ctx(t, [], {}))
gen, _, _ = VC.nonresult_list(rows)
say(f"N4 non-result list entries have no position ({len(gen)} entries, {len(set(gen))} distinct)", len(set(gen)) < len(gen))

# ---- N5: header labels cannot leak into body cells ----
tbl = "| Model | Size 27 |\n|---|---|\n| m | 27 |\n"
t = doc("Intro.\n\n" + tbl)
hdr_pos = t.index("Size 27") + 5
body_pos = t.index("| m | 27 |") + 6
entries = [{"position": body_pos, "token": "27", "header_line": "| m | 27 |", "cell": "27", "cell_index": 2, "table": "x"},
           {"position": hdr_pos, "token": "27", "header_line": "| Model | Size 27 |", "cell": "Size 27", "cell_index": 2, "table": "x"}]
s, rows, dg, st = verdict(t, base_ctx(t, [], {}, header_labels=entries))
body_row = next(r for r in rows if r["slot"].endswith(f"@{body_pos}"))
check(f"N5 a header-label entry on a body cell is refused (stop) and the body 27 is an orphan; the header 27 is a header label "
      f"(body {body_row['slot'].split('@')[0]}, stops {sum('header labels' in x for x in st)})",
      body_row["slot"].startswith("orphan@") and any("header labels" in x for x in st)
      and any(r["slot"] == f"header_label@{hdr_pos}" for r in rows))

# ---- N6: digit coverage: a superscript digit is a visible digit in no token (fails closed) ----
t = doc("Area 3 m² here.\n")
_, _, _, st = verdict(t, base_ctx(t, [], {}))
_, rows6, dg6, st = verdict(t, base_ctx(t, [], {}))
check("N6 (S1) a superscript digit is a token: no digit-coverage STOP, the glued '²' is an identifier",
      not any("visible digits" in x for x in st) and any(r.get("nr_class") == "identifier" and r["printed"] == "²" for r in rows6))
print(f"{sum(r for _, r in OUT)} of {len(OUT)} reproduced")
sys.exit(1 if any(r for _, r in OUT) or not all(CHECKS) else 0)
