"""Regression tests from the v84 adapter review (reviewer's fixtures_review.py, sha256 351e5107d7127ed5..., @ ca7a07e25), adapted to
the fixed APIs (rule claims by position, pinned patterns, apply_diff_map). Each must print NOT-REPRODUCED; the script exits 1 otherwise.
Synthetic text only.

  uv run --no-project --python 3.12 --with numpy==2.3.3 python -I -B test_review_regressions.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verify_core as VC  # noqa: E402
import verify_post as VP  # noqa: E402
import predicates as PR  # noqa: E402

PATS = VP.compile_patterns(json.loads((HERE / "config" / "v84_nonresult_patterns.json").read_text()))

SIG = {"shaS"}
OUT = []
H3, H4 = VP.TABLE_HEADERS["Table 3"], VP.TABLE_HEADERS["Table 4"]


def doc(body, t3="", t4=""):
    return body + "\n" + H3 + "\n|---|---|---|---|---|\n" + t3 + "\n" + H4 + "\n|---|---|---|\n" + t4



def say(name, reproduced):
    OUT.append((name, reproduced))
    print(("FALSE-PASS-REPRODUCED " if reproduced else "NOT-REPRODUCED ") + name)


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
    # Table 4's header "Real run 1" holds a digit: give it a map row so it is not an RQ3 orphan in these fixtures
    ctx = dict(ctx, provenance=ctx["provenance"] + [p(text, H4, "1", ["one"], {"op": "eq", "value": 1})],
               records=dict(ctx["records"], shaS=dict(ctx["records"].get("shaS", {}), one=1)))
    rows, plans, stops, dg = VP.post_rows("post", text, ctx)
    _, s_, n_ = VC.nonresult_list(rows)
    s = VC.summarize(rows, plans, stops, pins={"accepted_reasons": [], "nonresult_approved": {"sha256": s_, "count": n_}})
    return s, rows, dg, stops


# F1: printed "5" claims the post's token "15" (anchor_spans admits a token starting at a0 - 1 that is a digit, not a sign)
t = doc("We saw 15 runs.\n")
s, rows, _, _ = verdict(t, base_ctx(t, [p(t, "saw 15 runs", "5", ["n"], {"op": "eq", "value": 5})], {"shaS": {"n": 5}}))
say("F1 anchor_spans: printed '5' (record 5) verifies the post's '15'", s["verdict"] == "PASS" and any(r["printed"] == "15" and r["equal"] for r in rows))

# F1b: printed "2.5" verifies "12.5"
t = doc("The rate was 12.5 points.\n")
s, rows, _, _ = verdict(t, base_ctx(t, [p(t, "rate was 12.5 points", "2.5", ["x"], {"op": "round_eq", "dp": 1, "value": 2.5})], {"shaS": {"x": 2.5}}))
say("F1b anchor_spans: printed '2.5' (record 2.5) verifies the post's '12.5'", s["verdict"] == "PASS")

# F2: explicit sign in the post opposite to the record's sign
t = doc("The gap was +2.5 points.\n")
s, rows, _, _ = verdict(t, base_ctx(t, [p(t, "gap was +2.5 points", "2.5", ["d"], {"op": "abs_round_eq", "dp": 1, "value": 2.5, "sign": "-"})],
                                    {"shaS": {"d": -2.5}}))
say("F2 sign: the post prints '+2.5', the record is -2.5 (sign '-'), PASS", s["verdict"] == "PASS" and any(r["printed"] == "+2.5" for r in rows))

# F3: a multi-number printed span: only the point is checked, the interval ends are claimed as verified
t = doc("The rate was 12.3% (99.9 to 0.1).\n")
s, rows, _, _ = verdict(t, base_ctx(t, [p(t, "rate was 12.3% (99.9 to 0.1)", "12.3% (99.9 to 0.1)", ["r"],
                                          {"op": "all_round_pct_eq", "dp": 1, "value": 12.3})], {"shaS": {"r": 0.123}}))
say("F3 one-directional printed check: '99.9' and '0.1' (checked by nothing) verified", s["verdict"] == "PASS" and
    {r["printed"] for r in rows if r["artifact"] == "post" and r["equal"] and r["slot"].startswith("map:0@")} == {"12.3", "99.9", "0.1"})

# F4: abs_diff_le's target is masked in 09's table but not in VALUE_ARGS: never compared with the printed text
t = doc("It sits at about 0.7 here.\n")
s, rows, _, _ = verdict(t, base_ctx(t, [p(t, "about 0.7 here", "0.7", ["x"], {"op": "abs_diff_le", "target": 0.5, "tol": 0.01})], {"shaS": {"x": 0.5}}))
say("F4 abs_diff_le: target 0.5 (= record), the post prints 0.7, PASS", s["verdict"] == "PASS")

# F4b: a bound op: max_abs_le bound 0.05; the post prints 0.01 as the bound
t = doc("All stay within 0.01 of it.\n")
s, rows, _, _ = verdict(t, base_ctx(t, [p(t, "within 0.01 of it", "0.01", ["xs"], {"op": "max_abs_le", "bound": 0.05})], {"shaS": {"xs": [0.03, -0.04]}}))
say("F4b max_abs_le: bound 0.05 (values up to 0.04), the post prints 'within 0.01', PASS", s["verdict"] == "PASS")

# F5: a rule's inserted string claims EVERY occurrence in the post, incl. a prose number and a Table 3 cell (RQ3 bypass)
t = doc("Swapped: 6.\n\nIn an unrelated hand-written sentence we ran 6 seeds.\n", t3="| `a` | x | p | s | 6 |\n")
ver = {("rc1a", "k#0"): VC.row("rc1a", "k#0", "-", "-", "6", "6", "count", True, ["shaS"], SIG)}
ctx = base_ctx(t, [], {}, rule_claims=[{"rule": "R1", "keys": ["k"], "inserted": "6", "start": t.index("6"), "end": t.index("6") + 1}],
               released_all={"k": "6"},
               readout_of_slot={"k": ["rc1a"]}, verified=ver)
s, rows, dg, stops = verdict(t, ctx)
say("F5 rule claims all 3 occurrences of its inserted '6' (prose + Table 3 cell), PASS, tables34_printed_numbers_verified",
    s["verdict"] == "PASS" and sum(r["slot"].startswith("rule:") for r in rows) == 3 and dg["tables34_printed_numbers_verified"] is True)

# F6: a constants literal covers every occurrence post-wide (not bound to its anchor sentence)
t = doc("We use 5 folds.\n\nThe probe beat the baseline on 5 folds of the held-out data.\n")
cons = [{"row": 1, "text": "5 folds", "class": "constant", "checks_ok": True, "anchor_sentence": "We use 5 folds."}]
s, rows, dg, _ = verdict(t, base_ctx(t, [], {}, constants=cons))
say("F6 constants: the second '5 folds' (a result sentence) is classed constant, PASS once the list is approved",
    s["verdict"] == "PASS" and dg["tokens_by_class"].get("constant") == 2)

# F7: tokenizer: digits glued to letters, or after '.', are identifiers: no row at all
t = doc("Alarms rose 3x; the gap was 10pp; p<.001; 5k runs; the 2nd draw.\n")
toks, ident = VP.tokens(t.split(H3)[0])
say(f"F7 tokens: '3x', '10pp', '.001', '5k', '2nd' get no row ({len(toks)} tokens, {ident} identifiers)", len(toks) == 0 and ident == 5)

# F8: no_shadow under the 'fraction' reading: literal '5%' -> 0.05, the verified '5%' token -> 5.0: no shadow found
t = doc("The verified share was 5% here. Suppose a probe caught 5% of the runs.\n")
lit = "5% of the runs"
import hashlib  # noqa: E402
hx = {"9": {"expressions": [["0.05 * 100", 5]], "percent_reading": "fraction", "text_sha256": hashlib.sha256(lit.encode()).hexdigest()}}
chk = PR.hypothetical_checks([{"row": 9, "anchor_sentence": "Suppose a probe caught 5% of the runs.", "text": lit}], hx, t,
                             {VP.block_of(t, 0): {5.0}}, lambda o: VP.block_of(t, o))
say(f"F8 hypothetical no_shadow misses a verified '5%' beside a fraction-read '5%' literal (checks {chk[9]})", chk[9]["no_shadow"] is True)

# F9: added_line_spans trusts the hunk's +start; apply_unified_diff uses -start only, so a wrong +start still applies byte-exactly
src = "l1\nl2\nl3\nl4\nl5\nl6\n"
tgt = "l1\nl2\nNEW 7\nl4\nl5\nl6\n"
good = "--- a\n+++ b\n@@ -3 +3 @@\n-l3\n+NEW 7\n"
bad = "--- a\n+++ b\n@@ -3 +5 @@\n-l3\n+NEW 7\n"
tg_b, _, sp_bad, hb_bad = VP.apply_diff_map(src, bad)
tg_g, _, sp_good, hb_good = VP.apply_diff_map(src, good)
say("F9 hand-edit spans: a diff with a wrong +start applies byte-exactly and its error is not reported, or its added span is another line",
    tg_b == tgt and (hb_bad == 0 or tgt[sp_bad[0][0]:sp_bad[0][1]] != "NEW 7") and hb_good == 0)

# F10: Q-b: the verdict line with not-route-C numbers is the bare string "PASS"
t = doc("Layer 43 is used.\n")
ctx = base_ctx(t, [p(t, "Layer 43 is", "43", ["layer"], {"op": "eq", "value": 43}, rec="shaFIT")], {"shaFIT": {"layer": 43}}, not_route_C={"shaFIT"})
s, rows, _, _ = verdict(t, ctx)
say(f"F10 Q-b: verdict with 1 not-route-C number is {s['verdict']!r} (no 'N not route C, checked against unsigned records by key')",
    s["verdict"] == "PASS" and s["n_not_route_C_checked"] == 1)

# F11: a not-route-C map row inside Table 3 counts toward tables34_printed_numbers_verified (route C scope)
t = doc("Intro.\n", t3="| `a` | x | p | s | 43 |\n")
ctx = base_ctx(t, [p(t, "| `a` | x | p | s | 43 |", "43", ["layer"], {"op": "eq", "value": 43}, rec="shaFIT")], {"shaFIT": {"layer": 43}},
               not_route_C={"shaFIT"})
s, rows, dg, _ = verdict(t, ctx)
say("F11 a not-route-C map number in Table 3 sets tables34_printed_numbers_verified True (tables34 printed=True)",
    dg["tables34_printed_numbers_verified"] is True and s["verdict"] == "PASS")

# F12: REF_RX's ',' separator: 'In Section 4, 12 of 20 runs' classes 12 as a reference
t = doc("In Section 4, 12 of 20 runs failed.\n")
_, rows, dg, _ = verdict(t, base_ctx(t, [], {}))
say("F12 REF_RX: the '12' after 'Section 4, ' is a reference", any(r["printed"] == "12" and r.get("nr_class") == "reference" for r in rows))

# positive controls (author): the same harness CAN pass, so each NOT-REPRODUCED above is not vacuous
PC = []
t = doc("We saw 15 runs.\n")
s, rows, _, _ = verdict(t, base_ctx(t, [p(t, "saw 15 runs", "15", ["n"], {"op": "eq", "value": 15})], {"shaS": {"n": 15}}))
PC.append(("PC1 F1's text with the whole token '15' PASSes", s["verdict"] == "PASS"))
t = doc("The gap was −2.5 points.\n")
s, rows, _, _ = verdict(t, base_ctx(t, [p(t, "gap was −2.5 points", "−2.5", ["d"], {"op": "abs_round_eq", "dp": 1, "value": 2.5, "sign": "-"})],
                                    {"shaS": {"d": -2.5}}))
PC.append(("PC2 F2 with the sign printed as the record's PASSes", s["verdict"] == "PASS"))
t = doc("It sits at about 0.5 here.\n")
s, rows, _, _ = verdict(t, base_ctx(t, [p(t, "about 0.5 here", "0.5", ["x"], {"op": "abs_diff_le", "target": 0.5, "tol": 0.01})], {"shaS": {"x": 0.5}}))
PC.append(("PC3 F4 with the target printed PASSes", s["verdict"] == "PASS"))
t = doc("Swapped: 6.\n")
ver = {("rc1a", "k#0"): VC.row("rc1a", "k#0", "-", "-", "6", "6", "count", True, ["shaS"], SIG)}
ctx = base_ctx(t, [], {}, rule_claims=[{"rule": "R1", "keys": ["k"], "inserted": "6", "start": t.index("6"), "end": t.index("6") + 1}],
               released_all={"k": "6"}, readout_of_slot={"k": ["rc1a"]}, verified=ver)
s, rows, _, _ = verdict(t, ctx)
PC.append(("PC4 F5 with only the swap's own position PASSes (one rule row)", s["verdict"] == "PASS" and sum(r["slot"].startswith("rule:") for r in rows) == 1))
t = doc("We use 5 folds.\n")
cons = [{"row": 1, "text": "5 folds", "class": "constant", "checks_ok": True, "anchor": "We use 5 folds."}]
s, rows, _, _ = verdict(t, base_ctx(t, [], {}, constants=cons))
PC.append(("PC5 F6 with the literal at its anchor only PASSes", s["verdict"] == "PASS"))
for n_, ok_ in PC:
    print(("PASS " if ok_ else "FAIL ") + n_)
print(f"{sum(r for _, r in OUT)} of {len(OUT)} reproduced")
sys.exit(1 if any(r for _, r in OUT) or not all(ok_ for _, ok_ in PC) else 0)
