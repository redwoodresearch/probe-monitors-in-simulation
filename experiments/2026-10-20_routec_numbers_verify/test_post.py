"""Synthetic fixtures for the v84 pass (verify_post + summarize's C1 pins); no real value, no real text.

  uv run --no-project --python 3.12 --with numpy==2.3.3 python -I test_post.py
"""
import difflib
import hashlib
import json
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verify_core as VC  # noqa: E402
import verify_post as VP  # noqa: E402

FAILS = []
PATS = VP.compile_patterns(json.loads((HERE / "config" / "v84_nonresult_patterns.json").read_text()))


def check(n, ok):
    print(("PASS " if ok else "FAIL ") + n)
    if not ok:
        FAILS.append(n)


# ---------- the swap: writing's apply re-run in a subprocess, sub() observed, positions tracked ----------
APPLY = b'''import json, sys
din, rin, dout, rep = sys.argv[1:5]
text = open(din, encoding="utf-8").read()
S = json.load(open(rin, encoding="utf-8"))["slots"]
print("a value the observer must swallow: 12.3")


def sub(rule, old, new, keys, old_sign=None, check=None):
    global text
    if text.count(old) != 1:
        raise SystemExit("anchor occurs other than once")
    text = text.replace(old, new(*[S[k] for k in keys]) if callable(new) else new)


sub("R1", "the gap was OLD points", lambda a: f"the gap was {a.split(' (')[0]} points ({a.split(' (')[1]}", ["10.delta.x C"])
exec(open(__file__.replace("apply_v84.py", "appendix_rules.py"), encoding="utf-8").read())
open(dout, "w", encoding="utf-8").write(text)
open(rep, "w", encoding="utf-8").write("ok")
'''
RULES = b'''sub("A1", "catch OLD2", lambda a: f"catch {a}", ["09.catch.x C.sim"])
sub("A2", "and OLD3 more", lambda a: f"and {a} more", ["11.n.x"])
'''
SLOTS = {"10.delta.x C": "+1.2 (â0.3 to +2.7)", "09.catch.x C.sim": "12.3%", "17.share.B": "14%", "18.raw.B": "0.14", "11.n.x": "6"}
DRAFT_IN = "# Results\n\nIn the main text the gap was OLD points.\n\nThe catch OLD2 here.\n\nThen and OLD3 more.\n"
with tempfile.TemporaryDirectory() as d:
    SWAP, LOG, SPANS, INFO = VP.rerun_apply(APPLY, RULES, DRAFT_IN.encode(), json.dumps({"slots": SLOTS}).encode(), d)
SWAP = SWAP.decode()
check("swap: apply re-run in a subprocess, sub() observed, 3 insertions tracked to their final positions",
      [x["rule"] for x in LOG] == ["R1", "A1", "A2"] and len(SPANS) == 3 and INFO == {"unattributable": 0, "spans_lost": 0, "outside_changes": 0, "output_is_final_text": True}
      and all(SWAP[s:e] == LOG[x]["inserted"] for s, e, x in SPANS))
with tempfile.TemporaryDirectory() as d:
    try:
        VP.rerun_apply(APPLY.replace(b'"the gap was OLD points"', b'"no such anchor"'), RULES, DRAFT_IN.encode(),
                       json.dumps({"slots": SLOTS}).encode(), d)
        r = False
    except VC.Stop as e:
        r = "exited" in str(e)
check("swap: a failing re-run STOPs with its exit code only (its output swallowed)", r)

HAND = """
The share was 14%. The raw share rounds to 0.14. We used 1,313 tasks. The fit used layer 43. Another 9 runs. Two probes agreed.

| Command | What it does | Probe | Setting | Above the threshold |
|---|---|---|---|---|
| `a` | writes a 12-line file | p | s | +0.42 |
| `b` | reads the 7 most recent rows | p | s | +0.31 |
| `c` | other 5 things | p | s | +0.10 |

| Task (main task, hidden side task) | Real run 1 | Simulated run |
|---|---|---|
| task A | 3.1 | 2.0 |

## Appendix A. Deferred

Run-length median was 192 steps. Another 34 attempts before any model call.

## Appendix B. More

Suppose a probe scores 62% on real runs and 60% on simulated ones.

See Figure 4 and Table 3 (2026-10-20). We use 5 folds. About 5% of tokens there. A footnote[^1]. A model L48 ran 3x faster.

[^1]: A note. Later 5 folds again, and six probes.

| Name | Count |
|---|---|
| row a | 77 |
| row b | 78 |
"""
EDITED_SWAP = SWAP.replace("and 6 more", "and 6 more (edited)")
TEXT = EDITED_SWAP + HAND
DIFF = "\n".join(difflib.unified_diff(SWAP.split("\n"), TEXT.split("\n"), "a", "b", lineterm="")) + "\n"
TGT, KEEP, ADDED, BADH = VP.apply_diff_map(SWAP, DIFF)
check("diff: applying the hand-edit diff gives the post; header counts agree", TGT == TEXT and BADH == 0)
try:
    VP.apply_diff_map(SWAP.replace("Then and 6 more.", "Then and 6 mor."), DIFF)
    r = False
except VC.Stop:
    r = True
check("diff: a diff that does not apply STOPs", r)
bad_hdr = DIFF.replace("@@ -1,3 +1,", "@@ -1,3 +2,", 1) if "@@ -1,3 +1," in DIFF else DIFF.replace("@@ -", "@@ -", 1)
check("O2: a hunk whose +start disagrees with its body is reported", VP.apply_diff_map(SWAP, bad_hdr)[3] >= 1 or "@@ -1,3 +1," not in DIFF)
CLAIMS = [{"rule": LOG[x]["rule"], "keys": LOG[x]["keys"], "inserted": LOG[x]["inserted"], "start": s, "end": e}
          for s, e, x in VP.map_spans(SWAP, KEEP, TGT, SPANS)]
check("R3: a swap position on a line the hand edit changed is dropped; the others map into the post",
      sorted(c["rule"] for c in CLAIMS) == ["A1", "R1"] and all(TEXT[c["start"]:c["end"]] == c["inserted"] for c in CLAIMS))

# ---------- verified rows of this run (synthetic) ----------
SIG = {"shaA", "shaR", "shaP"}
V = {}
for k, v in SLOTS.items():
    for i, t in enumerate(VC.NUM_RX.findall(v)):
        V[("rc1a", f"{k}#{i}")] = VC.row("rc1a", f"{k}#{i}", "-", "-", t, t, "pct", True, ["shaR"], SIG)
    V[("rc1a/slots", k)] = VC.row("rc1a/slots", k, "-", "-", v, v, "pct/summ", True, ["shaR"], SIG, unit="slot")
SLOT_ROWS = {}
for (a_, s_), r_ in V.items():
    if a_ == "rc1a":
        SLOT_ROWS.setdefault((a_, s_.rsplit("#", 1)[0]), []).append(r_)


def side(i, z, T):
    return {"run_key": f"runs/x{i}#e0", "staged_row": {"staged": "s.jsonl", "staged_sha256": "0" * 64, "arm": "run1", "line": 1, "call": 0},
            "unit": {"row_unit": 3, "action": 0, "token": 10, "irid": "i", "step": 2}, "run_eval": {"sha256": "1" * 64}, "readout_kind": "call_end",
            "field": "cmd", "length": 9, "sha256": "2" * 64, "arguments_length": 20, "arguments_sha256": "3" * 64, "z": z, "T": T, "n_units": 12}


T = {"outcome_table": {"outcomes": {"both caught": {"n": 4, "drawn": {"real (run 1)": side(9, 5.1, 2.0), "sim (sim 1)": side(8, 4.0, 2.0)}}}},
     "false_alarms": {"draws": [side(0, 2.42, 2.0), side(1, 2.31, 2.0), side(2, 2.10, 2.0)]}}
BYJ = {}
for p_, v_ in VC.numeric_leaves(T):
    jp = "/".join(map(str, p_))
    BYJ[("tables34", jp)] = VC.row("tables34", jp, "-", "-", v_, v_, "exact", True, ["shaA"], SIG, jpath=jp)
SHA = VC.sha(TEXT.encode())


def prov(row, anchor, printed, rec_sha, kp, pred, **kw):
    return {"v84_sha256": SHA, "row": row, "anchor": anchor, "printed": printed, "record": f"x/r{row}.json@abcdef1", "record_sha256": rec_sha,
            "key_path": kp, "predicate": pred, "match": "whole_token"} | kw


PROV = [
    prov(1, "The share was 14%.", "14%", "shaRC1A", ["slots", "17.share.B"], {"op": "slot_string_eq", "transform": "identity"}),
    prov(2, "rounds to 0.14", "0.14", "shaRC1A", ["slots", "18.raw.B"], {"op": "round_eq", "dp": 2, "value": 0.14}),
    prov(3, "We used 1,313 tasks.", "1,313", "shaA", ["coverage", "n"], {"op": "eq", "value": 1313}),
    prov(4, "layer 43", "43", "shaFIT", ["directions", "layer"], {"op": "eq", "value": 43}),
    prov(5, "Another 9 runs.", "9", "shaUNKNOWN", ["n"], {"op": "eq", "value": 9}),
    prov(6, "Two probes agreed.", "Two", "shaA", ["probes"], {"op": "eq", "value": 2}),
    prov(7, "| `a` | writes a 12-line file | p | s | +0.42 |", "+0.42", "shaT", [["false_alarms", "draws", 0, "z"], ["false_alarms", "draws", 0, "T"]],
         {"op": "diff_round_eq", "expr": "z - T", "dp": 2, "value": 0.42, "sign": "+"}, cell_index=5),
    prov(8, "| `b` | reads the 7 most recent rows | p | s | +0.31 |", "+0.31", "shaT", [["false_alarms", "draws", 1, "z"], ["false_alarms", "draws", 1, "T"]],
         {"op": "diff_round_eq", "expr": "z - T", "dp": 2, "value": 0.31, "sign": "+"}),
    prov(9, "| `c` | other 5 things | p | s | +0.10 |", "+0.10", "shaT", [["false_alarms", "draws", 2, "z"], ["false_alarms", "draws", 2, "T"]],
         {"op": "diff_round_eq", "expr": "z - T", "dp": 2, "value": 0.10, "sign": "+"}),
    prov(10, "| task A | 3.1 | 2.0 |", "3.1", "shaT", ["outcome_table", "outcomes", "both caught", "drawn", "real (run 1)", "z"],
         {"op": "round_eq", "dp": 1, "value": 5.1}),
]
RECS = {"shaA": {"coverage": {"n": 1313}, "probes": 2}, "shaFIT": {"directions": {"layer": 43}}, "shaUNKNOWN": {"n": 9}}
DEF = [{"row": 1, "anchor": "## Appendix A. Deferred", "scope": "section", "reason": "not route C; behaviour record", "record": [{"sha256": "x"}]},
       {"row": 27, "anchor": "About 5% of tokens there", "scope": "clause", "reason": "UNSOURCED (09 ruling)", "record": None},
       {"row": 3, "anchor": "| row a | 77", "scope": "table_row", "reason": "not route C; behaviour record", "record": [{"sha256": "x"}]}]
CONST = [{"row": 1, "text": "5 folds", "class": "constant", "checks_ok": True, "anchor": "We use 5 folds."},
         {"row": 2, "text": "62% on real runs and 60%", "class": "hypothetical", "checks_ok": True,
          "anchor": "Suppose a probe scores 62% on real runs and 60% on simulated ones.",
          "anchor_sentence": "Suppose a probe scores 62% on real runs and 60% on simulated ones."}]
HYP = {"2": {"expressions": [["62 - 60", 2]], "percent_reading": "points", "text_sha256": hashlib.sha256(b"62% on real runs and 60%").hexdigest()}}
NOTES = [{"near": "before any model call", "note": "pending 0d's committed count"}]
H4 = VP.TABLE_HEADERS["Table 4"]
HL = [{"table": 2, "header_line": H4, "cell_index": 2, "cell": "Real run 1", "token": "1", "position": TEXT.index(H4) + H4.index("run 1") + 4}]
CALLS = []


def cmd_ok(side_, form, n):
    CALLS.append((form, n))
    return {"line_count": 12, "limit": 7}[form] == n


NP = sorted(f"false_alarms/draws/{i}/{k}" for i in range(3) for k in ("length", "arguments_length", "n_units", "unit/row_unit"))
ACC = ["not route C; behaviour record"]


def ctx(**kw):
    c = {"text_sha": SHA, "provenance": PROV, "deferred": DEF, "deferred_record_ok": {0: True, 1: False, 2: True}, "constants": CONST,
         "hyp_exprs": HYP, "rule_claims": CLAIMS, "released_all": SLOTS, "readout_of_slot": {k: ["rc1a"] for k in SLOTS}, "verified": V,
         "slot_rows": SLOT_ROWS, "by_jpath": BYJ, "artifact_of_sha": {"shaRC1A": "rc1a", "shaT": "tables34"},
         "kind_of": {"rc1a": "slots", "tables34": "tables34"}, "released": {"rc1a": SLOTS}, "slot_raw": {"rc1a": {"18.raw.B": 0.1404}},
         "records": RECS, "signed": SIG, "not_route_C": {"shaFIT"}, "T": T, "not_printed": NP, "not_printed_v84_approved": None,
         "command_check": cmd_ok, "added_spans": ADDED, "orphan_notes": NOTES, "patterns": PATS, "header_labels": HL, "accepted_reasons": ACC}
    return c | kw


rows, plans, stops, dg = VP.post_rows("post", TEXT, ctx())
by = {r["slot"]: r for r in rows}
PINS = {"accepted_reasons": ACC}


def summ(rows_, plans_, stops_, pins=PINS):
    p_ = dict(pins)
    _, s_, n_ = VC.nonresult_list(rows_)
    p_["nonresult_approved"] = {"sha256": s_, "count": n_}
    return VC.summarize(rows_, plans_, stops_, pins=p_)


def slot_of(prefix, txt, k=0):
    s = TEXT.index(txt)
    if k:
        s = TEXT.index(txt, s + 1)
    return next(x for x in by if x.startswith(prefix) and x.endswith(f"@{s}"))


def at(txt, off=0):
    s = TEXT.index(txt) + off
    return next(r for r in rows if r["artifact"] == "post" and r["slot"].endswith(f"@{s}"))


check("coverage: every token has one row, every visible digit lies in a token (an independent count)",
      plans["post"] == sum(r["artifact"] == "post" for r in rows) and dg["digits_outside_tokens"] == 0 and dg["visible_digits"] > 0)
check("R5: glued tokens get rows ('L48' an identifier, '3x' an orphan)", at("L48", 1)["nr_class"] == "identifier" and not at("3x faster")["equal"]
      and dg["tokens_by_kind"].get("glued_before", 0) >= 1 and dg["tokens_by_kind"].get("glued_after", 0) >= 1)
check("R3: the swap's tracked positions are rule rows (verified slot numbers)", at("+1.2 points")["slot"].startswith("rule:R1")
      and at("12.3%")["slot"].startswith("rule:A1") and at("+1.2 points")["equal"])
check("R3: a rule number on a line the hand edit changed is not a rule row (its position was dropped)", not at("and 6 more", 4)["slot"].startswith("rule"))
check("map: slot_string_eq, a raw-slot predicate, a signed record by key and a Tables 3/4 field all verify",
      all(at(t_)["equal"] and at(t_)["slot"].startswith("map:") for t_ in ("14%", "0.14", "1,313", "+0.42")))
check("map: a configured not-route-C record is checked and counted apart (unit not_route_C)", at("43")["unit"] == "not_route_C" and at("43")["equal"])
check("map: a record that is neither signed nor not-route-C fails its row (orphan bucket)", not at("9 runs")["equal"])
check("map: a words-only provenance row gets its own row", by["map:6@words"]["equal"] and plans["post"] == dg["tokens"] + 1)
check("R2: a predicate whose expected value is not what the post prints fails (record 5.1, printed 3.1)",
      not at("3.1")["equal"] and "not what the post prints" in next(r for r in rows if r["slot"] == "map:10")["note"])
check("C3: the line count and the limit are command-content rows, one boolean each", CALLS == [("line_count", 12), ("limit", 7)] and
      at("12-line")["equal"] and at("7 most recent")["equal"])
check("RQ3: a Table 3 body number with no map or command row is an orphan", at("5 things")["slot"].startswith("Table3@") and not at("5 things")["equal"])
check("RQ3: an unmapped Table 4 body number is an orphan", at("2.0 |")["slot"].startswith("Table4@"))
check("09 header labels: 'Real run 1' in Table 4's HEADER row is a header label at its listed position (not an RQ3 orphan)",
      at(H4, H4.index("run 1") + 4)["nr_class"] == "header_label" and dg["header_labels_matched"] == 1)
check("C2: drawn-side fields equal to a printed Table 3/4 number are reported by name and leave the not-printed list",
      "false_alarms/draws/0/n_units" in dg["not_printed_v84"]["fields_printed"] and dg["not_printed_v84"]["count"] == len(NP) - 1)
check("C2: without a fresh v84 approval the pass STOPs", any("not-printed list for this pass" in x for x in stops))
appr = {"sha256": dg["not_printed_v84"]["sha256"], "count": dg["not_printed_v84"]["count"], "scope": "v84"}
_, _, st2, _ = VP.post_rows("post", TEXT, ctx(not_printed_v84_approved=appr))
check("C2: with 09's approval of the regenerated list it passes that gate", not any("not-printed list" in x for x in st2))
_, _, st3, _ = VP.post_rows("post", TEXT, ctx(not_printed_v84_approved=appr | {"scope": "record-pass-only; provisional"}))
check("C2: the record-pass-only approval is refused", any("not-printed list for this pass" in x for x in st3))
check("deferred: a number in an accepted region with records read is a deferred row", at("192")["unit"] == "deferred" and at("192")["record_ok"])
check("09 precedence: a number with an explicit orphan note inside a deferred region is an ORPHAN, with the note",
      at("34 attempts")["note"] == "pending 0d's committed count" and not at("34 attempts")["equal"])
r5 = at("5% of tokens")
check("09: a null-record clause row with an unaccepted reason: reported with its reason and writing's row number; the pass continues",
      r5["reason"] == "UNSOURCED (09 ruling)" and r5["record_ok"] is False and r5["deferred_row"] == 27 and "#27@" in r5["slot"])
check("deferred: a table_row region is the whole line its anchor starts (77 deferred, 78 an orphan)",
      at("77")["unit"] == "deferred" and not at("78")["equal"])
_, _, st9, _ = VP.post_rows("post", TEXT, ctx(deferred=DEF + [{"row": 9, "anchor": "nowhere in the text", "scope": "paragraph", "reason": "x",
                                                                  "record": None}]))
check("deferred: a row whose anchor is missing gets no region and STOPs, but the pass continues", any("deferred row 9" in x for x in st9))
check("hypothetical: the worked example at its anchor is its own class with 09's three checks", at("62%")["checks_ok"] is True)
check("R4: a constants literal at its anchor is a constant; the same literal elsewhere is an orphan",
      at("5 folds")["nr_class"] == "constant" and not at("5 folds again")["equal"] and at("5 folds again")["slot"].startswith("orphan@"))
check("R9: references, a date and footnote markers come from 09's pinned patterns",
      {c: dg["tokens_by_class"].get(c, 0) for c in ("reference", "date", "footnote_marker")} == {"reference": 2, "date": 3, "footnote_marker": 2})
check("non-result list entries carry positions", all(isinstance(r.get("pos"), int) for r in rows if r.get("unit") == "nonresult"))
s = summ(rows, plans, stops)
check("verdict: STOP, naming the orphans and the unmatched rows; no duplicate slots, no count mismatch",
      s["verdict"].startswith("STOP") and f"post::{r5['slot']}" in s["stop"]["orphan"] and "post/map::map:5" in s["stop"]["unmatched"] + s["stop"]["orphan"]
      and not s["stop"]["duplicate_slots"] and not s["stop"]["count_mismatch"])
check("R7: the verdict line names the not-route-C bucket", "1 not route C, checked against unsigned records by key" in s["verdict"])
check("summary: deferred rows counted apart; the not-route-C value apart; non-results apart",
      s["n_not_verified_deferred"] == 2 and s["n_not_route_C_checked"] >= 1 and s["n_nonresult"] >= 8)
check("09 (10-20) accounting: per artifact the classes are a PARTITION of the printed rows (sum == n_printed); unmatched and orphan are "
      "sub-counts of 'failed' (a row both unmatched and orphan counts once)",
      all(sum(v for k_, v in d_.get("partition", {}).items() if not k_.startswith("failed_")) == d_["n_printed"] for d_ in s["per_artifact"].values())
      and all(d_.get("partition", {}).get(k_, 0) <= d_.get("partition", {}).get("failed", 0) for d_ in s["per_artifact"].values()
              for k_ in ("failed_unmatched", "failed_orphan"))
      and s["per_artifact"]["post"]["partition"].get("failed", 0) > 0 and not any("partition" in x for x in s["stop"]["other"]))
RI = VP.residual_items(s["stop"], "post", {str(TEXT.index("5 folds again")): "09 note for this position"})
check("09 (10-20) residual_items: every unmatched or orphan slot of the post once, by kind, position, row and note (09's note by "
      "position from config, else the kind's note); no text",
      len(RI) == len({(x["kind"], x["position"], x["row"]) for x in RI}) and all(set(x) == {"kind", "position", "row", "note"} for x in RI)
      and any(x["position"] == TEXT.index("5 folds again") and x["note"] == "09 note for this position" for x in RI)
      and any(x["kind"] == "map" and x["row"] == 5 for x in RI) and any(x["kind"] == "Table3" and x["note"] == "Tables 3/4 body cell: position only"
                                                                         for x in RI))
check("09 words: number words are listed with class, digit-masked sentence and covering row",
      dg["number_words"]["total"] >= 2 and any(x["word"] == "six" and x["class"] == "cardinal_noun" and x["covered_by"] == "NONE"
                                                for x in dg["number_words"]["list"])
      and any(x["word"] == "two" and x["covered_by"] == "map row 6" for x in dg["number_words"]["list"])
      and all("#" in x["sentence"] or not any(ch.isdigit() for ch in x["sentence"]) for x in dg["number_words"]["list"]))
# 2e / 09 (10-20): no Tables 3/4 text in the number-word list (fail closed); numv5's positions checked by key
T3ROW = "| `c` | other 5 things | p | s | +0.10 |\n"
TXW = TEXT.replace(T3ROW, T3ROW + "| `d` | reads one file | p | s | - |\n", 1)
_, _, _, dgw = VP.post_rows("post", TXW, ctx())
w_in = [x for x in dgw["number_words"]["list"] if x["position"] == TXW.index("reads one file") + len("reads ")]
check("2e/09: a number word in a Table 3 body row has no sentence and no word in the output (table, row, position, covering row only); counted",
      TEXT.count(T3ROW) == 1 and len(w_in) == 1 and set(w_in[0]) == {"position", "table", "row", "covered_by", "class"}
      and (w_in[0]["table"], w_in[0]["row"]) == ("Table 3", 4) and dgw["number_words"]["tables34_sentence_withheld"] == 1
      and dgw["number_words"]["all_sentences_withheld_tables34_unknown"] is False
      and any(x.get("word") == "two" and "sentence" in x for x in dgw["number_words"]["list"]))
check("2e/09: with no number word in Tables 3/4 nothing is withheld",
      dg["number_words"]["tables34_sentence_withheld"] == 0 and all("sentence" in x for x in dg["number_words"]["list"]))
TXU = TEXT.replace(H4, H4.replace("Simulated run", "Simulated runs"))
_, _, stu, dgu = VP.post_rows("post", TXU, ctx(numv5_positions=[0, 1]))
check("2e/09 fail closed: Tables 3/4 boundaries unknown -> EVERY number-word entry withheld; numv5's check is 'unknown' and STOPs",
      len(dgu["number_words"]["list"]) >= 2 and all("sentence" not in x and "word" not in x for x in dgu["number_words"]["list"])
      and dgu["number_words"]["all_sentences_withheld_tables34_unknown"] is True
      and dgu["numv5_number_word_positions"]["status"] == "unknown" and any("numv5's number-word positions" in x for x in stu))
BLK = {nm: VP.table_block(TEXT, h) for nm, h in VP.TABLE_HEADERS.items()}
p_in3, p_in4h, p_out = TEXT.index("| `b` |") + 3, TEXT.index(H4) + 2, TEXT.index("Two probes agreed.")
check("09 numv5 positions: inside a table body row / on a header line -> true; outside, or just past a block's end -> false",
      [VP.tables34_where(BLK, p) is not None for p in (p_in3, p_in4h, p_out, BLK["Table 3"][1], BLK["Table 4"][1])]
      == [True, True, False, False, False] and VP.tables34_where(BLK, p_in4h) == ("Table 4", "header"))
_, _, st5, dg5n = VP.post_rows("post", TEXT, ctx(numv5_positions=[p_out, p_in4h, p_in3], numv5_source="x@y"))
n5 = dg5n["numv5_number_word_positions"]
check("09 numv5 positions: count, inside count and the inside positions by key; a disclosure line when any is inside",
      (n5["count"], n5["inside_tables34"], n5["positions_inside"], n5["source"]) == (3, 2, sorted([p_in3, p_in4h]), "x@y")
      and n5["status"] == "disclosure: numv5's result file carries 2 sentence snippets from Tables 3/4, digits masked; unread"
      and not any("numv5" in x for x in st5))
check("09 numv5 positions: all outside -> cleared", VP.numv5_positions_check(BLK, [p_out])["status"] == "cleared"
      and "numv5_number_word_positions" not in dg)
# R10 / hand edits
check("R10: numbers on hand-edited lines count only when their row PASSES (orphans and failing rows are named check rows)",
      dg["hand_edit_numbers"]["uncovered"] > 0 and plans["post/hand_edit"] == dg["hand_edit_numbers"]["uncovered"]
      and any(r["slot"] == f"hand_edit@{TEXT.index('9 runs')}" for r in rows if r["artifact"] == "post/hand_edit"))
_, _, _, dg4 = VP.post_rows("post", TEXT, ctx(added_spans=[(TEXT.index("The share was"), TEXT.index("The share was") + 18)]))
check("hand edit (positive control): a line holding only a passing map number is covered", dg4["hand_edit_numbers"] == {"total": 1, "uncovered": 0,
                                                                                                                       "by_class": {"map": 1}})
_, _, _, dg5 = VP.post_rows("post", TEXT, ctx(command_check=lambda *a: False, added_spans=[(TEXT.index("| `a` |"), TEXT.index("| `a` |") + 45)]))
check("R10: a FAILING command count on a hand-edited line is not covered", dg5["hand_edit_numbers"]["uncovered"] >= 1)
# R8
check("R8: Tables 3/4's printed flag is False while a body number is an orphan or a gate stops", dg["tables34_printed_numbers_verified"] is False)
# R1 / R2 in the post (full fixtures in test_review_regressions.py)
t2 = TEXT.replace("Another 9 runs.", "Another 19 runs.")
P5 = [prov(5, "Another 19 runs.", "9", "shaA", ["n9"], {"op": "eq", "value": 9}) | {"v84_sha256": VC.sha(t2.encode())}]
rows6, _, _, _ = VP.post_rows("post", t2, ctx(text_sha=VC.sha(t2.encode()), provenance=P5, records=RECS | {"shaA": RECS["shaA"] | {"n9": 9}}))
check("R1: printed '9' does not verify the post's '19' (whole tokens)", not any(r["slot"].startswith("map:5@") for r in rows6))

def pclass(txt, tok):
    s = txt.rindex(tok)
    k = VP.tokens(txt)[1].get(s, "number")
    r = VP.pattern_class(txt, s, s + len(tok), tok, k, PATS)
    return r[0] if r else None


check("R9 (09, 10-20): dash separators only without whitespace: 'Table 7 – 0.52' is not a reference; 'Tables 3–4', 'Tables 3 and 4' are",
      pclass("See Table 7 – 0.52 here.", "0.52") is None and pclass("See Tables 3–4 here.", "4") == "reference"
      and pclass("See Tables 3 and 4 here.", "4") == "reference" and pclass("See Tables 3 - 4 here.", "4") is None
      and pclass("In Section 4, 12 of 20 runs.", "12") is None)
PD = [PROV[0] | {"anchor": "14%"}]
t9 = TEXT.replace("Another 9 runs.", "Another 9 runs at 14%.")
rows9, _, _, _ = VP.post_rows("post", t9, ctx(text_sha=VC.sha(t9.encode()), provenance=[p | {"v84_sha256": VC.sha(t9.encode())} for p in PD]))
check("09 (row 46): a provenance anchor that is not unique in the post fails its row",
      "anchor does not occur exactly once" in next(r for r in rows9 if r["slot"] == "map:1")["note"])

# ---------- round 2 (Q1-Q3, S1, N2, N3, token-level R2, args declarations) ----------
APPLY2 = b'''import json, sys
din, rin, dout, rep = sys.argv[1:5]
text = open(din, encoding="utf-8").read()
S = json.load(open(rin, encoding="utf-8"))["slots"]


def sub(rule, old, new, keys):
    global text
    text = text.replace(old, new(S))


sub("R1", "OLD", lambda S: S["k"], ["k"])
text = text.replace("P.", "P. Q 6")
exec(open(__file__.replace("apply_v84.py", "appendix_rules.py"), encoding="utf-8").read())
open(dout, "w", encoding="utf-8").write(text)
open(rep, "w", encoding="utf-8").write("ok")
'''
with tempfile.TemporaryDirectory() as d:
    out2, log2, sp2, inf2 = VP.rerun_apply(APPLY2, b"", b"P.\nQ OLD.\n", json.dumps({"slots": {"k": "6"}}).encode(), d)
b2 = out2.decode()
check("Q1: an edit OUTSIDE sub() shifts the span onto the number sub() wrote (never a stale span on another '6'); it is counted",
      [(s, e) for s, e, _ in sp2] == [(b2.index("Q 6.") + 2, b2.index("Q 6.") + 3)] and inf2["outside_changes"] == 1 and inf2["spans_lost"] == 0)
APPLY3 = APPLY2.replace(b'text = text.replace("P.", "P. Q 6")', b'text = text.replace("Q 6", "Q 6 6")')
with tempfile.TemporaryDirectory() as d:
    out3, log3, sp3, inf3 = VP.rerun_apply(APPLY3, b"", b"P.\nQ OLD.\n", json.dumps({"slots": {"k": "6"}}).encode(), d)
check("Q1: an outside edit touching the span drops it (lost, counted), never misplaces it", sp3 == [] and inf3["spans_lost"] == 1)
APPLY4 = APPLY2.replace(b'open(dout, "w", encoding="utf-8").write(text)', b'open(dout, "w", encoding="utf-8").write(text + "x")')
with tempfile.TemporaryDirectory() as d:
    out4, log4, sp4, inf4 = VP.rerun_apply(APPLY4, b"", b"P.\nQ OLD.\n", json.dumps({"slots": {"k": "6"}}).encode(), d)
check("Q1: a written output that is not the final text keeps no span and is reported", sp4 == [] and inf4["output_is_final_text"] is False)
gen, _, _ = VC.nonresult_list([VC.row("p", f"reference@{i}", "-", "Table 3", "3", None, "reference", True, ["nonresult"], {"nonresult"},
                                      unit="nonresult", nr_class="reference", matched="Table 3", pos=i) for i in (10, 90)])
check("Q2: two identical references are two list entries (each carries its position)", len(set(gen)) == 2 and all(g.endswith(("\t10", "\t90")) for g in gen))
okr = VC.row("post", "map:1@5", "-", "-", "4", None, "eq", True, ["shaA"], {"shaA"})
nrr = VC.row("post", "map:2@9", "-", "-", "43", None, "x", True, ["not-route-C"], {"not-route-C"}, unit="not_route_C")
chk_ = VC.row("post/map", "map:2", "-", "-", "<1 of 1 tokens>", None, "eq", True, ["not-route-C"], {"not-route-C"}, unit="not_route_C",
              check_row=True)
s_ = VC.summarize([okr, nrr, chk_], {"post": 2, "post/map": 1})
sc_, pf_ = VC.result_scope({"post": "post"}, {"post": {}}, (), None, s_)
check("Q3: a PASS with a not-route-C number: printed flag False by the stated rule (not by string mismatch); the entry counts 1, not 2",
      s_["verdict_base"] == "PASS" and pf_["post"] is False and "1 not route C" in sc_["post"] and "2 not route C" not in sc_["post"])
s_ = VC.summarize([okr], {"post": 1})
check("Q3: a plain PASS sets the post's printed flag", VC.result_scope({"post": "post"}, {"post": {}}, (), None, s_)[1]["post"] is True)
T2 = "Rates were 12 and 34 here.\n"
P2 = [{"v84_sha256": VC.sha(T2.encode()), "row": 1, "anchor": "Rates were 12 and 34 here.", "printed": "12 and 34", "record": "x@abcdef1",
       "record_sha256": "shaA", "key_path": ["a"], "predicate": {"op": "eq", "value": 12}},
      {"v84_sha256": VC.sha(T2.encode()), "row": 2, "anchor": "Rates were 12 and 34 here.", "printed": "34", "record": "x@abcdef1",
       "record_sha256": "shaA", "key_path": ["b"], "predicate": {"op": "eq", "value": 34}}]
c2 = ctx(text_sha=VC.sha(T2.encode()), provenance=P2, records={"shaA": {"a": 12, "b": 34}}, rule_claims=[], added_spans=[], header_labels=[],
         constants=[], deferred=[], not_printed=[], not_printed_v84_approved={"sha256": VC.sha(b""), "count": 0, "scope": "v84"})
r2_, p2_, st2_, d2_ = VP.post_rows("post", T2, c2)
s2_ = summ(r2_, p2_, [x for x in st2_ if "header line" not in x])
check("token-level R2: a sentence with 2 numbers, one row claiming each -> both verified; the wider span claims only its own value",
      s2_["verdict_base"] == "PASS" and d2_["map_spans"]["tokens_left_unclaimed_in_spans"] == 1)
r3_, p3_, st3_, _ = VP.post_rows("post", T2, c2 | {"provenance": P2[:1]})
s3_ = summ(r3_, p3_, [x for x in st3_ if "header line" not in x])
check("token-level R2: remove one row -> its token is an ORPHAN (STOP)", s3_["verdict_base"] == "STOP" and
      any(r["slot"] == f"orphan@{T2.index('34')}" for r in r3_))
T3 = "At 1 in 100 the 2-point fall was caught; at 1 in 30 it was not.\n"
AR = [{"text": "1", "value": 1, "role": "alarm level numerator (1 in 100)"}, {"text": "100", "value": 100, "role": "alarm level denominator"},
      {"text": "2", "value": 2, "role": "planted fall size (points)"}, {"text": "1", "value": 1, "role": "alarm level numerator (1 in 30)"},
      {"text": "30", "value": 30, "role": "alarm level denominator"}]
P3 = [{"v84_sha256": VC.sha(T3.encode()), "row": 1, "anchor": T3.strip(), "printed": T3.strip(), "record": "x@abcdef1", "record_sha256": "shaA",
       "key_path": [["fall_2pt", "0.01"], ["fall_2pt", "0.05"]], "predicate": {"op": "verdicts_eq", "levels": ["0.01", "0.05"],
                                                                               "values": ["detected", "not detected"], "args": AR}}]
recs3 = {"shaA": {"fall_2pt": {"0.01": "detected", "0.05": "not detected"}}}
c3 = c2 | {"text_sha": VC.sha(T3.encode()), "provenance": P3, "records": recs3}
r4_, p4_, st4_, d4_ = VP.post_rows("post", T3, c3)
cl4 = {r["printed"] for r in r4_ if r["slot"].startswith("map:1@")}
check("args: a numerator / denominator pair whose ratio is a level, and a planted size in the key path, are claimed; a pair whose ratio "
      "is no level (1 in 30) is not: both its tokens are orphans", {"1", "100", "2"} <= cl4
      and any(r["slot"] == f"orphan@{T3.index('30')}" for r in r4_) and any(r["slot"] == f"orphan@{T3.index('1 in 30')}" for r in r4_))
check("predicates: the 'args' declarations are not predicate inputs (check ignores them)",
      VP.PR.check(P3[0]["predicate"], [recs3["shaA"]["fall_2pt"]["0.01"], recs3["shaA"]["fall_2pt"]["0.05"]])[0] is True)
for nm, txt, pr, rec in (("N2a abs_round_eq without sign: '+2.5' for record -2.5", "The gap was +2.5 points.\n",
                          {"op": "abs_round_eq", "dp": 1, "value": 2.5}, -2.5),
                         ("N2b diff_round_eq without sign: '−0.42'", "z - T was −0.42 there.\n",
                          {"op": "diff_round_eq", "expr": "z - T", "dp": 2, "value": -0.42}, None),
                         ("N3 bound 0.05 accounting a bare '5' (no %)", "All stay within 5 of it.\n", {"op": "max_abs_le", "bound": 0.05}, None)):
    pr_tok = {"N2a": "+2.5", "N2b": "−0.42", "N3 ": "5"}[nm[:3]]
    kp = [["z"], ["T"]] if nm.startswith("N2b") else (["xs"] if nm.startswith("N3") else ["d"])
    rec_ = {"z": 2.42, "T": 2.0} if nm.startswith("N2b") else ({"xs": [0.03, -0.04]} if nm.startswith("N3") else {"d": rec})
    Pn = [{"v84_sha256": VC.sha(txt.encode()), "row": 1, "anchor": txt.strip(), "printed": pr_tok, "record": "x@abcdef1", "record_sha256": "shaA",
           "key_path": kp, "predicate": pr}]
    rn, pn, stn, _ = VP.post_rows("post", txt, c2 | {"text_sha": VC.sha(txt.encode()), "provenance": Pn, "records": {"shaA": rec_}})
    check(f"{nm}: STOP", summ(rn, pn, [x for x in stn if "header line" not in x])["verdict_base"] == "STOP")
T4 = "The fit has R² of 0.9 here.¹ And x³ again.\n"
tk4, kd4 = VP.tokens(T4)
c4 = c2 | {"text_sha": VC.sha(T4.encode()), "provenance": []}
r5_, _, st5_, d5_ = VP.post_rows("post", T4, c4)
check("S1: superscripts are tokens ('R²', 'x³' identifiers; a lone '¹' needs a row: orphan); every visible digit is in a token",
      d5_["digits_outside_tokens"] == 0 and next(r for r in r5_ if r["slot"].endswith(f"@{T4.index('²')}"))["nr_class"] == "identifier"
      and any(r["slot"] == f"orphan@{T4.index('¹')}" for r in r5_) and VP._num("³") == 3.0)

# ---------- after the first real run (numv4): 09's row-74 class waiver, upstream sources ----------
for nm, txt, printed, rec, want in (("waiver: a span with 0 digit tokens and a number word runs record-vs-target; the link is reported, not gated",
                                     "It came to about a fifth of them.\n", "about a fifth", 0.21, "PASS"),
                                    ("waiver keeps the record check: same words, record far from the target -> STOP",
                                     "It came to about a fifth of them.\n", "about a fifth", 0.5, "STOP"),
                                    ("waiver is by class, not by row: a digit token printed with a wrong value still STOPs",
                                     "It came to about 0.3 of them.\n", "0.3", 0.21, "STOP")):
    Pw = [{"v84_sha256": VC.sha(txt.encode()), "row": 74, "anchor": txt.strip(), "printed": printed, "record": "x@abcdef1",
           "record_sha256": "shaA", "key_path": ["x"], "predicate": {"op": "abs_diff_le", "target": 0.2, "tol": 0.025}}]
    rw, pw, stw, dw = VP.post_rows("post", txt, c2 | {"text_sha": VC.sha(txt.encode()), "provenance": Pw, "records": {"shaA": {"x": rec}}})
    sw = summ(rw, pw, [x for x in stw if "header line" not in x])
    note_w = dw["map_rows"]["74"]["note"]
    check(f"09 row 74: {nm}", sw["verdict_base"] == want and (want == "STOP" or "printed_link: number word, not gated" in note_w))
TU = "| `a` | x | p | s | 43 |\n"
Pu = [{"v84_sha256": VC.sha(T2.encode()), "row": 1, "anchor": "Rates were 12 and 34 here.", "printed": "12", "record": "x@abcdef1",
       "record_sha256": "shaT", "key_path": ["false_alarms", "draws", 0, "z"], "predicate": {"op": "eq", "value": 12}}]
byj_u = {("tables34", "false_alarms/draws/0/z"): VC.row("tables34", "false_alarms/draws/0/z", "-", "-", 12, 12, "exact", True, ["shaD1"], {"shaD1"},
                                                          jpath="false_alarms/draws/0/z")}
cu = c2 | {"provenance": Pu, "by_jpath": byj_u, "artifact_of_sha": {"shaT": "tables34"}, "kind_of": {"tables34": "tables34"}}
ru1, _, _, _ = VP.post_rows("post", T2, cu)
ru2, _, _, _ = VP.post_rows("post", T2, cu | {"upstream_sources": {"shaD1"}})
check("upstream sources: a Tables 3/4 field verified on a D1 file is an orphan without the run's accepted sources, verified with them",
      next(r for r in ru1 if r["slot"] == "map:1")["orphan"] and not next(r for r in ru2 if r["slot"] == "map:1")["orphan"])
check("map_rows: a keys-only status per provenance row (op, ok, bucket, note, claimed / span counts)",
      set(dw["map_rows"]["74"]) == {"op", "ok", "bucket", "note", "claimed", "span"})

# ---------- 09 (10-20): command definitions in the post, no count or text out; the non-result emitter; R4-1 ----------
DIG = lambda self, a: {"field": "cmd", "length": 9, "sha256": "2" * 64, "arguments_length": 20, "arguments_sha256": "3" * 64}  # noqa: E731
SEC7 = "cat > f <<'EOF'\n" + "\n".join(f"SECRET_LINE_{i}" for i in range(7)) + "\nEOF\n"
gx = type("GX", (), {"arguments_at": lambda self, e: {"args": {"cmd": SEC7}}, "digests": DIG,
                     "command_of": lambda self, a: ("cmd", a["cmd"], json.dumps(a))})()
rX, pX, sX, dX = VP.post_rows("post", TEXT, ctx(command_check=lambda s_, f_, n_: VP.command_equal(s_, f_, n_, gx, {})))
blobX = json.dumps([rX, pX, sX, dX], default=str)
check("09 (b) none True in the post: the row fails, a FINDING names the table and cell; only definition names and booleans leave "
      "(no count, line or text of the command anywhere)",
      any("line count matches no tested definition" in f for f in dX["findings"]) and "SECRET_LINE" not in blobX
      and all(set(x) == {"cell", "form", "true", "tested", "definitions"} and [d["name"] for d in x["definitions"]] == list(VP.ALL_DEFS)
              for x in dX["command_definitions"])
      and all(set(x["true"]) <= set(x["tested"]) <= {"splitlines", "newline_count", "newline_count_plus_1", "command_lines_without_heredoc_delimiters",
                                                     "integer_after_LIMIT"} for x in dX["command_definitions"]))
gc = type("GC", (), {"arguments_at": lambda self, e: {"args": {"cmd": "cat > f <<'EOF'\n" + "\n".join("x" * 12) + "\nEOF\n"}},
                     "digests": DIG, "command_of": lambda self, a: ("cmd", a["cmd"], json.dumps(a))})()
rC, _, _, dC = VP.post_rows("post", TEXT, ctx(command_check=lambda s_, f_, n_: VP.command_equal(s_, f_, n_, gc, {}) if f_ == "line_count"
                                              else (True, {"integer_after_LIMIT": True})))
check("09 (b) covered: the row names its definition(s) in the result", "covered under:" in at("12-line")["note"] if False else
      any(r["note"].startswith("covered under:") for r in rC if r["slot"].startswith("command@")) and not dC["findings"])
_, s_l, n_l = VC.nonresult_list(rows)
ent = dg["nonresult_entries"]
txt_, srt_ = VC.emit_nonresult_list(ent, s_l, n_l)
check("emitter: the list file's sha256 equals the list sha the run reports, by construction; entries carry class, token, matched, "
      "position and classifier", VC.sha(txt_) == s_l and len(srt_) == n_l and all({"class", "token", "matched", "pos", "by"} <= set(e) for e in ent))
for nm, bad_ in (("an altered matched text", [dict(ent[0], matched=ent[0]["matched"] + "x", line=ent[0]["line"].replace("\t", "x\t", 1))] + ent[1:]),
                 ("a dropped entry", ent[1:]), ("a line that is not class / matched / position", [dict(ent[0], line="x")] + ent[1:])):
    try:
        VC.emit_nonresult_list(bad_, s_l, n_l)
        r = False
    except VC.Stop:
        r = True
    check(f"emitter negative: {nm} -> STOP", r)
TH = "Intro.\n\n| Model | Size 27 |\n|---|---|\n| m | x |\n"
HLx = [{"table": 1, "header_line": "| Model | Size 27 |", "cell_index": 2, "cell": "Size 27", "token": "27", "position": TH.index("27")}]
CNx = [{"row": 9, "text": "Size 27", "class": "constant", "checks_ok": True, "anchor": "| Model | Size 27 |"}]
_, _, _, dH = VP.post_rows("post", TH, c2 | {"text_sha": VC.sha(TH.encode()), "provenance": [], "header_labels": HLx, "constants": CNx})
check("header labels taken by another class are counted with their positions (constants come first)",
      dH["header_labels_taken_by_other_classes"] == {"count": 1, "positions": [TH.index("27")], "by_class": {"constant": 1}})
for nm, txt, printed, pred, rec in (("W1 'three probes', expected 4, record 4", "We used three probes here.\n", "three probes",
                                     {"op": "eq", "value": 4}, 4),
                                    ("W2 'every one of the runs', count 12", "It held for every one of the runs.\n", "every one of the runs",
                                     {"op": "eq", "value": 12}, 12)):
    Pq = [{"v84_sha256": VC.sha(txt.encode()), "row": 9, "anchor": txt.strip(), "printed": printed, "record": "x@abcdef1", "record_sha256": "shaA",
           "key_path": ["x"], "predicate": pred}]
    rq, pq, stq, dq = VP.post_rows("post", txt, c2 | {"text_sha": VC.sha(txt.encode()), "provenance": Pq, "records": {"shaA": {"x": rec}}})
    check(f"R4-1 negative: {nm}: a readable number word is NOT waived -> STOP",
          summ(rq, pq, [x for x in stq if "header line" not in x])["verdict_base"] == "STOP" and "not gated" not in dq["map_rows"]["9"]["note"])

# ---------- numv6 (v12 / v13): literal classes, PRELIMINARY bucket, ops 23 / 24 in the post, record string cells, I1 / I2 ----------
LJ = json.loads((HERE / "config" / "v84_nonresult_literals.json").read_text())
TL = "In real run 1 caught 12 runs; Run 2 too. We used Qwen3.5-27B and GPT-6-Astra 7 times. [FIGURE 3. 45 runs] Hit localhost:80 9.\n"
LJx = json.loads(json.dumps(LJ))
LJx["classes"]["command_literal"]["entries"] = [{"literal": "localhost:80", "position": TL.index("localhost:80")}]
LJx["classes"]["citation_literal"]["entries"] = []
lits_ = VP.compile_literals(LJx)
cL = c2 | {"text_sha": VC.sha(TL.encode()), "provenance": [], "literals": lits_}
rL, _, stL, dL = VP.post_rows("post", TL, cL)
cls_at = {int(r["slot"].rsplit("@", 1)[1]): r.get("nr_class") or r["slot"].split("@")[0] for r in rL if r["artifact"] == "post"}
def cat(sub, off=0):
    return cls_at[TL.index(sub) + off]
check("09 literal classes claim their own tokens: run labels, identifier literals, a placeholder, a pinned command literal",
      cat("run 1", 4) == "run_label" and cat("Run 2", 4) == "run_label" and cat("Qwen3.5-27B", 4) == "identifier_literal"
      and cat("Qwen3.5-27B", 8) == "identifier_literal" and cat("GPT-6-Astra", 4) == "identifier_literal" and cat("[FIGURE 3.", 8) == "placeholder"
      and cat("localhost:80", 10) == "command_literal")
check("09 literal classes never swallow a following number: '12' after 'run 1 caught', '7' after GPT-6-Astra, '45' after the placeholder, "
      "'9' after the command literal are orphans", all(cat(x) == "orphan" for x in ("12 runs", "7 times", "45 runs", "9.")))
LJy = json.loads(json.dumps(LJx))
LJy["classes"]["command_literal"]["entries"] = [{"literal": "localhost:80", "position": TL.index("localhost:80") + 1}]
_, _, stY, _ = VP.post_rows("post", TL, cL | {"literals": VP.compile_literals(LJy)})
check("09 literal classes: a pinned literal that is not at its position STOPs", any("not at its pinned position" in x for x in stY))
try:
    VP.compile_literals({"classes": {"run_label": LJ["classes"]["run_label"]}})
    r = False
except VC.Stop:
    r = True
check("09 literal classes: a literal file without exactly the five classes STOPs", r)
TP = "The honest AUROC was 0.71 here.\n"
PP = [{"v84_sha256": VC.sha(TP.encode()), "row": 75, "anchor": TP.strip(), "printed": "0.71", "record": "x/honest.json@596bc767e",
       "record_sha256": "shaH", "key_path": ["cells", "c1"], "predicate": {"op": "slot_string_eq", "transform": "identity"}}]
LBH = "route C, out of v1 scope (PRELIMINARY); checked against a signed record by key (09 S2 marker)"
rP, pP, stP, dP = VP.post_rows("post", TP, c2 | {"text_sha": VC.sha(TP.encode()), "provenance": PP, "records": {"shaH": {"cells": {"c1": "0.71"}}},
                                                 "signed": SIG | {"shaH"}, "preliminary": {"shaH"}, "preliminary_labels": {"shaH": LBH}})
check("R6-1: a PRELIMINARY pin on a signed record carries its per-record label (never 'record unsigned')",
      next(r for r in rP if r["slot"] == f"map:75@{TP.index('0.71')}")["rule"] == LBH)
_, _, stP0, _ = VP.post_rows("post", TP, c2 | {"text_sha": VC.sha(TP.encode()), "provenance": PP, "records": {"shaH": {"cells": {"c1": "0.71"}}},
                                                "signed": SIG | {"shaH"}, "preliminary": {"shaH"}})
check("R6-1 negative: a record in both the PRELIMINARY pin and the signed set without a label override STOPs",
      any("also in the signed set and has no label override" in x for x in stP0))
sP = summ(rP, pP, [x for x in stP if "header line" not in x])
check("PRELIMINARY (09): a string cell of the pinned honest record is checked by key, counted apart and named in the verdict line, never "
      "route-C-verified (the preliminary pin wins over the signed set)",
      next(r for r in rP if r["slot"] == f"map:75@{TP.index('0.71')}")["unit"] == "preliminary" and sP["n_preliminary_checked"] == 1
      and "1 PRELIMINARY (route C, out of v1 scope), checked against records by key" in sP["verdict"] and sP["verdict_base"] == "PASS")
vlP = VC.verdict_line(sP["verdict_base"], sP["n_not_route_C_checked"], None, n_pre=sP["n_preliminary_checked"], pre_labels=dP["preliminary_labels_used"])
oP = VC.artifact_outcome("post", sP)[0]
check("R7-1: with the label set, neither the verdict line nor the post's outcome line says 'unsigned' for the signed record; the verdict "
      "points to the scope (2e / 09: the full label lives there, not in the verdict)",
      "unsigned" not in vlP and "unsigned" not in oP and "labels in scope" in vlP and LBH not in vlP)
_, _, stP3, _ = VP.post_rows("post", TP, c2 | {"text_sha": VC.sha(TP.encode()), "provenance": PP, "records": {"shaH": {"cells": {"c1": "0.71"}}},
                                                "preliminary": {"shaH"}, "preliminary_labels": {"shaH": LBH}})
check("R7-2: a label override on a PRELIMINARY-only (unsigned) record STOPs", any("not both PRELIMINARY and signed" in x for x in stP3))
rP2, _, _, dP2 = VP.post_rows("post", TP, c2 | {"text_sha": VC.sha(TP.encode()), "provenance": PP, "records": {"shaH": {"cells": {"c1": "0.72"}}},
                                              "preliminary": {"shaH"}})
check("PRELIMINARY: a record cell that differs from the printed text STOPs (the row fails; the token is an orphan)",
      dP2["map_rows"]["75"]["ok"] is False and any(r["slot"] == f"orphan@{TP.index('0.71')}" for r in rP2))
TC = "The gap was 0.3 points lower.\n"
TCT = TC + "\n" + VP.TABLE_HEADERS["Table 3"] + "\n|---|---|---|---|---|\n\n" + H4 + "\n|---|---|---|\n"  # both table blocks known (else fail closed)
SLc = {"k1": "+1.2 (−0.3 to +2.7)"}
Vc = {("rc1a", f"k1#{i}"): VC.row("rc1a", f"k1#{i}", "-", "-", t_, t_, "pp", True, ["shaR"], SIG) for i, t_ in enumerate(VC.NUM_RX.findall(SLc["k1"]))}
Vc[("rc1a/slots", "k1")] = VC.row("rc1a/slots", "k1", "-", "-", SLc["k1"], SLc["k1"], "pp/summ", True, ["shaR"], SIG, unit="slot")
SRc = {("rc1a", "k1"): [v for (a_, s_), v in Vc.items() if a_ == "rc1a"]}
def pcomp(comp, tr, vs, printed="0.3"):
    return [{"v84_sha256": VC.sha(TCT.encode()), "row": 5, "anchor": TC.strip(), "printed": printed, "record": "x@abcdef1", "record_sha256": "shaRC1A",
             "key_path": ["slots", "k1"], "predicate": {"op": "slot_component_eq", "component": comp, "transform": tr, "value_string": vs}}]
cC = c2 | {"text_sha": VC.sha(TCT.encode()), "verified": Vc, "slot_rows": SRc, "released": {"rc1a": SLc}, "artifact_of_sha": {"shaRC1A": "rc1a"},
           "kind_of": {"rc1a": "slots"}}
for nm, pr_, want in (("the right component with the abs transform passes and reports the sign word as not machine-checked",
                       pcomp("interval_low", "abs (sign stated in words)", SLc["k1"]), True),
                      ("a wrong component STOPs", pcomp("interval_high", "abs (sign stated in words)", SLc["k1"]), False),
                      ("an unknown transform STOPs", pcomp("interval_low", "negate", SLc["k1"]), False),
                      ("a value_string that is not the verified slot string STOPs", pcomp("interval_low", "abs (sign stated in words)", "+1.2 (-0.3 to +2.7)"), False)):
    rr, _, _, dd = VP.post_rows("post", TCT, cC | {"provenance": pr_})
    okr = dd["map_rows"]["5"]["ok"]
    check(f"op 23 in the post: {nm}", okr is want and (not want or ("sign word not machine-checked" in dd["map_rows"]["5"]["note"]
                                                                     and any(x["class"] == "sign_word_not_machine_checked" for x in dd["number_words"]["list"]))))
TW = "Near 100% catch on all three.\n"
SLw = {f"w{i}": "x" for i in range(3)}
def pall(keys):
    return [{"v84_sha256": VC.sha(TW.encode()), "row": 249, "anchor": TW.strip(), "printed": "100%", "record": "x@abcdef1", "record_sha256": "shaRC1A",
             "key_path": [["slots", k] for k in keys], "predicate": {"op": "all_within", "target": 100, "tol": 0.5, "unit": "percent (record fractions x 100)"}}]
Vw = {("rc1a", f"w{i}#0"): VC.row("rc1a", f"w{i}#0", "-", "-", "99.8", "99.8", "pct", True, ["shaR"], SIG) for i in range(3)}
Vw |= {("rc1a/slots", f"w{i}"): VC.row("rc1a/slots", f"w{i}", "-", "-", "x", "x", "pct/summ", True, ["shaR"], SIG, unit="slot") for i in range(3)}
SRw = {("rc1a", f"w{i}"): [Vw[("rc1a", f"w{i}#0")]] for i in range(3)}
cW = c2 | {"text_sha": VC.sha(TW.encode()), "verified": Vw, "slot_rows": SRw, "released": {"rc1a": SLw}, "artifact_of_sha": {"shaRC1A": "rc1a"},
           "kind_of": {"rc1a": "slots"}}
for nm, keys, raw, want in (("all three values within tol: passes, claiming only the '100'", ["w0", "w1", "w2"], {"w0": 0.998, "w1": 0.999, "w2": 1.0}, True),
                            ("one value out of tol STOPs", ["w0", "w1", "w2"], {"w0": 0.998, "w1": 0.98, "w2": 1.0}, False),
                            ("a missing key STOPs", ["w0", "w1", "wX"], {"w0": 0.998, "w1": 0.999}, False)):
    rr, _, _, dd = VP.post_rows("post", TW, cW | {"provenance": pall(keys), "slot_raw": {"rc1a": raw}})
    check(f"op 24 in the post: {nm}", dd["map_rows"]["249"]["ok"] is want and (not want or dd["map_rows"]["249"]["claimed"] == 1))
rI, _, _, dI = VP.post_rows("post", TEXT, ctx(command_check=lambda s_, f_, n_: (True, {"splitlines": False, "newline_count": False,
                                                                                    "newline_count_plus_1": True,
                                                                                    "command_lines_without_heredoc_delimiters": False})
                                                if f_ == "line_count" else (True, {"integer_after_LIMIT": True})))
check("09 I2: a cell matched only by +1 is reported by key ('matched only by +1 (possible trailing-newline off-by-one)')",
      any("matched only by +1 (possible trailing-newline off-by-one)" in f for f in dI["findings"]))
check("09 I1: every command cell lists the same five definition names, with an applicable flag by form",
      all([d["name"] for d in x["definitions"]] == list(VP.ALL_DEFS) for x in dI["command_definitions"]) and len(dI["command_definitions"]) == 2)

dfx = VC.nonresult_diff(["reference\tTable 3\t10", "constant\t5 folds\t20"], ["reference\tTable 3\t10", "run_label\treal run 1\t30"])
check("non-result diff by key: entries added / removed by class, against the approved list",
      dfx["added_by_class"] == {"run_label": 1} and dfx["removed_by_class"] == {"constant": 1} and dfx["unchanged"] == 1)

TF = "The drop was 2.3 points.\n"
def pfull(kp, pred):
    return [{"v84_sha256": VC.sha(TF.encode()), "row": 22, "anchor": TF.strip(), "printed": "2.3", "record": "x@abcdef1", "record_sha256": "shaRC1A",
             "key_path": kp, "predicate": pred}]
cF = cC | {"text_sha": VC.sha(TF.encode()), "slot_raw": {"rc1a": {"k1": 0.0231}}, "row_scope_notes": {"22": "verifies one cell only"}}
_, _, _, dF1 = VP.post_rows("post", TF, cF | {"provenance": pfull(["full", "k1"], {"op": "abs_round_eq", "dp": 1, "value": 2.3, "scale": 100})})
_, _, _, dF2 = VP.post_rows("post", TF, cF | {"provenance": pfull(["full", "k1"], {"op": "abs_round_eq", "dp": 1, "value": 2.3})})
_, _, _, dF3 = VP.post_rows("post", TF, cF | {"provenance": pfull(["full", "k1"], {"op": "slot_string_eq", "transform": "identity"})})
check("v14 [full, key]: the verified slot's unrounded value x scale 100 rounds to the printed value (scale is not a printed argument); the row "
      "carries its configured scope line", dF1["map_rows"]["22"]["ok"] is True and dF1["map_rows"]["22"]["claimed"] == 1
      and dF1["map_rows"]["22"]["scope"] == "verifies one cell only")
check("v14 [full, key] negative: without the scale the 100x mismatch fails (a printed-vs-record finding)", dF2["map_rows"]["22"]["ok"] is False)
check("v14 [full, key] negative: a string op must cite [slots, key]", dF3["map_rows"]["22"]["ok"] is False)

TR = "We saw run 1.5 here and run 2 there.\n"
rR, _, _, _ = VP.post_rows("post", TR, c2 | {"text_sha": VC.sha(TR.encode()), "provenance": [], "literals": lits_})
check("run-label guard (09): the '1' of 'run 1.5' is never claimed (whole token); 'run 2' is",
      next(r for r in rR if r["slot"].endswith(f"@{TR.index('1.5')}"))["slot"].startswith("orphan@")
      and next(r for r in rR if r["slot"].endswith(f"@{TR.index('run 2') + 4}"))["nr_class"] == "run_label")
TB = "Failures were rare. On real machines they were 3.7–6.8% of honest runs.\n"
DB = [{"anchor": "Failures were rare.", "scope": "sentence", "reason": "x", "record": None},
      {"row": None, "anchor": "On real machines they were 3.7–6.8% of honest runs", "scope": "clause",
       "reason": "not route C; unchanged on route C by 0d's 10-18 code map", "record": [{"sha256": "shaB"}]}]
KB = {"failed": ["h", "X_real", "failed"], "launched": ["h", "X_real", "launched"], "unseen": ["r", "unseen"]}
CB = [{"deferred_row": 2, "record_sha256": "shaB", "keys": KB}]
def run_b(rec):
    return VP.post_rows("post", TB, c2 | {"text_sha": VC.sha(TB.encode()), "provenance": [], "deferred": DB, "deferred_record_checks": CB,
                                          "deferred_record_ok": {0: True, 1: True}, "records": {"shaB": rec}})
# 100 * 37 / (969 + 31) = 3.7 and 100 * (37 + 31) / 1000 = 6.8 (synthetic)
rb1, pb1, sb1, db1 = run_b({"h": {"X_real": {"failed": 37, "launched": 969}}, "r": {"unseen": 31}})
check("[^failed] bounds (09): both printed bounds equal the record's by key -> the row reads not route C, checked against record by key",
      db1["deferred_record_checks"]["2"] == {"ran": True, "lower_equal": True, "upper_equal": True}
      and all(r["unit"] == "not_route_C" and r["equal"] for r in rb1 if r["slot"].startswith("deferred_check:2@")))
rb2, pb2, sb2, db2 = run_b({"h": {"X_real": {"failed": 38, "launched": 969}}, "r": {"unseen": 31}})
check("[^failed] bounds negative: a record that gives other bounds fails (booleans only; the tokens' rows fail)",
      db2["deferred_record_checks"]["2"]["lower_equal"] is False and any(not r["equal"] for r in rb2 if r["slot"].startswith("deferred_check:2@")))
rb3, pb3, sb3, db3 = run_b({"h": {"X_real": {"failed": 37}}, "r": {"unseen": 31}})
check("[^failed] bounds negative: a missing key STOPs (the check reports ran False)", db3["deferred_record_checks"]["2"] == {"ran": False}
      and any("its record check could not run" in x for x in sb3))
# 09 (10-20, row 29): U restricted to the headline's common sets, a config design constant; the record's U must equal the configured total
LB29 = "not route C; checked against cited record by key (record unsigned); U restricted to the headline's common sets"
def run_b2(rec, **kw):
    return VP.post_rows("post", TB, c2 | {"text_sha": VC.sha(TB.encode()), "provenance": [], "deferred": DB, "deferred_record_ok": {0: True, 1: True},
                                          "records": {"shaB": rec}, "deferred_record_checks": [CB[0] | kw]})
rc1, _, sc1, dc1 = run_b2({"h": {"X_real": {"failed": 37, "launched": 969}}, "r": {"unseen": 50}}, u_common=31, u_total=50, label=LB29)
check("09 row 29: with u_common the bounds use it (record U == the configured total) and the tokens carry 09's label",
      dc1["deferred_record_checks"]["2"] == {"ran": True, "lower_equal": True, "upper_equal": True}
      and all(r["equal"] and r["rule"] == LB29 for r in rc1 if r["slot"].startswith("deferred_check:2@")))
_, _, _, dc2 = run_b2({"h": {"X_real": {"failed": 37, "launched": 969}}, "r": {"unseen": 50}})
check("09 row 29 control: without u_common the record's own U gives other bounds (the printed upper fails)",
      dc2["deferred_record_checks"]["2"]["upper_equal"] is False)
_, _, sc3, dc3 = run_b2({"h": {"X_real": {"failed": 37, "launched": 969}}, "r": {"unseen": 49}}, u_common=31, u_total=50, label=LB29)
check("09 row 29 negative: a record U that is not the configured total STOPs (ran False)", dc3["deferred_record_checks"]["2"] == {"ran": False}
      and any("unseen total is not the configured total" in x for x in sc3))
# 09 op #25 in the post (row 81): claims only the size token; the qualitative word joins the number-word list; a missing key STOPs
TA0 = "At size 8 the planted fall was caught reliably.\n"
TA = TA0 + "\n" + VP.TABLE_HEADERS["Table 3"] + "\n|---|---|---|---|---|\n\n" + H4 + "\n|---|---|---|\n"  # both table blocks known
PA = [{"v84_sha256": VC.sha(TA.encode()), "row": 81, "anchor": TA0.strip(), "printed": "8", "record": "x/plant.json@abcdef1",
       "record_sha256": "shaK", "key_path": [["cells", "-8", "C", "0.01"]],
       "predicate": {"op": "all_detected_at_size", "expected": 8, "field": "detected", "of_field": "cases", "level": "0.01", "size": -8,
                     "qualitative_word": "reliably", "args": [{"role": "planted fall size", "text": "8", "value": 8}], "note": "x"}}]
def run_a(rec):
    return VP.post_rows("post", TA, c2 | {"text_sha": VC.sha(TA.encode()), "provenance": PA, "records": {"shaK": rec}, "signed": SIG | {"shaK"}})
ra, _, sa, da = run_a({"cells": {"-8": {"C": {"0.01": {"detected": 6, "cases": 6}}}}})
check("op 25 in the post: the row passes, claims only the size token, and lists the qualitative word as not machine-checked",
      da["map_rows"]["81"]["ok"] is True and da["map_rows"]["81"]["claimed"] == 1 and "qualitative word not machine-checked" in da["map_rows"]["81"]["note"]
      and next(r for r in ra if r["slot"] == f"map:81@{TA.index('8')}")["equal"]
      and any(x.get("word") == "reliably" and x["class"] == "qualitative_word_not_machine_checked" and x["covered_by"] == "map row 81"
              for x in da["number_words"]["list"]))
_, _, sa2, da2 = run_a({"cells": {"-8": {"C": {"0.01": {"detected": 5, "cases": 6}}}}})
check("op 25 in the post negative: detected < cases fails the row (no STOP)", da2["map_rows"]["81"]["ok"] is False
      and not any("map row 81" in x for x in sa2))
_, _, sa3, _ = run_a({"cells": {"-8": {"C": {"0.05": {"detected": 6, "cases": 6}}}}})
check("op 25 in the post: a missing key STOPs the pass", any("map row 81: all_detected_at_size: a listed key is absent (STOP)" in x for x in sa3))
_, _, sa4, _ = run_a({"cells": {"-8": {"C": {"0.01": {"detected": 6}}}}})
check("op 25 in the post: a missing field STOPs the pass", any("map row 81: all_detected_at_size: a node lacks field or of_field (STOP)" in x
                                                                 for x in sa4))
# 09 (10-20, v15 rows 75-78): round_eq on the pinned PRELIMINARY record's number cell, counted PRELIMINARY with 09's label
PPn = [PP[0] | {"predicate": {"op": "round_eq", "dp": 2, "value": 0.71}}]
rPn, _, _, dPn = VP.post_rows("post", TP, c2 | {"text_sha": VC.sha(TP.encode()), "provenance": PPn, "records": {"shaH": {"cells": {"c1": 0.7104}}},
                                                "signed": SIG | {"shaH"}, "preliminary": {"shaH"}, "preliminary_labels": {"shaH": LBH}})
check("09 v15 rows 75-78: round_eq on the PRELIMINARY record's number cell passes, counted PRELIMINARY with 09's label",
      dPn["map_rows"]["75"]["ok"] is True and dPn["map_rows"]["75"]["bucket"] == "preliminary"
      and next(r for r in rPn if r["slot"] == f"map:75@{TP.index('0.71')}")["rule"] == LBH)

# ---------- round 7 (R6-2, R6-3, run labels as verbs) ----------
TH2 = "Then real run 1 was cut, and we run 2 times more.\n"
lits2 = VP.compile_literals(LJx)
_, _, _, dH2 = VP.post_rows("post", TH2, c2 | {"text_sha": VC.sha(TH2.encode()), "provenance": [], "literals": lits2,
                                               "added_spans": [(0, len(TH2) - 1)]})
check("R6-2: run labels on a hand-edited line count as covered (their rows pass)", dH2["hand_edit_numbers"]["uncovered"] == 0
      and dH2["hand_edit_numbers"]["by_class"].get("run_label") == 2)
check("09: a run label used as a verb ('run 2 times') is listed apart in the result", dH2["run_label_verb_use"] == [TH2.index("run 2") + 4])
PRC = [{"v84_sha256": VC.sha(TP.encode()), "row": 76, "anchor": TP.strip(), "printed": "0.71", "record": "x/honest.json@596bc767e",
        "record_sha256": "shaH", "key_path": ["cells", "c1"],
        "predicate": {"op": "slot_component_eq", "component": "point", "transform": "identity", "value_string": "0.71"}}]
_, _, _, dRC = VP.post_rows("post", TP, c2 | {"text_sha": VC.sha(TP.encode()), "provenance": PRC, "records": {"shaH": {"cells": {"c1": "0.71"}}},
                                              "signed": SIG | {"shaH"}, "preliminary": {"shaH"}, "preliminary_labels": {"shaH": LBH}})
check("R6-3: op #23 on a record's string cell (even a pinned record) fails: only slot strings the readout gate verified",
      dRC["map_rows"]["76"]["ok"] is False and "not allowed" in dRC["map_rows"]["76"]["note"])
PU = [dict(PP[0], record_sha256="shaU")]
_, _, _, dU = VP.post_rows("post", TP, c2 | {"text_sha": VC.sha(TP.encode()), "provenance": PU, "records": {"shaU": {"cells": {"c1": "0.71"}}},
                                             "signed": SIG | {"shaU"}})
check("R6-3: slot_string_eq on an arbitrary (signed, not pinned) record's cell fails", dU["map_rows"]["75"]["ok"] is False)
TW2 = "Near 100% catch, within 0.5 points, on all three.\n"
PW2 = [dict(pall(["w0", "w1", "w2"])[0], v84_sha256=VC.sha(TW2.encode()), anchor=TW2.strip(), printed="100% catch, within 0.5")]
_, _, _, dW2 = VP.post_rows("post", TW2, cW | {"text_sha": VC.sha(TW2.encode()), "provenance": PW2, "slot_raw": {"rc1a": {"w0": 0.998, "w1": 0.999,
                                                                                                                          "w2": 1.0}}})
check("R6-3: op #24 claims ONLY the target token ('100'), never a printed tolerance ('0.5' is left unclaimed)",
      dW2["map_rows"]["249"]["ok"] is True and dW2["map_rows"]["249"]["claimed"] == 1 and dW2["map_rows"]["249"]["span"] == 2)
try:
    VP.PR.check({"op": "all_within", "target": 1, "tol": 0.1, "unit": "raw"}, [1.0])
    r = False
except VP.PR.PredicateError:
    r = True
check("R6-3: op #24's unit list is x100 only ('raw' raises)", r)

# ---------- C3: no echo ----------
SECRET = "SECRET_CMD_TEXT rm -rf /x"


class FakeX:
    def __init__(self, phase):
        self.phase = phase

    def arguments_at(self, e):
        if self.phase == "locate":
            raise ValueError(SECRET)
        return {"args": {"cmd": SECRET}}

    def digests(self, args):
        if self.phase == "digest":
            raise KeyError(SECRET)
        return {"field": "cmd", "length": 9, "sha256": "2" * 64, "arguments_length": 20, "arguments_sha256": "3" * 64}

    def command_of(self, args):
        if self.phase == "count":
            raise RuntimeError(SECRET)
        return "cmd", args["cmd"], json.dumps(args)


for ph in ("locate", "digest", "count"):
    cnt = {}
    try:
        VP.command_equal(side(0, 1, 1), "line_count", 12, FakeX(ph), cnt)
        tb, msg, noctx = "", "no raise", False
    except VC.Stop as e:
        tb, msg, noctx = "".join(traceback.format_exception(e)), str(e), e.__context__ is None and e.__cause__ is None
    check(f"C3 no-echo ({ph}): a generic Stop names the phase only; neither its message nor its traceback carries the command; counted",
          msg == f"command-content: {ph}" and SECRET not in tb and "rm -rf" not in tb and noctx and cnt == {ph: 1})
CNT = {}
rows8, plans8, stops8, dg8 = VP.post_rows("post", TEXT, ctx(command_check=lambda s_, f_, n_: VP.command_equal(s_, f_, n_, FakeX("count"), CNT)))
blob = json.dumps([rows8, plans8, stops8, dg8], default=str)
check("C3 no-echo (post level): the command text reaches no row, stop, diagnostic or count; the rows fail; errors counted by phase",
      SECRET not in blob and "rm -rf" not in blob and CNT == {"count": 2} and dg8["command_error_rows"]
      and all(not r["equal"] for r in rows8 if r["slot"].startswith("command@")))
good = type("G", (), {"arguments_at": lambda self, e: {"args": {"cmd": "cat > f <<'EOF'\na\nb\nc\nEOF\n"}},
                      "digests": lambda self, a: {"field": "cmd", "length": 9, "sha256": "2" * 64, "arguments_length": 20, "arguments_sha256": "3" * 64},
                      "command_of": lambda self, a: ("cmd", a["cmd"], json.dumps(a))})()
c3a, t3a = VP.command_equal(side(0, 1, 1), "line_count", 3, good, {})
c3b, t3b = VP.command_equal(side(0, 1, 1), "line_count", 2, good, {})
c3c, t3c = VP.command_equal(side(0, 1, 1), "line_count", 9, good, {})
check("09 (b) more than one True: 3 lines in a here-document -> splitlines, count+1 and the delimiter-free command lines; covered",
      c3a is True and sorted(k for k, v in t3a.items() if v) == ["command_lines_without_heredoc_delimiters", "newline_count_plus_1", "splitlines"])
check("09 (b) exactly one True: printed 2 -> newline_count only; covered", c3b is True and [k for k, v in t3b.items() if v] == ["newline_count"])
check("09 (b) none True: printed 9 -> not covered (a finding)", c3c is False and not any(t3c.values()) and len(t3c) == 4)
ft = type("F", (), {"arguments_at": lambda self, e: {"args": {"command": "create", "file_text": "a\nb\nc\n"}},
                    "digests": good.digests, "command_of": lambda self, a: ("command", a["command"], json.dumps(a))})()
c3d, t3d = VP.command_equal(side(0, 1, 1), "line_count", 4, ft, {})
check("09 (b) + I1: a create-file argument (no here-document): the same 4 definitions as with one; printed 4 -> newline_count_plus_1 only",
      c3d is True and sorted(t3d) == sorted(t3a) and [k for k, v in t3d.items() if v] == ["newline_count_plus_1"])
try:
    VP.command_equal(side(0, 1, 1) | {"sha256": "9" * 64}, "line_count", 3, good, {})[0]
    r = False
except VC.Stop as e:
    r = str(e) == "command-content: digest"
check("C3: a located call whose digests are not the record's fails in phase digest", r)
check("C3 count: the integer after LIMIT (exactly one)", VP.command_count({"cmd": "sqlite3 db 'SELECT * FROM t LIMIT 7'"}, "limit",
                                                                         lambda a: ("cmd", a["cmd"], "")) == 7)

print(f"{len(FAILS)} failed")
sys.exit(1 if FAILS else 0)
