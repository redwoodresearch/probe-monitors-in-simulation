"""Post-text (v84) checks for the route C numbers verifier: pure, run only inside the Modal container (tests use synthetic text).

Every digit token of the frozen post gets exactly one row (artifact "post"); an independent count checks that every visible digit lies
in a token. Tokens glued to letters ("L48", "3x", "5k", "2nd") and leading-dot decimals (".001") are tokens too (R5). Precedence (09,
10-20): a swap or provenance claim > an explicit orphan note > a non-result class > a deferred region > orphan by default. Classes:
  rule         the positions writing's swap wrote: apply_v84.py re-run in a subprocess with its sub() observed (Run A must reproduce the
               swap-only output byte for byte, the hand-edit diff must give the frozen post, Run B must reproduce it); each insertion's
               span is tracked through later substitutions and mapped through the diff's kept lines (R3). A token there is claimed only
               when it equals a verified slot number of the rule's keys (an explicit sign must be the slot's); never inside Tables 3/4.
  map          writing's provenance map (09-approved): the printed text matches WHOLE tokens of the anchor, sign included (R1); every
               number printed in the span is an argument of the row's predicate (expected or bound) and every expected value is printed,
               an explicit sign agreeing with the expected sign (R2); the predicate (closed set in predicates.py) holds on the cited
               values: a verified readout slot / Tables 3/4 field of this run, a signed record by key, or a configured not-route-C record
               (unit not_route_C: checked, counted apart, never route-C-verified).
  command      a Tables 3/4 body number printed as "N-line" or "N most recent": one boolean from the drawn side's command (C3).
  orphan note  a configured note for a known unresolved number: an orphan, whatever region (09).
  nonresult    constants rows at their own anchored position only (R4; hypothetical rows with 09's three checks); 09's header labels at
               their listed positions in table header rows; 09's pinned patterns (reference, appendix, footnote marker, date,
               citation, identifier; R9). Every entry of the generated list carries its position (09 approves {sha, count}).
  deferred     inside a deferred region: NOT VERIFIED, counted apart; summarize needs 09's reason and record_ok (C1).
Tables 3/4: only map or command rows count in body rows (RQ3); drawn-side fields equal to a printed body number leave the not-printed
list, which needs 09's approval for this pass (C2). Hand-edited lines: every number must be a passing map / constants / deferred row
(or 09's pattern classes, header labels, a passing command count) (R10). Number words are listed with their class and covering row.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import predicates as PR
import verify_core as VC

NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
HIDDEN = re.compile(r"<!--.*?-->", re.S)
LINK_TARGET = re.compile(r"\]\(([^)]*)\)")
PATTERN_KEYS = ("reference", "appendix", "footnote_marker", "date", "citation", "identifier")
WORDS = {"none": 0, "zero": 0, "one": 1, "a single": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
         "nine": 9, "ten": 10}
NUMBER_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen",
                "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty",
                "seventy", "eighty", "ninety", "hundred", "hundreds", "thousand", "thousands", "million", "millions", "billion", "dozen",
                "dozens", "half", "halves", "third", "thirds", "quarter", "quarters", "fifth", "fifths", "sixth", "sixths", "seventh",
                "eighth", "ninth", "tenth", "tenths", "twice", "thrice", "single", "double", "triple", "none")
FRACTION_WORDS = {"half", "halves", "third", "thirds", "quarter", "quarters", "fifth", "fifths", "sixth", "sixths", "seventh", "eighth",
                  "ninth", "tenth", "tenths"}
FUNCTION_WORDS = {"of", "to", "and", "or", "in", "on", "the", "a", "an", "per", "at", "for", "by", "with", "is", "are", "was", "were",
                  "be", "that", "than", "as", "from", "it", "this", "these", "those", "not", "no", "but", "if", "so", "then", "each"}
TABLE_HEADERS = {"Table 3": "| Command | What it does | Probe | Setting | Above the threshold |",
                 "Table 4": "| Task (main task, hidden side task) | Real run 1 | Simulated run |"}  # writing 23 (10-20), verbatim
SIDE_OF_COL = {2: "real (run 1)", 3: "sim (sim 1)"}  # Table 4 content cells (1-based) -> the drawn side (VC.SIDES)
COMMAND_FORMS = {"line_count": re.compile(r"-line\b"), "limit": re.compile(r" most recent\b")}  # 09: the form printed right after the number
EXPECTED_ARGS = {"eq": ("value",), "round_eq": ("value",), "abs_round_eq": ("value",), "diff_round_eq": ("value",), "sum_eq": ("value",),
                 "count_eq": ("value",), "count_eq_all": ("value",), "abs_round_eq_each": ("values",), "min_max_eq": ("min", "max"),
                 "all_round_pct_eq": ("value",), "ratio_pct_round_eq": ("value",), "abs_diff_le": ("target",), "all_within": ("target",)}
NOT_ARGS = {"op", "note", "report", "dp", "num", "den", "transform", "field", "expr", "flatten", "sign", "args", "scale", "scale_note", "unit", "component"}  # never printed numbers
MISSING = object()
BUCKET_LABEL = {"not_route_C": "value checked against cited record by key (record unsigned; not route C)",
                "preliminary": "value checked against cited record by key (record unsigned; route C, out of v1 scope: PRELIMINARY)"}
STRING_OPS = ("slot_string_eq", "slot_component_eq")  # ops that compare the printed text with a verified string, not a number


# ---------- text ----------

def masked_spans(text):
    return [m.span() for m in HIDDEN.finditer(text)] + [m.span(1) for m in LINK_TARGET.finditer(text)]


def tokens(text):
    """Every digit run of the visible text as a token -> ([(start, end, token)], {start: kind}). Kinds: number (a lone leading sign
    kept), glued_before (a letter, underscore or period right before), glued_after (a letter or underscore right after), leading_dot
    (".001", the dot kept), superscript (S1: a run of digit characters re's digit class misses, e.g. superscripts; glued to a letter it is
    glued_before, like "R²", else it needs a row)."""
    mask = masked_spans(text)
    out, kinds = [], {}
    for m in NUM.finditer(text):
        s = m.start()
        tok = m.group().rstrip(",")
        e = s + len(tok)
        if any(a <= s < b for a, b in mask):
            continue
        before = text[s - 1] if s else " "
        before2 = text[s - 2] if s >= 2 else " "
        after = text[e] if e < len(text) else " "
        if before == "." and not (before2.isalnum() or before2 in "_."):
            s, tok, kind = s - 1, "." + tok, "leading_dot"
        elif before.isalpha() or before in "_.":
            kind = "glued_before"
        elif after.isalpha() or after == "_":
            kind = "glued_after"
        else:
            kind = "number"
            if before in "+-−" and (s < 2 or not text[s - 2].isalnum()):
                s -= 1
                tok = text[s] + tok
        out.append((s, e, tok))
        kinds[s] = kind
    i = 0
    while i < len(text):  # S1: digit characters that re's \d misses (superscripts, subscripts: Unicode category No), as runs
        if text[i].isdigit() and not NUM.match(text[i]):
            j = i
            while j < len(text) and text[j].isdigit() and not NUM.match(text[j]):
                j += 1
            if not any(a <= i < b for a, b in mask):
                before = text[i - 1] if i else " "
                out.append((i, j, text[i:j]))
                kinds[i] = "glued_before" if (before.isalpha() or before == "_") else "superscript"
            i = j
        else:
            i += 1
    out.sort()
    return out, kinds


def digit_coverage(text, toks):
    """R5: an independent count: visible digits (outside hidden comments and link targets), and how many lie in no token."""
    mask = masked_spans(text)
    cov = set()
    for s, e, _ in toks:
        cov.update(range(s, e))
    vis = [i for i, ch in enumerate(text) if ch.isdigit() and not any(a <= i < b for a, b in mask)]
    return len(vis), sum(i not in cov for i in vis)


def _num(t):
    import unicodedata

    t = "".join(str(unicodedata.digit(ch)) if ch.isdigit() and not NUM.match(ch) else ch for ch in t)
    return float(t.replace(VC.MINUS, "-").replace(",", ""))


def _sign(t):
    return "-" if t[:1] in ("-", VC.MINUS) else ("+" if t[:1] == "+" else None)


def _signfree(t):
    return t.lstrip("+-" + VC.MINUS)


def table_block(text, header):
    """(start, end, (header start, header end), [body row spans], [[cell spans] per body row]) of the markdown table with that header
    line, or None."""
    if text.count(header) != 1:
        return None
    i = text.index(header)
    pos, rows, cells, end = i, [], [], i
    for j, ln in enumerate(text[i:].split("\n")):
        if not ln.lstrip().startswith("|"):
            break
        if j >= 2:
            rows.append((pos, pos + len(ln)))
            bars = [pos + k for k, ch in enumerate(ln) if ch == "|"]
            cells.append([(bars[c] + 1, bars[c + 1]) for c in range(len(bars) - 1)])
        end = pos + len(ln)
        pos += len(ln) + 1
    return i, end, (i, i + len(header)), rows, cells


def block_of(text, s):
    """(start, end) of the blank-line-delimited block holding offset s (a paragraph, or a whole table)."""
    a = text.rfind("\n\n", 0, s)
    b = text.find("\n\n", s)
    return (0 if a < 0 else a + 2), (len(text) if b < 0 else b)


RESIDUAL_NOTE = {"orphan": "orphan", "constant": "constants literal off its anchor: orphan until anchored",
                 "Table3": "Tables 3/4 body cell: position only", "Table4": "Tables 3/4 body cell: position only",
                 "hand_edit": "number on a hand-edited line: named"}


def residual_items(stop, art, notes):
    """09 (10-20): the post's residual STOP items by position, no text: every unmatched or orphan slot of the post and its sub-artifacts,
    once -> [{kind, position (None for a row-level map slot), row (a map or deferred row id, else None), note}]. The note is 09's
    per-position note from config (notes: {position: note}) or the kind's note."""
    out, seen = [], set()
    for k in ("unmatched", "orphan"):
        for x in stop.get(k) or []:
            a_, _, slot = x.partition("::")
            if not (a_ == art or a_.startswith(art + "/")) or x in seen:
                continue
            seen.add(x)
            kind, _, pos = slot.rpartition("@") if "@" in slot else (slot, "", "")
            pos = int(pos) if pos.isdigit() else None
            m = re.match(r"map:(\d+)$|deferred:[a-z_]+#(\d+)$", kind)
            row = int(m.group(1) or m.group(2)) if m else None
            base = kind.split(":")[0]
            note = (notes.get(str(pos)) if pos is not None else None) or (
                f"map row {row}: failed" if base == "map" else f"deferred row {row}: named" if base == "deferred" else RESIDUAL_NOTE.get(base, base))
            out.append({"kind": base, "position": pos, "row": row, "note": str(note)[:200]})
    return out


def tables34_where(blocks, s):
    """09's attack-text rule (10-20): (table, row) of a position inside a Tables 3/4 block (row: the body row number, or "header" on the
    header or separator line), else None. blocks: {table: table_block(...)}; when either block is unknown (None) the answer is
    ("unknown", "unknown") for EVERY position: callers fail closed."""
    if len(blocks) != len(TABLE_HEADERS) or not all(blocks.values()):
        return "unknown", "unknown"
    for nm, b in blocks.items():
        if b[0] <= s < b[1]:
            return nm, next((i + 1 for i, (rs, re_) in enumerate(b[3]) if rs <= s < re_), "header")
    return None


def withhold_tables34(nw, blocks):
    """2e / 09 (10-20): a number-word entry inside a Tables 3/4 block keeps only its table, row and position, plus its covering row id
    (no word, no sentence: those rows quote attack-run commands). Fails closed: if either block's boundaries are unknown, every entry is
    withheld. A sentence never crosses a line (sentence_of), so an entry outside the blocks holds no Tables 3/4 text.
    -> (entries, number inside Tables 3/4, all withheld because the boundaries are unknown)"""
    out, n_in, unknown = [], 0, False
    for x in nw:
        w = tables34_where(blocks, x["position"])
        if w is None:
            out.append(x)
            continue
        unknown |= w[0] == "unknown"
        n_in += w[0] != "unknown"
        out.append({"position": x["position"], "table": w[0], "row": w[1], "covered_by": x["covered_by"],
                    "class": "sentence_withheld_tables34" if w[0] != "unknown" else "sentence_withheld_tables34_unknown"})
    return out, n_in, unknown


def numv5_positions_check(blocks, positions):
    """09 (10-20): numv5's committed number-word positions (positions only, no text) against this post's Tables 3/4 blocks ->
    {count, inside_tables34, positions_inside, status}; status "unknown" (a STOP for the caller) when the boundaries are unknown."""
    where = [tables34_where(blocks, int(p)) for p in positions]
    if any(w is not None and w[0] == "unknown" for w in where):
        return {"count": len(positions), "inside_tables34": None, "positions_inside": None, "status": "unknown"}
    inside = sorted(int(p) for p, w in zip(positions, where) if w is not None)
    st = (f"disclosure: numv5's result file carries {len(inside)} sentence snippets from Tables 3/4, digits masked; unread" if inside
          else "cleared")
    return {"count": len(positions), "inside_tables34": len(inside), "positions_inside": inside, "status": st}


def sentence_of(text, s):
    a = max(text.rfind(". ", 0, s), text.rfind("\n", 0, s))
    m = re.search(r"\.(?:\s|$)|\n", text[s:])
    return a + 1, (s + m.end() if m else len(text))


def walk_key(rec, path):
    """A key path into a record: dict keys (a number matches one numeric key by value), list indices (non-negative ints)."""
    d = rec
    for p in path:
        if isinstance(d, list) and type(p) is int and 0 <= p < len(d):
            d = d[p]
        elif isinstance(d, dict) and not isinstance(p, bool) and p in d:
            d = d[p]
        elif isinstance(d, dict) and type(p) in (int, float):
            ks = [k for k in d if VC._num(k) and float(k) == float(p)]
            if len(ks) != 1:
                return MISSING
            d = d[ks[0]]
        else:
            return MISSING
    return d


def parse_record(s):
    """writing's record field "path@commit[ (note)]" -> (path, commit) or None."""
    m = re.fullmatch(r"([^@\s]+)@([0-9a-f]{7,40})(?:\s.*)?", s or "", re.S)
    return (m.group(1), m.group(2)) if m else None


def compile_patterns(pj):
    """09's pinned pattern file -> compiled patterns; exactly PATTERN_KEYS (R9)."""
    p = pj.get("patterns") or {}
    if set(p) != set(PATTERN_KEYS):
        raise VC.Stop("the non-result pattern file does not hold exactly the pattern classes (STOP)")
    return {k: re.compile(v) for k, v in p.items()}


# ---------- the swap (writing's apply_v84.py, re-run in a subprocess) ----------

RUNNER = r'''import ast, json, sys
apply_path, din, rin, dout, rep, logp = sys.argv[1:7]
tree = ast.parse(open(apply_path, encoding="utf-8").read())
defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "sub"]
if len(defs) != 1:
    sys.exit(3)
defs[0].name = "_nv_orig_sub"
ns = {"__name__": "__main__", "__file__": apply_path, "__builtins__": __builtins__}
LOG, SPANS = [], []
ST = {"unattributable": 0, "outside_changes": 0, "spans_lost": 0, "last": None}


def outside(cur):
    """A change made OUTSIDE sub() (the script rebinding `text` between calls): spans before the changed region stay, spans after it
    shift by the length change, spans touching it are dropped (lost; their tokens then need another row). The region is the one the
    common prefix and suffix leave, so several separate edits drop everything between them (never a misplaced span)."""
    last = ST["last"]
    if last is None or not isinstance(cur, str) or cur == last:
        return
    ST["outside_changes"] += 1
    p = 0
    while p < min(len(last), len(cur)) and last[p] == cur[p]:
        p += 1
    q = 0
    while q < min(len(last), len(cur)) - p and last[-1 - q] == cur[-1 - q]:
        q += 1
    a, b, d = p, len(last) - q, len(cur) - len(last)
    keep = []
    for sp in SPANS:
        if sp[0] <= b and sp[1] >= a:
            ST["spans_lost"] += 1
            continue
        keep.append([sp[0] + (d if sp[0] >= b else 0), sp[1] + (d if sp[0] >= b else 0), sp[2]])
    SPANS[:] = keep


def sub(rule, old, new, keys, *a, **k):
    outside(ns.get("text"))
    before = ns.get("text")
    out = ns["_nv_orig_sub"](rule, old, new, keys, *a, **k)
    after = ns.get("text")
    ent = {"rule": str(rule), "keys": [str(x) for x in (keys or []) if isinstance(x, str)], "changed": after != before, "inserted": None}
    if after != before:
        ok = isinstance(old, str) and bool(old) and isinstance(before, str) and before.count(old) == 1
        if ok:
            i, d = before.index(old), len(after) - len(before)
            ins = after[i:i + len(old) + d]
            ok = before[:i] + ins + before[i + len(old):] == after
        if ok:
            ent["inserted"] = ins
            keep = []
            for sp in SPANS:
                if sp[0] < i + len(old) and i < sp[1]:
                    ST["spans_lost"] += 1
                    continue
                sh = d if sp[0] >= i + len(old) else 0
                keep.append([sp[0] + sh, sp[1] + sh, sp[2]])
            keep.append([i, i + len(ins), len(LOG)])
            SPANS[:] = keep
        else:
            ST["unattributable"] += 1
            ST["spans_lost"] += len(SPANS)
            SPANS[:] = []
    ST["last"] = after if isinstance(after, str) else None
    LOG.append(ent)
    return out


ns["sub"] = sub
sys.argv = [apply_path, din, rin, dout, rep]
exec(compile(tree, apply_path, "exec"), ns)
final = ns.get("text") if isinstance(ns.get("text"), str) else ""
outside(final)
written = open(dout, encoding="utf-8").read()
out_ok = written == final
spans = [sp for sp in SPANS if out_ok and final[sp[0]:sp[1]] == LOG[sp[2]]["inserted"]]
ST["spans_lost"] += len(SPANS) - len(spans)
json.dump({"log": LOG, "spans": spans, "unattributable": ST["unattributable"], "spans_lost": ST["spans_lost"],
           "outside_changes": ST["outside_changes"], "output_is_final_text": out_ok}, open(logp, "w"))
'''


def rerun_apply(apply_src, rules_src, draft_bytes, slots_bytes, workdir, timeout=300):
    """Re-run writing's apply_v84.py (committed bytes) as invoked (argv: draft, slots JSON, output, report) in a SUBPROCESS with stdout
    and stderr to DEVNULL (O4), sub() renamed by AST and observed. Each insertion's span is tracked through later substitutions AND
    through changes the script makes outside sub() (Q1: shifted by the common prefix / suffix, or dropped when touched); a sub() change
    it cannot attribute clears every span; the written output must be the final text (else no span is kept). A lost span only removes
    a claim: its tokens then need another passing row or are orphans, so it can never make a false PASS (reported, not a STOP).
    -> (output bytes, log, spans [[start, end, log index]] in the output, info counts)."""
    w = Path(workdir)
    w.mkdir(parents=True, exist_ok=True)
    (w / "appendix_rules.py").write_bytes(rules_src)
    (w / "apply_v84.py").write_bytes(apply_src)
    (w / "in.md").write_bytes(draft_bytes)
    (w / "slots.json").write_bytes(slots_bytes)
    (w / "_nv_runner.py").write_text(RUNNER, encoding="utf-8")
    r = subprocess.run([sys.executable, "-I", str(w / "_nv_runner.py"), str(w / "apply_v84.py"), str(w / "in.md"), str(w / "slots.json"),
                        str(w / "out.md"), str(w / "report.md"), str(w / "nv_log.json")], cwd=w, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=timeout)
    if r.returncode != 0:
        raise VC.Stop(f"apply_v84 re-run exited {r.returncode} (STOP)")
    lg = json.loads((w / "nv_log.json").read_text())
    info = {k: lg[k] for k in ("unattributable", "spans_lost", "outside_changes", "output_is_final_text")}
    return (w / "out.md").read_bytes(), lg["log"], lg["spans"], info


_HUNK = r"(?m)^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@.*\n((?:[ +\-\\].*(?:\n|$))*)"


def apply_diff_map(src, diff):
    """A one-file unified diff applied to src -> (target, {src line: target line} for every kept line, [target spans of ADDED lines],
    header problems). Spans come from the application itself (O2); each hunk's header counts and +start must agree with its body."""
    sl, out, i, keep, added_ln, bad = src.split("\n"), [], 0, {}, [], 0
    for h in re.finditer(_HUNK, diff):
        alen, clen = int(h.group(2) or 1), int(h.group(4) or 1)
        a = int(h.group(1)) - (0 if alen == 0 else 1)
        if a < i:
            raise VC.Stop("the hand-edit diff's hunks overlap or run backwards (STOP)")
        for k in range(i, a):
            keep[k] = len(out)
            out.append(sl[k])
        i = a
        bad += (int(h.group(3)) - (0 if clen == 0 else 1)) != len(out)
        n_old = n_new = 0
        for ln in h.group(5).split("\n"):
            if not ln or ln.startswith("\\"):
                continue
            tag, body = ln[0], ln[1:]
            if tag in " -":
                if i >= len(sl) or sl[i] != body:
                    raise VC.Stop("the hand-edit diff does not apply to the swap-only output (STOP)")
                if tag == " ":
                    keep[i] = len(out)
                    out.append(body)
                    n_new += 1
                i += 1
                n_old += 1
            elif tag == "+":
                added_ln.append(len(out))
                out.append(body)
                n_new += 1
        bad += (n_old != alen) + (n_new != clen)
    for k in range(i, len(sl)):
        keep[k] = len(out)
        out.append(sl[k])
    starts = [0]
    for ln in out:
        starts.append(starts[-1] + len(ln) + 1)
    return "\n".join(out), keep, [(starts[t], starts[t] + len(out[t])) for t in added_ln], bad


def apply_unified_diff(src, diff):
    return apply_diff_map(src, diff)[0]


def map_spans(src, keep, target, spans):
    """Spans in src -> spans in target through the kept lines (a span must sit on one kept line; else it is dropped)."""
    ss, ts = [0], [0]
    for ln in src.split("\n"):
        ss.append(ss[-1] + len(ln) + 1)
    for ln in target.split("\n"):
        ts.append(ts[-1] + len(ln) + 1)
    out = []
    for s, e, x in spans:
        li = max(k for k in range(len(ss) - 1) if ss[k] <= s)
        if e > ss[li + 1] - 1 or li not in keep:
            continue
        d = ts[keep[li]] - ss[li]
        out.append((s + d, e + d, x))
    return out


# ---------- C3: the command-content count (container only; one boolean out) ----------

def command_count(args, form, command_of):
    """The count a printed command number states, from the tool call's arguments: line_count = the lines of the written content (a
    create-file argument, else the command's one here-document; exactly one candidate); limit = the integer after LIMIT (exactly one)."""
    _field, s, _full = command_of(args)
    if form == "line_count":
        cands = [args[k] for k in ("file_text", "content") if isinstance(args, dict) and isinstance(args.get(k), str)]
        cands += [m.group(3) for m in re.finditer(r"<<-?\s*(['\"]?)([A-Za-z_]\w*)\1[^\n]*\n(.*?)\n\2[ \t]*(?:\n|$)", s, re.S)]
        if len(cands) != 1:
            raise ValueError("not exactly one written content")
        return len(cands[0].splitlines())
    if form == "limit":
        ms = re.findall(r"\bLIMIT\s+(\d+)\b", s, re.I)
        if len(ms) != 1:
            raise ValueError("not exactly one LIMIT")
        return int(ms[0])
    raise ValueError("unknown form")


LINE_DEFS = ("splitlines", "newline_count", "newline_count_plus_1", "command_lines_without_heredoc_delimiters")
ALL_DEFS = LINE_DEFS + ("integer_after_LIMIT",)
HEREDOC_RX = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_]\w*)\1[^\n]*\n(.*?)\n\2[ \t]*(?:\n|$)", re.S)


def command_tests(args, form, printed_int, command_of):
    """09 (10-20), option (b): BOOLEANS only, never a count. line_count: printed == len(content.splitlines()) ("splitlines"),
    == content.count("\\n") ("newline_count"), == content.count("\\n") + 1 ("newline_count_plus_1"), and, when the argument carries a
    here-document, == the command's lines excluding the here-document delimiter lines ("command_lines_without_heredoc_delimiters").
    The content is the create-file argument, else the command's one here-document (exactly one candidate). limit: printed == the
    integer after LIMIT ("integer_after_LIMIT", exactly one). -> {definition: bool}."""
    _field, s, _full = command_of(args)
    if form == "line_count":
        heredocs = list(HEREDOC_RX.finditer(s))
        cands = [args[k] for k in ("file_text", "content") if isinstance(args, dict) and isinstance(args.get(k), str)]
        cands += [m.group(3) for m in heredocs]
        if len(cands) != 1:
            raise ValueError("not exactly one written content")
        c = cands[0]
        out = {"splitlines": len(c.splitlines()) == printed_int, "newline_count": c.count("\n") == printed_int,
               "newline_count_plus_1": c.count("\n") + 1 == printed_int}
        delims = {m.group(2) for m in heredocs}  # 09 I1: the 4th definition is always reported (False without a here-document)
        kept = [ln for ln in s.splitlines() if "<<" not in ln and ln.strip() not in delims]
        out["command_lines_without_heredoc_delimiters"] = bool(heredocs) and len(kept) == printed_int
        return out
    if form == "limit":
        ms = re.findall(r"\bLIMIT\s+(\d+)\b", s, re.I)
        if len(ms) != 1:
            raise ValueError("not exactly one LIMIT")
        return {"integer_after_LIMIT": int(ms[0]) == printed_int}
    raise ValueError("unknown form")


def command_equal(side, form, printed_int, X, counter):
    """09's command-content check for one printed number. side: the drawn side's dict in the tables record (its locator). X: the
    consumer module (arguments_at, digests, command_of). Phases: locate, digest (the located call hashes to the record's), count. Any
    exception: its phase is counted and a generic Stop naming the phase only is raised OUTSIDE the handler (no text, no context).
    -> (covered, {definition: bool}): covered when at least one tested definition holds (09: exactly one -> that definition; more than
    one -> all reported; none -> a finding). Only booleans and definition names leave."""
    phase, failed, tests = "locate", None, None
    try:
        e = {"key": side["run_key"], "staged": side["staged_row"]["staged"], "staged_sha256": side["staged_row"]["staged_sha256"],
             "arm": side["staged_row"]["arm"], "eval_sha256": side["run_eval"]["sha256"],
             "unit": {"unit": side["unit"]["row_unit"], "call_index": side["unit"]["action"], "token": side["unit"]["token"],
                      "kind": side["readout_kind"], "irid": side["unit"]["irid"]}}
        a = X.arguments_at(e)
        phase = "digest"
        d = X.digests(a["args"])
        if any(d[k] != side.get(k) for k in ("field", "length", "sha256", "arguments_length", "arguments_sha256")):
            raise ValueError("digest")
        phase = "count"
        tests = {k: v is True for k, v in command_tests(a["args"], form, printed_int, X.command_of).items()}
    except BaseException:  # noqa: BLE001  counted, never echoed (09, 10-20)
        failed = phase
    if failed is not None:
        counter[failed] = counter.get(failed, 0) + 1
        raise VC.Stop(f"command-content: {failed}")
    return any(tests.values()), tests


# ---------- deferred, non-results ----------

def deferred_row_id(r, j):
    """writing's row id, or the 1-based position for a row without one."""
    return r.get("row") if r.get("row") is not None else j + 1


def bounds_check(text, region, rec, keys, u_common=None, u_total=None):
    """09 (10-20), the [^failed] bounds: in the deferred region, exactly one printed "L–H%" pair; with f, l, U read by key from the record,
    lower = round(100 * f / (l + U), 1) and upper = round(100 * (f + U) / (l + U), 1) (the slot rounding rule). -> ((start, end) of each
    printed number, lower == printed L, upper == printed H). A missing key, a non-number, no single pair or l + U <= 0 raises.
    09 (10-20, row 29): with u_common (a design constant: U restricted to the headline's common sets) the record's U must equal u_total
    (else raises) and u_common is used in its place."""
    s0, e0 = region
    ms = list(re.finditer(r"(\d+(?:\.\d+)?)\s*[–-]\s*(\d+(?:\.\d+)?)%", text[s0:e0]))
    if len(ms) != 1:
        raise ValueError("not exactly one printed bounds pair in the region")
    m = ms[0]
    f, l_, u = (walk_key(rec, keys[k]) for k in ("failed", "launched", "unseen"))
    if any(v is MISSING or isinstance(v, bool) or type(v) not in (int, float) for v in (f, l_, u)) or l_ + u <= 0:
        raise ValueError("a missing or non-numeric key")
    if u_common is not None:
        if u_total is None or u != u_total or isinstance(u_common, bool) or type(u_common) is not int or u_common < 0:
            raise ValueError("the record's unseen total is not the configured total")
        u = u_common
    lo = PR._round_half_even(100 * f / (l_ + u), 1)
    hi = PR._round_half_even(100 * (f + u) / (l_ + u), 1)
    spans = [(s0 + m.start(1), s0 + m.end(1)), (s0 + m.start(2), s0 + m.end(2))]
    return spans, lo == float(m.group(1)), hi == float(m.group(2))


def deferred_regions(text, rows):
    """writing's deferred rows -> ([(start, end, reason, label, j)], problems). Scopes: section (its heading line to the next heading of
    the same or a higher level), paragraph / footnote (to the next blank line; a footnote also to the next footnote definition),
    sentence / clause (the anchor substring only), table_row (the whole table line the anchor starts). A row whose anchor does not
    occur exactly once, or that does not fit its scope, gets no region and is reported as a problem (a STOP); the pass continues."""
    heads = [(m.start(), len(m.group(1))) for m in re.finditer(r"(?m)^(#{1,6})\s+.+$", text)]
    out, problems = [], []
    for j, r in enumerate(rows):
        a, sc, rid = r.get("anchor") or "", r.get("scope"), deferred_row_id(r, j)
        if not a or text.count(a) != 1:
            problems.append(f"deferred row {rid}: its anchor does not occur exactly once")
            continue
        s = text.index(a)
        if sc == "section":
            lvl = next((lv for p, lv in heads if p == s), None)
            if lvl is None:
                problems.append(f"deferred row {rid}: a section anchor is not a heading line")
                continue
            e = next((p for p, lv in heads if p > s and lv <= lvl), len(text))
        elif sc in ("paragraph", "footnote"):
            m = re.search(r"\n\s*\n" + (r"|\n\[\^" if sc == "footnote" else ""), text[s:])
            e = s + m.start() if m else len(text)
        elif sc in ("sentence", "clause"):  # the anchor substring only
            e = s + len(a)
        elif sc == "table_row":
            if s != text.rfind("\n", 0, s) + 1 or not a.startswith("|"):
                problems.append(f"deferred row {rid}: a table_row anchor does not start a table line")
                continue
            le = text.find("\n", s)
            e = len(text) if le < 0 else le
        else:
            problems.append(f"deferred row {rid}: unknown scope")
            continue
        out.append((s, e, r.get("reason", "?"), f"{sc}#{rid}", j))
    return out, problems


def constants_positions(text, toks, constants):
    """R4 / 09 (10-20): each constants row's literal at ITS anchor only (the anchor occurs once in the post, the literal once in the
    anchor, at whole-token boundaries) -> ({token start: row}, problems, occurrence_of_literal disagreements)."""
    at, problems, occ_bad = {}, [], 0
    for c in constants:
        a, lit = c.get("anchor") or "", c.get("text") or ""
        if not a or text.count(a) != 1 or a.count(lit) != 1:
            problems.append(f"constants row {c.get('row')}: its anchor is not unique, or holds its literal other than once")
            continue
        l0 = text.index(a) + a.index(lit)
        l1 = l0 + len(lit)
        inside = [(s, e) for s, e, _ in toks if s < l1 and e > l0]
        if any(s < l0 or e > l1 for s, e in inside):
            problems.append(f"constants row {c.get('row')}: its literal does not sit on whole tokens")
            continue
        if c.get("occurrence_of_literal") is not None and text[:l0].count(lit) + 1 != c["occurrence_of_literal"]:
            occ_bad += 1
        for s, _ in inside:
            at[s] = c
    return at, problems, occ_bad


def header_positions(text, toks, entries):
    """09's header / label class: {token start: entry} for each approved entry whose token sits at its position in a table header row
    (the line just above a separator) with the same token and header cell text."""
    sep = re.compile(r"^\s*\|(?:\s*:?-{3,}:?\s*\|)+\s*$")
    pos = {s: t for s, _, t in toks}
    at, bad = {}, 0
    for x in entries or []:
        s = x.get("position")
        ls = text.rfind("\n", 0, s) + 1 if isinstance(s, int) else -1
        le = text.find("\n", ls)
        nxt = text[le + 1:text.find("\n", le + 1) if text.find("\n", le + 1) >= 0 else len(text)] if le >= 0 else ""
        line = text[ls:le if le >= 0 else len(text)]
        bars = [ls + k for k, ch in enumerate(line) if ch == "|"]
        c = x.get("cell_index")
        cell = text[bars[c - 1] + 1:bars[c]].strip() if isinstance(c, int) and 1 <= c < len(bars) else None
        if pos.get(s) == x.get("token") and sep.match(nxt) and line == x.get("header_line") and cell == x.get("cell"):
            at[s] = x
        else:
            bad += 1
    return at, bad


LITERAL_CLASSES = ("run_label", "identifier_literal", "placeholder", "command_literal", "citation_literal")


def compile_literals(lj):
    """09's pinned literal classes (10-20): exactly LITERAL_CLASSES."""
    c = lj.get("classes") or {}
    if set(c) != set(LITERAL_CLASSES):
        raise VC.Stop("the literal non-result file does not hold exactly the literal classes (STOP)")
    return {"run_label": re.compile(c["run_label"]["regex"]), "identifier_literal": list(c["identifier_literal"]["literals"]),
            "placeholder": re.compile(c["placeholder"]["regex"]), "command_literal": list(c["command_literal"]["entries"]),
            "citation_literal": list(c["citation_literal"]["entries"])}


def literal_problems(text, lits):
    """Every pinned literal must sit at its position (else a STOP)."""
    return [f"{k} literal at {x['position']} is not at its pinned position" for k in ("command_literal", "citation_literal") for x in lits[k]
            if text[x["position"]:x["position"] + len(x["literal"])] != x["literal"]]


def literal_class(text, s, e, tok, lits):
    """09's literal classes for one token -> (class, matched string, classifier) or None. Each claims only the tokens inside its own match."""
    for m in lits["run_label"].finditer(text, max(0, s - 16), min(len(text), e + 1)):
        if m.start(1) == s and m.end(1) == e:
            verb = re.match(r"\s+times?\b", text[e:e + 8], re.I) is not None  # 09: "run 2 times" is a verb use: listed apart in the result
            return "run_label", m.group(0), "literal run label" + (" (verb use)" if verb else "")
    for lit in lits["identifier_literal"]:
        for m in re.compile(re.escape(lit)).finditer(text, max(0, s - len(lit)), min(len(text), e + len(lit))):
            if m.start() <= s and e <= m.end():
                return "identifier_literal", lit, f"literal {lit}"
    for m in lits["placeholder"].finditer(text, max(0, s - 12), min(len(text), e + 2)):
        if m.start() <= s and e <= m.end():
            return "placeholder", m.group(0), "literal placeholder"
    for k in ("command_literal", "citation_literal"):
        for x in lits[k]:
            p0 = x["position"]
            if p0 <= s and e <= p0 + len(x["literal"]) and text[p0:p0 + len(x["literal"])] == x["literal"]:
                return k, x["literal"], f"literal at {p0}"
    return None


def pattern_class(text, s, e, tok, kind, pats):
    """09's pinned pattern classes (R9) for one token -> (class, matched string) or None."""
    pre = text[max(0, s - 60):s]
    for cls in ("reference", "appendix", "footnote_marker"):
        m = pats[cls].search(pre)
        if m:
            return cls, (pre[m.start():] + tok).strip()
    for cls in ("date", "citation"):
        for m in pats[cls].finditer(text, max(0, s - 40), min(len(text), e + 40)):
            if m.start() <= s and e <= m.end():
                return cls, m.group()
    if kind == "glued_before" and pats["identifier"].search(text[max(0, s - 1):s]):
        a = s
        while a > 0 and (text[a - 1].isalnum() or text[a - 1] in "_.-"):
            a -= 1
        b = e
        while b < len(text) and (text[b].isalnum() or text[b] in "_."):
            b += 1
        return "identifier", text[a:b][:60]
    return None


# ---------- provenance rows ----------

def _paths(kp):
    return kp if kp and all(isinstance(x, list) for x in kp) else [kp]


def _close(v, x):
    return abs(v - x) <= 1e-9 * max(1.0, abs(v))


def _arg_numbers(pred):
    """(expected numbers, other argument numbers) of a predicate: numeric arguments and the numbers inside its string arguments."""
    op = pred.get("op")
    exp, oth = [], []
    for k, v in pred.items():
        if k in NOT_ARGS:
            continue
        b = exp if k in EXPECTED_ARGS.get(op, ()) else oth
        for x in (v if isinstance(v, list) else [v]):
            if isinstance(x, bool) or x is None:
                continue
            if isinstance(x, (int, float)):
                b.append(float(x))
            elif isinstance(x, str):
                b += [float(t.replace(",", "")) for t in re.findall(r"\d[\d,]*(?:\.\d+)?", x)]
    return exp, oth


SIGN_FREE_OPS = {"abs_round_eq", "abs_round_eq_each", "diff_round_eq", "max_abs_lt", "max_abs_le"}  # they compare magnitudes only


def _hit(a, x, pc):
    """A printed magnitude x (pc: a "%" printed right after it) equals the argument a, sign-free; x100 only next to a "%" (N3)."""
    return _close(x, abs(a)) or (pc and _close(x, abs(100 * a)))


def expected_printed(pred, printed):
    """R2 (b): every expected argument of the row appears in its printed text: a number sign-free, or x100 next to a printed "%", or as a
    number word; a string verbatim. -> (ok, note)."""
    op = pred.get("op")
    nums = [(abs(_num(t)), e < len(printed) and printed[e] == "%") for _, e, t in tokens(printed)[0]]
    words = [float(WORDS[w]) for w in re.findall(r"a single|[a-z]+", printed.lower()) if w in WORDS]
    for k in EXPECTED_ARGS.get(op, ()):
        for v in (pred.get(k) if isinstance(pred.get(k), list) else [pred.get(k)]):
            if isinstance(v, bool) or v is None:
                return False, "an expected value is not a number or string"
            if isinstance(v, str):
                if v not in printed:
                    return False, "an expected value is not what the post prints"
            elif not (any(_hit(v, x, pc) for x, pc in nums) or any(_close(abs(v), w) for w in words)):
                return False, "an expected value is not what the post prints"
    return True, ""


def number_word_only(printed):
    """09 (10-20), row 74's class: a printed span with NO digit token, NONE of the words the code reads (WORDS: zero-ten, "a single",
    "none"), and at least one word of the number-word classes (NUMBER_WORDS).
    Its record-vs-predicate check runs as for any row; only the link "printed words <-> expected value" is not machine-checked
    (printed_link "number word, not gated"). Keyed on the class, never on a row id."""
    readable = [w for w in re.findall(r"a single|[a-z]+", printed.lower()) if w in WORDS]  # R4-1: words the code CAN read are not waived
    return (not tokens(printed)[0] and not readable
            and re.search(r"\b(" + "|".join(NUMBER_WORDS) + r")\b", printed, re.I) is not None)


def _keypath_numbers(kp):
    out = []
    for q in _paths(kp):
        for x in (q if isinstance(q, list) else [q]):
            if type(x) in (int, float):
                out.append(float(x))
            elif isinstance(x, str):
                out += [float(t) for t in re.findall(r"(?<![\d.])\d+(?:\.\d+)?(?![\d.])", x)]
    return out


def claim_tokens(p, spans, text, ctx):
    """R2 at the token level (09 / 2e, 10-20): the tokens of a row's printed span that the row itself VERIFIES; the others fall through
    to the next class (finally orphan). slot_string_eq: a token equal to a number of the released slot string (a "+" the post drops is
    allowed). Other ops: a token equal to one of the predicate's arguments (expected or bound; x100 only before a "%"), to a number of
    or declared in the predicate's "args" when the declaration is itself checked: a numerator / denominator pair
    whose ratio is one of the predicate's levels, or a value that is a predicate argument or a key-path number. An explicit printed sign
    must agree (the predicate's `sign`, else the matched argument's sign; on a magnitude-only op a sign needs `sign`): a disagreement
    fails the row. -> (claimed spans, problem or "")."""
    pred = p.get("predicate") or {}
    op = pred.get("op")
    if op in STRING_OPS:
        art = ctx["artifact_of_sha"].get(p.get("record_sha256"))
        q = _paths(p.get("key_path"))[0]
        sl = ctx["released"].get(art, {}).get(q[1]) if art and isinstance(q, list) and len(q) == 2 else None
        if sl is None and art is None and op == "slot_string_eq" and p.get("record_sha256") in (ctx.get("preliminary") or ()):
            # R6-3: a string cell only of a record 09 pinned by name (the PRELIMINARY honest record); never an arbitrary record's cell
            v_ = walk_key(ctx["records"].get(p.get("record_sha256"), {}), q) if isinstance(q, list) else MISSING
            sl = v_ if isinstance(v_, str) else None
        if op == "slot_component_eq":  # 09 op #23: the token claims only itself: the one component the row names
            try:
                want = PR.slot_component(sl or "", pred.get("component"), pred.get("transform"))
            except PR.PredicateError:
                return [], ""
            return [(s, e, t) for s, e, t in spans if PR.component_printed(want, t)], ""  # 09: a dropped "+" only
        stoks = VC.NUM_RX.findall(sl or "")
        return [(s, e, t) for s, e, t in spans if any(st == t or (_sign(t) is None and _sign(st) == "+" and _signfree(st) == t)
                                                       for st in stoks)], ""
    if op == "all_detected_at_size":  # 09 op #25: the row claims ONLY the printed size token (|size|, sign-free); nothing else
        sz = pred.get("size")
        return [(s, e, t) for s, e, t in spans if type(sz) in (int, float) and _close(abs(_num(t)), abs(sz))], ""
    exp, oth = _arg_numbers(pred)
    kp = _keypath_numbers(p.get("key_path"))
    decl = [a for a in pred.get("args") or [] if isinstance(a, dict) and type(a.get("value")) in (int, float)]
    levels = [float(x) for x in pred.get("levels") or [] if isinstance(x, (int, float, str)) and re.fullmatch(r"\d+(?:\.\d+)?", str(x))]
    nums_ = [a["value"] for a in decl if "numerator" in str(a.get("role", ""))]
    dens_ = [a["value"] for a in decl if "denominator" in str(a.get("role", ""))]
    ok_pairs = [(n, d) for n in nums_ for d in dens_ if d and any(_close(n / d, L) for L in levels)]
    checked = {float(a["value"]) for a in decl if "numerator" not in str(a.get("role", "")) and "denominator" not in str(a.get("role", ""))
               and any(_close(float(a["value"]), y) for y in exp + oth + kp)}
    paired = set()  # a numerator / denominator pair is claimed only as adjacent tokens "N in D" or "N/D" whose ratio is a level
    for (s1, e1, t1), (s2, e2, t2) in zip(spans, spans[1:]):
        if re.fullmatch(r"\s+in\s+|/", text[e1:s2]) and any(_close(_num(t1), n) and _close(_num(t2), d) for n, d in ok_pairs):
            paired |= {s1, s2}
    out = []
    for s, e, t in spans:
        if s in paired:
            out.append((s, e, t))
            continue
        x, pc = abs(_num(t)), text[e:e + 1] == "%"
        m_exp = [a for a in exp if _hit(a, x, pc)]
        m_oth = [] if op == "all_within" else (  # R6-3: op #24 claims ONLY its target, never a printed tolerance
            [a for a in oth if _hit(a, x, pc)] + [y for y in checked if _close(x, abs(y))])
        if not m_exp and not m_oth:
            continue
        sg = _sign(t)
        if sg:
            if op in SIGN_FREE_OPS and pred.get("sign") not in ("+", "-"):
                return [], "a printed sign on a magnitude-only op needs the predicate's sign argument"
            want = {pred["sign"]} if m_exp and pred.get("sign") in ("+", "-") else (
                {("-" if a < 0 else "+") for a in (m_exp or m_oth) if a != 0} or {"+", "-"})
            if sg not in want:
                return [], "a printed sign disagrees with the expected sign"
        out.append((s, e, t))
    return out, ""


def anchor_spans(text, toks, p):
    """R1: the token spans of a provenance row's printed text at every anchor, matched on WHOLE tokens: the post's tokens overlapping the
    printed span all lie inside it and equal tokens(printed), sign included. cell_index (when given): the 1-based content cell of a
    table-row anchor; it picks the occurrence when there are several. -> (spans, words_only, problem or "", report, printed spans)."""
    ptoks = [t for _, _, t in tokens(p["printed"])[0]]
    L = len(p["printed"])
    anchors = [p["anchor"]] + list(p.get("occurrence_anchors") or [])
    spans, rep, pspans = [], {}, []
    for a in anchors:
        if text.count(a) != 1:
            return [], False, "an anchor does not occur exactly once", rep, []
        base = text.index(a)
        good = []
        for m in re.finditer(re.escape(p["printed"]), a):
            a0 = base + m.start()
            ov = [(s, e, t) for s, e, t in toks if s < a0 + L and e > a0]
            if all(a0 <= s and e <= a0 + L for s, e, _ in ov) and [t for _, _, t in ov] == ptoks:
                good.append((m.start(), ov, a0))
        if not good:
            return [], False, "the printed text is not at whole-token boundaries in its anchor (sign included)", rep, []
        ci = p.get("cell_index")
        bars = [k for k, ch in enumerate(a) if ch == "|"]
        if type(ci) is int and 1 <= ci < len(bars):
            lo, hi = bars[ci - 1] + 1, bars[ci]
            in_cell = [g for g in good if lo <= g[0] and g[0] + L <= hi]
            rep = {"cell_index_holds_printed": len(in_cell) == 1}
            good = in_cell
        if len(good) != 1:
            return [], False, "the printed text is not unique (whole tokens) in its anchor or cell", rep, []
        spans += good[0][1]
        pspans.append((good[0][2], good[0][2] + L))
    return spans, not NUM.search(p["printed"]), "", rep, pspans


def map_check(p, ctx):
    """One provenance row -> (ok, sources, bucket, note, predicate report). bucket: verified | not_route_C | orphan."""
    pred = p.get("predicate")
    if not isinstance(pred, dict):
        return False, [], "orphan", "no structured predicate", {}
    rec_sha, paths = p.get("record_sha256"), _paths(p.get("key_path"))
    link = ""
    if pred.get("op") not in STRING_OPS:
        ok_p, note_p = expected_printed(pred, p["printed"])
        if not ok_p and number_word_only(p["printed"]):  # 09 (10-20): waived BY CLASS for a span with 0 digit tokens and a number word
            ok_p, link = True, "number word, not gated"
        if not ok_p:
            return False, [], "verified", note_p, {}
    art = ctx["artifact_of_sha"].get(rec_sha)
    if art is not None and ctx["kind_of"][art] == "slots":
        # [slots, key]: the released slot string (string ops) or its unrounded value; [full, key] (writing's v14): the unrounded value only
        slots = [q[1] for q in paths if isinstance(q, list) and len(q) == 2 and q[0] in ("slots", "full")]
        if len(slots) != len(paths) or not slots:
            return False, [], "verified", "a readout key path is not [slots or full, <key>]", {}
        if pred.get("op") in STRING_OPS and any(q[0] != "slots" for q in paths):
            return False, [], "verified", "a string op must cite [slots, <key>]", {}
        rows = [r for sl in slots for r in ctx["slot_rows"].get((art, sl), [])]
        summ = [ctx["verified"].get((f"{art}/slots", sl)) for sl in slots]
        srcs = sorted({x for r in rows for x in r["sources"]})
        if not all(r["equal"] and not r["orphan"] for r in rows) or not all(s_ and s_["equal"] and not s_["orphan"] for s_ in summ):
            return False, srcs, "verified", "a cited slot is not verified in this run", {}
        if pred["op"] in STRING_OPS:
            if len(slots) != 1:
                return False, srcs, "verified", f"{pred['op']} cites more than one slot", {}
            vals = [ctx["released"][art][slots[0]], p["printed"]]
        else:
            vals = [ctx["slot_raw"][art].get(sl) for sl in slots]
            if any(v is None for v in vals):
                return False, srcs, "verified", "a cited slot has no unrounded value equal to the re-run's", {}
        ok, rep = PR.check(pred, vals)
        if pred["op"] == "slot_component_eq" and pred.get("transform") == PR.COMPONENT_TRANSFORMS[1]:  # 09: the sign is stated in words
            link = "sign word not machine-checked"
        return ok, srcs, "verified", "predicate on the verified slot" + (f"; printed_link: {link}" if link else ""), rep
    if art is not None and ctx["kind_of"][art] == "tables34":
        vs = [ctx["by_jpath"].get((art, "/".join(map(str, q)))) for q in paths]
        if any(v is None or not v["equal"] or v["orphan"] for v in vs):
            return False, [], "verified", "a cited Tables 3/4 field is not a verified row", {}
        vals = [float(v["printed"]) if isinstance(v["printed"], str) else v["printed"] for v in vs]
        ok, rep = PR.check(pred, vals)
        return ok, sorted({x for v in vs for x in v["sources"]}), "verified", "Tables 3/4 record field" + (f"; printed_link: {link}" if link else ""), rep
    rec = ctx["records"].get(rec_sha, MISSING)
    if rec is MISSING:
        return False, [], "orphan", "the cited record is not among this run's records", {}
    if rec_sha in (ctx.get("preliminary") or ()):  # 09 (10-20): route C, out of v1 scope (PRELIMINARY), record unsigned; before signed
        bucket = "preliminary"
    elif rec_sha in ctx["signed"]:
        bucket = "verified"
    elif rec_sha in ctx["not_route_C"]:
        bucket = "not_route_C"
    else:
        return False, [], "orphan", "the cited record is neither signed nor a configured not-route-C record", {}
    vals = [walk_key(rec, q) for q in paths]
    if pred.get("op") == "all_detected_at_size":  # 09 op #25: [the nodes, the printed text, the key paths]; a missing key STOPs
        if not paths or any(v is MISSING for v in vals):
            raise PR.PredicateStop("all_detected_at_size: a listed key is absent")
        ok, rep = PR.check(pred, [vals, p["printed"], paths])
        return ok, [rec_sha], bucket, "record read by key; printed_link: qualitative word not machine-checked", rep
    if any(v is MISSING for v in vals):
        return False, [rec_sha], bucket, "a cited key is absent", {}
    if pred.get("op") in STRING_OPS:  # a string cell of a record (first-run finding: honest_all_routec's cells): [its string, the printed text]
        if pred["op"] == "slot_component_eq" or rec_sha not in (ctx.get("preliminary") or ()):
            # R6-3 (09): op #23 only on slot strings the readout gate verified; slot_string_eq on a record cell only for a record 09 pinned
            return False, [rec_sha], bucket, f"{pred['op']} on a record cell is not allowed (only verified readout slots, or a pinned record)", {}
        if len(vals) != 1 or not isinstance(vals[0], str):
            return False, [rec_sha], bucket, f"{pred['op']} needs one string-valued key", {}
        vals = [vals[0], p["printed"]]
    ok, rep = PR.check(pred, vals)
    return ok, [rec_sha], bucket, "record read by key" + (f"; printed_link: {link}" if link else ""), rep


# ---------- the post ----------

def side_of(T, target, col):
    """The tables record's drawn side for a Table 3/4 body row: ("outcome", oc) + Table 4 column -> that outcome's side; ("draw", i) ->
    the draw. -> (side dict, its key path) or (None, None)."""
    kind, key = target
    if T is None:
        return None, None
    if kind == "outcome" and col in SIDE_OF_COL:
        dr = (((T.get("outcome_table") or {}).get("outcomes") or {}).get(key) or {}).get("drawn")
        v = dr.get(SIDE_OF_COL[col]) if isinstance(dr, dict) else None
        return (v, ("outcome_table", "outcomes", key, "drawn", SIDE_OF_COL[col])) if isinstance(v, dict) else (None, None)
    if kind == "draw":
        ds = (T.get("false_alarms") or {}).get("draws") or []
        return (ds[key], ("false_alarms", "draws", key)) if 0 <= key < len(ds) and isinstance(ds[key], dict) else (None, None)
    return None, None


def number_words(text, cover):
    """09 (10-20): every number word of the visible text -> [{word, position, class, sentence (digits masked), covered_by}].
    cover: [(start, end, row id)]; class: fraction / compound_or_duration / cardinal_noun / other (mechanical)."""
    mask = masked_spans(text)
    rx = re.compile(r"\b(" + "|".join(NUMBER_WORDS) + r")\b", re.I)
    out = []
    for m in rx.finditer(text):
        s, e = m.span()
        if any(a <= s < b for a, b in mask):
            continue
        w = m.group(1).lower()
        nxt = re.match(r"\s+([A-Za-z]+)", text[e:e + 30])
        if w in FRACTION_WORDS:
            cls = "fraction"
        elif (e < len(text) and text[e] == "-" and re.match(r"-[A-Za-z]", text[e:e + 2])) or (s > 0 and text[s - 1] == "-"):
            cls = "compound_or_duration"
        elif nxt and nxt.group(1).lower() not in FUNCTION_WORDS:
            cls = "cardinal_noun"
        else:
            cls = "other"
        a_, b_ = sentence_of(text, s)
        cov = next((rid for c0, c1, rid in cover if c0 <= s and e <= c1), "NONE")
        out.append({"word": w, "position": s, "class": cls, "sentence": re.sub(r"\d", "#", text[a_:b_].strip())[:200], "covered_by": cov})
    return out


def post_rows(art, text, ctx):
    """-> (rows, plan counts, stops, diagnostics). ctx: see run_numbers_verify._post."""
    toks, kinds = tokens(text)
    pos = {s: (s, e, t) for s, e, t in toks}
    claims, rows, stops = {}, [], []
    diag = {"predicate_reports": {}, "cell_index": {}, "command_errors": {}, "command_error_rows": [], "command_definitions": [], "findings": []}
    cmd_note = {}
    # a cited verified row of this run was accepted with its own sources (e.g. the D1 calibration files tables34 reads, a readout's
    # constants): the post accepts exactly those, plus the signed set (first-run finding: map rows on such rows were false orphans)
    sig = set(ctx["signed"]) | set(ctx.get("upstream_sources") or ())
    pre_labels = ctx.get("preliminary_labels") or {}
    for x in sorted(set(ctx.get("preliminary") or ()) & set(ctx["signed"])):  # R6-1 (09): a PRELIMINARY pin on a signed record needs its label
        if x not in pre_labels:
            stops.append(f"{art}: a PRELIMINARY record ({x[:12]}) is also in the signed set and has no label override (STOP)")
    for x in sorted(pre_labels):  # R7-2: a label override is only for a record that is BOTH pinned PRELIMINARY and signed
        if x not in set(ctx.get("preliminary") or ()) or x not in set(ctx["signed"]):
            stops.append(f"{art}: a PRELIMINARY label override ({x[:12]}) names a record that is not both PRELIMINARY and signed (STOP)")
    diag["preliminary_labels_used"] = sorted({v for x, v in pre_labels.items()  # only overrides on records both pinned and signed apply
                                              if x in set(ctx.get("preliminary") or ()) and x in set(ctx["signed"])})
    pats = ctx["patterns"]
    lits = ctx.get("literals")
    if lits:
        stops += [f"{art}: {x}" for x in literal_problems(text, lits)]
    acc = set(ctx.get("accepted_reasons") or ())
    n_vis, n_unc = digit_coverage(text, toks)
    if n_unc:
        stops.append(f"{art}: {n_unc} of {n_vis} visible digits lie in no token")
    blocks = {nm: table_block(text, h) for nm, h in TABLE_HEADERS.items()}
    for nm, b in blocks.items():
        if b is None:
            stops.append(f"{art}: {nm}'s header line does not occur exactly once")

    def body_cell(s):  # a Tables 3/4 BODY position -> (table, row, cell); header and separator lines are not body
        for nm, b in blocks.items():
            if not b:
                continue
            ri = next((i + 1 for i, (rs, re_) in enumerate(b[3]) if rs <= s < re_), None)
            if ri:
                ci = next((c + 1 for c, (cs, ce) in enumerate(b[4][ri - 1]) if cs <= s < ce), None)
                return nm, ri, ci
        return None

    def in_tables34(s):
        return any(b and b[0] <= s < b[1] for b in blocks.values())

    def claim(s, kind, ok, srcs, bucket):
        claims.setdefault(s, []).append((kind, ok, srcs, bucket))

    # R3: the positions the swap wrote (Run A's tracked insertions, mapped through the hand-edit diff), never inside Tables 3/4
    rc = {"claims": 0, "in_tables34_dropped": 0, "position_mismatch": 0, "tokens_matched": 0, "tokens_unmatched": 0}
    for c in ctx["rule_claims"]:
        rc["claims"] += 1
        if text[c["start"]:c["end"]] != c["inserted"]:
            rc["position_mismatch"] += 1
            continue
        if in_tables34(c["start"]) or in_tables34(max(c["start"], c["end"] - 1)):
            rc["in_tables34_dropped"] += 1
            continue
        cand = [(k, i_, t) for k in c["keys"] for i_, t in enumerate(VC.NUM_RX.findall(ctx["released_all"].get(k, "")))]
        for s, e, t in toks:
            if not (c["start"] <= s and e <= c["end"]):
                continue
            hit = [(k, i_) for k, i_, ct in cand if ct == t or (_sign(t) is None and _sign(ct) == "+" and _signfree(ct) == t)
                   or (_sign(t) and _sign(ct) == _sign(t) and _signfree(ct) == _signfree(t))]
            vr = [ctx["verified"].get((ra, f"{k}#{i_}")) for k, i_ in hit for ra in ctx["readout_of_slot"].get(k, [])]
            good = [r for r in vr if r and r["equal"] and not r["orphan"]]
            if good:
                rc["tokens_matched"] += 1
                claim(s, "rule:" + re.sub(r"[^A-Za-z0-9_.-]", "_", c["rule"])[:40], True, sorted({x for r in good for x in r["sources"]}),
                      "verified")
            else:
                rc["tokens_unmatched"] += 1
    if rc["position_mismatch"]:
        stops.append(f"{art}: {rc['position_mismatch']} swap insertions are not at their mapped position in v84")
    diag["rule_claims"] = rc
    map_rows, by_bucket, words_rows, word_cover = [], {"verified": 0, "not_route_C": 0, "preliminary": 0, "orphan": 0}, [], []
    unclaimed_in_span = rows_with_unclaimed = 0
    sign_rows, qual_rows = [], []
    for j, p in enumerate(ctx["provenance"]):  # writing's provenance map
        rid = p.get("row", j)
        if p.get("v84_sha256") != ctx["text_sha"]:
            spans, words_only, problem, crep, pspans = [], False, "the row is not for this v84", {}, []
        else:
            spans, words_only, problem, crep, pspans = anchor_spans(text, toks, p)
        if crep:
            diag["cell_index"][str(rid)] = crep
        if problem:
            ok, srcs, bucket, note, rep = False, [], "orphan", problem, {}
        else:
            try:
                ok, srcs, bucket, note, rep = map_check(p, ctx)
            except PR.PredicateError as e:
                ok, srcs, bucket, note, rep = False, [], "verified", f"predicate error: {str(e)[:80]}", {}
                if isinstance(e, PR.PredicateStop):  # 09 op #25: a missing key, field or argument STOPs the pass
                    stops.append(f"{art}: map row {rid}: {str(e)[:80]} (STOP)")
        if rep:  # counts only leave the container (a list is reported by its length; the row keeps it)
            diag["predicate_reports"][f"map:{rid}"] = {(f"n_{k}" if isinstance(v, list) else k): (len(v) if isinstance(v, list) else v)
                                                      for k, v in rep.items() if isinstance(v, (list, int)) and not isinstance(v, bool)}
        claimed = []
        if not problem:  # R2 at the token level: the row claims only the tokens it verifies; the rest of its span falls through
            claimed, cprob = claim_tokens(p, spans, text, ctx)
            if cprob:
                ok, note = False, cprob
                claimed = spans
        unclaimed_in_span += len(spans) - len(claimed)
        rows_with_unclaimed += len(claimed) < len(spans)
        by_bucket[bucket] += 1
        word_cover += [(a0, a1, f"map row {rid}") for a0, a1 in pspans]
        pr_ = p.get("predicate") or {}
        if pr_.get("op") == "slot_component_eq" and pr_.get("transform") == PR.COMPONENT_TRANSFORMS[1]:  # 09: sign stated in words
            sign_rows += [(rid, s) for s, _, _ in claimed]
        if pr_.get("op") == "all_detected_at_size":  # 09 op #25: the qualitative word joins the number-word list, not machine-checked
            qual_rows += [(rid, s, str(pr_.get("qualitative_word"))[:40]) for s, _, _ in claimed]
        for s, _, _ in claimed:
            claim(s, f"map:{rid}", ok, srcs, bucket)
        if words_only and not problem:
            words_rows.append((rid, ok, srcs, bucket))
        diag.setdefault("map_rows", {})[str(rid)] = {"op": (p.get("predicate") or {}).get("op", "-"), "ok": bool(ok and (claimed or words_only)),
                                                    "bucket": bucket, "note": note[:120], "claimed": len(claimed), "span": len(spans)}
        if str(rid) in (ctx.get("row_scope_notes") or {}):  # 09 (10-20): a row whose result line must state its scope
            diag["map_rows"][str(rid)]["scope"] = ctx["row_scope_notes"][str(rid)][:200]
        rec_nm = (parse_record(p.get("record")) or ("-", "-"))[0].rsplit("/", 1)[-1]
        map_rows.append(VC.row(f"{art}/map", f"map:{rid}", rec_nm, json.dumps(p.get("key_path"))[:200], f"<{len(claimed)} of {len(spans)} tokens>",
                               None, (p.get("predicate") or {}).get("op", "-"), ok and (bool(claimed) or words_only),
                               srcs if bucket == "verified" else ([bucket] if bucket in BUCKET_LABEL else []),
                               sig | set(BUCKET_LABEL), note, unit=bucket if bucket in BUCKET_LABEL else "check", bucket=bucket,
                               report=rep, check_row=True))
    row_target = {}  # Tables 3/4: which outcome / draw a body row shows, from the map rows anchored in it
    for p in ctx["provenance"]:
        if p.get("v84_sha256") != ctx["text_sha"] or text.count(p["anchor"]) != 1:
            continue
        bc = body_cell(text.index(p["anchor"]))
        if not bc:
            continue
        for q in _paths(p.get("key_path")):
            q = list(q)
            if q[:2] == ["outcome_table", "outcomes"] and len(q) > 2:
                row_target.setdefault(bc[:2], set()).add(("outcome", q[2]))
            elif q[:2] == ["false_alarms", "draws"] and len(q) > 2 and type(q[2]) is int:
                row_target.setdefault(bc[:2], set()).add(("draw", q[2]))
    field_hits, matched_paths = {}, set()
    for s, e, t in toks:  # Tables 3/4 body numbers: drawn-side field matches (C2), and the command-content class (C3)
        bc = body_cell(s)
        if not bc:
            continue
        nm, ri, ci = bc
        tg = row_target.get((nm, ri)) or set()
        side, base = side_of(ctx.get("T"), next(iter(tg)), ci) if len(tg) == 1 else (None, None)
        if side is not None:
            names = sorted("/".join(map(str, p_)) for p_, v_ in VC.numeric_leaves(side) if v_ == _num(t))
            matched_paths |= {"/".join(map(str, base)) + "/" + n_ for n_ in names}
            field_hits[f"{nm} row {ri} cell {ci} @{s}"] = names
        form = next((f for f, rx in COMMAND_FORMS.items() if rx.match(text, e)), None)
        if s not in claims and form and side is not None and re.fullmatch(r"\d+", t):
            try:
                res_ = ctx["command_check"](side, form, int(t))
                eq, tests_, ph = (res_[0], dict(res_[1]), "") if isinstance(res_, tuple) else (res_, {}, "")
            except VC.Stop as st:
                eq, tests_, ph = None, {}, str(st).rsplit(": ", 1)[-1][:12]
            claim(s, "command", eq is True, ["command-content"] if eq is True else [], "verified")
            cell_ = f"{nm} row {ri} cell {ci}"
            if ph:
                diag["command_error_rows"].append(f"{cell_}: {ph}")
            true_ = sorted(k for k, v in tests_.items() if v is True)  # 09 (b): definition names and booleans only
            app_ = set(LINE_DEFS) if form == "line_count" else {"integer_after_LIMIT"}  # 09 I1: every cell lists the same names
            diag["command_definitions"].append({"cell": cell_, "form": form, "true": true_, "tested": sorted(tests_),
                                               "definitions": [{"name": n_, "applicable": n_ in app_, "true": n_ in true_} for n_ in ALL_DEFS]})
            cmd_note[s] = ("covered under: " + ", ".join(true_)) if true_ else ("matches no tested definition" if tests_ else "")
            if true_ == ["newline_count_plus_1"]:  # 09 I2
                cmd_note[s] += "; matched only by +1 (possible trailing-newline off-by-one)"
                diag["findings"].append(f"{cell_}: matched only by +1 (possible trailing-newline off-by-one)")
            if tests_ and not true_:
                diag["findings"].append(f"{nm} " + ("line count" if form == "line_count" else "limit") + f" matches no tested definition ({cell_})")
    cpos, cprob, occ_bad = constants_positions(text, toks, ctx["constants"])
    stops += [f"{art}: {x}" for x in cprob]
    hpos, h_bad = header_positions(text, toks, ctx.get("header_labels") or [])
    if h_bad:
        stops.append(f"{art}: {h_bad} approved header labels are not at their listed position in a table header row")
    verified_by_block = {}
    for s, cl in claims.items():
        if all(c[1] for c in cl) and all(c[3] == "verified" for c in cl):
            verified_by_block.setdefault(block_of(text, s), set()).add(abs(_num(pos[s][2])))
    hyp = [c for c in ctx["constants"] if c.get("class") == "hypothetical" and c.get("anchor_sentence")]
    hyp_chk = PR.hypothetical_checks(hyp, ctx["hyp_exprs"], text, verified_by_block, lambda o: block_of(text, o))
    dz_all, dz_problems = deferred_regions(text, ctx["deferred"])
    stops += [f"{art}: {x}" for x in dz_problems]
    diag["deferred_record_checks"] = {}
    dc_labels = {str(c_["deferred_row"]): c_["label"] for c_ in ctx.get("deferred_record_checks") or [] if c_.get("label")}
    for chk in ctx.get("deferred_record_checks") or []:  # 09 (10-20): a deferred row checked against its record by key, booleans only
        rid_ = chk["deferred_row"]
        reg = next(((r[0], r[1]) for r in dz_all if deferred_row_id(ctx["deferred"][r[4]], r[4]) == rid_), None)
        rec_ = ctx["records"].get(chk["record_sha256"], MISSING)
        try:
            if reg is None or rec_ is MISSING:
                raise ValueError("no region or no record")
            spans_, lo_ok, hi_ok = bounds_check(text, reg, rec_, chk["keys"], chk.get("u_common"), chk.get("u_total"))
        except (ValueError, TypeError, ZeroDivisionError):
            stops.append(f"{art}: deferred row {rid_}: its record check could not run (a missing key, not one printed bounds pair, "
                         "or the record's unseen total is not the configured total)")
            diag["deferred_record_checks"][str(rid_)] = {"ran": False}
            continue
        diag["deferred_record_checks"][str(rid_)] = {"ran": True, "lower_equal": lo_ok, "upper_equal": hi_ok}
        for (a0, _a1), ok_ in zip(spans_, (lo_ok, hi_ok), strict=True):
            if a0 in pos:
                claim(a0, f"deferred_check:{rid_}", ok_, ["not_route_C"], "not_route_C")
    notes = ctx.get("orphan_notes") or []
    final, final_ok = {}, {}
    for s, e, t in toks:
        cl = claims.get(s)
        if cl:
            kinds_ = "+".join(c[0] for c in cl)
            ok = all(c[1] for c in cl)
            srcs = sorted({x for c in cl for x in c[2]})
            bucket = "orphan" if any(c[3] == "orphan" for c in cl) else next((b_ for b_ in ("preliminary", "not_route_C") if any(c[3] == b_ for c in cl)),
                                                                                 "verified")
            final[s], final_ok[s] = cl[0][0].split(":")[0], ok and bucket != "orphan"
            if len(cl) > 1:
                diag["tokens_claimed_twice"] = diag.get("tokens_claimed_twice", 0) + 1
            if bucket in ("not_route_C", "preliminary"):
                lab_ = next((pre_labels[x] for c in cl for x in c[2] if x in pre_labels and x in set(ctx["signed"])),
                            BUCKET_LABEL[bucket]) if bucket == "preliminary" \
                    else next((dc_labels[c[0].split(":", 1)[1]] for c in cl if c[0].startswith("deferred_check:")  # 09: row 29's own label
                               and c[0].split(":", 1)[1] in dc_labels), BUCKET_LABEL[bucket])  # R6-1: a PRELIMINARY record's per-record label (09)
                rows.append(VC.row(art, f"{kinds_}@{s}", "-", kinds_, t, None, lab_, ok, [bucket], sig | {bucket}, unit=bucket, kind=kinds[s]))
            else:
                rows.append(VC.row(art, f"{kinds_}@{s}", "-", kinds_, t, None, kinds_.split(":")[0], ok, srcs if bucket == "verified" else [],
                                   sig | {"command-content"}, cmd_note.get(s, ""), kind=kinds[s]))
            continue
        bc = body_cell(s)
        if bc:  # RQ3: inside Tables 3/4 body rows only map or command rows count
            final[s], final_ok[s] = "orphan", False
            rows.append(VC.row(art, f"{bc[0].replace(' ', '')}@{s}", "-", "-", t, None, "a Table 3/4 number with no map or command row", False,
                               [], sig, "RQ3: only map rows on the tables record or command rows count here", kind=kinds[s]))
            continue
        a_, b_ = sentence_of(text, s)
        nt = [n["note"] for n in notes if n["near"] in text[a_:b_]]
        if nt:  # 09 (10-20): an explicit orphan note overrides any deferred region
            final[s], final_ok[s] = "orphan", False
            rows.append(VC.row(art, f"orphan@{s}", "-", "-", t, None, "orphan note", False, [], sig, "; ".join(nt), kind=kinds[s]))
            continue
        c = cpos.get(s)
        if c is not None and c.get("class") == "hypothetical":
            chk = hyp_chk.get(c["row"]) or {}
            okh = bool(chk) and all(chk.values()) and c.get("checks_ok") is True
            final[s], final_ok[s] = "hypothetical", okh
            rows.append(VC.row(art, f"hypothetical@{s}", "-", f"constants row {c['row']}", t, None, "hypothetical: no data claim", okh,
                               ["nonresult"], {"nonresult"}, ",".join(f"{k}={v}" for k, v in sorted(chk.items())), unit="nonresult",
                               nr_class="hypothetical", matched=c["text"], pos=s, checks_ok=okh, kind=kinds[s], by=f"constants row {c['row']}"))
            continue
        if c is not None:
            okc = c.get("checks_ok") is True
            final[s], final_ok[s] = c["class"], okc
            rows.append(VC.row(art, f"{c['class']}@{s}", "-", f"constants row {c['row']}", t, None, c["class"], True, ["nonresult"], {"nonresult"},
                               unit="nonresult", nr_class=c["class"], matched=c["text"], pos=s, checks_ok=okc, kind=kinds[s],
                               by=f"constants row {c['row']}"))
            continue
        hx = hpos.get(s)
        if hx is not None:
            final[s], final_ok[s] = "header_label", True
            rows.append(VC.row(art, f"header_label@{s}", "-", f"table {hx.get('table')} header", t, None, "header_label", True, ["nonresult"],
                               {"nonresult"}, unit="nonresult", nr_class="header_label", matched=hx.get("cell"), pos=s, checks_ok=True,
                               by=f"header label table {hx.get('table')} cell {hx.get('cell_index')}",
                               kind=kinds[s]))
            continue
        lc = literal_class(text, s, e, t, lits) if lits else None
        if lc:
            cls, matched, by_ = lc
            final[s], final_ok[s] = cls, True
            rows.append(VC.row(art, f"{cls}@{s}", "-", matched, t, None, cls, True, ["nonresult"], {"nonresult"}, unit="nonresult", nr_class=cls,
                               matched=matched, pos=s, checks_ok=True, kind=kinds[s], by=by_))
            continue
        nc = pattern_class(text, s, e, t, kinds[s], pats)
        if nc:
            cls, matched = nc
            final[s], final_ok[s] = cls, True
            rows.append(VC.row(art, f"{cls}@{s}", "-", matched, t, None, cls, True, ["nonresult"], {"nonresult"}, unit="nonresult",
                               nr_class=cls, matched=matched, pos=s, checks_ok=True, kind=kinds[s], by=f"pattern {cls}"))
            continue
        dz = [r for r in dz_all if r[0] <= s < r[1]]
        if dz:
            j = dz[0][4]
            rok = bool(ctx["deferred_record_ok"].get(j))
            final[s], final_ok[s] = "deferred", rok and dz[0][2] in acc
            rows.append(VC.row(art, f"deferred:{dz[0][3]}@{s}", "-", dz[0][3], t, None, "NOT VERIFIED (" + dz[0][2] + ")", True, ["deferred"],
                               {"deferred"}, "" if rok else "record null or not read by sha (orphan)", unit="deferred", reason=dz[0][2],
                               record_ok=rok, deferred_row=deferred_row_id(ctx["deferred"][j], j), kind=kinds[s]))
            continue
        final[s], final_ok[s] = "orphan", False
        rows.append(VC.row(art, f"orphan@{s}", "-", "-", t, None, "no class", False, [], sig, "outside every class", kind=kinds[s]))
    for rid, ok, srcs, bucket in words_rows:  # a provenance row printed in words only (no digit token): one row of its own
        rows.append(VC.row(art, f"map:{rid}@words", "-", f"map:{rid}", "<words>", None, "map", ok,
                           srcs if bucket == "verified" else ([bucket] if bucket in BUCKET_LABEL else []), sig | set(BUCKET_LABEL),
                           unit=bucket if bucket in BUCKET_LABEL else "number"))
    # R10 / 09 (a): a number on a hand-edited line is covered only by a PASSING map, constants, deferred, pattern, header or command row
    ok_cls = {"map", "deferred", "constant", "definitional", "citation", "hypothetical", "reference", "appendix", "footnote_marker", "date",
              "identifier", "header_label", "command",
              "run_label", "identifier_literal", "placeholder", "command_literal", "citation_literal", "deferred_check"}  # R6-2 (09)
    he = [s for s, _, _ in toks if any(a <= s < b for a, b in ctx.get("added_spans") or [])]
    he_bad = [s for s in he if final.get(s) not in ok_cls or not final_ok.get(s)]
    he_by = {}
    for s in he:
        he_by[final.get(s, "?")] = he_by.get(final.get(s, "?"), 0) + 1
    he_rows = [VC.row(f"{art}/hand_edit", f"hand_edit@{s}", "-", final.get(s, "?"), pos[s][2], None,
                      "a number on a hand-edited line must be a passing provenance, constants or deferred row", False, [], sig) for s in he_bad]
    if he_bad:
        stops.append(f"{art}: {len(he_bad)} numbers on hand-edited lines are not covered by a passing provenance, constants or deferred row")
    np_v84 = sorted(set(ctx.get("not_printed") or []) - matched_paths)  # C2: a fresh 09 approval for this pass
    np_sha = VC.sha("\n".join(np_v84).encode())
    a2 = ctx.get("not_printed_v84_approved") or {}
    if "record-pass-only" in str(a2.get("scope", "")) or (a2.get("sha256"), a2.get("count")) != (np_sha, len(np_v84)):
        stops.append(f"{art}: the not-printed list for this pass (count {len(np_v84)}, sha256 {np_sha}) has no v84 approval from 09")
    # R8: Tables 3/4's printed flag: every body number a PASSING route-C row (map on verified sources, or command), and no post stop
    t34 = [r for r in rows if r["artifact"] == art and "@" in r["slot"] and r["slot"].rsplit("@", 1)[1].isdigit()
           and body_cell(int(r["slot"].rsplit("@", 1)[1]))]
    t34_ok = bool(t34) and all(r["equal"] and not r["orphan"] and r["unit"] == "number" for r in t34) and not stops
    # number words (09, 10-20): listed with class and covering row
    cover = word_cover + [(text.index(c["anchor"]) + c["anchor"].index(c["text"]),
                           text.index(c["anchor"]) + c["anchor"].index(c["text"]) + len(c["text"]), f"constants row {c['row']}")
                          for c in ctx["constants"] if c.get("anchor") and text.count(c["anchor"]) == 1 and c["text"] in c["anchor"]]
    cover += [(r[0], r[1], f"deferred row {deferred_row_id(ctx['deferred'][r[4]], r[4])}") for r in dz_all]
    nw = number_words(text, cover)
    for rid, s in sign_rows:  # 09 (10-20): the abs-transform rows join the number-word list (the sign word is not machine-checked)
        a_, b_ = sentence_of(text, s)
        nw.append({"word": "(sign stated in words)", "position": s, "class": "sign_word_not_machine_checked",
                   "sentence": re.sub(r"\d", "#", text[a_:b_].strip())[:200], "covered_by": f"map row {rid}"})
    for rid, s, w in qual_rows:  # 09 op #25: the qualitative word, reported "not machine-checked"
        a_, b_ = sentence_of(text, s)
        nw.append({"word": w, "position": s, "class": "qualitative_word_not_machine_checked",
                   "sentence": re.sub(r"\d", "#", text[a_:b_].strip())[:200], "covered_by": f"map row {rid}"})
    nw, nw_t34, nw_unknown = withhold_tables34(nw, blocks)  # 2e / 09 (10-20): no Tables 3/4 text in the number-word list; fail closed
    if ctx.get("numv5_positions") is not None:  # 09 (10-20): numv5's committed positions, booleans and a count only
        diag["numv5_number_word_positions"] = numv5_positions_check(blocks, ctx["numv5_positions"]) | {"source": ctx.get("numv5_source")}
        if diag["numv5_number_word_positions"]["status"] == "unknown":
            stops.append(f"{art}: numv5's number-word positions cannot be checked (Tables 3/4 boundaries unknown) (STOP)")
    nw_by = {}
    for x in nw:
        k_ = f"{x['class']}|{'covered' if x['covered_by'] != 'NONE' else 'uncovered'}"
        nw_by[k_] = nw_by.get(k_, 0) + 1
    # 09 (10-20): the non-result entries (class, token as printed, matched post text, position, classifier), no record values; and
    # the approved header-label positions another class took first (precedence: constants before header labels)
    nr_entries = [{"class": r["nr_class"], "token": r["printed"], "matched": r["matched"], "pos": r["pos"], "by": r.get("by", "-"),
                   "line": VC.nonresult_line(r)} for r in rows if r.get("unit") == "nonresult"]
    taken = {x["position"]: final.get(x["position"], "?") for x in ctx.get("header_labels") or [] if final.get(x["position"]) != "header_label"}
    tk_by = {}
    for c_ in taken.values():
        tk_by[c_] = tk_by.get(c_, 0) + 1
    diag["nonresult_entries"] = nr_entries
    diag["header_labels_taken_by_other_classes"] = {"count": len(taken), "positions": sorted(taken), "by_class": tk_by}
    diag["run_label_verb_use"] = sorted(e_["pos"] for e_ in nr_entries if e_["class"] == "run_label" and e_["by"].endswith("(verb use)"))
    by_cls = {}
    for c_ in final.values():
        by_cls[c_] = by_cls.get(c_, 0) + 1
    by_kind = {}
    for s in kinds:
        by_kind[kinds[s]] = by_kind.get(kinds[s], 0) + 1
    diag["map_spans"] = {"tokens_left_unclaimed_in_spans": unclaimed_in_span, "rows_with_unclaimed_tokens": rows_with_unclaimed}
    diag |= {"tokens": len(toks), "tokens_by_kind": by_kind, "visible_digits": n_vis, "digits_outside_tokens": n_unc,
             "tokens_by_class": by_cls, "provenance_by_bucket": by_bucket, "words_only_map_rows": len(words_rows),
             "hand_edit_numbers": {"total": len(he), "uncovered": len(he_bad), "by_class": he_by},
             "not_printed_v84": {"paths": np_v84, "sha256": np_sha, "count": len(np_v84), "fields_printed": sorted(matched_paths)},
             "table_cell_field_matches": field_hits, "hypothetical_checks": {str(k): v for k, v in hyp_chk.items()},
             "constants_occurrence_index_disagreements": occ_bad, "header_labels_matched": len(hpos),
             "tables34_numbers": len(t34), "tables34_printed_numbers_verified": t34_ok,
             "number_words": {"total": len(nw), "covered": sum(x["covered_by"] != "NONE" for x in nw),
                              "uncovered": sum(x["covered_by"] == "NONE" for x in nw), "by_class": nw_by,
                              "tables34_sentence_withheld": nw_t34, "all_sentences_withheld_tables34_unknown": nw_unknown, "list": nw}}
    return rows + map_rows + he_rows, {art: len(toks) + len(words_rows), f"{art}/map": len(map_rows), f"{art}/hand_edit": len(he_rows)}, stops, diag
