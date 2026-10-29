"""Rebuild every printed number of the post from the committed records.

Step 1, the slot readouts. The post's numbers were read from four slot readouts (run_slot_readout.py, run on 2026-10-20): slot_table.build
turns the pooled records into the strings the post prints. config/slot_runs.json names each readout's records; build() reruns here on the
same records with byte-identical copies of the code that ran (slots/), and its output must equal the committed slot file exactly.
Step 2, the tokens. Each token of expected/expected_post_numbers.json in class R (rebuilt) or Q (read as recorded; none at present) has
one rule in number_rules.json: "slot" reads a recomputed slot string, "rec" reads a key path of a record, "fn" calls a named function
below. A rule whose claim fails (a bound, a "less than", a "near 100%") raises RuleError; it never returns a printable value.
"""

import importlib.util
import json
import re
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
TOKEN = re.compile(r"(?:−|\+)?[0-9][0-9,]*(?:\.[0-9]+)?%?")  # the tokenizer that made expected_post_numbers.json
MINUS = "−"
LEAD = "pooled_all_origins"
SIX = ("sol_a_uncut", "FIBS|uncut", "harmfulness|uncut", "refusal|uncut", "high-stakes|uncut", "own_all_rs_uncut_L48")  # slot_table.SIX
DISPLAY = {"own_all_rs_uncut_L48": "the on-policy probe", "sol_a_uncut": "the off-policy probe", "FIBS|uncut": "the deception probe",
           "harmfulness|uncut": "the harmfulness direction", "refusal|uncut": "the refusal direction", "high-stakes|uncut": "the high-stakes probe"}
WORDS = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
PLANT_SIZES = (3, 5, 8, 12)  # the planted falls and rises, in points (Table 6)


class RuleError(Exception):
    """A rule that cannot be evaluated (a missing key, members that disagree): reported as a mismatch, never skipped."""


def toks(s):
    return [t.rstrip(",") for t in TOKEN.findall(s)]


def num(t):
    return float(t.replace(MINUS, "-").replace(",", "").replace("%", ""))


def fmt(x, f):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise RuleError(f"{x!r} is not a number")
    return f.format(x).replace("-", MINUS)


def get(d, path):
    for k in path:
        try:
            d = d[k]
        except (KeyError, IndexError, TypeError):
            raise RuleError(f"no key {path}") from None
    return d


def words(r, s):
    if "words" not in r:
        return s
    if s not in r["words"]:
        raise RuleError(f"{s} has no word in the rule")
    return r["words"][s]


def strip(t, r):
    for x in r.get("strip", []):
        t = t.removesuffix("%") if x == "pct" else t[1:] if x == "sign" and t[:1] in (MINUS, "+") else t
    return t


# ---- step 1: the slot readouts ------------------------------------------------------------------------------------------------------

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def slot_readouts(records):
    """Rerun slot_table.build for each readout and compare with the committed slot file. -> ({run: build output}, problems)."""
    cfg = json.loads((HERE / "config" / "slot_runs.json").read_text())
    got, probs = {}, []
    for run, c in cfg.items():
        if run.startswith("_"):
            continue
        st = load_module(HERE / c["code"], f"slot_table_{run}")
        ins = c["inputs"]
        try:
            res = st.build({r: records.json(i["record"]) for r, i in ins.items()}, {r: records.sha(i["record"]) for r, i in ins.items()},
                           {r: i["path"] for r, i in ins.items()})
        except SystemExit as e:  # slot_table STOPs on a label or integrity failure
            probs.append(f"slots {run}: slot_table stopped: {e}")
            continue
        want = records.json(c["committed"])
        bad = [f"slots {run}.{p}[{k!r}] differs from {c['committed']}" for p in ("slots", "full", "missing", "not_carried")
               for k in sorted(set(res[p]) | set(want[p])) if res[p].get(k) != want[p].get(k)]  # one problem per differing key
        probs += bad
        print(f"  slots {run}: {len(res['slots'])} slots, " + (f"equal to {c['committed']}" if not bad else f"{len(bad)} key(s) DIFFERENT"))
        got[run] = res
    return got, probs


# ---- step 2: the rules --------------------------------------------------------------------------------------------------------------

def slot_rule(r, slots):
    if r["slot"] not in slots:
        raise RuleError(f"slot readout {r['slot']} did not run")
    d = slots[r["slot"]][r.get("src", "slots")]
    vals = [get(d, [k]) for k in (r.get("entries") or [r["entry"]])]
    try:
        if r.get("take", "str") == "str":
            if r.get("agg_tokens") == "same" and len({strip(t, r) for t in toks(vals[0])}) != 1:
                raise RuleError(f"the numbers in {vals[0]!r} differ")
            ts = [strip(toks(v)[r["i"]], r) for v in vals]
            agg = r.get("agg", "same")
            if agg == "same" and len(set(ts)) != 1:
                raise RuleError(f"members differ: {ts}")
            return words(r, ts[0] if agg == "same" else (min if agg == "min" else max)(ts, key=num))
        xs = [num(toks(v)[r["i"]]) * r.get("scale", 1) for v in vals]
    except IndexError:
        raise RuleError(f"no number {r['i']} in {vals}") from None
    xs = [abs(x) for x in xs] if r.get("abs") else xs
    agg = r.get("agg", "same")
    if agg == "le_bound":  # "within B points": every member at most B (inclusive); a failed claim is an error, never a printable value
        if max(xs) > r["bound"]:
            raise RuleError(f"the bound fails: the largest member is {max(xs)!r}, above {r['bound']!r}")
        return f"{r['bound']:g}"
    if agg == "same" and len({fmt(x, r['fmt']) for x in xs}) != 1:
        raise RuleError(f"members differ: {xs}")
    x = xs[0] if agg == "same" else min(xs) if agg == "min" else max(xs)
    return words(r, fmt(x, r["fmt"]) + r.get("suffix", ""))


def value_str(x, r):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise RuleError(f"{x!r} is not a number")
    x = abs(x * r.get("scale", 1)) if r.get("abs") else x * r.get("scale", 1)
    if "round_to" in r:  # an "about" count: rounded to the nearest round_to
        x = int(round(x / r["round_to"]) * r["round_to"])
    return fmt(x, r["fmt"]) + r.get("suffix", "")


def rec_rule(r, records):
    d = records.json(r["rec"])
    op = r.get("op", "get")
    if op == "count":  # rows whose field (or list index) equals `eq` or starts with `prefix`
        rows = get(d, r["path"])
        f = (lambda x: x[r["index"]]) if "index" in r else (lambda x: x[r["field"]])
        n = sum(1 for x in rows if (f(x) == r["eq"] if "eq" in r else str(f(x)).startswith(r["prefix"])))
        return words(r, str(n)) if "words" in r else fmt(n, r["fmt"])
    if "regex" in r:
        m = re.search(r["regex"], str(get(d, r["path"])))
        if not m:
            raise RuleError(f"{r['regex']} does not match")
        return m.group(1)
    v = [get(d, r["path"])] if "path" in r else [get(d, p) for p in r["paths"]]
    if op == "same":  # every path must print the same
        ss = {value_str(x, r) for x in v}
        if len(ss) != 1:
            raise RuleError(f"members differ: {sorted(ss)}")
        return ss.pop()
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in v):
        raise RuleError(f"not numbers: {v}")
    ops = {"get": lambda: v[0], "sum": lambda: sum(v), "ratio": lambda: v[0] / v[1], "diff": lambda: v[0] - v[1], "min": lambda: min(v),
           "max": lambda: max(v)}
    x = ops[op]()
    if "below" in r:  # printed as "less than +B": the claim 0 < x < B must hold; a failed claim is an error, never a printable value
        if not 0 < x < r["below"]:
            raise RuleError(f"the claim 0 < x < {r['below']} fails: x = {x!r}")
        return f"+{r['below']:.2f}"
    return value_str(x, r)


def failed_range(r, records, slots):
    """[^failed] 3.7-6.8%: real honest runs that failed, as a share of launched runs (headline common sets), where the upper end counts
    every unrecorded earlier attempt as failed. Unrecorded attempts are the gap keys' redraw indices in the headline sets' launches
    (the launch -> set map comes from the record's own failed-run keys)."""
    d = records.json("appendix_a_rates")
    sets_of = {}
    for set_arm, runs in get(d, ["real", "failed_runs"]).items():
        for run in runs:
            sets_of.setdefault(run["key"].split("/")[0], set()).add(set_arm.split("|")[0])
    gaps = []
    for k in get(d, ["real_lower_bound", "gap_keys"]):
        s, m = sets_of.get(k.split("/")[0], set()), re.search(r"redraw(\d+)", k)
        if len(s) != 1 or not m:
            raise RuleError(f"gap key {k}: no single set or no redraw index")
        gaps.append((next(iter(s)), int(m.group(1))))  # not s.pop(): that would empty the shared set before the launch's next gap key
    if sum(n for _, n in gaps) != get(d, ["real_lower_bound", "unseen_attempts_at_least"]):
        raise RuleError("the gap keys do not add up to unseen_attempts_at_least")
    u = sum(n for s, n in gaps if s in get(d, ["headline_common_sets", "sets"]))
    x = get(d, ["headline_common_sets", "X_real"])
    lo, hi = x["failed"] / (x["launched"] + u), (x["failed"] + u) / (x["launched"] + u)
    return fmt(100 * (lo if r["end"] == "lo" else hi), "{:.1f}") + r.get("suffix", "")


def mostly_range(r, records, slots):
    """App A "mostly layers 45 to 49": the off-policy probe's per-copy tool-output layer picks, without the picks more than 5 layers from
    the median; at least 60% of the picks must remain."""
    picks = [int(v) for pair in get(records.json("sola_fit"), ["fit_spec", "layer_choice_a"]) for v in pair]
    kept = [v for v in picks if abs(v - median(picks)) <= 5]
    if len(kept) < 0.6 * len(picks):
        raise RuleError(f"only {len(kept)} of {len(picks)} picks near the median")
    return str(min(kept) if r["end"] == "lo" else max(kept))


def near_100(r, records, slots):
    """App C "Near 100% catch": the on-policy probe's run 1, run 2 and simulated catch points all at least 99%."""
    q = get(records.json("A"), ["pooled", "own_all_rs_uncut_L48 C", LEAD, "joint", "0.01", "catch"])
    pts = [get(q, [a, "point"]) for a in ("run1", "run2", "sim")]
    if not all(p >= 0.99 for p in pts):
        raise RuleError(f"the claim fails: catch points {pts}, not all at least 0.99")
    return "100%"


def realline_mean(r, records, slots):
    """[^realline] points: the six-probe mean of simulated minus real catch against the real threshold (record A's per-probe real_line
    points, draw 0). The interval is realline_interval's."""
    xs = []
    for lb in SIX:
        q = get(records.json("A"), ["pooled", f"{lb} {r['ro']}", LEAD, "joint", "0.01"])
        if q.get("readable_real_line") is not True:
            raise RuleError(f"{lb} {r['ro']}: real_line not readable")
        xs.append(get(q, ["gap", "real_line", "point"]))
    return fmt(100 * sum(xs) / len(xs), "{:.1f}")


def summ(v, D):
    """Record A's interval rule (run_fprsweep_507.summ @ a936a5d97, verbatim): point = draw 0; 2.5 / 97.5 percentiles of the finite
    resampled draws 1..D; withheld below 95% complete draws."""
    import numpy as np

    f = v[1:][np.isfinite(v[1:])]
    o = {"point": float(v[0]), "finite_draws": int(len(f)), "draws": D}
    if len(f) < 0.95 * D:
        return o | {"ci95": None, "withheld": "fewer than 95% complete draws"}
    return o | {"ci95": [float(np.percentile(f, 2.5)), float(np.percentile(f, 97.5))], "degenerate": bool(np.ptp(f) == 0)}


_SIX_REALLINE = {}


def realline_six(records, ro):
    """The six-probe real_line mean at pooled_all_origins / joint / 1%, summarised as record A summarises (realline_draws.npz holds record
    A's 12 per-draw real_line arrays, copied by run_extract_realline_draws.py). Gates, each a RuleError: the file has exactly the 12
    arrays; each array has 10,001 draws and its own summary equals record A's stored real_line node exactly; the mean is taken draw by
    draw in record A's co-headline row order (as run_realline_boot_read.py did) and must equal the realline record exactly."""
    import numpy as np

    if ro in _SIX_REALLINE:
        return _SIX_REALLINE[ro]
    Z = records.npz("realline_draws")
    member = {f"{lb} {r_}": f"{lb}|{r_}".replace("|", "__").replace(" ", "_") for lb in SIX for r_ in ("C", "R")}  # the copy job's names
    if sorted(Z.files) != sorted(member.values()):
        raise RuleError(f"realline_draws.npz holds {sorted(Z.files)}, not the 12 expected arrays")
    A = records.json("A")
    rows = get(A, ["coheadline_signed", LEAD, "joint", "0.01", f"headline_six_full|{ro}", "rows"])
    if sorted(rows) != sorted(f"{lb} {ro}" for lb in SIX):
        raise RuleError(f"record A's co-headline rows are {rows}")
    arrs, Ds = [], set()
    for lab in rows:
        q = get(A, ["pooled", lab, LEAD, "joint", "0.01"])
        node = get(q, ["gap", "real_line"])
        a = np.asarray(Z[member[lab]], dtype=np.float64)
        if q.get("readable_real_line") is not True or a.shape != (node["draws"] + 1,):
            raise RuleError(f"{lab}: not readable, or {a.shape[0]} draws where record A has {node['draws']} + 1")
        s = summ(a, node["draws"])
        if any(s.get(k) != node.get(k) for k in ("point", "ci95", "draws", "finite_draws", "degenerate")):
            raise RuleError(f"{lab}: the draws do not reproduce record A's stored real_line summary")
        arrs.append(a)
        Ds.add(node["draws"])
    if len(Ds) != 1:
        raise RuleError(f"the six probes have different draw counts {sorted(Ds)}")
    six = summ(np.mean(arrs, 0), Ds.pop())
    if six.get("ci95") is None:
        raise RuleError("the six-probe interval is withheld (fewer than 95% complete draws)")
    rec = get(records.json("realline"), ["results", "six_probe", f"{ro}|0.01", "real_line_mean"])
    if 100 * six["point"] != rec["point"] or [100 * x for x in six["ci95"]] != rec["ci95"]:
        raise RuleError("the rebuilt six-probe summary differs from the realline record")
    _SIX_REALLINE[ro] = six
    return six


def realline_interval(r, records, slots):
    """[^realline] intervals: one end of the six-probe real_line mean's 95% interval, in points."""
    six = realline_six(records, r["ro"])
    return fmt(100 * six["ci95"][0 if r["end"] == "lo" else 1], "{:+.1f}")


def plant_parts(slots):
    """Table 6 / App D: the aggregate verdicts and the per-probe counts of the planted-error test (slot readout rc1b, PLANT rows)."""
    if "rc1b" not in slots:
        raise RuleError("slot readout rc1b did not run")
    d = slots["rc1b"]["slots"]

    def verdict(lv, k, agg):
        return get(d, [f"25.plant.aggregate.{lv}.C.-{k}.{agg}"]).split(";")[0]

    def count(tag):
        s = get(d, [f"25.plant.count.0.01.{tag}.C"])
        m = re.fullmatch(r"([\d,]+)/([\d,]+) detected, ([\d,]+) missed, ([\d,]+) unavailable(?: \((.*)\))?", s)
        if not m:
            raise RuleError(f"count slot {tag}: unexpected form")
        det, n, _, u = (int(x.replace(",", "")) for x in m.groups()[:4])
        named = set()
        for part in (m.group(5) or "").split("; "):
            if part:
                named |= set(part.split(": ", 1)[1].split(", "))
        return {"detected": det, "available": n - u, "detected_names": [lb for lb in SIX if f"{lb} C" not in named]}
    return verdict, count


def single_all(count, k):
    f, r = count(f"-{k}"), count(f"+{k}")
    return f["detected"] == f["available"] and r["detected"] == r["available"]


def plant(r, records, slots):
    verdict, count = plant_parts(slots)
    w = r["what"]

    def smallest(ok):  # the smallest size from which every larger size is also ok
        for i, k in enumerate(PLANT_SIZES):
            if all(ok(k2) for k2 in PLANT_SIZES[i:]):
                return k
        return None
    if w in ("signed_smallest", "gap_smallest"):
        k = smallest(lambda k: verdict("0.01", k, "S" if w == "signed_smallest" else "excess") == "detected")
    elif w == "gap_missed_at_1":
        ks = [k for k in PLANT_SIZES if verdict("0.01", k, "excess") != "detected"]
        k = ks[0] if len(ks) == 1 else None
    else:
        k_all = smallest(lambda k: single_all(count, k))
        if w == "single_all_smallest":
            k = k_all
        else:
            below = [k for k in PLANT_SIZES if k_all is None or k < k_all]
            k = (min(below) if w == "single_notall_min" else max(below)) if below else None
    if k is None:
        raise RuleError(f"plant {w}: no planted size satisfies the claim")
    return str(k)


def t6(r, records, slots):
    """One Table 6 cell, in the table's words."""
    verdict, count = plant_parts(slots)
    k, col = r["k"], r["col"]
    if col in ("signed", "gap"):
        a, b = (verdict(lv, k, "S" if col == "signed" else "excess") == "detected" for lv in ("0.01", "0.05"))
        if col == "signed" and a and b:
            return "noticed at 1% and at 5%"
        if col == "gap" and a:
            return "noticed at 1%"
        return "noticed at 1%, missed at 5%" if a else "missed at 1%, noticed at 5%" if b else "missed at 1% and at 5%"
    c = count(f"-{k}" if col == "falls" else f"+{k}")
    if c["detected"] == 0:
        return "none"
    if c["detected"] == c["available"]:
        return f"all {WORDS[c['available']]}"
    if c["detected"] == 1:
        return f"{DISPLAY[c['detected_names'][0]]} only"
    return WORDS[c["detected"]]


def t4_nothing_to_score(r, records, slots):
    """App B: Table 4 rows in which a drawn run had nothing to score (no tool calls; the record stores a status string, not a score)."""
    oc = get(records.json("tables"), ["outcome_table", "outcomes"])
    return words(r, str(sum(1 for o in oc.values() if any(isinstance(get(o, ["drawn", a]), str) for a in ("real (run 1)", "sim (sim 1)")))))


def one_in_n(r, records, slots):
    """App A "about one in seven": 1 / (the share of the public probes' scored calls the trained probes skip), the same in every pool."""
    f = slots["rc1a"]["full"] if "rc1a" in slots else {}
    ns = {round(1 / float(get(f, [f"17.readout_share.{p}"]))) for p in ("A", "B", "S")}
    if len(ns) != 1:
        raise RuleError(f"pools differ: {sorted(ns)}")
    return words(r, str(ns.pop()))


def per_main(r, records, slots):
    """App A "about two and about one per main task": honest runs behind the threshold per attacked main task, rounded."""
    f = slots["rc1a"]["full"] if "rc1a" in slots else {}
    lab = "own_all_rs_uncut_L48 C"
    return words(r, str(round(int(get(f, [f"14.after_runs.{lab}.{r['pool']}"])) / int(get(f, [f"14.attack_mains.{lab}"])))))


FN = {"failed_range": failed_range, "mostly_range": mostly_range, "near_100": near_100, "realline_mean": realline_mean,
      "realline_interval": realline_interval, "plant": plant, "t6": t6, "t4_nothing_to_score": t4_nothing_to_score, "one_in_n": one_in_n,
      "per_main": per_main}


def rebuild(records, slots, rules):
    """rules: number_rules.json without _about (sha-pinned in MANIFEST.json). -> ({token id: rebuilt string}, {token id: why it failed}).
    A token with an error gets no rebuilt value, and compare() counts it as a difference."""
    out, errors = {}, {}
    for tid, r in rules.items():
        try:
            out[tid] = slot_rule(r, slots) if "slot" in r else rec_rule(r, records) if "rec" in r else FN[r["fn"]](r, records, slots)
        except (RuleError, KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as e:
            errors[tid] = f"{type(e).__name__}: {e}"
    return out, errors
