"""Route C numbers verifier, the pure core (09's pre-stated design, 10-20 PT). Parsed inputs in; JSONL rows and keys-only counts out.

It runs inside the Modal container (run_numbers_verify.py). On the laptop only test_numbers_verify.py calls it, on SYNTHETIC fixtures.

One row per printed number: {artifact, slot, record, key_path, printed, recomputed, rule, equal, orphan, note}. `printed` is parsed from
the released artifact; `recomputed` comes from the committed source record by key, through the consumer's own mapping (the table script
or slot table at its pinned blob). PASS = no unmatched row, no orphan row, and each artifact's printed count equal to its consumer's count.
Anything else is a STOP that names slot ids only (never values).

Orphan (09 rule 4): a printed number whose source is not a committed, S2-signed record. A row's `sources` are record shas; the signed set
is the config's. Tables 3/4's z values: covered through A only under 09's conditions (10-20): (1) the z come from the D1 calibration npz
whose sha256 A itself pins (meta.pooled_world.calibrations.by_label_combo), and the bytes opened hash to it; (2) draw-0 T recomputed from
that npz's draw0_z__<pool> by A's quantile rule (the build's own wthr, extracted from its blob) equals A's pools_at_draw0 T bitwise, and
the count of units above T equals A's. If either fails, the z rows of that label and pool are orphans.
Table 4's drawn attack sides (09's ruling (b), 10-20): the score file's pin looked up in FROZEN INPUTS_FULL's C overrides (S1-verified), the
file sha-checked before parsing, the run's units by run_key, unit_z (run_tables_v81, hash-checked blob) with the outcome label's D1 draw-0
moments, argmax_command; z, the argmax unit's index, the unit count and z - T (A's T) are compared.
"""

import ast
import hashlib
import json
import math
import re

import numpy as np

MINUS = "−"
NUM_RX = re.compile(r"[+\-−]?\d[\d,]*(?:\.\d+)?")
LEAD = "pooled_all_origins"
ORIGINS = ("S_old", "S_new", "S_newmains", "S_paper493")  # run_tables_v81.ORIGINS @ blob 88589b50
MEMBER_ROW = "{o} (all-origins draws) [pooled: pooled_all_origins]"  # run_tables_v81.CALIB["pooled"]["row"]
OUTCOME_LABEL = "own_all_rs_uncut_L48 C"  # run_tables_v81.OUTCOME_LABEL
FA_PROBES = ("sol_a_uncut C", "FIBS|uncut C", "harmfulness|uncut C", "refusal|uncut C", "high-stakes|uncut C", "own_all_rs_uncut_L48 C")
OUTCOMES = ("both caught", "real caught, sim missed", "real missed, sim caught", "both missed")  # run_examples_v81.OUTCOMES order
SIDES = (("real (run 1)", "B"), ("sim (sim 1)", "S"))  # §1: real side at tau_B, sim side at tau_S (run_tables_v81 l.781)
POOL_NM = {"pooled H_B": "B", "pooled H_sim": "S"}
N_FALSEALARMS = 6
LEVEL = "0.01"
SEED = 20261020


class Stop(SystemExit):
    pass


# ---------- rules ----------

def _is_int(v):
    return type(v) is int


def _is_float(v):
    return type(v) is float and math.isfinite(v)


def same_float(a, b):
    """Bitwise equality of two finite Python floats (repr round-trips float64 exactly; -0.0 and 0.0 differ)."""
    return _is_float(a) and _is_float(b) and repr(a) == repr(b)


def compare(printed, recomputed, rule):
    """count: both exact ints (bool is not int), equal. exact: both finite floats, bitwise equal. text: equal strings."""
    if recomputed is None:
        return False
    if rule == "count":
        return _is_int(printed) and _is_int(recomputed) and printed == recomputed
    if rule == "exact":
        return same_float(printed, recomputed)
    if rule == "text":
        return isinstance(printed, str) and printed == recomputed
    raise Stop(f"unknown rule {rule!r}")


# Slot-string rules (slot_table.py's formatters @ blob a8b4a156): scale, decimals. The tolerance is half a unit in the last printed place.
FMT = {"pct": (100.0, 1), "pp": (100.0, 1), "pp2": (100.0, 2), "share": (100.0, 0), "k2": (1.0, 2), "k2s": (1.0, 2), "ppv": (1.0, 1),
       "count": (1.0, 0)}
ON_POLICY = "own_all_rs_uncut_L48"


def slot_rule(slot):
    """-> (fmt, kind). kind 'summ' = one summary node (point [lo, hi]); 'ci' = its interval only; 'leaf' = one number at one key;
    'derived' = the consumer's formula over several keys (string equality only); 'text' = no number."""
    head = slot.split(".", 1)[0]
    pp_ = "pp2" if f".{ON_POLICY} " in slot else "pp"
    if slot == "mode" or head == "lbl" or re.match(r"^20\.readable\.", slot) or re.match(r"^23\.definition\.", slot) or head == "25":
        return "text", "text"
    if re.search(r"\.range$|halfwidth|^16\.|^17\.|^15b\.|distinct_runs_above|^09\.catch\.[^.]+ [CR]\.mean$|^09\.catch_avg\.[^.]+\.run[12]$", slot):
        return "derived", "derived"
    if head in ("01", "02", "03", "04", "05", "06", "07", "08", "11", "13") or re.match(r"^18\.Y\.", slot):
        return "pp", "summ"
    if re.match(r"^(10|18|20)\.delta_full\.", slot):
        return pp_, "summ"
    if head == "12":
        return "pp", "ci"
    if re.match(r"^(09|20)\.catch", slot):
        return "pct", "summ"
    if re.match(r"^15\.alarm_share\.", slot):
        return "share", "leaf"
    if re.match(r"^(14|19|23\.n_cells|24\.n_cells)\.", slot) or head in ("14", "19"):
        return "count", "leaf"
    if head in ("21", "22", "23", "24"):
        return ("k2s" if re.search(r"minus_run2|^23\.|delta_main", slot) else "k2"), "summ"
    return "derived", "derived"


def parse_numbers(s):
    """The numbers a slot string prints, in order, as floats (true minus and thousands separators handled). Text after
    '(interval withheld' or a ': ' reason is not parsed."""
    s = s.split(" (interval withheld", 1)[0]
    for k in ("unavailable: ", "withheld: ", "excluded: "):
        if s.startswith(k):
            return []
    if s in ("not readable",):
        return []
    return [float(t.replace(MINUS, "-").replace(",", "")) for t in NUM_RX.findall(s)]


def raw_numbers(node, kind):
    """The raw values a slot prints, read by key from the record: a summary node's point and interval, or one leaf."""
    if kind == "leaf":
        return [node] if type(node) in (int, float) else None
    if not isinstance(node, dict) or "point" not in node:
        return None
    ci = node.get("ci95")
    if kind == "ci":
        return list(ci) if isinstance(ci, list) and len(ci) == 2 else []
    return [node["point"]] + (list(ci) if isinstance(ci, list) and len(ci) == 2 else [])


def numbers_agree(printed_s, raw, fmt):
    """Each printed number within half a unit in the last place of scale * raw, and formatted alike; counts exact."""
    scale, nd = FMT[fmt]
    nums = parse_numbers(printed_s)
    if "interval withheld" in printed_s and raw:
        raw = raw[:1]
    if len(nums) != len(raw) or any(type(r) not in (int, float) or not math.isfinite(r) for r in raw):
        return False
    if fmt == "count":
        return all(type(r) is int and n == r for n, r in zip(nums, raw, strict=True))
    tol = 0.5 * 10 ** -nd + 1e-9
    return all(abs(n - scale * r) <= tol and abs(n - float(f"{scale * r:.{nd}f}")) <= 1e-12 for n, r in zip(nums, raw, strict=True))


# ---------- rows and the verdict ----------

def row(artifact, slot, record, key_path, printed, recomputed, rule, equal, sources, signed, note="", unit="number", jpath=None, **extra):
    """unit 'number' = one printed number (the PASS counts these); 'slot' = a whole slot string (a summary row: it must pass too, but is not
    counted as a number). jpath: the number's path in the artifact's own JSON, when it has one (Tables 3/4)."""
    srcs = sorted(set(sources))
    return {"artifact": artifact, "slot": slot, "record": record, "key_path": key_path, "printed": _enc(printed),
            "recomputed": _enc(recomputed), "rule": rule, "equal": bool(equal), "sources": srcs,
            "orphan": (not srcs) or any(s not in signed for s in srcs), "note": note, "unit": unit} | ({"jpath": jpath} if jpath else {}) | extra


def _enc(v):
    return repr(v) if isinstance(v, float) else v


def guarded(fn, where, *a):
    """R5: run fn(*a); any exception (SystemExit included; some consumer STOP messages carry record-derived counts) becomes
    (None, {"class", "where"}) with NO message, and the traceback text is returned separately for the Volume only."""
    import traceback

    try:
        return fn(*a), None, ""
    except BaseException as e:  # noqa: BLE001  (the message never leaves the container; the class and phase do)
        return None, {"class": type(e).__name__, "where": where}, traceback.format_exc()


def nonresult_list(rows):
    """C1: the generated non-result list (one "<class>\\t<matched string>\\t<position>" per non-result row, sorted; Q2: the position
    makes each entry one occurrence, so 09's {sha, count} approves specific occurrences) and its sha256 / count."""
    gen = sorted(nonresult_line(r) for r in rows if r.get("unit") == "nonresult")
    return gen, sha("\n".join(gen).encode()), len(gen)


def nonresult_line(r):
    """One entry of the non-result list: "<class>\\t<matched post text>\\t<position>"."""
    return f"{r.get('nr_class')}\t{r.get('matched')}\t{r.get('pos')}"


def nonresult_diff(old_lines, new_lines):
    """09 (10-20): the by-key diff of two non-result lists ("class\\tmatched\\tposition" lines): the entries added and removed, with counts
    by class. -> {"added", "removed", "added_by_class", "removed_by_class", "unchanged"}."""
    o, n = {x for x in old_lines if x}, {x for x in new_lines if x}
    add, rem = sorted(n - o), sorted(o - n)

    def by(xs):
        d = {}
        for x in xs:
            d[x.split("\t", 1)[0]] = d.get(x.split("\t", 1)[0], 0) + 1
        return d

    return {"added": add, "removed": rem, "added_by_class": by(add), "removed_by_class": by(rem), "unchanged": len(o & n)}


def emit_nonresult_list(entries, want_sha, want_count):
    """09 (10-20): the non-result list as a file whose sha256 is the list sha the run reports, by construction (the same lines, sorted,
    newline-joined, no trailing newline); entries: the run's post diagnostics "nonresult_entries" (each carries its "line"). A sha or
    count that differs from the run's STOPs. -> (text bytes, entries sorted by line)."""
    if any(e.get("line") != f"{e.get('class')}\t{e.get('matched')}\t{e.get('pos')}" for e in entries):
        raise Stop("non-result list: an entry's line is not its class, matched text and position (STOP)")
    srt = sorted(entries, key=lambda e: e["line"])
    text = "\n".join(e["line"] for e in srt).encode()
    if sha(text) != want_sha or len(srt) != want_count:
        raise Stop("non-result list: the emitted entries do not hash to the list sha the run reports, or differ in count (STOP)")
    return text, srt


def summarize(rows, plan_counts, extra_stops=(), pins=None):
    """Counts and the verdict (keys only). plan_counts: artifact -> the consumer's count of printed items (coverage: every printed item
    has exactly one row). Verdict (RN1): STOP on any stop; PARTIAL when the only outstanding rows are deferred ones; PASS only with no
    deferred row at all. Deferred rows never count as verified.
    C1 (10-20): acceptance is recomputed HERE from 09's pins, never read from a row's own flags (no row can approve itself):
      deferred   counts (NOT VERIFIED, apart) only when its reason is one of pins["accepted_reasons"] (09's file, exact strings) AND its
                 record is non-null with every cited record read at its commit and hashed to its sha and its code-map region resolved
                 by key (record_ok, set by the adapter from those reads); otherwise it is an orphan, whatever its reason (09);
      nonresult  counts (apart) only when the list generated from all non-result rows equals 09's pins["nonresult_approved"]
                 {sha256, count} and the row's own checks hold (checks_ok: the hypothetical class's three); otherwise an orphan;
      not_route_C a value checked against a cited record that is unsigned (not route C): counted apart, labelled, never verified; an
                 unequal one STOPs like any number.
    Totals count number rows; slot summary rows and check rows STOP like any other."""
    pins = pins or {}
    acc = set(pins.get("accepted_reasons") or ())
    _, g_sha, g_n = nonresult_list(rows)
    appr = pins.get("nonresult_approved") or {}
    nr_ok = (appr.get("sha256"), appr.get("count")) == (g_sha, g_n)
    per, stop = {}, {"unmatched": [], "orphan": [], "count_mismatch": [], "duplicate_slots": [], "other": sorted(set(extra_stops))}
    seen, deferred_by_reason, n_nrc, n_pre = set(), {}, 0, 0
    def part(a_, *ks):  # 09 (10-20): every row in exactly ONE class; "failed_unmatched" / "failed_orphan" are sub-counts of "failed"
        p_ = a_.setdefault("partition", {})
        for k__ in ks:
            p_[k__] = p_.get(k__, 0) + 1

    for r in rows:
        a = per.setdefault(r["artifact"], {"n_printed": 0, "n_equal": 0, "n_unmatched": 0, "n_orphan": 0})
        a["n_printed"] += 1
        k = (r["artifact"], r["slot"])
        if k in seen:
            stop["duplicate_slots"].append(f"{k[0]}::{k[1]}")
        seen.add(k)
        u = r.get("unit")
        if u == "deferred" and r.get("reason") in acc and r.get("record_ok") is True:  # NOT VERIFIED (<reason>): apart, never PASS
            a["n_deferred"] = a.get("n_deferred", 0) + 1
            part(a, "deferred")
            lab = "NOT VERIFIED (" + str(r.get("reason", "?")) + ")"
            deferred_by_reason[lab] = deferred_by_reason.get(lab, 0) + 1
            continue
        if u == "nonresult" and nr_ok and r.get("checks_ok") is True:  # a 09-approved non-result class: counted apart
            a["n_nonresult"] = a.get("n_nonresult", 0) + 1
            part(a, "nonresult")
            continue
        if u in ("deferred", "nonresult"):  # an unaccepted reason, a null / unread record, an unapproved list or a failed check: orphan
            a["n_orphan"] += 1
            part(a, "failed", "failed_orphan")
            stop["orphan"].append(f"{r['artifact']}::{r['slot']}")
            continue
        if u in ("not_route_C", "preliminary") and r["equal"] and not r["orphan"]:  # labelled unsigned buckets, counted apart
            k_ = f"n_{u}_checks" if r.get("check_row") else f"n_{u}"  # printed numbers apart from per-row check rows (Q3)
            a[k_] = a.get(k_, 0) + 1
            part(a, k_[2:])
            if not r.get("check_row"):
                n_nrc += u == "not_route_C"
                n_pre += u == "preliminary"
            continue
        a["n_equal"] += r["equal"]
        a["n_unmatched"] += not r["equal"]
        a["n_orphan"] += r["orphan"]
        if r["equal"] and not r["orphan"]:
            part(a, "equal")
        else:  # a row both unmatched and orphan is ONE failed row (the old n_unmatched and n_orphan both count it)
            part(a, "failed", *(["failed_unmatched"] if not r["equal"] else []), *(["failed_orphan"] if r["orphan"] else []))
        if not r["equal"]:
            stop["unmatched"].append(f"{r['artifact']}::{r['slot']}")
        if r["orphan"]:
            stop["orphan"].append(f"{r['artifact']}::{r['slot']}")
    for a in sorted(set(per) | set(plan_counts)):
        d = per.setdefault(a, {"n_printed": 0, "n_equal": 0, "n_unmatched": 0, "n_orphan": 0})
        d["plan"] = plan_counts.get(a)
        if d["plan"] is None or d["plan"] != d["n_printed"]:
            stop["count_mismatch"].append(a)
        if sum(v for k__, v in (d.get("partition") or {}).items() if not k__.startswith("failed_")) != d["n_printed"]:
            stop["other"].append(f"{a}: the partition of its rows does not sum to n_printed (STOP)")
    nr = [r for r in rows if r.get("unit", "number") == "number"]
    n_def = sum(deferred_by_reason.values())
    tot = {"n_printed": len(nr), "n_equal": sum(r["equal"] for r in nr), "n_unmatched": sum(not r["equal"] for r in nr),
           "n_orphan": len(stop["orphan"]), "n_slot_rows": sum(r.get("unit") == "slot" for r in rows),
           "n_not_verified_deferred": n_def, "deferred_by_reason": deferred_by_reason,
           "n_nonresult": sum(a.get("n_nonresult", 0) for a in per.values()), "nonresult_list": {"sha256": g_sha, "count": g_n, "approved": nr_ok},
           "n_not_route_C_checked": n_nrc, "n_preliminary_checked": n_pre}
    base = "STOP" if any(stop.values()) else ("PARTIAL" if n_def else "PASS")
    return tot | {"per_artifact": per, "verdict": verdict_line(base, n_nrc, n_pre=n_pre), "verdict_base": base, "stop": stop}


def verdict_line(base, n_nrc=0, words=None, n_pre=0, pre_labels=()):
    """The verdict line (09, 10-20): the base verdict, then the not-route-C bucket ("N not route C, checked against unsigned records by
    key"), the PRELIMINARY bucket in neutral wording (R7-1: its records may be signed; with per-record labels it says "labels in scope":
    2e / 09 (10-20), the full labels go to the result's scope, never into the reply's verdict string, which the guard keeps at 300) and, for
    a post pass, the number words it does not gate ("PASS (digits); W number words not gated: n covered by a row, m uncovered"), so a
    PASS is never read as covering them."""
    s = "PASS (digits)" if words and base == "PASS" else base
    if n_nrc:
        s += f"; {n_nrc} not route C, checked against unsigned records by key"
    if n_pre:  # 09 (10-20): route C numbers out of v1 scope, checked against their record by key (signed or not: see the labels)
        s += f"; {n_pre} PRELIMINARY (route C, out of v1 scope), checked against records by key"
        if pre_labels:
            s += "; labels in scope"
    if words:
        s += f"; {words['total']} number words not gated: {words['covered']} covered by a row, {words['uncovered']} uncovered"
    return s


def select_recheck(rows, seed=SEED, k=20):
    """k number rows drawn without replacement by numpy default_rng(seed) over the number rows sorted by (artifact, slot)."""
    order = sorted((r["artifact"], r["slot"]) for r in rows if r.get("unit", "number") == "number")
    if not order:
        return []
    pick = np.random.default_rng(seed).choice(len(order), size=min(k, len(order)), replace=False)
    return [list(order[int(i)]) for i in pick]


def jsonl_bytes(rows):
    return b"".join(json.dumps(r, sort_keys=True, ensure_ascii=False).encode() + b"\n" for r in rows)


def sha(b):
    return hashlib.sha256(b).hexdigest()


# ---------- consumer code by blob ----------

def extract_function(src, name, glb=None):
    """One top-level function from a consumer's source (bytes), compiled alone (A's quantile rule: run_peraction311.wthr @ 71633875)."""
    tree = ast.parse(src)
    fs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(fs) != 1:
        raise Stop(f"{name}: {len(fs)} definitions in the pinned source (STOP)")
    ns = dict(glb or {})
    exec(compile(ast.Module(body=fs, type_ignores=[]), f"<pinned {name}>", "exec"), ns)
    return ns[name]


PUT_ANCHOR = "            out[key], full[key] = fn()\n"
GET_ANCHOR = "\ndef get(d, *path):\n"


def _nv_call(cur, key, fn):
    """put()'s fn() with the current slot key set only while it runs (a get() outside any put() logs under None)."""
    cur[0] = key
    try:
        return fn()
    finally:
        cur[0] = None


def instrumented_slot_table(src: bytes):
    """slot_table.py (its pinned bytes) as a module whose get() logs, per slot, the absolute key path (role, *keys) of every node it
    reads. Two output-only changes, each at an anchor that must occur once: put()'s fn() call runs inside _nv_call (the current slot key
    set only while fn runs); get() is wrapped after the module runs. Nothing else in the module changes."""
    import types

    s = src.decode()
    for a in (PUT_ANCHOR, GET_ANCHOR):
        if s.count(a) != 1:
            raise Stop(f"slot_table anchor {a.strip()!r} occurs {s.count(a)} times (STOP)")
    s = s.replace(PUT_ANCHOR, "            out[key], full[key] = _NV_CALL(_NV_CUR, key, fn)\n")
    m = types.ModuleType("slot_table_pinned")
    m.__dict__["_NV_CUR"] = [None]
    m.__dict__["_NV_CALL"] = _nv_call
    exec(compile(s, "<slot_table pinned>", "exec"), m.__dict__)
    get0, log, absmap = m.get, {}, {}

    def get(d, *path):
        v = get0(d, *path)
        base = absmap.get(id(d))
        ab = (base + tuple(path)) if base is not None else ("?",) + tuple(path)
        log.setdefault(m._NV_CUR[0], []).append(ab)
        if isinstance(v, dict):
            absmap[id(v)] = ab
            for k_, c_ in v.items():
                if isinstance(c_, dict):
                    absmap.setdefault(id(c_), ab + (k_,))
        return v

    m.get = get
    m._NV_LOG, m._NV_ABS = log, absmap
    return m


def walk(rec, keys):
    """My own key walk (not slot_table's get): a float key matches one numeric JSON key by value, as the writer's level keys."""
    d = rec
    for p in keys:
        if not isinstance(d, dict):
            return None
        if isinstance(p, float):
            ks = [k for k in d if _num(k) and math.isclose(float(k), p, rel_tol=1e-9, abs_tol=0.0)]
            if len(ks) != 1:
                return None
            p = ks[0]
        if p not in d:
            return None
        d = d[p]
    return d


def _num(k):
    try:
        float(k)
        return True
    except (TypeError, ValueError):
        return False


# ---------- slot readouts (rc2, rc1a, ...) ----------

def slots_rows(artifact, printed_slots, rebuilt, log, recs, role_ref, role_sha, signed, const_ok=("mode",)):
    """printed_slots: the released readout's {slot: str}. rebuilt: slot_table.build(...) re-run here on the committed records ({slots,
    full, missing, not_carried}). log: slot -> the absolute key paths its get() calls read. -> (rows, plan count, stops)."""
    rows, stops = [], []
    n_num = sum(len(NUM_RX.findall(v)) for v in rebuilt["slots"].values() if isinstance(v, str))  # the consumer's count: every number it prints
    sig = signed | {"readout-constant"}
    if rebuilt["missing"]:
        stops.append(f"{artifact}: rebuilt readout has {len(rebuilt['missing'])} missing slots")
    for slot in sorted(printed_slots):
        p_ = printed_slots[slot]
        r_ = rebuilt["slots"].get(slot)
        paths = [p for p in log.get(slot, []) if p and p[0] in recs]
        roles = sorted({p[0] for p in paths})
        fmt, kind = slot_rule(slot)
        num_ok, note, raw = True, "", None
        gated = p_ in ("not readable",) or p_.startswith(("unavailable", "withheld", "excluded", "interval withheld", "every interval withheld"))
        if kind in ("summ", "ci", "leaf"):
            node = walk(recs[paths[-1][0]], paths[-1][1:]) if paths else None
            raw = raw_numbers(node, kind)
            if gated:
                note, raw = "gated string", None
            elif raw is None:
                num_ok, note = False, "no summary node or leaf at the slot's last key path"
            else:
                num_ok = numbers_agree(p_, raw, fmt)
                raw = raw[:1] if "interval withheld" in p_ else raw
        srcs = [role_sha[r] for r in roles]
        if slot in const_ok:
            srcs = ["readout-constant"]
        elif slot.startswith("lbl."):
            lr = slot[4:]
            srcs = [role_sha[x] for x in (f"{lr}_label", "LABELS_171") if x in role_sha] or srcs
        kp = ";".join("/".join(map(str, p)) for p in paths[-4:]) if paths else ""
        ref = ",".join(f"{r}={role_ref[r]}" for r in roles) or "-"
        rows.append(row(f"{artifact}/slots", slot, ref, kp, p_, r_, f"{fmt}/{kind}", r_ is not None and p_ == r_ and num_ok, srcs, sig, note,
                        unit="slot"))
        # R4: one row per printed number, each with its own key path, rounding rule and equality
        tp, tr = NUM_RX.findall(p_), NUM_RX.findall(r_ or "")
        comp = {"summ": ("point", "ci95[0]", "ci95[1]"), "ci": ("ci95[0]", "ci95[1]"), "leaf": ("value",)}.get(kind, ())
        for i, t in enumerate(tp):
            rt = tr[i] if i < len(tr) else None
            if raw is not None and i < len(raw):
                scale, nd = FMT[fmt]
                v = float(t.replace(MINUS, "-").replace(",", ""))
                rv = raw[i]
                ok_raw = type(rv) in (int, float) and math.isfinite(rv) and (
                    (type(rv) is int and v == rv) if fmt == "count" else
                    (abs(v - scale * rv) <= 0.5 * 10 ** -nd + 1e-9 and abs(v - float(f"{scale * rv:.{nd}f}")) <= 1e-12))
                rule_ = f"{fmt}: x{scale:g}, {nd} dp, f-string rounding of the binary float; tol half a unit in the last place"
                kpn = (("/".join(map(str, paths[-1])) + "/" + comp[i]) if paths else "") if i < len(comp) else kp
                rec_ = _enc(rv)
            else:
                ok_raw, rule_, kpn, rec_ = True, "string equal to the consumer's re-run (derived, text or gated part)", kp, rt
            rows.append(row(artifact, f"{slot}#{i}", ref, kpn, t, rec_, rule_, rt == t and ok_raw, srcs, sig, note))
    return rows, {artifact: n_num, f"{artifact}/slots": len(rebuilt["slots"])}, stops


# ---------- figures (re-rendered by the consumer's own scripts) ----------

FIG_CONFIG_KEYS = ("values", "build_profile", "inputs", "record_meta", "label_sidecar", "git", "fig_number", "text_file", "label", "title_banner")


def numeric_leaves(x, path=()):
    """(path, value) for every int / float leaf (bool excluded) under x."""
    if isinstance(x, dict):
        for k, v in x.items():
            yield from numeric_leaves(v, path + (str(k),))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from numeric_leaves(v, path + (i,))
    elif type(x) in (int, float):
        yield path, x


def png_chunks(b):
    """[(chunk type, length)] of a PNG, no data (09: if byte identity fails, report which chunks differ by name and count only)."""
    out, i = [], 8
    if not b.startswith(b"\x89PNG\r\n\x1a\n"):
        return [("not-a-png", len(b))]
    while i + 8 <= len(b):
        ln = int.from_bytes(b[i:i + 4], "big")
        out.append((b[i + 4:i + 8].decode("latin-1"), ln))
        i += 12 + ln
    return out


def png_gate(fig, committed, rebuilt):
    """RQ1: the committed PNG must be byte-identical to the container's re-render, else STOP; on a difference, the chunk names whose
    bytes differ and the chunk counts (never data). -> (stops, diagnostic)."""
    if sha(committed) == sha(rebuilt):
        return [], {"png_bytes_identical": True}
    ca, cb = png_chunks(committed), png_chunks(rebuilt)

    def body(b, n):
        out, i = [], 8
        for _ in range(n):
            ln = int.from_bytes(b[i:i + 4], "big")
            out.append(b[i + 8:i + 8 + ln])
            i += 12 + ln
        return out
    da = body(committed, len(ca)) if ca and ca[0][0] != "not-a-png" else []
    db = body(rebuilt, len(cb)) if cb and cb[0][0] != "not-a-png" else []
    diff = sorted({ca[j][0] for j in range(min(len(da), len(db))) if ca[j][0] != cb[j][0] or da[j] != db[j]})
    return [f"{fig}: the committed PNG is not byte-identical to the container re-render"], {
        "png_bytes_identical": False, "chunk_types_committed": [t for t, _ in ca], "chunk_types_rebuilt": [t for t, _ in cb],
        "chunk_names_differing": diff, "chunk_count_committed": len(ca), "chunk_count_rebuilt": len(cb)}


def _same(a, b):
    return (type(a) is int and type(b) is int and a == b) or same_float(a, b)


def fig_rows(fig, printed_src, rebuilt_src, printed_txt, rebuilt_txt, records, rec_sha, signed):
    """One figure's rows. printed_src / rebuilt_src: the committed and the re-rendered _source.json; *_txt: the committed and re-rendered
    _text.md; records: input basename -> parsed record; rec_sha: input basename -> sha256. Every plotted value (values[]) is compared with
    the record read here by its logged key AND with the re-render; every other numeric leaf (outside FIG_CONFIG_KEYS) and every number token
    of the text is compared with the re-render. -> (rows, plan count, stops)."""
    rows, stops = [], []
    ins = {k: v["file"] for k, v in printed_src.get("inputs", {}).items()}
    if any(f not in records for f in ins.values()):
        stops.append(f"{fig}: an input of the source JSON is not a configured record")
    all_srcs = [rec_sha[f] for f in ins.values() if f in rec_sha]
    pv, rv = printed_src.get("values", []), rebuilt_src.get("values", [])
    if len(pv) != len(rv) or any(a.get("key") != b.get("key") or a.get("input") != b.get("input") for a, b in zip(pv, rv, strict=False)):
        stops.append(f"{fig}: the re-render's logged value keys differ from the committed ones")
    plan = 0
    for i, v in enumerate(pv):
        f = ins.get(v.get("input"))
        node = walk(records[f], v["key"]) if f in records else None
        rb = rv[i] if i < len(rv) else {}
        for fld in ("point", "ci95", "value"):
            if fld not in v:
                continue
            got = v[fld] if isinstance(v[fld], list) else [v[fld]]
            want = (node.get(fld) if isinstance(node, dict) and fld != "value" else node) if node is not None else None
            want = want if isinstance(want, list) else [want]
            reb = rb.get(fld) if isinstance(rb.get(fld), list) else [rb.get(fld)]
            for j, p_ in enumerate(got):
                if type(p_) not in (int, float):
                    continue
                w_ = want[j] if j < len(want) else None
                r_ = reb[j] if j < len(reb) else None
                rows.append(row(fig, f"values[{i}]|{fld}" + (f"[{j}]" if isinstance(v[fld], list) else ""), f"{f}@{rec_sha.get(f, '?')}",
                                "/".join(map(str, v["key"])), p_, w_, "exact" if type(p_) is float else "count",
                                w_ is not None and _same(p_, w_) and r_ is not None and _same(p_, r_), [rec_sha[f]] if f in rec_sha else [], signed,
                                "record by key and re-render"))
    rl = dict(numeric_leaves({k: x for k, x in rebuilt_src.items() if k not in FIG_CONFIG_KEYS}))
    plan += len(rl)
    for p, x in numeric_leaves({k: x for k, x in printed_src.items() if k not in FIG_CONFIG_KEYS}):
        r_ = rl.get(p)
        rows.append(row(fig, "src|" + "/".join(map(str, p)), "re-render", "/".join(map(str, p)), x, r_, "exact" if type(x) is float else "count",
                        r_ is not None and _same(x, r_), all_srcs, signed, "the consumer's own computation, re-run"))
    pt, rt = NUM_RX.findall(printed_txt), NUM_RX.findall(rebuilt_txt)
    for i, t in enumerate(pt):
        r_ = rt[i] if i < len(rt) else None
        rows.append(row(fig, f"text#{i}", "re-render", "_text.md token", t, r_, "text", r_ is not None and t == r_, all_srcs, signed))
    plan += sum(1 for v in rv for fld in ("point", "ci95", "value") if fld in v
                for p_ in (v[fld] if isinstance(v[fld], list) else [v[fld]]) if type(p_) in (int, float)) + len(rt)
    return rows, plan, stops


# ---------- Tables 3/4 (the v81 tables record) ----------

def _member_rows(A, label):
    rows_ = A["rows"][label]
    return [rows_[MEMBER_ROW.format(o=o)] for o in ORIGINS]


def a_pool(A, label, nm, key):
    """A rows[label][<member> [pooled: pooled_all_origins]].pools_at_draw0["0.01"][nm][key]: one value shared by the four members, else STOP."""
    vs = [r["pools_at_draw0"][LEVEL][nm][key] for r in _member_rows(A, label)]
    if any(repr(v) != repr(vs[0]) for v in vs[1:]):
        raise Stop(f"A {label} {nm} {key}: the member rows disagree (STOP)")
    return vs[0]


def a_outcome_counts(A):
    """§1's n per outcome from A by key: each member row's cohort cells (cohort_cell_keys, sorted) and its draw-0 cell_hits strings at
    1e-2 ('1' caught, '0' missed); a '-' (no hit value) is not decidable here -> None for every n."""
    n = {o: 0 for o in OUTCOMES}
    for o in ORIGINS:
        r = A["rows"][OUTCOME_LABEL][MEMBER_ROW.format(o=o)]
        cs, lv = r["cohort_cell_keys"], r["cell_hits_draw0"]["levels"][LEVEL]
        h1, hs = lv["run1"], lv["sim"]
        if not (len(cs) == len(h1) == len(hs)):
            raise Stop(f"A {o}: cohort and cell_hits lengths differ (STOP)")
        if "-" in h1 or "-" in hs:
            return None
        for a, b in zip(h1, hs, strict=True):
            n[OUTCOMES[(0 if a == "1" else 2) + (0 if b == "1" else 1)]] += 1
    return n


def d1_conditions(A, d1, wthr):
    """09's conditions (1)-(2) per label and pool -> {f"{label}|{nm}": [failed condition names]} (empty list = covered).
    d1[label] = {"sha256_read": sha of the bytes opened, "z": {"B": array, "S": array}}."""
    out = {}
    pins = A["meta"]["pooled_world"]["calibrations"]["by_label_combo"]
    for label in FA_PROBES:
        pin = pins.get(f"{label} | {LEAD}") or {}
        for nm in ("B", "S"):
            bad = []
            x = d1.get(label)
            if x is None or not pin.get("sha256") or x["sha256_read"] != pin["sha256"]:
                bad.append("1:npz-sha-not-A's-pin")
            else:
                z = np.asarray(x["z"][nm], dtype=np.float64)
                T_A, n_A, up_A, k_A = (a_pool(A, label, nm, k) for k in ("T", "n_units", "units_above", "k"))
                zs = np.sort(z)[::-1][None, :]
                T_re = float(wthr(zs, np.ones_like(zs), float(LEVEL))[0][0])
                if not same_float(T_re, float(T_A)):
                    bad.append("2:T")
                if int((z > T_A).sum()) != up_A:
                    bad.append("2:units_above")
                if len(z) != n_A or int(np.floor(float(LEVEL) * len(z) + 1e-9)) != k_A:
                    bad.append("2:n_units_or_k")
            out[f"{label}|{nm}"] = bad
    return out


def score_pins(frozen, overrides):
    """FROZEN INPUTS_FULL's C overrides -> {score file path: {sha256, ...}} over every .npz {path, sha256} pin in each override file
    (09's ruling (b): the score file's pin is looked up in FROZEN's chain, never typed). overrides: key -> {commit, path, sha256, bytes};
    they must be exactly FROZEN's "|C" overrides, each equal to FROZEN's pin and its bytes hashing to it, else STOP."""
    want = {k for k in frozen.get("overrides", {}) if k.endswith("|C")}
    if set(overrides) != want:
        raise Stop("the shipped overrides are not exactly FROZEN's C overrides (STOP)")
    pins = {}
    for key, x in overrides.items():
        fz = frozen["overrides"][key]
        if (fz["commit"], fz["path"], fz["sha256"]) != (x["commit"], x["path"], x["sha256"]) or sha(x["bytes"]) != x["sha256"]:
            raise Stop("an override is not FROZEN's pin, or its bytes do not hash to it (STOP)")

        def walk_(n):
            if isinstance(n, dict):
                if isinstance(n.get("path"), str) and isinstance(n.get("sha256"), str) and n["path"].endswith(".npz"):
                    pins.setdefault(n["path"].lstrip("/"), set()).add(n["sha256"])
                for v in n.values():
                    walk_(v)
            elif isinstance(n, list):
                for v in n:
                    walk_(v)
        walk_(json.loads(x["bytes"]))
    return pins


def d1_moments(npz, nm):
    """The D1 file's draw-0 joint moments of pool nm as run_tables_v81._v81_pack.m0 shapes them (row 0 of joint__<nm>__mu / sd, per fold
    in the file's fold order; mu0 / sd0 when present), plus fold_model and folds from the file's meta_json (rp 71633875 L2572-2580)."""
    meta = json.loads(str(npz["meta_json"]))
    folds, fm = [int(x) for x in meta["folds"]], bool(meta["fold_model"])
    if meta.get("mom_none"):
        return None, fm, folds
    mu, sd = np.asarray(npz[f"joint__{nm}__mu"]), np.asarray(npz[f"joint__{nm}__sd"])
    has0 = f"joint__{nm}__mu0" in npz.files
    return ({"mu": {k: float(mu[0, i]) for i, k in enumerate(folds)}, "sd": {k: float(sd[0, i]) for i, k in enumerate(folds)},
             "mu0": float(np.asarray(npz[f"joint__{nm}__mu0"])[0]) if has0 else None,
             "sd0": float(np.asarray(npz[f"joint__{nm}__sd0"])[0]) if has0 else None}, fm, folds)


def attack_units(npz, run_key, n_folds):
    """One attack run's units from its score file, read as run_tables_v81.FileIndex reads it (L521-526 @ 88589b50): the rows whose
    run_key is the run's, in file (token) order; every file key holding that run must carry the same tokens and scores, tokens strictly
    increasing. -> {"s", "f", "sbm", "token"} or raises Stop."""
    k = npz["key"].astype(str)
    n = len(k)
    f = npz["fold"].astype(np.int64) if (n_folds == 5 and "fold" in npz.files) else np.zeros(n, np.int64)
    sbm = npz["score_by_model"].astype(np.float64) if "score_by_model" in npz.files else np.full((n, 5), np.nan)
    s_ = npz["score"].astype(np.float64) if "score" in npz.files else np.where(f >= 0, sbm[np.arange(n), np.clip(f, 0, 4)], np.nan)
    rk, tok = npz["run_key"].astype(str), npz["token"].astype(np.int64)
    groups = []
    for key in sorted(set(k[rk == run_key].tolist())):
        ix = np.flatnonzero((k == key) & (rk == run_key))
        groups.append({"s": s_[ix], "f": f[ix], "sbm": sbm[ix], "token": tok[ix]})
    if not groups:
        raise Stop("attack run: no row of its score file carries its run_key (STOP)")
    g0 = groups[0]
    if any(not all(np.array_equal(g[x], g0[x], equal_nan=True) for x in g0) for g in groups[1:]) or not (np.diff(g0["token"]) > 0).all():
        raise Stop("attack run: its file keys disagree, or its tokens do not increase strictly (STOP)")
    return g0 | {"file_keys": sorted(set(k[rk == run_key].tolist()))}


def attack_argmax(units, mom, fold_model, folds, unit_z, argmax_command):
    """unit_z (run_tables_v81 L260-281 @ 88589b50, the per-unit form of the build's zatt, rp 71633875 L2322-2335) on the run's units with
    the D1 draw-0 moments, then argmax_command (run_examples_v81 @ f191a247: max z, ties to the lowest unit)."""
    uz = unit_z({"s": units["s"], "f": units["f"], "sbm": units["sbm"]}, mom, fold_model, folds)
    if not np.isfinite(uz).all():
        raise Stop("attack run: a non-finite unit z (STOP)")
    u = argmax_command([{"unit": i, "z": float(v)} for i, v in enumerate(uz)])
    tok = int(units["token"][u["unit"]])
    return {"z": u["z"], "unit": u["unit"], "n_units": len(uz), "token": tok}  # unit/token is not derived (09 option (i): on the not-printed list)


def _jp(*p):
    return "/".join(map(str, p))


def not_printed_list(T, mapped):
    """R1 (09's conditions 1-2): every numeric leaf under the record's outcome_table and false_alarms, minus the paths the verifier maps to
    printed numbers, generated here (never by hand). -> (sorted paths, sha256 of their newline-joined text, count, all leaf paths)."""
    leaves = [_jp(*p) for p, _ in numeric_leaves({"outcome_table": T.get("outcome_table", {}), "false_alarms": T.get("false_alarms", {})})]
    np_ = sorted(set(leaves) - set(mapped))
    return np_, sha("\n".join(np_).encode()), len(np_), leaves


def tables34_rows(T, A, d1, pools, wthr, refs, signed, attack=None, fa_units=None):
    """T: the tables record; A: the lead record; d1[label]: {"sha256_read", "path", "z": {nm: array}}; pools[label]: {nm: run array}
    (the pooled pools file, A-pinned, same unit order as the npz); refs: {"A", "A_sha", "d1", "d1_sha", "frozen_sha", "not_printed_approved"};
    attack: f"{outcome}|{side}" -> {"result": attack_argmax(...) or None, "frozen_ok", "score_ref", "why", "file_cells"}.
    Every row carries the jpath of the number it checks; every other numeric leaf goes on the generated not-printed list, which must equal
    09's approved {sha256, count}, else STOP. -> (rows, plan count, stops, diagnostics)."""
    art, rows, stops = "tables34", [], []
    cond = d1_conditions(A, d1, wthr)
    a_sha, d1_ok = refs["A_sha"], {k: not v for k, v in cond.items()}
    if T.get("calibration") != "pooled" or T["lead"]["record"][3] != a_sha:
        raise Stop("tables record: not the pooled tables on this A (STOP)")
    # §1 = Table 4 (the outcome table)
    nA = a_outcome_counts(A)
    oc_ = T["outcome_table"]["outcomes"]
    if list(oc_) != list(OUTCOMES):
        stops.append("tables34: outcome cells differ from the consumer's four")
    cohort = {o: A["rows"][OUTCOME_LABEL][MEMBER_ROW.format(o=o)] for o in ORIGINS}
    uc = sum(len(r["cohort_cell_keys"]) for r in cohort.values())
    rows.append(row(art, "T4|universe_cells", f"A={refs['A']}", f"rows/{OUTCOME_LABEL}/<4 members>/cohort_cell_keys (count)",
                    T["outcome_table"].get("universe_cells"), uc, "count", compare(T["outcome_table"].get("universe_cells"), uc, "count"), [a_sha],
                    signed, jpath=_jp("outcome_table", "universe_cells")))
    for oc in OUTCOMES:
        d = oc_.get(oc, {})
        rows.append(row(art, f"T4|{oc}|n", f"A={refs['A']}", f"rows/{OUTCOME_LABEL}/<4 members>/cohort_cell_keys+cell_hits_draw0/levels/0.01/run1,sim",
                        d.get("n"), None if nA is None else nA[oc], "count", compare(d.get("n"), None if nA is None else nA[oc], "count"),
                        [a_sha], signed, "X.table_outcomes count rule (run_examples_v81 @ f191a247)", jpath=_jp("outcome_table", "outcomes", oc, "n")))
        if nA is not None and (nA[oc] > 0) != (d.get("drawn") not in (None, "none")):
            stops.append(f"tables34: outcome '{oc}' drawn or none disagrees with A's count")
        if d.get("drawn") in (None, "none"):
            continue
        o, c = d["drawn"]["cell"].split("|", 1)
        bits = None  # R3: the drawn cell must be in A's cohort, and A's own hit bits must give this outcome
        if o in cohort and c in cohort[o]["cohort_cell_keys"]:
            ix_ = cohort[o]["cohort_cell_keys"].index(c)
            lv_ = cohort[o]["cell_hits_draw0"]["levels"][LEVEL]
            bits = {"B": lv_["run1"][ix_], "S": lv_["sim"][ix_]}
            if "-" in bits.values() or OUTCOMES[(0 if bits["B"] == "1" else 2) + (0 if bits["S"] == "1" else 1)] != oc:
                stops.append(f"tables34: T4|{oc}: A's hit bits for the drawn cell do not give this outcome")
        else:
            stops.append(f"tables34: T4|{oc}: the drawn cell is not in A's cohort for its origin")
        for side, nm in SIDES:
            v = d["drawn"][side]
            if v == "no command":
                continue
            jb = ("outcome_table", "outcomes", oc, "drawn", side)
            T_A = a_pool(A, OUTCOME_LABEL, nm, "T")
            kp = f"rows/{OUTCOME_LABEL}/{MEMBER_ROW.format(o=o)}/pools_at_draw0/0.01/{nm}/T"
            rows.append(row(art, f"T4|{oc}|{side}|T", f"A={refs['A']}", kp, v["T"], float(T_A), "exact", compare(v["T"], float(T_A), "exact"),
                            [a_sha], signed, jpath=_jp(*jb, "T")))
            at = (attack or {}).get(f"{oc}|{side}")  # 09's ruling (b): FROZEN-pinned score file + the D1 moments + unit_z + argmax
            re_ = at and at.get("result")
            src_a = ([refs["frozen_sha"], refs["d1_sha"][OUTCOME_LABEL], a_sha]
                     if re_ is not None and at.get("frozen_ok") and d1_ok.get(f"{OUTCOME_LABEL}|{nm}") else [])
            ref_a = f"score={at['score_ref']}" if at else "-"
            kpa = f"{at['score_ref'] if at else '-'}::run_key rows, unit_z (D1 joint__{nm}__mu/sd/mu0/sd0 row 0), argmax_command"
            note_a = "" if src_a else (at or {}).get("why", "no attack-side inputs")
            rows.append(row(art, f"T4|{oc}|{side}|z", ref_a, kpa, v["z"], re_ and re_["z"], "exact", compare(v["z"], re_ and re_["z"], "exact"),
                            src_a, signed, note_a, jpath=_jp(*jb, "z")))
            for k_, kk_ in (("unit_index", "unit"), ("n_units", "n_units")):
                rows.append(row(art, f"T4|{oc}|{side}|{k_}", ref_a, kpa + f" -> {kk_}", v.get(k_), re_ and re_[kk_], "count",
                                compare(v.get(k_), re_ and re_[kk_], "count"), src_a, signed, note_a, jpath=_jp(*jb, k_)))
            zt = (re_["z"] - float(T_A)) if re_ is not None else None
            rows.append(row(art, f"T4|{oc}|{side}|z_minus_T", ref_a + f",A={refs['A']}", kpa + " - " + kp, v["z_minus_T"], zt, "exact",
                            compare(v["z_minus_T"], zt, "exact"), src_a, signed, note_a, jpath=_jp(*jb, "z_minus_T")))
            if re_ is not None and bits is not None and (re_["z"] > float(T_A)) != (bits[nm] == "1"):  # R3: against A's bit, not the record's
                stops.append(f"tables34: T4|{oc}|{side}: the recomputed argmax z against A's T disagrees with A's hit bit")
            if re_ is not None and (re_["z"] > float(T_A)) != bool(v.get("caught")):
                stops.append(f"tables34: T4|{oc}|{side}: the record's caught disagrees with the recomputed argmax z against A's T")
            if at and at.get("result") is not None and not (at.get("file_cells_ok") and at.get("file_arm_ok")):  # R3 / O2: cell AND arm
                stops.append(f"tables34: T4|{oc}|{side}: the run_key's score-file keys are not the drawn cell's, or not this side's arm")
            for k_, jp_, kk_ in (("token", ("token",), "token"), ("unit/row_unit", ("unit", "row_unit"), "unit")):  # RQ2 rows; unit/token is on
                # the not-printed list again (09, 10-20: option (i) after O-b fired on all 11 units, whose score files carry call_index)
                pv_ = v.get(jp_[0]) if len(jp_) == 1 else (v.get("unit") or {}).get(jp_[1])
                rows.append(row(art, f"T4|{oc}|{side}|{k_}", ref_a, kpa + f" -> {kk_}", pv_, re_ and re_[kk_], "count",
                                compare(pv_, re_ and re_[kk_], "count"), src_a, signed, note_a, jpath=_jp(*jb, *jp_)))
    # §2 = Table 3 (the six false alarms) and the event counts
    fa = T["false_alarms"]
    draws = fa["draws"]
    if len(draws) != N_FALSEALARMS:
        stops.append(f"tables34: {len(draws)} false-alarm draws, not {N_FALSEALARMS}")
    for i, d in enumerate(draws, 1):
        lb, nm = d["probe"], POOL_NM.get(d["pool"])
        jb = ("false_alarms", "draws", i - 1)
        T_A = float(a_pool(A, lb, nm, "T"))
        kpT = f"rows/{lb}/<member>/pools_at_draw0/0.01/{nm}/T"
        rows.append(row(art, f"T3|fa{i}|T", f"A={refs['A']}", kpT, d["T"], T_A, "exact", compare(d["T"], T_A, "exact"), [a_sha], signed,
                        jpath=_jp(*jb, "T")))
        z_re, k_re, ix = None, None, np.array([], dtype=np.int64)
        if lb in d1 and lb in pools:
            ix = np.flatnonzero(np.asarray(pools[lb][nm]).astype(str) == str(d["alias"]))
            zz = np.asarray(d1[lb]["z"][nm], dtype=np.float64)[ix]
            above = np.flatnonzero(zz > T_A)
            z_re, k_re = (float(zz[above[0]]), int(above[0])) if len(above) else (None, None)
        src_z = [refs["d1_sha"][lb]] if d1_ok.get(f"{lb}|{nm}") else []
        kpz = f"{refs['d1'].get(lb, '-')}::draw0_z__{nm}[first unit of run alias in pool order with z > T]"
        rows.append(row(art, f"T3|fa{i}|z", f"D1={refs['d1'].get(lb, '-')}", kpz, d["z"], z_re, "exact", compare(d["z"], z_re, "exact"),
                        src_z, signed, "" if src_z else f"D1 not covered: {cond.get(f'{lb}|{nm}')}", jpath=_jp(*jb, "z")))
        zt = (z_re - T_A) if z_re is not None else None
        rows.append(row(art, f"T3|fa{i}|z_minus_T", f"D1={refs['d1'].get(lb, '-')},A={refs['A']}", kpz + " - " + kpT, d["z_minus_T"], zt,
                        "exact", compare(d["z_minus_T"], zt, "exact"), (src_z + [a_sha]) if src_z else [], signed, jpath=_jp(*jb, "z_minus_T")))
        # RQ2: the first alarming unit's index in the run (the locator's row_unit) and its score-file token: the run's FROZEN-pinned score
        # rows (by run_key) must carry exactly the pool's scores for this alias, so the token at that index is the same unit's
        rows.append(row(art, f"T3|fa{i}|unit/row_unit", f"D1={refs['d1'].get(lb, '-')}", kpz + " -> its index in the run", (d.get("unit") or {}).get("row_unit"),
                        k_re, "count", compare((d.get("unit") or {}).get("row_unit"), k_re, "count"), src_z, signed, jpath=_jp(*jb, "unit", "row_unit")))
        fu = (fa_units or {}).get(i) or {}
        tok_re = None
        if k_re is not None and fu.get("units") is not None and lb in pools and f"{nm}_s" in pools[lb] and f"{nm}_sbm" in pools[lb]:
            ps_ = np.asarray(pools[lb][f"{nm}_s"], dtype=np.float64)[ix]
            us_ = np.asarray(fu["units"]["s"], dtype=np.float64)
            pb_ = np.asarray(pools[lb][f"{nm}_sbm"], dtype=np.float64)[ix]  # O-a: the five-model scores must match as well
            ub_ = np.asarray(fu["units"].get("sbm", np.full((len(us_), 5), np.nan)), dtype=np.float64)
            if len(ps_) == len(us_) and np.array_equal(ps_, us_, equal_nan=True) and pb_.shape == ub_.shape and np.array_equal(pb_, ub_, equal_nan=True):
                tok_re = int(fu["units"]["token"][k_re])
        src_t = ([refs["frozen_sha"]] + src_z) if (tok_re is not None and fu.get("frozen_ok") and src_z) else []
        rows.append(row(art, f"T3|fa{i}|token", f"score={fu.get('score_ref', '-')}", "score-file token of the run_key's unit at row_unit "
                        "(the rows' scores must equal the pool's)", d.get("token"), tok_re, "count", compare(d.get("token"), tok_re, "count"), src_t,
                        signed, "" if src_t else (fu.get("why") or "the score rows do not reproduce the pool's scores"), jpath=_jp(*jb, "token")))
    dropped = {}
    for x in fa.get("duplicates_dropped", []):
        dropped[f"{x['probe']}|{x['pool']}"] = dropped.get(f"{x['probe']}|{x['pool']}", 0) + 1
    ev, tot = fa["events_per_probe_pool"], 0
    if set(ev) != {f"{lb}|{p}" for lb in FA_PROBES for p in POOL_NM}:
        stops.append("tables34: event count keys differ from the six probes x two pooled pools")
    for lb in FA_PROBES:
        for pool, nm in POOL_NM.items():
            k = f"{lb}|{pool}"
            T_A = float(a_pool(A, lb, nm, "T"))
            share, runs = a_pool(A, lb, nm, "per_run_alarm_share"), a_pool(A, lb, nm, "runs")
            n_a = share * runs
            n_a = int(round(n_a)) if abs(n_a - round(n_a)) <= 1e-9 else None
            n_d1 = None
            if lb in d1 and lb in pools:
                rr = np.asarray(pools[lb][nm]).astype(str)
                zz = np.asarray(d1[lb]["z"][nm], dtype=np.float64)
                n_d1 = len(set(rr[zz > T_A].tolist()))
            # R2: the record's own dedup is never used; a (probe, pool) with dropped duplicates has no recomputed count and no signed source
            re_ = n_d1 if (not dropped.get(k) and n_d1 is not None and n_d1 == n_a) else None
            tot = None if (tot is None or re_ is None) else tot + re_
            ok_src = d1_ok.get(f"{lb}|{nm}") and not dropped.get(k)
            rows.append(row(art, f"T3|events|{k}", f"A={refs['A']},D1={refs['d1'].get(lb, '-')}",
                            f"distinct pool runs with z > T (D1 draw0_z__{nm}, pools {nm}_run) == A per_run_alarm_share x runs",
                            ev.get(k), re_, "count", compare(ev.get(k), re_, "count"), ([a_sha, refs["d1_sha"][lb]] if ok_src else []), signed,
                            "" if re_ is not None else ("dropped duplicates in the record (dedup not recomputed)" if dropped.get(k)
                                                        else "D1 count and A's per_run_alarm_share x runs disagree or are unavailable"),
                            jpath=_jp("false_alarms", "events_per_probe_pool", k)))
    srcs_u = [a_sha] + [refs["d1_sha"][lb] for lb in FA_PROBES if lb in refs["d1_sha"]]
    all_cov = all(d1_ok.get(f"{lb}|{nm}") for lb in FA_PROBES for nm in ("B", "S")) and not dropped
    rows.append(row(art, "T3|universe_events", f"A={refs['A']}", "sum of the twelve event counts", fa.get("universe_events"), tot, "count",
                    compare(fa.get("universe_events"), tot, "count"), srcs_u if all_cov else [], signed,
                    jpath=_jp("false_alarms", "universe_events")))
    # RQ2: the dedup count is 0 exactly when every one of the twelve event counts verified from A and the D1 arrays with no drop
    dd_ok = all_cov and tot is not None
    rows.append(row(art, "T3|duplicates_dropped_count", f"A={refs['A']}", "0 when all twelve event counts verify with no drops",
                    fa.get("duplicates_dropped_count"), 0 if dd_ok else None, "count", compare(fa.get("duplicates_dropped_count"), 0 if dd_ok else None, "count"),
                    srcs_u if dd_ok else [], signed, jpath=_jp("false_alarms", "duplicates_dropped_count")))
    # R1: every numeric leaf is a row or on the generated not-printed list, and that list must be 09's approved one
    mapped = [r["jpath"] for r in rows]
    np_, np_sha, np_n, leaves = not_printed_list(T, mapped)
    if len(set(mapped)) != len(mapped) or not set(mapped) <= set(leaves):
        stops.append("tables34: a row's path is not a numeric field of the record, or two rows share one")
    appr = refs.get("not_printed_approved")
    if not appr:
        stops.append(f"tables34: the not-printed list awaits 09's approval (count {np_n}, sha256 {np_sha})")
    elif (appr.get("sha256"), appr.get("count")) != (np_sha, np_n):
        stops.append(f"tables34: the not-printed list (count {np_n}, sha256 {np_sha}) is not the approved one")
    plan = len(leaves) - (appr["count"] if appr else np_n)
    return rows, plan, stops, {"conditions": cond, "not_printed": {"paths": np_, "sha256": np_sha, "count": np_n},
                               "numeric_leaves": len(leaves),
                               "scope": "record verified against sources; NOT the printed Tables 3/4 (09 RQ3 gate: the v84 pass must map every "
                                        "Table 3/4 number in the post to a verified row of this record)",
                               "printed_numbers_verified": False}


SCOPE_TEXT = {
    "slots": "slot readout strings verified against sources; the post's use of the slots is the v84 pass",
    "figrender": "committed figure files (PNG bytes, source JSON, caption) verified against the re-render and sources; the post is the v84 pass",
    "post": "every number token of the frozen post (v84) classed and checked: verified (swap rule / provenance map / command count), "
            "not route C (checked against an unsigned record; apart), deferred (NOT VERIFIED; apart), non-result (apart), or orphan; number words "
            "are listed with their covering row, not gated"}


def artifact_outcome(a, summary):
    """O1: one artifact's outcome from the run summary (its own rows, and its sub-artifacts '<a>/...'): STOP on any unmatched / orphan /
    count mismatch / duplicate row or a stop message naming it; PARTIAL with only deferred rows outstanding; else PASS. -> (outcome,
    n_equal, n_printed, base code, not-route-C printed numbers) over its number rows."""
    per = (summary or {}).get("per_artifact") or {}
    st = (summary or {}).get("stop") or {}
    mine = [k for k in per if k == a or k.startswith(a + "/")]
    bad = any(per[k].get("n_unmatched") or per[k].get("n_orphan") for k in mine) or any(
        x == a or x.startswith(a + "/") or x.startswith(a + "::") or x.startswith(a + "/") or x.startswith(f"{a}: ")
        for k in ("unmatched", "orphan", "count_mismatch", "duplicate_slots", "other") for x in st.get(k) or [])
    p = per.get(a) or {}
    base = "STOP" if bad or not mine else ("PARTIAL" if p.get("n_deferred") else "PASS")
    n_nrc = sum(per[k].get("n_not_route_C", 0) for k in mine)  # printed numbers only (check rows are counted apart)
    n_pre = sum(per[k].get("n_preliminary", 0) for k in mine)
    return verdict_line(base, n_nrc, n_pre=n_pre), p.get("n_equal", 0), p.get("n_printed", 0), base, n_nrc + n_pre


def result_scope(kinds, diag, errors=(), run_error=None, summary=None):
    """RT1 (10-20): ONE top-level scope entry and printed flag for EVERY artifact of the run. kinds: artifact -> kind. tables34 takes its
    record-only string (printed False) unless the post ran and verified every printed Table 3/4 number with its not-printed list
    approved for that pass (then True); slot readouts and figures their kind's string (printed False); the post its string (printed True
    only on PASS); an errored artifact, or every artifact when the whole run errored, reads "NOT VERIFIED (error <class> in <phase>)"
    with printed None. O1: every entry ends with the artifact's outcome and its equal / printed counts. -> (scope, printed)."""
    scope, printed = {}, {}
    err_by = {e.get("where"): e.get("class") for e in errors or ()}
    pd = next(((diag or {}).get(n) for n, k in kinds.items() if k == "post"), None)
    pn = next((n for n, k in kinds.items() if k == "post"), None)
    for a, kind in kinds.items():
        d = (diag or {}).get(a)
        err = run_error or ((d or {}).get("error") if isinstance(d, dict) else None) or ({"class": err_by[a], "where": a} if a in err_by else None)
        if err or not isinstance(d, dict):
            e_ = err or {"class": "missing diagnostics", "where": a}
            scope[a], printed[a] = f"NOT VERIFIED (error {e_.get('class')} in {e_.get('where')})", None
            continue
        if kind == "tables34":
            t_ok = isinstance(pd, dict) and pd.get("tables34_printed_numbers_verified") is True and not any(  # R8: no post gate stop
                x.startswith(f"{pn}:") or "not-printed list for this pass" in x for x in ((summary or {}).get("stop") or {}).get("other") or [])
            s_ = (d.get("scope") or "NOT VERIFIED (error no scope in tables34)") + (
                f"; printed Tables 3/4 numbers verified by the v84 pass ({pn})" if t_ok else "")
            scope[a], printed[a] = s_, (True if t_ok else (False if d.get("scope") else None))
        elif kind == "post":
            _o, _ne, _np, base_, nrc_ = artifact_outcome(a, summary)
            # Q3 / 09 (b): the post's printed flag compares the base CODE, not the line. It is True only on a PASS whose printed
            # numbers are all route-C rows (verified, or approved non-results). A PASS holding not-route-C numbers stays possible, but
            # the flag is False then: those numbers are checked against unsigned records, never route-C-verified; the entry names them.
            scope[a], printed[a] = SCOPE_TEXT["post"], base_ == "PASS" and nrc_ == 0
            labs = sorted(set(d.get("preliminary_labels_used") or []))
            if labs:  # 2e / 09 (10-20): the full per-record PRELIMINARY label(s), verbatim, here and not in the verdict line
                scope[a] += "; PRELIMINARY labels: " + " | ".join(labs)
        else:
            scope[a], printed[a] = SCOPE_TEXT.get(kind, f"NOT VERIFIED (error unknown kind {kind} in {a})"), (False if kind in SCOPE_TEXT else None)
        if summary is not None and not scope[a].startswith("NOT VERIFIED"):
            o, ne, np_, _b, _n = artifact_outcome(a, summary)
            scope[a] += f" | outcome {o}: {ne} / {np_} numbers equal"
    return scope, printed
