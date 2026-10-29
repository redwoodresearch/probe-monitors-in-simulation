"""Writing's slot strings from pooled records (route C slot readout; 0d via 2e, 10-17). Pure: parsed records in, {slot_key: str} out, no
I/O. It runs only inside the Modal container (run_slot_readout.py); nothing on the laptop parses a record.

Key paths are the writer's (run_peraction311.py; NOTE_slot_readout.md maps every slot to its writer line). Each slot is one put(...) call in
build(); to add a slot, add a call. Integrity failures (a tie to the wrong record, member rows that disagree, a non-integer run count) STOP;
a slot whose key path is absent is listed in `missing` with the path where it broke (no value), never filled in."""

import math

MINUS = "−"
LEAD = "pooled_all_origins"
CR = ("C", "R")
SIX = ("sol_a_uncut", "FIBS|uncut", "harmfulness|uncut", "refusal|uncut", "high-stakes|uncut", "own_all_rs_uncut_L48")  # FO.HEADLINE_SIX
FOUR = ("sol_a_uncut", "FIBS|uncut", "harmfulness|uncut", "own_all_rs_uncut_L48")  # writing's four-probe group (#8, #9)
ON_POLICY = "own_all_rs_uncut_L48"
PUBLIC = ("FIBS|uncut", "harmfulness|uncut", "refusal|uncut", "high-stakes|uncut")
TRAINED = ("sol_a_uncut", "own_all_rs_uncut_L48")
C_ASIS = "c_asis_uncut"
R37_LEVELS = (1e-4, 2e-4, 5e-4)
POOLS = ("B", "A", "S")
POOL_NAME = {"B": "honest pool B", "A": "honest pool A", "S": "sim honest pool S"}  # 09 (10-18): rendered text never says a bare "B"
RS_PARTIAL = ("own_all_rs10_uncut_L48", "own_all_rs25_uncut_L48")  # writing via 0d (10-18, drop 1a): the on-policy probes trained on a
# fraction of tasks; in the writer's PW_FAMS (run_peraction311.py L862 @ fbb59a213, blob 71633875; L829 @ 704122edf), labelled "<fam> <C|R>"
# (L6748, L6763), each pooled block carrying catch.run1 / catch.run2 (L3374)
LEAD_MEMBERS = 4  # PW_MEMBERS["pooled_all_origins"]: S_old, S_new, S_newmains, S_paper493 (all-origins draws)


class Missing(Exception):
    """A key path the record does not carry (reported by slot key; never a value)."""


class NotInt(Exception):
    """A count that is not an int, or a number that is not an int / float (a bool, a list, an array ...): the slot STOPs, by key (a0b05
    R-1; a63478 L1-L3)."""


def intv(v, what):
    if type(v) is not int:  # bool is not int here
        raise NotInt(f"{what} is {type(v).__name__}, not int")
    return v


def numv(v, what):
    if type(v) not in (int, float):  # bool, list, array, str: never a number here
        raise NotInt(f"{what} is {type(v).__name__}, not a number")
    return v


def cnt(v):  # an integer slot: (string with thousands separators, e.g. "1,906" (ctrl1 fold-in, 10-17), full)
    return f"{intv(v, 'the count'):,}", str(v)


def get(d, *path):
    """d[p0][p1]...; a float in the path matches a level key by value (JSON keys are strings such as "0.01")."""
    for i, p in enumerate(path):
        if not isinstance(d, dict):
            raise Missing("/".join(map(str, path[:i + 1])))
        if isinstance(p, float):
            ks = [k for k in d if _isnum(k) and math.isclose(float(k), p, rel_tol=1e-9, abs_tol=0.0)]
            if len(ks) != 1:
                raise Missing("/".join(map(str, path[:i + 1])))
            p = ks[0]
        if p not in d:
            raise Missing("/".join(map(str, path[:i + 1])))
        d = d[p]
    return d


def _isnum(k):
    try:
        float(k)
        return True
    except (TypeError, ValueError):
        return False


def levels(d):
    return sorted(float(k) for k in d if _isnum(k))


def num(x, nd, signed):
    numv(x, "the value")
    s = f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"
    return s.replace("-", MINUS)


def pct(x, nd=1):  # catch rates: % with 1 decimal
    return num(100 * x, nd, False) + "%"


def pp(x, nd=1):  # gaps: percentage points, signed, a true minus
    return num(100 * x, nd, True)


def pp2(x):  # the on-policy probe's Δ_full: 2 decimals
    return pp(x, 2)


def share(x):  # shares: whole %
    return num(100 * x, 0, False) + "%"


def g17(x):
    return "%.17g" % x


def summ_str(node, fmt):
    """A summ() node (run_fprsweep_507.summ: point = draw 0; ci95 = 2.5/97.5 percentiles of the finite draws, None when withheld)."""
    if not isinstance(node, dict) or "point" not in node:
        raise Missing("summ node")
    if node.get("ci95") is None:
        return f"{fmt(node['point'])} (interval withheld: {node.get('withheld', 'no interval')})", g17(node["point"]) + " [withheld]"
    if not isinstance(node["ci95"], list) or len(node["ci95"]) != 2:
        raise NotInt("ci95 is not a two-number list")
    lo, hi = node["ci95"]
    return f"{fmt(node['point'])} ({fmt(lo)} to {fmt(hi)})", f"{g17(node['point'])} [{g17(lo)}, {g17(hi)}]"


def ci_str(node, fmt):
    if not isinstance(node, dict) or "point" not in node:
        raise Missing("summ node")
    if node.get("ci95") is None:
        return f"interval withheld: {node.get('withheld', 'no interval')}", "[withheld]"
    if not isinstance(node["ci95"], list) or len(node["ci95"]) != 2:
        raise NotInt("ci95 is not a two-number list")
    lo, hi = node["ci95"]
    return f"({fmt(lo)} to {fmt(hi)})", f"[{g17(lo)}, {g17(hi)}]"


def q_str(q, *path, fmt):
    """A per-label block's summary, except that a block with `reading` prints "not readable", never the interval (writer L3245-3246)."""
    if "reading" in q:
        return "not readable", "not readable"
    return summ_str(get(q, *path), fmt)


def real_line_str(q, fmt):
    """real_line has its own gate (a63478 R4): RULE 45's B-only flag q["readable_real_line"] (writer L3196-3198 at 704122edf, L3247-3249
    now), not the all-pool `reading` gate (L3175 / L3226). The writer sets that flag on every pooled block, under the same pooled_world
    condition as g["real_line"] itself (L3149-3151 / L3200-3202); so if the flag is absent there is no real_line to print, and the slot is
    listed as missing rather than gated by `reading`."""
    if "readable_real_line" not in q:
        raise Missing("readable_real_line")
    if q["readable_real_line"] is not True:
        return "not readable", "not readable"
    return summ_str(get(q, "gap", "real_line"), fmt)


def member_rows(rec, lab, cb=LEAD):
    rows = get(rec, "rows", lab)
    mem = sorted(o for o in rows if o.endswith(f" [pooled: {cb}]"))
    if len(mem) != LEAD_MEMBERS:
        raise SystemExit(f"{lab}: {len(mem)} member rows for {cb}, expected {LEAD_MEMBERS} (STOP)")
    return [rows[o] for o in mem]


def draw0(rec, lab, lvl, pool, key):
    """rows[lab][<member> [pooled: LEAD]].pools_at_draw0[lvl][pool][key]: the combo's member rows share the pooled pools, so every member
    must carry the same value (finish_rows checks the alarm shares; here any key), else STOP."""
    vs = [get(r, "pools_at_draw0", lvl, pool, key) for r in member_rows(rec, lab)]
    if any(v != vs[0] for v in vs[1:]):
        raise SystemExit(f"{lab} {pool} {key}: the combo's member rows disagree (STOP)")
    return vs[0]


def distinct_runs_above(rec, lab, lvl, pool):
    """Runs with any unit above T at draw 0 = per_run_alarm_share x runs (writer L2609-2610 at 704122edf: the share is the mean over runs of
    max unit > T); must be an integer to 1e-9, else STOP."""
    n = numv(draw0(rec, lab, lvl, pool, "per_run_alarm_share"), "per_run_alarm_share") * intv(draw0(rec, lab, lvl, pool, "runs"), "runs")
    if abs(n - round(n)) > 1e-9:
        raise SystemExit(f"{lab} {pool}: per_run_alarm_share x runs is not an integer (STOP)")
    return int(round(n))


def zero_unit_runs(rec, lab, lvl, pool):
    """Honest runs with no scored unit at draw 0. per_run_alarm_share's denominator is `runs` = the runs that have a unit in the pool (runs_ is
    built from the pool's units; writer L2649-2653 now, L2605-2609 at 704122edf); ..._with_no_unit_runs adds the admitted zero-unit runs and
    the structural no-row cells, as non-alarming (L2656-2667 / L2612-2623), and the record counts them in no_unit_runs_counted_non_alarming.
    The count is zero_unit_admitted + len(structural_no_row_cells); STOP unless the two shares agree with it."""
    nu = draw0(rec, lab, lvl, pool, "no_unit_runs_counted_non_alarming")
    st = get(nu, "structural_no_row_cells")
    if not isinstance(st, list):
        raise NotInt(f"structural_no_row_cells is {type(st).__name__}, not a list")
    n = intv(get(nu, "zero_unit_admitted"), "zero_unit_admitted") + len(st)
    runs, s1, s2 = (draw0(rec, lab, lvl, pool, k) for k in ("runs", "per_run_alarm_share", "per_run_alarm_share_with_no_unit_runs"))
    intv(runs, "runs"), numv(s1, "per_run_alarm_share"), numv(s2, "per_run_alarm_share_with_no_unit_runs")
    if abs(s1 * runs - s2 * max(runs + n, 1)) > 1e-6:
        raise SystemExit(f"{lab} {pool}: the two alarm shares do not agree with {n} no-unit runs (STOP)")
    return n


def group_units(rec, labs, pool):
    """(units, runs) at draw 0, C, 0.01, shared by every member of the group (writing: the members have identical units), else STOP."""
    vs = {(intv(draw0(rec, f"{lb} C", 0.01, pool, "n_units"), "n_units"), intv(draw0(rec, f"{lb} C", 0.01, pool, "runs"), "runs")) for lb in labs}
    if len(vs) != 1:
        raise SystemExit(f"{labs}: the group's members differ in units or runs in pool {pool} (STOP)")
    return vs.pop()


def skip_share(rec, pool):
    """(public units - trained units) / public units: the share of the public probes' scored calls the trained probes skip (writing, #17)."""
    up, ut = group_units(rec, PUBLIC, pool)[0], group_units(rec, TRAINED, pool)[0]
    return (up - ut) / up


def round5(x):  # to the nearest 5, halves up
    return int(math.floor(x / 5 + 0.5) * 5)


def check_tie(doc, role, want_sha, sha_path):
    got = get(doc, *sha_path)
    if got != want_sha:
        raise SystemExit(f"{role}: built from another record than A (its {'/'.join(sha_path)} differs; STOP)")


# ---- downstream records (0d via 2e, 10-17): kappa, the five-probe kappa extra, detector_vs_drop, the R² anchor, PLANT Table 6 --------------
# Writers (blobs): kappa = run_peraction311.py 0c7714f8 (at 540ae554d); kappa extra = run_kappa_extra.py 90cebcf0; detector_vs_drop =
# run_detector_vs_drop.py 3dd26da4; anchor = run_pertask_anchor.py 91702033; PLANT = experiments/2026-10-15_panel_noise_matched/
# run_plant_rows.py 32812bb6. NOTE_slot_readout.md maps every slot to its writer line.
DOWN = ("KAPPA", "KAPPA_EXTRA", "DETDROP", "ANCHOR", "PLANT")
KAPPA_PAIRS = ("kappa_sim_vs_run1", "kappa_run2_vs_run1", "kappa_sim_minus_run2")
DET_ANALYSES = ("sim_score_vs_drop", "null_run2_score_vs_run2_minus_run1")
DETECTORS = ("commands L63", "replies L48")
PLANT_TAGS = ("-12", "-8", "-5", "-3", "+3", "+5", "+8", "+12")
ANCHOR_SET = "pooled_all_origins (union stream)"


def k2(x):  # kappa, R², Spearman: 2 decimals, a true minus
    return num(x, 2, False)


def k2s(x):  # a kappa or R² difference: 2 decimals, signed
    return num(x, 2, True)


def ppv(x):  # PLANT's values are already in percentage points (writer L520-526, L188-195): 1 decimal, signed
    return num(x, 1, True)


RECORD_ROLES = ("A", "A_group_catch", "A_group_excess", "B", "R37") + DOWN  # every record a slot reads (09: one label rule for all)
REGISTRY = "LABELS_171"  # the committed 171-pool registry (results/LABELS_171pool_2026-10-17.json; CUT builds it, 09 verifies it)
REGISTRY_SHA256 = "5cbc8fb41a73d9b7c69f65ca90e1e705f40abebaace81f971574e890c0c52d81"  # 09's verified registry @ 429232cee (2e, 10-17): any other STOPs
APPENDIX_ROLES = ("KAPPA", "KAPPA_EXTRA", "DETDROP", "ANCHOR", "PLANT", "B", "R37")  # Aswin 10-17 (via 0d, 2e): the appendix is route C only
CONTROL_MODE = "CONTROL_ONLY: a control of the slot reader against v83's PRELIMINARY strings; never used by writing; not an appendix"


def registry_entries(reg):
    """The 171-pool registry: a list of {record, record_sha256, label} (or {"entries": [...]}), one entry per record path and per sha; a
    malformed or duplicated entry STOPs."""
    es = reg.get("entries") if isinstance(reg, dict) else reg
    if not isinstance(es, list) or not es:
        raise SystemExit("LABELS_171: not a non-empty list of entries (STOP)")
    for e in es:
        if not isinstance(e, dict) or not {"record", "record_sha256", "label"} <= set(e) or not all(isinstance(e[k], str) for k in ("record", "record_sha256", "label")):
            raise SystemExit("LABELS_171: an entry is not {record, record_sha256, label} as strings (STOP)")
    for k in ("record", "record_sha256"):
        if len({e[k] for e in es}) != len(es):
            raise SystemExit(f"LABELS_171: a {k} appears twice (STOP)")
    return es


def record_label(role, recs, shas, paths, control):
    """09's ONE label rule for every slot (10-17): the record's sha in the 171-pool registry -> the registry's label, which must contain
    "171-pool"; else a route C record -> its route C sidecar ({record, record_sha256, label} tied to the record's sha, a label naming route
    C); else STOP. A registry entry naming this record (same path, or same file name less any .gz: a Volume copy of a committed record)
    with another sha STOPs, and so does a registry record that also has a sidecar (two labels for one record). Aswin's ruling (10-17): the
    registry's labels only cite v83's PRELIMINARY records, so an appendix-bound record (APPENDIX_ROLES) with a 171-pool label STOPs
    unless the run is a CONTROL run."""
    sha, path, sc = shas[role], paths[role], recs.get(f"{role}_label")
    es = registry_entries(recs[REGISTRY]) if REGISTRY in recs else []
    nm = lambda s: s.rsplit("/", 1)[-1].removesuffix(".gz")  # noqa: E731
    if any(e["record_sha256"] != sha and (e["record"] == path or nm(e["record"]) == nm(path)) for e in es):
        raise SystemExit(f"{role}: a 171-pool registry entry names this record with another sha (STOP)")
    hit = [e for e in es if e["record_sha256"] == sha]
    if hit:
        if sc is not None:
            raise SystemExit(f"{role}: in the 171-pool registry and also given a sidecar (STOP)")
        if "171-pool" not in hit[0]["label"]:
            raise SystemExit(f"{role}: its registry label does not contain 171-pool (STOP)")
        if role in APPENDIX_ROLES and not control:
            raise SystemExit(f"{role}: a 171-pool label on an appendix-bound record; the appendix is route C only (Aswin 10-17), so only a "
                             "CONTROL run reads it (STOP)")
        return hit[0]["label"]
    if sc is None:
        raise SystemExit(f"{role}: neither in the 171-pool registry nor given a route C sidecar (09; STOP)")
    if not isinstance(sc, dict) or set(sc) != {"record", "record_sha256", "label"} or not isinstance(sc["label"], str):
        raise SystemExit(f"{role}_label: the sidecar is not {{record, record_sha256, label}} (STOP)")
    if sc["record_sha256"] != sha:
        raise SystemExit(f"{role}_label: the sidecar names another record than {role} (STOP)")
    if "route C" not in sc["label"]:
        raise SystemExit(f"{role}_label: the sidecar's label does not name route C (09; STOP)")
    return sc["label"]


def gate_pass(role, rec, *path):
    if get(rec, *path) is not True:
        raise SystemExit(f"{role}: its gate ({'.'.join(path)}) did not pass (STOP)")


def node_str(e, key, fmt):
    """A summary node under a block that may carry withheld / unavailable / excluded instead (the kappa writer L4018, L4060, L4063)."""
    for k in ("excluded", "unavailable", "withheld"):
        if k in e:
            return f"{k}: {e[k]}", k
    return summ_str(get(e, key), fmt)


def downstream(recs, shas, put, nc):
    for r in DOWN:
        if r not in recs:
            nc[r] = f"no {r} record given"
    def ksumm(node, key):  # the kappa writer's summ_ (L3988-3993): point None (draw 0 undefined) comes with ci95 None and withheld
        if isinstance(node, dict) and node.get("point") is None and "withheld" in node:
            return f"withheld: {node['withheld']}", "withheld"
        return summ_str(node, k2s if key.endswith("minus_run2") else k2)

    def kblock(e, mode, key):  # a combo / group entry: excluded, unavailable or withheld as a whole (L4018, L4060, L4063, L4096-4103), else the mode
        for k in ("excluded", "unavailable", "withheld"):
            if k in e:
                return f"{k}: {e[k]}", k
        return ksumm(get(e, mode, key), key)

    K = recs.get("KAPPA")
    if K is not None:
        gate_pass("KAPPA", K, "gate", "pass")
        for lb in SIX:
            for ro in CR:
                for mode in ("joint", "fixed"):
                    for key in KAPPA_PAIRS:
                        put(f"21.kappa.{lb} {ro}.{mode}.{key}", lambda lb=lb, ro=ro, mode=mode, key=key: kblock(
                            get(K, "kappa", f"{lb} {ro}", "combos", LEAD), mode, key))
        for ro in CR:
            for mode in ("joint", "fixed"):
                for key in ("mean_kappa_sim_vs_run1", "mean_kappa_run2_vs_run1", "mean_kappa_sim_minus_run2"):
                    put(f"21.kappa_group.headline_six|{ro}.{mode}.{key}", lambda ro=ro, mode=mode, key=key: kblock(
                        get(K, "kappa_groups", f"headline_six|{ro}|{LEAD}"), mode, key))
    KX = recs.get("KAPPA_EXTRA")
    if KX is not None:
        gate_pass("KAPPA_EXTRA", KX, "gate", "pass")
        if "KAPPA" in recs and get(KX, "meta", "record_sha256") != shas.get("KAPPA"):
            raise SystemExit("KAPPA_EXTRA: built against another kappa record than KAPPA (STOP)")
        if sorted(get(KX, "five_probe_sensitivity", "members")) != sorted(x for x in SIX if x != ON_POLICY):
            raise SystemExit("KAPPA_EXTRA: its five members are not the six minus the on-policy probe (STOP)")
        for ro in CR:
            for mode in ("joint", "fixed"):
                for key in ("mean_kappa_sim_vs_run1", "mean_kappa_run2_vs_run1", "mean_kappa_sim_minus_run2"):
                    put(f"22.kappa_five.{ro}.{mode}.{key}", lambda ro=ro, mode=mode, key=key: kblock(
                        get(KX, "five_probe_sensitivity", "groups", f"five_probe_sensitivity|{ro}|{LEAD}"), mode, key))
    D = recs.get("DETDROP")
    if D is not None:
        def ydef():  # 2e (10-18): y's definition verbatim beside the slot-23 values, so nobody misreads the sign (y = sim - real). Writer
            v = get(D, "meta", "definitions", "y")  # meta.definitions.y: run_detector_vs_drop.py L268-269 @ 3dd26da4; L358-359 @ a4ee8256
            if not isinstance(v, str) or not v.strip():
                raise NotInt("meta.definitions.y is not a non-empty string")
            return v, v
        put("23.definition.y", ydef)
        for pr in SIX:
            for dn in DETECTORS:
                for an in DET_ANALYSES:
                    put(f"23.spearman.{pr} C.{dn}.{an}", lambda pr=pr, dn=dn, an=an: summ_str(  # signed point and ends, as the post writes them (0d, 10-17)
                        get(D, "result", "probes", f"{pr} C", dn, an), k2s))
                    put(f"23.n_cells.{pr} C.{dn}.{an}", lambda pr=pr, dn=dn, an=an: cnt(get(D, "result", "probes", f"{pr} C", dn, an, "n_cells")))
    AN = recs.get("ANCHOR")
    if AN is not None:
        gate_pass("ANCHOR", AN, "gate", "pass")

        def yhw(blk, L):  # half the yardstick R² interval's width (the post's "about 0.04"); None when the interval is withheld
            n_ = get(AN, "union_stream", blk, L, "sets", ANCHOR_SET, "r2", "yardstick")
            if not isinstance(n_, dict) or "point" not in n_:
                raise Missing("summ node")
            if n_.get("ci95") is None:
                return None
            if not isinstance(n_["ci95"], list) or len(n_["ci95"]) != 2:
                raise NotInt("ci95 is not a two-number list")
            return (numv(n_["ci95"][1], "ci95") - numv(n_["ci95"][0], "ci95")) / 2

        for lb in SIX:
            for ro in CR:
                L = f"{lb} {ro}"
                for blk in ("lead", "sensitivity_run2_anchor"):
                    if blk == "sensitivity_run2_anchor" and lb != ON_POLICY:
                        continue  # the run2 block is the on-policy probes' only (writer L86, L475)
                    for key in ("main", "yardstick", "delta_main_minus_yardstick"):
                        put(f"24.r2.{blk}.{L}.{key}", lambda L=L, blk=blk, key=key: summ_str(
                            get(AN, "union_stream", blk, L, "sets", ANCHOR_SET, "r2", key), k2s if key.startswith("delta") else k2))
                    # the anchor's own task count (writer one_set L370, union sets L490: n_cells), not detector_vs_drop's (0d's ctrl2 diff)
                    put(f"24.n_cells.{blk}.{L}", lambda L=L, blk=blk: cnt(get(AN, "union_stream", blk, L, "sets", ANCHOR_SET, "n_cells")))
                    put(f"24.r2.yardstick_halfwidth.{blk}.{L}", lambda L=L, blk=blk: (lambda h: ("interval withheld", "withheld") if h is None
                                                                                      else (k2(h), g17(h)))(yhw(blk, L)))

        def yrange():  # min to max over the twelve lead labels with an interval (withheld ones counted, not filled)
            hs = [yhw("lead", f"{lb} {ro}") for lb in SIX for ro in CR]
            ok, w = [h for h in hs if h is not None], sum(h is None for h in hs)
            if not ok:
                return "every interval withheld", "withheld"
            tail = f" ({w} withheld)" if w else ""
            return f"{k2(min(ok))} to {k2(max(ok))}{tail}", f"{g17(min(ok))} to {g17(max(ok))}{tail}"
        put("24.r2.yardstick_halfwidth.lead.range", yrange)
    P = recs.get("PLANT")
    if P is not None:
        if get(P, "gates", "n_fail") != 0 or get(P, "shape_gate", "pass") is not True:
            raise SystemExit("PLANT: its gates did not pass (gates.n_fail / shape_gate.pass) (STOP)")
        for f in get(P, "levels"):
            lv = float(f)
            for ro in CR:
                for lb in SIX:
                    for tag in PLANT_TAGS:
                        def f_(lv=lv, ro=ro, lb=lb, tag=tag):
                            v = get(P, "rows", lv, ro, f"{lb} {ro}", tag)
                            s = get(v, "verdict")
                            return (f"{s} ({v['reason']})" if s == "unavailable" and "reason" in v else s), s
                        put(f"25.plant.verdict.{float(f):g}.{lb} {ro}.{tag}", f_)
                for tag in PLANT_TAGS:
                    for agg in ("S", "excess"):
                        def f_(lv=lv, ro=ro, tag=tag, agg=agg):
                            a = get(P, "aggregate", "by_level", lv, ro, tag, agg)
                            s = get(a, "verdict")
                            if "shift" not in a:
                                return s, s
                            sh = summ_str(get(a, "shift"), ppv)
                            return f"{s}; shift {sh[0]}", f"{s}; shift {sh[1]}"
                        put(f"25.plant.aggregate.{float(f):g}.{ro}.{tag}.{agg}", f_)
            for tag in PLANT_TAGS:
                for ro in ("C", "R", "C+R"):
                    def f_(lv=lv, tag=tag, ro=ro):  # detected / missed / unavailable apart: a ceiling is never a miss (0d's ctrl2 diff, 10-17)
                        c = get(P, "counts", lv, tag, ro)  # writer L553-556: detected, not_detected, unavailable, cases; misses = name -> verdict
                        k, m, u, n = (intv(get(c, x), x) for x in ("detected", "not_detected", "unavailable", "cases"))
                        ms = get(c, "misses")
                        if not isinstance(ms, dict):
                            raise NotInt("counts.misses is not a dict")
                        by = {v: sorted(x for x, w in ms.items() if w == v) for v in ("not detected", "unavailable")}
                        if k + m + u != n or (len(by["not detected"]), len(by["unavailable"]), len(ms)) != (m, u, m + u):
                            raise SystemExit(f"PLANT counts {lv:g} {tag} {ro}: detected + missed + unavailable is not the cases, or the named misses "
                                             "disagree (STOP)")
                        tail = "; ".join(f"{w}: {', '.join(by[v])}" for w, v in (("missed", "not detected"), ("unavailable", "unavailable")) if by[v])
                        return (f"{k:,}/{n:,} detected, {m:,} missed, {u:,} unavailable" + (f" ({tail})" if tail else ""),
                                f"{k}/{n}; missed {m}; unavailable {u}")
                    put(f"25.plant.count.{float(f):g}.{tag}.{ro}", f_)


def build(recs, shas, paths, control=False):
    """recs: role -> parsed record; shas: role -> its verified sha256; paths: role -> its pointer path; control: a CONTROL run (its output
    is never used by writing). -> {"slots", "full", "missing", "not_carried"}. Every record's label is checked first (09's rule and
    Aswin's ruling); a failing label STOPs before any slot. The slot `mode` says which kind of run made the output."""
    out, full, missing, nc = {}, {}, {}, {}
    out["mode"] = full["mode"] = CONTROL_MODE if control else "writing"
    if REGISTRY in recs and shas.get(REGISTRY) != REGISTRY_SHA256:  # shas are the container-verified bytes' (load_inputs)
        raise SystemExit("LABELS_171: its bytes do not hash to 09's verified registry (REGISTRY_SHA256; STOP)")
    stray = sorted(k for k in recs if k != REGISTRY and k not in RECORD_ROLES and not (k.endswith("_label") and k[:-6] in recs))
    if stray:
        raise SystemExit(f"{stray}: inputs with no record they label (STOP)")
    # a63478 R-1: only the registry tells a 171-pool record from a route C one, so a writing run without it could take a relabelled one
    if not control and REGISTRY not in recs and any(r in recs for r in APPENDIX_ROLES):
        raise SystemExit("a writing run with appendix-bound records needs the 171-pool registry, to tell a 171-pool record from a route C "
                         "one (a63478 R-1; STOP)")
    lbl_ = {r: record_label(r, recs, shas, paths, control) for r in RECORD_ROLES if r in recs}
    for r, lab in lbl_.items():
        out[f"lbl.{r}"], full[f"lbl.{r}"] = lab, lab

    def put(key, fn):
        try:
            out[key], full[key] = fn()
        except Missing as e:
            missing[key] = str(e)
        except NotInt as e:
            raise SystemExit(f"{key}: {e} (STOP)") from None

    A = recs.get("A")
    if A is None:
        nc["01-17"] = "no A record given"
    else:
        coh = ("coheadline_signed", LEAD, "joint")
        for n, grp, ro, lv, flds in ((1, "headline_six_full", "C", 0.01, ("S", "Y")), (2, "auroc06_subset", "C", 0.01, ("S", "Y")),
                                     (3, "headline_six_full", "R", 0.01, ("S", "Y")), (4, "auroc06_subset", "R", 0.01, ("S", "Y")),
                                     (5, "headline_six_full", "C", 0.05, ("S",))):
            for fld in flds:
                def f_(grp=grp, ro=ro, lv=lv, fld=fld):
                    q = get(A, *coh, lv, f"{grp}|{ro}")
                    if "unavailable" in q:
                        return f"unavailable: {q['unavailable']}", "unavailable"
                    return summ_str(get(q, fld), pp)
                put(f"{n:02d}.coheadline.{grp}|{ro}@{lv:g}.{fld}", f_)
        for n, ro in ((6, "C"), (7, "R")):
            def f_(ro=ro):
                pr = get(A, "panel_groups", LEAD, "pooled", "joint", 0.01, f"headline_six_full|{ro}", "primary")
                for k in ("withheld", "unavailable"):
                    if k in pr:
                        return f"{k}: {pr[k]}", k
                return summ_str(get(pr, "mean_G_minus_mean_Y"), pp)
            put(f"{n:02d}.excess.headline_six_full|{ro}", f_)
        for lb in SIX:
            for ro in CR:
                L = f"{lb} {ro}"
                q_ = ("pooled", L, LEAD, "joint", 0.01)
                for arm in ("run1", "run2", "sim"):
                    put(f"09.catch.{L}.{arm}", lambda q_=q_, arm=arm: q_str(get(A, *q_), "catch", arm, fmt=pct))
                put(f"10.delta_full.{L}", lambda q_=q_, lb=lb: q_str(get(A, *q_), "gap", "delta_full", fmt=pp2 if lb == ON_POLICY else pp))
                put(f"11.run2_minus_run1_at_tauB.{L}", lambda q_=q_: q_str(get(A, *q_), "gap", "run2_minus_run1_at_tauB", fmt=pp))
        for lb in RS_PARTIAL:  # 9, rs10 / rs25 (writing via 0d, 10-18): the pooled real-attack catch at 1%, as slot 9 reads it (same combo,
            for ro in CR:  # joint, 0.01 and `reading` rule); run1 and run2 by key, and their mean in the container (writing picks one)
                L = f"{lb} {ro}"
                q_ = ("pooled", L, LEAD, "joint", 0.01)
                for arm in ("run1", "run2"):
                    put(f"09.catch.{L}.{arm}", lambda q_=q_, arm=arm: q_str(get(A, *q_), "catch", arm, fmt=pct))

                def f_(q_=q_):  # the probe's own run 1 and run 2 points, equal weight (point only), as the 9 avg rows do
                    q = get(A, *q_)
                    if "reading" in q:
                        return "not readable", "not readable"
                    v = (numv(get(q, "catch", "run1", "point"), "the point") + numv(get(q, "catch", "run2", "point"), "the point")) / 2
                    return pct(v), g17(v) + " [point only: the mean of real run 1's and real run 2's catch points]"
                put(f"09.catch.{L}.mean", f_)
        for gname, labs in (("six", SIX), ("four", FOUR)):  # 9 avg, run1 / run2 columns (Tables 5/8; ctrl1 fold-in, 10-17): A stores no group
            for ro in CR:  # run1 / run2, so the point is the equal-weight mean of the members' recorded points (run_group_catch's own point rule)
                for arm in ("run1", "run2"):
                    def f_(labs=labs, ro=ro, arm=arm):
                        qs = [get(A, "pooled", f"{x} {ro}", LEAD, "joint", 0.01) for x in labs]
                        if any("reading" in q for q in qs):
                            return "not readable", "not readable"
                        v = sum(numv(get(q, "catch", arm, "point"), "the point") for q in qs) / len(qs)
                        return pct(v), g17(v) + " [point only]"
                    put(f"09.catch_avg.{gname}|{ro}.{arm}", f_)

        def hw(lb):  # half the 95% interval's width, in percentage points (C, joint, 0.01)
            q = get(A, "pooled", f"{lb} C", LEAD, "joint", 0.01)
            if "reading" in q:
                raise Missing(f"{lb} C not readable")
            ci = get(q, "gap", "delta_full", "ci95")
            if ci is None:
                raise Missing(f"{lb} C delta_full interval withheld")
            if not isinstance(ci, list) or len(ci) != 2:
                raise NotInt("delta_full ci95 is not a two-number list")
            return (numv(ci[1], "ci95[1]") - numv(ci[0], "ci95[0]")) / 2
        for key, labs, pick in (("min_excl_on_policy", [x for x in SIX if x != ON_POLICY], min),
                                ("max_excl_on_policy", [x for x in SIX if x != ON_POLICY], max), ("max_all", list(SIX), max)):
            def f_(labs=labs, pick=pick):
                v, lb = pick((hw(x), x) for x in labs)
                return num(100 * v, 1, False), f"{g17(v)} at {lb}"
            put(f"10.halfwidth_C.{key}", f_)
        for lv in (0.01, 0.05):
            def f_(lv=lv):
                q = get(A, "pooled", "sol_a_uncut C", LEAD, "fixed", lv)
                return ("not readable", "not readable") if "reading" in q else ci_str(get(q, "gap", "delta_full"), pp)
            put(f"12.fixed.delta_full_ci.sol_a_uncut C@{lv:g}", f_)
        for ro in CR:
            for g in ("delta_full", "real_line"):
                put(f"13.{g}.{C_ASIS} {ro}", lambda ro=ro, g=g: (real_line_str if g == "real_line" else lambda q, fmt: q_str(q, "gap", g, fmt=fmt))(
                    get(A, "pooled", f"{C_ASIS} {ro}", LEAD, "joint", 0.01), pp))
        for lb in SIX:
            for ro in CR:
                L = f"{lb} {ro}"
                for pool in ("B", "S"):
                    put(f"14.after_runs.{L}.{pool}", lambda L=L, pool=pool: cnt(
                        get(A, "meta", "pooled_world", "per_label", f"{L} | {LEAD}", "sizes", pool, "after", "runs")))
                put(f"14.attack_mains.{L}", lambda L=L: cnt(get(A, "combo_weights", f"{L}|{LEAD}", "n_mains")))
        for pool in ("B", "S"):  # 15: per_run_alarm_share (writing: not ..._with_no_unit_runs); 15b: the runs it leaves out
            for lb in SIX:
                put(f"15.alarm_share.{lb} C.{pool}", lambda lb=lb, pool=pool: (lambda v: (share(v), g17(v)))(draw0(A, f"{lb} C", 0.01, pool, "per_run_alarm_share")))
                put(f"15b.zero_unit_runs.{lb} C.{pool}", lambda lb=lb, pool=pool: cnt(zero_unit_runs(A, f"{lb} C", 0.01, pool)))

        def f_():
            vs = [(draw0(A, f"{lb} C", 0.01, pool, "per_run_alarm_share"), f"{lb} C, {POOL_NAME[pool]}") for lb in SIX for pool in ("B", "S")]
            (lo, l1), (hi, l2) = min(vs), max(vs)
            return f"{share(lo)} to {share(hi)}", f"{g17(lo)} at {l1}; {g17(hi)} at {l2}"
        put("15.alarm_share.range", f_)

        def f_():
            vs = [(zero_unit_runs(A, f"{lb} C", 0.01, pool), f"{lb} C, {POOL_NAME[pool]}") for lb in SIX for pool in ("B", "S")]
            (lo, l1), (hi, l2) = min(vs), max(vs)
            return f"{lo} to {hi}", f"{lo} at {l1}; {hi} at {l2}"
        put("15b.zero_unit_runs.range", f_)
        for gname, labs in (("trained", TRAINED), ("public", PUBLIC)):  # 16: units / runs per group and pool (members identical, else STOP)
            for pool in ("B", "S"):
                put(f"16.units_per_run.{gname}.{pool}", lambda labs=labs, pool=pool: (lambda u, r: (num(u / r, 1, False), g17(u / r)))(*group_units(A, labs, pool)))

        def f_():
            vs = [(lambda u, r: (u / r, f"{g}, {POOL_NAME[pool]}"))(*group_units(A, labs, pool)) for g, labs in (("trained", TRAINED), ("public", PUBLIC))
                  for pool in ("B", "S")]
            (lo, l1), (hi, l2) = min(vs), max(vs)
            return f"about {round5(lo)} to {round5(hi)}", f"{g17(lo)} at {l1}; {g17(hi)} at {l2}"
        put("16.units_per_run.range", f_)
        for pool in POOLS:  # 17: the share of the public probes' scored calls the trained probes skip
            put(f"17.readout_share.{pool}", lambda pool=pool: (lambda v: (share(v), g17(v)))(skip_share(A, pool)))

        def f_():
            vs = [(skip_share(A, pool), POOL_NAME[pool]) for pool in POOLS]
            (lo, l1), (hi, l2) = min(vs), max(vs)
            return f"{share(lo)} to {share(hi)}", f"{g17(lo)} at {l1}; {g17(hi)} at {l2}"
        put("17.readout_share.range", f_)

    GE = recs.get("A_group_excess")
    if GE is None:
        nc["08"] = ("A carries no four-probe PANEL group (pool_reduce_n's group families); pass run_group_excess.py's record on A as "
                    "A_group_excess")
    else:
        check_tie(GE, "A_group_excess", shas.get("A"), ("meta", "record", "sha256"))
        if get(GE, "gate", "pass") is not True:
            raise SystemExit("A_group_excess: its gate did not pass (STOP)")
        mt = get(GE, "meta")
        if (mt.get("combo"), mt.get("mode"), mt.get("readout")) != (LEAD, "joint", "C") or not math.isclose(float(mt.get("level")), 0.01):
            raise SystemExit("A_group_excess: not the lead combo / joint / C / 0.01 (STOP)")
        if sorted(get(GE, "four_probe", "rows")) != sorted(f"{x} C" for x in FOUR):
            raise SystemExit("A_group_excess: its four members are not writing's four (STOP)")
        put("08.excess.four_probe|C", lambda: summ_str(get(GE, "four_probe", "mean_G_minus_mean_Y"), pp))

    GC = recs.get("A_group_catch")
    if GC is None:
        nc["09.catch_avg"] = "A carries no group catch (real_mean, sim); pass run_group_catch.py's record on A as A_group_catch"
    else:
        check_tie(GC, "A_group_catch", shas.get("A"), ("meta", "record", "sha256"))
        if get(GC, "gate", "pass") is not True:
            raise SystemExit("A_group_catch: its gate did not pass (STOP)")
        for gname, labs, key in (("six", SIX, "six"), ("four", FOUR, "auroc_subset")):
            for ro in CR:
                node = get(GC, "groups", f"{key}|{ro}")
                if sorted(get(node, "members")) != sorted(f"{x} {ro}" for x in labs) or (node.get("combo"), node.get("mode")) != (LEAD, "joint"):
                    raise SystemExit(f"A_group_catch: {key}|{ro} is not writing's {gname} at the lead combo / joint (STOP)")
                for arm in ("real_mean", "sim"):
                    put(f"09.catch_avg.{gname}|{ro}.{arm}", lambda node=node, arm=arm: summ_str(get(node, "catch", arm), pct))

    B = recs.get("B")
    if B is None:
        nc["18-19"] = "no pooled B (sweep15) record given"
    else:
        for lb in SIX:
            for ro in CR:
                L = f"{lb} {ro}"
                for mode in ("joint", "fixed"):
                    try:
                        lvs = levels(get(B, "pooled", L, LEAD, mode))
                    except Missing as e:
                        missing[f"18.{mode}.{L}"] = str(e)
                        continue
                    for lv in lvs:
                        put(f"18.delta_full.{mode}.{L}@{lv:g}", lambda L=L, mode=mode, lv=lv, lb=lb: q_str(get(B, "pooled", L, LEAD, mode, lv), "gap", "delta_full",
                                                                                                     fmt=pp2 if lb == ON_POLICY else pp))
                        put(f"18.Y.{mode}.{L}@{lv:g}", lambda L=L, mode=mode, lv=lv: q_str(get(B, "pooled", L, LEAD, mode, lv), "gap", "run2_minus_run1_at_tauB", fmt=pp))
            put(f"19.units_above_tau_sim.{lb} C@0.001", lambda lb=lb: cnt(draw0(B, f"{lb} C", 0.001, "S", "units_above")))
            put(f"19.distinct_runs_above_tau_sim.{lb} C@0.001", lambda lb=lb: cnt(distinct_runs_above(B, f"{lb} C", 0.001, "S")))
            put(f"19.real_units_above_tau_B.{lb} C@0.001", lambda lb=lb: cnt(draw0(B, f"{lb} C", 0.001, "B", "units_above")))

    R = recs.get("R37")
    if R is None:
        nc["20"] = "no R37 record given"
    else:
        for lv in R37_LEVELS:
            for lb in SIX:
                for ro in CR:
                    L = f"{lb} {ro}"
                    for mode in ("joint", "fixed"):
                        q_ = ("pooled", L, LEAD, mode, lv)
                        put(f"20.delta_full.{mode}.{L}@{lv:g}", lambda q_=q_, lb=lb: q_str(get(R, *q_), "gap", "delta_full", fmt=pp2 if lb == ON_POLICY else pp))
                        for arm in ("sim", "real_mean"):
                            put(f"20.catch.{mode}.{L}@{lv:g}.{arm}", lambda q_=q_, arm=arm: q_str(get(R, *q_), "catch", arm, fmt=pct))

                        def f_(q_=q_):
                            q = get(R, *q_)
                            rr, ra = get(q, "readable_rule37"), get(q, "readable", "all")
                            return f"rule 37 {'readable' if rr else 'not readable'}; gate {'readable' if ra else 'not readable'}", f"{rr} {ra}"
                        put(f"20.readable.{mode}.{L}@{lv:g}", f_)
                    put(f"20.distinct_runs_above_tau_sim.{L}@{lv:g}", lambda L=L, lv=lv: cnt(distinct_runs_above(R, L, lv, "S")))
    downstream(recs, shas, put, nc)  # kappa, the kappa extra, detector_vs_drop, the R² anchor, PLANT (labels by 09's rule)
    res = {"slots": out, "full": full, "missing": missing, "not_carried": nc}
    bad = [f"{k}/{s}" for k, d_ in res.items() for s, v in d_.items() if type(v) is not str]  # a63478 REQUIRED-1: strings only out
    if bad:
        raise SystemExit(f"a slot output is not a string (raw value or array): {bad[:5]} (STOP)")
    return res
