"""Revision 6s NEW keys: independent recompute from the spec text, blinded comparison with 09's record. PRELIMINARY.

Spec (never 09's code): priority-1 RULE "Append 21j ACCEPTED" (@ 579ebeb17, latest @ afd4eed43), which adopts
PROPOSED_append21j_2026-10-15.md @ 816bfdd55 §2b (X_err_strat / U_strat, per-probe between-origin D with 90% and 95%
intervals and MDD_0) and §4 (normal-approximation power); acceptance §2 (verdicts, >= 9,500 jointly finite draws);
Append 18 (strategy strata); Append 37 §1 (D_std, w = 85/90); scale-up RULE Revision 6l (catches beside gaps) and
Revision 6s (key layout and the power formulas) @ 2bf70468d. Readings are listed in RULE.md, "Revision 6s new keys".

The record written here holds key paths, match booleans, tolerances and counts only: no recomputed or target value.
"""

from __future__ import annotations

import copy
import math
import re

import numpy as np

from probe_shift import xerr_recheck as core
from probe_shift import xerr_recheck_full as fullmod
from probe_shift.xerr_recheck import COMPS, NDRAW

TARGET_SHA256 = "d0933874561627c947f67af2cc0900c50d077ed0e6752eaf93620cd035a0890b"
TOL = 1e-9  # absolute, percent units as recorded (fractions for weights, shares and power); stated before the run
PRES = {"cross_strategy": "", "with_strategy": "|with_strategy", "without_strategy": "|without_strategy"}
ORIGINS = ("S_old", "S_new")
W37 = 85 / 90  # Append 37 §1, held fixed across draws
MIX_LABEL = "equal-main-weighted estimate for the registered cohort mixture ({st} stratum)"  # spec phrase @ e6ded1811
PRIMARY_CN = "sim_minus_mean_real"
NEWPAT = re.compile(r"/(append21j_2b|append37_item1_POST_HOC)/|/combined\|with(out)?_strategy(/|$)")
INSCOPE = re.compile(r"^result/results/([RC]\|0\.0[15])/primary/")
U_MINE = fullmod.U_LABEL
FIELD_MAP = {"U_label": {U_MINE: U_MINE + " (Append 19)"},  # as in make_key_map.py (RULE.md "Label value maps")
             "note": {core.FEW: "too few eligible probes (< 2): spread undefined, no X_err"}}
FAM_FIELDS = ("complete_draws", "unstable_over_1pct_nan", "X_err", "X_err_wording", "bias_mean", "bias_ci95",
              "bias2_corrected_floored", "spread", "spread_floored", "spread_plugin_ci95_inflated_by_noise",
              "X_err_plugin_no_deconvolution", "U_approx_upper_95", "fixed_family_rms", "certification", "U_label", "note")

# target leaves NOT recomputed here (suffix after .../primary/), with the reason
NOT_RECOMPUTED = [
    (r"^append21j_2b/(note|source)$", "free text; the spec gives no verbatim string"),
    (r"/mixture_row/row$", "free text (row name); the spec gives no verbatim string"),
    (r"/between_origin_D/[^/]+/(contrast|mdd_label)$", "free text; the spec gives no verbatim string"),
    (r"/power/label$", "free text; the spec gives no verbatim string"),
    (r"/X_err_strat/[^/]+/note$", "free text (unavailable companion); the spec gives no verbatim string"),
    (r"/fold_stratified/(?!strata_label$)", "fold-stratified sensitivity: its draw scheme is not implemented by this harness"),
    (r"/fold_vs_env_95pct_widths_differ_gt10pct$", "needs the fold-stratified sensitivity (not implemented here)"),
]
# decision-free label checks: the target string must contain the spec phrase
WORDING = [
    (r"/mixture_row/label$", "equal-main-weighted estimate for the registered cohort mixture"),
    (r"/X_err_strat/[^/]+/label$", "weighted RMS of origin-specific X_err values; opposite-signed origin gaps cannot cancel"),
    (r"/U_strat_label$", "approximate bootstrap upper limit, conditional on fixed calibration, eligibility and mixture "
                         "weights; coverage not guaranteed"),
    (r"^append37_item1_POST_HOC/label$", "POST HOC"),
    (r"^append21j_2b/status$", "registered"),  # Revision 6s item 5: these are REGISTERED 21j quantities (case-insensitive)
]
VERDICT_PREFIX = (("unavailable", "decision unavailable"), ("equivalent", "equivalent"),
                  ("materially different", "materially different"), ("inconclusive", "inconclusive at this n"))


def _same_float(a, b) -> bool:
    return a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _q(x: np.ndarray, p: float) -> float:
    return 100 * float(np.quantile(x, p))


def d_stats(D: np.ndarray, final: bool, verdict: bool, withheld: bool = False) -> dict:
    """One between-origin difference (fractions per draw, draw 0 = point) on its jointly finite draws. `withheld`: the
    readout is incomplete (7b item 5), an inherited validity rule, so the 21j decision is unavailable."""
    ok = np.isfinite(D[1:])
    f = D[1:][ok]
    n = int(ok.sum())
    pt = 100 * float(D[0]) if np.isfinite(D[0]) else None
    c90 = [_q(f, 0.05), _q(f, 0.95)] if n else [None, None]
    c95 = [_q(f, 0.025), _q(f, 0.975)] if n else [None, None]
    out = {"D": pt, "ci90": c90, "ci95": c95, "jointly_finite_draws": n, "draws": NDRAW,
           "unstable_over_1pct_nan": bool(NDRAW - n > 0.01 * NDRAW),
           "ci95_excludes_0_direction_only": bool(n and (c95[0] > 0 or c95[1] < 0))}
    if not verdict:  # Append 37's Δ_std: ">= 9,500 jointly finite or unavailable" (point and intervals None)
        if n < 9500:
            out.update({"D": None, "ci90": [None, None], "ci95": [None, None], "ci95_excludes_0_direction_only": False})
        return out
    if n < 9500 or pt is None or None in c90 or None in c95 or withheld:
        cat = "unavailable"
    elif not final:
        cat = "unavailable"
    elif c90[0] > -5 and c90[1] < 5:
        cat = "equivalent"
    elif c95[0] > 5 or c95[1] < -5:
        cat = "materially different"
    else:
        cat = "inconclusive"
    se = 100 * float(np.std(f, ddof=1)) if n >= 2 else None
    out.update({"verdict_21j": ("__category__", cat), "SE_diff": se,
                "MDD_0": (1.645 + 0.842) * se if se is not None else None,
                "D_true_for_about_80pct_material_detection": 5 + (1.960 + 0.842) * se if se is not None else None})
    if se is not None and se > 0:
        det = {f"abs_D_true_{d}": _phi((d - 5 - 1.96 * se) / se) + _phi((-d - 5 - 1.96 * se) / se) for d in (6, 10)}
        out["power"] = {"equivalence_power_at_true_D_0": max(0.0, 2 * _phi(5 / se - 1.645) - 1),
                        "material_difference_detection_power": det}
    else:
        out["power"] = {"equivalence_power_at_true_D_0": None,
                        "material_difference_detection_power": {"abs_D_true_6": None, "abs_D_true_10": None}}
    return out


def mains_of(cells: list, main_of: dict) -> int:
    return len({main_of[c] for c in cells})


def draw0_mean(v: np.ndarray, cells: list, cpos: dict, main_of: dict) -> float:
    """Equal-main-weighted mean at draw 0: the mean over a main's cells, then the plain mean over mains."""
    by = {}
    for c in cells:
        by.setdefault(main_of[c], []).append(float(v[cpos[c]]))
    return float(np.mean([np.mean(x) for x in by.values()])) if by else float("nan")


def cell_mirror(cap: dict, final: bool, counts: dict) -> dict:
    """My values for the Revision 6s new keys of one readout x alpha, laid out like the target's `primary` block."""
    Mp, H, elig, sets, sub, fam, main_of = (cap[k] for k in ("Mp", "H", "elig", "sets", "sub", "fam", "main_of"))
    wh = bool(cap["withheld"])
    P = len(elig)
    est_cache = {}

    def est(cn: str, s: str):
        """(ndraw+1, P) member deltas on the shared primary draws, and the subset's cells with every member finite."""
        if (cn, s) not in est_cache:
            X = np.stack([Mp[p][cn] for p in elig]) if P else np.zeros((0, len(sub.cpos)))
            ok = [c for c in sets[s] if np.isfinite(X[:, sub.cpos[c]]).all()] if P else list(sets[s])
            est_cache[(cn, s)] = (sub.est(X, ok), ok)
        return est_cache[(cn, s)]

    def v6_sd_fields(cn: str, s: str, b: dict) -> dict:
        """09's stated conventions (4f's relay; RULE.md "v6 SD fields"): bootstrap_sd_of_plugin = 100 x the ddof-1 SD,
        over draws 1..D complete for every member, of sqrt(mean_p^2 + var_p (ddof 1)); present only where the family
        block computes X_err/U. observed_sd_among_mains = 100 x the ddof-1 SD (NaN-ignoring) of the probe's draw-0
        per-main difference of this row's comparison (within-main mean over the row's cells); None with one main."""
        e, ok = est(cn, s)
        f = {}
        if P >= 2 and "U_label" in b:
            full = np.isfinite(e).all(1)
            d = e[1:][full[1:]]
            f["bootstrap_sd_of_plugin"] = 100 * float(np.sqrt(d.mean(1) ** 2 + d.var(1, ddof=1)).std(ddof=1))
        by = {}
        for c in ok:
            by.setdefault(main_of[c], []).append(sub.cpos[c])
        f["per_probe"] = {}
        for p in elig:
            v = np.array([np.mean(Mp[p][cn][ix]) for ix in by.values()])
            f["per_probe"][p] = {"observed_sd_among_mains": 100 * float(np.nanstd(v, ddof=1)) if len(v) > 1 else None}
        return f

    for s in sets:  # internal: my deltas equal the family blocks' (same machinery), and filters agree across comparisons
        for cn in COMPS:
            e, ok = est(cn, s)
            if s in fam[cn] and P:
                counts["family_delta_checked"] += P
                counts["family_delta_differs"] += sum(
                    1 for i, p in enumerate(elig) if not _same_float(fam[cn][s]["per_probe"][p]["delta"], 100 * float(e[0, i])))
            if set(ok) != set(est(PRIMARY_CN, s)[1]):
                counts["subset_filter_differs_across_comparisons"] += 1

    out = {}
    # 1. mixture rows within each strategy stratum (Append 18 x 21j; Revision 6s item 1) and their catches (6l)
    for st in ("with_strategy", "without_strategy"):
        s = f"combined|{st}"
        for cn in COMPS:
            b = fam[cn][s]
            e, ok = est(cn, s)
            pooled = {f: b[f] for f in FAM_FIELDS if f in b}
            for f, m in FIELD_MAP.items():
                if isinstance(pooled.get(f), str):
                    pooled[f] = m.get(pooled[f], pooled[f])
            pooled["per_probe"] = {}
            for i, p in enumerate(elig):
                x = e[1:, i][np.isfinite(e[1:, i])]
                pp = b["per_probe"][p]
                pooled["per_probe"][p] = {"delta": pp["delta"], "ci95": pp["ci95"], "U_abs_approx": pp["U_abs_approx"],
                                          "discordant_mains": pp["discordant_tasks"],
                                          "ci90": [_q(x, 0.05), _q(x, 0.95)] if len(x) else [None, None]}
            sd = v6_sd_fields(cn, s, b)
            for p, x in sd.pop("per_probe").items():  # merge per probe (never replace the per_probe dict)
                pooled["per_probe"][p].update(x)
            pooled.update(sd)
            # spec phrase @ e6ded1811 (priority-1 RULE, "Spec phrase: the strata_label on the 21j within-strategy
            # mixture rows"): exact, on both the pooled and the fold_stratified blocks
            pooled["strata_label"] = MIX_LABEL.format(st=st)
            out.setdefault(cn, {})[s] = {"pooled": pooled, "mains": b["n_mains"], "cells": b["n_cells"],
                                        "under_30_mains": b["n_mains"] < 30,
                                        "fold_stratified": {"strata_label": MIX_LABEL.format(st=st)}}
    for cn in COMPS:  # control (OLD keys, never compared before): the two v6 SD fields on the pre-existing rows
        for s in ("combined", "S_old", "S_old|with_strategy", "S_old|without_strategy", "S_new", "S_new|with_strategy",
                  "S_new|without_strategy"):
            out.setdefault(cn, {})[s] = {"pooled": v6_sd_fields(cn, s, fam[cn][s])}
    out["catch_percent"] = {}
    for s in sets:
        if s.startswith("S_new|same_launch"):
            continue
        _, ok = est(PRIMARY_CN, s)
        blk = {}
        for key, arms in (("catch_run1", ("run1",)), ("catch_run2", ("run2",)), ("catch_mean_real", ("run1", "run2")),
                          ("catch_sim", ("sim",))):
            per = {p: 100 * float(np.mean([draw0_mean(H[p][a], ok, sub.cpos, main_of) for a in arms])) for p in elig}
            per["family_mean"] = float(np.mean([per[p] for p in elig])) if elig else None
            blk[key] = per
        out["catch_percent"][s] = blk

    # 2. append21j_2b per presentation
    a2 = {}
    for pres, suf in PRES.items():
        mix = f"combined{suf}"
        _, okm = est(PRIMARY_CN, mix)
        nm = {o: mains_of([c for c in okm if c in set(sets[o + suf])], main_of) for o in ORIGINS}
        # point weights: each main 1/N, split across origins by its filtered cells (origins share no mains here)
        share = {o: 0.0 for o in ORIGINS}
        mains_all = {main_of[c] for c in okm}
        for m in mains_all:
            cs = [c for c in okm if main_of[c] == m]
            for o in ORIGINS:
                share[o] += sum(1 for c in cs if c in set(sets[o + suf])) / len(cs) / len(mains_all)
        if not mains_all:
            share = {o: None for o in ORIGINS}
        blk = {"mixture_row": {"origins": list(ORIGINS), "estimation_mains": nm, "effective_origin_contributions": share},
               "X_err_strat": {}, "between_origin_D": {}}
        for cn in COMPS:
            es = {o: est(cn, o + suf) for o in ORIGINS}
            n_s = {o: mains_of(es[o][1], main_of) for o in ORIGINS}
            tot = sum(n_s.values())
            w = {o: (n_s[o] / tot if tot else None) for o in ORIGINS}
            xo = {o: fam[cn][o + suf].get("X_err") for o in ORIGINS}
            row = {"weights": w, "estimation_mains": n_s, "X_err_by_origin": xo}
            if any(v is None for v in xo.values()) or P < 2 or not tot:
                row.update({"X_err_strat": None, "U_strat": None})
            else:
                row["X_err_strat"] = math.sqrt(sum(w[o] * xo[o] ** 2 for o in ORIGINS))
                joint = np.ones(NDRAW + 1, dtype=bool)
                for o in ORIGINS:
                    joint &= np.isfinite(es[o][0]).all(1)
                jd = joint[1:]
                plug2 = sum(w[o] * (es[o][0][1:][jd].mean(1) ** 2 + es[o][0][1:][jd].var(1, ddof=1)) for o in ORIGINS)
                n = int(jd.sum())
                row.update({"U_strat": 100 * float(np.quantile(np.sqrt(plug2), 0.95)) if n else None,
                            "complete_draws_joint": n, "unstable_over_1pct_nan": bool(NDRAW - n > 0.01 * NDRAW),
                            "certification": core.CERT_OK if n >= 0.95 * NDRAW else core.CERT_NO})
            blk["X_err_strat"][cn] = row
        ea, oka = est(PRIMARY_CN, "S_new" + suf)
        eb, okb = est(PRIMARY_CN, "S_old" + suf)
        blk["between_origin_D"]["S_new-S_old"] = {
            "mains": [mains_of(oka, main_of), mains_of(okb, main_of)],
            "per_probe": {p: d_stats(ea[:, i] - eb[:, i], final, True, wh) for i, p in enumerate(elig)}}
        a2[pres] = blk
    out["append21j_2b"] = a2

    # 3. Append 37 §1 (POST HOC): strategy-standardized S_new gap and D_std, on the shared primary draws
    ew, okw = est(PRIMARY_CN, "S_new|with_strategy")
    eo, oko = est(PRIMARY_CN, "S_new|without_strategy")
    e0, ok0 = est(PRIMARY_CN, "S_old")
    _, ok0w = est(PRIMARY_CN, "S_old|with_strategy")
    std = W37 * ew + (1 - W37) * eo
    out["append37_item1_POST_HOC"] = {
        "w_fixed_append37": W37,
        "this_rows_S_old_with_strategy_share": (mains_of(ok0w, main_of) / mains_of(ok0, main_of)) if ok0 else None,
        "mains": {"S_old": mains_of(ok0, main_of), "S_old|with_strategy": mains_of(ok0w, main_of),
                  "S_new|with_strategy": mains_of(okw, main_of), "S_new|without_strategy": mains_of(oko, main_of)},
        "Delta_S_new_std": {p: d_stats(std[:, i], final, False) for i, p in enumerate(elig)},
        "D_std": {p: d_stats(std[:, i] - e0[:, i], final, True, wh) for i, p in enumerate(elig)}}
    return out


def flat(x, path: str, out: dict) -> dict:
    if isinstance(x, tuple) and len(x) == 2 and x[0] == "__category__":
        out[path] = x
    elif isinstance(x, dict):
        for k, v in x.items():
            flat(v, f"{path}/{k}", out)
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            flat(v, f"{path}[{i}]", out)
    else:
        out[path] = x
    return out


def _kind(x) -> str:
    if x is None:
        return "none"
    if isinstance(x, bool):
        return "bool"
    if isinstance(x, (int, float)):
        return "number"
    return "str" if isinstance(x, str) else type(x).__name__


def judge(mine, theirs, suffix: str) -> tuple[bool, str]:
    """(match, tolerance_used) for one target leaf; never returns a value."""
    for pat, phrase in WORDING:
        if re.search(pat, suffix):
            if not isinstance(theirs, str):
                return False, "wording: contains the spec phrase"
            if phrase == "registered":  # whole word (not un-/de-/pre-registered); no negation in the 3 words before
                hits = list(re.finditer(r"(?<![a-z-])registered\b", theirs.lower()))
                neg = re.compile(r"\b(not|no|never|non|isn't|aren't|wasn't|weren't|nor)\b")
                ok = bool(hits) and all(not neg.search(" ".join(theirs.lower()[:h.start()].split()[-3:])) for h in hits)
                return ok, "wording: the word 'registered' (case-insensitive), not negated in the 3 words before it"
            return phrase in theirs, "wording: contains the spec phrase"
    if isinstance(mine, tuple):  # a 21j verdict category
        cat = next((c for c, pre in VERDICT_PREFIX if isinstance(theirs, str) and theirs.startswith(pre)), "unrecognized")
        return cat == mine[1], "category (spec verdict phrase as prefix)"
    if mine is _MISS:
        return False, "exact (key not produced here)"
    km, kt = _kind(mine), _kind(theirs)
    if km != kt:
        return False, "exact (type)"
    if km in ("none", "bool", "str"):
        return mine == theirs, "exact"
    ia = isinstance(mine, int) and not isinstance(mine, bool)
    it = isinstance(theirs, int) and not isinstance(theirs, bool)
    if ia or it:  # an integer quantity: both sides must be int, and equal
        return ia and it and mine == theirs, "exact (integer; int/float type must agree)"
    a, b = float(mine), float(theirs)
    if not (math.isfinite(a) and math.isfinite(b)):
        return (math.isnan(a) and math.isnan(b)) or a == b, "exact (non-finite)"
    return abs(a - b) <= TOL, f"abs {TOL:g}"


_MISS = object()

PRIORITY = [f"result/results/R|0.05/primary/{s}" for s in (
    *(f"append21j_2b/with_strategy/between_origin_D/S_new-S_old/per_probe/sol_a/{f}"
      for f in ("D", "ci90[0]", "ci90[1]", "ci95[0]", "ci95[1]", "verdict_21j")),
    "append21j_2b/cross_strategy/X_err_strat/sim_minus_mean_real/X_err_strat",
    "append21j_2b/cross_strategy/X_err_strat/sim_minus_mean_real/U_strat",
    "sim_minus_mean_real/combined|with_strategy/pooled/X_err",
    "sim_minus_mean_real/combined|with_strategy/pooled/U_approx_upper_95")]


def recheck6s(inp, target_meta: dict, note) -> dict:
    if target_meta.get("sha256") != TARGET_SHA256:
        raise core.Stop("target is not 09's Revision 6s FINAL record (sha256 pin)")
    inp.embedded["__diag__"] = {"json": True}
    ctx = fullmod.full(inp, target_meta, note)  # the selftest gate and every input read, as in full mode
    if not ctx["final"]:
        raise core.Stop("target is not a final-mode record")
    if not (ctx["override"] or {}).get("key_map"):
        raise core.Stop("--override with the 6r key map is required (the old-key gate)")
    cap = {}
    res = fullmod.analyze(ctx["b"], ctx["keep"], ctx["units"], ctx["cells"], ctx["cohorts"], ctx["pools"], ctx["reg"],
                          ctx["rt"], ctx["missing"], ctx["final"], ctx["il_keylist"], note, capture=cap)
    tgt = ctx["tgt"]
    counts = {"family_delta_checked": 0, "family_delta_differs": 0, "subset_filter_differs_across_comparisons": 0}
    mine = {}
    for cell, c in cap.items():
        flat(cell_mirror(c, ctx["final"], counts), f"result/results/{cell}/primary", mine)
    note(f"6s mirror built for {len(cap)} cells")

    # the target's new keys (pattern = exactly the 6r -> 6s new leaves; checked locally, RULE.md)
    tl = {}
    flat(tgt["result"], "result", tl)
    new = {p: v for p, v in tl.items() if NEWPAT.search(p)}
    rows, not_rec, out_scope, control = [], [], {}, []
    for p in sorted(new):
        m = INSCOPE.match(p)
        if not m:
            mm = re.match(r"^result/(results|expansion)/[RC]\|0\.0[15]/([^/]+)", p)
            blk = "other" if not mm else ("expansion" if mm.group(1) == "expansion" else mm.group(2))
            out_scope[blk] = out_scope.get(blk, 0) + 1
            continue
        suf = p[m.end():]
        why = next((r for pat, r in NOT_RECOMPUTED if re.search(pat, suf)), None)
        if m.group(1) not in cap:
            why = "primary calibration undefined in this harness for this readout x alpha"
        if why:
            not_rec.append({"key_path": p, "reason": why})
            continue
        ok, tol = judge(mine.get(p, _MISS), new[p], suf)
        rows.append({"key_path": p, "match": bool(ok), "tolerance_used": tol})
    # control (OLD keys, never compared by the 6r key map): catches and the two v6 SD fields of the pre-existing rows
    sdpat = re.compile(r"^(sim_minus_mean_real|sim_minus_run1|sim_minus_run2|reexecution_run2_minus_run1)/[^/]+/pooled/"
                       r"(bootstrap_sd_of_plugin|per_probe/[^/]+/observed_sd_among_mains)$")
    for p, v in sorted(tl.items()):
        m = INSCOPE.match(p)
        if m and not NEWPAT.search(p) and m.group(1) in cap and ("/catch_percent/" in p or sdpat.search(p[m.end():])):
            ok, tol = judge(mine.get(p, _MISS), v, p[m.end():])
            control.append({"key_path": p, "match": bool(ok), "tolerance_used": tol})
    for p in sorted(mine):  # an old-row SD field produced here must exist in the target
        m = INSCOPE.match(p)
        if m and not NEWPAT.search(p) and sdpat.search(p[m.end():]) and p not in tl:
            control.append({"key_path": p, "match": False, "tolerance_used": "exact (key presence: produced here, absent in target)"})
    # the fold-stratified blocks of the new rows must NOT carry observed_sd_among_mains (09's stated convention)
    fold_abs = []
    for cell, c in cap.items():
        for cn in COMPS:
            for st in ("with_strategy", "without_strategy"):
                for p in c["elig"]:
                    base = f"result/results/{cell}/primary/{cn}/combined|{st}/fold_stratified/per_probe/{p}/"
                    blk = fullmod.get_path(tgt, base)  # the member's own dict (any value type under the key fails)
                    fold_abs.append({"key_path": base + "observed_sd_among_mains",
                                     "match": isinstance(blk, dict) and "delta" in blk and "observed_sd_among_mains" not in blk,
                                     "tolerance_used": "exact (key absent; the fold-stratified member block exists)"})
    in_scope_mine = {p for p in mine if NEWPAT.search(p)}
    extra = sorted(in_scope_mine - set(new))
    rows += [{"key_path": p, "match": False, "tolerance_used": "exact (key presence: produced here, absent in target)"}
             for p in extra]
    missing_cells = sorted({f"{rd}|{a}" for rd in ("R", "C") for a in ("0.01", "0.05")} - set(cap))
    required = []  # spec-required label keys that the mirror does not produce (it never produces label text)
    for cell in cap:
        P0 = f"result/results/{cell}/primary/"
        required += [P0 + "append21j_2b/status", P0 + "append37_item1_POST_HOC/label"]
        for pres in PRES:
            required.append(P0 + f"append21j_2b/{pres}/mixture_row/label")
            for cn in COMPS:
                X0 = P0 + f"append21j_2b/{pres}/X_err_strat/{cn}/"
                required.append(X0 + "label")
                if mine.get(X0 + "U_strat") is not None:
                    required.append(X0 + "U_strat_label")
    absent_req = sorted(p for p in required if p not in new)
    rows += [{"key_path": p, "match": False, "tolerance_used": "exact (required key absent in target)"}
             for p in absent_req]

    # gate: the 6r key map on the 6s record (old keys); its two combined-strategy excuses no longer hold, since the
    # target now carries those rows (compared above), so only their absence preconditions are dropped
    ov = copy.deepcopy(ctx["override"])
    for e in ov.get("key_map", []):
        if "excuse" in e and e["mine"].endswith(("/combined|with_strategy", "/combined|without_strategy")):
            e["checks"] = []
    mine_old = {"cohort": {st: {"cohort_cells_sha256": core.sha256("\n".join(sorted(ctx["cohorts"][st])).encode())}
                           for st in ("S_new", "S_newmains")}, **res}
    c_old = fullmod.compare_with_target(mine_old, tgt, ov, lambda *_: None)
    sm = c_old["summary"]
    gate = {"verdict": c_old["verdict"], "compared": sm["compared"], "n_mismatch": sm["n_mismatch"],
            "mismatch_paths": sm["mismatch_paths"], "unmapped": len(c_old["unmapped"]),
            "excuse_precondition_failures": len(c_old["excuse_precondition_failures"]),
            "n_required_leaves_not_compared": c_old["n_required_leaves_not_compared"],
            "required_not_computed_here": c_old["required_not_computed_here"]}

    mism = [r["key_path"] for r in rows if not r["match"]]
    pri = [next((r for r in rows if r["key_path"] == k), {"key_path": k, "match": None, "tolerance_used": "not compared"})
           for k in PRIORITY]
    groups = {}
    for r in rows:
        g = re.sub(r"^result/results/[RC]\|0\.0[15]/primary/", "", r["key_path"]).split("/")
        g = g[0] if g[0] in ("append21j_2b", "append37_item1_POST_HOC", "catch_percent") else "family rows " + g[1]
        t = groups.setdefault(g, {"compared": 0, "matched": 0, "mismatched": 0})
        t["compared"] += 1
        t["matched" if r["match"] else "mismatched"] += 1
    ctrl_bad = [r["key_path"] for r in control if not r["match"]]
    n_num_unrec = sum(1 for r in not_rec if r["reason"].startswith("numeric"))
    fold_bad = [r["key_path"] for r in fold_abs if not r["match"]]
    ok_new = (rows and not mism and not missing_cells and not counts["family_delta_differs"]
              and not counts["subset_filter_differs_across_comparisons"] and fold_abs and not fold_bad)
    qual = f" ON RECOMPUTED KEYS ({n_num_unrec} numeric leaves not recomputed, listed)" if n_num_unrec else ""
    v_new = ("MATCH" + qual) if ok_new else "NOT MATCH"
    verdict = (("MATCH" + qual) if ok_new and gate["verdict"] == "MATCH" and control and not ctrl_bad
               else "NOT MATCH")
    note(f"6s: {len(rows)} compared, {len(mism)} mismatched, {len(not_rec)} not recomputed; new keys {v_new}; "
         f"old-key gate {gate['verdict']}; control mismatches {len(ctrl_bad)}; overall {verdict}")
    return {
        "label": "PRELIMINARY; 4f harness, Revision 6s new-key recheck: verdicts, counts and key paths only (no values)",
        "verdict": verdict, "verdict_new_keys": v_new,
        "verdict_rule": ("verdict_new_keys = MATCH iff every compared new-key leaf matches (including key presence both "
                         "ways and the spec-required label keys), every readout x alpha was recomputed, and the internal "
                         "checks are clean, and observed_sd_among_mains is absent from every new fold-stratified member "
                         "block; it is qualified 'ON RECOMPUTED KEYS' when numeric leaves were not recomputed; "
                         "verdict = the same, and also requires the old-key gate MATCH and every control catch to match"),
        "readouts_not_recomputed": missing_cells, "priority_keys": pri,
        "tolerances": {"float": f"absolute {TOL:g} (percent units as recorded; fractions for weights, shares, power)",
                       "draw_based": f"exact registered draws (seed [20261014, 1], 10,000), judged at absolute {TOL:g}",
                       "integers_booleans_strings_key_presence": "exact",
                       "verdict_21j": "category: the target string starts with the spec phrase of my category",
                       "labels": "wording: the target string contains the spec phrase (listed in RULE.md)"},
        "totals": {"target_new_leaves": len(new),
                   "in_scope_target_leaves": len(rows) - len(extra) - len(absent_req) + len(not_rec),
                   "keys_produced_here_absent_in_target": len(extra), "required_keys_absent_in_target": len(absent_req),
                   "numeric_leaves_not_recomputed": n_num_unrec, "compared": len(rows),
                   "by_field_v6_sd": {f: {"compared": sum(1 for r in rows if r["key_path"].endswith(f)),
                                          "matched": sum(1 for r in rows if r["key_path"].endswith(f) and r["match"])}
                                      for f in ("bootstrap_sd_of_plugin", "observed_sd_among_mains", "pooled/strata_label",
                                                "fold_stratified/strata_label")},
                   "matched": len(rows) - len(mism), "mismatched": len(mism), "not_recomputed": len(not_rec),
                   "out_of_scope_new_leaves": sum(out_scope.values()), "by_group": groups},
        "mismatched_key_paths": mism, "not_recomputed": not_rec,
        "out_of_scope_new_leaves_by_block": {"reason": "sensitivities and the expansion family are not recomputed by "
                                             "this harness (README, 'Not recomputed')", "counts": out_scope},
        "produced_here_absent_in_target": extra,
        "control_old_keys": {"compared": len(control), "matched": sum(r["match"] for r in control),
                             "by_kind": {k: {"compared": sum(1 for r in control if f(r["key_path"])),
                                             "matched": sum(1 for r in control if f(r["key_path"]) and r["match"])}
                                         for k, f in (("catch_percent", lambda x: "/catch_percent/" in x),
                                                      ("bootstrap_sd_of_plugin", lambda x: x.endswith("bootstrap_sd_of_plugin")),
                                                      ("observed_sd_among_mains", lambda x: x.endswith("observed_sd_among_mains")))},
                             "mismatched_key_paths": ctrl_bad},
        "fold_stratified_observed_sd_absent": {"compared": len(fold_abs), "matched": len(fold_abs) - len(fold_bad),
                                               "mismatched_key_paths": fold_bad},
        "gate_old_keys_6r_map_on_6s": gate, "internal_checks": counts,
        "rows": rows,
    }
