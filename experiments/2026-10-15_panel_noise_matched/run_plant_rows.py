"""Planted check (a) per probe x readout (09 RULE item 24 @ aa58468f9; 2e for the writing session's slot #7). For EACH lead probe and EACH readout (C, R):
a verdict on that probe's run-mean Delta_full, comparator (A), from Stage A's planted pooled arrays (adds (a):
<label>__<combo>__<mode>__<f>__pooled__gap_delta_full_plant_<+-d>), i.e. the same plant (a per-row constant solved at draw 0), draws and deltas (+-3, +-5,
+-8) as S and the excess. DETECTED iff |(planted - unplanted) - delta| <= 0.5|delta| AND the planted 95% interval excludes the unplanted point (points at
draw 0); a move outside that range is flagged "mechanism mismatch". Per-delta detected counts by script.

Gates (fail closed): input shas; the record's window and plant deltas; every unplanted array's summary equal to the record's pooled delta_full (and to a
base record's, when given); each planted array's draw-0 move equal to the D-weighted achieved shift the record states per origin; a planted array absent
only where a member's arm is not "ok"; the mean of the probes' planted arrays equal to the record's coheadline_signed_plant S, whose rows are the probes.
synthetic_TEST plants a constant (factor x delta) on a real record's arrays and writes the matching record fields in the container (labelled TEST).

  PYTHONPATH=src modal run experiments/2026-10-15_panel_noise_matched/run_plant_rows.py --cfg <config name>
"""

import hashlib
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
VOL = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13"
APP = "probe-shift_aswin-surya_sub-e2esim_pooledplant_kill-2026-11-16"  # E2E port for the pooled PLANT verdicts (2e 10-17); the PANEL fork ran sub-panelnm
DELTAS = [-8, -5, -3, 3, 5, 8]  # 09 RULE item 24
DELTAS_12 = [-12, -8, -5, -3, 3, 5, 8, 12]  # Aswin "all by 12" (via 0d / 2e, 10-16): +-12 added, pre-stated in the config NOTE before any full-length planted record
RULE47 = {"rule": "09 RULE item 47 @ 1ce4a94c8 (experiments/2026-10-14_priority1_spec_09/RULE.md)",
          "aggregate": ("the six-probe aggregate reports negative-shift verdicts only (delta = -3 / -5 / -8 / -12; the intervals stay two-sided, only the plant direction is "
                        "restricted); the positive side names the members unattainable at the ceiling, never an average over the attainable subset"),
          "per_probe": "per-probe x readout verdicts (item 24) wherever attainable, positive delta included; 'unavailable' where not; never combined into subset means",
          "caption": "the six-probe planted check tests detection of a simulator that UNDER-catches. Over-catching is testable only per probe, where the ceiling allows."}
RULE47_POS = "positive shifts unattainable (ceiling) for >= 1 member"
POOLED_Y1_DEF = "run1@B on every set's pooled row (pooled calibration: no B2, item 45(1)); Y = |run2@B - run1@B| at the pooled tau_B"  # Stage A @ 8a15fb10f, verbatim


def deltas_of(cfg):
    """The config's planted deltas: exactly 09's +-3/5/8 (the default) or the +-12 extension; anything else is not pre-stated."""
    d_ = sorted(cfg.get("deltas", DELTAS))
    if d_ not in (DELTAS, DELTAS_12):
        raise SystemExit(f"STOP: deltas {d_} are neither {DELTAS} nor {DELTAS_12}")
    return d_
LABEL = "pipeline test on fixed data (planted at draw 0), not calibrated power"
RULE = ("09 RULE item 24 @ aa58468f9 (standing rule @ 43e884177 sec. 2): DETECTED iff |(planted - unplanted) - delta| <= 0.5|delta| AND the planted 95% "
        "interval excludes the unplanted point (points at draw 0); a move outside that range is flagged 'mechanism mismatch'")
POOL_STEP = "plant_rows (Table 6 verdicts)"  # 09's label ruling (10-17): the step named in route C's composed pool label
SIDECAR_PTR = r"git:([0-9a-f]{40}):([A-Za-z0-9_./-]+\.json)@([0-9a-f]{64})"  # A's LABEL SIDECAR pointer (SIMPROBE's sidecar_bytes form)
COMPARATOR = "comparator (A): the run-mean Delta_full (sim - mean of run 1 and run 2); labelled (A) even when the all-runs comparator (B) leads (RULE item 24)"
TOL = 1e-12

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3").add_local_python_source("probe_shift")
SOL = modal.Volume.from_name(VOL, version=2)


def npzkey(*parts):
    """507-prep's npz key v2: parts joined by '__'; inside a part '@' -> '_at_', '|' -> '_', ' ' -> '_', parentheses dropped; no part may hold '__'."""
    out = []
    for p_ in parts:
        q = str(p_).replace("@", "_at_").replace("|", "_").replace(" ", "_").replace("(", "").replace(")", "")
        if "__" in q or not q:
            raise SystemExit(f"npz key part {p_!r} would be ambiguous")
        out.append(q)
    return "__".join(out)


def okey(cfg, o):
    """E2E port: an origin's npz key part. The pooled build names per-origin arrays by the origin WITHOUT its " [pooled: <combo>]" suffix
    (run_peraction311 @ b91071dfe l.3223/3231: o_.split(" [pooled: ", 1)[0]); the record's members carry the suffix."""
    return o.split(" [pooled: ", 1)[0] if cfg.get("pooled_world") else o


def jget(d, path):
    for p in path:
        d = d[p]
    return d


def fin(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def summ_ok(a):
    """A well-formed summary: finite point, integer draws / finite_draws, and ci95 either exactly two finite ordered ends or None with a 'withheld' reason."""
    if not (isinstance(a, dict) and fin(a.get("point")) and isinstance(a.get("draws"), int) and isinstance(a.get("finite_draws"), int)):
        return False
    ci = a.get("ci95")
    return (ci is None and bool(a.get("withheld"))) or (isinstance(ci, list) and len(ci) == 2 and all(fin(x) for x in ci) and ci[0] <= ci[1])


def same_summ(a, b):
    """Two well-formed summaries {point, ci95, draws, finite_draws} equal within TOL (rate units) (Codex plant-rows r1)."""
    if not (summ_ok(a) and summ_ok(b)) or a["draws"] != b["draws"] or a["finite_draws"] != b["finite_draws"] or (a["ci95"] is None) != (b["ci95"] is None):
        return False
    return abs(a["point"] - b["point"]) <= TOL and all(abs(x - y) <= TOL for x, y in zip(a["ci95"] or [], b["ci95"] or []))


def levels_of(points):
    """The planted record's levels (meta.plant.points, e.g. [0.01, 0.05]) as the record's level keys (f"{f:g}": "0.01"); None if malformed."""
    if not (isinstance(points, list) and points and all(fin(x) and x > 0 for x in points)):
        return None
    ks = [f"{float(x):g}" for x in points]
    if len(set(ks)) != len(ks) or any(float(k) != float(x) for k, x in zip(ks, points)):  # each key must name its level exactly (0.010000001 is not "0.01"; Codex r3)
        return None
    return ks


def arm_ok(s_):
    """A member's planted-arm record: status 'ok' with finite c and achieved_points, or 'unattainable'; anything else is malformed (Codex plant-rows r1)."""
    return isinstance(s_, dict) and ((s_.get("status") == "ok" and fin(s_.get("c")) and fin(s_.get("achieved_points"))) or s_.get("status") == "unattainable")


AGG_RULE = ("09 item 35: S = the six-probe mean of the run-mean Delta_full, item 24's delta rule (|(planted - unplanted) - delta| <= 0.5|delta| AND the planted 95% "
            "interval excludes the unplanted point); the EXCESS (mean G - mean Y; the plant shifts each probe's signed gaps, G = 0.5|sim - run1| + 0.5|sim - run2| is "
            "recomputed, Y = |Delta_rr| is unchanged): Delta_exp = excess(planted, draw 0) - excess(unplanted, draw 0); DETECTED iff |bootstrap-mean move - Delta_exp| <= "
            "0.5|Delta_exp| AND the planted 95% interval excludes the unplanted point; |Delta_exp| < 0.5 points -> 'not testable at this delta'")


def unattainable_rows(rec, lb, cb, f, tag):
    """How many of the row's combo members state the sim arm 'unattainable' at this delta and level (rows.<label>.<origin>.plant.solve_draw0.<f>.sim.arms.<tag>)."""
    mem = ((((rec.get("pooled") or {}).get(lb) or {}).get(cb) or {}).get("members")) or []
    return sum(((((((rec.get("rows") or {}).get(lb) or {}).get(o) or {}).get("plant") or {}).get("solve_draw0", {}).get(f, {}).get("sim", {}).get("arms", {}).get(tag) or {})
                .get("status") == "unattainable") for o in mem)


def aggregate(cfg, rec, base, z, keys, DS, fkeys, g):
    """09 item 35: the aggregate planted check on the headline combo from the SAME planted pooled arrays: the six-probe signed S (item 24's delta rule) and
    the registered excess (Delta_exp rule), per level, readout and signed delta, with Delta_exp printed beside each verdict. Gates: each probe's
    0.5 (sim - run1 + sim - run2) equals its unplanted Delta_full; the unplanted S and excess equal Stage A's coheadline_signed S and panel_groups excess
    of the UNPLANTED base record (a PLANT record withholds its panel_groups: Stage A @ 20c6fbd20 run_peraction311.py:3021-3022), read by the explicit
    group names in the config; an absent base record or key is a gate failure, never a fallback."""
    import numpy as np

    from probe_shift.fpr_core import summ

    A = cfg["aggregate"]
    cb, md, out = cfg["combo"], cfg["mode"], {}
    YG = "gap_" + (cfg["pooled_world"]["y_gap"] if cfg.get("pooled_world") else "delta_rr")  # pooled (RULE 45.1): the pooled tau_B yardstick, explicit (main() requires it)
    if base is None:
        g(False, check="aggregate: needs the unplanted base record (the planted record's panel_groups are withheld)")
        return {"rule": AGG_RULE, "unavailable": "no base record"}
    pc_ = lambda s_: None if s_ is None else {"point": 100 * s_["point"], "ci95": [100 * x for x in s_["ci95"]] if s_.get("ci95") else None}
    for f in fkeys:
        for ro in cfg["readouts"]:
            labs = [f"{p} {ro}" for p in cfg["probes"]]
            ar = {}
            for lb in labs:
                for nm in ("gap_delta_full", "gap_sim_minus_run1", "gap_sim_minus_run2", YG):
                    k_ = npzkey(lb, cb, md, f, "pooled", nm)
                    ar[(lb, nm)] = np.asarray(z[k_], float) if k_ in keys else None
            rd_ = all(((((rec.get("pooled") or {}).get(lb) or {}).get(cb) or {}).get(md, {}).get(f, {}).get("readable") or {}).get("all") is True for lb in labs)
            if any(v is None for v in ar.values()) or not rd_:
                g(False, f=f, readout=ro, check="aggregate: every probe's pooled gaps present and the probe readable")
                continue
            for lb in labs:
                g(np.allclose(0.5 * (ar[(lb, "gap_sim_minus_run1")] + ar[(lb, "gap_sim_minus_run2")]), ar[(lb, "gap_delta_full")], rtol=0, atol=1e-12, equal_nan=True),
                  row=lb, f=f, check="aggregate: 0.5 (sim - run1 + sim - run2) == the unplanted Delta_full")
            Y = np.mean([np.abs(ar[(lb, YG)]) for lb in labs], 0)
            S0 = np.mean([ar[(lb, "gap_delta_full")] for lb in labs], 0)
            X0 = np.mean([0.5 * np.abs(ar[(lb, "gap_sim_minus_run1")]) + 0.5 * np.abs(ar[(lb, "gap_sim_minus_run2")]) for lb in labs], 0) - Y
            try:
                sa_ = jget(base, ["coheadline_signed", cb, md, f, f"{cfg['coheadline_group']}|{ro}", "S"])
                xa_ = jget(base, ["panel_groups", cb, "pooled", md, f, f"{A['group']}|{ro}", "primary", "mean_G_minus_mean_Y"])
            except (KeyError, TypeError):
                sa_ = xa_ = None
            g(same_summ(summ(S0), sa_), f=f, readout=ro, check="aggregate: unplanted six-probe S == Stage A's coheadline_signed S")
            g(same_summ(summ(X0), xa_), f=f, readout=ro, check=f"aggregate: unplanted excess == Stage A's panel_groups {A['group']} excess")
            s0, x0 = summ(S0), summ(X0)
            for d in DS:
                tag = f"{d:+d}"
                kp = [npzkey(lb, cb, md, f, "pooled", f"gap_delta_full_plant_{tag}") for lb in labs]
                un_ = {lb: n_ for lb in labs if (n_ := unattainable_rows(rec, lb, cb, f, tag))}  # member arm-rows marked 'unattainable' at this delta, by probe
                if d > 0:  # RULE 47 (1) @ 1ce4a94c8: no six-probe verdict for a positive shift; name the ceiling members, never average the attainable subset
                    e_ = ({"verdict": RULE47_POS, "unattainable_members": sorted(un_), "unattainable_arm_rows": un_, "unattainable_arm_rows_total": sum(un_.values())} if un_ else
                          {"verdict": "not reported (RULE 47: the six-probe aggregate reports negative-shift verdicts only)"})
                    out.setdefault(f, {}).setdefault(ro, {})[tag] = {"S": e_, "excess": e_}
                    continue
                if any(k_ not in keys for k_ in kp):  # a probe's arm is unattainable (gated per row above): unavailable, never filled in
                    why_ = ("unattainable (floor) in at least one member row; never averaged over a subset of the probes (not a failure)")
                    out.setdefault(f, {}).setdefault(ro, {})[tag] = {"S": {"verdict": "unavailable", "reason": why_, "unattainable_arm_rows": un_},
                                                                     "excess": {"verdict": "unavailable", "reason": why_, "unattainable_arm_rows": un_}}
                    continue
                dl = {lb: np.asarray(z[k_], float) - ar[(lb, "gap_delta_full")] for lb, k_ in zip(labs, kp)}
                S1 = np.mean([ar[(lb, "gap_delta_full")] + dl[lb] for lb in labs], 0)
                X1 = np.mean([0.5 * np.abs(ar[(lb, "gap_sim_minus_run1")] + dl[lb]) + 0.5 * np.abs(ar[(lb, "gap_sim_minus_run2")] + dl[lb]) for lb in labs], 0) - Y
                e_ = {}
                for nm, P0, P1, s_un in (("S", S0, S1, s0), ("excess", X0, X1, x0)):
                    sp = summ(P1)
                    u, p_ = 100 * s_un["point"], 100 * sp["point"]
                    dexp = p_ - u
                    ci = [100 * x for x in sp["ci95"]] if sp.get("ci95") else None
                    mv = (P1 - P0)[1:]
                    boot = 100 * float(np.mean(mv[np.isfinite(mv)])) if np.isfinite(mv).any() else None
                    sh_ = summ(P1 - P0)  # E2E port (2e 10-17, writing's Table 6): the shift planted - unplanted, paired by draw, and its 95% interval
                    v_ = {"unplanted_point": u, "planted": pc_(sp), "delta_exp_points": dexp, "bootstrap_mean_move_points": boot, "shift": pc_(sh_)}
                    if ci is None or boot is None:
                        v_["verdict"] = "unavailable"
                        v_["reason"] = "planted interval withheld or no finite draws"
                    elif nm == "excess" and abs(dexp) < 0.5:
                        v_["verdict"] = "not testable at this delta"
                    else:
                        in_ = abs(dexp - d) <= 0.5 * abs(d) if nm == "S" else abs(boot - dexp) <= 0.5 * abs(dexp)
                        ex_ = not (ci[0] <= u <= ci[1])
                        v_ |= {"move_in_range": bool(in_), "interval_excludes_unplanted": bool(ex_), "verdict": "detected" if (in_ and ex_) else "not detected",
                               "mechanism_mismatch": None if in_ else "mechanism mismatch"}
                    e_[nm] = v_
                out.setdefault(f, {}).setdefault(ro, {})[tag] = e_
    rule_ = (AGG_RULE.replace("Y = |Delta_rr| is unchanged", "Y = |run2@B - run1@B| at the pooled tau_B (RULE 45.1; gap run2_minus_run1_at_tauB) is unchanged")
             if cfg.get("pooled_world") else AGG_RULE)  # Codex r5 note: the yardstick named is the one computed
    return {"rule": rule_, "group": A["group"], "comparator": "(A) run-mean Delta_full", "rule47": RULE47, "caption": RULE47["caption"], "by_level": out}


def shape_gate(cfg, rec, base, z, keys):
    """0d's shape-check gate (E2E, 2e 10-17 ~07:3x): the pinned records' keys and the npz arrays' shapes, checked BEFORE any computation (a synthetic
    TEST runs it on the real record and npz, before synth). Any problem is a STOP naming keys and shapes only, never a value."""
    import numpy as np

    bad, n_ = [], None
    cb, md = cfg["combo"], cfg["mode"]
    for k_ in ("meta", "pooled", "rows", "coheadline_signed_plant", "npz"):
        if not isinstance(rec.get(k_), dict):
            bad.append(f"record: {k_} is not an object")
    pl_ = (rec.get("meta") or {}).get("plant") or {}
    if not (isinstance(pl_.get("deltas"), list) and isinstance(pl_.get("points"), list) and all(fin(x) for x in pl_["points"])):
        bad.append("record: meta.plant.deltas / points are not lists of numbers")
    fk_ = [f"{float(x):g}" for x in (pl_.get("points") or []) if fin(x)]
    if base is not None:
        for k_ in ("pooled", "coheadline_signed", "panel_groups"):
            if not isinstance(base.get(k_), dict):
                bad.append(f"base record: {k_} is not an object")
    yg_ = "gap_" + ((cfg.get("pooled_world") or {}).get("y_gap") or "delta_rr")
    for p in cfg["probes"]:
        for ro in cfg["readouts"]:
            lb = f"{p} {ro}"
            e_ = ((rec.get("pooled") or {}).get(lb) or {}).get(cb)
            mem = e_.get("members") if isinstance(e_, dict) else None
            if not (isinstance(mem, list) and mem and all(isinstance(o, str) for o in mem)):
                bad.append(f"record: pooled[{lb}][{cb}].members missing")
                continue
            bad += [f"record: rows[{lb}][{o}] missing" for o in mem if not isinstance(((rec.get("rows") or {}).get(lb) or {}).get(o), dict)]
            for f in fk_:
                if not isinstance((e_.get(md) or {}).get(f), dict):
                    bad.append(f"record: pooled[{lb}][{cb}][{md}][{f}] missing")
                if base is not None and not isinstance(((((((base.get("pooled") or {}).get(lb) or {}).get(cb) or {}).get(md) or {}).get(f) or {}).get("gap")
                                                        or {}).get("delta_full"), dict):
                    bad.append(f"base record: pooled[{lb}][{cb}][{md}][{f}].gap.delta_full missing")
                need = [npzkey(lb, cb, md, f, "pooled", "gap_delta_full")] + [npzkey(lb, cb, md, f, okey(cfg, o), "D") for o in mem]
                if cfg.get("aggregate"):
                    need += [npzkey(lb, cb, md, f, "pooled", nm) for nm in ("gap_sim_minus_run1", "gap_sim_minus_run2", yg_)]
                need += sorted(k for k in keys if k.startswith(npzkey(lb, cb, md, f, "pooled", "gap_delta_full_plant_")))
                for k in need:
                    if k not in keys:
                        bad.append(f"npz: {k} missing")
                        continue
                    sh = np.asarray(z[k]).shape
                    if len(sh) != 1 or sh[0] < 2 or (n_ is not None and sh[0] != n_):
                        bad.append(f"npz: {k} has shape {sh} (expected 1-D over the draws, one length for every array)")
                    elif n_ is None:
                        n_ = sh[0]
    if bad:
        raise SystemExit(f"STOP: shape-check gate: {len(bad)} problems (keys and shapes only), e.g. {bad[:6]}")
    return {"pass": True, "draws_plus_draw0": n_, "levels": fk_, "base_checked": base is not None}


def synth(cfg, rec, z):
    """synthetic_TEST: planted = unplanted + factor x delta / 100 in every draw (an 'unattainable' arm in one member of one row; optionally 10% NaN draws in
    one row, or the NEGATIVE cases drop_key (an absent array where every arm is 'ok') and bad_status (a malformed status)), and the record fields a planted
    record states (meta.window, meta.plant, rows[..].plant.solve_draw0, coheadline_signed_plant) written to match. Exercises the readers and gates only."""
    import numpy as np

    from probe_shift.fpr_core import summ

    T = cfg["synthetic_TEST"]
    cb, md = cfg["combo"], cfg["mode"]
    zz, base_ = {}, {}  # base_: the synthetic UNPLANTED base record (the aggregate's Stage A fields)
    rec.setdefault("meta", {})["window"] = cfg["calibration_expected"]
    rec["meta"]["plant"] = {"deltas": sorted(deltas_of(cfg) + [0]), "points": list(T["points"]), "labels": [f"{p} {r}" for r in cfg["readouts"] for p in cfg["probes"]
                                                                                                   if f"{p} {r}" != T.get("drop_label")]}
    for f in dict.fromkeys(f"{float(x):g}" for x in T["points"]):  # the npz's level keys (a NEGATIVE test may state a point no key names exactly)
        for ro in cfg["readouts"]:
            labs = [f"{p} {ro}" for p in cfg["probes"]]
            for lb in labs:
                base = z[npzkey(lb, cb, md, f, "pooled", "gap_delta_full")]
                zz[npzkey(lb, cb, md, f, "pooled", "gap_delta_full")] = base
                mem = rec["pooled"][lb][cb]["members"]
                for o in mem:
                    zz[npzkey(lb, cb, md, f, okey(cfg, o), "D")] = z[npzkey(lb, cb, md, f, okey(cfg, o), "D")]
                if T.get("bad_d_draw0") == lb:  # NEGATIVE test (E2E port): the first member's draw-0 D moved by 1.0, so only the D_draw0 gate fails
                    kd_ = npzkey(lb, cb, md, f, okey(cfg, mem[0]), "D")  # (every member's achieved shift is equal, so the D-weighted draw-0 check still holds)
                    zz[kd_] = np.array(z[kd_], dtype=float, copy=True)
                    zz[kd_][0] += 1.0
                zz[npzkey(lb, cb, md, f, "pooled", "gap_delta_full_plant_+0")] = base + (1e-6 if lb == T.get("bad_zero") else 0.0)  # the delta-0 arm (bad_zero: NEGATIVE)
                yg_ = "gap_" + ((cfg.get("pooled_world") or {}).get("y_gap") or "delta_rr")
                for nm in (("gap_sim_minus_run1", "gap_sim_minus_run2", yg_) if cfg.get("aggregate") else ()):  # the aggregate's inputs, only when configured
                    zz[npzkey(lb, cb, md, f, "pooled", nm)] = z[npzkey(lb, cb, md, f, "pooled", nm)]
                for d in deltas_of(cfg):
                    fac = T["factor"].get(lb.rsplit(" ", 1)[0], 1.0)
                    un_ = lb == T["unattainable"]["row"] and d == T["unattainable"]["delta"]
                    hit = lambda k_: bool(T.get(k_)) and lb == T[k_]["row"] and d == T[k_]["delta"]  # noqa: B023
                    for o in mem:
                        st_ = "typo" if (hit("bad_status") and o == mem[0]) else "unattainable" if (un_ and o == mem[0]) else "ok"
                        a_ = {"status": st_} | ({"c": fac * d / 100.0, "achieved_points": 100 * (fac * d / 100.0)} if st_ == "ok" else {})
                        rec["rows"][lb][o].setdefault("plant", {}).setdefault("solve_draw0", {}).setdefault(f, {}).setdefault("sim", {}).setdefault("arms", {})[f"{d:+d}"] = a_
                    if not (un_ or hit("drop_key") or hit("bad_status")):  # drop_key / bad_status: NEGATIVE tests (must GATE_FAIL)
                        pa_ = (np.full_like(base, base[0]) if T.get("constant_planted") else base) + fac * d / 100.0  # E2E port: constant_planted = a zero-width planted interval
                        if hit("nan_draws"):  # 10% of draws 1..D NaN: the planted interval is withheld, the verdict 'unavailable'
                            pa_ = pa_.copy()
                            pa_[1 : 1 + (len(pa_) - 1) // 10] = np.nan
                        zz[npzkey(lb, cb, md, f, "pooled", f"gap_delta_full_plant_{d:+d}")] = pa_
            q = {}
            for d in deltas_of(cfg):
                ks = [npzkey(lb, cb, md, f, "pooled", f"gap_delta_full_plant_{d:+d}") for lb in labs]
                q[f"{d:+d}"] = ({"S": summ(np.mean([zz[k] for k in ks], 0)), "rows": labs} if all(k in zz for k in ks)
                                else {"unavailable": "unattainable or missing in a member row"})
            pg_ = T.get("plant_group_written", cfg["plant_coheadline_group"])  # NEGATIVE test: the record names the group differently (must GATE_FAIL, no fallback)
            rec.setdefault("coheadline_signed_plant", {}).setdefault(cb, {}).setdefault(md, {}).setdefault(f, {})[f"{pg_}|{ro}"] = q
            if cfg.get("aggregate"):  # the UNPLANTED base record's fields the aggregate's gates read, made here from the same arrays (TEST), as Stage A lays them out
                gk_ = lambda nm: [zz[npzkey(lb, cb, md, f, "pooled", nm)] for lb in labs]  # noqa: B023
                S0_ = np.mean(gk_("gap_delta_full"), 0)
                X0_ = np.mean([0.5 * np.abs(a_) + 0.5 * np.abs(b_) for a_, b_ in zip(gk_("gap_sim_minus_run1"), gk_("gap_sim_minus_run2"))], 0) - np.mean(np.abs(gk_(yg_)), 0)
                X0_ = X0_ + (1e-6 if T.get("bad_stageA_excess") else 0.0)  # NEGATIVE test: Stage A's excess differs (must GATE_FAIL)
                base_.setdefault("coheadline_signed", {}).setdefault(cb, {}).setdefault(md, {}).setdefault(f, {})[f"{cfg['coheadline_group']}|{ro}"] = {"S": summ(S0_), "rows": labs}
                base_.setdefault("panel_groups", {}).setdefault(cb, {}).setdefault("pooled", {}).setdefault(md, {}).setdefault(f, {})[f"{cfg['aggregate']['group']}|{ro}"] = {
                    "primary": {"mean_G_minus_mean_Y": summ(X0_), "rows": labs}}
    if cfg.get("aggregate"):  # as Stage A's PLANT record (@ 20c6fbd20 run_peraction311.py:3021-3022): its own panel_groups are withheld
        rec["panel_groups"] = {"withheld": "plant job: the plant labels only (the headline six); the final record's panel_groups are authoritative"}
        base_["pooled"] = rec["pooled"]
        if cfg.get("pooled_world"):  # E2E port: the synthetic base is the same pooled build on (A) as the planted record it is made from (its meta copied)
            base_["meta"] = {k_: rec["meta"][k_] for k_ in ("pooled_world", "full_length", "estimator") if k_ in rec["meta"]}
    return rec, zz, (base_ if cfg.get("aggregate") and not T.get("no_base") else None)  # no_base: NEGATIVE test (the aggregate must GATE_FAIL)


def pool_label(cfg: dict, side_blob) -> str | None:
    """09's label ruling (10-17), in the container: a route C config names A's LABEL SIDECAR (cfg["a_label_sidecar"], git:<40-hex>:<path>.json@<sha256>;
    SIMPROBE's schema {record, record_sha256, label}, run_group_catch.a_label @ eb1b3478). The record's pool label is COMPOSED from it, never typed:
    STOP unless the bytes hash to the pin BEFORE parsing, the keys are exactly those three, record_sha256 is the base record's (A's) sha256, and the
    label names route C and PRELIMINARY. -> "<label> | plant_rows (Table 6 verdicts) on route C pooled A @ <A sha256>". No sidecar (the 171-pool
    configs): None."""
    import re

    ptr = cfg.get("a_label_sidecar")
    if ptr is None:
        return None
    m_, br_ = re.fullmatch(SIDECAR_PTR, str(ptr)), cfg.get("base_record") or {}
    a_sha = br_.get("decompressed_sha256") or br_.get("sha256")
    if not m_ or not isinstance(side_blob, bytes) or hashlib.sha256(side_blob).hexdigest() != m_[3]:
        raise SystemExit("STOP: A's label sidecar: not git:<40-hex>:<path>.json@<sha256>, or its bytes differ from the pin")
    sc = json.loads(side_blob)
    if not isinstance(sc, dict) or set(sc) != {"record", "record_sha256", "label"} or not isinstance(sc["record"], str) or not sc["record"]:
        raise SystemExit("STOP: A's label sidecar is not {record, record_sha256, label}")
    if not a_sha or sc["record_sha256"] != a_sha:
        raise SystemExit("STOP: A's label sidecar names another A record than this config's base_record")
    lab = sc["label"]
    if not isinstance(lab, str) or not lab.strip() or "route C" not in lab or "PRELIMINARY" not in lab:
        raise SystemExit("STOP: A's label sidecar's label is empty, does not name route C, or is not PRELIMINARY (09)")
    return f"{lab} | {POOL_STEP} on route C pooled A @ {a_sha}"


@app.function(image=image, cpu=(4, 4), memory=(16384, 16384), timeout=900, retries=0, scaledown_window=2, volumes={"/sol": SOL})
def plant_rows(cfg: dict, rec_blob: bytes, base_blob: bytes | None, ctx: dict | None = None, side_blob: bytes | None = None) -> dict:
    """plant_rows_body on Modal; with a launch context its return value (and the config it ran) is also saved to the Volume (exclusive name, committed), so a
    detached run whose client dropped can be re-materialised with fetch_panel_output.py (Aswin 10-16 via 2e)."""
    import copy

    cfg0 = copy.deepcopy(cfg)  # the config as launched (never a runtime object; Codex detach r1)
    try:
        plab_ = pool_label(cfg0, side_blob)  # 09's label ruling: route C's pool label, composed here BEFORE any work (None: a 171-pool config)
        out = plant_rows_body(cfg, rec_blob, base_blob)
        reply = finalize(cfg0, out, ctx or {}, plab_)  # 0d's rule (2e 10-17): record, expectations, tallies and token lines built HERE; inside the try, so a
        # failure in finalize() or expectations() is saved to the Volume too (2e's optional)
    except (Exception, SystemExit):  # the detach gap (2e): the failure itself (an error or a STOP) is saved to the Volume, then re-raised (never swallowed)
        if ctx:
            import traceback

            from probe_shift.panel_out import save_out

            save_out(SOL, "/sol", ctx["out_name"], {"context": ctx, "cfg": cfg0, "exception": traceback.format_exc()})
        raise
    if ctx:
        from probe_shift.panel_out import save_out

        reply = reply | {"volume_copy": save_out(SOL, "/sol", ctx["out_name"], {"context": ctx, "cfg": cfg0, "out": reply})}
    return reply


def finalize(c, out, ctx, plab=None):
    """In the container (0d's rule, 2e 10-17): the results record exactly as write_outputs used to build it (expectations included), as BYTES; the
    fixed-format token lines printed to stdout (counts, booleans and check names with every digit run replaced by '#', never a value); and the exit
    verdict. The laptop only writes the returned bytes (xb) and prints the file name and its sha256."""
    import re

    test_ = bool(c.get("synthetic_TEST"))
    out = {k: v for k, v in out.items() if k != "volume_copy"}
    exp_ = expectations(c, out) if test_ else None
    rec = {"label": ("SYNTHETIC TEST (a constant planted on a real record's arrays; not a result): " if test_ else "") + LABEL, "rule": RULE, "comparator": COMPARATOR,
           "config_label": c["label"], **({"pool_label": plab} if plab is not None else {}), **out, **({"test_expectations": exp_} if exp_ is not None else {}),
           "meta": {"git": ctx.get("git"), "stamp_pt": ctx.get("stamp"), "script_sha256": ctx.get("script_sha256"), "config": c}}
    fail = bool(out.get("GATE_FAIL"))
    sg_ = out.get("shape_gate") or {}
    names = sorted({re.sub(r"\d+", "#", str(x.get("check", ""))) for x in (out.get("gates") or {}).get("failures", [])}) if fail else []
    exit_ok = not ((exp_ is not None and not exp_["pass"]) or (fail and not (exp_ and exp_["pass"])))
    tokens = {"N_FAIL": int((out.get("gates") or {}).get("n_fail", -1)), "SHAPE_GATE": "pass" if sg_.get("pass") is True else "fail",
              "BASE_CHECKED": "true" if sg_.get("base_checked") is True else "false",
              "TEST_EXPECTATIONS_PASS": "none" if exp_ is None else ("true" if exp_["pass"] else "false"), "GATE_FAIL": "true" if fail else "false"}
    for k_, v_ in tokens.items():
        print(f"{k_} {v_}", flush=True)
    if fail:
        print("FAIL_CHECKS " + " | ".join(names), flush=True)
    print(f"EXIT_OK {'true' if exit_ok else 'false'}", flush=True)
    return {"record_bytes": json.dumps(rec, indent=1).encode(), "exit_ok": exit_ok, "tokens": tokens}


def unpack(b, ent):
    """E2E port (2e 10-17): a gzipped pinned record is decompressed and its decompressed sha256 checked HERE, in the container, never on the laptop."""
    if b is not None and isinstance(ent, dict) and ent.get("git", "").endswith(".gz"):
        import gzip

        b = gzip.decompress(b)
        if hashlib.sha256(b).hexdigest() != ent.get("decompressed_sha256"):
            raise SystemExit(f"STOP: {ent['git'].split(':', 1)[0][:12]}... decompressed sha256 != config")
    return b


def plant_rows_body(cfg: dict, rec_blob: bytes, base_blob: bytes | None) -> dict:
    import numpy as np

    from probe_shift.fpr_core import summ

    rec_blob, base_blob = unpack(rec_blob, cfg.get("record")), unpack(base_blob, cfg.get("base_record"))
    rec = json.loads(rec_blob)
    npz_ = rec["npz"]
    if npz_.get("volume") != VOL:
        raise SystemExit(f"STOP: the record's npz lives on {npz_.get('volume')!r}, not {VOL}")
    p_ = Path("/sol") / npz_["path"]
    if hashlib.sha256(p_.read_bytes()).hexdigest() != npz_["sha256"]:
        raise SystemExit(f"STOP: npz {npz_['path']} sha256 != the record's")
    z0 = np.load(p_)
    shape_ = shape_gate(cfg, rec, json.loads(base_blob) if base_blob else None, z0, set(z0.files))  # 0d's gate: STOP before any computation
    base_syn = None
    if cfg.get("synthetic_TEST"):
        rec, z, base_syn = synth(cfg, rec, z0)
    else:
        z = z0
    keys = set(z.files) if hasattr(z, "files") else set(z)
    cb, md = cfg["combo"], cfg["mode"]
    gates = []
    g = lambda ok, **kw: gates.append(kw | {"pass": bool(ok)})

    try:
        win_ = jget(rec, cfg["calibration_field"])
    except (KeyError, TypeError):
        win_ = None
    g(win_ == cfg["calibration_expected"], check=f"record window {win_!r} == {cfg['calibration_expected']!r}")
    pl_ = (rec.get("meta") or {}).get("plant") or {}
    dl_ = pl_.get("deltas") or []
    zero_ = 0 in dl_
    DS = deltas_of(cfg)
    g(sorted(x for x in dl_ if x != 0) == DS and len(dl_) == len(set(dl_)), check=f"record plant deltas {dl_} are the config's {DS} (and optionally 0)")
    fkeys = levels_of(pl_.get("points"))  # 2e: the levels come from the planted record's own meta, never a config list; 0.01 must be among them
    if not fkeys or "0.01" not in fkeys or 0.01 not in pl_["points"]:
        g(False, check=f"record plant points {pl_.get('points')!r} are not well-formed levels including 0.01")
        bad = [x for x in gates if not x["pass"]]
        return {"GATE_FAIL": True, "shape_gate": shape_, "gates": {"n": len(gates), "n_fail": len(bad), "failures": bad[:40]}}
    # the planted ROWS by key (2e 10-16: meta.plant.labels is descriptive only; the full-length PLANT record lists 32k-style names there): every probe x readout row of the
    # record carries a plant block on at least one origin row (each member's per-delta statement is gated row by row below)
    prow_ = {lb for lb, by_o in (rec.get("rows") or {}).items() if isinstance(by_o, dict) and any(isinstance((v_ or {}).get("plant"), dict) for v_ in by_o.values() if isinstance(v_, dict))}
    miss_p = [f"{p} {ro}" for p in cfg["probes"] for ro in cfg["readouts"] if f"{p} {ro}" not in prow_]
    g(not miss_p, check=f"every probe x readout row carries planted data by key (missing: {miss_p})")
    base_rec = json.loads(base_blob) if base_blob else base_syn  # the unplanted base record (a synthetic one in a TEST)
    pw_ = cfg.get("pooled_world")
    if pw_:  # Codex r3 #2 / RULE 45.1 (via 2e): the pooled contract on BOTH records - a pooled build on (A), y1 = run1@B stated everywhere, no run1@B2 arm
        for nm_, r_ in (("planted", rec), ("base", base_rec)):
            if r_ is None:
                g(False, check=f"pooled: the {nm_} record is required")
                continue
            yd_ = {(q_.get("catch") or {}).get("y1_def") for e_ in (r_.get("pooled") or {}).values() if isinstance(e_, dict) for c_ in e_.values() if isinstance(c_, dict)
                   for m_ in ("joint", "fixed") for q_ in (c_.get(m_) or {}).values() if isinstance(q_, dict)}
            g(yd_ == {pw_["y1_def"]}, check=f"pooled: the {nm_} record states y1_def {pw_['y1_def']!r} in every pooled catch block (found {sorted(map(str, yd_))})")
            mt_ = r_.get("meta") or {}
            fe_ = (mt_.get("full_length") or {}).get("estimator")  # Stage A @ 8a15fb10f states it at meta.full_length.estimator.used; an older record at meta.estimator.used
            est_ = fe_.get("used") if isinstance(fe_, dict) else (mt_.get("estimator") or {}).get("used")
            g(bool(mt_.get("pooled_world")) and est_ == "A", check=f"pooled: the {nm_} record is a pooled build on (A) (meta.pooled_world, estimator used)")
        g(not [k_ for k_ in keys if "run1_at_B2" in k_], check="pooled: no run1@B2 array in the planted npz (RULE 45.1: y1 = run1@B)")
    rows, counts = {}, {}
    for f in fkeys:
        for ro in cfg["readouts"]:
            labs = [f"{p} {ro}" for p in cfg["probes"]]
            try:  # the PLANT record's own group name (Stage A @ 20c6fbd20 run_peraction311.py:2679/2689 writes "headline_six|<ro>"), mapped explicitly in the config
                coh = jget(rec, ["coheadline_signed_plant", cb, md, f, f"{cfg['plant_coheadline_group']}|{ro}"])
            except (KeyError, TypeError):
                coh = None
            g(isinstance(coh, dict), check=f"coheadline_signed_plant {cb}/{md}/{f}/{cfg['plant_coheadline_group']}|{ro} present")
            planted_by_d = {}
            for lb in labs:
                e_ = (rec.get("pooled") or {}).get(lb, {}).get(cb)
                if not isinstance(e_, dict) or md not in e_ or f not in e_[md]:
                    g(False, row=lb, f=f, check="row missing in the record's pooled section")
                    continue
                kb = npzkey(lb, cb, md, f, "pooled", "gap_delta_full")
                if kb not in keys:
                    g(False, row=lb, f=f, check=f"unplanted array {kb} missing")
                    continue
                base = np.asarray(z[kb], float)
                sb = summ(base)
                g(same_summ(sb, e_[md][f]["gap"]["delta_full"]), row=lb, f=f, check="unplanted array summary == the record's pooled delta_full")
                if base_rec is not None:
                    g(same_summ(sb, jget(base_rec, ["pooled", lb, cb, md, f, "gap", "delta_full"])), row=lb, f=f, check="unplanted == the base record's pooled delta_full")
                readable = e_[md][f].get("readable", {}).get("all") is True
                mem = e_.get("members")
                if not (isinstance(mem, list) and mem and all(isinstance(o, str) and o for o in mem) and len(set(mem)) == len(mem)):
                    g(False, row=lb, f=f, check=f"members {mem!r} are not a non-empty list of unique origins")
                    continue
                D0 = {o: float(np.asarray(z[npzkey(lb, cb, md, f, okey(cfg, o), "D")])[0]) for o in mem if npzkey(lb, cb, md, f, okey(cfg, o), "D") in keys}
                if cfg.get("pooled_world") and (not cfg.get("synthetic_TEST") or cfg["synthetic_TEST"].get("check_d_draw0")):  # E2E port: the npz draw-0 D
                    # equals the record's own D_draw0 per member (a TEST runs it with check_d_draw0: TEST_pos must pass it, TEST_dd0neg must fail it)
                    dd_ = e_[md][f].get("D_draw0") or {}
                    g(set(dd_) == set(mem) and all(o in D0 and abs(D0[o] - float(dd_[o])) <= 1e-9 for o in mem), row=lb, f=f,
                      check="pooled: the npz draw-0 D of every member == the record's D_draw0")
                g(len(D0) == len(mem) and all(fin(x) and x >= 0 for x in D0.values()) and sum(D0.values()) > 0, row=lb, f=f,
                  check="every member's draw-0 D present, finite and >= 0; total > 0")
                u = 100 * float(base[0])
                if zero_:  # the record states delta 0: its planted array must equal the unplanted one in every draw
                    k0 = npzkey(lb, cb, md, f, "pooled", "gap_delta_full_plant_+0")
                    g(k0 in keys and np.array_equal(np.asarray(z[k0], float), base, equal_nan=True), row=lb, f=f, check="the delta-0 planted array equals the unplanted array")
                for d in DS:
                    tag = f"{d:+d}"
                    st_ = {o: (((rec["rows"].get(lb) or {}).get(o) or {}).get("plant") or {}).get("solve_draw0", {}).get(f, {}).get("sim", {}).get("arms", {}).get(tag) for o in mem}
                    g(all(arm_ok(s_) for s_ in st_.values()), row=lb, f=f, delta=tag, check="every member states the arm: 'ok' (finite c, achieved_points) or 'unattainable'")
                    allok = all(arm_ok(s_) and s_["status"] == "ok" for s_ in st_.values())
                    kp = npzkey(lb, cb, md, f, "pooled", f"gap_delta_full_plant_{tag}")
                    v_ = {"delta": d, "readable": readable, "unplanted_point": u}
                    if kp not in keys:
                        g(all(arm_ok(s_) for s_ in st_.values()) and any(s_["status"] == "unattainable" for s_ in st_.values()), row=lb, f=f, delta=tag,
                          check="planted array absent only where a member's arm is 'unattainable'")
                        rows.setdefault(f, {}).setdefault(ro, {}).setdefault(lb, {})[tag] = v_ | {"verdict": "unavailable", "reason": f"unattainable ({'ceiling' if d > 0 else 'floor'}) in a member: "
                                                                                                  + ", ".join(f"{o} ({(s_ or {}).get('status')})" for o, s_ in st_.items() if (s_ or {}).get("status") != "ok")}
                        continue
                    g(allok, row=lb, f=f, delta=tag, check="planted array present only where every member's arm is 'ok'")
                    pa = np.asarray(z[kp], float)
                    g(pa.shape == base.shape, row=lb, f=f, delta=tag, check="planted array has the unplanted array's shape")
                    planted_by_d.setdefault(tag, []).append(pa)
                    if allok and len(D0) == len(mem) and sum(D0.values()) > 0:
                        ach = sum(st_[o]["achieved_points"] * D0[o] for o in mem) / sum(D0.values())
                        g(abs(100 * (pa[0] - base[0]) - ach) <= 1e-9, row=lb, f=f, delta=tag, check="draw-0 move == the D-weighted achieved shift the record states per origin")
                    sp = summ(pa)
                    ch = summ(pa - base)
                    p = 100 * sp["point"]
                    move = p - u
                    ci = [100 * sp["ci95"][0], 100 * sp["ci95"][1]] if sp.get("ci95") else None
                    in_range = abs(move - d) <= 0.5 * abs(d)
                    v_ |= {"planted_point": p, "move_points": move, "move_in_range": bool(in_range), "planted_ci95": ci, "finite_draws": sp["finite_draws"], "draws": sp["draws"],
                           "paired_change": {"point": 100 * ch["point"], "ci95": [100 * x for x in ch["ci95"]] if ch.get("ci95") else None},
                           "mechanism_mismatch": None if in_range else "mechanism mismatch"}
                    if not readable or ci is None:
                        v_ |= {"verdict": "unavailable", "reason": "row unreadable" if not readable else "planted interval withheld (fewer than 95% complete draws)"}
                    else:
                        excl = not (ci[0] <= u <= ci[1])
                        marg = (ci[0] - u) if u < ci[0] else (u - ci[1]) if u > ci[1] else -min(u - ci[0], ci[1] - u)
                        v_ |= {"interval_excludes_unplanted": bool(excl), "margin_points": marg, "verdict": "detected" if (in_range and excl) else "not detected"}
                    rows.setdefault(f, {}).setdefault(ro, {}).setdefault(lb, {})[tag] = v_
            for d in DS:  # the probes' planted arrays average to the record's planted S (same arrays as the aggregate)
                tag = f"{d:+d}"
                q_ = (coh or {}).get(tag)
                if len(planted_by_d.get(tag, [])) == len(labs):
                    g(isinstance(q_, dict) and sorted(q_.get("rows") or []) == sorted(labs) and same_summ(summ(np.mean(planted_by_d[tag], 0)), q_.get("S")),
                      f=f, readout=ro, delta=tag, check="mean of the probes' planted arrays == coheadline_signed_plant S; its rows are the probes")
                else:
                    g(isinstance(q_, dict) and "unavailable" in q_, f=f, readout=ro, delta=tag, check="coheadline_signed_plant states this delta unavailable (a probe's arm is)")
    for f in fkeys:  # complete scope: every level, readout and signed delta has one case per probe (Codex plant-rows r1)
        for ro in cfg["readouts"]:
            for d in DS:
                n_ = sum(f"{d:+d}" in v for v in rows.get(f, {}).get(ro, {}).values())
                g(n_ == len(cfg["probes"]) == 6, f=f, readout=ro, delta=f"{d:+d}", check=f"{n_} cases == the six probes")
    for f, by in rows.items():
        for d in DS:
            tag = f"{d:+d}"
            c_ = {}
            for ro in list(by) + ["+".join(by)]:
                vs = [by[r_][lb][tag] for r_ in (ro.split("+")) for lb in by[r_]]
                c_[ro] = {"detected": sum(v["verdict"] == "detected" for v in vs), "not_detected": sum(v["verdict"] == "not detected" for v in vs),
                          "unavailable": sum(v["verdict"] == "unavailable" for v in vs), "mechanism_mismatch": sum(v.get("mechanism_mismatch") is not None for v in vs), "cases": len(vs),
                          "misses": {f"{lb} ({r_})" if "+" in ro else lb: by[r_][lb][tag]["verdict"] for r_ in ro.split("+") for lb in by[r_]  # E2E port: names of misses
                                     if by[r_][lb][tag]["verdict"] != "detected"}}
            counts.setdefault(f, {})[tag] = c_
    agg_ = aggregate(cfg, rec, base_rec, z, keys, DS, fkeys, g) if cfg.get("aggregate") else None  # 09 item 35
    bad = [x for x in gates if not x["pass"]]
    if bad or not gates:  # a failed gate withholds every value (fail closed)
        return {"GATE_FAIL": True, "shape_gate": shape_, "gates": {"n": len(gates), "n_fail": max(len(bad), 1), "failures": bad[:40]}}
    return {"gates": {"n": len(gates), "n_fail": 0}, "shape_gate": shape_, "levels": fkeys, "levels_source": "meta.plant.points of the planted record", "rule47": RULE47,
            "rows": rows, "counts": counts} | (
        {"aggregate": agg_} if agg_ is not None else {})


def pinned_six(git, pin):
    """HEADLINE_SIX from 507-prep's pinned label constants (full_override.py @ sha), read from its top-level UPPERCASE assignments only (as run_pooled.py)."""
    import ast

    b_ = git("show", pin["git"])
    if hashlib.sha256(b_).hexdigest() != pin["sha256"]:
        raise SystemExit("STOP: the pinned label constants file sha256 != config")
    ns_ = {"__builtins__": {"tuple": tuple, "dict": dict, "list": list, "set": set, "sorted": sorted}}
    for n_ in ast.parse(b_.decode()).body:
        if isinstance(n_, ast.Assign) and len(n_.targets) == 1 and isinstance(n_.targets[0], ast.Name) and n_.targets[0].id.isupper():
            exec(compile(ast.Module(body=[n_], type_ignores=[]), "full_override_constants", "exec"), ns_)  # noqa: S102 (pinned project constants, restricted builtins)
    return list(ns_["HEADLINE_SIX"])


def expectations(c, out):
    """synthetic_TEST: the stated outcome must hold - a GATE_FAIL whose failures include the named check (and no values), or the stated per-delta counts."""
    ex = c.get("expect")
    if not ex:
        return None
    if "gate_fail_check" in ex:
        hit_ = [x for x in out["gates"].get("failures", []) if ex["gate_fail_check"] in x.get("check", "")]
        return {"expected": ex, "pass": bool(out.get("GATE_FAIL")) and bool(hit_) and "rows" not in out and "counts" not in out and "aggregate" not in out}
    got = {k: {d: out["counts"][ex["f"]][d]["+".join(c["readouts"])][k] for d in ex[k]} for k in ("detected", "mechanism_mismatch", "unavailable")} if not out.get("GATE_FAIL") else None
    ok_ = got is not None and all(got[k] == {d: ex[k][d] for d in ex[k]} for k in got)
    if ex.get("aggregate_complete") and ok_:  # 09 item 35: every level, readout and signed delta has an S and an excess verdict from the allowed set
        V_ = {"detected", "not detected", "unavailable", "not testable at this delta"}
        by_ = (out.get("aggregate") or {}).get("by_level") or {}
        ok_ = bool(by_) and set(by_) == set(out.get("levels") or []) and all(  # every expected level (Codex agg r1), readout and delta
            set((by_.get(f) or {}).get(ro, {})) == {f"{d:+d}" for d in deltas_of(c)} and all(
                {(e_.get(k_) or {}).get("verdict") for k_ in ("S", "excess")} <= V_ and set(e_) == {"S", "excess"} for e_ in by_[f][ro].values()) for f in by_ for ro in c["readouts"])
        for f, by_ro in (ex.get("aggregate_unavailable") or {}).items():  # the fixture's unattainable / NaN-draw arms read unavailable in the aggregate, and only those
            for ro in c["readouts"]:
                for tag, e_ in ((by_.get(f) or {}).get(ro) or {}).items():
                    want_ = tag in (by_ro.get(ro) or [])
                    ok_ = ok_ and all(((e_.get(k_) or {}).get("verdict") == "unavailable") == want_ for k_ in ("S", "excess"))
    if ex.get("aggregate_S_negative") and ok_:  # E2E port: the six-probe S verdict at every level, readout and NEGATIVE delta is the stated one
        by_ = (out.get("aggregate") or {}).get("by_level") or {}
        ok_ = bool(by_) and set(by_) == set(out.get("levels") or []) and all(
            ((((by_.get(f) or {}).get(ro) or {}).get(f"{d:+d}") or {}).get("S") or {}).get("verdict") == ex["aggregate_S_negative"]
            for f in by_ for ro in c["readouts"] for d in deltas_of(c) if d < 0)
    return {"expected": ex, "got": got, "pass": ok_}


@app.local_entrypoint()
def main(cfg: str):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("uncommitted tracked changes")
    c = json.loads((HERE / "config" / f"{cfg}.json").read_text())
    if "TO_FILL" in json.dumps(c) or "TO_CONFIRM" in json.dumps(c):
        raise SystemExit(f"config {cfg} still has TO_FILL fields")
    if deltas_of(c) is None or sorted(c["readouts"]) != ["C", "R"] or len(c["probes"]) != 6 or len(set(c["probes"])) != 6:
        raise SystemExit("STOP: readouts C and R, probes the six lead probes without repeats (deltas: see deltas_of)")
    if "f_keys" in c:  # 2e: the levels come from the planted record's meta.plant.points (0.01 required there), never from the config
        raise SystemExit("STOP: f_keys is not a config field here (levels come from the planted record's meta.plant.points)")
    test_ = bool(c.get("synthetic_TEST"))
    if c.get("pooled_world") and (c["pooled_world"].get("y_gap") != "run2_minus_run1_at_tauB" or c["pooled_world"].get("y1_def") != POOLED_Y1_DEF):
        raise SystemExit(f"STOP: a pooled plant config needs pooled_world.y_gap 'run2_minus_run1_at_tauB' and y1_def {POOLED_Y1_DEF!r} (RULE 45.1; no fallback to delta_rr)")
    if not all(isinstance(c.get(k_), str) and c[k_] for k_ in ("plant_coheadline_group", "coheadline_group")):  # explicit key maps, no fallback (0d's planner, 10-16)
        raise SystemExit("STOP: the config must name plant_coheadline_group (the PLANT record's coheadline_signed_plant group) and coheadline_group (the base record's)")
    if c.get("aggregate") and not test_ and not c.get("base_record"):  # the PLANT record withholds its panel_groups: the aggregate reads the unplanted base record
        raise SystemExit("STOP: the aggregate needs base_record (the unplanted record whose coheadline_signed and panel_groups it reads)")
    if not test_:  # production: the six are 507-prep's pinned HEADLINE_SIX, in order (a TEST names its stand-ins in its label)
        if not c.get("labels_pin") or c["probes"] != pinned_six(git, c["labels_pin"]):
            raise SystemExit("STOP: a non-TEST run needs labels_pin and probes equal to its HEADLINE_SIX")
        if c.get("expect"):
            raise SystemExit("STOP: expect is for synthetic TESTs only")
    blobs = []
    for k_ in ("record", "base_record"):
        if c.get(k_):
            b_ = git("show", c[k_]["git"])
            if hashlib.sha256(b_).hexdigest() != c[k_]["sha256"]:
                raise SystemExit(f"STOP: {k_} {c[k_]['git']} sha256 != config")
            if c[k_]["git"].endswith(".gz") and not (isinstance(c[k_].get("decompressed_sha256"), str) and len(c[k_]["decompressed_sha256"]) == 64):
                raise SystemExit(f"STOP: {k_} is gzipped but the config pins no decompressed_sha256")  # it is checked in the container (2e 10-17)
            blobs.append(b_)  # the committed (gzipped) bytes as pinned; decompressed and checked only in the container (unpack)
        else:
            blobs.append(None)
    side_ = None
    if c.get("a_label_sidecar") is not None:  # 09's label ruling (route C): A's label sidecar bytes, read unparsed (the container checks and composes)
        import re

        m_ = re.fullmatch(SIDECAR_PTR, str(c["a_label_sidecar"]))
        if test_ or not c.get("base_record") or "171-pool" in json.dumps(c.get("label")) or not m_:
            raise SystemExit("STOP: a route C config (a_label_sidecar) is a non-TEST run with base_record, a git:<40-hex>:<path>.json@<sha256> "
                             "sidecar pointer, and a label that names no 171-pool")
        side_ = git("show", f"{m_[1]}:{m_[2]}")
        if hashlib.sha256(side_).hexdigest() != m_[3]:
            raise SystemExit("STOP: A's label sidecar's bytes differ from the config's pin")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    if not app.app_id:
        raise SystemExit("STOP: no app id at launch (the Volume copy needs one)")
    ctx = {"kind": "plant_rows", "cfg_name": cfg, "stamp": stamp, "git": git("rev-parse", "HEAD").decode().strip(), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "app_id": app.app_id, "out_name": f"plant_rows_{cfg}_{stamp}_{app.app_id}"}
    print(json.dumps({"app_id": ctx["app_id"], "volume_copy_name": ctx["out_name"]}))
    out = plant_rows.remote(c, blobs[0], blobs[1], ctx, side_)
    print(json.dumps({"app_id": ctx["app_id"], "volume_copy": out.get("volume_copy")}))
    write_outputs(cfg, c, out, ctx)


def write_outputs(cfg, c, reply, ctx):
    """results/ from a plant_rows() reply (directly after the run, or later from its Volume copy via fetch_panel_output.py): the laptop only WRITES the
    record bytes the container built (exclusive create) and prints the file name and sha256; it reads no field of the record (0d's rule, 2e 10-17).
    The exit verdict is the container's (finalize)."""
    from probe_shift.panel_out import write_once

    b_ = reply["record_bytes"]
    p_ = HERE / "results" / f"panel_plant_rows_{cfg}_{ctx['stamp']}.json"
    write_once(p_, b_)
    print(f"RECORD {p_.name} sha256 {hashlib.sha256(b_).hexdigest()}", flush=True)
    if reply.get("exit_ok") is not True:
        raise SystemExit(1)
