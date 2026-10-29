"""PLANT V3 block (v): a LINE-BY-LINE PORT of run_plant_rows.py @ fd12eca35 (blob fd6790f925cf, file sha256
199b1ab31983da827b7db210fef2380a0ad3019d398312f9919fa1f79040ef5b = plant_rows' meta.script_sha256), labelled
"producer's field definitions (RULE defines only the verdict rule)"; the excess move is the producer's reading (RULE ambiguous;
GATES_routec @ 4e4ab4142 L580). PRESTATED_V3 fb34c414 + AMEND1 5e54a709 / AMEND3 94fc0084 / AMEND3a 1d0f5696 / AMEND4 e84c8e1e.

Ported ranges (AMEND4 D5): module L15-21 (imports; L23 `import modal` dropped) and L25-127 WHOLE except L27 (APP), L53 (app), L54 (image),
L55 (SOL); aggregate L128-213; shape_gate L215-266; unpack L415-423; plant_rows_body L426-593; fpr_core.summ L37-44 @ fd12eca35 inline.
Explicit diffs from the producer's lines (each marked "# V3 DIFF" in place):
  D-a  L136 and L429 `from probe_shift.fpr_core import summ` -> `pass` (summ is this module's inline port; same body).
  D-b  after L187 (X1 = ...): one added line, `X1 = _v3_xctl(X1, f, ro, tag)`, the A4 control hook; the identity unless the A4 control
       call sets V3_XCTL (AMEND4 D3 A4).
  D-d  after L445 (z = z0): one added line, `z = _v3_zctl(z)`, the P4 overlay (AMEND5 F1); the identity unless V3_CTL names P4.
  D-e  L589: the withhold is skipped ONLY in the P4 control copy, which then returns its failed gates beside the values computed
       before the withhold (AMEND5 F1); a normal call (V3_CTL None) takes the producer's path exactly.
  D-c  the container: this module runs inside the G7b checker's own app, image and Volume mounts (/sol = the same solavg Volume, by name,
       read-only); nothing here creates an app, image or Volume object.
Generated from git by extraction (sed ranges), then the three diffs above applied; plant_rows_v3.py holds V3's own code (comparator,
printed-row booleans, controls, self-tests).
"""

import hashlib
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# ---- run_plant_rows.py @ fd12eca35 L25-26 ----
HERE = Path(__file__).parent
VOL = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13"
# ---- L27 (APP) not ported (V3 DIFF D-c) ----
# ---- L28-52 ----
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

# ---- L53-55 (app, image, SOL) not ported (V3 DIFF D-c) ----
# ---- L56-127 ----


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


# ---- L128-213 (aggregate) ----
def aggregate(cfg, rec, base, z, keys, DS, fkeys, g):
    """09 item 35: the aggregate planted check on the headline combo from the SAME planted pooled arrays: the six-probe signed S (item 24's delta rule) and
    the registered excess (Delta_exp rule), per level, readout and signed delta, with Delta_exp printed beside each verdict. Gates: each probe's
    0.5 (sim - run1 + sim - run2) equals its unplanted Delta_full; the unplanted S and excess equal Stage A's coheadline_signed S and panel_groups excess
    of the UNPLANTED base record (a PLANT record withholds its panel_groups: Stage A @ 20c6fbd20 run_peraction311.py:3021-3022), read by the explicit
    group names in the config; an absent base record or key is a gate failure, never a fallback."""
    import numpy as np

    pass  # V3 DIFF D-a (L136): summ is this module's inline port of fpr_core.summ (below)

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
                X1 = _v3_xctl(X1, f, ro, tag)  # V3 DIFF D-b (after L187): the A4 control hook; the identity unless V3_CTL names A4
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

# ---- L214-266 (shape_gate) ----

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




# ---- L415-423 (unpack) ----
def unpack(b, ent):
    """E2E port (2e 10-17): a gzipped pinned record is decompressed and its decompressed sha256 checked HERE, in the container, never on the laptop."""
    if b is not None and isinstance(ent, dict) and ent.get("git", "").endswith(".gz"):
        import gzip

        b = gzip.decompress(b)
        if hashlib.sha256(b).hexdigest() != ent.get("decompressed_sha256"):
            raise SystemExit(f"STOP: {ent['git'].split(':', 1)[0][:12]}... decompressed sha256 != config")
    return b


# ---- L426-593 (plant_rows_body) ----
def plant_rows_body(cfg: dict, rec_blob: bytes, base_blob: bytes | None) -> dict:
    import numpy as np

    pass  # V3 DIFF D-a (L429): summ is this module's inline port of fpr_core.summ (below)

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
        z = _v3_zctl(z)  # V3 DIFF D-d (after L445): the P4 overlay; the identity unless V3_CTL names P4
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
    if (bad or not gates) and not _v3_p4_keep():  # V3 DIFF D-e (L589): the withhold is skipped ONLY in the P4 control copy
        return {"GATE_FAIL": True, "shape_gate": shape_, "gates": {"n": len(gates), "n_fail": max(len(bad), 1), "failures": bad[:40]}}
    if _v3_p4_keep():  # V3 DIFF D-e: the P4 copy returns its failed gates beside the values computed before the withhold (never a normal call)
        return {"v3_failed_gates": bad, "rows": rows, "counts": counts, "aggregate": agg_}
    return {"gates": {"n": len(gates), "n_fail": 0}, "shape_gate": shape_, "levels": fkeys, "levels_source": "meta.plant.points of the planted record", "rule47": RULE47,
            "rows": rows, "counts": counts} | (
        {"aggregate": agg_} if agg_ is not None else {})


# ---- src/probe_shift/fpr_core.py @ fd12eca35 L37-44 (summ), inline (V3 DIFF D-a) ----
import numpy as np  # fpr_core.py L9 (module import used by summ)


def summ(v):
    """Point (draw 0) and percentile 95% interval over draws 1..D; withheld if under 95% of draws are finite."""
    D = len(v) - 1
    f = v[1:][np.isfinite(v[1:])]
    o = {"point": float(v[0]), "finite_draws": int(len(f)), "draws": D}
    if len(f) < 0.95 * D:
        return o | {"ci95": None, "withheld": "fewer than 95% complete draws"}
    return o | {"ci95": [float(np.percentile(f, 2.5)), float(np.percentile(f, 97.5))]}


# ---- V3 control hooks (AMEND4 D3, AMEND5 F1): each is the IDENTITY while V3_CTL is None (the normal block (v) call) ----
V3_CTL = None  # set ONLY by plant_rows_v3 for one control copy: {"kind": "A4", "f", "ro", "tag", "shift"} or {"kind": "P4", "key", "shift"}


def _v3_xctl(X1, f, ro, tag):  # A4: the planted per-draw excess shifted on draws 1..D only (rate units), at one (level, readout, delta)
    c_ = V3_CTL
    if c_ is None or c_.get("kind") != "A4" or (c_["f"], c_["ro"], c_["tag"]) != (f, ro, tag):
        return X1
    X1 = np.array(X1, dtype=float, copy=True)
    X1[1:] = X1[1:] + c_["shift"]
    return X1


class _V3Overlay:  # P4: the PLANT npz with exactly one array replaced (same key); .files and [k] as an NpzFile
    def __init__(self, z, key, arr):
        self._z, self._k, self._a = z, key, arr
        self.files = list(z.files)

    def __getitem__(self, k):
        return self._a if k == self._k else self._z[k]


def _v3_zctl(z):  # P4: one probe's pooled planted array shifted at draw 0 only (rate units, 2 x delta / 100)
    c_ = V3_CTL
    if c_ is None or c_.get("kind") != "P4":
        return z
    assert c_["key"] in set(z.files), ("P4: the overlay key is not in the npz", c_["key"])
    a_ = np.array(z[c_["key"]], dtype=float, copy=True)
    a_[0] = a_[0] + c_["shift"]
    return _V3Overlay(z, c_["key"], a_)


def _v3_p4_keep():
    return V3_CTL is not None and V3_CTL.get("kind") == "P4"
