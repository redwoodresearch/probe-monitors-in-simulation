"""Clean-room PLANT, tier 1 (addendum 21, a1, 2026-10-20): route C's planted-shift check (Table 6, App D, the main text's
"as little as 3 points"), from the registered plant and verdict rules and my own draw-0 units; intervals from PLANT's per-draw npz.

  draw 0 (independent): my unchanged reducer (cleanroom_reduce_v2.py, sha256 pinned) under the A18 config gives every cell's largest
      transformed sim unit (g_sim coordinates; raw scores for single-model probes), the cell weights and the draw-0 thresholds. A constant
      c added to every sim attack unit moves that maximum by c, so the plant is solved exactly per probe row x readout x level x ORIGIN
      (plant455 RULE @ a82f13724 L10-13, L107; 09's ruling (ii)): the smallest |c| whose draw-0 origin sim catch at tau_sim moves by
      >= |delta| points, "unattainable" if no c can. Planted draw-0 points (per-probe Delta_full, S, the excess, Delta_exp) follow.
  draws (shared): PLANT's per-draw npz (a355f071) and A's (5e6cd10e) give the planted / unplanted per-draw pooled arrays. The index
      rule and percentile method are identified on PLANT's own unplanted pooled delta_full nodes; PLANT's unplanted arrays must equal
      A's draw for draw; my planted-S rule must reproduce PLANT's coheadline_signed_plant node.
  verdicts: RULE item 24 / 34 / 35 / 47 (@ 993da3957 l.4777, l.4820ff, l.4829ff, l.4943ff), 09's rulings (i)-(iii) and the
      bootstrap-mean-move readings r1 / r2 / r3.
The laptop gets booleans and counts only; every value goes to the output Volume.

  CR_CFG=<pinned clean-room inputs> PL_CFG=<PLANT gz + plant_rows> uv run --no-project --python 3.12 --with modal==1.6.1 modal run cleanroom_plant.py
"""

import json
import os
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).parent
OUT = "probe-shift_aswin-surya_sub-repro_cleanroom_kill-2026-11-16"
SCORES = "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14"
POOLED = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13"
ACTS = "probe-shift_aswin-surya_sub-q3cutfull_acts_kill-2026-11-14"
HF = "probe-shift_aswin-surya_hf-cache"
REDUCER_SHA = "991a34be9f290797bfbf89c543341e59990b53a72ec11bf045ec6d426cf21f30"  # cleanroom_reduce_v2.py @ bd4c2a802 (blob b8c136c8)
A18_VALUES = ("cleanroom_values_A18_2026-10-20_040452.json", "9a6c50f41e1e394f13f1cf0feda7ac5d2b2a8bcea6a0d0c2154df0f12de1bbe2")
A18_CMP = ("cleanroom_comparison_A18_2026-10-20_040452.json", "584c9d8c9564df728aaa692bc03c696446100bd1e68ba17406d64f24571a61d7")
PLANT_GZ = ("routec_PLANT_alltasks_CRD_FULL_plant_r2_allorigins_pooledworld_scored_e12_2026-10-19_230620.json.gz",  # @ b90a38f98
            "df2c36a66dd79e50f9fb71216b906fbca6120f2c0f8a2ca4d766acf12c980261", "66fda847576f30bd23fa6d6b3bde80290257115e6f2afca59e1ee7a92a86851b")
PLANT_ROWS = ("panel_plant_rows_plant_rows_POOLED_FULL_routeC_2026-10-20_004247_2026-10-20_004833.json",  # @ 699f21403
              "a864fad74436f0b5052adfaae7ba4ff54a0f192670b74b9b88ec8235b7048a21")
PLANT_NPZ = ("xfer/fpr507prep/alltasks_CRD_FULL_plant_r2_allorigins_pooledworld_scored_e12_2026-10-19_230620_perdraw.npz",
             "a355f0715688818cda7a2e7b337c49f38377da38b4d3094c8643b5ed79c7d452")
A_NPZ = ("xfer/fpr507prep/alltasks_CRD_FULL_r2_allorigins_pooledworld_scored_e12_2026-10-19_180327_perdraw.npz",
         "5e6cd10e382d8b03d409982c67efd72512176d7e95d2f3c94752d2b90e5650d1")
COMBO, MODE = "pooled_all_origins", "joint"
LVLS = ("0.01", "0.05")
DELTAS = (-12, -8, -5, -3, 3, 5, 8, 12)
SIX = ["sol_a_uncut", "FIBS|uncut", "harmfulness|uncut", "refusal|uncut", "high-stakes|uncut", "own_all_rs_uncut_L48"]
SETS = ["S_old", "S_new", "S_newmains", "S_paper493"]
METHODS = ("linear", "hazen", "weibull", "median_unbiased", "normal_unbiased", "interpolated_inverted_cdf", "inverted_cdf",
           "averaged_inverted_cdf", "closest_observation", "lower", "higher", "nearest", "midpoint")
RTOL = 1e-9
CUM_EPS = 1e-12  # a cumulative catch change counts as reaching |delta| / 100 when it is within this of it (float sums of 1 / n weights)

app = modal.App(OUT)
OUT_VOL = modal.Volume.from_name(OUT, create_if_missing=True, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "threadpoolctl==3.6.0", "tokenizers==0.22.1")
if modal.is_local():
    _root = HERE.resolve().parents[1]
    image = (image.add_local_dir(os.environ["CR_CFG"], "/cfg")
             .add_local_file(str(HERE / "config/routec_A18_pins.json"), "/run.json")
             .add_local_file(str(_root / "src/probe_shift/q1_eval.py"), "/root/q1_eval.py")
             .add_local_file(str(HERE / "cleanroom_reduce_v2.py"), "/root/cleanroom_reduce_v2.py")
             .add_local_dir(os.environ["PL_CFG"], "/pl"))


def tol(x):
    return RTOL * max(1.0, abs(x))


def enc(s):
    return s.replace("@", "_at_").replace("|", "_").replace(" ", "_").replace("(", "").replace(")", "")


def origin_key(s):
    return f"{s} (all-origins draws) [pooled: {COMBO}]"


def solve(cells, w, zmax, T, delta):
    """Exact draw-0 plant for one origin: cells, their within-origin weights w, their largest sim unit zmax (g_sim), threshold T.
    delta > 0: an uncaught cell (zmax <= T) is caught once c > t = T - zmax; delta < 0: a caught cell (zmax > T) falls once c <= t.
    Returns status, the achieved change (signed, catch fraction), and (lo, hi, lo_open, hi_open): every c in that interval gives
    exactly that change; t ties enter together. Unattainable: the whole reachable change is < |delta| / 100."""
    import math

    target = abs(delta) / 100.0
    if delta > 0:
        cand = sorted((T - zmax[c], w[c]) for c in cells if math.isfinite(zmax[c]) and not zmax[c] > T)
    else:
        cand = sorted(((T - zmax[c], w[c]) for c in cells if zmax[c] > T), reverse=True)
    groups = []
    for t, wt in cand:
        if groups and groups[-1][0] == t:
            groups[-1][1] += wt
        else:
            groups.append([t, wt])
    cum = 0.0
    for i, (t, wt) in enumerate(groups):
        cum += wt
        if cum >= target - CUM_EPS:
            nxt = groups[i + 1][0] if i + 1 < len(groups) else (math.inf if delta > 0 else -math.inf)
            if delta > 0:
                return "attainable", cum, (t, nxt, True, False)
            return "attainable", -cum, (nxt, t, True, False)
    return "unattainable", (cum if delta > 0 else -cum), None


@app.function(image=image, cpu=8, memory=32768, timeout=3600, retries=0,
              volumes={"/s": modal.Volume.from_name(SCORES, version=2).read_only(),
                       "/p": modal.Volume.from_name(POOLED, version=2).read_only(),
                       "/a": modal.Volume.from_name(ACTS, version=2).read_only(),
                       "/hf": modal.Volume.from_name(HF).read_only(),
                       "/o": OUT_VOL})
def run(stamp: str) -> dict:
    import gzip
    import hashlib
    import math

    import numpy as np

    def sha(b):
        return hashlib.sha256(b).hexdigest()

    def fsha(p):
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        return h.hexdigest()

    def num(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)

    stops, rows, controls = [], [], []

    def row(fam, key, m, ref, bound, verdict=True, kind="value", note=None, scale=1.0):
        r = {"family": fam, "key": key, "verdict": verdict, "kind": kind, "mine": m, "ref": ref, "bound": bound, "note": note}
        if kind == "bool":
            r["match"] = bool(isinstance(ref, bool) and m is ref)
        elif kind == "int":
            r["match"] = bool(isinstance(ref, int) and not isinstance(ref, bool) and m == ref)
        elif kind == "cat":
            r["match"] = bool(m == ref)
        elif kind == "unresolved":
            r["match"] = None
        else:
            mm = None if m is None else scale * m
            r["mine_scaled"] = mm
            r["match"] = bool(num(mm) and num(ref) and abs(mm - ref) <= scale * (bound or 0.0) + tol(ref))
        rows.append(r)

    def ci_rows(fam, key, ci, ref):  # 21c (Codex): withheld on both sides is agreement; withheld on one side is a mismatch
        if ci is None or not ref:
            row(fam, f"{key}.withheld", ci is None, not ref, None, kind="bool")
            return
        for i in (0, 1):
            row(fam, f"{key}[{i}]", ci[i], ref[i] if len(ref) > i else None, 0.0, scale=SCALE)

    # ---- pins: the reducer is the reviewed, unchanged file; every PLANT / A input by sha
    assert sha(Path("/root/cleanroom_reduce_v2.py").read_bytes()) == REDUCER_SHA, "reducer sha"
    graw = Path("/pl", PLANT_GZ[0]).read_bytes()
    assert sha(graw) == PLANT_GZ[1], "PLANT gz sha"
    praw = gzip.decompress(graw)
    assert sha(praw) == PLANT_GZ[2], "PLANT decompressed sha"
    P = json.loads(praw)
    del graw, praw
    prow_b = Path("/pl", PLANT_ROWS[0]).read_bytes()
    assert sha(prow_b) == PLANT_ROWS[1], "plant_rows sha"
    PR = json.loads(prow_b)
    vraw = Path("/o", A18_VALUES[0]).read_bytes()
    assert sha(vraw) == A18_VALUES[1], "A18 values sha"
    V18 = json.loads(vraw)["values"]
    craw = Path("/o", A18_CMP[0]).read_bytes()
    assert sha(craw) == A18_CMP[1], "A18 comparison sha"
    fb18 = {r["what"]: float(r.get("flip_bound") or 0.0) for r in json.loads(craw)["comparison"]}
    del vraw, craw
    for nm, (p, want) in (("PLANT npz", PLANT_NPZ), ("A npz", A_NPZ)):
        assert fsha(Path("/p", p)) == want, f"{nm} sha"
    if (P.get("npz") or {}).get("sha256") != PLANT_NPZ[1] or (P.get("npz") or {}).get("path") != PLANT_NPZ[0]:
        stops.append("PLANT record's npz pin != the pinned PLANT npz")
    cfgp = (PR.get("meta") or {}).get("config") or {}
    if (cfgp.get("record") or {}).get("decompressed_sha256") != PLANT_GZ[2]:
        stops.append("plant_rows is not built on the pinned PLANT record")
    if cfgp.get("combo") != COMBO or cfgp.get("mode") != MODE or sorted(cfgp.get("probes") or []) != sorted(SIX):
        stops.append("plant_rows combo / mode / probes != registered")
    if sorted(PR.get("levels") or []) != sorted(LVLS) or sorted(cfgp.get("deltas") or []) != sorted(DELTAS):
        stops.append("plant_rows levels / deltas != pre-stated")

    # ---- draw 0: my unchanged reducer, A18 config; capture the per-label results instead of comparing them
    sys.path.insert(0, "/root")
    import cleanroom_reduce_v2 as R

    cap = {}

    def capture(report, results, six, allp, main, stamp_, t0, run_cfg):
        cap.update(results=results, main=main, run_cfg=run_cfg)
        return {}

    R.compare_and_control = capture
    R.run_stage.local("compute", stamp)  # the reducer's own compute stage, in this process; only its final comparison is replaced
    res, main = cap["results"], cap["main"]
    assert cap["run_cfg"].get("tag") == "A18", "run config tag"
    # control 0: the rerun reproduces my committed A18 values exactly (same code, same inputs)
    keys0 = ("tpr", "real_mean", "delta_full", "sim_minus_run1", "sim_minus_run2", "run2_minus_run1")
    eq0 = []
    for lab, r in res.items():
        for lvl, d in r["levels"].items():
            v = V18[lab]["levels"][lvl]
            eq0.append(all(d[k] == v[k] for k in keys0) and d["tau_S"]["T"] == v["tau_S"]["T"] and d["tau_B"]["T"] == v["tau_B"]["T"])
    controls.append({"name": "C0_reproduces_A18_values", "n": len(eq0), "n_true": int(sum(eq0)), "pass": bool(eq0) and all(eq0)})
    if not all(eq0):
        stops.append("draw-0 rerun != committed A18 values")

    def nofb(lab, lvl, q):
        return fb18.get(f"{lab}|{MODE}|{lvl}|{q}", None)

    # ---- the plant solve per label x level x origin x delta, and the planted draw-0 points
    # 21e: the per-origin catch's weighting is IDENTIFIED (control C6 must hold for all 96 under one reading), then used for the solve:
    #   "equal_main": main weight 1 / n_mains in the origin, split over the main's cells in the origin (21's reading);
    #   "pooled_w":   the pooled, main-deduplicated cell weights (W_main / n_main over the label's cells in ALL origins), renormalized
    #                 within the origin (A meta.merge.rule: "a main in several origins is ONE main ... cell weight W_main / n_main").
    def origin_weights(r, cells, reading):
        if reading == "equal_main":
            bm = {}
            for c in cells:
                bm.setdefault(main[c], []).append(c)
            return {c: 1.0 / (len(bm) * len(v)) for v in bm.values() for c in v}
        tot = float(sum(r["weight"][c] for c in cells))
        return {c: r["weight"][c] / tot for c in cells}

    READINGS = ("equal_main", "pooled_w")
    ok6 = {rd: [] for rd in READINGS}
    for n in SIX:
        for ro in ("C", "R"):
            lab = f"{n} {ro}"
            r = res[lab]
            zmax = {c: r["zmax"][f"sim|{c}"] for s in SETS for c in r["cells_by_set"][s]}
            for lvl in LVLS:
                T = r["levels"][lvl]["tau_S"]["T"]
                for s in SETS:
                    cells = r["cells_by_set"][s]
                    node = ((P["rows"][lab].get(origin_key(s)) or {}).get("points") or {})
                    pt = (((node.get(MODE) or {}).get(lvl) or {}).get("catch") or {}).get("sim@S|primary") or {}
                    for rd in READINGS:
                        w_ = origin_weights(r, cells, rd)
                        c0 = float(sum(w_[c] for c in cells if zmax[c] > T))
                        ok6[rd].append(bool(num(pt.get("point")) and abs(c0 - pt["point"]) <= (nofb(lab, lvl, "catch_sim") or 0.0) + tol(pt["point"])))
    WREAD = next((rd for rd in READINGS if ok6[rd] and all(ok6[rd])), None)
    controls.append({"name": "C6_origin_catch0", "n": len(ok6["equal_main"]), "n_true_by_reading": {rd: int(sum(v)) for rd, v in ok6.items()},
                     "reading": WREAD, "pass": WREAD is not None})
    if WREAD is None:
        stops.append("per-origin catch weighting not identified (C6): no reading reproduces all 96 per-origin points")
        return {"stamp": stamp, "n_stops": len(stops), "stops": stops[:40], "controls": controls}
    mine = {}
    for n in SIX:
        for ro in ("C", "R"):
            lab = f"{n} {ro}"
            r = res[lab]
            zmax = {c: r["zmax"][f"sim|{c}"] for s in SETS for c in r["cells_by_set"][s]}
            wP = r["weight"]
            for lvl in LVLS:
                d = r["levels"][lvl]
                T = d["tau_S"]["T"]
                caught0 = {c: zmax[c] > T for c in zmax}
                per = {}
                for s in SETS:
                    cells = r["cells_by_set"][s]
                    wO = origin_weights(r, cells, WREAD)
                    c0 = float(sum(wO[c] for c in cells if caught0[c]))
                    per[s] = {"cells": cells, "w": wO, "catch0": c0, "solve": {}}
                    for dl in DELTAS:
                        per[s]["solve"][dl] = solve(cells, wO, zmax, T, dl)
                mine[(lab, lvl)] = {"T": T, "per": per, "d": d, "zmax": zmax, "wP": wP, "caught0": caught0}

    def planted(lab, lvl, dl):
        """Pooled draw-0 planted sim catch with each origin's own minimal plant; None if any origin is unattainable."""
        m = mine[(lab, lvl)]
        if any(m["per"][s]["solve"][dl][0] != "attainable" for s in SETS):
            return None
        tp = 0.0
        for s in SETS:
            lo, hi, _, _ = m["per"][s]["solve"][dl][2]
            for c in m["per"][s]["cells"]:
                if dl > 0:
                    cc = m["caught0"][c] or (math.isfinite(m["zmax"][c]) and (m["T"] - m["zmax"][c]) <= lo)
                else:
                    cc = m["caught0"][c] and not ((m["T"] - m["zmax"][c]) >= hi)
                tp += m["wP"][c] * float(cc)
        return tp

    def near_w_origin(m, s, dl):  # within-origin weight of cells whose breakpoint is within 2b of my step's ends
        sv = m["per"][s]["solve"][dl]
        if sv[2] is None:
            return 0.0
        ends = [e for e in sv[2][:2] if math.isfinite(e)]
        return float(sum(m["per"][s]["w"][c] for c in m["per"][s]["cells"] if math.isfinite(m["zmax"][c])
                         and any(abs((m["T"] - m["zmax"][c]) - e) <= 2 * tol(e) for e in ends)))

    def near_w(lab, lvl, dl):  # weight of the cells whose plant breakpoint is within 2b of my step's ends (ties with the producer's c)
        m = mine[(lab, lvl)]
        tot = 0.0
        for s in SETS:
            sv = m["per"][s]["solve"][dl]
            if sv[2] is None:
                continue
            ends = [e for e in sv[2][:2] if math.isfinite(e)]
            for c in m["per"][s]["cells"]:
                if math.isfinite(m["zmax"][c]):
                    t = m["T"] - m["zmax"][c]
                    if any(abs(t - e) <= 2 * tol(e) for e in ends):
                        tot += m["wP"][c]
        return tot

    # ---- (A) the record's per-origin solves vs mine (verdict: status, achieved change, c inside my exact step)
    def in_step(c, iv):  # c within my exact step (lo exclusive, hi inclusive), widened by 2b at each finite end
        lo, hi = iv[0], iv[1]
        return bool(num(c) and (not math.isfinite(lo) or c > lo - 2 * tol(lo)) and (not math.isfinite(hi) or c <= hi + 2 * tol(hi)))

    sv_votes = {1.0: 0, 100.0: 0}  # the solve record's scale (catch0 / achieved_points): identified on catch0, one scale
    for (lab, lvl), m in mine.items():
        for s in SETS:
            rec = ((((P["rows"][lab].get(origin_key(s)) or {}).get("plant") or {}).get("solve_draw0") or {}).get(lvl) or {}).get("sim") or {}
            for k_s in sv_votes:
                bd = tol(rec["catch0"]) + k_s * (nofb(lab, lvl, "catch_sim") or 0.0) if num(rec.get("catch0")) else 0.0
                sv_votes[k_s] += num(rec.get("catch0")) and abs(k_s * m["per"][s]["catch0"] - rec["catch0"]) <= bd
    SSC = 100.0 if sv_votes[100.0] > sv_votes[1.0] else 1.0
    controls.append({"name": "C7_solve_scale", "votes": {str(k): v for k, v in sv_votes.items()}, "scale": SSC,
                     "pass": sv_votes[SSC] == 4 * len(mine)})
    ac_votes, ac_n = {1.0: 0, 100.0: 0}, 0  # 21c (R1 P0-2): achieved_points' own scale, identified on arms attainable on both sides
    for (lab, lvl), m in mine.items():
        for s in SETS:
            rec = ((((P["rows"][lab].get(origin_key(s)) or {}).get("plant") or {}).get("solve_draw0") or {}).get(lvl) or {}).get("sim") or {}
            for dl in DELTAS:
                st, ach, _ = m["per"][s]["solve"][dl]
                ra = (rec.get("arms") or {}).get(f"{dl:+d}") or {}
                if st == "attainable" and str(ra.get("status", "")).lower() == "ok" and num(ra.get("achieved_points")):
                    ac_n += 1
                    for k_s in ac_votes:
                        ac_votes[k_s] += abs(k_s * ach - ra["achieved_points"]) <= tol(ra["achieved_points"]) + k_s * (
                            (nofb(lab, lvl, "catch_sim") or 0.0) + near_w_origin(m, s, dl))
    SAC = 100.0 if ac_votes[100.0] > ac_votes[1.0] else 1.0
    controls.append({"name": "C8_achieved_scale", "votes": {str(k): v for k, v in ac_votes.items()}, "n": ac_n, "scale": SAC,
                     "pass": ac_n > 0 and ac_votes[SAC] == ac_n})
    for (lab, lvl), m in mine.items():
        for s in SETS:
            rec = ((((P["rows"][lab].get(origin_key(s)) or {}).get("plant") or {}).get("solve_draw0") or {}).get(lvl) or {}).get("sim") or {}
            row("plant_solve", f"{lab}|{lvl}|{s}|T0", m["T"], rec.get("T0"), 0.0)
            row("plant_solve", f"{lab}|{lvl}|{s}|catch0", m["per"][s]["catch0"], rec.get("catch0"), nofb(lab, lvl, "catch_sim") or 0.0, scale=SSC)
            for dl in DELTAS:
                st, ach, iv = m["per"][s]["solve"][dl]
                ra = (rec.get("arms") or {}).get(f"{dl:+d}") or {}
                rst = str(ra.get("status", "")).lower()  # 21c (Codex): whitelist "ok" / "unattainable"; anything else is a mismatch
                rcat = "unattainable" if rst == "unattainable" else "attainable" if rst == "ok" else (f"other:{rst}" if rst else None)
                row("plant_solve", f"{lab}|{lvl}|{s}|{dl:+d}|status", st, rcat, None, kind="cat")
                if st == "attainable" and rcat == "attainable":  # 21c (R1 / R2): unattainable arms carry no achieved_points or c
                    row("plant_solve", f"{lab}|{lvl}|{s}|{dl:+d}|achieved", ach, ra.get("achieved_points"),
                        (nofb(lab, lvl, "catch_sim") or 0.0) + near_w_origin(m, s, dl), scale=SAC)
                    row("plant_solve", f"{lab}|{lvl}|{s}|{dl:+d}|c_in_my_step", in_step(ra.get("c"), iv), True, None, kind="bool")
                if rcat is None:
                    stops.append(f"PLANT solve arm absent: {lab} {lvl} {s} {dl:+d}")

    # ---- (B) per-probe planted draw-0 points (independent) and the npz-based intervals / verdicts (shared)
    ZP = np.load(Path("/p", PLANT_NPZ[0]))
    ZA = np.load(Path("/p", A_NPZ[0]))
    namesP, namesA = set(ZP.files), set(ZA.files)

    def k_(lab, lvl, name):
        return "__".join([enc(lab), enc(COMBO), MODE, lvl, "pooled", name])

    absentP = []

    def getP(lab, lvl, name, dl=None):  # 21e: PLANT stores no planted arrays for some unattainable plants
        k = k_(lab, lvl, name)
        if k not in namesP:
            if dl is not None and dl != 0 and planted(lab, lvl, dl) is None:
                absentP.append(k)  # allowed: my plant is unavailable too; the row's verdict is "unavailable"
            else:
                stops.append(f"PLANT npz key absent: {k}")
            return None
        return np.asarray(ZP[k], dtype=float)

    def getA(lab, lvl, name):
        k = k_(lab, lvl, name)
        if k not in namesA:
            stops.append(f"A npz key absent: {k}")
            return None
        return np.asarray(ZA[k], dtype=float)

    arrs = {}
    for n in SIX:
        for ro in ("C", "R"):
            lab = f"{n} {ro}"
            for lvl in LVLS:
                for dl in (0,) + DELTAS:
                    arrs[(lab, lvl, dl, "dfull")] = getP(lab, lvl, f"gap_delta_full_plant_{dl:+d}", dl)
                    arrs[(lab, lvl, dl, "csim")] = getP(lab, lvl, f"catch_sim_plant_{dl:+d}", dl)
                for g in ("delta_full", "sim_minus_run1", "sim_minus_run2", "run2_minus_run1_at_tauB"):
                    arrs[(lab, lvl, "A", g)] = getA(lab, lvl, f"gap_{g}")
    if stops:
        return {"stamp": stamp, "n_stops": len(stops), "stops": stops[:40], "controls": controls}

    # control 1: PLANT's unplanted arrays are A's, draw for draw (same draws)
    same = [bool(np.array_equal(arrs[(lab, lvl, 0, "dfull")], arrs[(lab, lvl, "A", "delta_full")], equal_nan=True))
            for (lab, lvl) in mine]
    controls.append({"name": "C1_plant0_equals_A_draws", "n": len(same), "n_true": int(sum(same)), "pass": all(same)})
    # control 2: the per-draw catch shift is the delta_full shift (scale 1), every finite draw
    sc = []
    for (lab, lvl) in mine:
        for dl in DELTAS:
            if arrs[(lab, lvl, dl, "dfull")] is None or arrs[(lab, lvl, dl, "csim")] is None:  # 21e: absent only for unavailable plants
                continue
            shp = {arrs[(lab, lvl, x, y)].shape for x in (dl, 0) for y in ("dfull", "csim")}
            if len(shp) != 1:  # 21d (Codex probe): unequal shapes fail C2 instead of raising
                sc.append(False)
                continue
            a = arrs[(lab, lvl, dl, "dfull")] - arrs[(lab, lvl, 0, "dfull")]
            b = arrs[(lab, lvl, dl, "csim")] - arrs[(lab, lvl, 0, "csim")]
            f = np.isfinite(a) & np.isfinite(b)
            need = planted(lab, lvl, dl) is not None  # 21d (Codex): only an attainable plant must have finite draws; unavailable may be all-NaN
            sc.append(bool(a.shape == b.shape and np.array_equal(np.isfinite(a), np.isfinite(b)) and (f.any() or not need)
                           and np.all(np.abs(a[f] - b[f]) <= RTOL * np.maximum(1.0, np.abs(a[f])))))  # 21c (Codex): same finite support
    controls.append({"name": "C2_catch_shift_is_gap_shift", "n": len(sc), "n_true": int(sum(sc)), "pass": all(sc), "n_absent_unavailable": len(absentP)})

    def summ(x, idx, method):
        x = x[1:] if idx == "drop0" else x
        f = x[np.isfinite(x)]
        ok = len(f) >= 0.95 * len(x) and len(f) > 0  # the registered summ: withheld below 95% complete
        ci = [float(v) for v in np.percentile(f, [2.5, 97.5], method=method)] if ok else None
        return {"draws": int(len(x)), "finite_draws": int(len(f)), "ci95": ci, "mean": float(np.mean(f)) if len(f) else None}

    def same_node(s_, node):
        c = node.get("ci95")
        if s_["ci95"] is None or not c:
            return s_["ci95"] is None and not c and s_["draws"] == node.get("draws") and s_["finite_draws"] == node.get("finite_draws")
        return (s_["draws"] == node.get("draws") and s_["finite_draws"] == node.get("finite_draws")
                and all(num(c[i]) and abs(s_["ci95"][i] - c[i]) <= tol(c[i]) for i in (0, 1)))

    # control 3: identify the index rule and percentile method on PLANT's own unplanted pooled delta_full nodes (24)
    nodes = {(lab, lvl): P["pooled"][lab][COMBO][MODE][lvl]["gap"]["delta_full"] for (lab, lvl) in mine}
    rules_ = {("all" if len(arrs[(lab, lvl, 0, "dfull")]) == nd.get("draws") else
               "drop0" if len(arrs[(lab, lvl, 0, "dfull")]) == (nd.get("draws") or -9) + 1 else None) for (lab, lvl), nd in nodes.items()}
    IDX = rules_.pop() if len(rules_) == 1 and None not in rules_ else None
    METHOD = next((mth for mth in METHODS if IDX and all(same_node(summ(arrs[(lab, lvl, 0, "dfull")], IDX, mth), nd)
                                                         for (lab, lvl), nd in nodes.items())), None)
    controls.append({"name": "C3_member_unplanted", "n": len(nodes), "idx_rule": IDX, "method": METHOD, "pass": bool(IDX and METHOD)})

    def cat_of(v):
        s_ = str(v or "").lower()
        if "unavail" in s_ or "unattain" in s_:
            return "unavailable"
        if "not testable" in s_:
            return "not_testable"
        if s_.startswith("not") or "not detected" in s_:
            return "not_detected"
        if s_.startswith("detected"):
            return "detected"
        return f"other:{bool(s_)}"

    # identify plant_rows' point scale on its unplanted points (x1 or x100), one scale for every *_points field
    sc_votes = {"x1": 0, "x100": 0}
    for lvl in LVLS:
        for ro in ("C", "R"):
            for n in SIX:
                for dl in DELTAS:
                    pr = (((PR["rows"].get(lvl) or {}).get(ro) or {}).get(f"{n} {ro}") or {}).get(f"{dl:+d}") or {}  # 21c: keyed by label
                    u, mu = pr.get("unplanted_point"), mine[(f"{n} {ro}", lvl)]["d"]["delta_full"]
                    if num(u):
                        sc_votes["x1"] += abs(u - mu) <= tol(u) + (nofb(f"{n} {ro}", lvl, "delta_full") or 0)
                        sc_votes["x100"] += abs(u - 100 * mu) <= tol(u) + 100 * (nofb(f"{n} {ro}", lvl, "delta_full") or 0)
    SCALE = 100.0 if sc_votes["x100"] > sc_votes["x1"] else 1.0
    n_votes = sum(1 for lvl in LVLS for ro in ("C", "R") for n in SIX for dl in DELTAS
                  if num(((((PR["rows"].get(lvl) or {}).get(ro) or {}).get(f"{n} {ro}") or {}).get(f"{dl:+d}") or {}).get("unplanted_point")))
    controls.append({"name": "C4_points_scale", "votes": sc_votes, "n": n_votes, "scale": SCALE,
                     "pass": n_votes > 0 and sc_votes["x100" if SCALE == 100.0 else "x1"] == n_votes})

    verd = {}
    ok_iv = bool(IDX and METHOD and all(same) and all(sc))
    for (lab, lvl), m in mine.items():
        n, ro = lab.rsplit(" ", 1)
        d = m["d"]
        bnd = nofb(lab, lvl, "delta_full") or 0.0
        for dl in DELTAS:
            pr = (((PR["rows"].get(lvl) or {}).get(ro) or {}).get(lab) or {}).get(f"{dl:+d}") or {}  # 21c: keyed by label
            k0 = f"rows.{lvl}.{ro}.{n}.{dl:+d}"
            tp = planted(lab, lvl, dl)
            if tp is None:
                verd[(lab, lvl, dl)] = {"cat": "unavailable", "mm": False}
                row("plant_rows", f"{k0}.verdict", "unavailable", cat_of(pr.get("verdict")), None, kind="cat")
                continue
            pl_df = tp - d["real_mean"]
            move = 100 * (pl_df - d["delta_full"])  # points
            nw = near_w(lab, lvl, dl)
            in_rng = abs(move - dl) <= 0.5 * abs(dl)
            row("plant_rows", f"{k0}.planted_point", pl_df, pr.get("planted_point"), bnd + nw, scale=SCALE)
            row("plant_rows", f"{k0}.unplanted_point", d["delta_full"], pr.get("unplanted_point"), bnd, scale=SCALE)
            row("plant_rows", f"{k0}.move_points", move / 100, pr.get("move_points"), 2 * bnd + nw, scale=SCALE)
            row("plant_rows", f"{k0}.move_in_range", bool(in_rng), pr.get("move_in_range"), None, kind="bool")
            mref = pr.get("mechanism_mismatch")  # 21c (R1 P1-2): null in the record -> a diagnostic row; counts carry the totals
            row("plant_rows", f"{k0}.mechanism_mismatch", not in_rng, mref, None, verdict=mref is not None, kind="bool",
                note=None if mref is not None else "record null")
            row("plant_rows", f"{k0}.readable", bool(d["tau_B"]["readable"] and d["tau_S"]["readable"]), pr.get("readable"), None, kind="bool")
            if not ok_iv:
                for part in ("planted_ci95", "draws", "finite_draws", "interval_excludes_unplanted", "verdict"):
                    row("plant_rows", f"{k0}.{part}", None, None, None, kind="unresolved", note="control C1 / C2 / C3 failed")
                verd[(lab, lvl, dl)] = {"cat": "unresolved", "mm": not in_rng}  # 21c (R1 / R2 / Codex): no crash downstream
                continue
            s_ = summ(arrs[(lab, lvl, dl, "dfull")], IDX, METHOD)
            ci = s_["ci95"]
            ci_rows("plant_rows", f"{k0}.planted_ci95", ci, pr.get("planted_ci95"))
            row("plant_rows", f"{k0}.draws", s_["draws"], pr.get("draws"), None, kind="int")
            row("plant_rows", f"{k0}.finite_draws", s_["finite_draws"], pr.get("finite_draws"), None, kind="int")
            excl = ci is not None and not (ci[0] <= d["delta_full"] <= ci[1])
            row("plant_rows", f"{k0}.interval_excludes_unplanted", bool(excl), pr.get("interval_excludes_unplanted"), None, kind="bool")
            cat = "detected" if (in_rng and excl) else "not_detected"
            verd[(lab, lvl, dl)] = {"cat": cat, "mm": not in_rng}
            row("plant_rows", f"{k0}.verdict", cat, cat_of(pr.get("verdict")), None, kind="cat")

    # ---- (C) counts per level x delta x readout (C, R, C+R)
    for lvl in LVLS:
        for dl in DELTAS:
            for rk, ros in (("C", ("C",)), ("R", ("R",)), ("C+R", ("C", "R"))):
                vs = [(n, verd[(f"{n} {ro}", lvl, dl)]) for ro in ros for n in SIX]
                node = (((PR.get("counts") or {}).get(lvl) or {}).get(f"{dl:+d}") or {}).get(rk) or {}
                if any(v["cat"] == "unresolved" for _, v in vs):  # 21c: counts that depend on an unresolved verdict are unresolved
                    for k in ("cases", "detected", "not_detected", "unavailable", "mechanism_mismatch") + (("misses.keys", "misses.values") if rk != "C+R" else ()):
                        row("counts", f"counts.{lvl}.{dl:+d}.{rk}.{k}", None, None, None, kind="unresolved", note="a member verdict is unresolved")
                    continue
                cnt = {"cases": len(vs), "detected": sum(v["cat"] == "detected" for _, v in vs),
                       "not_detected": sum(v["cat"] == "not_detected" for _, v in vs),
                       "unavailable": sum(v["cat"] == "unavailable" for _, v in vs),
                       "mechanism_mismatch": sum(v["mm"] for _, v in vs)}
                node = (((PR.get("counts") or {}).get(lvl) or {}).get(f"{dl:+d}") or {}).get(rk) or {}
                for k, v in cnt.items():
                    row("counts", f"counts.{lvl}.{dl:+d}.{rk}.{k}", int(v), node.get(k), None, kind="int")
                if rk in ("C", "R"):  # 21b (09: row 49 reads misses): its key set and mapped values == my not-detected / unavailable probes
                    mm = node.get("misses")
                    mine_m = {n: v["cat"] for n, v in vs if v["cat"] != "detected"}
                    if isinstance(mm, dict):
                        ents = [(str(k).removesuffix(f" {rk}"), cat_of(val.get("verdict") if isinstance(val, dict) else val)) for k, val in mm.items()]
                        keys_ = [k for k, _ in ents]  # 21c (Codex P2): every original entry is checked; a normalized-key collision fails
                        row("counts", f"counts.{lvl}.{dl:+d}.{rk}.misses.keys", sorted(mine_m), sorted(keys_) if len(set(keys_)) == len(keys_) else
                            ["<collision>"] + sorted(keys_), None, kind="cat")
                        row("counts", f"counts.{lvl}.{dl:+d}.{rk}.misses.values", sorted(mine_m.items()), sorted(ents), None, kind="cat")
                    else:
                        row("counts", f"counts.{lvl}.{dl:+d}.{rk}.misses.keys", sorted(mine_m), None, None, kind="cat", note="misses absent or not a map")

    # ---- (D) the six-probe aggregate, C: S and the excess (negative delta verdicts; positive delta "unattainable" members, item 47)
    for lvl in LVLS:
        labs = [f"{n} C" for n in SIX]
        ds = [mine[(lab, lvl)]["d"] for lab in labs]
        for dl in DELTAS:
            node = ((((PR.get("aggregate") or {}).get("by_level") or {}).get(lvl) or {}).get("C") or {}).get(f"{dl:+d}") or {}
            k0 = f"aggregate.by_level.{lvl}.C.{dl:+d}"
            tps = [planted(lab, lvl, dl) for lab in labs]
            unatt = sorted(n for n, t in zip(SIX, tps) if t is None)
            for stat in ("S", "excess"):
                nd = node.get(stat) or {}
                if unatt:
                    row("aggregate", f"{k0}.{stat}.verdict", "unavailable", cat_of(nd.get("verdict")), None, kind="cat")
                    row("aggregate", f"{k0}.{stat}.unattainable_members", sorted(f"{n} C" for n in unatt), sorted(nd.get("unattainable_members") or []),
                        None, kind="cat")  # 21c (R1 / R2): the record lists labels
                    continue
                bnds = [nofb(lab, lvl, "delta_full") or 0.0 for lab in labs]
                nws = [near_w(lab, lvl, dl) for lab in labs]
                pl = [t - d["real_mean"] for t, d in zip(tps, ds)]
                if stat == "S":
                    un_pt = float(np.mean([d["delta_full"] for d in ds]))
                    pl_pt = float(np.mean(pl))
                    b_pt = float(np.mean(bnds)) + float(np.mean(nws))
                    mv = 100 * (pl_pt - un_pt)
                    in_rng = abs(mv - dl) <= 0.5 * abs(dl)
                else:
                    def G(d, shift):
                        return 0.5 * (abs(d["sim_minus_run1"] + shift) + abs(d["sim_minus_run2"] + shift))
                    un_pt = float(np.mean([G(d, 0.0) for d in ds]) - np.mean([abs(d["run2_minus_run1"]) for d in ds]))
                    pl_pt = float(np.mean([G(d, t - d["tpr"]["sim"]) for d, t in zip(ds, tps)]) - np.mean([abs(d["run2_minus_run1"]) for d in ds]))
                    bnds = [nofb(lab, lvl, "G_minus_Y~r3") or 0.0 for lab in labs]  # 21c (Codex): the excess's own (r3) allowance
                    b_pt = float(np.mean(bnds)) + float(np.mean(nws))
                    dexp = 100 * (pl_pt - un_pt)
                    row("aggregate", f"{k0}.excess.delta_exp_points", (pl_pt - un_pt), nd.get("delta_exp_points"), 2 * b_pt, scale=SCALE)
                row("aggregate", f"{k0}.{stat}.planted.point", pl_pt, (nd.get("planted") or {}).get("point"), b_pt, scale=SCALE)
                row("aggregate", f"{k0}.{stat}.unplanted_point", un_pt, nd.get("unplanted_point"), float(np.mean(bnds)), scale=SCALE)
                if not ok_iv:
                    row("aggregate", f"{k0}.{stat}.verdict", None, None, None, kind="unresolved", note="control C1 / C2 / C3 failed")
                    continue
                # per-draw statistic on the common finite draws of the six (l.4936)
                if stat == "S":
                    Ms = np.vstack([arrs[(lab, lvl, dl, "dfull")] for lab in labs])
                    Mu = np.vstack([arrs[(lab, lvl, 0, "dfull")] for lab in labs])
                    pdr = np.where(np.all(np.isfinite(Ms), 0), Ms.mean(0), np.nan)
                    udr = np.where(np.all(np.isfinite(Mu), 0), Mu.mean(0), np.nan)
                else:
                    def Gd(lab, shift):
                        a1, a2 = arrs[(lab, lvl, "A", "sim_minus_run1")], arrs[(lab, lvl, "A", "sim_minus_run2")]
                        return 0.5 * (np.abs(a1 + shift) + np.abs(a2 + shift))
                    sh = {lab: arrs[(lab, lvl, dl, "csim")] - arrs[(lab, lvl, 0, "csim")] for lab in labs}
                    Gp = np.vstack([Gd(lab, sh[lab]) for lab in labs])
                    Gu = np.vstack([Gd(lab, 0.0) for lab in labs])
                    Yv = np.vstack([np.abs(arrs[(lab, lvl, "A", "run2_minus_run1_at_tauB")]) for lab in labs])
                    okp, oku = np.all(np.isfinite(Gp), 0) & np.all(np.isfinite(Yv), 0), np.all(np.isfinite(Gu), 0) & np.all(np.isfinite(Yv), 0)
                    pdr = np.where(okp, Gp.mean(0) - Yv.mean(0), np.nan)
                    udr = np.where(oku, Gu.mean(0) - Yv.mean(0), np.nan)
                s_ = summ(pdr, IDX, METHOD)
                ci = s_["ci95"]
                if stat == "S":  # control 5: my planted-S rule == PLANT's coheadline_signed_plant node (item 35 l.4831)
                    cn = ((((P.get("coheadline_signed_plant") or {}).get(COMBO) or {}).get(MODE) or {}).get(lvl) or {}).get("headline_six|C") or {}
                    cs = (cn.get(f"{dl:+d}") or {}).get("S") or {}
                    controls.append({"name": f"C5_planted_S|{lvl}|{dl:+d}", "pass": bool(same_node(s_, cs)) and num(cs.get("point"))
                                     and abs(pl_pt - cs["point"]) <= b_pt + tol(cs["point"])})
                ci_rows("aggregate", f"{k0}.{stat}.planted.ci95", ci, (nd.get("planted") or {}).get("ci95"))
                excl = ci is not None and not (ci[0] <= un_pt <= ci[1])
                row("aggregate", f"{k0}.{stat}.interval_excludes_unplanted", bool(excl), nd.get("interval_excludes_unplanted"), None, kind="bool")
                if stat == "S":
                    row("aggregate", f"{k0}.S.move_in_range", bool(in_rng), nd.get("move_in_range"), None, kind="bool")
                    cat = "detected" if (in_rng and excl) else "not_detected"
                    row("aggregate", f"{k0}.S.verdict", cat, cat_of(nd.get("verdict")), None, kind="cat")
                    continue
                # the excess: Delta_exp, three bootstrap-mean-move readings; the verdict on the reading that reproduces the record, r1 beside
                if IDX == "drop0":  # 21c (Codex): the bootstrap means use the bootstrap draws only, as the intervals do
                    pdr, udr = pdr[1:], udr[1:]
                fp, fu = np.isfinite(pdr), np.isfinite(udr)
                both = fp & fu
                mv = {"r1": float(np.mean(pdr[fp])) - un_pt, "r2": float(np.mean(pdr[both] - udr[both])),
                      "r3": float(np.mean(pdr[fp])) - float(np.mean(udr[fu]))}
                ref_m = nd.get("bootstrap_mean_move_points")
                ident = [k for k, v in mv.items() if num(ref_m) and abs(SCALE * v - ref_m) <= tol(ref_m)]
                rows.append({"family": "aggregate_excess_move", "key": f"{k0}.excess.move_reading", "verdict": False, "kind": "cat",
                             "mine": ident, "ref": None, "match": bool(ident)})
                testable = abs(dexp) >= 0.5

                def ev(move):
                    if not testable:
                        return "not_testable"
                    return "detected" if (abs(100 * move - dexp) <= 0.5 * abs(dexp) and excl) else "not_detected"
                rd = ident[0] if ident else None
                if rd:
                    row("aggregate", f"{k0}.excess.verdict", ev(mv[rd]), cat_of(nd.get("verdict")), None, kind="cat", note=f"reading {rd}")
                else:
                    row("aggregate", f"{k0}.excess.verdict", None, None, None, kind="unresolved", note="no reading reproduces bootstrap_mean_move_points")
                rows.append({"family": "aggregate_excess_r1", "key": f"{k0}.excess.verdict|r1", "verdict": False, "kind": "cat",
                             "mine": ev(mv["r1"]), "ref": cat_of(nd.get("verdict")), "match": ev(mv["r1"]) == cat_of(nd.get("verdict"))})

    # ---- summary (booleans and counts); values to the Volume
    vr = [r for r in rows if r["verdict"]]
    fam = {}
    for r in rows:
        f_ = fam.setdefault(r["family"], {"rows": 0, "verdict": 0, "verdict_match": 0, "unresolved": 0, "diag": 0, "diag_match": 0})
        f_["rows"] += 1
        if r["verdict"]:
            f_["verdict"] += 1
            f_["verdict_match"] += r["match"] is True
            f_["unresolved"] += r["kind"] == "unresolved"
        else:
            f_["diag"] += 1
            f_["diag_match"] += r["match"] is True
    failing = [r["key"] for r in vr if r["match"] is not True]
    n_pr_verdict = sum(1 for r in rows if r["family"] == "plant_rows" and r["key"].endswith(".verdict"))  # 21c: one per level x readout x probe x delta
    printed = [f"aggregate.by_level.{lv}.C.{dl:+d}.{st}.verdict" for lv in LVLS for dl in (-3, -5, -8, -12) for st in ("S", "excess")] + \
              [f"counts.0.01.{dl:+d}.C.{k}" for dl in DELTAS for k in ("cases", "detected", "not_detected", "unavailable", "mechanism_mismatch")] + \
              ["counts.0.01.-3.C.misses.keys", "counts.0.01.-3.C.misses.values"]  # 21b: row 49 (09)
    pmap = {r["key"]: r["match"] for r in rows if r["verdict"]}
    vals = {"rows": rows, "controls": controls, "idx_rule": IDX, "method": METHOD, "scale": SCALE, "stops": stops}
    vb = json.dumps(vals, default=float).encode()
    vpath = Path("/o", f"cleanroom_plant_values_{stamp}.json")
    with vpath.open("xb") as fh:
        fh.write(vb)
    OUT_VOL.commit()
    return {"stamp": stamp, "n_stops": len(stops), "stops": stops[:40], "controls": controls, "by_family": fam,
            "n_verdict": len(vr), "n_verdict_match": sum(r["match"] is True for r in vr),
            "n_unresolved": sum(r["kind"] == "unresolved" for r in vr), "n_failing_verdict": len(failing), "failing_verdict_keys": failing[:200],
            "n_plant_rows_verdict": n_pr_verdict,
            "printed": {"n": len(printed), "n_present": sum(k in pmap for k in printed), "n_match": sum(pmap.get(k) is True for k in printed),
                        "failing": [k for k in printed if pmap.get(k) is not True]},
            "values_file": {"volume": OUT, "path": vpath.name, "sha256": hashlib.sha256(vb).hexdigest()},
            "PASS": not stops and all(c["pass"] for c in controls) and not failing}


@app.local_entrypoint()
def main():
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    out = run.remote(stamp)
    path = HERE / "results" / f"cleanroom_plant_{stamp}.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    print(json.dumps({k: out.get(k) for k in ("n_stops", "n_verdict", "n_verdict_match", "n_unresolved", "n_failing_verdict", "PASS")}))
    print("wrote", path)
