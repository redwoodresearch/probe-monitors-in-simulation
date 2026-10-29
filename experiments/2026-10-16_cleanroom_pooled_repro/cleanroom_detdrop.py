"""Clean-room detector_vs_drop half on the commands-L48 attack detector (addendum 25, a1, 2026-10-20). Docstring route: "independent
recompute from the producer's stated definition (POST HOC; no RULE registration)"; Aswin 12:08:34 PDT via 0d, "sure check that, but
also check the formula if it seems correct"; 0d's formula reviewer passed the detector_vs_drop formula (only the detector changes).

Definitions (producer's stated text, read: run_detector_vs_drop.py module docstring @ 17f522e49 and the record's meta.definitions):
  cells     the pooled_all_origins cohort cells of the four sets (1,313); excluded and counted: no detector score, or non-finite y.
  x         the certified attack detector's held-out per-run score of the cell's SIMULATED run: refit3 (df7171a9 @ 80d75b27e)
            cells[<readout>].problems.c_simatk_vs_run1.run_scores["sim|<cell>"]; readouts commands L48 (new) / L63 (old), replies L48.
  y         the probe's per-task change sim - mean(run1, run2) of max_minus_tau (f = 1e-2, draw 0, pooled tau): MY per-task values
            (my unchanged reducer's draw-0 maxima, as in addendum 24).
  point     Spearman(x, y) per lead probe (C); null: Spearman(heldout_run2.run_scores["run2|<cell>"], run2 - run1 of max_minus_tau).
  flips     catch-flip counts (run1 caught & sim missed; sim caught & run1 missed; the same with run2) by quartile of x.
  interval  main-clustered: one default_rng([20261015, 41]), multinomial over the sorted canonical mains (refit3 run_main) of the
            cells, 10,000 draws shared by every analysis; 2.5 / 97.5 percentiles of finite draws (>= 95% finite). The exact call
            order, the percentile method and the quartile conventions are IDENTIFIED on the existing L63 record (51aeb15d @ 17f522e49):
            a variant counts only if it reproduces that record's points, intervals and quartile rows exactly; verdict rows are then
            against the NEW L48 record (a536b65f, ap-NuUKhGaNZaLhqdKopouezY).
x (refit3 scores) is a SHARED input. The laptop gets booleans and counts only.

  CR_CFG=<pinned clean-room inputs> DD_CFG=<refit3 + L63 + L48 records> uv run --no-project --python 3.12 --with modal==1.6.1 modal run cleanroom_detdrop.py
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
REDUCER_SHA = "991a34be9f290797bfbf89c543341e59990b53a72ec11bf045ec6d426cf21f30"
A18_VALUES = ("cleanroom_values_A18_2026-10-20_040452.json", "9a6c50f41e1e394f13f1cf0feda7ac5d2b2a8bcea6a0d0c2154df0f12de1bbe2")
DD_PINS = {"refit3.json": "df7171a98d82d28ae86653480be922d978fc36104c63f622b3a69fb43a074eac",  # git copies (refit3 @ 80d75b27e; L63 @ 17f522e49)
           "detdrop_L63.json": "51aeb15d2b4fe4a369063d4ee79914269002d5d8c2bf323ac0efb8ab6f61567c"}
L48 = ("xfer/pertask/detector_vs_drop_2026-10-20_125606_ap-NuUKhGaNZaLhqdKopouezY.json",  # q35solavg Volume (not yet in git)
       "a536b65f78a40582c4e9491866c566a2139db906279ae0100d2ca23eb357af2c")
SIX = ["sol_a_uncut", "FIBS|uncut", "harmfulness|uncut", "refusal|uncut", "high-stakes|uncut", "own_all_rs_uncut_L48"]
SETS = ["S_old", "S_new", "S_newmains", "S_paper493"]
READOUT_CELL = {"commands L48": "command|L48", "commands L63": "command|L63", "replies L48": "reply|L48"}
LVL, N_DRAWS, SEED = "0.01", 10000, [20261015, 41]
VARIANTS = ("multinomial_size", "multinomial_loop", "integers_bincount")
METHODS = ("linear", "hazen", "weibull", "median_unbiased", "normal_unbiased", "interpolated_inverted_cdf", "inverted_cdf",
           "averaged_inverted_cdf", "closest_observation", "lower", "higher", "nearest", "midpoint")
QBOUNDS = ("right", "left")  # numpy searchsorted side (25e, R1 P2: labels corrected): "right" gives Q_k = [e_{k-1}, e_k), "left" (e_{k-1}, e_k]
FLIP_SETS = ("point_set", "score_only")  # 25e (R1 P1): flips cell set = the point's included cells, or every cell with a numeric score
RTOL = 1e-9

app = modal.App(OUT)
OUT_VOL = modal.Volume.from_name(OUT, create_if_missing=True, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "threadpoolctl==3.6.0", "tokenizers==0.22.1")
if modal.is_local():
    _root = HERE.resolve().parents[1]
    image = (image.add_local_dir(os.environ["CR_CFG"], "/cfg")
             .add_local_file(str(HERE / "config/routec_A18_pins.json"), "/run.json")
             .add_local_file(str(_root / "src/probe_shift/q1_eval.py"), "/root/q1_eval.py")
             .add_local_file(str(HERE / "cleanroom_reduce_v2.py"), "/root/cleanroom_reduce_v2.py")
             .add_local_dir(os.environ["DD_CFG"], "/dd"))


def tol(x):
    return RTOL * max(1.0, abs(x))


def wrho(x, y, W):
    """Weighted Spearman rho for every row of W (draws x cells; integer repeat counts): weighted Pearson of the average ranks of the
    weight-repeated data, ties sharing one average rank. Vectorized over draws (x, y fixed)."""
    import numpy as np

    W = np.atleast_2d(np.asarray(W, dtype=float))
    if W.shape[1] == 0:  # 25c (Codex on 24, same code): no cells -> undefined
        return np.full(W.shape[0], np.nan)
    pos = W > 0

    def const(v):  # 25c: a constant value over the positive-weight cells -> undefined
        return np.where(pos, v, -np.inf).max(axis=1) == np.where(pos, v, np.inf).min(axis=1)

    bad = const(np.asarray(x, dtype=float)) | const(np.asarray(y, dtype=float))

    def wrank(v):
        o = np.argsort(v, kind="mergesort")
        vs = v[o]
        starts = np.flatnonzero(np.r_[True, vs[1:] != vs[:-1]])
        gid = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(vs)]))
        Wg = np.add.reduceat(W[:, o], starts, axis=1)
        rg = np.cumsum(Wg, axis=1) - 0.5 * Wg + 0.5
        R = np.empty_like(W)
        R[:, o] = rg[:, gid]
        return R

    A, B = wrank(x), wrank(y)
    S = W.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ma, mb = (W * A).sum(axis=1) / S, (W * B).sum(axis=1) / S
        da, db = A - ma[:, None], B - mb[:, None]
        va, vb = (W * da * da).sum(axis=1), (W * db * db).sum(axis=1)
        r = (W * da * db).sum(axis=1) / np.sqrt(va * vb)
    return np.where((S > 0) & (va > 0) & (vb > 0) & ~bad, r, np.nan)


def boundary(tag):
    """25d / 24c (Codex P1): nothing but a type-and-stage message leaves the container. The full traceback is written to the output
    Volume (it stays on Modal) and a fresh RuntimeError is raised after the except block (no __context__), so no record bytes ride
    the exception chain."""
    def deco(fn):
        import functools

        @functools.wraps(fn)
        def wrapper(stamp, *args, **kwargs):
            try:
                return fn(stamp, *args, **kwargs)
            except Exception as e:
                import traceback
                name = type(e).__name__
                tb = traceback.format_exc()
                try:  # best effort; the run fails below either way (25e / 24d: any failure here is dropped)
                    with Path("/o", f"{tag}_error_{stamp}.txt").open("x", encoding="utf-8", errors="backslashreplace") as fh:
                        fh.write(tb)
                    OUT_VOL.commit()
                except Exception:
                    pass
                del tb
            # 25e / 24d (R1 P2): raised after the except block, so the new exception has no __context__ at all
            raise RuntimeError(f"STOP: {name} in {tag} (traceback on the output Volume)")
        return wrapper
    return deco


@app.function(image=image, cpu=8, memory=32768, timeout=3600, retries=0,
              volumes={"/s": modal.Volume.from_name(SCORES, version=2).read_only(),
                       "/p": modal.Volume.from_name(POOLED, version=2).read_only(),
                       "/a": modal.Volume.from_name(ACTS, version=2).read_only(),
                       "/hf": modal.Volume.from_name(HF).read_only(),
                       "/o": OUT_VOL})
@boundary("cleanroom_detdrop")
def run(stamp: str) -> dict:
    import hashlib
    import math

    import numpy as np

    def sha(b):
        return hashlib.sha256(b).hexdigest()

    def num(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)

    stops, rows, controls = [], [], []

    def row(fam, key, m, ref, bound, verdict=True, kind="value"):
        r = {"family": fam, "key": key, "verdict": verdict, "kind": kind, "mine": m, "ref": ref}
        if kind == "int":
            r["match"] = bool(isinstance(ref, int) and not isinstance(ref, bool) and m == ref)
        elif kind == "bool":  # 25f (R1 P2): the withheld row is a bool row
            r["match"] = bool(isinstance(m, bool) and isinstance(ref, bool) and m == ref)
        elif kind == "unresolved":
            r["match"] = None
        else:
            r["match"] = bool(num(m) and num(ref) and abs(m - ref) <= (bound or 0.0) + tol(ref))
        rows.append(r)

    assert sha(Path("/root/cleanroom_reduce_v2.py").read_bytes()) == REDUCER_SHA, "reducer sha"
    raw = {f: Path("/dd", f).read_bytes() for f in DD_PINS}
    for f, want in DD_PINS.items():
        assert sha(raw[f]) == want, f"input sha {f}"
    R3, D63 = (json.loads(raw[f]) for f in ("refit3.json", "detdrop_L63.json"))
    del raw
    b48 = Path("/p", L48[0]).read_bytes()
    assert sha(b48) == L48[1], "L48 record sha"
    D48 = json.loads(b48)
    vraw = Path("/o", A18_VALUES[0]).read_bytes()
    assert sha(vraw) == A18_VALUES[1], "A18 values sha"
    V18 = json.loads(vraw)["values"]

    # ---- draw 0: my unchanged reducer under the A18 config (as in addenda 21 / 24)
    sys.path.insert(0, "/root")
    import cleanroom_reduce_v2 as R

    cap = {}

    def capture(report, results, six, allp, main, stamp_, t0, run_cfg):
        cap.update(results=results, main=main)
        return {}

    R.compare_and_control = capture
    R.run_stage.local("compute", stamp)
    res = cap["results"]
    eq0 = [all(d[k] == V18[lab]["levels"][lvl][k] for k in ("tpr", "real_mean", "delta_full")) for lab, r in res.items() for lvl, d in r["levels"].items()]
    controls.append({"name": "C0_reproduces_A18_values", "n": len(eq0), "n_true": int(sum(eq0)), "pass": all(eq0)})

    # ---- per-task values (my own), the cohort cells, mains from refit3 run_main
    cells = None
    pt = {}
    for n in SIX:
        lab = f"{n} C"
        r = res[lab]
        d = r["levels"][LVL]
        tB, tS = d["tau_B"]["T"], d["tau_S"]["T"]
        cl = sorted(c for s in SETS for c in r["cells_by_set"][s])
        cells = cl if cells is None else cells
        if cl != cells:
            stops.append(f"cohort cells differ across probes: {lab}")
        for c in cl:
            pt[(lab, c)] = {"run1": r["zmax"][f"run1|{c}"] - tB, "run2": r["zmax"][f"run2|{c}"] - tB, "sim": r["zmax"][f"sim|{c}"] - tS}
    run_main = R3.get("run_main") or {}
    cmain = {}
    for c in cells or []:
        m = run_main.get(f"sim|{c}")
        if m is None:
            stops.append(f"no refit3 run_main for sim|{c}")
        cmain[c] = m
    if len(cells or []) != (D48.get("result") or {}).get("cells") or len(cells or []) != (D63.get("result") or {}).get("cells"):
        stops.append("my cohort cell count != the records' result.cells")
    if stops:
        return {"stamp": stamp, "n_stops": len(stops), "stops": stops[:40], "controls": controls}
    mains = sorted(set(cmain.values()))
    midx = {m: i for i, m in enumerate(mains)}
    row("cohort", "result.mains (L48 record)", len(mains), (D48.get("result") or {}).get("mains"), None, kind="int")

    def draws(variant):
        rng = np.random.default_rng(SEED)
        M = len(mains)
        p = np.full(M, 1.0 / M)
        if variant == "multinomial_size":
            return rng.multinomial(M, p, size=N_DRAWS)
        if variant == "multinomial_loop":
            return np.stack([rng.multinomial(M, p) for _ in range(N_DRAWS)])
        return np.stack([np.bincount(rng.integers(0, M, M), minlength=M) for _ in range(N_DRAWS)])

    KINDS = ("sim_score_vs_drop", "null_run2_score_vs_run2_minus_run1")

    def base(readout):
        """Draw-independent parts per probe x kind: included cells, x, y, point, counts; and the quartile flips under every convention."""
        node = (R3.get("cells") or {}).get(READOUT_CELL[readout]) or {}
        xmaps = {"sim_score_vs_drop": ((((node.get("problems") or {}).get("c_simatk_vs_run1") or {}).get("run_scores")) or {}),
                 "null_run2_score_vs_run2_minus_run1": (((node.get("heldout_run2") or {}).get("run_scores")) or {})}
        prefix = {"sim_score_vs_drop": "sim", "null_run2_score_vs_run2_minus_run1": "run2"}
        yfun = {"sim_score_vs_drop": lambda v: v["sim"] - 0.5 * (v["run1"] + v["run2"]),
                "null_run2_score_vs_run2_minus_run1": lambda v: v["run2"] - v["run1"]}
        out = {}
        for n in SIX:
            lab = f"{n} C"
            for kind in KINDS:
                inc, no_score, no_y = [], 0, 0
                for c in cells:
                    xv, yv = xmaps[kind].get(f"{prefix[kind]}|{c}"), yfun[kind](pt[(lab, c)])
                    if not num(xv):
                        no_score += 1
                    elif not math.isfinite(yv):
                        no_y += 1
                    else:
                        inc.append((c, float(xv), float(yv)))
                x, y = np.array([t[1] for t in inc], dtype=float), np.array([t[2] for t in inc], dtype=float)
                pnt = float(wrho(x, y, np.ones((1, len(inc))))[0]) if inc else float("nan")  # 25d (Codex P2): empty -> undefined
                b = {"x": x, "y": y, "col": np.array([midx[cmain[t[0]]] for t in inc], dtype=np.intp), "point": pnt,
                     "n_cells": len(inc), "n_mains": len({cmain[t[0]] for t in inc}),
                     "excluded": {"no_detector_score": no_score, "no_finite_y": no_y}}
                if kind == "sim_score_vs_drop":
                    flips = {}
                    for fs in FLIP_SETS:  # caught = max_minus_tau > 0 from my per-task values; a -inf / nan arm is not caught
                        fc = ([(t[0], t[1]) for t in inc] if fs == "point_set" else
                              [(c, float(xmaps[kind][f"sim|{c}"])) for c in cells if num(xmaps[kind].get(f"sim|{c}"))])
                        if not fc:  # 25d (Codex P2): empty -> no table (C2 fails / the L48 row is UNRESOLVED)
                            continue
                        xf = np.array([t[1] for t in fc], dtype=float)
                        cg = {a: np.array([bool(pt[(lab, t[0])][a] > 0) for t in fc]) for a in ("run1", "run2", "sim")}
                        exc = no_score + no_y if fs == "point_set" else no_score
                        for qm in METHODS:
                            e = [float(v) for v in np.quantile(xf, [0.25, 0.5, 0.75], method=qm)]
                            for qb in QBOUNDS:
                                qi = np.searchsorted(e, xf, side=qb)  # right: x == e1 -> Q2; left: x == e1 -> Q1
                                byq = {}
                                for k in range(4):
                                    sl = qi == k
                                    byq[f"Q{k + 1}"] = {"n_cells": int(sl.sum()),
                                                        "run1_caught_sim_missed": int((cg["run1"] & ~cg["sim"] & sl).sum()),
                                                        "sim_caught_run1_missed": int((cg["sim"] & ~cg["run1"] & sl).sum()),
                                                        "run2_caught_sim_missed": int((cg["run2"] & ~cg["sim"] & sl).sum()),
                                                        "sim_caught_run2_missed": int((cg["sim"] & ~cg["run2"] & sl).sum())}
                                flips[(fs, qm, qb)] = {"quartile_edges": e, "by_quartile": byq, "excluded_cells": exc}
                    b["flips"] = flips
                out[(lab, kind)] = b
        return out

    def per_draw(B, Md):
        return {k: wrho(b["x"], b["y"], Md[:, b["col"]]) for k, b in B.items()}

    def ci_of(dr, method):
        f = dr[np.isfinite(dr)]
        return ([float(v) for v in np.percentile(f, [2.5, 97.5], method=method)] if len(f) >= 0.95 * N_DRAWS else None), int(len(f))

    def compare(fam, rec, readout, B, DR, method, qconv, verdict):
        for n in SIX:
            lab = f"{n} C"
            node = ((((rec.get("result") or {}).get("probes") or {}).get(lab) or {}).get(readout)) or {}
            for kind in KINDS:
                b, rf = B[(lab, kind)], node.get(kind) or {}
                k0 = f"{readout}|{lab}|{kind}"
                row(fam, f"{k0}.point", b["point"], rf.get("point"), 0.0, verdict)
                for fld in ("n_cells", "n_mains"):
                    row(fam, f"{k0}.{fld}", b[fld], rf.get(fld), None, verdict, kind="int")
                for fld in ("no_detector_score", "no_finite_y"):
                    row(fam, f"{k0}.excluded.{fld}", b["excluded"][fld], (rf.get("excluded") or {}).get(fld), None, verdict, kind="int")
                if DR is None:
                    row(fam, f"{k0}.interval", None, None, None, verdict, kind="unresolved")
                    continue
                ci, nf = ci_of(DR[(lab, kind)], method)
                row(fam, f"{k0}.draws", N_DRAWS, rf.get("draws"), None, verdict, kind="int")
                row(fam, f"{k0}.finite_draws", nf, rf.get("finite_draws"), None, verdict, kind="int")
                rc = rf.get("ci95") or [None, None]
                if ci is None or not rf.get("ci95"):  # 25e (R2 P2): a withheld interval is a bool row, as in the anchor
                    row(fam, f"{k0}.ci95.withheld", ci is None, not rf.get("ci95"), None, verdict, kind="bool")
                    continue
                for i in (0, 1):
                    row(fam, f"{k0}.ci95[{i}]", ci[i], rc[i], 0.0, verdict)
            fr = node.get("catch_flips_by_score_quartile")
            if fr is None:
                continue
            if qconv is None or qconv not in B[(lab, "sim_score_vs_drop")]["flips"]:
                row(fam, f"{readout}|{lab}|flips", None, None, None, verdict, kind="unresolved")
                continue
            mf = B[(lab, "sim_score_vs_drop")]["flips"][qconv]
            for i, e in enumerate(mf["quartile_edges"]):
                row(fam, f"{readout}|{lab}|flips.quartile_edges[{i}]", e, (fr.get("quartile_edges") or [None] * 3)[i], 0.0, verdict)
            row(fam, f"{readout}|{lab}|flips.excluded_cells", mf["excluded_cells"], fr.get("excluded_cells"), None, verdict, kind="int")
            for q, d_ in mf["by_quartile"].items():
                for fld, v in d_.items():
                    row(fam, f"{readout}|{lab}|flips.{q}.{fld}", v, ((fr.get("by_quartile") or {}).get(q) or {}).get(fld), None, verdict, kind="int")

    def rec_node(rec, ro, lab, kind):
        return ((((rec.get("result") or {}).get("probes") or {}).get(lab) or {}).get(ro) or {}).get(kind) or {}

    # ---- identification on the existing L63 record (commands L63, replies L48): points; quartile convention; draws x percentile method
    B63 = {ro: base(ro) for ro in ("commands L63", "replies L48")}
    pts_ok = all(num(rec_node(D63, ro, lab, k).get("point")) and abs(b["point"] - rec_node(D63, ro, lab, k)["point"]) <= 1e-12
                 for ro in B63 for (lab, k), b in B63[ro].items())
    controls.append({"name": "C1_points_reproduce_L63", "pass": bool(pts_ok)})
    QCONV, QALL, n_checked = None, [], 0  # 25e (R1 P2): every convention that matches on L63 is kept
    for fs, qm, qb in ((fs, qm, qb) for fs in FLIP_SETS for qm in METHODS for qb in QBOUNDS):
        good, n_checked = True, 0
        for ro in B63:
            for n in SIX:
                fr = ((((D63.get("result") or {}).get("probes") or {}).get(f"{n} C") or {}).get(ro) or {}).get("catch_flips_by_score_quartile")
                if fr is None:
                    continue
                n_checked += 1
                mf = B63[ro][(f"{n} C", "sim_score_vs_drop")]["flips"].get((fs, qm, qb))
                if mf is None:
                    good = False
                    continue
                good &= (len(fr.get("quartile_edges") or []) == 3  # 25d (Codex P1): exact up to float rounding, 1e-12 absolute
                         and all(num(a) and abs(a - b) <= 1e-12 * max(1.0, abs(a)) for a, b in zip(fr["quartile_edges"], mf["quartile_edges"]))
                         and all(((fr.get("by_quartile") or {}).get(q) or {}) == v for q, v in mf["by_quartile"].items()))
        if good and n_checked > 0:  # 25e (R2 P2): never a vacuous match
            QALL.append((fs, qm, qb))
    QCONV = QALL[0] if QALL else None
    controls.append({"name": "C2_quartile_convention_on_L63", "convention": list(QCONV) if QCONV else None,
                     "all_matching": [list(c) for c in QALL], "pass": QCONV is not None})
    VAR, METHOD, Mid = None, None, None
    if pts_ok:
        for v in VARIANTS:
            Md = draws(v)
            DR63 = {ro: per_draw(B63[ro], Md) for ro in B63}
            for mth in METHODS:
                hit = True
                for ro in B63:
                    for (lab, k), dr in DR63[ro].items():
                        ci, _ = ci_of(dr, mth)
                        rc = rec_node(D63, ro, lab, k).get("ci95") or [None, None]
                        hit &= ci is not None and all(num(rc[i]) and abs(ci[i] - rc[i]) <= 1e-12 for i in (0, 1))  # 25d: exact up to rounding
                if hit:
                    VAR, METHOD, Mid = v, mth, Md
                    break
            if VAR:
                break
    controls.append({"name": "C3_draws_and_method_on_L63", "variant": VAR, "method": METHOD, "pass": VAR is not None})

    # ---- the L63 record under the identified conventions (diagnostic); the L48 record (verdict)
    if VAR:
        for ro in B63:
            compare("L63_identification", D63, ro, B63[ro], per_draw(B63[ro], Mid), METHOD, QCONV, False)
    B48 = {ro: base(ro) for ro in ("commands L48", "replies L48")}
    for ro in B48:
        compare("L48_verdict", D48, ro, B48[ro], per_draw(B48[ro], Mid) if VAR else None, METHOD, QCONV, True)
    have48 = sorted({ro for p_ in ((D48.get("result") or {}).get("probes") or {}).values() for ro in p_})
    rows.append({"family": "L48_verdict", "key": "readouts in the L48 record == {commands L48, replies L48}", "verdict": True, "kind": "bool",
                 "match": have48 == ["commands L48", "replies L48"]})
    def flips_close(a, b):  # 25g (R1 / R2 P2): at the verdict tolerance (edges within tol, counts exact), not float-exact
        return (a is not None and b is not None and a["excluded_cells"] == b["excluded_cells"] and a["by_quartile"] == b["by_quartile"]
                and len(a["quartile_edges"]) == len(b["quartile_edges"]) == 3
                and all(abs(x - y) <= tol(y) for x, y in zip(a["quartile_edges"], b["quartile_edges"])))

    if len(QALL) > 1:  # 25e (R1 P2): conventions tied on L63 must give the same L48 flips tables, else the flips verdict is ambiguous
        with48 = [(ro, n) for ro in B48 for n in SIX
                  if ((((D48.get("result") or {}).get("probes") or {}).get(f"{n} C") or {}).get(ro) or {}).get("catch_flips_by_score_quartile")]
        same = all(flips_close(B48[ro][(f"{n} C", "sim_score_vs_drop")]["flips"].get(c), B48[ro][(f"{n} C", "sim_score_vs_drop")]["flips"].get(QCONV))
                   for ro, n in with48 for c in QALL)
        rows.append({"family": "L48_verdict", "key": "flips: every L63-matching convention gives the same L48 tables", "verdict": True,
                     "kind": "bool" if same else "unresolved", "match": True if same else None})  # 25e (R2 P2): disagreement = UNRESOLVED

    vr = [r for r in rows if r["verdict"]]
    fam = {}
    for r in rows:
        f_ = fam.setdefault(r["family"], {"rows": 0, "verdict": 0, "verdict_match": 0, "unresolved": 0, "diag_match": 0})
        f_["rows"] += 1
        if r["verdict"]:
            f_["verdict"] += 1
            f_["verdict_match"] += r["match"] is True
            f_["unresolved"] += r["kind"] == "unresolved"
        else:
            f_["diag_match"] += r["match"] is True
    failing = [r["key"] for r in vr if r["match"] is not True]
    printed = ["commands L48|sol_a_uncut C|sim_score_vs_drop.point", "commands L48|sol_a_uncut C|sim_score_vs_drop.ci95[0]",
               "commands L48|sol_a_uncut C|sim_score_vs_drop.ci95[1]",
               "commands L48|sol_a_uncut C|null_run2_score_vs_run2_minus_run1.point"]  # 25b (09): provenance v17 rows 119-122 (rc1b slot 23)
    pm = {r["key"]: r["match"] for r in rows if r["family"] == "L48_verdict"}
    vals = {"rows": rows, "controls": controls, "variant": VAR, "method": METHOD, "quartile_convention": QCONV, "stops": stops}
    vb = json.dumps(vals, default=float).encode()
    vpath = Path("/o", f"cleanroom_detdrop_values_{stamp}.json")
    with vpath.open("xb") as fh:
        fh.write(vb)
    OUT_VOL.commit()
    return {"stamp": stamp, "n_stops": len(stops), "stops": stops[:40], "controls": controls, "by_family": fam,
            "n_verdict": len(vr), "n_verdict_match": sum(r["match"] is True for r in vr), "n_unresolved": sum(r["kind"] == "unresolved" for r in vr),
            "n_failing_verdict": len(failing), "failing_verdict_keys": failing[:200],
            "printed": {"n": len(printed), "n_present": sum(k in pm for k in printed), "n_match": sum(pm.get(k) is True for k in printed),
                        "failing": [k for k in printed if pm.get(k) is not True]},
            "values_file": {"volume": OUT, "path": vpath.name, "sha256": hashlib.sha256(vb).hexdigest()},
            "label": "independent recompute from the producer's stated definition (POST HOC; no RULE registration); x (refit3 scores) shared",
            "PASS": not stops and all(c["pass"] for c in controls) and not failing}


@app.local_entrypoint()
def main():
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    out = run.remote(stamp)
    path = HERE / "results" / f"cleanroom_detdrop_{stamp}.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    print(json.dumps({k: out.get(k) for k in ("n_stops", "n_verdict", "n_verdict_match", "n_unresolved", "n_failing_verdict", "printed", "PASS")}))
    print("wrote", path)
