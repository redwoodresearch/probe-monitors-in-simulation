"""Clean-room anchor package (addendum 24, a1, 2026-10-20): Stage D's per-task points (Fig 7), the anchor's R^2 / Spearman (Fig 8;
rc1b slot 24), and their union-stream intervals. Docstring route (Aswin 12:08:34 via 0d; the formula review passed): "independent
recompute from the producer's stated definition (POST HOC; no RULE registration)".

Definitions (producer's stated text, read: run_pertask_anchor.py L1-9 and post_pertask.py L1-6 docstrings; detector_vs_drop's
meta.definitions; A's meta.merge.rule; the anchor's union_stream.rule) + registered RULE items 16 / 18 / 29 / 30 (@ 993da3957):
  per task (cell) at f = 1e-2, readout as the label, draw 0, pooled-world tau: max_minus_tau(arm) = the arm's largest transformed unit
      minus its own alarm line (run1, run2: tau_B; sim: tau_S); caught = max_minus_tau > 0. From my unchanged reducer's draw-0 maxima.
  anchor (LEAD, every probe): main = run1 vs sim, yardstick = run1 vs run2; Pearson r^2 and Spearman rho; delta = main - yardstick.
      run2 block (sensitivity_run2_anchor; on-policy probes): main = run2 vs sim, yardstick = run2 vs run1.
      Cells: finite run1, run2 and sim values (RULE item 18). Sets: the four pooled origin rows and pooled_all_origins.
  intervals (pooled_all_origins, union stream; RULE item 30(1) / (3), tau fixed at draw 0): per draw, each cell weighted by its main's
      multiplicity (weighted Pearson; Spearman = weighted Pearson of average ranks with tied repeats), 95% percentile interval.
      The draw mechanics (seed [20261015, 40], env-stratified multinomial over each env's sorted mains, one multiplicity per main shared
      across everything) come from producer text; the exact call order is IDENTIFIED: four pre-stated variants, and only one that
      reproduces A's stored FIXED-mode pooled delta_full arrays exactly (all 18 labels) is used; else the intervals are UNRESOLVED.
      The percentile method is identified on A's own fixed-mode pooled delta_full nodes (as in addendum 20).
The laptop gets booleans and counts only.

  CR_CFG=<pinned clean-room inputs> AN_CFG=<Stage D + anchor + rc1b> uv run --no-project --python 3.12 --with modal==1.6.1 modal run cleanroom_anchor.py
"""

import json
import os
import re
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
AN_PINS = {"stage_d.json": "468655154f826ec1a93e08333eb802e68abea6db65a8e2f80a3e0f23465f36d9",
           "anchor.json": "2763ad99b7d37ee25784d9b58563a60d5d6c4b425656e45a97f71b82684cfbad",
           "rc1b.json": "ab8dcb439b942c92d30b784ae1e6a3a193b18127391ff1753426663b726e728b"}
A_NPZ = ("xfer/fpr507prep/alltasks_CRD_FULL_r2_allorigins_pooledworld_scored_e12_2026-10-19_180327_perdraw.npz",
         "5e6cd10e382d8b03d409982c67efd72512176d7e95d2f3c94752d2b90e5650d1")
COMBO, LVL, N_DRAWS = "pooled_all_origins", "0.01", 10000
SETS = ["S_old", "S_new", "S_newmains", "S_paper493"]
METHODS = ("linear", "hazen", "weibull", "median_unbiased", "normal_unbiased", "interpolated_inverted_cdf", "inverted_cdf",
           "averaged_inverted_cdf", "closest_observation", "lower", "higher", "nearest", "midpoint")
VARIANTS = ("draw_outer_multinomial", "env_outer_multinomial_size", "draw_outer_integers", "env_outer_integers_size")
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
             .add_local_dir(os.environ["AN_CFG"], "/an"))


def tol(x):
    return RTOL * max(1.0, abs(x))


def enc(s):
    return s.replace("@", "_at_").replace("|", "_").replace(" ", "_").replace("(", "").replace(")", "")


def origin_key(s):
    return f"{s} (all-origins draws) [pooled: {COMBO}]"


def wstats(x, y, W, ranks="repeat"):
    """Weighted Pearson r^2 and Spearman rho for every row of W (draws x cells; nonnegative integer counts = repeats). Spearman =
    weighted Pearson of average ranks of the weight-repeated data (ties share one average rank). Vectorized over draws: the values are
    fixed (tau fixed at draw 0), only the weights change. Returns (r2, rho), each of length W.shape[0].
    ranks="unique" (24d, diagnostic reading only): average ranks of the unique cells, unweighted, then the same weighted Pearson."""
    import numpy as np

    W = np.atleast_2d(np.asarray(W, dtype=float))
    if W.shape[1] == 0:  # 24b (Codex P2): no cells -> undefined, not a crash
        nan = np.full(W.shape[0], np.nan)
        return nan, nan.copy()
    pos = W > 0

    def const(v):  # 24b (Codex P1): a constant value over the positive-weight cells -> undefined
        return np.where(pos, v, -np.inf).max(axis=1) == np.where(pos, v, np.inf).min(axis=1)

    bad = const(np.asarray(x, dtype=float)) | const(np.asarray(y, dtype=float))

    def wrank(v, Wr):
        o = np.argsort(v, kind="mergesort")
        vs = v[o]
        starts = np.flatnonzero(np.r_[True, vs[1:] != vs[:-1]])
        gid = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(vs)]))
        Wg = np.add.reduceat(Wr[:, o], starts, axis=1)  # draws x groups
        cum = np.cumsum(Wg, axis=1)
        rg = cum - 0.5 * Wg + 0.5  # average of ranks (cum - Wg + 1) .. cum
        R = np.empty_like(Wr)
        R[:, o] = rg[:, gid]
        return R

    def wpear(A, B):
        S = W.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            ma, mb = (W * A).sum(axis=1) / S, (W * B).sum(axis=1) / S
            da, db = A - ma[:, None], B - mb[:, None]
            va, vb = (W * da * da).sum(axis=1), (W * db * db).sum(axis=1)
            r = (W * da * db).sum(axis=1) / np.sqrt(va * vb)
        return np.where((S > 0) & (va > 0) & (vb > 0), r, np.nan)

    X, Y = np.broadcast_to(x, W.shape), np.broadcast_to(y, W.shape)
    r = np.where(bad, np.nan, wpear(X, Y))
    Wr = W if ranks == "repeat" else np.ones((1, W.shape[1]))
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    rho = np.where(bad, np.nan, wpear(wrank(xa, Wr), wrank(ya, Wr)))
    return r * r, rho


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
@boundary("cleanroom_anchor")
def run(stamp: str) -> dict:
    import gzip
    import hashlib
    import math

    import numpy as np

    def sha(b):
        return hashlib.sha256(b).hexdigest()

    def num(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)

    stops, rows, controls = [], [], []

    def row(fam, key, m, ref, bound, verdict=True, kind="value", total=False):
        r = {"family": fam, "key": key, "verdict": verdict, "kind": kind, "mine": m, "ref": ref}
        if kind == "value" and total:  # 24b (Codex P1): bound is the whole tolerance (b of tau), nothing added
            r["match"] = bool(num(m) and num(ref) and abs(m - ref) <= bound)
            rows.append(r)
            return
        if kind in ("int", "bool", "cat"):
            r["match"] = bool(m == ref and type(m) is type(ref)) if kind != "cat" else bool(m == ref)
        elif kind == "unresolved":
            r["match"] = None
        else:
            r["match"] = bool(num(m) and num(ref) and abs(m - ref) <= (bound or 0.0) + tol(ref))
        rows.append(r)

    assert sha(Path("/root/cleanroom_reduce_v2.py").read_bytes()) == REDUCER_SHA, "reducer sha"
    raw = {f: Path("/an", f).read_bytes() for f in AN_PINS}
    for f, want in AN_PINS.items():
        assert sha(raw[f]) == want, f"input sha {f}"
    SD, AN, RC = (json.loads(raw[f]) for f in ("stage_d.json", "anchor.json", "rc1b.json"))
    del raw
    vraw = Path("/o", A18_VALUES[0]).read_bytes()
    assert sha(vraw) == A18_VALUES[1], "A18 values sha"
    V18 = json.loads(vraw)["values"]
    h = hashlib.sha256()
    with open(Path("/p", A_NPZ[0]), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    assert h.hexdigest() == A_NPZ[1], "A npz sha"

    # ---- draw 0: my unchanged reducer under the A18 config (as in addendum 21); capture instead of comparing
    sys.path.insert(0, "/root")
    import cleanroom_reduce_v2 as R

    cap = {}

    def capture(report, results, six, allp, main, stamp_, t0, run_cfg):
        cap.update(results=results, main=main, run_cfg=run_cfg)
        return {}

    R.compare_and_control = capture
    R.run_stage.local("compute", stamp)
    res, main = cap["results"], cap["main"]
    run_cfg = cap["run_cfg"]
    eq0 = [all(d[k] == V18[lab]["levels"][lvl][k] for k in ("tpr", "real_mean", "delta_full"))
           and d["tau_S"]["T"] == V18[lab]["levels"][lvl]["tau_S"]["T"] and d["tau_B"]["T"] == V18[lab]["levels"][lvl]["tau_B"]["T"]
           for lab, r in res.items() for lvl, d in r["levels"].items()]
    controls.append({"name": "C0_reproduces_A18_values", "n": len(eq0), "n_true": int(sum(eq0)), "pass": all(eq0)})
    A = json.loads(gzip.decompress(Path("/cfg", run_cfg["pooled_A"]["name"]).read_bytes()))
    labels = sorted(AN["lead"])
    if sorted(res) != labels or len(labels) != 18:
        stops.append("anchor lead labels != my 18 labels")
        return {"stamp": stamp, "n_stops": len(stops), "stops": stops, "controls": controls}
    for blk, anc in (("lead", "run1"), ("sensitivity_run2_anchor", "run2")):  # 24d (R2 P2): the record's anchor field, as post_pertask gates
        bad_anc = [lab for lab, nd in (AN.get(blk) or {}).items() if not isinstance(nd, dict) or nd.get("anchor") != anc]
        if bad_anc:
            stops.append(f"{blk}: anchor field != {anc} for {len(bad_anc)} labels")

    # ---- per-task values (draw 0, f = 1e-2)
    pt = {}
    for lab in labels:
        r = res[lab]
        d = r["levels"][LVL]
        tB, tS = d["tau_B"]["T"], d["tau_S"]["T"]
        for s in SETS:
            for c in r["cells_by_set"][s]:
                pt[(lab, c)] = {"set": s, "main": main[c], "run1": r["zmax"][f"run1|{c}"] - tB, "run2": r["zmax"][f"run2|{c}"] - tB,
                                "sim": r["zmax"][f"sim|{c}"] - tS, "tB": tB, "tS": tS}

    # ---- (1) Stage D per_task rows (Fig 7's points): max_minus_tau and caught per cell x arm, and tau, per pooled origin row
    near = 0
    for lab in labels:
        for s in SETS:
            node = (SD["rows"].get(lab) or {}).get(origin_key(s)) or {}
            tau = node.get("tau") or {}
            any_c = next((v for (l_, c), v in pt.items() if l_ == lab and v["set"] == s), None)
            if any_c is None:
                stops.append(f"no cells: {lab} {s}")
                continue
            row("stage_d_tau", f"{lab}|{s}|tau.B", any_c["tB"], tau.get("B"), 0.0)
            row("stage_d_tau", f"{lab}|{s}|tau.S", any_c["tS"], tau.get("S"), 0.0)
            if node.get("f") != 0.01:
                stops.append(f"Stage D f != 0.01: {lab} {s}")
            seen = set()
            expected = {(c, a) for (l_, c), v in pt.items() if l_ == lab and v["set"] == s for a in ("run1", "run2", "sim")}
            for e in node.get("per_task") or []:
                c, arm = e.get("cell"), e.get("arm")
                m = pt.get((lab, c))
                if m is None or m["set"] != s or arm not in ("run1", "run2", "sim") or (c, arm) in seen:  # 24b (Codex P1): origin checked
                    row("stage_d_pt", f"{lab}|{s}|{c}|{arm}|present", False, True, None, kind="bool")
                    continue
                seen.add((c, arm))
                mv, rv = m[arm], e.get("max_minus_tau")
                if not math.isfinite(mv):  # a zero-unit arm: the record must carry no finite value either
                    row("stage_d_pt", f"{lab}|{s}|{c}|{arm}|max_minus_tau", True, not num(rv), None, kind="bool")
                    row("stage_d_pt", f"{lab}|{s}|{c}|{arm}|caught", False, e.get("caught"), None, kind="bool")
                    continue
                b = tol(m["tB" if arm != "sim" else "tS"])
                row("stage_d_pt", f"{lab}|{s}|{c}|{arm}|max_minus_tau", mv, rv, b, total=True)  # 24b: b(tau) is the whole bound
                cg = mv > 0
                if abs(mv) <= 2 * b and e.get("caught") != cg:  # 24b: exactly 2 b(tau)
                    near += 1
                    row("stage_d_pt", f"{lab}|{s}|{c}|{arm}|caught", cg, e.get("caught"), None, verdict=False, kind="bool")
                else:
                    row("stage_d_pt", f"{lab}|{s}|{c}|{arm}|caught", cg, e.get("caught"), None, kind="bool")
            row("stage_d_pt", f"{lab}|{s}|n_entries", len(expected), len(seen), None, kind="int")
            row("stage_d_pt", f"{lab}|{s}|entry_set_equal", True, seen == expected, None, kind="bool")  # 24b: the exact (cell, arm) set

    # ---- (2) anchor points per label x set (lead: run1 anchor; run2 block where the record has it)
    def stats_for(lab, cells, anchor, w_over_n=False):
        other = "run2" if anchor == "run1" else "run1"
        rows_ = [pt[(lab, c)] for c in cells if all(math.isfinite(pt[(lab, c)][a]) for a in ("run1", "run2", "sim"))]
        a = np.array([v[anchor] for v in rows_], dtype=float)
        sv, ov = np.array([v["sim"] for v in rows_], dtype=float), np.array([v[other] for v in rows_], dtype=float)
        w = np.ones((1, len(rows_)))
        rk = "repeat"
        if w_over_n:  # 24d diagnostic reading (R2 P2 iii): cell weight W_main / n_main (A's merge.rule wording); W_main = 1 at draw 0
            nm = {}
            for v in rows_:
                nm[v["main"]] = nm.get(v["main"], 0) + 1
            w, rk = np.array([[1.0 / nm[v["main"]] for v in rows_]]), "unique"
        r2m, rhom = (float(z[0]) for z in wstats(a, sv, w, rk))
        r2y, rhoy = (float(z[0]) for z in wstats(a, ov, w, rk))
        return {"n_cells": len(rows_), "r2": {"main": r2m, "yardstick": r2y, "delta_main_minus_yardstick": r2m - r2y},
                "spearman": {"main": rhom, "yardstick": rhoy, "delta_main_minus_yardstick": rhom - rhoy}}

    mine_pts = {}
    for blk, anchor in (("lead", "run1"), ("sensitivity_run2_anchor", "run2")):
        for lab in sorted((AN.get(blk) or {})):
            for setname in [origin_key(s) for s in SETS] + [COMBO]:
                carried = (((AN.get(blk) or {}).get(lab) or {}).get("sets") or {})
                if setname not in carried:  # 24d (R1 P1-3): lead must carry all 5 sets (verdict); the run2 block's absences are diagnostic
                    row("anchor_pt", f"{blk}|{lab}|{setname}|carried", False, True, None, verdict=(blk == "lead"), kind="bool")
                    continue
                cells = [c for (l_, c), v in pt.items() if l_ == lab and (setname == COMBO or origin_key(v["set"]) == setname)]
                st = stats_for(lab, cells, anchor)
                mine_pts[(blk, lab, setname)] = st
                node = carried.get(setname) or {}
                row("anchor_pt", f"{blk}|{lab}|{setname}|n_cells", st["n_cells"], node.get("n_cells"), None, kind="int")
                dg = stats_for(lab, cells, anchor, w_over_n=True) if setname == COMBO else None
                for fam in ("r2", "spearman"):
                    for q in ("main", "yardstick", "delta_main_minus_yardstick"):
                        ref = (node.get(fam) or {}).get(q)  # 24b (Codex P2): a numeric 0 stays 0
                        ref = ref.get("point") if isinstance(ref, dict) else ref
                        row("anchor_pt", f"{blk}|{lab}|{setname}|{fam}.{q}", st[fam][q], ref, 0.0)
                        if dg is not None:
                            row("anchor_diag", f"{blk}|{lab}|{setname}|{fam}.{q}|w_over_n_main", dg[fam][q], ref, 0.0, verdict=False)

    # ---- (3) union-stream draw mechanics, identified on A's FIXED-mode pooled delta_full arrays (exact), then the percentile method
    mrg = A["meta"]["merge"]
    env_of = mrg["env_of_main"]
    # 24d (R1 P2): two pre-stated environment orders; mains within an environment in sorted order (A's list is sorted by main)
    ENV_ORDERS = {"sorted": sorted(set(env_of.values())), "first_appearance": list(dict.fromkeys(env_of[m] for m in mrg["mains"]))}

    def layout(envs):
        by_env = {e: sorted(m for m in mrg["mains"] if env_of[m] == e) for e in envs}
        return by_env, {m: i for i, m in enumerate(m for e in envs for m in by_env[e])}

    miss = sorted({v["main"] for v in pt.values()} - set(mrg["mains"]))
    if miss:  # every cohort main must be in the union stream's main list
        stops.append(f"{len(miss)} cohort mains absent from A meta.merge.mains")

    def draws(variant, envs):
        by_env, _ = layout(envs)
        rng = np.random.default_rng(list(mrg["seed"]))
        M = np.zeros((N_DRAWS, len(mrg["mains"])), dtype=np.int64)
        off = {e: sum(len(by_env[x]) for x in envs[:k]) for k, e in enumerate(envs)}
        if variant == "draw_outer_multinomial":
            for d in range(N_DRAWS):
                for e in envs:
                    n = len(by_env[e])
                    M[d, off[e]:off[e] + n] = rng.multinomial(n, np.full(n, 1.0 / n))
        elif variant == "env_outer_multinomial_size":
            for e in envs:
                n = len(by_env[e])
                M[:, off[e]:off[e] + n] = rng.multinomial(n, np.full(n, 1.0 / n), size=N_DRAWS)
        elif variant == "draw_outer_integers":
            for d in range(N_DRAWS):
                for e in envs:
                    n = len(by_env[e])
                    M[d, off[e]:off[e] + n] = np.bincount(rng.integers(0, n, n), minlength=n)
        else:
            for e in envs:
                n = len(by_env[e])
                ix = rng.integers(0, n, size=(N_DRAWS, n))
                M[:, off[e]:off[e] + n] = np.stack([np.bincount(row_, minlength=n) for row_ in ix])
        return M

    if mrg.get("seed") != [20261015, 40] or AN["union_stream"].get("seed") != [20261015, 40] or AN["union_stream"].get("draws") != N_DRAWS:
        stops.append("union stream seed / draws != pre-stated")
    ZA = np.load(Path("/p", A_NPZ[0]))
    fixed = {}
    for lab in labels:
        k = "__".join([enc(lab), enc(COMBO), "fixed", LVL, "pooled", "gap_delta_full"])
        fixed[lab] = np.asarray(ZA[k], dtype=float) if k in ZA.files else None
    if any(v is None for v in fixed.values()):
        stops.append("A npz fixed-mode pooled gap_delta_full absent for some label")

    def fixed_delta(lab, M, midx):  # per draw, tau fixed at draw 0: main-level mean catch per arm, main weight = multiplicity
        r = res[lab]
        cells = [c for s in SETS for c in r["cells_by_set"][s]]
        d0 = r["levels"][LVL]
        cm = {}
        for c in cells:
            v = pt[(lab, c)]
            cm.setdefault(v["main"], []).append(c)
        mains_l = sorted(cm)
        cols = np.array([midx[m] for m in mains_l], dtype=np.intp)
        mc = {a: np.array([np.mean([float(pt[(lab, c)][a] > 0) for c in cm[m]]) for m in mains_l]) for a in ("run1", "run2", "sim")}
        W = M[:, cols].astype(float)
        den = W.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            tpr = {a: (W * mc[a]).sum(axis=1) / den for a in mc}
        _ = d0
        return tpr["sim"] - 0.5 * (tpr["run1"] + tpr["run2"])

    VAR, Mid, midx, c1_candidates = None, None, None, []
    if not stops:
        for oname, envs in ENV_ORDERS.items():
            _, mx = layout(envs)
            for v in VARIANTS:
                M = draws(v, envs)
                ok = all(fixed[lab] is not None and len(fixed[lab]) == N_DRAWS + 1 and
                         np.allclose(fixed_delta(lab, M, mx), fixed[lab][1:], rtol=0, atol=1e-12, equal_nan=True) for lab in labels)
                c1_candidates.append({"env_order": oname, "variant": v, "reproduces": bool(ok)})  # 24b: candidates are diagnostic
                if ok:
                    VAR, Mid, midx = f"{v}|env_order={oname}", M, mx
                    break
            if VAR:
                break
    # 24d (R2 P2): C1 is not vacuous only if some label's fixed array varies across draws
    varies = any(fixed.get(lab) is not None and np.isfinite(fixed[lab][1:]).sum() > 1
                 and np.nanmax(fixed[lab][1:]) > np.nanmin(fixed[lab][1:]) for lab in labels) if not stops else False
    controls.append({"name": "C1_union_stream_identified", "variant": VAR, "candidates": c1_candidates, "some_label_varies": bool(varies),
                     "pass": VAR is not None and bool(varies)})
    METHOD = None
    if VAR:
        nodes = {lab: A["pooled"][lab][COMBO]["fixed"][LVL]["gap"]["delta_full"] for lab in labels}
        for mth in METHODS:
            okm = True
            for lab in labels:
                x = fixed[lab][1:]
                f = x[np.isfinite(x)]
                ci = np.percentile(f, [2.5, 97.5], method=mth)
                nd = nodes[lab]
                okm &= (nd.get("draws") == N_DRAWS and nd.get("finite_draws") == len(f)
                        and all(num((nd.get("ci95") or [None, None])[i]) and abs(ci[i] - nd["ci95"][i]) <= tol(nd["ci95"][i]) for i in (0, 1)))
            if okm:
                METHOD = mth
                break
    controls.append({"name": "C2_percentile_method_on_A_fixed", "method": METHOD, "pass": METHOD is not None})

    # ---- (4) anchor intervals on the identified union stream (pooled_all_origins (union stream)), lead and run2 block
    us = AN.get("union_stream") or {}

    def cint(x):
        f = x[np.isfinite(x)]
        return [float(v) for v in np.percentile(f, [2.5, 97.5], method=METHOD)] if len(f) >= 0.95 * N_DRAWS else None
    for blk, anchor in (("lead", "run1"), ("sensitivity_run2_anchor", "run2")):
        for lab in sorted((us.get(blk) or {})):
            node = ((((us.get(blk) or {}).get(lab) or {}).get("sets") or {}).get(f"{COMBO} (union stream)")) or {}
            key0 = f"{blk}|{lab}|union"
            if not (VAR and METHOD):
                row("anchor_ci", f"{key0}|all", None, None, None, kind="unresolved")
                continue
            other = "run2" if anchor == "run1" else "run1"
            cells = [c for (l_, c), v in pt.items() if l_ == lab and all(math.isfinite(v[a]) for a in ("run1", "run2", "sim"))]
            a = np.array([pt[(lab, c)][anchor] for c in cells], dtype=float)
            sv = np.array([pt[(lab, c)]["sim"] for c in cells], dtype=float)
            ov = np.array([pt[(lab, c)][other] for c in cells], dtype=float)
            col = np.array([midx[pt[(lab, c)]["main"]] for c in cells], dtype=np.intp)  # 24c (Codex P2): empty -> int index, reaches wstats guard
            Wd = Mid[:, col]  # draws x cells: each cell weighted by its main's multiplicity
            r2m, rhom = wstats(a, sv, Wd)
            r2y, rhoy = wstats(a, ov, Wd)
            per = {("r2", "main"): r2m, ("r2", "yardstick"): r2y, ("r2", "delta_main_minus_yardstick"): r2m - r2y,
                   ("spearman", "main"): rhom, ("spearman", "yardstick"): rhoy, ("spearman", "delta_main_minus_yardstick"): rhom - rhoy}
            # 24d diagnostic readings (R1 / R2 P2; never gating): unique-cell ranks; cell weight = multiplicity / n_main
            nmain = np.bincount(col, minlength=len(midx)).astype(float)[col] if len(col) else np.zeros(0)
            Wn = Wd / nmain if len(col) else Wd.astype(float)
            alt = {}
            _, um = wstats(a, sv, Wd, "unique")
            _, uy = wstats(a, ov, Wd, "unique")
            alt["spearman_unique_ranks"] = {("spearman", "main"): um, ("spearman", "yardstick"): uy, ("spearman", "delta_main_minus_yardstick"): um - uy}
            nm_, nrm = wstats(a, sv, Wn, "unique")
            ny_, nry = wstats(a, ov, Wn, "unique")
            alt["w_over_n_main"] = {("r2", "main"): nm_, ("r2", "yardstick"): ny_, ("r2", "delta_main_minus_yardstick"): nm_ - ny_,
                                    ("spearman", "main"): nrm, ("spearman", "yardstick"): nry, ("spearman", "delta_main_minus_yardstick"): nrm - nry}
            for rd, per_alt in alt.items():
                for (fam, q), x in per_alt.items():
                    ci_a = cint(x)
                    rc = (((node.get(fam) or {}).get(q)) or {}).get("ci95") or [None, None]
                    row("anchor_diag", f"{key0}|{fam}.{q}.ci95|{rd}", ci_a is not None and all(num(rc[i]) and abs(ci_a[i] - rc[i]) <= tol(rc[i])
                                                                                               for i in (0, 1)), True, None, verdict=False, kind="bool")
            st0 = mine_pts.get((blk, lab, COMBO))
            for (fam, q) in per:
                rp = (((node.get(fam) or {}).get(q)) or {}).get("point")
                row("anchor_diag", f"{key0}|{fam}.{q}.point_vs_unweighted", st0[fam][q] if st0 else None, rp, 0.0, verdict=False)
            row("anchor_ci", f"{key0}|n_cells", len(cells), node.get("n_cells"), None, kind="int")
            for (fam, q), x in per.items():
                f = x[np.isfinite(x)]
                ci = cint(x)
                nd = ((node.get(fam) or {}).get(q)) or {}
                row("anchor_ci", f"{key0}|{fam}.{q}.draws", N_DRAWS, nd.get("draws"), None, kind="int")
                row("anchor_ci", f"{key0}|{fam}.{q}.finite_draws", int(len(f)), nd.get("finite_draws"), None, kind="int")
                if ci is None or not nd.get("ci95"):
                    row("anchor_ci", f"{key0}|{fam}.{q}.ci95.withheld", ci is None, not nd.get("ci95"), None, kind="bool")
                else:
                    for i in (0, 1):
                        row("anchor_ci", f"{key0}|{fam}.{q}.ci95[{i}]", ci[i], nd["ci95"][i], 0.0)
                if q == "delta_main_minus_yardstick" and ci is not None:
                    row("anchor_ci", f"{key0}|{fam}.{q}.excludes_0", not (ci[0] <= 0.0 <= ci[1]), nd.get("excludes_0"), None, kind="bool")
                mine_pts.setdefault(("union", blk, lab), {})[(fam, q)] = ci

    # ---- (5) rc1b slot 24 (printed rows 69-74 read own_all_rs_uncut_L48 C). 24d (R1 / R2): compared against rc1b's FULL-precision strings
    # (RC["full"], same 71 keys as the display "slots"), parsed as pre-stated: n_cells = all digits -> int; r2 = "<point> [<lo>, <hi>]" -> 3
    # floats; halfwidth = one float. Pairs = the (block, label) pairs rc1b carries (14), not the anchor's 24.
    full = RC.get("full") or {}
    k24 = sorted(k for k in full if k.startswith("24."))
    NOT_COMPARED = {"24.r2.yardstick_halfwidth.lead.range"}  # an "x to y" summary across labels
    printed = ["24.n_cells.lead.own_all_rs_uncut_L48 C", "24.r2.lead.own_all_rs_uncut_L48 C.yardstick",  # provenance v17 rows 69-74
               "24.r2.yardstick_halfwidth.lead.own_all_rs_uncut_L48 C",
               "24.r2.lead.own_all_rs_uncut_L48 C.main", "24.r2.lead.own_all_rs_uncut_L48 C.delta_main_minus_yardstick",  # 24e (09): rows 241-248
               "24.r2.sensitivity_run2_anchor.own_all_rs_uncut_L48 C.delta_main_minus_yardstick"]
    printed += [f"24.r2.{b_}.own_all_rs_uncut_L48 C.delta_main_minus_yardstick.ci95[{i}]"  # 24g (R1 / R2): rows 244/245, 247/248 = interval ends
                for b_ in ("lead", "sensitivity_run2_anchor") for i in (0, 1)]
    used = []

    def fparse(k, kind):  # a parse failure gives None (the row fails); the ValueError text is never kept
        used.append(k)
        s_ = full.get(k)
        if not isinstance(s_, str):
            return None
        try:
            if kind == "int":
                return int(s_) if re.fullmatch(r"[0-9]+", s_) else None
            if kind == "float":
                return float(s_)
            mt = re.fullmatch(r"(\S+) \[(\S+), (\S+)\]", s_)
            return tuple(float(g) for g in mt.groups()) if mt else None
        except ValueError:
            return None

    pairs = sorted({tuple(k.split(".", 3)[2:]) for k in k24 if k.startswith("24.n_cells.")})
    an_pairs = {(b_, l_) for b_ in ("lead", "sensitivity_run2_anchor") for l_ in (AN.get(b_) or {})}
    row("rc1b_24", "coverage|rc1b pairs within the anchor's labels", True, bool(pairs) and set(pairs) <= an_pairs, None, kind="bool")
    row("rc1b_24", "coverage|printed keys present", True,  # 24h (Codex P1): a .ci95[i] component lives inside its parent key's string
        all(re.sub(r"\.ci95\[[01]\]$", "", k) in full for k in printed), None, kind="bool")
    hw_hits = {}
    for blk, lab in pairs:
        st = mine_pts.get((blk, lab, COMBO))
        cu = mine_pts.get(("union", blk, lab)) or {}
        row("rc1b_24", f"24.n_cells.{blk}.{lab}", st["n_cells"] if st else None, fparse(f"24.n_cells.{blk}.{lab}", "int"), None, kind="int")
        for q in ("main", "yardstick", "delta_main_minus_yardstick"):
            ref = fparse(f"24.r2.{blk}.{lab}.{q}", "triple")
            row("rc1b_24", f"24.r2.{blk}.{lab}.{q}", st["r2"][q] if st else None, ref[0] if ref else None, 0.0)
            ci = cu.get(("r2", q))
            for i in (0, 1):
                if not (VAR and METHOD):  # 24f (R1 P2): no identified union stream -> UNRESOLVED, not a fail
                    row("rc1b_24", f"24.r2.{blk}.{lab}.{q}.ci95[{i}]", None, ref[1 + i] if ref else None, None, kind="unresolved")
                else:
                    row("rc1b_24", f"24.r2.{blk}.{lab}.{q}.ci95[{i}]", ci[i] if ci else None, ref[1 + i] if ref else None, 0.0)
        ref = fparse(f"24.r2.yardstick_halfwidth.{blk}.{lab}", "float")
        ci = cu.get(("r2", "yardstick"))
        hits = set()
        if ci is not None and st is not None and num(ref):  # readings: (hi - lo) / 2, or max(hi - point, point - lo); all hits kept
            y0 = st["r2"]["yardstick"]
            hw = {"half_range": 0.5 * (ci[1] - ci[0]), "max_side": max(ci[1] - y0, y0 - ci[0])}
            hits = {k for k, v in hw.items() if abs(v - ref) <= tol(ref)}
        if not (VAR and METHOD):  # 24f (R1 P2)
            row("rc1b_24", f"24.r2.yardstick_halfwidth.{blk}.{lab}", None, ref, None, kind="unresolved")
            continue
        hw_hits[(blk, lab)] = hits
        row("rc1b_24", f"24.r2.yardstick_halfwidth.{blk}.{lab}", bool(hits), True, None, kind="bool")
        rows[-1]["readings"] = sorted(hits)
    if not (VAR and METHOD):
        row("rc1b_24", "halfwidth reading identified (one reading fits every pair)", None, None, None, kind="unresolved")
    else:
        common = set.intersection(*hw_hits.values()) if hw_hits else set()
        row("rc1b_24", "halfwidth reading identified (one reading fits every pair)", bool(common), True, None, kind="bool")
        rows[-1]["readings"] = sorted(common)
    # 24g (R2 P1-a): the paper prints rc1b's 2-dp display strings ("slots"), so each compared display string must agree with its
    # full-precision string within 0.005 + 1e-9 (n_cells exactly). n_cells "d,ddd" -> int; r2 "p (lo to hi)" with U+2212 / "+" signs.
    disp = RC.get("slots") or {}
    row("rc1b_24", "coverage|display keys == full keys (24.*)", True, sorted(k for k in disp if k.startswith("24.")) == k24, None, kind="bool")

    def dparse(k, kind):
        s_ = disp.get(k)
        if not isinstance(s_, str):
            return None
        try:
            if kind == "int":
                return int(s_.replace(",", "")) if re.fullmatch(r"[0-9][0-9,]*", s_) else None
            s_ = s_.replace("\u2212", "-")
            if kind == "float":
                return float(s_)
            mt = re.fullmatch(r"([+-]?[0-9]+\.[0-9]+) \(([+-]?[0-9]+\.[0-9]+) to ([+-]?[0-9]+\.[0-9]+)\)", s_)
            return tuple(float(g) for g in mt.groups()) if mt else None
        except ValueError:
            return None

    def fval(k, kind):  # the full string again, without touching the coverage list
        s_ = full.get(k)
        try:
            if kind == "int":
                return int(s_) if isinstance(s_, str) and re.fullmatch(r"[0-9]+", s_) else None
            if kind == "float":
                return float(s_) if isinstance(s_, str) else None
            mt = re.fullmatch(r"(\S+) \[(\S+), (\S+)\]", s_) if isinstance(s_, str) else None
            return tuple(float(g) for g in mt.groups()) if mt else None
        except ValueError:
            return None

    def dclose(a, b):
        return num(a) and num(b) and abs(a - b) <= 0.005 + 1e-9

    for blk, lab in pairs:
        k = f"24.n_cells.{blk}.{lab}"
        a, b = dparse(k, "int"), fval(k, "int")
        row("rc1b_24_display", f"display|{k}", True, a is not None and a == b, None, kind="bool")
        for q in ("main", "yardstick", "delta_main_minus_yardstick"):
            k = f"24.r2.{blk}.{lab}.{q}"
            a, b = dparse(k, "triple"), fval(k, "triple")
            for j, suf in enumerate(("", ".ci95[0]", ".ci95[1]")):
                row("rc1b_24_display", f"display|{k}{suf}", True, bool(a and b and dclose(a[j], b[j])), None, kind="bool")
        k = f"24.r2.yardstick_halfwidth.{blk}.{lab}"
        row("rc1b_24_display", f"display|{k}", True, bool(dclose(dparse(k, "float"), fval(k, "float"))), None, kind="bool")
    row("rc1b_24", "coverage|every 24.* key used once or declared not compared", True,
        len(used) == len(set(used)) and set(used) == set(k24) - NOT_COMPARED, None, kind="bool")

    vr = [r for r in rows if r["verdict"]]
    fam = {}
    for r in rows:
        f_ = fam.setdefault(r["family"], {"rows": 0, "verdict": 0, "verdict_match": 0, "unresolved": 0})
        f_["rows"] += 1
        if r["verdict"]:
            f_["verdict"] += 1
            f_["verdict_match"] += r["match"] is True
            f_["unresolved"] += r["kind"] == "unresolved"
    failing = [r["key"] for r in vr if r["match"] is not True]
    pm = {r["key"]: r["match"] for r in rows if r["family"] == "rc1b_24"}
    pdm = {r["key"]: r["match"] for r in rows if r["family"] == "rc1b_24_display"}
    pboth = {k: pm.get(k) is True and pdm.get(f"display|{k}") is True for k in printed}  # 24g: my value AND the printed string
    vals = {"rows": rows, "controls": controls, "variant": VAR, "method": METHOD, "stops": stops, "near_caught_flips": near}
    vb = json.dumps(vals, default=float).encode()
    vpath = Path("/o", f"cleanroom_anchor_values_{stamp}.json")
    with vpath.open("xb") as fh:
        fh.write(vb)
    OUT_VOL.commit()
    return {"stamp": stamp, "n_stops": len(stops), "stops": stops[:40], "controls": controls, "by_family": fam,
            "n_verdict": len(vr), "n_verdict_match": sum(r["match"] is True for r in vr), "n_unresolved": sum(r["kind"] == "unresolved" for r in vr),
            "n_failing_verdict": len(failing), "failing_verdict_keys": failing[:200], "near_caught_flips": near,
            "printed": {"n": len(printed), "n_match": sum(pboth.values()), "failing": [k for k in printed if not pboth[k]],
                        "n_value_match": sum(pm.get(k) is True for k in printed), "n_display_match": sum(pdm.get(f"display|{k}") is True for k in printed)},
            "values_file": {"volume": OUT, "path": vpath.name, "sha256": hashlib.sha256(vb).hexdigest()},
            "label": "independent recompute from the producer's stated definition (POST HOC; no RULE registration)",
            "PASS": not stops and all(c["pass"] for c in controls) and not failing}


@app.local_entrypoint()
def main():
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    out = run.remote(stamp)
    path = HERE / "results" / f"cleanroom_anchor_{stamp}.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    print(json.dumps({k: out.get(k) for k in ("n_stops", "n_verdict", "n_verdict_match", "n_unresolved", "n_failing_verdict", "printed", "PASS")}))
    print("wrote", path)
