"""Continuous-score estimator on the full current cohort (09; this folder's RULE.md; POST HOC; PRELIMINARY).
Reads the CONTFULL dump of the all-origins analysis (run_dump_cont.py = run_xerr_scaleup.py 6v + a read-only capture) and
applies the overnight continuous estimator (contcore.py, byte-identical to experiments/2026-10-15_continuous_score_09) per
row x stratum: standardized shift D (own-pool honest-SD units), Append 19 on D, the first-order KDE mapping to catch
points, the rerun yardstick on the same estimator, and the location-shift diagnostics. Gates: contcore selftest;
hit identity; G0 (the binary family and per-probe values reproduce the all-origins record); single-origin equivalence
(the generalized mixed-origin code equals contcore.cell where both apply). Modal CPU.

  PYTHONPATH=src $MODAL run --detach experiments/2026-10-15_continuous_full_09/run_cont_full.py \
      --dump /work/contfull/<dump>.json --dump-sha <sha256> [--alltasks-note <commit>]
"""

import hashlib
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-convaudit_contfull_kill-2026-11-14"
VOL = "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14"
G0_REF = "g0_reference_ext_2026-10-15.json"
SEL_SEED = [20261014, 2]  # the all-tasks note's selection-half stream
ORIGINS = ("old", "new", "newmains")
REQ_ROWS = ("C|0.01", "C|0.05", "R|0.01", "R|0.05")
GATE2_U = 4.754140064621326  # the overnight record's v6 with-strategy reply-5% mapped U
SEL_LABEL = ("POST HOC: includes tasks used to choose settings (layer/readout, frozen eligibility); reuses selection-half "
             "outcomes; not the registered held-out family estimate")
STRATA = {
    "combined": lambda o, s: True,
    "S_old": lambda o, s: o == "old",
    "S_new": lambda o, s: o == "new",
    "S_newmains": lambda o, s: o == "newmains",
    "combined|with_strategy": lambda o, s: s,
    "combined|without_strategy": lambda o, s: not s,
}

app = modal.App(APP)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy==2.3.3")
    .add_local_file(HERE / "contcore.py", "/root/contcore.py")
    .add_local_file(HERE / G0_REF, f"/root/{G0_REF}")
)
vol = modal.Volume.from_name(VOL, version=2)


class Stop(Exception):
    pass


def make_draws(units, stratum, seed, draws):
    """Verbatim from run_xerr_scaleup.py (make_draws), with DRAWS passed in."""
    import numpy as np

    rng = np.random.default_rng(seed)
    W = np.zeros((draws + 1, len(units)))
    W[0] = 1.0
    g = [stratum[u] for u in units]
    for s in sorted(set(g)):
        idx = np.array([i for i, v in enumerate(g) if v == s], dtype=np.intp)
        W[1:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=draws)
    return W, {u: i for i, u in enumerate(units)}


def _analyze(dump: dict, ref: dict, sel_on: bool) -> dict:  # noqa: C901
    import numpy as np

    import contcore as cc

    out = {"selftest": cc.selftest()}
    ctx = dump["context"]
    cohort = ctx["cohort"]
    pos = {c: i for i, c in enumerate(cohort)}
    origin, main_of, has_strat = ctx["origin"], ctx["main_of"], ctx["has_strat"]
    D = int(ctx["draws"])
    strat_x = {m: tuple(v) for m, v in ctx["univ_x_stratum"].items()}
    Wx, px = make_draws(ctx["univ_x"], strat_x, list(ctx["seed"]), D)
    mains_est = ctx["mains_est"]
    com = {m: list(cs) for m, cs in ctx["cells_of_main"].items()}
    if not all(ctx["cells_eff_extended"][c] for m in mains_est for c in com[m]):
        raise Stop("an estimation cell is outside the extended analyzed cohort")
    sel = ctx["sel_mains"]
    sel_cells = {}
    for c in cohort:
        if main_of[c] in sel:
            sel_cells.setdefault(main_of[c], []).append(c)
    W_sel, p_sel = make_draws(sel, {m: ctx["sel_main_env"][m] for m in sel}, SEL_SEED, D)
    if tuple(sorted(dump["rows"])) != REQ_ROWS:  # Codex r1 item 2: exactly the four registered rows
        raise Stop(f"dump rows {sorted(dump['rows'])} != {list(REQ_ROWS)}")
    for K_ in REQ_ROWS:
        if dump["rows"][K_]["eligible"] != ref["rows"][K_]["eligible"]:
            raise Stop(f"eligible set differs from the record: {K_}")
        for st_ in STRATA:
            for cmp_ in ("sim_minus_mean_real", "reexecution_run2_minus_run1"):
                r_ = ref["rows"][K_]["strata"].get(st_, {}).get(cmp_)
                if r_ is None or any(p_ not in r_["per_probe"] for p_ in dump["rows"][K_]["eligible"]):
                    raise Stop(f"incomplete G0 reference: {K_} {st_} {cmp_}")
    est_ix = [pos[c] for m in mains_est for c in com[m]]
    sel_ix = [pos[c] for cs in sel_cells.values() for c in cs]
    for K_, row_ in dump["rows"].items():  # Codex r1 item 3: finite scores and thresholds for every analyzed cell
        for p_, pr_ in row_["probes"].items():
            if not all(isinstance(pr_["T"].get(o), (int, float)) and math.isfinite(pr_["T"][o]) for o in ORIGINS):
                raise Stop(f"non-finite threshold: {K_} {p_}")
            for w_ in ("run1", "run2", "sim"):
                for i_ in est_ix + (sel_ix if sel_on else []):
                    v_ = pr_["cohort_z"][w_][i_]
                    if v_ is None or not math.isfinite(v_):
                        raise Stop(f"non-finite transformed score: {K_} {p_} {w_} {cohort[i_]}")

    def design(sets):
        """sets: 'est' (the 606) or 'sel' (SEL alone) or 'all' (606 + SEL). Returns (mains, cells_of_main, W)."""
        ms, cm, cols = [], {}, []
        if sets in ("est", "all"):
            for m in mains_est:
                ms.append(("est", m))
                cm[("est", m)] = com[m]
            cols.append(Wx[:, [px[m] for m in mains_est]])
        if sets in ("sel", "all"):
            for m in sorted(sel_cells):
                ms.append(("sel", m))
                cm[("sel", m)] = sel_cells[m]
            cols.append(W_sel[:, [p_sel[m] for m in sorted(sel_cells)]])
        return ms, cm, np.hstack(cols)

    def pool_for(pr, o):
        se, sp = pr["scheme_extended"], pr["scheme_primary"]
        if se == "full_union":
            return pr["pool_keys_old"] + pr["pool_keys_new"] + pr["pool_keys_newmains"]
        if se != "separate_newmains":
            raise Stop(f"unknown extended scheme {se}")
        if o == "newmains":
            return pr["pool_keys_newmains"]
        if sp == "union":
            return pr["pool_keys_old"] + pr["pool_keys_new"]
        if sp == "stratum_specific":
            return pr["pool_keys_old"] if o == "old" else pr["pool_keys_new"]
        raise Stop(f"unknown primary scheme {sp}")

    def probe_arrays(pr):
        """Per-origin standardization s' = (z - mu_o) / sd_o on the pool that set origin o's threshold; T'_o likewise.
        Hit identity: s' > T'_o equals z > T_o for every cohort cell and arm, else stop."""
        z = {w: np.array([np.nan if v is None else v for v in pr["cohort_z"][w]], float) for w in ("run1", "run2", "sim")}
        mom, Tp = {}, {}
        for o in ORIGINS:
            try:
                mu, sd = cc.pool_moments([pr["pool_z"][k] for k in pool_for(pr, o)])
            except cc.CoreError as e:
                raise Stop(f"pool moments: {e}") from e
            mom[o], Tp[o] = (mu, sd), (pr["T"][o] - mu) / sd
        org = np.array([origin[c] for c in cohort])
        mu_v = np.array([mom[o][0] for o in org])
        sd_v = np.array([mom[o][1] for o in org])
        T_v = np.array([pr["T"][o] for o in org])
        Tp_v = np.array([Tp[o] for o in org])
        s = {w: (z[w] - mu_v) / sd_v for w in z}
        h = {w: (z[w] > T_v).astype(float) for w in z}
        for w in z:
            fin = np.isfinite(z[w])
            if not ((s[w][fin] > Tp_v[fin]) == (z[w][fin] > T_v[fin])).all():
                raise Stop("hit identity failed")
        return s, h, z, Tp, mom

    def row_cell(s, h, Tp, mains, cm, W, keep):
        """Generalized cell over mains (cell-within-main means; equal-main weights by W), with the per-origin KDE mapping
        M_d = 100 sum_o f_o,d (W_d . a_o) / (W_d . 1), a_o[m] = (n_mo / n_m) d_m(o). Reduces to contcore.cell for one
        origin with one cell per main (checked by the equivalence gate)."""
        nM = len(mains)
        sel_c = [[pos[c] for c in cm[m] if keep(c)] for m in mains]
        idx = [j for j in range(nM) if sel_c[j]]
        mains_k = [mains[j] for j in idx]
        cells = [sel_c[j] for j in idx]
        Wk = W[:, idx]

        def mm(v):
            return np.array([np.mean(v[cs]) for cs in cells])

        hs, h1, h2 = mm(h["sim"]), mm(h["run1"]), mm(h["run2"])
        ss, s1, s2 = mm(s["sim"]), mm(s["run1"]), mm(s["run2"])
        out_ = {"b": cc.wmean(Wk, hs - (h1 + h2) / 2), "br": cc.wmean(Wk, h2 - h1),
                "d": cc.wmean(Wk, ss - (s1 + s2) / 2), "dr": cc.wmean(Wk, s2 - s1)}
        f_by, h_by = {}, {}
        m_tot = np.zeros(W.shape[0])
        mr_tot = np.zeros(W.shape[0])
        mh_tot = np.zeros(W.shape[0])
        md_tot = np.zeros(W.shape[0])
        wsum = Wk.sum(1)
        any_nan = False
        for o in ORIGINS:
            oc = [[c for c in cs if origin[cohort[c]] == o] for cs in cells]
            oj = [j for j, x in enumerate(oc) if x]
            if not oj:
                continue
            real = np.concatenate([np.concatenate([s["run1"][x], s["run2"][x]]) for x in (oc[j] for j in oj)])
            ho = cc.bandwidth(real)
            h_by[o] = ho if math.isfinite(ho) else None
            share = np.array([len(oc[j]) / len(cells[j]) if oc[j] else 0.0 for j in range(len(cells))])
            a_d = np.array([share[j] * (np.mean(s["sim"][oc[j]] - (s["run1"][oc[j]] + s["run2"][oc[j]]) / 2) if oc[j] else 0.0) for j in range(len(cells))])
            a_r = np.array([share[j] * (np.mean(s["run2"][oc[j]] - s["run1"][oc[j]]) if oc[j] else 0.0) for j in range(len(cells))])
            Wo = Wk[:, oj]

            def dens(hh, oc=oc, oj=oj, Wo=Wo, o=o):
                if not (math.isfinite(hh) and hh > 0):
                    return np.full(W.shape[0], np.nan)
                kv = np.array([np.mean((cc.phi((Tp[o] - s["run1"][oc[j]]) / hh) + cc.phi((Tp[o] - s["run2"][oc[j]]) / hh)) / (2 * hh)) for j in oj])
                return cc.wmean(Wo, kv)

            fo = dens(ho)
            f_by[o] = fo
            with np.errstate(invalid="ignore", divide="ignore"):
                m_tot = m_tot + 100 * fo * (Wk @ a_d) / wsum
                mr_tot = mr_tot + 100 * fo * (Wk @ a_r) / wsum
                mh_tot = mh_tot + 100 * dens(ho / 2) * (Wk @ a_d) / wsum
                md_tot = md_tot + 100 * dens(2 * ho) * (Wk @ a_d) / wsum
            any_nan |= not math.isfinite(ho)
        out_["m"], out_["mr"], out_["m_h_half"], out_["m_h_double"] = m_tot, mr_tot, mh_tot, md_tot
        with np.errstate(invalid="ignore", divide="ignore"):
            share_o = {o: np.array([sum(1 for c in cs if origin[cohort[c]] == o) / len(cs) for cs in cells]) for o in f_by}
            out_["f"] = sum(f_by[o] * (Wk @ share_o[o]) / wsum for o in f_by)
        out_["n"] = np.full(W.shape[0], np.nan)  # the nonlinear diagnostic is computed only by contcore.cell (single origin)
        out_["desc"] = {"bandwidth_by_origin": h_by, "mains": len(mains_k), "cells": int(sum(len(c) for c in cells)),
                        "nonlinear_diagnostic": "computed only for single-origin rows (contcore.cell)"}
        out_["_single"] = (len(f_by) == 1 and all(len(c) == 1 for c in cells), mains_k, cells, Wk)
        return out_

    def single_cell(s, h, Tp, o, cells, Wk):
        ix = [c[0] for c in cells]
        return cc.cell(s["run1"][ix], s["run2"][ix], s["sim"][ix], h["run1"][ix], h["run2"][ix], h["sim"][ix], Tp[o], Wk)

    gates = {"hit_identity": True, "g0": [], "single_origin_equivalence": []}
    results = {}
    sets = ["est"] + (["sel", "all"] if sel_on else [])
    for K, row in sorted(dump["rows"].items()):
        el = row["eligible"]
        arrs = {p: probe_arrays(row["probes"][p]) for p in el}
        results[K] = {"eligible": el, "standardization": {p: {o: {"mu": arrs[p][4][o][0], "sd": arrs[p][4][o][1], "T_std": arrs[p][3][o]} for o in ORIGINS} for p in el}}
        for st in sets:
            mains, cm, W = design(st)
            for sname, sf in STRATA.items():
                if st != "est" and sname not in ("combined", "S_old"):
                    continue
                keep = lambda c, sf=sf: sf(origin[c], has_strat[c])  # noqa: E731  (c is a cell name)
                cells_ = []
                for p in el:
                    s_, h_, z_, Tp_, mom_ = arrs[p]
                    c_ = row_cell(s_, h_, Tp_, mains, cm, W, keep)
                    single, mk, cl, Wk = c_.pop("_single")
                    if single:  # use contcore.cell where it applies; the generalized code must agree (gate)
                        o_ = next(o for o in ORIGINS if all(origin[cohort[x[0]]] == o for x in cl))
                        cs_ = single_cell(s_, h_, Tp_, o_, cl, Wk)
                        diffs, ok = {}, True
                        for k in ("b", "br", "d", "dr", "m", "mr", "m_h_half", "m_h_double"):  # Codex r1 item 4
                            a1, a2 = cs_[k], c_[k]
                            if np.isinf(a1).any() or np.isinf(a2).any() or not (np.isfinite(a1) == np.isfinite(a2)).all():
                                ok, diffs[k] = False, float("inf")
                                continue
                            f_ = np.isfinite(a1)
                            diffs[k] = float(np.max(np.abs(a1[f_] - a2[f_]))) if f_.any() else 0.0
                        ok = ok and all(v <= 1e-9 for v in diffs.values())
                        gates["single_origin_equivalence"].append({"row": K, "set": st, "stratum": sname, "probe": p, "max_abs_diff": max(diffs.values()), "pass": ok})
                        if not ok:
                            raise Stop(f"single-origin equivalence failed: {K} {sname} {p}")
                        c_ = cs_
                    cells_.append(c_)
                blk = cc.family_block(cells_, el)
                blk["mains"] = sum(1 for m in mains if any(keep(c) for c in cm[m]))
                blk["set"] = st
                results[K].setdefault(st, {})[sname] = blk
                if st != "est":
                    blk["label"] = SEL_LABEL  # Codex r1 item 5
                if st == "est":  # G0 against the all-origins record (Codex r1 item 1: both comparisons, full coverage)
                    chk = {"row": K, "stratum": sname}
                    errs, avail = [], []
                    for cmp_, fam_key, pp_key in (
                        ("sim_minus_mean_real", "family_binary_points", "binary_delta_points"),
                        ("reexecution_run2_minus_run1", "family_reexecution_binary_points", "reexecution_binary_points"),
                    ):
                        rf = ref["rows"][K]["strata"][sname][cmp_]
                        need = ("mains", "X_err", "U", "bias_mean", "per_probe")  # Codex r2: explicit presence
                        if any(k_ not in rf for k_ in need) or any(
                            k_ not in rf["per_probe"].get(p, {}) for p in el for k_ in ("delta", "ci95")
                        ):
                            raise Stop(f"reference field missing: {K} {sname} {cmp_}")
                        fb = blk[fam_key]
                        if blk["mains"] != rf["mains"]:
                            avail.append(f"{cmp_} mains {blk['mains']} vs {rf['mains']}")
                        for a_, b_ in (("X_err", "X_err"), ("U_approx_upper_95", "U"), ("bias_mean", "bias_mean")):
                            if (fb.get(a_) is None) != (rf.get(b_) is None):
                                avail.append(f"{cmp_} {a_} availability")
                            elif fb.get(a_) is not None:
                                errs.append(abs(fb[a_] - rf[b_]))
                        for p in el:
                            pp, mine = rf["per_probe"][p], blk["per_probe"][p][pp_key]
                            if (mine["point"] is None) != (pp["delta"] is None):
                                avail.append(f"{cmp_} {p} delta availability")
                            elif mine["point"] is not None:
                                errs.append(abs(mine["point"] - pp["delta"]))
                            if (mine["ci95"] is None) != (pp["ci95"] is None):
                                avail.append(f"{cmp_} {p} ci95 availability")
                            elif mine["ci95"] is not None:
                                errs.extend(abs(x - y) for x, y in zip(mine["ci95"], pp["ci95"], strict=True))
                    chk["n_values_compared"] = len(errs)
                    if any(not math.isfinite(e_) for e_ in errs):  # Codex r2: a non-finite difference fails G0
                        avail.append("non-finite compared value")
                    chk["max_abs_diff"] = max(errs) if errs and all(math.isfinite(e_) for e_ in errs) else None
                    chk["availability_or_count_mismatch"] = avail
                    chk["pass"] = (not avail) and chk["max_abs_diff"] is not None and chk["max_abs_diff"] <= 1e-9
                    gates["g0"].append(chk)
                    if not chk["pass"]:
                        raise Stop(f"G0 failed: {K} {sname} {chk}")
    out["gates"] = gates
    out["results"] = results
    return out


@app.function(image=image, cpu=4, memory=16384, timeout=1800, retries=0, volumes={"/work": vol})
def run(dump_path: str, dump_sha: str, sel_on: bool) -> dict:
    vol.reload()
    raw = Path(dump_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != dump_sha:
        return {"fatal": "dump sha256 mismatch"}
    ref = json.loads(Path(f"/root/{G0_REF}").read_text())
    try:
        res = _analyze(json.loads(raw), ref, sel_on)
    except Stop as e:
        res = {"fatal": "Stop", "why": str(e)}
    return res


@app.local_entrypoint()
def main(dump: str = "", dump_sha: str = "", alltasks_note: str = "", gate2_record: str = ""):
    def git(*a):
        return subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()

    if git("status", "--porcelain", "--untracked-files=no", "--", str(HERE)):
        raise SystemExit("refusing to run: commit this folder first")
    if not dump or not dump_sha:
        raise SystemExit("--dump and --dump-sha are required")
    if not gate2_record:  # gate 2 (machine-checked): the overnight re-run's record must reproduce U exactly
        raise SystemExit("--gate2-record <committed overnight re-run record> is required")
    raw2 = (REPO / gate2_record).read_bytes()  # Codex r2: committed at HEAD, its own gates passing, U finite and equal
    if subprocess.run(["git", "show", f"HEAD:{gate2_record}"], capture_output=True, check=True, cwd=REPO).stdout != raw2:
        raise SystemExit("gate 2: the record is not the committed file at HEAD")
    r2_ = json.loads(raw2)["result"]
    g2g = r2_["gates"]
    if not (g2g.get("pass") is True and all(v is True for v in g2g["selftest_fixtures"].values())
            and g2g["gate1_max_abs_float_diff"] <= 1e-12 and len(g2g["gate1_rows_reproduce_v6_primary"]) == 4):
        raise SystemExit("gate 2: the re-run's own gates did not pass")
    u2 = r2_["results"]["R_reply_meanpooled|mechanical|alpha=0.05"]["continuous"]["with_strategy"]["family_mapped_points"]["U_approx_upper_95"]
    if not (isinstance(u2, float) and math.isfinite(u2) and abs(u2 - GATE2_U) <= 1e-9):
        raise SystemExit("gate 2 failed: the overnight re-run does not reproduce U")
    if alltasks_note:  # ALL/SEL only after 09's all-tasks note is committed, verified at that commit (Codex r1 item 5)
        rule_ = git("show", f"{alltasks_note}:experiments/2026-10-14_priority1_spec_09/RULE.md")
        if '### POST HOC "all tasks" robustness row' not in rule_ or "seed [20261014, 2]" not in rule_:
            raise SystemExit("the given commit does not hold 09's all-tasks note with its draw design")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    res = run.remote(dump, dump_sha, bool(alltasks_note))
    rec = {"git": git("rev-parse", "HEAD"), "stamp": stamp, "gate2_record": gate2_record, "label": "POST HOC; PRELIMINARY (continuous estimator, full cohort)",
           "dump": {"path": dump, "sha256": dump_sha}, "alltasks_note": alltasks_note or None,
           "contcore_sha256": hashlib.sha256((HERE / "contcore.py").read_bytes()).hexdigest(),
           "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "result": res}
    dest = HERE / "results" / f"contfull_{stamp}.json"
    dest.parent.mkdir(exist_ok=True)
    dest.open("x").write(json.dumps(rec, indent=1, default=float))
    g = (res.get("gates") or {})
    print(json.dumps({"fatal": res.get("fatal"), "why": res.get("why"), "selftest": res.get("selftest"),
                      "g0_pass": all(c["pass"] for c in g.get("g0", [])) if g else None, "g0_n": len(g.get("g0", [])),
                      "single_origin_equiv_n": len(g.get("single_origin_equivalence", []))}, indent=1))
    print("wrote", dest.name)
