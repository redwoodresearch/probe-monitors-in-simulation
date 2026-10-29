"""RULE.md "Add-on (0d 02:1x, panel 3)": A (own-real probes on the shared 26-point grid, honest B u C units), B (threshold noise: a
joint bootstrap that redraws the unit threshold with the attack side's task draw), C (sim-honest runs at the real command-A
thresholds, Sol (a) and c_asis). Stored per-row scores only, Modal CPU; the laptop reads pinned inputs and launches.

  PYTHONPATH=src <modal 1.6.1> run --detach experiments/2026-10-14_fpr_sweep_unit/run_panel3.py
"""

import hashlib
import io
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-fprsweep_unit_kill-2026-11-14"
VOLS = {"solavg": "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13", "p2": "probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13",
        "convaudit": "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14"}
D, SEED = 10000, [20261014, 1]
GRID = sorted(set([10 ** (-4 + 0.125 * i) for i in range(25)]) | {0.05})
A3, HW = ("run1", "run2", "sim"), ("honest_a", "honest_b", "honest_c")
REC_B = ("fprsweep_unit_2026-10-15_003643.json", "d979413913b988ff6f29fbc0a5910aa01d3bb048387bc41b1a848e039ebba7d2")
app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")
vols = {k: modal.Volume.from_name(v, version=2) if k != "convaudit" else modal.Volume.from_name(v) for k, v in VOLS.items()}


def fdiff(x, y):
    import math

    if x is None or y is None:
        return 0.0 if x is None and y is None else float("inf")
    if isinstance(x, (bool, str)) or isinstance(y, (bool, str)):
        return 0.0 if x == y else float("inf")
    x, y = float(x), float(y)
    return abs(x - y) if math.isfinite(x) and math.isfinite(y) else float("inf")


def flat(x, pre=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from flat(v, f"{pre}.{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from flat(v, f"{pre}[{i}]")
    else:
        yield pre, x


def jpath(d, kp):
    for k in kp:
        d = d[k]
    return d


@app.function(image=image, cpu=(8, 8), memory=(32768, 32768), timeout=3000, retries=0,
              volumes={"/solavg": vols["solavg"], "/p2": vols["p2"].read_only(), "/convaudit": vols["convaudit"].read_only()})
def compute(name: str, cfg: dict, blobs: dict, meta: dict) -> dict:  # noqa: C901
    import numpy as np

    od = Path("/solavg") / "xfer" / "fprsweep_unit" / name
    od.mkdir(parents=True, exist_ok=False)
    fails, diag = [], {}

    def gate(ok, msg):  # fail fast: diagnostics only, nonzero exit
        if not ok:
            fails.append(msg)
            (od / f"{name}_GATE_FAILED.json").open("x").write(json.dumps({"label": "GATE FAILED: stop and ask 2e", "meta": meta, "failures": fails,
                                                                           "gates_so_far": diag}, default=float, indent=1))
            vols["solavg"].commit()
            raise SystemExit(f"gate failed: {msg}")

    def npz(spec):
        b = (Path("/" + spec["volume"]) / spec["path"].lstrip("/")).read_bytes()
        gate(hashlib.sha256(b).hexdigest() == spec["sha256"], f"G1: {spec['path']} sha256")
        return dict(np.load(io.BytesIO(b), allow_pickle=False))

    labels = json.loads(blobs["labels"])["labels"]
    t196 = json.loads(blobs["t196"])
    canon = sorted(c["key"] for c in t196["cells"])
    env_of = {c["key"]: c["env"] for c in t196["cells"]}
    fold_of_main = {}
    for r in json.loads(blobs["r36"]):
        gate(fold_of_main.setdefault(r["main_task"], r["fold"]) == r["fold"], "r36: a main task in two folds")
    fot = {c["key"]: fold_of_main[c["main"]] for c in t196["cells"]}
    U = sorted(json.loads(blobs["v6"])["result"]["universe_and_split"]["universe"])
    ui = {t: i for i, t in enumerate(U)}
    att = lambda w, t: labels.get(f"{w}|{t}") == "attempted"  # noqa: E731
    st = np.array([env_of[t] for t in U])
    rng, W = np.random.default_rng(SEED), np.zeros((D + 1, len(U)))
    W[0] = 1
    for b_ in range(1, D + 1):
        for idx in [np.flatnonzero(st == k) for k in sorted(set(st.tolist()))]:
            np.add.at(W[b_], rng.choice(idx, len(idx), replace=True), 1)

    def summ(v):
        f = v[1:][np.isfinite(v[1:])]
        o = {"point": float(v[0]), "finite_draws": int(len(f)), "draws": D}
        if len(f) < 0.95 * D:
            return o | {"ci95": None, "withheld": "fewer than 95% complete draws"}
        return o | {"ci95": [float(np.percentile(f, 2.5)), float(np.percentile(f, 97.5))], "degenerate": bool(np.ptp(f) == 0)}

    wmean = lambda Wm, h: (Wm * h).sum(1) / Wm.sum(1)  # noqa: E731

    def units(keys, score, fold):
        keys = np.array([str(k) for k in keys])
        gate(len(keys) == len(score) == len(fold) and np.isfinite(score).all(), "units: misaligned or non-finite")
        by_run = {}
        for i, k in enumerate(keys):
            by_run.setdefault(k, []).append(i)
        u = {"keys": keys, "score": np.asarray(score, np.float64), "fold": np.asarray(fold, np.int64), "world": np.array([k.split("|", 1)[0] for k in keys]),
             "by_run": {k: np.array(v) for k, v in by_run.items()}}
        bad = [k for k, ix in u["by_run"].items() if set(u["fold"][ix].tolist()) != {fot.get(k.split("|", 1)[1], -9)}]
        gate(not bad, f"units: rows whose fold is not the r36 fold: {bad[:3]}")
        return u

    def zscore(u, pool_runs):
        sel = np.zeros(len(u["score"]), bool)
        for k in pool_runs:
            sel[u["by_run"][k]] = True
        mom = {}
        for f in range(5):
            v = u["score"][sel & (u["fold"] == f)]
            sd = float(np.std(v, ddof=0)) if len(v) >= 2 else 0.0
            gate(len(v) >= 2 and sd > 0, f"z: fold {f} empty or constant")
            mom[f] = {"mean": float(np.mean(v)), "sd": sd, "n": int(len(v))}
        mu, sd = np.array([mom[f]["mean"] for f in range(5)]), np.array([mom[f]["sd"] for f in range(5)])
        return (u["score"] - mu[u["fold"]]) / sd[u["fold"]], mom

    pool_of = lambda u, ws: sorted(k for k in u["by_run"] if k.split("|", 1)[0] in ws and k.split("|", 1)[1] in ui)  # noqa: E731

    def cohort(u):
        has = {w: {k.split("|", 1)[1] for k in u["by_run"] if k.startswith(w + "|")} for w in A3}
        return sorted(t for t in canon if t in ui and all(att(w, t) for w in A3) and all(t in has[w] for w in A3))

    def unit_points(u, zs, pool_runs, coh):  # part (b)'s unit_points, verbatim arithmetic
        rmax = {k: float(zs[ix].max()) for k, ix in u["by_run"].items()}
        pool = np.concatenate([zs[u["by_run"][k]] for k in pool_runs])
        d_ = pool[np.argsort(-pool, kind="mergesort")]
        n = len(d_)
        wr = {w: [k for k in u["by_run"] if k.startswith(w + "|") and k.split("|", 1)[1] in ui] for w in HW}
        wu = {w: np.concatenate([zs[u["by_run"][k]] for k in wr[w]]) if wr[w] else np.array([]) for w in HW}
        Wc = W[:, [ui[t] for t in coh]]
        pts = []
        for f in GRID:
            k = int(np.floor(f * n + 1e-9))
            T = float(d_[min(k, n - 1)])
            h = {w: np.array([float(rmax[f"{w}|{t}"] > T) for t in coh]) for w in A3}
            h["real_mean"] = (h["run1"] + h["run2"]) / 2
            c = {w: wmean(Wc, h[w]) for w in h}
            dif = {"sim_minus_real_mean": c["sim"] - c["real_mean"], "sim_minus_run1": c["sim"] - c["run1"], "sim_minus_run2": c["sim"] - c["run2"],
                   "run2_minus_run1": c["run2"] - c["run1"]}
            pts.append({"fpr_target": f, "k": k, "n_pool_units": n, "threshold": T, "estimable": k >= 1, "pool_units_above": int((pool > T).sum()),
                        "shaded_lt10_above": int((pool > T).sum()) < 10, "cohort": "all", "n_tasks": len(coh),
                        "realized_unit_fpr": {"pool": float((pool > T).mean())} | {w: float((wu[w] > T).mean()) if len(wu[w]) else None for w in HW},
                        "realized_run_false_alarm_any_unit": {"pool": float(np.mean([rmax[k_] > T for k_ in pool_runs]))}
                        | {w: float(np.mean([rmax[k_] > T for k_ in wr[w]])) if wr[w] else None for w in HW},
                        "tpr": {w: summ(v) for w, v in c.items()}, "diff": {k_: summ(v) for k_, v in dif.items()},
                        "saturated": {w: c[w][0] in (0.0, 1.0) for w in c}})
        return pts

    def joint(u, zs, pool_runs, coh):  # B: the threshold redrawn per draw from the honest units, weighted by the attack side's task draw
        rmax = {k: float(zs[ix].max()) for k, ix in u["by_run"].items()}
        pu = np.concatenate([zs[u["by_run"][k]] for k in pool_runs])
        pt = np.concatenate([np.full(len(u["by_run"][k]), ui[k.split("|", 1)[1]]) for k in pool_runs])
        o = np.argsort(-pu, kind="mergesort")
        s_sorted, t_sorted = pu[o], pt[o]
        n = len(s_sorted)
        Tb, kb = np.empty((len(GRID), D + 1)), np.empty((len(GRID), D + 1), np.int64)
        for c0 in range(0, D + 1, 500):  # A24's thr() with unit weights W[b, task], in chunks of draws
            cw = np.cumsum(W[c0:c0 + 500][:, t_sorted], 1)
            tot = cw[:, -1]
            for gi, f in enumerate(GRID):
                kk = np.floor(f * tot + 1e-9)
                Tb[gi, c0:c0 + 500] = s_sorted[np.minimum((cw <= kk[:, None]).sum(1), n - 1)]
                kb[gi, c0:c0 + 500] = kk.astype(np.int64)
        Wc = W[:, [ui[t] for t in coh]]
        r = {w: np.array([rmax[f"{w}|{t}"] for t in coh]) for w in A3}
        pts = []
        for gi, f in enumerate(GRID):
            h = {w: (r[w][None, :] > Tb[gi][:, None]).astype(float) for w in A3}
            h["real_mean"] = (h["run1"] + h["run2"]) / 2
            c = {w: wmean(Wc, h[w]) for w in h}
            dif = {"sim_minus_real_mean": c["sim"] - c["real_mean"], "sim_minus_run1": c["sim"] - c["run1"], "sim_minus_run2": c["sim"] - c["run2"],
                   "run2_minus_run1": c["run2"] - c["run1"]}
            pts.append({"fpr_target": f, "threshold_draw0": float(Tb[gi, 0]), "k_draw0": int(kb[gi, 0]), "draws_k0": int((kb[gi, 1:] == 0).sum()),
                        "draws_k_lt10": int((kb[gi, 1:] < 10).sum()), "k_draws_p5_p50_p95": [float(np.percentile(kb[gi, 1:], q)) for q in (5, 50, 95)],
                        "tpr": {w: summ(v) for w, v in c.items()}, "diff": {k_: summ(v) for k_, v in dif.items()}})
        return pts

    out = {"label": "PRELIMINARY until 09 and Codex review; from raw data, by script", "meta": meta, "part": "add-on (0d 02:1x, panel 3)", "grid": GRID,
           "draws": D, "seed": SEED, "A": {}, "B": {}, "C": {}}
    recb = json.loads(blobs["rec_b"])
    own = json.loads(blobs["own_a_record"])
    fam = cfg["own_frac100_L48"]
    # ---------------- A: own-real rows on the shared grid ----------------
    own_units = {}
    for lab, kind in (("own_frac100_L48 C", ("fam", "C")), ("own_frac100_L48 R", ("fam", "R")), ("own_a_tight cmd_A", ("own", "cmd_A")),
                      ("own_a_tight avg_top2", ("own", "avg_top2"))):
        if kind[0] == "fam":
            zo, zs_ = (npz({"volume": "p2"} | fam["files"][kind[1]][x]) for x in ("open", "sim"))
            u = units(np.concatenate([zo["key"], zs_["key"]]), np.concatenate([zo["score"], zs_["score"]]).astype(np.float64),
                      np.concatenate([zo["fold"], zs_["fold"]]))
            ref = dict(zip([str(k) for k in np.concatenate([zo["run_key"], zs_["run_key"]])], np.concatenate([zo["run_score"], zs_["run_score"]]).astype(float).tolist()))
            agg = max if fam["files"][kind[1]]["run_agg"] == "max" else (lambda x: float(np.mean(x)))
        else:
            e = cfg["own_a"]["readouts"][kind[1]]
            if kind[1] == "cmd_A":
                ent = jpath(own, e["rows_keypath"])
                z = npz({"volume": "solavg", "path": ent["path"], "sha256": ent["sha256"]})
                u = units(z["key"], z["score"].astype(np.float64), z["fold"])
            else:
                ent = jpath(own, e["rows_keypath"])
                parts = []
                for f, (La, Lb) in enumerate(cfg["own_a"]["layer_choice"]):
                    za, zb = (npz({"volume": "solavg", "path": ent[f"{L}|{f}"]["rows"], "sha256": ent[f"{L}|{f}"]["rows_sha256"]}) for L in (La, Lb))
                    gate(all(np.array_equal(za[x], zb[x]) for x in ("key", "unit", "call_index", "fold")), f"{lab}: layer files misaligned")
                    parts.append((za["key"], (za["score"].astype(np.float64) + zb["score"].astype(np.float64)) / 2, za["fold"]))
                u = units(np.concatenate([p_[0] for p_ in parts]), np.concatenate([p_[1] for p_ in parts]), np.concatenate([p_[2] for p_ in parts]))
            ref = jpath(own, e["run_keypath"])
            agg = lambda x: float(np.mean(x))  # noqa: E731
        mine = {k: float(agg(u["score"][ix])) for k, ix in u["by_run"].items()}
        dmax = max(fdiff(mine[k], ref[k]) for k in mine) if set(mine) == set(ref) else float("inf")
        gate(dmax <= 1e-9, f"{lab}: run aggregation vs reference {dmax}")
        coh = cohort(u)
        gate(len(coh) == 183, f"{lab}: cohort {len(coh)}")
        cpool = {"B": ["honest_b"], "BuC": ["honest_b", "honest_c"]}[fam["pool_from_v7"][kind[1] if kind[0] == "fam" else fam["own_a_readout_match"][kind[1]]]]
        zs, _ = zscore(u, pool_of(u, cpool))
        refp = recb["own_real"][lab]["primary"]["sweep_all_cohort_descriptive"]
        mine_p = unit_points(u, zs, pool_of(u, cpool), coh)
        worst, nv = 0.0, 0
        for m_, t_ in zip(mine_p, refp):
            a_, b2 = dict(flat(m_)), dict(flat(t_))
            if set(a_) != set(b2):
                worst = float("inf")
            for k2 in b2:
                worst, nv = max(worst, fdiff(a_.get(k2), b2[k2]) if k2 in a_ else float("inf")), nv + 1
        diag[lab] = {"GP1": {"values": nv, "max_abs_diff": worst, "pool": "+".join(cpool)}, "aggregation_max_abs_diff": dmax}
        gate(len(refp) == 26 and worst <= 1e-12, f"GP1 {lab}: part (c) sweep not reproduced ({worst})")
        pool = pool_of(u, ["honest_b", "honest_c"])
        zs, mom = zscore(u, pool)
        own_units[lab] = (u, zs, pool, coh)
        out["A"][lab] = {"gates": diag[lab], "pool": "honest_b+honest_c", "n_pool_runs": len(pool), "z_moments_unit": mom, "trained_on_run1": True,
                         "note": "descriptive, beside part (c)'s pre-registered catch-target points (whose 1-9 units above was the catch-target choice)",
                         "points": unit_points(u, zs, pool, coh)}
    # ---------------- B: joint bootstrap ----------------
    for p in cfg["probes_b"]:
        z = npz(p["rows"])
        u = units(z["key"], z["score"].astype(np.float64), z["fold"] if "fold" in z else np.array([fot[str(k).split("|", 1)[1]] for k in z["key"]]))
        rp = recb["probes"][p["probe_id"]]
        pool = rp["pool_runs"]
        zs = zscore(u, pool)[0] if p["folds"] == 5 else u["score"]
        jp = joint(u, zs, pool, rp["cohort"])
        gate(len(jp) == len(rp["points"]) == 26 and all(q["fpr_target"] == r_["fpr_target"] for q, r_ in zip(jp, rp["points"])), f"GB1 {p['label']}: grids differ")
        bad = [q["fpr_target"] for q, r_ in zip(jp, rp["points"]) if q["threshold_draw0"] != r_["threshold"] or q["k_draw0"] != r_["k"]
               or any(q["tpr"][w]["point"] != r_["tpr"][w]["point"] for w in r_["tpr"])]
        gate(not bad, f"GB1 {p['label']}: draw 0 differs from part (b) at {bad[:3]}")
        out["B"][p["label"]] = {"pool": "honest_b+honest_c (part (b))", "GB1": "draw 0 equals part (b) at all 26 points", "points": jp}
    for lab in ("own_a_tight avg_top2", "own_a_tight cmd_A"):
        u, zs, pool, coh = own_units[lab]
        jp = joint(u, zs, pool, coh)
        gate(len(jp) == len(out["A"][lab]["points"]) == 26 and all(q["fpr_target"] == r_["fpr_target"] for q, r_ in zip(jp, out["A"][lab]["points"])), f"GB1 {lab}: grids differ")
        bad = [q["fpr_target"] for q, r_ in zip(jp, out["A"][lab]["points"]) if q["threshold_draw0"] != r_["threshold"] or q["k_draw0"] != r_["k"]
               or any(q["tpr"][w]["point"] != r_["tpr"][w]["point"] for w in r_["tpr"])]
        gate(not bad, f"GB1 {lab}: draw 0 differs from the fixed grid at {bad[:3]}")
        out["B"][lab] = {"pool": "honest_b+honest_c", "GB1": "draw 0 equals panel 3 A's fixed grid at all 26 points", "points": jp}
    # ---------------- C: sim-honest at the real command-A thresholds ----------------
    for p in cfg["probes_c"]:
        z = npz(p["bc_rows"])
        u = units(z["key"], z["score"].astype(np.float64), z["fold"])
        pool = pool_of(u, ["honest_b", "honest_c"])
        zs, mom = zscore(u, pool)
        mu, sd = np.array([mom[f]["mean"] for f in range(5)]), np.array([mom[f]["sd"] for f in range(5)])
        sh_keys, sh_sc, sh_fo, gc = [], [], [], {}
        for bt in p["simh"]:  # GC1 (amended before the run): the per-row files carry no fold and no command run-score reference exists
            zh = npz(bt["rows"])
            man = json.loads((Path("/" + bt["manifest"]["volume"]) / bt["manifest"]["path"]).read_bytes())
            gate(hashlib.sha256((Path("/" + bt["manifest"]["volume"]) / bt["manifest"]["path"]).read_bytes()).hexdigest() == bt["manifest"]["sha256"],
                 f"{p['label']} {bt['batch']}: manifest sha256")
            kk, ss = zh["key"].astype(str), zh["score"].astype(np.float64)
            gate(np.isfinite(ss).all() and len(kk) == len(ss) == len(zh["unit"]), f"{p['label']} {bt['batch']}: simh rows non-finite or misaligned")
            runs_ = sorted(set(kk.tolist()))
            gate(len(kk) == bt["n_rows"] and len(runs_) == bt["n_runs"], f"GC1 {p['label']} {bt['batch']}: {len(kk)} rows / {len(runs_)} runs, record says {bt['n_rows']} / {bt['n_runs']}")
            rf = man["run_folds"]
            bad = [k_ for k_ in runs_ if k_ not in rf or k_.split("|", 1)[1] not in fot or int(rf[k_]) != fot[k_.split("|", 1)[1]]]
            gate(not bad, f"GC1 {p['label']} {bt['batch']}: runs without a manifest fold, off T196, or whose fold differs from r36: {bad[:2]}")
            gate(not (set(runs_) & set(sh_keys)) and not (set(runs_) & set(u["by_run"])), f"GC1 {p['label']}: simh keys overlap")
            pairs = list(zip(kk.tolist(), zh["unit"].tolist()))
            gate(len(set(pairs)) == len(pairs), f"GC1 {p['label']} {bt['batch']}: a (key, unit) pair repeats")
            gc[bt["batch"]] = {"runs": len(runs_), "rows": int(len(kk)), "folds_checked_against_r36": len(runs_), "world": sorted({k_.split("|", 1)[0] for k_ in runs_})}
            sh_keys += list(kk)
            sh_sc.append(ss)
            sh_fo.append(np.array([int(rf[k_]) for k_ in kk], dtype=np.int64))
        sh_keys = np.array(sh_keys)
        sh_z = (np.concatenate(sh_sc) - mu[np.concatenate(sh_fo)]) / sd[np.concatenate(sh_fo)]
        sh_runs = {}
        for i, k_ in enumerate(sh_keys):
            sh_runs.setdefault(k_, []).append(i)
        sh_max = {k_: float(sh_z[ix].max()) for k_, ix in sh_runs.items()}
        pts_c = []
        pu = np.concatenate([zs[u["by_run"][k]] for k in pool])
        d_ = pu[np.argsort(-pu, kind="mergesort")]
        rmax = {k: float(zs[ix].max()) for k, ix in u["by_run"].items()}
        wr = {w: [k for k in u["by_run"] if k.startswith(w + "|") and k.split("|", 1)[1] in ui] for w in HW}
        wu = {w: np.concatenate([zs[u["by_run"][k]] for k in wr[w]]) for w in HW}
        for f in GRID:
            k = int(np.floor(f * len(d_) + 1e-9))
            T = float(d_[min(k, len(d_) - 1)])
            pts_c.append({"fpr_target": f, "k": k, "threshold": T, "pool_units_above": int((pu > T).sum()), "shaded_lt10_above": int((pu > T).sum()) < 10,
                          "sim_honest": {"unit_fpr": float((sh_z > T).mean()), "run_false_alarm_any_unit": float(np.mean([v > T for v in sh_max.values()])),
                                         "units": int(len(sh_z)), "runs": len(sh_max)},
                          "real_honest": {w: {"unit_fpr": float((wu[w] > T).mean()), "run_false_alarm_any_unit": float(np.mean([rmax[k_] > T for k_ in wr[w]])),
                                              "runs": len(wr[w])} for w in HW} | {"pool": {"unit_fpr": float((pu > T).mean()),
                                                                                           "run_false_alarm_any_unit": float(np.mean([rmax[k_] > T for k_ in pool]))}}})
        out["C"][p["label"]] = {"readout": "command A (content-only call end, L63)", "gates": {"GC1": gc}, "pool": "honest_b+honest_c", "z_moments_unit": mom,
                                "note": "sim-honest per-row reply (avg_top2) scores do not exist; command A is the only sim-honest per-row readout pinned here",
                                "points": pts_c}
    blob = json.dumps(out, default=float).encode()
    fn = f"{name}.json"
    (od / fn).open("xb").write(blob)
    vols["solavg"].commit()
    return {"file": fn, "bytes": blob, "sha256": hashlib.sha256(blob).hexdigest()}


@app.local_entrypoint()
def main(config: str = "config/inputs_panel3.json"):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("uncommitted tracked changes")
    cfg = json.loads((HERE / config).read_text())
    blobs = {}
    for k, spec in cfg["local"].items():
        b = git("show", spec["git"]) if "git" in spec else (HERE / spec["path"]).read_bytes() if not spec["path"].startswith("/") else Path(spec["path"]).read_bytes()
        if hashlib.sha256(b).hexdigest() != spec["sha256"]:
            raise SystemExit(f"G1: {k} sha256 mismatch")
        blobs[k] = b
    head = git("rev-parse", "HEAD").decode().strip()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    name = f"panel3_{stamp}"
    meta = {"git": head, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "config_sha256": hashlib.sha256((HERE / config).read_bytes()).hexdigest(),
            "rule": "experiments/2026-10-14_fpr_sweep_unit/RULE.md, Add-on (0d 02:1x, panel 3)", "stamp_pt": stamp, "inputs": {k: hashlib.sha256(v).hexdigest() for k, v in blobs.items()}}
    out = compute.remote(name, cfg, blobs, meta)
    R = HERE / "results"
    (R / out["file"]).open("xb").write(out["bytes"])
    print(json.dumps({"record": f"experiments/2026-10-14_fpr_sweep_unit/results/{out['file']}", "sha256": out["sha256"]}, indent=1))
