"""Sizing line (2e via 0d, 10-15): at the strict unit thresholds (target unit FPR 1e-3, 1.33e-3, 1.78e-3; honest B u C pool) for the
five main probes, how many tasks would bring the 95% interval half-width of sim - run1 to +/-5 and +/-1 points?

Method (task sampling only; a launch effect like run2 - run1 does not shrink with more tasks from one launch):
  - per cell, the paired per-task difference d_t = hit_sim - hit_run1 (one draw resamples the TASK, so sim and run1 move together;
    nothing is added in quadrature), recomputed from the pinned per-row files and gated against part (b)'s record (moments,
    threshold, catch points exact);
  - h0 = the record's 95% half-width (A24's bootstrap, 183 tasks);
  - scaling: A24's estimator is a ratio (draws resample the 195 universe tasks within environments; the catch is the weighted mean
    over the cohort tasks drawn). Growing every environment's universe in proportion (cohort share unchanged), its delta-method
    variance is O(1/n), so the half-width scales as 1/sqrt(n): n' = n (h0 / h)^2, asymptotically;
  - simulation check of that scaling with the actual estimator: synthetic universes of size u' (universe tasks, each with its cohort
    flag and d_t, resampled with replacement within environments; integer sizes by largest remainder, summing exactly to u'),
    each re-bootstrapped as A24 does (2,000 draws within environments, ratio over the cohort tasks drawn) -> median half-width
    over 200 synthetic universes; run at u' = 195 (sanity against h0) and at each scaled u'.
Modal CPU; the laptop reads pinned inputs and launches.

  PYTHONPATH=src <modal 1.6.1> run --detach experiments/2026-10-14_fpr_sweep_unit/analyze_sizing.py
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
VOLS = {"solavg": "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13", "p2": "probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13"}
REC_B = ("fprsweep_unit_2026-10-15_003643.json", "d979413913b988ff6f29fbc0a5910aa01d3bb048387bc41b1a848e039ebba7d2")
STRICT = [10 ** (-4 + 0.125 * i) for i in (8, 9, 10)]  # 1e-3, 1.33e-3, 1.78e-3 (the record's grid expression)
TARGETS = (5.0, 1.0)  # half-width targets, points
R_SYN, B_BOOT = 200, 2000
app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")
sol_vol = modal.Volume.from_name(VOLS["solavg"], version=2)
p2_vol = modal.Volume.from_name(VOLS["p2"], version=2)


@app.function(image=image, cpu=(8, 8), memory=(16384, 16384), timeout=1800, retries=0,
              volumes={"/solavg": sol_vol.read_only(), "/p2": p2_vol.read_only()})
def run(cfg: dict, blobs: dict, meta: dict) -> dict:  # noqa: C901
    import numpy as np

    recb = json.loads(blobs["rec_b"])
    env_of = {c["key"]: c["env"] for c in json.loads(blobs["t196"])["cells"]}
    fails = []
    out = {"label": "PRELIMINARY until 09 and Codex review; from raw data, by script", "meta": meta,
           "scope": "task sampling only: a launch effect (as in run2 - run1) does not shrink with more tasks from one launch; intervals conditional on the threshold",
           "method": __doc__, "cells": []}
    for pi, p in enumerate([p for p in cfg["probes"] if p["role"] == "main"]):
        pid = p["probe_id"]
        b = (Path("/" + p["rows"]["volume"]) / p["rows"]["path"]).read_bytes()
        if hashlib.sha256(b).hexdigest() != p["rows"]["sha256"]:
            raise SystemExit(f"{pid}: sha256")
        z = np.load(io.BytesIO(b), allow_pickle=False)
        keys, sc = z["key"].astype(str), z["score"].astype(np.float64)
        fo = z["fold"].astype(np.int64) if p["folds"] == 5 else None
        by_run = {}
        for i, k in enumerate(keys):
            by_run.setdefault(k, []).append(i)
        rp = recb["probes"][pid]
        pool_runs, coh = rp["pool_runs"], rp["cohort"]
        if fo is not None:  # part (b)'s per-fold unit z, gated against the record's moments
            sel = np.zeros(len(sc), bool)
            for k in pool_runs:
                sel[by_run[k]] = True
            mu, sd = np.zeros(5), np.zeros(5)
            for f in range(5):
                v = sc[sel & (fo == f)]
                mu[f], sd[f] = float(np.mean(v)), float(np.std(v, ddof=0))
                m_ = rp["z_moments_unit"][str(f)]
                if abs(mu[f] - m_["mean"]) > 1e-12 or abs(sd[f] - m_["sd"]) > 1e-12:
                    fails.append(f"{pid}: fold {f} moments differ from the record")
            zs = (sc - mu[fo]) / sd[fo]
        else:
            zs = sc
        pool = np.concatenate([zs[by_run[k]] for k in pool_runs])
        d_ = pool[np.argsort(-pool, kind="mergesort")]
        rmax = {k: float(zs[ix].max()) for k, ix in by_run.items()}
        for f in STRICT:
            q = next(q for q in rp["points"] if abs(q["fpr_target"] - f) <= 1e-12 * f)
            k = int(np.floor(f * len(d_) + 1e-9))
            T = float(d_[min(k, len(d_) - 1)])
            h = {w: np.array([float(rmax[f"{w}|{t}"] > T) for t in coh]) for w in ("sim", "run1")}
            if T != q["threshold"] or k != q["k"] or any(abs(h[w].mean() - q["tpr"][w]["point"]) > 1e-12 for w in h) or q["shaded_lt10_above"]:
                fails.append(f"{pid} {f:.2e}: threshold, k or catch differ from the record (or the point is shaded)")
            dd = h["sim"] - h["run1"]
            iv95 = q["diff"]["sim_minus_run1"]["ci95"]
            h0 = 100 * (iv95[1] - iv95[0]) / 2
            n0 = len(coh)

            # A24's estimator: draws resample the UNIVERSE tasks (195) within environments; the catch is the weighted mean over the
            # cohort tasks drawn (a ratio: non-cohort draws add weight to neither numerator nor denominator)
            uni = sorted(json.loads(blobs["v6"])["result"]["universe_and_split"]["universe"])
            dd_of = dict(zip(coh, dd.tolist()))
            ucat = {s: np.array([(0 if t not in dd_of else int(dd_of[t]) + 2) for t in uni if env_of[t] == s], dtype=np.int64)  # 0 = not in cohort; 1, 2, 3 = d -1, 0, +1
                    for s in sorted({env_of[t] for t in uni})}
            levels = sorted(ucat)
            u0 = len(uni)
            rng = np.random.default_rng([20261015, pi, int(round(f * 1e6))])

            def alloc(total):  # integer universe sizes per environment, proportional, summing exactly to total (largest remainder)
                raw = {s: len(ucat[s]) * total / u0 for s in levels}
                n_ = {s: int(np.floor(raw[s])) for s in levels}
                for s in sorted(levels, key=lambda s_: raw[s_] - n_[s_], reverse=True)[: total - sum(n_.values())]:
                    n_[s] += 1
                return n_

            def sim_hw(uprime):  # median 95% half-width (points) at universe size u' over R_SYN synthetic universes
                ns = alloc(uprime)
                hws, elig = np.empty(R_SYN), np.empty(R_SYN)
                vals_ = np.array([0.0, -1.0, 0.0, 1.0])  # categories: not in cohort, -1, 0, +1
                for r in range(R_SYN):
                    num, den, ne = np.zeros(B_BOOT), np.zeros(B_BOOT), 0
                    for s in levels:
                        syn = rng.choice(ucat[s], ns[s], replace=True)
                        cnt = np.bincount(syn, minlength=4).astype(float)
                        ne += int(cnt[1:].sum())
                        draw = rng.multinomial(ns[s], cnt / ns[s], size=B_BOOT)
                        num += draw @ vals_
                        den += draw[:, 1:].sum(1)
                    m = num / den
                    hws[r] = 100 * (np.percentile(m, 97.5) - np.percentile(m, 2.5)) / 2
                    elig[r] = ne
                return float(np.median(hws)), float(np.median(elig))

            cell = {"probe": p["label"], "fpr_target": f, "k": k, "n_tasks": n0, "discordant": {"sim_only": int((dd == 1).sum()), "run1_only": int((dd == -1).sum())},
                    "point_pts": 100 * float(dd.mean()), "h0_record_pts": h0}
            if h0 == 0:
                cell["note"] = "zero-width interval: no discordant tasks; sizing undefined"
                out["cells"].append(cell)
                continue
            num, den = np.zeros(10000), np.zeros(10000)  # direct bootstrap of the original universe (no outer resampling): compare with h0
            for s in levels:
                cnt = np.bincount(ucat[s], minlength=4).astype(float)
                draw = rng.multinomial(len(ucat[s]), cnt / len(ucat[s]), size=10000)
                num += draw @ np.array([0.0, -1.0, 0.0, 1.0])
                den += draw[:, 1:].sum(1)
            m0 = num / den
            cell["h_direct_bootstrap_pts"] = float(100 * (np.percentile(m0, 97.5) - np.percentile(m0, 2.5)) / 2)
            cell["h_sim_at_universe_195_pts"] = sim_hw(u0)[0]
            for tg in TARGETS:
                c_ = (h0 / tg) ** 2  # the 1/sqrt(n) scaling factor (proportional growth of the universe and its cohort share)
                uprime = int(np.ceil(u0 * c_))
                hw, ne = sim_hw(uprime)
                cell[f"n_for_pm{tg:g}"] = {"scaled_cohort_tasks": int(np.ceil(n0 * c_)), "scaled_universe_tasks": uprime, "simulated_half_width_pts": hw,
                                           "simulated_median_cohort_tasks": ne}
            out["cells"].append(cell)
    out["gate_failures"] = fails
    if fails:
        raise SystemExit("gates failed: " + " | ".join(fails))
    for tg in TARGETS:
        ns = [c[f"n_for_pm{tg:g}"]["scaled_cohort_tasks"] for c in out["cells"] if f"n_for_pm{tg:g}" in c]
        out[f"summary_pm{tg:g}"] = {"cells": len(ns), "median_n": float(np.median(ns)), "max_n": int(max(ns)), "min_n": int(min(ns))}
    return out


@app.local_entrypoint()
def main(config: str = "config/inputs_b.json"):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("uncommitted tracked changes")
    cfg = json.loads((HERE / config).read_text())
    blobs = {}
    b = (HERE / "results" / REC_B[0]).read_bytes()
    if hashlib.sha256(b).hexdigest() != REC_B[1]:
        raise SystemExit("record sha256")
    blobs["rec_b"] = b
    for k in ("t196", "v6"):
        t = git("show", cfg["local"][k]["git"])
        if hashlib.sha256(t).hexdigest() != cfg["local"][k]["sha256"]:
            raise SystemExit(f"{k} sha256")
        blobs[k] = t
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    meta = {"git": git("rev-parse", "HEAD").decode().strip(), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "stamp_pt": stamp,
            "record": {"file": REC_B[0], "sha256": REC_B[1]}}
    out = run.remote(cfg, blobs, meta)
    dest = HERE / "results" / f"sizing_{stamp}.json"
    dest.open("x").write(json.dumps(out, indent=1, default=float))
    print(json.dumps({"file": str(dest.relative_to(HERE.parents[1])), "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()}
                     | {k: v for k, v in out.items() if k.startswith("summary")}, indent=1))
