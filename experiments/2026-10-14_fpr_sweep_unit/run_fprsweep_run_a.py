"""Part (a) of RULE.md in this folder: the run-level FPR sweep (A24's fixed_cal_std arithmetic) with the real-honest relaunch's
958 usable runs added to the honest B u C pool. Sol (a) and c_asis only (4f's external relaunch scores are pending). Stored
scores only, Modal CPU; the laptop only reads pinned inputs, checks sha256 and launches. Gate G7: with the pool restricted to
B u C the code reproduces A24's sweep rows exactly. --mode render draws the F1/F2 template from the saved record.

  PYTHONPATH=src <modal 1.6.1> run --detach experiments/2026-10-14_fpr_sweep_unit/run_fprsweep_run_a.py --mode compute
  PYTHONPATH=src <modal 1.6.1> run --detach experiments/2026-10-14_fpr_sweep_unit/run_fprsweep_run_a.py --mode render --record <name> --sha <sha> --titles <json>
"""

import gzip
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
VOLS = {"solavg": "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13", "convaudit": "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14"}
D, SEED = 10000, [20261014, 1]
A3, HW = ("run1", "run2", "sim"), ("honest_a", "honest_b", "honest_c")
app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "scipy==1.16.2", "matplotlib==3.10.6")
sol_vol = modal.Volume.from_name(VOLS["solavg"], version=2)
conv_vol = modal.Volume.from_name(VOLS["convaudit"])


def tango(b, c, n, level):  # verbatim from A24's run_priority1.py @ 9866abcb7
    from scipy.stats import norm

    z = norm.ppf(0.5 + level / 2)

    def stat(dl):
        A_, B_, C_ = 2 * n, -b - c + (2 * n - b + c) * dl, -c * dl * (1 - dl)
        p21 = (-B_ + max(B_ * B_ - 4 * A_ * C_, 0.0) ** 0.5) / (2 * A_)
        v, num = n * (2 * p21 + dl * (1 - dl)), b - c - n * dl
        return num / v**0.5 if v > 0 else (0.0 if num == 0 else (1e300 if num > 0 else -1e300))

    def root(lo, hi, target):
        for _ in range(200):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if stat(mid) > target else (lo, mid)
        return (lo + hi) / 2

    est, eps = (b - c) / n, 1e-12
    lo = -1.0 if stat(-1 + eps) <= z else root(-1 + eps, est, z)
    hi = 1.0 if stat(1 - eps) >= -z else root(est, 1 - eps, -z)
    return [lo, hi]


def jpath(d, kp):
    for k in kp:
        d = d[k]
    return d


def fdiff(x, y):
    import math

    if x is None or y is None:
        return 0.0 if x is None and y is None else float("inf")
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


@app.function(image=image, cpu=(8, 8), memory=(32768, 32768), timeout=2400, retries=0,
              volumes={"/sol": sol_vol, "/conv": conv_vol.read_only()})
def compute(name: str, cfg: dict, blobs: dict, meta: dict) -> dict:  # noqa: C901
    import numpy as np

    root = {"solavg": "/sol", "convaudit": "/conv"}
    od = Path("/sol") / "xfer" / "fprsweep_unit" / name
    od.mkdir(parents=True, exist_ok=False)
    fails, diag = [], {}

    def gate(ok, msg):  # fail fast: diagnostics only (no sweep numbers), nonzero exit
        if not ok:
            fails.append(msg)
            (od / f"{name}_GATE_FAILED.json").open("x").write(json.dumps({"label": "GATE FAILED: stop and ask 2e", "meta": meta, "failures": fails,
                                                                           "gates_so_far": diag}, default=float, indent=1))
            sol_vol.commit()
            raise SystemExit(f"gate failed: {msg}")

    def vread(spec):
        b = (Path(root[spec["volume"]]) / spec["path"]).read_bytes()
        gate(hashlib.sha256(b).hexdigest() == spec["sha256"], f"G1: {spec['path']} sha256")
        return b

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
    groups = [np.flatnonzero(st == k) for k in sorted(set(st.tolist()))]
    for b_ in range(1, D + 1):
        for idx in groups:
            np.add.at(W[b_], rng.choice(idx, len(idx), replace=True), 1)

    def thr(s, w, a):
        o = np.argsort(-s, kind="mergesort")
        cw = np.cumsum(w[:, o], 1)
        tot = cw[:, -1]
        out = s[o][np.minimum((cw <= np.floor(a * tot + 1e-9)[:, None]).sum(1), len(o) - 1)].astype(float)
        out[tot <= 0] = np.nan
        return out

    def summ(v):
        f = v[1:][np.isfinite(v[1:])]
        o = {"point": float(v[0]), "finite_draws": int(len(f)), "draws": D}
        if len(f) < 0.95 * D:
            return o | {"ci95": None, "withheld": "fewer than 95% complete draws"}
        return o | {"ci95": [float(np.percentile(f, 2.5)), float(np.percentile(f, 97.5))], "degenerate": bool(np.ptp(f) == 0)}

    wmean = lambda Wm, h: (Wm * h).sum(1) / Wm.sum(1)  # noqa: E731

    # ---- the usable relaunch set (RULE part (a) item 1): d9's counted manifest minus the 22 counted unusable runs ----
    cm = json.loads(blobs["counted_manifest"])["files"]
    gate(len(cm) == 980 and len({(e["launch"], e["task"]) for e in cm}) == 980, "counted manifest: not 980 distinct counted runs")
    counted = {(e["launch"], e["task"]): e for e in cm}
    un = json.loads(blobs["unusable"])["classes"]
    excl = {}
    for cl in cfg["unusable_classes_counted"]:
        for x in un[cl]["runs"]:
            if (x["launch"], x["key"]) in counted:
                excl[(x["launch"], x["key"])] = cl
    gate(len(excl) == 22, f"usable: {len(excl)} counted unusable runs, not 22")

    def rkey(e):
        base, rep = e["task"].rsplit("_r", 1)
        gate(e["world"] == f"honest_r{rep}" and base in fot, f"usable: key mapping for {e['task']}")
        return f"{e['world']}|{base}"

    usable = {rkey(e): e for k_, e in counted.items() if k_ not in excl}
    gate(len(usable) == 958, f"usable: {len(usable)} keys, not 958")
    launches = sorted({e["launch"] for e in usable.values()})
    gate(set(launches) <= {"2026-10-14_135334_honest_rh980", "2026-10-14_160335_honest_rh_relaunch"}, f"usable: unexpected launch directories {launches}")
    excluded = sorted({rkey(counted[k_]): cl for k_, cl in excl.items()}.items())
    cue = {}
    for r in json.loads(blobs["cue_d"])["rows"]:  # cue_D rows are keyed by the physical eval path plus its sample suffix ("#e0")
        gate(r["key"] not in cue, f"cue_D: duplicate row {r['key'][-40:]}")
        cue[r["key"]] = r.get(cfg["local"]["cue_d"]["flag_field"])  # None = unassessed
    flag = {k: cue.get(e["path"] + "#e0", "missing") for k, e in usable.items()}
    gate(all(v != "missing" for v in flag.values()), "cue_D: a usable run has no cue_D row")
    asis = json.loads(blobs["asis"])
    a24 = {r["probe_id"]: r for r in (json.loads(ln) for ln in gzip.decompress(blobs["a24_jsonl_gz"]).decode().splitlines()) if r["row_type"] == "sweep"}
    man = json.loads(vread(cfg["relaunch_manifest"]))
    rec = {"label": "PRELIMINARY until 09 and Codex review; from raw data, by script", "meta": meta, "part": "a", "draws": D, "seed": SEED,
           "relaunch": {"usable": len(usable), "excluded_with_cause": excluded, "label": "a new honest batch (honest_r): no registered analysis reads it for calibration yet",
                        "cue_d_flagged": sum(1 for v in flag.values() if v is True), "cue_d_unassessed": sorted(k for k, v in flag.items() if v is None),
                        "cue_d_bc": cfg["cue_d_bc"], "launch_dirs": launches,
                        "physical_disjointness": "relaunch runs come only from the two relaunch launch directories (10-14), distinct from B's and C's launches"}, "externals": cfg["externals"], "probes": {}}
    for p in cfg["probes"]:
        pid, lab = p["probe_id"], p["label"]
        g = {}
        diag[pid] = g
        # B u C, A and attack-arm run scores exactly as A24 read them
        if p["bc_runs"] == "npz_mean":
            z = np.load(io.BytesIO(vread(p["bc_rows"])), allow_pickle=False)
            agg = {}
            for k, v in zip(z["key"].astype(str), z["score"].astype(np.float64)):
                agg.setdefault(k, []).append(v)
            rs = {k: float(np.mean(v)) for k, v in agg.items()}
        else:
            rs = {k: float(v) for k, v in jpath(asis, p["bc_runs"]["keypath"]).items()}
        # relaunch rows and run scores (2e's scorer; its record reproduces A24's B scores exactly)
        zr = np.load(io.BytesIO(vread(p["relaunch"]["rows"])), allow_pickle=False)
        rr = jpath(json.loads(vread(p["relaunch"]["runs"])), p["relaunch"]["runs"]["keypath"])
        keys, sc, fo = zr["key"].astype(str), zr["score"].astype(np.float64), zr["fold"].astype(np.int64)
        gate(len(keys) == len(sc) == len(fo) and np.isfinite(sc).all(), f"{lab}: relaunch rows misaligned or non-finite")
        agg = {}
        for k, v, f in zip(keys, sc, fo):
            agg.setdefault(k, []).append((v, f))
        rmean = {k: float(np.mean([v for v, _ in x])) for k, x in agg.items()}
        dmax = max(fdiff(rmean[k], rr[k]) for k in rmean) if set(rmean) == set(rr) else float("inf")
        g["relaunch_aggregation"] = {"runs": len(rmean), "max_abs_diff_vs_json": dmax}
        gate(dmax <= 1e-9, f"{lab}: relaunch row means differ from the run-score JSON ({dmax})")
        gate(set(rmean) == set(usable), f"{lab}: relaunch keys != the 958 usable keys (only scored {len(set(rmean) - set(usable))}, only usable {len(set(usable) - set(rmean))})")
        badf = [k for k, x in agg.items() if {f for _, f in x} != {fot[k.split("|", 1)[1]]} or man["run_folds"].get(k) != fot[k.split("|", 1)[1]]]
        gate(not badf, f"{lab}: relaunch folds differ from r36 or the manifest: {badf[:3]}")
        gate(not (set(rmean) & set(rs)), f"{lab}: relaunch keys overlap B/C/attack keys")
        # A24's run-level structure
        A = {w: {} for w in A3 + HW}
        for k_, v in rs.items():
            w, t = k_.split("|", 1)
            gate(np.isfinite(v), f"{lab}: non-finite run score {k_}")
            if t in ui and w in A:
                A[w][t] = v
        coh = sorted(t for t in canon if all(att(w, t) for w in A3) and all(t in A[w] for w in A3))
        cf = np.array([fot[t] for t in coh])
        Wc = W[:, [ui[t] for t in coh]]
        n = len(coh)
        hon = {h: sorted(A[h]) for h in HW}
        raw = {w: np.array([A[w][t] for t in coh]) for w in A3} | {h: np.array([A[h][t] for t in hon[h]]) for h in HW}
        hfold = {h: np.array([fot[t] for t in hon[h]]) for h in HW}
        rkeys = sorted(rmean)
        raw["honest_r"], hfold["honest_r"] = np.array([rmean[k] for k in rkeys]), np.array([fot[k.split("|", 1)[1]] for k in rkeys])

        def sweep(pool_parts, keep_r=None, gate_a24=False):
            pr = raw["honest_r"] if keep_r is None else raw["honest_r"][keep_r]
            pfr = hfold["honest_r"] if keep_r is None else hfold["honest_r"][keep_r]
            parts = [(raw[h], hfold[h]) for h in pool_parts if h != "honest_r"] + ([(pr, pfr)] if "honest_r" in pool_parts else [])
            cal = np.concatenate([x for x, _ in parts])
            calf = np.concatenate([f for _, f in parts])
            mu, sd = np.zeros(5), np.ones(5)
            for f in range(5):
                v = cal[calf == f]
                mu[f], sd[f] = v.mean(), v.std()
                gate(len(v) >= 2 and sd[f] > 0, f"{lab}: fold {f} empty or constant")
            z_ = lambda x, f: (x - mu[f]) / sd[f]  # noqa: E731
            pv = z_(cal, calf)
            std = {w: z_(raw[w], cf) for w in A3} | {h: z_(raw[h], hfold[h]) for h in HW} | {"honest_r": z_(pr, pfr)}
            pts = []
            for fpr in np.logspace(np.log10(1 / len(pv)), -1, 25):
                T0 = thr(pv, np.ones((1, len(pv))), fpr)[0]
                h = {w: (std[w] > T0).astype(float) for w in A3}
                h["real_mean"] = (h["run1"] + h["run2"]) / 2
                c = {w: wmean(Wc, h[w]) for w in h}
                b_, c_ = int(((h["run2"] == 1) & (h["run1"] == 0)).sum()), int(((h["run2"] == 0) & (h["run1"] == 1)).sum())
                above = int((pv > T0).sum())
                pts.append({"fpr_target": float(fpr), "k": int(np.floor(fpr * len(pv) + 1e-9)), "n_pool_runs": int(len(pv)), "threshold": float(T0),
                            "pool_runs_above": above, "shaded_lt10_above": above < 10,
                            "realized_fa": {"pool": float(np.mean(pv > T0))} | {h_: float(np.mean(std[h_] > T0)) for h_ in HW + ("honest_r",)},
                            "tpr": {w: summ(v) for w, v in c.items()},
                            "diff": {"sim_minus_real_mean": summ(c["sim"] - c["real_mean"]), "sim_minus_run1": summ(c["sim"] - c["run1"]),
                                     "sim_minus_run2": summ(c["sim"] - c["run2"]), "run2_minus_run1": summ(c["run2"] - c["run1"])},
                            "control_tango": {"point": float(h["run2"].mean() - h["run1"].mean()), "ci95": tango(b_, c_, n, 0.95), "discordant": [b_, c_]},
                            "saturated": {w: c[w][0] in (0.0, 1.0) for w in c}})
            if gate_a24:  # G7: B u C only must reproduce A24's sweep rows exactly
                worst, nv = 0.0, 0
                for q, a_ in zip(pts, a24[pid]["points"]):
                    mine = {"fpr_target": q["fpr_target"], "k": q["k"], "fa": q["realized_fa"]["pool"], "faA": q["realized_fa"]["honest_a"],
                            "tpr": {w: [q["tpr"][w]["point"]] + (q["tpr"][w]["ci95"] or [None, None]) for w in q["tpr"]},
                            "delta": [q["diff"]["sim_minus_real_mean"]["point"]] + (q["diff"]["sim_minus_real_mean"]["ci95"] or [None, None]),
                            "control": [q["control_tango"]["point"]] + q["control_tango"]["ci95"] + q["control_tango"]["discordant"]}
                    theirs = {"fpr_target": a_["fpr_target"], "k": a_["k"], "fa": a_["realized_fa_cal_in_sample"], "faA": a_["realized_fa_A_out_of_sample"],
                              "tpr": {w: [a_["tpr"][w]["point"]] + (a_["tpr"][w]["ci95"] or [None, None]) for w in a_["tpr"]},
                              "delta": [a_["delta"]["point"]] + (a_["delta"]["ci95"] or [None, None]),
                              "control": [a_["control_run2_minus_run1"]["point"]] + a_["control_run2_minus_run1"]["ci95"] + list(a_["control_run2_minus_run1"]["discordant"])}
                    m_, t_ = dict(flat(mine)), dict(flat(theirs))
                    if set(m_) != set(t_):
                        worst = float("inf")
                    for k2 in t_:
                        worst, nv = max(worst, fdiff(m_.get(k2), t_[k2]) if k2 in m_ else float("inf")), nv + 1
                g["G7_bc_vs_A24"] = {"points": len(pts), "values": nv, "max_abs_diff": worst}
                gate(len(pts) == len(a24[pid]["points"]) == 25 and worst <= 1e-12, f"G7 {lab}: B u C sweep differs from A24 by {worst}")
            return {"pool": "+".join(pool_parts), "n_pool_runs": int(len(pv)), "moments": {"mean": mu.tolist(), "sd": sd.tolist()}, "points": pts}

        gate(a24[pid]["calibration"] == "honest_b+honest_c" and n == a24[pid]["n_tasks"], f"{lab}: A24 pool or cohort differs")
        sweep(["honest_b", "honest_c"], gate_a24=True)
        prim = sweep(["honest_b", "honest_c", "honest_r"])
        keep = np.array([flag[k] is not True for k in rkeys])
        sens = sweep(["honest_b", "honest_c", "honest_r"], keep_r=keep)
        sens["label"] = f"cue_D-flagged relaunch runs dropped ({int((~keep).sum())}); {cfg['cue_d_bc']}"
        rec["probes"][pid] = {"label": lab, "role": p["role"], "cohort": coh, "gates": g, "primary": prim, "sensitivity_cue_d_excluded": sens,
                              "a24_reference_bc": a24[pid]}
    blob = json.dumps(rec, default=float).encode()
    fn = f"{name}.json"
    (od / fn).open("xb").write(blob)
    sol_vol.commit()
    return {"file": fn, "bytes": blob, "sha256": hashlib.sha256(blob).hexdigest()}


@app.function(image=image, cpu=(2, 2), memory=(8192, 8192), timeout=900, retries=0, volumes={"/sol": sol_vol})
def render(name: str, sha: str, titles: dict, tag: str = "") -> dict:
    import textwrap

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    b = (Path("/sol") / "xfer" / "fprsweep_unit" / name / f"{name}.json").read_bytes()
    if hashlib.sha256(b).hexdigest() != sha:
        raise SystemExit("record sha256 differs")
    rec = json.loads(b)
    plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6"})
    BLUE, ORANGE, GREY, BAND5, HATCH = "#3182bd", "#e6550d", "#6f6f6f", "#f5f5f5", "#bdbdbd"
    pids = list(rec["probes"])
    out, view = {}, {"label": rec["label"], "record": name, "record_sha256": sha, "panels": {}, "titles": titles}
    foot = ("Threshold on honest B \u222a C + 958 relaunch runs (per-fold z); run score = mean over reply units; 183 attempted tasks; "
            "pointwise 95% task bootstrap, conditional on the threshold")
    for kind, title in (("gap", titles["F1a"]), ("tpr", titles["F2a"])):
        for layout in ("wide", "narrow"):
            wide = layout == "wide"
            fig, axes = plt.subplots(1, 2, figsize=(10, 4.9)) if wide else plt.subplots(2, 1, figsize=(4.6, 9.2))
            for ax, pid in zip(axes.flat, pids):
                qs = rec["probes"][pid]["primary"]["points"]
                x = [q["fpr_target"] for q in qs]
                specs = ((("run2_minus_run1", "diff", GREY, "--", 0.3, "run 2 − run 1 (rerun + launch)"),
                          ("sim_minus_real_mean", "diff", ORANGE, "-", 0.25, "sim − real (mean of run 1, run 2)")) if kind == "gap" else
                         (("real_mean", "tpr", BLUE, "-", 0.2, "real (mean of run 1, run 2)"), ("sim", "tpr", ORANGE, "-", 0.2, "simulated")))
                if kind == "gap":
                    ax.axhspan(-5, 5, facecolor=BAND5, edgecolor="#8c8c8c", ls=":", lw=0.9, zorder=0, label="±5 points")
                    ax.axhline(0, color="black", lw=0.8)
                vp = {}
                for key, grp, col, ls, al, lb in specs:
                    y = np.array([100 * q[grp][key]["point"] for q in qs])
                    lo = np.array([100 * q[grp][key]["ci95"][0] if q[grp][key]["ci95"] else np.nan for q in qs])
                    hi = np.array([100 * q[grp][key]["ci95"][1] if q[grp][key]["ci95"] else np.nan for q in qs])
                    ax.fill_between(x, lo, hi, color=col, alpha=al, lw=0, label=lb + ", 95% band")
                    ax.plot(x, y, color=col, ls=ls, lw=2.0 if col != GREY else 1.3)
                    vp[key] = {"x": x, "point": y.tolist(), "lo": lo.tolist(), "hi": hi.tolist()}
                sh = [q["fpr_target"] for q in qs if q["shaded_lt10_above"]]
                un = [q["fpr_target"] for q in qs if not q["shaded_lt10_above"]]
                if sh:
                    right = (max(sh) * min(u for u in un if u > max(sh))) ** 0.5 if any(u > max(sh) for u in un) else 0.105
                    ax.axvspan(x[0] / 1.08, right, facecolor="none", hatch="////", edgecolor=HATCH, lw=0, zorder=0)
                if kind == "gap":
                    dg = [(q["fpr_target"], 100 * q["diff"]["sim_minus_real_mean"]["point"]) for q in qs if q["diff"]["sim_minus_real_mean"].get("degenerate")]
                else:
                    dg = [(q["fpr_target"], 100 * q["tpr"][k_]["point"]) for q in qs for k_ in ("real_mean", "sim") if q["saturated"][k_]]
                if dg:
                    ax.plot(*zip(*dg), "x", color="black", ms=5, mew=1.2, label="zero-width or saturated (not read)")
                view["panels"][f"{kind}|{pid}"] = vp | {"shaded_lt10_above": [q["shaded_lt10_above"] for q in qs], "fa_pool": [q["realized_fa"]["pool"] for q in qs]}
                ax.set_xscale("log")
                ax.set_xlim(x[0] / 1.08, 0.105)
                if kind == "tpr":
                    ax.set_ylim(0, 100)
                ax.set_title(rec["probes"][pid]["label"].replace(" (positive control)", ": positive control, not a monitor"), loc="left",
                             color="#a63603" if "positive" in rec["probes"][pid]["role"] else "black")
                ax.set_xlabel(f"target run-level FPR on honest B ∪ C + relaunch ({qs[0]['n_pool_runs']} runs)")
                ax.set_ylabel("catch difference (points)" if kind == "gap" else "catch (% of attack runs)")
            hs = {}
            for ax in axes.flat:
                for h_, l_ in zip(*ax.get_legend_handles_labels()):
                    hs.setdefault(l_, h_)
            h = list(hs.values()) + [matplotlib.patches.Patch(facecolor="white", hatch="////", edgecolor=HATCH, label="fewer than 10 honest runs above the threshold (not read)")]
            fig.legend(handles=h, loc="lower center", ncol=2 if wide else 1, frameon=False, fontsize=9 if wide else 8.3, bbox_to_anchor=(0.5, 0.04 if wide else 0.02))
            fig.suptitle(textwrap.fill(title, 105 if wide else 52), fontsize=12 if wide else 10.5, x=0.01, ha="left", y=0.995)
            fig.text(0.01, 0.005, foot if wide else "Threshold on B ∪ C + 958 relaunch runs; 95% task bootstrap",
                     fontsize=7.2 if wide else 7.0, color="#444444", ha="left", va="bottom")
            fig.tight_layout(rect=(0, 0.2 if wide else 0.13, 1, 0.9 if wide else 0.93))
            w_in = fig.get_size_inches()[0]
            for sfx, dpi in ((("", 150), ("_900px", 900 / w_in)) if wide else (("_narrow_400px", 400 / w_in), ("_narrow_800px", 800 / w_in))):
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=dpi)
                out[f"{name}{tag}_F{'1' if kind == 'gap' else '2'}a_{'gap' if kind == 'gap' else 'curves'}{sfx}.png"] = buf.getvalue()
            plt.close(fig)
    view["footer"] = foot
    out[f"{name}{tag}_viewer.json"] = json.dumps(view, default=float).encode()
    od = Path("/sol") / "xfer" / "fprsweep_unit" / name
    for fn, blob in out.items():
        (od / fn).open("xb").write(blob)
    sol_vol.commit()
    return out


@app.local_entrypoint()
def main(mode: str = "compute", config: str = "config/inputs_a.json", record: str = "", sha: str = "", titles: str = "", tag: str = ""):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("uncommitted tracked changes")
    R = HERE / "results"
    if mode == "render":
        out = render.remote(record, sha, json.loads((HERE / titles).read_text()), tag)
        for fn, blob in out.items():
            (R / fn).open("xb").write(blob)
        print(json.dumps({fn: hashlib.sha256(blob).hexdigest() for fn, blob in out.items()}, indent=1))
        return
    cfg = json.loads((HERE / config).read_text())
    blobs = {}
    for k, spec in cfg["local"].items():
        if k in ("probe_list", "z22i"):
            continue
        b = git("show", spec["git"]) if "git" in spec else Path(spec["path"]).read_bytes()
        if hashlib.sha256(b).hexdigest() != spec["sha256"]:
            raise SystemExit(f"G1: {k} sha256 mismatch")
        blobs[k] = b
    if hashlib.sha256(gzip.decompress(blobs["a24_jsonl_gz"])).hexdigest() != cfg["local"]["a24_jsonl_gz"]["decompressed_sha256"]:
        raise SystemExit("G1: A24 JSONL (decompressed) sha256 mismatch")
    rr = json.loads(blobs["relaunch_record"])
    for p in cfg["probes"]:  # the relaunch files must be the ones 2e's published record lists
        for part in ("rows", "runs"):
            if rr["outputs"]["files"][Path(p["relaunch"][part]["path"]).name] != p["relaunch"][part]["sha256"]:
                raise SystemExit(f"G1: {p['label']} relaunch {part} sha256 is not the record's")
    if rr["outputs"]["manifest_sha256"] != cfg["relaunch_manifest"]["sha256"]:
        raise SystemExit("G1: relaunch manifest sha256 is not the record's")
    head = git("rev-parse", "HEAD").decode().strip()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    name = f"fprsweep_run_a_{stamp}"
    meta = {"git": head, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "config_sha256": hashlib.sha256((HERE / config).read_bytes()).hexdigest(),
            "rule": "experiments/2026-10-14_fpr_sweep_unit/RULE.md part (a)", "stamp_pt": stamp, "inputs": {k: hashlib.sha256(v).hexdigest() for k, v in blobs.items()}}
    out = compute.remote(name, cfg, blobs, meta)
    (R / out["file"]).open("xb").write(out["bytes"])
    print(json.dumps({"record": f"experiments/2026-10-14_fpr_sweep_unit/results/{out['file']}", "sha256": out["sha256"],
                      "volume": f"{VOLS['solavg']}:/xfer/fprsweep_unit/{name}/"}, indent=1))
