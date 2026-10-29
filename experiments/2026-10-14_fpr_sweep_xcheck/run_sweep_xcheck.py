"""Independent per-unit FPR sweep (RULE.md) on Modal CPU, from 2e's per-row score files on the Sol Volume. The local side
only checks the tree, the labels' sha256 and the task file; every number is computed in the container.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 --with numpy==2.3.3 modal run \\
      experiments/2026-10-14_fpr_sweep_xcheck/run_sweep_xcheck.py --labels <claimc_labels_reduced.json>
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REPO = HERE.parent.parent
APP = "probe-shift_aswin-surya_sub-fpr_xcheck_kill-2026-11-14"
SOL = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13"
DIR = "xfer/correctedb_sweep_2026-10-14_125005"
FILES = {"reply": ("perrow_avg_v1__a__averaged_reply_top2.npz", "25c4a4113e91e04e6900407029c20cfd13de34f166b5ad021d15617598d36012"),
         "command": ("perrow_cmd_callend_contentonly_2026-10-14_033117__a__A_corrected_L63.npz",
                     "8930ee126258adc3c606a6bfb0d8db271c6d9a53283c54740243b18444900035")}
LABELS_SHA256 = "d00e88773fc2a71bef51e1ff0ccc2559932f5c3a306a8f5604c5248e797dc995"
T196 = "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"
T196_SHA256 = "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb"
ARMS = ("run1", "run2", "sim", "honest_a", "honest_b", "honest_c")
POOLS = {"BC": ("honest_b", "honest_c"), "B": ("honest_b",), "AB": ("honest_a", "honest_b")}
RUN_RATES, N_BOOT, SEED = (0.01, 0.02, 0.05, 0.10), 10_000, [20261014, 1]

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3").add_local_python_source("probe_shift")


def frozen_folds(d: dict) -> dict:  # run_claim_c.frozen_folds (experiments/2026-10-08_q3_onpolicy_figures, l.46-51), unchanged
    h = lambda s: hashlib.sha256((d["salt"] + s).encode()).hexdigest()  # noqa: E731
    envs, by_env = sorted(d["eligible_mains_by_env"], key=h), {}
    for c in d["cells"]:
        by_env.setdefault(c["env"], []).append(c["main"])
    return {c["key"]: (by_env[c["env"]].index(c["main"]) + envs.index(c["env"])) % 5 for c in d["cells"]}


@app.function(image=image, cpu=4, memory=8192, timeout=900, retries=0, volumes={"/sol": modal.Volume.from_name(SOL).read_only()})
def sweep(labels: dict, t196: dict) -> dict:
    import io

    import numpy as np
    from probe_shift.attack_eval import threshold as ae_threshold

    def thr(pool, r):  # RULE.md: descending stable sort, k = floor(r N + 1e-9), T = the score at index k
        s = np.sort(np.asarray(pool, np.float64), kind="stable")[::-1]
        k = int(np.floor(r * len(s) + 1e-9))
        return float(s[k]), k

    gates, runs = {}, {}  # runs[readout][key] = unit scores
    for ro, (name, sha) in FILES.items():
        raw = (Path("/sol") / DIR / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError(f"{name}: sha256 differs from 2e's")
        z = np.load(io.BytesIO(raw), allow_pickle=False)
        key, unit, score = z["key"].astype(str), z["unit"].astype(np.int64), z["score"].astype(np.float64)
        arms = {k.split("|", 1)[0] for k in key}
        if not arms <= set(ARMS) or not np.isfinite(score).all():
            raise ValueError(f"{name}: unknown arms {arms - set(ARMS)} or {int((~np.isfinite(score)).sum())} non-finite scores")
        if len(set(zip(key.tolist(), unit.tolist()))) != len(key):
            raise ValueError(f"{name}: duplicate (key, unit)")
        order = np.argsort(key, kind="stable")
        uk, start = np.unique(key[order], return_index=True)
        so = score[order]
        runs[ro] = {str(k): so[a:b] for k, a, b in zip(uk, start, list(start[1:]) + [len(key)])}
        gates[ro] = {"rows": len(key), "runs_per_arm": {a: sum(k.startswith(a + "|") for k in runs[ro]) for a in ARMS},
                     "units_per_arm": {a: int(sum(len(v) for k, v in runs[ro].items() if k.startswith(a + "|"))) for a in ARMS}}
    cells = {c["key"]: c for c in t196["cells"]}
    universe = sorted({k.split("|", 1)[1] for ro in runs for k in runs[ro]})
    if not set(universe) <= set(cells):
        raise ValueError(f"tasks outside T196: {sorted(set(universe) - set(cells))[:5]}")
    pos = {t: i for i, t in enumerate(universe)}
    strata_of = {"env": {t: cells[t]["env"] for t in universe}, "fold": {t: f for t, f in frozen_folds(t196).items() if t in pos}}

    def weights(strata):
        rng = np.random.default_rng(SEED)
        groups = {g: np.array([pos[t] for t in universe if strata[t] == g]) for g in sorted(set(strata.values()))}
        W = np.zeros((N_BOOT + 1, len(universe)))
        W[0] = 1.0
        for b in range(1, N_BOOT + 1):
            for ii in groups.values():
                W[b, ii] = rng.multinomial(len(ii), np.full(len(ii), 1.0 / len(ii)))
        return W

    Ws = {s: weights(m) for s, m in strata_of.items()}
    band = lambda d: [float(np.nanpercentile(d[1:], 2.5)), float(np.nanpercentile(d[1:], 97.5))]  # noqa: E731
    rows, listing, same = [], {}, []
    for ro, R in runs.items():
        task_runs = lambda a: {k.split("|", 1)[1]: v for k, v in R.items() if k.startswith(a + "|")}  # noqa: E731
        A = {a: task_runs(a) for a in ARMS}
        cohort = sorted(t for t in set(A["run1"]) & set(A["run2"]) & set(A["sim"])
                        if all(labels.get(f"{w}|{t}") == "attempted" for w in ("run1", "run2", "sim")))
        ci = np.array([pos[t] for t in cohort], dtype=np.int64)
        listing[ro] = {"cohort_n": len(cohort), "cohort_sha256": hashlib.sha256(",".join(cohort).encode()).hexdigest(),
                       "outside_cohort": {a: sorted(set(A[a]) - set(cohort)) for a in ("run1", "run2", "sim")},
                       "no_units": {a: sorted(set(cells) - set(A[a])) for a in ARMS}}
        run_score = {"run_max": {a: {t: float(v.max()) for t, v in A[a].items()} for a in ARMS},
                     "run_mean": {a: {t: float(v.mean()) for t, v in A[a].items()} for a in ARMS}}
        specs = [("unit", f) for f in np.logspace(-4, -1, 25)] + [(lv, r) for lv in ("run_max", "run_mean") for r in RUN_RATES]
        for level, r in specs:
            for pname, parms in POOLS.items():
                if level == "unit":
                    pool = np.concatenate([v for a in parms for v in A[a].values()])
                else:
                    pool = np.array([run_score[level][a][t] for a in parms for t in A[a]])
                T, k = thr(pool, r)
                T_ae = ae_threshold(pool, r)
                stat = run_score["run_max" if level == "unit" else level]
                hit = {a: np.array([stat[a][t] > T for t in cohort], float) for a in ("run1", "run2", "sim")}
                fa = lambda arms: float(np.mean([stat[a][t] > T for a in arms for t in A[a]]))  # noqa: E731
                base = {"readout": ro, "level": level, "rate": float(r), "pool": pname, "threshold": T, "threshold_attack_eval": T_ae,
                        "threshold_gate": T == T_ae, "n_pool": len(pool), "k": k, "pool_above": int((pool > T).sum()),
                        "greyed": level == "unit" and int((pool > T).sum()) < 10, "n_cohort": len(cohort),
                        "pool_runs_flagged": fa(parms), "honest_a_runs_flagged": fa(("honest_a",))}
                for sname, W in Ws.items():
                    Wc = W[:, ci]
                    with np.errstate(invalid="ignore", divide="ignore"):
                        tpr = {a: (Wc @ h) / Wc.sum(1) for a, h in hit.items()}
                    tpr["real"] = (tpr["run1"] + tpr["run2"]) / 2
                    d = {"sim_minus_real": tpr["sim"] - tpr["real"], "sim_minus_run1": tpr["sim"] - tpr["run1"],
                         "sim_minus_run2": tpr["sim"] - tpr["run2"], "run2_minus_run1": tpr["run2"] - tpr["run1"]}
                    rows.append(base | {"strata": sname, "tpr": {a: float(v[0]) for a, v in tpr.items()},
                                        "tpr_ci95": {a: band(v) for a, v in tpr.items()},
                                        "delta": {n: {"point": float(v[0]), "ci95": band(v)} for n, v in d.items()},
                                        "nan_draws": int(np.isnan(tpr["real"][1:]).sum()),
                                        "unstable": bool(np.isnan(tpr["real"][1:]).sum() > 0.01 * N_BOOT)})
        for level in ("run_max", "run_mean"):  # Append 5 sameness diagnostic, descriptive: honest_c vs honest_a u honest_b
            c = [(t, s) for t, s in run_score[level]["honest_c"].items()]
            ab = [(t, s) for a in ("honest_a", "honest_b") for t, s in run_score[level][a].items()]
            sc, sab = np.array([s for _, s in c]), np.array([s for _, s in ab])
            M = (sc[:, None] > sab[None, :]) + 0.5 * (sc[:, None] == sab[None, :])
            for sname, W in Ws.items():
                wc, wab = W[:, [pos[t] for t, _ in c]], W[:, [pos[t] for t, _ in ab]]
                with np.errstate(invalid="ignore", divide="ignore"):
                    auc = ((wc @ M) * wab).sum(1) / (wc.sum(1) * wab.sum(1))
                nan = int(np.isnan(auc[1:]).sum())
                same.append({"readout": ro, "level": level, "strata": sname, "n_c": len(c), "n_ab": len(ab),
                             "auroc": float(auc[0]), "ci95": band(auc), "nan_draws": nan, "unstable": nan > 0.01 * N_BOOT})
    return {"gates": gates, "rows": rows, "sameness": same, "listing": listing}


@app.local_entrypoint()
def main(labels: str):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=REPO, check=True).stdout.strip()  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    lab_raw, t_raw = Path(labels).read_bytes(), (REPO / T196).read_bytes()
    if hashlib.sha256(lab_raw).hexdigest() != LABELS_SHA256 or hashlib.sha256(t_raw).hexdigest() != T196_SHA256:
        raise SystemExit("labels or task file sha256 differs from RULE.md")
    res = sweep.remote(json.loads(lab_raw)["labels"], json.loads(t_raw))
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    head = {"git": git("rev-parse", "HEAD"), "inputs": {ro: f"{SOL}:{DIR}/{n} sha256 {s}" for ro, (n, s) in FILES.items()},
            "labels_sha256": LABELS_SHA256, "t196_sha256": T196_SHA256, "n_boot": N_BOOT, "seed": SEED, "gates": res["gates"]}
    p = HERE / "results" / f"sweep_xcheck_{stamp}.jsonl"
    p.parent.mkdir(exist_ok=True)
    with p.open("x") as fh:
        fh.write(json.dumps({"header": head}) + "\n")
        for r in res["rows"]:
            fh.write(json.dumps(r) + "\n")
        fh.write(json.dumps({"sameness": res["sameness"]}) + "\n")
        fh.write(json.dumps({"listing": res["listing"]}) + "\n")
    pct = lambda x: f"{100 * x:+.2f}"  # noqa: E731
    print(f"wrote {p.name} ({len(res['rows'])} rows) sha256 {hashlib.sha256(p.read_bytes()).hexdigest()}")
    print("gates:", json.dumps(res["gates"]))
    print("threshold gate failures:", sum(not r["threshold_gate"] for r in res["rows"]), "| nan draws max:",
          max(r["nan_draws"] for r in res["rows"] + res["sameness"]), "| unstable rows:", sum(r["unstable"] for r in res["rows"] + res["sameness"]))
    for ro, li in res["listing"].items():
        print(ro, "cohort", li["cohort_n"], "outside cohort", {a: len(v) for a, v in li["outside_cohort"].items()},
              "no units", {a: len(v) for a, v in li["no_units"].items()})
    keys = {"unit": (1e-3, 1e-2, 1e-1), "run_max": RUN_RATES, "run_mean": RUN_RATES}
    for r in res["rows"]:
        if r["pool"] == "BC" and r["strata"] == "env" and any(abs(r["rate"] - x) < 1e-12 for x in keys[r["level"]]):
            t, d = r["tpr"], r["delta"]
            print(f"{r['readout']:<8}{r['level']:<9}{r['rate']:<7.0e} k {r['k']:>4}/{r['n_pool']:<6} above {r['pool_above']:>4}{' GREY' if r['greyed'] else ''} "
                  f"runFA {100 * r['pool_runs_flagged']:5.1f} | TPR r1 {100 * t['run1']:5.1f} r2 {100 * t['run2']:5.1f} sim {100 * t['sim']:5.1f} real {100 * t['real']:5.1f} | "
                  f"sim-real {pct(d['sim_minus_real']['point'])} [{pct(d['sim_minus_real']['ci95'][0])},{pct(d['sim_minus_real']['ci95'][1])}] "
                  f"r2-r1 {pct(d['run2_minus_run1']['point'])} [{pct(d['run2_minus_run1']['ci95'][0])},{pct(d['run2_minus_run1']['ci95'][1])}]")
    for s in res["sameness"]:
        print("sameness", s["readout"], s["level"], s["strata"], f"AUROC C vs AuB {s['auroc']:.3f} [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}]")
