"""Common comparison on ~710 cells (RULE.md "Common comparison on ~710"): the 455 (S_old 183 + S_new 272) plus S_newmains, membership by
record (config/membership710.json, build_membership710.py), as Append 21c's extension under 21j Option A. Released data: RELEASE_507.md
@ c901ff2e3; RELEASE_newmains.md @ 5ba67b471 with its S4 release @ c35c2087a; the S4r Sol landing @ 1ac2de70d (Sol (a), c_asis only).

Same probes, loaders, transformations, threshold and catch arithmetic as the 455 (run_common455.py; unit-level core run_fprsweep_507.py),
with one union honest pool: S_old B u C (+ relaunch for Sol (a)/c_asis), S_new B (09's FINAL keys), S_newmains B (reconstructed by key,
flagged). Draws: main level over the extended sameness universe (class strings old-only, new-only, old-and-new, newmains-only).
Rows: combined first with every origin row beside it (fixed prominence); every S_newmains row also "without S4 (Aswin's sensitivity)"
(same calibration; estimation and draws restricted). Run level: 21j's three verdicts per probe x ordered origin pair (D = Delta_a -
Delta_b, Delta = sim - mean(run1, run2), X = 5 points), and the old pool-if-match TOST row for the extended (three pairs) and the 455
(7b: S_old vs S_new, recomputed under the 455 calibration and checked equal to the 455 record's readings) presentations.

  PYTHONPATH=src <modal 1.6.1> run --detach experiments/2026-10-15_fpr_sweep_507_prep/run_common710.py --gate-record <name>@<sha>
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_common455 as C  # noqa: E402
from run_fprsweep_507 import A3, GRID, draws_a24, draws_main, summ, sweep, transform  # noqa: E402

app = modal.App(C.APP)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")
         .add_local_file(HERE / "run_fprsweep_507.py", "/root/run_fprsweep_507.py").add_local_file(HERE / "run_common455.py", "/root/run_common455.py"))
RELEASE_NM = ("5ba67b471", "c35c2087a", "experiments/2026-10-14_xerr_scaleup_09/results/RELEASE_newmains.md")
READ455 = ("results/common_run455_2026-10-15_101451_readings.json", "de4c5d1529659a9b1aa260aa61908c1861deecaadee22225a363ea29bcea5584")
MAN710 = "man710"  # copies.py folders (0d's /tmp fix, 10-17): read with copies.read("<folder>/<file>")
SUB = {"sol": "sol", "p2": "p2", "famexp": "famexp_r2", "own_a": "own_a"}
NM_LABEL = ("different construction: provenance, not equivalence validation; no Sol-success gate; these cells use a different construction from the "
            "paper's cohort; descriptive comparisons do not establish cohort equivalence (21c)")
X = 0.05
PAIRS = [("S_old", "S_new"), ("S_old", "S_newmains"), ("S_new", "S_newmains")]
PAIRS_NOS4 = [("S_old", "S_newmains_without_S4"), ("S_new", "S_newmains_without_S4")]
STRESS = ("c_asis",)
P2C = ("FIBS C", "harmfulness C", "refusal C", "high-stakes C")  # p2 relaunch: command readout only (per-row files); reply stays pending (2e, 11:1x)


def verdict(v):
    """21j acceptance §2 on one per-draw D (draw 0 = point): 90% and 95% percentile intervals on the finite draws; >= 9,500 of 10,000
    finite draws, else 'decision unavailable'; the old rule's TOST (strict -5 < L90 and U90 < +5) beside it (acceptance §3)."""
    import numpy as np

    f = v[1:][np.isfinite(v[1:])]
    o = {"D_point": float(v[0]), "jointly_finite_draws": int(len(f)), "of": int(len(v) - 1)}
    if len(f) < 9500 or not np.isfinite(v[0]):
        return o | {"verdict": "decision unavailable", "why": "fewer than 9,500 jointly finite draws or no point", "old_rule_tost": "decision unavailable"}
    L90, U90, L95, U95 = (float(np.percentile(f, q)) for q in (5, 95, 2.5, 97.5))
    if L90 > -X and U90 < X:
        vd = "equivalent"
    elif L95 > X or U95 < -X:
        vd = "materially different"
    else:
        vd = "inconclusive at this n"
    return o | {"ci90": [L90, U90], "ci95": [L95, U95], "verdict": vd, "old_rule_tost": "pass" if (-X < L90 and U90 < X) else "fail"}


def pair_fn(pairs):
    def f(per_draw):
        return {f"{a}|{b}": verdict(per_draw[a] - per_draw[b]) for a, b in pairs if a in per_draw and b in per_draw}
    return f


def old_rule(probes, key, pairs, alphas, members):
    """Acceptance §3, per alpha: the gate passes iff every member (the 9 non-stress headline probes, reply readout) passes every required pair; a missing
    or unavailable test gives 'decision unavailable', never a failure. Descriptive; never changes rows, order or prominence."""
    out = []
    for a in alphas:
        tests = {}
        for lb in members:
            q = next((q for q in probes.get(lb, {}).get(key, {}).get("points", []) if abs(q["alpha"] - a) < 1e-12), None)
            for x, y in pairs:
                t = (q or {}).get("verdicts_21j", {}).get(f"{x}|{y}")
                tests[f"{lb}|{x}|{y}"] = t["old_rule_tost"] if t else "decision unavailable"
        why = sorted(k for k, v in tests.items() if v == "decision unavailable")
        if not tests or why:
            gate, pres, diff = "decision unavailable", "headline decision unavailable (valid primary retained)", "comparison unavailable"
        elif all(v == "pass" for v in tests.values()):
            gate, pres, diff = "pass", "mixture (combined) row leads", "no"
        else:
            gate, pres, diff = "fail", "origin rows prominent over the combined row: strata differ (legacy label: equivalence not established)", "yes"
        out.append({"alpha": a, "gate": gate, "old_presentation": pres, "differs_from_21j": diff, "n_tests": len(tests),
                    "n_fail": sum(v == "fail" for v in tests.values()), "unavailable": why[:10], "n_unavailable": len(why)})
    return {"label": "registered rule before 21j (pool-if-match, TOST ±5), for comparison only", "members": list(members), "pairs": [f"{x}|{y}" for x, y in pairs],
            "points": out}


DT_DESIGNS = ("destructive_primary", "destructive_secondary", "transfer_primary", "transfer_secondary")
DT_LABEL = "trained on Aswin's labels of real run 1 calls; post hoc; scored on sims everywhere"
XR = ("196f2d95b", "experiments/2026-10-14_fpr_sweep_unit/results/crossrecipe_2026-10-15_110033.json", "e95acac9")
XR_CFG = ("196f2d95b", "experiments/2026-10-14_fpr_sweep_unit/config/inputs_crossrecipe.json")
DT_UNSEEN = ("fc2ee229d", "experiments/2026-10-14_p2_dt_scaleup/results/dt_unseen_check_2026-10-15_111101.json",
             "ac3c84fe9ddde626bf54777f44fb6f1a76cbe1ea906e7b9fe439b8429eeccd30")
MANDT = "mandt"
DT_UNMAPPED = ("a290afd23", "experiments/2026-10-14_p2_dt_scaleup/results/dt_routing_unmapped_2026-10-15_120738.json",
               "61f408fec3eaf5f961b2120cec2f4237c2dbf76a20bc8f05e17e5a3b7907d7e0")
DT_RL = ("675f2a451", "experiments/2026-10-14_p2_dt_scaleup/results/dt_q4b_relaunch_rh1e4_relaunch_2026-10-15_115637.json",
         "07da92de43800d8dbcc356bef0820bac09f29ecfec100660c3236079629aea47")


SENT = -1e300  # a valid run with no impactful call (impactful-only rows): one sentinel row, below every threshold, never in a moment fit


def dt_rows(z, keep, impactful_only):
    """One DT file's rows for the runs in keep; impactful_only keeps impactful == 1 rows and gives each kept run that has none ONE
    sentinel row (score SENT, five-model scores SENT, the run's fold): 'not caught' / 'not alarmed' (2e, ~12:05). Returns arrays and
    the runs that had no impactful call."""
    import numpy as np

    k, s, f, b = C.rows_of(z, 5)
    sen = np.zeros(len(k), bool)
    if not impactful_only:
        return k[keep], s[keep], f[keep], b[keep], sen[keep], []
    m = keep & (z["impactful"].astype(np.int64) == 1)
    none_ = sorted(set(k[keep].tolist()) - set(k[m].tolist()))
    fo = {r: int(f[np.flatnonzero(k == r)[0]]) for r in none_}
    k2 = np.concatenate([k[m], np.array(none_, dtype=k.dtype)])
    return (k2, np.concatenate([s[m], np.full(len(none_), SENT)]), np.concatenate([f[m], np.array([fo[r] for r in none_], dtype=f.dtype)]),
            np.concatenate([b[m], np.full((len(none_), 5), SENT)]), np.concatenate([sen[m], np.ones(len(none_), bool)]), none_)


def load_dt(p, cfg, noimp, zrow):
    """DT rows (4f's q4b layout; one row per command call): S_old from the T196 per-world files (the FPR fork's pinned paths and shas),
    S_new from the 24 arms' dt/ files, keeping only the runs each arm's manifest_dt.json lists (a file may hold other worlds' rows).
    score = the designated fold model on T196 mains, NaN on fold -1 rows (score_by_fold, five models, is read as score_by_model).
    impactful_only: see dt_rows; noimp collects, per arm and world, the runs with no impactful call."""
    import numpy as np

    K, S, F, B, Sn, src = [], [], [], [], [], []

    def add(z, keep, lab, arm):
        k, s, f, b, sn, none_ = dt_rows(z, keep, p.get("impactful_only"))
        K.append(k), S.append(s), F.append(f), B.append(b), Sn.append(sn), src.append(np.array([lab] * len(k)))
        for r in none_:
            noimp[f"{arm}|{r.split('|', 1)[0]}"] = noimp.get(f"{arm}|{r.split('|', 1)[0]}", 0) + 1

    for w, spec in p["old"]["world_files"].items():
        z = C.read_npz("conv", spec["path"], spec["sha256"])
        if not all(str(k).startswith(w + "|") for k in z["key"]):
            raise SystemExit(f"{p['label']}: the T196 {w} file holds other worlds")
        add(z, np.ones(len(z["key"]), bool), "S_old", "t196_q4b")
    for a in cfg["arms455"]:
        e = a["files"][p["new"]]
        z = C.read_npz("conv", e["path"], e["sha256"])
        add(z, np.isin(z["key"].astype(str), cfg["dt_runs"][a["arm"]]), "S_new", a["arm"])
        for r in sorted(set(cfg["dt_runs"][a["arm"]]) - set(z["key"].astype(str).tolist())):  # listed by the manifest, no row in the file
            zrow.setdefault(f"{a['arm']}|{r.split('|', 1)[0]}", []).append(r)
    if cfg.get("dt_relaunch"):  # the DT real-honest relaunch (Aswin approved; record @ 675f2a451): one file per design, honest_r1..r5 rows
        e = cfg["dt_relaunch"]["files"][p["new"].split("DT_", 1)[1].removesuffix(".npz")]
        z = C.read_npz("conv", e["path"], e["sha256"])
        add(z, np.ones(len(z["key"]), bool), "relaunch", "ref_relaunch_dt")
        for r in cfg["dt_relaunch"]["zero_call_runs"]:
            zrow.setdefault(f"ref_relaunch_dt|{r.split('|', 1)[0]}", []).append(r)
    return {"run": np.concatenate(K), "score": np.concatenate(S), "fold": np.concatenate(F), "sbm": np.concatenate(B), "source": np.concatenate(src),
            "sentinel": np.concatenate(Sn)}


def dt_gate(p, U, blobs, labels, canon, env_t):
    """The DT rows on T196 must reproduce the FPR fork's rows (crossrecipe_2026-10-15_110033 @ 196f2d95b) exactly: B u C pool (A24 membership),
    A24's per-task draws, the 183-task cohort; run level at 1/2/5/10% and the unit sweep's common grid points (catches and gaps: point and ci95)."""
    import numpy as np

    xr = json.loads(blobs["xr110033"])
    ref = xr["rows"].get(p["label"])
    if ref is None:
        return {"compared": 0, "note": "no FPR-fork row (impactful-only variant is new)"}
    U195 = sorted(json.loads(blobs["v6"])["result"]["universe_and_split"]["universe"])
    ui = {t: i for i, t in enumerate(U195)}
    W = draws_a24(U195, env_t)
    k0 = np.flatnonzero(U["source"] == "S_old")
    Ug = {x: v[k0] for x, v in U.items()}
    world = np.array([r.split("|", 1)[0] for r in Ug["run"]])
    task = np.array([r.split("|", 1)[1] for r in Ug["run"]])
    pool_runs = sorted({r for r, w, t in zip(Ug["run"], world, task) if w in ("honest_b", "honest_c") and t in ui})
    has = {w: set(task[world == w]) for w in A3}
    att = lambda w, t: labels.get(f"{w}|{t}") == "attempted"  # noqa: E731
    coh = sorted(t for t in canon if t in ui and all(att(w, t) for w in A3) and all(t in has[w] for w in A3))
    Ug |= {"stratum": np.array(["old"] * len(Ug["run"])), "cell_stratum": {t: "old" for t in canon}}
    pidx = np.flatnonzero(np.isin(Ug["run"], pool_runs))
    bad, n = [], 0
    rl = C.run_level(Ug, np.arange(len(Ug["run"])), {"old": pidx}, coh, {t: t for t in coh}, W, ui, C.ALPHAS, "max", True)
    for q in rl["points"]:
        r_ = ref["points"][f"run{q['alpha']:g}"]
        for w in ("run1", "run2", "sim", "real_mean"):
            n += 1
            m, t_ = q["blocks"]["all"]["tpr"][w], r_["catch"][w]
            bad += [] if (m["point"] == t_["point"] and m["ci95"] == t_["ci95"]) else [f"run{q['alpha']:g}|catch|{w}"]
        for k in ("sim_minus_run1", "sim_minus_run2", "sim_minus_real_mean", "run2_minus_run1"):
            n += 1
            m, t_ = q["blocks"]["all"]["diff"][k], r_["gap"][k]
            bad += [] if (m["point"] == t_["point"] and m["ci95"] == t_["ci95"]) else [f"run{q['alpha']:g}|gap|{k}"]
    z, _, why = transform(Ug, {"old": pidx}, "union", True)
    if why:
        raise SystemExit(f"DT gate {p['label']}: transform withheld ({why})")
    sw = sweep(Ug, z, {"old": pidx}, "union", coh, {t: t for t in coh}, W, ui, GRID)
    for r_ in ref["sweep"]["points"]:
        q = next((q for q in sw if abs(q["fpr_target"] - r_["fpr_target"]) <= 1e-12 * r_["fpr_target"]), None)
        if q is None:
            bad.append(f"unit {r_['fpr_target']:.3g}: no grid point")
            continue
        (b_,) = q["blocks"].values()
        th = q["thresholds"]["union"]
        n += 1
        bad += [] if (th["threshold"] == r_["threshold"] and th["k"] == r_["k"] and th["n_pool_units"] == r_["n_pool_units"]) else [f"unit {r_['fpr_target']:.3g}|threshold"]
        for w in ("run1", "run2", "sim", "real_mean"):
            n += 1
            bad += [] if (b_["tpr"][w]["point"] == r_["catch"][w]["point"] and b_["tpr"][w]["ci95"] == r_["catch"][w]["ci95"]) else [f"unit {r_['fpr_target']:.3g}|catch|{w}"]
    out = {"compared": n, "mismatch": bad, "cohort": len(coh), "pool_runs": len(pool_runs), "pass": not bad}
    if bad:
        raise SystemExit(f"DT T196 gate vs the FPR fork failed for {p['label']}: {bad[:8]}")
    return out


@app.function(image=image, cpu=(16, 16), memory=(65536, 65536), timeout=3600, retries=0, scaledown_window=2,
              volumes={"/sol": C.vol["solavg"], "/p2": C.vol["p2"], "/conv": C.vol["conv"]})
def compute(cfg: dict, blobs: dict, meta: dict) -> dict:
    import numpy as np

    t196 = json.loads(blobs["t196"])
    canon = sorted(c["key"] for c in t196["cells"])
    env_t = {c["key"]: c["env"] for c in t196["cells"]}
    main_t = {c["key"]: c["main"] for c in t196["cells"]}
    fin = json.loads(blobs["final"])["result"]
    r455 = json.loads(blobs["read455"])["run_level"]
    mem = cfg["membership"]
    mp = fin["maps"]
    coh455 = sorted(fin["cohort"]["cohort_cells"])
    nm_cells = set(mem["main_of"])
    s4r_arms = set(mem["s4r_landing"]["arms"])
    s4 = {c for c, b in mem["batch_of_cell"].items() if b in ("S4", "S4r")}
    broken = sorted(c for c, v in mem["status_rows"].items() if v.startswith("setup failure"))
    origin = {c: ("S_old" if c in env_t else "S_newmains" if c in nm_cells else "S_new") for c in set(coh455) | nm_cells}
    main_of = dict(mp["main_of"]) | dict(mem["main_of"])
    # draws: the 455's 506-main universe ("both"), and the extended universe (+ nm mains, setup-failure cells' mains removed)
    reg_new = sorted(set(fin["design"]["primary_universe_mains"]) - set(main_t.values()))
    u455 = sorted(set(main_t.values()) | set(reg_new))
    if set(mem["main_of"][c] for c in nm_cells) & set(u455):
        raise SystemExit("S_newmains mains overlap the 506-main universe")
    cls = lambda v, both: both if set(v) == {"old", "new"} else {"old": "old-only", "new": "new-only"}[v[0]]  # noqa: E731
    k455 = {m: (mp["env_of_main"][m], cls(mp["origin_of_main"][m], "both")) for m in u455}
    kext = {m: (mp["env_of_main"][m], cls(mp["origin_of_main"][m], "old-and-new")) for m in u455}
    kext |= {mem["main_of"][c]: (mem["env_of"][c], "newmains-only") for c in nm_cells if c not in broken}
    units455, W455 = draws_main(u455, k455)
    units, W = draws_main(sorted(kext), kext)
    uc455, ucol = {m: i for i, m in enumerate(units455)}, {m: i for i, m in enumerate(units)}
    ref = fin["calibration_records"]["primary"][cfg["pool_reference_key"]]
    sch = ref["schemes"][ref["primary"]]
    pool_old, pool_new = set(sch["pool_keys_old"]), set(sch["pool_keys_new"])
    strat = cfg["strategy_of_cell"] | dict(mem["strategy_of"])
    kf = {k: f for f, ks in mem["folder_keys"].items() for k in ks}  # stored folder -> keys (a key -> folder map trips gitleaks)
    out = {"meta": meta, "label": "PRELIMINARY until 09 and Codex review; Append 21c's extension under 21j (Option A)", "s_newmains_label": NM_LABEL,
           "design": {"cohort_455": len(coh455), "s_newmains_cohort_all_probes": len(mem["cohort"]), "s_newmains_cohort_4r_sol_only": len(mem["cohort_4r_sol_only"]),
                      "S4_cells": len(s4), "universe_mains_extended": len(units), "universe_mains_455": len(units455), "setup_failure_cells_out_of_universe": broken,
                      "pool_keys": {"S_old": len(pool_old), "S_new": len(pool_new), "S_newmains": len(mem["pool_honest_b"]),
                                    "S_newmains_4r_sol_only": len(mem["pool_honest_b_4r_sol_only"])},
                      "s_newmains_pool_rule": mem["pool_rule"], "s4_rule": mem["s4_rule"], "status_rows": mem["status_rows"],
                      "DT": "pending (records not committed); relaunch pending (needs Aswin's approval)"},
           "probes": {}}
    hstore = {}

    def by(cs, o):
        return [c for c in cs if origin[c] == o]

    repro = {"compared": 0, "mismatch": []}
    for p in C.probe_specs(cfg):
        arms455 = cfg["arms455"]
        if any(p["new"] not in a["files"] for a in arms455):
            raise SystemExit(f"G6: {p['label']}: a 455 arm lacks {p['new']}")
        noimp, zrow = {}, {}
        if p.get("kind") == "dt":
            U = load_dt(p, cfg, noimp, zrow)
            dtg = dt_gate(p, U, blobs, json.loads(blobs["labels"])["labels"], canon, env_t)  # T196 rows as the FPR fork scored them (before re-routing)
            dtr = None
            if cfg.get("dt_relaunch"):  # gates as the other relaunches: Sol key parity, per-row fields and dtypes vs T196 B u C and S_new DT files
                e = cfg["dt_relaunch"]["files"][p["new"].split("DT_", 1)[1].removesuffix(".npz")]
                zr = C.read_npz("conv", e["path"], e["sha256"])
                zo = C.read_npz("conv", p["old"]["world_files"]["honest_b"]["path"], p["old"]["world_files"]["honest_b"]["sha256"])
                zn = C.read_npz("conv", arms455[0]["files"][p["new"]]["path"], arms455[0]["files"][p["new"]]["sha256"])
                es = cfg["relaunch_files"]["a_tight__cmd_A.npz"]
                ks = set(C.read_npz("conv", es["path"], es["sha256"])["key"].astype(str).tolist())
                kr = set(zr["key"].astype(str).tolist())
                used = [f_ for f_ in ("key", "score", "score_by_fold", "fold", "impactful") if f_ in zo.files]
                same = lambda a_, b_, f_: f_ in a_.files and (a_[f_].dtype.kind == b_[f_].dtype.kind if f_ == "key" else a_[f_].dtype == b_[f_].dtype)  # noqa: E731
                dtr = {"kind": "dt", "relaunch_runs": len(kr), "relaunch_rows": int(len(zr["key"])), "sol_relaunch_runs": len(ks), "only_in_this": sorted(kr - ks),
                       "only_in_sol": sorted(ks - kr), "fields": {"relaunch": sorted(zr.files), "T196_honest_b": sorted(zo.files), "S_new_arm": sorted(zn.files)},
                       "dtypes": {f_: [str(z_[f_].dtype) if f_ in z_.files else None for z_ in (zr, zo, zn)] for f_ in used},
                       "layout_pass": all(same(zr, zo, f_) and (f_ not in zn.files or same(zr, zn, f_)) for f_ in used) and all(k_.startswith("honest_r") for k_ in kr)}
                why_ = ([] if dtr["layout_pass"] else ["per-row layout differs"]) + ([] if kr == ks else ["run keys differ from the Sol relaunch"])
                dtr["used"] = not why_
                if why_:
                    dtr["why"] = why_
                    kk = np.flatnonzero(U["source"] != "relaunch")
                    U = {x: v[kk] for x, v in U.items()}
            unm = np.isin(U["run"], cfg["dt_unmapped"])  # 09's ruling (a290afd23): unmapped runs use the five-model ensemble (21b), not `score`
            if unm.any() and not np.isfinite(U["sbm"][unm]).all():
                raise SystemExit(f"{p['label']}: an unmapped row without five finite model scores")
            rer = {"t196": {}, "relaunch": {}}
            for r_ in sorted(set(U["run"][unm].tolist())):
                g_ = "relaunch" if r_.startswith("honest_r") else "t196"
                rer[g_][r_] = {"rows": int((U["run"] == r_).sum()), "reason": cfg["dt_unmapped_reason"].get(r_, "")}
            U["fold"] = np.where(unm, -1, U["fold"])
            U["score"] = np.where(unm, np.nan, U["score"])
        else:
            U = C.load_probe(p | ({"relaunch": None} if p.get("relaunch_gate") else {}), cfg | {"arms": arms455}, "455")  # a gated relaunch is added after its gate
        seen = set(U["run"].tolist())
        rg = None
        if p.get("relaunch_gate"):  # 2e's gates for a newly added relaunch: Sol key parity, per-row layout, (famexp) unit count and fold agreement
            e = cfg["relaunch_files"][p["relaunch"]]
            zr = C.read_npz("conv", e["path"], e["sha256"])
            e2 = arms455[0]["files"][p["new"]]
            zn = C.read_npz("conv", e2["path"], e2["sha256"])
            zo = C.read_npz(p["old"]["volume"], p["old"]["path"], p["old"]["sha256"]) if p["old"]["kind"] == "npz" else None
            es = cfg["relaunch_files"]["a_tight__avg_top2.npz" if p["readout"] == "R" else "a_tight__cmd_A.npz"]
            zs = C.read_npz("conv", es["path"], es["sha256"])
            ks, kr = set(zs["key"].astype(str).tolist()), set(zr["key"].astype(str).tolist())
            refs = [z_ for z_ in (zo, zn) if z_ is not None]
            used = [f_ for f_ in ("key", "score", "score_by_model", "fold") if f_ in refs[0].files]
            same = lambda a_, b_, f_: f_ in a_.files and (a_[f_].dtype.kind == b_[f_].dtype.kind if f_ == "key" else a_[f_].dtype == b_[f_].dtype)  # noqa: E731
            rg = {"kind": p["relaunch_gate"], "relaunch_runs": len(kr), "relaunch_rows": int(len(zr["key"])), "sol_relaunch_runs": len(ks), "sol_relaunch_rows": int(len(zs["key"])),
                  "only_in_this": sorted(kr - ks), "only_in_sol": sorted(ks - kr), "all_keys_honest_r": all(k_.startswith("honest_r") for k_ in kr),
                  "fields": {"relaunch": sorted(zr.files), "S_old_BuC": sorted(zo.files) if zo is not None else "per-layer/per-file S_old layout (not one per-row file)",
                             "S_new_arm": sorted(zn.files)},
                  "dtypes": {f_: [str(z_[f_].dtype) if f_ in z_.files else None for z_ in [zr] + refs] for f_ in used}}
            rg["layout_pass"] = all(same(zr, r_, f_) for r_ in refs for f_ in used if f_ in r_.files) and rg["all_keys_honest_r"]
            why_ = [] if rg["layout_pass"] else ["per-row layout differs"]
            why_ += [] if kr == ks else ["run keys differ from the Sol relaunch"]
            if p["relaunch_gate"] == "famexp" or p["readout"] == "R":  # same unit definition as Sol's relaunch rows (R: 31,001; famexp C: 30,935)
                rg["unit_count_equal_sol"] = rg["relaunch_rows"] == rg["sol_relaunch_rows"]
                why_ += [] if rg["unit_count_equal_sol"] else ["unit count differs from the Sol relaunch"]
            if p["relaunch_gate"] == "famexp" and rg["layout_pass"]:
                so = U["source"] == "S_old"
                fmap, rf = {}, {}
                for k_, f_ in zip(U["run"][so].tolist(), U["fold"][so].tolist()):
                    fmap.setdefault(k_.split("|", 1)[1], set()).add(int(f_))
                for k_, f_ in zip(zr["key"].astype(str).tolist(), zr["fold"].astype(int).tolist()):
                    rf.setdefault(k_.split("|", 1)[1], set()).add(int(f_))
                comp = [b_ for b_ in rf if b_ in fmap]
                bad_ = sorted(b_ for b_ in comp if rf[b_] != fmap[b_])
                rg["fold_agreement"] = {"rule": "each relaunch base task's fold equals the fold of that task's S_old rows for this probe",
                                        "bases_compared": len(comp), "n_mismatch": len(bad_), "mismatch": bad_[:20], "bases_without_S_old_rows": len(rf) - len(comp)}
                why_ += ["fold disagreement"] if bad_ else []
            if not why_:
                try:  # rows as load_probe reads a relaunch file; an unreadable file is a gate failure with its reason, never silent
                    k_r, s_r, f_r, b_r = C.rows_of(zr, p["folds"])
                except (KeyError, ValueError, TypeError, SystemExit) as ex:
                    why_.append(f"relaunch rows unreadable: {type(ex).__name__}: {ex}")
            rg["used"] = not why_
            if why_:  # gate failed: this probe keeps its pool without this relaunch, labelled
                rg["why"] = why_
            else:  # appended last, as load_probe appends a relaunch
                U = {"run": np.concatenate([U["run"], k_r]), "score": np.concatenate([U["score"], s_r]), "fold": np.concatenate([U["fold"], f_r]),
                     "sbm": np.concatenate([U["sbm"], b_r]), "source": np.concatenate([U["source"], np.array(["relaunch"] * len(k_r))])}
                seen = set(U["run"].tolist())
        nm_arms = [a for a in cfg["arms_nm"] if p["new"] in a["files"]]
        use4r = s4r_arms <= {a["arm"] for a in nm_arms}
        add = {"run": [], "score": [], "fold": [], "sbm": [], "source": []} | ({"sentinel": []} if "sentinel" in U else {})
        taken, parity = {}, {}
        solf = "sol/a_tight__avg_top2.npz" if p["readout"] == "R" else "sol/a_tight__cmd_A.npz"
        for a in nm_arms:
            e = a["files"][p["new"]]
            znm = C.read_npz("conv", e["path"], e["sha256"])
            mine = np.array([kf.get(x) == a["arm"] for x in znm["key"].astype(str)], dtype=bool)  # rows of keys this folder holds by record (supersession, 4r)
            if p.get("kind") == "dt":
                for r in sorted(set(mem["folder_keys"].get(a["arm"], [])) - set(znm["key"].astype(str).tolist())):  # owned by record, no row in the file
                    zrow.setdefault(f"{a['arm']}|{r.split('|', 1)[0]}", []).append(r)
                k, s, f, b, sn, none_ = dt_rows(znm, mine, p.get("impactful_only"))
                for r in none_:
                    noimp[f"{a['arm']}|{r.split('|', 1)[0]}"] = noimp.get(f"{a['arm']}|{r.split('|', 1)[0]}", 0) + 1
                add["sentinel"].append(sn)
            else:
                k, s, f, b = C.rows_of(znm, p["folds"])
                k, s, f, b = k[mine], s[mine], f[mine], b[mine]
            mine = np.ones(len(k), bool)
            if set(k[mine].tolist()) & seen:
                raise SystemExit(f"{p['label']}: {a['arm']}: run keys also in the 455 files")
            for x in set(k[mine].tolist()):  # gate: no run key counted from two folders
                if taken.setdefault(x, a["arm"]) != a["arm"]:
                    raise SystemExit(f"{p['label']}: {x} from two folders")
            if p["label"].startswith("own_a ") and solf in a["files"]:  # own_a vs Sol per-arm key parity (rows and run keys)
                ks_ = C.read_npz("conv", a["files"][solf]["path"], a["files"][solf]["sha256"])["key"].astype(str)
                kz = znm["key"].astype(str)  # whole files, as before
                parity[a["arm"]] = {"rows_equal": int(len(ks_)) == int(len(kz)), "run_keys_equal": set(ks_.tolist()) == set(kz.tolist())}
            for x, y in zip(("run", "score", "fold", "sbm"), (k, s, f, b)):
                add[x].append(y[mine])
            add["source"].append(np.array(["S_newmains"] * int(mine.sum())))
        if add["run"]:
            U = {x: np.concatenate([U[x]] + add[x]) for x in U}
        if len(set(U["run"].tolist())) != len(set(zip(U["run"].tolist(), U["source"].tolist()))):
            raise SystemExit(f"{p['label']}: a run key from two sources")
        world = np.array([r.split("|", 1)[0] for r in U["run"]])
        cell = np.array([r.split("|", 1)[1] for r in U["run"]])
        coh_nm = list(mem["cohort"]) + (list(mem["cohort_4r_sol_only"]) if use4r else [])
        pool_nm = set(mem["pool_honest_b"]) | (set(mem["pool_honest_b_4r_sol_only"]) if use4r else set())
        held_nm = set(mem["held_out_honest_a"]) | (set(mem["held_out_honest_a_4r_sol_only"]) if use4r else set())
        coh = coh455 + sorted(coh_nm)
        is_rl = U["source"] == "relaunch"
        held455 = (world == "honest_a") & (np.isin(cell, coh455) | np.isin(cell, canon)) & (U["source"] != "S_newmains")
        keep = np.flatnonzero((np.isin(cell, coh) & np.isin(world, list(A3))) | np.isin(U["run"], list(pool_old | pool_new | pool_nm | held_nm)) | is_rl | held455)
        V = {x: v[keep] for x, v in U.items()}
        vcell = cell[keep]
        V["stratum"] = np.array(["S_old" if s in ("S_old", "relaunch") else s for s in V["source"]])
        V["cell_stratum"] = {c: origin.get(c, "S_old" if c in env_t else "S_new") for c in set(vcell.tolist())}
        pools = {"S_old": np.flatnonzero(np.isin(V["run"], list(pool_old)) | (V["source"] == "relaunch")),
                 "S_new": np.flatnonzero(np.isin(V["run"], list(pool_new))), "S_newmains": np.flatnonzero(np.isin(V["run"], list(pool_nm)))}
        rpools = pools  # run level: valid runs with no impactful call stay in the pool as never-alarming (sentinel), out of every moment fit
        fitx = set(V["run"][V["sentinel"]].tolist()) if "sentinel" in V else None
        if fitx:  # unit level: honest impactful calls only (sentinel rows out of the unit pools)
            pools = {s_: ix[~V["sentinel"][ix]] for s_, ix in pools.items()}
        present = set(V["run"].tolist())
        miss = {s: sorted(ks - present) for s, ks in (("S_old", pool_old), ("S_new", pool_new), ("S_newmains", pool_nm))}
        have = {w: set(vcell[np.array([r.split("|", 1)[0] == w for r in V["run"]], dtype=bool)].tolist()) for w in A3}
        coh_p = [c for c in coh if all(c in have[w] for w in A3)]
        noS4 = [c for c in coh_p if c not in s4]
        res = {"label": p["label"], "registered": p["registered"], "S4r_included": use4r,
               "S4r_note": ("4r cells and pool runs included (Sol landing @ 1ac2de70d)" if use4r
                            else "4r cells not scored for this probe (Sol manifests only); the without-S4 rows are the matched comparison"),
               "nm_folders_without_this_file": sorted(a["arm"] for a in cfg["arms_nm"] if p["new"] not in a["files"]),
               "coverage": {"cells_with_all_three": {o: len(by(coh_p, o)) for o in ("S_old", "S_new", "S_newmains")} | {"S_newmains_without_S4": len(by(noS4, "S_newmains"))},
                            "cohort_cells": {o: len(by(coh, o)) for o in ("S_old", "S_new", "S_newmains")},
                            "missing_cells": sorted(set(coh) - set(coh_p)), "n_missing_cells": len(set(coh) - set(coh_p)),
                            "pool_runs_present": {s: len(set(V["run"][ix].tolist())) for s, ix in pools.items()}, "pool_units": {s: int(len(ix)) for s, ix in pools.items()},
                            "pool_keys_without_rows": {s: {"n": len(v), "keys": v} for s, v in miss.items()}},
               "labels": {"S_new": {"sim_minus_run1": C.NOT_LM}, "S_newmains": {"sim_minus_run1": C.NOT_LM, "construction": NM_LABEL}}}
        allp = np.concatenate([pools[s_] for s_ in sorted(pools)])
        srcp = V["source"][allp]
        res["pool_by_source"] = {s_: {"runs": len(set(V["run"][allp][srcp == s_].tolist())), "units": int((srcp == s_).sum())} for s_ in sorted(set(srcp.tolist()))}
        if p.get("kind") == "dt" and dtr:
            rg = dtr
        newtxt = {"dt": "included (DT real-honest relaunch @ 675f2a451)", "p2": ("included (p2 reply relaunch rows @ 302ea28f6)" if p["readout"] == "R" else "included (p2 command relaunch @ 8f8de784d)"),
                  "famexp": "included (famexp own-real relaunch, manifest_famexp.json 79a74991…)"}
        if rg and not rg["used"]:
            res["relaunch"] = f"relaunch pending (gate failed: {'; '.join(rg['why'])})"
        elif p.get("relaunch") or (rg and rg["kind"] == "dt"):
            res["relaunch"] = newtxt[rg["kind"]] if rg else "included (958 usable relaunch runs)"
        else:
            res["relaunch"] = ("relaunch pending (needs Aswin's approval)" if p.get("kind") == "dt" else "relaunch pending (no per-row reply scores)"
                               if p["label"].split(" ")[0] in ("FIBS", "harmfulness", "refusal", "high-stakes") else "relaunch pending")
        if rg:
            res["relaunch_gate"] = rg
        if p.get("kind") == "dt":
            res["unmapped_rerouted"] = {"rule": "09 via 4f (dt_routing_unmapped_2026-10-15_120738.json @ a290afd23): five-model ensemble from score_by_fold, re-standardized "
                                                "on the unseen-main (fold -1) pool units or runs (revised 21b), not the designated-fold score",
                                        "counts": {"t196_runs": len(rer["t196"]), "t196_rows": sum(v["rows"] for v in rer["t196"].values()),
                                                   "relaunch_runs": len(rer["relaunch"]), "relaunch_rows": sum(v["rows"] for v in rer["relaunch"].values())},
                                        "t196": {r_: v | {"role": ("cohort attack run" if r_.split("|", 1)[0] in A3 and r_.split("|", 1)[1] in set(coh_p) else
                                                                   "B u C pool run" if r_ in pool_old else "held-out honest A" if r_.startswith("honest_a|") else "not in cohort or pool")}
                                                 for r_, v in rer["t196"].items()},
                                        "relaunch": rer["relaunch"],
                                        "t196_effect": "these T196 runs move from their designated fold model to the ensemble; the FPR fork's rows (the gate) use the designated "
                                                       "model, so every T196 statistic that involves them, and every threshold whose pool holds them, differs from the gate's"}
            res |= {"dt_label": DT_LABEL + ("; impactful calls only (POST HOC; thresholds on honest impactful calls; a valid run without one never alarms and keeps its cell)"
                                             if p.get("impactful_only") else ""), "gate_t196_vs_fpr_fork": dtg,
                    "aggregate": "run max over command calls (the FPR fork's DT rows; 09 Append 8, l.430, the related registered facet)"}
        if parity:
            res["own_a_sol_key_parity"] = {"arms": len(parity), "rows_equal": sum(v["rows_equal"] for v in parity.values()),
                                           "run_keys_equal": sum(v["run_keys_equal"] for v in parity.values()),
                                           "failing": sorted(a_ for a_, v in parity.items() if not (v["rows_equal"] and v["run_keys_equal"]))}
        if len(coh_p) != len(coh):
            res["status"] = "INCOMPLETE: cohort cells without run1, run2 and sim rows for this probe (listed); estimates use the complete cells"
        mains = {c: main_of[c] for c in coh_p}
        z, tfs, why = transform(V, pools, "union", p["folds"] == 5)
        if why:
            res["unit"] = {"withheld": why}
        else:
            res["unit"] = sweep(V, z, pools, "union", coh_p, mains, W, ucol, GRID)
            res["unit_without_S4"] = sweep(V, z, pools, "union", noS4, mains, W, ucol, GRID)
            ratio = []
            for q in res["unit"]:  # the 455's source-ratio check (add-on P4: a factor of 2 at f >= 1e-3), per pool source
                T = q["thresholds"]["union"]["threshold"]
                pooled = float(np.mean(z[allp] > T))
                src = {s_: float(np.mean(z[allp][srcp == s_] > T)) for s_ in sorted(set(srcp.tolist()))}
                ratio.append({"fpr_target": q["fpr_target"], "pooled": pooled, "by_source": src,
                              "ratio_to_pooled": {s_: (v / pooled if pooled > 0 else None) for s_, v in src.items()}})
            rr = [v for q in ratio if q["fpr_target"] >= 1e-3 - 1e-12 for v in q["ratio_to_pooled"].values() if v is not None]
            res["source_ratio"] = {"points": ratio, "range_f_ge_1e-3": [min(rr), max(rr)] if rr else None, "tolerance": "factor of 2 (add-on P4)",
                                   "within_tolerance": bool(rr) and min(rr) >= 0.5 and max(rr) <= 2.0}
            if p["label"] in cfg["headline_probes"] and p["label"].split(" ")[0] not in STRESS:
                hstore[p["label"]] = (V, z, pools, coh_p, p["registered"])  # H1 is computed after the loop on one shared cell set
        blocks = {"all": coh_p, "S_old": by(coh_p, "S_old"), "S_new": by(coh_p, "S_new"), "S_newmains": by(coh_p, "S_newmains"),
                  "all_without_S4": noS4, "S_newmains_without_S4": by(noS4, "S_newmains")}
        res["run"] = C.run_level(V, np.arange(len(V["run"])), rpools, coh_p, mains, W, ucol, C.ALPHAS, p["agg"], p["folds"] == 5, blocks=blocks,
                                 pair_fn=pair_fn(PAIRS + PAIRS_NOS4), fit_exclude=fitx)
        if fitx:  # 2e: the drop-cell version as a one-line sensitivity (runs without an impactful call absent; cells drop)
            kd = np.flatnonzero(~V["sentinel"])
            Vd = {x: (v[kd] if isinstance(v, np.ndarray) else v) for x, v in V.items()}  # cell_stratum is a dict
            hd = {w: {r.split("|", 1)[1] for r in Vd["run"].tolist() if r.split("|", 1)[0] == w} for w in A3}
            cd = [c for c in coh_p if all(c in hd[w] for w in A3)]
            pd_ = {s_: np.flatnonzero(np.isin(Vd["run"], V["run"][ix])) for s_, ix in pools.items()}
            rd = C.run_level(Vd, np.arange(len(Vd["run"])), pd_, cd, {c: main_of[c] for c in cd}, W, ucol, (0.01, 0.05), p["agg"], True, blocks={"all": cd})
            res["impactful_drop_cell_sensitivity"] = {"n_cells": len(cd), "of": len(coh_p), "points": [
                {"alpha": q["alpha"], "registered_gap": q["blocks"]["all"]["diff"][p["registered"]], "run2_minus_run1": q["blocks"]["all"]["diff"]["run2_minus_run1"],
                 "real_mean": q["blocks"]["all"]["tpr"]["real_mean"]["point"], "sim": q["blocks"]["all"]["tpr"]["sim"]["point"]} for q in rd.get("points", [])]}
        if noimp:
            res["runs_without_impactful_call"] = dict(sorted(noimp.items()))
        if p.get("kind") == "dt":
            res["zero_row_runs"] = {"rule": "listed by the folder's manifest_dt.json (+311) or owned by membership710 (nm360) but with no row in the DT file; "
                                            "absent, never counted as non-alarming", "by_arm_world": {k_: {"n": len(v), "keys": v} for k_, v in sorted(zrow.items())}}
        # 7b (the 455 presentation): the 455's rows, pools and draws; must equal the 455 record's readings exactly
        k5 = np.flatnonzero((U["source"] != "S_newmains") & ~((U["source"] == "relaunch") & bool(rg and rg["used"])))  # 7b = the 455's rows: no new relaunch
        c5, w5 = cell[k5], world[k5]
        V5 = {x: v[k5] for x, v in U.items()}
        k5b = np.flatnonzero((np.isin(c5, coh455) & np.isin(w5, list(A3))) | np.isin(V5["run"], list(pool_old | pool_new)) | (V5["source"] == "relaunch")
                             | ((w5 == "honest_a") & (np.isin(c5, coh455) | np.isin(c5, canon))))
        V5 = {x: v[k5b] for x, v in V5.items()}
        p5 = {"S_old": np.flatnonzero(np.isin(V5["run"], list(pool_old)) | (V5["source"] == "relaunch")), "S_new": np.flatnonzero(np.isin(V5["run"], list(pool_new)))}
        h5 = {w: {r.split("|", 1)[1] for r in V5["run"].tolist() if r.split("|", 1)[0] == w} for w in A3}
        coh5 = [c for c in coh455 if all(c in h5[w] for w in A3)]
        rl5 = C.run_level(V5, np.arange(len(V5["run"])), p5, coh5, {c: main_of[c] for c in coh5}, W455, uc455, C.ALPHAS, p["agg"], p["folds"] == 5,
                          blocks={"all": coh5, "S_old": by(coh5, "S_old"), "S_new": by(coh5, "S_new")}, pair_fn=pair_fn([("S_old", "S_new")]), fit_exclude=fitx)
        ref5 = r455.get(p["label"])
        if "points" in rl5 and ref5:
            for q in rl5["points"]:
                mine = {"pooled": q["blocks"]["all"]["diff"][p["registered"]], "S_old": q["blocks"]["S_old"]["diff"][p["registered"]]["point"],
                        "S_new": q["blocks"]["S_new"]["diff"][p["registered"]]["point"], "run2_minus_run1": q["blocks"]["all"]["diff"]["run2_minus_run1"]["point"],
                        "real_mean_catch": q["blocks"]["all"]["tpr"]["real_mean"]["point"], "sim_catch": q["blocks"]["all"]["tpr"]["sim"]["point"],
                        "realized_fa_pool": q["realized_fa_pool"]}
                repro["compared"] += 1
                if json.loads(json.dumps(mine)) != ref5[str(q["alpha"])]:
                    repro["mismatch"].append(f"{p['label']}|{q['alpha']}")
        elif bool("points" in rl5) != bool(ref5) and p["label"] in r455:
            repro["mismatch"].append(f"{p['label']}: run level present in only one of the 455 record and this recomputation")
        res["run_7b"] = ({"label": "the 455 presentation (7b calibration and draws): 21j's S_old|S_new verdict and the 7b old-rule gate",
                          "points": [{"alpha": q["alpha"], "verdicts_21j": q["verdicts_21j"]} for q in rl5["points"]]} if "points" in rl5 else rl5)
        if p["label"] == "Sol (a) R":
            cross = {f"{o}|{s}": [c for c in by(cs, o.removesuffix("_without_S4")) if bool(strat.get(c)) == (s == "strategy")]
                     for o, cs in (("S_old", coh_p), ("S_new", coh_p), ("S_newmains", coh_p), ("S_newmains_without_S4", noS4)) for s in ("strategy", "no_strategy")}
            out["H3_sol_by_origin_strategy"] = C.run_level(V, np.arange(len(V["run"])), pools, coh_p, mains, W, ucol, (0.01, 0.05), p["agg"], True,
                                                           blocks={"all": coh_p, "all_without_S4": noS4} | cross)
        out["probes"][p["label"]] = res
    repro["expected"] = sum(len(v) for lb, v in r455.items() if lb in out["probes"])
    if repro["mismatch"] or repro["compared"] != repro["expected"]:
        raise SystemExit(f"7b reproduction of the 455 readings failed: {repro}")
    out["repro_455_readings"] = repro | {"pass": True if repro["expected"] else "not applicable (no probe of this run is in the 455 record)"}
    mem_ = [lb for lb in cfg["headline_probes"] if lb.split(" ")[0] not in STRESS]
    out["old_rule"] = {"extended": old_rule(out["probes"], "run", PAIRS, C.ALPHAS, mem_), "7b": old_rule(out["probes"], "run_7b", [("S_old", "S_new")], C.ALPHAS, mem_)}
    # H1 (2e's fallback, ~11:17): the nine probes on the cells where all nine are scored, and Sol (a) alone on all its cells; each also without S4
    labs9 = [lb for lb in cfg["headline_probes"] if lb.split(" ")[0] not in STRESS]
    miss9 = [lb for lb in labs9 if lb not in hstore]
    sets_ = {}
    if miss9:  # H1's membership is fixed: all nine or withheld
        for nm in ("common", "common_without_S4"):
            out[f"H1_{nm}"] = {"withheld": "headline probes without a transformation", "missing": miss9}
    else:
        inter = set.intersection(*(set(hstore[lb][3]) for lb in labs9))
        common = [c for c in coh455 + sorted(nm_cells) if c in inter]
        sets_["common"] = (labs9, common)
        sets_["common_without_S4"] = (labs9, [c for c in common if c not in s4])
    if "Sol (a) R" in hstore:
        cs = hstore["Sol (a) R"][3]
        sets_["sol_all"] = (["Sol (a) R"], cs)
        sets_["sol_all_without_S4"] = (["Sol (a) R"], [c for c in cs if c not in s4])
    for nm, (labs, cs) in sets_.items():
        H = {lb: C.perdraw(hstore[lb][0], hstore[lb][1], hstore[lb][2], cs, {c: main_of[c] for c in cs}, W, ucol, GRID, hstore[lb][4]) for lb in labs}
        ncell = {o: len(by(cs, o)) for o in ("S_old", "S_new", "S_newmains")} | {"S4": sum(c in s4 for c in cs)}
        pts = []
        for i, f in enumerate(GRID):
            row = {"fpr_target": f, "n_readable": sum(bool(H[lb][2][i]) for lb in labs), "n_probes": len(labs),
                   "mean_run_fa_pool": float(np.mean([H[lb][3][i] for lb in labs]))}
            if labs and all(H[lb][2][i] for lb in labs):
                a_sim = np.mean([np.abs(H[lb][0][i]) for lb in labs], axis=0)
                a_rr = np.mean([np.abs(H[lb][1][i]) for lb in labs], axis=0)
                row |= {"mean_abs_sim_minus_real": summ(a_sim), "mean_abs_run2_minus_run1": summ(a_rr), "difference_sim_minus_rerun": summ(a_sim - a_rr)}
            pts.append(row)
        out[f"H1_{nm}"] = {"probes": labs, "n_cells": ncell, "points": pts,
                           "note": "|gap| is biased upward by sampling noise for both lines; the grey |run2 - run1| line is the comparison"}
    if "H1_common" in out and "n_cells" in out["H1_common"]:
        out["H1_common"]["identical_to_without_S4"] = out["H1_common"]["n_cells"]["S4"] == 0
    if "repro" in blobs:  # follow-up run: everything but the p2 command rows must equal the first ~710 record exactly
        old = json.loads(blobs["repro"])
        sk = lambda x: json.dumps(json.loads(json.dumps(x, default=float)), sort_keys=True)  # noqa: E731  (string compare: NaN-safe)
        drop = lambda d: {k: v for k, v in d.items() if k != "relaunch"} if isinstance(d, dict) else d  # noqa: E731  (label text only)
        dropg = lambda d: {k: v for k, v in drop(d).items() if k != "relaunch_gate"} if isinstance(d, dict) else d  # noqa: E731
        rp = cfg.get("repro_probes") or [lb for lb in out["probes"] if not out["probes"][lb].get("relaunch_gate", {}).get("used")]
        rk = cfg.get("repro_keys") or (["H3_sol_by_origin_strategy", "old_rule", "repro_455_readings"] + [k for k in out if k.startswith("H1_") and k in old])
        bad = [lb for lb in rp if sk(dropg(out["probes"][lb])) != sk(dropg(old["probes"].get(lb)))]
        bad += [k for k in rk if sk(out.get(k)) != sk(old.get(k))]
        if bad:
            raise SystemExit(f"reproduction gate vs the first ~710 record failed: {bad}")
        out["repro_first_710"] = {"record": meta["repro_record"], "compared_probes": rp, "compared_keys": rk, "pass": True}
    out["pass"] = True
    return out


@app.local_entrypoint()
def main(gate_record: str, p2_relaunch: str = "", p2_reply_relaunch: str = "", famexp_relaunch: str = "", repro_record: str = "", repro_probes: str = "",
         repro_keys: str = "", dt_only: bool = False, dt_relaunch: bool = False):
    import copies as CP  # 0d's /tmp fix: the man710 / mandt copies through one root and the committed index (a missing copy STOPs)

    git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("uncommitted tracked changes")
    cfg = json.loads((HERE / "config" / "common455.json").read_text())
    ucfg = json.loads((C.UNIT / "config" / "inputs_b.json").read_text())
    blobs = {}
    for k in ("v6", "t196", "labels"):
        spec = ucfg["local"][k]
        b = git("show", spec["git"]) if "git" in spec else Path(spec["path"]).read_bytes()
        if C.sha_b(b) != spec["sha256"]:
            raise SystemExit(f"G1: {k}")
        blobs[k] = b
    rel, reln = git("show", f"{C.RELEASE[0]}:{C.RELEASE[1]}"), git("show", f"{RELEASE_NM[1]}:{RELEASE_NM[2]}")
    if b"Released:" not in rel or C.FINAL[1].encode() not in rel or b"S4 release" not in reln:
        raise SystemExit("refused: the release records do not release the FINAL record and S1-S5 with S4")
    gname, gsha = gate_record.rsplit("@", 1)
    gb = (HERE / "results" / gname).read_bytes()
    gj = json.loads(gb)
    if C.sha_b(gb) != gsha or not gj.get("pass") or gj.get("mode") != "gate" or gj["meta"].get("core_sha256") != C.sha_b((HERE / "run_fprsweep_507.py").read_bytes()):
        raise SystemExit("refused: a passing gate record of the current core is required")
    fb = git("show", f"{C.FINAL[0]}:{C.FINAL[1]}")
    if C.sha_b(fb) != C.FINAL[2]:
        raise SystemExit("FINAL record sha256")
    blobs["final"] = fb
    rb = (HERE / READ455[0]).read_bytes()
    if C.sha_b(rb) != READ455[1]:
        raise SystemExit("455 readings sha256")
    blobs["read455"] = rb
    ob = git("show", cfg["own_a_record"]["git"])
    if C.sha_b(ob) != cfg["own_a_record"]["sha256"]:
        raise SystemExit("G1: own_a record")
    own = json.loads(ob)
    for p in cfg["probes"]:
        if p["old"]["kind"] == "own_a_R":
            p["old"]["files"] = own["saved"]["tight"]["avg"]
        if p["old"]["kind"] == "own_a_C":
            e = own["saved"]["cmd"]["cmdA"]["rows|own_a_tight"]
            p["old"] |= {"path": e["path"], "sha256": e["sha256"]}
    c311 = git("show", f"{C.CELLS311[0]}:{C.CELLS311[1]}")
    if C.sha_b(c311) != C.CELLS311[2]:
        raise SystemExit("+311 list sha256")
    cfg["strategy_of_cell"] = ({c["key"]: (c.get("approach_chars") or 0) > 0 for c in json.loads(c311)["cells"]}
                               | {c["key"]: (c.get("approach_chars") or 0) > 0 for c in json.loads(blobs["t196"])["cells"]})
    cfg["arms455"] = C.arms_from_spec(json.loads(git("show", f"{C.SPEC[0]}:{C.SPEC[1]}")), git)
    mb = (HERE / "config" / "membership710.json").read_bytes()
    mem = json.loads(mb)
    arms_nm = []
    for folder, fi in sorted(mem["folders"].items()):
        files = {}
        for kind, h in fi["manifests"].items():
            b = CP.read(f"{MAN710}/{folder}__{kind}.json")
            if C.sha_b(b) != h:
                raise SystemExit(f"{folder} {kind}: manifest copy differs from membership710's sha")
            for fn, s in json.loads(b)["files"].items():
                files[f"{kind}/{Path(fn).name}"] = {"path": f"scaleup/{folder}/{SUB[kind]}/{Path(fn).name}", "sha256": s}
        arms_nm.append({"arm": folder, "files": files})
    cfg["arms_nm"] = arms_nm
    rr = git("show", cfg["relaunch_record"]["git"])
    if C.sha_b(rr) != cfg["relaunch_record"]["sha256"]:
        raise SystemExit("G1: relaunch record")
    cfg["relaunch_files"] = {fn: {"path": f"ref/relaunch/sol/{fn}", "sha256": json.loads(rr)["outputs"]["files"][fn]} for fn in cfg["relaunch_names"]}
    cfg["membership"] = mem
    extra = {}
    if dt_only and (repro_record or p2_relaunch or p2_reply_relaunch or famexp_relaunch or repro_probes or repro_keys):
        raise SystemExit("--dt-only takes no other relaunch or reproduction options")
    if dt_relaunch and not dt_only:
        raise SystemExit("--dt-relaunch needs --dt-only")
    if dt_only:  # the DT add-on (2e, ~11:55): the four DT designs (all calls) and their impactful-only variants; the same pools, membership and draws
        xb = git("show", f"{XR[0]}:{XR[1]}")
        if not C.sha_b(xb).startswith(XR[2]):
            raise SystemExit("FPR fork record sha256")
        blobs["xr110033"] = xb
        ub = git("show", f"{DT_UNSEEN[0]}:{DT_UNSEEN[1]}")
        if C.sha_b(ub) != DT_UNSEEN[2]:
            raise SystemExit("DT unseen-check record sha256")
        pins = {Path(k).name.removesuffix(".json"): v["sha256"] for k, v in json.loads(ub)["manifests_dt"].items()}
        xcfg = {r["label"]: r for r in json.loads(git("show", f"{XR_CFG[0]}:{XR_CFG[1]}"))["rows"] if r.get("source") == "dt"}
        cfg["dt_runs"], zero = {}, {}
        for a in cfg["arms455"] + cfg["arms_nm"]:
            mb_ = CP.read(f"{MANDT}/{a['arm']}.json")
            if C.sha_b(mb_) != pins[a["arm"]]:
                raise SystemExit(f"{a['arm']}: manifest_dt.json differs from 4f's pin")
            md = json.loads(mb_)
            for d in DT_DESIGNS:
                (ent,) = [v for k, v in md["files"].items() if k.endswith(f"/DT_{d}")]
                a["files"][f"dt/DT_{d}.npz"] = {"path": ent["path"].replace("/work/", ""), "sha256": ent["sha256"]}
            cfg["dt_runs"][a["arm"]] = sorted(md["runs"])
            zero[a["arm"]] = sorted(k for k, v in md["runs"].items() if v.get("calls") == 0)  # where the manifest has calls; the job also lists runs without rows
        cfg["probes"] = [{"label": f"DT_{d}{' impactful' if imp else ''} C", "recipe": "DT", "family": f"DT {d.split('_')[0]}", "readout": "C", "agg": "max",
                          "folds": 5, "registered": "sim_minus_run2", "new": f"dt/DT_{d}.npz", "kind": "dt", "impactful_only": imp,
                          "old": {"kind": "dt", "world_files": {w: {"path": v["path"], "sha256": v["sha256"]} for w, v in xcfg[f"DT_{d} C"]["world_files"].items()}}}
                         for imp in (False, True) for d in DT_DESIGNS]
        cfg["headline_probes"] = [f"DT_{d} C" for d in DT_DESIGNS]
        extra["dt"] = {"unseen_check": f"{DT_UNSEEN[1]} @ {DT_UNSEEN[0]}", "t196_files": f"{XR_CFG[1]} @ {XR_CFG[0]}", "fpr_fork_record": f"{XR[1]} @ {XR[0]}",
                       "zero_call_runs_listed_in_manifests": {k: v for k, v in zero.items() if v}, "relaunch": "relaunch pending (needs Aswin's approval)"}
        ub2 = git("show", f"{DT_UNMAPPED[0]}:{DT_UNMAPPED[1]}")
        if C.sha_b(ub2) != DT_UNMAPPED[2]:
            raise SystemExit("DT unmapped-routing record sha256")
        um = json.loads(ub2)["unmapped_runs"]
        cfg["dt_unmapped"] = sorted(k for g in um.values() for k in g)
        cfg["dt_unmapped_reason"] = {k: v["reason"] for g in um.values() for k, v in g.items()}
        extra["dt"]["unmapped_routing"] = f"{DT_UNMAPPED[1]} @ {DT_UNMAPPED[0]} ({len(cfg['dt_unmapped'])} runs)"
        if dt_relaunch:
            rb_ = git("show", f"{DT_RL[0]}:{DT_RL[1]}")
            if C.sha_b(rb_) != DT_RL[2]:
                raise SystemExit("DT relaunch record sha256")
            rl_ = json.loads(rb_)
            cfg["dt_relaunch"] = {"files": {d: {"path": v["path"].replace("/work/", ""), "sha256": v["sha256"]} for d in DT_DESIGNS
                                            for k, v in rl_["files"].items() if k.endswith(f"/DT_{d}")}, "zero_call_runs": rl_["zero_call_runs"]}
            if sorted(cfg["dt_relaunch"]["files"]) != sorted(DT_DESIGNS):
                raise SystemExit("DT relaunch record: a design file is missing")
            extra["dt"]["relaunch"] = f"{DT_RL[1]} @ {DT_RL[0]} (manifest {rl_['manifest_sha256']})"
    if p2_relaunch:  # <local copy of /ref/relaunch/p2/manifest_p2.json>@<sha256>
        mp_, ms_ = p2_relaunch.rsplit("@", 1)
        mb2 = Path(mp_).read_bytes()
        if C.sha_b(mb2) != ms_:
            raise SystemExit("p2 relaunch manifest sha256")
        man2 = json.loads(mb2)
        for p in cfg["probes"]:
            if p["label"] in P2C:
                fn = p["new"].split("/", 1)[1]
                cfg["relaunch_files"][f"p2rl/{fn}"] = {"path": f"ref/relaunch/p2/{fn}", "sha256": man2["files"][fn]}
                p |= {"relaunch": f"p2rl/{fn}", "relaunch_gate": "p2"}
        extra["p2_relaunch_manifest"] = f"/ref/relaunch/p2/manifest_p2.json @ {ms_} (scorer 70831ab3d; record @ 8f8de784d)"
    if p2_reply_relaunch:  # <local copy of /ref/relaunch/p2/manifest_p2_reply_rows.json>@<sha256> (4f; record @ 302ea28f6); names: <probe>__reply_mean.npz
        mp_, ms_ = p2_reply_relaunch.rsplit("@", 1)
        mb4 = Path(mp_).read_bytes()
        if C.sha_b(mb4) != ms_:
            raise SystemExit("p2 reply relaunch manifest sha256")
        man4 = json.loads(mb4)
        for p in cfg["probes"]:
            if p["label"] in ("FIBS R", "harmfulness R", "refusal R", "high-stakes R"):
                fn = p["new"].split("/", 1)[1]  # p2/<probe>__reply_mean.npz, the same name in the relaunch folder (B u C use other names; the gate checks fields)
                cfg["relaunch_files"][f"p2rl/{fn}"] = {"path": f"ref/relaunch/p2/{fn}", "sha256": man4["files"][fn]}
                p |= {"relaunch": f"p2rl/{fn}", "relaunch_gate": "p2"}
        extra["p2_reply_relaunch_manifest"] = f"/ref/relaunch/p2/manifest_p2_reply_rows.json @ {ms_} (record @ 302ea28f6)"
    if repro_probes:
        cfg["repro_probes"] = repro_probes.split(",")
    if repro_keys:
        cfg["repro_keys"] = repro_keys.split(",")
    if famexp_relaunch:  # <local copy of /ref/relaunch/famexp/manifest_famexp.json>@<sha256> (4f; code @ 1e96056ab)
        mp_, ms_ = famexp_relaunch.rsplit("@", 1)
        mb3 = Path(mp_).read_bytes()
        if C.sha_b(mb3) != ms_:
            raise SystemExit("famexp relaunch manifest sha256")
        man3 = json.loads(mb3)
        for p in cfg["probes"]:
            if p["label"].startswith(("own_frac", "own_a ")):
                fn = p["new"].split("/", 1)[1]
                cfg["relaunch_files"][f"famrl/{fn}"] = {"path": f"ref/relaunch/famexp/{fn}", "sha256": man3["files"][fn]}
                p |= {"relaunch": f"famrl/{fn}", "relaunch_gate": "famexp"}
        extra["famexp_relaunch_manifest"] = f"/ref/relaunch/famexp/manifest_famexp.json @ {ms_} (4f; code, RULE and SPEND @ 1e96056ab)"
    if repro_record:
        rn, rs = repro_record.rsplit("@", 1)
        rb2 = (HERE / "results" / rn).read_bytes()
        if C.sha_b(rb2) != rs:
            raise SystemExit("repro record sha256")
        blobs["repro"] = rb2
        extra["repro_record"] = repro_record
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    meta = {"git": git("rev-parse", "HEAD").decode().strip(), "stamp": stamp, "script_sha256": C.sha_b(Path(__file__).read_bytes()),
            "core_sha256": C.sha_b((HERE / "run_fprsweep_507.py").read_bytes()), "common455_sha256": C.sha_b((HERE / "run_common455.py").read_bytes()),
            "membership_sha256": C.sha_b(mb), "gate_record": gate_record, "readings_455": f"{READ455[0]} @ {READ455[1]}",
            "release": [f"{C.RELEASE[1]} @ {C.RELEASE[0]}", f"{RELEASE_NM[2]} @ {RELEASE_NM[0]} (+ S4 release @ {RELEASE_NM[1]})", "S4r Sol landing @ 1ac2de70d"],
            "inputs": {k: C.sha_b(v) for k, v in blobs.items()}} | extra
    out = compute.remote(cfg, blobs, meta)
    blob = json.dumps(out, default=float).encode()
    dest = HERE / "results" / f"common_run710_{stamp}.json"
    dest.open("xb").write(blob)
    print(json.dumps({"record": dest.name, "sha256": C.sha_b(blob), "pass": out.get("pass"), "repro_455": out.get("repro_455_readings")}), flush=True)
