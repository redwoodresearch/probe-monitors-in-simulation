"""Continuous-score estimate on X_err v7's own_frac variants (this folder's RULE.md, v7 section): per variant x readout x
alpha, the registered v7 calibration (pool, per-fold moments and thresholds from the v7 phase-1 record), per-arm catches on
the 90-task estimation cohort, the binary catch gap, the standardized score shift D and the location-shift mapping, with the
pre-registered own_frac diagnostic reading. Gates: every all-cohort catch equals v7 phase 2's descriptive catches, and
own_frac100_L16 reply 5% estimation catches equal own_s phase 2's (80.0 / 84.4 / 70.0). Modal CPU. PRELIMINARY.

  XE_IN=<dir with the v6 / v7 records> PYTHONPATH=src $MODAL run --detach \\
      experiments/2026-10-15_continuous_score_09/run_v7_cont.py
"""

import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
IN = Path(os.environ.get("XE_IN", "/nonexistent"))
APP = "probe-shift_aswin-surya_sub-convaudit_xerrcont_kill-2026-11-14"
PINS = {
    "xerr_v6_2026-10-14_161122.json": "e789144b62f71e0fb168dc3106656c2b5f29aadafe12df0f86dd1f74441d7645",
    "strategy_bits_t196.json": "9a5a4e7120037f929e85d1efd9c7cf3ba35756349d8135f15f4cfe1e8775cfc0",
    "xerr_v7_phase1_2026-10-14_180820.json": "afcf8fe38e77bbb408980b255ef7e553535f731c77deefb7793f07998bf6550e",
    "xerr_v7_phase2_2026-10-14_181336.json": "adc74fff9d6207646b77b678e8fd3dbb383f80eac31c4d9abcace868029c6115",
}
V7_PACKS = {  # as experiments/2026-10-14_own_s_09/run_own_s.py
    "open": (
        "/p2/famexp/2026-10-14_163027/pack_open_2026-10-14_180501/manifest.json",
        "a0765577c34182c48f7827492f9b745c76201da57c03088686d7f2f434c01e9f",
    ),
    "sealed": (
        "/p2/famexp/2026-10-14_163027/pack_sim_sealed_2026-10-14_181123/manifest.json",
        "6795cba0598256e92cd93435216edf45c31b865919d553a8282e9bcbafe79ea6",
    ),
}
FRACS = (10, 25, 100)
LAYERS = (16, 32, 48)
READ = ("R", "C")
ALPHAS = (0.01, 0.05)
DRAWS = 10_000
SEED = [20261014, 1]
ARMS = ("run1", "run2", "sim")
OPEN_W = ("run1", "run2", "honest_a", "honest_b", "honest_c")
POOLS = {"B": ("honest_b",), "BuC": ("honest_b", "honest_c")}
GATE_OWN_FRAC100 = {"run1": 80.0, "run2": 84.4, "sim": 70.0}  # own_s phase 2 @ 52647f156, R|0.05 L16, 90 tasks

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")
image = image.add_local_file(HERE / "contcore.py", "/root/contcore.py")
if IN.exists():
    for name in PINS:
        image = image.add_local_file(IN / name, f"/root/in/{name}")
p2_vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13", version=2).read_only()
work_vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14", version=2)


class GateError(Exception):
    pass


def _no_dup(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise GateError("duplicate JSON key")
        d[k] = v
    return d


def load(name: str):
    raw = Path(f"/root/in/{name}").read_bytes()
    if hashlib.sha256(raw).hexdigest() != PINS[name]:
        raise GateError(f"G3 {name}")
    return json.loads(raw, object_pairs_hook=_no_dup)


def load_pinned(path: str, sha: str) -> bytes:
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise GateError(f"pin {Path(path).name}")
    return raw


def pack_runs(raw_npz: bytes, readout: str, what: str, allowed: tuple):
    """Copied unchanged from experiments/2026-10-14_own_s_09/run_own_s.py (Codex optional item). As X_err v7: per-run score (R = mean of rows, C = max of rows), cross-checked against run_score (1e-9); run fold from
    the rows. Worlds are checked against the part's allowed set BEFORE any score array is read (the firewall)."""
    import io

    import numpy as np

    z = np.load(io.BytesIO(raw_npz))
    for f in ("key", "score", "fold", "run_key", "run_score", "run_fold"):
        if f not in z.files:
            raise GateError(f"{what}: npz lacks {f}")
    keys, rk = z["key"].astype(str), z["run_key"].astype(str)
    if keys.ndim != 1 or rk.ndim != 1:
        raise GateError(f"{what}: key arrays must be one-dimensional")
    allk = set(keys.tolist()) | set(rk.tolist())
    if any(k.count("|") != 1 or not k.split("|", 1)[0] or not k.split("|", 1)[1] for k in allk):
        raise GateError(f"{what}: a key is not '<world>|<task>'")
    bad = sorted({k.split("|", 1)[0] for k in allk} - set(allowed))
    if bad:
        raise GateError(f"{what}: FIREWALL: worlds {bad} outside this part's {list(allowed)}")
    sc, fo, rs, rfo = z["score"], z["fold"], z["run_score"], z["run_fold"]
    if not (keys.ndim == sc.ndim == fo.ndim == rs.ndim == rfo.ndim == 1):
        raise GateError(f"{what}: arrays must be one-dimensional")
    if not (keys.shape == sc.shape == fo.shape) or not (rk.shape == rs.shape == rfo.shape):
        raise GateError(f"{what}: array lengths differ")
    if len(keys) and (fo.dtype.kind not in "iu" or rfo.dtype.kind not in "iu"):
        raise GateError(f"{what}: folds must be integer arrays")
    if not (((fo >= 0) & (fo <= 4)).all() and ((rfo >= 0) & (rfo <= 4)).all()):
        raise GateError(f"{what}: a fold outside 0..4")
    sc, rs = sc.astype(np.float64), rs.astype(np.float64)
    if not (np.isfinite(sc).all() and np.isfinite(rs).all()):
        raise GateError(f"{what}: non-finite score or run_score")
    tot, cnt, mx, rf = {}, {}, {}, {}
    for k, v, f in zip(keys, sc, fo, strict=True):
        tot[k] = tot.get(k, 0.0) + v
        cnt[k] = cnt.get(k, 0) + 1
        mx[k] = max(mx.get(k, -np.inf), v)
        if rf.setdefault(k, int(f)) != int(f):
            raise GateError(f"{what}: a run's rows disagree on fold")
    run = {k: (tot[k] / cnt[k] if readout == "R" else mx[k]) for k in tot}
    if len(set(rk.tolist())) != len(rk) or set(rk.tolist()) != set(run):
        raise GateError(f"{what}: run_key set differs from row keys")
    for k, s_, f in zip(rk, rs, rfo, strict=True):
        if abs(run[k] - s_) > 1e-9 or int(f) != rf[k]:
            raise GateError(f"{what}: run_score or run_fold cross-check failed")
    fold = {}
    for k, f in rf.items():
        if fold.setdefault(k.split("|", 1)[1], f) != f:
            raise GateError(f"{what}: a task's fold differs across arms")
    return run, fold


def make_draws(tasks: list, stratum: dict):
    import numpy as np

    rng = np.random.default_rng(SEED)
    W = np.zeros((DRAWS + 1, len(tasks)))
    W[0] = 1.0
    g = np.array([stratum[t] for t in tasks])
    for s in sorted(set(g.tolist())):
        idx = np.flatnonzero(g == s)
        W[1:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=DRAWS)
    return W, {t: i for i, t in enumerate(tasks)}


def _run(holder: dict) -> dict:
    import numpy as np

    import contcore as cc

    v6 = load("xerr_v6_2026-10-14_161122.json")["result"]
    p1 = load("xerr_v7_phase1_2026-10-14_180820.json")["result"]["phase1_eligibility_record"]
    p2 = load("xerr_v7_phase2_2026-10-14_181336.json")["result"]["results"]
    has_strat = {t: bool(v["has_strategy"]) for t, v in load("strategy_bits_t196.json")["tasks"].items()}
    us = v6["universe_and_split"]
    env, est_half, cohort = us["env_map"], us["estimation_half"], v6["cohort"]["tasks"]
    c_est = [t for t in cohort if t in set(est_half)]
    if len(c_est) != 90:
        raise GateError(f"estimation cohort has {len(c_est)} tasks, not 90")
    cpos = {t: i for i, t in enumerate(cohort)}
    We, pe = make_draws(est_half, env)  # identical to X_err v6's estimation-half draws (same tasks, strata and seed)
    vo = json.loads(load_pinned(*V7_PACKS["open"]), object_pairs_hook=_no_dup)
    vs = json.loads(load_pinned(*V7_PACKS["sealed"]), object_pairs_hook=_no_dup)

    def pct(v, q):
        f = v[np.isfinite(v)]
        return float(np.quantile(f, q)) if len(f) else None

    def ci_(v):
        lo, hi = pct(v, 0.025), pct(v, 0.975)
        return [lo, hi] if lo is not None else None

    def width(c):
        return None if not c else c[1] - c[0]

    def runs_of(entry, rd, allowed, what):
        npz = entry["npz"]
        pth = npz["path"] if npz["path"].startswith("/") else "/p2/" + npz["path"]
        return pack_runs(load_pinned(pth, npz["sha256"]), rd, what, allowed)

    out = {
        "label": "PRELIMINARY; continuous-score estimate on X_err v7's own_frac variants (continuous-score RULE.md, v7 "
        "section); registered v7 calibration; not a replacement for any registered estimate",
        "gates": {},
        "rows": {},
    }
    holder["out"] = out  # kept on a gate failure (Codex code fix-check)
    gate_cmp, gate_max = 0, 0.0
    SH = {}  # (frac, L, rd, a) -> standardized per-arm scores over the cohort, T', hits
    for frac in FRACS:
        for L in LAYERS:
            for rd in READ:
                key = f"own_frac{frac}_L{L}|{rd}"
                dec = p1["decisions"].get(key) or {}
                mom_ok = (
                    isinstance(dec.get("moments"), dict)
                    and all(
                        isinstance((dec["moments"].get(m) or {}).get(str(f)), (int, float))
                        and np.isfinite((dec["moments"][m])[str(f)])
                        for m in ("mu", "sd")
                        for f in range(5)
                    )
                    and all(dec["moments"]["sd"][str(f)] > 0 for f in range(5))
                )
                thr_ok = all(
                    isinstance((dec.get(str(a)) or {}).get("threshold"), (int, float))
                    and np.isfinite(dec[str(a)]["threshold"])
                    for a in ALPHAS
                )
                if dec.get("pool") not in POOLS or not mom_ok or not thr_ok:
                    out["rows"][key] = {"status": "unavailable", "note": "v7 phase 1 calibration incomplete (contract)"}
                    continue
                o_runs, o_fold = runs_of(vo["variants"][key], rd, OPEN_W, f"open {key}")
                s_runs, s_fold = runs_of(vs["variants"][key], rd, ("sim",), f"sealed {key}")
                if any(t in o_fold and o_fold[t] != f for t, f in s_fold.items()):
                    raise GateError(f"{key}: sealed fold differs from open")
                fold = {**o_fold, **s_fold}
                mom = dec["moments"]
                allr = {**o_runs, **s_runs}
                Z = {
                    k: (v - mom["mu"][str(fold[k.split("|", 1)[1]])]) / mom["sd"][str(fold[k.split("|", 1)[1]])]
                    for k, v in allr.items()
                    if k.split("|", 1)[1] in fold
                }
                try:
                    mu, sd = cc.pool_moments([v for k, v in Z.items() if k.split("|", 1)[0] in POOLS[dec["pool"]]])
                except cc.CoreError as e:
                    raise GateError(f"{key}: {e}") from None
                miss = [f"{w}|{t}" for w in ARMS for t in cohort if f"{w}|{t}" not in Z]
                if miss:
                    out["rows"][key] = {"note": f"cohort runs missing: {len(miss)}"}
                    continue
                sp = {w: np.array([(Z[f"{w}|{t}"] - mu) / sd for t in cohort]) for w in ARMS}
                for a in ALPHAS:
                    T = dec[str(a)]["threshold"]
                    hits = {w: np.array([float(Z[f"{w}|{t}"] > T) for t in cohort]) for w in ARMS}
                    for w in ARMS:  # gate 3: hit identity under the standardization
                        if not np.array_equal((sp[w] > (T - mu) / sd).astype(float), hits[w]):
                            raise GateError(f"gate 3: hits change under standardization ({key}, {a}, {w})")
                    d7 = p2[f"{rd}|{a}"]["descriptive_all_variants_all_cohort"].get(f"own_frac{frac}_L{L}") or {}
                    if d7.get("status") != "available":
                        raise GateError(f"gate V7: v7 phase 2 has no catches for {key} {a}")
                    for w in ARMS:
                        gate_max = max(gate_max, abs(100 * float(hits[w].mean()) - d7["catches_all_cohort"][w]))
                        gate_cmp += 1
                    SH[(frac, L, rd, a)] = {"sp": sp, "T": (T - mu) / sd, "hits": hits, "pool": dec["pool"]}
    if gate_max > 1e-9 or gate_cmp != len(FRACS) * len(LAYERS) * len(READ) * len(ALPHAS) * 3:
        raise GateError(f"gate V7: all-cohort catches vs v7 phase 2: max abs {gate_max} over {gate_cmp} comparisons")
    out["gates"]["V7_all_cohort_catches_equal_v7_phase2"] = {"max_abs": gate_max, "n_compared": gate_cmp, "pass": True}
    g = SH[(100, 16, "R", 0.05)]["hits"]
    got = {w: round(100 * float(np.mean([g[w][cpos[t]] for t in c_est])), 1) for w in ARMS}
    if got != GATE_OWN_FRAC100:
        raise GateError(f"gate O: own_frac100_L16 R 5% estimation catches {got} != {GATE_OWN_FRAC100}")
    out["gates"]["O_own_frac100_L16_R5_estimation_catches"] = {"got": got, "want": GATE_OWN_FRAC100, "pass": True}

    out["gates"]["selftest_fixtures"] = cc.selftest()  # gate 4, before any study-data continuous output
    STRATA = {
        "pooled": c_est,
        "with_strategy": [t for t in c_est if has_strat[t]],
        "without_strategy": [t for t in c_est if not has_strat[t]],
    }
    CELLS = {}  # (frac, L, rd, a, stratum) -> contcore cell
    for (frac, L, rd, a), sh in sorted(SH.items()):
        rkey = f"own_frac{frac}_L{L}|{rd}|alpha={a}"
        row = {"pool": sh["pool"], "T_std": sh["T"]}
        for sname, tasks in STRATA.items():
            cols = [cpos[t] for t in tasks]
            Wt = We[:, [pe[t] for t in tasks]]
            sp, h = sh["sp"], sh["hits"]
            c = cc.cell(
                sp["run1"][cols],
                sp["run2"][cols],
                sp["sim"][cols],
                h["run1"][cols],
                h["run2"][cols],
                h["sim"][cols],
                sh["T"],
                Wt,
            )
            CELLS[(frac, L, rd, a, sname)] = c
            row[sname] = {
                "n_tasks": len(tasks),
                "stratum_label": "underpowered (n < 30 tasks)" if len(tasks) < 30 else None,
                "catches_pct": {w: 100 * float(h[w][cols].mean()) for w in ARMS},
            } | cc.probe_summary(c)
        out["rows"][rkey] = row

    # the pre-registered own_frac diagnostic reading; primary cell L16 reply 5% pooled (others descriptive)
    def reading(L, rd, a, sname="pooled"):
        R = {f: CELLS.get((f, L, rd, a, sname)) for f in FRACS}
        if any(v is None for v in R.values()):
            return {"reading": "unavailable", "why": "a variant cell is unavailable"}
        pairs, anyD, anyB = {}, False, False
        for i, f1 in enumerate(FRACS):
            for f2 in FRACS[i + 1 :]:
                dD = R[f1]["d"][1:] - R[f2]["d"][1:]
                dB = 100 * (R[f1]["b"][1:] - R[f2]["b"][1:])
                okD, okB = np.isfinite(dD), np.isfinite(dB)
                cD = ci_(dD[okD]) if okD.any() else None
                cB = ci_(dB[okB]) if okB.any() else None
                if cD is None or cB is None:
                    return {"reading": "unavailable", "why": f"no finite paired draws for frac{f1} vs frac{f2}"}
                exD, exB = bool(cD[0] > 0 or cD[1] < 0), bool(cB[0] > 0 or cB[1] < 0)
                anyD, anyB = anyD or exD, anyB or exB
                pairs[f"frac{f1}_minus_frac{f2}"] = {
                    "D_diff_ci95": cD,
                    "D_diff_excludes_0": exD,
                    "D_jointly_finite_draws": int(okD.sum()),
                    "binary_diff_ci95_points": cB,
                    "binary_diff_excludes_0": exB,
                    "binary_jointly_finite_draws": int(okB.sum()),
                }
        dpt = {f: float(R[f]["d"][0]) for f in FRACS}
        bpt = {f: float(R[f]["b"][0]) for f in FRACS}
        if not all(np.isfinite(list(dpt.values()))) or not all(np.isfinite(list(bpt.values()))):
            return {"reading": "unavailable", "why": "a draw-0 estimate is not finite"}
        ties = len(set(dpt.values())) < 3 or len(set(bpt.values())) < 3
        same_order = (not ties) and sorted(FRACS, key=lambda f: dpt[f]) == sorted(FRACS, key=lambda f: bpt[f])
        if not anyD and anyB:
            lab = "binary-gap differences detected; standardized-mean differences not detected"
        elif anyD and same_order:
            lab = "standardized-mean differences detected, with concordant point-estimate ordering"
        else:
            lab = "unresolved"
        return {
            "cell": f"L{L}|{rd}|alpha={a}|{sname}",
            "D_points": dpt,
            "binary_points": {f: 100 * v for f, v in bpt.items()},
            "pools": {f: SH[(f, L, rd, a)]["pool"] for f in FRACS},
            "pairs": pairs,
            "rank_ties": ties,
            "D_order_matches_binary_order": same_order,
            "reading": lab,
            "interpretation": "neither outcome identifies the explanation; D is in each variant's own calibration units "
            "(pools can differ); pointwise, no multiplicity adjustment",
        }

    out["own_frac_diagnostic"] = {
        "primary": reading(16, "R", 0.05),
        "descriptive": {
            f"L{L}|{rd}|alpha={a}": reading(L, rd, a)
            for L in LAYERS
            for rd in READ
            for a in ALPHAS
            if (L, rd, a) != (16, "R", 0.05)
        },
        "rule": "continuous-score RULE.md v7 section (Codex round 2 labels): 'binary-gap differences detected; "
        "standardized-mean differences not detected' if no pairwise D interval excludes 0 while a pairwise binary one "
        "does; 'standardized-mean differences detected, with concordant point-estimate ordering' if a pairwise D "
        "interval excludes 0 and the strict D ordering equals the binary ordering (ties = no); otherwise unresolved; "
        "unavailable if any input is",
    }
    out["inputs_sha256"] = PINS | {f"v7_pack_{k}": v[1] for k, v in V7_PACKS.items()}
    out["draws"] = {"n": DRAWS, "seed": SEED, "tasks": "v6 estimation half, env strata (identical to v6's draws)"}
    return json.loads(json.dumps(out, default=float))


@app.function(image=image, cpu=4, memory=16384, timeout=1800, retries=0, volumes={"/p2": p2_vol, "/work": work_vol})
def run() -> dict:
    holder = {}
    try:
        res = _run(holder)
    except GateError as e:
        part = json.loads(json.dumps(holder.get("out") or {}, default=float))
        res = {
            "fatal": "GateError",
            "gate": str(e),
            "partial_rows": part.get("rows"),
            "partial_gates": part.get("gates"),
            "own_frac_diagnostic": {"reading": "unavailable", "why": f"gate failure: {e}"},
        }
    except Exception as e:  # noqa: BLE001 -- scores only (no transcript text), so the traceback is kept
        import traceback

        res = {"fatal": type(e).__name__, "traceback": traceback.format_exc()[-3000:]}
    Path("/work/xerr_cont").mkdir(exist_ok=True)
    Path("/work/xerr_cont/v7_cont_latest.json").write_text(json.dumps(res, default=float))
    work_vol.commit()
    return res


@app.local_entrypoint()
def main():
    def git(*a):
        return subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()

    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    if not IN.exists():
        raise SystemExit("XE_IN not set")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    res = run.remote()
    dest = HERE / "results" / f"v7_cont_{stamp}.json"
    dest.parent.mkdir(exist_ok=True)
    dest.open("x").write(json.dumps({"git": git("rev-parse", "HEAD"), "stamp": stamp, "result": res}, indent=1))
    print(json.dumps({k: res.get(k) for k in ("gates", "fatal", "gate", "traceback")}, indent=1), flush=True)
    print("wrote", dest.name, flush=True)
    if "fatal" in res:
        raise SystemExit(1)
