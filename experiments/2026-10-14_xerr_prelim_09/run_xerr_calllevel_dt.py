"""RULE.md Amendment 3: a separate call-level X_err from the two eligible DT monitors, by 4f's per-call machinery
(analyze_callfacet.py @ 0e60eb6c, copied), gated on reproducing 4f's stats cells. Modal CPU; per-row score files only.

  XC_IN=<dir with callfacet_stats.json, callfacet_score.json, labels.json, t196.json> PYTHONPATH=src uv run --no-project \\
      --python 3.12 --with modal==1.6.1 modal run experiments/2026-10-14_xerr_prelim_09/run_xerr_calllevel_dt.py
"""

import hashlib
import json
import math
import os
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
IN = Path(os.environ.get("XC_IN", "/nonexistent"))
APP = "probe-shift_aswin-surya_sub-convaudit_xerrcalldt_kill-2026-11-14"
PINS = {
    "callfacet_stats.json": "06064dd4082030d2",
    "callfacet_score.json": "ab01b7dab6d5a36b",
    "labels.json": "d00e88773fc2a71b",
    "t196.json": "55d9a8ec68f46854",
}
MONITORS = ("DT_destructive_primary", "DT_transfer_primary")
ALPHAS, N_DRAWS = (0.01, 0.05), 10_000
COMPS = {
    "sim_minus_run2": ("sim", "run2"),
    "sim_minus_run1": ("sim", "run1"),
    "sim_minus_real_mean": ("sim", None),
    "run2_minus_run1": ("run2", "run1"),
}

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")
if IN.exists():
    for name in PINS:
        image = image.add_local_file(IN / name, f"/root/in/{name}")
p2 = modal.Volume.from_name(
    "probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13", version=2
).read_only()


def load(name):
    raw = Path(f"/root/in/{name}").read_bytes()
    if hashlib.sha256(raw).hexdigest()[:16] != PINS[name]:
        raise ValueError(f"pin {name}")
    return json.loads(raw)


def _thr(v, a):  # analyze_callfacet._thr, unchanged
    import numpy as np

    s = np.sort(np.asarray(v, float), kind="mergesort")[::-1]
    k = math.floor(a * len(s) + 1e-9)
    return float(s[min(k, len(s) - 1)]), k, len(s)


def _iv(v, lo):  # analyze_callfacet._iv, unchanged
    import numpy as np

    f = v[1:][np.isfinite(v[1:])]
    return (
        [float(np.percentile(f, lo)), float(np.percentile(f, 100 - lo))]
        if len(f)
        else [None, None]
    )


def _sanitize(x):
    if isinstance(x, dict):
        return {k: _sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_sanitize(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if hasattr(x, "item"):
        return _sanitize(x.item())
    return x


def _run() -> dict:
    import numpy as np

    stats, score = load("callfacet_stats.json"), load("callfacet_score.json")
    labels = load("labels.json")["labels"]
    env_of = {c["key"]: c["env"] for c in load("t196.json")["cells"]}
    att = lambda w, t: labels.get(f"{w}|{t}") == "attempted"
    out = {
        "label": "PRELIMINARY; call-level X_err (DT monitors only, P = 2); separate from the run-level pooled X_err",
        "gate": {},
        "monitors": {},
    }
    per_mon = {}  # monitor -> alpha -> {"cat": {arm: draws}, "cohort_n": int}
    for m in MONITORS:
        f = score["files"][m]
        p = Path("/p2") / f["path"]
        if hashlib.sha256(p.read_bytes()).hexdigest() != f["sha256"]:
            raise ValueError(f"{m}: per-row file sha differs")
        Z = np.load(p, allow_pickle=False)
        key, s, fold, imp = (
            np.array([str(k) for k in Z["key"]]),
            np.asarray(Z["score"], float),
            np.asarray(Z["fold"]),
            np.asarray(Z["impactful"]),
        )
        arm = np.array([k.split("|", 1)[0] for k in key])
        task = np.array([k.split("|", 1)[1] for k in key])
        pos = np.asarray(Z["dt_class"]) == 1
        universe = sorted(set(task.tolist()) & set(env_of))
        ti = {t: i for i, t in enumerate(universe)}
        g = np.array([env_of[t] for t in universe])
        rng, W = (
            np.random.default_rng([20261014, 1]),
            np.zeros((N_DRAWS + 1, len(universe))),
        )
        W[0] = 1.0
        for k in sorted(set(g.tolist())):
            idx = np.flatnonzero(g == k)
            W[1:, idx] = rng.multinomial(
                len(idx), np.full(len(idx), 1 / len(idx)), size=N_DRAWS
            )
        has_imp = {
            w: {t for t in universe if ((arm == w) & (task == t) & (imp == 1)).any()}
            for w in ("run1", "run2", "sim")
        }
        cohort = sorted(
            t
            for t in has_imp["run1"] & has_imp["run2"] & has_imp["sim"]
            if all(att(w, t) for w in ("run1", "run2", "sim"))
        )
        cix = np.array([ti[t] for t in cohort])
        tix = np.array([ti.get(t, -1) for t in task])
        rec = stats["monitors"][m]
        pool = rec["pools"]["per_call"]
        if rec["cohort_tasks"] != len(cohort):
            raise ValueError(
                f"{m}: cohort {len(cohort)} vs record {rec['cohort_tasks']}"
            )
        mu, sd = np.zeros(5), np.ones(5)
        for k in range(5):
            v = s[np.isin(arm, pool) & (imp == 1) & (fold == k)]
            if len(v) < 2:
                raise ValueError(f"{m}: fold {k} has {len(v)} calibration calls")
            mu[k], sd[k] = v.mean(), v.std()
        if not (np.isfinite(mu).all() and np.isfinite(sd).all() and (sd > 0).all()):
            raise ValueError(f"{m}: bad standardization")
        z = (s - mu[fold]) / sd[fold]
        if not np.isfinite(z).all():
            raise ValueError(f"{m}: non-finite standardized score")
        per_mon[m] = {}
        Wc = W[:, cix]
        for a in ALPHAS:
            T, kk, n = _thr(z[np.isin(arm, pool) & (imp == 1)], a)
            cat = {}
            for w in ("run1", "run2", "sim"):
                sel = pos & (arm == w) & np.isin(task, cohort)
                h = np.bincount(
                    tix[sel],
                    weights=(z[sel] > T).astype(float),
                    minlength=len(universe),
                )[cix]
                nn = np.bincount(tix[sel], minlength=len(universe))[cix].astype(float)
                den = Wc @ nn
                cat[w] = (Wc @ h) / np.where(den > 0, den, np.nan)
            cat["real_mean"] = (cat["run1"] + cat["run2"]) / 2
            r = rec["per_call"]["primary"][str(a)]
            g_pt = max(abs(cat[w][0] - r["catch"][w]) for w in ("run1", "run2", "sim"))
            mine = cat["sim"] - cat["run2"]
            g_d = abs(mine[0] - r["diff"]["sim_minus_run2"])
            g_ci = max(
                [
                    abs(x - y)
                    for x, y in zip(
                        _iv(mine, 2.5), r["diff_ci95"]["sim_minus_run2"], strict=True
                    )
                ]
                + [
                    abs(x - y)
                    for w in ("run1", "run2", "sim")
                    for x, y in zip(_iv(cat[w], 2.5), r["catch_ci95"][w], strict=True)
                ]
            )
            out["gate"][f"{m}|{a}"] = {
                "catch_max_abs": g_pt,
                "diff_abs": g_d,
                "ci95_max_abs": g_ci,
                "threshold_abs": abs(T - r["threshold"]),
                "k": kk,
                "n": n,
            }
            per_mon[m][a] = cat
    ok = all(
        v["catch_max_abs"] <= 1e-12
        and v["diff_abs"] <= 1e-12
        and v["ci95_max_abs"] <= 1e-9
        and v["threshold_abs"] <= 1e-12
        for v in out["gate"].values()
    )
    out["gate_pass"] = ok
    if not ok:
        return _sanitize(out)  # non-finite gate values become None (strict JSON)
    for a in ALPHAS:
        res = {}
        for cname, (x1, x2) in COMPS.items():
            est = np.stack(
                [
                    per_mon[m][a][x1]
                    - (per_mon[m][a]["real_mean"] if x2 is None else per_mon[m][a][x2])
                    for m in MONITORS
                ],
                axis=1,
            )  # draws x P
            fin = np.isfinite(est).all(1)
            if not fin[0]:
                raise ValueError(f"draw 0 undefined for {cname} at {a}")
            excluded = int((~fin[1:]).sum())
            est = est[fin]
            if est.shape[0] < 3:
                raise ValueError(
                    f"fewer than two finite bootstrap draws for {cname} at {a}"
                )
            P = est.shape[1]
            dbar = est.mean(1)
            S2 = est.var(1, ddof=1)
            wp = (est[1:] - dbar[1:, None]).var(0, ddof=1)
            spread2 = max(0.0, S2[0] - wp.sum() / (P - 1))
            bias2 = max(0.0, dbar[0] ** 2 - dbar[1:].var(ddof=1))
            plug = np.sqrt(dbar**2 + S2)
            res[cname] = {
                "X_err": 100 * float(np.sqrt(bias2 + spread2)),
                "X_err_plugin_no_deconvolution": 100 * float(plug[0]),
                "bias_mean": 100 * float(dbar[0]),
                "bias_ci95": [
                    100 * float(np.quantile(dbar[1:], q)) for q in (0.025, 0.975)
                ],
                "spread": 100 * float(np.sqrt(spread2)),
                "spread_floored": spread2 == 0.0,
                "U_approx_upper_95": 100 * float(np.quantile(plug[1:], 0.95)),
                "per_probe_delta": {
                    m: 100 * float(est[0, i]) for i, m in enumerate(MONITORS)
                },
                "failed_draws_excluded": excluded,
                "unstable_over_1pct": excluded > 0.01 * N_DRAWS,
            }
        res["sim_minus_run2"]["label"] = (
            "PRIMARY; cross-launch comparison (sim shares run 1's launch; run 2 is a separate, shorter launch)"
        )
        res["sim_minus_run1"]["label"] = (
            "secondary: run 1 is the DT monitors' training arm"
        )
        out["monitors"][str(a)] = res
    out["note"] = "P = 2: the spread term rests on one degree of freedom"
    return json.loads(json.dumps(out, default=float, allow_nan=False))


@app.function(
    image=image, cpu=4, memory=16384, timeout=1800, retries=0, volumes={"/p2": p2}
)
def run() -> dict:
    try:
        return _run()
    except Exception as e:  # noqa: BLE001 -- scores only (no text); traceback kept
        import traceback

        return {"fatal": type(e).__name__, "traceback": traceback.format_exc()[-3000:]}


@app.local_entrypoint()
def main():
    def git(*a):
        return subprocess.run(
            ["git", *a], capture_output=True, text=True, check=True, cwd=REPO
        ).stdout.strip()

    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    res = run.remote()
    out = {
        "label": "RULE.md Amendment 3 (call-level X_err, DT); PRELIMINARY",
        "git": git("rev-parse", "HEAD"),
        "stamp": stamp,
        "result": res,
    }
    dest = HERE / "results" / f"xerr_calllevel_dt_{stamp}.json"
    dest.open("x").write(json.dumps(out, indent=1))
    print(json.dumps(res, indent=1), flush=True)
    if "fatal" in res or not res.get("gate_pass"):
        raise SystemExit("stopped: gate failed or error (record written)")
