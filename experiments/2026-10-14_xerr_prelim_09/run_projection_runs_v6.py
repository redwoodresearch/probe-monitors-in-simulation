"""PROJECTION (not a measurement): U under frozen eligibility (Amendment 7b) when each task gets MORE RUNS per arm.

Per eligible probe, from v6's estimation half (xerr_v6_2026-10-14_161122.json):
  sigma_d^2  = Var over tasks of d_t = hit_sim - mean(hit_run1, hit_run2)   (task_sd of sim_minus_mean_real)
  sigma_W^2  = mean_t (hit_run2 - hit_run1)^2 / 2   (within-task run-to-run variance, from the re-execution row; includes the
               run1/run2 launch difference) = (sd_rx^2 + delta_rx^2) / 2, assuming the sim's within-task variance is the same
  sigma_B^2  = max(0, sigma_d^2 - sigma_W^2 (1/1 + 1/2))   (between-task part; current design is 1 sim run, 2 real runs)
  with n_s sim runs and n_r real runs per task: sigma_d^2(n_s, n_r) = sigma_B^2 + sigma_W^2 (1/n_s + 1/n_r)
Then the same Gaussian nested simulation as run_projection_v6.py, with v6's backed-out between-probe correlation.

  PJ_IN=<dir with the v6 record> $MODAL run --detach experiments/2026-10-14_xerr_prelim_09/run_projection_runs_v6.py
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
IN = Path(os.environ.get("PJ_IN", "/nonexistent"))
APP = "probe-shift_aswin-surya_sub-convaudit_xerrprojruns_kill-2026-11-14"
V6 = (
    "xerr_v6_2026-10-14_161122.json",
    "e789144b62f71e0fb168dc3106656c2b5f29aadafe12df0f86dd1f74441d7645",
)
ATT = 183 / 196
COHORTS = {  # estimation tasks under frozen eligibility (cohort minus v6's 93 selection tasks)
    "current_183_posthoc": 183 - 93,
    "507cells_attempt93": round(183 + 311 * ATT) - 93,
    "507cells_attempt50": round(183 + 311 * 0.50) - 93,
}
RUNS = [
    (1, 2),
    (2, 2),
    (2, 4),
    (3, 3),
    (3, 6),
]  # (sim runs, real runs) per task; (1, 2) is the current design
FUTURE, BOOT = 400, 4000

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")
if IN.exists():
    image = image.add_local_file(IN / V6[0], f"/root/in/{V6[0]}")


def _run() -> dict:
    import numpy as np

    raw = Path(f"/root/in/{V6[0]}").read_bytes()
    if hashlib.sha256(raw).hexdigest() != V6[1]:
        raise ValueError("pin")
    res = json.loads(raw)["result"]["results"]
    rng = np.random.default_rng([20261014, 8])
    out = {
        "label": "PROJECTION (Gaussian model from v6's own estimates); more runs per task; not a measurement; PRELIMINARY",
        "assumptions": [
            "within-task variance sigma_W^2 from v6's run2 - run1 rows; it includes the run1/run2 launch difference, so "
            "it may overstate pure rerun noise",
            "the sim's within-task variance equals the real arm's (no sim replicates exist to measure it)",
            "between-task variance = observed per-task variance minus the within-task part of the current design "
            "(1 sim run, 2 real runs), floored at 0 (where floored, the (1, 2) row exceeds the observed variance)",
            "runs are conditionally independent across arms and within a task given the task (repeatable launches)",
            "the between-probe correlation (v6's backed-out rho) is unchanged by extra runs",
            "U conditions on fixed thresholds: more honest runs per task make the threshold more precise and lift k at "
            "1% above the thin-pool flag, which this projection does not show",
            "truth T0: true Delta = 0; T1: true Delta = v6's observed per-probe Delta (an observed-effect scenario)",
            "+6 probes: SD and within/between split = the current probes' means, same rho",
        ],
        "rows": {},
    }

    def plug(D):
        return np.sqrt(D.mean(-1) ** 2 + D.var(-1, ddof=1))

    def u_single(mu, L, n, k):
        fut = mu + (rng.standard_normal((FUTURE, k)) @ L.T) / np.sqrt(n)
        bs = fut[:, None, :] + (rng.standard_normal((FUTURE, BOOT, k)) @ L.T) / np.sqrt(
            n
        )
        return np.quantile(plug(bs), 0.95, axis=1)

    for ro in ("R_reply_meanpooled", "C_anycommand_max"):
        for a in ("0.01", "0.05"):
            prim = res[f"{ro}|mechanical|alpha={a}"]["primary_eligible_estimation_half"]
            blk = (prim.get("sim_minus_mean_real") or {}).get("pooled") or {}
            rx = (prim.get("reexecution_run2_minus_run1") or {}).get("pooled") or {}
            pp, pr = blk.get("per_probe"), rx.get("per_probe")
            if (
                not isinstance(pp, dict)
                or len(pp) < 2
                or not isinstance(pr, dict)
                or blk.get("bias_ci95") is None
            ):
                out["rows"][f"{ro}|{a}"] = {
                    "note": "fewer than 2 eligible probes in v6: no projection"
                }
                continue
            names = list(pp)
            sd = np.array([pp[p]["task_sd"] for p in names], float)
            dl = np.array([pp[p]["delta"] for p in names], float) / 100
            sw2 = np.array(
                [
                    # task_sd is ddof 1; mean((run2 - run1)^2) = ((n-1)/n) sd^2 + mean^2 (Codex review14)
                    (
                        ((rx["n_tasks"] - 1) / rx["n_tasks"]) * pr[p]["task_sd"] ** 2
                        + (pr[p]["delta"] / 100) ** 2
                    )
                    / 2
                    for p in names
                ],
                float,
            )
            if not (
                np.isfinite(sd).all() and (sd > 0).all() and np.isfinite(sw2).all()
            ):
                out["rows"][f"{ro}|{a}"] = {
                    "note": "non-finite or zero SD: no projection"
                }
                continue
            sb2 = np.maximum(0.0, sd**2 - sw2 * 1.5)
            n0, P = blk["n_tasks"], len(names)
            vbar = ((blk["bias_ci95"][1] - blk["bias_ci95"][0]) / 100 / 3.92) ** 2
            off = np.outer(sd, sd).sum() - (sd**2).sum()
            rho_raw = float((P**2 * n0 * vbar - (sd**2).sum()) / off)
            rho = float(np.clip(rho_raw, -1 / (P + 6 - 1) + 1e-6, 0.999))
            row = {
                "probes": names,
                "sigma_d": sd.tolist(),
                "sigma_W2_within": sw2.tolist(),
                "sigma_B2_between": sb2.tolist(),
                "between_share_of_current_variance": (sb2 / sd**2).tolist(),
                "between_floored_at_0": [bool(x == 0) for x in sb2],
                "rho_raw": rho_raw,
                "rho_used": rho,
                "projections": {},
            }
            for pname, extra in (("P_current", 0), ("P_plus_6", 6)):
                b2 = np.concatenate([sb2, np.full(extra, sb2.mean())])
                w2 = np.concatenate([sw2, np.full(extra, sw2.mean())])
                k = len(b2)
                for truth in ("T0", "T1"):
                    mu = (
                        np.zeros(k)
                        if truth == "T0"
                        else np.concatenate([dl, np.full(extra, dl.mean())])
                    )
                    cells = {}
                    for ns, nr in RUNS:
                        s_ = np.sqrt(b2 + w2 * (1 / ns + 1 / nr))
                        Sig = rho * np.outer(s_, s_)
                        np.fill_diagonal(Sig, s_**2)
                        L = np.linalg.cholesky(Sig)
                        for c, n in COHORTS.items():
                            U = u_single(mu, L, n, k)
                            cells[f"{c}|sim{ns}_real{nr}"] = {
                                "n_estimation_tasks": int(n),
                                "U_median": 100 * float(np.median(U)),
                                "U_p10_p90": [
                                    100 * float(np.quantile(U, q)) for q in (0.1, 0.9)
                                ],
                            }
                    row["projections"][f"{pname}|{truth}"] = cells
            out["rows"][f"{ro}|{a}"] = row
    return out


@app.function(image=image, cpu=4, memory=8192, timeout=900, retries=0)
def run() -> dict:
    try:
        return _run()
    except Exception as e:  # noqa: BLE001 -- numbers only
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
    dest = HERE / "results" / f"projection_runs_v6_{stamp}.json"
    dest.open("x").write(
        json.dumps(
            {"git": git("rev-parse", "HEAD"), "stamp": stamp, "result": res}, indent=1
        )
    )
    print("wrote", dest.name, "fatal" in res, flush=True)
