"""PROJECTION (not a measurement): the approximate upper limit U of X_err under v6's method at larger cohorts, from v6's own
per-probe task SDs, deltas and bias interval (xerr_v6_2026-10-14_161122.json). Gaussian model of the per-probe estimates with
a common between-probe correlation backed out from v6's bias interval. Nested: future sample, then bootstrap around it.

  PJ_IN=<dir with the v6 record> $MODAL run --detach experiments/2026-10-14_xerr_prelim_09/run_projection_v6.py
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
APP = "probe-shift_aswin-surya_sub-convaudit_xerrproj_kill-2026-11-14"
V6 = (
    "xerr_v6_2026-10-14_161122.json",
    "e789144b62f71e0fb168dc3106656c2b5f29aadafe12df0f86dd1f74441d7645",
)
T196_ATTEMPT = 183 / 196  # attempted in all three arms, T196 (v6 cohort)
# cohort sizes (attempted in all three arms) and what they assume; effective n counts distinct main tasks
COHORTS = {
    "current_183": 183,
    "507cells_507mains_attempt93": round(183 + 311 * T196_ATTEMPT),
    "507cells_507mains_attempt50": round(183 + 311 * 0.50),
    "1000cells_by542mains_attempt93": round(183 + 346 * T196_ATTEMPT),
    "1000cells_by542mains_attempt50": round(183 + 346 * 0.50),
    "1000cells_by1000cells_attempt93_upper": round(183 + 804 * T196_ATTEMPT),
}
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
    rng = np.random.default_rng([20261014, 7])
    out = {
        "label": "PROJECTION (Gaussian model from v6's own estimates); not a measurement; PRELIMINARY",
        "assumptions": [
            "per-probe per-task differences have v6's estimation-half task SDs; a common between-probe correlation rho "
            "is backed out from v6's bias 95% interval width (Var(mean) = (width/3.92)^2)",
            "truth T0: every probe's true Delta = 0 (consistent with v6's floored X_err); truth T1: true Delta = v6's "
            "observed per-probe Delta (an observed-effect scenario, not necessarily pessimistic)",
            "extra probes (P+6): SD = mean of current SDs, same rho, Delta = 0 (T0) or the mean observed Delta (T1)",
            "U = 95th percentile of the plug-in sqrt(mean^2 + S^2) over a bootstrap around each simulated future "
            "sample; the reported value is the median over future samples, with the 10th-90th percentile",
            "rules: single_split = v6's rule on the combined cohort (estimate on half); crossfit_A7_mean_of_folds = "
            "Amendment 7 as written (mean of the two folds' plug-ins, each on half; same eligible set assumed in both "
            "folds); frozen_eligibility_from_v6_half = eligible sets frozen from v6's 93-task eligibility half, "
            "estimation on every other cohort task (a proposal, not yet registered); n//2 approximates the attempted "
            "count in a universe-defined half",
            "1,000 cells span only 542 distinct mains: the main task is the bootstrap unit, so 542-main rows are the "
            "realistic case; the 1,000-cell row assumes cells on one main are independent (optimistic precision)",
            "attempt 93% = T196's attempted-in-all-three rate (183/196); T196 was Sol-success gated, new cells are not, "
            "so 50% is shown as a pessimistic case",
        ],
        "rows": {},
    }

    def plug(D):  # D: (..., P)
        return np.sqrt(D.mean(-1) ** 2 + D.var(-1, ddof=1))

    def u_single(mu, L, n, k):
        fut = mu + (rng.standard_normal((FUTURE, k)) @ L.T) / np.sqrt(n)
        bs = fut[:, None, :] + (rng.standard_normal((FUTURE, BOOT, k)) @ L.T) / np.sqrt(
            n
        )
        return np.quantile(plug(bs), 0.95, axis=1), plug(fut)

    def u_crossfit(
        mu, L, n_half, k
    ):  # Amendment 7 as written: mean of the two folds' plug-ins, each at n/2
        f1 = mu + (rng.standard_normal((FUTURE, k)) @ L.T) / np.sqrt(n_half)
        f2 = mu + (rng.standard_normal((FUTURE, k)) @ L.T) / np.sqrt(n_half)
        b1 = f1[:, None, :] + (rng.standard_normal((FUTURE, BOOT, k)) @ L.T) / np.sqrt(
            n_half
        )
        b2 = f2[:, None, :] + (rng.standard_normal((FUTURE, BOOT, k)) @ L.T) / np.sqrt(
            n_half
        )
        return np.quantile((plug(b1) + plug(b2)) / 2, 0.95, axis=1), (
            plug(f1) + plug(f2)
        ) / 2

    for ro in ("R_reply_meanpooled", "C_anycommand_max"):
        for a in ("0.01", "0.05"):
            prim = res[f"{ro}|mechanical|alpha={a}"]["primary_eligible_estimation_half"]
            blk = (prim.get("sim_minus_mean_real") or {}).get("pooled") or {}
            pp = blk.get("per_probe")
            if not isinstance(pp, dict) or len(pp) < 2 or blk.get("bias_ci95") is None:
                out["rows"][f"{ro}|{a}"] = {
                    "note": "fewer than 2 eligible probes in v6: no projection"
                }
                continue
            names = list(pp)
            sd = np.array(
                [
                    pp[p]["task_sd"] if pp[p]["task_sd"] is not None else np.nan
                    for p in names
                ],
                float,
            )
            dl = np.array([pp[p]["delta"] for p in names], float) / 100
            if not (np.isfinite(sd).all() and (sd > 0).all() and np.isfinite(dl).all()):
                out["rows"][f"{ro}|{a}"] = {
                    "note": "a non-finite or zero task SD: no projection"
                }
                continue
            n0, P = blk["n_tasks"], len(names)
            vbar = ((blk["bias_ci95"][1] - blk["bias_ci95"][0]) / 100 / 3.92) ** 2
            off = np.outer(sd, sd).sum() - (sd**2).sum()
            rho_raw = float((P**2 * n0 * vbar - (sd**2).sum()) / off)
            # one feasible common rho for every scenario (k up to P + 6)
            rho = float(np.clip(rho_raw, -1 / (P + 6 - 1) + 1e-6, 0.999))
            row = {
                "probes": names,
                "task_sd": sd.tolist(),
                "rho_raw_backed_out": rho_raw,
                "rho_used_all_scenarios": rho,
                "rho_clipped": rho != rho_raw,
                "v6_observed_U": blk["U_approx_upper_95"],
                "v6_n_estimation": n0,
                "projections": {},
            }
            for pname, extra in (("P_current", 0), ("P_plus_6", 6)):
                s_ = np.concatenate([sd, np.full(extra, sd.mean())])
                k = len(s_)
                Sig = rho * np.outer(s_, s_)
                np.fill_diagonal(Sig, s_**2)
                L = np.linalg.cholesky(Sig)
                for truth in ("T0", "T1"):
                    mu = (
                        np.zeros(k)
                        if truth == "T0"
                        else np.concatenate([dl, np.full(extra, dl.mean())])
                    )
                    cells = {}
                    U, pl = u_single(mu, L, n0, k)
                    cells["repeat_of_v6_design_n90"] = {
                        "n_estimation_tasks": n0,
                        "U_median": 100 * float(np.median(U)),
                        "U_p10_p90": [
                            100 * float(np.quantile(U, q)) for q in (0.1, 0.9)
                        ],
                    }
                    for c, nc in COHORTS.items():
                        for rule, nn in (
                            ("single_split", nc // 2),
                            ("crossfit_A7_mean_of_folds", nc // 2),
                            ("frozen_eligibility_from_v6_half", nc - 93),
                        ):
                            U, pl = (
                                u_crossfit(mu, L, nn, k)
                                if rule.startswith("crossfit")
                                else u_single(mu, L, nn, k)
                            )
                            cells[f"{c}|{rule}"] = {
                                "n_estimation_tasks": int(nn)
                                * (2 if rule.startswith("crossfit") else 1),
                                "U_median": 100 * float(np.median(U)),
                                "U_p10_p90": [
                                    100 * float(np.quantile(U, q)) for q in (0.1, 0.9)
                                ],
                                "plain_median": 100 * float(np.median(pl)),
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
    dest = HERE / "results" / f"projection_v6_{stamp}.json"
    dest.open("x").write(
        json.dumps(
            {"git": git("rev-parse", "HEAD"), "stamp": stamp, "result": res}, indent=1
        )
    )
    print("wrote", dest.name, "fatal" in res, flush=True)
