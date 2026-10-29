"""PROJECTION using a pilot variance estimate (not a measurement): run_projection_runs_v6.py's model with the MEASURED
per-probe sim within-task variance from run A (run_sim_rerunvar.py's record) in place of "sim = real". RULE.md here.

  sigma_W,real^2 = v6's run2 - run1 rows (as before); sigma_W,sim^2 = the rerunvar record's sigma2_W_sim point
  sigma_B^2 = max(0, sigma_d^2 - sigma_W,sim^2 / 1 - sigma_W,real^2 / 2)   (current design: 1 sim run, 2 real runs)
  sigma_d^2(n_s, n_r) = sigma_B^2 + sigma_W,sim^2 / n_s + sigma_W,real^2 / n_r; rows (1, 2) and (2, 2)

  PJ_IN=<dir with the v6 record and the rerunvar record> RV_NAME=<rerunvar file> RV_SHA=<its sha256> \\
      $MODAL run --detach experiments/2026-10-14_sim_rerun_variance_09/run_projection_runs_measured.py
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
RVN, RVS = os.environ.get("RV_NAME", ""), os.environ.get("RV_SHA", "")
APP = "probe-shift_aswin-surya_sub-convaudit_xerrprojmeas_kill-2026-11-14"
V6 = (
    "xerr_v6_2026-10-14_161122.json",
    "e789144b62f71e0fb168dc3106656c2b5f29aadafe12df0f86dd1f74441d7645",
)
ATT = 183 / 196
COHORTS = {  # estimation tasks under frozen eligibility (cohort minus v6's 93 selection tasks), as run_projection_runs_v6
    "current_183_posthoc": 183 - 93,
    "507cells_attempt93": round(183 + 311 * ATT) - 93,
    "507cells_attempt50": round(183 + 311 * 0.50) - 93,
}
RUNS = [(1, 2), (2, 2)]
FUTURE, BOOT = 400, 4000

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")
if IN.exists() and RVN:
    image = image.add_local_file(IN / V6[0], f"/root/in/{V6[0]}").add_local_file(
        IN / RVN, f"/root/in/{RVN}"
    )


def _run(rv_name: str, rv_sha: str) -> dict:
    import numpy as np

    raw = Path(f"/root/in/{V6[0]}").read_bytes()
    rraw = Path(f"/root/in/{rv_name}").read_bytes()
    if (
        hashlib.sha256(raw).hexdigest() != V6[1]
        or hashlib.sha256(rraw).hexdigest() != rv_sha
    ):
        raise ValueError("pin")
    res = json.loads(raw)["result"]["results"]
    rv = json.loads(rraw)
    rv = rv.get("result", rv)
    if rv.get("fatal") or not rv["gates"]["V"].get("pass"):
        raise ValueError("rerunvar record is fatal or its gate V did not pass")
    test = rv["meta"]["mode"] == "test"
    rng = np.random.default_rng([20261014, 9])
    out = {
        "label": "PROJECTION using a pilot variance estimate (n ~ 19 paired tasks); not a measurement; PRELIMINARY"
        + ("; PIPELINE TEST INPUT (sim epoch 1 stood in for epoch 2)" if test else ""),
        "rerunvar_record": {"name": rv_name, "sha256": rv_sha, "label": rv["label"]},
        "assumptions": [
            (
                "sigma_W,sim^2 = run A's measured point (pilot); sigma_W,real^2 = v6's run2 - run1 rows, which include the "
                "run1/run2 launch difference"
            ),
            (
                "between-task variance = observed per-task variance minus the current design's within-task part, floored "
                "at 0 (where floored, the (1, 2) row exceeds the observed variance)"
            ),
            "runs are conditionally independent given the task; v6's backed-out between-probe rho is unchanged",
            "U conditions on fixed thresholds; truth T0: true Delta = 0; T1: true Delta = v6's observed per-probe Delta",
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
            meas = rv["rows"][f"{ro}|alpha={a}"]
            sd = np.array([pp[p]["task_sd"] for p in names], float)
            dl = np.array([pp[p]["delta"] for p in names], float) / 100
            sw2_real = np.array(
                [
                    (
                        ((rx["n_tasks"] - 1) / rx["n_tasks"]) * pr[p]["task_sd"] ** 2
                        + (pr[p]["delta"] / 100) ** 2
                    )
                    / 2
                    for p in names
                ],
                float,
            )
            sw2_sim = np.array([meas[p]["sigma2_W_sim"]["point"] for p in names], float)
            if not (
                np.isfinite(sd).all()
                and (sd > 0).all()
                and np.isfinite(sw2_real).all()
                and np.isfinite(sw2_sim).all()
            ):
                out["rows"][f"{ro}|{a}"] = {
                    "note": "non-finite or zero SD, or no measured sim variance: no projection"
                }
                continue
            sb2 = np.maximum(0.0, sd**2 - sw2_sim - sw2_real / 2)
            n0, P = blk["n_tasks"], len(names)
            vbar = ((blk["bias_ci95"][1] - blk["bias_ci95"][0]) / 100 / 3.92) ** 2
            off = np.outer(sd, sd).sum() - (sd**2).sum()
            rho_raw = float((P**2 * n0 * vbar - (sd**2).sum()) / off)
            rho = float(
                np.clip(rho_raw, -1 / (P + 6 - 1) + 1e-6, 0.999)
            )  # as run_projection_runs_v6
            row = {
                "probes": names,
                "sigma_d": sd.tolist(),
                "sigma_W2_real": sw2_real.tolist(),
                "sigma_W2_sim_measured": sw2_sim.tolist(),
                "sigma_W2_sim_measured_ci95": [
                    meas[p]["sigma2_W_sim"]["ci95"] for p in names
                ],
                "sigma_B2_between": sb2.tolist(),
                "between_floored_at_0": [bool(x == 0) for x in sb2],
                "rho_raw": rho_raw,
                "rho_used": rho,
                "projections": {},
            }
            for truth in ("T0", "T1"):
                mu = np.zeros(P) if truth == "T0" else dl
                cells = {}
                for ns, nr in RUNS:
                    s_ = np.sqrt(sb2 + sw2_sim / ns + sw2_real / nr)
                    Sig = rho * np.outer(s_, s_)
                    np.fill_diagonal(Sig, s_**2)
                    L = np.linalg.cholesky(Sig)
                    for c, n in COHORTS.items():
                        U = u_single(mu, L, n, P)
                        cells[f"{c}|sim{ns}_real{nr}"] = {
                            "n_estimation_tasks": int(n),
                            "U_median": 100 * float(np.median(U)),
                            "U_p10_p90": [
                                100 * float(np.quantile(U, q)) for q in (0.1, 0.9)
                            ],
                        }
                row["projections"][f"P_current|{truth}"] = cells
            out["rows"][f"{ro}|{a}"] = row
    return out


@app.function(image=image, cpu=4, memory=8192, timeout=900, retries=0)
def run(rv_name: str, rv_sha: str) -> dict:
    try:
        return _run(rv_name, rv_sha)
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
    if not (RVN and RVS):
        raise SystemExit("RV_NAME and RV_SHA are required")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    res = run.remote(RVN, RVS)
    dest = HERE / "results" / f"projection_runs_measured_{stamp}.json"
    dest.parent.mkdir(exist_ok=True)
    dest.open("x").write(
        json.dumps(
            {"git": git("rev-parse", "HEAD"), "stamp": stamp, "result": res}, indent=1
        )
    )
    print("wrote", dest.name, flush=True)
    if "fatal" in res:  # Codex review1: a fatal record exits nonzero
        print(
            json.dumps({k: res.get(k) for k in ("fatal", "traceback")}, indent=1),
            flush=True,
        )
        raise SystemExit(1)
