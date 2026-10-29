"""RULE.md here: per-probe and joint sign-flip tests of run 2 − run 1 on corrected B. Modal CPU on committed records only.

  LFX_IN=<dir with the three records + labels.json> PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 \\
      modal run experiments/2026-10-14_launch_effect_joint_09/run_launch_joint.py
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
IN = Path(os.environ.get("LFX_IN", "/nonexistent"))
APP = "probe-shift_aswin-surya_sub-convaudit_launchjoint_kill-2026-11-14"
PINS = {
    "scores_2026-10-14_130313.json": "3c6ed0e542373109",
    "fibs_scores_2026-10-14_124948.json": "366d594cd719d52e",
    "correctedb_sweep_125005.json": "877429fd302114d4",
    "labels.json": "d00e88773fc2a71b",
}
DRAWS = 100_000
ALPHAS = ("0.05", "0.01")

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")
if IN.exists():
    for name in PINS:
        image = image.add_local_file(IN / name, f"/root/in/{name}")


def load(name: str):
    raw = Path(f"/root/in/{name}").read_bytes()
    if hashlib.sha256(raw).hexdigest()[:16] != PINS[name]:
        raise ValueError(f"pin mismatch: {name}")
    return json.loads(raw)


def thresholds(honest: list) -> dict:
    """Spec §3: k = floor(alpha n + 1e-9), the (k+1)-th largest, caught when strictly above."""
    import numpy as np

    s = np.sort(np.asarray(honest))[::-1]
    return {
        a: float(s[min(int(np.floor(float(a) * len(s) + 1e-9)), len(s) - 1)])
        for a in ALPHAS
    }


def _run() -> dict:
    import numpy as np

    rh, fb, sw = (
        load("scores_2026-10-14_130313.json"),
        load("fibs_scores_2026-10-14_124948.json"),
        load("correctedb_sweep_125005.json"),
    )
    labels = load("labels.json")["labels"]
    fam, sec = {}, {}
    for name, rec in (
        ("harmfulness", rh["probes"]["harmfulness"]),
        ("refusal", rh["probes"]["refusal"]),
        ("fibs", fb),
    ):
        for ro, target in (("reply_mean", fam), ("command", sec)):
            st = rec["stats"][ro]
            if sorted(st["pool"]) != ["honest_b", "honest_c"]:
                raise ValueError(f"{name}/{ro}: pool is not honest B u C")
            target[f"{name}|{ro}"] = (
                rec["run_scores"][ro],
                {a: float(st["thresholds"][a]) for a in ALPHAS},
            )
    for ro, key in (
        ("averaged_reply", "sol_a|averaged_reply"),
        ("command_call_end", "sol_a|command"),
    ):
        rm = sw["sweep"][ro]["run_means"]
        hon = [
            v for k, v in rm.items() if k.split("|", 1)[0] in ("honest_b", "honest_c")
        ]
        fam[key] = (rm, thresholds(hon))
    tasks = None
    for scores, _ in fam.values():
        have = {k.split("|", 1)[1] for k in scores if k.startswith("run1|")} & {
            k.split("|", 1)[1] for k in scores if k.startswith("run2|")
        }
        tasks = have if tasks is None else tasks & have
    cohort = sorted(
        t
        for t in tasks
        if all(labels.get(f"{w}|{t}") == "attempted" for w in ("run1", "run2", "sim"))
    )
    rng = np.random.default_rng([20261014, 2])
    flips = rng.choice(np.array([-1.0, 1.0]), size=(DRAWS, len(cohort)))

    def series(group):
        out = {}
        for name, (scores, thr) in group.items():
            hon = np.array(
                [
                    v
                    for k, v in scores.items()
                    if k.split("|", 1)[0] in ("honest_b", "honest_c")
                ]
            )
            sd = float(np.std(hon, ddof=1))
            r1 = np.array([scores[f"run1|{t}"] for t in cohort])
            r2 = np.array([scores[f"run2|{t}"] for t in cohort])
            out[name] = {"d": (r2 - r1) / sd} | {
                f"h{a}": (r2 > thr[a]).astype(float) - (r1 > thr[a]).astype(float)
                for a in ALPHAS
            }
            out[name]["catch"] = {
                a: {
                    "run1": 100 * float((r1 > thr[a]).mean()),
                    "run2": 100 * float((r2 > thr[a]).mean()),
                }
                for a in ALPHAS
            }
            out[name]["thresholds"] = thr
            out[name]["honest_sd"] = sd
        return out

    def test(x):
        obs = float(x.mean())
        null = flips @ x / len(x)
        return (
            obs,
            (1 + int((np.abs(null) >= abs(obs) - 1e-12).sum())) / (1 + DRAWS),
            null,
        )

    res = {
        "cohort_tasks": len(cohort),
        "draws": DRAWS,
        "per_probe": {},
        "secondary_command": {},
        "joint": {},
    }
    fam_s, sec_s = series(fam), series(sec)
    nulls = {"d": [], "h0.05": []}
    obs_all = {"d": [], "h0.05": []}
    for group, s, dest in (
        (fam, fam_s, "per_probe"),
        (sec, sec_s, "secondary_command"),
    ):
        for name in group:
            row = {
                "catch_pct": s[name]["catch"],
                "thresholds": s[name]["thresholds"],
                "honest_sd": s[name]["honest_sd"],
            }
            for m in ("d", *(f"h{a}" for a in ALPHAS)):
                o, p, null = test(s[name][m])
                row[m] = {"mean": o * (100 if m != "d" else 1), "p_two_sided": p}
                if dest == "per_probe" and m in nulls:
                    nulls[m].append(null)
                    obs_all[m].append(o)
            res[dest][name] = row
    names = list(fam)
    for m in (
        "d",
        *(f"h{a}" for a in ALPHAS),
    ):  # Holm across the five family probes, per statistic
        ps = sorted((res["per_probe"][n][m]["p_two_sided"], n) for n in names)
        run_max = 0.0
        for i, (pv, n) in enumerate(ps):
            run_max = max(run_max, min(1.0, (len(ps) - i) * pv))
            res["per_probe"][n][m]["p_holm"] = run_max
    for m in ("d", "h0.05"):
        N = np.stack(nulls[m])  # probes x draws
        O = np.array(obs_all[m])
        t_dir, t_dir_null = float(O.mean()), N.mean(0)
        t_max, t_max_null = float(np.abs(O).max()), np.abs(N).max(0)
        res["joint"][m] = {
            "T_dir": t_dir * (100 if m != "d" else 1),
            "T_dir_p": (1 + int((np.abs(t_dir_null) >= abs(t_dir) - 1e-12).sum()))
            / (1 + DRAWS),
            "T_max": t_max * (100 if m != "d" else 1),
            "T_max_p": (1 + int((t_max_null >= t_max - 1e-12).sum())) / (1 + DRAWS),
            "probes": names,
        }
    return res


@app.function(image=image, cpu=2, memory=8192, timeout=900, retries=0)
def run() -> dict:
    try:
        return _run()
    except Exception as e:  # noqa: BLE001 -- the error type only, never text
        return {"fatal": type(e).__name__}


@app.local_entrypoint()
def main():
    def git(*a):
        return subprocess.run(
            ["git", *a], capture_output=True, text=True, check=True, cwd=REPO
        ).stdout.strip()

    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    if not IN.exists():
        raise SystemExit("LFX_IN not set")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    res = run.remote()
    out = {
        "label": "RULE.md (launch effect, joint sign-flip); corrected B; PRELIMINARY",
        "git": git("rev-parse", "HEAD"),
        "stamp": stamp,
        "result": res,
    }
    dest = HERE / "results" / f"launch_joint_{stamp}.json"
    dest.parent.mkdir(exist_ok=True)
    dest.open("x").write(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)
