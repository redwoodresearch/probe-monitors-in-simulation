"""R44 and the scoring the RULE_simtrained.md Addendum 3 analysis needs (= RULE_scaled.md Amendment 2; POST HOC).

One Modal CPU container (read-only volumes) loads the layer-30 call_end readings for: the R44 draws' training rows (from the
committed results/r44_selection_manifest.json), the label-free job's 10,973 rows, and the gate's 4,452 staged rows
(`/score/sabotage_gate_v1/rows.json`, pinned). It fits R44 for draws 0-4 with S's recipe (fit_dual copied verbatim from
run_simtrained.py), and fits draw 0 a second time. It returns the scores of R (the frozen gate), S (the sim-trained pilot,
no refit) and the five R44 draws on all those rows. Numbers only.

Reproduction gates, all checked locally before anything is written:
- R reproduces the label-free job's stored per-call and staged scores (<= 1e-6);
- S reproduces the pilot's stored per-call scores (<= 1e-6);
- R44 draw 0, fitted twice, agrees (<= 1e-6);
- the selection manifest equals a fresh selection (make_r44_selection.select).
Writes results/r44_scores_<stamp>.json and results/r44_weights_<stamp>.npz. Refuses to run unless this file, RULE_simtrained.md,
make_simtrained_rows.py, make_r44_selection.py and the selection manifest are committed unchanged at HEAD.

  uv run --no-project --python 3.12 --with "modal==1.6.1" --with numpy modal run run_r44.py --manifests <dir> --sim-votes <csv> \\
      --real-votes <csv> --flag-rows <csv> --flag-scores <json> --npz <gate_refit npz> --gate-rows <rows_v2.json> \\
      --s-weights <simtrained_weights npz> --s-scores <simtrained_scores json>
"""

import csv
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

APP = "probe-shift_aswin-surya_sub-q3r44_fit_kill-2026-11-12"
ACTS = "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08"
SCORE = "probe-shift_aswin-surya_sub-q3onpol_score_kill-2026-11-08"
OUT, LAYER, HIDDEN, CS = "sabotage_gate_v1", 30, 5120, [1e-4]
SIMTRAINED_MANIFEST_SHA = "bf75c3dfcacf5e7a4d8ede24e64229fed31c19913eaa83fc1596807c955df8d4"  # results/manifest_simtrained_s3_v1.json @ 3dbb913c
HERE = Path(__file__).parent

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2)
score = modal.Volume.from_name(SCORE, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy")


def fit_dual(K, y, w, Cs, iters=100, tol=1e-9):
    """Copied from run_persist_gate.py (from run_score_onpolicy.py, main): alpha [L, c, n], beta = Z^T alpha minimizing
    C sum w_i (log(1 + e^f) - y f) + 1/2 |beta|^2, f = K alpha, by Newton steps with backtracking; raises if not converged."""
    import torch
    import torch.nn.functional as F

    L, n, _ = K.shape
    C = torch.tensor(Cs, dtype=K.dtype)[None, :, None]
    Kb, eye = K[:, None], torch.eye(n, dtype=K.dtype)
    a = torch.zeros(L, len(Cs), n, dtype=K.dtype)

    def obj(a):
        f = (Kb @ a[..., None])[..., 0]
        return (C * w * (F.softplus(f) - y * f)).sum(-1) + 0.5 * (a * f).sum(-1)

    for _ in range(iters):
        f = (Kb @ a[..., None])[..., 0]
        p = torch.sigmoid(f)
        r = a - C * w * (y - p)
        res = r.abs().amax(-1)
        if float(res.max()) < tol:
            return a
        delta = torch.linalg.solve(eye + (C * w * p * (1 - p))[..., :, None] * Kb, r[..., None])[..., 0]
        step, j0 = torch.ones(L, len(Cs), 1, dtype=K.dtype), obj(a)
        for _ in range(40):
            worse = obj(a - step * delta) > j0 + 1e-12 * j0.abs()
            if not worse.any():
                break
            step = torch.where(worse[..., None], step / 2, step)
        a = a - step * delta
    raise RuntimeError(f"dual Newton did not converge: max residual {float(res.max()):.3g}")


@app.function(image=image, cpu=4, memory=24576, timeout=1500, retries=0, volumes={"/acts": acts.read_only(), "/score": score.read_only()})
def fit_job(rows_sha: str, draws: list, eval_locs: list, frozen: dict) -> dict:
    """draws: [[(extraction, chunk, row, y)]]; eval_locs: [(extraction, chunk, row)]; frozen: name -> (weights, mu, sd)."""
    import numpy as np
    import torch

    raw = Path(f"/score/{OUT}/rows.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != rows_sha:
        raise SystemExit("staged rows on the Volume differ from rows_v2.json")
    staged = [(r["x"], r["chunk"], int(r["row"])) for r in json.loads(raw)["rows"]]
    locs = sorted({(x, c, int(r)) for d in draws for x, c, r, _ in d} | {(x, c, int(r)) for x, c, r in eval_locs} | set(staged))
    pos = {v: i for i, v in enumerate(locs)}
    groups = {}
    for i, (x, chunk, row) in enumerate(locs):
        groups.setdefault((x, chunk), []).append((i, row))
    X = np.empty((len(locs), HIDDEN), np.float64)
    for (x, chunk), lst in groups.items():
        a = np.load(f"/acts/{x}/by_layer/L{LAYER}/{chunk}.npy")
        X[[i for i, _ in lst]] = a[[r for _, r in lst]]
    Xe = X[[pos[(x, c, int(r))] for x, c, r in eval_locs]]
    Xs = X[[pos[v] for v in staged]]
    out = {}
    for name, (w, mu, sd) in frozen.items():
        w, mu, sd = (np.array(v) for v in (w, mu, sd))
        out[name] = {"eval": (((Xe - mu) / sd) @ w).tolist(), "staged": (((Xs - mu) / sd) @ w).tolist()}

    def fit(d):
        Xt = X[[pos[(x, c, int(r))] for x, c, r, _ in d]]
        y = torch.tensor([float(v) for *_, v in d], dtype=torch.float64)
        mu, sd = Xt.mean(0), Xt.std(0)
        sd = np.where(sd < 1e-6, 1.0, sd)
        Zt = torch.from_numpy((Xt - mu) / sd)
        a = fit_dual((Zt @ Zt.T)[None], y[None, None, :], torch.ones(len(d), dtype=torch.float64), CS)
        return (a[0, 0] @ Zt).numpy(), mu, sd

    weights = {}
    for i, d in enumerate(draws):
        beta, mu, sd = fit(d)
        weights[f"R44_{i}"] = {"weights": beta.tolist(), "mu": mu.tolist(), "sd": sd.tolist()}
        out[f"R44_{i}"] = {"eval": (((Xe - mu) / sd) @ beta).tolist(), "staged": (((Xs - mu) / sd) @ beta).tolist()}
    beta2, mu2, sd2 = fit(draws[0])
    refit_diff = float(np.abs(((Xe - mu2) / sd2) @ beta2 - np.array(out["R44_0"]["eval"])).max())
    return {"scores": out, "weights": weights, "refit0_max_abs_diff": refit_diff, "chunks_read": len(groups), "n_staged": len(staged)}


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@app.local_entrypoint()
def main(manifests: str, sim_votes: str, real_votes: str, flag_rows: str, flag_scores: str, npz: str, gate_rows: str, s_weights: str, s_scores: str):
    import sys

    import numpy as np

    sys.path.insert(0, str(HERE))
    import make_r44_selection as SEL
    import make_simtrained_rows as M

    for f in (Path(__file__).name, "RULE_simtrained.md", "make_simtrained_rows.py", "make_r44_selection.py",
              "results/r44_selection_manifest.json", "results/manifest_simtrained_s3_v1.json"):
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", f], cwd=HERE, capture_output=True)
        status = subprocess.run(["git", "status", "--porcelain", "--", f], cwd=HERE, capture_output=True, text=True)
        if tracked.returncode != 0 or status.returncode != 0 or status.stdout:
            raise SystemExit(f"{f} is not committed unchanged at HEAD")
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True, check=True).stdout.strip()
    sm = HERE / "results" / "manifest_simtrained_s3_v1.json"
    if sha(sm) != SIMTRAINED_MANIFEST_SHA:
        raise SystemExit("manifest_simtrained_s3_v1.json: sha256 differs from its pin")
    m = M.manifests(manifests)
    got = {Path(p).name: M.sha(M.checked(mm, p)) for p, mm in {flag_scores: m["manifest_flagrate_s3_v1.json"], npz: m["manifest_gate_refit_v1.json"],
                                                              gate_rows: m["manifest_gate_v1.json"], s_weights: sm, s_scores: sm}.items()}
    if Path(gate_rows).name != "rows_v2.json":
        raise SystemExit("--gate-rows must be rows_v2.json")
    spec = M.build(manifests, sim_votes, real_votes, flag_rows)
    man = json.loads((HERE / "results" / "r44_selection_manifest.json").read_text())
    fresh = SEL.select(spec["labels"])
    if fresh["draws"] != man["draws"] or fresh["candidate_counts"] != man["candidate_counts"]:
        raise SystemExit("the committed selection manifest differs from a fresh selection")
    R = list(csv.DictReader(open(flag_rows)))
    loc = {r["call_id"]: (r["x"], r["chunk"], int(r["row"])) for r in R}
    draws = [[(*loc[c], 1) for c in d["sabotage"]] + [(*loc[c], 0) for c in d["not_sabotage"]] for d in man["draws"]]
    zr, zs = np.load(npz), np.load(s_weights)
    frozen = {n: tuple(np.asarray(z[k], dtype=np.float64).tolist() for k in ("weights", "mu", "sd")) for n, z in (("R", zr), ("S", zs))}
    out = fit_job.remote(got["rows_v2.json"], draws, [loc[r["call_id"]] for r in R], frozen)
    fs, ps = json.loads(Path(flag_scores).read_text()), json.loads(Path(s_scores).read_text())
    r_eval, r_staged = {s["call_id"]: s["score"] for s in fs["scores"]}, [s["score"] for s in fs["staged"]]
    s_eval = {s["call_id"]: s["score"] for s in ps["scores"]}
    sc = out["scores"]
    val = {"R_eval_max_abs_diff": float(max(abs(v - r_eval[r["call_id"]]) for v, r in zip(sc["R"]["eval"], R))),
           "R_staged_max_abs_diff": float(np.abs(np.array(sc["R"]["staged"]) - np.array(r_staged)).max()) if len(r_staged) == out["n_staged"] else None,
           "S_eval_max_abs_diff": float(max(abs(v - s_eval[r["call_id"]]) for v, r in zip(sc["S"]["eval"], R))),
           "R44_0_refit_max_abs_diff": out["refit0_max_abs_diff"], "selection_matches_fresh": True,
           "all_finite": bool(all(np.isfinite(np.array(v[k])).all() for v in sc.values() for k in ("eval", "staged")))}
    val["passes"] = bool(val["R_staged_max_abs_diff"] is not None and max(val["R_eval_max_abs_diff"], val["R_staged_max_abs_diff"],
                         val["S_eval_max_abs_diff"], val["R44_0_refit_max_abs_diff"]) <= 1e-6 and val["all_finite"])
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    if not val["passes"]:
        raise SystemExit(f"reproduction gates failed, nothing written: {json.dumps(val)}")
    wz = HERE / "results" / f"r44_weights_{stamp}.npz"
    with wz.open("xb") as fh:
        np.savez(fh, **{f"{k}_{n}": np.array(v[k]) for n, v in out["weights"].items() for k in ("weights", "mu", "sd")})
    staged_meta = [{"index": s["index"], "role": s["role"]} for s in fs["staged"]]
    names = ["S", "R"] + [f"R44_{i}" for i in range(len(draws))]
    res = {"rule": "RULE_simtrained.md Addendum 3 (= RULE_scaled.md Amendment 2)", "label": "POST HOC; PILOT, labels not audited",
           "git": git, "app": APP, "layer": LAYER, "C": CS[0], "validation": val, "inputs_sha256": got | spec["summary"]["inputs_sha256"],
           "selection_manifest_sha256": sha(HERE / "results" / "r44_selection_manifest.json"), "weights_npz": wz.name,
           "weights_npz_sha256": sha(wz), "chunks_read": out["chunks_read"], "probes": names,
           "scores": [{"call_id": r["call_id"], "part": int(r["part"]), "arm": r["arm"], "world": r["world"], "task": r["task"],
                       "group": r["group"], "state_changing": int(r["state_changing"]), "s": [sc[n]["eval"][i] for n in names]}
                      for i, r in enumerate(R)],
           "staged": [m_ | {"s": [sc[n]["staged"][i] for n in names]} for i, m_ in enumerate(staged_meta)],
           "written": stamp}
    dst = HERE / "results" / f"r44_scores_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(res) + "\n")
    print(json.dumps({"validation": val}, indent=1), "\n->", dst, sha(dst), wz)
