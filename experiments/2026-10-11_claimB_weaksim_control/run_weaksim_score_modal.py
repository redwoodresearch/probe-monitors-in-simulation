"""Weak-simulator positive control (PLAN.md): score la18 with claim B's frozen realism probes. One H100; numbers only.

Probes (the meaning control's frozen refit, `run_slips_evened_modal.py::refit` on main, its body copied in `fit_rows`:
fixed sim vs real e1 on pack `claimB/level1/packedB_2cfd1aec393c`, C 0.001, standardized on the training rows):
  heldout:<env>  for each claim B environment la18 shares (8) and, for the gate only, `tiktok`: trained on every pack
                 environment except <env>, at step A's layer for <env>;
  all10          trained on all ten pack environments at L63, for la18's 10 other environments.
Run score = mean of the run's kept `output_end` scores (`step_table`), as in step A.

Writes (Volume `claimB/weaksim_v1/scores.json`, returned to the laptop): the G2 gate (pooled claim B test AUROC over the 5
test environments vs 0.8221); per la18 run: arm, cell, environment, main task, probe, score, outputs, tokens, truncated;
per claim B run in the shared environments: arm, cell, environment, main task, score (catch thresholds, yardstick,
cohort row); the fresh-probe diagnostic (la18 R0 vs real, leave one la18 environment out, L63, same fit); and the
scrambled refits (each probe refit with claim B's labels swapped per main task with probability 1/2, same layer and C,
20 per probe, or 10 if the first 3 project the scrambled part past 6,000 s). Fit diagnostics (convergence) for every fit.

  uv run modal run --detach experiments/2026-10-11_claimB_weaksim_control/run_weaksim_score_modal.py
"""

import hashlib
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REPO = HERE.parents[1] if modal.is_local() else HERE  # the container holds only this file under /root
APP = "probe-shift_aswin-surya_sub-claimb-weaksim_score_kill-2026-11-11"
PACK = "claimB/level1/packedB_2cfd1aec393c"
EXTRACT = "2026-10-11_la18_weaksim-scrub1_qwen3-32b_alllayers_v1"
OUT = "claimB/weaksim_v1"
C, REAL, SIM, E2 = 1e-3, "real-attack-e1", "sim-attack-fixed", "real-attack-e2"
A_LAYER = {"clinical_trial": 58, "media_processing": 58, "tiktok": 58, "shared_dotfiles": 63, "user_tracking": 63,  # step A
           "auto_workflow": 63, "file_indexing": 58, "model_registry": 63, "slack_clone": 63, "web_scraping": 63}
TEST = ("clinical_trial", "media_processing", "shared_dotfiles", "tiktok", "user_tracking")
STEP_A_TEST_AUROC, G2_TOL = 0.8221, 0.02
N_SCRAMBLE, SEED = 20, 20261012
RATE_PER_H, BILLED_X, SCORE_BUDGET = 5.4, 1.45, 15.0  # all-in H100 container rate, billed / container time, PLAN.md cap ($)
STAGED = "2026-10-11_la18_weaksim_probe-inputs-l1_v1"
ARMS = ("real-attack-la18", "sim-attack-la18-r0", "real-attack-la18-plant1")

app = modal.App(APP)
acts = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("torch==2.8.0", "numpy", "scikit-learn")
         .add_local_python_source("probe_shift"))


def fit_rows(X, y):
    """The meaning control's frozen refit (run_slips_evened_modal.py::refit's body on main), on given rows and labels."""
    import torch

    from probe_shift import torch_probe as tp

    Xa = torch.from_numpy(X).cuda().double()
    Ya = torch.from_numpy(y.astype(float)[:, None]).cuda().double()
    mean, std = Xa.mean(0), Xa.std(0, unbiased=False)
    std = std.where(std > 0, std.new_ones(()))
    Xa = (Xa - mean) / std * C**0.5
    tol, mi, prev = 1e-4 * min(C**0.5, 1.0), 5000, None
    while True:
        W, b, evals = tp.fit(Xa, Ya, 1.0, mi, tol=tol)
        if evals < mi or evals == prev or mi * 4 > 80000:
            break
        mi, prev = mi * 4, evals
    Wg, bg = W.clone().requires_grad_(True), b.clone().requires_grad_(True)
    loss = (torch.nn.functional.binary_cross_entropy_with_logits(Xa @ Wg + bg, Ya, reduction="sum") + 0.5 * (Wg * Wg).sum()) / len(Xa)
    loss.backward()
    grad = float(max(Wg.grad.abs().max(), bg.grad.abs().max()))
    return (mean, std, W, b), {"converged": grad <= tol, "grad_max": grad, "evals": int(evals)}


def score_rows(probe, X):
    import numpy as np
    import torch

    mean, std, W, b = probe
    return ((((torch.from_numpy(np.asarray(X, np.float32)).cuda().double() - mean) / std * C**0.5) @ W + b)[:, 0]).cpu().numpy()


def run_means(scores, run_of_row, n_runs):
    import numpy as np

    s, c = np.zeros(n_runs), np.zeros(n_runs)
    np.add.at(s, run_of_row, scores)
    np.add.at(c, run_of_row, 1)
    return s / c


@app.function(image=image, gpu="H100!", cpu=8, memory=131072, volumes={"/acts": acts}, timeout=150 * 60, retries=0)
def score(env_to_fold: dict) -> dict:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    from probe_shift.q1_eval import step_table

    t0 = time.time()
    if (Path("/acts") / OUT / "scores.json").exists():
        raise FileExistsError(f"{OUT}/scores.json exists; scoring runs once")
    if json.loads((Path("/acts") / STAGED / "manifest.json").read_text()).get("stop"):
        raise RuntimeError("staging stopped on the eligibility rule; no reading")
    out = {"fits": {}}

    def finish(reason=None):
        out["stopped"] = reason
        out["seconds"] = round(time.time() - t0)
        (Path("/acts") / OUT).mkdir(parents=True, exist_ok=True)
        with open(Path("/acts") / OUT / "scores.json", "x") as fh:
            fh.write(json.dumps(out))
        acts.commit()
        return out
    # claim B pack
    pruns = json.loads((Path("/acts") / PACK / "runs.L63.json").read_text())
    for r in pruns:
        mt = r["cell"].split(":")[0]
        if r.get("main_task") not in (None, mt):
            raise ValueError(f"pack run {r['cell']}: main_task {r.get('main_task')} vs cell {mt}")
        r["main_task"] = mt
    t = step_table(pruns, "output_end")
    arm_of = np.array([pruns[i]["arm"] for i in t["run"]])
    env_of = np.array([pruns[i]["environment"] for i in t["run"]])
    mt_of = np.array([pruns[i]["main_task"] for i in t["run"]])
    efold = np.array([env_to_fold[e] for e in env_of])
    XP = {l: np.load(Path("/acts") / PACK / f"L{l}_output_end.npy") for l in (58, 63)}
    assert all(len(x) == len(t["run"]) for x in XP.values()), "pack rows and step table differ"
    train = np.isin(arm_of, [REAL, SIM])
    y_pack = arm_of == SIM

    # la18 rows from the extraction's metadata and per-layer chunk files (each chunk file loaded whole)
    run_dir = Path("/acts") / EXTRACT
    metas = [json.loads(p.read_text()) for arm in ARMS for p in sorted((run_dir / "meta" / arm).glob("*.json"))]
    chunks = sorted({m["chunk"] for m in metas})
    XL = {l: {c: np.load(run_dir / "by_layer" / f"L{l}" / f"{c}.npy") for c in chunks} for l in (58, 63)}
    runs, rows = [], {l: [] for l in (58, 63)}
    for m in metas:
        idx = [m["row_start"] + p["row"] for p in m["positions"] if p["kind"] == "output_end" and p["kept"]]
        if not idx:
            continue
        for l in (58, 63):
            rows[l].append(XL[l][m["chunk"]][idx])
        runs.append({k: m[k] for k in ("arm", "cell", "environment", "main_task")} | {"n_outputs": len(idx), "n_tokens": m["n_tokens"],
                                                                                     "truncated": m["truncated"]})
    staged = json.loads((Path("/acts") / STAGED / "manifest.json").read_text())["arms"]
    got = {a: sum(r["arm"] == a for r in runs) for a in ARMS}
    if got != {a: staged.get(a, 0) for a in ARMS}:  # every staged run must come back from the extraction with rows
        return finish(f"incomplete extraction: read {got}, staged {staged}")
    run_of = np.concatenate([np.full(len(x), i) for i, x in enumerate(rows[63])])
    XL = {l: np.concatenate(rows[l]) for l in (58, 63)}
    if not all(np.isfinite(x).all() for x in XL.values()):
        return finish("non-finite la18 activations")
    out["la18_runs_read"] = len(runs)
    la_env = np.array([r["environment"] for r in runs])
    shared = sorted(set(la_env) & set(env_to_fold))

    # frozen probes: held-out refits (shared environments and the 5 test environments for G2) and the all-ten refit
    probes = {}
    for e in sorted(set(shared) | set(TEST)):
        m = train & (efold != env_to_fold[e])
        probes[f"heldout:{e}"], out["fits"][f"heldout:{e}"] = fit_rows(XP[A_LAYER[e]][m], y_pack[m])
    probes["all10"], out["fits"]["all10"] = fit_rows(XP[63][train], y_pack[train])
    layer_of = {f"heldout:{e}": A_LAYER[e] for e in env_to_fold} | {"all10": 63}

    # G2: pooled claim B test AUROC over the 5 test environments, each by its held-out refit (step A's estimand)
    test_runs = sorted({int(r) for r, a, e in zip(t["run"], arm_of, env_of) if a in (REAL, SIM) and e in TEST})
    sc = np.zeros(len(pruns))
    for e in TEST:
        k = (env_of == e) & train
        s = score_rows(probes[f"heldout:{e}"], XP[A_LAYER[e]][k])
        sums, cnt = np.zeros(len(pruns)), np.zeros(len(pruns))
        np.add.at(sums, t["run"][k], s)
        np.add.at(cnt, t["run"][k], 1)
        sc[cnt > 0] = (sums / np.maximum(cnt, 1))[cnt > 0]
    yt = np.array([pruns[i]["arm"] == SIM for i in test_runs])
    g2 = float(roc_auc_score(yt, sc[test_runs]))
    out["G2"] = {"pooled_test_auroc": g2, "step_a": STEP_A_TEST_AUROC, "tolerance": G2_TOL, "runs": len(test_runs),
                 "pass": abs(g2 - STEP_A_TEST_AUROC) <= G2_TOL}
    unconverged = sorted(k for k, f in out["fits"].items() if not f["converged"])
    if unconverged:
        return finish(f"frozen refits did not converge: {unconverged}")
    if not out["G2"]["pass"]:
        return finish("G2 failed")

    def la_probe(e):
        return f"heldout:{e}" if e in env_to_fold else "all10"

    # la18 frozen scores
    plist = sorted({la_probe(e) for e in la_env})
    ids_of = {p: np.where([la_probe(e) == p for e in la_env])[0] for p in plist}  # sorted run indices per probe

    def la_read(pr, p):
        """Run means of probe pr (fitted for slot p) over the la18 runs that slot reads."""
        ids = ids_of[p]
        k = np.isin(run_of, ids)
        return ids, run_means(score_rows(pr, XL[layer_of[p]][k]), np.searchsorted(ids, run_of[k]), len(ids))

    la_scores = np.full(len(runs), np.nan)
    for p in plist:
        ids, means = la_read(probes[p], p)
        la_scores[ids] = means
    assert not np.isnan(la_scores).any()
    if not np.isfinite(la_scores).all():
        return finish("non-finite la18 scores")
    out["la18"] = [r | {"probe": la_probe(r["environment"]), "score": float(la_scores[i])} for i, r in enumerate(runs)]

    # claim B runs in the shared environments, each by its held-out refit (thresholds, yardstick, cohort row)
    cb = []
    for e in shared:
        k = env_of == e
        s = score_rows(probes[f"heldout:{e}"], XP[A_LAYER[e]][k])
        ids = np.unique(t["run"][k])
        means = run_means(s, np.searchsorted(ids, t["run"][k]), len(ids))
        cb += [{k2: pruns[i][k2] for k2 in ("arm", "cell", "environment", "main_task")} | {"score": float(v)} for i, v in zip(ids, means)]
    out["claimB_shared"] = cb

    # fresh-probe diagnostic: la18 R0 vs real, leave one la18 environment out, L63, same fit
    fresh = np.full(len(runs), np.nan)
    lab = np.array([r["arm"] for r in runs])
    for e in sorted(set(la_env)):
        tr = np.isin(run_of, np.where((la_env != e) & np.isin(lab, ARMS[:2]))[0])
        te_runs = np.where((la_env == e) & np.isin(lab, ARMS[:2]))[0]
        pr, out["fits"][f"fresh:{e}"] = fit_rows(XL[63][tr], lab[run_of[tr]] == ARMS[1])
        te = np.isin(run_of, te_runs)
        s = score_rows(pr, XL[63][te])
        fresh[te_runs] = run_means(s, np.searchsorted(te_runs, run_of[te]), len(te_runs))
    out["fresh_scores"] = [None if np.isnan(v) else float(v) for v in fresh]
    out["fresh_unconverged"] = sorted(k for k, f in out["fits"].items() if k.startswith("fresh:") and not f["converged"])

    # scrambled refits: claim B labels swapped per main task with probability 1/2, same layer, C and rows
    rng = np.random.default_rng(SEED)
    tasks = np.unique(mt_of[train])
    scr, t_s, n_per = {p: [] for p in plist}, [], N_SCRAMBLE
    for d in range(N_SCRAMBLE):
        if d >= n_per:
            break
        flip = dict(zip(tasks, rng.random(len(tasks)) < 0.5))
        y_s = np.where([flip[m] for m in mt_of], ~y_pack, y_pack)
        for p in plist:
            t1 = time.time()
            m = train & (efold != env_to_fold[p.split(":")[1]]) if p != "all10" else train
            pr, out["fits"][f"scramble{d}:{p}"] = fit_rows(XP[layer_of[p]][m], y_s[m])
            ids, means = la_read(pr, p)
            scr[p].append({"run_index": ids.tolist(), "score": means.tolist()})
            t_s.append(time.time() - t1)
            if len(t_s) == 3:  # project the whole scoring container's bill; drop to 10 per probe, else stop after draw 0
                bill = lambda n: (time.time() - t0 + np.mean(t_s) * (n * len(plist) - 3)) / 3600 * RATE_PER_H * BILLED_X
                n_per = next((n for n in (20, 10, 1) if bill(n) <= SCORE_BUDGET), 0)
                out["scrambled_projection"] = {"bill_20": bill(20), "bill_10": bill(10), "bill_1": bill(1), "chosen": n_per}
                if n_per == 0:  # even one full draw would pass the cap: stop the scrambled refits here
                    break
    out["scrambled"] = {"per_probe": n_per, "seconds_mean": float(np.mean(t_s)), "draws": scr,
                        "unconverged": sorted(k for k, f in out["fits"].items() if k.startswith("scramble") and not f["converged"])}
    return finish()


@app.local_entrypoint()
def main():
    env_to_fold = json.loads((REPO / "splits" / "folds_v2.json").read_text())["environment_folds"]["environment_to_fold"]
    res = score.remote(env_to_fold)
    res |= {"git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"scores_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(res))
    print({"stopped": res["stopped"], "G2": res.get("G2"), "la18_runs_read": res.get("la18_runs_read"),
           "scrambled_per_probe": (res.get("scrambled") or {}).get("per_probe"), "seconds": res["seconds"]})
    print("->", dst)
