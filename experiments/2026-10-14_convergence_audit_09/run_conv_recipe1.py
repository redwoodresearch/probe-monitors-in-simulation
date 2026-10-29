"""Convergence audit, part 2 (RULE.md, Part 2): recipe #1's C = 0.001 simulated-cell candidates (run 2026-10-13_173403, git
bf121ce2), refit with a tight C-scaled float64 L-BFGS on the as-run training rows and saved standardization, swapped into the
stored candidate scores, and the per-fold simulated-only choice redone (the as-run recipe_select, copied whole). Reads d9's
packs and weights on the q3attack_work Volume read only (packs: recipe1/2026-10-13_172450 and sce_fixfull/2026-10-13_152618).

  CONV_R1_IN=<dir with recipe1_scores_2026-10-13_173403.npz and recipe1_run_2026-10-13_173403.json> PYTHONPATH=src \\
      uv run --no-project --python 3.12 --with 'modal==1.6.1' --with 'numpy==2.3.3' modal run --detach \\
      experiments/2026-10-14_convergence_audit_09/run_conv_recipe1.py --out <results json>
"""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
IN = Path(os.environ.get("CONV_R1_IN", "/nonexistent"))
APP = "probe-shift_aswin-surya_sub-convaudit_recipe1_kill-2026-11-14"
WORK = "probe-shift_aswin-surya_sub-q3attack_work_kill-2026-11-06"
RUN, PACKS, P1_PACK = "2026-10-13_173403", "2026-10-13_172450", "sce_fixfull/2026-10-13_152618"
SCORES_SHA, RECORD_SHA = "3e42da3f206c8154d2defc442ac63722addb9affedf29f2f058b74a6a77eb5b3", "88aea1b89a339e8e"
P1_SHA = {48: "d3339fdabbad1c662ca4e98df5b63f6679f2300a646377194f539222707dfb80",
          63: "759d8b8bf07dc404f5fc24a32c7364d8a2456ae43d1e0cd1da355abde5bca6cd"}  # run_recipe1.P1_SHA
L7 = (16, 24, 32, 40, 48, 56, 63)
PARTS = {"P1": {"layers": (48, 63), "readouts": ("output_end",), "sim": ("sim-attack-fullfix", "sim-honest-fullfix")},
         "P2": {"layers": L7, "readouts": ("output_end", "output_end_mean"), "sim": ("sim-attack-fixed", "sim-honest-edited")},
         "P3": {"layers": L7, "readouts": ("output_end",), "sim": ("sim-attack-fx-full", "sim-honest-fx-full")}}  # run_recipe1.PARTS
CS, C_TEST, ALPHA = (0.001, 0.01, 0.1, 1.0), 0.001, 0.01
PAIRS = [(f, v) for f in range(5) for v in range(5) if v != f]
TOL, MAXIT, GRAD_OK = 1e-8, 5000, 1e-6

app = modal.App(APP)
work = modal.Volume.from_name(WORK, version=2).read_only()
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy==2.3.3")
         .add_local_python_source("probe_shift")
         .add_local_file(HERE / "recipe_select_bf121ce2.py", "/root/recipe_select_bf121ce2.py")
         .add_local_file(HERE / "recipe_select_757d07ac.py", "/root/recipe_select_757d07ac.py"))
if IN.exists():
    image = image.add_local_dir(IN, "/root/in")


def objective(Xs, y, W, b, scale, C):
    """run_conv_titlecells.objective with C as an argument: scale * (C * sum log loss + 0.5 ||W||^2) / n and its largest
    absolute gradient component (W and b)."""
    import torch
    import torch.nn.functional as F

    W = W.detach().clone().requires_grad_(True)
    b = b.detach().clone().requires_grad_(True)
    loss = scale * (C * F.binary_cross_entropy_with_logits((Xs @ W + b).reshape(-1), y, reduction="sum") + 0.5 * (W * W).sum()) / len(y)
    loss.backward()
    return float(loss.detach()), float(torch.cat([W.grad.reshape(-1), b.grad.reshape(-1)]).abs().max())


def fit_tight(Xs, y, C):
    """run_conv_titlecells.fit_tight with C as an argument: L-BFGS on G = (sum log loss + ||W||^2 / (2C)) / n, float64, from zero."""
    import torch
    import torch.nn.functional as F

    n, d = Xs.shape
    W = torch.zeros(d, 1, device=Xs.device, dtype=Xs.dtype, requires_grad=True)
    b = torch.zeros(1, device=Xs.device, dtype=Xs.dtype, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=MAXIT, max_eval=int(MAXIT * 1.25), tolerance_grad=TOL, tolerance_change=0.0,
                            history_size=20, line_search_fn="strong_wolfe")
    ev = [0]

    def closure():
        opt.zero_grad()
        loss = (F.binary_cross_entropy_with_logits((Xs @ W + b).reshape(-1), y, reduction="sum") + (W * W).sum() / (2 * C)) / n
        loss.backward()
        ev[0] += 1
        return loss

    opt.step(closure)
    return W.detach(), b.detach(), ev[0]


def load_in():
    import numpy as np

    z = Path("/root/in/recipe1_scores_2026-10-13_173403.npz")
    if hashlib.sha256(z.read_bytes()).hexdigest() != SCORES_SHA:
        raise SystemExit("scores file sha256 differs")
    r = Path("/root/in/recipe1_run_2026-10-13_173403.json")
    if not hashlib.sha256(r.read_bytes()).hexdigest().startswith(RECORD_SHA):
        raise SystemExit("run record sha256 differs")
    return np.load(z), json.loads(r.read_text())


@app.function(image=image, gpu="H100!", cpu=(8, 8), memory=(65536, 65536), timeout=2400, retries=0, volumes={"/work": work})
def refit(part: str, layer: int) -> dict:
    """For each readout: gate P (the pack is the as-run one), gate A (the saved C 0.001 weights reproduce the stored nested
    scores), then the tight refit of the 20 nested models from zero; returns the tight nested scores and diagnostics."""
    import numpy as np
    import torch

    from probe_shift.attack_eval import pool

    torch.backends.cuda.matmul.allow_tf32 = False
    Z, rec = load_in()
    arm, fold = Z[f"runs_{part}__arm"].astype(str), Z[f"runs_{part}__fold"].astype(int)
    n = len(arm)
    pos, neg = PARTS[part]["sim"]
    y = np.where(arm == pos, 1.0, np.where(arm == neg, 0.0, np.nan))
    trainable = ~np.isnan(y)
    shas = {(p["part"], p["layer"], k): v["features_sha256"] for p in rec["pack"] for k, v in p["readouts"].items()}
    out = {"part": part, "layer": layer, "readouts": {}}
    for ro in PARTS[part]["readouts"]:
        if part == "P1":
            xf, rf = Path("/work") / P1_PACK / f"L{layer}.npy", Path("/work") / P1_PACK / f"rows_L{layer}.npz"
            want = P1_SHA[layer]
        else:
            d = Path("/work/recipe1") / PACKS / part
            xf, rf, want = d / f"{ro}_L{layer}.npy", d / f"{ro}_rows_L{layer}.npz", shas[(part, layer, ro)]
        Xh = np.load(xf)
        gateP = hashlib.sha256(Xh.tobytes()).hexdigest() == want
        r_run = np.load(rf)["run"].astype(np.int64)
        X = torch.from_numpy(Xh).cuda().float()
        del Xh
        key = f"{part}__L{layer}__{ro}__C{C_TEST:g}__sim"
        ref = Z[key]
        Wz = np.load(Path("/work/recipe1") / RUN / part / "weights" / f"L{layer}__{ro}__C{C_TEST:g}__sim__C.npz")
        nt = np.full((5, 5, n), np.nan, np.float32)
        fits, gA = [], []
        for f, v in PAIRS:
            te = (fold == f) | (fold == v)
            tr, ti = np.flatnonzero((trainable & ~te)[r_run]), np.flatnonzero(te[r_run])
            mean, std = torch.from_numpy(Wz["mean"][f, v]).cuda(), torch.from_numpy(Wz["std"][f, v]).cuda()
            W32 = torch.from_numpy(Wz["W"][f, v]).cuda().reshape(-1, 1)
            b32 = torch.tensor([float(Wz["b"][f, v])], device="cuda")
            Xti = (X[torch.from_numpy(ti).cuda()] - mean) / std
            S_as = pool(r_run[ti], (Xti @ W32 + b32).double().cpu().numpy(), n, "mean")[:, 0]
            S_as[~te] = np.nan
            same = bool(np.array_equal(np.isnan(S_as), np.isnan(ref[f, v])))
            gA.append({"same_missing": same, "max_abs_diff": float(np.nanmax(np.abs(S_as - ref[f, v])))})
            Xtr = ((X[torch.from_numpy(tr).cuda()] - mean) / std).double()
            ytr = torch.from_numpy(y[r_run[tr]]).cuda().double()
            L_as, g_as = objective(Xtr, ytr, W32.double(), b32.double(), 1.0, C_TEST)
            W, b, ev = fit_tight(Xtr, ytr, C_TEST)
            L_t, _ = objective(Xtr, ytr, W, b, 1.0, C_TEST)
            _, g_tG = objective(Xtr, ytr, W, b, 1.0 / C_TEST, C_TEST)
            S_t = pool(r_run[ti], (Xti.double() @ W + b).cpu().numpy(), n, "mean")[:, 0]
            S_t[~te] = np.nan
            nt[f, v] = S_t
            fin = ~np.isnan(ref[f, v])
            ok = bool(np.array_equal(np.isnan(nt[f, v]), ~fin) and np.isfinite(nt[f, v][fin]).all() and np.isfinite([L_t, g_tG]).all())
            fits.append({"f": f, "v": v, "train_rows": int(len(tr)), "L_asrun": L_as, "L_tight": L_t, "rel_L_gap": (L_as - L_t) / L_t,
                         "maxgrad_L_asrun_float64": g_as, "maxgrad_G_tight": g_tG, "evals_tight": ev, "tight_finite_same_missing": ok,
                         "max_abs_run_score_diff": float(np.nanmax(np.abs(nt[f, v] - ref[f, v])))})
            del Xtr, Xti
            torch.cuda.empty_cache()
        out["readouts"][ro] = {"key": key, "gateP_pack_sha": gateP,
                               "gateA_pass": all(g["same_missing"] and np.isfinite(g["max_abs_diff"]) and g["max_abs_diff"] <= 1e-3 for g in gA),
                               "gateA": gA, "fits": fits,
                               "gateC_pass": all(np.isfinite(x["maxgrad_G_tight"]) and x["maxgrad_G_tight"] <= GRAD_OK and x["tight_finite_same_missing"] for x in fits),
                               "nested_tight": nt}
        del X
        torch.cuda.empty_cache()
    return out


@app.function(image=image, cpu=(8, 8), memory=(32768, 32768), timeout=3000, retries=0)
def orchestrate(git: str) -> dict:
    import sys

    sys.path.insert(0, "/root")
    import recipe_select_757d07ac as rs_new
    import recipe_select_bf121ce2 as rs_asrun

    Z, rec = load_in()
    res = {"git": git, "run": RUN, "packs": PACKS, "settings": {"C_test": C_TEST, "tol_on_G": TOL, "max_iter": MAXIT, "grad_ok": GRAD_OK},
           "parts": {}}
    asrun_choice = {p: {int(f): tuple(c["chosen"]) for f, c in ch.items()} for p, ch in rec["choice"].items()}
    jobs = [(p, l) for p, P in PARTS.items() for l in P["layers"]]
    outs = list(refit.starmap(jobs))
    tight = {(o["part"], o["layer"], ro): r for o in outs for ro, r in o["readouts"].items()}
    for p, P in PARTS.items():
        arm, fold = Z[f"runs_{p}__arm"].astype(str), Z[f"runs_{p}__fold"].astype(int)
        cand = {(l, ro, C): Z[f"{p}__L{l}__{ro}__C{C:g}__sim"] for l in P["layers"] for ro in P["readouts"] for C in CS}
        kw = {"readout_order": ("output_end", "output_end_mean"), "alpha": ALPHA}
        old = rs_asrun.select(cand, arm, fold, *P["sim"], **kw)
        gateS = {f: tuple(old[f]["chosen"]) == asrun_choice[p][f] for f in range(5)}
        old_new_code = rs_new.select(cand, arm, fold, *P["sim"], **kw)
        diag = {f"L{l}|{ro}": {k: v for k, v in tight[(p, l, ro)].items() if k != "nested_tight"} for l in P["layers"] for ro in P["readouts"]}
        gates_ok = all(gateS.values()) and all(t["gateP_pack_sha"] and t["gateA_pass"] for (pp, _, _), t in tight.items() if pp == p)
        cert = all(t["gateC_pass"] for (pp, _, _), t in tight.items() if pp == p)
        pr = {"gateS_asrun_choice_reproduced": gateS, "gates_P_A_S_pass": gates_ok, "gateC_certified": cert, "fits": diag,
              "asrun_choice": {f: list(old[f]["chosen"]) for f in range(5)},
              "asrun_choice_757d07ac_select": {f: list(old_new_code[f]["chosen"]) for f in range(5)}}
        if gates_ok and cert:  # the swap and the new choice, only when every gate passes for this part
            cand_t = dict(cand) | {(l, ro, C_TEST): tight[(p, l, ro)]["nested_tight"] for l in P["layers"] for ro in P["readouts"]}
            new = rs_asrun.select(cand_t, arm, fold, *P["sim"], **kw)
            new2 = rs_new.select(cand_t, arm, fold, *P["sim"], **kw)
            crit = lambda sel, f, k: next(({"catch": c["catch"], "auroc": c["auroc"]} for c in sel[f]["candidates"]  # noqa: E731
                                           if (c["layer"], c["readout"], c["C"]) == k), None)
            pr["tight"] = {
                "choice": {f: list(new[f]["chosen"]) for f in range(5)}, "choice_757d07ac_select": {f: list(new2[f]["chosen"]) for f in range(5)},
                "folds_choice_changed": [f for f in range(5) if tuple(new[f]["chosen"]) != tuple(old[f]["chosen"])],
                "folds_choosing_C0.001": {"asrun": [f for f in range(5) if old[f]["chosen"][2] == C_TEST], "tight": [f for f in range(5) if new[f]["chosen"][2] == C_TEST]},
                "C0.001_criteria_by_fold": {f: {f"L{l}|{ro}": {"asrun": crit(old, f, (l, ro, C_TEST)), "tight": crit(new, f, (l, ro, C_TEST))}
                                                for l in P["layers"] for ro in P["readouts"]} for f in range(5)},
                "chosen_criteria_by_fold": {f: {"asrun": crit(old, f, tuple(old[f]["chosen"])), "tight": crit(new, f, tuple(new[f]["chosen"]))} for f in range(5)}}
        res["parts"][p] = pr
    return res


@app.local_entrypoint()
def main(out: str = ""):
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=REPO).stdout:
        raise SystemExit("uncommitted tracked changes")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    print(f"git {git}", flush=True)
    t0 = time.time()
    r = orchestrate.remote(git)
    lines = []
    for p, pr in r["parts"].items():
        ev = sorted(x["evals_tight"] for d in pr["fits"].values() for x in d["fits"])
        gr = max(x["maxgrad_G_tight"] for d in pr["fits"].values() for x in d["fits"])
        rel = sorted(x["rel_L_gap"] for d in pr["fits"].values() for x in d["fits"])
        sd = max(x["max_abs_run_score_diff"] for d in pr["fits"].values() for x in d["fits"])
        g = (f"{p}: gate S {pr['gateS_asrun_choice_reproduced']} | gates P/A/S {pr['gates_P_A_S_pass']} | certified {pr['gateC_certified']}"
             f" | tight evals {ev[0]}/{ev[len(ev) // 2]}/{ev[-1]} | max|grad G| {gr:.1e} | rel L gap median {rel[len(rel) // 2]:.2e} max {rel[-1]:.2e}"
             f" | max |run score diff| {sd:.3f} | as-run choice {pr['asrun_choice']}")
        if "tight" in pr:
            t = pr["tight"]
            g += f" | tight choice {t['choice']} | folds changed {t['folds_choice_changed']} | C 0.001 chosen: as-run {t['folds_choosing_C0.001']['asrun']} tight {t['folds_choosing_C0.001']['tight']}"
        lines.append(g)
    print("\n".join(lines), flush=True)
    if out:
        Path(out).open("x").write(json.dumps(r, indent=1, default=lambda a: None))
        Path(out[:-5] + ".printout.txt").open("x").write("\n".join(lines) + f"\nclient seconds {round(time.time() - t0)}\n")
