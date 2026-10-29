"""RULE.md Amendment 1: convergence certificate for the high-stakes logistic fits (09's bar, RULE_sanity_bundle.md C2). One Modal
CPU container: max|grad G| in float64 at the chosen L43 refit's saved weights (all train rows), and for the 64 layer-selection fits
(80% split) the as-run refit's gradient plus a tight refit (float64 L-BFGS on G, tolerance_grad 1e-8, tolerance_change 0, <= 5,000
iterations) with its held-out AUROC; and, if L43 is over the bar, a tight L43 refit saved as a folded probe for the movement check.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run experiments/2026-10-14_p2_highstakes/run_convergence.py
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-p2probes_hsconv_kill-2026-11-13"
STAMP = "2026-10-14_131233"
PROBE = HERE / "results" / f"highstakes_probe_{STAMP}.safetensors"
BAR, C = 1e-6, 1.0

app = modal.App(APP)
reads = modal.Volume.from_name("probe-shift_aswin-surya_sub-p2probes_reads_kill-2026-11-13", version=2).read_only()
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "scikit-learn==1.7.2", "joblib", "safetensors==0.8.0", "torch==2.8.0")
         .add_local_file(HERE / "run_read_fit.py", "/root/hs_reader.py"))


def grad_G(Z, y, w, b):  # sklearn's objective / n: (sum logloss + ||w||^2 / (2C)) / n, intercept unpenalized; float64
    import numpy as np

    p = 1 / (1 + np.exp(-(Z @ w + b)))
    n = len(y)
    return np.concatenate([(Z.T @ (p - y) + w / C) / n, [(p - y).sum() / n]])


def tight(Z, y):  # float64 L-BFGS on G from zero (09's bar)
    import torch

    Zt, yt = torch.from_numpy(Z), torch.from_numpy(y.astype("float64"))
    w = torch.zeros(Z.shape[1] + 1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w], lr=1, max_iter=5000, tolerance_grad=1e-8, tolerance_change=0, history_size=50, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = Zt @ w[:-1] + w[-1]
        g = (torch.nn.functional.softplus(z) - yt * z).sum() + (w[:-1] @ w[:-1]) / (2 * C)
        loss = g / len(yt)
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().numpy()[:-1], float(w.detach().numpy()[-1])


@app.function(image=image, cpu=(16, 16), memory=(65536, 65536), timeout=3600, retries=0, volumes={"/reads": reads})
def certify(probe_bytes: bytes) -> dict:
    import random

    import hs_reader as hs
    import numpy as np
    import safetensors.numpy as stn
    import torch
    from joblib import Parallel, delayed
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    torch.set_num_threads(1)
    items, _ = hs.load_items()
    d = Path("/reads") / "highstakes" / STAMP
    plan = hs.shard_plan(items)
    A, where = {}, {}
    for s, ps in plan["parts"].items():  # hs.fit's bookkeeping and integrity checks (Codex): anchors in part 00 skipped, excluded dropped
        for p, pt in enumerate(ps):
            f = f"shard{s}_part{p:02d}"
            A[f] = np.load(d / f"{f}.npy", mmap_mode="r")
            m = json.loads((d / f"{f}.json").read_text())
            if m["items"] != pt or [e["item"] for e in m["meta"]] != pt or A[f].shape != (len(pt), 2, 64, 5120):
                raise ValueError(f"{f}: items, meta or shape differ from the plan")
            for k, (i, e) in enumerate(zip(m["items"], m["meta"])):
                if p == 0 and k < hs.N_ANCHOR:
                    continue
                if i in where:
                    raise ValueError(f"item {i} appears twice")
                if not e["excluded"]:
                    where[i] = (f, k)
    tr_items = [i for i, it in enumerate(items) if it[0] == "train" and i in where]
    y = np.array([items[i][1] for i in tr_items], dtype=np.float64)
    rng, ho = random.Random(20261014), set()  # hs.fit's stratified 20% held out
    for c in (0, 1):
        idx = [j for j in range(len(tr_items)) if y[j] == c]
        ho |= set(rng.sample(idx, round(0.2 * len(idx))))
    fit_i, ho_i = np.array([j for j in range(len(tr_items)) if j not in ho]), np.array(sorted(ho))
    if (len(tr_items), len(ho_i)) != (8000, 1600):  # the record's n (read_fit_2026-10-14_131233: train 8000, held out 1600)
        raise ValueError(f"train {len(tr_items)} / held out {len(ho_i)} differ from the record")

    def X(l):
        out = np.empty((len(tr_items), 5120), np.float32)
        for f in A:
            pos = [j for j, i in enumerate(tr_items) if where[i][0] == f]
            if pos:
                out[pos] = A[f][[where[tr_items[j]][1] for j in pos], 0, l]
        return out

    t = stn.load(probe_bytes)
    L = 43
    X43 = X(L)  # float32, as fit() had it
    mu, sd = t["mean"], t["std"]  # float32, as fit() standardized
    if not np.allclose(mu, X43.mean(0), atol=1e-3) or not np.allclose(sd, X43.std(0) + 1e-6, rtol=1e-3):
        raise ValueError("saved standardization does not match the training rows")
    Z43 = ((X43 - mu) / sd).astype(np.float64)  # standardized in float32 then float64, as sklearn received it (Codex)
    g = grad_G(Z43, y, t["W"].astype(np.float64), float(t["intercept"].reshape(-1)[0]))
    chosen = {"layer": L, "n": len(y), "max_abs_grad": float(np.abs(g).max()), f"max|grad G| <= {BAR}": bool(np.abs(g).max() <= BAR)}

    def one(l):
        torch.set_num_threads(1)
        Xl = X(l)
        m_, s_ = Xl[fit_i].mean(0), Xl[fit_i].std(0) + 1e-6
        Zf = ((Xl[fit_i] - m_) / s_).astype(np.float64)
        Zh = ((Xl[ho_i] - m_) / s_).astype(np.float64)
        clf = LogisticRegression(C=C, max_iter=2000, tol=1e-6).fit((Xl[fit_i] - m_) / s_, y[fit_i])  # the as-run fit, reproduced
        ga = np.abs(grad_G(Zf, y[fit_i], clf.coef_[0].astype(np.float64), float(clf.intercept_[0]))).max()
        wt, bt = tight(Zf, y[fit_i])
        gt = np.abs(grad_G(Zf, y[fit_i], wt, bt)).max()
        return l, {"n": len(fit_i), "as_run_max_abs_grad": float(ga), f"as_run max|grad G| <= {BAR}": bool(ga <= BAR), "as_run_heldout_auroc": float(roc_auc_score(y[ho_i], clf.decision_function((Xl[ho_i] - m_) / s_))),
                   "tight_max_abs_grad": float(gt), "tight_certified": bool(gt <= BAR), "tight_heldout_auroc": float(roc_auc_score(y[ho_i], Zh @ wt + bt))}

    from joblib import parallel_backend

    with parallel_backend("loky", inner_max_num_threads=1):  # 16 workers x 1 thread each (Codex)
        sel = dict(Parallel(n_jobs=16)(delayed(one)(l) for l in range(64)))
    out = {"chosen_refit": chosen, "selection_fits": sel,
           "layer_by_as_run": max(sel, key=lambda l: sel[l]["as_run_heldout_auroc"]), "layer_by_tight": max(sel, key=lambda l: sel[l]["tight_heldout_auroc"])}
    if not chosen[f"max|grad G| <= {BAR}"]:
        wt, bt = tight(Z43, y)
        gt = float(np.abs(grad_G(Z43, y, wt, bt)).max())
        gs = float(np.abs(grad_G(Z43, y, wt.astype(np.float32).astype(np.float64), float(np.float32(bt)))).max())  # after float32 serialization
        r, c = (wt / sd).astype(np.float32), np.float32(bt - float((mu.astype(np.float64) * wt / sd).sum()))
        out["tight_L43"] = {"max_abs_grad": gt, "max_abs_grad_serialized_float32": gs, "certified": gt <= BAR,
                            "status": "intermediate: the movement check (catch on B, as-run vs tight) is pending",
                            "file": stn.save({"direction": r, "bias": np.array([c], np.float32), "W": wt.astype(np.float32), "intercept": np.array([bt], np.float32),
                                              "mean": mu.astype(np.float32), "std": sd.astype(np.float32)}, metadata={"layer": str(L), "fit": "tight float64 L-BFGS"})}
    return out


@app.local_entrypoint()
def main():
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    if git("status", "--porcelain", "--", ".") or git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("uncommitted changes")
    head = git("rev-parse", "HEAD")
    if not git("branch", "-r", "--contains", head):
        raise SystemExit("HEAD is not pushed")
    r = certify.remote(PROBE.read_bytes())
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rec = {"label": "RULE.md Amendment 1 (convergence certificate)", "git": head, "app": APP, "probe": PROBE.name,
           "probe_sha256": hashlib.sha256(PROBE.read_bytes()).hexdigest(), "objective": "G = (sum logloss + ||w||^2 / (2C)) / n, C = 1, intercept unpenalized (sklearn's)",
           "bar": BAR} | {k: v for k, v in r.items() if k != "tight_L43"}
    if "tight_L43" in r:
        f = HERE / "results" / f"highstakes_probe_tight_{stamp}.safetensors"
        f.open("xb").write(r["tight_L43"]["file"])
        rec["tight_L43"] = {k: v for k, v in r["tight_L43"].items() if k != "file"} | {"file": {"path": f.name, "sha256": hashlib.sha256(r["tight_L43"]["file"]).hexdigest()}}
    dest = HERE / "results" / f"convergence_{stamp}.json"
    dest.open("x").write(json.dumps(rec, indent=1))
    s = r["selection_fits"]
    lines = [f"chosen L43 refit: n {r['chosen_refit']['n']}, max|grad G| {r['chosen_refit']['max_abs_grad']:.3g}, certified {r['chosen_refit'][f'max|grad G| <= {BAR}']}",
             f"selection fits: as-run max|grad| max over layers {max(v['as_run_max_abs_grad'] for v in s.values()):.3g}; tight certified {sum(v['tight_certified'] for v in s.values())}/64",
             f"layer by as-run held-out AUROC {r['layer_by_as_run']}; by tight {r['layer_by_tight']}"]
    if "tight_L43" in rec:
        lines.append(f"tight L43: max|grad| {rec['tight_L43']['max_abs_grad']:.3g}, certified {rec['tight_L43']['certified']}")
    dest.with_suffix(".printout.txt").open("x").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(dest.name)
