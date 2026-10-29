"""Realism probe at the 196-task monitor's command tokens (RULE_POSTHOC_realism_196.md; POST HOC). One Modal job on the
monitor's own rows and pipeline:
- the realism probe: run_score_onpolicy.one_draw with world as the label (real run 1 = 0, simulated = 1);
- the control: the same with real run 2 = 1 and real run 1 = 0;
- n_null task-block label shuffles of the realism probe;
- per fold, the realism probe at the monitor's layer (C by inner CV at that layer), the monitor's saved model
  re-scored, and the projection terms.
It writes /score/<key>/result.json before returning, so a run whose client died can still be read. Launched from this
file, detached, with a frozen copy of watch_layers.py on the app ID (the spend stop):

  uv run --no-project --python 3.14 --with 'modal>=1.4.3' --with pyyaml --with numpy python run_realism_probe.py \\
      --config score_scaleup196.yaml --watcher <frozen watch_layers.py> --prior-usd <billed so far> --cap-usd 9.00 \\
      --max-minutes 95 --n-null 6
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

sys.path.insert(0, str(Path(__file__).parent))
import run_score_onpolicy as R  # noqa: E402

APP = "probe-shift_aswin-surya_sub-q3onpol_realism_kill-2026-11-08"
PROBES = "scaleup196_probes_14b33da3d97a_percall/probes_draw0.npz"  # the 196-task monitor's draw-0 fold models
SEED = 20261012
app = modal.App(APP)
image = R.image.add_local_python_source("run_score_onpolicy")


def relabel(D, pos, neg, name, y=None):
    """D with one training world `name`: `neg` rows labelled 0 and `pos` rows 1 (or y); other rows keep their world."""
    import numpy as np

    m = np.isin(D["world"], [pos, neg])
    return D | {"world": np.where(m, name, D["world"]), "y": (D["world"] == pos).astype(int) if y is None else y,
                "worlds": [name], "cells": [(name, name)], "warm_start": False}


@app.function(image=image, gpu="H200", memory=65536, timeout=6600, retries=0,  # H200 (the realism world doubles the rows); 110 min (0d, reply arm)
              volumes={"/acts": R.acts_vol.read_only(), "/score": R.score_vol})
def realism(cfg: dict, folds: dict, key: str, n_null: int, alignment: bool = True) -> dict:
    import numpy as np
    import torch

    t0 = time.time()
    D = R.load(cfg, folds)
    load_s, d = time.time() - t0, Path("/score") / key
    print(f"{key}: loaded {D['X'].shape} in {load_s:.0f} s", flush=True)
    R.score_vol.reload()
    if d.exists() and any(d.iterdir()):  # never overwrite an earlier run's folder
        raise RuntimeError(f"{key} already holds files")
    d.mkdir(parents=True, exist_ok=True)
    X, fold, dev = D["X"], D["fold"], D["X"].device
    out = {"summary": D["summary"], "load_seconds": round(load_s, 1), "meta": D["meta"]}

    def run(Dx, save):
        R._WARM.clear()
        R._SAVED.clear()
        R._LAYERS.clear()
        t = time.time()
        r = R.one_draw(Dx | {"save_probes": save}, 0, cfg["seed"], observed=True, per_call=True) | {"seconds": round(time.time() - t, 2)}
        print(f"{key}: one_draw in {r['seconds']} s", flush=True)
        return r, {k: v for k, v in R._SAVED.items()}

    def save(name, obj):  # each stage as it finishes, so a stopped run keeps what it finished
        (d / f".{name}.json.tmp").write_text(json.dumps(obj))
        (d / f".{name}.json.tmp").replace(d / f"{name}.json")
        R.score_vol.commit()

    def fit_at(Dx, rows, L, Cs):  # fit_rows' primal fit at one layer, draw-0 weights; (mu, sd, B [d, c])
        idx = torch.tensor(np.flatnonzero(rows), device=dev)
        w = torch.ones(len(idx), dtype=torch.float64, device=dev)
        y = torch.tensor(Dx["y"][rows], dtype=torch.float64, device=dev)
        Z = X[L: L + 1, idx].double()
        mu, sd = R.standardize(Z, w)
        Z.sub_(mu).div_(sd)
        B, kkt, _ = R.fit_primal(Z, y, w, Cs)
        for _ in range(R.RESTARTS):
            if not np.isfinite(kkt) or kkt <= 1e-4:
                break
            B, kkt, _ = R.fit_primal(Z, y, w, Cs, beta0=B)
        if not np.isfinite(kkt) or kkt > 1e-4:
            raise RuntimeError(f"primal fit not converged: relative KKT residual {kkt:.2e}")
        return mu[0, 0], sd[0, 0], B[0]

    def score_at(rows, L, mu, sd, beta):
        idx = torch.tensor(np.flatnonzero(rows), device=dev)
        return ((X[L, idx].double() - mu) / sd) @ beta

    Dr = relabel(D, "sim", "real", "rs")
    main, saved = run(Dr, True)
    out["realism"] = main
    save("stage_realism", out)
    out["control"] = run(relabel(D, "run2", "real", "rr"), False)[0]
    save("stage_control", {"control": out["control"]})
    out["nulls"] = []
    for i in range(1, n_null + 1):  # task-block shuffle: per task, the real and simulated runs swap labels with prob 1/2
        flip = np.random.default_rng([SEED, i]).integers(0, 2, len(D["fold_of_task"]))
        y = np.where(np.isin(D["world"], ["real", "sim"]), (D["world"] == "sim").astype(int) ^ flip[D["task"]], 0)
        r = run(relabel(D, "sim", "real", "rs", y), False)[0]
        out["nulls"].append({"i": i, "values": r["cells"]["rs->rs"]["values"], "per_call": r["cells"]["rs->rs"]["per_call"],
                             "seconds": r["seconds"]})
        save(f"stage_null_{i}", out["nulls"][-1])
    if not alignment:  # the reply-token arm: separation only (the monitor reads command tokens, not these rows)
        out["folds"] = []
        out["seconds"] = round(time.time() - t0, 1)
        (d / ".result.json.tmp").write_text(json.dumps(out))
        (d / ".result.json.tmp").replace(d / "result.json")
        R.score_vol.commit()
        print(f"{key}: done in {out['seconds']} s", flush=True)
        return out
    P = np.load(Path("/score") / PROBES)
    mon = {(W, k): (L, C) for W, k, L, C in json.loads(str(P["meta"]))}
    inW = Dr["world"] == "rs"
    out["folds"] = []
    for k in range(5):
        cal, T = (k + 1) % 5, [f for f in range(5) if f not in (k, (k + 1) % 5)]
        Lm = mon[("real", k)][0]
        mu_m, sd_m, b_m = (torch.tensor(P[f"real|{k}|{n}"], dtype=torch.float64, device=dev) for n in ("mu", "sd", "beta"))
        held, ys = [], []
        for j in T:  # the inner CV of one_draw, at the monitor's layer only
            mu, sd, B = fit_at(Dr, inW & np.isin(fold, [t for t in T if t != j]), Lm, R.CS)
            ho = inW & (fold == j)
            held.append(score_at(ho, Lm, mu, sd, B))
            ys.append(Dr["y"][ho])
        s = torch.cat(held, 0).T
        yy = torch.tensor(np.concatenate(ys), dtype=torch.float64, device=dev)
        auc = R.auroc_batch(s, yy, torch.ones_like(yy)).cpu().numpy()
        c = int(np.flatnonzero(auc >= auc.max() - R.TIE).min())
        mu_r, sd_r, B_r = fit_at(Dr, inW & np.isin(fold, T), Lm, [R.CS[c]])
        b_r = B_r[:, 0]
        v = b_r * sd_m / sd_r  # the realism direction in the monitor's standardized coordinates
        u = v / v.norm()
        raw_r, raw_m = b_r / sd_r, b_m / sd_m
        rows = np.isin(fold, [k, cal])
        idx = torch.tensor(np.flatnonzero(rows), device=dev)
        z = (X[Lm, idx].double() - mu_m) / sd_m
        Lr, sv = saved[("rs", k)]["layer"], saved[("rs", k)]
        s_r = score_at(rows, Lr, *(torch.tensor(sv[n], dtype=torch.float64, device=dev) for n in ("mu", "sd", "beta")))
        out["folds"].append({
            "fold": k, "monitor_layer": Lm, "monitor_C": mon[("real", k)][1], "realism_layer": Lr, "realism_C": sv["C"],
            "realism_at_monitor_layer_C": R.CS[c], "realism_at_monitor_layer_inner_auroc": auc.tolist(),
            "cos_std": float(v @ b_m / (v.norm() * b_m.norm())), "cos_raw": float(raw_r @ raw_m / (raw_r.norm() * raw_m.norm())),
            "coef": float(u @ b_m), "rows": np.flatnonzero(rows).tolist(), "monitor_score": (z @ b_m).cpu().tolist(),
            "proj": (z @ u).cpu().tolist(), "realism_score": s_r.cpu().tolist(),
            "realism_score_monitor_layer": score_at(rows, Lm, mu_r, sd_r, b_r).cpu().tolist()})
        del z
    out["seconds"] = round(time.time() - t0, 1)
    (d / ".result.json.tmp").write_text(json.dumps(out))
    (d / ".result.json.tmp").replace(d / "result.json")
    R.score_vol.commit()
    print(f"{key}: done in {out['seconds']} s", flush=True)
    return out


@app.function(image=image, gpu="H100", memory=65536, timeout=1500, retries=0, volumes={"/acts": R.acts_vol.read_only(), "/score": R.score_vol})
def alignment_only(cfg: dict, folds: dict, key: str) -> dict:
    """The command job's alignment terms after its wall stop (0d's OK, 12:33 PT): realism()'s per-fold block, unchanged, except
    that the realism score at the probe's own chosen layer is the held-out score saved in /score/<key>/stage_realism.json
    (fold k's real and simulated rows; NaN elsewhere, never used there). Writes /score/<key>/alignment.json."""
    import numpy as np
    import torch

    t0 = time.time()
    d = Path("/score") / key
    R.score_vol.reload()
    st = json.loads((d / "stage_realism.json").read_text())
    if (d / "alignment.json").exists():
        raise RuntimeError(f"{key}/alignment.json already exists")
    D = R.load(cfg, folds)
    if D["meta"] != st["meta"]:
        raise RuntimeError("the rows differ from the stage file's")
    X, fold, dev = D["X"], D["fold"], D["X"].device
    Dr = relabel(D, "sim", "real", "rs")
    pc = st["realism"]["cells"]["rs->rs"]["per_call"]
    held = dict(zip(pc["row"], pc["score"]))
    chosen = {f["fold"]: (f["layer"], f["C"]) for f in st["realism"]["cells"]["rs->rs"]["folds"] if f["threshold_kind"] == "calibrated"}

    def fit_at(rows, L, Cs):
        idx = torch.tensor(np.flatnonzero(rows), device=dev)
        w = torch.ones(len(idx), dtype=torch.float64, device=dev)
        y = torch.tensor(Dr["y"][rows], dtype=torch.float64, device=dev)
        Z = X[L: L + 1, idx].double()
        mu, sd = R.standardize(Z, w)
        Z.sub_(mu).div_(sd)
        B, kkt, _ = R.fit_primal(Z, y, w, Cs)
        for _ in range(R.RESTARTS):
            if not np.isfinite(kkt) or kkt <= 1e-4:
                break
            B, kkt, _ = R.fit_primal(Z, y, w, Cs, beta0=B)
        if not np.isfinite(kkt) or kkt > 1e-4:
            raise RuntimeError(f"primal fit not converged: relative KKT residual {kkt:.2e}")
        return mu[0, 0], sd[0, 0], B[0]

    def score_at(rows, L, mu, sd, beta):
        idx = torch.tensor(np.flatnonzero(rows), device=dev)
        return ((X[L, idx].double() - mu) / sd) @ beta

    P = np.load(Path("/score") / PROBES)
    mon = {(W, k): (L, C) for W, k, L, C in json.loads(str(P["meta"]))}
    inW = Dr["world"] == "rs"
    out = {"folds": []}
    for k in range(5):
        cal, T = (k + 1) % 5, [f for f in range(5) if f not in (k, (k + 1) % 5)]
        Lm = mon[("real", k)][0]
        mu_m, sd_m, b_m = (torch.tensor(P[f"real|{k}|{n}"], dtype=torch.float64, device=dev) for n in ("mu", "sd", "beta"))
        hs, ys = [], []
        for j in T:
            mu, sd, B = fit_at(inW & np.isin(fold, [t for t in T if t != j]), Lm, R.CS)
            ho = inW & (fold == j)
            hs.append(score_at(ho, Lm, mu, sd, B))
            ys.append(Dr["y"][ho])
        s = torch.cat(hs, 0).T
        yy = torch.tensor(np.concatenate(ys), dtype=torch.float64, device=dev)
        auc = R.auroc_batch(s, yy, torch.ones_like(yy)).cpu().numpy()
        c = int(np.flatnonzero(auc >= auc.max() - R.TIE).min())
        mu_r, sd_r, B_r = fit_at(inW & np.isin(fold, T), Lm, [R.CS[c]])
        b_r = B_r[:, 0]
        v = b_r * sd_m / sd_r
        u = v / v.norm()
        raw_r, raw_m = b_r / sd_r, b_m / sd_m
        rows = np.isin(fold, [k, cal])
        idx = torch.tensor(np.flatnonzero(rows), device=dev)
        z = (X[Lm, idx].double() - mu_m) / sd_m
        out["folds"].append({
            "fold": k, "monitor_layer": Lm, "monitor_C": mon[("real", k)][1], "realism_layer": chosen[k][0], "realism_C": chosen[k][1],
            "realism_at_monitor_layer_C": R.CS[c], "realism_at_monitor_layer_inner_auroc": auc.tolist(),
            "cos_std": float(v @ b_m / (v.norm() * b_m.norm())), "cos_raw": float(raw_r @ raw_m / (raw_r.norm() * raw_m.norm())),
            "coef": float(u @ b_m), "rows": np.flatnonzero(rows).tolist(), "monitor_score": (z @ b_m).cpu().tolist(),
            "proj": (z @ u).cpu().tolist(),
            "realism_score": [held.get(int(r), float("nan")) if fold[r] == k else float("nan") for r in np.flatnonzero(rows)],
            "realism_score_monitor_layer": score_at(rows, Lm, mu_r, sd_r, b_r).cpu().tolist()})
        del z
    out["seconds"] = round(time.time() - t0, 1)
    (d / ".alignment.json.tmp").write_text(json.dumps(out))
    (d / ".alignment.json.tmp").replace(d / "alignment.json")
    R.score_vol.commit()
    return {"seconds": out["seconds"], "folds": len(out["folds"])}


@app.function(image=image, cpu=8, memory=98304, timeout=3600, retries=0, volumes={"/acts": R.acts_vol.read_only(), "/score": R.score_vol})
def cache_rows(cfg: dict, folds: dict, name: str) -> dict:
    """The config's rows (load's selection, incl. readout_kind) read once on CPU and saved as /score/<name>/X.npy with
    rows.json (the rows' sha256, as load checks it), like the x caches the scorer reads. Never overwrites."""
    import numpy as np

    d = Path("/score") / name
    R.score_vol.reload()
    if d.exists():
        raise RuntimeError(f"{name} already exists")
    t0 = time.time()
    rows = R.load(cfg | {"_rows_only": True}, folds)["rows"]
    X = R.read_rows(rows, cfg["n_layers"], cfg["hidden"])
    d.mkdir(parents=True)
    np.save(d / "X.npy", X)
    (d / "rows.json").write_text(json.dumps({"rows_sha256": R.rows_sha(rows), "n_rows": len(rows)}))
    R.score_vol.commit()
    return {"name": name, "rows": len(rows), "shape": list(X.shape), "seconds": round(time.time() - t0, 1)}


if __name__ == "__main__":
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--watcher", required=True)
    ap.add_argument("--prior-usd", type=float, required=True)
    ap.add_argument("--cap-usd", type=float, default=9.00)  # the whole job's hard stop (0d, about 12:16 PT), counting the check
    ap.add_argument("--max-minutes", type=float, default=95)
    ap.add_argument("--n-null", type=int, default=6)
    ap.add_argument("--no-alignment", action="store_true", help="separation only (the reply-token arm)")
    ap.add_argument("--cache-as", default="", help="only build the config's row cache under this name, then exit")
    ap.add_argument("--alignment-for", default="", help="only run the alignment follow-up on this earlier job's key, then exit")
    a = ap.parse_args()
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=R.HERE).stdout.strip():
        raise SystemExit("the repo has uncommitted tracked changes")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=R.HERE).stdout.strip()
    cfg = yaml.safe_load((R.HERE / "config" / a.config).read_text())
    folds = R.frozen_folds(R.HERE.parents[1] / cfg["pilot_tasks"])
    ident = {"config": cfg, "folds": folds, "n_null": a.n_null, "git": git, "alignment": not a.no_alignment, "cache_as": a.cache_as, "alignment_for": a.alignment_for,
             "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    key = f"{cfg['label']}_realism_{hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:12]}"
    t0 = time.time()
    with modal.enable_output(), app.run(detach=True):
        app_id = app.app_id
        wlog = Path(a.watcher).with_name(f"watch_{key}.log")
        with open(wlog, "a") as fh:  # the watchdog's own uv environment outlives this launcher
            subprocess.Popen(["uv", "run", "-q", "--isolated", "--no-project", "--python", "3.14", "--with", "modal==1.6.1", "python",
                              a.watcher, app_id, str(a.prior_usd), str(a.cap_usd), str(a.max_minutes)],
                             stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        print(f"LAUNCHED app {app_id} key {key} at {datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(timespec='seconds')}; "
              f"watchdog log {wlog}", flush=True)
        if a.alignment_for:
            print(json.dumps(alignment_only.remote(cfg, folds, a.alignment_for)))
            raise SystemExit(0)
        if a.cache_as:
            print(json.dumps(cache_rows.remote(cfg, folds, a.cache_as)))
            raise SystemExit(0)
        r = realism.remote(cfg, folds, key, a.n_null, not a.no_alignment)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = R.HERE / "data" / f"realism_{cfg['label']}_{stamp}.json"  # per-row arrays: S3, not git
    path.parent.mkdir(exist_ok=True)
    with path.open("x") as fh:
        fh.write(json.dumps(r | {"git": git, "app": APP, "app_id": app_id, "key": key, "config": cfg, "folds_map": folds,
                                 "wall_seconds": round(time.time() - t0)}))
    print(json.dumps({"key": key, "app_id": app_id, "summary": r["summary"], "seconds": r["seconds"], "load_seconds": r["load_seconds"]}))
    print("->", path)
