"""Control 4b: refit the #36 tool-output probe at L63 without the 200 base200 real honest runs, then rescore. Modal GPU
(A10G, cap $2; Aswin's go via the orchestrator). #36's script is not changed: this reuses its recipe on its packed data.

- Data: #36's data 307cbfee925c, packed L63 output_end features (`packed/L63_output_end.npy`, one row per tool output).
- Labels: real attack e1 = 1, real honest = 0, as #36, except that real honest runs whose key starts with base200 are
  dropped from training (they are still scored).
- Recipe: #36's (config/q3_attack.yaml fit: standardized L2 logistic regression, C = 0.01, max_iter 1000, torch_probe),
  task folds: 5 outer and 20 nested models, mean pooling.
- No label-shuffle null: the refit is reported only as a comparison against the committed detector (how the shortfall
  and the swap headline change when base200 is not trained on), with no claim about chance. #36's null covers the
  detector itself.
- Scored: every run of the data (outer and nested run means), and the both-fixes swap arms (the combined extraction's
  100 runs and the extension's 117) with the model of each run's fold.
- Gate: the same refit with base200 kept must reproduce #36's committed L63 output_end scores (scores_L63.npz), so the
  dropped-base200 refit differs from the detector in the training set only.
Writes run-level scores only; analyze_refit_no_base200.py applies the frozen protocol.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.2' --with numpy modal run \
      experiments/2026-10-08_q3_swap_cue/run_refit_no_base200.py
"""

import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q3swapcue_refit_kill-2026-11-06"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
WORK = "probe-shift_aswin-surya_sub-q3attack_work_kill-2026-11-06"
DATA_ID, LAYER, C, MAX_ITER = "307cbfee925c", 63, 0.01, 1000
BASE_RUN = "2026-10-06_181548"  # #36's committed scores, for the gate
GATE_MAX_ABS = 0.05  # logits, on #36's own hardware (H100). An A10G run differed by up to 0.13 (corr 0.9998), so it failed this gate
EXPECTED = {"COMB": 100, "COMB_ext": 117}
FIXED = {"COMB": ("2026-10-08_q2_swap_combined_qwen3-32b_l48-63_v1", "sim-attack-note-sr-both"),
         "COMB_ext": ("2026-10-08_q2_swap_ext_qwen3-32b_l48-63_v1", "sim-attack-note-sr-both")}

app = modal.App(APP)
acts_vol = modal.Volume.from_name(ACTS, version=2)
work_vol = modal.Volume.from_name(WORK, version=2)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "torch==2.8.0")
         .add_local_python_source("probe_shift"))


@app.function(image=image, gpu="H100!", cpu=4, memory=65536, timeout=600,  # #36's hardware; each refit takes under a minute, 10-min cap
              volumes={"/acts": acts_vol.read_only(), "/work": work_vol.read_only()})
def refit(drop_base200: bool) -> dict:
    import numpy as np
    import torch

    from probe_shift import torch_probe as tp

    t0 = time.time()
    d = Path("/work/data") / DATA_ID
    runs = json.loads((d / "runs.json").read_text())
    t = np.load(d / "tables.npz")
    X = torch.from_numpy(np.load(d / "packed" / f"L{LAYER}_output_end.npy")).cuda().float()
    r_run = t["output_end__run"]
    if X.shape[0] != len(r_run):
        raise ValueError(f"{X.shape[0]} packed rows, {len(r_run)} in the table")
    n = len(runs)
    arm = np.array([r["arm"] for r in runs])
    fold = np.array([r["fold"] for r in runs])
    base200 = np.array([r["arm"] == "real-honest" and r["key"].startswith("base200") for r in runs])
    y = np.where(arm == "real-attack-e1", 1.0, np.where(arm == "real-honest", 0.0, np.nan))
    trainable = ~np.isnan(y) & (~base200 if drop_base200 else True)
    fix_rows = {}  # label -> (key, fold, activation rows)
    for label, (run, a) in FIXED.items():
        rows = []
        fold_of = {r["key"]: r["fold"] for r in runs if r["arm"] == "sim-attack"}
        for mp in sorted((Path("/acts") / run / "meta" / a).glob("*.json")):
            m = json.loads(mp.read_text())
            pos = [p["row"] for p in m["positions"] if p["kind"] == "output_end" and p["kept"]]
            chunk = np.load(Path("/acts") / run / "by_layer" / f"L{LAYER}" / f"{m['chunk']}.npy", mmap_mode="r")
            rows.append((m["key"], fold_of[m["key"]], np.asarray(chunk[[m["row_start"] + r for r in pos]], np.float32)))
        if len(rows) != EXPECTED[label] or any(len(x) == 0 for _, _, x in rows):
            raise ValueError(f"{label}: {len(rows)} runs (expected {EXPECTED[label]}), {sum(len(x) == 0 for _, _, x in rows)} without kept outputs")
        fix_rows[label] = rows
    fits, outer, nested = [], np.full(n, np.nan), np.full((5, 5, n), np.nan)
    fixed = {label: {} for label in FIXED}  # label -> key -> {"outer": s, "nested": {v: s}}

    def fit_score(train_runs, score_runs, name):
        tr = np.flatnonzero(train_runs[r_run])
        te = np.flatnonzero(score_runs[r_run])
        Xtr = X[torch.from_numpy(tr).cuda()]
        mean, std = Xtr.mean(0), Xtr.std(0, unbiased=False)
        std = std.where(std > 0, std.new_ones(()))
        W, b, n_eval = tp.fit((Xtr - mean) / std, torch.from_numpy(y[r_run[tr]]).cuda().float()[:, None], C, MAX_ITER)
        s = tp.decision((X[torch.from_numpy(te).cuda()] - mean) / std, W, b).double().cpu().numpy()[:, 0]
        sums = np.bincount(r_run[te], weights=s, minlength=n)
        cnt = np.bincount(r_run[te], minlength=n)
        fits.append({"model": name, "train_rows": int(len(tr)), "train_runs": int(train_runs.sum()), "n_eval": int(n_eval),
                     "may_have_hit_cap": bool(n_eval >= MAX_ITER)})
        score_fixed = lambda x: float(tp.decision((torch.from_numpy(x).cuda() - mean) / std, W, b).double().cpu().numpy()[:, 0].mean())  # noqa: E731
        return np.where(cnt > 0, sums / np.maximum(cnt, 1), np.nan), score_fixed

    for f in range(5):
        rs, sf = fit_score(trainable & (fold != f), fold == f, f"task_outer_{f}")
        outer[fold == f] = rs[fold == f]
        for label, rows in fix_rows.items():
            for key, fk, x in rows:
                if fk == f and len(x):
                    fixed[label].setdefault(key, {"nested": {}})["outer"] = sf(x)
    for f in range(5):
        for v in range(5):
            if v == f:
                continue
            te = (fold == f) | (fold == v)
            rs, sf = fit_score(trainable & ~te, te, f"task_nested_{f}_{v}")
            nested[f, v][te] = rs[te]
            for label, rows in fix_rows.items():
                for key, fk, x in rows:
                    if fk == f and len(x):
                        fixed[label].setdefault(key, {"nested": {}})["nested"][str(v)] = sf(x)
    for label, rows in fix_rows.items():  # every fixed-arm run: an outer score and four nested scores, all finite
        for key, fk, _ in rows:
            r = fixed[label].get(key, {})
            vals = [r.get("outer")] + [r.get("nested", {}).get(str(v)) for v in range(5) if v != fk]
            if any(x is None or not np.isfinite(x) for x in vals):
                raise ValueError(f"{label} {key}: incomplete or nonfinite scores {vals}")
    gate = None
    if not drop_base200:  # the same recipe with base200 kept must reproduce #36's committed scores
        z = np.load(Path("/work/runs") / BASE_RUN / f"scores_L{LAYER}.npz")
        co, cn = z["output_end__mean__task_outer"].astype(np.float64), z["output_end__mean__task_nested"].astype(np.float64)
        if co.shape != outer.shape or cn.shape != nested.shape or (np.isnan(co) != np.isnan(outer)).any() or (np.isnan(cn) != np.isnan(nested)).any():
            raise ValueError("gate: shapes or missingness differ from the committed scores")
        m, mn = ~np.isnan(co), ~np.isnan(cn)
        gate = {"outer_max_abs": float(np.abs(co[m] - outer[m]).max()), "nested_max_abs": float(np.abs(cn[mn] - nested[mn]).max()),
                "outer_corr": float(np.corrcoef(co[m], outer[m])[0, 1])}
        if gate["outer_max_abs"] > GATE_MAX_ABS or gate["nested_max_abs"] > GATE_MAX_ABS:
            raise ValueError(f"gate failed: {gate}")
    if not np.isfinite(outer[~np.isnan(outer)]).all():
        raise ValueError("nonfinite outer scores")
    return {"drop_base200": drop_base200, "gate": gate, "seconds": round(time.time() - t0), "fits": fits, "trained_honest": int((trainable & (arm == "real-honest")).sum()),
            "dropped_base200_honest": int(base200.sum()), "outer": [None if np.isnan(x) else float(x) for x in outer],
            "nested": [[[None if np.isnan(x) else float(x) for x in nested[f, v]] for v in range(5)] for f in range(5)], "fixed": fixed}


@app.local_entrypoint()
def main():
    import gzip

    t0 = datetime.now(ZoneInfo("America/Los_Angeles"))
    print("start", t0.isoformat(), flush=True)
    outs = list(refit.map([True, False]))  # dropped, and the gate (base200 kept)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"refit_no_base200_{stamp}.json.gz"
    with gzip.open(path, "xt") as fh:  # run-level scores only
        json.dump({"stamp": stamp, "start": t0.isoformat(), "data_id": DATA_ID, "layer": LAYER, "C": C, "max_iter": MAX_ITER,
                   "fixed_arms": FIXED, "refits": outs}, fh)
    for out in outs:
        bad = [x for x in out["fits"] if x["may_have_hit_cap"]]
        print("drop_base200", out["drop_base200"], "gate", out["gate"], f"{out['seconds']} s;", f"{len(bad)} fits may have hit the cap;", "trained honest", out["trained_honest"])
    print("stop", datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(), "->", path)
