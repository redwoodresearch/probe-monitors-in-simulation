"""R7b, the Hewitt and Liang control task (Aswin's ranked row 7; approved via 0d at the priced +$3.0 with 30%, stop $4).
EXPLORATORY. Output level, on the fixed-sim vs real e1 runs' kept output_end rows (step-table order of step B's level-1
pack), at four readings: L58 and L63 (step A's chosen layers, step B's pack), E (raw token embeddings) and B0 (block 0).
Three label columns fitted together, the same linear probe, folds and C grid as the grid copy:
  real     each output labelled by its run's arm (fixed sim = 1): the real task at output level;
  control  g(id) = sha256("l0ctl:<id>") mod 2 of the output's readout token id: Hewitt and Liang's control task, a fixed
           random function of the surface token (per-token-id variant);
  task     h(main task) = sha256("l0task:<main task>") mod 2: the per-main-task random label. At chance by construction:
           the folds hold out whole environments and each main task sits in one environment, so a test output's label is
           independent of everything the probe saw (expected AUROC 0.5 within each held-out environment).
Fitter as the grid copy (experiments/2026-10-10_claimB_qwen35_reader/run_claimB_grid.py): standardize on training rows,
features x sqrt(C), torch_probe.fit at C = 1, gradient check <= 1e-4·min(sqrt(C), 1), refits at 4x iterations up to
80,000; per held-out environment, C chosen per column on validation tasks (task fold = outer fold mod 5) among converged
fits; test AUROC pooled over the 5 test environments (and all 10). Selectivity = AUROC(real) - AUROC(control), per
reading. Also reported: the share of test outputs whose token type was seen in training. One (reading, C) record per
file on the Volume as it finishes; a rerun skips finished records.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' modal deploy experiments/2026-10-11_claimB_layer0_control/run_control_task_modal.py
  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' python experiments/2026-10-11_claimB_layer0_control/run_control_task_modal.py spawn
  ... run_control_task_modal.py collect <call id>
"""

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-claimb-layer0_ctl_kill-2026-11-11"
B_PACK, E_PACK, B0_PACK = "claimB/level1/packedB_2cfd1aec393c", "claimB/level1/packedB_d1260a6f6139", "claimB/level1/packedB_86e11a8a29de"
READINGS = {"L58": (B_PACK, 58), "L63": (B_PACK, 63), "E": (E_PACK, 0), "B0": (B0_PACK, 0)}
C_GRID, MAX_ITER, MAX_ITER_CAP = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], 5000, 80000
ARMS, POS, OUT = ("real-attack-e1", "sim-attack-fixed"), "output_end", "claimB/level1/control_task_v1"
TEST_ENVS = ["clinical_trial", "media_processing", "shared_dotfiles", "tiktok", "user_tracking"]
COLS = ["real", "control", "task"]

app = modal.App(APP)
vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy", "scipy")
         .add_local_python_source("probe_shift"))


def bit(prefix: str, x) -> int:
    return int(hashlib.sha256(f"{prefix}:{x}".encode()).hexdigest(), 16) % 2


def col_auc(Y, S):
    from scipy.stats import rankdata

    R = rankdata(S, axis=0)
    n1 = Y.sum(0)
    return ((R * Y).sum(0) - n1 * (n1 + 1) / 2) / (n1 * (len(Y) - n1))


@app.function(image=image, gpu="H100!", cpu=8, memory=98304, volumes={"/acts": vol}, timeout=45 * 60, retries=0, scaledown_window=5)
def fit_all(readings: list[str], folds_text: str, tag: str) -> list[dict]:
    """One H100 for all readings in turn: at about $5.1/h the 45-minute timeout caps the bill near $3.8 (stop $4)."""
    from probe_shift import torch_probe

    mine = tag_of(Path(__file__).read_bytes(), Path(torch_probe.__file__).read_bytes(), folds_text.encode())
    if mine != tag:  # the deployed code, fitter and folds must be the ones the launcher hashed
        raise ValueError(f"deployed tag {mine} != launcher tag {tag}")
    folds = json.loads(folds_text)
    e2f, t2f = folds["environment_folds"]["environment_to_fold"], folds["task_folds"]["main_task_to_fold"]
    return [fit_reading(r, e2f, t2f, tag) for r in readings]


def tag_of(script: bytes, fitter: bytes, folds: bytes) -> str:
    return hashlib.sha256(script + fitter + folds + json.dumps(READINGS).encode()).hexdigest()[:12]


def fit_reading(reading: str, e2f: dict, t2f: dict, tag: str) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F

    from probe_shift import torch_probe as tp
    from probe_shift.q1_eval import step_table

    A = Path("/acts")
    pack, layer = READINGS[reading]
    runs = json.loads((A / B_PACK / "runs.L58.json").read_text())
    if json.loads((A / pack / f"runs.L{layer}.json").read_text()) != runs:
        raise ValueError(f"{reading}: run list differs from step B's pack")
    t = step_table(runs, POS)
    ids = np.load(A / E_PACK / "token_ids.npy")
    X = np.load(A / pack / f"L{layer}_{POS}.npy")
    if not (len(X) == len(ids) == len(t["run"])):
        raise ValueError(f"{reading}: rows {len(X)}, token ids {len(ids)}, step table {len(t['run'])}")
    arm = np.array([runs[i]["arm"] for i in t["run"]])
    use = np.isin(arm, ARMS)
    env = np.array([runs[i]["environment"] for i in t["run"]])
    efold = np.array([e2f[e] for e in env])
    tfold = np.array([t2f[runs[i]["main_task"]] for i in t["run"]])
    Y = np.stack([(arm == "sim-attack-fixed").astype(float), [bit("l0ctl", int(x)) for x in ids],
                  [bit("l0task", runs[i]["main_task"]) for i in t["run"]]], 1)
    Xg = torch.from_numpy(X).cuda().float()
    idx = lambda m: torch.from_numpy(np.flatnonzero(m)).cuda()
    tol_of = lambda C: 1e-4 * min(C**0.5, 1.0)

    def grad_max(Xs, Ys, W, b):
        W, b = W.clone().requires_grad_(True), b.clone().requires_grad_(True)
        loss = (F.binary_cross_entropy_with_logits(Xs @ W + b, Ys, reduction="sum") + 0.5 * (W * W).sum()) / len(Xs)
        loss.backward()
        return torch.maximum(W.grad.abs().amax(0), b.grad.abs()).cpu().numpy()

    folds = sorted(set(efold[use]))
    for C in C_GRID:
        p = A / OUT / tag / reading / f"C{C:g}.npz"  # tag: code, folds and packs (a changed input never reuses a record)
        if p.exists():
            continue
        r = {"val_auroc": np.full((len(folds), 3), np.nan), "ok": np.zeros((len(folds), 3), bool), "test_score": np.full((len(X), 3), np.nan)}
        for i, f in enumerate(folds):
            tr, te = use & (efold != f), use & (efold == f)
            va = tr & (tfold == f % 5)
            ok = np.ones(3, bool)
            for name, a, b_ in (("full", tr, te), ("val", tr & ~va, va)):
                Xa, Xb = tp.standardize(Xg[idx(a)].double(), Xg[idx(b_)].double())
                Xa, Xb, Ya = Xa * C**0.5, Xb * C**0.5, torch.from_numpy(Y[a]).cuda().double()
                mi, prev = MAX_ITER, None
                while True:  # as the grid copy: refit only fits that used all their iterations
                    W, b, evals = tp.fit(Xa, Ya, 1.0, mi, tol=tol_of(C))
                    g = grad_max(Xa, Ya, W, b)
                    if g.max() <= tol_of(C) or evals < mi or evals == prev or mi * 4 > MAX_ITER_CAP:
                        break
                    mi, prev = mi * 4, evals
                ok &= g <= tol_of(C)
                s = tp.decision(Xb, W, b).cpu().numpy()
                if name == "full":
                    r["test_score"][b_] = s
                else:
                    r["val_auroc"][i] = col_auc(Y[b_], s)
                del Xa, Xb
            r["ok"][i] = ok
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p.with_suffix(".tmp"), "wb") as fh:
            np.savez(fh, folds=np.array(folds), **r)
        p.with_suffix(".tmp").replace(p)
        vol.commit()
        print(f"{reading} C {C:g} done", flush=True)
    R = {C: np.load(A / OUT / tag / reading / f"C{C:g}.npz") for C in C_GRID}
    if any(R[C]["folds"].tolist() != folds for C in C_GRID):
        raise ValueError(f"{reading}: a record's folds differ")
    V = np.stack([np.where(R[C]["ok"], R[C]["val_auroc"], -np.inf) for C in C_GRID])  # [C, fold, column]
    pick = V.argmax(0)
    valid = ~np.isinf(V).all(0).any(0)  # a column with any fold lacking a converged fit is not reported (as the grid)
    s = np.full((len(X), 3), np.nan)
    for i, f in enumerate(folds):
        m = use & (efold == f)
        for k in range(3):
            s[m, k] = R[C_GRID[pick[i, k]]]["test_score"][m, k]
    out = {"reading": reading, "rows_used": int(use.sum()), "folds": [int(x) for x in folds],
           "C_chosen": {COLS[k]: {int(f): C_GRID[pick[i, k]] for i, f in enumerate(folds)} for k in range(3)},
           "no_converged_fit": {COLS[k]: int(np.isinf(V[:, :, k]).all(0).sum()) for k in range(3)}, "tag": tag}
    for sub, m in (("test_envs", use & np.isin(env, TEST_ENVS)), ("all_envs", use)):
        a = col_auc(Y[m], s[m])
        au = {COLS[k]: (round(float(a[k]), 4) if valid[k] else "invalid (a fold had no converged fit)") for k in range(3)}
        out[sub] = {"auroc": au, "selectivity": round(float(a[0] - a[1]), 4) if valid[0] and valid[1] else None, "outputs": int(m.sum())}
        seen = sum(int(np.isin(ids[m & (efold == f)], ids[use & (efold != f)]).sum()) for f in folds)
        out[sub]["test_token_type_seen_share"] = round(seen / int(m.sum()), 4)
    out["task_label_by_environment"] = {}  # the per-main-task label's chance claim holds within each held-out environment
    for e in sorted(set(env[use])):
        m = use & (env == e)
        n1 = int(Y[m, 2].sum())
        out["task_label_by_environment"][e] = {"auroc": round(float(col_auc(Y[m][:, 2:], s[m][:, 2:])[0]), 4) if 0 < n1 < m.sum() and valid[2] else None,
                                               "outputs_label1": n1, "outputs_label0": int(m.sum()) - n1}
    (A / OUT / f"{reading}_summary.json").write_text(json.dumps(out))
    vol.commit()
    return out


def main():
    mode, rest = sys.argv[1], sys.argv[2:]
    if mode == "spawn":
        root = HERE.parents[1]  # local only
        folds_text = (root / "splits" / "folds_v2.json").read_text()
        tag = tag_of(Path(__file__).read_bytes(), (root / "src/probe_shift/torch_probe.py").read_bytes(), folds_text.encode())
        f = modal.Function.from_name(APP, "fit_all")
        print(json.dumps({"tag": tag, "call": f.spawn(rest or list(READINGS), folds_text, tag).object_id}))
        return
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    res = {r["reading"]: r for cid in rest for r in modal.FunctionCall.from_id(cid).get()}
    out = {"label": "R7b control task (EXPLORATORY)", "calls": res,
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    dst = HERE / "results" / f"control_task_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out) + "\n")
    for r in res.values():
        print(f"EXPLORATORY | {r['reading']} test envs: {r['test_envs']} | all envs: {r['all_envs']} | C {r['C_chosen']} | no converged {r['no_converged_fit']}")
    print("->", dst)


if __name__ == "__main__":
    main()
