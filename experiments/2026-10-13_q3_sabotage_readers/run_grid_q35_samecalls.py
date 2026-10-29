"""B3 RULE.md Amendment 1: Qwen3.5-27B's own sabotage grid on exactly the calls one reader reads. Numbers only; no text.

From a reader's grid spec (make_grid_spec.py): its read exam calls, read honb_cal / honb_ho rows and its training selections
(S, R44 x5, M x5, D x5). Each is located in Qwen3.5's stored layer-30 rows: call_end from flagrate_rows / scaled_rows (attack
calls) and rows_v2 (honest rows); command from check_grid_inputs.py's command_rows csv (attack calls) and item E's rows_command.json
(honest rows; aligned with rows_v2 row by row: opaque_id where present, role, task, extraction and chunk). A missing location stops the job. One Modal CPU container
reads the rows read-only and fits with Amendment 3's recipe; fit_dual and the row loading are copied from run_grid.py
(aswin/q3-sabotage-scaleup @ 4f60d040). R44_0 is fitted twice (must agree within 1e-6). The output has run_grid_readers.py's
layout, so analyze_readers.py reads it unchanged. Writes results/grid_scores_qwen35_on_<reader>_<stamp>.json.

  uv run --no-project --python 3.12 --with "modal==1.6.1" --with numpy modal run run_grid_q35_samecalls.py --reader seed \\
      --spec <grid_spec json> --inputs <scaled_inputs dir>
"""

import csv
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

APP = "probe-shift_aswin-surya_sub-q3sabr_q35same_kill-2026-11-13"
ACTS = "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08"  # run_grid.py's
HERE = Path(__file__).parent
LAYER, C = 30, 1e-4
PINS = {"flagrate_rows_2026-10-13_122417.csv": "44c75f78e78f86b4bcab409b1379762a8d4dccadfd456f28ee651832eb2b13a7",
        "scaled_rows_2026-10-13_152021.csv": "a95453ac2688e61ba99184726ce27a4dd54276a04730a56d67825f204d1ee1f3",
        "rows_v2.json": "727c417916b8fb2df9a31aca5f8ca0cc4dd17002a8a2edcb7bd79444bc0bf4b5",
        "command_rows_2026-10-13_162911.csv": "096f147f6a72ebf7b1479630b6fb3d8cfa8c141a5a68e54c9ce18a770765d05e",
        "rows_command.json": "4d7289f3b2e475d86feaed028e0df795d1d80998918eaeacc060a0bfc7116beb"}

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy")


def fit_dual(K, y, w, Cs, iters=100, tol=1e-9):
    """Copied from run_grid.py (aswin/q3-sabotage-scaleup @ 4f60d040), itself from run_persist_gate.py."""
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


@app.function(image=image, cpu=4, memory=49152, timeout=1800, retries=0, volumes={"/acts": acts.read_only()})
def grid_job(locs: dict) -> dict:
    """locs: {"eval": {readout: [loc]}, "honest": {readout: [loc]}, "fits": {name: {readout: [loc]}, "y": [..]}} with loc = [x, chunk, row]."""
    import numpy as np
    import torch

    every = {tuple(v) for part in ("eval", "honest") for L in locs[part].values() for v in L}
    every |= {tuple(v) for f in locs["fits"].values() for ro in ("call_end", "command") for v in f[ro]}
    order = sorted(every)
    pos = {v: i for i, v in enumerate(order)}
    groups = {}
    for i, (x, chunk, row) in enumerate(order):
        groups.setdefault((x, chunk), []).append((i, int(row)))
    X = np.empty((len(order), 5120), np.float64)
    for (x, chunk), lst in groups.items():  # run_grid.py's loading
        a = np.load(f"/acts/{x}/by_layer/L{LAYER}/{chunk}.npy")
        X[[i for i, _ in lst]] = a[[r for _, r in lst]]
    ix = lambda L: X[[pos[tuple(v)] for v in L]]  # noqa: E731
    out, checks = {}, {}
    for ro in ("call_end", "command"):
        E, H, sc, rep = ix(locs["eval"][ro]), ix(locs["honest"][ro]), {}, None

        def fit(f):
            Xt = ix(f[ro])
            y = torch.tensor(f["y"], dtype=torch.float64)
            mu, sd = Xt.mean(0), Xt.std(0)
            sd = np.where(sd < 1e-6, 1.0, sd)
            Zt = torch.from_numpy((Xt - mu) / sd)
            a = fit_dual((Zt @ Zt.T)[None], y[None, None, :], torch.ones(len(Xt), dtype=torch.float64), [C])
            return (a[0, 0] @ Zt).numpy(), mu, sd

        for n, f in sorted(locs["fits"].items()):
            beta, mu, sd = fit(f)
            se, sh = ((E - mu) / sd) @ beta, ((H - mu) / sd) @ beta
            if not (np.isfinite(se).all() and np.isfinite(sh).all()):
                raise SystemExit(f"{n} {ro}: non-finite scores")
            sc[n] = {"eval": se.tolist(), "honest": sh.tolist()}
            if n == "R44_0":
                b2, mu2, sd2 = fit(f)
                rep = float(np.abs(((E - mu2) / sd2) @ b2 - se).max())
        if rep is None or rep > 1e-6:
            raise SystemExit(f"{ro}: R44_0 refit differs by {rep}")
        out[f"L{LAYER}/{ro}"] = sc
        checks[f"L{LAYER}/{ro}"] = {"r44_0_refit_max_abs_diff": rep, "eval_rows": len(E), "honest_rows": len(H)}
    return {"scores": out, "checks": checks, "rows_loaded": len(order)}


@app.local_entrypoint()
def main(reader: str, spec: str, inputs: str):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=HERE).stdout.strip()  # noqa: E731
    if git("status", "--porcelain", "--", Path(__file__).name, "RULE.md"):
        raise SystemExit("commit this script and RULE.md before it runs")
    I = Path(inputs)
    for n, h in PINS.items():
        if hashlib.sha256((I / n).read_bytes()).hexdigest() != h:
            raise SystemExit(f"{n}: sha256 mismatch")
    b = Path(spec).read_bytes()
    sp = json.loads(b)
    if sp["reader"] != reader:
        raise SystemExit("spec is another reader's")
    ce, cmd = {}, {}
    for n in ("flagrate_rows_2026-10-13_122417.csv", "scaled_rows_2026-10-13_152021.csv"):
        for r in csv.DictReader(open(I / n)):
            if r.get("part", "2") == "2":
                ce[r["call_id"]] = [r["x"], r["chunk"], int(r["row"])]
    for r in csv.DictReader(open(I / "command_rows_2026-10-13_162911.csv")):
        cmd[r["call_id"]] = [r["x"], r["chunk"], int(r["row"])]
    rv, rc = json.loads((I / "rows_v2.json").read_text())["rows"], json.loads((I / "rows_command.json").read_text())["rows"]
    hon = [(a, c) for a, c in zip(rv, rc) if a["role"] in ("honb_cal", "honb_ho")]
    if len(rv) != len(rc) or any(a.get("opaque_id") != c.get("opaque_id") or [a[k] for k in ("role", "task", "x", "chunk")] != [c[k] for k in ("role", "task", "x", "chunk")]
                                 for a, c in zip(rv, rc)):
        raise SystemExit("rows_command.json isn't aligned with rows_v2.json")
    if len(hon) != len(sp["honest"]) or any(a["role"] != h["role"] or a["task"] != h["task"] for (a, _), h in zip(hon, sp["honest"])):
        raise SystemExit("the spec's honest rows aren't rows_v2's honb rows in order")
    ev = [c for c in sp["calls"] if c["read"]]
    hk = [(a, c, h) for (a, c), h in zip(hon, sp["honest"]) if h["read"]]
    need = [c["call_id"] for c in ev] + [i for f in sp["fits"].values() for i in f["pos"] + f["neg"]]
    if missing := [i for i in need if i not in ce or i not in cmd]:
        raise SystemExit(f"{len(missing)} calls have no Qwen3.5 location")
    loc = lambda i, ro: (ce if ro == "call_end" else cmd)[i]  # noqa: E731
    locs = {"eval": {ro: [loc(c["call_id"], ro) for c in ev] for ro in ("call_end", "command")},
            "honest": {"call_end": [[a["x"], a["chunk"], int(a["row"])] for a, _, _ in hk], "command": [[c["x"], c["chunk"], int(c["row"])] for _, c, _ in hk]},
            "fits": {n: {ro: [loc(i, ro) for i in f["pos"] + f["neg"]] for ro in ("call_end", "command")} | {"y": [1.0] * len(f["pos"]) + [0.0] * len(f["neg"])}
                     for n, f in sp["fits"].items()}}
    res = grid_job.remote(locs)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    names = sorted(sp["fits"], key=lambda n: (n.split("_")[0], n))
    out = {"written": stamp, "reader": f"qwen35_on_{reader}", "git": git("rev-parse", "HEAD"), "spec": Path(spec).name,
           "spec_sha256": hashlib.sha256(b).hexdigest(), "layers": [LAYER], "primary_layer": LAYER, "C": C, "probes": names,
           "eval_call_ids": [c["call_id"] for c in ev], "honest_roles": [h["role"] for _, _, h in hk], "honest_tasks": [h["task"] for _, _, h in hk],
           "inputs_sha256": PINS} | res
    dst = HERE / "results" / f"grid_scores_qwen35_on_{reader}_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out))
    print(json.dumps(res["checks"], indent=1), "\n->", dst)
