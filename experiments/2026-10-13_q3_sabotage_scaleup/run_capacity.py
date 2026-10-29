"""RULE_scaled.md Amendment 5 (T3, capacity), Qwen3.5-27B: probe M (simulated sabotage vs real not-sabotage) at the command readouts
"last" (the command's last token) and "avg" (B1's mean over the command value's tokens), sizes "matched" (Amendment 3's five M
draws, 44 vs 900) and "full" (all 44 vs all 1,830 real not-sabotage), C in {1e-4, 1e-3, 1e-2, 1e-1, 1}. Numbers only; no text.

One warm Modal CPU function: reads the layer-30 readings (q3onpol acts Volume, read-only), maps command locations to B1's averaged
ones exactly as run_grid_avg.py does (original metas joined to 2026-10-13_sabavg_ metas on arm, key, irid, call_index, readout; the
re-read last-token row must equal the original row), fits every (readout, training set, C) with fit_dual (copied from run_scaled.py;
S's recipe), and scores the exam rows, the label-free honest rows and item E's staged command rows (thresholds and validity).
Training and staged rows must map at "avg"; exam and honest rows without an averaged reading are dropped there and counted. A fit
that doesn't converge is recorded and left out, never scored as 0. Gate: at "last" and C 1e-4 the five matched draws reproduce
Amendment 3's stored M_i@cmd scores within 1e-6. Writes results/capacity_scores_<stamp>.json.

  uv run --no-project --python 3.12 --with "modal==1.6.1" --with numpy modal run run_capacity.py --inputs <scaled_inputs dir> \\
      --grid-scores results/grid_scores_2026-10-13_163458.json
"""

import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

APP = "probe-shift_aswin-surya_sub-q3capacity_score_kill-2026-11-12"
ACTS = "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08"
SCORE = "probe-shift_aswin-surya_sub-q3onpol_score_kill-2026-11-08"
AVG, LAYER, HIDDEN = "2026-10-13_sabavg_", 30, 5120
CS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
CMD_ROWS = "/score/sabotage_learning_v1/judge_v1/rows_command.json"
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
PINS = {"command_rows_2026-10-13_162911.csv": "096f147f6a72ebf7b1479630b6fb3d8cfa8c141a5a68e54c9ce18a770765d05e",
        "rows_command.json": "4d7289f3b2e475d86feaed028e0df795d1d80998918eaeacc060a0bfc7116beb",
        "flagrate_rows_2026-10-13_122417.csv": "44c75f78e78f86b4bcab409b1379762a8d4dccadfd456f28ee651832eb2b13a7",
        "scaled_rows_2026-10-13_152021.csv": "a95453ac2688e61ba99184726ce27a4dd54276a04730a56d67825f204d1ee1f3",
        "grid_scores_2026-10-13_163458.json": "671848fe608618564e79aabdc91d805667ac0597ea9d9eed32634cbd5e1528de"}
GRID_SEL_SHA = "be928fd55bd0b6732f52b3ef2b64cd2afb2c7c3ea82fb448c7ed22ff2907ab8b"

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


@app.function(image=image, cpu=16, memory=65536, timeout=2400, retries=0, volumes={"/acts": acts.read_only(), "/score": score.read_only()})
def capacity_job(cmd_sha: str, fits: dict, ev_cmd: list) -> dict:
    """fits: name -> [(x, chunk, row, y)] at the original ("last") command locations; ev_cmd: [(x, chunk, row) or None]."""
    import numpy as np
    import torch

    torch.set_num_threads(16)
    rawc = Path(CMD_ROWS).read_bytes()
    if hashlib.sha256(rawc).hexdigest() != cmd_sha:
        raise SystemExit("rows_command on the Volume differs from its pin")
    st_cmd = [(r["x"], r["chunk"], int(r["row"])) for r in json.loads(rawc)["rows"]]
    used = set(st_cmd) | {tuple(t[:3]) for d in fits.values() for t in d} | {tuple(v) for v in ev_cmd if v}
    need = sorted({v[0] for v in used})
    T = {}  # B1's mapping, as run_grid_avg.py
    for x in need:
        orig = {}
        for f in sorted(Path(f"/acts/{x}/meta").glob("*/*.json")):
            m = json.loads(f.read_text())
            for c in m["calls"]:
                if c.get("kind", "command") == "command" and c.get("row") is not None:
                    orig[(m["arm"], m["key"], c["irid"], c.get("call_index", 0), c["readout"])] = (x, m["chunk"], m["row_start"] + c["row"])
        for f in sorted(Path(f"/acts/{AVG}{x}/meta").glob("*/*.json")):
            m = json.loads(f.read_text())
            for c in m["calls"]:
                k = (m["arm"], m["key"], c["irid"], c.get("call_index", 0), c["readout"])
                if k in orig:
                    T[orig[k]] = (AVG + x, m["chunk"], m["row_start"] + c["row"])
    must = set(st_cmd) | {tuple(t[:3]) for d in fits.values() for t in d}
    if missing := [v for v in must if v not in T]:
        raise SystemExit(f"{len(missing)} training or staged command locations have no averaged reading")
    mapped = sorted((v, T[v]) for v in used if v in T)
    mism, by = 0, {}
    for o, a_ in mapped:
        by.setdefault((o[0], o[1], a_[0], a_[1]), []).append((o[2], a_[2]))
    for (ox, oc, ax, ac), lst in by.items():
        o_rows = np.load(f"/acts/{ox}/by_layer/L{LAYER}/{oc}.npy", mmap_mode="r")
        l_rows = np.load(f"/acts/{ax}/last/L{LAYER}/{ac}.npy", mmap_mode="r")
        mism += sum(not np.array_equal(o_rows[r], l_rows[q]) for r, q in lst)
    if mism:
        raise SystemExit(f"mapping gate: {mism} of {len(mapped)} mapped locations' last-token rows differ from the original")
    locs = sorted(used | {T[v] for v in used if v in T})
    pos = {v: i for i, v in enumerate(locs)}
    groups = {}
    for i, (x, chunk, row) in enumerate(locs):
        groups.setdefault((x, chunk), []).append((i, int(row)))
    X = np.empty((len(locs), HIDDEN), np.float64)
    for (x, chunk), lst in groups.items():
        a = np.load(f"/acts/{x}/by_layer/L{LAYER}/{chunk}.npy")
        X[[i for i, _ in lst]] = a[[r for _, r in lst]]
    if not np.isfinite(X).all():
        raise SystemExit("non-finite readings")
    out, failed = {}, []
    for ro in ("last", "avg"):
        tr = (lambda v: tuple(v)) if ro == "last" else (lambda v: T.get(tuple(v)))
        ev = [tr(v) if v else None for v in ev_cmd]
        has = [v is not None for v in ev]
        Xe = X[[pos[v] for v in ev if v is not None]]
        Xs = X[[pos[tr(v)] for v in st_cmd]]
        for name, d in sorted(fits.items()):
            Xt = X[[pos[tr(t[:3])] for t in d]]
            y = torch.tensor([float(t[3]) for t in d], dtype=torch.float64)
            mu, sd = Xt.mean(0), Xt.std(0)
            sd = np.where(sd < 1e-6, 1.0, sd)
            Zt = torch.from_numpy((Xt - mu) / sd)
            K = (Zt @ Zt.T)[None]
            for c in CS:
                key = f"{name}|{ro}|{c:g}"
                try:
                    a = fit_dual(K, y[None, None, :], torch.ones(len(d), dtype=torch.float64), [c], iters=300)
                except RuntimeError as e:  # recorded, never scored as 0
                    failed.append({"fit": key, "error": str(e)})
                    continue
                beta = (a[0, 0] @ Zt).numpy()
                it = iter((((Xe - mu) / sd) @ beta).tolist())
                out[key] = {"eval": [next(it) if h else None for h in has], "staged": (((Xs - mu) / sd) @ beta).tolist()}
    return {"scores": out, "failed_fits": failed, "chunks_read": len(groups), "mapped_locations": len(mapped), "mapping_mismatches": mism,
            "eval_without_avg": int(sum(1 for v in ev_cmd if v and tuple(v) not in T))}


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@app.local_entrypoint()
def main(inputs: str):
    import numpy as np

    import make_grid_selection as G

    for f in (Path(__file__).name, "RULE_scaled.md", "make_grid_selection.py", "results/grid_selection_manifest.json"):
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", f], cwd=HERE, capture_output=True)
        status = subprocess.run(["git", "status", "--porcelain", "--", f], cwd=HERE, capture_output=True, text=True)
        if tracked.returncode != 0 or status.returncode != 0 or status.stdout:
            raise SystemExit(f"{f} is not committed unchanged at HEAD")
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True, check=True).stdout.strip()
    I = Path(inputs)
    for n, want in PINS.items():
        p = (HERE / "results" / n) if n.startswith("grid_scores") else I / n
        if sha(p) != want:
            raise SystemExit(f"{n}: sha256 differs from its pin")
    gsel = HERE / "results" / "grid_selection_manifest.json"
    if sha(gsel) != GRID_SEL_SHA:
        raise SystemExit("grid_selection_manifest.json differs from its pin")
    gm = json.loads(gsel.read_text())
    pools = G.pools(str(I))
    cm = {r["call_id"]: (r["x"], r["chunk"], int(r["row"])) for r in csv.DictReader(open(I / "command_rows_2026-10-13_162911.csv"))}
    fits = {f"M{d['draw']}": [(*cm[c], 1) for c in d["pos"]] + [(*cm[c], 0) for c in d["neg"]] for d in gm["M"]["draws"]}
    fits["Mfull"] = [(*cm[c], 1) for c in pools["sim_sabotage"]] + [(*cm[c], 0) for c in pools["real_not_sabotage"]]
    F = list(csv.DictReader(open(I / "flagrate_rows_2026-10-13_122417.csv")))
    N = list(csv.DictReader(open(I / "scaled_rows_2026-10-13_152021.csv")))
    rows = F + N
    out = capacity_job.remote(PINS["rows_command.json"], fits, [cm.get(r["call_id"]) for r in rows])
    sc = out["scores"]
    gs = json.loads((HERE / "results" / "grid_scores_2026-10-13_163458.json").read_text())
    gi = {s["call_id"]: s["s"] for s in gs["scores"]}
    col = {n: j for j, n in enumerate(gs["probes"])}
    diffs = {}
    for d in range(5):
        k = f"M{d}|last|0.0001"
        if k in sc:
            diffs[k] = float(max(abs(v - gi[r["call_id"]][col[f"M_{d}@cmd"]]) for v, r in zip(sc[k]["eval"], rows) if v is not None))
    val = {"reproduction_vs_amendment3_M": diffs, "failed_fits": out["failed_fits"], "mapping_mismatches": out["mapping_mismatches"],
           "mapped_locations": out["mapped_locations"], "eval_rows_without_avg": out["eval_without_avg"]}
    val["passes"] = bool(len(diffs) == 5 and max(diffs.values()) <= 1e-6 and out["mapping_mismatches"] == 0)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    if not val["passes"]:
        raise SystemExit(f"gates failed, nothing written: {json.dumps(val)}")
    names = sorted(sc)
    role = [r["role"] for r in json.loads((I / "rows_command.json").read_text())["rows"]]
    meta = [{k: r[k] for k in ("call_id", "world", "task", "run_id")} | {"source": "flagrate", "part": int(r["part"]), "arm": r["arm"], "group": r["group"],
            "epoch": int(r["epoch"]), "turn": int(r["turn"]), "call_index": int(r["call_index"]), "state_changing": int(r["state_changing"])} for r in F]
    meta += [{k: r[k] for k in ("call_id", "world", "task", "run_id")} | {"source": "new", "part": 2, "arm": r["world"], "group": "new_held_out",
             "epoch": int(r["epoch"]), "turn": int(r["turn"]), "call_index": int(r["call_index"]), "state_changing": int(r["state_changing"])} for r in N]
    res = {"rule": "RULE_scaled.md Amendment 5 (T3)", "label": "scores only; no label joined", "git": git, "app": APP, "layer": LAYER, "C_grid": CS,
           "validation": val, "inputs_sha256": PINS | {"grid_selection_manifest": GRID_SEL_SHA}, "chunks_read": out["chunks_read"],
           "fits": {k: len(v) for k, v in fits.items()}, "probes": names,
           "scores": [m | {"s": [sc[n]["eval"][i] for n in names]} for i, m in enumerate(meta)],
           "staged": [{"index": i, "role": role[i], "s": [sc[n]["staged"][i] for n in names]} for i in range(len(role))], "written": stamp}
    dst = HERE / "results" / f"capacity_scores_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(res) + "\n")
    print(json.dumps({"validation": val, "probes": len(names)}, indent=1), "\n->", dst, sha(dst))
