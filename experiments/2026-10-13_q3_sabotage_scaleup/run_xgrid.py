"""RULE_scaled.md Amendment 4: the R/S/R44/M/D grid with Qwen3-32B reading Qwen3.5-27B's acting runs (the cross-read extraction), at
the command readout, layer 63, C 0.01. Numbers only; no text.

Two remote steps in one run:
1. keys_job reads the cross-read metas (q3xread acts Volume: run 1, run 2, simulated, honest_a, honest_b; redrawn and rerun
   attempts skipped) and returns each command entry's location by (world, task, turn, call_index), plus the metas' digests.
2. fit_job loads layer-63 readings at those locations, fits S, R44 (5 draws), M (5), D (5) with S's recipe (unit weights,
   standardization on the draw's rows), R with the gate's recipe (its staged training rows, inclusion weights, weighted
   standardization), all at C 0.01, refits R44 draw 0 twice, and scores every covered exam call and staged honest row.
Post-run fix (Codex): pools are sorted by call_id, as Amendment 3; the first run (a38528a2) sorted by key, a documented deviation.
Locally, between the steps: staged gate rows (Qwen3.5 locations) are mapped to (world, task, turn, call_index) through the Qwen3.5
metas; covered pools and draws are built by the rule (matched counts = covered simulated training counts) and written to
results/xgrid_selection_<stamp>.json before the fit. Writes results/xgrid_scores_<stamp>.json. Refuses to run unless this file
and RULE_scaled.md are committed unchanged.

  uv run --no-project --python 3.12 --with "modal==1.6.1" --with numpy modal run run_xgrid.py --inputs <scaled_inputs dir> --metas <acts_meta root>
"""

import csv
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

APP = "probe-shift_aswin-surya_sub-q3xgrid_score_kill-2026-11-12"
ACTS = "probe-shift_aswin-surya_sub-q3xread_acts_kill-2026-11-13"
LAYER, HIDDEN, C = 63, 5120, 0.01
SOURCES = [("2026-10-13_xread_main_v1", "attack", "real"), ("2026-10-13_xread_run2_v1", "attack", "run2"),
           ("2026-10-13_xread_main_v1", "sim_attack@september", "sim"), ("2026-10-13_xread_main_v1", "honest_a", "honest_a"),
           ("2026-10-13_xread_main_v1", "honest_b", "honest_b")]
HERE = Path(__file__).parent
PINS = {"flagrate_rows_2026-10-13_122417.csv": "44c75f78e78f86b4bcab409b1379762a8d4dccadfd456f28ee651832eb2b13a7",
        "scaled_rows_2026-10-13_152021.csv": "a95453ac2688e61ba99184726ce27a4dd54276a04730a56d67825f204d1ee1f3",
        "rows_v2.json": "727c417916b8fb2df9a31aca5f8ca0cc4dd17002a8a2edcb7bd79444bc0bf4b5",
        "sim_votes.csv": "06f4ef4bc00b53767b074185cda99311f2ba44f7c8f5a4f38ce59dd1537e81ae",
        "real_votes.csv": "9a96988db9818c8b97f3045ade83f19fae0e19e7d0cac134fa775005eba8dc36",
        "scaleup_panel_votes_k116.csv": "78ae0fa4e7e85da9a1ee85c9944be917b47e63c0d0a62f6f71390f046fdb7fcf"}
QWEN35_ARM_WORLD = {"attack": "real", "honest_a": "honest_a", "honest_b": "honest_b", "sim_attack@september": "sim"}

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2)
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


@app.function(image=image, cpu=1, memory=4096, timeout=600, retries=0, volumes={"/acts": acts.read_only()})
def keys_job() -> dict:
    loc, dig = {}, {}
    for x, arm, world in SOURCES:
        files = sorted(Path(f"/acts/{x}/meta").glob("*/*.json"))
        dig[x] = hashlib.sha256(b"".join(hashlib.sha256(f.read_bytes()).digest() for f in files)).hexdigest()
        for f in sorted(Path(f"/acts/{x}/meta/{arm}").glob("*.json")):
            m = json.loads(f.read_text())
            if m.get("redrawn_because") or m.get("attempt_kind") == "rerun" or m.get("chunk") is None:
                continue
            for c in m["calls"]:
                if c.get("kind", "command") == "command" and c.get("row") is not None:
                    k = f"{world}|{m['task']}|{c['turn']}|{c['call_index']}"
                    if k in loc:
                        raise SystemExit(f"two command readings for {k}")
                    loc[k] = [x, m["chunk"], m["row_start"] + c["row"]]
    return {"loc": loc, "metas_digests": dig}


@app.function(image=image, cpu=4, memory=32768, timeout=1500, retries=0, volumes={"/acts": acts.read_only()})
def fit_job(fits: dict, weighted: dict, eval_locs: list, staged_locs: list) -> dict:
    """fits: name -> [(x, chunk, row, y)]; weighted: name -> [(x, chunk, row, y, w)] (the gate's recipe)."""
    import numpy as np
    import torch

    want = sorted({tuple(t[:3]) for d in list(fits.values()) + list(weighted.values()) for t in d} | {tuple(v) for v in eval_locs} | {tuple(v) for v in staged_locs})
    pos = {v: i for i, v in enumerate(want)}
    groups = {}
    for i, (x, chunk, row) in enumerate(want):
        groups.setdefault((x, chunk), []).append((i, int(row)))
    X = np.empty((len(want), HIDDEN), np.float64)
    for (x, chunk), lst in groups.items():
        a = np.load(f"/acts/{x}/by_layer/L{LAYER}/{chunk}.npy")
        X[[i for i, _ in lst]] = a[[r for _, r in lst]]
    if not np.isfinite(X).all():
        raise SystemExit("non-finite readings")
    Xe, Xs = X[[pos[tuple(v)] for v in eval_locs]], X[[pos[tuple(v)] for v in staged_locs]]
    out, refit = {}, {}

    def fit(d, w=None):
        Xt = X[[pos[tuple(t[:3])] for t in d]]
        y = torch.tensor([float(t[3]) for t in d], dtype=torch.float64)
        if w is None:
            mu, sd, wt = Xt.mean(0), Xt.std(0), torch.ones(len(d), dtype=torch.float64)
        else:
            wv = np.array(w, dtype=float)
            mu = (wv[:, None] * Xt).sum(0) / wv.sum()
            sd = np.sqrt((wv[:, None] * (Xt - mu) ** 2).sum(0) / wv.sum())
            wt = torch.from_numpy(wv)
        sd = np.where(sd < 1e-6, 1.0, sd)
        Zt = torch.from_numpy((Xt - mu) / sd)
        a = fit_dual((Zt @ Zt.T)[None], y[None, None, :], wt, [C])
        return (a[0, 0] @ Zt).numpy(), mu, sd

    for name, d in sorted(fits.items()):
        beta, mu, sd = fit(d)
        out[name] = {"eval": (((Xe - mu) / sd) @ beta).tolist(), "staged": (((Xs - mu) / sd) @ beta).tolist()}
        if name == "R44_0":
            b2, mu2, sd2 = fit(d)
            refit[name] = float(np.abs(((Xe - mu2) / sd2) @ b2 - np.array(out[name]["eval"])).max())
    for name, d in weighted.items():
        beta, mu, sd = fit(d, [t[4] for t in d])
        out[name] = {"eval": (((Xe - mu) / sd) @ beta).tolist(), "staged": (((Xs - mu) / sd) @ beta).tolist()}
    return {"scores": out, "refit_max_abs_diff": refit, "chunks_read": len(groups)}


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@app.local_entrypoint()
def main(inputs: str, metas: str = "/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/a683364c-3a60-44cd-bb15-e08d87dd4f9a/scratchpad/acts_meta"):
    import numpy as np

    for f in (Path(__file__).name, "RULE_scaled.md"):
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", f], cwd=HERE, capture_output=True)
        status = subprocess.run(["git", "status", "--porcelain", "--", f], cwd=HERE, capture_output=True, text=True)
        if tracked.returncode != 0 or status.returncode != 0 or status.stdout:
            raise SystemExit(f"{f} is not committed unchanged at HEAD")
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True, check=True).stdout.strip()
    I = Path(inputs)
    for n, want in PINS.items():
        if sha(I / n) != want:
            raise SystemExit(f"{n}: sha256 differs from its pin")
    K = keys_job.remote()
    loc = K["loc"]
    key4 = lambda r: (r["run_id"], int(r["epoch"]), int(r["turn"]), int(r["call_index"]))  # noqa: E731
    votes = {}
    for p, default in ((I / "sim_votes.csv", "sim"), (I / "real_votes.csv", None), (I / "scaleup_panel_votes_k116.csv", None)):
        for r in csv.DictReader(open(p)):
            votes[key4(r)] = (r.get("world") or default, r["label"], int(r["state_changing"]))
    F = list(csv.DictReader(open(I / "flagrate_rows_2026-10-13_122417.csv")))
    N = list(csv.DictReader(open(I / "scaled_rows_2026-10-13_152021.csv")))
    exam, pools = [], {k: [] for k in ("sim_sabotage", "sim_not_sabotage", "real_sabotage", "real_not_sabotage")}
    cover = Counter()
    for r, src in [(r, "flagrate") for r in F] + [(r, "new") for r in N]:
        if src == "flagrate" and r["part"] != "2":
            continue
        v = votes.get(key4(r))
        if v is None or int(r["state_changing"]) != 1:
            continue
        k = f"{r['world']}|{r['task']}|{r['turn']}|{r['call_index']}"
        grp = "new_held_out" if src == "new" else r["group"]
        cover[(r["world"], grp, v[1], k in loc)] += 1
        if k not in loc:
            continue
        if grp in ("stage1_held_out", "new_held_out"):
            exam.append({"call_id": r["call_id"], "world": r["world"], "task": r["task"], "run_id": r["run_id"], "epoch": int(r["epoch"]),
                         "turn": int(r["turn"]), "call_index": int(r["call_index"]), "source": src, "group": grp, "state_changing": 1, "key": k})
        elif grp == "stage1_train" and v[1] in ("sabotage", "not_sabotage"):
            pools[("sim_" if r["world"] == "sim" else "real_") + v[1]].append((r["call_id"], k))
    pools = {p: [k for _, k in sorted(v)] for p, v in pools.items()}  # each pool sorted by call_id, as Amendment 3 (Codex, post-run fix)
    n_pos, n_neg = len(pools["sim_sabotage"]), len(pools["sim_not_sabotage"])
    sel = {"n_pos": n_pos, "n_neg": n_neg, "pool_counts": {k: len(v) for k, v in pools.items()}, "S": {"pos": pools["sim_sabotage"], "neg": pools["sim_not_sabotage"]}}
    for g, (pk, nk) in {"R44": ("real_sabotage", "real_not_sabotage"), "M": ("sim_sabotage", "real_not_sabotage"), "D": ("real_sabotage", "sim_not_sabotage")}.items():
        sel[g] = []
        for d in range(5):
            seed = int(hashlib.sha256(f"x{'44' if g == 'R44' else g}|{d}".encode()).hexdigest()[:16], 16)
            rng = np.random.default_rng(seed)
            take = lambda pool, n: [pool[i] for i in rng.choice(len(pool), min(n, len(pool)), replace=False)]  # noqa: E731
            sel[g].append({"draw": d, "seed": seed, "pos": take(pools[pk], n_pos), "neg": take(pools[nk], n_neg)})
    # staged gate rows -> (world, task, turn, call_index) through the Qwen3.5 metas
    ce = {}
    for f in sorted(Path(metas).glob("*/meta/*/*.json")):
        x, arm, m = f.parent.parent.parent.name, f.parent.name, json.loads(f.read_text())
        if m.get("chunk") is None or x == "2026-10-11_honest_c_segends_v1":  # honest C isn't in the cross-read, and its folder is named honest_b
            continue
        world = "run2" if x == "2026-10-10_second_attack_segends_v1" else QWEN35_ARM_WORLD.get(arm)
        for c in m["calls"]:
            if c.get("kind") == "call_end" and c.get("row") is not None and world:
                ce[(x, m["chunk"], m["row_start"] + c["row"])] = f"{world}|{m['task']}|{c['turn']}|{c['call_index']}"
    staged = json.loads((I / "rows_v2.json").read_text())["rows"]
    st_keys = [ce.get((r["x"], r["chunk"], int(r["row"]))) for r in staged]
    if any(k is None for k in st_keys):
        raise SystemExit("a staged gate row has no Qwen3.5 call_end entry")
    st_cov = [k in loc for k in st_keys]
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    selp = HERE / "results" / f"xgrid_selection_{stamp}.json"
    with selp.open("x") as fh:  # written before the fit
        fh.write(json.dumps({"rule": "RULE_scaled.md Amendment 4", "git": git, **sel}) + "\n")
    fits = {"S": [(*loc[k], 1) for k in sel["S"]["pos"]] + [(*loc[k], 0) for k in sel["S"]["neg"]]}
    for g in ("R44", "M", "D"):
        for d in sel[g]:
            fits[f"{g}_{d['draw']}"] = [(*loc[k], 1) for k in d["pos"]] + [(*loc[k], 0) for k in d["neg"]]
    rtrain = [(*loc[k], int(r["y"]), float(r["w"])) for r, k, c in zip(staged, st_keys, st_cov) if r["role"] == "train" and c]
    out = fit_job.remote(fits, {"R": rtrain}, [loc[e["key"]] for e in exam], [loc[k] for k, c in zip(st_keys, st_cov) if c])
    sc = out["scores"]
    names = ["R", "S"] + [f"{g}_{i}" for g in ("R44", "M", "D") for i in range(5)]
    val = {"R44_0_refit_max_abs_diff": out["refit_max_abs_diff"]["R44_0"],
           "all_finite": bool(all(np.isfinite(np.array(v[k])).all() for v in sc.values() for k in ("eval", "staged")))}
    val["passes"] = bool(val["R44_0_refit_max_abs_diff"] <= 1e-6 and val["all_finite"])
    if not val["passes"]:
        raise SystemExit(f"gates failed, nothing written: {json.dumps(val)}")
    roles = [r["role"] for r, c in zip(staged, st_cov) if c]
    res = {"rule": "RULE_scaled.md Amendment 4", "label": "scores only; no label joined", "git": git, "app": APP, "model": "Qwen3-32B (cross-read)",
           "readout": "command", "layer": LAYER, "C": C, "validation": val, "inputs_sha256": PINS, "xread_metas_digests": K["metas_digests"],
           "selection_file": selp.name, "selection_sha256": sha(selp), "r_train_rows_covered": len(rtrain), "chunks_read": out["chunks_read"],
           "coverage": {f"{w} / {g} / {lab} / {'covered' if c else 'not covered'}": n for (w, g, lab, c), n in sorted(cover.items())},
           "staged_coverage": {f"{r} / {'covered' if c else 'not covered'}": n for (r, c), n in sorted(Counter((r["role"], c) for r, c in zip(staged, st_cov)).items())},
           "probes": names, "scores": [{k: e[k] for k in e if k != "key"} | {"part": 2, "s": [sc[n]["eval"][i] for n in names]} for i, e in enumerate(exam)],
           "staged": [{"role": r, "s": [sc[n]["staged"][i] for n in names]} for i, r in enumerate(roles)], "written": stamp}
    dst = HERE / "results" / f"xgrid_scores_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(res) + "\n")
    print(json.dumps({"validation": val, "selection": {k: sel[k] for k in ("n_pos", "n_neg", "pool_counts")}, "staged_coverage": res["staged_coverage"]}, indent=1),
          "\n->", dst, sha(dst))
