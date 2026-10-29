"""Halves on Qwen3.5-27B (PLAN.md in this folder): pack the Qwen3.5 reads of the evened texts, check them against follow-up
(b)'s reads, then halves' refit copied with the reader swapped. Saves keys, numbers and score arrays only.

  --mode packgate  CPU: every staged evened transcript's kept output_end rows at L58/L63 from the extraction
                   (`2026-10-12_la_fixed-sim-v2-b-evened-halves-scrub1_qwen35-27b_alllayers_v1`, Volume claimb-reader),
                   with each run's "Invalid `view_range`" outputs mapped to (step, action) through the extractor's own
                   positions (one output_end per tool message, in order, as B-dev mapped them); then the Qwen3.5
                   reproduction gate: the 4 unchanged gate runs must have the same positions as follow-up (b)'s dump and
                   rows bitwise equal or cosine >= 0.9999 at L58 and L63 (bitwise reported).
  --mode refit     1 H100: halves' refit (aswin/claimb-halves run_halves_modal.py @a2494dc4) with the rows assembled from
                   Qwen3.5: changed runs (H3's test, B-dev's dev) take their evened Qwen3.5 rows without range outputs,
                   unchanged runs keep follow-up (b)'s Qwen3.5 level-1 pack rows (claimB/level1/packedB_06a0329937c1),
                   B-dev's 100 dev planted copies are plant-2 copies. Fitting, columns and seeds are halves' code
                   unchanged; one addition that changes no fit: the fixed-vs-e1 probes also score the held-out e2 runs
                   (the direct yardstick, reported beside the registered one).

probe_shift comes from the as-run commit 23b5c37a (PYTHONPATH=<asrun>/src), so H3's check_b() holds, as in staging.

  PYTHONPATH=<asrun>/src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy modal deploy \\
      experiments/2026-10-11_claimB_halves_qwen35/run_halves_qwen35_modal.py
  PYTHONPATH=<asrun>/src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy python \\
      experiments/2026-10-11_claimB_halves_qwen35/run_halves_qwen35_modal.py packgate|refit
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).parent
H3_DIR = HERE.parent / "2026-10-11_claimB_h3_fix_to_plateau"
sys.path[:0] = [str(H3_DIR), str(HERE.parent / "2026-10-11_claimB_slips_evened_probe")] if modal.is_local() else ["/root"]
import run_h3_small_modal as R  # noqa: E402  (H3's runner: B-dev's code and constants; hash-checked below)

H3_SHA12 = "04c3953b139e"
B, M = R.B, R.M
SIM, E1, E2, PL2 = M.SIM, M.REAL, "real-attack-e2", "sim-attack-fixed-plant2"
TEST = list(R.TEST)
DEV = ["auto_workflow", "file_indexing", "model_registry", "slack_clone", "web_scraping"]
DIRS = {"dev_to_test": (DEV, TEST), "test_to_dev": (TEST, DEV)}
COMPS = {"fixed_vs_e1": (E1, SIM), "e2_vs_e1": (E1, E2), "plant2": (SIM, PL2)}  # (negative arm, positive arm)
K, KP, VAL_FOLD, LAYERS = 101, 21, 0, (58, 63)
STAGED = "2026-10-12_la_fixed-sim-v2_probe-inputs-b-l1-evened-halves_v1"
STAGE_SHA = "23e04d5cef13542c398ccfc190d0663d025508297e461c15a380535f2a785ac6"  # run_stage_evened_modal.py that staged it
DUMP = "2026-10-12_la_fixed-sim-v2-b-evened-halves-scrub1_qwen35-27b_alllayers_v1"
B_DUMP = "2026-10-10_la_fixed-sim-v2-b-scrub1_qwen35-27b_alllayers_v1"  # follow-up (b)'s dump: the gate's reference
PACK_Q35 = "claimB/level1/packedB_06a0329937c1"  # follow-up (b)'s level-1 pack (output_end, L19..L63)
SCRIPT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
OUT = f"claimB/halves_qwen35_v1/{SCRIPT_SHA[:12]}"

app = modal.App("probe-shift_aswin-surya_sub-claimb-hq35refit_kill-2026-11-12")
image = R.image.add_local_file(H3_DIR / "run_h3_small_modal.py", "/root/run_h3_small_modal.py")
q35 = modal.Volume.from_name("probe-shift_aswin-surya_sub-claimb-reader_acts_kill-2026-12-10", version=2)


def check_code():
    R.check_b()
    if R.SCRIPT_SHA[:12] != H3_SHA12:
        raise ValueError("H3's runner differs from the one whose reads defined the changed sets")


def filename(key: str) -> str:
    """run_extract_qwen35.filename, copied: the dump's per-transcript file stem."""
    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


def dump_rows(dump: str, arm: str, key: str, chunks: dict):
    """(kept output_end rows per layer, kept output_end (step, action) list, all output_end positions) of one transcript."""
    import numpy as np

    m = json.loads((Path("/q35") / dump / "meta" / arm / f"{filename(key)}.json").read_text())
    outs = [p for p in m["positions"] if p["kind"] == "output_end"]
    kept = [p for p in outs if p["kept"]]
    rows = {}
    for l in LAYERS:
        ck = (dump, l, m["chunk"])
        if ck not in chunks:  # whole chunk files, read once (row-indexed memory maps read page by page from a Volume)
            chunks[ck] = np.load(Path("/q35") / dump / "by_layer" / f"L{l}" / f"{m['chunk']}.npy")
        rows[l] = chunks[ck][m["row_start"] + np.array([p["row"] for p in kept], int)].astype(np.float16)
    return rows, [[p["step"], p["action"]] for p in kept], outs


@app.function(image=image, cpu=16, memory=131072, volumes={"/acts": B.P.vol, "/q35": q35.read_only()}, timeout=60 * 60, retries=0)
def packgate() -> dict:
    import numpy as np

    from probe_shift.models import QWEN35_27B

    check_code()
    man = json.loads((Path("/acts") / STAGED / "manifest.json").read_text())
    if man["script_sha256"] != STAGE_SHA:
        raise ValueError("the staged run was made by another staging script")
    cfg, bcfg = (json.loads((Path("/q35") / d / "config.json").read_text()) for d in (DUMP, B_DUMP))
    same = ("model", "revision", "dtype", "attn_implementation", "max_tokens", "pad_to")
    if cfg["inputs_from"] != STAGED or cfg["model"] != QWEN35_27B["name"] or cfg["revision"] != QWEN35_27B["revision"] \
            or any(cfg[k] != bcfg[k] for k in same):
        raise ValueError("the evened dump was not read with follow-up (b)'s reader settings")
    chunks, index, X, start, gate_rows = {}, [], {l: [] for l in LAYERS}, 0, {}
    for it in man["items"]:
        rows, steps, outs = dump_rows(DUMP, it["arm"], it["key"], chunks)
        if it["range_tool_ordinals"] and max(it["range_tool_ordinals"]) >= len(outs):
            raise ValueError(f"{it['key']}: a range output has no output_end position")
        rng = [[outs[k]["step"], outs[k]["action"]] for k in it["range_tool_ordinals"]]
        n = len(steps)
        if n == 0 or not all(np.isfinite(rows[l].astype(np.float32)).all() for l in LAYERS):
            raise ValueError(f"{it['key']}: no kept output_end rows or non-finite values")
        for l in LAYERS:
            X[l].append(rows[l])
        if it["kind"] == "h3_gate_unchanged":
            gate_rows[it["key"]] = rows
        index.append({k: it[k] for k in ("kind", "arm", "key", "cell", "environment", "main_task")}
                     | {"row_start": start, "rows": n, "steps": steps, "range": rng})
        start += n
    gate = []
    for it in (x for x in index if x["kind"] == "h3_gate_unchanged"):
        mine = gate_rows[it["key"]]
        ref, rsteps, _ = dump_rows(B_DUMP, it["arm"], it["key"], chunks)
        g = {"key": it["key"], "positions_equal": rsteps == it["steps"]}
        for l in LAYERS:
            a, b = mine[l].astype(np.float32), ref[l].astype(np.float32)
            g[f"L{l}"] = {"bitwise": bool(np.array_equal(mine[l], ref[l])),
                          "min_cosine": float(((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))).min()) if a.shape == b.shape else None,
                          "max_abs_diff": float(np.abs(a - b).max()) if a.shape == b.shape else None}
        g["pass"] = g["positions_equal"] and all(g[f"L{l}"]["bitwise"] or (g[f"L{l}"]["min_cosine"] or 0) >= 0.9999 for l in LAYERS)
        gate.append(g)
    out_dir = Path("/acts") / OUT / "evened"
    out_dir.mkdir(parents=True, exist_ok=True)
    for l in LAYERS:
        with open(out_dir / f".L{l}.tmp", "wb") as fh:
            np.save(fh, np.concatenate(X[l]))
        os.replace(out_dir / f".L{l}.tmp", out_dir / f"L{l}.npy")
    res = {"gate": gate, "passed": len(gate) == 4 and all(g["pass"] for g in gate), "transcripts": len(index), "rows": start,
           "by_kind": {k: sum(x["kind"] == k for x in index) for k in sorted({x["kind"] for x in index})},
           "script_sha256": SCRIPT_SHA, "dump": DUMP, "staged": STAGED}
    (out_dir / "index.json").write_text(json.dumps({"runs": index} | res))
    B.P.vol.commit()
    return res


@app.function(image=image, gpu="H100!", cpu=8, memory=131072, volumes={"/acts": B.P.vol}, timeout=120 * 60, retries=0)
def refit(cfg: dict) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scipy.stats import rankdata

    from probe_shift import torch_probe as tp
    from probe_shift.q1_eval import step_table

    check_code()
    B.P.vol.reload()
    t0 = time.time()
    log = lambda m: print(f"[halves-q35] {time.time() - t0:6.0f}s {m}", flush=True)  # noqa: E731
    ev = json.loads((Path("/acts") / OUT / "evened" / "index.json").read_text())
    if not ev.get("passed") or ev["script_sha256"] != SCRIPT_SHA:
        raise ValueError("no passed Qwen3.5 reproduction gate for this script version")
    EV = {l: np.load(Path("/acts") / OUT / "evened" / f"L{l}.npy") for l in LAYERS}
    pruns = json.loads((Path("/acts") / PACK_Q35 / "runs.L63.json").read_text())
    t = step_table(pruns, "output_end")
    XP = {l: np.load(Path("/acts") / PACK_Q35 / f"L{l}_output_end.npy") for l in LAYERS}
    rows_of = {}
    for k in range(len(t["run"])):
        rows_of.setdefault(int(t["run"][k]), {})[(int(t["step"][k]), int(t["action"][k]))] = k
    ev_by = {(x["cell"], x["arm"]): x for x in ev["runs"] if x["kind"] in ("h3_test_changed", "bdev_dev_changed")}
    planted_dev = [x for x in ev["runs"] if x["kind"] == "bdev_dev_planted"]

    def ev_rows(x):  # evened rows without the run's range outputs (halves' read_rows)
        rng = {tuple(s) for s in x["range"]}
        keep = [x["row_start"] + j for j, s in enumerate(x["steps"]) if tuple(s) not in rng]
        return {l: np.asarray(EV[l][keep], np.float16) for l in LAYERS}, len(keep)

    runs, rr, Xl = [], [], {l: [] for l in LAYERS}

    def add(entry, rows, n):
        if n == 0:
            raise ValueError(f"{entry['cell']} {entry['arm']}: no rows")
        for l in LAYERS:
            Xl[l].append(rows[l])
        rr.extend([len(runs)] * n)
        runs.append(entry | {"rows": n})

    used = set()
    for i, m in enumerate(pruns):
        if m["arm"] not in (SIM, E1, E2, PL2):
            continue
        src = ev_by.get((m["cell"], m["arm"]))
        if src:
            rows, n = ev_rows(src)
            how = "evened Qwen3.5 read"
            used.add((m["cell"], m["arm"]))
        else:
            ix = sorted(rows_of.get(i, {}).values())
            rows, n = {l: XP[l][ix] for l in LAYERS}, len(ix)
            how = "Qwen3.5 level-1 pack rows"
        add({"cell": m["cell"], "arm": m["arm"], "environment": m["environment"], "main_task": m["main_task"], "source": how}, rows, n)
    if used != set(ev_by):
        raise ValueError(f"{len(set(ev_by) - used)} evened reads matched no pack run")
    for x in planted_dev:  # B-dev's evened dev planted reads (R2 undone), as plant-2 copies
        rows, n = ev_rows(x)
        add({"cell": x["cell"], "arm": PL2, "environment": x["environment"], "main_task": x["main_task"], "source": "B-dev planted (Qwen3.5)"}, rows, n)
    rr = np.array(rr)
    X = {l: np.concatenate(Xl[l]) for l in LAYERS}
    del XP, EV
    log(f"rows assembled: {len(rr)} rows, {len(runs)} runs")
    env = np.array([r["environment"] for r in runs])
    arm = np.array([r["arm"] for r in runs])
    task = np.array([r["main_task"] for r in runs])
    cell = np.array([r["cell"] for r in runs])
    tfold = np.array([cfg["main_task_to_fold"][x] for x in task])
    pl2_cells = set(cell[arm == PL2])
    dev = torch.device("cuda")
    out_dir = Path("/acts") / OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "runs.json").write_text(json.dumps({"runs": [{k: r[k] for k in ("cell", "arm", "environment", "main_task", "source", "rows")} for r in runs],
                                                   "script_sha256": SCRIPT_SHA}))
    B.P.vol.commit()

    def linear(Xa, Xb, Ya, C, cap):  # step A's fitter and convergence loop (halves' copy)
        Xa, Xb = tp.standardize(Xa.double(), Xb.double())
        Xa, Xb = Xa * C**0.5, Xb * C**0.5
        tol, mi, prev = 1e-4 * min(C**0.5, 1.0), R.MAX_ITER, None
        while True:
            W, b, evals = tp.fit(Xa, Ya, 1.0, mi, tol=tol)
            Wg, bg = W.clone().requires_grad_(True), b.clone().requires_grad_(True)
            loss = (F.binary_cross_entropy_with_logits(Xa @ Wg + bg, Ya, reduction="sum") + 0.5 * (Wg * Wg).sum()) / len(Xa)
            loss.backward()
            gk = torch.maximum(Wg.grad.abs().amax(0), bg.grad.abs()).cpu().numpy()
            if float(gk.max()) <= tol or evals < mi or evals == prev or mi * 4 > cap:
                return tp.decision(Xb, W, b), gk
            mi, prev = mi * 4, evals

    def labels(comp):
        neg, pos = COMPS[comp]
        if comp == "plant2":  # pairs: each plant-2 copy and its own evened fixed-sim twin
            use = ((arm == PL2) | (arm == SIM)) & np.isin(cell, list(pl2_cells))
            unit, kk, tag = cell, KP, "plant2"
        else:
            use = (arm == neg) | (arm == pos)
            unit, kk, tag = task, K, "real"  # the same coins for both real comparisons
        y = (arm == pos).astype(np.float32)
        Y = np.zeros((len(runs), kk), np.float32)
        Y[:, 0] = y
        units = sorted(set(unit[use])) if comp == "plant2" else sorted(set(task[np.isin(arm, [SIM, E1, E2])]))
        for k in range(1, kk):
            rngk = np.random.default_rng(R.seed_of("halves-shuffle", tag, k))
            flip = dict(zip(units, rngk.random(len(units)) < 0.5))
            Y[:, k] = [1 - v if (use[i] and flip[unit[i]]) else v for i, v in enumerate(y)]
        return use, Y

    subs = [(l, C) for l in LAYERS for C in R.C_GRID]
    res = {"rows": int(len(rr)), "runs": len(runs), "planted_dev": len(planted_dev), "planted_test": int(((arm == PL2) & np.isin(env, TEST)).sum())}
    for dname, (train_envs, test_envs) in DIRS.items():
        for comp in COMPS:
            use, Y = labels(comp)
            KK = Y.shape[1]
            cap = R.MAX_ITER if comp == "plant2" else R.MAX_ITER_CAP
            u_r, e_r, f_r = use[rr], env[rr], tfold[rr]
            tr, te = u_r & np.isin(e_r, train_envs), u_r & np.isin(e_r, test_envs)
            te_full = te | ((arm[rr] == E2) & np.isin(e_r, test_envs)) if comp == "fixed_vs_e1" else te  # + the direct yardstick
            va = tr & (f_r == VAL_FOLD)
            r = {"val_auroc": np.full((len(subs), KK), np.nan), "test_run_score": np.full((len(subs), len(runs), KK), np.nan),
                 "grad": np.full((len(subs), 2, KK), np.nan)}
            for j, (part, a, b_) in enumerate((("full", tr, te_full), ("val", tr & ~va, va))):
                ia, ib = np.flatnonzero(a), np.flatnonzero(b_)
                Ya = torch.from_numpy(Y[rr[ia]]).to(dev).double()
                for s, (l, C) in enumerate(subs):
                    S, ok = linear(torch.from_numpy(np.asarray(X[l][ia], np.float32)).to(dev), torch.from_numpy(np.asarray(X[l][ib], np.float32)).to(dev), Ya, C, cap)
                    kk_, inv = np.unique(rr[ib], return_inverse=True)
                    tot = torch.zeros(len(kk_), KK, dtype=torch.float64, device=dev).index_add_(0, torch.from_numpy(inv).to(dev), S.double())
                    Mv = (tot / torch.from_numpy(np.bincount(inv)).to(dev)[:, None]).cpu().numpy()
                    r["grad"][s, j] = ok
                    if part == "full":
                        r["test_run_score"][s, kk_] = Mv
                    else:
                        R_, Yv = rankdata(Mv, axis=0), Y[kk_]
                        n1 = Yv.sum(0)
                        r["val_auroc"][s] = ((R_ * Yv).sum(0) - n1 * (n1 + 1) / 2) / (n1 * (len(Yv) - n1))
            held = use & np.isin(env, test_envs)
            if not np.isfinite(r["test_run_score"][:, held]).all():
                raise ValueError(f"{dname} {comp}: a held-out run got no score")
            (out_dir / dname).mkdir(exist_ok=True)
            with open(out_dir / dname / f"{comp}.tmp", "wb") as fh:
                np.savez(fh, subs=json.dumps(subs), Y=Y, use=use, script_sha256=SCRIPT_SHA, train_envs=json.dumps(train_envs), test_envs=json.dumps(test_envs), **r)
            os.replace(out_dir / dname / f"{comp}.tmp", out_dir / dname / f"{comp}.npz")
            B.P.vol.commit()
            res[f"{dname}|{comp}"] = {"train_runs": int(len(np.unique(rr[tr]))), "held_runs": int(held.sum())}
            log(f"{dname} {comp} done")
    return res | {"seconds": round(time.time() - t0)}


@app.function(image=image, cpu=1, memory=1024, timeout=120, retries=0)
def version() -> str:
    return SCRIPT_SHA


def main(mode: str):
    if subprocess.run(["git", "status", "--porcelain", str(Path(__file__))], capture_output=True, text=True, check=True).stdout:
        raise SystemExit(f"commit {Path(__file__).name} first")
    check_code()  # locally too: probe_shift must be the as-run version
    if modal.Function.from_name(app.name, "version").remote() != SCRIPT_SHA:
        raise SystemExit("the deployed app runs other code: `modal deploy` this revision first")
    if mode == "packgate":
        call = modal.Function.from_name(app.name, "packgate").spawn()
    elif mode == "refit":
        folds = json.loads((HERE.parents[1] / "splits/folds_v2.json").read_text())
        call = modal.Function.from_name(app.name, "refit").spawn({"main_task_to_fold": folds["task_folds"]["main_task_to_fold"]})
    else:
        raise SystemExit(f"unknown mode {mode}")
    print(f"{mode}: {call.object_id}", flush=True)
    print(json.dumps(call.get(timeout=130 * 60), indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
