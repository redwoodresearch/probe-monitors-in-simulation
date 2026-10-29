"""Matched label-shuffle null for realism left after removing k = 10 world directions (RULE.md; POST HOC).

One app per job family (MNULL_FAMILY=wd: wd48, wd63; MNULL_FAMILY=q1b: q1b_avg, q1b_lt). One H100 container per (job, chunk
of 7 label columns), the chunk's columns fit concurrently (one thread and CUDA stream each: small fits use an H100 at only
4-28%, d9's measurement via 0d). Per column c (0 = real labels, 1-20 = the first 20 as-run within-task shuffles) and nested model: the
as-run nullspace iteration (run_null_gpu.py @ 5d1e5332 lines 145-158; run_q1b.py @ 162d860b lines 184-197) with the 10
directions AND the held-out realism probe fit on column c's labels, then the held-out run AUROC against column c. Each
column's 20 AUROCs are checkpointed with provenance (code commit, input sha256s, seed, labels) as it finishes; a rerun
reuses a column only if its provenance matches. `report` (Modal CPU) requires one provenance and 20 finite AUROCs in every
column, checks the column-0 gate, then applies the committed reading.

  MNULL_FAMILY=wd NULL_IN=<folder with d9's scores npz> PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' \\
      --with numpy modal run --detach experiments/2026-10-13_matched_world_null_POSTHOC/run_matched_null.py::main --tag <tag>
  MNULL_FAMILY=wd NULL_IN=... (same) modal run experiments/2026-10-13_matched_world_null_POSTHOC/run_matched_null.py::report --tag <tag> --out <json>
"""
import hashlib
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
FAMILY = os.environ.get("MNULL_FAMILY", "wd")
APP = f"probe-shift_aswin-surya_sub-mnull-{FAMILY}_fit_kill-2026-11-13"
WORK, DFV = "probe-shift_aswin-surya_sub-q3attack_work_kill-2026-11-06", "probe-shift_aswin-surya_sub-whysteps_work_kill-2026-11-13"
CKPT = "probe-shift_aswin-surya_sub-realism-null_ckpt_kill-2026-11-13"
IN = Path(os.environ.get("NULL_IN", "/nonexistent"))
D9_SHA = "48fc5eb52546f8884bf1083a3d8530a6c80624b52c619678edf72c28443826be"
R36 = REPO / "experiments/2026-10-06_q3_attack_detector/results"
H = "real-honest"
C, MAX_ITER, N_DIR, N_COLS, N_ALL = 0.01, 1000, 10, 20, 100
PAIRS = [(f, v) for f in range(5) for v in range(5) if v != f]
FF, DF = "/work/sce_fixfull/2026-10-13_152618", "/df/packs/2026-10-13_185308"
JOBS = {  # as_run = the as-run k = 10 realism (the gate); unmatched_p95 = the earlier unmatched null's 95th (RULE.md table)
    "wd48": {"family": "wd", "runs": "d9", "sh": "sim-honest-fullfix", "x": f"{FF}/L48.npy", "rows": f"{FF}/rows_L48.npz", "seed": [20261017, 48],
             "as_run": 0.5825, "unmatched_p95": 0.535},
    "wd63": {"family": "wd", "runs": "d9", "sh": "sim-honest-fullfix", "x": f"{FF}/L63.npy", "rows": f"{FF}/rows_L63.npz", "seed": [20261017, 63],
             "as_run": 0.5654, "unmatched_p95": 0.534},
    "q1b_avg": {"family": "q1b", "runs": "avg", "sh": "sim-honest", "x": f"{DF}/ffavg_output_end_mean_L48.npy", "rows": f"{DF}/rows_ffavg_output_end_mean_L48.npz",
                "seed": [20261021, 0], "as_run": 0.5392, "unmatched_p95": 0.5266},
    "q1b_lt": {"family": "q1b", "runs": "d9", "sh": "sim-honest-fullfix", "x": f"{FF}/L48.npy", "rows": f"{FF}/rows_L48.npz", "seed": [20261021, 1],
               "as_run": 0.5831, "unmatched_p95": 0.5325},
}
CHUNKS = [list(range(i, i + 7)) for i in range(0, N_COLS + 1, 7)]
READINGS = ("still above the matched null: the removal was incomplete", "can't tell", "realism at chance under the matched null")
app = modal.App(APP)
work = modal.Volume.from_name(WORK, version=2).read_only()
dfv = modal.Volume.from_name(DFV, version=2).read_only()
ckpt = modal.Volume.from_name(CKPT, version=2)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy==2.3.3").env({"MNULL_FAMILY": FAMILY}).add_local_python_source("probe_shift")
         .add_local_file(R36 / "run_scores_2026-10-06_181548/runs.json", "/root/in/runs36.json")
         .add_local_file(R36 / "sim_honest_scores_2026-10-07_124717/runs.json", "/root/in/runs_sh.json"))
if IN.exists():
    image = image.add_local_file(IN / "sce_fixfull_scores_2026-10-13_152618.npz", "/root/in/d9.npz")


def load_runs(J):
    """arm, main_task, fold in the job's run order."""
    if J["runs"] == "avg":  # df's run order (run_q1b.py lines 99-101): #36's 3,807 runs, then the simulated honest runs
        runs = json.loads(Path("/root/in/runs36.json").read_text()) + [r for r in json.loads(Path("/root/in/runs_sh.json").read_text()) if r["arm"] == J["sh"]]
        return [np.array([r[k] for r in runs]) for k in ("arm", "main_task", "fold")]
    if hashlib.sha256(Path("/root/in/d9.npz").read_bytes()).hexdigest() != D9_SHA:
        raise RuntimeError("d9's scores npz differs")
    R = np.load("/root/in/d9.npz")
    return [R[f"runs__{k}"] for k in ("arm", "main_task", "fold")]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 26), b""):
            h.update(b)
    return h.hexdigest()


@app.function(image=image, gpu="H100!", cpu=8, memory=32768, timeout=1800, retries=0, volumes={"/work": work, "/df": dfv, "/ckpt": ckpt})
def run(job: str, cols: list, tag: str, commit: str) -> dict:
    import torch

    from probe_shift import torch_probe as tp
    from probe_shift.attack_eval import auc, pool, shuffle_within_task

    t0, J = time.time(), JOBS[job]
    arm, task, fold = load_runs(J)
    fold, n = fold.astype(int), len(arm)
    raw = np.load(J["x"])
    r_run = np.load(J["rows"])["run"].astype(np.int64)
    if len(r_run) != raw.shape[0] or r_run.max() >= n or np.any(np.diff(r_run) < 0):
        raise RuntimeError("row table does not match the runs")
    X = torch.from_numpy(raw).cuda().float()
    del raw
    torch.cuda.synchronize()  # X is read from every column's stream
    y = np.where(arm == J["sh"], 1.0, np.where(arm == H, 0.0, np.nan))
    rng = np.random.default_rng(J["seed"])
    Y = np.stack([y] + [shuffle_within_task(y, task, ~np.isnan(y), rng) for _ in range(N_ALL)], 1)  # the as-run 101 columns, same order
    honest = np.isin(arm, [H, J["sh"]])
    runs_sha = hashlib.sha256(b"".join(np.ascontiguousarray(a.astype(str)).tobytes() for a in (arm, task, fold))).hexdigest()
    prov = {"code_commit": commit, "x": J["x"], "x_sha256": sha256_file(J["x"]), "rows": J["rows"], "rows_sha256": sha256_file(J["rows"]),
            "runs_sha256": runs_sha, "seed": J["seed"], "labels_sha256": hashlib.sha256(np.ascontiguousarray(Y[:, :N_COLS + 1]).tobytes()).hexdigest()}
    out_dir = Path("/ckpt") / f"matched_{tag}" / job
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{job} cols {cols}: loaded in {time.time() - t0:.0f} s", flush=True)

    def fit(Z, tr_runs, te_runs, ycol):  # run_null_gpu.fit / run_q1b.fit (restandardized, no row weights), one label column
        tr, te = np.flatnonzero(tr_runs[r_run]), np.flatnonzero(te_runs[r_run])
        Ztr = Z[torch.from_numpy(tr).cuda()]
        mean, std = Ztr.mean(0), Ztr.std(0, unbiased=False)
        std = std.where(std > 0, std.new_ones(()))
        Yt = torch.from_numpy(ycol.reshape(n, 1)[r_run[tr]]).cuda().float()
        W, b, _ = tp.fit((Ztr - mean) / std, Yt, C, MAX_ITER)
        del Ztr
        s = tp.decision((Z[torch.from_numpy(te).cuda()] - mean) / std, W, b).double().cpu().numpy()
        if not np.isfinite(s).all():
            raise RuntimeError("non-finite scores")
        S = pool(r_run[te], s, n, "mean")
        S[~te_runs] = np.nan
        return S, W / std[:, None]

    lock = threading.Lock()

    def column(c):
        p = out_dir / f"col_{c:03d}.json"
        if p.exists() and json.loads(p.read_text())["provenance"] == prov:
            return c
        tc, ycol, aucs = time.time(), Y[:, c], []
        with torch.cuda.stream(torch.cuda.Stream()):
            for f, v in PAIRS:
                te = (fold == f) | (fold == v)
                htr, hte = honest & ~te, honest & te
                rows_h = torch.from_numpy(np.flatnonzero(htr[r_run])).cuda()
                mu, sd = X[rows_h].mean(0), X[rows_h].std(0, unbiased=False)
                sd = sd.where(sd > 0, sd.new_ones(()))
                Z = (X - mu) / sd
                U = torch.zeros(X.shape[1], 0, device="cuda")
                for _ in range(N_DIR):  # the as-run nullspace iteration, on column c's labels
                    Zp = Z - (Z @ U) @ U.T if U.shape[1] else Z
                    w = fit(Zp, htr, hte, ycol)[1][:, 0]
                    w = w - U @ (U.T @ w) if U.shape[1] else w
                    U = torch.cat([U, (w / w.norm())[:, None]], 1)
                    del Zp
                Zp = Z - (Z @ U) @ U.T
                S = fit(Zp, htr, hte, ycol)[0]
                m = hte & np.isfinite(S[:, 0])
                aucs.append(float(auc(S[m, 0], ycol[m], np.ones((1, m.sum())))[0]))
                del Zp, Z
        if not np.isfinite(aucs).all():
            raise RuntimeError(f"{job} column {c}: non-finite AUROC")
        with lock:
            p.write_text(json.dumps({"job": job, "column": c, "aurocs": aucs, "mean": float(np.mean(aucs)), "seconds": round(time.time() - tc), "provenance": prov}, indent=1))
            ckpt.commit()
        if c == 0:  # the gate, first: column 0 against the as-run k = 10 value
            print(f"{job} GATE column 0 {np.mean(aucs):.4f} vs as-run {J['as_run']}: {'passes' if abs(np.mean(aucs) - J['as_run']) <= 0.005 else 'FAILS'}", flush=True)
        print(f"{job} col {c} done in {time.time() - tc:.0f} s", flush=True)
        return c

    with ThreadPoolExecutor(len(cols)) as ex:
        done = list(ex.map(column, cols))  # a failed column re-raises here, after the others finish
    return {"job": job, "cols": done, "seconds": round(time.time() - t0)}


@app.function(image=modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3").env({"MNULL_FAMILY": FAMILY}), cpu=1, memory=2048,
              timeout=600, retries=0, volumes={"/ckpt": ckpt})
def summarize(tag: str, family: str) -> dict:
    ckpt.reload()
    res = {}
    for job, J in ((j, J) for j, J in JOBS.items() if J["family"] == family):
        d = Path("/ckpt") / f"matched_{tag}" / job
        cols = {int(p.stem[4:]): json.loads(p.read_text()) for p in sorted(d.glob("col_*.json"))}
        provs = {json.dumps(x["provenance"], sort_keys=True) for x in cols.values()}
        r = {"columns_found": sorted(cols), "as_run": J["as_run"], "unmatched_p95": J["unmatched_p95"], "provenances": [json.loads(s) for s in sorted(provs)]}
        ok = all(len(x["aurocs"]) == len(PAIRS) and np.isfinite(x["aurocs"]).all() and np.isclose(x["mean"], np.mean(x["aurocs"])) for x in cols.values())
        if sorted(cols) != list(range(N_COLS + 1)) or len(provs) != 1 or not ok:
            res[job] = r | {"complete": False, "one_provenance": len(provs) == 1, "aurocs_ok": bool(ok)}
            continue
        obs = cols[0]["mean"]
        gate = abs(obs - J["as_run"]) <= 0.005
        r |= {"complete": True, "observed": obs, "gate_passes": bool(gate), "gate_abs_diff": abs(obs - J["as_run"])}
        if gate:  # the null is read only after the gate passes
            null = np.array([cols[c]["mean"] for c in range(1, N_COLS + 1)])
            p95, sd = float(np.percentile(null, 95)), float(null.std(ddof=1))
            reading = READINGS[0] if obs > p95 + sd else READINGS[1] if obs >= p95 - sd else READINGS[2]
            r |= {"null": null.tolist(), "null_mean": float(null.mean()), "null_p95": p95, "null_sd": sd,
                  "p": float((1 + (null >= obs).sum()) / (1 + N_COLS)), "reading": reading}
        res[job] = r
    return res


@app.local_entrypoint()
def main(tag: str):
    me = str(Path(__file__).resolve())
    if subprocess.run(["git", "status", "--porcelain", "--", me, str(HERE / "RULE.md")], capture_output=True, text=True, cwd=HERE).stdout.strip():
        raise SystemExit("commit the runner and RULE.md first")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    calls = [run.spawn(job, ch, tag, commit) for job, J in JOBS.items() if J["family"] == FAMILY for ch in CHUNKS]
    print(json.dumps({"app": APP, "family": FAMILY, "tag": tag, "commit": commit, "calls": [c.object_id for c in calls]}), flush=True)
    for c in calls:
        print(json.dumps(c.get()), flush=True)


@app.local_entrypoint()
def report(tag: str, out: str):
    res = summarize.remote(tag, FAMILY)
    with Path(out).open("x") as fh:
        fh.write(json.dumps({"label": "POST HOC (RULE.md)", "family": FAMILY, "tag": tag, "jobs": res}, indent=1))
    for job, r in res.items():
        if not r.get("complete"):
            print(f"{job}: INCOMPLETE or INCONSISTENT, columns {r['columns_found']}, one provenance {r['one_provenance'] if 'one_provenance' in r else '-'}, aurocs ok {r.get('aurocs_ok', '-')}")
        elif not r["gate_passes"]:
            print(f"{job}: GATE FAILS, column 0 {r['observed']:.4f} vs as-run {r['as_run']} (no null reported)")
        else:
            print(f"{job}: observed {r['observed']:.4f} (as-run {r['as_run']}) | matched null mean {r['null_mean']:.4f}, 95th {r['null_p95']:.4f}, "
                  f"SD {r['null_sd']:.4f} | p {r['p']:.3f} | unmatched 95th {r['unmatched_p95']} | {r['reading']}")
