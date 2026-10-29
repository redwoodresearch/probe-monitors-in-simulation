"""False alarms counted per task, real vs simulated, Qwen acting (PLAN_POSTHOC_per_task_false_alarms_2026-10-11.md;
EXPLORATORY; $0, local; keys and scores only). The read-only calls of the attack runs (real run 1, simulated, run 2),
scored by draw 0's real-trained models at the frozen 1% threshold: per-task mean and pooled per-call rate per world,
real - sim and run 1 - run 2 paired by task, 2,000 within-fold task resamples (seed [20261011, 80, b]) with thresholds
re-estimated.

  uv run --no-project --with numpy python analyze_POSTHOC_per_task_false_alarms.py <main per-call>
"""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from analyze_scaleup_plants import threshold

HERE = Path(__file__).parent
N_BOOT = 2000
CELLS = {"real": "real->real", "sim": "real->sim", "run2": "real->run2"}


def main(main_path: str):
    if subprocess.run(["git", "status", "--porcelain", __file__], capture_output=True, text=True).stdout:
        raise SystemExit("commit the script first")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    raw = Path(main_path).read_bytes()
    M = json.loads(raw)
    folds, meta = M["folds"], M["meta"]
    tasks_all = sorted({m["task"] for m in meta})
    ti = {t: i for i, t in enumerate(tasks_all)}
    fold_of = np.array([folds[t] for t in tasks_all])
    cal = M["draws"][0]["cells"]["real->real"]["per_call"]["calibration"]
    cs = [(ti[meta[i]["task"]], s, k) for i, s, k in zip(cal["row"], cal["score"], cal["model_fold"]) if meta[i]["world"] == "real" and meta[i]["y"] == 0]
    c_t, c_s, c_k = (np.array(x) for x in zip(*cs))

    def thresholds(w):
        out = []
        for k in range(5):
            m = (c_k == k) & (w[c_t] > 0)
            if not m.any():
                return None
            out.append(threshold(c_s[m], w[c_t[m]]))
        return np.array(out)

    one = np.ones(len(tasks_all))
    th0 = thresholds(one)
    D = {}
    for w, cell in CELLS.items():
        pc = M["draws"][0]["cells"][cell]["per_call"]
        if not len(pc["row"]) == len(pc["score"]) == len(pc["flag_frozen"]) or len(set(pc["row"])) != len(pc["row"]):
            raise SystemExit(f"{cell}: misaligned or repeated per-call rows")
        if any(meta[r]["world"] != w for r in pc["row"]):
            raise SystemExit(f"{cell}: a row from another world")
        t = np.array([ti[meta[r]["task"]] for r in pc["row"]])
        f = np.array([folds[meta[r]["task"]] for r in pc["row"]])
        s, y = np.array(pc["score"]), np.array([meta[r]["y"] for r in pc["row"]])
        if not np.array_equal(s > th0[f], np.array(pc["flag_frozen"], dtype=bool)):
            raise SystemExit(f"{cell}: the recomputed flags differ from flag_frozen")
        if abs((s > th0[f])[y == 0].mean() - M["draws"][0]["cells"][cell]["values"]["false_alarm_frozen"]) > 1e-9:
            raise SystemExit(f"{cell}: the pooled false-alarm rate differs from the file's")
        ro = y == 0
        D[w] = {"t": t[ro], "f": f[ro], "s": s[ro], "n": np.bincount(t[ro], minlength=len(tasks_all))}

    def per_world(d, th):
        """Per task: read-only calls and flagged ones."""
        flag = d["s"] > th[d["f"]]
        return d["n"], np.bincount(d["t"], weights=flag, minlength=len(tasks_all))

    def rates(d, th, w, tasks):
        """Per-task mean (each task once, weighted by resample multiplicity) and pooled per-call rate, over `tasks`."""
        n, k = per_world(d, th)
        wt = w * tasks
        if (wt * (n > 0)).sum() == 0:
            return None, None
        per_task = np.divide(k, n, out=np.zeros_like(k), where=n > 0)
        return (wt * per_task).sum() / wt.sum(), (wt * k).sum() / (wt * n).sum()

    W = [np.zeros(len(tasks_all)) for _ in range(N_BOOT)]
    for b in range(N_BOOT):
        rng = np.random.default_rng([20261011, 80, b + 1])
        for k in range(5):
            idx = np.flatnonzero(fold_of == k)
            np.add.at(W[b], rng.choice(idx, len(idx), replace=True), 1)
    TH = [thresholds(w) for w in W]
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    out = {"label": "EXPLORATORY: false alarms counted per task (PLAN_POSTHOC_per_task_false_alarms_2026-10-11.md)", "git": git,
           "inputs": {Path(main_path).name: hashlib.sha256(raw).hexdigest()}, "thresholds_point": th0.tolist(),
           "checks": "flags reproduce flag_frozen on every row of the three cells; pooled false alarms equal the file's",
           "worlds": {}, "comparisons": {}}
    for w, d in D.items():
        has = (d["n"] > 0).astype(float)
        pt, pooled = rates(d, th0, one, has)
        out["worlds"][w] = {"tasks_with_read_only_calls": int(has.sum()), "tasks_without": [tasks_all[i] for i in np.flatnonzero(d["n"] == 0)],
                            "read_only_calls": int(d["n"].sum()), "per_task_mean": pt, "pooled": pooled}
    for name, a, b in (("real - sim", "real", "sim"), ("real - run2", "real", "run2")):
        both = ((D[a]["n"] > 0) & (D[b]["n"] > 0)).astype(float)
        pa, pb = rates(D[a], th0, one, both), rates(D[b], th0, one, both)
        re_t, re_p, fx_t = [], [], []
        for th, w in zip(TH, W):
            if th is None or (w * both).sum() == 0:
                continue
            ra, rb = rates(D[a], th, w, both), rates(D[b], th, w, both)
            re_t.append(ra[0] - rb[0]); re_p.append(ra[1] - rb[1])
            fa, fb = rates(D[a], th0, w, both), rates(D[b], th0, w, both)
            fx_t.append(fa[0] - fb[0])
        ok = N_BOOT - len(re_t) <= 0.05 * N_BOOT
        out["comparisons"][name] = {"paired_tasks": int(both.sum()), "dropped": N_BOOT - len(re_t),
                                    "per_task": {"point": pa[0] - pb[0], a: pa[0], b: pb[0], "ci95": ci(re_t) if ok else None, "ci95_fixed": ci(fx_t) if ok else None},
                                    "pooled": {"point": pa[1] - pb[1], a: pa[1], b: pb[1], "ci95": ci(re_p) if ok else None}}

    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"per_task_false_alarms_POSTHOC_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out))
    pct = lambda v: [round(x * 100, 2) for x in v] if v else None
    for w, v in out["worlds"].items():
        print(f"{w}: tasks {v['tasks_with_read_only_calls']} (without: {len(v['tasks_without'])}), read-only calls {v['read_only_calls']}, "
              f"per-task mean {v['per_task_mean'] * 100:.2f}%, pooled {v['pooled'] * 100:.2f}%")
    for k, v in out["comparisons"].items():
        print(f"{k} on {v['paired_tasks']} tasks: per task {v['per_task']['point'] * 100:+.2f} {pct(v['per_task']['ci95'])} (fixed {pct(v['per_task']['ci95_fixed'])}); "
              f"pooled {v['pooled']['point'] * 100:+.2f} {pct(v['pooled']['ci95'])}; dropped {v['dropped']}")
    print("->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("main_path")
    main(ap.parse_args().main_path)
