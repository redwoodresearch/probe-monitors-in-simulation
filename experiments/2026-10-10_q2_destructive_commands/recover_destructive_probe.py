"""Destructive-command probe: rebuild the job's output from its checkpoints (Modal CPU, about a cent).

The 14:51 PT launch (ap-9lXAKlLNaJAxGS65AUzYI7, stamp 2026-10-10_145058) lost its local client in a network outage
and was stopped at 15:40 PT near its spend stop, with 50 of 105 (draw, fold) results checkpointed. This rebuilds the
rows exactly as run_destructive_probe.run() does (the same code, copied, so the same order), checks each checkpoint's
test indices against the rebuilt folds, and assembles the same output for the draws whose five folds are all present.
Partial draws are listed and left out.

  uv run modal run experiments/2026-10-10_q2_destructive_commands/recover_destructive_probe.py --labels <labels jsonl> --stamp 2026-10-10_145058
"""
import hashlib
import json
from pathlib import Path

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-destructive_probe_kill-2026-11-08"
DUMP = "2026-10-06_la_segment-end-v4_qwen35-27b_alllayers_v1"
ARMS = ("real-honest", "sim-honest")
LAYERS = list(range(3, 64, 4))
CS = (0.001, 0.01, 0.1)
app = modal.App(APP)
acts = modal.Volume.from_name("probe-shift_aswin-surya_sub-qwen35_acts_kill-2026-11-06")
ckpt = modal.Volume.from_name(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy", "scikit-learn==1.7.2")


def fold_of(task: str) -> int:
    return int(hashlib.md5(task.encode()).hexdigest(), 16) % 5


@app.function(image=image, volumes={"/acts": acts.read_only(), "/ckpt": ckpt.read_only()}, cpu=8, memory=16384, timeout=900, retries=0)
def recover(labels: list[dict], stamp: str) -> dict:
    import re
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    from sklearn.metrics import roc_auc_score

    # Rows: copied from run_destructive_probe.run()
    root = Path("/acts") / DUMP
    lab = {(r["arm"], r["file"], r["sample_id"], r["epoch"], r["step"], r["action"]): r for r in labels}
    paths = [p for arm in ARMS for p in sorted((root / "meta" / arm).glob("*.json"))]
    with ThreadPoolExecutor(32) as ex:
        metas = list(ex.map(lambda p: json.loads(p.read_text()), paths))
    rows, unmatched, kept_calls = [], 0, 0  # rows: (world, task, run, chunk, row, y, step, action)
    for m in metas:
        run_key = f"{m['arm']}|{m['file']}|{m['sample_id']}|{m['epoch']}"
        for p in m["positions"]:
            if p["kind"] != "call_end" or not p.get("kept", True) or p["row"] is None:
                continue
            kept_calls += 1
            r = lab.get((m["arm"], m["file"], m["sample_id"], m["epoch"], p["step"], p["action"]))
            if r is None:
                unmatched += 1
                continue
            if r["impactful"]:
                rows.append((0 if m["arm"] == "real-honest" else 1, r["main_task"], run_key, m["chunk"], m["row_start"] + p["row"], int(r["destructive"]), p["step"], p["action"]))
    if unmatched > 0.005 * kept_calls:
        raise ValueError(f"{unmatched} of {kept_calls} kept call_end positions have no label")
    world = np.array([r[0] for r in rows]); y = np.array([r[5] for r in rows])
    task = np.array([r[1] for r in rows]); run_id = np.array([r[2] for r in rows])
    fold = np.array([fold_of(t) for t in task])

    def shuffled(seed):
        """Labels permuted within each run (the run keeps its count of destructive calls)."""
        if seed == 0:
            return y
        rng = np.random.default_rng(seed)
        out = y.copy()
        for r in np.unique(run_id):
            i = np.flatnonzero(run_id == r)
            out[i] = rng.permutation(out[i])
        return out

    # Checkpoints: each one's test indices must be exactly the rebuilt fold
    found = {}
    for f in sorted((Path("/ckpt") / stamp).glob("d*_f*.npz")):
        d, k = map(int, re.fullmatch(r"d(\d+)_f(\d)\.npz", f.name).groups())
        z = np.load(f)
        if not np.array_equal(z["idx"], np.flatnonzero(fold == k)):
            raise ValueError(f"{f.name}: test indices differ from the rebuilt fold {k}")
        meta = json.loads(z["meta"].item())
        found[(d, k)] = (d, k, meta["layer"], meta["C"], meta["threshold"], z["idx"], z["s"], meta["selection fits not converged"], meta["refit not converged"])
    complete = sorted(d for d in {d for d, _ in found} if all((d, k) in found for k in range(5)))
    partial = sorted({d for d, _ in found} - set(complete))
    if 0 not in complete:
        raise ValueError("the real-label draw is incomplete")
    res = [found[(d, k)] for d in complete for k in range(5)]

    # Assembly: copied from run_destructive_probe.run(), over the complete draws
    scores = {d: np.full(len(y), np.nan) for d in complete}
    caught = {d: np.zeros(len(y), bool) for d in complete}
    picks = {}
    sel_bad = ref_bad = 0
    for d, k, L, C, thr, idx, s, nsel, nref in res:
        scores[d][idx] = s; caught[d][idx] = s > thr
        picks.setdefault(d, {})[k] = {"layer": L, "C": C, "threshold": thr, "refit converged": not nref}
        sel_bad += nsel; ref_bad += nref

    def fold_auroc(yy, sc, w):
        """Mean over test folds of the AUROC within the fold (scores from different folds' probes are not pooled)."""
        a = []
        for k in range(5):
            m = (world == w) & (fold == k)
            if len(set(yy[m])) == 2:
                a.append(roc_auc_score(yy[m], sc[m]))
        return float(np.mean(a)), len(a)

    draws = []
    for d in complete:
        yy = shuffled(d)
        tpr = [caught[d][(world == w) & (yy == 1)].mean() for w in (0, 1)]
        (ar, nr), (as_, ns) = fold_auroc(yy, scores[d], 0), fold_auroc(yy, scores[d], 1)
        draws.append({"draw": d, "TPR real minus sim": float(tpr[0] - tpr[1]), "AUROC real (fold mean)": ar, "AUROC sim (fold mean)": as_,
                      "folds in AUROC real": nr, "folds in AUROC sim": ns, "picks": picks[d],
                      "labels": "".join(map(str, yy)), "caught": "".join("1" if c else "0" for c in caught[d]),
                      "scores": [round(float(x), 5) for x in scores[d]]})
    return {"rows": [{"world": int(world[i]), "task": task[i], "run": run_id[i], "step": rows[i][6], "action": rows[i][7], "y": int(y[i]),
                      "fold": int(fold[i])} for i in range(len(y))],
            "draws": draws, "unmatched": unmatched, "kept call_end positions": kept_calls,
            "selection fits not converged": sel_bad, "refits not converged": f"{ref_bad} of {len(res)}", "checkpoints": f"{APP}:/{stamp}",
            "recovered from checkpoints": {"checkpoints found": len(found), "complete draws": complete, "partial draws left out": partial}}


@app.local_entrypoint()
def main(labels: str, stamp: str):
    rows = [json.loads(line) for line in Path(labels).read_text().splitlines()]
    (HERE.parents[1] / "data" / "destructive").mkdir(parents=True, exist_ok=True)  # before any remote work
    res = recover.remote(rows, stamp) | {"labels sha256": hashlib.sha256(Path(labels).read_bytes()).hexdigest(), "layers": LAYERS, "C": CS}
    full = HERE.parents[1] / "data" / "destructive" / f"destructive_probe_{stamp}.json"
    if full.exists():
        raise FileExistsError(full)
    full.write_text(json.dumps(res) + "\n")
    keep = ("draw", "TPR real minus sim", "AUROC real (fold mean)", "AUROC sim (fold mean)", "folds in AUROC real", "folds in AUROC sim", "picks")
    summary = {k: v for k, v in res.items() if k not in ("rows", "draws")} | {
        "draws": [{k: d[k] for k in keep} for d in res["draws"]], "rows": len(res["rows"]),
        "full output": str(full.relative_to(HERE.parents[1])), "full output sha256": hashlib.sha256(full.read_bytes()).hexdigest()}
    out = HERE / "results" / f"destructive_probe_{stamp}.json"
    out.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary["draws"][0] | {"recovered": res["recovered from checkpoints"], "selection fits not converged": res["selection fits not converged"],
                                            "refits not converged": res["refits not converged"]}))
    print(out, full)
