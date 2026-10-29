"""Standardized partial AUROC over false alarms 0-5% (sklearn's max_fpr=0.05, McClish's standardization: 0.5 chance,
1 perfect; amendment 40's second main measure) for every replicate and shuffle draw of the refit bootstrap, from the
per-row scores each item saved. Runs on Modal CPU next to the Volume; nothing is downloaded.

Each cell pools its five test folds (each row scored by its own fold's probe) and weights rows by their main task's
multiplicity, as the replicate did by repeating rows. Check first: the AUROC rebuilt this way must match the AUROC
each item stored, or the run stops. The local client writes results/refit_pauc_<stamp>.json (per reader: each cell's
point estimate, 95% percentile interval and shuffle null; the four contrasts signed simulated minus real, amendment 43)
and results/refit_pauc_items_<stamp>.json (every item's values, for S3).

  uv run modal run experiments/2026-10-08_impactful_refit_bootstrap/analyze_refit_pauc.py
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal
import yaml

HERE = Path(__file__).parent
CFG = yaml.safe_load((HERE / "config" / "refit.yaml").read_text())
app = modal.App("probe-shift_aswin-surya_sub-q2-refit_pauc_kill-2026-11-08")
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy", "scikit-learn", "pyyaml")
         .add_local_file(HERE / "config" / "refit.yaml", "/root/config/refit.yaml"))  # the module reads it at import
out_vol = modal.Volume.from_name(CFG["out_volume"], version=2)
CELLS = {"real->real": (0, 0), "real->sim": (0, 1), "sim->real": (1, 0), "sim->sim": (1, 1)}  # (training, test world)
CONTRASTS = {"domain gap, trained on real: real->sim − real->real": ("real->sim", "real->real"),  # simulated minus real
             "domain gap, trained on sim: sim->sim − sim->real": ("sim->sim", "sim->real"),
             "transfer, tested on sim: sim->sim − real->sim": ("sim->sim", "real->sim"),
             "transfer, tested on real: sim->real − real->real": ("sim->real", "real->real")}


def score_item(args) -> dict:
    """One item's AUROC and pAUC per cell, from its scores file."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    npz, world, fold, task = args
    z = np.load(npz)
    s, y, w = z["scores"], z["labels"], z["task_counts"][task]
    out = {}
    for cell, (t, v) in CELLS.items():
        i = np.flatnonzero((world == v) & (w > 0))
        ss = s[fold[i], t, i]
        if np.isnan(ss).any():
            raise ValueError(f"{npz} {cell}: NaN scores")
        out[cell] = {"auroc": roc_auc_score(y[i], ss, sample_weight=w[i]), "pauc_0_5pct": roc_auc_score(y[i], ss, sample_weight=w[i], max_fpr=0.05)}
    return out | {"rows_sha256": str(z["rows_sha256"]), "tasks": z["tasks"].tolist()}


def configured_items(items: list[dict]) -> tuple[list[dict], int]:
    """The items in the config's (kind, replicate) set, and how many others were left out. The fingerprint leaves out
    the replicate counts, so the cached aggregate can hold items beyond the config (as the runner's aggregate step
    allows); every configured item must be present."""
    want = {("boot", i) for i in range(CFG["n_boot"] + 1)} | {("null", i) for i in range(1, CFG["n_null"] + 1)}
    have = {(it["kind"], it["replicate"]) for it in items}
    if want - have:
        raise ValueError(f"{len(want - have)} configured items missing")
    return [it for it in items if (it["kind"], it["replicate"]) in want], len(have - want)


@app.function(image=image, volumes={"/out": out_vol}, cpu=8, memory=16384, timeout=1800, retries=0)
def rescore() -> dict:
    from multiprocessing import Pool

    import numpy as np

    agg = json.loads((Path("/out") / CFG["out_run"] / "stats" / "aggregate_items.json").read_text())
    res = {}
    for reader in CFG["readers"]:
        d = Path("/out") / CFG["out_run"] / reader
        meta = json.loads((d / "rows.json").read_text())
        rows_sha = hashlib.sha256((d / "rows.json").read_bytes()).hexdigest()
        world = np.array([0 if m["arm"] == CFG["arms"]["real"] else 1 for m in meta])
        fold = np.array([m["fold"] for m in meta])
        tasks, task = np.unique([m["main_task"] for m in meta], return_inverse=True)
        reps = d / "replicates" / agg[reader]["fingerprint"]
        configured, extra = configured_items(agg[reader]["items"])
        items = [it for it in configured if "undefined" not in it]
        with Pool() as pool:
            got = pool.map(score_item, [(reps / it["scores_file"], world, fold, task) for it in items])
        diffs = []
        for it, g in zip(items, got):
            if g["rows_sha256"] != rows_sha or g["tasks"] != tasks.tolist():
                raise ValueError(f"{reader} {it['scores_file']}: rows or task order differ from rows.json")
            diffs += [abs(g[c]["auroc"] - it[c]["auroc"]) for c in CELLS]
        if max(diffs) > 1e-9:
            raise ValueError(f"{reader}: rebuilt AUROC differs from the stored AUROC by up to {max(diffs)}")
        res[reader] = {"fingerprint": agg[reader]["fingerprint"], "auroc check (max abs difference)": max(diffs),
                       "undefined items": len(configured) - len(items), "left out (beyond the configured counts)": extra,
                       "items": [{"kind": it["kind"], "replicate": it["replicate"]} | {c: g[c]["pauc_0_5pct"] for c in CELLS} for it, g in zip(items, got)]}
    return res


def summarize(items: list[dict]) -> dict:
    """As run_refit_bootstrap.summarize: point (replicate 0), 95% percentile interval over replicates 1..n, shuffle null."""
    import numpy as np

    point = next(r for r in items if r["kind"] == "boot" and r["replicate"] == 0)
    boots = [r for r in items if r["kind"] == "boot" and r["replicate"] != 0]
    nulls = [r for r in items if r["kind"] == "null"]
    ci = lambda vals: [round(float(x), 4) for x in np.nanpercentile(vals, [2.5, 97.5])]
    out = {"replicates": {"bootstrap": len(boots), "null draws": len(nulls)}}
    for c in CELLS:
        v = np.array([r[c] for r in nulls])
        out[c] = {"value": round(point[c], 4), "ci95": ci([b[c] for b in boots]),
                  "null": {"mean": round(float(v.mean()), 4), "max": round(float(v.max()), 4), "p": round(float((1 + (v >= point[c]).sum()) / (1 + len(v))), 4)}}
    for name, (a, b) in CONTRASTS.items():
        out[name] = {"value": round(point[a] - point[b], 4), "ci95": ci([r[a] - r[b] for r in boots])}
    return out


@app.local_entrypoint()
def main():
    res = rescore.remote()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    summary = {"measure": "standardized pAUC, false alarms 0-5% (sklearn max_fpr=0.05)", "config": CFG,
               "readers": {r: {k: v[k] for k in ("fingerprint", "auroc check (max abs difference)", "undefined items", "left out (beyond the configured counts)")} | summarize(v["items"])
                           for r, v in res.items()}}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"refit_pauc_{stamp}.json").write_text(json.dumps(summary, indent=1))
    (HERE / "results" / f"refit_pauc_items_{stamp}.json").write_text(json.dumps(res))
    for r, v in summary["readers"].items():
        print(f"== {r}: {v['replicates']}; AUROC check max abs difference {v['auroc check (max abs difference)']}; undefined {v['undefined items']}")
        for k in list(CELLS) + list(CONTRASTS):
            print(f"  {k} | pAUC 0-5% {v[k]['value']:+.4f} {v[k]['ci95']}" + (f" (null mean {v[k]['null']['mean']}, max {v[k]['null']['max']})" if "null" in v[k] else ""))
    print(HERE / "results" / f"refit_pauc_{stamp}.json")
