"""Honest-arm probe-vs-TF-IDF comparisons on batch-matched cells, from saved scores (no fits).

A cell's first real honest run by key can come from the base200 batch (container umask: group-writable `ls -l` modes),
while the simulated honest runs reference overlap800 runs. So in base200 cells real vs simulated also differs by batch.
For each analysis this recomputes its protocol report's comparison exactly as `protocol_checks` does: per output, the
headline's out-of-fold step score (each outer fold's chosen layer) and TF-IDF's own-output score; run means over the
outputs both scored; run AUROCs and their paired main-task bootstrap gap. It must reproduce the report on all cells,
then gives the same numbers on batch-matched runs (honest runs only where the first real run is overlap800; attack runs,
which carry no batch prefix, always kept) and on the base200 cells' honest runs.

`--analyses` is a JSON list of {label, results, report, run_scores, steps, tfidf_env, tfidf_task, readout} (local
paths; the output records every input by name, sha256 and S3 URI instead). The TF-IDF baseline is the report's. The
report's bootstrap (`q1_eval.boot_aucs`) uses torch, which only the Modal image installs, so add it for this run:

  uv run --with torch python experiments/2026-10-06_q1_segment_end/analyze_batch_matched_gap.py --analyses <list.json>
"""

import argparse
import gzip
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from probe_shift.provenance import capture
from probe_shift.q1_eval import Scorer

HERE = Path(__file__).parent
SPLITS = {"A_held_out_environment": ("headline_held_out_environment", "__held_out_environment"),
          "B_main_task": ("headline", "")}
OUTPUTS = {"all_outputs": "all", "no_editor_outputs": "no_editor"}
HONEST = {"real-honest", "real-honest-2", "sim-honest"}


def archive_index() -> dict:
    """Archived copy of each input: results files by name, step scores by '<analysis id>/<name>', TF-IDF files by name."""
    uris = {}
    for r in map(json.loads, (HERE / "s3_archive_manifest.jsonl").read_text().splitlines()):
        uris["/".join(r["path"].split("/")[-2:]) if "/step_scores/" in r["uri"] else r["path"].split("/")[-1]] = r
    for k, v in json.loads((HERE / "tfidf_inputs_s3.json").read_text())["files"].items():
        uris[Path(k).name] = v
    return uris


def archived(path: Path, uris: dict, key: str | None = None) -> dict:
    """An input as a fresh checkout finds it: name, sha256 and S3 URI (the archive must hold these exact bytes)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    row = uris.get(key or path.name)
    if row is None or row["sha256"] != h.hexdigest():
        raise ValueError(f"{key or path.name}: {'not in' if row is None else 'differs from'} the archive manifests")
    return {"name": key or path.name, "sha256": row["sha256"], "uri": row["uri"]}


def tfidf_swap(sc, runs: list[dict], d: dict, base200) -> dict:
    """TF-IDF alone, same cells, real run swapped: each cell's first real run is replaced by the task's second real run,
    which the real-vs-sim TF-IDF also scored out of fold. The second run is always overlap800, so in base200 cells the
    swap changes the batch and the run; in overlap800 cells (both runs overlap800) it changes only the run, which is the
    control for run identity. The probe never scored second runs (its weights are not kept), so it has no such row.
    Run means over each run's own scored outputs (not the probe's shared outputs), original and swapped, per cell set.
    Cells without a second real run are left out of both."""
    by_key = {}
    for k, s in d.items():
        by_key.setdefault(k.split("|")[0], []).append(s)
    run_tf = np.array([np.mean(by_key[m["key"]]) if m["key"] in by_key else np.nan for m in runs])
    second = {m["cell"]: i for i, m in enumerate(runs) if m["arm"] == "real-honest-2"}
    first_real = {runs[i]["cell"]: i for i in np.flatnonzero(sc.runs_in) if runs[i]["arm"] == "real-honest"}
    two_run = {c for c in first_real if c in second}
    out = {}
    for cells, pick in (("base200_cells", lambda c: base200[first_real[c]]), ("overlap800_cells", lambda c: not base200[first_real[c]])):
        chosen = {c for c in two_run if pick(c)}
        idx = np.array([i for i in np.flatnonzero(sc.runs_in) if runs[i]["cell"] in chosen])
        swapped = np.array([second[runs[i]["cell"]] if runs[i]["arm"] == "real-honest" else i for i in idx])
        y = sc.y_run[idx]  # a swapped-in second run keeps its first run's label (real)
        out[cells] = {"cells": len(chosen), "tfidf_original": sc.auc_ci(y, run_tf[idx], sc.main[idx]),
                      "tfidf_swapped": sc.auc_ci(y, run_tf[swapped], sc.main[idx])}
    return out


def analyze(a: dict, uris: dict) -> tuple[dict, list[str]]:
    res, report = json.loads(Path(a["results"]).read_text()), json.loads(Path(a["report"]).read_text())
    cfg, readout, steps_dir = res["config"], a["readout"], Path(a["steps"])
    runs = [{k: r[k] for k in ("arm", "key", "cell", "main_task", "fold", "environment")}
            for r in map(json.loads, Path(a["run_scores"]).read_text().splitlines())]
    keys = np.array([m["key"] for m in runs], object)
    real_first = {m["cell"]: m["key"].split("_")[0] for m in runs if m["arm"] == "real-honest"}
    if set(real_first.values()) != {"base200", "overlap800"}:
        raise ValueError(f"{a['label']}: unexpected batches {set(real_first.values())}")
    honest = np.array([m["arm"] in HONEST for m in runs])
    base200 = honest & np.array([real_first.get(m["cell"]) == "base200" for m in runs])
    tfidf = {}
    for split, path in (("A_held_out_environment", a["tfidf_env"]), ("B_main_task", a["tfidf_task"])):
        (baseline,) = report["tfidf_baselines"][split]
        for r in map(json.loads, gzip.open(path, "rt")):
            if r["baseline"] == baseline:
                d = tfidf.setdefault(f"{split}|{r['comparison']}|{r['outputs']}", {})
                k = f"{r['key']}|{r['step']}|{r['action']}"
                if k in d:
                    raise ValueError(f"{Path(path).name}: two scores for {k}")
                d[k] = r["score"]
    steps, rows_out, md, md_swap = {}, {}, [], []
    fmt = lambda m: f"{m['auroc']:.3f} [{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}]"
    for split, (hkey, suffix) in SPLITS.items():
        fold = np.array([cfg["environment_to_fold"][m["environment"]] if split.startswith("A") else m["fold"] for m in runs])
        for variant, short in OUTPUTS.items():
            h = res[hkey][readout]["real_vs_sim"][variant]
            t = hs = None
            for f, l in h["chosen_layer_by_fold"].items():
                p = steps_dir / f"L{l}_step_scores_{readout}.npz"
                if p not in steps:
                    steps[p] = np.load(p)
                z = steps[p]
                if t is None:
                    t = {k: z[k] for k in ("run", "step", "action")}
                    hs = np.full(len(t["run"]), np.nan)
                elif not all(np.array_equal(t[k], z[k]) for k in t):
                    raise ValueError(f"{p.name}: rows differ from the other layers'")
                rows = fold[t["run"]] == int(f)
                hs[rows] = z[f"real_vs_sim__{variant}{suffix}"][rows]
            sc = Scorer(cfg, runs, "real_vs_sim")
            d = tfidf[f"{split}|real_vs_sim|{short}"]
            rows = sc.runs_in[t["run"]] & ~np.isnan(hs)
            v = np.array([d.get(f"{keys[i]}|{st}|{ac}", np.nan) for i, st, ac in zip(t["run"], t["step"], t["action"])])
            hit = rows & ~np.isnan(v)
            tf, ps = sc.run_scores(t["run"][hit], v[hit]), sc.run_scores(t["run"][hit], hs[hit])
            row = {}
            for name, m in (("all_cells", sc.runs_in), ("batch_matched", sc.runs_in & ~base200), ("base200_cells", sc.runs_in & base200)):
                row[name] = {"probe_on_shared_outputs": sc.auc_ci(sc.y_run[m], ps[m], sc.main[m]),
                             "tfidf": sc.auc_ci(sc.y_run[m], tf[m], sc.main[m]),
                             "gap_probe_minus_tfidf": sc.auc_gap(sc.y_run[m], ps[m], tf[m], sc.main[m])}
            rep = report["results"][split][f"{readout}__real_vs_sim__{variant}"]
            for k in ("probe_on_shared_outputs", "tfidf", "gap_probe_minus_tfidf"):
                if row["all_cells"][k] != rep[k]:
                    raise ValueError(f"{a['label']} {split} {variant} {k}: recomputed {row['all_cells'][k]} vs reported {rep[k]}")
            row["tfidf_swap"] = sw = tfidf_swap(sc, runs, d, base200)
            md_swap.append(f"| {a['label']} | {split.split('_', 1)[1].replace('_', ' ')} | {variant.replace('_', ' ')} | "
                           f"{fmt(sw['base200_cells']['tfidf_original'])} | {fmt(sw['base200_cells']['tfidf_swapped'])} "
                           f"({sw['base200_cells']['cells']}) | {fmt(sw['overlap800_cells']['tfidf_original'])} | "
                           f"{fmt(sw['overlap800_cells']['tfidf_swapped'])} ({sw['overlap800_cells']['cells']}) |")
            rows_out[f"{split}|{variant}"] = row
            for name, r in ((k, v) for k, v in row.items() if k != "tfidf_swap"):
                g = r["gap_probe_minus_tfidf"]
                md.append(f"| {a['label']} | {split.split('_', 1)[1].replace('_', ' ')} | {variant.replace('_', ' ')} | "
                          f"{name.replace('_', ' ')} ({g['n_tasks']}) | {fmt(r['probe_on_shared_outputs'])} | {fmt(r['tfidf'])} | "
                          f"{g['gap']:+.4f} [{g['ci95'][0]:+.4f}, {g['ci95'][1]:+.4f}] |")
    # Every input by role, so the analyses list can be rebuilt from the archive alone.
    inputs = {k: archived(Path(a[k]), uris) for k in ("results", "report", "run_scores", "tfidf_env", "tfidf_task")}
    inputs["steps"] = [archived(p, uris, f"{cfg['analysis_id']}/{p.name}") for p in sorted(steps)]
    cells = {"all": len(real_first), "batch_matched": int(sum(v == "overlap800" for v in real_first.values())),
             "base200": int(sum(v == "base200" for v in real_first.values()))}
    return {"readout": readout, "honest_cells": cells, "inputs": inputs, "rows": rows_out}, md, md_swap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyses", type=Path, required=True)
    args = ap.parse_args()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    uris = archive_index()
    out = {"stamp": stamp, "provenance": capture(HERE.parent.parent, [Path(__file__)]), "analyses": {}}
    md = [f"# Probe vs TF-IDF on batch-matched cells ({stamp})", "",
          ("Probe (headline step scores) and TF-IDF on the outputs both scored, run means; paired main-task bootstrap. "
           + "Batch-matched: honest runs only where the first real run is overlap800 (attack runs always kept)."), "",
          "| Analysis | Split | Outputs | Runs (tasks) | Probe on shared outputs | TF-IDF | Gap [95% CI] |", "|---|---|---|---|---|---|---|"]
    swap = []
    for a in json.loads(args.analyses.read_text()):
        out["analyses"][a["label"]], rows, rows_swap = analyze(a, uris)
        md += rows
        swap += rows_swap
    md += ["", ("TF-IDF alone, same cells, first real run swapped for the task's second (always overlap800). In base200 "
                + "cells that changes batch and run; in overlap800 cells only the run (the control). Run means over each "
                + "run's own TF-IDF-scored outputs; main-task bootstrap; two-run cells only."), "",
           "| Analysis | Split | Outputs | base200 cells, original | base200 cells, swapped (cells) | overlap800 cells, original | overlap800 cells, swapped (cells) |",
           "|---|---|---|---|---|---|---|"] + swap
    md += ["", "All-cell rows reproduce each analysis's protocol report exactly (checked). Honest cells: "
           + "; ".join(f"{k} {v['honest_cells']}" for k, v in out["analyses"].items())]
    (HERE / "results" / f"batch_matched_gap_{stamp}.json").write_text(json.dumps(out, indent=1))
    (HERE / "results" / f"batch_matched_gap_{stamp}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
