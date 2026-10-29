"""The honest real-vs-real control split by batch, from saved headline scores (no fits).

A task's two real honest runs are its first and second by key. The first can come from the base200 batch (keys sort
`base200_…` < `overlap800_…`), whose container umask makes `ls -l` modes group-writable; the second is always
overlap800. So a pair is base200/overlap800 (mixed) or overlap800/overlap800 (same batch). For each analysis, split
and output variant, this recomputes the control's run AUROC (second run vs first, the headline layer chosen by fold)
on all pairs, which must match the report, then on same-batch and mixed pairs. Pooled analyses keep their attack
pairs (e1 vs e2, no batch prefix) and drop or keep honest pairs the same way.

  uv run python experiments/2026-10-06_q1_segment_end/analyze_honest_floor_batches.py \\
      --analysis amendment5=results/q1_gpu_<id>.json:<run_scores.jsonl>:output_end_ema0.95 ...
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from probe_shift.provenance import capture
from probe_shift.scoring import auroc_ci

HERE = Path(__file__).parent
SECOND = {"real-honest-2", "real-attack-e2"}
CONTROL = {"real-honest", "real-honest-2", "real-attack-e1", "real-attack-e2"}
SPLITS = {"A_held_out_environment": "headline_held_out_environment", "B_main_task": "headline"}


def archived(path: str, manifest: dict) -> dict:
    """An input as a fresh checkout can find it: its name, sha256 and S3 URI from the archive manifest (which must hold
    these exact bytes)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    name = Path(path).name
    row = manifest.get(name)
    if row is None or row["sha256"] != h.hexdigest():
        raise ValueError(f"{name}: {'not in' if row is None else 'differs from'} s3_archive_manifest.jsonl")
    return {"name": name, "sha256": row["sha256"], "uri": row["uri"]}


def runs_of(path: str) -> list[dict]:
    """Arm, key and main task of every run, in the order of the headline's run_scores."""
    out = []
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            out.append({"arm": r["arm"], "key": r["key"], "main_task": r["main_task"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", nargs="+", required=True, help="label=results.json:run_scores.jsonl:readout")
    args = ap.parse_args()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = {"stamp": stamp, "provenance": capture(HERE.parent.parent, [Path(__file__)]), "analyses": {}}
    manifest = {r["path"]: r for r in map(json.loads, (HERE / "s3_archive_manifest.jsonl").read_text().splitlines())
                if "/" not in r["path"]}  # results files are listed by name
    md = [f"# Honest real-vs-real control by batch ({stamp})", "",
          "Run AUROC, second real run vs first, headline layer by fold; 95% main-task bootstrap.", "",
          "| Analysis | Split | Outputs | Reported | All pairs (recomputed) | Same batch (overlap800/overlap800) | Mixed (base200/overlap800) |",
          "|---|---|---|---|---|---|---|"]
    fmt = lambda m: "n/a" if m is None else f"{m['auroc']:.3f} [{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}] ({m['n_main_tasks']} tasks)"
    for spec in args.analysis:
        label, rest = spec.split("=", 1)
        res_path, scores_path, readout = rest.split(":")
        res, runs = json.loads(Path(res_path).read_text()), runs_of(scores_path)
        batch = defaultdict(dict)  # honest task -> {arm: batch}
        for r in runs:
            if r["arm"] in ("real-honest", "real-honest-2"):
                batch[r["main_task"]][r["arm"]] = r["key"].split("_")[0]
        pattern = {t: f"{b['real-honest']}/{b['real-honest-2']}" for t, b in batch.items() if len(b) == 2}
        if not set(pattern.values()) <= {"base200/overlap800", "overlap800/overlap800"}:
            raise ValueError(f"{label}: unexpected batch patterns {Counter(pattern.values())}")
        entry = {"results": archived(res_path, manifest), "run_scores": archived(scores_path, manifest), "readout": readout,
                 "honest_pairs_by_pattern": dict(Counter(pattern.values())), "rows": {}}
        for split, hkey in SPLITS.items():
            for variant, h in res[hkey][readout]["null_e1_vs_e2"].items():
                s = h["run_scores"]
                if len(s) != len(runs):
                    raise ValueError(f"{label} {split} {variant}: {len(s)} scores for {len(runs)} runs")
                idx = [i for i, r in enumerate(runs) if s[i] is not None]
                if {runs[i]["arm"] for i in idx} - CONTROL:
                    raise ValueError(f"{label}: control scores on non-control arms")

                def auc(keep):
                    sel = [i for i in idx if keep(runs[i])]
                    y = np.array([runs[i]["arm"] in SECOND for i in sel])
                    return auroc_ci(y, np.array([s[i] for i in sel]), np.array([runs[i]["main_task"] for i in sel])) \
                        if 0 < y.sum() < len(y) else None

                honest = lambda r: r["arm"].startswith("real-honest")
                row = {"reported": h["run"],
                       "all_pairs": auc(lambda r: True),
                       "same_batch": auc(lambda r: not honest(r) or pattern.get(r["main_task"]) == "overlap800/overlap800"),
                       "mixed": auc(lambda r: honest(r) and pattern.get(r["main_task"]) == "base200/overlap800")}
                if abs(row["all_pairs"]["auroc"] - h["run"]["auroc"]) > 1e-4:
                    raise ValueError(f"{label} {split} {variant}: recomputed {row['all_pairs']['auroc']} vs reported {h['run']['auroc']}")
                entry["rows"][f"{split}|{variant}"] = row
                md.append(f"| {label} | {split.split('_', 1)[1].replace('_', ' ')} | {variant.replace('_', ' ')} | "
                          f"{h['run']['auroc']:.3f} | {fmt(row['all_pairs'])} | {fmt(row['same_batch'])} | {fmt(row['mixed'])} |")
        out["analyses"][label] = entry
    md += ["", "Honest pairs by batch pattern: " + "; ".join(f"{k} {v['honest_pairs_by_pattern']}" for k, v in out["analyses"].items())]
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"honest_floor_batches_{stamp}.json").write_text(json.dumps(out, indent=1))
    (HERE / "results" / f"honest_floor_batches_{stamp}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
