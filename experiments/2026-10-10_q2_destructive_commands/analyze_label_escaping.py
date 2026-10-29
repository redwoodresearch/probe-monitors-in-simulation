"""Destructive-command probe: how much the label rule's JSON escaping matters (local, free; added after review, Codex on
#440).

The rule ran on json.dumps(arguments). There a newline is the two characters "\\n", so a command written right after a newline
follows the letter n and a leading `\\b` fails: `\\nrm -rf` and `\\npkill` are missed. This counts, per world, the calls whose
verdict changes when the same rule runs on each raw argument string (real newlines), and rescores the probe's existing
scores and thresholds against the raw-string rule (nothing refit; descriptive). The rescore's null (added after review,
Codex on #450): the raw-string labels shuffled within each run, 1,000 draws (seed 0), on the same fixed scores and
thresholds, with no refit. It is a fixed-probe null, not selection-aware: the probes and thresholds were fit on the
original labels, a subset of these, so it is descriptive only. No command text is written.

  uv run --no-project --python 3.14 --with numpy --with scikit-learn --with boto3 --with pyyaml python \
      experiments/2026-10-10_q2_destructive_commands/analyze_label_escaping.py <transcript cache> <labels jsonl> <probe full output> [tag]
"""
import glob
import hashlib
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_destructive_counts import COHORTS, DESTRUCTIVE, PATTERNS
from analyze_destructive_probe import Tasks, fold_auroc, world_stats
from build_labels import ARM, calls

HERE = Path(__file__).parent


def raw_rule(args: dict) -> bool:
    return any(DESTRUCTIVE.search(v if isinstance(v, str) else json.dumps(v)) for v in args.values())


def null_draws(yr, run, world, fold, score, caught, n=1000, seed=0):
    """Fold-mean AUROC per world and the TPR gap, with the labels shuffled within each run (counts kept)."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(run == r) for r in np.unique(run)]
    out = []
    for _ in range(n):
        yp = yr.copy()
        for i in groups:
            yp[i] = rng.permutation(yp[i])
        tpr = [caught[(world == w) & (yp == 1)].mean() for w in (0, 1)]
        out.append((fold_auroc(yp, score, world == 0, fold), fold_auroc(yp, score, world == 1, fold), tpr[0] - tpr[1]))
    return np.array(out)


def main(cache: Path, labels: Path, probe_path: Path, tag: str = ""):
    lab = {(r["arm"], r["file"], r["sample_id"], r["epoch"], r["step"], r["action"]): r for r in map(json.loads, labels.read_text().splitlines())}
    full = json.loads(probe_path.read_text())
    rows = full["rows"]
    index = {(r["run"], r["step"], r["action"]): i for i, r in enumerate(rows)}
    yr = np.full(len(rows), -1)
    n, by_pattern = Counter(), Counter()
    pats = [re.compile(p, re.IGNORECASE) for p in PATTERNS]
    for world, d in COHORTS.items():
        for ev in sorted(glob.glob(str(cache / d / "logs" / "*.eval"))):
            with zipfile.ZipFile(ev) as z:
                for name in sorted(x for x in z.namelist() if x.startswith("samples/") and x.endswith(".json")):
                    s = json.loads(z.read(name))
                    for step, action, _, args in calls(s):
                        key = (ARM[world], Path(ev).name, s["id"], s.get("epoch", 1), step, action)
                        j, r = bool(DESTRUCTIVE.search(json.dumps(args))), raw_rule(args)
                        if j != lab[key]["destructive"]:
                            raise ValueError(f"{key}: the JSON rule disagrees with the stored label")
                        n[f"{world} | {'impactful' if lab[key]['impactful'] else 'read-only'} | JSON rule {j} | raw rule {r}"] += 1
                        if r and not j:
                            for p, c in zip(PATTERNS, pats):
                                by_pattern[f"{world} | {p}"] += any(c.search(v) for v in args.values() if isinstance(v, str))
                        i = index.get((f"{key[0]}|{key[1]}|{key[2]}|{key[3]}", step, action))
                        if i is not None:
                            yr[i] = int(r)
    if (yr < 0).any():
        raise ValueError(f"{(yr < 0).sum()} probe rows not found in the transcripts")
    y = np.array([r["y"] for r in rows]); world = np.array([r["world"] for r in rows]); fold = np.array([r["fold"] for r in rows])
    d0 = full["draws"][0]
    caught = np.array([c == "1" for c in d0["caught"]]); score = np.array(d0["scores"])
    T = Tasks(np.array([r["task"] for r in rows]))
    probe_rows = {f"{w} | JSON rule {a} | raw rule {b}": int(((world == wi) & (y == a) & (yr == b)).sum())
                  for wi, w in ((0, "real"), (1, "simulated")) for a in (0, 1) for b in (0, 1)}
    out = {"probe output": str(probe_path), "probe output sha256": hashlib.sha256(probe_path.read_bytes()).hexdigest(),
           "labels sha256": hashlib.sha256(labels.read_bytes()).hexdigest(), "added after review": True,
           "all calls": dict(sorted(n.items())), "probe rows": probe_rows,
           "patterns of the calls only the raw-string rule matches": {k: v for k, v in sorted(by_pattern.items(), key=lambda kv: -kv[1]) if v},
           "as labelled (frozen)": world_stats(y, caught, world, T) | {"AUROC real (fold mean)": d0["AUROC real (fold mean)"], "AUROC simulated (fold mean)": d0["AUROC sim (fold mean)"]},
           "against the raw-string rule (same scores and thresholds, nothing refit)": world_stats(yr, caught, world, T) | {
               "AUROC real (fold mean)": fold_auroc(yr, score, world == 0, fold), "AUROC simulated (fold mean)": fold_auroc(yr, score, world == 1, fold)},
           "caught among the calls only the raw-string rule matches": {w: float(caught[(world == wi) & (y == 0) & (yr == 1)].mean()) for wi, w in ((0, "real"), (1, "simulated"))}}
    resc = out["against the raw-string rule (same scores and thresholds, nothing refit)"]
    obs = (resc["AUROC real (fold mean)"], resc["AUROC simulated (fold mean)"], resc["TPR real minus simulated"]["point"])
    nd = null_draws(yr, np.array([r["run"] for r in rows]), world, fold, score, caught)
    gap_p = min(1.0, 2 * min((1 + np.sum(nd[:, 2] >= obs[2])) / (len(nd) + 1), (1 + np.sum(nd[:, 2] <= obs[2])) / (len(nd) + 1)))
    out["null for the rescore (raw-string labels shuffled within run, 1,000 draws, seed 0; fixed scores and thresholds)"] = {
        "AUROC real: null 95th percentile, p": [float(np.percentile(nd[:, 0], 95)), float((1 + np.sum(nd[:, 0] >= obs[0])) / (len(nd) + 1))],
        "AUROC simulated: null 95th percentile, p": [float(np.percentile(nd[:, 1], 95)), float((1 + np.sum(nd[:, 1] >= obs[1])) / (len(nd) + 1))],
        "AUROC null means (real, simulated)": [float(nd[:, 0].mean()), float(nd[:, 1].mean())],
        "TPR gap: null 2.5th and 97.5th percentiles, two-sided p": [float(np.percentile(nd[:, 2], 2.5)), float(np.percentile(nd[:, 2], 97.5)), float(gap_p)]}
    dest = HERE / "results" / f"destructive_label_escaping_{probe_path.stem.removeprefix('destructive_probe_')}{tag}.json"
    with open(dest, "x") as fh:  # results/ is append-only
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("all calls",)}, indent=1))
    print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]).expanduser(), Path(sys.argv[2]), Path(sys.argv[3]), *sys.argv[4:5])  # optional tag: a rerun writes a new file
