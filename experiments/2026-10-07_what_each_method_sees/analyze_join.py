"""One table of every kept tool output with both methods' out-of-fold scores, at levels 0 and 7, on both splits.

Rows: the TF-IDF staging's kept outputs (real attack e1, e2, simulated attack; 939 paired cells). Columns per level
(L0, L7), method (tfidf, probe), split (env = held-out environment, task = main-task folds) and comparison (real_vs_sim
scores every arm; null_e1_vs_e2 scores e1 and e2): the per-output log-odds, NaN where that method did not score the
output. The probe's score is the frozen headline's: each outer fold's rows from the layer that fold chose by inner CV.

Checks (fail loudly): staging hash; the probe's step rows identical across layers; the headline's run scores rebuilt
from the step scores equal the stored ones; TF-IDF and probe run AUROCs equal the published ones.

  uv run python experiments/2026-10-07_what_each_method_sees/analyze_join.py
Writes .run_cache/joined.npz and results/join_<stamp>.json.
"""

import argparse
import gzip
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

from probe_shift.provenance import capture

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
SUFFIX = {"env": "__held_out_environment", "task": ""}
HEAD = {"env": "headline_held_out_environment", "task": "headline"}
E1, E2, SIM = "real-attack-e1", "real-attack-e2", "sim-attack"
POS = {"real_vs_sim": (E1, SIM), "null_e1_vs_e2": (E1, E2)}  # (label 0, label 1)


def sha256_file(path: Path, gunzip: bool = False) -> str:
    h = hashlib.sha256()
    with (gzip.open(path, "rb") if gunzip else open(path, "rb")) as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_name(key: str) -> str:
    """The extraction's file name for a run key (run_extract_all_layers.py): long keys are cut and hashed."""
    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


def run_auroc(score, key, arm, neg, pos):
    """AUROC of the run mean of `score` (NaN rows dropped), pos-arm runs over neg-arm runs."""
    m = np.isin(arm, [neg, pos]) & ~np.isnan(score)
    keys, inv = np.unique(key[m], return_inverse=True)
    mean = np.bincount(inv, score[m]) / np.bincount(inv)
    run_arm = arm[m][np.unique(inv, return_index=True)[1]]
    return roc_auc_score(run_arm == pos, mean), len(keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config/what_sees.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    prov = capture(ROOT, [Path(__file__), Path(args.config)])
    staging = Path(cfg["staging_dir"]).expanduser()
    got = sha256_file(staging / "outputs.jsonl.gz", gunzip=True)
    assert got == cfg["staging_outputs_sha256"], f"staging outputs.jsonl sha256 {got} != config"

    rows = [r for r in map(json.loads, gzip.open(staging / "outputs.jsonl.gz", "rt")) if r["kept"]]
    command = {}
    for c in map(json.loads, gzip.open(staging / "calls.jsonl.gz", "rt")):
        if c["tool"] == "text_editor" and (m := re.search(r'"command": "(\w+)"', c["text"])):
            command[(c["key"], c["step"], c["action"])] = m.group(1)
    rows.sort(key=lambda r: (r["key"], r["step"], r["action"]))
    T = {k: np.array([r[k] for r in rows]) for k in ("key", "arm", "tool", "environment", "main_task", "cell", "step", "action", "n_tokens")}
    T["editor_command"] = np.array([command.get((r["key"], r["step"], r["action"]), "") for r in rows])
    T["readout_chars"] = np.array([len(r["readout_text"]) for r in rows])
    index = {(r["key"], r["step"], r["action"]): i for i, r in enumerate(rows)}
    n = len(rows)
    report = {"stamp": stamp, "provenance": prov, "config": cfg, "rows": n, "rows_by_arm": dict(Counter(T["arm"])), "checks": {}}
    folds = json.loads((ROOT / "splits" / cfg["folds_file"]).read_text())

    # TF-IDF: one score per (level, split, comparison), all outputs.
    for level, files in cfg["tfidf"].items():
        for split, name in files.items():
            path = Path(cfg["tfidf_dir"]).expanduser() / name
            report.setdefault("inputs", {})[f"tfidf_{level}_{split}"] = {"path": str(path), "sha256": sha256_file(path)}
            cols = {c: np.full(n, np.nan) for c in cfg["comparisons"]}
            missing = Counter()
            for r in map(json.loads, gzip.open(path, "rt")):
                if r["outputs"] != "all":
                    continue
                i = index.get((r["key"], r["step"], r["action"]))
                if i is None:
                    missing[r["comparison"]] += 1
                    continue
                cols[r["comparison"]][i] = r["score"]
            assert not missing, f"TF-IDF {level} {split}: rows not in the staging {dict(missing)}"
            for c, s in cols.items():
                T[f"tfidf_{level}_{split}_{c}"] = s
                auc, nr = run_auroc(s, T["key"], T["arm"], *POS[c])
                report["checks"][f"tfidf_{level}_{split}_{c}_run_auroc"] = {"auroc": round(auc, 4), "runs": nr}

    # Probe: compose the headline from the chosen layer per outer fold, then align to the staging rows.
    for level, p in cfg["probe"].items():
        summary = json.loads((HERE / ".run_cache/q1_results" / f"q1_gpu_{p['analysis']}.json").read_text())
        key_of = {file_name(k): k for k in set(T["key"])}
        assert len(key_of) == len(set(T["key"])), "two run keys share an extraction file name"
        missing_runs = [name for _, name in summary["manifest"]["runs"] if name not in key_of]
        assert not missing_runs, f"probe {level}: {len(missing_runs)} runs not in the staging, e.g. {missing_runs[:2]}"
        names = [key_of[name] for _, name in summary["manifest"]["runs"]]
        run_arm = np.array([a for a, _ in summary["manifest"]["runs"]])
        env_of = dict(zip(T["key"], T["environment"]))
        task_of = dict(zip(T["key"], T["main_task"]))
        run_fold = {"env": np.array([summary["config"]["environment_to_fold"][env_of[k]] for k in names]),
                    "task": np.array([folds["task_folds"]["main_task_to_fold"][task_of[k]] for k in names])}
        base = HERE / ".run_cache/probe_step_scores" / p["analysis"]
        loaded, ref = {}, None
        for split in cfg["splits"]:
            for c in cfg["comparisons"]:
                chosen = summary[HEAD[split]]["output_end"][c]["all_outputs"]["chosen_layer_by_fold"]
                s = None
                for f, l in chosen.items():
                    if l not in loaded:
                        loaded[l] = np.load(base / f"L{l}_step_scores_output_end.npz")
                        if ref is None:
                            ref = {k: loaded[l][k] for k in ("run", "step", "action")}
                        assert all(np.array_equal(loaded[l][k], ref[k]) for k in ref), f"L{l}: step rows differ from other layers"
                    if s is None:
                        s = np.full(len(ref["run"]), np.nan)
                    rows_f = run_fold[split][ref["run"]] == int(f)
                    s[rows_f] = loaded[l][f"{c}__all_outputs{SUFFIX[split]}"][rows_f]
                # The stored headline run scores must be the run means of these step scores.
                ok = ~np.isnan(s)
                tot = np.bincount(ref["run"][ok], s[ok], minlength=len(names))
                cnt = np.bincount(ref["run"][ok], minlength=len(names))
                mine = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
                stored = np.array([np.nan if x is None else x for x in summary[HEAD[split]]["output_end"][c]["all_outputs"]["run_scores"]])
                both = ~np.isnan(stored)
                assert np.array_equal(both, ~np.isnan(mine) & both) and np.nanmax(np.abs(mine[both] - stored[both])) < 1e-4, \
                    f"probe {level} {split} {c}: rebuilt run scores differ from the stored headline"
                neg, pos = POS[c]
                keep = both & np.isin(run_arm, [neg, pos])
                auc_all = roc_auc_score(run_arm[keep] == pos, mine[keep])
                published = summary[HEAD[split]]["output_end"][c]["all_outputs"]["run"]["auroc"]
                assert abs(auc_all - published) < 1e-3, f"probe {level} {split} {c}: {auc_all:.4f} vs published {published}"
                # Align to staging rows.
                col = np.full(n, np.nan)
                hit = Counter()
                for r_, st, a, v in zip(ref["run"], ref["step"], ref["action"], s):
                    i = index.get((names[r_], int(st), int(a)))
                    hit["matched" if i is not None else "probe_only"] += not np.isnan(v)
                    if i is not None:
                        col[i] = v
                T[f"probe_{level}_{split}_{c}"] = col
                auc, nr = run_auroc(col, T["key"], T["arm"], neg, pos)
                report["checks"][f"probe_{level}_{split}_{c}"] = {
                    "chosen_layer_by_fold": chosen, "published_run_auroc": published, "rebuilt_run_auroc": round(auc_all, 4),
                    "run_auroc_on_staging_rows": round(auc, 4), "scored_rows": dict(hit)}
        report.setdefault("inputs", {})[f"probe_{level}"] = {
            "analysis": p["analysis"], "extraction_run": summary["config"]["extraction_run"],
            "files": {f"L{l}": sha256_file(base / f"L{l}_step_scores_output_end.npz") for l in sorted(loaded)}}

    for k, v in report["checks"].items():
        print(k, json.dumps(v))
    out = HERE / ".run_cache/joined.npz"
    np.savez_compressed(out, **T)
    report["joined"] = {"path": str(out.relative_to(HERE)), "sha256": sha256_file(out), "columns": sorted(T)}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"join_{stamp}.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"wrote {out} and results/join_{stamp}.json")


if __name__ == "__main__":
    main()
