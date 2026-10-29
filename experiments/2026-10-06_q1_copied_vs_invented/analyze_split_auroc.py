"""Split the per-output real-vs-simulated TF-IDF separation by reference match (Q1, copied vs invented).

Joins run_flag_outputs.py's flags with baselines-q3's out-of-fold per-output TF-IDF scores (cleaned text, model fit
on all outputs) and reports, per slice (all outputs, text_editor only, other tools, base200, overlap800) and per
match group (out_match; and cmd_in_ref x out_match):
  - each arm's share of outputs in the group;
  - per-output AUROC, real epoch 1 (0) vs simulated (1), and the null, epoch 1 (0) vs epoch 2 (1, the null model);
  - run-level AUROC from that group's outputs only (a run's score = mean of its outputs in the group).
95% intervals from bootstrapping main tasks. Sanity check: two outputs with byte-identical text must get identical
TF-IDF scores, so within a cell, an epoch-1 and a simulated output that both copy the same reference output tie.
Writes results/split_auroc_<stamp>.{json,md}, plus the flags and joined scores it used (identifiers, flags and
floats, no text) to results/flags_used_<stamp>.jsonl.gz and results/scores_used_<stamp>.jsonl.gz, so the table's
inputs survive if their sources move (the identical-text sanity check and the excerpts still read the staged text
table, `staged_outputs`, which holds transcript text and stays out of git); example excerpts go to .run_cache/ only.

  uv run python experiments/2026-10-06_q1_copied_vs_invented/analyze_split_auroc.py [--flags .run_cache/flags_X.jsonl.gz]
  uv run python experiments/2026-10-06_q1_copied_vs_invented/analyze_split_auroc.py --render results/split_auroc_X.json
"""

import argparse
import gzip
import json
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import yaml
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
EDITOR = "text_editor"
E1, E2, SIM = "real-attack-e1", "real-attack-e2", "sim-attack"
MIN_PER_CLASS = 20  # below this many outputs (or runs) per class, report counts but no AUROC


def auroc_ci(y, score, groups, n_boot, seed):
    """AUROC of score for label 1 over label 0, 95% interval from resampling groups (main tasks) with replacement."""
    y, score, groups = np.asarray(y), np.asarray(score, float), np.asarray(groups)
    assert not np.isnan(score).any()
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = np.concatenate([members[k] for k in rng.integers(0, len(uniq), len(uniq))])
        if len(np.unique(y[i])) == 2:
            vals.append(roc_auc_score(y[i], score[i]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"auroc": round(float(roc_auc_score(y, score)), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "n": int(len(y)), "n_main_tasks": int(len(uniq))}


def render_md(json_path: Path) -> str:
    """Write the readable tables for a split_auroc_<stamp>.json next to it (same stamp, .md) and return them.
    Refuses if that .md exists: results/ is append-only."""
    out = json.loads(json_path.read_text())

    def fmt(r):
        return f"{r['auroc']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]" if r else "–"

    table = out["table"]
    lines = [f"# Split AUROC ({out['stamp']}, flags {Path(out['flags_file']).name})", "",
             f"Sanity, identical text ties: {json.dumps(out['sanity_identical_text'])}", ""]
    for gname in dict.fromkeys(r["grouping"] for r in table):
        for sname in dict.fromkeys(r["slice"] for r in table):
            lines += [f"## {sname} · {gname.replace('|', ' × ')}", "",
                      "| group | e1 share | e2 share | sim share | per output: real vs sim | per output: e1 vs e2 | per run: real vs sim (runs) |",
                      "|---|---|---|---|---|---|---|"]
            for r in table:
                if r["slice"] == sname and r["grouping"] == gname:
                    sh = r["shares"]
                    lines.append(f"| {r['group'].replace('|', ' × ')} | {sh[E1]['share']:.3f} ({sh[E1]['n']}) | "
                                 f"{sh[E2]['share']:.3f} | {sh[SIM]['share']:.3f} ({sh[SIM]['n']}) | "
                                 f"{fmt(r.get('output_real_vs_sim'))} | {fmt(r.get('output_null_e1_vs_e2'))} | "
                                 f"{fmt(r.get('run_real_vs_sim'))} "
                                 f"({r['runs_with_group_outputs'][E1]}/{r['runs_with_group_outputs'][SIM]}) |")
            lines.append("")
    text = "\n".join(lines)
    md = json_path.with_suffix(".md")
    assert not md.exists(), f"{md} exists; results/ is append-only"
    md.write_text(text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flags", help="flags file (default: newest .run_cache/flags_*.jsonl.gz)")
    ap.add_argument("--render", help="only write the missing .md tables of a results/split_auroc_<stamp>.json")
    args = ap.parse_args()
    if args.render:
        print(render_md(Path(args.render)))
        return
    cfg = yaml.safe_load((HERE / "config" / "copied_vs_invented.yaml").read_text())
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    flags_path = Path(args.flags) if args.flags else sorted((HERE / ".run_cache").glob("flags_*.jsonl.gz"))[-1]
    with gzip.open(flags_path, "rt") as f:
        flags = [json.loads(line) for line in f]

    scores = {}
    with gzip.open(Path(cfg["tfidf_scores"]).expanduser(), "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["baseline"] == cfg["tfidf_baseline"] and r["outputs"] == cfg["tfidf_outputs"]:
                scores[(r["comparison"], r["arm"], r["key"], r["step"], r["action"])] = r["score"]
    for x in flags:
        uid = (x["arm"], x["key"], x["step"], x["action"])
        x["s_rvs"] = scores.get(("real_vs_sim", *uid))
        x["s_null"] = scores.get(("null_e1_vs_e2", *uid)) if x["arm"] != SIM else None
    missing = sum(x["s_rvs"] is None for x in flags) + sum(x["s_null"] is None for x in flags if x["arm"] != SIM)
    assert missing == 0, f"{missing} flagged outputs have no TF-IDF score: units differ, fix before scoring"

    # Sanity: byte-identical text gets an identical score. Within a cell, e1 and sim outputs that both copy the same
    # reference output have the same text; check their real-vs-sim scores tie.
    text = {}
    with gzip.open(Path(cfg["staged_outputs"]).expanduser(), "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["kept"]:
                text[(r["arm"], r["key"], r["step"], r["action"])] = r["text"].rstrip()
    same_text = defaultdict(lambda: defaultdict(list))
    for x in flags:
        if x["out_match"] == "exact" and x["arm"] in (E1, SIM):
            same_text[(x["cell"], text[(x["arm"], x["key"], x["step"], x["action"])])][x["arm"]].append(x["s_rvs"])
    tied = [max(abs(a - b) for a in d[E1] for b in d[SIM]) for d in same_text.values() if d[E1] and d[SIM]]
    sanity = {"identical_text_pairs_groups": len(tied), "max_score_gap": round(max(tied), 6) if tied else None,
              "share_with_gap_over_1e-4": round(float(np.mean(np.array(tied) > 1e-4)), 4) if tied else None}

    slices = {"all": lambda x: True, "text_editor": lambda x: x["tool"] == EDITOR,
              "other_tools": lambda x: x["tool"] != EDITOR, "base200": lambda x: x["sim_reference"] == "base200",
              "overlap800": lambda x: x["sim_reference"] == "overlap800"}
    groupings = {"out_match": lambda x: x["out_match"],
                 "cmd_in_ref|out_match": lambda x: f"{'same_cmd' if x['cmd_in_ref'] else 'new_cmd'}|{x['out_match']}",
                 "everything": lambda x: "all"}

    jobs, table = [], []
    for sname, sf in slices.items():
        sub = [x for x in flags if sf(x)]
        for gname, gf in groupings.items():
            for gval in sorted({gf(x) for x in sub}):
                rows = [x for x in sub if gf(x) == gval]
                arm_n = {a: sum(x["arm"] == a for x in sub) for a in (E1, E2, SIM)}
                share = {a: {"n": sum(x["arm"] == a for x in rows),
                             "share": round(sum(x["arm"] == a for x in rows) / max(arm_n[a], 1), 4)} for a in (E1, E2, SIM)}
                rec = {"slice": sname, "grouping": gname, "group": gval, "shares": share}
                table.append(rec)
                rvs = [x for x in rows if x["arm"] in (E1, SIM)]
                null = [x for x in rows if x["arm"] in (E1, E2)]
                for name, rs, lab, key in (("output_real_vs_sim", rvs, SIM, "s_rvs"), ("output_null_e1_vs_e2", null, E2, "s_null")):
                    y = [int(x["arm"] == lab) for x in rs]
                    if min(sum(y), len(y) - sum(y)) >= MIN_PER_CLASS:
                        jobs.append((rec, name, y, [x[key] for x in rs], [x["main_task"] for x in rs]))
                runs = defaultdict(list)
                for x in rvs:
                    runs[(x["arm"], x["key"], x["main_task"])].append(x["s_rvs"])
                y = [int(a == SIM) for a, _, _ in runs]
                if min(sum(y), len(y) - sum(y)) >= MIN_PER_CLASS:
                    jobs.append((rec, "run_real_vs_sim", y, [np.mean(v) for v in runs.values()], [m for _, _, m in runs]))
                rec["runs_with_group_outputs"] = {a: sum(k[0] == a for k in runs) for a in (E1, SIM)}

    results = Parallel(n_jobs=cfg["n_jobs"])(delayed(auroc_ci)(y, s, g, cfg["bootstrap"], cfg["seed"]) for _, _, y, s, g in jobs)
    for (rec, name, *_), res in zip(jobs, results):
        rec[name] = res

    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    out = {"config": cfg, "git": git, "stamp": stamp, "flags_file": str(flags_path.resolve()),
           "flags_used_file": f"results/flags_used_{stamp}.jsonl.gz",
           "scores_used_file": f"results/scores_used_{stamp}.jsonl.gz",
           "n_outputs": len(flags), "sanity_identical_text": sanity, "table": table}
    # The flags and scores used (no text), the scores in the source file's schema, kept in results/ so `--flags` and
    # `tfidf_scores` can point here if the sources move or a cache is cleaned (`staged_outputs` is still needed).
    (HERE / ".run_cache").mkdir(exist_ok=True)  # for the excerpts below; --flags may point outside this folder
    shutil.copyfile(flags_path, HERE / "results" / f"flags_used_{stamp}.jsonl.gz")
    with gzip.open(HERE / "results" / f"scores_used_{stamp}.jsonl.gz", "wt") as f:
        for x in flags:
            for comparison, key in (("real_vs_sim", "s_rvs"), ("null_e1_vs_e2", "s_null")):
                if x[key] is not None:
                    f.write(json.dumps({"baseline": cfg["tfidf_baseline"], "comparison": comparison,
                                        "outputs": cfg["tfidf_outputs"], "arm": x["arm"], "key": x["key"],
                                        "step": x["step"], "action": x["action"], "score": x[key]}) + "\n")
    json_path = HERE / "results" / f"split_auroc_{stamp}.json"
    json_path.write_text(json.dumps(out, indent=1))
    print(render_md(json_path))

    # Example excerpts per group (transcript text: .run_cache only, never results/).
    rng = np.random.default_rng(cfg["seed"])
    ex = ["# Example outputs per group (e1 vs sim, same cell when possible)", ""]
    by = defaultdict(lambda: defaultdict(list))
    for x in flags:
        if x["arm"] in (E1, SIM):
            by[f"{'same_cmd' if x['cmd_in_ref'] else 'new_cmd'}|{x['out_match']}"][x["cell"]].append(x)
    for g in sorted(by):
        both = [c for c, xs in by[g].items() if {x["arm"] for x in xs} == {E1, SIM}]
        ex.append(f"## {g}")
        for c in rng.choice(both, size=min(3, len(both)), replace=False) if both else []:
            for arm in (E1, SIM):
                x = next(x for x in by[g][c] if x["arm"] == arm)
                t = text[(x["arm"], x["key"], x["step"], x["action"])]
                ex.append(f"- {arm} · {c} · step {x['step']} · {x['tool']} · sim {x['best_similarity']} · score {x['s_rvs']:.3f}\n"
                          f"  ```\n  {t[:300]!r}\n  ```")
        ex.append("")
    (HERE / ".run_cache" / f"examples_{stamp}.md").write_text("\n".join(ex))


if __name__ == "__main__":
    main()
