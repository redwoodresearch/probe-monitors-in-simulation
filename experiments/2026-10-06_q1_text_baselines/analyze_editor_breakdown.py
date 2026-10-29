"""What still separates real from simulated inside editor outputs after scrub level 6?

Per text_editor command (view, create, str_replace, insert, undo_edit; read from each output's tool call): per-output
AUROC of the level-6 out-of-fold scores, with counts per arm. Then one model fit on editor outputs only (level-6 text, for
reading only) for the top features each way, and 3 real + 3 simulated examples from the best-separating command. Writes
results/editor_breakdown_<stamp>.md.

  uv run python experiments/2026-10-06_q1_text_baselines/analyze_editor_breakdown.py [scores file in results/] [level] [config]
"""

import gzip
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

from analyze_tfidf import E1, SIM, featurize, fit, name_columns, scrub

HERE = Path(__file__).parent
EXAMPLES = "examples_2026-10-06_142532"
SCORES = "harness_normalized_empty_diff_lines_scores_2026-10-06_150020.jsonl.gz"  # level 6, main-task folds
LEVEL, TOP = 6, 15
SHOW = {"\n": "⏎", "\t": "⇥"}


def visible(s: str) -> str:
    return "".join(SHOW.get(c, c) for c in s).replace("|", "\\|").replace("`", "'")


def main():
    global SCORES, LEVEL
    SCORES = sys.argv[1] if len(sys.argv) > 1 else SCORES
    LEVEL = int(sys.argv[2]) if len(sys.argv) > 2 else LEVEL
    cfg = yaml.safe_load((HERE / "config" / (sys.argv[3] if len(sys.argv) > 3 else "tfidf_harness_empty_diff_lines.yaml")).read_text())
    ex = HERE / ".run_cache" / EXAMPLES
    command = {}
    for line in gzip.open(ex / "calls.jsonl.gz", "rt"):
        c = json.loads(line)
        if c["tool"] == "text_editor":
            m = re.search(r'"command": "(\w+)"', c["text"])
            command[(c["key"], c["step"], c["action"])] = m.group(1) if m else "unknown"
    outputs = {(r["key"], r["step"], r["action"]): r for r in map(json.loads, gzip.open(ex / "outputs.jsonl.gz", "rt"))}
    scored = [json.loads(l) for l in gzip.open(HERE / "results" / SCORES, "rt")]
    scored = [s for s in scored if s["comparison"] == "real_vs_sim" and s["outputs"] == "all" and s["arm"] in (E1, SIM)
              and s["tool"] == "text_editor"]
    by_cmd = {}
    for s in scored:
        by_cmd.setdefault(command.get((s["key"], s["step"], s["action"]), "unknown"), []).append(s)
    lines = [f"# What separates real from simulated inside editor outputs, after scrub level {LEVEL}", "",
             f"Level-{LEVEL} out-of-fold scores (`{SCORES}`, main-task folds), editor outputs only, real epoch 1 vs simulated.", "",
             "| command | real outputs | simulated outputs | per-output AUROC |", "|---|---|---|---|"]
    aucs = {}
    for cmd, rows in sorted(by_cmd.items(), key=lambda kv: -len(kv[1])):
        y = np.array([r["arm"] == SIM for r in rows])
        n = Counter(r["arm"] for r in rows)
        aucs[cmd] = roc_auc_score(y, [r["score"] for r in rows]) if 0 < y.sum() < len(y) else float("nan")
        lines.append(f"| {cmd} | {n[E1]} | {n[SIM]} | {aucs[cmd]:.3f} |")
    y_all = np.array([r["arm"] == SIM for r in scored])
    lines += [f"| all editor outputs | {(~y_all).sum()} | {y_all.sum()} | {roc_auc_score(y_all, [r['score'] for r in scored]):.3f} |", ""]

    # Top features of a model fit on editor outputs only (reading only)
    rows = [outputs[(s["key"], s["step"], s["action"])] for s in scored]
    texts = [scrub(r["readout_text"], LEVEL) for r in rows]
    X = featurize(texts, cfg)
    _, clf = fit(X, y_all.astype(int), cfg, cfg["C"])
    coef = clf.coef_[0]
    order = np.argsort(coef)
    cols = np.concatenate([order[::-1][:TOP], order[:TOP]])
    names = name_columns(X, texts, cols, cfg)
    lines += [f"## Top {TOP} features, model fit on editor outputs only (level-{LEVEL} text)", "",
              "| # | toward simulated | weight | toward real | weight |", "|---|---|---|---|---|"]
    for k, (a, b) in enumerate(zip(order[::-1][:TOP], order[:TOP]), 1):
        lines.append(f"| {k} | {' / '.join('`' + visible(g) + '`' for g in names[int(a)])} | {coef[a]:+.2f} | "
                     f"{' / '.join('`' + visible(g) + '`' for g in names[int(b)])} | {coef[b]:+.2f} |")
    lines.append("")

    # Examples from the best-separating command (with at least 100 outputs per arm)
    eligible = {c: a for c, a in aucs.items() if min(Counter(r["arm"] for r in by_cmd[c]).values()) >= 100}
    best = max(eligible, key=lambda c: abs(eligible[c] - 0.5))
    random.seed(0)
    lines += [f"## 3 real and 3 simulated `{best}` outputs (AUROC {aucs[best]:.3f})", ""]
    for arm in (E1, SIM):
        pick = random.sample([r for r in by_cmd[best] if r["arm"] == arm], 3)
        for k, s in enumerate(pick, 1):
            r = outputs[(s["key"], s["step"], s["action"])]
            lines += [f"### {arm} {k}: `{s['key'][:80]}` step {s['step']} (score {s['score']:+.2f})", "",
                      "Cleaned text (first 700 characters):", "", "```", r["readout_text"][:700], "```", "",
                      f"After the level-{LEVEL} scrub (first 500 characters):", "", "```", scrub(r["readout_text"], LEVEL)[:500], "```", ""]
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = HERE / "results" / f"editor_breakdown_{stamp}.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:16]))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
