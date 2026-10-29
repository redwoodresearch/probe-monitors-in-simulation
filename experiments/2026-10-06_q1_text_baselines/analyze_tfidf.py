"""Q1 text baselines: can TF-IDF + logistic regression tell real from simulated LinuxArena tool outputs?

The same classifier on different texts per tool output (rows from run_stage_examples.py), named in the config:
  - cleaned text (`readout_text`): the output as Qwen3-32B reads it, up to the probe's readout (output_end);
  - raw text (`raw`): the tool message as stored in the log (output, error record, tool name);
or tool calls (`segments: calls`: the call JSON as rendered, through its last content token, call_end);
each at a scrub level, applied cumulatively to every arm:
  0 nothing; 1 clock times and dates masked; 2 also every digit run, hex strings of 6+ characters (with a digit and a
  letter), UUIDs and IPv4 addresses masked; 3 also whitespace normalized (trailing whitespace stripped on every line and
  at the end, runs of blank lines collapsed); 4 also harness format (real editor diffs' line-number gutter stripped,
  stderr-looking lines dropped, weekday and month names masked, blank lines removed, spaces collapsed); 5 also diff lines
  canonicalized (the space real diffs put after the +/- sign removed, hunk headers reduced to "@@"); 6 also empty
  added or removed diff lines dropped (the simulator writes blank lines as added where the real diff shows context); 7
  also editor output format canonicalized (diffs and `cat -n` snippets both reduced to plain code). `no_insert_undo` drops
  text_editor insert and undo_edit outputs, whose output format differs by harness.
  Output variants: `all`, `no_editor` (text_editor segments dropped) and
  `no_errors` (tool-error outputs dropped: a harness tell), each refit.
Features: word 1-2-grams and character 2-5-grams, case and whitespace kept, hashed into 2^20 columns each, TF-IDF
weights fit on the training folds only, then L2 logistic regression with a fixed C. Examples: one per kept tool output
(its last content token inside the first 32,768 tokens, the probe's rule). Out-of-fold scores on the shared main-task
folds. A run's score is the mean of its outputs' scores.

Comparisons: real attack epoch 1 (0) vs simulated attack (1), whose models also score the epoch-2 runs; and the null,
epoch 1 (0) vs epoch 2 (1). Each with all outputs and refit with text_editor outputs removed. Metrics, with 95%
intervals from bootstrapping main tasks: output-level AUROC, run-level AUROC, and run-level pairwise accuracy (share of
cells whose label-1 run scores above its epoch-1 run). Slices: all cells, base200, overlap800, no run over 32k.
Optional: a label-shuffle null for real vs simulated (each main task's runs swap labels with probability 1/2, one flip
per main task as for the probe; refit on the same folds) and a C sensitivity check.

  uv run python experiments/2026-10-06_q1_text_baselines/analyze_tfidf.py [config/tfidf.yaml]
  (run_tfidf_modal.py runs the same baselines in parallel Modal CPU containers)
"""

import gzip
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from functools import partial
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import scipy.sparse as sp
import yaml
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.utils import murmurhash3_32

from probe_shift.scoring import auroc_ci, pairwise_accuracy_ci

HERE = Path(__file__).parent
SPLITS = HERE.parent.parent / "splits"
E1, E2, SIM = "real-attack-e1", "real-attack-e2", "sim-attack"
COMPARISONS = {"real_vs_sim": (E1, SIM, [E1, E2, SIM]), "null_e1_vs_e2": (E1, E2, [E1, E2])}


def arms_of(side) -> list:
    """A comparison side's arms: one arm name, or a tuple of arms pooled into one label."""
    return [side] if isinstance(side, str) else list(side)


def on_side(arm, side):
    return np.isin(arm, arms_of(side))


def side_name(side) -> str:
    return side if isinstance(side, str) else "+".join(side)


def comparisons(cfg: dict) -> dict:
    """{name: (label-0 arm, label-1 arm, arms scored)}. Config `arms` ({real, sim, real2 optional, also_scored
    optional}) replaces the attack arms, e.g. for the honest arms; without a second real arm there is no real-vs-real
    null. `also_scored` arms (e.g. another simulator's runs) are scored by the real-vs-sim model, never trained on."""
    if "arms" not in cfg:
        return COMPARISONS
    group = lambda x: tuple(x) if isinstance(x, list) else x  # a side is one arm, or a group of arms pooled
    real, sim, real2 = group(cfg["arms"]["real"]), group(cfg["arms"]["sim"]), group(cfg["arms"].get("real2"))
    out = {"real_vs_sim": (real, sim, [a for x in (real, real2, sim) if x for a in arms_of(x)] + cfg["arms"].get("also_scored", []))}
    if real2:
        out["null_e1_vs_e2"] = (real, real2, arms_of(real) + arms_of(real2))
    return out

# Level 1. Wall-clock times identify when a cohort ran (real epochs 1 and 2 ran about 2 h apart): ISO dates, clock
# times (not inside IPs or ports), month-day dates, Unix epoch seconds or ms.
MONTHS = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
TIME_PATTERNS = [(re.compile(r"\d{4}-\d{2}-\d{2}"), "<date>"),
                 (re.compile(r"(?<![\d.:])\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?(?![\d:])"), "<time>"),
                 (re.compile(rf"\b(?:{MONTHS})\s+\d{{1,2}}\b"), "<date>"),
                 (re.compile(r"\b1[5-9]\d{8}(?:\d{3})?\b"), "<epoch>")]
# Level 2. Numbers and IDs: PIDs, sizes, inodes, ports, hashes, container IDs. UUIDs and hex IDs are bounded like
# `\b` except that `_` also separates (`(?<![^\W_])`, `(?![^\W_])`): with `\b`, an ID joined by an underscore
# (`run_46a7d3b90e5c`) escaped the hex rule and the digit rule chopped it into `<n>a<n>d<n>b…`, while the hyphen-joined
# form (`run-46a7d3b90e5c`) became `<hex>`, a tell made by the scrub itself (fixed 2026-10-07).
ID_PATTERNS = [(re.compile(r"(?<![^\W_])[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}(?![^\W_])"), "<uuid>"),
               (re.compile(r"(?<![\d.])\d{1,3}(?:\.\d{1,3}){3}(?![\d.])"), "<ip>"),
               (re.compile(r"(?<![^\W_])(?=[0-9a-fA-F]*[a-fA-F])(?=[0-9a-fA-F]*\d)[0-9a-fA-F]{6,}(?![^\W_])"), "<hex>"),
               (re.compile(r"\d+"), "<n>")]
# Level 3. Whitespace.
TRAILING_SPACE, BLANK_RUNS = re.compile(r"[^\S\n]+$", re.M), re.compile(r"\n{3,}")
# Level 4. Harness format: real editor diffs carry a line-number gutter (" 142|+ code"), simulated ones don't, so the
# gutter is stripped from real diffs; stderr-looking lines are dropped (their order in the output differs by harness);
# weekday and month names masked; blank lines removed and runs of spaces or tabs collapsed.
EDITOR_DIFF, GUTTER = "Here's a diff showing the changes made", re.compile(r"^ *\d+\|", re.M)
STDERR_LINE = re.compile(r"^(?:(?:fatal|error|warning|bash|sh|/bin/sh|sed|grep|ls|cat|find|rm|cp|mv|mkdir|chmod|head|tail|awk|"
                         r"wc|xargs|cd|python3?|node|npm WARN|E): .*|.*(?:No such file or directory|Permission denied|"
                         r"command not found|not a git repository).*)\n?", re.M | re.I)
# Level 5. Diff lines canonicalized: real diff lines put one space between the +/-/context sign and the code (" 46|+ code"),
# simulated ones don't ("+code"), so that space goes with the gutter; hunk headers are reduced to "@@" (the simulator
# sometimes writes the short "@@ -N +N @@" form).
GUTTER_AND_SPACE, HUNK = re.compile(r"^ *\d+\|([+\- ]?) ?", re.M), re.compile(r"^@@ [^\n]*@@[^\n]*$", re.M)
# Level 6. Empty diff lines dropped: the real tool's diff algorithm matches new blank lines against existing ones and shows
# them as context, while the simulator writes them as added lines ("+" alone; 0.88 per simulated diff vs 0.47 real), so
# lines that are only a sign are removed from editor diffs (blank context lines are already gone at level 4).
EMPTY_DIFF_LINE = re.compile(r"^[+-]$\n?", re.M)
# Level 7. Editor output format canonicalized: the real tool answers some edits (all inserts) with a `cat -n` snippet of
# the edited file plus a review note, the simulator with a diff, so both are reduced to plain code: preambles, the review
# note, diff headers and removed lines dropped, +/space diff markers and `cat -n` line numbers stripped.
EDITED = re.compile(r"^The file \S+ has been edited\. Here's (?:a diff showing the changes made|the result of running `cat -n` "
                    r"on a snippet of (?:the edited file|\S+)):\n?", re.M)
CAT_N_VIEW = re.compile(r"^Here's the result of running `cat -n` on \S+:\n?", re.M)
REVIEW_NOTE = re.compile(r"\n?Review the changes and make sure they are as expected \(correct indentation, no duplicate lines, "
                         r"etc\)\. Edit the file again if necessary\.?")
DIFF_HEADER, REMOVED_LINE = re.compile(r"^(?:--- old_version|\+\+\+ edited_version|@@.*)$\n?", re.M), re.compile(r"^-.*$\n?", re.M)
# A `cat -n` line number is followed by a tab, or ends the line when the readout cut falls right after it.
DIFF_MARKER, CAT_N_PREFIX = re.compile(r"^[+ ]", re.M), re.compile(r"^ *\d+(?:\t|$)", re.M)
WEEKDAYS = re.compile(r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b")
MONTH_NAMES = re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December|"
                         r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)\b")


def mask_times(text: str) -> str:
    for pattern, placeholder in TIME_PATTERNS:
        text = pattern.sub(placeholder, text)
    return text


def scrub(text: str, level: int) -> str:
    """Cumulative scrub: 1 times and dates, 2 numbers and IDs, 3 whitespace, 4 harness format, 5 diff lines, 6 empty diff
    lines, 7 editor output format (diff and `cat -n` snippet both reduced to plain code)."""
    if level >= 4:  # before the digit masks, which would turn the gutter into "<n>|"
        diff = EDITOR_DIFF in text
        if diff:
            text = GUTTER_AND_SPACE.sub(r"\1", text) if level >= 5 else GUTTER.sub("", text)
            if level >= 5:
                text = HUNK.sub("@@", text)
        if level >= 7 and (diff or "result of running `cat -n`" in text):
            text = REVIEW_NOTE.sub("", CAT_N_VIEW.sub("", EDITED.sub("", text)))
            if diff:
                text = DIFF_MARKER.sub("", REMOVED_LINE.sub("", DIFF_HEADER.sub("", text)))
            else:
                text = CAT_N_PREFIX.sub("", text)
        text = STDERR_LINE.sub("", text)
    if level >= 1:
        text = mask_times(text)
    if level >= 2:
        for pattern, placeholder in ID_PATTERNS:
            text = pattern.sub(placeholder, text)
    if level >= 3:
        text = BLANK_RUNS.sub("\n\n", TRAILING_SPACE.sub("", text)).rstrip()
    if level >= 4:
        text = MONTH_NAMES.sub("<month>", WEEKDAYS.sub("<wday>", text))
        text = re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n", "\n", text))
    if level >= 6 and EDITOR_DIFF in text:
        text = EMPTY_DIFF_LINE.sub("", text).rstrip("\n")
    return text


def char_ngrams(text: str, lo: int, hi: int, lowercase: bool = False) -> list[str]:
    """Every character n-gram with whitespace untouched (sklearn's "char" analyzer collapses runs of whitespace)."""
    text = text.lower() if lowercase else text
    return [text[i:i + n] for n in range(lo, hi + 1) for i in range(len(text) - n + 1)]


def vectorizers(cfg) -> list[tuple[str, HashingVectorizer]]:
    """[(kind, vectorizer)]: word n-grams, then character n-grams, each in its own 2^n block of columns; either is
    left out when its config range is null."""
    common = dict(n_features=cfg["n_features"], alternate_sign=False, norm=None)
    out = []
    if cfg["word_ngrams"]:
        out.append(("word", HashingVectorizer(analyzer="word", ngram_range=tuple(cfg["word_ngrams"]),
                                              token_pattern=cfg["token_pattern"], lowercase=cfg["lowercase"], **common)))
    if cfg["char_ngrams"]:
        lo, hi = cfg["char_ngrams"]
        out.append(("char", HashingVectorizer(analyzer=partial(char_ngrams, lo=lo, hi=hi, lowercase=cfg["lowercase"]), **common)))
    return out


def _featurize_chunk(texts, cfg):
    return sp.hstack([v.transform(texts) for _, v in vectorizers(cfg)]).tocsr()


def featurize(texts, cfg):
    chunks = [texts[i:i + 1000] for i in range(0, len(texts), 1000)]
    return sp.vstack(Parallel(n_jobs=cfg["n_jobs"])(delayed(_featurize_chunk)(c, cfg) for c in chunks)).tocsr()


def fit(X, y, cfg, C):
    tfidf = TfidfTransformer(sublinear_tf=cfg["sublinear_tf"]).fit(X)
    return tfidf, LogisticRegression(C=C, solver=cfg["solver"], max_iter=cfg["max_iter"]).fit(tfidf.transform(X), y)


def _fold_scores(X, y, train, test, cfg, C):
    tfidf, clf = fit(X[train], y[train], cfg, C)
    return clf.decision_function(tfidf.transform(X[test]))


def inner_cv_C(X, y, fold, train_rows, cfg) -> dict:
    """Per outer fold, the C from `c_grid` with the best per-output AUROC in an inner cross-validation over the other
    folds (each inner fold left out once; with environment folds, over the training environments)."""
    outer, inner = sorted(set(fold)), sorted(set(fold[train_rows]))  # every outer fold gets a C, even one with no training rows
    jobs = [(f, g, C) for f in outer for g in inner if g != f for C in cfg["c_grid"]]
    parts = Parallel(n_jobs=cfg["n_jobs"])(
        delayed(_fold_scores)(X, y, train_rows & ~np.isin(fold, [f, g]), train_rows & (fold == g), cfg, C) for f, g, C in jobs)
    scores = {}  # (outer fold, C) -> inner out-of-fold scores
    for (f, g, C), s in zip(jobs, parts):
        scores.setdefault((f, C), np.full(len(y), np.nan))[train_rows & (fold == g)] = s
    best = {}
    for f in outer:
        m = train_rows & (fold != f)
        auc = {C: roc_auc_score(y[m], scores[(f, C)][m]) for C in cfg["c_grid"]}
        best[int(f)] = {"C": max(auc, key=auc.get), "inner_auroc": {str(C): round(float(a), 4) for C, a in auc.items()}}
    return best


def oof(X, y, fold, train_rows, test_rows, cfg, C, chosen: dict | None = None):
    """Out-of-fold scores for every test row (trained on the other folds' train rows); NaN elsewhere. C = "inner_cv"
    picks C per outer fold (inner_cv_C), recorded in `chosen`."""
    folds = sorted(set(fold))
    jobs = [(train_rows & (fold != f), test_rows & (fold == f)) for f in folds]
    if C == "inner_cv":
        best = inner_cv_C(X, y, fold, train_rows, cfg)
        if chosen is not None:
            chosen.update(best)
        Cs = [best[int(f)]["C"] for f in folds]
    else:
        Cs = [C] * len(folds)
    parts = Parallel(n_jobs=min(len(folds), cfg["n_jobs"]))(
        delayed(_fold_scores)(X, y, tr, te, cfg, c) for (tr, te), c in zip(jobs, Cs))
    score = np.full(len(y), np.nan)
    for (_, te), s in zip(jobs, parts):
        score[te] = s
    return score


def oof_many(X, sets, fold, cfg, C):
    """Out-of-fold scores for several (y, train_rows, test_rows) sets in one pool: every (set, fold) fit runs at once,
    so all cores stay busy. Yields each set's scores, in order, as soon as its folds are done."""
    folds = sorted(set(fold))
    jobs = [(k, tr & (fold != f), te & (fold == f)) for k, (_, tr, te) in enumerate(sets) for f in folds]
    parts = Parallel(n_jobs=cfg["n_jobs"], return_as="generator")(
        delayed(_fold_scores)(X, sets[k][0], tr, te, cfg, C) for k, tr, te in jobs)
    for i, ((k, _, te), s) in enumerate(zip(jobs, parts)):
        if i % len(folds) == 0:
            score = np.full(len(fold), np.nan)
        score[te] = s
        if i % len(folds) == len(folds) - 1:
            yield score


def run_means(score, key, rows):
    """Mean score per run over `rows`, with each run's first row index."""
    keys, inv = np.unique(key[rows], return_inverse=True)
    first = np.flatnonzero(rows)[np.unique(inv, return_index=True)[1]]
    return np.bincount(inv, score[rows]) / np.bincount(inv), first


def name_columns(X, texts, columns, cfg, per_column=5):
    """Each hashed column's n-grams, read back from a few documents that contain it (hash collisions show as several)."""
    analyzers = [(k * cfg["n_features"], v.build_analyzer()) for k, (_, v) in enumerate(vectorizers(cfg))]
    Xc = X.tocsc()
    names = {}
    for c in columns:
        docs = Xc.indices[Xc.indptr[c]:Xc.indptr[c + 1]][:per_column]
        found = Counter()
        for d in docs:
            off, analyze = analyzers[c // cfg["n_features"]]
            found.update(g for g in analyze(texts[d]) if off + abs(murmurhash3_32(g, seed=0)) % cfg["n_features"] == c)
        names[int(c)] = [g for g, _ in found.most_common(3)]
    return names


def with_cells(cfg: dict) -> dict:
    """The config with `cells` filled from `cells_file` (relative to the experiment folder), so a Modal container
    gets the list itself."""
    if cfg.get("cells_file") and "cells" not in cfg:
        cfg = cfg | {"cells": json.loads((HERE / cfg["cells_file"]).read_text())["cells"]}
    if cfg.get("slices") and "slice_cells" not in cfg:  # extra slices, each from a cells file
        cfg = cfg | {"slice_cells": {name: json.loads((HERE / v["cells_file"]).read_text())["cells"] for name, v in cfg["slices"].items()}}
    return cfg


def add_slices(d: dict, cfg: dict) -> None:
    """Config `slices` ({name: {cells_file, exclude}}, lists filled in by with_cells): each the analysis's cells in
    that file, or with `exclude: true` all its cells but those (e.g. a sensitivity check without a few redrawn tasks)."""
    every = d["slices"]["all_cells"]
    for name, cells in cfg.get("slice_cells", {}).items():
        d["slices"][name] = every - set(cells) if cfg["slices"][name].get("exclude") else every & set(cells)
    d["counts"]["cells_per_slice"] = {k: len(v) for k, v in d["slices"].items()}


def fold_map(cfg: dict, folds: dict) -> tuple[str, dict]:
    """(row field, value -> fold): main-task folds by default; `fold_scheme: environment` holds out one environment."""
    if cfg.get("fold_scheme", "main_task") == "environment":
        return "environment", folds["environment_folds"]["environment_to_fold"]
    return "main_task", folds["task_folds"]["main_task_to_fold"]


def load(examples_dir: Path, fold_of: dict, segments: str = "outputs", fold_key: str = "main_task",
         cells: list | None = None, holdout_arm: str | None = None) -> dict:
    """Kept segments of a staging run (`outputs` or `calls`) as arrays, plus the cell slices and counts. `fold_of` maps
    each row's `fold_key` (main_task, or environment for held-out-environment folds) to its fold. `cells` restricts
    everything (training, scoring, slices) to those cells. `holdout_arm`: every main task that arm has a run on moves
    to a fold of its own, so its runs are scored by a model trained on none of those tasks."""
    rows = [json.loads(l) for l in gzip.open(examples_dir / f"{segments}.jsonl.gz", "rt")]
    runs = [json.loads(l) for l in (examples_dir / "runs.jsonl").read_text().splitlines()]
    if cells is not None:
        keep_cells = set(cells)
        assert keep_cells <= {r["cell"] for r in runs}, "cells_file names cells that are not in the staging"
        rows, runs = [r for r in rows if r["cell"] in keep_cells], [r for r in runs if r["cell"] in keep_cells]
    counts = {"outputs": dict(Counter(r["arm"] for r in rows)), "kept_outputs": dict(Counter(r["arm"] for r in rows if r["kept"])),
              "runs": dict(Counter(r["arm"] for r in runs))}
    rows = [r for r in rows if r["kept"]]
    d = {"rows": rows, "counts": counts}
    for k in ("arm", "tool", "key", "cell", "main_task"):
        d[k] = np.array([r[k] for r in rows])
    # A tool error is a harness tell: the simulator accepts any argument shape, so real runs have far more of them.
    d["error"] = np.array([bool(json.loads(r["raw"]).get("error")) if "raw" in r else False for r in rows])
    # Each output's text_editor command (view, create, str_replace, insert, undo_edit), from its tool call.
    command = {}
    if (examples_dir / "calls.jsonl.gz").exists():
        for line in gzip.open(examples_dir / "calls.jsonl.gz", "rt"):
            c = json.loads(line)
            if c["tool"] == "text_editor" and (m := re.search(r'"command": "(\w+)"', c["text"])):
                command[(c["key"], c["step"], c["action"])] = m.group(1)
    d["editor_command"] = np.array([command.get((r["key"], r["step"], r["action"]), "") for r in rows])
    d["fold"] = np.array([fold_of[r[fold_key]] for r in rows])
    if holdout_arm:
        held = {r["main_task"] for r in runs if r["arm"] == holdout_arm}
        assert held, f"no runs of {holdout_arm}"
        d["fold"][np.isin(d["main_task"], list(held))] = d["fold"].max() + 1
    every = {r["cell"] for r in runs}  # staging keeps only cells with a run in every arm
    sim_ref = {r["cell"]: r["sim_reference"] for r in runs if r["sim_reference"]}  # attack sims only
    over_32k = {r["cell"] for r in runs if r["truncated"]}
    d["slices"] = {"all_cells": every, "no_run_over_32k": every - over_32k}
    if sim_ref:
        d["slices"] |= {"base200": {c for c, t in sim_ref.items() if t == "base200"},
                        "overlap800": {c for c, t in sim_ref.items() if t == "overlap800"}}
    for a in sorted({r["arm"] for r in runs}):  # an arm on only some cells (e.g. a pilot) gets a slice of its own
        if (on := {r["cell"] for r in runs if r["arm"] == a}) != every:
            d["slices"][f"cells_with_{a}"] = on
    counts["cells_per_slice"] = {s: len(c) for s, c in d["slices"].items()}
    return d


def texts_of(d: dict, spec: dict) -> list[str]:
    return [scrub(r[spec["text"]], spec["scrub"]) for r in d["rows"]]


def output_variants(d: dict, cfg: dict) -> dict:
    """Which segments a model is fit and scored on: all, no text_editor segments, no tool-error outputs, no text_editor
    insert or undo_edit outputs (their output format differs by harness)."""
    wanted = cfg.get("output_variants", ["all", "no_editor"])
    if "no_insert_undo" in wanted:  # needs each editor output's command, from the staging's calls.jsonl.gz
        missing = int(((d["tool"] == "text_editor") & (d["editor_command"] == "")).sum())
        assert not missing, f"{missing} text_editor outputs have no command (staging without calls.jsonl.gz?)"
    every = {"all": np.ones(len(d["rows"]), bool), "no_editor": d["tool"] != "text_editor", "no_errors": ~d["error"],
             "no_insert_undo": ~((d["tool"] == "text_editor") & np.isin(d["editor_command"], ["insert", "undo_edit"]))}
    return {v: every[v] for v in wanted}


def evaluate_scores(d: dict, score, keep, neg: str, pos: str, cfg: dict, run_score_fn=None) -> dict:
    """Per-segment and run-level AUROC and run-level pairwise accuracy of `pos` over `neg`, per slice. A run's score
    is the mean of its segments' scores, unless `run_score_fn(m)` gives (run scores, each run's first row index)."""
    arm, key, cell, main_task = d["arm"], d["key"], d["cell"], d["main_task"]
    res = {}
    for name, cells in d["slices"].items():
        if not cells:  # e.g. a cells_file with no overlap800 cells
            res[name] = {"cells": 0}
            continue
        m = keep & (on_side(arm, neg) | on_side(arm, pos)) & np.isin(cell, list(cells)) & ~np.isnan(score)
        r = {"outputs": auroc_ci(on_side(arm[m], pos), score[m], main_task[m], cfg["bootstrap"], cfg["seed"])}
        run_score, first = run_score_fn(m) if run_score_fn else run_means(score, key, m)
        run_arm, run_cell, run_task = arm[first], cell[first], main_task[first]
        run_pos = on_side(run_arm, pos)
        r["runs"] = auroc_ci(run_pos, run_score, run_task, cfg["bootstrap"], cfg["seed"])
        by = {(bool(p), c): s for p, c, s in zip(run_pos, run_cell, run_score)}
        assert len(by) == len(run_score), "a cell has two runs on one side of the comparison"
        pairs = sorted(c for c in cells if (False, c) in by and (True, c) in by)
        correct = [float(by[(True, c)] > by[(False, c)]) + 0.5 * float(by[(True, c)] == by[(False, c)]) for c in pairs]
        r["pairs"] = pairwise_accuracy_ci(correct, [c.split(":")[0] for c in pairs], cfg["bootstrap"], cfg["seed"])
        r["cells_without_a_scored_run"] = len(cells) - len(pairs)
        res[name] = r
    return res


def run_baseline(cfg: dict, d: dict, baseline: str, spec: dict, out_dir: Path, stamp: str,
                 checkpoint_dir: Path | None = None, commit=lambda: None) -> dict:
    """One baseline: out-of-fold scores (written to out_dir), metrics, optional nulls, top features; returns its summary.

    With `checkpoint_dir`, every finished set of out-of-fold scores is saved there (then `commit()`, e.g. a Modal
    Volume commit) and reused on a rerun, so a preempted job resumes instead of refitting."""
    cfg = cfg | spec.get("settings", {})  # a baseline's own featurizer or C settings, if any
    assert not (cfg["C"] == "inner_cv" and cfg["shuffle_null"]["draws"]), "the label-shuffle null needs a fixed C"
    add_slices(d, cfg)
    arm, tool, key, cell, main_task, fold, rows = (d[k] for k in ("arm", "tool", "key", "cell", "main_task", "fold", "rows"))
    ckpt = checkpoint_dir / baseline if checkpoint_dir else None
    if ckpt is not None:  # a resumed run must have exactly the settings, inputs and code its checkpoints were made with
        inputs = hashlib.sha256()
        for r, f, err, cmd in zip(rows, fold, d["error"], d["editor_command"]):  # rows, folds, text, and what selects rows
            inputs.update(f"{r['key']}|{r['step']}|{r['action']}|{r['arm']}|{r['cell']}|{r['tool']}|{err}|{cmd}|{f}|{r[spec['text']]}\n".encode())
        identity = json.dumps({"config": {k: v for k, v in cfg.items() if k != "n_jobs"}, "spec": spec,
                               "inputs": inputs.hexdigest(), "code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                              sort_keys=True)
        saved = ckpt / "identity.json"
        if saved.exists():
            assert saved.read_text() == identity, f"{baseline}: checkpoints in {ckpt} were made with other settings; use a new stamp"
        else:
            assert not list(ckpt.glob("*.npy")), f"{baseline}: checkpoints in {ckpt} have no recorded identity; use a new stamp"
            ckpt.mkdir(parents=True, exist_ok=True)
            saved.write_text(identity)
            commit()

    def cached(name, compute):
        if ckpt is None:
            return compute()
        path = ckpt / f"{name}.npy"
        if path.exists():
            print(f"  {baseline}: reusing checkpoint {name}", flush=True)
            return np.load(path)
        value = compute()
        ckpt.mkdir(parents=True, exist_ok=True)
        np.save(path, value)
        commit()
        return value

    def evaluate(score, keep, neg, pos):
        return evaluate_scores(d, score, keep, neg, pos, cfg)

    texts = texts_of(d, spec)
    X = featurize(texts, cfg)
    print(f"{baseline}: {X.shape[0]} outputs, {X.nnz:,} nonzeros", flush=True)
    res, path = {}, out_dir / f"{baseline}_scores_{stamp}.jsonl.gz"
    rows_written = Counter()

    def fit_sets(outputs, keep, neg, pos, scored):  # (y, train rows, scored rows): cells with a run on each side only
        both = keep & np.isin(cell, list(set(cell[on_side(arm, neg)]) & set(cell[on_side(arm, pos)])))
        return on_side(arm, pos).astype(int), both & (on_side(arm, neg) | on_side(arm, pos)), both & np.isin(arm, scored)

    # With a fixed C, every (comparison, output variant) set of fold fits goes into one pool so all cores stay busy;
    # each set is checkpointed as soon as its folds finish. (Inner-CV C runs its own pool per set below.)
    precomputed = {}
    if cfg["C"] != "inner_cv":
        todo = [(f"{c}__{o}", fit_sets(o, k, neg, pos, scored)) for o, k in output_variants(d, cfg).items()
                for c, (neg, pos, scored) in comparisons(cfg).items() if ckpt is None or not (ckpt / f"{c}__{o}.npy").exists()]
        for (name, _), s in zip(todo, oof_many(X, [t for _, t in todo], fold, cfg, cfg["C"])):
            precomputed[name] = cached(name, lambda s=s: s)
    with gzip.open(path, "wt") as fh:
        for outputs, keep in output_variants(d, cfg).items():
            for comparison, (neg, pos, scored) in comparisons(cfg).items():
                name = f"{comparison}__{outputs}"
                chosen = {}  # C per outer fold, when C is picked by inner cross-validation; checkpointed with the scores
                y_set, train_set, test_set = fit_sets(outputs, keep, neg, pos, scored)

                def scores_and_C():
                    s = oof(X, y_set, fold, train_set, test_set, cfg, cfg["C"], chosen)
                    if chosen and ckpt is not None:
                        ckpt.mkdir(parents=True, exist_ok=True)
                        (ckpt / f"{name}__inner_cv_C.json").write_text(json.dumps(chosen))
                    return s

                score = precomputed[name] if name in precomputed else cached(name, scores_and_C)
                if not chosen and ckpt is not None and (saved_C := ckpt / f"{name}__inner_cv_C.json").exists():
                    chosen = json.loads(saved_C.read_text())
                res[f"{comparison}/{outputs}"] = evaluate(score, keep, neg, pos)
                if chosen:
                    res[f"{comparison}/{outputs}"]["inner_cv_C"] = chosen
                    print(f"  {baseline} {comparison} {outputs} C per fold: {json.dumps({f: v['C'] for f, v in chosen.items()})}", flush=True)
                for other in scored if comparison == "real_vs_sim" else []:  # arms this model scores but never trained on
                    if other == E2 and neg == E1:  # epoch 2: should look like epoch 1
                        res[f"{comparison}/{outputs}"]["epoch2_vs_epoch1_scored_by_this_model"] = evaluate(score, keep, neg, other)
                    elif other not in arms_of(neg) + arms_of(pos):  # e.g. another simulator's runs, on the cells it has runs on
                        on = d | {"slices": {f"cells_with_{other}": set(cell[arm == other])}}
                        res[f"{comparison}/{outputs}"][f"{other}_vs_{side_name(neg)}_scored_by_this_model"] = evaluate_scores(
                            on, score, keep, neg, other, cfg)
                for i in np.flatnonzero(~np.isnan(score)):
                    rows_written[(comparison, outputs, arm[i])] += 1
                    fh.write(json.dumps({"baseline": baseline, "comparison": comparison, "outputs": outputs,
                                         "arm": arm[i], "key": key[i], "cell": cell[i], "main_task": main_task[i],
                                         "step": rows[i]["step"], "action": rows[i]["action"], "tool": tool[i],
                                         "fold": int(fold[i]), "score": round(float(score[i]), 5)}) + "\n")
                a = res[f"{comparison}/{outputs}"]["all_cells"]
                print(f"  {baseline} {comparison:14} {outputs:9} outputs {a['outputs']['auroc']:.3f} {a['outputs']['ci95']}  "
                      f"runs {a['runs']['auroc']:.3f} {a['runs']['ci95']}  pairs {a['pairs']['accuracy']:.3f}", flush=True)
    real, sim = comparisons(cfg)["real_vs_sim"][:2]
    keep, pair_rows = np.ones(len(rows), bool), on_side(arm, real) | on_side(arm, sim)
    y = on_side(arm, sim).astype(int)
    # C sensitivity on the main comparison, reported only (C is fixed in config, not chosen on these numbers)
    if cfg["c_sensitivity"]:
        res["c_sensitivity_real_vs_sim_all"] = {
            str(C): evaluate(cached(f"c_sensitivity__{C}", lambda: oof(X, y, fold, pair_rows, pair_rows, cfg, C)), keep, real, sim)["all_cells"]
            for C in cfg["c_sensitivity"]}
    # Label-shuffle null for the main comparison (all outputs): per main task, with probability 1/2 every one of its
    # cells swaps which run is called simulated (one flip per main task, as the folds and the bootstrap group by main
    # task, and as the probe's null does); every output of a run keeps one label; same folds, refit, scored against
    # the shuffled labels.
    if cfg["shuffle_null"]["draws"]:
        observed = res["real_vs_sim/all"]["all_cells"]
        tasks_u, task_idx = np.unique(main_task, return_inverse=True)
        perms = [np.where((np.random.default_rng(cfg["seed"] + 1 + k).random(len(tasks_u)) < 0.5)[task_idx], 1 - y, y)
                 for k in range(cfg["shuffle_null"]["draws"])]
        folds = sorted(set(fold))
        # Every (draw, fold) fit in one pool, so all CPUs stay busy.
        jobs = [(k, pair_rows & (fold != f), pair_rows & (fold == f)) for k in range(len(perms)) for f in folds]

        def shuffle_scores():
            parts = Parallel(n_jobs=cfg["n_jobs"])(delayed(_fold_scores)(X, perms[k], tr, te, cfg, cfg["C"]) for k, tr, te in jobs)
            out = np.full((len(perms), len(y)), np.nan)
            for (k, _, te), s in zip(jobs, parts):
                out[k, te] = s
            return out

        scores = cached(f"shuffle_null_per_main_task__{len(perms)}", shuffle_scores)
        draws = []
        for y_perm, s in zip(perms, scores):
            run_score, first = run_means(s, key, pair_rows)
            draws.append((roc_auc_score(y_perm[pair_rows], s[pair_rows]), roc_auc_score(y_perm[first], run_score)))
        draws = np.array(draws)
        res["label_shuffle_null_real_vs_sim_all"] = {
            level: {"n": len(draws), "mean": round(float(v.mean()), 4), "p95": round(float(np.percentile(v, 95)), 4),
                    "max": round(float(v.max()), 4), "observed": observed[level]["auroc"],
                    "p_value": round(float((1 + (v >= observed[level]["auroc"]).sum()) / (1 + len(v))), 4)}
            for level, v in (("outputs", draws[:, 0]), ("runs", draws[:, 1]))}
        print(f"  {baseline} label-shuffle null ({len(draws)} draws): {json.dumps(res['label_shuffle_null_real_vs_sim_all'])}", flush=True)
    # Top features, for reading only: one model per comparison on all its outputs
    top, comps = {}, list(comparisons(cfg).items())
    masks = [(on_side(arm, neg) | on_side(arm, pos)) & np.isin(cell, list(set(cell[on_side(arm, neg)]) & set(cell[on_side(arm, pos)])))
             for _, (neg, pos, _) in comps]  # paired cells, as scored
    fits = Parallel(n_jobs=len(comps))(delayed(fit)(X[m], on_side(arm[m], pos).astype(int), cfg, 1.0 if cfg["C"] == "inner_cv" else cfg["C"])
                                       for m, (_, (neg, pos, _)) in zip(masks, comps))  # the comparisons' fits at once; reading only
    for (comparison, (neg, pos, _)), (_, clf) in zip(comps, fits):
        coef = clf.coef_[0]
        order = np.argsort(coef)
        cols = {f"toward_{side_name(pos)}": order[::-1][:cfg["top_features"]], f"toward_{side_name(neg)}": order[:cfg["top_features"]]}
        names = name_columns(X, texts, np.concatenate(list(cols.values())), cfg)
        top[comparison] = {side: [{"ngrams": names[int(c)], "weight": round(float(coef[c]), 3),
                                   "kind": vectorizers(cfg)[c // cfg["n_features"]][0]} for c in cs] for side, cs in cols.items()}
    out = {"baseline": baseline, "spec": spec, "counts": d["counts"], "results": res, "top_features": top, "score_file": path.name,
           "score_rows": {"/".join(k): n for k, n in sorted(rows_written.items())}}
    (out_dir / f"{baseline}_summary_{stamp}.json").write_text(json.dumps(out, indent=1))  # checkpoint per baseline
    return out


def curve_subsets(cfg: dict, d: dict, fold_of: dict) -> dict:
    """Nested random subsets of each fold's training main tasks: per (seed, fold) one permutation from
    numpy default_rng([seed, fold]); subset n is its first n tasks, sorted. "all" is every training main task."""
    tasks = sorted(set(d["main_task"]))
    folds = {}
    for f in sorted(set(d["fold"].tolist())):
        train = sorted(t for t in tasks if fold_of[t] != f)
        subsets = {}
        for seed in cfg["seeds"]:
            perm = [str(t) for t in np.random.default_rng([seed, f]).permutation(train)]
            for n in cfg["train_main_tasks"]:
                if n != "all":
                    assert n <= len(train), f"fold {f} has only {len(train)} training main tasks"
                    subsets.setdefault(str(n), {})[str(seed)] = sorted(perm[:n])
        subsets["all"] = {"all": train}
        folds[str(f)] = {"train_main_tasks": train, "test_main_tasks": sorted(t for t in tasks if fold_of[t] == f),
                         "subsets": subsets}
    return {"rule": curve_subsets.__doc__.strip(), "examples": cfg["examples"], "folds_file": cfg["folds_file"],
            "seeds": cfg["seeds"], "train_main_tasks": cfg["train_main_tasks"], "folds": folds}


def curve_fold(cfg: dict, d: dict, variant: str, fold: int, fold_subsets: dict) -> dict:
    """Train on each task subset of one fold's training tasks (epoch-1 vs simulated outputs), score the held-out fold."""
    spec = cfg["variants"][variant]
    keep = np.ones(len(d["rows"]), bool) if spec["outputs"] == "all" else d["tool"] != "text_editor"
    pair = keep & np.isin(d["arm"], [E1, SIM])
    test = pair & (d["fold"] == fold)
    X, y = featurize(texts_of(d, spec), cfg), (d["arm"] == SIM).astype(int)
    jobs = [(n, s, pair & np.isin(d["main_task"], tasks)) for n, by_seed in fold_subsets["subsets"].items() for s, tasks in by_seed.items()]
    parts = Parallel(n_jobs=min(len(jobs), cfg["n_jobs"]))(delayed(_fold_scores)(X, y, tr, test, cfg, cfg["C"]) for _, _, tr in jobs)
    return {"variant": variant, "fold": fold, "test_rows": np.flatnonzero(test).tolist(),
            "scores": {f"{n}/{s}": np.round(p, 5).tolist() for (n, s, _), p in zip(jobs, parts)},
            "train_rows": {f"{n}/{s}": int(tr.sum()) for n, s, tr in jobs}}


def curve_summary(cfg: dict, d: dict, fold_results: list[dict]) -> dict:
    """Pool each (variant, subset size, seed)'s held-out scores over the folds: output-level and run-level AUROC."""
    out = {}
    for variant in cfg["variants"]:
        per = [r for r in fold_results if r["variant"] == variant]
        assert sorted(r["fold"] for r in per) == sorted(set(d["fold"].tolist())), f"{variant}: missing folds"
        rows = np.concatenate([r["test_rows"] for r in per])
        res = {}
        for size_seed in per[0]["scores"]:
            score = np.full(len(d["rows"]), np.nan)
            for r in per:
                score[r["test_rows"]] = r["scores"][size_seed]
            m = np.zeros(len(score), bool)
            m[rows] = True
            run_score, first = run_means(score, d["key"], m)
            n, seed = size_seed.split("/")
            res.setdefault(n, {})[seed] = {"outputs": round(float(roc_auc_score(d["arm"][m] == SIM, score[m])), 4),
                                           "runs": round(float(roc_auc_score(d["arm"][first] == SIM, run_score)), 4),
                                           "train_rows_mean_per_fold": round(float(np.mean([r["train_rows"][size_seed] for r in per])), 1)}
        out[variant] = {}
        for n, by_seed in res.items():
            stats = {}
            for level in ("outputs", "runs"):
                v = np.array([x[level] for x in by_seed.values()])
                stats[f"{level}_mean"], stats[f"{level}_sd"] = round(float(v.mean()), 4), round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 4)
            out[variant][n] = {"seeds": by_seed} | stats
    return out


def provenance(cfg: dict, counts: dict) -> dict:
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=HERE).stdout.strip())
    # A dirty run also records its uncommitted code (the repo's .py and .yaml files: the diff against HEAD, and untracked
    # files in full), so the source that ran can be rebuilt.
    diff, untracked = "", {}
    if dirty:
        root = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, cwd=HERE, check=True).stdout.strip())
        diff = subprocess.run(["git", "diff", "HEAD", "--", "*.py", "*.yaml"], capture_output=True, text=True, cwd=root, check=True).stdout
        new = subprocess.run(["git", "ls-files", "--others", "--exclude-standard"], capture_output=True, text=True, cwd=root, check=True).stdout
        untracked = {f: (root / f).read_text() for f in new.splitlines() if f.endswith((".py", ".yaml"))}
    return {"config": cfg, "git": git, "git_dirty": dirty, "git_diff_code": diff, "git_untracked_code": untracked, "counts": counts}


def main():
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "config" / "tfidf.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    fold_key, fold_of = fold_map(cfg, json.loads((SPLITS / cfg["folds_file"]).read_text()))
    cfg = with_cells(cfg)
    d = load(HERE / ".run_cache" / cfg["examples"], fold_of, fold_key=fold_key, cells=cfg.get("cells"), holdout_arm=cfg.get("holdout_arm"))
    print(json.dumps(d["counts"]), flush=True)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    summary = provenance(cfg, d["counts"]) | {"baselines": {}}
    for baseline, spec in cfg["baselines"].items():
        segs = spec.get("segments", "outputs")
        data = d if segs == "outputs" else load(HERE / ".run_cache" / cfg["examples"], fold_of, segs, fold_key, cfg.get("cells"),
                                                cfg.get("holdout_arm"))
        summary["baselines"][baseline] = run_baseline(cfg, data, baseline, spec, HERE / "results", stamp)
    out = HERE / "results" / f"{cfg['name']}_summary_{stamp}.json"
    out.write_text(json.dumps(summary, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
