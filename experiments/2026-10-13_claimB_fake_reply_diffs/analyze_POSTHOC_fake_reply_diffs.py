"""POST HOC, descriptive (RULE.md in this folder @ 4fb94059, committed before this script): what differs between fake
(simulated) and real tool replies at the meaning judge's 91 strict honest sites, beyond the two editor slips.
(A) surface features, paired per site with a task-clustered bootstrap; (B) a character 1-5-gram classifier with a
within-task label-shuffle null and its top features; each fake vs e1 with e2 vs e1 as the control, before and after the
repairs (fixes.apply R1/R2/R5, then the evening; R6 is not applied, see RULE.md). Local CPU, $0, no Modal call.
Writes results/fake_reply_diffs_POSTHOC_<PT stamp>.json: numbers and n-grams of at most 5 characters, no reply text.

  PYTHONPATH=src uv run --no-project --python 3.12 --with numpy --with scipy --with scikit-learn --with joblib \\
      --with pyyaml --with 'modal>=1.4.3' python \\
      experiments/2026-10-13_claimB_fake_reply_diffs/analyze_POSTHOC_fake_reply_diffs.py --data <meaning judge data dir>
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "experiments/2026-10-09_fixed_sim"), str(REPO / "experiments/2026-10-06_q1_text_baselines")]
import clock  # noqa: E402  (R6's timestamp matcher, for the n_clock_stamps feature only)
import fixes  # noqa: E402  (the frozen fixed-sim rules R1, R2, R5, unchanged)
import run_tfidf_slips_evened_modal as SL  # noqa: E402  (the evening: even() and RANGE, unchanged; no Modal call)
from analyze_tfidf import char_ngrams  # noqa: E402  (whitespace-preserving character n-grams, unchanged)

SEED, DRAWS, SHUFFLES, TOP, N_SITES = 20261013, 2000, 100, 30, 91
SL.even_order("")  # compiles SL.ERROR_LINE exactly as the evening does (scrub.STDERR_LINE, whole line, any case)
TRUNC = ("<START_TOOL_OUTPUT>", "<END_TOOL_OUTPUT>", "…", "[...]")
TRUNC_CI = ("truncated", "output too long")
NUMBERS = {"n_ints": r"\d+", "n_thousands_sep": r"\b\d{1,3}(?:,\d{3})+\b", "n_decimals": r"\d+\.\d+",
           "n_percent": r"\d+(?:\.\d+)?%", "n_human_sizes": r"\b\d+(?:\.\d+)?[KMGT]i?B?\b", "n_hex_ids": r"\b[0-9a-f]{12,}\b"}
NUMBERS = {k: re.compile(v) for k, v in NUMBERS.items()}


def load(data: Path) -> list[dict]:
    """The strict sites with their three replies; the fake reply mapped exactly as run_meaning_judge.py's jobs()."""
    sites = [s for s in map(json.loads, (data / "sites.jsonl").read_text().splitlines()) if s["set"] == "strict"]
    new = {r["pair"]: r["result"]["reader_output"] for r in map(json.loads, (data / "sim_new.jsonl").read_text().splitlines())
           if r["status"] == "ok"}
    out = []
    for s in sites:
        sim = s["sim"] if s["sim"] is not None else new.get(s["site"])
        assert isinstance(sim, str) and isinstance(s["e1"], str) and isinstance(s["e2"], str), s["site"]
        assert (s["sim"] is None) == (s["sim_source"] == "new"), s["site"]
        out.append({"site": s["site"], "task": s["task"], "source": s["sim_source"], "command": s["command"],
                    "text": {"sim": sim, "e1": s["e1"], "e2": s["e2"]}})
    assert len(out) == N_SITES and len({s["site"] for s in out}) == N_SITES, len(out)
    return out


def repair(text: str, cmd: dict) -> tuple[str, list[str], bool]:
    """(repaired text, names of the rules that changed it, holds RANGE). fixes.apply on a one-call transcript, then the
    evening's even(); the editor command is the text_editor call's `command`, None for shell calls."""
    msgs = [{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": cmd["function"], "arguments": cmd["arguments"]}}]},
            {"role": "tool", "content": text}]
    new, recs = fixes.apply(msgs, ("R1", "R2", "R5"))
    t = new[1]["content"]
    hits = sorted({r["rule"] for r in recs if r.get("changed")})
    editor = cmd["arguments"].get("command") if cmd["function"] == "text_editor" else None
    t2, ev = SL.even(t, editor, cmd["function"])
    return t2, hits + ev, SL.RANGE in t


def features(t: str) -> dict:
    lines = t.split("\n")
    inner = t.strip("\n").split("\n") if t.strip("\n") else []
    lead, trail = len(t) - len(t.lstrip()), len(t) - len(t.rstrip())
    err = [SL.ERROR_LINE.fullmatch(l) is not None for l in lines]
    low = t.lower()
    n_trunc = sum(t.count(m) for m in TRUNC) + sum(low.count(m) for m in TRUNC_CI)
    f = {"n_chars": len(t), "n_lines": len(t.splitlines()), "n_newlines": t.count("\n"),
         "max_line_len": max(map(len, lines)), "mean_line_len": float(np.mean([len(l) for l in lines])),
         "leading_ws": lead, "trailing_ws": trail, "leading_newlines": t[:lead].count("\n"),
         "trailing_newlines": t[len(t) - trail:].count("\n"), "ends_with_newline": int(t.endswith("\n")),
         "n_tabs": t.count("\t"), "tab_indented_lines": sum(l.startswith("\t") for l in lines),
         "space_indented_lines": sum(l.startswith(" ") for l in lines),
         "trailing_space_lines": sum(l.endswith((" ", "\t")) for l in lines),
         "blank_lines": sum(l.strip() == "" for l in inner), "n_cr": t.count("\r"), "n_esc": t.count("\x1b"),
         "n_other_ctrl": sum((ord(c) < 32 and c not in "\t\n\r\x1b") or ord(c) == 127 for c in t),
         "n_non_ascii": sum(ord(c) > 127 for c in t), "n_trunc_markers": n_trunc, "has_trunc_marker": int(n_trunc > 0)}
    f |= {k: len(rx.findall(t)) for k, rx in NUMBERS.items()}
    f["n_clock_stamps"] = sum(s["kind"] != "FAILED" for s in clock.stamps(t, clock.LS_YEAR))
    f |= {"is_empty": int(t == ""), "is_blank": int(t.strip() == ""), "n_error_lines": sum(err),
          "has_error_line": int(any(err)), "first_line_error": int(err[0])}
    return f


def interval(x: np.ndarray, W: np.ndarray) -> dict:
    """Mean of per-site values and its 95% percentile interval over the task-clustered draws (rows of W: site weights)."""
    tot = W.sum(1)
    ok = tot > 0
    boot = (W[ok] @ x) / tot[ok]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": round(float(x.mean()), 6), "lo": round(float(lo), 6), "hi": round(float(hi), 6),
            "excludes_0": 1 if lo > 0 else -1 if hi < 0 else 0,  # from the unrounded interval
            "draws_used": int(ok.sum()), "pos": int((x > 0).sum()), "neg": int((x < 0).sum()), "zero": int((x == 0).sum())}


def excludes0(d: dict) -> int:
    return d["excludes_0"]


def surface(F: dict, keep: np.ndarray, W: np.ndarray, source: np.ndarray) -> dict:
    """Per feature: world means and the paired differences (fake - e1, e2 - e1, fake - e2), all sites and per source."""
    out = {}
    for k in F["sim"][0]:
        v = {w: np.array([F[w][i][k] for i in range(len(keep))], float) for w in ("sim", "e1", "e2")}
        row = {}
        for name, m in (("all", keep), ("natural", keep & (source == "natural")), ("new", keep & (source == "new"))):
            row[name] = {"n_sites": int(m.sum()), "world_means": {w: round(float(v[w][m].mean()), 6) for w in v},
                         "fake_minus_e1": interval((v["sim"] - v["e1"])[m], W[:, m]),
                         "e2_minus_e1": interval((v["e2"] - v["e1"])[m], W[:, m]),
                         "fake_minus_e2": interval((v["sim"] - v["e2"])[m], W[:, m])}
        out[k] = row
    return out


def fit_scores(X, y, train, test):
    """Vocabulary limited to n-grams seen in the training rows; tf-idf fit on them; logistic regression, C fixed."""
    seen = np.flatnonzero(np.asarray(X[train].sum(0)).ravel() > 0)
    Xs = X[:, seen]
    tf = TfidfTransformer(sublinear_tf=True).fit(Xs[train])
    clf = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000).fit(tf.transform(Xs[train]), y[train])
    return clf.decision_function(tf.transform(Xs[test]))


def cv_auroc(X, y, folds) -> float:
    s = np.full(len(y), np.nan)
    for tr, te in folds:
        s[te] = fit_scores(X, y, tr, te)
    assert not np.isnan(s).any()
    return float(roc_auc_score(y, s))


def classifier(pos: list[str], neg: list[str], groups: np.ndarray, names: tuple[str, str]) -> dict:
    """Label 1 = `pos` replies, 0 = `neg` replies of the same sites; CV grouped by main task; within-task shuffle null."""
    texts, y, g = pos + neg, np.r_[np.ones(len(pos), int), np.zeros(len(neg), int)], np.r_[groups, groups]
    cv = CountVectorizer(analyzer=lambda t: char_ngrams(t, 1, 5))
    X = cv.fit_transform(texts).tocsr()
    vocab = cv.get_feature_names_out()
    folds = list(GroupKFold(n_splits=5).split(X, y, g))
    obs = cv_auroc(X, y, folds)
    rng = np.random.default_rng(SEED)
    shuffled = []
    for _ in range(SHUFFLES):
        yy = y.copy()
        for task in sorted(set(g)):
            idx = np.flatnonzero(g == task)
            yy[idx] = y[idx][rng.permutation(len(idx))]
        shuffled.append(yy)
    null = np.array(Parallel(n_jobs=-1)(delayed(cv_auroc)(X, yy, folds) for yy in shuffled))
    tf = TfidfTransformer(sublinear_tf=True).fit(X)
    coef = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000).fit(tf.transform(X), y).coef_[0]
    cnt = {1: np.asarray(X[y == 1].sum(0)).ravel(), 0: np.asarray(X[y == 0].sum(0)).ravel()}
    docs = {1: np.asarray((X[y == 1] > 0).sum(0)).ravel(), 0: np.asarray((X[y == 0] > 0).sum(0)).ravel()}
    order = np.argsort(coef, kind="stable")

    def feat(j):
        return {"ngram": repr(str(vocab[j])), "coef": round(float(coef[j]), 6),
                f"count_{names[0]}": int(cnt[1][j]), f"count_{names[1]}": int(cnt[0][j]),
                f"replies_{names[0]}": int(docs[1][j]), f"replies_{names[1]}": int(docs[0][j])}

    assert all(len(str(vocab[j])) <= 5 for j in np.r_[order[:TOP], order[-TOP:]])
    return {"rows": len(y), "positives": names[0], "negatives": names[1], "folds": 5, "vocabulary": int(X.shape[1]),
            "auroc": round(obs, 4), "null_mean": round(float(null.mean()), 4),
            "null_p95": round(float(np.percentile(null, 95)), 4), "null_max": round(float(null.max()), 4),
            "p": round(float((1 + (null >= obs).sum()) / (SHUFFLES + 1)), 4), "above_null_p95": bool(obs > np.percentile(null, 95)),
            "null_draws": [round(float(v), 4) for v in null],
            f"top_toward_{names[0]}": [feat(j) for j in order[::-1][:TOP]],
            f"top_toward_{names[1]}": [feat(j) for j in order[:TOP]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="the meaning judge's data dir (sites.jsonl, sim_new.jsonl)")
    args = ap.parse_args()
    me = Path(__file__).resolve()
    if subprocess.run(["git", "status", "--porcelain", str(me)], capture_output=True, text=True, check=True, cwd=REPO).stdout:
        raise SystemExit(f"commit {me.name} first: the result records the commit that ran")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()
    sites = load(args.data)
    n, source = len(sites), np.array([s["source"] for s in sites])
    tasks = sorted({s["task"] for s in sites})
    assert len(tasks) == 72 and Counter(source.tolist()) == {"natural": 33, "new": 58}, (len(tasks), Counter(source.tolist()))
    site_task = np.array([tasks.index(s["task"]) for s in sites])
    draws = np.random.default_rng(SEED).integers(0, len(tasks), size=(DRAWS, len(tasks)))
    W = np.stack([np.bincount(d, minlength=len(tasks))[site_task] for d in draws]).astype(float)

    texts = {"before": {w: [s["text"][w] for s in sites] for w in ("sim", "e1", "e2")}, "after": {}}
    changed = Counter({f"{w}|{r}": 0 for w in ("sim", "e1", "e2") for r in ("R1", "R2", "R5", "create", "view", "order", "any")})
    has_range = np.zeros(n, bool)
    for w in ("sim", "e1", "e2"):
        texts["after"][w] = []
        for i, s in enumerate(sites):
            t, hits, rng_hit = repair(s["text"][w], s["command"])
            texts["after"][w].append(t)
            changed.update(f"{w}|{h}" for h in hits)
            changed[f"{w}|any"] += bool(hits)
            has_range[i] |= rng_hit
    assert all(changed[f"{w}|{r}"] == 0 for w in ("sim", "e1", "e2") for r in ("R1", "R2", "create")), changed  # out of scope (RULE.md)
    keep = {"before": np.ones(n, bool), "after": ~has_range}

    out = {"label": "POST HOC, descriptive, preliminary until reviewed", "rule": "RULE.md @ 4fb94059", "git": git,
           "input_digest": {f: hashlib.sha256((args.data / f).read_bytes()).hexdigest() for f in ("sites.jsonl", "sim_new.jsonl")},
           "seed": SEED, "draws": DRAWS, "shuffles": SHUFFLES,
           "sites": {"n": n, "tasks": len(tasks), "by_source": dict(Counter(source.tolist())),
                     "by_tool": dict(Counter(s["command"]["function"] for s in sites)),
                     "dropped_after_by_range": int(has_range.sum())},
           "repair_changes": dict(sorted(changed.items())), "r6": "not applied (RULE.md: run-level offset, attack-run donors)",
           "identical": {}, "surface": {}, "classifier": {}}
    F = {}
    for ver in ("before", "after"):
        T, m = texts[ver], keep[ver]
        out["identical"][ver] = {pair: int(sum(T[a][i] == T[b][i] for i in np.flatnonzero(m)))
                                 for pair, (a, b) in {"sim_eq_e1": ("sim", "e1"), "e2_eq_e1": ("e2", "e1"), "sim_eq_e2": ("sim", "e2")}.items()}
        F[ver] = {w: [features(t) for t in T[w]] for w in T}
        out["surface"][ver] = surface(F[ver], m, W, source)
        idx = np.flatnonzero(m)
        for comp, (a, names) in {"fake_vs_e1": ("sim", ("fake", "e1")), "e2_vs_e1": ("e2", ("e2", "e1"))}.items():
            res = classifier([T[a][i] for i in idx], [T["e1"][i] for i in idx], site_task[idx], names)
            out["classifier"][f"{comp}|{ver}"] = res
            print(f"{comp} {ver}: AUROC {res['auroc']} null mean {res['null_mean']} p95 {res['null_p95']} p {res['p']}", flush=True)

    both = keep["before"] & keep["after"]
    out["repair_change"], out["reading"] = {}, {}
    for k in F["before"]["sim"][0]:
        v = {(ver, w): np.array([F[ver][w][i][k] for i in range(n)], float) for ver in F for w in ("sim", "e1", "e2")}
        out["repair_change"][k] = {
            d: interval(((v["after", a] - v["after", b]) - (v["before", a] - v["before", b]))[both], W[:, both])
            for d, (a, b) in {"fake_minus_e1": ("sim", "e1"), "e2_minus_e1": ("e2", "e1"), "fake_minus_e2": ("sim", "e2")}.items()}
        seen = {}
        for ver in ("before", "after"):
            row = out["surface"][ver][k]["all"]
            s1, s2 = excludes0(row["fake_minus_e1"]), excludes0(row["fake_minus_e2"])
            seen[ver] = s1 if s1 != 0 and s1 == s2 else 0
        out["reading"][k] = {"before": seen["before"], "after": seen["after"],
                             "call": "survives" if seen["before"] and seen["after"] else "removed" if seen["before"]
                             else "new_after_repair" if seen["after"] else "none"}
    out["reading_counts"] = dict(Counter(r["call"] for r in out["reading"].values()))
    out["reading_note"] = (f"{len(out['reading'])} features x 2 versions, no multiplicity correction: 1-3 chance flags expected")
    print("repair changes:", out["repair_changes"])
    print("identical:", out["identical"])
    print("readings:", {k: r["call"] for k, r in out["reading"].items() if r["call"] != "none"})
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"fake_reply_diffs_POSTHOC_{stamp}.json"
    path.parent.mkdir(exist_ok=True)
    with open(path, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print("->", path)


if __name__ == "__main__":
    main()
