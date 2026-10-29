"""POST HOC, descriptive (RULE.md Amendment 3, committed before this script; the last step on this thread). A made-up marker
planted in the seeded 28 first-real replies against their unplanted twins, in (A) the masked text and (B) the masked text
with permission strings dropped; the pass is read from (B). Only if (B) passes, the fake vs first-real feature lists of (B)
are written, with counts in all three worlds and simulator-specific flags read against the second-vs-first-real control.
Same sites, repairs, mask, drop, insert position, classifier and null as before (imported unchanged). $0, local.
Writes results/marker_control_POSTHOC_<PT stamp>.json: numbers and n-grams of at most 5 characters, no reply text.

  PYTHONPATH=src uv run --no-project --python 3.12 --with numpy --with scipy --with scikit-learn --with joblib \\
      --with pyyaml --with 'modal>=1.4.3' python \\
      experiments/2026-10-13_claimB_fake_reply_diffs/analyze_POSTHOC_marker_control.py --data <meaning judge data dir>
"""

import argparse
import ast
import hashlib
import json
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_POSTHOC_controls_v2 as C  # noqa: E402  (PERM: Amendment 2's pinned regex)
import analyze_POSTHOC_fake_reply_diffs as F  # noqa: E402  (load, repair, classifier, char_ngrams)
import analyze_POSTHOC_masked_planted as M  # noqa: E402  (SEED, PLANTED_N)
from probe_shift.scrub import scrub  # noqa: E402

MARKER = "gfpdkvlgzrrwtgvhfrtbzkbg"
assert "".join(random.Random("fake-reply-diffs-marker").choices("bcdfghjklmnpqrstvwxz", k=24)) == MARKER
INSERT = "\n" + MARKER
TOP = 30


def plant(t: str) -> str:
    """The marker on its own line after the reply's last non-newline character; the trailing newlines stay (as item 3)."""
    body = t.rstrip("\n")
    return body + INSERT + t[len(body):]


def occurrences(t: str, g: str) -> int:
    """Overlapping occurrences, as the n-gram counts count them."""
    n, i = 0, t.find(g)
    while i != -1:
        n, i = n + 1, t.find(g, i + 1)
    return n


def fit_all(pos: list[str], neg: list[str]):
    """classifier()'s all-rows fit: (feature names, coefficients)."""
    cv = CountVectorizer(analyzer=lambda t: F.char_ngrams(t, 1, 5))
    X = cv.fit_transform(pos + neg).tocsr()
    y = np.r_[np.ones(len(pos), int), np.zeros(len(neg), int)]
    coef = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000).fit(TfidfTransformer(sublinear_tf=True).fit_transform(X), y).coef_[0]
    return cv.get_feature_names_out(), coef


def feature_table(texts: dict[str, list[str]]) -> dict:
    """(B)'s fake vs first real top features toward each side, with counts and replies in all three worlds, the control
    fit's rank, and the mechanical simulator-specific flag (Amendment 3)."""
    names, coef = fit_all(texts["sim"], texts["e1"])
    cnames, ccoef = fit_all(texts["e2"], texts["e1"])
    cidx, csorted = {g: k for k, g in enumerate(cnames)}, np.sort(ccoef)
    # 1 + the features strictly ahead (ties share the best rank), by binary search on the sorted coefficients
    crank_e2 = lambda g: None if g not in cidx else int(len(csorted) - np.searchsorted(csorted, ccoef[cidx[g]], side="right")) + 1  # noqa: E731
    crank_e1 = lambda g: None if g not in cidx else int(np.searchsorted(csorted, ccoef[cidx[g]], side="left")) + 1  # noqa: E731
    order = np.argsort(coef, kind="stable")
    rows = {"toward_fake": [], "toward_e1": []}
    for side, idx in (("toward_fake", order[::-1][:TOP]), ("toward_e1", order[:TOP])):
        for j in idx:
            g = str(names[j])
            occ = {w: int(sum(occurrences(t, g) for t in ts)) for w, ts in texts.items()}
            rep = {w: int(sum(g in t for t in ts)) for w, ts in texts.items()}
            flag = (rep["sim"] > rep["e1"] and rep["e2"] <= rep["e1"]) if side == "toward_fake" else (rep["sim"] < rep["e1"] and rep["e2"] >= rep["e1"])
            rows[side].append({"ngram": repr(g), "coef": round(float(coef[j]), 6), "count": occ, "replies": rep,
                               "control_rank_toward_e2": crank_e2(g), "control_rank_toward_e1": crank_e1(g),
                               "simulator_specific": bool(flag)})
    assert all(len(ast.literal_eval(r["ngram"])) <= 5 for side in rows.values() for r in side)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="the meaning judge's data dir (sites.jsonl, sim_new.jsonl)")
    args = ap.parse_args()
    for f in (Path(__file__).resolve(), Path(F.__file__).resolve(), Path(M.__file__).resolve(), Path(C.__file__).resolve()):
        if subprocess.run(["git", "status", "--porcelain", str(f)], capture_output=True, text=True, check=True, cwd=F.REPO).stdout:
            raise SystemExit(f"commit {f.name} first: the result records the commit that ran")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=F.REPO).stdout.strip()
    sites = F.load(args.data)
    tasks = sorted({s["task"] for s in sites})
    groups = np.array([tasks.index(s["task"]) for s in sites])
    masked = {w: [] for w in ("sim", "e1", "e2")}
    for s in sites:
        for w in masked:
            t, _, has_range = F.repair(s["text"][w], s["command"])
            assert not has_range
            masked[w].append(scrub(t, 1))
    versions = {"A_masked": masked, "B_masked_perm": {w: [C.PERM.sub("<perm>", t) for t in ts] for w, ts in masked.items()}}
    chosen = sorted(np.random.default_rng(M.SEED).choice(len(sites), M.PLANTED_N, replace=False).tolist())
    out = {"label": "POST HOC (added after Amendment 2's (i) failed), descriptive, preliminary until reviewed", "rule": "RULE.md Amendment 3",
           "git": git, "input_digest": {f: hashlib.sha256((args.data / f).read_bytes()).hexdigest() for f in ("sites.jsonl", "sim_new.jsonl")},
           "marker": MARKER, "planted_sites": chosen, "shuffles": F.SHUFFLES, "versions": {}}
    for v, texts in versions.items():
        unplanted = texts["e1"]
        grams = {g for n in range(1, 6) for g in F.char_ngrams(MARKER, n, n)}  # the marker itself, not the newline before it
        marker_grams = {g for g in grams if not any(g in t for t in unplanted)}  # counted on unplanted text, before planting
        if not marker_grams:  # Amendment 3: an empty eligible set stops this version's check; the marker is not changed
            out["versions"][v] = {"marker_ngrams_before_planting": 0, "pass": False, "stopped": "no eligible marker n-gram"}
            print(v, "stopped: no eligible marker n-gram", flush=True)
            continue
        res = F.classifier([plant(t) if i in chosen else t for i, t in enumerate(unplanted)], unplanted, groups, ("e1_marked", "e1"))
        top = ast.literal_eval(res["top_toward_e1_marked"][0]["ngram"])
        first = next((i + 1 for i, f in enumerate(res["top_toward_e1_marked"]) if ast.literal_eval(f["ngram"]) in marker_grams), None)
        res |= {"marker_distinct_ngrams": len(grams), "marker_ngrams_before_planting": len(marker_grams), "top_is_marker_ngram": top in marker_grams,
                "first_marker_ngram_rank": first, "pass_auroc": res["above_null_p95"]}
        res["pass"] = bool(res["pass_auroc"] and res["top_is_marker_ngram"])
        out["versions"][v] = res
        print(v, {k: res[k] for k in ("auroc", "null_mean", "null_p95", "p", "marker_distinct_ngrams", "marker_ngrams_before_planting",
                                      "top_is_marker_ngram", "first_marker_ngram_rank", "pass")}, flush=True)
    out["primary_pass"] = out["versions"]["B_masked_perm"]["pass"]
    if out["primary_pass"]:
        out["feature_lists_B"] = feature_table(versions["B_masked_perm"])
        for side, rows in out["feature_lists_B"].items():
            print(side, "simulator-specific:", sum(r["simulator_specific"] for r in rows), "of", len(rows))
    else:
        out["feature_lists_B"] = "not written: (B) failed the registered bar (Amendment 3)"
        print(out["feature_lists_B"])
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"marker_control_POSTHOC_{stamp}.json"
    with open(path, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print("->", path)


if __name__ == "__main__":
    main()
