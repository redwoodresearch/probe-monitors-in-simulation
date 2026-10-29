"""POST HOC, descriptive (RULE.md Amendment 2, committed before this script; added after item 3's control failed its bar).
(i) the review note planted in the seeded 28 first-real replies, against the same 91 replies unplanted; (ii) every `ls`-style
permission string replaced by `<perm>` in all arms after the level-1 mask, then fake vs first real, second vs first real and
item 3's plant control rerun; and the double-newline n-gram's rank, coefficient and counts in every comparison of items 1, 3
and this amendment. Same sites, repairs, mask, plant, classifier and null as items 1 and 3 (imported unchanged). $0, local.
Writes results/controls_v2_POSTHOC_<PT stamp>.json: numbers and n-grams of at most 5 characters, no reply text.

  PYTHONPATH=src uv run --no-project --python 3.12 --with numpy --with scipy --with scikit-learn --with joblib \\
      --with pyyaml --with 'modal>=1.4.3' python \\
      experiments/2026-10-13_claimB_fake_reply_diffs/analyze_POSTHOC_controls_v2.py --data <meaning judge data dir>
"""

import argparse
import ast
import hashlib
import json
import re
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
import analyze_POSTHOC_fake_reply_diffs as F  # noqa: E402  (load, repair, classifier, char_ngrams: unchanged)
import analyze_POSTHOC_masked_planted as M  # noqa: E402  (plant, INSERT, PLANTED_N, SEED: unchanged)
from probe_shift.scrub import scrub  # noqa: E402

PERM = re.compile(r"(?<![\w-])(?!-{10}(?![\w-]))[-dlcbpsD][-r][-w][-xsStT][-r][-w][-xsStT][-r][-w][-xsStT][.+@]?(?![\w-])")
GRAM = "\n\n"


def gram_report(pos: list[str], neg: list[str]) -> dict:
    """The double newline in the all-rows fit (the same fit as classifier()'s top-feature lists): rank toward each class
    (1 = top), coefficient, counts and replies per class; None if it occurs in no reply."""
    texts, y = pos + neg, np.r_[np.ones(len(pos), int), np.zeros(len(neg), int)]
    cv = CountVectorizer(analyzer=lambda t: F.char_ngrams(t, 1, 5))
    X = cv.fit_transform(texts).tocsr()
    vocab = {g: j for j, g in enumerate(cv.get_feature_names_out())}
    if GRAM not in vocab:
        return None
    coef = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000).fit(TfidfTransformer(sublinear_tf=True).fit_transform(X), y).coef_[0]
    j, col = vocab[GRAM], X[:, vocab[GRAM]].toarray().ravel()
    return {"coef": round(float(coef[j]), 6), "rank_toward_pos": int((coef > coef[j]).sum()) + 1, "rank_toward_neg": int((coef < coef[j]).sum()) + 1,
            "count_pos": int(col[y == 1].sum()), "count_neg": int(col[y == 0].sum()),
            "replies_pos": int((col[y == 1] > 0).sum()), "replies_neg": int((col[y == 0] > 0).sum()), "features": int(len(coef))}


def plant_check(res: dict, key: str, unplanted: list[list[str]]) -> dict:
    """Pass (a) and (b): (b) = the top feature toward the planted class is in the insert and in no unplanted reply."""
    top = ast.literal_eval(res[key][0]["ngram"])
    hits = int(sum(t.count(top) for ts in unplanted for t in ts))
    res |= {"top_feature_in_insert": top in M.INSERT, "top_feature_in_unplanted": hits,
            "pass_top_feature": bool(top in M.INSERT and hits == 0), "pass_auroc": res["above_null_p95"]}
    res["pass"] = res["pass_top_feature"] and res["pass_auroc"]
    first = next((i for i, f in enumerate(res[key]) if ast.literal_eval(f["ngram"]) in M.INSERT), None)
    res["first_insert_feature_rank"] = None if first is None else first + 1  # within the reported top 30
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="the meaning judge's data dir (sites.jsonl, sim_new.jsonl)")
    args = ap.parse_args()
    for f in (Path(__file__).resolve(), Path(F.__file__).resolve(), Path(M.__file__).resolve()):
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
    chosen = sorted(np.random.default_rng(M.SEED).choice(len(sites), M.PLANTED_N, replace=False).tolist())
    plant = lambda ts: [M.plant(t) if i in chosen else t for i, t in enumerate(ts)]  # noqa: E731
    dropped = {w: [PERM.sub("<perm>", t) for t in ts] for w, ts in masked.items()}
    out = {"label": "POST HOC (added after item 3 failed), descriptive, preliminary until reviewed", "rule": "RULE.md Amendment 2",
           "git": git, "input_digest": {f: hashlib.sha256((args.data / f).read_bytes()).hexdigest() for f in ("sites.jsonl", "sim_new.jsonl")},
           "planted_sites": chosen, "shuffles": F.SHUFFLES,
           "perm_drop": {w: {"replies_changed": int(sum(a != b for a, b in zip(masked[w], dropped[w]))),
                             "chars_removed": int(sum(len(a) - len(b) for a, b in zip(masked[w], dropped[w]))),
                             "fields_replaced": int(sum(len(PERM.findall(t)) for t in masked[w])),
                             "source_chars_replaced": int(sum(len(m.group(0)) for t in masked[w] for m in PERM.finditer(t)))} for w in masked},
           "classifier": {}, "double_newline": {}}
    # (i) first, alone: the plant against its own unplanted twins
    res = F.classifier(plant(masked["e1"]), masked["e1"], groups, ("e1_planted", "e1"))
    out["classifier"]["i_planted_vs_unplanted|masked"] = plant_check(res, "top_toward_e1_planted", [masked["e1"]])
    print("(i)", {k: res[k] for k in ("auroc", "null_mean", "null_p95", "p", "top_feature_in_insert", "top_feature_in_unplanted", "pass")}, flush=True)
    # (ii) permission strings dropped in every arm
    for comp, (a, names) in {"fake_vs_e1|masked_perm": ("sim", ("fake", "e1")), "e2_vs_e1|masked_perm": ("e2", ("e2", "e1"))}.items():
        out["classifier"][comp] = F.classifier(dropped[a], dropped["e1"], groups, names)
    res = F.classifier(plant(dropped["e1"]), dropped["e2"], groups, ("e1_planted", "e2"))
    out["classifier"]["planted_e1_vs_e2|masked_perm"] = plant_check(res, "top_toward_e1_planted", [dropped["e1"], dropped["e2"]])
    for k, c in out["classifier"].items():
        print(f"{k}: AUROC {c['auroc']} null mean {c['null_mean']} p95 {c['null_p95']} p {c['p']}" + (f" pass {c['pass']}" if "pass" in c else ""), flush=True)
    # the double newline in every comparison of items 1, 3 and this amendment, on the same text each used
    versions = {"item1 fake_vs_e1|masked": (masked["sim"], masked["e1"]), "item1 e2_vs_e1|masked": (masked["e2"], masked["e1"]),
                "item3 planted_e1_vs_e2|masked": (plant(masked["e1"]), masked["e2"]),
                "i planted_vs_unplanted|masked": (plant(masked["e1"]), masked["e1"]),
                "ii fake_vs_e1|masked_perm": (dropped["sim"], dropped["e1"]), "ii e2_vs_e1|masked_perm": (dropped["e2"], dropped["e1"]),
                "ii planted_e1_vs_e2|masked_perm": (plant(dropped["e1"]), dropped["e2"])}
    for k, (p, n) in versions.items():
        out["double_newline"][k] = gram_report(p, n)
        print("double newline", k, out["double_newline"][k], flush=True)
    print("perm drop:", out["perm_drop"])
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"controls_v2_POSTHOC_{stamp}.json"
    with open(path, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print("->", path)


if __name__ == "__main__":
    main()
