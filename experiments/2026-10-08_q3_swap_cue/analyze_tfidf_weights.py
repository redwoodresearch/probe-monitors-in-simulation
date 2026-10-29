"""Which words does the tool-output text detector rely on? Summarizes run_tfidf_weights.py's per-feature contributions
into plain-word categories. $0, local. The n-grams stay in data/ (and S3); results/ gets categories and numbers only.

A feature's contribution is combined over the 5 outer models, weighted by their cells (a model whose vocabulary lacks
the n-gram contributes 0 for it). Every feature of every model is included, so the categories add up to the net gap.
Categories, by the n-gram's form:
- date-time stamp piece, unambiguous: an ISO day-T-hour ("NNtNN", optionally after the month or year) with a valid day
  (01-31) and hour (00-23), or a month name with a valid day;
- number pair that may be a time (ambiguous): any other two-number form the first version counted as a stamp piece
  ("NN NN", the year or "09" followed by two digits). These can be a day and hour, but also sizes, counts or IDs;
- bare one- or two-digit number;
- common English word (a short fixed list);
- other.

  python analyze_tfidf_weights.py --weights data/q3_swap_cue/tfidf_weights_<stamp>.json.gz
"""

import argparse
import gzip
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
STAMP = re.compile(r"^((sep|oct|2026|09)\s)?\d{2}t\d{2}$|^(sep|oct)\s\d{1,2}$|^2026\s\d{2}$|^\d{2}\s\d{2}$|^09\s\d{2}$")  # the first version
ISO = re.compile(r"^(?:(?:sep|oct|2026|09)\s)?(\d{2})t(\d{2})$")
MONTH_DAY = re.compile(r"^(?:sep|oct)\s(\d{1,2})$")
NUMBER = re.compile(r"^\d{1,2}$")
WORDS = {"the", "its", "is", "than", "are", "and", "on", "no", "has", "made", "of", "to", "a", "it", "this", "that", "in",
         "for", "was", "be", "with", "not", "as", "by", "an", "or"}


def category(ngram):
    m, d = ISO.match(ngram), MONTH_DAY.match(ngram)
    if (m and 1 <= int(m[1]) <= 31 and int(m[2]) <= 23) or (d and 1 <= int(d[1]) <= 31):
        return "date-time stamp piece (unambiguous)"
    if STAMP.match(ngram):
        return "number pair that may be a time (ambiguous)"
    if NUMBER.match(ngram):
        return "bare one- or two-digit number"
    return "common English word" if ngram in WORDS else "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    args = ap.parse_args()
    with gzip.open(args.weights, "rt") as fh:
        D = json.load(fh)
    n = sum(f["cells"] for f in D["fits"])
    gap = sum(f["cells"] * f["gap_logits"] for f in D["fits"]) / n
    contrib, weights = defaultdict(float), defaultdict(list)
    for f in D["fits"]:
        F = f["features"]
        for g, w, e1, sim in zip(F["ngram"], F["weight"], F["e1_mean"], F["sim_mean"]):
            contrib[g] += f["cells"] * w * (e1 - sim) / n
            weights[g].append(w)
    by_cat, count = defaultdict(float), Counter()
    for k, c in contrib.items():
        by_cat[category(k)] += c
        count[category(k)] += 1
    top = sorted(contrib.items(), key=lambda kv: -kv[1])[:15]
    top_w = defaultdict(list)  # each model's 30 largest positive weights
    for f in D["fits"]:
        F = f["features"]
        for j in sorted(range(len(F["weight"])), key=lambda j: -F["weight"][j])[:30]:
            top_w[F["ngram"][j]].append(F["weight"][j])
    out = {"inputs": {"weights": Path(args.weights).name, "gate_max_abs": max(f["gate_max_abs"] for f in D["fits"])},
           "cells": n, "net_gap_e1_minus_sim_logits": round(gap, 4),
           "positive_contributions_logits": round(sum(f["cells"] * f["contrib_pos_sum"] for f in D["fits"]) / n, 4),
           "negative_contributions_logits": round(sum(f["cells"] * f["contrib_neg_sum"] for f in D["fits"]) / n, 4),
           "sum_of_all_contributions_logits": round(sum(contrib.values()), 4),
           "by_category": {c: {"features": count[c], "logits": round(v, 4), "share_of_net_gap": round(v / gap, 3)} for c, v in by_cat.items()},
           "top_15_contributors": [{"rank": i + 1, "category": category(k), "logits": round(c, 4), "share_of_net_gap": round(c / gap, 3),
                                    "mean_weight": round(sum(weights[k]) / len(weights[k]), 2)} for i, (k, c) in enumerate(top)],
           "top_30_attack_weights_by_category (features in all 5 models)": dict(Counter(category(k) for k, v in top_w.items() if len(v) == 5))}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"tfidf_weights_summary_{stamp}.json"
    with path.open("x") as fh:  # one line per top-level entry
        fh.write("{\n" + ",\n".join(f"{json.dumps(k)}: {json.dumps(v)}" for k, v in out.items()) + "\n}\n")
    print(json.dumps(out, indent=1))
    print("->", path)


if __name__ == "__main__":
    main()
