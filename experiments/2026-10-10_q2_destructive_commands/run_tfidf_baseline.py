"""Destructive-command probe, text baseline (local, free): TF-IDF (character n-grams of the call's arguments, the same
text the label rule reads) with logistic regression, on the probe's own calls and folds and the same threshold rule:
train on the real rows of the folds other than k and k+1, threshold at the 1% false-alarm point of fold k+1's real
negatives, score fold k's real and simulated rows. No layer or C selection (C = 1). The label is a rule on this text,
so a perfect text reader exists (the rule); this is a simple text baseline. No command text is written; per-call
scores only.

  uv run --no-project --python 3.14 --with scikit-learn==1.7.2 --with numpy \
      python experiments/2026-10-10_q2_destructive_commands/run_tfidf_baseline.py <transcript cache> <probe full output json>
"""
import glob
import hashlib
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))
from analyze_destructive_counts import COHORTS, DESTRUCTIVE
from build_labels import ARM, calls

HERE = Path(__file__).parent


def main(cache: Path, probe_path: Path):
    rows = json.loads(probe_path.read_text())["rows"]
    index = {(r["run"], r["step"], r["action"]): i for i, r in enumerate(rows)}
    text = [None] * len(rows)
    for world, d in COHORTS.items():
        for ev in sorted(glob.glob(str(cache / d / "logs" / "*.eval"))):
            with zipfile.ZipFile(ev) as z:
                for name in sorted(n for n in z.namelist() if n.startswith("samples/") and n.endswith(".json")):
                    s = json.loads(z.read(name))
                    run = f"{ARM[world]}|{Path(ev).name}|{s['id']}|{s.get('epoch', 1)}"
                    for step, action, _, args in calls(s):
                        if (i := index.get((run, step, action))) is not None:
                            text[i] = json.dumps(args)
    missing = sum(t is None for t in text)
    if missing:
        raise ValueError(f"{missing} of {len(rows)} probe rows have no call text")
    y = np.array([r["y"] for r in rows]); world = np.array([r["world"] for r in rows]); fold = np.array([r["fold"] for r in rows])
    relabel = np.array([bool(DESTRUCTIVE.search(t)) for t in text])
    if (relabel != y.astype(bool)).any():  # the join check: the rule on the joined text must give the probe's label
        raise ValueError(f"{(relabel != y.astype(bool)).sum()} rows: the rule on the joined text disagrees with the label")
    scores, caught, thresholds = np.full(len(y), np.nan), np.zeros(len(y), bool), {}
    for k in range(5):
        cal = (k + 1) % 5
        tr = np.flatnonzero((world == 0) & ~np.isin(fold, [k, cal]))
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, max_features=200_000)
        Xtr = vec.fit_transform([text[i] for i in tr])
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, y[tr])
        score = lambda idx, clf=clf, vec=vec: clf.decision_function(vec.transform([text[i] for i in idx]))
        negc = np.flatnonzero((world == 0) & (fold == cal) & (y == 0))
        neg = np.sort(score(negc))[::-1]
        thr = float(neg[int(np.floor(0.01 * len(neg)))])
        te = np.flatnonzero(fold == k)
        scores[te] = score(te); caught[te] = scores[te] > thr
        thresholds[k] = thr
        print(f"fold {k}: {len(tr)} training rows, {Xtr.shape[1]} features, threshold {thr:.3f}", flush=True)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = HERE.parents[1] / "data" / "destructive" / f"tfidf_{stamp}.json"
    out.write_text(json.dumps({"probe output": str(probe_path), "probe output sha256": hashlib.sha256(probe_path.read_bytes()).hexdigest(),
                               "thresholds": thresholds, "scores": [round(float(x), 5) for x in scores],
                               "caught": "".join("1" if c else "0" for c in caught)}) + "\n")
    print(out)


if __name__ == "__main__":
    main(Path(sys.argv[1]).expanduser(), Path(sys.argv[2]))
