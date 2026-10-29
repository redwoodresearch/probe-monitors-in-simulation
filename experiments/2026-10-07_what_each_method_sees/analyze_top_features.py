"""Top TF-IDF features at levels 0 and 7, and which n-grams the probe's score follows.

1. TF-IDF: the baselines' featurizer and model (word 1-2-grams and character 2-5-grams, case and whitespace kept,
   hashed to 2^20 columns each, sublinear TF-IDF, L2 logistic regression, C = 1), fit once on every real-e1 and
   simulated output at the level (for reading only, as the baselines' own top features). Top N columns each way, each
   named by the n-grams that hash there in a few documents, with the share of real and simulated outputs containing it.
2. Probe, read through text: a ridge regression of the probe's held-out-environment log-odds on the same TF-IDF
   features. Out-of-fold R^2 on the main-task folds says how much of the probe's per-output score the output's own
   words explain; the top columns of a fit on all rows say which words move it.

Heavy (one 2-million-column fit per level plus ten ridge fits): run it on Modal, not the laptop:
  uv run modal run --detach experiments/2026-10-07_what_each_method_sees/run_top_features_modal.py::main
which calls `main` here with the inputs uploaded to a Volume. Writes results/top_features_L<level>_<stamp>.json and .md.
"""

import argparse
import gzip
import json
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
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.utils import murmurhash3_32

from probe_shift.provenance import capture
from probe_shift.scrub import scrub

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
E1, SIM = "real-attack-e1", "sim-attack"
N_FEATURES = 1 << 20
TOKEN_PATTERN = r"(?u)\b\w+\b|[^\w\s]"


def char_ngrams(text: str, lo: int = 2, hi: int = 5) -> list[str]:
    """Every character n-gram, whitespace untouched (as the baselines; sklearn's "char" analyzer collapses whitespace)."""
    return [text[i:i + n] for n in range(lo, hi + 1) for i in range(len(text) - n + 1)]


def vectorizers():
    common = dict(n_features=N_FEATURES, alternate_sign=False, norm=None)
    return [("word", HashingVectorizer(analyzer="word", ngram_range=(1, 2), token_pattern=TOKEN_PATTERN, lowercase=False, **common)),
            ("char", HashingVectorizer(analyzer=partial(char_ngrams), **common))]


def _chunk(texts):
    return sp.hstack([v.transform(texts) for _, v in vectorizers()]).tocsr()


def featurize(texts, n_jobs):
    chunks = [texts[i:i + 500] for i in range(0, len(texts), 500)]
    return sp.vstack(Parallel(n_jobs=n_jobs)(delayed(_chunk)(c) for c in chunks)).tocsr()


def name_columns(X, texts, columns, per_column=8):
    """Each hashed column's n-grams, read back from documents that contain it (collisions show as several)."""
    analyzers = [(k * N_FEATURES, v.build_analyzer()) for k, (_, v) in enumerate(vectorizers())]
    Xc = X[:, columns].tocsc()
    names = {}
    for j, c in enumerate(columns):
        off, analyze = analyzers[c // N_FEATURES]
        found = Counter()
        for d in Xc.indices[Xc.indptr[j]:Xc.indptr[j + 1]][:per_column]:
            found.update(g for g in analyze(texts[d]) if off + abs(murmurhash3_32(g, seed=0)) % N_FEATURES == c)
        names[int(c)] = [g for g, _ in found.most_common(3)]
    return names


def describe(X, texts, coef, is_sim, top):
    order = np.argsort(coef)
    sides = {"toward_sim": order[::-1][:top], "toward_real": order[:top]}
    names = name_columns(X, texts, np.concatenate(list(sides.values())))
    present = (X[:, np.concatenate(list(sides.values()))] > 0).toarray()
    share = {int(c): (float(present[~is_sim, j].mean()), float(present[is_sim, j].mean()))
             for j, c in enumerate(np.concatenate(list(sides.values())))}
    return {side: [{"ngrams": names[int(c)], "weight": round(float(coef[c]), 4), "kind": "word" if c < N_FEATURES else "char",
                    "share_real": round(share[int(c)][0], 4), "share_sim": round(share[int(c)][1], 4)} for c in cs]
            for side, cs in sides.items()}


def show(ngrams):
    return " / ".join("`" + g.replace("\n", "\\n").replace("\t", "\\t").replace("|", "\\|").replace("`", "'") + "`" for g in ngrams)


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config/what_sees.yaml"))
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 7])
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--ridge-alpha", type=float, nargs="+", default=[1.0, 10.0])
    ap.add_argument("--fold-jobs", type=int, default=5)  # each fold copies the matrix (~4 GB)
    ap.add_argument("--n-jobs", type=int, default=-1)
    # Inputs and outputs, for a run away from this checkout (Modal): defaults are the local cache and config.
    ap.add_argument("--joined", default=str(HERE / ".run_cache/joined.npz"))
    ap.add_argument("--staging-outputs", default="")
    ap.add_argument("--folds", default="")
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    ap.add_argument("--stamp", default="")
    ap.add_argument("--provenance", default="", help="JSON captured at launch (a container has no git)")
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    stamp = args.stamp or datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    prov = json.loads(Path(args.provenance).read_text()) if args.provenance else capture(ROOT, [Path(__file__), Path(args.config)])
    T = dict(np.load(args.joined))
    staging = Path(args.staging_outputs or Path(cfg["staging_dir"]).expanduser() / "outputs.jsonl.gz")
    text = {(r["key"], r["step"], r["action"]): r["readout_text"]
            for r in map(json.loads, gzip.open(staging, "rt")) if r["kept"]}
    m = np.isin(T["arm"], [E1, SIM])
    rows = np.flatnonzero(m)
    raw = [text[(T["key"][i], int(T["step"][i]), int(T["action"][i]))] for i in rows]
    is_sim = T["arm"][rows] == SIM
    folds = json.loads(Path(args.folds or ROOT / "splits" / cfg["folds_file"]).read_text())["task_folds"]["main_task_to_fold"]
    task_fold = np.array([folds[t] for t in T["main_task"][rows]])
    out = {"stamp": stamp, "provenance": prov, "args": vars(args), "rows": int(len(rows)), "levels": {}}
    md = [f"# Top features, levels {args.levels} ({stamp})", ""]
    for level in args.levels:
        texts = [scrub(t, level) for t in raw]
        X = featurize(texts, args.n_jobs)
        print(f"L{level}: {X.shape}, {X.nnz:,} nonzeros", flush=True)
        tfidf = TfidfTransformer(sublinear_tf=True).fit(X)
        Xt = tfidf.transform(X)
        clf = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000).fit(Xt, is_sim.astype(int))
        res = {"tfidf": describe(X, texts, clf.coef_[0], is_sim, args.top)}
        # The probe through text: ridge on the same features, out-of-fold R^2 by main-task folds (TF-IDF refit per fold).
        probe = T[f"probe_L{level}_env_real_vs_sim"][rows]
        ok = ~np.isnan(probe)
        r2 = {}
        for alpha in args.ridge_alpha:
            pred = np.full(len(rows), np.nan)

            def fold_fit(f):
                tr, te = ok & (task_fold != f), ok & (task_fold == f)
                tf = TfidfTransformer(sublinear_tf=True).fit(X[tr])
                return te, Ridge(alpha=alpha, solver="sparse_cg").fit(tf.transform(X[tr]), probe[tr]).predict(tf.transform(X[te]))

            for te, p in Parallel(n_jobs=args.fold_jobs)(delayed(fold_fit)(f) for f in sorted(set(task_fold))):
                pred[te] = p
            r2[str(alpha)] = round(float(1 - np.mean((probe[ok] - pred[ok]) ** 2) / np.var(probe[ok])), 4)
            print(f"L{level} ridge alpha {alpha}: out-of-fold R^2 {r2[str(alpha)]}", flush=True)
        best = max(r2, key=r2.get)
        ridge = Ridge(alpha=float(best), solver="sparse_cg").fit(Xt[ok], probe[ok])
        res["probe_through_text"] = {"target": f"probe_L{level}_env_real_vs_sim", "rows": int(ok.sum()), "oof_r2_by_alpha": r2,
                                     "alpha": float(best), "top": describe(X, texts, ridge.coef_, is_sim, args.top)}
        out["levels"][f"L{level}"] = res
        for title, top in (("TF-IDF (logistic regression, real e1 vs simulated)", res["tfidf"]),
                           (f"Probe read through text (ridge on the probe's held-out-environment log-odds; out-of-fold R^2 "
                            f"{r2[best]} at alpha {best})", res["probe_through_text"]["top"])):
            md += [f"## Level {level}: {title}", ""]
            for side in ("toward_sim", "toward_real"):
                md += [f"**{side.replace('_', ' ')}**", "", "| # | n-grams in the hashed column | kind | weight | share of real outputs | share of simulated |",
                       "|---|---|---|---|---|---|"]
                md += [f"| {i + 1} | {show(f['ngrams'])} | {f['kind']} | {f['weight']:+.3f} | {f['share_real']:.3f} | {f['share_sim']:.3f} |"
                       for i, f in enumerate(top[side])]
                md.append("")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"top_features_{'_'.join(f'L{lv}' for lv in args.levels)}_{stamp}"
    (out_dir / f"{name}.json").write_text(json.dumps(out, indent=1))
    (out_dir / f"{name}.md").write_text("\n".join(md))
    print(f"wrote {out_dir}/{name}.json and .md")
    return [out_dir / f"{name}.json", out_dir / f"{name}.md"]


if __name__ == "__main__":
    main()
