"""Author-split character n-gram classifier, fix-full simulated honest vs real honest (RULE.md in this folder; POST HOC,
descriptive). Numbers and n-grams of at most 5 characters only; no reply or command text is written or printed.

Steps: build the real (overlap800) and original-n437 rows locally from the eval logs with the v4 staging's code; check
them against q2's fix-full rows (non-tool messages byte-identical; command identity count); then, for one split
(replies or commands, clock masked at scrub level 1), fit fake vs real, the same-batch control and the marker control.

  AUTHOR_SPLIT_STAGE=<scratch dir outside git> uv run --no-project --python 3.14 --with numpy --with scipy \\
      --with scikit-learn --with joblib --with boto3 --with pyyaml python -I \\
      experiments/2026-10-13_claimB_author_split/analyze_POSTHOC_author_split.py <fx-full shard_12.jsonl> <replies|commands>
Python 3.14: the eval logs' Zstandard ZIP entries need it. python -I ignores PYTHONPATH, so the script puts the repo's
src and simulated-control-evals' src on sys.path itself.
"""

import hashlib
import json
import os
import random
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "src"), str(Path.home() / "simulated-control-evals" / "src")]

import numpy as np
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from probe_shift.scrub import scrub
from probe_shift.transcripts import (
    file_samples,
    main_task_id,
    reader_messages,
    system_prompt,
)

FX_SHA = "94ae36d6d3bbd420d607254fd7c389910aee80b512d2281e19d2f08ae693b799"  # stage_2026-10-13_142154.json @ 82909df0
REAL, SIM = "la_public10_real_honest_n1000", "la_public10_sim_astra_honest_n437"
SEP, SEED, SHUFFLES, TOP, NMAX = "\x1e\n", 20261013, 100, 20, 5
MARKER = "gfpdkvlgzrrwtgvhfrtbzkbg"  # the earlier thread's marker (aswin/claimb-fake-reply-diffs, Amendment 3)
assert (
    "".join(
        random.Random("fake-reply-diffs-marker").choices("bcdfghjklmnpqrstvwxz", k=24)
    )
    == MARKER
)
N_PLANT = 131  # 30% of 437, RULE.md
STAGE_DIR = Path(
    os.environ["AUTHOR_SPLIT_STAGE"]
)  # rebuilt rows hold raw run text: a scratch dir outside git, deleted after


def plant(t: str) -> str:
    """The marker on its own line after the document's last non-newline character; trailing newlines stay."""
    body = t.rstrip("\n")
    return body + "\n" + MARKER + t[len(body) :]


# ---------------------------------------------------------------- data
def cohort_rows():
    ex = json.loads((REPO / "splits" / "exclusions_v4.json").read_text())
    ok = [r for r in ex["samples"] if not r["reason"]]
    sim = {r["key"]: r for r in ok if r["arm"] == "sim-honest"}
    real = sorted(
        (
            r
            for r in ok
            if r["arm"] == "real-honest" and r["key"].startswith("overlap800_")
        ),
        key=lambda r: r["key"],
    )
    by = {}
    for r in real:
        by.setdefault(r["cell"], []).append(r)
    tasks = sorted({r["cell"] for r in sim.values()})
    first = {t: by[t][0] for t in tasks}
    second = {t: by[t][1] for t in tasks if len(by[t]) > 1}
    return (
        sim,
        first,
        second,
        hashlib.sha256(
            (REPO / "splits" / "exclusions_v4.json").read_bytes()
        ).hexdigest(),
    )


def build(alias, rows, prompts=None):
    """Reader messages for the given exclusions rows, as the v4 staging builds them (key, honest system prompt)."""
    files = sorted({r["file"] for r in rows})
    want = {r["key"]: r for r in rows}
    with ThreadPoolExecutor(16) as pool:
        loaded = list(pool.map(lambda f: (f, file_samples(alias, f)), files))
    out = {}
    for f, samples in loaded:
        for s in samples:
            key = (
                f"{f.removesuffix('.eval')}__{s['id'].replace('/', '_')}__{s['epoch']}"
            )
            if key in want:
                p = system_prompt(s) if prompts is None else prompts[main_task_id(s)]
                out[key] = {
                    "key": key,
                    "cell": want[key]["cell"],
                    "messages": reader_messages(s, p),
                    "prompt": p,
                }
    missing = sorted(set(want) - set(out))
    if missing:
        raise SystemExit(f"{len(missing)} {alias} runs not found in their eval files")
    return out


def stage(fx_path):
    sim, first, second, ex_sha = cohort_rows()
    code = hashlib.sha256(
        (REPO / "src/probe_shift/transcripts.py").read_bytes()
        + Path(__file__).read_bytes()
    ).hexdigest()
    cache = (
        STAGE_DIR / f"stage_{ex_sha[:12]}_{code[:12]}.json"
    )  # rebuilt rows, keyed by the exclusions and the code
    if cache.exists():
        st = json.loads(cache.read_text())
    else:
        real = build(REAL, list(first.values()) + list(second.values()))
        prompts = {}
        for r in real.values():
            prompts.setdefault(r["cell"], set()).add(r["prompt"])
        if any(len(v) != 1 for v in prompts.values()):
            raise SystemExit("a main task has more than one honest system prompt")
        prompts = {k: v.pop() for k, v in prompts.items()}
        orig = build(SIM, list(sim.values()), prompts)
        st = {
            "real": {k: v["messages"] for k, v in real.items()},
            "orig": {k: v["messages"] for k, v in orig.items()},
        }
        STAGE_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(st))
    raw = Path(fx_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != FX_SHA:
        raise SystemExit("the fix-full shard's sha256 does not match the stage record")
    fx = {r["key"]: r for r in map(json.loads, raw.decode().splitlines())}
    if set(fx) != set(sim) or any(
        r["arm"] != "sim-honest-fx-full" for r in fx.values()
    ):
        raise SystemExit("the fix-full rows are not the 437 simulated honest runs")
    return sim, first, second, ex_sha, st, fx


def canon(m):
    return json.dumps(m, sort_keys=True, ensure_ascii=False)


def command_text(msgs, masked):
    def args(v):
        if isinstance(v, str):
            return scrub(v, 1) if masked else v
        if isinstance(v, list):
            return [args(x) for x in v]
        if isinstance(v, dict):
            return {k: args(x) for k, x in v.items()}
        return v  # numbers, booleans, None: untouched at level 1

    return SEP.join(
        f"{c['function']['name']} {json.dumps(args(c['function']['arguments']), sort_keys=True, ensure_ascii=False)}"
        for m in msgs
        if m["role"] == "assistant"
        for c in (m.get("tool_calls") or [])
    )


def reply_text(msgs):
    outs = [m["content"] for m in msgs if m["role"] == "tool"]
    if not all(isinstance(t, str) for t in outs):
        raise SystemExit("a tool message's content is not a string")
    return SEP.join(scrub(t, 1) for t in outs)


def format_check(st, fx):
    c = Counter()
    for k, row in fx.items():
        a, b = st["orig"][k], row["messages"]
        if len(a) != len(b) or any(x["role"] != y["role"] for x, y in zip(a, b)):
            c["runs with a different message structure"] += 1
            continue
        nontool = all(canon(x) == canon(y) for x, y in zip(a, b) if x["role"] != "tool")
        c["runs with every non-tool message byte-identical"] += nontool
        c["runs with a non-tool message differing"] += not nontool
        c["runs with byte-identical command text (unmasked)"] += command_text(
            a, False
        ) == command_text(b, False)
        for x, y in zip(a, b):
            if x["role"] == "tool":
                c["tool replies byte-identical"] += x["content"] == y["content"]
                c["tool replies changed by fix-full"] += x["content"] != y["content"]
        c["runs with at least one reply changed"] += any(
            x["content"] != y["content"] for x, y in zip(a, b) if x["role"] == "tool"
        )
    return dict(c)


# ---------------------------------------------------------------- n-grams (exact integer codes, no hashing)
BITS = 12  # each character is coded by its index in the comparison's alphabet (+1); 5 x 12 bits + the length on top


def alphabet(texts):
    """Sorted code points of every character in the comparison's documents and the marker; at most 4,095 (fail loudly)."""
    chars = set(MARKER).union(*map(set, texts))
    if len(chars) >= 2**BITS:
        raise SystemExit(
            f"{len(chars)} distinct characters: more than {2**BITS - 1} cannot be coded exactly"
        )
    return np.array(sorted(map(ord, chars)), dtype=np.uint64)


def gram_codes(text: str, alpha):
    """[(n, exact uint64 code per start position)] for n = 1..5: code = n << 60 | sum_k (index+1 of char k) << 12k."""
    cp = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32).astype(np.uint64)
    idx = np.searchsorted(alpha, cp).astype(np.uint64) + np.uint64(1)
    out, code = [], np.zeros(len(idx), dtype=np.uint64)
    for n in range(1, NMAX + 1):
        if len(idx) < n:
            break
        code = code[: len(idx) - n + 1] | (idx[n - 1 :] << np.uint64(BITS * (n - 1)))
        out.append((n, code | (np.uint64(n) << np.uint64(60))))
    return out


def decode(code: int, alpha) -> str:
    n = code >> 60
    return "".join(
        chr(int(alpha[((code >> (BITS * k)) & (2**BITS - 1)) - 1])) for k in range(n)
    )


def doc_counts(text: str, alpha):
    cs = [c for _, c in gram_codes(text, alpha)]
    return np.unique(
        np.concatenate(cs) if cs else np.array([], np.uint64), return_counts=True
    )


def matrix(texts, alpha):
    per = Parallel(n_jobs=-1)(delayed(doc_counts)(t, alpha) for t in texts)
    allu, df = np.unique(np.concatenate([u for u, _ in per]), return_counts=True)
    vocab = allu[
        df >= 2
    ]  # label-free document frequency >= 2 over the comparison's documents (RULE.md)
    if not len(vocab):
        raise SystemExit("empty vocabulary")
    rows, cols, vals = [], [], []
    for i, (u, cnt) in enumerate(per):
        j = np.searchsorted(vocab, u)
        keep = (j < len(vocab)) & (vocab[np.minimum(j, len(vocab) - 1)] == u)
        rows.append(np.full(int(keep.sum()), i))
        cols.append(j[keep])
        vals.append(cnt[keep].astype(np.float64))
    X = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(texts), len(vocab)),
    )
    return X, vocab


# ---------------------------------------------------------------- classifier
def fold_mats(X, folds):
    """Label-independent per fold: training-seen columns, IDF fit on training rows."""
    out = []
    for tr, te in folds:
        seen = np.flatnonzero(np.asarray(X[tr].sum(0)).ravel() > 0)
        Xs = X[:, seen]
        tf = TfidfTransformer(sublinear_tf=True).fit(Xs[tr])
        out.append((tf.transform(Xs[tr]).tocsr(), tf.transform(Xs[te]).tocsr(), tr, te))
    return out


def cv_scores(mats, y):
    s, per = np.full(len(y), np.nan), []
    for Xtr, Xte, tr, te in mats:
        clf = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000).fit(
            Xtr, y[tr]
        )
        s[te] = clf.decision_function(Xte)
        per.append(float(roc_auc_score(y[te], s[te])) if len(set(y[te])) == 2 else None)
    assert not np.isnan(s).any()
    return float(roc_auc_score(y, s)), per


def classify(pos, neg, groups, names, extra_feats=None):
    """Label 1 = `pos`, 0 = `neg`; row i of each is the same task; CV grouped by task; within-task shuffle null."""
    texts, y, g = (
        pos + neg,
        np.r_[np.ones(len(pos), int), np.zeros(len(neg), int)],
        np.r_[groups, groups],
    )
    alpha = alphabet(texts)
    X, vocab = matrix(texts, alpha)
    say(names, "matrix", X.shape, "nnz", X.nnz)
    folds = list(GroupKFold(n_splits=5).split(X, y, g))
    mats = fold_mats(X, folds)
    obs, per = cv_scores(mats, y)
    say(names, "observed fit done; null refits start")
    rng = np.random.default_rng(SEED)
    shuffled = []
    for _ in range(SHUFFLES):
        yy = y.copy()
        for task in sorted(set(g)):
            idx = np.flatnonzero(g == task)
            yy[idx] = y[idx][rng.permutation(len(idx))]
        shuffled.append(yy)
    null = np.array(
        [
            a
            for a, _ in Parallel(n_jobs=-1)(
                delayed(cv_scores)(mats, yy) for yy in shuffled
            )
        ]
    )
    tf = TfidfTransformer(sublinear_tf=True).fit(X)
    coef = (
        LogisticRegression(C=1.0, solver="liblinear", max_iter=1000)
        .fit(tf.transform(X), y)
        .coef_[0]
    )
    order = np.argsort(coef, kind="stable")
    top_pos, top_neg = order[::-1][:TOP], order[:TOP]
    names_of = {j: decode(int(vocab[j]), alpha) for j in np.r_[top_pos, top_neg]}
    cnt = {
        1: np.asarray(X[y == 1].sum(0)).ravel(),
        0: np.asarray(X[y == 0].sum(0)).ravel(),
    }
    docs = {
        1: np.asarray((X[y == 1] > 0).sum(0)).ravel(),
        0: np.asarray((X[y == 0] > 0).sum(0)).ravel(),
    }

    def feat(j):
        g_ = names_of[j]
        assert len(g_) <= NMAX
        return {
            "ngram": repr(g_),
            "coef": round(float(coef[j]), 6),
            f"count_{names[0]}": int(cnt[1][j]),
            f"count_{names[1]}": int(cnt[0][j]),
            f"runs_{names[0]}": int(docs[1][j]),
            f"runs_{names[1]}": int(docs[0][j]),
        }

    res = {
        "rows": len(y),
        "positives": names[0],
        "negatives": names[1],
        "folds": 5,
        "vocabulary_df2": len(vocab),
        "auroc": round(obs, 4),
        "fold_aurocs": [None if a is None else round(a, 4) for a in per],
        "null_mean": round(float(null.mean()), 4),
        "null_p95": round(float(np.percentile(null, 95)), 4),
        "null_max": round(float(null.max()), 4),
        "p": round(float((1 + (null >= obs).sum()) / (SHUFFLES + 1)), 4),
        "above_null_p95": bool(obs > np.percentile(null, 95)),
        "null_draws": [round(float(v), 4) for v in null],
        f"top_toward_{names[0]}": [feat(j) for j in top_pos],
        f"top_toward_{names[1]}": [feat(j) for j in top_neg],
    }
    if extra_feats is not None:
        res |= extra_feats(vocab, order, alpha)
    return res


def marker_control(real_docs, tasks, first):
    """RULE.md: the marker in 131 real runs drawn from their sorted run keys; planted vs unplanted twins."""
    chosen = set(
        random.Random("author-split-marker").sample(
            sorted(first[t]["key"] for t in tasks), N_PLANT
        )
    )
    unplanted = [real_docs[t] for t in tasks]
    planted = [
        plant(real_docs[t]) if first[t]["key"] in chosen else real_docs[t]
        for t in tasks
    ]
    alpha = alphabet(
        planted + unplanted
    )  # the same alphabet classify() builds from these documents
    present = np.unique(
        np.concatenate(
            [
                u
                for u, _ in Parallel(n_jobs=-1)(
                    delayed(doc_counts)(t, alpha) for t in unplanted
                )
            ]
        )
    )
    mk = np.unique(
        np.concatenate([c for _, c in gram_codes(MARKER, alpha)])
    )  # the marker's own n-grams (1-5)
    unique = {
        int(c) for c in mk[~np.isin(mk, present)]
    }  # 0 occurrences in the unplanted documents, before planting

    def extra(vocab, order, alpha_):
        assert np.array_equal(alpha_, alpha)
        ranks = [
            i + 1 for i, j in enumerate(order[::-1][:5000]) if int(vocab[j]) in unique
        ]
        first_rank = ranks[0] if ranks else None
        return {
            "marker_distinct_ngrams": len(mk),
            "marker_unique_ngrams_before_planting": len(unique),
            "first_marker_unique_rank": first_rank,
            "marker_unique_in_top5": bool(first_rank is not None and first_rank <= 5),
            "top_is_marker_unique": first_rank == 1,
        }

    res = classify(planted, unplanted, np.array(tasks), ("planted", "unplanted"), extra)
    res |= {
        "planted_runs": len(chosen),
        "pass_auroc": res["above_null_p95"],
        "pass": bool(res["above_null_p95"] and res["marker_unique_in_top5"]),
    }
    return res


# ---------------------------------------------------------------- main
def say(*a):
    print(
        datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%H:%M:%S"),
        *a,
        flush=True,
    )


def analyze(split, sim, first, second, ex_sha, st, fx, git, source):
    """Everything after the rows are in hand; `st` holds the original and real messages, `fx` the fix-full rows."""
    if split not in ("replies", "commands"):
        raise SystemExit("split must be replies or commands")
    t0 = datetime.now(ZoneInfo("America/Los_Angeles"))
    check = format_check(st, fx)
    say(split, "format check", json.dumps(check))
    out = {
        "label": "POST HOC, descriptive, preliminary until reviewed",
        "rule": "RULE.md (this folder)",
        "split": split,
        "git": git,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows_source": source,
        "inputs": {"fx_full_shard_sha256": FX_SHA, "exclusions_v4_sha256": ex_sha},
        "cohorts": {
            "fix_full": len(fx),
            "real_overlap800_first": len(first),
            "real_overlap800_second": len(second),
        },
        "format_check": check,
    }
    if check.get("runs with every non-tool message byte-identical") != len(fx):
        out["stopped"] = (
            "the original's non-tool messages differ from fix-full: the rows don't match the staging"
        )
    else:
        doc = (lambda m: command_text(m, True)) if split == "commands" else reply_text
        tasks = sorted(first)
        key_of = {r["cell"]: k for k, r in sim.items()}
        fake = [doc(fx[key_of[t]]["messages"]) for t in tasks]
        real = {t: doc(st["real"][first[t]["key"]]) for t in tasks}
        lens = {
            "fix_full": [len(x) for x in fake],
            "real": [len(real[t]) for t in tasks],
        }
        out["doc_chars"] = {
            k: {
                "median": float(np.median(v)),
                "mean": round(float(np.mean(v)), 1),
                "total": int(sum(v)),
            }
            for k, v in lens.items()
        }
        out["fake_vs_real"] = classify(
            fake, [real[t] for t in tasks], np.array(tasks), ("fix_full", "real")
        )
        say(
            split,
            "fake_vs_real",
            {k: out["fake_vs_real"][k] for k in ("auroc", "null_p95", "p")},
        )
        ct = sorted(second)
        out["control"] = classify(
            [doc(st["real"][second[t]["key"]]) for t in ct],
            [real[t] for t in ct],
            np.array(ct),
            ("real_second", "real_first"),
        )
        say(
            split, "control", {k: out["control"][k] for k in ("auroc", "null_p95", "p")}
        )
        out["positive_control"] = marker_control(real, tasks, first)
        say(
            split,
            "positive_control",
            {
                k: out["positive_control"][k]
                for k in ("auroc", "null_p95", "first_marker_unique_rank", "pass")
            },
        )
        out["feature_lists_readable"] = out["positive_control"]["pass"]
    out["elapsed_s"] = round(
        (datetime.now(ZoneInfo("America/Los_Angeles")) - t0).total_seconds(), 1
    )
    return out, f"author_split_{split}_POSTHOC_{t0.strftime('%Y-%m-%d_%H%M%S')}.json"


def main(fx_path, split):
    """Local path (RULE.md, first draft): rows rebuilt from the eval logs. The run itself is on Modal (Amendment 1)."""
    sim, first, second, ex_sha, st, fx = stage(fx_path)
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=HERE,
    ).stdout.strip()
    out, name = analyze(
        split, sim, first, second, ex_sha, st, fx, git, "eval logs, rebuilt locally"
    )
    path = HERE / "results" / name
    path.parent.mkdir(exist_ok=True)
    with open(path, "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:3])
