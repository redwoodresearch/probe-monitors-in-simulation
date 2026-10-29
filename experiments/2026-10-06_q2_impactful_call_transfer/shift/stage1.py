"""Stage 1 of the real->sim shift analysis: is the shift a ranking change or an offset?

Per reader, readout, view and training domain (headline probes, saved test scores, frozen 1% thresholds):
  - AUROC on real and on sim (ranking);
  - median margin (score minus the fold's frozen threshold, in logits) of negatives and of positives on each domain,
    and the separation (positive median minus negative median). Offset: both medians move together, separation and
    AUROC stay. Separation change: not fixable by moving the threshold;
  - recall and FPR at the frozen threshold, and recall with the threshold recalibrated on each domain's own negatives
    (per fold, in sample: the most a perfect recalibration could recover).
95% intervals resample main tasks (1000 draws, seed 0, the same draws as two_by_two), every quantity on the same draws.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from probe_shift.transfer import _weighted_metrics, threshold_at_fpr  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent))
from paths import DATA, RESULTS, ROOT, stamp as now  # noqa: E402

HERE = Path(__file__).parent
N_BOOT, SEED, FPR, CHUNK = 1000, 0, 0.01, 100
TF_TRAIN = {"real-attack-e1": "real", "sim-attack": "sim"}  # TF-IDF training sets by arm; the real+honest one is left out
KEEP = None  # optional set of (arm, key): attack runs to keep (e.g. attempted only); other arms untouched


def task_counts(groups):
    uniq, inv = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(SEED)
    counts = np.stack([np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)) for _ in range(N_BOOT)])
    return counts, inv


def wmedian(x, W):
    """Weighted lower median per row of W."""
    o = np.argsort(x, kind="mergesort")
    c = np.cumsum(W[:, o], 1)
    i = (c < c[:, -1:] / 2).sum(1)
    return x[o][np.minimum(i, len(x) - 1)]


def wthreshold(neg, W):
    """threshold_at_fpr with rows counted W times: the (k+1)-th largest score, k = floor(fpr * n)."""
    o = np.argsort(-neg, kind="mergesort")
    c = np.cumsum(W[:, o], 1)
    k = np.floor(FPR * c[:, -1] + 1e-9)
    return neg[o][np.minimum((c <= k[:, None]).sum(1), len(neg) - 1)]


def domain_stats(d, W):
    """Every Stage 1 quantity for one domain's rows, per draw (rows of W)."""
    y, s, flag, m, fold = d.y.to_numpy(), d.score.to_numpy(np.float64), d.flag.to_numpy(float), d.margin.to_numpy(), d.fold.to_numpy()
    out = _weighted_metrics(y, s, flag, W)
    pos, neg = y == 1, y == 0
    out["med_neg"], out["med_pos"] = wmedian(m[neg], W[:, neg]), wmedian(m[pos], W[:, pos])
    out["separation"] = out["med_pos"] - out["med_neg"]
    hit, tot, shift = np.zeros(len(W)), np.zeros(len(W)), []
    for f in np.unique(fold):  # recalibrated: each fold's model gets the 1% threshold of its own test negatives here
        nf, pf = neg & (fold == f), pos & (fold == f)
        t = wthreshold(s[nf], W[:, nf])
        hit += (W[:, pf] * (s[pf][None, :] > t[:, None])).sum(1)
        tot += W[:, pf].sum(1)
        shift.append(t - d.threshold.to_numpy()[nf][0])  # this domain's 1% threshold minus the frozen one
    out["recall_recal"] = hit / tot
    out["threshold_shift"] = np.mean(shift, 0)  # mean over folds, in logits: the negatives' tail offset
    return out


def analyze(table: Path, train: str):
    df = pd.read_json(table, lines=True)
    if "command_text" in table.name:
        df = df[df.train_set.isin(TF_TRAIN)].assign(train_set=lambda x: x.train_set.map(TF_TRAIN))
    df = df[df.train_set == train].reset_index(drop=True)
    if KEEP is not None:
        attack = df.arm.isin(["real-attack-e1", "sim-attack"])
        df = df[~attack | pd.Series(list(zip(df.arm, df.key))).isin(KEEP).to_numpy()].reset_index(drop=True)
    df["margin"] = df.score - df.threshold
    counts, inv = task_counts(df.main_task.to_numpy())
    W_all = np.vstack([np.ones((1, len(counts[0]))), counts])  # row 0: the point estimate
    per = {"real": [], "sim": []}
    for k in range(0, len(W_all), CHUNK):
        Wc = W_all[k:k + CHUNK][:, inv].astype(float)
        for dom in per:
            i = np.flatnonzero(df.domain == dom)
            per[dom].append(domain_stats(df.iloc[i], Wc[:, i]))
    res = {dom: {m: np.concatenate([c[m] for c in chunks]) for m in chunks[0]} for dom, chunks in per.items()}
    res["sim minus real"] = {m: res["sim"][m] - res["real"][m] for m in res["real"]}
    report = {dom: {m: {"value": round(float(v[0]), 4), "ci95": [round(float(x), 4) for x in np.nanpercentile(v[1:], [2.5, 97.5])]}
                    for m, v in r.items()} for dom, r in res.items()}
    for dom in ("real", "sim"):  # the in-sample recalibrated threshold, checked against the project's own rule at the point
        d = df[df.domain == dom]
        point = np.mean([(d[(d.fold == f) & (d.y == 1)].score > threshold_at_fpr(d[(d.fold == f) & (d.y == 0)].score.to_numpy(), FPR)).sum()
                         for f in sorted(d.fold.unique())]) * d.fold.nunique() / (d.y == 1).sum()
        assert abs(point - report[dom]["recall_recal"]["value"]) < 1e-4, (table, train, dom, point)
        report[dom]["n"] = {"impactful": int((d.y == 1).sum()), "read_only": int((d.y == 0).sum()), "main_tasks": int(d.main_task.nunique())}
    reader, readout, view = (["Qwen3-32B", "command_text", "attack_only"] if "command_text" in table.name
                             else table.name.removesuffix(".jsonl.gz").split("__"))
    return {"reader": {"qwen3-32b": "Qwen3-32B", "qwen3.5": "Qwen3.5"}.get(reader, reader), "readout": readout, "view": view,
            "train": train, **report}


if __name__ == "__main__":
    tables = sorted((DATA / "tables").glob("*.jsonl.gz"))
    tables = [t for t in tables if len(sys.argv) < 2 or any(a in t.name for a in sys.argv[1:])]
    rows = Parallel(n_jobs=-1)(delayed(analyze)(t, tr) for t in tables for tr in ("real", "sim"))
    stamp = pd.Timestamp.now(tz="America/Los_Angeles").strftime("%Y-%m-%d_%H%M%S")
    out = RESULTS / f"shift_stage1_{stamp}.json"
    out.write_text(json.dumps({"n_boot": N_BOOT, "seed": SEED, "fpr": FPR, "tables": [t.name for t in tables], "rows": rows}, indent=1))
    print(out)
