"""Cleaned-text test: the real->sim pre_call gap at text levels 0, 4 and 7, on the same calls and layers.

Per level: Qwen3-32B fits at layers 9, 16, 17 (level 0 from alllayers/, levels 4 and 7 from scrub4/ and scrub7/), the
headline layer per outer fold chosen as everywhere in Q2 (earliest layer within layer_tolerance of the best validation
AUROC, here among these three layers), scored on the calls every level kept. Stage 1's quantities per level and
training set (AUROC, recall and FPR at the frozen threshold, recall recalibrated on the test domain), and the change in
each gap from level 0, all on the same task-resampled draws (1000, seed 0) so level differences get paired intervals.
Inputs (downloaded exact files) under levels/: L{level}/manifest.json, L{level}/fits_index.jsonl, L{level}/fits/*.npz.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).parent
from stage1 import CHUNK, N_BOOT, domain_stats, task_counts  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent))
from paths import DATA, RESULTS, ROOT, stamp as now  # noqa: E402

Q2 = ROOT / "experiments/2026-10-06_q2_impactful_call_transfer"
TOL = yaml.safe_load((Q2 / "config" / "q2_probes.yaml").read_text())["layer_tolerance"]
LAYERS, LEVELS, POSITIONS = [9, 16, 17], [0, 4, 5, 6, 7], ["pre_call", "call_end", "prev_content"]
TRAIN = {"real-attack-e1": "real", "sim-attack": "sim"}
K = ["arm", "key", "step", "action"]
KEEP = None  # optional set of (arm, key) attack runs to keep


def level_table(level: int, position: str) -> pd.DataFrame:
    d = DATA / "levels" / f"L{level}"
    man = pd.DataFrame(json.loads((d / "manifest.json").read_text()))
    test = man[man.arm.isin(TRAIN)].reset_index(drop=True)
    idx = [json.loads(l) for l in (d / "fits_index.jsonl").read_text().splitlines()]
    fits = {(f["position"], f["train_set"], f["layer"]): f for f in idx
            if f["seed"] is None and f["layer"] in LAYERS and f.get("view", "") == ""}
    parts = []
    for t in TRAIN:
        score, flag, thr, pick = np.zeros(len(test)), np.zeros(len(test), bool), np.zeros(len(test)), np.zeros(len(test), int)
        for f in sorted(test.fold.unique()):
            val = {l: next(fo["val_auroc"] for fo in fits[(position, t, l)]["folds"] if fo["fold"] == f) for l in LAYERS}
            p = min(l for l in LAYERS if val[l] >= max(val.values()) - TOL)
            a = np.load(d / "fits" / fits[(position, t, p)]["file"])
            if len(a["score"]) != len(test):
                raise ValueError(f"level {level} {position} {t} L{p}: {len(a['score'])} scores for {len(test)} test rows")
            m = (test.fold == f).to_numpy()
            score[m], flag[m], thr[m], pick[m] = a["score"][m], a["flag"][m], a["threshold"][m], p
        parts.append(test[K + ["main_task", "fold", "y"]].assign(domain=test.arm.map(TRAIN), train_set=TRAIN[t], score=score,
                                                                   flag=flag, threshold=thr, layer=pick))
    return pd.concat(parts, ignore_index=True)


def main(tag: str = "") -> Path:
    out = {}
    for position in POSITIONS:
        tabs = {lv: level_table(lv, position) for lv in LEVELS}
        common = set.intersection(*[set(map(tuple, t[K].drop_duplicates().to_numpy())) for t in tabs.values()])
        if KEEP is not None:
            common = {c for c in common if (c[0], c[1]) in KEEP}
        kept = {lv: len(t[K].drop_duplicates()) for lv, t in tabs.items()}
        out[position] = {"calls": {"common": len(common), **{f"level {lv}": n for lv, n in kept.items()}}}
        for train in ("real", "sim"):
            res = {}
            for lv, t in tabs.items():
                t = t[(t.train_set == train) & pd.Series(list(map(tuple, t[K].to_numpy())), index=t.index).isin(common)]
                res[lv] = t.sort_values(K).reset_index(drop=True)
            groups = res[0].main_task.to_numpy()
            counts, inv = task_counts(groups)
            W_all = np.vstack([np.ones((1, counts.shape[1])), counts])
            draws = {lv: {"real": [], "sim": []} for lv in LEVELS}
            for k in range(0, len(W_all), CHUNK):
                Wc = W_all[k:k + CHUNK][:, inv].astype(float)
                for lv, t in res.items():
                    t = t.assign(margin=t.score - t.threshold)
                    for dom in ("real", "sim"):
                        i = np.flatnonzero(t.domain == dom)
                        draws[lv][dom].append(domain_stats(t.iloc[i], Wc[:, i]))
            v = {lv: {dom: {m: np.concatenate([c[m] for c in ch]) for m in ch[0]} for dom, ch in per.items()} for lv, per in draws.items()}
            gap = {lv: {m: v[lv]["real"][m] - v[lv]["sim"][m] for m in ("auroc", "recall", "fpr", "recall_recal")} for lv in LEVELS}
            s = lambda x: {"value": round(float(x[0]), 4), "ci95": [round(float(q), 4) for q in np.nanpercentile(x[1:], [2.5, 97.5])]}
            out[position][f"train {train}"] = {
                f"level {lv}": {"real": {m: s(v[lv]["real"][m]) for m in ("auroc", "recall", "fpr", "recall_recal")},
                                "sim": {m: s(v[lv]["sim"][m]) for m in ("auroc", "recall", "fpr", "recall_recal")},
                                "gap real minus sim": {m: s(g) for m, g in gap[lv].items()},
                                **({f"gap change from level 0": {m: s(gap[lv][m] - gap[0][m]) for m in gap[lv]}} if lv else {}),
                                "layers": res[lv].groupby("fold").layer.first().to_dict()}
                for lv in LEVELS}
    stamp = pd.Timestamp.now(tz="America/Los_Angeles").strftime("%Y-%m-%d_%H%M%S")
    path = RESULTS / f"shift_compare_levels{tag}_{stamp}.json"
    path.write_text(json.dumps({"n_boot": N_BOOT, "layers": LAYERS, "layer_tolerance": TOL, **out}, indent=1, default=str))
    print(path)
    return path


if __name__ == "__main__":
    main()
