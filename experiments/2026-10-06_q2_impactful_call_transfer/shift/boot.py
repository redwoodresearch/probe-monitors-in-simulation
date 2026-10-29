"""Shared: recall, FPR and AUROC per subset for real and sim calls, with one set of task-resampled draws per table."""
import sys
from collections import defaultdict

import numpy as np

from probe_shift.transfer import _weighted_metrics  # noqa: E402
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from paths import DATA, RESULTS, ROOT, stamp as now  # noqa: E402

N_BOOT, SEED, CHUNK = 1000, 0, 100


def report(df, subs: dict) -> dict:
    """subs: name -> (real rows mask, sim rows mask). Gap = real minus sim. 95% intervals from 1000 main-task draws."""
    uniq, inv = np.unique(df.main_task, return_inverse=True)
    rng = np.random.default_rng(SEED)
    counts = np.stack([np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)) for _ in range(N_BOOT)])
    W_all = np.vstack([np.ones((1, len(uniq))), counts])
    y, s, fl = df.y.to_numpy(), df.score.to_numpy(np.float64), df.flag.to_numpy(float)
    acc = defaultdict(lambda: defaultdict(list))
    for k in range(0, len(W_all), CHUNK):
        Wc = W_all[k:k + CHUNK][:, inv].astype(float)
        for name, (mr, ms) in subs.items():
            for d, m in (("real", mr), ("sim", ms)):
                i = np.flatnonzero(m)
                acc[name][d].append(_weighted_metrics(y[i], s[i], fl[i], Wc[:, i]))
    rep = {}
    for name, per in acc.items():
        res = {d: {met: np.concatenate([c[met] for c in chunks]) for met in chunks[0]} for d, chunks in per.items()}
        res["real minus sim"] = {met: res["real"][met] - res["sim"][met] for met in res["real"]}
        with np.errstate(all="ignore"):
            rep[name] = {d: {met: {"value": round(float(v[0]), 4),
                                   "ci95": [round(float(x), 4) for x in np.nanpercentile(v[1:], [2.5, 97.5])] if np.isfinite(v[1:]).any() else None}
                             for met, v in r.items()} for d, r in res.items()}
        for d, m in zip(("real", "sim"), subs[name]):
            rep[name][d]["n"] = {"impactful": int(y[m].sum()), "read_only": int((y[m] == 0).sum())}
    return rep


def fmt(rep: dict) -> str:
    f = lambda v: "nan" if v["value"] != v["value"] else (f"{v['value']:+.3f} [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]" if v["ci95"] else f"{v['value']:+.3f}")
    out = []
    for name, s in rep.items():
        R, S, G = s["real"], s["sim"], s["real minus sim"]
        out.append(f"{name[:58]:58} n imp/ro real {R['n']['impactful']:5}/{R['n']['read_only']:4} sim {S['n']['impactful']:5}/{S['n']['read_only']:4} | "
                   f"recall {R['recall']['value']:.3f}/{S['recall']['value']:.3f} gap {f(G['recall']):>24} | FPR {R['fpr']['value']:.3f}/{S['fpr']['value']:.3f} | "
                   f"AUROC {R['auroc']['value']:.3f}/{S['auroc']['value']:.3f} gap {f(G['auroc']):>24}")
    return "\n".join(out)
