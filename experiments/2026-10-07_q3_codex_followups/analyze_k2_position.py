"""K2: is Q2's "readable at the command, barely before it" (impactful-call recall 0.86–0.89 at call_end vs 0.22 at
pre_call) a property of identical actions, or of which calls each readout covers? q2's saved out-of-fold per-call
tables (64-layer run, per-fold headline layers) for call_end, pre_call and prev_content, inner-joined on
(arm, key, step, action) so every readout scores the same calls. Stratified by tool and by the call_end token kind
(content vs a timeout digit, view range, empty string or escaped newline; q2's base manifest). $0, no model calls.

Per stratum and readout: recall on impactful calls and false alarms on read-only calls at each table's own frozen
threshold (column `flag`), and AUROC. Real→real (train real, test real) and real→sim (train real, test simulated).

  python analyze_k2_position.py --tables <dir with qwen3-32b__{readout}__attack_only.jsonl.gz and base_manifest.json>
"""

import argparse
import gzip
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
READOUTS = ("call_end", "pre_call", "prev_content")


def auroc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, bool)
    if y.all() or not y.any():
        return None
    order = np.argsort(s, kind="mergesort")
    ss, r, i = s[order], np.empty(len(s)), 0
    while i < len(ss):  # average ranks over ties
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        r[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True)
    args = ap.parse_args()
    d = Path(args.tables)
    key = lambda r: (r["arm"], r["key"], r["step"], r["action"])  # noqa: E731
    manifest = {key(r): r for r in json.loads((d / "base_manifest.json").read_text())}
    tab = {}
    for ro in READOUTS:
        f = d / f"qwen3-32b__{ro}__attack_only.jsonl.gz"
        tab[ro] = {(r["train_set"], r["domain"]) + key(r): r for r in map(json.loads, gzip.open(f, "rt"))}
    shared = set.intersection(*(set(t) for t in tab.values()))
    out = {"tables": {ro: hashlib.sha256((d / f"qwen3-32b__{ro}__attack_only.jsonl.gz").read_bytes()).hexdigest() for ro in READOUTS},
           "rows_per_readout": {ro: len(t) for ro, t in tab.items()}, "shared_rows": len(shared), "results": {}}
    for split in (("real", "real"), ("real", "sim")):
        keys = sorted(k for k in shared if k[:2] == split)
        meta = [manifest[k[2:]] for k in keys]
        strata = {"all": [True] * len(keys)}
        for tool in ("bash", "text_editor"):
            strata[f"tool={tool}"] = [m["tool"] == tool for m in meta]
        strata["tool=other"] = [m["tool"] not in ("bash", "text_editor") for m in meta]
        strata["call_end=content"] = [bool(m["call_end_content"]) for m in meta]
        strata["call_end=not content"] = [not m["call_end_content"] for m in meta]
        res = {}
        for name, sel in strata.items():
            sel = np.array(sel)
            if not sel.any():
                continue
            ks = [k for k, s_ in zip(keys, sel) if s_]
            y = np.array([tab["call_end"][k]["y"] for k in ks], bool)
            r = {"calls": int(len(ks)), "impactful": int(y.sum())}
            for ro in READOUTS:
                s = np.array([tab[ro][k]["score"] for k in ks]); fl = np.array([tab[ro][k]["flag"] for k in ks], bool)
                r[ro] = {"recall": float(fl[y].mean()) if y.any() else None, "false_alarms": float(fl[~y].mean()) if (~y).any() else None,
                         "auroc": auroc(s, y)}
            res[name] = r
        out["results"]["train_%s__test_%s" % split] = res
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path, text = HERE / "results" / f"k2_position_{stamp}.json", json.dumps(out, indent=1)
    with path.open("x") as f:
        f.write(text)
    print("rows", out["rows_per_readout"], "shared", out["shared_rows"])
    for split, res in out["results"].items():
        print(split)
        for name, r in res.items():
            cells = " | ".join(f"{ro} rec {r[ro]['recall']:.3f} fa {r[ro]['false_alarms']:.3f} auc {r[ro]['auroc']:.3f}"
                               if r[ro]["recall"] is not None and r[ro]["auroc"] is not None else f"{ro} n/a" for ro in READOUTS)
            print(f"  {name:22s} n={r['calls']:6d} imp={r['impactful']:5d} | {cells}")
    print("->", path)


if __name__ == "__main__":
    main()
