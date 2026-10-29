"""The attack detector's nulls for every reported world (added after review). $0, local (Python 3.14 reads the .eval logs
for the errored runs). Layer 51, tool-output readout.
- All outputs, the refit null: the 50 saved draws (labels shuffled among each main task's real runs, full refit), each
  scoring every on-policy run; p = (1 + #{null >= observed}) / 51, for real, the honest_a replicate, September, today.
- First k outputs (10, 20, 27) and the absolute-position windows (1-10 ... 41-50), a fixed-score null: attack and
  honest labels permuted among each task's runs, 1,000 draws (seed [0, 9]); p = (1 + #{null >= observed}) / 1,001.

  uv run --no-project --python 3.14 --with numpy --with scikit-learn python analyze_attack_detector_nulls.py \
      --runs results/attack_detector_onpolicy_<stamp>.json --per-output results/attack_detector_per_output_<stamp>.json --run-dir <rerun folder>
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from analyze_attack_detector import WORLDS, aucs, errored_runs

HERE = Path(__file__).parent
CELL = "L51_output_end"


def fixed_null(y, s, t, rng, n=1000):
    obs, null = aucs(y, s)[0], []
    for _ in range(n):
        yy = y.copy()
        for task in np.unique(t):
            i = np.flatnonzero(t == task)
            yy[i] = rng.permutation(yy[i])
        null.append(aucs(yy, s)[0])
    return {"auroc": round(float(obs), 4), "p": round(float((1 + (np.array(null) >= obs).sum()) / (1 + n)), 4)}


def main(a):
    bad = set(errored_runs(Path(a.run_dir).expanduser()))
    R, P = json.loads(Path(a.runs).read_text()), json.loads(Path(a.per_output).read_text())
    c = R["cells"][CELL]
    runs = np.array(c["on_runs"])
    arm, keep = np.array([r.split("|")[0] for r in runs]), ~np.isin(runs, list(bad))
    out = {"inputs": {"runs": Path(a.runs).name, "per_output": Path(a.per_output).name}, "all_outputs_refit_null": {}, "first_k": {}, "windows": {}}
    for w, (at, hn) in WORLDS.items():
        m = keep & np.isin(arm, [at, hn])
        y = (arm[m] == at).astype(int)
        obs = aucs(y, np.array(c["on_score"])[m])[0]
        nv = np.array([aucs(y, np.array(d["on_score"])[m])[0] for d in c["null"]])
        out["all_outputs_refit_null"][w] = {"auroc": round(float(obs), 4), "p": round(float((1 + (nv >= obs).sum()) / (1 + len(nv))), 4), "draws": int(len(nv))}
    seq = {}
    for r, s in zip(P["rows"], P["scores"][CELL]):
        if r["kind"] == "output_end" and s is not None:
            seq.setdefault(f"{r['arm']}|{r['task']}", []).append((r["turn"], r["call_index"], s))
    seq = {k: [x[2] for x in sorted(v)] for k, v in seq.items() if k not in bad}
    rng = np.random.default_rng([0, 9])
    for name, sel, f in [(f"first {k}", lambda v, k=k: True, lambda v, k=k: np.mean(v[:k])) for k in (10, 20, 27)] + \
                        [(f"outputs {lo + 1}-{hi}", lambda v, hi=hi: len(v) >= hi, lambda v, lo=lo, hi=hi: np.mean(v[lo:hi])) for lo, hi in ((0, 10), (10, 20), (20, 30), (30, 40), (40, 50))]:
        res = {}
        for w, (at, hn) in WORLDS.items():
            ks = sorted(k for k in seq if k.split("|")[0] in (at, hn) and sel(seq[k]))
            y = np.array([k.split("|")[0] == at for k in ks], int)
            res[w] = fixed_null(y, np.array([f(seq[k]) for k in ks]), np.array([k.split("|")[1] for k in ks]), rng) | {"runs": int(len(ks))}
        out["first_k" if name.startswith("first") else "windows"][name] = res
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"attack_detector_nulls_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print("all outputs, refit null:", json.dumps(out["all_outputs_refit_null"]))
    for sec in ("first_k", "windows"):
        for k, r in out[sec].items():
            print(k, {w: (v["auroc"], v["p"]) for w, v in r.items()})
    print("->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("--runs", "--per-output", "--run-dir"):
        ap.add_argument(k, required=True)
    main(ap.parse_args())
