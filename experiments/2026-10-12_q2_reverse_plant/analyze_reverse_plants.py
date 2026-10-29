"""Reverse plant, part (a) (PLAN_A.md, committed with this script before it runs): the swap's ±10 plants on the
single-habit surface-edit arms. Reproduces the committed paired means first (gate), then plants. Local, $0.

  python3 experiments/2026-10-12_q2_reverse_plant/analyze_reverse_plants.py <attack runs jsonl> <honest runs jsonl> \\
      <surface_edits_2026-10-08_171856.json>
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
SHA = {"swap_frozen_runs_2026-10-08_171744.jsonl": "42a7f06d65b0b2947a2fe35dfeea60cb99646a7c7e0b8bc46d8c67bf1df49b72",
       "swap_frozen_runs_2026-10-08_171727.jsonl": "4e884cdbc896d7bbf43ad7676daaf3282da8cc9ac16346ddca64d1cfb3791e62",
       "surface_edits_2026-10-08_171856.json": "d05502fea054f04c98613d07df5a924e2f3ec6eac39a0108ee0b409b835e36a9"}
PLANT, TOL, SEED, B, ALPHAS = 0.10, 0.01, 20261011, 2000, ("0.05", "0.01")
READOUTS = ("output_end", "output_end@48")  # L63 primary (the frozen detector's), L48 beside


def main(att_p: Path, hon_p: Path, ref_p: Path):
    me = Path(__file__).resolve()
    if subprocess.run(["git", "ls-files", "--error-unmatch", str(me)], cwd=HERE, capture_output=True).returncode \
            or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(me)], cwd=HERE).returncode:
        raise SystemExit("the script differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    for p, role in ((att_p, "swap_frozen_runs_2026-10-08_171744.jsonl"), (hon_p, "swap_frozen_runs_2026-10-08_171727.jsonl"),
                    (ref_p, "surface_edits_2026-10-08_171856.json")):  # each argument must be its own pinned file
        if hashlib.sha256(p.read_bytes()).hexdigest() != SHA[role]:
            raise SystemExit(f"{p.name} is not the pinned {role}")
    ref = json.loads(ref_p.read_text())["probe"]
    g = np.random.default_rng(SEED)

    def boot(d, tasks):
        u, ti = np.unique(tasks, return_inverse=True)
        W = np.stack([np.bincount(g.integers(0, len(u), len(u)), minlength=len(u))[ti] for _ in range(B)]).astype(float)
        m = (W @ d) / W.sum(1)
        return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
    out = {"plan": "PLAN_A.md (this folder)", "git": head, "inputs sha256": SHA, "arms": {}}
    for path in (att_p, hon_p):
        by = {}
        for r in map(json.loads, path.open()):
            by.setdefault(r["key"], {})[r["label"]] = r
        labels = sorted({lab for v in by.values() for lab in v} - {"V1"} - {"A_CTL", "H_CTL"})
        for lab in labels:
            attack = lab.startswith("A_")
            keys = sorted(k for k, v in by.items() if lab in v and v[lab]["first_changed_output"] is not None)
            res = {"changed runs": len(keys)}
            for ro in READOUTS:
                for a in ALPHAS:
                    ks = [k for k in keys if by[k][lab]["caught"][ro][a] is not None and by[k]["V1"]["caught"][ro][a] is not None]
                    ed = np.array([by[k][lab]["caught"][ro][a] for k in ks], float)
                    un = np.array([by[k]["V1"]["caught"][ro][a] for k in ks], float)
                    tasks = np.array([by[k]["V1"]["main_task"] for k in ks])
                    d = 100 * (ed - un); mean = float(d.mean())
                    want = ref[lab][ro]["changed"][f"catch_change_pp@{a}"]["mean"]
                    if round(mean, 4) != want:
                        raise SystemExit(f"{lab} {ro} {a}: mean {mean:.4f} doesn't reproduce the committed {want}")
                    r_ = {"n": len(ks), "mean change (points)": mean, "95% (task bootstrap)": boot(d, tasks)}
                    if attack:
                        n = len(ks); k_ = round(PLANT * n); pl = {}
                        for sign, want_c, new in (("-10", 1.0, 0.0), ("+10", 0.0, 1.0)):
                            elig = np.flatnonzero(ed == want_c)
                            if len(elig) < k_:
                                pl[sign] = {"constructible": False, "eligible": int(len(elig)), "needed": k_}; continue
                            pick = np.sort(g.choice(elig, k_, replace=False)); edp = ed.copy(); edp[pick] = new
                            dp = 100 * (edp - un); mp = float(dp.mean()); lo, hi = boot(dp, tasks); shift = mp - mean
                            target = (-1 if sign == "-10" else 1) * 100 * PLANT
                            pl[sign] = {"constructible": True, "k": k_, "expected shift (points)": 100 * (new - want_c) * k_ / n,
                                        "planted mean": mp, "shift": shift, "planted 95%": [lo, hi],
                                        "detected": bool(abs(shift - target) <= 100 * TOL and not (lo <= mean <= hi))}
                        r_["plants"] = pl
                        if ro == "output_end":  # the verdict is L63's only; L48 is descriptive (PLAN_A.md)
                            r_["passes its plants"] = all(p.get("detected", False) for p in pl.values())
                    res[f"{ro} @ {a}"] = r_
            out["arms"][lab] = res
    if set(out["arms"]) != {"A_SG", "A_NL", "A_NN", "H_SG", "H_NL", "H_NN"}:
        raise SystemExit(f"unexpected arms {sorted(out['arms'])}")
    out["L63 vs L48 (stated as found)"] = {lab: {ro: out["arms"][lab][f"{ro} @ 0.05"]["mean change (points)"] for ro in READOUTS}
                                           for lab in out["arms"] if lab.startswith("A_")}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"reverse_plants_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1)); print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
