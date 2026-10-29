"""The attack detector's off-policy reference restricted to the 20 Phase 2 pilot main tasks, the on-policy cohort, so the
reading-vs-acting comparison uses the same tasks. $0, local. Held-out outer-fold run scores from
run_attack_detector.py (`off_outer`; the pilot main tasks were never in any reference model's training). AUROC real
e1 vs real honest and simulated attack vs simulated honest, with 2,000 resamples of main tasks (seed 0).

  uv run --no-project --with numpy --with scikit-learn python analyze_attack_detector_cohort.py \
      results/attack_detector_onpolicy_<stamp>.json [--cell L51_output_end]
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from analyze_attack_detector import aucs, ci

HERE = Path(__file__).parent
PILOT = HERE.parents[1] / "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9.json"


def main(path: str, cell: str):
    off = json.loads(Path(path).read_text())["cells"][cell]["off_outer"]
    mains = {c["main"] for c in json.loads(PILOT.read_text())["cells"]}
    out = {"input": Path(path).name, "cell": cell, "pilot_main_tasks": len(mains), "pairs": {}}
    for name, (a, h) in {"real e1 vs real honest": ("real-attack-e1", "real-honest"), "sim attack vs sim honest": ("sim-attack", "sim-honest")}.items():
        y, s, t = [], [], []
        for arm, lab in ((a, 1), (h, 0)):
            for m, sc in zip(off[arm]["main_task"], off[arm]["score"]):
                if m in mains:
                    y.append(lab), s.append(sc), t.append(m)
        y, s = np.array(y), np.array(s)
        tasks = sorted(set(t))
        ti = np.searchsorted(tasks, t)
        rng = np.random.default_rng(0)
        bs = [aucs(y, s, np.bincount(rng.integers(0, len(tasks), len(tasks)), minlength=len(tasks))[ti])[0] for _ in range(2000)]
        out["pairs"][name] = {"auroc": round(float(aucs(y, s)[0]), 4), "ci95": ci(bs), "runs": {a: int(y.sum()), h: int((1 - y).sum())},
                              "main_tasks": len(tasks)}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"attack_detector_cohort_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps(out["pairs"]))
    print("->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--cell", default="L51_output_end")
    a = ap.parse_args()
    main(a.path, a.cell)
