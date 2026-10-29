"""POST HOC diagnostic for analyze_q22_matched_fa_POSTHOC.py: per fold model k, each world's own 1% threshold (from its
calibration fold k + 1) and the share of that world's test-fold read-only calls above it, plus the share of each world's
test false alarms carried by its top 5 tasks. Numbers only. $0, local.

  uv run --no-project --with numpy python analyze_q22_matched_fa_folds_POSTHOC.py <per-call draw-0 file>
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from analyze_scaleup_plants import threshold

WORLDS = {"real": "real->real", "sim": "real->sim", "run2": "real->run2"}


def main(path: str):
    P = json.loads(Path(path).read_bytes())
    meta, cells, folds = P["meta"], P["draws"][0]["cells"], P["folds"]
    cal = cells["real->real"]["per_call"]["calibration"]
    out = {}
    for w, cell in WORLDS.items():
        p, rows, alarms = cells[cell]["per_call"], [], Counter()
        for k in range(5):
            c = np.array([s for i, s, mk in zip(cal["row"], cal["score"], cal["model_fold"]) if mk == k and meta[i]["world"] == w and meta[i]["y"] == 0])
            t = threshold(c, np.ones(len(c)), fpr=0.01)
            neg = [(meta[i]["task"], s) for i, s in zip(p["row"], p["score"]) if meta[i]["world"] == w and meta[i]["y"] == 0 and folds[meta[i]["task"]] == k]
            alarms.update(task for task, s in neg if s > t)
            rows.append({"fold": k, "threshold": t, "calibration_calls": len(c), "test_calls": len(neg), "test_flagged": sum(s > t for _, s in neg)})
        total = sum(alarms.values())
        out[w] = {"by_fold": rows, "test_false_alarms": total, "top5_task_share": sum(n for _, n in alarms.most_common(5)) / total if total else None}
        print(w, "| top-5-task share of false alarms:", None if not total else round(out[w]["top5_task_share"], 3), "| total", total)
        for r in rows:
            print(f"  fold {r['fold']}: threshold {r['threshold']:+.3f} | test {r['test_flagged']}/{r['test_calls']} = {r['test_flagged'] / r['test_calls']:.4f}")
    print(json.dumps(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    main(ap.parse_args().path)
