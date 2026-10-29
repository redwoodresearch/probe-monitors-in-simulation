"""Label-tail audit list: every false positive and every missed impactful call at the frozen 1% threshold, for the
cross-domain cells (trained on real, tested on sim; trained on sim, tested on real), per readout. The list is large,
so it goes to .run_cache/q2_tail/, not results/.

Call identities and rule labels only (arm, key, step, action, tool, label, category, rules, score, threshold); no
command text, so the list can go to a separate blind adjudication.

  uv run python experiments/2026-10-06_q2_impactful_call_transfer/analyze_q2_tail.py RESULTS_JSON
"""

import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
CELLS = [("real-attack-e1", "sim-attack"), ("sim-attack", "real-attack-e1")]


def main(results_json: str):
    res = json.loads(Path(results_json).read_text())
    labels = ROOT / res["config"]["labels_file"]
    rows = {(r["arm"], r["key"], r["step"], r["action"]): r for r in map(json.loads, labels.read_text().splitlines())}
    preds_path = Path(results_json).parent / res["predictions_file"]
    if not preds_path.exists():
        preds_path = ROOT / ".run_cache" / "q2_predictions" / res["predictions_file"]
    out = []
    with gzip.open(preds_path, "rt") as f:
        for line in f:
            p = json.loads(line)
            if (p["train_set"], p["arm"]) not in CELLS:
                continue
            kind = "false_positive" if p["flag"] and p["y"] == 0 else "missed_impactful" if not p["flag"] and p["y"] == 1 else None
            if kind:
                r = rows[(p["arm"], p["key"], p["step"], p["action"])]
                out.append({"readout": p["position"], "train": p["train_set"], "test": p["arm"], "kind": kind,
                            **{k: p[k] for k in ("key", "step", "action", "tool")}, "rule_label": r["label"],
                            "scored_label": "impactful" if p["y"] == 1 else "read_only",  # after the run's label_rule
                            "label_rule": res["config"].get("label_rule"),
                            **{k: r[k] for k in ("net_method", "net_dest", "net_sends_data")},
                            "category": r["category"], "rules": r["rules"], "score": round(p["score"], 4),
                            "threshold": round(p["threshold"], 4)})
    stem = Path(results_json).stem.replace("q2_probes_", "")
    path = ROOT / ".run_cache" / "q2_tail" / f"q2_tail_{stem}.jsonl"  # 20+ MB of call identities: not in results/
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(o) + "\n" for o in out))
    from collections import Counter
    print(Counter((o["readout"], o["train"], o["kind"]) for o in out))
    print(f"labeler {res['labeler_sha']}; {len(out)} calls -> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1])
