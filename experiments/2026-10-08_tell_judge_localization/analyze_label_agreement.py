"""Agreement between the provisional labels (this session, blind) and the pre-registered Sol labels, on the same blind
list. Plan, fixed before the Sol labels exist:
- Cohen's kappa over the 8 categories and over the pooled content indicator (state inconsistency, unearned success,
  environment rule violation), with the confusion matrix;
- the localization analysis rerun on the Sol labels is the pre-registered result; the provisional one is reported next to
  it. If content kappa < 0.6, the provisional result is read as unreliable rather than as a replication.

  uv run --with scikit-learn python experiments/2026-10-08_tell_judge_localization/analyze_label_agreement.py \
      --a config/flag_labels_provisional.yaml --b config/flag_labels_sol.yaml
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml
from sklearn.metrics import cohen_kappa_score

HERE = Path(__file__).parent
CODES = {"SI": "state_inconsistency", "US": "unearned_success", "RV": "environment_rule_violation", "TR": "too_regular",
         "HA": "harness_artifact", "RL": "reference_leak", "TO": "thin_output", "OT": "other"}
CONTENT = {"state_inconsistency", "unearned_success", "environment_rule_violation"}


def load(p):
    return {k: CODES.get(v, v) for k, v in yaml.safe_load((HERE / p).read_text())["labels"].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    args = ap.parse_args()
    a, b = load(args.a), load(args.b)
    ids = sorted(set(a) & set(b))
    assert len(ids) == len(a) == len(b), "the two label files cover different flags"
    xa, xb = [a[i] for i in ids], [b[i] for i in ids]
    out = {"n": len(ids), "kappa_8": round(cohen_kappa_score(xa, xb), 3),
           "kappa_content": round(cohen_kappa_score([x in CONTENT for x in xa], [x in CONTENT for x in xb]), 3),
           "raw_agreement": round(sum(p == q for p, q in zip(xa, xb)) / len(ids), 3),
           "confusion": {f"{p} -> {q}": n for (p, q), n in Counter(zip(xa, xb)).most_common()}}
    print(json.dumps(out, indent=1))
    (HERE / "results" / "label_agreement.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
