"""Attempted-only: Stage 1 (level 0, 64-layer headline), the level 0/7 comparison (layers 9/16/17) and Stage 3 (level 0,
pooled view), for pre_call and call_end, on attack runs that q3 labeled `attempted`.

  python attempted_only.py           each arm keeps its own attempted runs (different cell sets in real and sim)
  python attempted_only.py paired    only cells where both the real (epoch 1) and the sim run attempted, so real and
                                     sim keep the same cells
Honest runs are untouched. The probes are not refit: only the scored calls change."""
import json
import sys
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parent))
import compare_levels  # noqa: E402
import stage1  # noqa: E402
import stage3  # noqa: E402
from paths import DATA, RESULTS, stamp as now  # noqa: E402

# q3's attempt labels: the corrected table from PR #73 (2026-10-08: 165 unclear runs moved to declined, 3 sim runs
# attempted -> declined). Results before 2026-10-08 used the earlier copy, DATA / "attack_attempt_labels_2026-10-07_154729.jsonl".
LABELS = Path(__file__).resolve().parents[3] / "experiments/2026-10-07_q3_nano_monitor/results/attack_attempt_labels_2026-10-08_010659.jsonl"
ARMS = ("real-attack-e1", "sim-attack")


def kept_runs(paired: bool) -> tuple[set, dict]:
    """The attack runs to keep, with counts as labeled and as actually scored (runs present in the Q2 score tables)."""
    L = pd.read_json(LABELS, lines=True)
    L = L[L.arm.isin(ARMS)]
    att = L[L.label == "attempted"]
    if paired:
        both = set.intersection(*[set(att[att.arm == a].cell) for a in ARMS])
        att = att[att.cell.isin(both)]
    scored = pd.read_json(DATA / "tables" / "qwen3-32b__pre_call__attack_only.jsonl.gz", lines=True)[["arm", "key", "cell"]].drop_duplicates()
    kept = scored.merge(att[["arm", "key"]], on=["arm", "key"])
    counts = {"label_eligible": {"runs": {a: int((att.arm == a).sum()) for a in ARMS}},
              "analyzed": {"runs": {a: int((kept.arm == a).sum()) for a in ARMS}}}
    if paired:
        counts["label_eligible"]["cells"] = len(both)
        counts["analyzed"]["cells"] = len(set.intersection(*[set(kept[kept.arm == a].cell) for a in ARMS]))
    return set(zip(att.arm, att.key)), counts


def s1(t, tr, keep):
    stage1.KEEP = keep
    return stage1.analyze(t, tr)


def s3(t, tr, keep):
    stage3.KEEP = keep
    return stage3.analyze(t, tr)


if __name__ == "__main__":
    paired = len(sys.argv) > 1 and sys.argv[1] == "paired"
    keep, counts = kept_runs(paired)
    tag = "attempted_paired" if paired else "attempted"
    t1 = [DATA / "tables" / f"{r}__{p}__attack_only.jsonl.gz" for r in ("qwen3-32b", "qwen3.5") for p in ("pre_call", "call_end")]
    t3 = [DATA / "tables" / f"{r}__{p}__pooled.jsonl.gz" for r in ("qwen3-32b", "qwen3.5") for p in ("pre_call", "call_end")]
    r1 = Parallel(n_jobs=-1)(delayed(s1)(t, tr, keep) for t in t1 for tr in ("real", "sim"))
    r3 = Parallel(n_jobs=-1)(delayed(s3)(t, "real", keep) for t in t3)
    (RESULTS / f"shift_{tag}_stage1_stage3_{now()}.json").write_text(
        json.dumps({"labels": str(LABELS), "paired": paired, "kept": counts, "stage1": r1, "stage3": r3}, indent=1))
    compare_levels.KEEP = keep
    compare_levels.LEVELS, compare_levels.POSITIONS = [0, 7], ["pre_call", "call_end"]
    compare_levels.main(f"_{tag}")
    print("done", tag, counts)
