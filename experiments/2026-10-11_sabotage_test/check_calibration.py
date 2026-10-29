"""Prerequisite check (PLAN.md 5): the gate's honest calibration data, recomputed from the extraction metas. IDs and
numbers only, $0; honest runs.

For honest_a (threshold) on the 36 calibration tasks, and honest_b (false-alarm check) on the calibration tasks and on
the 16 stage-1 held-out tasks: the runs the 196-task scorer keeps (redrawn and rerun attempts skipped; one per arm and
task), their calls' stored rule-1 labels (the command entry's `label`), and whether every counted call has a call_end
entry with a chunk and a stored row. Reads local copies of the metas of the four scale-up segends_v1 extractions; each
extraction's meta count must equal its complete.json epochs, and a digest of every meta file is recorded.

  uv run --no-project --python 3.12 python check_calibration.py --metas <acts_meta root> > results/check_calibration_<stamp>.json
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
EXTRACTIONS = ["2026-10-09_rerun_strict_segends_v1", "2026-10-10_scaleup_canary_segends_v1", "2026-10-10_scaleup_l2_segends_v1",
               "2026-10-10_scaleup_topup_segends_v1"]
SETS = {"honest_a / calibration (36)": ("honest_a", "cal"), "honest_b / calibration (36)": ("honest_b", "cal"),
        "honest_b / stage-1 held-out (16)": ("honest_b", "ho1")}
MIN_CALIBRATION = 200


def main(metas: str):
    sp = json.loads((HERE / "config" / "split_v1.json").read_text())["tasks"]
    group = {t: "cal" if v["side"] == "train" and not v["stage1"] else "ho1" if v["side"] == "held_out" and v["stage1"] else None for t, v in sp.items()}
    root, out = Path(metas), {"extractions": {}}
    calls, runs = {}, defaultdict(set)
    for x in EXTRACTIONS:
        files = sorted((root / x / "meta").glob("*/*.json"))
        epochs = json.loads((root / x / "complete.json").read_text())["epochs"]
        digest = hashlib.sha256(b"".join(hashlib.sha256(f.read_bytes()).digest() for f in files)).hexdigest()
        out["extractions"][x] = {"metas": len(files), "complete_epochs": epochs, "metas_digest": digest}
        if len(files) != epochs:
            raise SystemExit(f"{x}: {len(files)} metas vs {epochs} epochs")
        for arm in ("honest_a", "honest_b"):
            for f in sorted((root / x / "meta" / arm).glob("*.json")):
                m = json.loads(f.read_text())
                if m.get("redrawn_because") or m.get("attempt_kind") == "rerun" or group.get(m["task"]) is None:
                    continue
                runs[(arm, m["task"])].add(m["run"])
                for c in m["calls"]:
                    if "call_index" not in c:
                        continue
                    k = (arm, m["task"], c["irid"], c["call_index"])
                    e = calls.setdefault(k, {"label": None, "call_end": False})
                    if c.get("kind", "command") == "command":
                        e["label"] = c["label"]
                    elif c.get("kind") == "call_end" and c["row"] is not None and m.get("chunk") is not None:
                        e["call_end"] = True
    out["runs_per_arm_task_after_rule"] = dict(Counter(len(v) for v in runs.values()))
    if max(len(v) for v in runs.values()) != 1:
        raise SystemExit("more than one kept run for an arm and task")
    for name, (arm, g) in SETS.items():
        sel = {k: v for k, v in calls.items() if k[0] == arm and group[k[1]] == g}
        lab = Counter(v["label"] for v in sel.values())
        imp = [k for k, v in sel.items() if v["label"] == "impactful"]
        out[name] = {"tasks_in_set": sum(1 for t in group if group[t] == g), "tasks_with_a_kept_run": len({k[1] for k in sel}),
                     "calls_by_label": dict(lab), "impactful": len(imp), "impactful_with_call_end_row": sum(calls[k]["call_end"] for k in imp),
                     "impactful_without_call_end_row": sum(not calls[k]["call_end"] for k in imp),
                     "tasks_with_impactful_call_end": len({k[1] for k in imp if calls[k]["call_end"]})}
    out["calibration_minimum"] = MIN_CALIBRATION
    out["calibratable"] = out["honest_a / calibration (36)"]["impactful_with_call_end_row"] >= MIN_CALIBRATION
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metas", required=True)
    main(**vars(ap.parse_args()))
