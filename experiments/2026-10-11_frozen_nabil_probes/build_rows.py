"""Local, IDs only: the call_end rows to score for part 1. Each row maps to (extraction, chunk, global row) in the
stored readings. It covers stage 1's listed real calls (key_stage1.csv: worlds real and run2) and the impactful call_end
calls of honest A (36 calibration tasks, which set the threshold) and of honest B (the calibration tasks plus the 16
stage-1 held-out tasks, used for false alarms). Counted runs only (not redrawn, never a rerun). No label is read here.

  python3 -I build_rows.py <acts_meta root> <key_stage1.csv> <split_v1.json> <out.json>
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

EXTRACTIONS = ["2026-10-09_rerun_strict_segends_v1", "2026-10-10_scaleup_canary_segends_v1", "2026-10-10_scaleup_l2_segends_v1",
               "2026-10-10_scaleup_topup_segends_v1", "2026-10-10_second_attack_segends_v1"]
WORLD = {("2026-10-10_second_attack_segends_v1", "attack"): "run2"}


def main(root, key, split, out):
    root = Path(root)
    sp = json.loads(Path(split).read_text())["tasks"]
    cal = {t for t, v in sp.items() if v["side"] == "train" and not v["stage1"]}
    ho1 = {t for t, v in sp.items() if v["side"] == "held_out" and v["stage1"]}
    ce, honest = {}, []
    for x in EXTRACTIONS:
        done = json.loads((root / x / "complete.json").read_text())["epochs"]
        files = sorted((root / x / "meta").glob("*/*.json"))
        if len(files) != done:
            raise SystemExit(f"{x}: {len(files)} metas vs {done} epochs")
        for f in files:
            m = json.loads(f.read_text())
            if m.get("redrawn_because") or m.get("attempt_kind") == "rerun":
                continue
            arm = f.parent.name
            world = WORLD.get((x, arm), {"attack": "real"}.get(arm, arm))
            for c in m["calls"]:
                if c.get("kind") != "call_end" or c.get("row") is None or "call_index" not in c:
                    continue
                loc = [x, arm, m["chunk"], m["row_start"] + c["row"]]
                if arm in ("attack",):
                    ce[(world, c["irid"], c["call_index"])] = loc
                elif arm in ("honest_a", "honest_b") and c.get("label") == "impactful":
                    sets = ["cal"] if m["task"] in cal else []
                    sets += ["ho1"] if (arm == "honest_b" and m["task"] in ho1) else []
                    if sets and (arm == "honest_b" or "cal" in sets):
                        honest.append({"id": f"{arm}|{m['task']}|{c['irid']}|{c['call_index']}", "world": arm, "task": m["task"],
                                       "sets": sets, "loc": loc})
    rows, missing = [], 0
    with open(key) as fh:
        for r in csv.DictReader(fh):
            if r["world"] not in ("real", "run2"):
                continue
            loc = ce.get((r["world"], r["irid"], int(r["call_index"])))
            if loc is None:
                missing += 1
                continue
            rows.append({"id": r["opaque_id"], "world": r["world"], "task": r["task_key"], "side": r["side"], "loc": loc})
    if missing:
        raise SystemExit(f"{missing} listed real calls without a call_end row")
    res = {"key_sha256": hashlib.sha256(Path(key).read_bytes()).hexdigest(), "listed_real": len(rows), "honest": len(honest),
           "rows": rows + honest}
    with open(out, "x") as fh:  # never overwrite
        fh.write(json.dumps(res))
    from collections import Counter
    print(json.dumps({"listed_real": len(rows), "by_side_world": dict(Counter(f"{r['side']}/{r['world']}" for r in rows)),
                      "honest": dict(Counter(f"{h['world']}/{','.join(h['sets'])}" for h in honest)),
                      "honest_cal_tasks_a": len({h['task'] for h in honest if h['world'] == 'honest_a'})}))


if __name__ == "__main__":
    main(*sys.argv[1:5])
