"""Per-draw provenance for the 196-task scoring (thread PRRT_kwDOU9RNXc6rImym on #417): for every draw record in the
main and null checkpoint folders, its id, file, save time (the volume's file time) and the scorer that made it, assigned
from the app that wrote it and that app's window in Modal's app list (#418's disclosure). A _failed_v1 record is a failed
original kept by Amendment 8's redo; it was made by e2bd13c8 (the rename gives it a later file time). Reads names and
times only, no values. Raises if any record falls outside every window.

  uv run --no-project --python 3.12 --with 'modal==1.2.6' python draw_provenance_scaleup196.py <out.json>
"""
import datetime as dt
import json
import re
import sys
from zoneinfo import ZoneInfo

import modal

PT = ZoneInfo("America/Los_Angeles")
t = lambda s: dt.datetime.fromisoformat(f"2026-10-10T{s}").replace(tzinfo=PT)
SCORERS = {"7cf6fdae": "353015fca706", "e2bd13c8": "2c933bae5241", "b82dd04d": "4d0c69147250"}  # scorer blob prefixes
WINDOWS = {  # folder -> [(start, end, commit, app)]
    "scaleup196_86a195d3d2e5": [(t("14:19:21"), t("14:47:02"), "7cf6fdae", "ap-oDRiGxzY4JAO7qp7mhBCoX"),
                                (t("15:41:20"), t("23:59:59"), "b82dd04d", "ap-pihbE2tc1RCYD7mm4qdLu0")],
    "scaleup196_null9_9a74d7009164": [(t("13:22:00"), t("13:31:00"), "7cf6fdae", "ap-43SUTup55a4RPuxki6Hx80"),
                                      (t("14:29:41"), t("14:46:53"), "e2bd13c8", "ap-M7di0kNoWOxslcYOwSfUn2"),
                                      (t("15:41:20"), t("23:59:59"), "b82dd04d", "ap-pJoEfhk6Zs7nFFkRD7RrMU")]}
v = modal.Volume.from_name("probe-shift_aswin-surya_sub-q3onpol_score_kill-2026-11-08", version=2)
rows = []
for folder, wins in WINDOWS.items():
    for e in v.listdir(folder):
        f = e.path.rsplit("/", 1)[-1]
        if not f.startswith("draw_"):
            continue
        saved = dt.datetime.fromtimestamp(e.mtime, PT)
        if f.endswith("_failed_v1.json"):
            commit, app = "e2bd13c8", "ap-M7di0kNoWOxslcYOwSfUn2"
        else:
            hit = [(c, a) for s, en, c, a in wins if s <= saved <= en]
            if len(hit) != 1:
                raise ValueError(f"{folder}/{f} saved {saved:%H:%M:%S} falls in {len(hit)} windows")
            commit, app = hit[0]
        rows.append({"folder": folder, "file": f, "draw": int(re.search(r"draw_(\d+)", f).group(1)), "saved_pt": saved.strftime("%H:%M:%S"),
                     "kind": "failed original (kept)" if "failed" in f else "record", "scorer_commit": commit,
                     "scorer_blob": SCORERS[commit], "app": app})
summary = {}
for r in rows:
    k = f"{r['folder']} | {r['scorer_commit']} | {r['kind']}"
    summary[k] = summary.get(k, 0) + 1
json.dump({"summary": summary, "draws": sorted(rows, key=lambda r: (r["folder"], r["draw"], r["file"]))}, open(sys.argv[1], "x"), indent=1)
print(json.dumps(summary, indent=1))
