"""Bugbot's #1173 thread (rxVuU): the readings' "(the gap grew)" suffix tests abs(fixed - e2) > abs(as simulated - e2), while
fix939's reading (../2026-10-08_q3_swap_cue/analyze_fix939.py, lines 253-261) appends it when the as-simulated gap lies
outside the fixed range. This $0 check re-labels three committed part-2 readings from their stored bar fields, at both rates:
  - as run: each script's own rule, which must reproduce its committed label exactly (else it exits);
  - fix939: analyze_fix939.py's rule (explains it if within and shrinks; partial if shrinks; else "doesn't explain it",
    with "(the gap grew)" if outside), where shrinks = outside and |fixed - e2| < |as simulated - e2|.
Each file is read from git at its pinned commit (sha256 recorded). usage: python check_gap_grew_label.py"""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
BAR = 5.0
FILES = {  # reader: (commit, path, the as-run script's rule)
    "Qwen3.5-27B": ("34c7c020", "experiments/2026-10-12_q2_qwen35_repaired/results/edit_effect_q35_2026-10-13_003512.json", "shrink restored"),
    "Llama-3.3-70B": ("dddc8324", "experiments/2026-10-12_q2_llama_reader/results/edit_effect_llama_2026-10-12_153720.json", "outside only"),
    "Olmo-3.1-32B": ("03af438b", "experiments/2026-10-12_q2_olmo_reader/results/edit_effect_olmo_2026-10-12_115023.json", "outside only"),
}


def as_run(rule, gap, asis, outside):  # analyze_edit_effect_q35.py:154-161; _llama.py:142-148 and _olmo.py:140-146
    if rule == "shrink restored":
        shrinks = outside and abs(gap) < abs(asis)
        explains, partial = abs(gap) <= BAR and shrinks, shrinks
    else:
        explains, partial = abs(gap) <= BAR and outside, outside and abs(gap) < abs(asis)
    if explains:
        return "explains it"
    return "partial" if partial else "doesn't explain it" + (" (the gap grew)" if abs(gap) > abs(asis) else "")


def fix939(gap, asis, outside):  # analyze_fix939.py:253-261
    shrinks = outside and abs(gap) < abs(asis)
    if abs(gap) <= BAR and shrinks:
        return "explains it"
    return "partial" if shrinks else ("doesn't explain it (the gap grew)" if outside else "doesn't explain it")


out = {"thread": "#1173 PRRT_kwDOU9RNXc6rxVuU", "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True, check=True).stdout.strip(),
       "files": {}, "rows": []}
for reader, (commit, path, rule) in FILES.items():
    raw = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=HERE, capture_output=True, check=True).stdout
    out["files"][reader] = {"commit": commit, "path": path, "sha256": hashlib.sha256(raw).hexdigest()}
    e = json.loads(raw)
    for rate in ("0.05", "0.01"):
        R = e["rates"][rate]; b = R["fix939's bar (on fixed - e2)"]
        gap, asis, (lo, hi) = b["fixed - e2"], b["as simulated - e2"], b["95%"]
        outside = not (lo <= asis <= hi)
        assert outside == b["as-simulated gap outside the fixed range"], (reader, rate)
        assert (abs(gap) <= BAR) == b["point within +-5"], (reader, rate)  # BAR = 5.0 points in all three readings
        committed = R["reading"].replace(" (1%, reported)", "")
        mine = as_run(rule, gap, asis, outside)
        if mine != committed:
            raise SystemExit(f"{reader} {rate}: the as-run rule gives {mine!r}, the file says {committed!r}")
        new = fix939(gap, asis, outside)
        out["rows"].append({"reader": reader, "rate": rate, "fixed - e2": gap, "as simulated - e2": asis, "fixed 95%": [lo, hi],
                            "outside": outside, "|fixed| > |as simulated|": abs(gap) > abs(asis), "committed label": committed,
                            "fix939's rule": new, "changes": new != committed})
        print(f"{reader:14} {rate}  gap {gap:+.2f} [{lo:+.2f}, {hi:+.2f}]  as simulated {asis:+.2f}  outside {outside!s:5}  "
              f"|gap|>|asis| {abs(gap) > abs(asis)!s:5}  committed: {committed:20}  fix939's rule: {new:20}  changes: {new != committed}")
out["any label changes"] = any(r["changes"] for r in out["rows"])
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
with open(dest := HERE / "results" / f"gap_grew_check_{stamp}.json", "x") as fh:
    fh.write(json.dumps(out, indent=1) + "\n")
print("any label changes:", out["any label changes"]); print(dest)
