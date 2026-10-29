"""POST HOC, $0, local: the drop sets for the matched-false-alarm redo (RULE.md), from false-alarm flags only.

Ranking rule, copied exactly from experiments/2026-10-12_q3_fa_concentration/RULE.md at bf593217 (blob eb66652f):
read-only calls are rows with y = 0; flagged means `flag_frozen` (the frozen 1% threshold); per task, n = read-only calls
and k = flagged ones; tasks are ranked by k descending, ties to the higher k/n, then to the task id ascending. Here only
the simulated world's ranking is used (draw 0's real-trained cell real->sim, per-call rows). Drop sets are its top 1, 3
and 5 tasks. Gate: the pooled sim k/n equals the cell's stored `false_alarm_frozen` within 1e-9. No catch number is read.

  uv run --no-project --with numpy python experiments/2026-10-12_q3_matchedfa_drop_top/compute_drop_sets.py <per-call draw-0 file>
"""
import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
INPUT_SHA = "c2656658baaf3aa150f83f4d105e7572bfd854f98032831621419e1f8e8f36bf"  # s3_manifest_scaleup196_v1.json
RULE_SOURCE = {"commit": "bf593217", "path": "experiments/2026-10-12_q3_fa_concentration/RULE.md", "blob": "eb66652fb5e9a0407dbbb0b83f6bb8a6c3a8ef8a"}


def main(path):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != INPUT_SHA:
        raise SystemExit("input sha256 differs from the manifest")
    P = json.loads(raw)
    meta, cell = P["meta"], P["draws"][0]["cells"]["real->sim"]
    p = cell["per_call"]
    n, k = Counter(), Counter()
    for i, f in zip(p["row"], p["flag_frozen"]):
        m = meta[i]
        if m["world"] != "sim" or m["y"] != 0:
            continue
        n[m["task"]] += 1
        k[m["task"]] += int(bool(f))
    pooled = sum(k.values()) / sum(n.values())
    stored = cell["values"]["false_alarm_frozen"]
    if abs(pooled - stored) > 1e-9:
        raise SystemExit(f"gate failed: pooled sim false alarms {pooled} != stored {stored}")
    rank = sorted(n, key=lambda t: (-k[t], -(k[t] / n[t]), t))
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    out = {"label": "post hoc: tasks chosen after seeing the false-alarm concentration; ranking from false-alarm flags only",
           "git": head, "input": {"file": Path(path).name, "sha256": INPUT_SHA}, "rule_source": RULE_SOURCE,
           "gate_pooled_sim_false_alarm": {"recomputed": pooled, "stored": stored, "flagged": sum(k.values()), "total": sum(n.values())},
           "top10_sim": [{"task": t, "flagged": k[t], "read_only": n[t]} for t in rank[:10]],
           "drop_sets": {f"top{j}": rank[:j] for j in (1, 3, 5)}}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"drop_sets_POSTHOC_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps({k2: out[k2] for k2 in ("gate_pooled_sim_false_alarm", "drop_sets")}, indent=1), "->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    main(ap.parse_args().path)
