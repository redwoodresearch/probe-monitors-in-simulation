"""Batch C's timing, the evidence for the one-launch design (RULE.md, Amendment 2). Reads batch C's launch.json
(s3://individual-aswin-surya-probe-shift-kill-2027-04-06/phase2/runs/2026-10-11_112714_honest_c/launch.json; timestamps
and counts only) and prints: wall time, run durations, the tail after the last start, and per server the idle time before
the batch ended (servers_september_scaleup.yaml stops a server after idle_exit_min = 20 idle minutes). Run on the devbox:
    ~/sims/simulated-control-evals-sept/.venv/bin/python batchc_timing.py <launch.json>"""
import hashlib
import json
import sys
from datetime import datetime

raw = open(sys.argv[1], "rb").read()
d = json.loads(raw)
T = lambda s: datetime.fromisoformat(s).timestamp()  # noqa: E731
t0, t1 = T(d["started"]), T(d["finished"])
spans = [(r["server"], T(a["start"]), T(a["end"])) for r in d["runs"] for a in r["attempts"]]
durs = sorted(e - s for _, s, e in spans)
q = lambda p: durs[int(p * (len(durs) - 1))] / 60  # noqa: E731
print(f"launch.json sha256 {hashlib.sha256(raw).hexdigest()}; {len(d['runs'])} runs, {len(spans)} attempts, "
      f"{len(d['servers'])} servers, max_real_runs {d['pilot']['max_real_runs']}")
print(f"wall {(t1 - t0) / 3600:.2f} h; run minutes mean {sum(durs) / len(durs) / 60:.1f}, p50 {q(.5):.1f}, "
      f"p90 {q(.9):.1f}, p99 {q(.99):.1f}, max {durs[-1] / 60:.1f}")
print(f"last start +{(max(s for _, s, _ in spans) - t0) / 60:.1f} min; last end +{(max(e for *_, e in spans) - t0) / 60:.1f} min")
for i, srv in enumerate(d["servers"]):
    end = max(e for x, _, e in spans if x == srv)
    print(f"server {i}: {sum(x == srv for x, _, _ in spans)} attempts; idle before the batch ended {(t1 - end) / 60:.1f} min")
