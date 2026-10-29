"""Serial rel9+ cap projection for the ordered queue, from dt4f/cap_report.json (run cap_report.py first). Each launch's bound = the current
bound + earlier queued apps' wall bounds (median, or largest normal, of observed apps of that kind) + this launch's ceiling. Usage:
python3 -I queue_projection.py"""
import json
import statistics
from pathlib import Path

P = Path(__file__).resolve().parent
d = json.loads((P / "cap_report.json").read_text())
s, apps = d["summary"], d["apps"]
led = [json.loads(x) for x in (P / "rel9plus_ledger.jsonl").read_text().splitlines() if x.strip()]
rate = {x["kind"]: (x["ceiling_usd"], x["rate_usd_s"]) for x in led if "reserve" in x}
rate.setdefault("arm_honest", rate["simh_arm"])  # same containers as simh_arm (Amendment 39)
walls = sorted(a["wall_s"] for a in apps if a.get("kind") == "simh_arm" and a.get("wall_s") is not None and a["wall_s"] < 200)
aa = sorted(a["wall_s"] for a in apps if a.get("kind") == "arm_attack" and a.get("wall_s") is not None)
queue = [("Q1 _rlr", "simh_arm"), ("S4r s2", "simh_arm"), ("S4r s3", "simh_arm"), ("S4r s4", "simh_arm"), ("S4r s5", "simh_arm"),
         ("S5 s2", "simh_arm"), ("S5 s3", "simh_arm"), ("S5 s4", "simh_arm"), ("S5 s5", "simh_arm"), ("S5 final", "simh_arm"), ("r46", "simh_arm"),
         ("ug", "arm_attack"), ("ha25f q1", "arm_honest"), ("ha25f q2", "arm_honest"), ("ha25f q4", "arm_honest")]
for lab, pick in (("median", statistics.median), ("largest normal", max)):
    w = {"simh_arm": pick(walls), "arm_honest": pick(walls), "arm_attack": pick(aa) if aa else 90.0}
    b, out = s["bound_now_usd"], []
    for name, kind in queue:
        ceil, r = rate[kind]
        out.append((name, round(b + ceil, 2)))
        b += (w[kind] + 60) * r
    print(f"{lab} (simh wall {w['simh_arm']:.0f} s, attack wall {w['arm_attack']:.0f} s):", out, "| fit", sum(v <= 15 for _, v in out), "of", len(out),
          "| first refused:", next((n for n, v in out if v > 15), None))
print("bound_now", s["bound_now_usd"], "billed_posted", s["billed_posted_usd"], "apps_final", s["apps_final"], "at", s["at_pt"])
