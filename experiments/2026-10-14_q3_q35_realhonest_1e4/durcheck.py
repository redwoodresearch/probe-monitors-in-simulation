import json, statistics as st, sys
from datetime import datetime
T = lambda s: datetime.fromisoformat(s).timestamp()
d = json.load(open("/home/aswin/phase2/runs_rh1e4/2026-10-14_135334_honest_rh980/launch.json"))
c = {r["task"]: r for r in json.load(open("/home/aswin/phase2/rh_test/batchc_launch.json"))["runs"]}
dur = lambda r: T(r["attempts"][-1]["end"]) - T(r["attempts"][-1]["start"])
ended = [r for r in d["runs"] if r["state"] in ("done", "no policy call") and r["attempts"] and r["attempts"][-1].get("end")]
pairs = [(dur(r), dur(c[r["task"].rsplit("_", 1)[0]])) for r in ended]
ratios = sorted(a / b for a, b in pairs)
q = lambda xs, p: xs[int(p * (len(xs) - 1))]
print(f"ended {len(ended)}; ours median {st.median([a / 60 for a, _ in pairs]):.1f} min; same tasks in batch C median "
      f"{st.median([b / 60 for _, b in pairs]):.1f} min; per-task ratio ours/C median {st.median(ratios):.2f} "
      f"(p25 {q(ratios, .25):.2f}, p75 {q(ratios, .75):.2f}), n {len(ratios)}")
print("running", sum(r["state"] == "running" for r in d["runs"]), "started", sum(bool(r["attempts"]) for r in d["runs"]),
      "states", dict(__import__("collections").Counter(r["state"] for r in d["runs"])))
rep = __import__("collections").defaultdict(__import__("collections").Counter)
for r in d["runs"]:
    k = r["task"].rsplit("_", 1)[1]
    rep[k]["started"] += bool(r["attempts"])
    rep[k]["running"] += r["state"] == "running"
    rep[k]["ended"] += r["state"] in ("done", "no policy call")
print("per replicate:", {k: dict(v) for k, v in sorted(rep.items())})
