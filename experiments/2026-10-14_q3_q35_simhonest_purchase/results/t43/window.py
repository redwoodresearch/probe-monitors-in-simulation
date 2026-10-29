import glob, collections
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor
from inspect_ai.log import read_eval_log
W0 = datetime(2026,10,14,23,15,tzinfo=timezone.utc); W1 = datetime(2026,10,14,23,55,tzinfo=timezone.utc)
def scan(f):
    out = {"ok": 0, "timeouts": [], "maxwall": 0.0}
    try:
        log = read_eval_log(f)
    except Exception as e:
        return f, {"read_error": str(e)[:120]}
    for s in log.samples or []:
        for e in s.events:
            if e.event != "model" or not str(e.model).startswith("openai/"): continue
            ts = e.timestamp
            if ts is None or not (W0 <= ts <= W1): continue
            if e.error: out["timeouts"].append(ts.strftime("%H:%M:%S"))
            else:
                out["ok"] += 1
                if e.completed: out["maxwall"] = max(out["maxwall"], (e.completed - ts).total_seconds())
    return f, out
files = glob.glob("/home/aswin/phase2/runs/2026-10-14_160932_simh_cont/sim_honest@september/*/*.eval")
tot_ok = 0; hits = []; maxw = []
with ProcessPoolExecutor(16) as ex:
    for f, o in ex.map(scan, files):
        if "read_error" in o: print("READERR", f.split("/")[-2], o["read_error"]); continue
        tot_ok += o["ok"]
        if o["timeouts"]: hits.append((f.split("/")[-2], o["timeouts"]))
        if o["ok"]: maxw.append((round(o["maxwall"]), f.split("/")[-2]))
print("files", len(files), "simulator calls OK in 23:15-23:55 UTC:", tot_ok)
print("runs with simulator attempt timeouts in window:", hits)
print("longest successful simulator call wall-time (s) in window, top 5:", sorted(maxw)[-5:])
