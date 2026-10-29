"""Read-only, ON A BOX (d9 / 09's Append 35 @ d99ce7997): every SIMULATED run of the given launches with at least one
attempt whose terminal sample error is a RateLimitError (a 429 after the simulator's retries). Per run: host, launch run
dir, launch name, world "sim", cell (task key), run_key, state, and per attempt: index, start, end, exit code, the terminal
error class of each sample error (the exception a RetryError wraps, else the error's class), and the attempt's .eval path
and sha256. pure_429 = every attempt that failed has only RateLimitError terminal errors (Append 35). Metadata only.
  python3 sim_ratelimit_list.py <host label> <launch dir> [<launch dir> ...]"""
import glob, hashlib, json, re, sys
from pathlib import Path
host, launches = sys.argv[1], sys.argv[2:]
out = []
def cls(e):
    m = re.search(r"raised (\w+)", e)
    return (m.group(1) if m else e.split("(")[0])[:60]
for run in launches:
    try:
        d = json.load(open(run + "/launch.json"))
    except (OSError, ValueError):
        continue
    for r in d.get("runs") or []:
        if r.get("real") or not r.get("attempts"):
            continue
        atts = []
        for i, a in enumerate(r["attempts"]):
            errs = [cls(str(e)) for e in ((a.get("outcome") or {}).get("sample_errors") or [])]
            ev = sorted(glob.glob(a.get("log_dir", "") + "/*.eval")) if a.get("log_dir") else []
            atts.append({"attempt": i + 1, "start": a.get("start"), "end": a.get("end"), "exit_code": a.get("exit_code"),
                         "terminal_errors": errs, "redrawn_because": (str(a["redrawn_because"])[:80] if a.get("redrawn_because") else None),
                         "eval_path": ev[0] if len(ev) == 1 else (ev if ev else None),
                         "eval_sha256": hashlib.sha256(Path(ev[0]).read_bytes()).hexdigest() if len(ev) == 1 else None})
        if not any("RateLimitError" in x["terminal_errors"] for x in atts):
            continue
        failed = [x for x in atts if x["terminal_errors"]]
        task = r["task"] if isinstance(r.get("task"), str) else (r.get("task") or {}).get("key")
        out.append({"host": host, "launch_run_dir": run, "launch": run.rstrip("/").split("/")[-1], "world": "sim", "cell": task,
                    "run_key": r["key"], "state": r.get("state"), "attempts": atts,
                    "pure_429": all(set(x["terminal_errors"]) == {"RateLimitError"} for x in failed)})
print(json.dumps(out))
