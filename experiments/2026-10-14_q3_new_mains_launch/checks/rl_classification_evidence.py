"""Append 35 item 1, ON THE BOX (read-only): classification evidence for rate-limit-failed simulated runs, from the
ORIGINAL logs only: per run, every attempt's .eval path and sha256, the sample error's exception chain head (the class a
RetryError wraps), the error's timestamp, and the model of the last model call before the error (policy vs simulator),
plus the number of simulator model calls before it. Metadata only (classes, model names, counts, times, hashes).
  <sept venv python> rl_classification_evidence.py <launch dir> <cell> [<cell> ...]"""
import glob, hashlib, json, re, sys
from pathlib import Path
from inspect_ai.event import ModelEvent
from inspect_ai.log import read_eval_log
run, cells = sys.argv[1], set(sys.argv[2:])
d = json.load(open(run + "/launch.json"))
policy = d.get("policy")
out = []
for r in d["runs"]:
    task = r["task"] if isinstance(r.get("task"), str) else r["task"]["key"]
    if r["real"] or task not in cells:
        continue
    atts = []
    for i, a in enumerate(r["attempts"]):
        ev = sorted(glob.glob(a["log_dir"] + "/*.eval"))
        rec = {"attempt": i + 1, "start": a.get("start"), "end": a.get("end"), "eval_path": ev[0] if len(ev) == 1 else ev,
               "eval_sha256": hashlib.sha256(Path(ev[0]).read_bytes()).hexdigest() if len(ev) == 1 else None}
        if len(ev) == 1:
            s = read_eval_log(ev[0]).samples[0]
            msg = (s.error.message if s.error else "") or ""
            m = re.search(r"raised (\w+)", msg)
            models = [e for e in s.events if isinstance(e, ModelEvent)]
            rec |= {"sample_error_class": msg.split("(")[0][:40] if msg else None, "wrapped_exception": m.group(1) if m else None,
                    "last_model_call": models[-1].model if models else None, "last_model_call_time": str(models[-1].timestamp) if models else None,
                    "simulator_calls": sum(1 for e in models if e.model != policy), "policy_calls": sum(1 for e in models if e.model == policy)}
        atts.append(rec)
    out.append({"launch": run.rstrip("/").split("/")[-1], "run_key": r["key"], "cell": task, "state": r["state"], "attempts": atts,
                "infra_rate_limit": all(x.get("wrapped_exception") == "RateLimitError" for x in atts if x.get("sample_error_class"))
                and any(x.get("sample_error_class") for x in atts)})
print(json.dumps({"policy": policy, "runs": out}))
