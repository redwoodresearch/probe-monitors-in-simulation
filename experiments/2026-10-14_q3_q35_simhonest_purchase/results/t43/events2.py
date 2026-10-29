import json, glob
from inspect_ai.log import read_eval_log
f = glob.glob("/home/aswin/phase2/runs/2026-10-14_160932_simh_cont/sim_honest@september/t43_auto_workflow/*.eval")[0]
log = read_eval_log(f)
print("status", log.status, "error", str(log.error)[:300] if log.error else None)
s = log.samples[0]
print("sample error:", str(s.error.message)[:400] if s.error else None)
tb = (s.error.traceback if s.error else "") or ""
print("traceback tail (frames only):")
for line in tb.splitlines()[-40:]:
    if line.strip().startswith("File ") or "Error" in line or "Timeout" in line:
        print("  ", line.strip()[:220])
print("total_time", s.total_time, "working_time", s.working_time)
for e in s.events:
    t = e.event
    row = {"ts": str(getattr(e, "timestamp", "")), "event": t}
    if t == "model":
        out = e.output
        ch = out.choices[0] if (out and out.choices) else None
        u = out.usage if out else None
        row.update(model=e.model, error=(str(e.error)[:300] if e.error else None), retries=getattr(e, "retries", None),
                   completed=str(getattr(e, "completed", "")), working_time=getattr(e, "working_time", None),
                   in_tok=getattr(u, "input_tokens", None), out_tok=getattr(u, "output_tokens", None),
                   reasoning_tok=getattr(u, "reasoning_tokens", None),
                   stop_reason=(ch.stop_reason if ch else None))
    elif t == "tool":
        row.update(function=e.function, error=(str(e.error)[:200] if e.error else None), completed=str(getattr(e, "completed", "")))
    elif t in ("error", "sample_limit"):
        row.update(detail=str(getattr(e, "error", "") or getattr(e, "message", ""))[:400], limit=str(getattr(e, "type", "")))
    elif t == "logger":
        m = e.message
        lvl = getattr(m, "level", "")
        msg = str(getattr(m, "message", ""))
        if lvl not in ("warning", "error") and "retry" not in msg.lower() and "timeout" not in msg.lower():
            continue
        row.update(level=lvl, msg=msg[:260])
    else:
        continue
    print(json.dumps(row, default=str))
