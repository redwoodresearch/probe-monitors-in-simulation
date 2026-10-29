"""Apply RELABEL_RULE_drainstop.txt to one run (ON THE BOX, the September venv: inspect_ai reads the .eval headers only):
every run recorded "stopped" is DONE iff its last attempt's exit_code is 0 and its .eval status is "success" with exactly
one sample; else it stays "stopped". Writes <run>/relabel_drainstop.json (per run: key, arm, exit code, eval status,
samples, label) and prints the counts. Never edits launch.json.
  <sept venv python> relabel_drainstop.py <run dir>"""
import json, sys
from collections import Counter
from pathlib import Path
from inspect_ai.log import read_eval_log
run = Path(sys.argv[1])
d = json.loads((run / "launch.json").read_text())
rows = []
for r in d["runs"]:
    if r["state"] != "stopped":
        continue
    a = r["attempts"][-1]
    evals = sorted(Path(a["log_dir"]).glob("*.eval"))
    status, samples = None, None
    if len(evals) == 1:
        h = read_eval_log(str(evals[0]), header_only=True)
        status, samples = h.status, (h.results.total_samples if h.results else None)
    done = a.get("exit_code") == 0 and status == "success" and samples == 1
    rows.append({"key": r["key"], "real": r["real"], "exit_code": a.get("exit_code"), "evals": len(evals), "eval_status": status,
                 "samples": samples, "label": "done" if done else "stopped"})
(run / "relabel_drainstop.json").write_text(json.dumps(rows, indent=1))
print(json.dumps({"stopped": len(rows), "labels": dict(Counter(x["label"] for x in rows)),
                  "not_run": sum(1 for r in d["runs"] if r["state"] == "not run")}))
