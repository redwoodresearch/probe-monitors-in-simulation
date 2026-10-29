"""Quick check: does the tool-output detector's real-vs-simulated gap sit on error outputs or on outputs with a
timestamp? Modal CPU, no model calls. Flags each tool output the #36 readout reads (data 307cbfee925c, original text):
- error: the failure-hiding study's text definition (experiments/2026-10-07_sim_failure_hiding/analyze_failure_hiding.py):
  a text_editor output matching TE_FAIL, or a bash-like output matching BASH_FAIL;
- timestamp: any of scrub level 1's time patterns (probe_shift.scrub.TIME_PATTERNS).
The texts are the ones run_tfidf_outputs.py staged on the work Volume (aligned to #36's output_end rows). Returns the
two flags per row, no text; analyze_output_flags.py splits the gap.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.2' --with numpy modal run \
      experiments/2026-10-08_q3_swap_cue/run_output_flags.py --texts-run q3_tfidf_outputs_<stamp>
"""

import gzip
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q3swapcue_score_kill-2026-11-06"
WORK = "probe-shift_aswin-surya_sub-q3attack_work_kill-2026-11-06"
DATA_ID = "307cbfee925c"
# copied from experiments/2026-10-07_sim_failure_hiding/analyze_failure_hiding.py (lines 46-56)
TE_FAIL = (r"does not exist\. Please provide a valid path|No replacement was performed|File already exists at|"
           r"Cannot overwrite files|Multiple occurrences of old_str|Invalid `view_range`|"
           r"Invalid `insert_line`|is a directory\. |Ran into \[Errno \d+\]")
BASH_FAIL = (r"command not found|No such file or directory|Permission denied|cannot access|"
             r"Traceback \(most recent call last\)|syntax error|Connection refused|npm ERR!|"
             r"timed out before completing|failed because it timed out|exited unexpectedly|"
             r"Segmentation fault|core dumped|\bpanicked at\b|\b[1-9]\d* failed\b|ERROR at (?:setup|teardown) of|"
             r"^(?:bash|sh|/bin/sh): |^fatal: |^error(?:\[E\d+\])?: |^Error: |^ERROR[: ]|^E: |"
             r"^[A-Z][A-Za-z]+Error: |^FAILED|^FAIL\s|^Diff in \S+:\d+:")

app = modal.App(APP)
work_vol = modal.Volume.from_name(WORK, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3").add_local_python_source("probe_shift")


@app.function(image=image, cpu=2, memory=16384, timeout=1800, volumes={"/work": work_vol.read_only()})
def flags(texts_run: str) -> dict:
    import numpy as np

    from probe_shift.scrub import TIME_PATTERNS

    d = Path("/work/data") / DATA_ID
    t = np.load(d / "tables.npz")
    tools = json.loads((d / "summary.json").read_text())["tools"]
    with gzip.open(Path("/work/runs") / texts_run / "texts.json.gz", "rt") as fh:
        base = json.load(fh)["base"]
    if len(base) != len(t["output_end__tool"]):
        raise ValueError(f"{len(base)} texts vs {len(t['output_end__tool'])} rows")
    te, sh = re.compile(TE_FAIL, re.I), re.compile(BASH_FAIL, re.M)
    names = {i: n for n, i in tools.items()} if isinstance(tools, dict) else dict(enumerate(tools))
    err, ts, tool_name = [], [], []
    for text, tid in zip(base, t["output_end__tool"]):
        name = names.get(int(tid), "")
        tool_name.append(name)
        err.append(bool((name == "text_editor" and te.search(text)) or (name.endswith("bash") and sh.search(text))))
        ts.append(any(p.search(text) for p, _ in TIME_PATTERNS))
    return {"error": err, "timestamp": ts, "tool": tool_name}


@app.local_entrypoint()
def main(texts_run: str):
    out = flags.remote(texts_run)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"output_flags_{stamp}.json.gz"
    with gzip.open(path, "xt") as fh:  # per-row flags and tool names only
        json.dump({"data_id": DATA_ID, "texts_run": texts_run, "rows": len(out["error"]), **out}, fh)
    print(f"{sum(out['error'])} error rows, {sum(out['timestamp'])} timestamp rows of {len(out['error'])} ->", path)
