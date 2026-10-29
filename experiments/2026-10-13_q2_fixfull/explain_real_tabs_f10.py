"""F10 (PLAN_F10.md, Amendment 8a): explain every real text_editor message F10 would change (the invariance control's 9, 9
and 35). Modal CPU; counts only (no text): per real arm, the changed messages by command, environment, whether the output
is a `cat -n` view, the line class of each tabbed line, and whether a tabbed `cat -n` line's content tab sits in the content
(the real tool did not expand it) or the line failed the `cat -n` pattern.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' modal run experiments/2026-10-13_q2_fixfull/explain_real_tabs_f10.py
"""
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import stage_f10 as S  # noqa: E402  (its f10_text, content_tabs, constants; its app is not run here)

app = modal.App("probe-shift_aswin-surya_sub-q2-f10_explain_kill-2026-11-13")
image = (modal.Image.debian_slim(python_version="3.12").add_local_python_source("probe_shift")
         .add_local_file(S.SWAP / "rerender.py", "/root/swap/rerender.py").add_local_file(S.SWAP / "str_replace_tells.py", "/root/swap/str_replace_tells.py")
         .add_local_file(S.FSIM / "fixes.py", "/root/fsim/fixes.py").add_local_file(HERE / "stage_f10.py", "/root/stage_f10.py"))  # beside this module in the container


@app.function(image=image, volumes={"/acts": S.acts}, cpu=4, memory=32768, timeout=1800, retries=0)
def explain(real_keys: dict) -> dict:
    try:
        sys.path[:0] = ["/root/swap", "/root/fsim"]
        import fixes
        import stage_f10 as T
        c = Counter()
        for p in sorted((Path("/acts") / T.BASE / "inputs").glob("shard_*.jsonl")):
            for line in p.read_text().splitlines():
                it = json.loads(line)
                if it["arm"] not in T.REAL or it["key"] not in real_keys[it["arm"]]:
                    continue
                a, m = it["arm"], it["messages"]
                for x in fixes.calls_of(m):
                    o = m[x["msg"]]["content"]
                    if x["tool"] != "text_editor" or not isinstance(o, str) or T.f10_text(o) == o:
                        continue
                    cmd = x["args"].get("command") if x["args"].get("command") in T.COMMANDS else "unknown"
                    c[f"{a}|changed|{cmd}"] += 1
                    c[f"{a}|changed|environment {it.get('environment')}"] += 1
                    c[f"{a}|changed|starts as a cat -n view"] += o.startswith("Here's the result of running `cat -n`")
                    for ln in o.split("\n"):
                        if "\t" not in ln:
                            continue
                        mm = T.CAT_N.match(ln)
                        if mm and "\t" in mm.group(2):
                            c[f"{a}|tabbed lines|cat -n line with a content tab (not expanded)"] += 1
                        elif not mm:
                            c[f"{a}|tabbed lines|not a cat -n line"] += 1
        return dict(sorted(c.items()))
    except Exception as e:  # noqa: BLE001 — type and location only
        import traceback
        tb = traceback.extract_tb(e.__traceback__)[-1]
        raise RuntimeError(f"explain failed: {type(e).__name__} at {Path(tb.filename).name}:{tb.lineno}") from None


@app.local_entrypoint()
def main():
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(Path(__file__))], cwd=HERE).returncode:
        raise SystemExit("commit this script before running")
    runs = json.loads(S.RUNS36.read_text())
    out = explain.remote({a: sorted(r["key"] for r in runs if r["arm"] == a) for a in S.REAL})
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(p := HERE / "results" / f"f10_real_tabs_{stamp}.json", "x") as fh:
        fh.write(json.dumps({"POST HOC": True, "plan": "PLAN_F10.md Amendment 8a", "git": head, "counts": out}, indent=1) + "\n")
    print(p)
    for k, v in out.items():
        print(f"  {k}: {v}")
