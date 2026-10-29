"""POST HOC (PLAN_FEATURES_POSTHOC.md): structural features of every bash output in B-dev's dev runs, computed on the
evened text the probe read (B-dev's evened(), imported unchanged). One Modal CPU container. Keys and numbers only: no
output text is printed, saved or returned.

  modal deploy experiments/2026-10-11_q3_q11_remainder/run_q11_features_modal.py
  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy python \\
      experiments/2026-10-11_q3_q11_remainder/run_q11_features_modal.py --features-out <path outside the repo>
"""

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
BDEV = HERE.parent / "2026-10-11_claimB_slips_evened_probe"
sys.path.insert(0, str(BDEV) if modal.is_local() else "/root")
import run_slips_evened_modal as B  # noqa: E402

APP = "probe-shift_aswin-surya_sub-q3-q11rem_features_kill-2026-11-11"
OUT = "claimB/q11_features_v1"
B_SCRIPT = "0abee4a4d8730c0d3330e4702d179dfd8d3a11a228279e18e84fc4baa1d4cd06"
B_DEPS = "33d539e9a646c5adfeb3b8069c708c3f696f1ac794e09c26a281448bf0c6a65d"
ERR_MARKERS = ("Traceback (most recent call last)", "Error", "error", "ERROR", "Exception", "exit code", "Exit code",
               "exit status", "returned non-zero", "command not found", "No such file or directory", "Permission denied",
               "fatal:", "FAILED", "Killed", "timed out", "Timeout")
TS = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{2}:\d{2}:\d{2}\b")
PATH = re.compile(r"(?:/[\w.\-]+){2,}")

app = modal.App(APP)
image = B.image.add_local_file(BDEV / "run_slips_evened_modal.py", "/root/run_slips_evened_modal.py")


def step_action(msgs) -> dict:
    """(step, action) -> tool-message index; copied from run_q11_rescore_modal.step_action (the reader's counting)."""
    out, step, action = {}, -1, 0
    for i, m in enumerate(msgs):
        if m["role"] == "assistant":
            step, action = step + 1, 0
        elif m["role"] == "tool":
            out[(step, action)] = i
            action += 1
    return out


def features(c: str, err_line) -> dict:
    lines = c.split("\n") if c else []
    n = len(lines)
    return {"n_chars": len(c), "n_lines": c.count("\n") + 1 if c else 0, "empty": c.strip() == "",
            "trailing_newline": c.endswith("\n"),
            "stderr_lines": sum(err_line.fullmatch(x) is not None for x in lines),
            "err_marker": any(m in c for m in ERR_MARKERS),
            "blank_share": sum(x.strip() == "" for x in lines) / n if n else 0.0,
            "max_line_len": max((len(x) for x in lines), default=0), "ansi": "\x1b[" in c,
            "ts_tokens": len(TS.findall(c)), "path_tokens": len(PATH.findall(c)),
            "digit_share": sum(ch.isdigit() for ch in c) / max(len(c), 1), "leading_ws": c[:1].isspace() if c else False,
            "trailing_ws_share": sum(x.endswith((" ", "\t")) for x in lines) / n if n else 0.0,
            "json_like": c.lstrip()[:1] in ("{", "[") and c.strip() != "", "tab": "\t" in c,
            "non_ascii": any(ord(ch) > 127 for ch in c),
            "repeated_line_share": 1 - len(set(lines)) / n if n else 0.0}


@app.function(image=image, cpu=8, memory=32768, volumes={"/acts": B.P.vol}, timeout=20 * 60, retries=0)
def compute() -> dict:
    from probe_shift.scrub import STDERR_LINE

    if (B.SCRIPT_SHA, B.DEPS_SHA) != (B_SCRIPT, B_DEPS):
        raise ValueError("B-dev's code here differs from the code that ran")
    B.P.vol.reload()
    err_line = re.compile(STDERR_LINE.pattern.replace(r")\n?", ")"), re.I)  # the per-line form even_text uses
    rows, nonstr, tools = [], 0, {}
    for (cell, arm), r in sorted(B.dev_runs().items()):
        msgs = r["messages"]
        calls, ev = B.calls_of(msgs), B.evened(msgs)[0]
        for (step, action), i in step_action(msgs).items():
            tool = calls.get(i, (msgs[i].get("tool_name"), None))[0]
            tools[tool] = tools.get(tool, 0) + 1
            if tool != "bash":
                continue
            c = ev[i]["content"]
            if not isinstance(c, str):
                nonstr += 1
                continue
            rows.append({"cell": cell, "arm": arm, "environment": r["environment"], "main_task": r["main_task"], "step": step,
                         "action": action, **features(c, err_line)})
    body = json.dumps(rows).encode()
    (Path("/acts") / OUT).mkdir(parents=True, exist_ok=True)
    (Path("/acts") / OUT / "features.json").write_bytes(body)
    B.P.vol.commit()
    return {"rows": rows, "n_rows": len(rows), "nonstr_bash": nonstr, "tool_messages": tools,
            "features_sha256": hashlib.sha256(body).hexdigest(), "volume_path": f"{OUT}/features.json",
            "b_script_sha256": B_SCRIPT, "b_deps_sha256": B_DEPS}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-out", required=True)
    args = ap.parse_args()
    call = modal.Function.from_name(APP, "compute").spawn()
    print("spawned", call.object_id, flush=True)
    res = call.get()
    rows = res.pop("rows")
    Path(args.features_out).write_text(json.dumps(rows))
    res["features_local_copy_sha256"] = hashlib.sha256(Path(args.features_out).read_bytes()).hexdigest()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(HERE / "results" / f"q11_features_run_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("n_rows", "nonstr_bash", "tool_messages", "features_sha256")}, indent=1))
