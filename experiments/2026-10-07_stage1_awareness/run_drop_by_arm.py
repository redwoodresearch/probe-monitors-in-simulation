"""Which calls run_score_stage1.py dropped, by arm. Modal CPU, no activations read.

The scorer keeps a call only when its last parameter is the command parameter (the v4 extractions hold each call's
last-parameter token), and it counted the drops over all arms together. If the drop rate differs between simulated
and real arms, the Es compare different subsets of calls. This recounts the same rule (same inputs, same order of
checks as run_score_stage1.py) by arm, by environment, by tool × call-position cell, and by the tool's subcommand
(text_editor's `command`; for other tools, the parameter that came last).

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.2' --with numpy modal run \
      experiments/2026-10-07_stage1_awareness/run_drop_by_arm.py
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REPO = HERE.parent.parent
APP = "probe-shift_aswin-surya_sub-stage1_drops_kill-2026-11-06"
Q35 = "probe-shift_aswin-surya_sub-qwen35_acts_kill-2026-11-06"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
CASES = {"a_original": ("2026-10-06_la_segment-end-v4_qwen35-27b_alllayers_v1", "2026-10-06_la_segment-end-v4_qwen3-32b_v1"),
         "b_level7": ("2026-10-06_la_segment-end-v4-scrub7_qwen35-27b_alllayers_v1", "2026-10-06_la_segment-end-v4-scrub7_qwen3-32b_v1")}

app = modal.App(APP)
q35_vol = modal.Volume.from_name(Q35, version=2)
acts_vol = modal.Volume.from_name(ACTS, version=2)
image = modal.Image.debian_slim(python_version="3.12").add_local_python_source("probe_shift")


@app.function(image=image, cpu=2, memory=16384, timeout=1800, volumes={"/q35": q35_vol.read_only(), "/acts": acts_vol.read_only()})
def count() -> dict:
    from collections import Counter, defaultdict

    from probe_shift.reader import command_param

    out = {}
    for case, (ext, staged) in CASES.items():
        ext_dir, st_dir = Path("/q35") / ext, Path("/acts") / staged / "inputs"
        calls = {}
        for shard in sorted(st_dir.glob("shard_*.jsonl")):
            for line in shard.read_text().splitlines():
                it = json.loads(line)
                turns = [m_ for m_ in it["messages"] if m_["role"] == "assistant"]
                calls[(it["arm"], it["key"])] = {(si, ai): c["function"] for si, t in enumerate(turns) for ai, c in enumerate(t.get("tool_calls") or [])}
        by = defaultdict(Counter)  # (group kind, arm, group value) -> outcome counts
        for p_ in sorted(ext_dir.glob("meta/*/*.json")):
            m_ = json.loads(p_.read_text())
            cs, arm = calls[(m_["arm"], m_["key"])], m_["arm"]
            n_call = 0
            for pos in m_["positions"]:
                if pos["kind"] != "call_end":
                    continue
                n_call += 1
                f = cs[(pos["step"], pos["action"])]
                args = f.get("arguments") or {}
                cp = command_param(f["name"], args)
                outcome = ("not_kept" if not pos["kept"] else "no_command_param" if cp is None
                           else "command_not_last" if list(args)[-1] != cp else "read")
                tool = "bash" if (f["name"] == "bash" or f["name"].endswith("_bash")) else ("text_editor" if f["name"] == "text_editor" else "other")
                bucket = 0 if n_call <= 10 else (1 if n_call <= 30 else 2)
                sub = f"{tool}:{args.get('command')}" if tool == "text_editor" else f"{tool}:last={list(args)[-1] if args else None}"
                for kind, val in (("all", "all"), ("environment", m_["environment"]), ("cell", f"{tool}|{bucket}"), ("subcommand", sub)):
                    by[(kind, arm, val)][outcome] += 1
        res = defaultdict(lambda: defaultdict(dict))
        for (kind, arm, val), c in sorted(by.items()):
            n = sum(c.values())
            res[kind][val][arm] = {"n": n, **dict(c), "share_command_not_last": c["command_not_last"] / n, "share_read": c["read"] / n}
        out[case] = res
    return json.loads(json.dumps(out))


@app.local_entrypoint()
def main():
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = {"stamp": stamp, "git": head, "app": APP, "cases": count.remote()}
    path = HERE / "results" / f"drop_by_arm_{stamp}.json"
    with path.open("x") as f:
        f.write(json.dumps(out, indent=1))
    for case, res in out["cases"].items():
        print(case)
        for arm, v in res["all"]["all"].items():
            print(f"  {arm:15s} calls {v['n']:6d}  command_not_last {v['share_command_not_last']:.3f}  read {v['share_read']:.3f}")
    print("->", path)
