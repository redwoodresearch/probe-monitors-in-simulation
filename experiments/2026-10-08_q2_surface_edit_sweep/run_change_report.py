"""Which outputs each sweep arm changed, per run, by step and action (Modal CPU, pennies), for q3's split of the score
change into edited vs unchanged later outputs. Diffs every staged item (2026-10-08_q2_surface_edits_inputs_v1) against
its original in the base dump. Format as plan A's stage report: {key: {"changed <arm>": [{step, action, msg, tool,
command}]}}. Step = the agent turn's index; action = the call's index in that turn (as the swap's helpers count).

  uv run modal run experiments/2026-10-08_q2_surface_edit_sweep/run_change_report.py
"""
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
SWAP = HERE.parent / "2026-10-07_q2_rendering_swap"
BASE, STAGED = "2026-10-06_la_segment-end_qwen3-32b_v1", "2026-10-08_q2_surface_edits_inputs_v1"
app = modal.App("probe-shift_aswin-surya_sub-q2-surfedit_report_kill-2026-11-08")
acts = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
image = modal.Image.debian_slim(python_version="3.12").add_local_file(SWAP / "rerender.py", "/root/rerender.py")


@app.function(image=image, volumes={"/acts": acts}, cpu=2, memory=16384, timeout=1800, retries=0)
def report() -> dict:
    import sys

    sys.path.insert(0, "/root")
    from rerender import args_of

    base = {}
    for shard in sorted((Path("/acts") / BASE / "inputs").glob("shard_*.jsonl")):
        for it in map(json.loads, shard.read_text().splitlines()):
            if it["arm"] in ("real-attack-e1", "real-honest"):
                base[(it["arm"], it["key"])] = it["messages"]
    out, totals = {}, {}
    for it in map(json.loads, (Path("/acts") / STAGED / "inputs" / "shard_0.jsonl").read_text().splitlines()):
        orig_arm = next(a for a in ("real-attack-e1", "real-honest") if it["arm"].startswith(a))
        orig, new, rows = base[(orig_arm, it["key"])], it["messages"], []
        step, action, calls = -1, 0, []
        for i, m in enumerate(orig):
            if m["role"] == "assistant":
                step, action = step + 1, 0
                calls = [(c["function"]["name"], args_of(c["function"])) for c in m.get("tool_calls") or []]
                continue
            if m["role"] == "tool":
                tool, a = calls[action] if action < len(calls) else (None, {})
                if m != new[i]:
                    rows.append({"step": step, "action": action, "msg": i, "tool": tool, "command": a.get("command")})
                action += 1
        if any(a != b for a, b in zip(orig, new)) and not rows:
            raise ValueError(f"{it['key']} {it['arm']}: a non-tool message changed")
        out.setdefault(it["key"], {})[f"changed {it['arm']}"] = rows
        totals[it["arm"]] = totals.get(it["arm"], 0) + len(rows)
    return {"base": BASE, "staged": STAGED, "outputs changed per arm": totals, "runs": out}


@app.local_entrypoint()
def main():
    res = report.remote()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    f = HERE / "results" / f"change_report_{stamp}.json"
    f.parent.mkdir(exist_ok=True)
    f.write_text(json.dumps(res))
    print(f, json.dumps(res["outputs changed per arm"]))
