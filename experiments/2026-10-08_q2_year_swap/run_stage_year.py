"""Year swap, staging (Modal CPU; config/year.yaml; PREDICTIONS.md first): 2025 -> 2026 in the date-shaped strings of
simulated tool outputs (ISO dates, with or without a time, and log-style `08/Oct/2025:`), nothing else. Only runs with
a change are staged, as arm `sim-attack-year2026`, keys unchanged; every changed message is checked to be a tool output
whose only differences are those years. The staged run is never overwritten.

  uv run modal run experiments/2026-10-08_q2_year_swap/run_stage_year.py
"""
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
app = modal.App("probe-shift_aswin-surya_sub-q2-year-swap_stage_kill-2026-11-08")
acts = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
image = modal.Image.debian_slim(python_version="3.12")
YEAR = [re.compile(r"\b2025(?=-\d{2}-\d{2}(?!\d))"), re.compile(r"(?<=\b\d{2}/(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)/)2025(?=:)")]


def swap_years(text: str) -> str:
    for p in YEAR:
        text = p.sub("2026", text)
    return text


@app.function(image=image, volumes={"/acts": acts}, cpu=2, memory=16384, timeout=1800, retries=0)
def stage(cfg: dict, keys: list[str]) -> dict:
    dst = Path("/acts") / cfg["staged_run"]
    if dst.exists():
        raise FileExistsError(f"{dst} exists: a staged run is never overwritten")
    want, out, outputs = set(keys), [], 0
    items = [it for p in sorted((Path("/acts") / cfg["base"] / "inputs").glob("shard_*.jsonl"))
             for it in map(json.loads, p.read_text().splitlines()) if it["arm"] == "sim-attack" and it["key"] in want]
    if len(items) != len(want):
        raise ValueError(f"found {len(items)} of {len(want)} runs")
    for it in items:
        new = [m | {"content": swap_years(m["content"])} if m["role"] == "tool" and isinstance(m["content"], str) else m for m in it["messages"]]
        changed = [i for i, (a, b) in enumerate(zip(it["messages"], new)) if a != b]
        for i in changed:  # only those years differ
            if it["messages"][i]["role"] != "tool" or new[i]["content"].replace("2026", "2025") != it["messages"][i]["content"].replace("2026", "2025"):
                raise ValueError(f"{it['key']} message {i}: changed beyond the year")
        if changed:
            out.append(it | {"arm": cfg["arm"], "messages": new})
            outputs += len(changed)
    (dst / "inputs").mkdir(parents=True)
    (dst / "inputs" / "shard_0.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out))
    acts.commit()
    return {"runs": len(items), "runs changed": len(out), "outputs changed": outputs}


@app.local_entrypoint()
def main():
    import yaml

    cfg = yaml.safe_load((HERE / "config" / "year.yaml").read_text())
    keys = sorted({k for f in cfg["runs_files"] for k in json.loads((HERE / f).read_text())["runs"]})
    res = stage.remote(cfg, keys)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = HERE / "results" / f"stage_report_{stamp}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res | {"config": cfg}, indent=1))
    print(out, res)
