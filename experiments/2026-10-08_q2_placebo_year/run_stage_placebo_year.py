"""Placebos and year swap, staging (Modal CPU; config/placebo_year.yaml; predictions committed first, here and in
../2026-10-08_q2_year_swap). Three arms in one staged run, for one extraction with one model load:
  <arm>-gutter-colon   real str_replace diffs: `{n:>4}|{sign} {code}` -> `{n:>4}:{sign} {code}` (the sweep's attack and
                       honest runs; placebo for strip-gutter)
  <arm>-note-swap      real insert outputs ending in the review note: the note -> a neutral sentence of the same length
                       (placebo for no-note)
  sim-attack-year2026  simulated tool outputs: 2025 -> 2026 in ISO dates and log-style dates (the swap's 100 runs and
                       the extension's 117)
Runs are chosen as the sweep chose them (../2026-10-08_q2_surface_edit_sweep/run_stage_edits.py); only changed runs are
staged; every changed message is checked to be a target output changed only as its arm says. Never overwrites.

  aws s3 cp s3://individual-aswin-surya-probe-shift-kill-2027-04-06/experiments/2026-10-07_q2_rendering_swap/config/swap_runs_ext.json \
      experiments/2026-10-07_q2_rendering_swap/config/
  uv run modal run experiments/2026-10-08_q2_placebo_year/run_stage_placebo_year.py
"""
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
SWAP = HERE.parent / "2026-10-07_q2_rendering_swap"
app = modal.App("probe-shift_aswin-surya_sub-q2-placebo-year_stage_kill-2026-11-08")
acts = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
image = modal.Image.debian_slim(python_version="3.12").add_local_file(SWAP / "rerender.py", "/root/rerender.py")
GUTTER = re.compile(r"^(\s*\d+)\|([ +\-] .*)$")
YEAR = [re.compile(r"\b2025(?=-\d{2}-\d{2}(?!\d))"), re.compile(r"(?<=\b\d{2}/(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)/)2025(?=:)")]


def cell_of(key: str) -> tuple[str, str]:
    parts = key.rsplit("__", 2)[-2].split(":")
    return parts[0], parts[1] if len(parts) == 3 else ""


def gutter_colon(out: str) -> str:
    head, sep, body = out.partition("+++ edited_version\n")
    return head + sep + "\n".join(f"{m.group(1)}:{m.group(2)}" if (m := GUTTER.match(x)) else x for x in body.split("\n")) if sep else out


def swap_years(text: str) -> str:
    for p in YEAR:
        text = p.sub("2026", text)
    return text


@app.function(image=image, volumes={"/acts": acts}, cpu=2, memory=16384, timeout=3600, retries=0)
def stage(cfg: dict, cells: list[list[str]], sim_keys: list[str]) -> dict:
    import sys

    sys.path.insert(0, "/root")
    from rerender import DIFF, REVIEW, args_of

    if len(cfg["note_placebo"]) != len(REVIEW):
        raise ValueError(f"the note placebo has {len(cfg['note_placebo'])} characters, the note {len(REVIEW)}")
    dst = Path("/acts") / cfg["staged_run"]
    if dst.exists():
        raise FileExistsError(f"{dst} exists: a staged run is never overwritten")
    cells, want, attack, honest, sims = {tuple(c) for c in cells}, set(sim_keys), [], {}, []
    for shard in sorted((Path("/acts") / cfg["base"] / "inputs").glob("shard_*.jsonl")):
        for it in map(json.loads, shard.read_text().splitlines()):
            main, side = cell_of(it["key"])
            if it["arm"] == "real-attack-e1" and (main, side) in cells:
                attack.append(it)
            elif it["arm"] == "real-honest" and main in {m for m, _ in cells}:
                honest.setdefault(main, []).append(it)
            elif it["arm"] == "sim-attack" and it["key"] in want:
                sims.append(it)
    honest = [min(v, key=lambda x: x["key"]) for v in honest.values()]  # as the sweep: the first key per main task
    if len(attack) != len(cells) or len(sims) != len(want):
        raise ValueError(f"{len(attack)} attack runs for {len(cells)} cells; {len(sims)} of {len(want)} simulated runs")
    out, report = [], {}

    def add(it, arm, new, allowed):
        changed = [i for i, (a, b) in enumerate(zip(it["messages"], new)) if a != b]
        if not set(changed) <= allowed:
            raise ValueError(f"{it['key']} {arm}: changed messages {changed} outside the targets")
        r = report.setdefault(arm, {"runs": 0, "outputs": 0, "unchanged runs": 0})
        if changed:
            r["runs"], r["outputs"] = r["runs"] + 1, r["outputs"] + len(changed)
            out.append(it | {"arm": arm, "messages": new})
        else:
            r["unchanged runs"] += 1

    for it in sorted(attack, key=lambda x: x["key"]) + sorted(honest, key=lambda x: x["key"]):
        msgs, pending, edits = it["messages"], [], {}
        for i, m in enumerate(msgs):
            if m["role"] == "assistant":
                pending = [(c["function"]["name"], args_of(c["function"])) for c in m.get("tool_calls") or []]
            elif m["role"] == "tool" and pending:
                tool, a = pending.pop(0)
                if tool == "text_editor" and isinstance(m["content"], str):
                    edits[i] = a.get("command")
        sr = {i for i, c in edits.items() if c == "str_replace" and DIFF in msgs[i]["content"][:300]}
        ins = {i for i, c in edits.items() if c == "insert" and msgs[i]["content"].endswith("\n" + REVIEW)}
        add(it, f"{it['arm']}-gutter-colon", [m | {"content": gutter_colon(m["content"])} if i in sr else m for i, m in enumerate(msgs)], sr)
        add(it, f"{it['arm']}-note-swap", [m | {"content": m["content"][: -len(REVIEW)] + cfg["note_placebo"]} if i in ins else m
                                           for i, m in enumerate(msgs)], ins)
    for it in sorted(sims, key=lambda x: x["key"]):
        tools = {i for i, m in enumerate(it["messages"]) if m["role"] == "tool" and isinstance(m["content"], str)}
        new = [m | {"content": swap_years(m["content"])} if i in tools else m for i, m in enumerate(it["messages"])]
        for i in tools:  # only those years may differ
            if new[i]["content"].replace("2026", "2025") != it["messages"][i]["content"].replace("2026", "2025"):
                raise ValueError(f"{it['key']} message {i}: changed beyond the year")
        add(it, "sim-attack-year2026", new, tools)
    (dst / "inputs").mkdir(parents=True)
    (dst / "inputs" / "shard_0.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out))
    (dst / "stage_report.json").write_text(json.dumps(report))
    acts.commit()
    return {"items": len(out), "attack runs": len(attack), "honest runs": len(honest), "simulated runs": len(sims), "report": report}


@app.local_entrypoint()
def main():
    import yaml

    cfg = yaml.safe_load((HERE / "config" / "placebo_year.yaml").read_text())
    swap = json.loads((HERE / cfg["swap_runs"]).read_text())["runs"]
    ext = json.loads((HERE / cfg["ext_runs"]).read_text())["runs"]
    cells = sorted({cell_of(k) for k in swap})
    res = stage.remote(cfg, [list(c) for c in cells], sorted(set(swap) | set(ext)))
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = HERE / "results" / f"stage_report_{stamp}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res | {"config": cfg}, indent=1))
    print(out, json.dumps(res["report"], indent=1))
