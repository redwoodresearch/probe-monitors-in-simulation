"""Surface-edit sweep, staging (Modal CPU): simulator tells applied in reverse to real runs, one edit at a time
(config/sweep.yaml; PREDICTIONS.md was committed first).

Reads the base dump's staged reader messages and writes a new staged run for the extractor
(../2026-10-06_all_layers_extraction/config/alllayers_q2_surface_edits.yaml). Each edit touches only its own outputs:
  strip-gutter  real text_editor str_replace diffs: `{n:>4}|{sign} {code}` hunk lines back to `{sign}{code}`
  add-newline   real str_replace diffs that end without a newline get one (the simulator's ending)
  no-note       real insert outputs ending in the review note lose it (the simulator writes none)
Only runs an edit changes are staged, as arm `<arm>-<edit>`; the control arm `<arm>-control` repeats some runs unchanged.
Before writing, every changed message is checked to be one the edit targeted, and the staged run is never overwritten.
Returns per-edit counts and writes 3 before/after pairs per edit to data/ (raw text, not results/).

  uv run modal run experiments/2026-10-08_q2_surface_edit_sweep/run_stage_edits.py
"""
import gzip
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
SWAP = HERE.parent / "2026-10-07_q2_rendering_swap"
ROOT = HERE.parents[1] if len(HERE.parents) > 1 else HERE
app = modal.App("probe-shift_aswin-surya_sub-q2-surface-edits_stage_kill-2026-11-08")
acts = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
image = (modal.Image.debian_slim(python_version="3.12")
         .add_local_file(SWAP / "rerender.py", "/root/rerender.py"))
GUTTER = re.compile(r"^\s*\d+\|([ +\-]) (.*)$")


def cell_of(key: str) -> tuple[str, str]:
    """(main task, side task or '') from a key `<prefix>__<main>:<side>:<hash>__<epoch>`."""
    parts = key.rsplit("__", 2)[-2].split(":")
    return parts[0], parts[1] if len(parts) == 3 else ""


def edit_output(edit: str, command: str | None, out: str, diff: str, review: str) -> str:
    """The output after `edit`, or the same string when the edit does not apply to it."""
    if command == "str_replace" and diff in out[:300] and edit in ("strip-gutter", "add-newline"):
        if edit == "add-newline":
            return out if out.endswith("\n") else out + "\n"
        head, sep, body = out.partition("+++ edited_version\n")
        return head + sep + "\n".join(f"{m.group(1)}{m.group(2)}" if (m := GUTTER.match(x)) else x for x in body.split("\n")) if sep else out
    if command == "insert" and edit == "no-note" and out.endswith("\n" + review):
        return out[: -len("\n" + review)]
    return out


@app.function(image=image, volumes={"/acts": acts}, cpu=2, memory=16384, timeout=3600, retries=0)
def stage(cfg: dict, cells: list[list[str]]) -> dict:
    import sys

    sys.path.insert(0, "/root")
    from rerender import DIFF, REVIEW, args_of

    dst = Path("/acts") / cfg["staged_run"]
    if dst.exists():
        raise FileExistsError(f"{dst} exists: a staged run is never overwritten")
    cells, items = {tuple(c) for c in cells}, {}
    for shard in sorted((Path("/acts") / cfg["base"] / "inputs").glob("shard_*.jsonl")):
        for it in map(json.loads, shard.read_text().splitlines()):
            main, side = cell_of(it["key"])
            if it["arm"] == "real-attack-e1" and (main, side) in cells:
                items[it["key"]] = it
            elif it["arm"] == "real-honest" and main in {m for m, _ in cells}:
                items.setdefault(("honest", main), []).append(it)
    attack = sorted((v for k, v in items.items() if isinstance(k, str)), key=lambda x: x["key"])
    honest = sorted((min(v, key=lambda x: x["key"]) for k, v in items.items() if isinstance(k, tuple)), key=lambda x: x["key"])
    if len(attack) != len(cells):
        raise ValueError(f"{len(attack)} real attack runs for {len(cells)} cells")
    out, report, pairs = [], {}, []
    half = cfg["n_control"] // 2
    for it in attack[:half] + honest[:half]:
        out.append(it | {"arm": f"{it['arm']}-control"})
    for it in attack + honest:
        msgs, pending, targets = it["messages"], [], []
        for i, m in enumerate(msgs):  # calls and outputs pair in order, as in the swap's helpers
            if m["role"] == "assistant":
                pending = [args_of(c["function"]) | {"_tool": c["function"]["name"]} for c in m.get("tool_calls") or []]
            elif m["role"] == "tool" and pending:
                a = pending.pop(0)
                if a["_tool"] == "text_editor":
                    targets.append((i, a.get("command")))
        for edit in cfg["edits"]:
            new = [dict(m) for m in msgs]
            for i, command in targets:
                if isinstance(msgs[i]["content"], str):
                    new[i]["content"] = edit_output(edit, command, msgs[i]["content"], DIFF, REVIEW)
            changed = [i for i, (a, b) in enumerate(zip(msgs, new)) if a != b]
            if not set(changed) <= {i for i, _ in targets}:
                raise ValueError(f"{it['key']} {edit}: changed messages {changed} outside the text_editor outputs")
            r = report.setdefault(f"{it['arm']}-{edit}", {"runs": 0, "outputs": 0, "unchanged runs": 0})
            if not changed:
                r["unchanged runs"] += 1
                continue
            r["runs"], r["outputs"] = r["runs"] + 1, r["outputs"] + len(changed)
            out.append(it | {"arm": f"{it['arm']}-{edit}", "messages": new})
            if sum(p["edit"] == edit for p in pairs) < 3:
                pairs.append({"key": it["key"], "edit": edit, "before": msgs[changed[0]]["content"], "after": new[changed[0]]["content"]})
    (dst / "inputs").mkdir(parents=True)
    (dst / "inputs" / "shard_0.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out))
    (dst / "stage_report.json").write_text(json.dumps(report))
    acts.commit()
    return {"items": len(out), "attack runs": len(attack), "honest runs": len(honest), "control": 2 * half, "report": report, "pairs": pairs}


@app.local_entrypoint()
def main():
    import yaml  # local only: stage() gets the config as an argument

    CFG = yaml.safe_load((HERE / "config" / "sweep.yaml").read_text())
    runs = json.loads((HERE / CFG["runs_file"]).read_text())["runs"]
    cells = sorted({cell_of(k) for k in runs})
    res = stage.remote(CFG, [list(c) for c in cells])
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    pairs = ROOT / "data" / "surface_edits" / f"pairs_{stamp}.json.gz"
    pairs.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(pairs, "wt") as f:
        json.dump(res.pop("pairs"), f)
    out = HERE / "results" / f"stage_report_{stamp}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res | {"config": CFG, "pairs file (not in git)": str(pairs.relative_to(ROOT))}, indent=1))
    print(out, json.dumps(res["report"], indent=1))
