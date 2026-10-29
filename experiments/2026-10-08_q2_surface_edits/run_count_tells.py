"""Free count before a surface-edit sweep (Modal CPU, about $0.02): per edit, how many of the swap's 100 cells' real runs
it would change, and how common each tell is in real vs simulated runs. Reads the staged reader messages of the base
dump (all arms), the inputs the extractor would read. No model, no GPU. PREDICTIONS.md was committed first.

Edits (simulator tells applied to real text in reverse):
  a  strip the str_replace diff gutter        (real str_replace diffs with numbered hunk lines)
  b  add the trailing newline                  (real str_replace diffs that end without one)
  c  remove the insert review note             (real insert outputs that carry it)
  d  rewrite timestamps                        (outputs with an ISO, ls- or log-style timestamp)
  e  randomize IDs and hashes                  (outputs with a UUID, or a hex string: 12-15 characters with a
                                                  letter and a digit, or 16+ hex characters)

  uv run modal run experiments/2026-10-08_q2_surface_edits/run_count_tells.py
"""
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
SWAP = HERE.parent / "2026-10-07_q2_rendering_swap"
APP = "probe-shift_aswin-surya_sub-q2-surfedit_count_kill-2026-11-08"  # the first run used a 66-character name
BASE = "2026-10-06_la_segment-end_qwen3-32b_v1"
app = modal.App(APP)
acts = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
image = (modal.Image.debian_slim(python_version="3.12")
         .add_local_file(SWAP / "rerender.py", "/root/rerender.py")
         .add_local_file(SWAP / "str_replace_tells.py", "/root/str_replace_tells.py"))

MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
TIMESTAMPS = {"iso": re.compile(r"\b(\d{4})-\d{2}-\d{2}[T ]\d{2}:\d{2}"), "ls": re.compile(rf"\b{MONTH}\s+\d{{1,2}}\s+(?:\d{{2}}:\d{{2}}|\d{{4}})\b"),
              "log": re.compile(rf"\b\d{{2}}/{MONTH}/(\d{{4}}):\d{{2}}:\d{{2}}"), "epoch": re.compile(r"\b1[67]\d{8}\b")}
# `\b` except that `_` also separates, as in src/probe_shift/scrub.py's ID patterns (`run_46a7d3b90e5c` counts).
L, R = r"(?<![^\W_])", r"(?![^\W_])"
UUID = re.compile(rf"{L}[0-9a-f]{{8}}-[0-9a-f]{{4}}-[0-9a-f]{{4}}-[0-9a-f]{{4}}-[0-9a-f]{{12}}{R}", re.IGNORECASE)
HEX = re.compile(rf"{L}(?:(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{{12,}}|[a-f]{{16,}}){R}", re.IGNORECASE)  # all-letter: 16+ only
DIGITS = "0123456789abcdef"
LOW_ENTROPY = {c * 4 for c in DIGITS} | {DIGITS[i:i + 4] for i in range(13)} | {DIGITS[i:i + 4][::-1] for i in range(13)}  # every 4-run


def cell_of(key: str) -> tuple[str, str]:
    """(main task, side task or '') from a key `<prefix>__<main>:<side>:<hash>__<epoch>`."""
    parts = key.rsplit("__", 2)[-2].split(":")
    return parts[0], parts[1] if len(parts) == 3 else ""


@app.function(image=image, volumes={"/acts": acts}, cpu=2, memory=8192, timeout=1800, retries=0)
def count(cells: list[list[str]]) -> dict:
    import sys

    sys.path.insert(0, "/root")
    from rerender import DIFF, REVIEW, args_of
    from str_replace_tells import add_gutter

    cells = {tuple(c) for c in cells}
    mains = {m for m, _ in cells}
    runs, years, hexes, keys_seen = [], Counter(), Counter(), Counter()
    for shard in sorted((Path("/acts") / BASE / "inputs").glob("shard_*.jsonl")):
        for line in shard.read_text().splitlines():
            it = json.loads(line)
            arm, msgs = it["arm"], it["messages"]
            pending, step, edits = [], -1, Counter()
            for m in msgs:
                if m["role"] == "assistant":
                    step += 1
                    pending = [(c["function"]["name"], args_of(c["function"])) for c in m.get("tool_calls") or []]
                    continue
                if m["role"] != "tool" or not pending:
                    continue
                name, a = pending.pop(0)
                out = m["content"] if isinstance(m["content"], str) else json.dumps(m["content"])
                cmd = a.get("command") if name == "text_editor" else None
                if cmd == "str_replace" and DIFF in out[:300]:
                    edits["str_replace diffs"] += 1
                    edits["a: gutter"] += add_gutter(out)[1] == "numbered"  # every hunk line numbered, as the swap's parser decides
                    edits["b: no trailing newline"] += not out.endswith("\n")
                if cmd == "insert":
                    edits["inserts"] += 1
                    edits["c: review note"] += REVIEW in out
                edits["tool outputs"] += 1
                stamps = {k for k, p in TIMESTAMPS.items() if p.search(out)}
                edits["d: timestamp"] += bool(stamps - {"epoch"})  # bare 10-digit numbers are only counted on their own
                for k in stamps:
                    edits[f"d: {k}"] += 1
                years.update((arm, y) for y in TIMESTAMPS["iso"].findall(out) + TIMESTAMPS["log"].findall(out))
                hexes_out = HEX.findall(UUID.sub(" ", out))  # UUIDs first, as scrub.py does: a UUID's 12-character tail is not its own hex string
                ids = UUID.findall(out) + hexes_out
                edits["e: id or hash"] += bool(ids)
                for h in map(str.lower, hexes_out):  # the patterns are lowercase
                    hexes[(arm, "hex strings")] += 1
                    hexes[(arm, "with a repeated or sequential 4-run")] += any(s in h for s in LOW_ENTROPY)
                    p = [h.count(c) / len(h) for c in set(h)]  # Shannon entropy per character, in bits (random hex: up to 4)
                    hexes[(arm, "entropy bits x1000 (sum)")] += round(-1000 * sum(q * math.log2(q) for q in p))
            main, side = cell_of(it["key"])
            keys_seen[arm] += 1
            row = {"arm": arm, "key": it["key"], "main": main, "side": side, **edits}
            row["in 100 cells"] = (main, side) in cells if "attack" in arm else main in mains  # honest runs: by main task
            runs.append(row)
    by_arm = defaultdict(Counter)
    for r in runs:
        for k, v in r.items():
            if isinstance(v, int) and not isinstance(v, bool):
                by_arm[r["arm"]][f"outputs: {k}"] += v
                by_arm[r["arm"]][f"runs with any: {k}"] += v > 0
        by_arm[r["arm"]]["runs"] += 1
    affected = {}
    for arm in ("real-attack-e1", "real-honest"):
        sub = [r for r in runs if r["arm"] == arm and r["in 100 cells"]]
        affected[arm] = {"runs": len(sub), **{e: sum(r.get(e, 0) > 0 for r in sub) for e in
                                               ("a: gutter", "b: no trailing newline", "c: review note", "d: timestamp", "e: id or hash")},
                         "outputs changed": {e: sum(r.get(e, 0) for r in sub) for e in
                                             ("a: gutter", "b: no trailing newline", "c: review note", "d: timestamp", "e: id or hash")},
                         "keys (first 3)": [r["key"] for r in sub[:3]]}
    return {"base dump": BASE, "runs per arm": dict(keys_seen), "prevalence by arm": {a: dict(c) for a, c in by_arm.items()},
            "affected, 100 cells": affected, "timestamp years": {f"{a} {y}": n for (a, y), n in sorted(years.items())},
            "hex strings": {f"{a} {k}": n for (a, k), n in sorted(hexes.items())}}


@app.local_entrypoint()
def main():
    runs = json.loads((SWAP / "config" / "swap_runs.json").read_text())["runs"]
    cells = sorted({cell_of(k) for k in runs})
    if len(cells) != len(runs):
        raise ValueError(f"{len(runs)} swap runs map to {len(cells)} cells")
    res = count.remote([list(c) for c in cells]) | {"cells": len(cells)}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = HERE / "results" / f"tell_counts_{stamp}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(out, json.dumps(res["affected, 100 cells"], indent=1))
