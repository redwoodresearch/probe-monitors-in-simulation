"""James's check (f), free part: how much of each tool output scrub level 7 changes, per world. Modal CPU; pennies.

From the level-0 staged reader messages of the Q1 extraction (the text the reader read, before any scrub), for the 939
paired attack cells' real epoch-1 and simulated runs: every tool output at levels 6 and 7 (probe_shift.scrub), tokenized
with Qwen3-32B's tokenizer. Per world: the share of outputs level 7 changes; the share of level-6 tokens not kept at
level 7 (difflib matching blocks over token ids); total tokens at levels 0, 6 and 7; by tool and by text_editor command.
Check: the share of outputs whose level-7 text here equals the staged level-7 extraction's (that staging used scrub
version 1; the versions differ only on IDs joined by "_"). Counts only, no text: results/level7_changes_<stamp>.json.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' modal run \
      experiments/2026-10-09_q1_method_checks/run_level7_changes.py
"""

import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
APP = "probe-shift_aswin-surya_sub-q1-method-checks_l7changes_kill-2026-11-09"
LEVEL0 = "2026-10-06_la_segment-end_qwen3-32b_alllayers_v1"
LEVEL7 = "2026-10-06_la_segment-end-scrub7_qwen3-32b_alllayers_v1"
WORLDS = {"real-attack-e1": "real", "sim-attack": "simulated"}

app = modal.App(APP)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("transformers==4.56.2", "jinja2")  # the Q1 extraction's version
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}).add_local_python_source("probe_shift"))
VOLUMES = {"/acts": modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2).read_only(),
           "/hf": modal.Volume.from_name("probe-shift_aswin-surya_hf-cache").read_only()}


def staged(run: str, cells: set) -> tuple[dict, dict]:
    """An extraction's config and its staged real epoch-1 and simulated items of these cells, by (arm, key) (its own
    inputs/, or its source run's); other arms are skipped while reading."""
    cfg = json.loads((Path("/acts") / run / "config.json").read_text())
    d = Path("/acts") / run / "inputs"
    d = d if d.exists() else Path("/acts") / cfg["inputs_from"] / "inputs"
    out = {}
    for sh in sorted(d.glob("shard_*.jsonl")):
        with sh.open() as fh:
            for line in fh:
                it = json.loads(line)
                if it["arm"] in WORLDS and it["cell"] in cells:
                    if (it["arm"], it["key"]) in out:
                        raise ValueError(f"{run}: {it['arm']} {it['key']} staged twice")
                    out[(it["arm"], it["key"])] = it
    return cfg, out


def outputs(messages: list[dict]) -> list[tuple[str, str, str]]:
    """(tool, text_editor command or "", content) per tool output, matched to the calls of the turn before it, in order."""
    calls, out = [], []
    for m in messages:
        if m["role"] == "assistant":
            calls = list(m.get("tool_calls") or [])
        elif m["role"] == "tool":
            if not isinstance(m["content"], str):
                raise TypeError("a tool output that is not a string")
            c = calls.pop(0) if calls else None
            name = c["function"]["name"] if c else "?"
            cmd = str((c["function"]["arguments"] or {}).get("command")) if name == "text_editor" else ""
            out.append((name, cmd, m["content"]))
    return out


def per_run(job):
    """Token counts for one run's outputs: [(tool, command, changed, n0, n6, n7, kept, staged_equal)]. staged_equal is
    None where the level-7 staging holds no such output (its 32k cut falls elsewhere)."""
    import difflib

    from transformers import AutoTokenizer

    from probe_shift.models import QWEN3_32B
    from probe_shift.scrub import scrub

    global TOK
    if "TOK" not in globals():
        TOK = AutoTokenizer.from_pretrained(QWEN3_32B["name"], revision=QWEN3_32B["revision"])
    outs0, outs7 = job
    rows = []
    for i, (tool, cmd, text) in enumerate(outs0):
        t6, t7 = scrub(text, 6), scrub(text, 7)
        n0, a, b = (len(TOK(x, add_special_tokens=False)["input_ids"]) if x else 0 for x in (text, t6, t7))
        if t6 == t7:
            kept = a
        else:
            ia, ib = TOK(t6, add_special_tokens=False)["input_ids"], TOK(t7, add_special_tokens=False)["input_ids"]
            kept = sum(bl.size for bl in difflib.SequenceMatcher(None, ia, ib, autojunk=False).get_matching_blocks())
        rows.append((tool, cmd, t6 != t7, n0, a, b, kept, (t7 == outs7[i][2]) if i < len(outs7) else None))
    return rows


@app.function(image=image, cpu=16, memory=65536, timeout=1800, retries=0, volumes=VOLUMES)
def count(cells: list[str]) -> dict:
    from multiprocessing import Pool

    cells = set(cells)
    c0, by0 = staged(LEVEL0, cells)
    c7, by7 = staged(LEVEL7, cells)
    if set(by0) != set(by7):
        raise ValueError(f"level-0 and level-7 stagings hold different runs: {len(set(by0) ^ set(by7))} differ")
    per_cell = Counter((a, by0[(a, k)]["cell"]) for a, k in by0)
    if set(per_cell) != {(a, c) for a in WORLDS for c in cells} or set(per_cell.values()) != {1}:
        raise ValueError(f"not exactly one run per arm in each of the {len(cells)} cells")
    keys = sorted(by0)
    jobs, mismatch = [], []
    for k in keys:
        o0, o7 = outputs(by0[k]["messages"]), outputs(by7[k]["messages"])
        if len(o0) != len(o7):
            mismatch.append(k)
        jobs.append((o0, o7))
    with Pool(16) as pool:
        res = pool.map(per_run, jobs, chunksize=4)
    agg = defaultdict(Counter)
    for (arm, _), rows in zip(keys, res):
        w = WORLDS[arm]
        agg[(w, "runs", "")]["n"] += 1
        for tool, cmd, changed, n0, n6, n7, kept, eq in rows:
            for g in {(w, "all", ""), (w, tool, ""), (w, tool, cmd)}:  # a set: an empty command is not counted twice
                a = agg[g]
                a["outputs"] += 1
                a["changed"] += changed
                a["tokens0"] += n0
                a["tokens6"] += n6
                a["tokens7"] += n7
                a["kept6to7"] += kept
                a["staged7_compared"] += eq is not None
                a["staged7_equal"] += bool(eq)
    out = {}
    for (w, tool, cmd), a in sorted(agg.items()):
        if tool == "runs":
            out[f"{w}|runs"] = a["n"]
            continue
        out[f"{w}|{tool}|{cmd}".rstrip("|")] = dict(a) | {
            "share_outputs_changed": round(a["changed"] / a["outputs"], 4),
            "share_tokens6_not_kept": round(1 - a["kept6to7"] / max(a["tokens6"], 1), 4),
            "share_staged7_equal": round(a["staged7_equal"] / max(a["staged7_compared"], 1), 4)}
    return {"stagings": {"level0": {"run": LEVEL0, "inputs_from": c0.get("inputs_from"), "scrub_level": c0.get("scrub_level"),
                                    "scrub_version": c0.get("scrub_version")},
                         "level7": {"run": LEVEL7, "inputs_from": c7.get("inputs_from"), "scrub_level": c7.get("scrub_level"),
                                    "scrub_version": c7.get("scrub_version")}},
            "runs_with_different_output_counts": len(mismatch), "groups": out}


@app.local_entrypoint()
def main():
    from probe_shift.scrub import SCRUB_VERSION

    ex = json.loads((ROOT / "splits" / "exclusions_v3.json").read_text())
    r = count.remote(ex["paired_attack_cells"]["cells"])
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dst = HERE / "results" / f"level7_changes_{stamp}.json"
    dst.parent.mkdir(exist_ok=True)
    with dst.open("x") as fh:
        fh.write(json.dumps({"meta": {"git": git, "scrub_version_here": SCRUB_VERSION, "cells": "exclusions_v3.json paired_attack_cells"}} | r, indent=1))
    print(json.dumps(r["stagings"]), "| runs with different output counts:", r["runs_with_different_output_counts"])
    for g, a in r["groups"].items():
        if isinstance(a, dict) and (g.count("|") == 1 or g.startswith(("real|text_editor", "simulated|text_editor")) or a["outputs"] > 2000):
            print(f"{g:40s} outputs {a['outputs']:>6} changed {a['share_outputs_changed']:.3f} tokens6 {a['tokens6']:>9} "
                  f"not kept {a['share_tokens6_not_kept']:.3f} (tokens0 {a['tokens0']}, tokens7 {a['tokens7']}) staged7 equal {a['share_staged7_equal']:.3f}")
        elif not isinstance(a, dict):
            print(g, a)
    print("->", dst)
