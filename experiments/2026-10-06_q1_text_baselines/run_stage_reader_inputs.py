"""Stage per-output examples from the reader inputs an activation extraction staged on its Modal Volume: the exact
messages the probe reads, so any arm the extraction has (e.g. the honest arms) and any cut (`max_tokens: null` for
whole transcripts) get the same rows as run_stage_examples.py, minus `raw`.

Keeps the cells where every listed arm (or every `paired_arms` arm) has a run; an arm with several runs per cell is
split by key (`split_by_key`), else two runs of one arm in a cell fail. Writes
.run_cache/examples_<stamp>/ (outputs.jsonl.gz, calls.jsonl.gz, runs.jsonl, stage.json).

  uv run --with transformers==4.56.2 --with jinja2 python experiments/2026-10-06_q1_text_baselines/run_stage_reader_inputs.py config/stage_reader_inputs_whole.yaml
"""

import gzip
import json
import shutil
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from probe_shift.models import QWEN3_32B
from run_stage_examples import stage_run

HERE = Path(__file__).parent
META = ("arm", "key", "file", "sample_id", "epoch", "environment", "main_task", "cell", "sim_reference")
_tok = None


def _stage(args):
    transcript, max_tokens, tokenizer = args
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer

        _tok = AutoTokenizer.from_pretrained(tokenizer["name"], revision=tokenizer["revision"])
    _, rows, calls, run = stage_run(_tok, transcript["messages"], max_tokens, {k: transcript[k] for k in META})
    assert run["tokens"] == transcript["tokens"], f"{transcript['key']}: token count disagrees with the extraction's"
    return rows, calls, run


def main():
    cfg = yaml.safe_load((HERE / sys.argv[1]).read_text())
    assert cfg["tokenizer"] == QWEN3_32B, f"config tokenizer {cfg['tokenizer']} is not the shared pin {QWEN3_32B}"
    transcripts = []
    for src in cfg["inputs"]:  # extraction folders on the Volume; downloaded once
        local = HERE / ".run_cache" / "reader_inputs" / src
        if not local.exists():  # download next to it, then rename: an interrupted download leaves no folder behind
            part = local.with_name(local.name + ".partial")
            shutil.rmtree(part, ignore_errors=True)
            part.mkdir(parents=True)
            subprocess.run(["modal", "volume", "get", cfg["volume"], f"{src}/inputs", str(part)], check=True)
            subprocess.run(["modal", "volume", "get", cfg["volume"], f"{src}/config.json", str(part / "config.json")], check=True)
            part.rename(local)
        shards = sorted(local.rglob("shard_*.jsonl"))
        expected = json.loads((local / "config.json").read_text())["shards"]
        assert len(shards) == expected, f"{local}: {len(shards)} shards, the extraction wrote {expected}"
        for f in shards:
            transcripts += [t for t in map(json.loads, f.read_text().splitlines()) if t["arm"] in cfg["arms"]]
    # An arm with several runs per cell is split by key: e.g. {real-honest: [real-honest, real-honest-2]} names a cell's
    # first run by key real-honest, its second real-honest-2, and drops the rest.
    arms, dropped = [], Counter()
    for arm in cfg["arms"]:
        names = cfg.get("split_by_key", {}).get(arm, [arm])
        arms += names
        by_cell = {}
        for t in sorted((t for t in transcripts if t["arm"] == arm), key=lambda t: t["key"]):
            by_cell.setdefault(t["cell"], []).append(t)
        for runs_of_cell in by_cell.values():
            assert len(runs_of_cell) == 1 or arm in cfg.get("split_by_key", {}), f"two {arm} runs for cell {runs_of_cell[0]['cell']}"
            for k, t in enumerate(runs_of_cell):
                t["arm"] = names[k] if k < len(names) else None
            dropped[f"{arm}: beyond {len(names)} per cell"] += max(len(runs_of_cell) - len(names), 0)
    transcripts = [t for t in transcripts if t["arm"]]
    # Cells: those where every `paired_arms` arm (default: every arm) has its run; other arms are kept where they exist.
    cells = set.intersection(*({t["cell"] for t in transcripts if t["arm"] == a} for a in cfg.get("paired_arms", arms)))
    dropped.update(f"{t['arm']}: unpaired cell" for t in transcripts if t["cell"] not in cells)
    transcripts = sorted((t for t in transcripts if t["cell"] in cells), key=lambda t: (arms.index(t["arm"]), t["cell"]))
    print(f"{len(cells)} cells with a run in every paired arm; dropped: {dict(dropped)}", flush=True)

    rows, calls, runs = [], [], []
    with ProcessPoolExecutor(cfg["workers"]) as ex:
        for r, c, run in ex.map(_stage, [(t, cfg["max_tokens"], cfg["tokenizer"]) for t in transcripts], chunksize=4):
            rows += r
            calls += c
            runs.append(run)
    for arm in arms:
        print(f"{arm}: {sum(r['arm'] == arm for r in runs)} runs, {sum(r['arm'] == arm for r in rows)} outputs "
              f"({sum(r['arm'] == arm and r['kept'] for r in rows)} kept), {sum(c['arm'] == arm for c in calls)} calls", flush=True)

    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = HERE / ".run_cache" / f"examples_{stamp}"
    out.mkdir(parents=True)
    with gzip.open(out / "outputs.jsonl.gz", "wt") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    with gzip.open(out / "calls.jsonl.gz", "wt") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in calls)
    (out / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in runs))
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    (out / "stage.json").write_text(json.dumps({"config": cfg, "git": git, "cells": len(cells),
                                                "dropped": dict(dropped)}, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
