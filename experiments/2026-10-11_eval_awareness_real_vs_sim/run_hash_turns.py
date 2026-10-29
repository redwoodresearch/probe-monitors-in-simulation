"""Confound check: do the arms' fixed opening turns differ? Hash each run's system turn and first user turn (Modal CPU).

README "Confound check". For every cell of claim B's level-1 packs, the sha256 of the system message and of the first
user message, for e1, e2 and fixed sim. Returns only counts: how many cells have sim's (and e2's) turns equal to e1's,
how many distinct hashes each arm has, and for cells where they differ, line-level counts (lines only in sim, only in
e1) and length differences in characters. No text leaves the container: failures come back as type and line numbers.
CPU only (2 cores, 4 GiB), timeout 240 s, retries 0.

  PYTHONPATH=src uv run modal run --detach experiments/2026-10-11_eval_awareness_real_vs_sim/run_hash_turns.py
"""

import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
APP = "probe-shift_aswin-surya_sub-evalaware_hashturns_kill-2026-11-09"
INPUTS, PACKS = "probe-shift_aswin-surya_sub-la-activations", "2026-10-10_la_fixed-sim-v2_probe-inputs-b-l1_v1"
MANIFEST_SHA = "e56a27e46c4f0093ab07c08fa5c994f2d76391bc6e36e12cc583fdaf6ec9bd1f"
ARMS = {"real-attack-e1": "e1", "real-attack-e2": "e2", "sim-attack-fixed": "sim"}

app = modal.App(APP)
in_vol = modal.Volume.from_name(INPUTS, version=2).read_only()
image = modal.Image.debian_slim(python_version="3.12")


@app.function(image=image, cpu=2, memory=4096, volumes={"/in": in_vol}, timeout=240, retries=0)
def tabulate() -> dict:
    import os
    import sys
    import traceback

    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1)
    os.dup2(null, 2)
    try:
        return _tabulate()
    except BaseException as e:  # never send LinuxArena text back: only the type and line numbers
        lines = [f.lineno for f in traceback.extract_tb(e.__traceback__) if f.filename.endswith("run_hash_turns.py")]
        raise RuntimeError(f"tabulate failed: {type(e).__name__} at run_hash_turns.py lines {lines}") from None
    finally:
        sys.stdout.flush()
        sys.stderr.flush()


def _tabulate() -> dict:
    base = Path("/in") / PACKS
    man = (base / "manifest.json").read_bytes()
    if hashlib.sha256(man).hexdigest() != MANIFEST_SHA:
        raise ValueError("manifest sha256 differs")
    files = json.loads(man)["files"]  # {path: sha256 or {sha256}} or [{path or name, sha256}]
    entries = files.items() if isinstance(files, dict) else [(e.get("path") or e.get("name") or e.get("file"), e) for e in files]
    want = {Path(k).name: (v if isinstance(v, str) else v["sha256"]) for k, v in entries if Path(k).name.startswith("shard_")}
    shard_files = sorted((base / "inputs").glob("shard_*.jsonl"))
    if sorted(p.name for p in shard_files) != sorted(want):
        raise ValueError("the shard set differs from the manifest")
    h = lambda x: hashlib.sha256(json.dumps(x).encode()).hexdigest()
    turns = {}
    for p in shard_files:
        raw = p.read_bytes()
        if hashlib.sha256(raw).hexdigest() != want[p.name]:
            raise ValueError("a shard differs from the manifest")
        for it in map(json.loads, raw.decode().splitlines()):
            if it["arm"] not in ARMS:
                continue
            if (it["cell"], ARMS[it["arm"]]) in turns:
                raise ValueError("a duplicate (cell, arm)")
            msgs = it["messages"]
            sys_m = msgs[0]["content"] if msgs and msgs[0]["role"] == "system" else None
            user = next((m.get("content") for m in msgs if m["role"] == "user"), None)
            turns[(it["cell"], ARMS[it["arm"]])] = (sys_m, user, it.get("environment"))
    cells = sorted({c for c, _ in turns if all((c, a) in turns for a in ARMS.values())})
    if len(cells) != 939:
        raise ValueError(f"{len(cells)} complete cells, not 939")
    out = {"cells with all three arms": len(cells)}
    for k, name in ((0, "system turn"), (1, "first user turn")):
        res = {}
        for arm in ("e2", "sim"):
            same = [h(turns[(c, arm)][k]) == h(turns[(c, "e1")][k]) for c in cells]
            diff = [c for c, s in zip(cells, same) if not s]
            only_a, only_e1, dlen = [], [], []
            for c in diff:
                a, b = turns[(c, arm)][k] or "", turns[(c, "e1")][k] or ""
                la, lb = Counter(map(hash, a.splitlines())), Counter(map(hash, b.splitlines()))
                only_a.append(sum((la - lb).values()))
                only_e1.append(sum((lb - la).values()))
                dlen.append(len(a) - len(b))
            q = lambda xs: [min(xs), sorted(xs)[(len(xs) - 1) // 2], max(xs)] if xs else None  # lower median
            res[f"{arm} vs e1"] = {"cells equal": sum(same), "cells differing": len(diff),
                                   f"lines only in {arm} (min/median/max)": q(only_a), "lines only in e1 (min/median/max)": q(only_e1),
                                   f"length {arm} - e1, characters (min/median/max)": q(dlen),
                                   "differing cells by environment": dict(Counter(turns[(c, 'e1')][2] for c in diff).most_common())}
        res["distinct hashes per arm"] = {arm: len({h(turns[(c, arm)][k]) for c in cells}) for arm in ARMS.values()}
        out[name] = res
    return out


@app.local_entrypoint()
def main():
    out = tabulate.remote()
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=True).stdout.strip()
    out = {"packs": f"{INPUTS}/{PACKS}", "manifest_sha256": MANIFEST_SHA, "git": git, "counts only": True} | out
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"hash_turns_{stamp}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text("{\n" + ",\n".join(f" {json.dumps(k)}: {json.dumps(v)}" for k, v in out.items()) + "\n}\n")
    print(json.dumps(out, indent=1))
    print(f"-> {path}")
