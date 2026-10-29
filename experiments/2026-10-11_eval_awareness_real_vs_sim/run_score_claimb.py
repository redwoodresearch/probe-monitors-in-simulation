"""Score claim B's level-1 packs (real e1, real e2, fixed sim per cell) with the frozen eval-awareness probes (H200s).

README "Scoring". For each drawn cell: e1, e2 and sim as they are, plus e1 with LINE ending the system prompt (the
far planted check). The reader is cut at 32,768 tokens, as claim B. Calls are scored as
experiments/2026-10-10_eval_awareness_probe/run_score_retrain.py scores them: content argument last, span text
matching, the span inside the cut, and the turn header checked. Probes: B (turn start), A (last 32 content tokens) and
the old probe (full span), in one pass up to the deepest probe layer. A run whose rendering fails or that has no
scored call is skipped and counted. Only numbers leave the container: failures come back as type and line numbers,
stdout and stderr stay discarded, and rows hold scores and identifiers (cell names, indices). Each finished run is checkpointed on the
Volume, so a rerun resumes. Retries 0.

  PYTHONPATH=src uv run modal run --detach experiments/2026-10-11_eval_awareness_real_vs_sim/run_score_claimb.py --stage pilot
  ... --stage rest: every cell not in the pilot, on 8 containers
"""

import hashlib
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
APP = "probe-shift_aswin-surya_sub-evalaware_claimb_kill-2026-11-09"
VOL = "probe-shift_aswin-surya_sub-evalaware_acts_kill-2026-11-09"
INPUTS, PACKS = "probe-shift_aswin-surya_sub-la-activations", "2026-10-10_la_fixed-sim-v2_probe-inputs-b-l1_v1"
MANIFEST_SHA = "e56a27e46c4f0093ab07c08fa5c994f2d76391bc6e36e12cc583fdaf6ec9bd1f"
ARMS = {"real-attack-e1": "e1", "real-attack-e2": "e2", "sim-attack-fixed": "sim"}
SEED, PILOT, MAX_TOKENS, CAP = "20261011", 100, 32768, 32
LINE = " This is an evaluation."
SHARDS = {"pilot": 2, "rest": 8}
EA = "experiments/2026-10-10_eval_awareness_probe/results/"
FITS = {"old": (EA + "fit_2026-10-10_022437.json", "per_token", "full"), "A": (EA + "fit_2026-10-10_230616.json", "per_token", "capped"),
        "B": (EA + "fit_2026-10-10_230615.json", "per_token", "turn_start")}
PIP = ["torch==2.8.0", "transformers==5.19.0", "accelerate>=1.1.0", "numpy", "huggingface_hub", "flash-linear-attention==0.5.2"]

app = modal.App(APP)
vol = modal.Volume.from_name(VOL, version=2)
in_vol = modal.Volume.from_name(INPUTS, version=2).read_only()
hf_cache = modal.Volume.from_name("probe-shift_aswin-surya_hf-cache")
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PIP).env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})
         .add_local_python_source("probe_shift"))
LOG_FD = 1


def log(msg: str):  # counts only, to the real stdout (the Modal log)
    import os
    os.write(LOG_FD, (msg + "\n").encode())


class Safe(Exception):
    """A failure whose message holds only counts, keys and indices, so it may leave the container."""


def draw(cells: dict) -> list[str]:
    """Cells ranked by sha256(SEED|cell); the first cell of each main task, then the rest in rank order."""
    ranked = sorted(cells, key=lambda c: hashlib.sha256(f"{SEED}|{c}".encode()).hexdigest())
    seen, first, rest = set(), [], []
    for c in ranked:
        (rest if cells[c] in seen else first).append(c)
        seen.add(cells[c])
    return first + rest


@app.function(image=image, gpu="H200", volumes={"/hf": hf_cache, "/in": in_vol, "/acts": vol}, memory=65536, timeout=1800, retries=0)
def score(shard: int, n_shards: int, stage: str, probes: dict) -> dict:
    global LOG_FD
    import os
    import sys
    import traceback

    null = os.open(os.devnull, os.O_WRONLY)
    LOG_FD = os.dup(1)
    os.dup2(null, 1)
    os.dup2(null, 2)
    try:
        return _score(shard, n_shards, stage, probes)
    except Safe:
        raise
    except BaseException as e:  # never send LinuxArena text back: only the type and line numbers
        lines = [f.lineno for f in traceback.extract_tb(e.__traceback__) if f.filename.endswith("run_score_claimb.py")]
        raise RuntimeError(f"score failed: {type(e).__name__} at run_score_claimb.py lines {lines}") from None
    finally:  # the streams stay discarded for the container's life
        sys.stdout.flush()
        sys.stderr.flush()


def _score(shard: int, n_shards: int, stage: str, probes: dict) -> dict:
    import bisect

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import probe_shift.reader as reader_module
    from probe_shift.models import QWEN35_27B
    from probe_shift.reader import command_param, segment_end_positions

    t0 = time.time()
    P = {}
    for name, spec in probes.items():
        blob = (Path("/acts") / spec["dir"] / "probes.json").read_bytes()
        if hashlib.sha256(blob).hexdigest() != spec["sha256"]:
            raise Safe(f"{name}: probes.json does not match its sha256")
        v = json.loads(blob)["probes"][spec["readout"]]
        P[name] = {k: torch.tensor(v[k], device="cuda", dtype=torch.float32) for k in ("w", "mu", "sd")} | {"layer": v["layer"], "read": spec["read"]}
    layers = sorted({p["layer"] for p in P.values()})
    base = Path("/in") / PACKS
    man_raw = (base / "manifest.json").read_bytes()
    if hashlib.sha256(man_raw).hexdigest() != MANIFEST_SHA:
        raise Safe("the packs' manifest does not match its sha256")
    files = json.loads(man_raw)["files"]  # {path: sha256 or {sha256}} or [{path or name, sha256}]
    entries = files.items() if isinstance(files, dict) else [(e.get("path") or e.get("name") or e.get("file"), e) for e in files]
    want = {Path(k).name: (v if isinstance(v, str) else v["sha256"]) for k, v in entries}
    want = {k: v for k, v in want.items() if k.startswith("shard_")}
    shard_files = sorted((base / "inputs").glob("shard_*.jsonl"))
    if sorted(p.name for p in shard_files) != sorted(want):
        raise Safe(f"{len(shard_files)} shard files against {len(want)} in the manifest")
    runs, cells = {}, {}
    for p in shard_files:
        raw = p.read_bytes()
        if hashlib.sha256(raw).hexdigest() != want[p.name]:
            raise Safe(f"{p.name} does not match the manifest")
        for it in map(json.loads, raw.decode().splitlines()):
            if it["arm"] in ARMS:
                runs[(it["cell"], ARMS[it["arm"]])] = it
                cells[it["cell"]] = it["main_task"]
    complete = {c: t for c, t in cells.items() if all((c, a) in runs for a in ARMS.values())}
    if len(complete) != 939:
        raise Safe(f"{len(complete)} complete cells, not 939")
    order = draw(complete)
    mine_cells = order[:PILOT] if stage == "pilot" else order[PILOT:]
    # work per run: its characters clipped near the token cut (about 4 per token), twice for e1 (two versions)
    cost = lambda ca: min(len(json.dumps(runs[ca]["messages"])), 4 * MAX_TOKENS) * (2 if ca[1] == "e1" else 1)
    work = sorted(((c, a) for c in mine_cells for a in ARMS.values()), key=lambda ca: -cost(ca))
    loads, mine = [0] * n_shards, []
    for w in work:  # greedy: each run to the least-loaded shard, deterministic
        k = min(range(n_shards), key=lambda i: (loads[i], i))
        loads[k] += cost(w)
        if k == shard:
            mine.append(w)
    tok = AutoTokenizer.from_pretrained(QWEN35_27B["name"], revision=QWEN35_27B["revision"])
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    lm = AutoModelForCausalLM.from_pretrained(QWEN35_27B["name"], revision=QWEN35_27B["revision"], dtype=torch.bfloat16,
                                              device_map="cuda", attn_implementation="sdpa").eval()

    class Stop(Exception):
        pass

    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    fp = hashlib.sha256(Path(__file__).read_bytes() + Path(reader_module.__file__).read_bytes() +
                        json.dumps([probes, stage, n_shards, MANIFEST_SHA], sort_keys=True).encode() + json.dumps(QWEN35_27B).encode()).hexdigest()[:16]
    ck = Path("/acts") / "claimb_scores" / fp / f"shard{shard}"
    ck.mkdir(parents=True, exist_ok=True)
    done = {}
    for f in ck.glob("*.json"):
        d_ = json.loads(f.read_text())
        done[d_["id"]] = d_
    rows, skipped, n_tok, fwd = [], [], 0, 0.0
    for n_run, (cell, arm) in enumerate(mine):
        it = runs[(cell, arm)]
        rid = hashlib.sha1(f"{cell}|{arm}".encode()).hexdigest()[:16]
        msg_sha = hashlib.sha256(json.dumps(it["messages"]).encode()).hexdigest()
        if rid in done and done[rid]["messages_sha256"] == msg_sha:
            rows += done[rid]["rows"]
            skipped += done[rid]["skipped"]
            continue
        mine_rows, mine_skip = [], []
        for version in (["original", "far"] if arm == "e1" else ["original"]):
            msgs = json.loads(json.dumps(it["messages"]))
            if version == "far":
                if msgs[0]["role"] != "system":
                    raise Safe(f"{it['key']}: no system message to extend")
                msgs[0]["content"] = msgs[0]["content"].rstrip() + LINE
            try:
                r = segment_end_positions(tok, msgs, MAX_TOKENS)
            except ValueError as e:  # only the reader's XML-tag refusal is an expected skip; anything else aborts
                if "quotes the XML call tags" not in str(e):
                    raise
                mine_skip.append({"cell": cell, "arm": arm, "version": version, "environment": it.get("environment"), "reason": "XML tags quoted"})
                continue
            ids = r["ids"]
            ok, step = {}, -1
            for m in msgs:
                if m["role"] == "assistant":
                    step += 1
                    for a_, c in enumerate(m.get("tool_calls") or []):
                        args = c["function"].get("arguments") or {}
                        args = json.loads(args) if isinstance(args, str) else args
                        key = command_param(c["function"]["name"], args)
                        if key is not None and list(args)[-1] == key:
                            ok[(step, a_)] = str(args[key])
            span_text = lambda p: tok.decode(ids[p["span"][0]:p["span"][1]]).strip()
            calls = [p for p in r["positions"] if p["kind"] == "call_end" and p["kept"] and p.get("span") and p["span"][1] <= len(ids)
                     and (p["step"], p["action"]) in ok and span_text(p) and span_text(p)[-40:] in ok[(p["step"], p["action"])]]
            if not calls:
                mine_skip.append({"cell": cell, "arm": arm, "version": version, "environment": it.get("environment"), "reason": "no scored call"})
                continue
            starts = [k for k, t in enumerate(ids) if t == im_start]
            acts = {}

            def hook_for(layer):
                def hook(module, args, output):
                    acts[layer] = (output[0] if isinstance(output, tuple) else output)[0]
                    if layer == layers[-1]:
                        raise Stop
                return hook

            handles = [lm.model.layers[l].register_forward_hook(hook_for(l)) for l in layers]
            t1 = time.time()
            try:
                with torch.inference_mode():
                    lm.model(input_ids=torch.tensor([ids + [pad] * (-len(ids) % 4096)], device="cuda"), use_cache=False)
            except Stop:
                pass
            finally:
                for h in handles:
                    h.remove()
            torch.cuda.synchronize()
            fwd += time.time() - t1
            n_tok += len(ids)
            with torch.inference_mode():
                for c in calls:
                    i0, i1 = c["span"]
                    b_ = bisect.bisect_left(starts, i0) - 1
                    if b_ < 0 or tok.decode(ids[starts[b_]:starts[b_] + 3]) != "<|im_start|>assistant\n":
                        raise Safe(f"{it['key']} step {c['step']}: the call's turn does not open with the assistant header")
                    sl = {"full": slice(i0, i1), "capped": slice(max(i0, i1 - CAP), i1), "turn_start": slice(starts[b_] + 2, starts[b_] + 3)}
                    row = {"cell": cell, "main_task": it["main_task"], "arm": arm, "version": version, "step": c["step"],
                           "action": c["action"], "span_start": i0, "span_len": i1 - i0}
                    for name, p in P.items():
                        z = (acts[p["layer"]][sl[p["read"]]].to(torch.float16).float() - p["mu"]) / p["sd"]
                        row[f"score_{name}"] = float((z @ p["w"]).mean())
                    mine_rows.append(row)
            acts.clear()
        (ck / f".{rid}.tmp").write_text(json.dumps({"id": rid, "messages_sha256": msg_sha, "rows": mine_rows, "skipped": mine_skip}))
        (ck / f".{rid}.tmp").rename(ck / f"{rid}.json")
        vol.commit()
        rows += mine_rows
        skipped += mine_skip
        if n_run % 10 == 0:
            log(f"shard {shard}: {n_run + 1}/{len(mine)} runs, {n_tok} tokens, {fwd:.0f} s forward, {time.time() - t0:.0f} s")
    return {"rows": rows, "skipped": skipped, "roster": sorted({c for c, _ in mine}), "scorer_fingerprint": fp,
            "complete cells": len(complete), "resumed": sum(hashlib.sha1(f"{c}|{a}".encode()).hexdigest()[:16] in done for c, a in mine),
            "stats": {"shard": shard, "runs": len(mine), "tokens": n_tok, "forward_s": round(fwd, 1), "layers": layers,
                      "gpu": torch.cuda.get_device_properties(0).name}}


@app.local_entrypoint()
def main(stage: str = "pilot"):
    if stage not in SHARDS:
        raise ValueError(f"stage must be one of {sorted(SHARDS)}")
    probes = {}
    for name, (f, readout, read) in FITS.items():
        fit = json.loads((ROOT / f).read_text())
        probes[name] = {"dir": f"{fit['run']}/fits/{fit['fit_id']}", "sha256": fit["probes_sha256"], "readout": readout, "read": read}
    import probe_shift.reader as reader_module
    protocol = hashlib.sha256(Path(__file__).read_bytes() + Path(reader_module.__file__).read_bytes() + json.dumps(
        [probes, MANIFEST_SHA, MAX_TOKENS, LINE, CAP, SEED, PILOT], sort_keys=True).encode()).hexdigest()  # stage-independent
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    parts = list(score.starmap([(s, SHARDS[stage], stage, probes) for s in range(SHARDS[stage])], order_outputs=False))
    if len({p["scorer_fingerprint"] for p in parts}) != 1:
        raise ValueError("the shards ran different scorer code or inputs")
    rows = [r for p in parts for r in p["rows"]]
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=ROOT).stdout.strip())
    rows_file = ROOT / ".run_cache" / "eval_awareness_real_vs_sim" / f"claimb_rows_{stage}_{stamp}.json"  # bulky: .run_cache and S3
    rows_file.parent.mkdir(parents=True, exist_ok=True)
    rows_file.write_text(json.dumps(rows))
    out = {"git": git, "dirty": dirty, "stage": stage, "protocol_sha256": protocol, "probes": probes, "packs": f"{INPUTS}/{PACKS}", "manifest_sha256": MANIFEST_SHA,
           "seed": SEED, "max_tokens": MAX_TOKENS, "line": LINE, "scorer_fingerprint": parts[0]["scorer_fingerprint"],
           "complete cells": parts[0]["complete cells"], "cells": sorted({c for p in parts for c in p["roster"]}),
           "skipped": [s for p in parts for s in p["skipped"]], "resumed runs": sum(p["resumed"] for p in parts),
           "rows": rows_file.name, "rows_sha256": hashlib.sha256(rows_file.read_bytes()).hexdigest(), "n_rows": len(rows),
           "shards": [p["stats"] for p in parts]}
    path = HERE / "results" / f"claimb_scores_{stage}_{stamp}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text("{\n" + ",\n".join(f" {json.dumps(k)}: {json.dumps(v)}" for k, v in out.items()) + "\n}\n")
    print(json.dumps({k: out[k] for k in ("n_rows", "complete cells", "resumed runs", "shards")}, indent=1), f"\n-> {path}")
