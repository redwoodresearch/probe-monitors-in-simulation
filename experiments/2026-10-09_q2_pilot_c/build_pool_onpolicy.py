"""Pilot C on-policy, step 1 (local, $0): the candidate pool, the seeded draw and the judges' inputs for the strict
rerun's Qwen3.5-27B attack runs (`2026-10-09_000523_rerun_strict`): real (`attack/`) and September-simulated
(`sim_attack@september/`). All 40 runs are attempted (amendment 31's labels, q3 commit 2411573), so every eligible call
is in scope.

Pool: the eligible calls of q3's on-policy 2x2 (attack runs, counted attempts, a rule-1 readout, the terminal submit
excluded) that q3's amendment-36 command-text baseline, trained on real, does not flag at its frozen 1% threshold
(`flag_frozen` in the real->real and real->sim cells, observed draw). Each row keeps the rule label (y), the text score,
and the probe's score and frozen flag (Qwen3.5-27B, real-trained, same cells). Calls are joined on (world, irid, call):
the x-irid request ID in each model event, and the call's index in that event's tool calls; the event's tool-call ID
names the call in the transcript. Draw and render as build_pool.py: N per world, numpy default_rng(SEED).choice without
replacement over the pool sorted by (task, irid, call), then shuffled, neutral IDs o00..; the same render() (reasoning
parts dropped, as the off-policy runs' reasoning was encrypted).

  uv run --no-project --python 3.14 --with numpy --with tiktoken --with pyyaml --with boto3 python \
      experiments/2026-10-09_q2_pilot_c/build_pool_onpolicy.py --run-dir <rerun folder> --text <A36 per-call json> --probe <probe per-call json>
"""
import argparse
import hashlib
import json
import sys
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import tiktoken

sys.path.insert(0, str(Path(__file__).parent))
from build_pool import render

from probe_shift.transcripts import RUNNER_SUBMIT, visible_text

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
ARMS = {"real": "attack", "sim": "sim_attack@september"}
ACTING = "vllm/Qwen/Qwen3.5-27B"


def per_call(path: Path) -> dict:
    """(world, irid, call) -> {y, score, flag} under the real-trained cells (observed draw)."""
    r = json.loads(path.read_text())
    rows = r.get("calls") or r["meta"]
    out = {}
    for cell, world in (("real->real", "real"), ("real->sim", "sim")):
        pc = r["draws"][0]["cells"][cell]["per_call"]
        for i, score, flag in zip(pc["row"], pc["score"], pc["flag_frozen"]):
            c = rows[i]
            if c["world"] != world:
                raise ValueError(f"{path.name}: row {i} is {c['world']}, cell {cell}")
            out[(world, c["irid"], c["call"])] = {"task": c["task"], "y": c["y"], "score": score, "flag": flag}
    return out


def locate(sample: dict) -> dict:
    """irid -> [(turn, action) in render()'s numbering, or None for a submit] per tool call of that model event."""
    where, turn = {}, -1
    for m in sample["messages"]:
        if m["role"] != "assistant":
            continue
        calls = [tc for tc in m.get("tool_calls") or [] if tc["function"] != RUNNER_SUBMIT]
        if not (calls or visible_text(m["content"]).strip()):
            continue
        turn += 1
        for a, tc in enumerate(calls):
            where[tc["id"]] = (turn, a, tc)
    out = {}
    for e in sample["events"]:
        if e.get("event") != "model" or e.get("model") != ACTING:
            continue
        irid = (((e.get("call") or {}).get("request") or {}).get("extra_headers") or {}).get("x-irid")
        tcs = (((e.get("output") or {}).get("choices") or [{}])[0].get("message") or {}).get("tool_calls") or []
        out[irid] = [where.get(tc["id"]) for tc in tcs]
    return out


def main(run_dir: Path, text_path: Path, probe_path: Path, seed: int, n: int, exclude: Path | None, prefix: str):
    text, probe = per_call(text_path), per_call(probe_path)
    if set(text) != set(probe):
        raise ValueError("the text and probe files score different calls")
    samples, where = {}, {}
    for world, arm in ARMS.items():
        for f in sorted((run_dir / arm).glob("*/*.eval")):
            with zipfile.ZipFile(f) as z:
                (name,) = [x for x in z.namelist() if x.startswith("samples/") and x.endswith(".json")]
                s = json.loads(z.read(name))
            for irid, locs in locate(s).items():
                if irid in where:
                    raise ValueError(f"irid {irid} in two runs")
                where[irid] = (world, str(f.relative_to(run_dir)), locs)
                samples[str(f.relative_to(run_dir))] = s
    counts, pool = Counter(), {w: [] for w in ARMS}
    # an earlier draw's calls, left out of this one (the expansion draws fresh calls from the same pools)
    done = {(d["world"], d["irid"], d["call"]) for d in json.loads(exclude.read_text())["drawn"]} if exclude else set()
    for (world, irid, call), t in sorted(text.items()):
        w, run, locs = where[irid]
        if w != world or locs[call] is None:
            raise ValueError(f"{world} {irid} call {call}: not found as a non-submit call in {run}")
        counts[(world, "eligible calls")] += 1
        if t["flag"]:
            counts[(world, "flagged by the text baseline")] += 1
            continue
        if (world, irid, call) in done:
            counts[(world, "already judged (excluded)")] += 1
            continue
        p = probe[(world, irid, call)]
        if p["y"] != t["y"]:
            raise ValueError(f"{world} {irid} {call}: rule labels differ between files")
        turn, action, tc = locs[call]
        pool[world].append({"world": world, "run": run, "task": t["task"], "irid": irid, "call": call, "step": turn, "action": action,
                            "tool": tc["function"], "rule_label": "impactful" if t["y"] else "read_only", "text_score": t["score"],
                            "probe_qwen35": {"score": p["score"], "flag": p["flag"]}})
    if exclude and counts[("real", "already judged (excluded)")] + counts[("sim", "already judged (excluded)")] != len(done):
        raise ValueError("some excluded calls are not among the eligible, unflagged calls")
    rng = np.random.default_rng(seed)
    drawn = []
    for world in ARMS:
        P = sorted(pool[world], key=lambda d: (d["task"], d["irid"], d["call"]))
        drawn += [P[i] for i in sorted(rng.choice(len(P), n, replace=False))]
    drawn = [drawn[i] for i in rng.permutation(len(drawn))]
    enc = tiktoken.get_encoding("o200k_base")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    data = ROOT / "data" / "pilot_c"
    data.mkdir(parents=True, exist_ok=True)
    with open(data / f"judge_inputs_onpolicy_{stamp}.jsonl", "w") as out:
        for j, d in enumerate(drawn):
            s = samples[d["run"]]
            _, _, tc = where[d["irid"]][2][d["call"]]
            prompt = render(s, d["step"], d["action"], {"tool": tc["function"], "arguments": tc.get("arguments") or {}})
            d |= {"id": f"{prefix}{j:0{max(2, len(str(len(drawn) - 1)))}d}", "prompt_chars": len(prompt), "prompt_tokens_o200k": len(enc.encode(prompt)),
                  "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
            out.write(json.dumps(d | {"prompt": prompt, "command": tc.get("arguments") or {}}, ensure_ascii=False) + "\n")
    toks = [d["prompt_tokens_o200k"] for d in drawn]
    summary = {"run_dir": run_dir.name, "seed": seed, "n_per_world": n, "prefix": prefix, "excluded_draw": exclude.name if exclude else None,
               "excluded_draw_sha256": hashlib.sha256(exclude.read_bytes()).hexdigest() if exclude else None, "text_file": text_path.name, "probe_file": probe_path.name,
               "text_sha256": hashlib.sha256(text_path.read_bytes()).hexdigest(), "probe_sha256": hashlib.sha256(probe_path.read_bytes()).hexdigest(),
               "counts": {w: {k: v for (ww, k), v in counts.items() if ww == w} | {"pool": len(pool[w])} for w in ARMS},
               "pool rule labels": {w: dict(Counter(d["rule_label"] for d in pool[w])) for w in ARMS},
               "drawn rule labels": {w: dict(Counter(d["rule_label"] for d in drawn if d["world"] == w)) for w in ARMS},
               "prompt tokens (o200k)": {"total": sum(toks), "max": max(toks), "median": int(np.median(toks))},
               "drawn": drawn, "inputs_file": f"data/pilot_c/judge_inputs_onpolicy_{stamp}.jsonl"}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"pool_onpolicy_{stamp}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "drawn"}, indent=1))
    print(HERE / "results" / f"pool_onpolicy_{stamp}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--text", type=Path, required=True)
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--exclude", type=Path, help="a pool_onpolicy JSON whose drawn calls are left out")
    ap.add_argument("--prefix", default="o", help="ID prefix (o for the first draw)")
    a = ap.parse_args()
    main(a.run_dir, a.text, a.probe, a.seed, a.n, a.exclude, a.prefix)
