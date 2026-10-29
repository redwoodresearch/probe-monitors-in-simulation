"""POST HOC (RULE.md in this folder): literal tabs in text_editor replies, by arm, file type and command. Counts only.

One Modal CPU container reads the staged reader inputs from the activations Volume and writes one result JSON back to it.
Nothing is computed on the laptop. It reads:
- v4's five arms (real-honest, real-attack-e1, real-attack-e2, sim-attack, sim-honest; valid samples, scrub level 0);
- q2's fix-full shards 10 (sim-attack-fx-full) and 12 (sim-honest-fx-full), each sha256-checked against the stage record.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' modal run --detach \\
      experiments/2026-10-13_claimB_tab_census/run_tab_census_modal.py
"""

import hashlib
import json
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-claimb-tab-census_kill-2026-11-12"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
V4 = "2026-10-06_la_segment-end-v4_qwen3-32b_v1/inputs"  # segment_end_v4.yaml: five arms, original text
FX = "2026-10-13_q2_fixfull_inputs_v2/inputs"
FX_SHARDS = {  # experiments/2026-10-13_q2_fixfull/results/stage_2026-10-13_142154.json @ 82909df0
    "shard_10.jsonl": ("sim-attack-fx-full", "5a3874814163b21e1c20dfedd00f1473dc01d2151b9769bb2930a1c16ae05652"),
    "shard_12.jsonl": ("sim-honest-fx-full", "94ae36d6d3bbd420d607254fd7c389910aee80b512d2281e19d2f08ae693b799"),
}
V4_ARMS = {"real-honest", "real-attack-e1", "real-attack-e2", "sim-attack", "sim-honest"}
SPLIT_BY_PREFIX = {"real-honest", "sim-attack", "sim-attack-fx-full"}  # eval-file prefix overlap800 / base200
OUT = "claimB/tab_census_v1"
COMMANDS = ("view", "str_replace", "insert", "create")

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2)
image = modal.Image.debian_slim(python_version="3.12")


def ftype(path) -> str:
    p = str(path or "")
    base = p.rsplit("/", 1)[-1]
    if p.endswith(".go"):
        return "go"
    if base == "Makefile" or p.endswith(".mk"):
        return "makefile"
    if p.endswith(".tsv"):
        return "tsv"
    return "rest"


def shard_rows(job):
    """Per run: (arm, batch, key, [(ftype, command, content_tab, gutter_tab)], tool_name mismatches), and the shard's
    sha256. A reply is paired with its call by position: the k-th tool message after an assistant turn answers that
    turn's k-th call (reader_messages drops submit calls and their results together)."""
    import re

    gutter = re.compile(r"^(\s*\d+)\t")  # a cat -n gutter: its one tab is the real view/insert format, not content
    path, forced_arm, want_sha = job
    raw = Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if want_sha is not None:
        assert sha == want_sha, f"{path} sha256 mismatch: stop"  # checked before any count
    rows = []
    for line in raw.decode().splitlines():
        r = json.loads(line)
        arm = forced_arm or r["arm"]
        reps, mis, pending = [], 0, []
        for m in r["messages"]:
            if m["role"] == "assistant":
                pending = [tc["function"] for tc in m.get("tool_calls") or []]
            elif m["role"] == "tool":
                call = pending.pop(0) if pending else None
                if call is None or call.get("name") != m.get("tool_name"):
                    mis += 1
                    continue
                if call["name"] != "text_editor":
                    continue
                args = call.get("arguments") or {}
                if isinstance(args, str):
                    args = json.loads(args)
                cmd = args.get("command")
                text = m["content"] if isinstance(m["content"], str) else ""
                lines = text.split("\n")
                reps.append((ftype(args.get("path")), cmd if cmd in COMMANDS else "other",
                             any("\t" in gutter.sub(r"\1", x, count=1) for x in lines), any(gutter.match(x) for x in lines)))
        batch = str(r.get("file", "")).split("_", 1)[0] if arm in SPLIT_BY_PREFIX else None
        rows.append((arm, batch, r["key"], reps, mis))
    return rows, sha


def wilson(n, N):
    import math

    if N == 0:
        return None
    p, z = n / N, 1.959963984540054
    d = 1 + z * z / N
    c = (p + z * z / (2 * N)) / d
    h = z * math.sqrt(p * (1 - p) / N + z * z / (4 * N * N)) / d
    return [round(c - h, 4), round(c + h, 4)]


def summarize(rows):
    from collections import defaultdict

    cell = defaultdict(lambda: [0, 0, 0])  # (ftype, command) -> [content_tab, gutter_tab, N]
    runs_go, runs_go_tab, mis = 0, 0, 0
    for _, _, _, reps, m in rows:
        mis += m
        go = [x for x in reps if x[0] == "go"]
        runs_go += bool(go)
        runs_go_tab += any(x[2] for x in go)  # run-level: any Go reply with a content tab
        for ft, cmd, ct, gt in reps:
            groups = {ft, "all"} | ({"non_go"} if ft != "go" else set())  # non_go is inclusive: makefile, tsv and rest
            for g in groups:
                for c in (cmd, "all"):
                    v = cell[(g, c)]
                    v[0] += ct
                    v[1] += gt
                    v[2] += 1
    return {"runs": len(rows), "tool_name_mismatches": mis, "runs_with_go_reply": runs_go,
            "runs_with_go_content_tab": runs_go_tab, "run_share": round(runs_go_tab / runs_go, 4) if runs_go else None,
            "run_wilson95": wilson(runs_go_tab, runs_go),
            "cells": {f"{g}|{c}": {"content_tab": v[0], "N": v[2], "share": round(v[0] / v[2], 4) if v[2] else None,
                                   "wilson95": wilson(v[0], v[2]), "gutter_tab_share": round(v[1] / v[2], 4) if v[2] else None}
                      for (g, c), v in sorted(cell.items())}}


@app.function(image=image, volumes={"/vol": acts}, cpu=8, memory=16384, timeout=1800, retries=0)
def census(git: str, script_sha256: str) -> dict:
    from collections import Counter, defaultdict
    from concurrent.futures import ProcessPoolExecutor
    from datetime import datetime
    from zoneinfo import ZoneInfo

    v4 = sorted(Path("/vol", V4).glob("shard_*.jsonl"))
    assert len(v4) == 64, f"v4 shards: {len(v4)}"
    jobs = [(str(p), None, None) for p in v4] + [(str(Path("/vol", FX, s)), a, sha) for s, (a, sha) in FX_SHARDS.items()]
    with ProcessPoolExecutor(max_workers=8) as ex:  # the container's 8 requested CPUs
        parts = list(ex.map(shard_rows, jobs))
    shas = {Path(j[0]).relative_to("/vol").as_posix(): sha for j, (_, sha) in zip(jobs, parts)}
    rows = [r for part, _ in parts for r in part]
    arms = defaultdict(list)
    for r in rows:
        arms[r[0]].append(r)
        if r[1]:
            arms[f"{r[0]}|{r[1]}"].append(r)
    assert V4_ARMS <= set(arms), sorted(arms)
    res = {"label": "POST HOC, descriptive, preliminary until reviewed", "rule": "RULE.md (this folder)", "git": git,
           "script_sha256": script_sha256, "inputs": {"v4": V4, "fixfull": FX, "shard_sha256": shas},
           "arms": {a: summarize(v) for a, v in sorted(arms.items())}}
    # fix-full vs its original (v4) arm, per run matched by key with equal text_editor reply counts
    comp = {}
    for orig, fx in (("sim-honest", "sim-honest-fx-full"), ("sim-attack", "sim-attack-fx-full")):
        o, f = {r[2]: r for r in arms[orig]}, {r[2]: r for r in arms[fx]}
        assert len(o) == len(arms[orig]) and len(f) == len(arms[fx]), "duplicate keys"
        same, count_diff, trans = 0, 0, Counter()
        for k in sorted(set(o) & set(f)):
            if len(o[k][3]) != len(f[k][3]):
                count_diff += 1
                continue
            same += 1
            for a, b in zip(o[k][3], f[k][3]):
                trans[f"{a[0]}|{a[1]}|{int(a[2])}->{int(b[2])}"] += 1
        comp[f"{orig} -> {fx}"] = {"matched_equal_counts": same, "matched_count_differs": count_diff,
                                   "only_original": len(set(o) - set(f)), "only_fixfull": len(set(f) - set(o)),
                                   "per_reply_content_tab_transitions": dict(sorted(trans.items()))}
    res["fixfull_vs_original"] = comp
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = Path("/vol", OUT)
    out.mkdir(parents=True, exist_ok=True)
    name = f"tab_census_POSTHOC_{stamp}.json"
    with open(out / name, "x") as fh:
        fh.write(json.dumps(res, indent=1) + "\n")
    acts.commit()
    lines = []
    for a, s in res["arms"].items():
        g, n = s["cells"].get("go|all", {}), s["cells"].get("non_go|all", {})
        lines.append(f"{a:30s} runs {s['runs']:5d} | go {g.get('content_tab')}/{g.get('N')} {g.get('wilson95')} | non-go "
                     f"{n.get('content_tab')}/{n.get('N')} {n.get('wilson95')} | runs with a go tab {s['runs_with_go_content_tab']}/"
                     f"{s['runs_with_go_reply']} | mismatches {s['tool_name_mismatches']}")
    return {"path": f"{OUT}/{name}", "lines": lines, "fixfull_vs_original": {k: {kk: vv for kk, vv in v.items() if kk !=
            "per_reply_content_tab_transitions"} for k, v in comp.items()}}


@app.local_entrypoint()
def main():
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", str(HERE)], capture_output=True, text=True, cwd=HERE).stdout
    assert not dirty.strip(), "commit the script before running it"
    out = census.remote(git, hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    print("\n".join(out["lines"]))
    print(json.dumps(out["fixfull_vs_original"]))
    print("result on the Volume:", out["path"])
