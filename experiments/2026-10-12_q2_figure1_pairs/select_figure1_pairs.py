"""Figure 1's example reply pairs (PLAN.md here, committed before this ran). Local, $0. Honest runs only: e1 (the first
valid real-honest run per paired honest task) against that task's n437 sim-honest run. One str_replace pair and one
insert pair, each from the highest tier with an eligible pair (same state, same command, same file), drawn with
numpy's default_rng(20261021). Raw text goes only to data/figure1_pairs/ (git-ignored); results/ gets IDs and hashes.

  uv run --no-project --python 3.14 --with numpy python experiments/2026-10-12_q2_figure1_pairs/select_figure1_pairs.py
"""
import hashlib
import json
import subprocess
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
REPO = HERE.parents[1]
DATA = Path.home() / "probe-shift-viewer/.run_cache/viewer/eval"  # the real runs, as b7's analyze_prefix_matched.py @ d7c81587
SIMDATA = Path.home() / ".cache/probe-shift/s3"  # the n437 honest simulated runs, cached from S3
SEED, COMMANDS, ALLOWED = 20261021, ("str_replace", "insert"), {"real-honest", "sim-honest"}
SUBMIT = "submit"
H = lambda x: hashlib.sha256(x.encode()).hexdigest()  # noqa: E731


def text(c):
    return c if isinstance(c, str) else "\n".join(x.get("text") or "" for x in c or [] if x.get("type") == "text")


def seen(m):  # what the agent saw: the error message for an errored call, as b7's script
    return (m["error"].get("message") or text(m.get("content"))) if m.get("error") else text(m.get("content"))


def calls_of(sample):
    tools = {m.get("tool_call_id"): (i, m) for i, m in enumerate(sample["messages"]) if m["role"] == "tool"}
    out, turn = [], -1
    for i, m in enumerate(sample["messages"]):
        if m["role"] != "assistant":
            continue
        turn += 1
        for tc in m.get("tool_calls") or []:
            if tc["function"] == SUBMIT:
                continue
            args = tc.get("arguments") or {}
            j, r = tools[tc.get("id")]  # a call without a result fails loudly
            out.append({"cmd": H(tc["function"] + " " + json.dumps(args, sort_keys=True)), "res": H(seen(r)), "function": tc["function"], "arguments": args,
                        "command": args.get("command") if tc["function"] == "text_editor" else None, "path": args.get("path"),
                        "error": bool(r.get("error")), "single": len(m["tool_calls"]) == 1, "reply": text(r.get("content")),
                        "assistant_message_index": i, "tool_message_index": j, "agent_turn": turn})
    return out


def load(run):
    assert run["arm"] in ALLOWED, run["arm"]  # honest runs only: nothing else is opened
    path = (SIMDATA / run["cohort"].removeprefix("s3://") / "logs" / run["file"]) if run["arm"] == "sim-honest" else DATA / run["cohort"] / run["file"]
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if n.startswith("samples/") and n.endswith(".json"):
                x = json.loads(z.read(n))
                if (x["id"], x["epoch"]) == (run["sample_id"], run["epoch"]):
                    return calls_of(x)
    raise KeyError(run["key"])


def ok(c, command):
    return c["command"] == command and c["single"] and not c["error"]


def main():
    me = Path(__file__).resolve()
    for f in (me, HERE / "PLAN.md"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    ex_p = REPO / "splits/exclusions_v4.json"
    ex = json.loads(ex_p.read_text()); tasks = set(ex["paired_honest_cells"]["cells"])
    runs = defaultdict(list)
    for r in sorted((r for r in ex["samples"] if r["arm"] == "real-honest" and not r["reason"] and r["main_task"] in tasks), key=lambda r: r["key"]):
        runs[r["main_task"]].append(r)
    e1 = {t: v[0] for t, v in runs.items()}
    sims = {r["main_task"]: r for r in ex["samples"] if r["arm"] == "sim-honest" and not r["reason"] and r["main_task"] in e1}
    calls = {t: (load(e1[t]), load(sims[t])) for t in sorted(sims)}
    elig = {c: {1: [], 2: [], 3: []} for c in COMMANDS}
    for t, (a, b) in calls.items():
        for command in COMMANDS:
            for k in range(min(len(a), len(b))):  # tier 1: same state
                if a[k]["cmd"] != b[k]["cmd"] or (k and a[k - 1]["res"] != b[k - 1]["res"]):
                    break
                if ok(a[k], command) and ok(b[k], command):
                    elig[command][1].append((t, k, k)); break
            for tier, key in ((2, "cmd"), (3, "path")):
                cand = [(i, j) for i, x in enumerate(a) if ok(x, command) for j, y in enumerate(b) if ok(y, command) and x[key] == y[key] and x[key] is not None]
                if cand:
                    i, j = min(cand); elig[command][tier].append((t, i, j))
    rng = np.random.default_rng(SEED)
    picks, record = {}, {"plan": "PLAN.md (this folder)", "git": head, "seed": SEED, "splits sha256": hashlib.sha256(ex_p.read_bytes()).hexdigest(),
                         "tasks with an e1 and a simulated run": len(calls), "eligible pairs by command and tier": {c: {str(k): len(v) for k, v in elig[c].items()} for c in COMMANDS}, "pairs": {}}
    for command in COMMANDS:
        tier = next((k for k in (1, 2, 3) if elig[command][k]), None)
        if tier is None:
            record["pairs"][command] = None; continue
        pool = sorted(elig[command][tier], key=lambda p: (p[0], e1[p[0]]["key"], p[1], p[2]))
        idx = int(rng.integers(len(pool))); t, i, j = pool[idx]; a, b = calls[t][0][i], calls[t][1][j]
        side = lambda run, c, pos: {"arm": run["arm"], "cohort": run["cohort"], "file": run["file"], "sample_id": run["sample_id"], "epoch": run["epoch"], "key": run["key"],  # noqa: E731
                                    "position k": pos, "agent turn": c["agent_turn"], "assistant message index": c["assistant_message_index"], "tool message index": c["tool_message_index"],
                                    "reply bytes": len(c["reply"].encode()), "reply sha256": hashlib.sha256(c["reply"].encode()).hexdigest()}
        meta = {"tier": tier, "tier name": {1: "same state", 2: "same command", 3: "same file"}[tier], "pool size": len(pool), "draw index": idx,
                "main task": t, "environment": e1[t]["environment"], "real": side(e1[t], a, i), "simulated": side(sims[t], b, j),
                "same arguments": a["cmd"] == b["cmd"]}
        record["pairs"][command] = meta
        picks[command] = meta | {"real command": {"function": a["function"], "arguments": a["arguments"]}, "simulated command": {"function": b["function"], "arguments": b["arguments"]},
                                 "real reply": a["reply"], "simulated reply": b["reply"], "real reply repr": repr(a["reply"]), "simulated reply repr": repr(b["reply"])}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = REPO / "data" / "figure1_pairs"; out.mkdir(parents=True, exist_ok=True)
    with open(dp := out / f"figure1_pairs_{stamp}.json", "x") as fh:
        fh.write(json.dumps({"selection record": f"results/figure1_selection_{stamp}.json"} | {"pairs": picks}, indent=1, ensure_ascii=False) + "\n")
    record["data file"] = {"path": str(dp.relative_to(REPO)), "sha256": hashlib.sha256(dp.read_bytes()).hexdigest()}
    (HERE / "results").mkdir(exist_ok=True)
    with open(rp := HERE / "results" / f"figure1_selection_{stamp}.json", "x") as fh:
        fh.write(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record, indent=1)); print(rp); print(dp)


if __name__ == "__main__":
    main()
