"""E1 stage 1 inputs (PLAN_E1_honest.md, #670; Aswin approved stage 1 at 10:19:53 PT; local, free). From the 196-task
scale-up's real honest A runs (Qwen3.5-27B acting), a seeded draw: the first 10 eligible runs are the profiling pilot,
the next 60 are stage 1. A run is eligible when it has a text_editor edit (create, str_replace or insert) whose tool reply
has no error; its last such edit's path is the action the instructed prompts name. For each run the index keeps the
final model call's request as vLLM received it (the messages up to the last tool reply, the tools and the reasoning
effort), so the job can append one user turn. The index holds transcript text, so it goes to the git-ignored data/ and
S3; results/ gets keys, path hashes and counts only.

  uv run --no-project --python 3.14 python experiments/2026-10-11_q2_e1_honest/build_e1_index.py data/e1_honest/runs_phase2_196.csv
"""
import csv
import hashlib
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
CACHE = Path.home() / ".cache/probe-shift/s3/individual-aswin-surya-probe-shift-kill-2027-04-06/phase2/runs"
SEED, N_PILOT, N_STAGE1 = 20261011, 10, 60
EDITS = ("create", "str_replace", "insert")


def run_entry(run_id: str) -> dict:
    """The final model call's request and the last successful text_editor edit, or why the run isn't eligible."""
    path = next(CACHE.glob(f"*/{run_id}"))
    with zipfile.ZipFile(path) as z:
        samples = [n for n in z.namelist() if n.startswith("samples/") and n.endswith(".json")]
        if len(samples) != 1:
            return {"problem": f"{len(samples)} samples"}
        s = json.loads(z.read(samples[0]))
    model = [e for e in s["events"] if e.get("event") == "model"][-1]
    call = model["call"]
    if call.get("call_key") != "messages":
        raise ValueError(f"{run_id}: unexpected call_key {call.get('call_key')}")
    pool = s["events_data"]["calls"]
    messages = [m for lo, hi in call["call_refs"] for m in pool[lo:hi]]
    if not messages or messages[-1]["role"] != "tool":
        return {"problem": "final request does not end with a tool reply"}
    in_request = {m.get("tool_call_id") for m in messages if m["role"] == "tool"}  # only edits the replayed history holds
    errors = {m.get("tool_call_id"): m.get("error") for m in s["messages"] if m["role"] == "tool"}
    edits = [tc for m in s["messages"] if m["role"] == "assistant" for tc in m.get("tool_calls") or []
             if tc["function"] == "text_editor" and tc["arguments"].get("command") in EDITS and tc["arguments"].get("path")]
    ok = [tc for tc in edits if tc["id"] in in_request and tc["id"] in errors and errors[tc["id"]] is None]
    if not ok:
        return {"problem": f"no successful text_editor edit in the final request ({len(edits)} edits in the run)"}
    req = call["request"]
    target = ok[-1]["arguments"]["path"]
    return {"messages": messages, "tools": req["tools"], "reasoning_effort": req.get("reasoning_effort"), "path": target,
            "path_sha256": hashlib.sha256(target.encode()).hexdigest(), "n_messages": len(messages), "n_edits_ok": len(ok)}


def main(run_list: Path):
    rows = [r for r in csv.DictReader(run_list.open()) if r["arm"] == "honest_a"]
    order = sorted(rows, key=lambda r: hashlib.sha256(f"{SEED}|{r['run_id']}".encode()).hexdigest())
    index, skipped = [], []
    for r in order:
        if len(index) == N_PILOT + N_STAGE1:
            break
        e = run_entry(r["run_id"])
        if "problem" in e:
            skipped.append({"run_id": r["run_id"], "problem": e["problem"]})
            continue
        index.append({"key": r["task_key"], "run_id": r["run_id"], "env": r["env"], "main_task": r["main_task"],
                      "part": "pilot" if len(index) < N_PILOT else "stage1"} | e)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    data = HERE.parents[1] / "data" / "e1_honest"
    with open(data / f"index_{stamp}.jsonl", "x") as fh:
        fh.writelines(json.dumps(x) + "\n" for x in index)
    summary = {"run list": str(run_list), "run list sha256": hashlib.sha256(run_list.read_bytes()).hexdigest(),
               "honest A runs in list": len(rows), "seed": SEED, "index file": f"data/e1_honest/index_{stamp}.jsonl",
               "index sha256": hashlib.sha256((data / f"index_{stamp}.jsonl").read_bytes()).hexdigest(),
               "skipped before filling": skipped,
               "runs": [{k: x[k] for k in ("part", "key", "run_id", "path_sha256", "n_messages", "n_edits_ok")} for x in index]}
    dest = HERE / "results" / f"index_summary_{stamp}.json"; dest.parent.mkdir(exist_ok=True)
    with open(dest, "x") as fh:
        fh.write(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"} | {"pilot": sum(x["part"] == "pilot" for x in index),
                                                                         "stage1": sum(x["part"] == "stage1" for x in index),
                                                                         "messages per run (min, max)": [min(x["n_messages"] for x in index), max(x["n_messages"] for x in index)]}))
    print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
