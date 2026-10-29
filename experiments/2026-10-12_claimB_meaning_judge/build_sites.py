"""Meaning judge: build the judge inputs (PLAN.md @ 2d35f421; Aswin's GO about 14:46 PT, "Sol + Gemini judge"). $0, local.

Strict sites 0-90 (results/prefix_matched_2026-10-12_145915.json, v5): the command at k (e1's call) and what the agent saw
from e1, e2 and, at sim-matched sites, the n437 run itself. For the other 58, a simulator record in the Oct-6 pipeline's
format (`run_simulate_pairs.py` @ cba62df4): e1's real prefix, the step, and the n437 run's own simulator metadata and
stored system prompt (the unrepaired September simulator's inputs for that task). Oct-6 sites 100-199: the 100 honest
pairs' real output and stored simulated reply, read from their Volume (attack rows are never kept).
Text goes only to data/ (git-ignored; S3 behind a manifest); results/sites_*.json holds hashes and lengths only.

  uv run --no-project --python 3.14 --with 'modal>=1.4.3' python experiments/2026-10-12_claimB_meaning_judge/build_sites.py
"""

import hashlib
import json
import re
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REPO = HERE.parents[1]
PM = HERE / "results/prefix_matched_2026-10-12_145915.json"
REAL, SIM = Path.home() / "probe-shift-viewer/.run_cache/viewer/eval", Path.home() / ".cache/probe-shift/s3"
DESCRIBED = re.compile(r"The agent called the `([^`]+)` tool \(the tool is described as: (.*?)\)\n")
SIM_KEYS = ["codebase_root", "env_context", "container_path_mappings", "deployment_tree", "static_filesystem", "setup_script",
            "codebase_exec", "exec_source_authority", "orig_trajectory", "orig_trajectory_mode", "orig_trajectory_out_cap",
            "session_notes", "self_critique", "seeded_prefix", "simulator_model", "simulator_fallback_model",
            "simulator_refusal_judge_model", "simulator_router_judge_model", "fork", "arena_reference_contract", "source_tree",
            "env", "main_id"]  # run_select_pairs.py @ cba62df4
H = lambda s: hashlib.sha256(s.encode()).hexdigest()  # noqa: E731


def text(c):
    return c if isinstance(c, str) else "\n".join(x.get("text") or "" for x in c or [] if x.get("type") == "text")


def seen(m):
    return (m["error"].get("message") or text(m.get("content"))) if m.get("error") else text(m.get("content"))


def sample(row):
    path = REAL / row["cohort"] / row["file"] if row["arm"] == "real-honest" else SIM / row["cohort"].removeprefix("s3://") / "logs" / row["file"]
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if n.startswith("samples/") and n.endswith(".json"):
                x = json.loads(z.read(n))
                if (x["id"], x["epoch"]) == (row["sample_id"], row["epoch"]):
                    return x
    raise KeyError(row["key"])


def actions(s):  # every non-submit call in order with what the agent saw (run_select_pairs.actions_of)
    res = {m.get("tool_call_id"): m for m in s["messages"] if m["role"] == "tool"}
    return [{"function": tc["function"], "arguments": tc.get("arguments") or {}, "id": tc["id"], "output": seen(res[tc["id"]])}
            for m in s["messages"] if m["role"] == "assistant" for tc in m.get("tool_calls") or [] if tc["function"] != "submit"]


def main():
    rows = {r["key"]: r for r in json.loads((REPO / "splits/exclusions_v4.json").read_text())["samples"]}
    pm = json.loads(PM.read_text())
    (HERE / "data").mkdir(exist_ok=True)
    sites, recs, summary = [], [], []
    for i, st in enumerate(pm["strict_sites"]):
        a1, a2 = actions(sample(rows[st["e1"]])), actions(sample(rows[st["e2"]]))
        k, step = st["k"], a1[st["k"]]
        assert a2[k]["function"] == step["function"] and a2[k]["arguments"] == step["arguments"]
        sm = sample(rows[st["sim"]])
        site = {"site": i, "set": "strict", "task": st["task"], "environment": rows[st["e1"]]["environment"], "split": st["split"], "k": k,
                "sim_source": "natural" if st["sim_matched"] else "new", "command": {"function": step["function"], "arguments": step["arguments"]},
                "e1": step["output"], "e2": a2[k]["output"], "sim": actions(sm)[k]["output"] if st["sim_matched"] else None}
        if not st["sim_matched"]:
            md = sm["metadata"]
            assert not {a["id"] for a in a1} & {a.get("id") for a in md["orig_trajectory"]}, "e1 shares call IDs with the reference"
            desc = dict(d for m in sm["store"]["SimulatorState:sim_messages"] for d in DESCRIBED.findall(text(m["content"])))
            prefix = [dict(a, description=desc[a["function"]]) if a["function"] not in ("bash", "text_editor") else a for a in a1[:k]]
            recs.append({"pair": i, "pilot": False, "key": st["e1"], "environment": site["environment"], "prefix": prefix,
                         "step": {x: step[x] for x in ("function", "arguments", "id")}, "sim_metadata": {x: md.get(x) for x in SIM_KEYS},
                         "stored_system_prompt": sm["store"]["SimulatorState:sim_messages"][0]["content"]})
        sites.append(site)
    vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-matched_pairs_kill-2026-11-06", version=2)
    pairs = [json.loads(x) for x in b"".join(vol.read_file("raw/pairs_2026-10-06_175844.jsonl")).decode().splitlines()]
    simo = {r["pair"]: r for r in map(json.loads, b"".join(vol.read_file("raw/sim_pilot_2026-10-06.jsonl")).decode().splitlines())}
    for p in (p for p in pairs if p["arm"] == "honest"):
        assert simo[p["pair"]]["status"] == "ok"
        sites.append({"site": 100 + p["pair"], "set": "oct6", "task": p["main_task"], "environment": p["environment"], "k": p["k"],
                      "sim_source": "oct6", "command": {"function": p["step"]["function"], "arguments": p["step"]["arguments"]},
                      "e1": p["real_output"], "e2": None, "sim": simo[p["pair"]]["result"]["reader_output"]})
    with open(HERE / "data/sites.jsonl", "x") as fh:
        fh.writelines(json.dumps(s) + "\n" for s in sites)
    with open(HERE / "data/sim_recs.jsonl", "x") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in recs)
    for s in sites:
        summary.append({k: s[k] for k in ("site", "set", "task", "split", "k", "sim_source") if k in s}
                       | {f"{r}_sha256": H(s[r]) if s[r] is not None else None for r in ("e1", "e2", "sim")}
                       | {f"{r}_chars": len(s[r]) if s[r] is not None else None for r in ("e1", "e2", "sim")}
                       | {"command_sha256": H(json.dumps(s["command"], sort_keys=True))})
    out = {"label": "Meaning judge inputs (hashes and lengths only)", "sites": summary, "new_sim_records": len(recs),
           "data_sha256": {f: hashlib.sha256((HERE / "data" / f).read_bytes()).hexdigest() for f in ("sites.jsonl", "sim_recs.jsonl")},
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(HERE / "results" / f"sites_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out) + "\n")
    print({"sites": len(sites), "strict": sum(s["set"] == "strict" for s in sites), "natural": sum(s["sim_source"] == "natural" for s in sites),
           "new_sim_records": len(recs), "oct6": sum(s["set"] == "oct6" for s in sites)})


if __name__ == "__main__":
    main()
