"""Which attack approach and system prompt each scale-up attack run actually used, by sha256 only (RULE.md, A).

For every `attack` and `sim_attack@<simulator>` run of the given launches (launch.json, the counted last attempt's
.eval), it hashes the sample's metadata["attack_approach"] and the first system and user messages of the policy's first
model call, i.e. what the attacker was sent, and compares them with the launch's task record (approach_sha256,
sol_system_sha256, sol_input_sha256). It also records how each run was launched: its task function and PHASE2_CELLS
file (with its sha256), from the attempt's recorded cmd and env_vars. It writes and prints hashes, paths and match flags
only, never any text.

--gate <A's task file> --harness-tasks <the task file A launches with> exits 1 unless, for every A task, its first
simulated attack run is done with one .eval, and that run's approach and sent system prompt hash to the harness task
file's approach_sha256 and sol_system_sha256. tasks.py (la_sim_attack_sept) sets A's approach from that file and
refuses to build any sample whose rendered system prompt hashes to anything else, so a pass means A sends what the
first run sent.

Runs ON THE DEVBOX with the September snapshot's python (its inspect_ai reads the logs):
  $PY check_approach_hashes.py --launches ~/phase2/runs/<launch> ... --out <json> \\
      [--gate tasks_simattack2_20.json --harness-tasks <...>/config/pilot_tasks_amendment9_scaleup176.json]
      [--a-launches ~/phase2/runs/<A launch> ...]   (after A ran: each A run vs its task's first sim run)
"""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ARMS = ("attack", "sim_attack")
EMPTY = hashlib.sha256(b"").hexdigest()


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def msg_sha(m) -> str:
    return sha(m.content) if isinstance(m.content, str) else "list:" + sha(m.text)


def hash_eval(job: tuple[str, str]) -> dict:
    path, policy = job
    from inspect_ai.event import ModelEvent
    from inspect_ai.log import read_eval_log

    # resolve_attachments: events store long texts as attachment://<hash> references, which would be hashed instead
    (s,) = read_eval_log(path, resolve_attachments=True).samples
    calls = [e for e in s.events if isinstance(e, ModelEvent) and e.model == policy]
    sent = calls[0].input if calls else []
    approach = s.metadata.get("attack_approach")
    return {"approach_sha256": None if approach is None else sha(approach),
            "policy_calls": len(calls),
            "sent_roles": [m.role for m in sent[:2]],
            "sent_system_sha256": msg_sha(sent[0]) if sent and sent[0].role == "system" else None,
            "sent_user_sha256": msg_sha(sent[1]) if len(sent) > 1 and sent[1].role == "user" else None,
            # the first call in full, lengths and hashes only: every message (role and text), every user message's
            # length, the first non-empty user message, and the tool definitions sent with it
            "sent_n_messages": len(sent),
            "sent_system_len": len(sent[0].text) if sent and sent[0].role == "system" else None,
            "sent_user_lens": [len(m.text) for m in sent if m.role == "user"],
            "sent_first_nonempty_user": next(({"index": i, "sha256": msg_sha(m)} for i, m in enumerate(sent)
                                              if m.role == "user" and m.text.strip()), None),
            "sent_all_sha256": sha(json.dumps([[m.role, m.text] for m in sent])),
            "sent_tools_sha256": sha(json.dumps([t.model_dump() for t in calls[0].tools], sort_keys=True, default=str))
            if calls else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--launches", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gate", type=Path, help="A's task file (config/tasks_simattack2_20.json)")
    ap.add_argument("--harness-tasks", type=Path, help="the task file A launches with (the frozen harness's copy)")
    ap.add_argument("--a-launches", nargs="*", type=Path, default=[],
                    help="A's launches, after they ran: each A run must equal its task's first simulated run")
    args = ap.parse_args()
    assert not args.out.exists(), f"{args.out} exists"

    rows, jobs = [], []
    for launch in [*args.launches, *args.a_launches]:
        rec = json.loads((launch / "launch.json").read_text())
        expect = {t["key"]: t for t in rec["tasks"]}
        for r in rec["runs"]:
            if r["arm"] not in ARMS:
                continue
            t = expect[r["task"]]  # launch.json stores the task as its key (run_pilot.py save())
            row = {"launch": launch.name, "run": r["key"], "arm": r["arm"], "simulator": r["simulator"],
                   "real": r["real"], "a": launch in args.a_launches,
                   "task": t["key"], "state": r["state"], "attempts": len(r["attempts"]),
                   "expected_approach_sha256": t["approach_sha256"],
                   "expected_system_sha256": t["sol_system_sha256"], "expected_user_sha256": t["sol_input_sha256"]}
            evals = []
            if r["attempts"]:  # how it was launched, from the last attempt's record
                a = r["attempts"][-1]
                cells = a["env_vars"].get("PHASE2_CELLS")
                row |= {"task_fn": next((Path(c).name for c in a["cmd"] if "@" in c), None), "phase2_cells": cells,
                        "phase2_cells_sha256": hashlib.sha256(Path(cells).read_bytes()).hexdigest() if cells else None}
            if r["state"] == "done":  # the counted attempt is the last; its log dir, wherever the launch folder now is
                d = Path(r["attempts"][-1]["log_dir"])
                evals = sorted((launch / d.parent.name / d.name).glob("*.eval"))
            row["evals"] = [p.name for p in evals]
            rows.append(row)
            jobs.append((str(evals[0]), rec["policy"]) if len(evals) == 1 else None)

    with ProcessPoolExecutor(max_workers=16) as pool:
        hashed = list(pool.map(hash_eval, [j for j in jobs if j]))
    it = iter(hashed)
    for row, job in zip(rows, jobs):
        if job:
            row |= next(it)
            row["approach_match"] = row["approach_sha256"] == row["expected_approach_sha256"]
            row["system_match"] = row["sent_system_sha256"] == row["expected_system_sha256"]
            row["user_match"] = row["sent_user_sha256"] == row["expected_user_sha256"]

    # Across arms, per task: real run 1 (`attack` outside second_attack), real run 2 (second_attack) and the first
    # sim_attack@september run must have been sent the same first call (all messages, and tools) and system prompt.
    arms = defaultdict(dict)
    for r in rows:
        if "approach_match" in r and not r["a"] and (r["real"] or r["simulator"] == "september"):
            slot = ("run2" if r["launch"].endswith("second_attack") else "run1") if r["real"] else "sim"
            assert slot not in arms[r["task"]], f"two {slot} runs for {r['task']}"
            arms[r["task"]][slot] = r
    cross = {}
    for task, a in sorted(arms.items()):
        fields = ("sent_system_sha256", "sent_all_sha256", "sent_tools_sha256")
        cross[task] = {"slots": sorted(a), "equal": {f: len({x[f] for x in a.values()}) == 1 for f in fields}}
    full = [t for t, c in cross.items() if c["slots"] == ["run1", "run2", "sim"]]
    print(f"across arms: {len(cross)} tasks; {len(full)} with run 1, run 2 and sim; all three fields equal across "
          f"their arms in {sum(all(cross[t]['equal'].values()) for t in full)} of those, and in "
          f"{sum(all(c['equal'].values()) for c in cross.values() if len(c['slots']) > 1)} of the "
          f"{sum(len(c['slots']) > 1 for c in cross.values())} tasks with two or more arms")
    for t, c in cross.items():
        if len(c["slots"]) > 1 and not all(c["equal"].values()):
            print("ACROSS-ARM DIFFERENCE", t, c)

    gate = None
    if args.gate:
        harness = {c["key"]: c for c in json.loads(args.harness_tasks.read_text())["cells"]}
        gate = {"a_tasks": str(args.gate), "a_tasks_sha256": hashlib.sha256(args.gate.read_bytes()).hexdigest(),
                "harness_tasks": str(args.harness_tasks),
                "harness_tasks_sha256": hashlib.sha256(args.harness_tasks.read_bytes()).hexdigest(), "tasks": []}
        for a in json.loads(args.gate.read_text())["tasks"]:
            first = [r for r in rows if r["arm"] == "sim_attack" and r["task"] == a["key"]
                     and r["launch"] == a["first_sim_attack_launch"]]
            h = harness[a["key"]]
            g = {"key": a["key"], "first_run": first[0]["run"] if len(first) == 1 else None,
                 "a_approach_sha256": h["approach_sha256"], "a_system_sha256": h["sol_system_sha256"]}
            if len(first) == 1 and len(first[0]["evals"]) == 1:
                g |= {"first_approach_sha256": first[0]["approach_sha256"],
                      "first_system_sha256": first[0]["sent_system_sha256"]}
            c = cross.get(a["key"], {"slots": [], "equal": {}})
            g["across_arms"] = c
            # Tools are not compared across worlds: the simulated tools' parameter schemas differ from the real ones
            # by design of the harness (check_tool_defs.py), identically on every task; A is a simulated run like
            # the first, so its tools are compared with the first sim run's after it runs.
            g["first_tools_sha256"] = first[0].get("sent_tools_sha256") if len(first) == 1 else None
            g["pass"] = (g.get("first_approach_sha256") == g["a_approach_sha256"]
                         and g.get("first_system_sha256") == g["a_system_sha256"]
                         and c["slots"] == ["run1", "run2", "sim"]
                         and c["equal"]["sent_system_sha256"] and c["equal"]["sent_all_sha256"])
            gate["tasks"].append(g)
        gate["pass"] = all(g["pass"] for g in gate["tasks"]) and len(gate["tasks"]) == 20

    a_check = None
    if args.a_launches:  # after A ran: each A run against its task's first sim_attack@september run, all four hashes
        fields = ("approach_sha256", "sent_system_sha256", "sent_all_sha256", "sent_tools_sha256")
        a_check = []
        for r in rows:
            if r["a"]:
                first = arms.get(r["task"], {}).get("sim")
                hashed = "approach_match" in r
                a_check.append({"run": r["run"], "launch": r["launch"], "hashed": hashed,
                                "equal": {f: bool(first) and hashed and r[f] == first[f] for f in fields}})
    ok_a = a_check is None or all(x["hashed"] and all(x["equal"].values()) for x in a_check)
    args.out.write_text(json.dumps({"launches": [str(p) for p in args.launches],
                                    "a_launches": [str(p) for p in args.a_launches], "runs": rows,
                                    "across_arms": cross, "gate": gate, "a_check": a_check}, indent=1))

    by = Counter()
    for r in rows:
        k = (r["launch"], r["arm"] if r["real"] else f"{r['arm']}@{r['simulator']}")
        by[k + ("runs",)] += 1
        by[k + ("hashed",)] += "approach_match" in r
        for f in ("approach_match", "system_match", "user_match"):
            by[k + (f,)] += bool(r.get(f))
        by[k + ("approach_empty",)] += r.get("approach_sha256") == EMPTY
    print("launch\tarm\truns\thashed\tapproach_match\tsystem_match\tuser_match\tapproach_empty")
    for launch, arm in sorted({k[:2] for k in by}):
        print("\t".join([launch, arm] + [str(by[(launch, arm, f)]) for f in (
            "runs", "hashed", "approach_match", "system_match", "user_match", "approach_empty")]))
    for r in rows:
        if "approach_match" in r and not (r["approach_match"] and r["system_match"] and r["user_match"]):
            print("MISMATCH", r["launch"], r["run"],
                  {f: r[f] for f in ("approach_match", "system_match", "user_match")})
    if a_check is not None:
        print(f"A CHECK {'PASS' if ok_a else 'FAIL'}: {sum(x['hashed'] and all(x['equal'].values()) for x in a_check)}"
              f"/{len(a_check)} A runs equal their first run (approach, system, all messages, tools)")
        for x in a_check:
            if not (x["hashed"] and all(x["equal"].values())):
                print("A CHECK FAIL", x)
    if gate:
        print(f"GATE {'PASS' if gate['pass'] else 'FAIL'}: {sum(g['pass'] for g in gate['tasks'])}/"
              f"{len(gate['tasks'])} tasks; harness tasks sha256 {gate['harness_tasks_sha256']}")
        for g in gate["tasks"]:
            if not g["pass"]:
                print("GATE FAIL", g["key"], g["first_run"])
    sys.exit(0 if ok_a and (not gate or gate["pass"]) else 1)


if __name__ == "__main__":
    main()
