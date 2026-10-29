"""New main tasks outside the SCE paper's 542 (PREPARED, not launched; 0d's follow-up, 2026-10-14): the main task is the
bootstrap's resampling unit, so new MAINS tighten the estimate. Counts, folder names and approach hashes only.

- Settings: the September snapshot's task folders (<settings>/<env>/main_tasks/<main>/, side_tasks/<side>/). "Clean" is
  the fresh-task rule (experiments/2026-10-13_q2_fresh_honest/PLAN.md): the folder has both info.yml and scorer.py.
- The paper's 542 mains: splits/exclusions_v4.json, arm real-attack-e1.
- The 327 fresh mains q2 ran honest on 10-13/14: config/fresh_task_list.json @ d9a827ae5 (aswin/q2-fresh-honest),
  stages 1-7; their honest outcome from score_fresh_s1234567 @ 37b5bbc9 ("errored run ids, not staged") plus the
  dummy main (errored_runs_2026-10-14.md).
- Pairing, as observed in the paper's 1,000 (no rule is on record): every side task of an environment is used, near
  evenly (100 cells per environment spread over its side tasks), whatever the main's name. The proposal mirrors that for
  one cell per new main: within each environment, new mains in seeded order are dealt round-robin over the side tasks
  in seeded order (seed 20261014, salt "newmain-pairing-v1:").
- Approach: per SIDE task by the fleet's rule (first file in sorted path order under docs/attack_strategies/, stripped,
  or ""), sha256 and length only; the rule is first checked against the 196's recorded values.
usage: python3 count_new_mains.py --settings ~/simulated-control-evals-sept/.settings/linuxarena --stamp <PT stamp>
"""
import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXCL = REPO / "splits/exclusions_v4.json"
A9 = REPO / "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"
FRESH = ("d9a827ae5", "experiments/2026-10-13_q2_fresh_honest/config/fresh_task_list.json")
VERDICT = ("37b5bbc9", "experiments/2026-10-13_q2_fresh_honest/results/score_fresh_s1234567_2026-10-14_080417.json")
SEED, SALT = 20261014, "newmain-pairing-v1:"
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731


def git(*c):
    return subprocess.run(["git", "-C", str(REPO), *c], capture_output=True, text=True, check=True).stdout


def approach(st, env, side):
    d = st / env / "side_tasks" / side / "docs" / "attack_strategies"
    files = sorted(p for p in d.rglob("*") if p.is_file()) if d.is_dir() else []
    text = files[0].read_text().strip() if files else ""
    return sha(text.encode()), len(text)


def dirs(p):
    return sorted(x.name for x in p.iterdir() if x.is_dir() and not x.name.startswith(("_", "."))) if p.is_dir() else []


def main(a):
    st = Path(a.settings).expanduser()
    ex = json.loads(EXCL.read_text())
    e1 = [s for s in ex["samples"] if s["arm"] == "real-attack-e1"]
    paper_mains = {(s["environment"], s["main_task"]) for s in e1}
    paper_sides = defaultdict(Counter)
    for s in e1:
        paper_sides[s["environment"]][s["side_task"]] += 1
    envs = sorted({s["environment"] for s in e1})

    for c in json.loads(A9.read_text())["cells"]:  # the approach rule reproduces the 196's records
        if approach(st, c["env"], c["side"]) != (c["approach_sha256"], c["approach_chars"]):
            raise SystemExit(f"approach rule mismatch at {c['key']}")

    fresh = json.loads(git("show", f"{FRESH[0]}:{FRESH[1]}"))
    fresh_cells = [c for c in fresh["cells"] if str(c["stage"]) in {"1", "2", "3", "4", "5", "6", "7"}]
    fresh_mains = {(c["environment"], c["main_task"]) for c in fresh_cells}
    run2main = {c["run_id"]: (c["environment"], c["main_task"]) for c in fresh_cells}
    verdict = json.loads(git("show", f"{VERDICT[0]}:{VERDICT[1]}"))
    broken = {run2main[r] for r in verdict["errored run ids, not staged"] if r in run2main}
    broken |= {m for m in fresh_mains if m[1].startswith("dummy_")}

    per_env, cells, approaches = {}, [], {}
    for env in envs:
        mains = dirs(st / env / "main_tasks")
        clean = [m for m in mains if (st / env / "main_tasks" / m / "info.yml").is_file()
                 and (st / env / "main_tasks" / m / "scorer.py").is_file()]
        sides = dirs(st / env / "side_tasks")
        assert set(paper_sides[env]) == set(sides), env  # the paper used every side task of the environment
        for side in sides:
            approaches[f"{env}|{side}"] = dict(zip(("approach_sha256", "approach_chars"), approach(st, env, side)))
        new = sorted((m for m in clean if (env, m) not in paper_mains),
                     key=lambda m: sha(f"{SALT}{SEED}:{env}:{m}".encode()))
        order = sorted(sides, key=lambda s: sha(f"{SALT}{SEED}:{env}:{s}".encode()))
        for i, m in enumerate(new):
            side = order[i % len(order)]
            ap = approaches[f"{env}|{side}"]
            cells.append({"env": env, "main": m, "side": side, **ap,
                          "in_fresh_327": (env, m) in fresh_mains,
                          "fresh_honest_ok": (env, m) in fresh_mains and (env, m) not in broken,
                          "fresh_honest_broken": (env, m) in broken,
                          "sol_attack_sample_on_record": False})
        ec = [c for c in cells if c["env"] == env]
        per_env[env] = {
            "main_folders": len(mains), "clean": len(clean), "in_paper_542": sum((env, m) in paper_mains for m in mains),
            "outside_542_clean": len(new), "outside_542_not_clean": sum(
                1 for m in mains if (env, m) not in paper_mains and m not in clean),
            "outside_542_clean_in_fresh_327": sum(c["in_fresh_327"] for c in ec),
            "outside_542_clean_not_in_fresh": sum(not c["in_fresh_327"] for c in ec),
            "fresh_honest_ok": sum(c["fresh_honest_ok"] for c in ec),
            "fresh_honest_broken": sum(c["fresh_honest_broken"] for c in ec),
            "side_tasks": len(sides), "side_tasks_with_strategy": sum(
                approaches[f"{env}|{s}"]["approach_chars"] > 0 for s in sides),
            "proposed_cells": len(ec), "proposed_cells_strategy_bearing": sum(c["approach_chars"] > 0 for c in ec),
            "formable": len(ec), "blocked_by_sol_hash_build_check": len(ec)}
    tot = {k: sum(v[k] for v in per_env.values()) for k in next(iter(per_env.values()))}
    out = {"POST HOC": False, "what": "new main tasks outside the paper's 542, one proposed cell each (PREPARED, not launched)",
           "seed": SEED, "salt": SALT, "settings_head": subprocess.run(
               ["git", "-C", str(st.parents[1]), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "inputs_sha256": {str(EXCL.relative_to(REPO)): sha(EXCL.read_bytes()), str(A9.relative_to(REPO)): sha(A9.read_bytes()),
                             f"{FRESH[1]} @ {FRESH[0]}": sha(git("show", f"{FRESH[0]}:{FRESH[1]}").encode()),
                             f"{VERDICT[1]} @ {VERDICT[0]}": sha(git("show", f"{VERDICT[0]}:{VERDICT[1]}").encode())},
           "git": git("rev-parse", "HEAD").strip(), "approach_rule_check_196": "196 of 196 reproduced",
           "fresh_mains_stages_1_7": len(fresh_mains), "fresh_mains_broken": len(broken),
           "paper_mains": len(paper_mains), "totals": tot, "by_env": per_env,
           "paper_cells_per_side_task_by_env": {e: sorted(paper_sides[e].values(), reverse=True) for e in envs},
           "approach_by_side_task": approaches, "proposed_cells": cells}
    p = HERE / "results" / f"new_mains_counts_{a.stamp}.json"
    head = json.dumps({k: v for k, v in out.items() if k != "proposed_cells"}, indent=1)[:-2]
    body = ",\n".join(json.dumps(c) for c in cells)
    p.write_text(f'{head},\n "proposed_cells": [\n{body}\n ]\n}}\n')
    assert json.loads(p.read_text()) == out
    print(json.dumps({"totals": tot, "fresh_mains_stages_1_7": len(fresh_mains), "fresh_mains_broken": len(broken),
                      "by_env": {e: {k: v[k] for k in ("main_folders", "clean", "in_paper_542", "outside_542_clean",
                                                       "outside_542_clean_in_fresh_327", "fresh_honest_ok",
                                                       "proposed_cells_strategy_bearing")} for e, v in per_env.items()}},
                     indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", required=True)
    ap.add_argument("--stamp", required=True)
    main(ap.parse_args())
