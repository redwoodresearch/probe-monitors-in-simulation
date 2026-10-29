"""Fresh-task honest runs (PLAN.md here): the fresh main tasks, the parity tasks and the task file run_pilot.py reads.
Runs on the devbox, which holds the September settings. Task names and counts only; no transcript is read.

Fresh = main tasks in the September settings' public10 environments (<settings>/<env>/main_tasks/<main>/ with both
info.yml and scorer.py), minus every main named in any LinuxArena cohort record we have: the references file
(`references` and `uncovered_mains`), #36's runs.json, the simulated honest runs.json, and every cohort manifest under
--manifests (a main counts as used if its exact name appears as any JSON key or string value in a manifest, or as a
piece of one split on non-word characters and '__').
Parity = 20 old tasks with at least two valid real honest runs in #36's runs.json, outside zoo C's 274 sealed tasks.

  python make_fresh_tasks.py --settings ~/sims/simulated-control-evals-sept/.settings/linuxarena --manifests <dir> \
      --refs <references json> --runs36 <#36 runs.json> --simhonest <sim honest runs.json> --zooc <zoo C batch spec> \
      --out-dir <results dir> --task-file <config/pilot_tasks_fresh_honest.json>
"""

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ENVS = ("auto_workflow", "clinical_trial", "file_indexing", "media_processing", "model_registry", "shared_dotfiles",
        "slack_clone", "tiktok", "user_tracking", "web_scraping")
SEED, N_PARITY, STAGE, TAG = 20261014, 20, 50, "fresh1013"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731


def tokens(s):
    """The string itself plus its pieces split on non-word characters and on '__' (keys like
    'overlap800_202_<hash>__<main>:<side hash>__1' and cells like '<main>:<side>' name a main inside)."""
    out = {s}
    for piece in re.split(r"[^A-Za-z0-9_]+", s):
        out.add(piece)
        out.update(piece.split("__"))
    return out


def walk(o, out):
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(tokens(k))
            walk(v, out)
    elif isinstance(o, list):
        for v in o:
            walk(v, out)
    elif isinstance(o, str):
        out.update(tokens(o))


def main(a):
    settings = Path(a.settings).expanduser()
    mains, no_scorer = {}, []
    for env in ENVS:
        for d in sorted((settings / env / "main_tasks").iterdir()):
            if not d.is_dir():
                continue
            if (d / "info.yml").is_file() and (d / "scorer.py").is_file():
                mains[d.name] = env
            else:
                no_scorer.append(d.name)
    assert not set(no_scorer) & set(mains), "a main name repeats across environments"

    refs = json.loads(Path(a.refs).read_bytes())
    runs36 = json.loads(Path(a.runs36).read_bytes())
    simh = json.loads(Path(a.simhonest).read_bytes())
    used = {"references file": set(refs["references"]) | set(refs["uncovered_mains"]),
            "#36 runs.json": {r["main_task"] for r in runs36},
            "sim honest runs.json": {r["main_task"] for r in simh}}
    inputs = {Path(p).name: sha(p) for p in (a.refs, a.runs36, a.simhonest, a.zooc)}
    for m in sorted(Path(a.manifests).glob("*.json")):
        strings = set()
        walk(json.loads(m.read_bytes()), strings)
        used[m.name] = strings & (set(mains) | set(no_scorer))
        inputs[m.name] = sha(m)
    all_used = set().union(*used.values())
    cohort = used["references file"]
    assert len(cohort) == 542, len(cohort)
    fresh = sorted(set(mains) - all_used)
    fresh_no_scorer = sorted(set(no_scorer) - all_used)

    zooc = json.loads(Path(a.zooc).read_bytes())["task order"]
    assert len(set(zooc)) == 274 and set(zooc) <= cohort
    honest = [r for r in runs36 if r["arm"] == "real-honest"]
    per_task = Counter(r["main_task"] for r in honest)
    eligible = sorted(t for t, n in per_task.items() if n >= 2 and t not in set(zooc) and t in mains)
    parity = sorted(random.Random(SEED).sample(eligible, N_PARITY))
    order = list(fresh)
    random.Random(SEED).shuffle(order)

    cells, stages = [], {}
    smoke = sorted(t for t in cohort if t not in set(zooc) and t not in set(parity) and t in mains)[0]
    cells.append({"key": f"{TAG}_smoke", "env": mains[smoke], "main": smoke, "position": None, "run": None,
                  "stage": "smoke", "approach": ""})
    stages["smoke"] = [f"{TAG}_smoke"]
    blocks = [("parity", [(i, t, "q") for i, t in enumerate(parity, 1)])]
    for s, start in enumerate(range(0, len(order), STAGE), 1):
        blocks.append((str(s), [(i, t, "p") for i, t in enumerate(order[start:start + STAGE], start + 1)]))
    for stage, block in blocks:
        keys = []
        for k in (1, 2):
            for pos, main_task, prefix in block:
                key = f"{TAG}_{prefix}{pos:03d}_r{k}"
                cells.append({"key": key, "env": mains[main_task], "main": main_task, "position": pos, "run": k,
                              "stage": stage, "approach": ""})  # honest runs carry no attack approach
                keys.append(key)
        stages[stage] = keys
    out = {"rule_name": f"fresh_honest_{TAG}", "batch_tag": TAG, "runs_per_task": 2, "seed": SEED,
           "stages": stages, "cells": cells}
    head = {k: v for k, v in out.items() if k not in ("stages", "cells")}
    text = (json.dumps(head)[:-1] + ', "stages": {\n' + ",\n".join(f" {json.dumps(s)}: {json.dumps(k)}" for s, k in stages.items())
            + '},\n"cells": [\n' + ",\n".join(json.dumps(c) for c in cells) + "\n]}\n")
    assert json.loads(text) == out
    with open(a.task_file, "x") as fh:
        fh.write(text)

    rec = {"POST HOC": True, "plan": "PLAN.md", "seed": SEED, "settings": str(settings), "inputs sha256": inputs,
           "counts": {"mains with info.yml and scorer.py": len(mains), "mains without them": len(no_scorer),
                      "cohort (references file)": len(cohort),
                      "used by source (mains named)": {k: len(v) for k, v in used.items()},
                      "used, union": len(all_used & (set(mains) | set(no_scorer))),
                      "fresh (with scorer)": len(fresh), "fresh without scorer (excluded)": len(fresh_no_scorer),
                      "fresh by environment": dict(Counter(mains[t] for t in fresh)),
                      "parity-eligible (>= 2 valid #36 real honest runs, outside zoo C's 274)": len(eligible)},
           "fresh": fresh, "fresh order (seeded)": order, "parity": parity, "smoke": smoke,
           "fresh without scorer": fresh_no_scorer,
           "task file": {"name": Path(a.task_file).name, "sha256": hashlib.sha256(text.encode()).hexdigest(),
                         "runs by stage": {s: len(k) for s, k in stages.items()}}}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dest = Path(a.out_dir) / f"fresh_tasks_{stamp}.json"
    with open(dest, "x") as fh:
        fh.write(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec["counts"], indent=1)); print(rec["task file"]); print(dest)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("settings", "manifests", "refs", "runs36", "simhonest", "zooc", "out_dir", "task_file"):
        ap.add_argument("--" + k.replace("_", "-"), required=True)
    main(ap.parse_args())
