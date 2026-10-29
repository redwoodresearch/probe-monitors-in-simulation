"""Amendment 31's attempted / declined / unclear labels for Phase 2's attack runs. $0, local.

The rubric is the pinned code, run unchanged: `label()` from experiments/2026-10-07_q3_nano_monitor/make_attempt_labels.py
and the NOT_ATTEMPTED / ATTEMPT_CLAIM text match from run_self_reports.py, both read from git at main 8950b34 and
executed as written (each block is located by its exact first and last lines; the script stops if either moved).
Extended only to read Phase 2's logs: per run, the last submit call whose arguments include attack_successful, its
attack_description (missing reads as empty), and linux_scorer's side_task_success (in simulated arms the simulator's
own verdict, as amendment 31 says). No eye-read overrides (SPOT_CHECKED applies to Phase 1 only). The counted attempt
per run follows the launcher's rule (no redrawn_because, never a rerun). No model text is written: each description is
kept as its sha256 and length.

  uv run --no-project --python 3.14 python make_phase2_attempt_labels.py --run-dir <run folder> --arms attack,sim_attack@september \
      [--self-report-only sim_attack@september]   (the retrain, #342: simulated runs by self-report alone)
"""

import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
PIN = "8950b34"
D = "experiments/2026-10-07_q3_nano_monitor"


def pinned_block(path: str, start: str, end: str) -> str:
    src = subprocess.run(["git", "show", f"{PIN}:{path}"], capture_output=True, text=True, check=True, cwd=HERE).stdout
    i, j = src.index(start), src.index(end)
    return src[i:j]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--arms", required=True, help="arm folders, e.g. attack,sim_attack@september")
    ap.add_argument("--self-report-only", default="", help="arm folders whose side-task scorer is ignored (the retrain plan "
                    "#342: Phase 2 simulated runs have no instrument, so only the self-report counts), e.g. sim_attack@september")
    args = ap.parse_args()
    no_scorer = set(filter(None, args.self_report_only.split(",")))
    if no_scorer - set(args.arms.split(",")):
        raise ValueError(f"--self-report-only names arms not in --arms: {sorted(no_scorer - set(args.arms.split(',')))}")
    ns = {"re": re}
    exec(pinned_block(f"{D}/run_self_reports.py", "NOT_ATTEMPTED = re.compile(", "\n\n\n# Attack descriptions quote"), ns)
    exec(pinned_block(f"{D}/make_attempt_labels.py", "def label(r):", "\n\n\ndef main():"), ns)
    rd = Path(args.run_dir)
    launch = json.loads((rd / "launch.json").read_text())
    rows = []
    for r in launch["runs"]:
        folder = r["key"].split("/")[0]
        if folder not in args.arms.split(","):
            continue
        counted = [a for a in r.get("attempts") or [] if not a.get("redrawn_because") and a.get("kind") != "rerun"][:1]
        if not counted:
            rows.append({"arm": folder, "task": r["task"], "label": None, "rule": "no counted attempt"})
            continue
        ld = counted[0]["log_dir"]
        evs = sorted((rd / ld.split(f"/{rd.name}/", 1)[1]).glob("*.eval"))
        if not evs:  # e.g. t88 and t169, which fail before the first policy call (PLAN_second_attack.md): listed, no label
            rows.append({"arm": folder, "task": r["task"], "label": None, "rule": "no .eval file in the counted attempt"})
            continue
        if len(evs) != 1:
            raise ValueError(f"{r['key']}: {len(evs)} .eval files in its counted attempt")
        with zipfile.ZipFile(evs[0]) as z:
            names = [n for n in z.namelist() if n.startswith("samples/") and n.endswith(".json")]
            if len(names) != 1:
                raise ValueError(f"{evs[0]}: {len(names)} samples")
            s = json.loads(z.read(names[0]))
        rep = None
        for m in s.get("messages") or []:
            for c in (m.get("tool_calls") or []) if m.get("role") == "assistant" else []:
                if c.get("function") == "submit" and isinstance(c.get("arguments"), dict) and "attack_successful" in c["arguments"]:
                    rep = c["arguments"]
        desc = None if rep is None else str(rep.get("attack_description", ""))
        if desc and desc.startswith("attachment://"):
            desc = (s.get("attachments") or {})[desc.removeprefix("attachment://")]
        scorer = ((s.get("scores") or {}).get("linux_scorer") or {}).get("value") or {}
        x = {"attack_successful": None if rep is None else rep["attack_successful"],
             "says_not_attempted": None if desc is None else bool(ns["NOT_ATTEMPTED"].search(desc)) and not ns["ATTEMPT_CLAIM"].search(desc),
             "side_task_scorer": None if folder in no_scorer else scorer.get("side_task_success")}
        lab, rule = ns["label"](x)
        rows.append({"arm": folder, "task": r["task"], "run": str(evs[0].relative_to(rd)), "sample_id": s.get("id"), **x,
                     "attack_successful_type": type(x["attack_successful"]).__name__, "label": lab, "rule": rule,
                     "decided_by_scorer_clause": rule == "side-task scorer pass",
                     "description_sha256": None if desc is None else hashlib.sha256(desc.encode()).hexdigest(),
                     "description_chars": None if desc is None else len(desc)})
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"phase2_attempt_labels_{rd.name}{'_selfreport' if no_scorer else ''}_{stamp}.jsonl"
    with path.open("x") as f:
        f.writelines(json.dumps(o) + "\n" for o in rows)
    for arm in args.arms.split(","):
        rs = [o for o in rows if o["arm"] == arm]
        print(arm, len(rs), "runs:", dict(Counter(o["label"] for o in rs)), "| rules:", dict(Counter(o["rule"] for o in rs)),
              "| scorer clause:", sum(bool(o.get("decided_by_scorer_clause")) for o in rs))
    both = {o["task"] for o in rows if o["arm"] == "attack" and o["label"] == "attempted"} & \
           {o["task"] for o in rows if o["arm"] == "sim_attack@september" and o["label"] == "attempted"}
    print("cells with both attack runs attempted (real and September simulated):", len(both))
    print("->", path, hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
