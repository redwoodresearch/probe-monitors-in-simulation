"""EXPLORATORY clock mask, the $0 count (laptop, read-only from S3): how many tool outputs and runs does the clock mask
change, per arm? These counts set the re-read volume and price (PLAN.md).

Reads claim B's 939 paired cells (splits/exclusions_v4.json `paired_attack_cells`) in real e1, real e2 and the
simulated arm, through `probe_shift.transcripts` (`reader_messages`, the reader's own serialization). For every tool
output: level 1 (`scrub(text, 1)`) vs level 1 + clock mask (`clock_mask.level1_clock`); changed if they differ.
Whole transcripts are counted, not only the first 32,768 tokens, so the counts are upper bounds on what a re-read
touches. The simulated arm gets fixed-sim v2's R1, R2 and R5 first (`fixed_sim/fixes.py`, its sha256 checked against
RULES.md's pin), as the build applied them: R1 can bring an earlier view's lines, clock strings included, into an
insert output (Codex). R6 is not re-applied: it moves clock values without adding or removing a clock string, so it
doesn't change which outputs the mask touches.

Writes counts per arm and scope (all 10 environments, the 5 test environments) and, per run, the (step, action) of
every changed output (keys and indices only, never text): `results/clock_mask_counts_<stamp>.json`.

  cd <worktree root> && PYTHONPATH="src:$HOME/simulated-control-evals/src:experiments/2026-10-11_claimB_clock_mask" \\
      uv run --no-project --python 3.14 --with boto3 --with pyyaml \\
      python experiments/2026-10-11_claimB_clock_mask/count_clock_mask.py
"""

import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import sys

from clock_mask import DAY_MONTH_YEAR, WEEKDAY_BEFORE_DATE, YEAR_AFTER_CLOCK, level1_clock
from probe_shift.scrub import scrub
from probe_shift.transcripts import file_samples, reader_messages

HERE = Path(__file__).parent
REPO = HERE.parents[1]  # laptop only (this script never runs on Modal)
FIXES = REPO / "experiments/2026-10-09_fixed_sim/fixes.py"
FIXES_PIN = "b1ba4a4e574e964c2ab2445066bf57c1b36b367959f2edc0e36454b26bd58655"  # RULES.md:75
assert hashlib.sha256(FIXES.read_bytes()).hexdigest() == FIXES_PIN, "fixes.py differs from RULES.md's pin"
sys.path.insert(0, str(FIXES.parent))
import fixes  # noqa: E402
ARMS = ("sim-attack", "real-attack-e1", "real-attack-e2")
TEST = {"clinical_trial", "media_processing", "shared_dotfiles", "tiktok", "user_tracking"}


def main():
    split = json.loads((REPO / "experiments/2026-10-09_fixed_sim/split.json").read_text())
    assert set(split["test"]) == TEST, split["test"]
    ex = json.loads((REPO / "splits/exclusions_v4.json").read_text())
    cells = set(ex["paired_attack_cells"]["cells"])
    assert len(cells) == 939, len(cells)
    want = defaultdict(dict)  # (cohort, file) -> (sample_id, epoch) -> sample row
    for s in ex["samples"]:
        if s["arm"] in ARMS and s["cell"] in cells and s["reason"] is None:
            want[(s["cohort"], s["file"])][(s["sample_id"], s["epoch"])] = s
    n_want = Counter(s["arm"] for d in want.values() for s in d.values())
    assert all(n_want[a] == 939 for a in ARMS), dict(n_want)

    c, per_run, seen = Counter(), {}, Counter()
    for (cohort, file), rows in sorted(want.items()):
        for sample in file_samples(cohort, file):
            row = rows.get((sample["id"], sample["epoch"]))
            if row is None:
                continue
            arm, scopes = row["arm"], ("all", "test") if row["environment"] in TEST else ("all",)
            seen[arm] += 1
            changed, step, action, n_out = [], -1, 0, 0
            msgs = reader_messages(sample, "")
            if arm == "sim-attack":
                msgs, _ = fixes.apply(msgs, ("R1", "R2", "R5"))
            for m in msgs:
                if m["role"] == "assistant":
                    step, action = step + 1, 0
                    continue
                if m["role"] != "tool":
                    continue
                n_out += 1
                l1 = scrub(m["content"], 1) if isinstance(m["content"], str) else None
                if l1 is not None and level1_clock(m["content"]) != l1:
                    changed.append([step, action])
                    for s in scopes:
                        c[(arm, s, "dmy_dates")] += len(DAY_MONTH_YEAR.findall(l1))
                        d1 = DAY_MONTH_YEAR.sub("<date>", l1)
                        c[(arm, s, "years_after_clock")] += len(YEAR_AFTER_CLOCK.findall(d1))
                        c[(arm, s, "weekdays_before_date")] += len(WEEKDAY_BEFORE_DATE.findall(YEAR_AFTER_CLOCK.sub(r"\1\2<year>", d1)))
                action += 1
            for s in scopes:
                c[(arm, s, "runs")] += 1
                c[(arm, s, "outputs")] += n_out
                c[(arm, s, "runs_changed")] += bool(changed)
                c[(arm, s, "outputs_changed")] += len(changed)
            per_run[row["key"]] = {"arm": arm, "environment": row["environment"], "tool_outputs": n_out, "changed": changed}
    assert all(seen[a] == 939 for a in ARMS), dict(seen)

    counts = {arm: {s: {k: c[(arm, s, k)] for k in ("runs", "runs_changed", "outputs", "outputs_changed", "dmy_dates",
                                                   "years_after_clock", "weekdays_before_date")}
                    for s in ("all", "test")} for arm in ARMS}
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    res = {"label": "EXPLORATORY clock mask: outputs and runs the mask changes, per arm (counts, keys and indices only)",
           "counts": counts, "per_run": per_run, "git": git,
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "clock_mask_sha256": hashlib.sha256((HERE / "clock_mask.py").read_bytes()).hexdigest(),
           "note": "whole transcripts (upper bounds); simulated arm with fixed-sim v2's R1, R2 and R5 applied (R6 not re-applied)"}
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")  # laptop local time (PT)
    (HERE / "results").mkdir(exist_ok=True)
    dst = HERE / "results" / f"clock_mask_counts_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(res))
    print(json.dumps(counts, indent=1))
    print("->", dst)


if __name__ == "__main__":
    main()
