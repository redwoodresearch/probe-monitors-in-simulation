"""Row 11 pilot: the 40 runs (PLAN.md: 40 runs, seed 20261011, held out of every reading), drawn from the E1
confirmation's resolved-context index. The index holds transcript text, so it goes to the git-ignored data/ folder;
results/ gets the keys, tasks and hashes only. Local, $0.

  python3 experiments/2026-10-11_q2_hide_same_instruction/make_pilot_index.py <confirm index jsonl>
"""
import hashlib
import json
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
INDEX_SHA = "a4217da0379df98b118584e6015f5fe7eb49d7739d5ccd5f5127937304d2a127"
SEED, N = 20261011, 40


def main(index: Path):
    if hashlib.sha256(index.read_bytes()).hexdigest() != INDEX_SHA:
        raise SystemExit("confirm index differs")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    me = Path(__file__).resolve()
    if (subprocess.run(["git", "ls-files", "--error-unmatch", str(me)], cwd=HERE, capture_output=True).returncode
            or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(me)], cwd=HERE).returncode):
        raise SystemExit("the script differs from HEAD: commit it before running")
    rows = sorted((json.loads(line) for line in index.open()), key=lambda r: r["key"])
    picked = sorted(random.Random(SEED).sample(range(len(rows)), N))
    pilot = [rows[i] | {"part": "pilot"} for i in picked]
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    d = ROOT / "data" / "hide_pilot"; d.mkdir(parents=True, exist_ok=True)
    with open(ix := d / f"index_pilot_{stamp}.jsonl", "x") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in pilot)
    out = {"plan": "PLAN.md (this folder)", "git": head, "confirm index sha256": INDEX_SHA, "seed": SEED, "runs": N,
           "pilot keys": [r["key"] for r in pilot], "tasks": len({r["main_task"] for r in pilot}), "arms": {a: sum(r["arm"] == a for r in pilot) for a in ("honest_a", "honest_b")},
           "index file": str(ix.relative_to(ROOT)), "index sha256": hashlib.sha256(ix.read_bytes()).hexdigest()}
    with open(dest := HERE / "results" / f"pilot_index_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "pilot keys"}, indent=1)); print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
