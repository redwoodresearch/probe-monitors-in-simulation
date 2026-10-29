"""E1 fresh confirmation, the frozen sentence rule on the new reports (PLAN.md here; committed and run before any score is
read). The rule comes unchanged from experiments/2026-10-11_q2_e1_sentences/select_sentences.py (a471d57b) through
masks_lib, which reproduces stage 1's committed masks exactly. Never opens scores/; writes indices and counts only.

  uv run --no-project --python 3.12 --with "transformers==5.19.0" --with jinja2 python \\
      experiments/2026-10-11_q2_e1_confirm/select_confirm.py <confirm stage folder> <confirm index jsonl>
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from masks_lib import CONDS, build_masks  # noqa: E402

INDEX_SHA = "a4217da0379df98b118584e6015f5fe7eb49d7739d5ccd5f5127937304d2a127"  # fixed by power_check_2026-10-11_164216.json
MIN_TOKENS = 10


def main(folder: Path, index: Path):
    if hashlib.sha256(index.read_bytes()).hexdigest() != INDEX_SHA:
        raise SystemExit("confirm index differs from the one the power check fixed")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    for f in (Path(__file__).resolve(), HERE / "masks_lib.py"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    path_of = {r["key"]: r["path"] for r in map(json.loads, index.open())}
    masks = build_masks(folder, path_of)
    summary = {}
    for cond in CONDS + ("honest+conceal", "honest+neutral"):
        ms = [m for k, m in masks.items() if k.endswith("|" + cond)]
        n = sorted(len(m["scored"]) for m in ms); tot = sorted(m["report tokens"] for m in ms)
        summary[cond] = {"staged reports": len(ms), f"with at least {MIN_TOKENS} scored tokens": sum(x >= MIN_TOKENS for x in n),
                         "median scored tokens": n[len(n) // 2] if n else None, "median report tokens": tot[len(tot) // 2] if tot else None}
    out = {"plan": "PLAN.md (this folder)", "git": head, "index sha256": INDEX_SHA,
           "staged.json sha256": hashlib.sha256((folder / "staged.json").read_bytes()).hexdigest(), "summary by condition": summary, "masks": masks}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"confirm_masks_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, separators=(",", ":")) + "\n")
    print(json.dumps(summary, indent=1)); print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
