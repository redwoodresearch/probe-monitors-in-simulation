"""AI-judge planning ($0): how many tokens a judge would read per run, and how many runs the 32k-token cut truncates,
from the level-1 staged reader inputs' metadata (`tokens`, `truncated`, arm, cell, environment). Each JSONL line is
parsed and its `messages` dropped at once; no message text is kept, printed or saved.

Per run: `read` = min(tokens, 32,768), the Qwen3-32B tokens the probe read under the cut; truncated = the staging's
`truncated` flag (checked against tokens > 32,768). Caveat: the staging kept these fields from before the level-1 scrub,
which only swaps dates and times for short placeholders, so totals are off by well under 1%; runs within a few tokens
of the cut could flip. For pricing this is immaterial; the run itself recounts on the text it sends. Cohorts as step A scored them: fixed sim and real e1 from cells
holding both; real e2 from cells holding e2 and e1. Reported for the 5 test environments (the runs step A scored) and
the 5 dev environments (where the prompt is frozen and the pilot runs).

  uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy python experiments/2026-10-11_claimB_judge/analyze_judge_tokens.py
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal
import numpy as np

HERE = Path(__file__).parent
VOL, STAGED = "probe-shift_aswin-surya_sub-la-activations", "2026-10-10_la_fixed-sim-v2_probe-inputs-b-l1_v1"
TEST = {"clinical_trial", "media_processing", "shared_dotfiles", "tiktok", "user_tracking"}
SIM, E1, E2 = "sim-attack-fixed", "real-attack-e1", "real-attack-e2"
CUT = 32768


def main():
    vol = modal.Volume.from_name(VOL, version=2)
    meta, digest = [], hashlib.sha256()
    for path in sorted(e.path for e in vol.listdir(f"{STAGED}/inputs")):
        raw = b"".join(vol.read_file(path))
        digest.update(hashlib.sha256(raw).digest())
        for line in raw.splitlines():
            r = json.loads(line)
            meta.append({k: r[k] for k in ("arm", "cell", "environment", "tokens", "truncated")})  # messages dropped here
        del raw
    arms_of = {}
    for m in meta:
        arms_of.setdefault(m["cell"], set()).add(m["arm"])
    cohort = {SIM: {SIM, E1}, E1: {SIM, E1}, E2: {E2, E1}}
    out = {"label": "AI-judge planning: tokens read and truncation from staged-input metadata (Qwen3-32B tokens; no text)", "cut": CUT,
           "arms_seen": sorted({m["arm"] for m in meta}), "rows": {}}
    for split in ("test", "dev"):
        for arm in (SIM, E1, E2):
            sel = [m for m in meta if m["arm"] == arm and (m["environment"] in TEST) == (split == "test") and arms_of[m["cell"]] >= cohort[arm]]
            tok = np.array([m["tokens"] for m in sel])
            trunc = np.array([bool(m["truncated"]) for m in sel])
            read = np.minimum(tok, CUT)
            out["rows"][f"{split}|{arm}"] = {"runs": len(sel), "truncated": int(trunc.sum()), "truncated_share": float(trunc.mean()),
                                            "flag_disagrees_with_tokens_over_cut": int((trunc != (tok > CUT)).sum()),
                                            "read_mean": float(read.mean()), "read_q": {q: float(np.percentile(read, q)) for q in (10, 50, 90)},
                                            "read_total": int(read.sum()), "full_mean": float(tok.mean()), "full_max": int(tok.max())}
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    out |= {"git": git, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "inputs": {"staged_inputs_sha256_of_shard_hashes": digest.hexdigest()}}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    dst = HERE / "results" / f"judge_tokens_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print("arms seen:", out["arms_seen"])
    for k, v in out["rows"].items():
        print(k, {x: (round(y) if isinstance(y, float) and y > 10 else round(y, 3) if isinstance(y, float) else y) for x, y in v.items() if x != "read_q"},
              "read q10/50/90", [round(x) for x in v["read_q"].values()])
    print("->", dst)


if __name__ == "__main__":
    main()
