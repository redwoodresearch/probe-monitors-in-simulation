"""Review check for #515 (thread LYDw) and #543 (LlKE), local and free: per judge-pilot design, how many judge inputs
(pair sides) had an output bisected by the probe's 32k cut (content tokens inside the cut, readout past it), which
as_probe_saw then dropped. Both sides of every pair, including the R2-undone copy of the planted pairs (rebuilt with
run_judge_fixed.r2_undone from the simulator's own outputs, as the pilot built it). The reader view is the level-1 scrub.
Writes results/recheck_judge_bisected_<stamp>.json (counts and provenance only).

  cd experiments/2026-10-09_fixed_sim && uv run --with pydantic-settings --with "transformers==4.56.2" --with jinja2 \\
      python recheck_judge_bisected.py <fixed-sim build dir> [results dir, default ./results]
"""
import glob
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml
from transformers import AutoTokenizer

import run_judge_fixed as J
from probe_shift.models import QWEN3_32B
from probe_shift.reader import segment_end_positions

HERE = Path(__file__).parent
run_dir = Path(sys.argv[1])
RES = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "results"  # judge_pilot_*.json are git-ignored outputs
rows = {}
for f in sorted((run_dir / "inputs").glob("shard_*.jsonl")):
    for line in f.read_text().splitlines():
        r = json.loads(line)
        rows[r["key"]] = r
cfg = yaml.safe_load((HERE / "config/judge_fixed.yaml").read_text())
tok = AutoTokenizer.from_pretrained(QWEN3_32B["name"], revision=QWEN3_32B["revision"])
out, inputs = {}, {}
for f in sorted(glob.glob(str(RES / "judge_pilot_*.json"))):
    inputs[Path(f).name] = hashlib.sha256(Path(f).read_bytes()).hexdigest()
    d, c = json.load(open(f)), Counter()
    for p in d["pairs"]:
        for slot, key in (("real_slot", p["real_key"]), ("other_slot", p["other_key"])):
            r = rows[key.removesuffix("__r2_undone")]
            msgs = r["messages"]
            if key.endswith("__r2_undone"):  # the planted copy: the fixed run with the simulator's own str_replace text back
                smp = J.samples_by_key(cfg["cohorts"]["sim-attack"], [r])[r["key"]]
                got = J.r2_undone(r["messages"], J.reader_messages(smp, ""), Counter())
                if got is None:
                    raise ValueError(f"{key}: no R2-only output to revert")
                msgs = got[0]
            seg = segment_end_positions(tok, J.scrubbed(msgs, 1), 32768)
            bis = [o for o in seg["outputs"] if o["end"] > o["start"] and o["n_tokens"] > o["end"] - o["start"]]
            c["sides"] += 1
            c[f"sides_{slot}"] += 1
            c["sides_with_bisected_output"] += bool(bis)
            c["bisected_outputs"] += len(bis)
    out[Path(f).name] = dict(c)
    print(Path(f).name, dict(c))
git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
res = {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "git": git, "inputs_sha256": inputs,
       "build_manifest_sha256": hashlib.sha256((run_dir / "manifest.json").read_bytes()).hexdigest(), "by_design": out}
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
path = RES / f"recheck_judge_bisected_{stamp}.json"
with open(path, "x") as fh:
    fh.write(json.dumps(res, indent=1))
print("->", path)
