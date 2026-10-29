"""PR review of #350 (local; stored per-output scores of A's chosen plant1 / plant2 probes, level 1, test environments):
does the 32k cut, rather than the plant, separate a planted copy from its twin? `over_32k`: pairs whose run the v2 run
marks truncated (its fixed text at level 0 is over the cut). Run-level AUROC (mean score) with all pairs, without the
over-32k pairs, and on only the readouts both copies have, each with analyze_review_checks.run_level's within-pair
null. Writes results/review_cutoff_<stamp>.json.

  uv run python experiments/2026-10-09_fixed_sim/analyze_review_cutoff.py <plant1 scores> <plant2 scores> <local v2 run>
"""

import gzip
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from analyze_review_checks import TEST, run_level  # noqa: E402


def main(plant1, plant2, run_dir):
    rng, res = np.random.default_rng(0), {}
    trunc = {r["key"]: r["truncated"] for f in sorted((Path(run_dir) / "inputs").glob("shard_*.jsonl"))
             for r in map(json.loads, f.read_text().splitlines()) if r["arm"] == "sim-attack"}
    for check, path in (("plant1", plant1), ("plant2", plant2)):
        rows = [r for r in json.loads(gzip.decompress(Path(path).read_bytes()))["levels"]["level1"]["rows"] if r["environment"] in TEST]
        sa = defaultdict(set)
        for r in rows:
            sa[r["key"]].add((r["step"], r["action"]))
        twins = sorted(k for k in sa if not k.endswith(f"__{check}") and f"{k}__{check}" in sa)
        hit = {k for k in twins if trunc[k]}
        both = {k: sa[k] & sa[f"{k}__{check}"] for k in twins}
        pos = f"sim-attack-fixed-{check}"
        res[check] = {"pairs": len(twins), "over_32k": len(hit), "readout_sets_differ": sum(sa[k] != sa[f"{k}__{check}"] for k in twins),
                      "all_pairs": run_level(rows, pos, rng),
                      "without_over_32k": run_level([r for r in rows if r["key"].removesuffix(f"__{check}") not in hit], pos, rng),
                      "readouts_in_both": run_level([r for r in rows if (r["step"], r["action"]) in both.get(r["key"].removesuffix(f"__{check}"), set())], pos, rng)}
        v = res[check]
        print(check, {k: v[k] for k in ("pairs", "over_32k", "readout_sets_differ")},
              *(f"{m}: {v[m]['auroc']} {v[m]['ci95']} runs {v[m]['runs']} null p95 {v[m]['null']['p95']}" for m in ("all_pairs", "without_over_32k", "readouts_in_both")), sep="\n  ")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = HERE / "results" / f"review_cutoff_{stamp}.json"
    with open(out, "x") as f:
        f.write(json.dumps({"files": [Path(plant1).name, Path(plant2).name], "checks": res}, indent=1))
    print(out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
