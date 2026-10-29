"""Which repetition settings would stop which saved completions, and what the check costs. Laptop, no spend, on local
copies of the proxy's token logs (archived on S3, manifest_v1.json in each archive):

  uv run --no-project --python 3.12 --with numpy --with pyyaml python \\
      experiments/2026-10-07_phase2_attack_check/analyze_repetition_settings.py <requests_*.jsonl.gz> ...

1. vLLM 0.30's own check (`check_sequence_repetition` and its helper, copied from vllm/v1/core/sched/utils.py),
   applied after every generated token as the scheduler does, for each setting in config/servers.yaml: where each
   completion would stop, and the check's cost per token (pure Python, as in vLLM, on this machine).
2. For every completion and every pattern length up to 512 tokens, the most consecutive repeats of a block of that
   length it ever reaches (the margin between ordinary output and a setting's min_count).

The known runaway turns (any completion that reached the 32,768-token cap) are reported apart from the ordinary ones.
Writes results/repetition_settings_<stamp>.json.
"""

import gzip
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import yaml

HERE = Path(__file__).parent
CAP, P_MAX = 32768, 512


def _has_repeating_pattern(token_ids, pattern_len: int, repetition_min_count: int) -> bool:  # vLLM 0.30, verbatim
    for n in range(1, pattern_len + 1):
        target_token = token_ids[-n]
        for m in range(1, repetition_min_count):
            if token_ids[-(pattern_len * m + n)] != target_token:
                return False
    return True


def check_sequence_repetition(token_ids, params) -> bool:  # vLLM 0.30, verbatim
    max_pattern_size, min_pattern_size, min_count = params.max_pattern_size, params.min_pattern_size, params.min_count
    if min_pattern_size <= 0:
        min_pattern_size = 1
    if max_pattern_size <= 0 or min_count < 2 or min_pattern_size > max_pattern_size:
        return False
    for pattern_len in range(min_pattern_size, max_pattern_size + 1):
        if pattern_len * min_count > len(token_ids):
            return False
        if _has_repeating_pattern(token_ids, pattern_len, min_count):
            return True
    return False


def max_repeats(ids: np.ndarray) -> dict[int, int]:
    """Pattern length -> the most consecutive repeats of a block of that length (only lengths repeating at least
    twice). A block of length p repeats c times where ids[k] == ids[k - p] holds for a run of p * (c - 1) tokens."""
    best = {}
    for p in range(1, min(P_MAX, len(ids) // 2) + 1):
        eq = ids[p:] == ids[:-p]
        if not eq.any():
            continue
        d = np.diff(np.concatenate(([0], eq.view(np.int8), [0])))
        run = int((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).max())
        if 1 + run // p >= 2:
            best[p] = 1 + run // p
    return best


def main() -> None:
    settings = yaml.safe_load((HERE / "config" / "servers.yaml").read_text())["repetition_settings"]
    completions = []
    for path in sys.argv[1:]:
        with gzip.open(path, "rt") as f:
            for r in map(json.loads, f):
                for c in r.get("choices") or []:
                    if c.get("token_ids"):
                        completions.append({"log": Path(path).name, "request_id": r["request_id"],
                                            "ids": c["token_ids"]})
    runaway = {c["request_id"] for c in completions if len(c["ids"]) >= CAP}
    out = {"logs": [Path(p).name for p in sys.argv[1:]], "completions": len(completions),
           "tokens": sum(len(c["ids"]) for c in completions), "runaway_turns": sorted(runaway), "settings": {}}
    window = max(s["max_pattern_size"] * s["min_count"] for s in settings.values())  # all the check ever reads
    for name, s in settings.items():
        params, stops, checked = SimpleNamespace(**s), [], 0
        t0 = time.perf_counter()
        for c in completions:
            ids = c["ids"]
            for k in range(1, len(ids) + 1):  # the scheduler checks after each new token
                checked += 1
                if check_sequence_repetition(ids[max(0, k - window):k], params):
                    stops.append({"request_id": c["request_id"], "stop_at": k, "tokens": len(ids),
                                  "runaway": c["request_id"] in runaway})
                    break
        out["settings"][name] = {"params": s, "stops": stops,
                                 "ordinary_stopped": sum(not x["runaway"] for x in stops),
                                 "runaway_stopped": sum(x["runaway"] for x in stops),
                                 "us_per_token": round((time.perf_counter() - t0) / checked * 1e6, 2)}
        print(name, {k: v for k, v in out["settings"][name].items() if k != "stops"}, flush=True)
    by_length = {}  # pattern-length band -> most repeats over ordinary completions
    for c in completions:
        if c["request_id"] in runaway:
            continue
        for p, reps in max_repeats(np.array(c["ids"], dtype=np.int64)).items():
            band = "1" if p == 1 else "2-4" if p <= 4 else "5-16" if p <= 16 else "17-63" if p <= 63 else "64-512"
            by_length[band] = max(by_length.get(band, 1), reps)
    out["ordinary_max_repeats_by_pattern_length"] = by_length
    print("ordinary completions, most repeats by pattern length:", by_length)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"repetition_settings_{stamp}.json"
    with open(path, "x") as f:
        json.dump(out, f, indent=1)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
