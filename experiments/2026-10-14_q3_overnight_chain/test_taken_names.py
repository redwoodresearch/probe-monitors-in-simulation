"""Write-once staging names stay taken (09 via d9 10-15 ~23:40): for every batch target in the chain state, every staged-set name already used
(a staging record in results/ naming it, or the state's own stage entry) has a try index <= the state's stage_tries for that target, so the next
staging try can never reuse one; and the names 09 listed as burned (refused-but-advanced, stopped in flight) stay at or below the counter. Names only.

  python test_taken_names.py <chain state dir> [extra results dir ...]
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BURNED = {("sp493-b", "sp493_b_main"): 3, ("sp493-g4", "sp493_g4_simh"): 2, ("sp493-g1_s4", "sp493_g1_sim_s4"): 1,  # try indices taken (09 @ d9 ~23:35)
          ("sp493-g2_s2", "sp493_g2_sim_s2"): 1, ("sp493-g2_s3", "sp493_g2_sim_s3"): 1}


def try_index(out: str) -> int:
    m = re.search(r"_v1_try([0-9]+)$", out)
    return int(m.group(1)) if m else 1


def main():
    st = json.loads((Path(sys.argv[1]) / "state.json").read_text())["batches"]
    dirs = [HERE / "results", *map(Path, sys.argv[2:])]
    outs = {}  # target -> set of try indices used by a staging record
    for d in dirs:
        for p in d.glob("chain_stage_*_20*.json"):
            r = json.loads(p.read_text())
            k = next((x for x in r if x not in ("git", "stamp")), None)
            out = (r.get(k) or {}).get("out") if k else None
            t = p.name[len("chain_stage_"):].rsplit("_20", 1)[0]
            if out:
                outs.setdefault(t, set()).add(try_index(out))
    bad, n_t = [], 0
    for bid, b in st.items():
        tries, stage = b.get("stage_tries") or {}, b.get("stage") or {}
        for t in set(b.get("targets") or []) | set(tries) | set(stage):  # a missing counter counts as 0 (step_stage's default)
            n_t += 1
            if t in stage and t not in tries:  # staged before counters existed: step_stage skips a staged target, so it never stages again
                continue
            n = tries.get(t, 0)
            used = outs.get(t, set()) | ({try_index(stage[t]["out"])} if t in stage else set())
            if used and max(used) > n:
                bad.append((bid, t, sorted(used), n))
    for (bid, t), k in BURNED.items():
        n = (st.get(bid, {}).get("stage_tries") or {}).get(t, 0)
        if n < k:
            bad.append((bid, t, f"burned try {k}", n))
    assert not bad, bad
    print(f"PASS: {n_t} batch targets: every used or burned name is at or below its try counter (a missing counter = 0)")


if __name__ == "__main__":
    main()
