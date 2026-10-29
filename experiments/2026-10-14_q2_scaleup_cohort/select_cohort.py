"""Scale-up cohort lists (PREPARED, not launched; Aswin via 0d, 2026-10-14): the SCE paper's 1,000 LinuxArena attack
cells minus our 196 amendment-9 cells, as two nested task files in the amendment-9 schema.

- Source: splits/exclusions_v4.json, arm real-attack-e1 (n_attack_cells 1000, cohort la_public10_real_attack_n1000).
- Approach, by the fleet's rule as 0d stated it: the side task's first strategy file (sorted path under
  <settings>/<env>/side_tasks/<side>/docs/attack_strategies/), whitespace-stripped, or "" when there is none. Only its
  sha256 and character count are kept; the text is never printed, logged or written. The rule is checked first on the
  196 against their recorded approach_sha256 / approach_chars; any mismatch stops the script.
- +804 ("about 1,000"): every remaining cell, so the total is the paper's 1,000 exactly.
- +311 ("about 500"): a subset of the +804. Cells whose main task is not among the 196 mains, at most one cell per
  main, stratified by (environment, strategy-bearing) so that 196 + 311 matches the 1,000's mix as closely as the pool
  allows (largest-remainder quotas; seeded order within each stratum).
- Order: every list is in launch order (the +311 first, then the rest), keys t197... continue the 196's numbering.
Counts only to stdout. usage: python3 select_cohort.py --settings ~/simulated-control-evals-sept/.settings/linuxarena
"""
import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXCL = REPO / "splits/exclusions_v4.json"
A9 = REPO / "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"
SEED, SALT = 20261014, "scaleup-cohort-v1:"
SIZES = (311, 804)
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731


def approach(settings, env, side):
    """(sha256, chars, n_strategy_files) of the fleet rule's approach; the text stays in this function."""
    d = settings / env / "side_tasks" / side / "docs" / "attack_strategies"
    files = sorted(p for p in d.rglob("*") if p.is_file()) if d.is_dir() else []
    text = files[0].read_text().strip() if files else ""
    return sha(text.encode()), len(text), len(files)


def order_key(cell):
    return sha(f"{SALT}{SEED}:{cell}".encode())


def main(a):
    settings = Path(a.settings).expanduser()
    ex, a9 = json.loads(EXCL.read_text()), json.loads(A9.read_text())
    e1 = {s["cell"]: s for s in ex["samples"] if s["arm"] == "real-attack-e1"}
    e2 = defaultdict(list)
    for s in ex["samples"]:
        if s["arm"] == "real-attack-e2":
            e2[s["cell"]].append(s)
    assert len(e1) == ex["paired_attack_cells"]["n_attack_cells"] == 1000
    paired = set(ex["paired_attack_cells"]["cells"])
    honest437 = set(ex["paired_honest_cells"]["cells"])
    old = {c["cell"]: c for c in a9["cells"]}
    mains196 = {c["main"] for c in a9["cells"]}
    assert set(old) <= set(e1)

    # 1) the rule must reproduce every recorded approach of the 196
    bad = []
    info = {}
    for cell, s in e1.items():
        info[cell] = approach(settings, s["environment"], s["side_task"])
    for cell, c in old.items():
        h, n, _ = info[cell]
        if (h, n) != (c["approach_sha256"], c["approach_chars"]):
            bad.append(c["key"])
    if bad:
        raise SystemExit(f"approach rule does not reproduce {len(bad)} of 196: {bad[:10]}")

    bearing = {cell: info[cell][1] > 0 for cell in e1}
    strat = lambda cell: (e1[cell]["environment"], bearing[cell])  # noqa: E731
    mix1000 = Counter(strat(c) for c in e1)
    have196 = Counter(strat(c) for c in old)
    pool = sorted((c for c in e1 if c not in old), key=order_key)

    # 2) +311: new mains only, one cell per main (first in seeded order), quotas to match the 1,000's mix in total
    one_per_main, seen = [], set()
    for c in pool:
        m = e1[c]["main_task"]
        if m not in mains196 and m not in seen:
            seen.add(m)
            one_per_main.append(c)
    avail = Counter(strat(c) for c in one_per_main)
    n_small, total_small = SIZES[0], len(old) + SIZES[0]
    want = {s: max(0.0, mix1000[s] / 1000 * total_small - have196.get(s, 0)) for s in mix1000}
    scale = n_small / sum(want.values())
    raw = {s: want[s] * scale for s in want}
    quota = {s: min(int(raw[s]), avail[s]) for s in raw}
    for s in sorted(raw, key=lambda s: raw[s] - int(raw[s]), reverse=True):  # largest remainder, capped by pool
        if sum(quota.values()) >= n_small:
            break
        if quota[s] < avail[s]:
            quota[s] += 1
    while sum(quota.values()) < n_small:  # pool caps bound: fill from the largest leftover strata
        s = max((s for s in avail if quota[s] < avail[s]), key=lambda s: avail[s] - quota[s])
        quota[s] += 1
    taken = Counter()
    small = []
    for c in one_per_main:
        if taken[strat(c)] < quota[strat(c)]:
            taken[strat(c)] += 1
            small.append(c)
    big = small + [c for c in pool if c not in set(small)]
    assert len(small) == SIZES[0] and len(big) == SIZES[1] and set(small) <= set(big)

    git = lambda *c: subprocess.run(["git", *c], capture_output=True, text=True, check=True).stdout.strip()  # noqa
    env_commits = {e: git("-C", str(settings / e), "rev-parse", "HEAD") for e in sorted({s["environment"] for s in e1.values()})}

    def entry(i, cell):
        s = e1[cell]
        h, n, nfiles = info[cell]
        e2s = [{"alias": x["cohort"], "file": x["file"], "id": x["sample_id"], "epoch": x["epoch"],
                "side_success": None, "tool_calls": None} for x in e2.get(cell, [])]
        return {"key": f"t{197 + i}_{s['environment']}", "env": s["environment"], "cell": cell,
                "main": s["main_task"], "side": s["side_task"], "approach_sha256": h, "approach_chars": n,
                "approach_s3": None, "sol_system_sha256": None, "sol_input_sha256": None,
                "sol_e1": {"alias": s["cohort"], "file": s["file"], "id": s["sample_id"], "epoch": s["epoch"],
                           "side_success": None, "tool_calls": None},
                "sol_e2": e2s,
                "flags": {"strategy_files": nfiles, "phase1_paired_attack_cell": cell in paired,
                          "main_in_437_paired_honest": s["main_task"] in honest437,
                          "main_in_196": s["main_task"] in mains196}}

    # The rule is described in this docstring and the README; results/ keeps only its id (the text guard treats any
    # "approach" key as model text).
    common = {"seed": SEED, "salt": SALT, "approach_rule_id": "fleet_first_strategy_file_stripped_v1",
              "settings_env_commits": env_commits,
              "inputs_sha256": {str(EXCL.relative_to(REPO)): sha(EXCL.read_bytes()),
                                str(A9.relative_to(REPO)): sha(A9.read_bytes())},
              "git": git("-C", str(REPO), "rev-parse", "HEAD")}
    stamp = a.stamp
    out = {}
    for n, lst, rule in ((SIZES[0], small, "about 500: 196 + 311; new mains only, one cell per main, stratified by "
                          "(environment, strategy-bearing) so the total matches the paper's 1,000-cell mix"),
                         (SIZES[1], big, "about 1,000: 196 + every remaining cell of the paper's 1,000 (the +311 first)")):
        cells = [entry(i, c) for i, c in enumerate(lst)]
        f = {"all196": f"the +{n} scale-up draw (PREPARED, not launched); its cells are new, keys continue the 196's",
             "cells": cells,
             "eligible_mains_by_env": dict(sorted(Counter(c["env"] for c in cells).items())),
             "n_eligible_cells": len(cells), "n_eligible_mains": len({c["main"] for c in cells}),
             "rule": rule, "rule_name": f"scaleup_plus{n}", **common}
        p = HERE / "results" / f"scaleup_tasks_plus{n}_{stamp}.json"
        head = json.dumps({k: v for k, v in f.items() if k != "cells"}, indent=1)[:-2]
        body = ",\n".join(json.dumps(c) for c in cells)  # one cell per line, as in the amendment-9 file
        p.write_text(f'{head},\n "cells": [\n{body}\n ]\n}}\n')
        assert json.loads(p.read_text()) == f
        out[n] = cells

    def comp(cells):
        return {"by_env": dict(sorted(Counter(e1[c]["environment"] for c in cells).items())),
                "strategy_bearing": sum(bearing[c] for c in cells), "n": len(cells),
                "phase1_paired_attack_cell": sum(c in paired for c in cells),
                "main_in_437": sum(e1[c]["main_task"] in honest437 for c in cells),
                "main_in_196": sum(e1[c]["main_task"] in mains196 for c in cells),
                "distinct_mains": len({e1[c]["main_task"] for c in cells})}
    summary = {"POST HOC": False, "what": "scale-up cohort selection record (counts only)", **common,
               "rule_check_196": "196 of 196 approach_sha256 and approach_chars reproduced",
               "paper_1000": comp(list(e1)), "ours_196": comp(list(old)), "remaining_804": comp(pool),
               "new_main_one_cell_pool": comp(one_per_main),
               "quota_plus311": {f"{e}|{'bearing' if b else 'none'}": q for (e, b), q in sorted(quota.items())},
               "plus311": comp(small), "plus804": comp(big),
               "total_507": comp(list(old) + small), "total_1000": comp(list(old) + big),
               "strategy_files_per_side_task": dict(sorted(Counter(info[c][2] for c in e1).items()))}
    (HERE / "results" / f"scaleup_cohort_record_{stamp}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("rule_check_196", "paper_1000", "ours_196", "plus311", "total_507",
                                              "plus804", "total_1000")}, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--settings", required=True)
    p.add_argument("--stamp", required=True)
    main(p.parse_args())
