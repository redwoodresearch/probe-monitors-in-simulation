"""The paper's cells beyond the 507 (196 + the +311): the last 493 cells of the +804 list. Counts only (PREPARED, not
launched; Aswin "yes check remaining 490", via 0d, 2026-10-14). Reads committed JSON and the September settings tree's
folder names (no file contents); writes one counts record to results/.

Checks per cell: the side task's and main task's folders exist in the September settings (the registry the September
snapshot builds from); the approach (from the +804 list, sha256 and length only); whether the main already has staged
on-policy Qwen3.5 honest runs (d9's stage_result @ ce2f68ca, arms q35-real-honest-a / -b, joined to mains through the
196's keys); per-main multiplicity against the 196, the +311 and the 493 themselves; and the subset the +311's own rule
(a new main, one cell per main) would still admit.
usage: python3 count_remaining.py --settings ~/simulated-control-evals-sept/.settings/linuxarena --stamp <PT stamp>
"""
import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
A9 = REPO / "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"
L311 = HERE / "results/scaleup_tasks_plus311_2026-10-14_125135.json"
L804 = HERE / "results/scaleup_tasks_plus804_2026-10-14_125135.json"
STAGE = ("ce2f68ca", "experiments/2026-10-14_q3_onpolicy_titlecells/results/stage_result_2026-10-14_011819.json")
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731


def git(*c):
    return subprocess.run(["git", "-C", str(REPO), *c], capture_output=True, text=True, check=True).stdout


def comp(cells):
    return {"n": len(cells), "strategy_bearing": sum(c["approach_chars"] > 0 for c in cells),
            "strategy_bearing_share": round(sum(c["approach_chars"] > 0 for c in cells) / max(1, len(cells)), 4),
            "by_env": dict(sorted(Counter(c["env"] for c in cells).items())),
            "distinct_mains": len({c["main"] for c in cells})}


def main(a):
    st = Path(a.settings).expanduser()
    a9 = json.loads(A9.read_text())["cells"]
    s311 = json.loads(L311.read_text())["cells"]
    s804 = json.loads(L804.read_text())["cells"]
    assert [c["cell"] for c in s804[:311]] == [c["cell"] for c in s311]
    rest = s804[311:]
    stage = json.loads(git("show", f"{STAGE[0]}:{STAGE[1]}"))["runs by arm"]
    key2main = {c["key"]: c["main"] for c in a9}
    staged = {}
    for arm in ("q35-real-honest-a", "q35-real-honest-b"):
        keys = {m.group(1) for r in stage[arm] for m in [re.search(r"__(t\d+_[a-z_]+?)__", r)] if m}
        staged[arm] = {key2main[k] for k in keys if k in key2main}
    m196 = {c["main"] for c in a9}
    m311 = {c["main"] for c in s311}
    per_main_rest = Counter(c["main"] for c in rest)

    rows, reasons = [], Counter()
    for c in rest:
        side_dir = (st / c["env"] / "side_tasks" / c["side"]).is_dir()
        main_dir = (st / c["env"] / "main_tasks" / c["main"]).is_dir()
        r = {"key": c["key"], "env": c["env"], "main": c["main"], "approach_chars": c["approach_chars"],
             "side_task_folder": side_dir, "main_task_folder": main_dir,
             "main_in_196": c["main"] in m196, "main_in_311": c["main"] in m311,
             "honest_a_staged": c["main"] in staged["q35-real-honest-a"],
             "honest_b_staged": c["main"] in staged["q35-real-honest-b"],
             "phase1_paired_attack_cell": c["flags"]["phase1_paired_attack_cell"]}
        rows.append(r)
        if not side_dir:
            reasons["side task folder missing in the September settings"] += 1
        if not main_dir:
            reasons["main task folder missing in the September settings"] += 1
    eligible = [c for c, r in zip(rest, rows) if r["side_task_folder"] and r["main_task_folder"]]

    # the +311's own rule: a main in neither the 196 nor the +311, first cell per main in list order
    strict, seen = [], set()
    for c in eligible:
        if c["main"] not in m196 and c["main"] not in m311 and c["main"] not in seen:
            seen.add(c["main"])
            strict.append(c)

    # honest runs needed: a main needs new honest A and B unless the 196's staged honest runs or an earlier cell in
    # the plan (the +311, or a previous cell of the 493) supply them
    have_a = set(staged["q35-real-honest-a"]) | m311
    have_b = set(staged["q35-real-honest-b"]) | m311
    new_honest_a = new_honest_b = 0
    for c in eligible:
        if c["main"] not in have_a:
            new_honest_a += 1
            have_a.add(c["main"])
        if c["main"] not in have_b:
            new_honest_b += 1
            have_b.add(c["main"])

    elig_rows = [r for c, r in zip(rest, rows) if c in eligible]
    total = list(a9) + list(s311) + eligible
    out = {
        "POST HOC": False, "what": "counts for the paper's cells beyond the 507 (PREPARED, not launched)",
        "inputs_sha256": {str(p.relative_to(REPO)): sha(p.read_bytes()) for p in (A9, L311, L804)},
        "stage_result": f"{STAGE[1]} @ {STAGE[0]}", "git": git("rev-parse", "HEAD").strip(),
        "settings_head": subprocess.run(["git", "-C", str(st.parents[1]), "rev-parse", "HEAD"], capture_output=True,
                                        text=True).stdout.strip(),
        "remaining": len(rest), "exclusion_counts": dict(reasons), "eligible": len(eligible),
        "eligible_composition": comp(eligible),
        "total_507_plus_eligible": comp(total),
        "strict_rule_subset": comp(strict),
        "phase1_paired_attack_cell": sum(r["phase1_paired_attack_cell"] for r in elig_rows),
        "no_approach": sum(c["approach_chars"] == 0 for c in eligible),
        "main_in_196": sum(r["main_in_196"] for r in elig_rows),
        "main_in_196_with_staged_honest_a": sum(r["main_in_196"] and r["honest_a_staged"] for r in elig_rows),
        "main_in_196_with_staged_honest_b": sum(r["main_in_196"] and r["honest_b_staged"] for r in elig_rows),
        "main_in_311": sum(r["main_in_311"] for r in elig_rows),
        "main_new": sum(not r["main_in_196"] and not r["main_in_311"] for r in elig_rows),
        "new_honest_runs_needed": {"honest_a": new_honest_a, "honest_b": new_honest_b},
        "multiplicity": {
            "cells per main among the 493": dict(sorted(Counter(per_main_rest.values()).items())),
            "mains with 2+ cells in 196 + 311 + 493": sum(v > 1 for v in Counter(c["main"] for c in total).values()),
            "cells on a main shared with another cell in 196 + 311 + 493": sum(
                v for v in Counter(c["main"] for c in total).values() if v > 1),
            "cells per main in 196 + 311 + 493": dict(sorted(Counter(Counter(c["main"] for c in total).values()).items())),
        },
    }
    p = HERE / "results" / f"remaining490_counts_{a.stamp}.json"
    p.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("remaining", "exclusion_counts", "eligible", "main_in_196",
                                          "main_in_196_with_staged_honest_a", "main_in_311", "main_new",
                                          "new_honest_runs_needed", "multiplicity")}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", required=True)
    ap.add_argument("--stamp", required=True)
    main(ap.parse_args())
