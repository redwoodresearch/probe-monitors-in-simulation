"""honest_b_relaunch: S_old's registered ALTERNATE real-honest calibration pool (09 RULE item 15 @ 4c095dec9; "R3 against B u C"), a labelled
sensitivity, never the lead's tau_B. Membership is EXACTLY the FPR sweep's frozen physical-run selection (RULE Part (a) section 1,
experiments/2026-10-14_fpr_sweep_unit/RULE.md @ a700b3bf6): d9's 980 counted keys (one physical run per logical key: the relaunch run where the
key was rerun, else the first launch's) minus the 22 unusable runs (17 error after start, 3 one-step submits, 2 environment start failures).
Key mapping <base task>_r<i> -> honest_r<i>|<base task>; all 958 enter, cue_D-flagged runs stay in. KEYS ONLY (no score is read).

Asserts (any failure STOPs): 980 counted files, one per logical key; the 22 are counted runs of exactly those launches; 958 = 980 - 22 with no
overlap; the 22 and their causes equal the Part (a) run record's excluded_with_cause by key and its usable count; the members' .eval sha256s
are disjoint from every S_old (T196) non-relaunch row of CUT's inventory_b_v2 (B, C, A and the attack arms); the members are a subset of the
963 staged relaunch rows, by key and eval sha, and the staged rows outside the membership are listed with their reason.

  python make_relaunch_pool.py      # writes results/honest_b_relaunch_members_<stamp>.json (exclusive create)
"""

import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SRC = {  # (commit, path, sha256): every input pinned
    "rule_part_a": ("a700b3bf6", "experiments/2026-10-14_fpr_sweep_unit/RULE.md", None),
    "rule_item15": ("4c095dec9", "experiments/2026-10-14_priority1_spec_09/RULE.md", None),
    "counted_manifest": ("9cd23708e", "experiments/2026-10-14_q3_q35_realhonest_1e4/config/manifest_rh1e4_eval_upload_v1.json",
                         "9785118d230c4ec4e67bb73ea6821547c672705689c8bc3d519bef56b49068ac"),
    "unusable": ("f0a371a29", "experiments/2026-10-14_q3_q35_realhonest_1e4/results/unusable_by_cause_2026-10-14_234651.json",
                 "80741883e54ee943674762fd0d97aa749063678d228c21e25f8c38be39cc97de"),
    "part_a_record": ("c9065498e", "experiments/2026-10-14_fpr_sweep_unit/results/fprsweep_run_a_2026-10-15_011009.json", None),
    "inventory_b_v2": ("a0bf963a7", "experiments/2026-10-15_cut_sensitivity/results/inventory_b_v2_2026-10-16_014820.json",
                       "fc0e84ea5260193f7e5f249f022b2f4b04de544794e06a7137d777c222d5bbf8"),
}
LAUNCH1, RELAUNCH = "2026-10-14_135334_honest_rh980", "2026-10-14_160335_honest_rh_relaunch"
UNUSABLE_CLASSES = {"error after start": 17, "1-step submit": 3, "environment start failure": 2}


def git_show(c, p):
    return subprocess.run(["git", "show", f"{c}:{p}"], capture_output=True, check=True, cwd=REPO).stdout


def logical(task):
    m = re.fullmatch(r"(.+)_r([1-9][0-9]*)", task)
    if not m:
        raise SystemExit(f"task {task}: not <base task>_r<i>")
    return f"honest_r{m.group(2)}|{m.group(1)}"


def main():
    raw, pins = {}, {}
    for k, (c, p, h) in SRC.items():
        b = git_show(c, p)
        sh = hashlib.sha256(b).hexdigest()
        if h and sh != h:
            raise SystemExit(f"{k}: sha256 {sh} != pinned {h}")
        raw[k], pins[k] = b, {"commit": c, "path": p, "sha256": sh}
    man, un, pa, inv = (json.loads(raw[k]) for k in ("counted_manifest", "unusable", "part_a_record", "inventory_b_v2"))
    files = man["files"]
    if len(files) != 980 or man["counts"]["counted_with_eval"] != 980:
        raise SystemExit(f"counted manifest: {len(files)} files, counted_with_eval {man['counts'].get('counted_with_eval')}")
    by_key = {}
    for f in files:
        if f["launch"] not in (LAUNCH1, RELAUNCH):
            raise SystemExit(f"{f['task']}: launch {f['launch']}")
        k = logical(f["task"])
        if k.split("|", 1)[0] != f["world"]:
            raise SystemExit(f"{f['task']}: world {f['world']} != key {k}")
        if k in by_key:
            raise SystemExit(f"{k}: two counted physical runs")
        if f["rerun_key"] and f["launch"] != RELAUNCH:  # a rerun key counts its relaunch run (keys never run in launch 1 are relaunch-only)
            raise SystemExit(f"{k}: a rerun key counted from {f['launch']}")
        by_key[k] = f
    excl = {}
    for cl, n in UNUSABLE_CLASSES.items():
        runs = un["classes"][cl]["runs"]
        if len(runs) != n or un["classes"][cl]["count"] != n:
            raise SystemExit(f"unusable {cl}: {len(runs)} runs, expected {n}")
        for r in runs:
            k = logical(r["key"])
            if k not in by_key or by_key[k]["launch"] != r["launch"]:
                raise SystemExit(f"unusable {k} ({r['launch']}): not the counted physical run")
            if k in excl:
                raise SystemExit(f"unusable {k}: listed twice")
            excl[k] = {"cause": cl, "launch": r["launch"], "eval_sha256": by_key[k]["sha256"]}
    members = sorted(set(by_key) - set(excl))
    if len(excl) != 22 or len(members) != 958 or set(members) & set(excl):
        raise SystemExit(f"958 = 980 - 22 fails: {len(members)} members, {len(excl)} excluded")
    pa_ex = {k: c for k, c in pa["relaunch"]["excluded_with_cause"]}
    if pa_ex != {k: v["cause"] for k, v in excl.items()} or pa["relaunch"]["usable"] != 958:
        raise SystemExit("the 22 (or their causes, or 958) differ from the FPR sweep Part (a) record")
    t196_other = {r["eval_sha256"]: r["world"] for r in inv["runs"] if r.get("task_set") == "T196" and r["world"] != "relaunch"}
    clash = sorted(k for k in members if by_key[k]["sha256"] in t196_other)
    if clash:
        raise SystemExit(f"members share an .eval with S_old's other worlds: {clash[:5]}")
    staged = {}
    for r in inv["runs"]:
        if r["world"] != "relaunch":
            continue
        k = logical(r["task"])
        staged.setdefault(k, []).append(r)
    bad = [k for k in members if k not in staged or not any(r["eval_sha256"] == by_key[k]["sha256"] for r in staged[k])]
    if bad:
        raise SystemExit(f"members not among the staged relaunch rows by key and eval sha: {bad[:5]}")
    n_staged = sum(len(v) for v in staged.values())
    outside = []
    for k, rows in sorted(staged.items()):
        for r in rows:
            if k in excl and r["eval_sha256"] == excl[k]["eval_sha256"]:
                why = f"unusable: {excl[k]['cause']} (unusable_by_cause @ f0a371a29)"
            elif k in by_key and r["eval_sha256"] != by_key[k]["sha256"]:
                why = f"not the counted physical run (counted: {by_key[k]['launch']}; this row: {r['launch']})"
            elif k not in by_key:
                why = "logical key not in the counted manifest"
            else:
                continue
            outside.append({"key": k, "staged_key": r["key"], "launch": r["launch"], "eval_sha256": r["eval_sha256"], "reason": why})
    if n_staged - len(outside) != len(members):
        raise SystemExit(f"reconciliation: {n_staged} staged - {len(outside)} outside != {len(members)} members")
    out = {"what": "honest_b_relaunch members: S_old's alternate real-honest calibration pool (09 item 15 @ 4c095dec9); keys only",
           "label": "honest_b_relaunch (alternate calibration); a labelled sensitivity, never the lead's tau_B (B u C + epoch 2 @ 64f00b09e)",
           "rule": ("membership = FPR sweep RULE Part (a) section 1 @ a700b3bf6 (980 counted keys, one physical run per logical key, minus the 22 "
                    "unusable runs); key <base task>_r<i> -> honest_r<i>|<base task>; all 958 enter; cue_D-flagged runs stay in"),
           "generator": "make_relaunch_pool.py", "stamp_pt": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S"),
           "sources": pins,
           "counts": {"counted": 980, "excluded": len(excl), "members": len(members), "members_by_launch": {
               L: sum(by_key[k]["launch"] == L for k in members) for L in (LAUNCH1, RELAUNCH)}, "staged_relaunch_rows": n_staged,
               "staged_outside_membership": len(outside), "excluded_by_cause": {c: sum(v["cause"] == c for v in excl.values()) for c in UNUSABLE_CLASSES}},
           "asserts": {"958_eq_980_minus_22_no_overlap": True, "equals_part_a_record_exclusions_and_usable": True,
                       "eval_disjoint_from_S_old_other_worlds": {"checked_against_rows": len(t196_other), "clashes": 0},
                       "members_subset_of_staged_by_key_and_eval_sha": True},
           "members": [{"key": k, "task": by_key[k]["task"], "launch": by_key[k]["launch"], "eval_path": by_key[k]["path"],
                        "eval_sha256": by_key[k]["sha256"]} for k in members],
           "excluded": [{"key": k} | v for k, v in sorted(excl.items())],
           "staged_outside_membership": outside}
    dest = HERE / "results" / f"honest_b_relaunch_members_{out['stamp_pt']}.json"
    blob = json.dumps(out, indent=1).encode()
    dest.open("xb").write(blob)
    print(json.dumps({"file": str(dest.relative_to(HERE)), "sha256": hashlib.sha256(blob).hexdigest(), "counts": out["counts"],
                      "outside": [(x["key"], x["reason"]) for x in outside]}, indent=1))


if __name__ == "__main__":
    main()
