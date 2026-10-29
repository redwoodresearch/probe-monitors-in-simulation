"""Epoch-2 staged-key lists for 09's Append 38 §4 pool-key note (d9 10-15 ~16:50): per launch, the folders <name>_v1__honest_a and
<name>_v1__honest_b, each with its keys "<arm>|<cell>" (epoch ep2). Counting (Append 38 §2 @ 0f5d1b429, clarified in the split record
@ 91feb3525): per launch x arm x task, the counted run is the unique non-rerun attempt without redrawn_because. Its .eval comes from the
launch's own upload manifests (the committed relay allowlist; each sha256 verified on the Volume by the relay record). Status rows (not in
a folder's key list): unusable by the launcher's own count (the counted attempt's policy calls have usable 0), counted without an .eval,
no counted attempt, and (+311) outside this launch's side of the split. Staging exclusions (the stager's c_gate) are not known before
staging and are added beside these lists after it. Inputs: the launch.json copy fetched from the Volume (its sha256 must equal the
allowlist's), the allowlist @ HEAD, and for +311 the split record @ 91feb3525. Keys and hashes only; no outcome is read.

  python3 make_ep2_keylists.py <name: ep2_t196 | ep2_311_l1 | ep2_311_l2 | ep2_nm> <launch.json copy> <allowlist path (results/hygiene/...)> <out.json>
"""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REL = "experiments/2026-10-14_q3_overnight_chain"
SPLIT = ("91feb3525", "experiments/2026-10-15_q3_realhonest_epoch2/results/split_311_P265_2026-10-15.json")
T196_ARM = {"_e2a": "honest_a", "_e2b": "honest_b"}
RUNS = {"ep2_t196": "2026-10-15_115153_ep2_t196", "ep2_311_l1": "2026-10-15_122516_ep2_311", "ep2_311_l2": "2026-10-15_144034_ep2_311b",
        "ep2_nm": "2026-10-15_123630_ep2_nm"}  # each list name is bound to its launch (the new-mains fork's run dirs)  # Append 38 §3: T196's e2a counts as honest_a epoch 2, e2b as honest_b epoch 2


def git(*a) -> bytes:
    return subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout


def main():
    name, lj_path, allow_rel, out = sys.argv[1:5]
    allow = json.loads(git("show", f"HEAD:{REL}/{allow_rel}"))
    run = allow["run"]
    if RUNS.get(name) != run:
        raise SystemExit(f"{name} is launch {RUNS.get(name)}, but the allowlist is for {run}")
    lj_raw = Path(lj_path).read_bytes()
    lj_sha = hashlib.sha256(lj_raw).hexdigest()
    want = next(f["sha256"] for f in allow["files"] if f["rel"] == "launch.json")
    if lj_sha != want:
        raise SystemExit(f"launch.json copy sha256 {lj_sha} != the relayed one {want}")
    lj = json.loads(lj_raw)
    if not lj.get("finished"):
        raise SystemExit("launch.json is not finished")
    evals = {f["rel"]: f for f in allow["files"] if f["rel"].endswith(".eval")}
    side, l2keys = None, None
    if name in ("ep2_311_l1", "ep2_311_l2"):
        sp = json.loads(git("show", f"{SPLIT[0]}:{SPLIT[1]}"))
        l2keys, side = set(sp["launch2_keys"]), ("l1" if name.endswith("l1") else "l2")
        assert len(l2keys) == sp["n_launch2_tasks"] == 44, len(l2keys)
    folders = {f"{name}_v1__honest_a": [], f"{name}_v1__honest_b": []}
    status, seen = [], set()
    for r in lj["runs"]:
        folder_arm, task = r["key"].split("/", 1)
        arm, cell = folder_arm, task
        if name == "ep2_t196":
            suf = next((s for s in T196_ARM if task.endswith(s)), None)
            if suf is None:
                raise SystemExit(f"{task}: no _e2a/_e2b suffix")
            arm, cell = T196_ARM[suf], task[: -len(suf)]
        key = f"{arm}|{cell}"
        if key in seen:
            raise SystemExit(f"{key}: two runs")
        seen.add(key)
        if side == "l1" and cell in l2keys:
            status.append({"key": key, "class": "outside launch 1's side of the split (index >= 265; excluded by rule @ 91feb3525)", "state": r.get("state")})
            continue
        if side == "l2" and cell not in l2keys:
            raise SystemExit(f"{key}: launch 2 holds a task below P")
        counted = [(i, a) for i, a in enumerate(r.get("attempts") or []) if not a.get("redrawn_because") and a.get("kind") != "rerun"]
        if len(counted) > 1:
            raise SystemExit(f"{key}: {len(counted)} counted attempts")
        if not counted:
            status.append({"key": key, "class": "no counted attempt", "state": r.get("state")})
            continue
        i, a = counted[0]
        rel_dir = a["log_dir"].split(f"/{run}/", 1)[1]
        if not re.fullmatch(re.escape(r["key"]) + r"(_redraw[1-9][0-9]*)?", rel_dir):  # this run's own <arm folder>/<task>, or its pre-start redraw folder
            raise SystemExit(f"{key}: counted attempt's log_dir {rel_dir} is not the run's folder {r['key']} (or its _redraw<n>)")
        evs = [e for rel, e in evals.items() if rel.startswith(rel_dir + "/") and "/" not in rel[len(rel_dir) + 1:]]
        if len(evs) != 1:
            status.append({"key": key, "class": "counted attempt without exactly one .eval in the upload manifests", "evals": len(evs), "state": r.get("state")})
            continue
        pcs = (a.get("outcome") or {}).get("policy_calls")
        row = {"key": key, "launch_task": task, "attempt_index": i, "attempt_kind": a.get("kind"), "eval": evs[0]["rel"], "eval_sha256": evs[0]["sha256"]}
        if not pcs or any("usable" not in pc for pc in pcs):  # never read a missing record as "unusable"
            status.append(row | {"class": "usability unknown (the counted attempt has no policy-call usable record)"})
            continue
        usable = sum(pc["usable"] for pc in pcs)
        if usable < 1:
            status.append(row | {"class": "unusable by the launcher's own count (counted attempt's policy calls: usable 0)"})
            continue
        folders[f"{name}_v1__{arm}"].append(row)
    used = [x["eval"] for f in folders.values() for x in f] + [x["eval"] for x in status if "eval" in x]
    if len(used) != len(set(used)):  # one transcript, one key
        raise SystemExit("an .eval is assigned to two keys")
    order = [t["key"] for t in lj["tasks"]]  # the launch's own task list, in its strict start order (launch.json is hash-verified above)
    if name == "ep2_t196":
        expected = {t[: -len(next(x for x in T196_ARM if t.endswith(x)))] for t in order}
    elif side == "l2":
        expected = set(order)
        if expected != l2keys:
            raise SystemExit(f"launch 2's task list ({len(expected)}) != the split's 44 launch2_keys")
    elif side == "l1":
        if len(order) != 309 or set(order[265:]) != l2keys:
            raise SystemExit("launch 1's task list is not 309 tasks whose positions >= 265 are the split's 44 keys")
        expected = set(order[:265])  # 1-based positions 1-265 of the hash-verified list
    else:
        expected = set(order)
    got = [x["key"] for f in folders.values() for x in f] + [x["key"] for x in status if not x["class"].startswith("outside")]
    want = {f"{arm}|{c}" for c in expected for arm in ("honest_a", "honest_b")}
    if len(got) != len(set(got)) or set(got) != want:  # every expected arm x cell exactly once, in a folder or as a status row
        raise SystemExit(f"arm x cell coverage: {len(set(got))} of {len(want)}; missing {sorted(want - set(got))[:5]}, extra {sorted(set(got) - want)[:5]}")
    doc = {"what": f"epoch-2 staged-key lists for {name} (09 Append 38 §4 pool-key note); keys '<arm>|<cell>', epoch ep2; keys and hashes only",
           "launch": run, "counting_rule": "Append 38 §2 @ 0f5d1b429 (counted run per task x arm: the unique non-rerun attempt without redrawn_because), as clarified in the split record @ 91feb3525",
           "split": None if side is None else {"record": f"{SPLIT[1]} @ {SPLIT[0]}", "side": side, "P_index0": 265},
           "arm_labels": "Append 38 §3: T196 _e2a -> honest_a, _e2b -> honest_b (suffix dropped from the cell)" if name == "ep2_t196" else "as configured (honest_a, honest_b)",
           "inputs": {"launch_json": {"volume_path": f"evals_v1/{run}/launch.json", "sha256": lj_sha}, "allowlist": f"{REL}/{allow_rel} @ {git('log', '-1', '--format=%h', '--', allow_rel).decode().strip()}"},
           "staging_exclusions": "not yet known (the stager decides them); added beside these lists after staging",
           "counts": {**{f: len(v) for f, v in folders.items()}, "status_rows": len(status), "runs": len(lj["runs"])},
           "folders": {f: sorted(v, key=lambda x: x["key"]) for f, v in folders.items()}, "status_rows": sorted(status, key=lambda x: x["key"])}
    Path(out).open("x").write(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc["counts"]), json.dumps(sorted({s["class"] for s in status})))


if __name__ == "__main__":
    main()
