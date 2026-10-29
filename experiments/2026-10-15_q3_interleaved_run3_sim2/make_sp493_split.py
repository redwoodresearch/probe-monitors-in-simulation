"""S_paper493 sim L1/L2 split at the 21:46-21:47 PAUSE_SIMS (d9 ~21:50; 09 accepted the split NOW): per cell key, did its run start
before its launch's pause? Read from the launcher's own start records (each attempt's "start" in launch.json), never from results.
- L1 (box3): started -> g1, unstarted -> g6 (tail); L2 (box4): started -> g2, unstarted -> g7 (tail). A cell started before the
  pause stays in g1/g2 forever (cut or failed runs follow the normal replacement rules; never moved to the tail).
Inputs: results/sp493_split_reads_2026-10-15/{L1,L2}_rows.txt (box reads ~21:59: PAUSE_SIMS mtime, launch.json sha256, then per
run: index, run key, task key, state, attempts, each attempt's start); the committed run lists
results/sp493_sim_runlist_L{1,2}_2026-10-15.json. Keys, states and timestamps only: no outcome read.
  python make_sp493_split.py            # writes results/sp493_sim_split_2026-10-15.json (never overwrites)
  python make_sp493_split.py --check a1|a2|a3|a4   # one assert per call (d9): a1 g1+g6 = the L1 list; a2 g2+g7 = the L2 list;
                                                   # a3 the four groups pairwise disjoint; a4 their union = the two lists
"""

import argparse
import datetime as dt
import json
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
READS = HERE / "results/sp493_split_reads_2026-10-15"
OUT = HERE / "results/sp493_sim_split_2026-10-15.json"
GROUPS = {"L1": ("g1", "g6"), "L2": ("g2", "g7")}


def runlist(L):
    return [c["key"] for c in json.loads((HERE / f"results/sp493_sim_runlist_{L}_2026-10-15.json").read_text())["cells"]]


def build():
    assert not OUT.exists(), f"{OUT} exists"
    rec = {"what": "S_paper493 sim L1/L2 split by key at the PAUSE_SIMS (launcher start records only; make_sp493_split.py)",
           "rule": "started before the pause (an attempt start earlier than the PAUSE_SIMS mtime) -> g1 (L1) / g2 (L2), forever; "
                   "unstarted -> the tail g6 (L1) / g7 (L2). The 22:45 time guard was never deployed (moot after the pause).",
           "pause": {}, "reads": {}, "counts": {}, "rows": []}
    for L, (gs, gu) in GROUPS.items():
        lines = (READS / f"{L}_rows.txt").read_text().splitlines()
        p = [ln.split() for ln in lines if ln.startswith("PAUSE ")][0]
        h = [ln.split() for ln in lines if ln.startswith("HDR ")][0]
        pause = int(p[1])
        rec["pause"][L] = {"path": f"/home/aswin/phase2/runs/2026-10-15_sp493_sim_{L}/PAUSE_SIMS", "mtime_epoch": pause, "mtime": " ".join(p[2:]),
                            "mtime_pt": dt.datetime.fromtimestamp(pause, ZoneInfo("America/Los_Angeles")).isoformat()}
        rec["reads"][L] = {"read": h[1], "launch_json_sha256": h[2], "finished": h[3]}
        assert h[3] == "None" and int(h[4]) == len(runlist(L)), h
        rows = [ln.split(" ", 6) for ln in lines if ln.startswith("ROW ")]
        assert [r[3] for r in rows] == runlist(L), f"{L}: launch order differs from the run list"
        for _, i, run_key, key, state, n, starts in rows:
            ts = [] if starts == "-" else [dt.datetime.fromisoformat(s).timestamp() for s in starts.split()]
            assert len(ts) == int(n) and run_key == f"sim_attack@september/{key}", (L, key)
            before = [t for t in ts if t < pause]
            assert len(before) == len(ts), f"{L} {key}: an attempt started at/after the pause"
            assert (state == "ready") == (not ts), f"{L} {key}: state {state} with {len(ts)} attempts"
            rec["rows"].append({"key": key, "launch": L, "group": gs if ts else gu, "file_position": int(i), "state_at_read": state,
                                "first_attempt_start": starts.split()[0] if ts else None, "attempts": len(ts)})
        for g in (gs, gu):
            rec["counts"][g] = sum(r["group"] == g for r in rec["rows"])
    OUT.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({"counts": rec["counts"], "pause": rec["pause"], "reads": rec["reads"]}, indent=1))


def check(a):
    rows = json.loads(OUT.read_text())["rows"]
    g = {x: [r["key"] for r in rows if r["group"] == x] for x in ("g1", "g2", "g6", "g7")}
    L1, L2 = runlist("L1"), runlist("L2")
    if a == "a1":
        ok = sorted(g["g1"] + g["g6"]) == sorted(L1) and len(L1) == 238
    elif a == "a2":
        ok = sorted(g["g2"] + g["g7"]) == sorted(L2) and len(L2) == 217
    elif a == "a3":
        sets = [set(v) for v in g.values()]
        ok = all(len(set(v)) == len(v) for v in g.values()) and all(not (x & y) for i, x in enumerate(sets) for y in sets[i + 1:])
    else:
        ok = set().union(*map(set, g.values())) == set(L1) | set(L2) and sum(map(len, g.values())) == len(L1) + len(L2) == 455
    print(f"{a}: {'PASS' if ok else 'FAIL'} {({k: len(v) for k, v in g.items()})}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", choices=["a1", "a2", "a3", "a4"])
    args = ap.parse_args()
    check(args.check) if args.check else build()
