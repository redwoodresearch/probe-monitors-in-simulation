"""server_set_map.py: which sim-honest launches use which policy-server set (d9 ~12:45: so we know which launches die if a
set dies). Reads the server records (app IDs) and a small launch table kept here; writes a timestamped snapshot
results/server_set_map_<HHMM>.json (append-only: one file per update).
  python3 server_set_map.py <servers dir>
"""
import datetime as dt
import json
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

SETS = {"25bccf": "servers_2026-10-15_090645_25bccf.json", "6ab18f": "servers_2026-10-15_090717_6ab18f.json",
        "aad41f": "servers_2026-10-15_100913_aad41f.json", "caf71f": "servers_2026-10-15_112527_caf71f.json",
        "353f90": "servers_2026-10-15_221729_353f90.json"}  # g5's own set (d9 ~21:5x; servers_g5.yaml)
# launch: (set, host, run dir, started PT or None if planned, state)
LAUNCHES = {
    "simh_q4": ("25bccf", "xbox1", "/home/aswin/phase2/runs/2026-10-15_simh_q4", "2026-10-15T09:27:26", "finished 2026-10-15T12:59:17"),
    "simh_q3": ("6ab18f", "astra", "/home/aswin/phase2/runs/2026-10-15_simh_q3", "2026-10-15T09:51:34", "finished 2026-10-15T14:04:58"),
    "simh_q1": ("aad41f", "xbox2", "/home/aswin/phase2/runs/2026-10-15_simh_q1", "2026-10-15T11:04:02", "finished 2026-10-15T14:23:17"),
    "simh_nm_s1": ("caf71f", "box1", "/home/aswin/phase2/runs/2026-10-15_simh_nm_s1", "2026-10-15T11:49:00", "finished 2026-10-15T14:22:50"),
    "simh_nm_s4d": ("caf71f", "box4", "/home/aswin/phase2/runs/2026-10-15_simh_nm_s4d", "2026-10-15T11:56:09", "finished (~13:01)"),
    "simh_nm_s3": ("caf71f", "box3", "/home/aswin/phase2/runs/2026-10-15_simh_nm_s3", "2026-10-15T12:37:45", "finished 13:37:20"),
    "simh_nm_s2": ("aad41f", "xbox3", "/home/aswin/phase2/runs/2026-10-15_simh_nm_s2", "2026-10-15T12:43:47", "finished 2026-10-15T14:25:11"),
    "simh_q2": ("aad41f", "xbox1", "/home/aswin/phase2/runs/2026-10-15_simh_q2", "2026-10-15T13:00:27", "finished 2026-10-15T15:06:03"),
    "simh_nm_s5": ("6ab18f", "astra", "/home/aswin/phase2/runs/2026-10-15_simh_nm_s5", "2026-10-15T14:13:06", "finished 2026-10-15T15:55:46"),
    "simh_nm_s3_rlr": ("caf71f", "box3", "/home/aswin/phase2/runs/2026-10-15_simh_nm_s3_rlr", "2026-10-15T14:15 (starter 14:15:51)", "finished 2026-10-15T14:57:39"),
    "simh_nm_s4r": ("caf71f", "box4", "/home/aswin/phase2/runs/2026-10-15_simh_nm_s4r", "2026-10-15T14:01 (starter 14:01:55)", "finished 2026-10-15T15:43:58"),
    # replacement launches (d9 ~14:30: "Pick whichever gets both started sooner, and update the set map"): planned, one at a time per host
    "simh_q1_rlr": ("aad41f", "xbox3", "/home/aswin/phase2/runs/2026-10-15_simh_q1_rlr", "2026-10-15T15:06:44", "finished 2026-10-15T16:19:43"),
    "simh_q4_rlr": ("caf71f", "xbox3", "/home/aswin/phase2/runs/2026-10-15_simh_q4_rlr", "2026-10-15T16:41:34", "finished (~17:11; v3: t361, t411, t413)"),
    # d9 ~15:30: S2_rlr moved to box1; ~15:41 withdrawn to xbox3 (settings digest differs); 15:4x restored to box1 (see below)
    # d9 15:4x: S2_rlr back to box1 on caf71f: box1's settings-tree digest gap is host-local, not an identity (09, 25h §13), the same kind as the acked Q1/Q2/Q4_rlr placements
    "simh_nm_s2_rlr": ("caf71f", "box1", "/home/aswin/phase2/runs/2026-10-15_simh_nm_s2_rlr", "2026-10-15T15:58:32", "finished 2026-10-15T16:39:40"),
    "simh_q2_rlr": ("caf71f", "box3", "/home/aswin/phase2/runs/2026-10-15_simh_q2_rlr", "2026-10-15T16:02:22", "finished 2026-10-15T17:12:01 (v2: 5 RL cells; t293 held; finished field read from box3 at 23:5x)"),
    "simh_nm_s4d_rlr": ("caf71f", "box4", "/home/aswin/phase2/runs/2026-10-15_simh_nm_s4d_rlr", "2026-10-15T15:59:28", "finished 2026-10-15T16:51:41 (nm258)"),
    "simh_nm_s4r_rlr": ("caf71f", "box4", "/home/aswin/phase2/runs/2026-10-15_simh_nm_s4r_rlr", "2026-10-15T16:56:47", "finished 2026-10-15T18:31:07 (nm353)"),
    "simh_nm_s4d_rlr2": ("caf71f", "box1", "/home/aswin/phase2/runs/2026-10-15_simh_nm_s4d_rlr2", "2026-10-15T16:53:29", "finished 2026-10-15T17:00:06 (nm107)"),
    "simh_q2_rlr2": ("caf71f", "box3", "/home/aswin/phase2/runs/2026-10-15_simh_q2_rlr2", "2026-10-15T17:22:21", "finished 2026-10-15T17:48:30 (t293)"),
    "simh_q3_rlr2": ("caf71f", "xbox3", "/home/aswin/phase2/runs/2026-10-15_simh_q3_rlr2", "2026-10-15T17:13:20", "finished 2026-10-15T18:20:33 (t298, t310)"),
    # S_paper493 / A25 (d9 ~17:20-18:20): on caf71f too
    "simh_t196fill": ("caf71f", "xbox3", "/home/aswin/phase2/runs/2026-10-15_simh_t196fill", "2026-10-15T18:29:25", "finished 2026-10-15T19:10:34 (A25 fill-in, 8/8 usable)"),
    "simh_sp493_t196": ("caf71f", "xbox3", "/home/aswin/phase2/runs/2026-10-15_simh_sp493_t196", "2026-10-15T19:12:32", "finished 2026-10-15T20:19:44 (S_paper493 g4, 14/14 usable)"),
    "simh_sp493_new": ("353f90", "xbox3", "/home/aswin/phase2/runs/2026-10-15_simh_sp493_new", "2026-10-15T22:28:35", "finished 2026-10-15T23:45:54 (S_paper493 g5, 34/35 usable; t699 EmptyPolicyCompletionError; set 353f90 stopped 23:56:37 by the design fork's waiter v4)"),
    "simh_q1_rlr2": ("caf71f", "box3", "/home/aswin/phase2/runs/2026-10-15_simh_q1_rlr2", "2026-10-15T18:24:28", "finished 2026-10-15T18:34:06 (t320; moved box1 -> box4 -> box3 for the 20:15 cut-off)"),
    "simh_q3_rlr": ("caf71f", "astra", "/home/aswin/phase2/runs/2026-10-15_simh_q3_rlr", "2026-10-15T16:06:17", "CUT by astra's EC2 stop (d9 read 16:31:12); superseded by q3_rlr2"),
}
src = Path(sys.argv[1])
now = dt.datetime.now(ZoneInfo("America/Los_Angeles"))
out = {"written_pt": now.strftime("%F %T"), "note": "25bccf: no further launch (0d: stopped after simh_q4 exits and its outputs are relayed). d9 ~16:00 (0d Modal audit): all remaining replacement launches on caf71f; aad41f stops after q1_rlr exits, 6ab18f after nm_s5 exits (the design fork's stop waiter, by procedure); caf71f stays until the last rlr launch. The set is a start argument only; no manifest pins it.", "sets": {}}
for s, f in SETS.items():
    rec = json.loads((src / f).read_text())
    out["sets"][s] = {"servers_record": f, "app_ids": [x["app_id"] for x in rec["servers"]],
                      "launches": [{"launch": k, "host": v[1], "run_dir": v[2], "started_pt": v[3], "state": v[4]}
                                   for k, v in LAUNCHES.items() if v[0] == s]}
p = Path(__file__).parent / "results" / f"server_set_map_{now:%H%M}.json"
p.write_text(json.dumps(out, indent=1) + "\n")
for s, v in out["sets"].items():
    print(s, len(v["app_ids"]), "servers:", [(x["launch"], x["state"].split(":")[0]) for x in v["launches"]])
print("->", p)
