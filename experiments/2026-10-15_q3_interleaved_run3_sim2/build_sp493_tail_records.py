"""Build the S_paper493 X/Y post-start record and the rollout rows for 21:46-23:08 (pause, SIGINT, set stops, X/Y starts) from the
box reads and logs. Keys, states, shas and times only; no outcome field.
  python build_sp493_tail_records.py <scratchpad>   # writes results/sp493_tail_postlaunch_checks_2026-10-15.json + prints the rows
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
SP = Path(sys.argv[1])
OUT = HERE / "results/sp493_tail_postlaunch_checks_2026-10-15.json"
L = {"X": dict(box="xbox1 i-04331f272d6778df0", started="2026-10-15T23:05:31-07:00", set="39bf31", servers="sp493/srv/results/servers_2026-10-15_225559_39bf31.json", cap=4809, n=146,
               groups={"g6": 72, "g7": 54, "g3": 20}, gate="rr spX ALL CONDITIONS HOLD 23:04:54 (servers 39bf31 6/6 live; census sim_error_census_2026-10-15_2255.json @ 47c68ccba: 22:55:35 RL 0/18, running 19 <= 65)"),
     "Y": dict(box="box3 i-0cc7555acb95c694b", started="2026-10-15T23:08:25-07:00", set="a47cb6", servers="sp493/srv/results/servers_2026-10-15_225606_a47cb6.json", cap=4862, n=149,
               groups={"g6": 65, "g7": 67, "g3": 17}, gate="rr spY: dispatched after spX released the pacing lock (servers a47cb6 6/6 live; census as spX)")}
assert not OUT.exists(), f"{OUT} exists"
rec = {"what": "S_paper493 sim tail + L3 X/Y post-start checks (09 ack and d9 GO ~22:58, as relayed by d9): consumed references vs the index pins by key; server checks by canonical sha256; recorded inputs; exempt-file references; start-rate check. Read-only on xbox1/box3; no outcome field",
       "launches": {}}
rows = []
for k, m in L.items():
    lines = (SP / f"sp493/postlaunch_{k}_read.txt").read_text().splitlines()
    pid = [ln for ln in lines if ln.startswith("PID")][-1].split()[1]
    fr = json.loads([ln for ln in lines if ln.startswith("POST ")][-1][5:])
    bk = json.loads((SP / f"sp493/postlaunch_{k}.byKey.json").read_text())
    refs = bk["refs"]; neq = sum(r["equal"] for r in refs)
    assert len(refs) == neq == m["n"] and all(r["reference_state"] == "usable" for r in refs), k
    assert fr["server_checks"]["n"] == fr["server_checks"]["exit0"] == 6 and not any(fr["exempt_refs"].values()), k
    sr = [json.loads(ln) for ln in (SP / f"sp493/startrate_{k}.jsonl").read_text().splitlines() if ln.strip()]
    rate = [{"read": s["read"], "states": s["states"], "starts": s["starts"], "first_start": s["first_start"], "last_start": s["last_start"]} for s in sr]
    rec["launches"][k] = {**m, "run": bk["run"], "launcher_pid": int(pid), "first_read": {x: fr[x] for x in ("launch_json_sha256", "server_checks", "inputs", "exempt_refs", "first_start")},
                          "by_key_read_utc": bk["read_utc"], "launch_json_sha256_at_second_read": bk["launch_json_sha256_at_second_read"], "refindex": bk["refindex"],
                          "refindex_sha256": bk["refindex_sha256"], "equal_by_key": f"{neq}/{len(refs)}", "start_rate": rate,
                          "start_rate_expectation": "replay (results/sp493_tail_replay_2026-10-15/replay.json): 30 starts within the first 10 min; L1's environment-sorted order: 7", "refs": refs}
    sc, inp = fr["server_checks"]["sha256_canonical"][:8], fr["inputs"]
    cfg = [v for x, v in inp.items() if x.startswith("config/pilot_")][0][:8]; tsk = [v for x, v in inp.items() if x.startswith("config/tasks_")][0][:8]
    rows.append({"time_pt": m["started"], "action": "launch_start", "box": m["box"], "launch": f"sp493_sim_{k} ({bk['run']}; session sp493sim{k}; launcher PID {pid})", "il_launch": False,
                 "reason": (f"S_paper493 sim tail + L3 launch {k}: {m['n']} cells (g6 {m['groups']['g6']} + g7 {m['groups']['g7']} + g3 {m['groups']['g3']}; group per row), sims only, K 30, fresh set {m['set']}. "
                            "Approval as relayed by d9: 09 ACKED the X/Y run lists and both exemption files @ a400f9af0 (keys 146/149, 0 overlap, moved 258 = 258, t987 absent); d9 GO ~22:58; SPEND @ a400f9af0. "
                            f"Cap ${m['cap']:,}. {m['gate']}. Post-start (results/sp493_tail_postlaunch_checks_2026-10-15.json): consumed references = the {neq} index pins by key (refindex {bk['refindex_sha256'][:8]}...), all usable; "
                            f"server checks 6/6 exit 0 ({sc}...); inputs config {cfg} / tasks {tsk}; exempt files referenced 0 times; start rate {rate[0]['states'].get('running')} running about 5 min after the first start (read {rate[0]['read'][11:19]} UTC; replay: 30 within 10 min). Outcomes sealed until 09's read gates"),
                 "source": f"scratchpad il_watch/rr_sp{k}.log; {m['box'].split()[0]} reads 23:08-23:20", "outcomes_read": False, "outcome_info_available": "none", "reason_class": "registered", "reported_by": "design fork"})
OUT.write_text(json.dumps(rec, indent=1) + "\n")
ev = lambda t, a, b, launch, reason, src: {"time_pt": t, "action": a, "box": b, "launch": launch, "il_launch": False, "reason": reason, "source": src,  # noqa: E731
                                          "outcomes_read": False, "outcome_info_available": "none", "reason_class": "registered", "reported_by": "design fork"}
pre = [ev("2026-10-15T21:46:45-07:00", "pause", "box3 i-0cc7555acb95c694b / box4 i-07bce857a818b6f69", "sp493_sim_L1 / sp493_sim_L2",
          "09 accepted the split NOW (relayed by d9 ~21:46): PAUSE_SIMS touched (touch only, no signal) at /home/aswin/phase2/runs/2026-10-15_sp493_sim_L1/PAUSE_SIMS 21:46:45.82 and .../2026-10-15_sp493_sim_L2/PAUSE_SIMS 21:47:04.21. Cause: the task files were environment-sorted (make_sp493_sim.py), so the frozen launcher's strict order + 90 s per-environment stagger started ~0.7 runs/min (16-19 running, not 30); the tail (unstarted at the pause) relaunches shuffled. The on-box 22:45 guard was never deployed (moot); the laptop 22:45 guard (PID 91713) was stopped 21:47:12. Split: results/sp493_sim_split_2026-10-15.json @ 59f333d03",
          "scratchpad sp493/split/pause_now.txt; box reads"),
       ev("2026-10-15T22:35:46-07:00", "stop", "box3 i-0cc7555acb95c694b / box4 i-07bce857a818b6f69", "sp493_sim_L1 (PID 2405493) / sp493_sim_L2 (PID 3652205)",
          "drained (running 0 at 22:34:30, its own call); SIGINT by exact PID (verified our run_pilot for that --out, running 0 re-read): L1 22:35:46, L2 22:36:01. Exited: L1 finished 22:35:50 (done 101 = g1, not run 137 = g6; launch.json 486d0cc1...), L2 finished 22:36:06 (done 96 = g2, not run 121 = g7; 20195129...)",
          "results/sp493_tail_prelaunch_2026-10-15/l1_l2_drain_sigint.txt"),
       ev("2026-10-15T22:55:28-07:00", "server_stop", "Modal", "server sets 97cdc9 / 99398b / 423232 (S_paper493 real arms, B fork)",
          "waiter v3h (PID 377) was stuck: the sets were unused on every box from 21:48 (keepalive off), but its quiet rule needs >= 5 load lines and these apps had 0 containers (0 lines), so it looped 'not quiet yet'. App list read just before (operator-reported ~22:54; the list JSON carries no timestamp): all 18 exact IDs 'deployed', tasks 0. v3h stopped 22:54:57 (own process; logged in v3h_stop_waiter.log); the 18 IDs stopped from 22:54:57; app list after (operator-reported 22:55:28, the command's laptop clock): 18/18 stopped",
          "scratchpad sp493/rec/applist_bsets_{pre,post}.json, il_watch/apps_{97cdc9,99398b,423232}.txt"),
       ev("2026-10-15T22:55:38-07:00", "start", "Modal", "server sets 39bf31 (X) / a47cb6 (Y)",
          "two fresh 6 x H200 sets, servers_sp493tail.yaml @ a400f9af0 (servers_sp493sim.yaml with hard_timeout_h 5; idle exit 30 min, startup idle 60 min); launched at operator-reported 22:55:38 / 22:55:41 (the laptop clock at the commands; corroborated by the server records' names 225559 / 225606) while 09 reviewed (d9 ~22:55); SPEND @ a400f9af0",
          "scratchpad sp493/srv/launch_{X,Y}.log, results/servers_2026-10-15_225559_39bf31.json, servers_2026-10-15_225606_a47cb6.json")]
print("\n".join(json.dumps(r) for r in pre + rows))
