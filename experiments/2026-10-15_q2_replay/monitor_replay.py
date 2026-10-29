"""Counts-only progress of the replay's live pass, for q2's reports (+15 min, then every 30 min).

It reads the data Volume mounted READ-ONLY, so it can never write or remove /replay/PAUSE, a ledger or the status file.
Per arm (attack = the 90 Phase-1 run1 runs, honest_b = the 45 placebo runs): attempts by status, failed attempts by error
class name, refusals, kept-real and substituted shares (stage_replay.py gate 5's definitions: kept-real outputs over
processed outputs; calls in steps with any substitution over simulated calls), $ per simulated call, per ok step and per
request, ok steps per minute; plus live.json, its age and the pass events. It returns no output text: counts, error class
names and times only. The flags are q2's stop rules (10-15): trip-wire, spend ceiling, a share above 5% in an arm. Failed
attempts are handled by the rule (backoff, two failures -> kept real, refusals) and count only in the 5% shares; an
UNHANDLED error (a worker crash, an exception outside the rule's paths, a ledger or journal write failure) aborts the whole
pass in this runner (no pass_end), so it shows as a stale or missing live.json, flagged here, and as a stopped app.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 modal run --detach \\
      experiments/2026-10-15_q2_replay/monitor_replay.py --label t196_v1
"""

import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-replay_monitor_kill-2026-11-14"
VOL = "probe-shift_aswin-surya_sub-q2-onpolicysim_data_kill-2026-11-14"
MANIFESTS = {"phase1": ("604bc7c60", "experiments/2026-10-15_replay_sim_outputs/phase1_manifest.json",
                        "872ddc7c68bd856f1b21d3effb5f368068d058372cedc9d749532c57bc9e2ce7"),
             "placebo": ("604bc7c60", "experiments/2026-10-15_replay_sim_outputs/placebo_manifest.json",
                         "8eb57dae18ff40fd1c6da7fdc178868d8773415252a57fe4b83510c7ea58cf94")}
CEILING_USD, STALE_S = 2700.0, 60.0
PLACEHOLDER = "[tool output withheld]"  # stage_replay.py's; counted in raw simulator outputs, never returned
FIT_USD_ON_K = (0.00067, 0.231023)  # $ per call = slope * position + intercept (Step 2 fit, sim_throughput @ 3f8936e2)
PT = ZoneInfo("America/Los_Angeles") if modal.is_local() else None  # the slim image has no tzdata

app = modal.App(APP)
vol = modal.Volume.from_name(VOL, version=2).read_only()


@app.function(image=modal.Image.debian_slim(python_version="3.12"), cpu=1, memory=2048, timeout=600, retries=0,
              volumes={"/data": vol})
def monitor(arm_of: dict, label: str, share_max: float) -> dict:
    root, now = Path("/data/replay"), datetime.now(timezone.utc)
    arms = {a: Counter() for a in sorted(set(arm_of.values()))}
    errors, stops, ok_t, start_t, bad_tail = {a: Counter() for a in arms}, Counter(), [], [], []
    per_run = {}  # run -> K_r, terminal output positions, $ so far, done
    for f in sorted((root / label / "ledger").glob("*.jsonl")):
        arm = arm_of[f.stem]
        c = arms[arm]
        c["runs_with_ledger"] += 1
        pr = per_run[f.stem] = {"K_r": None, "terminal": set(), "cost": 0.0, "done": False, "ok_pos": [], "ok_cost": 0.0, "calls": 0, "sub_calls": 0}
        lines = f.read_bytes().split(b"\n")
        lines = lines[:-1] if lines[-1] == b"" else lines
        for i, line in enumerate(lines):
            try:
                r = json.loads(line)
            except ValueError:  # a live ledger's last line can be mid-write in the Volume snapshot: counted, never silent
                if i != len(lines) - 1:
                    raise ValueError(f"{f.name}: line {i} of {len(lines)} unparseable (not the last line)")
                bad_tail.append({"run": f.stem, "line": i, "bytes": len(line), "nul_bytes": line.count(b"\0")})
                continue
            k = r["kind"]
            if k == "header":
                pr["K_r"] = r["K_r"]
            if k in ("result", "kept_real", "rejected"):
                pr["cost"] += r.get("cost_usd", 0.0) if k == "result" else 0.0
                if k != "result" or r["status"] == "ok":
                    pr["terminal"].update(r["js"])
                if k == "result" and r["status"] == "ok":
                    pr["ok_pos"] += r["js"]
                    pr["ok_cost"] += r["cost_usd"]
            if k == "start":
                start_t.append(r["t_utc"])
            elif k == "result":
                c["attempts"] += 1
                c[f"attempts_{r['status']}"] += 1
                c["cost_usd"] += r.get("cost_usd", 0.0)
                c["charged_usd"] += r.get("charge_usd", 0.0)
                if r["status"] == "ok":
                    n, subs = len(r["js"]), r.get("subs", {})
                    c["ok_steps"] += 1
                    c["simulated_calls"] += n
                    c["requests_in_ok_steps"] += r.get("requests", 0)
                    c["cost_ok_usd"] += r["cost_usd"]
                    subbed = any(v > 0 for v in subs.values())
                    c["calls_in_substituted_steps"] += n if subbed else 0
                    pr["calls"] += n
                    pr["sub_calls"] += n if subbed else 0
                    for o in r["outputs"]:
                        if PLACEHOLDER in o:
                            c["outputs_with_placeholder"] += 1
                            c["outputs_with_placeholder_in_substituted_steps"] += 1 if subbed else 0
                    for sk, v in subs.items():
                        c[sk] += v
                    ok_t.append((r["t_utc"], n))
                elif r["status"] == "failed":
                    errors[arm][r.get("error")] += 1
                    c["refusal_attempts" if r.get("refusal") else "failed_other_attempts"] += 1  # handled by the rule
            elif k == "kept_real":
                c["outputs_kept_real"] += len(r["js"])
                c[f"outputs_{r['status']}"] += len(r["js"])
            elif k == "rejected":
                c["outputs_rejected_by_inspect"] += len(r["js"])
            elif k == "backoff":
                c["backoffs"] += 1
            elif k == "stop":
                c["runs_stopped"] += 1
                stops[r["reason"]] += 1
            elif k == "done":
                c["runs_done"] += 1
                pr["done"] = True
    out, flags = {}, []
    for a, c in arms.items():
        processed = c["simulated_calls"] + c["outputs_kept_real"] + c["outputs_rejected_by_inspect"]  # as staging: kept_real_rejected counts
        sh = {"kept_real_share": c["outputs_kept_real"] / processed if processed else None,
              "refusal_share": c["outputs_kept_real_after_refusal"] / processed if processed else None,
              "substituted_share": c["calls_in_substituted_steps"] / c["simulated_calls"] if c["simulated_calls"] else None,
              "failed_attempts_share": c["attempts_failed"] / c["attempts"] if c["attempts"] else None,  # handled; no flag
              "usd_per_call": c["cost_ok_usd"] / c["simulated_calls"] if c["simulated_calls"] else None,
              "usd_per_ok_step": c["cost_ok_usd"] / c["ok_steps"] if c["ok_steps"] else None,
              "usd_per_request": c["cost_ok_usd"] / c["requests_in_ok_steps"] if c["requests_in_ok_steps"] else None}
        for name in ("kept_real_share", "substituted_share"):
            if sh[name] is not None and sh[name] > share_max:
                flags.append(f"{a}: {name} {sh[name]:.4f} > {share_max}")
        out[a] = {"runs_in_manifest": sum(v == a for v in arm_of.values()), **{k: round(v, 4) for k, v in sorted(c.items())},
                  **{k: (round(v, 6) if v is not None else None) for k, v in sh.items()}, "failed_by_error": dict(errors[a]),
                  "runs_with_substitution": {r: [q["sub_calls"], q["calls"], round(q["sub_calls"] / q["calls"], 4)]  # [sub calls, calls, share]
                                             for r, q in sorted(per_run.items()) if arm_of[r] == a and q["sub_calls"]},
                  "runs_over_10pct_substituted": sum(1 for r, q in per_run.items() if arm_of[r] == a and q["calls"] and q["sub_calls"] / q["calls"] > 0.10)}
    # projection: each unfinished run's remaining positions below its K_r, priced by the per-position fit with the slope
    # from Step 2 and the intercept calibrated on this pass's ok calls; K_r of runs not started yet from the dry_main headers
    dry_k = {}
    for f in (root / "dry_main" / "ledger").glob("*.jsonl"):
        dry_k[f.stem] = json.loads(f.open().readline())["K_r"]
    slope = FIT_USD_ON_K[0]
    pos = [j for pr in per_run.values() for j in pr["ok_pos"]]
    a_live = (sum(pr["ok_cost"] for pr in per_run.values()) - slope * sum(pos)) / len(pos) if pos else FIT_USD_ON_K[1]
    remaining, remaining_step2, mismatch, k_all = 0.0, 0.0, [], {}
    for rid in arm_of:
        pr = per_run.get(rid, {"K_r": None, "terminal": set(), "done": False})
        k_r = k_all[rid] = pr["K_r"] if pr["K_r"] is not None else dry_k[rid]
        if pr["K_r"] is not None and pr["K_r"] != dry_k[rid]:
            mismatch.append(rid)
        if not pr["done"]:
            left = [j for j in range(k_r) if j not in pr["terminal"]]
            remaining += sum(a_live + slope * j for j in left)
            remaining_step2 += sum(FIT_USD_ON_K[1] + slope * j for j in left)
    done = [rid for rid, pr in per_run.items() if pr["done"]]
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None  # noqa: E731
    charged = sum(c["charged_usd"] for c in arms.values())
    projection = {"slope_usd_per_position": slope, "intercept_live": round(a_live, 4), "intercept_step2": FIT_USD_ON_K[1],
                  "charged_so_far_usd": round(charged, 2), "remaining_usd": round(remaining, 2),
                  "projected_total_usd": round(charged + remaining, 2), "projected_total_usd_step2_intercept": round(charged + remaining_step2, 2),
                  "runs_done": len(done), "done_mean_K_r": mean([per_run[r]["K_r"] for r in done]),
                  "all_mean_K_r": mean(list(k_all.values())), "done_mean_cost_usd": mean([per_run[r]["cost"] for r in done]),
                  "remaining_mean_cost_per_unfinished_run_usd": round(remaining / max(len(arm_of) - len(done), 1), 2),
                  "K_r_live_vs_dry_mismatch": mismatch}
    rate = {}
    for m in (15, 30):
        t0 = (now - timedelta(minutes=m)).isoformat(timespec="seconds")
        recent = [n for t, n in ok_t if t >= t0]
        rate[f"last_{m}min"] = {"ok_steps_per_min": round(len(recent) / m, 2), "calls_per_min": round(sum(recent) / m, 2)}
    live_p = root / "status" / "live.json"
    live = json.loads(live_p.read_text()) if live_p.exists() else None
    ev = root / label / "pass_events.jsonl"
    events = [json.loads(x) for x in ev.read_text().splitlines()][-5:] if ev.exists() else []
    ended = bool(events) and events[-1]["kind"] == "pass_end"
    if not live and not ended:
        flags.append("no live.json and no pass_end")
    if live:
        live["age_s"] = round((now - datetime.fromisoformat(live["status_utc"])).total_seconds(), 1)
        if live.get("overrun_stop"):
            flags.append(f"trip-wire: {live['overrun_stop']}")
        if live.get("spend_logged_usd", 0) + live.get("reserved_in_flight_usd", 0) >= CEILING_USD:
            flags.append("spend ceiling reached")
        if live["age_s"] > STALE_S and not ended:
            flags.append(f"live.json {live['age_s']} s old without a pass_end: the pass may have crashed (check the app)")
    return {"read_utc": now.isoformat(timespec="seconds"), "label": label, "share_max": share_max, "arms": out, "rates": rate,
            "first_start_utc": min(start_t, default=None), "last_ok_utc": max((t for t, _ in ok_t), default=None),
            "run_stops_by_reason": dict(stops), "unparsed_last_lines": bad_tail, "projection": projection, "live": live, "pause_present": (root / "PAUSE").exists(),
            "pass_events_last5": events, "flags": flags}


@app.local_entrypoint()
def main(label: str = "t196_v1", share_max: float = 0.05):  # 0.05 = q2's rule; lower only to test that the flag fires
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, cwd=HERE, check=True).stdout  # noqa: E731
    arm_of = {}
    for c, p, sha in MANIFESTS.values():
        b = git("show", f"{c}:{p}")
        assert hashlib.sha256(b).hexdigest() == sha, p
        arm_of |= {f"{r['world']}__{r['task']}": r["arm_of"] for r in json.loads(b)["runs"]}
    assert len(arm_of) == 135
    res = monitor.remote(arm_of, label, share_max)
    stamp = datetime.now(PT).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"monitor_{label}_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps({"git": git("rev-parse", "HEAD").decode().strip(), "app": APP, "read_pt": stamp} | res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "pass_events_last5"})[:6000], "->", dst.name)
