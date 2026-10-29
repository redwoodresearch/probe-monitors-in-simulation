"""Today's running spend (2026-10-15 PT, 00:00 to the read time), by source, from raw reads saved in <raw dir>.
- Modal: `modal billing report --for today -r h --tz America/Los_Angeles --json` (billed cost per app per hour; the
  current hour is partial). Ours = app names starting probe-shift_aswin-surya_; grouped by sub-<project>, with the
  phase2 server sets (sub-phase2_srv<hex>) and the chain (sub-q3chain) split out.
- Simulator API: every launch.json under /home/aswin (depth 6, dir name 2026-10-14* or 2026-10-15*) on each running box,
  read with jq over ssh: the spend[] series (time, actual_usd). Today = last actual_usd minus the last value before
  2026-10-15T00:00-07:00. Copies of one launch on several boxes are counted once (they must agree). Plus q2's replay
  spend_logged_usd from /replay/status/live.json on its Modal volume. Stopped boxes cannot be read.
- EC2: Cost Explorer is reachable but cannot attribute cost to our instances (cost-allocation tags not active, resource-
  level data not enabled), so EC2 is an ESTIMATE: on-demand Linux us-west-2 rate (AWS Pricing API) x instance-hours today,
  as a range. The lower bound counts only the current run of each running instance (since its last start). The upper
  bound counts from 00:00 (or the first start today) to the read time, for every instance that ran today; OS-initiated
  stops leave no CloudTrail event, so the true value lies in between. EBS, network and other services are excluded.
Usage: python3 -I spend_today.py <raw dir> <out json>
"""
import datetime as dt
import glob
import hashlib
import json
import os
import sys

RAW, OUT = sys.argv[1:3]
DAY0 = "2026-10-15T00:00:00-07:00"
PT = dt.timezone(dt.timedelta(hours=-7))
T0 = dt.datetime(2026, 10, 15, tzinfo=PT)
OURS = "probe-shift_aswin-surya_"


def load(n):
    with open(os.path.join(RAW, n)) as f:
        return json.load(f)


def read_at(n):
    return dt.datetime.strptime(open(os.path.join(RAW, n)).read().strip().replace(" PDT", "-0700"), "%Y-%m-%d %H:%M:%S%z")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


# ---------- Modal ----------
rows = load("modal_billing_today_h_pt.json")
hours = sorted({r["interval_start"] for r in rows})
assert all(h.startswith("2026-10-15T") and h.endswith("-07:00") for h in hours)
ours = [r for r in rows if r["description"].startswith(OURS)]
groups = {}
for r in ours:
    sub = r["description"][len(OURS):].split("_", 1)[0]
    rest = r["description"][len(OURS) + len(sub) + 1:]
    g = ("server sets (sub-phase2_srv*)" if sub == "sub-phase2" and rest.startswith("srv")
         else "chain (sub-q3chain)" if sub == "sub-q3chain" else f"other: {sub}")
    groups.setdefault(g, {"usd": 0.0, "apps": set()})
    groups[g]["usd"] += float(r["cost"])
    groups[g]["apps"].add(r["object_id"])
by_hour = {}
for r in ours:
    by_hour[r["interval_start"][11:16]] = by_hour.get(r["interval_start"][11:16], 0.0) + float(r["cost"])
modal_total = sum(float(r["cost"]) for r in ours)
other_ws = sum(float(r["cost"]) for r in rows if not r["description"].startswith(OURS))
top = sorted(groups.items(), key=lambda kv: -kv[1]["usd"])
modal = {
    "source": "modal billing report --for today -r h --tz America/Los_Angeles --json (workspace redwood-default), read "
              + open(os.path.join(RAW, "modal_read_at.txt")).read().strip(),
    "kind": "billed (Modal billing API), hourly; the last hour is partial",
    "hours_covered_pt": [hours[0][11:16], hours[-1][11:16]],
    "ours_usd": round(modal_total, 2),
    "ours_n_apps": len({r["object_id"] for r in ours}),
    "groups": {g: {"usd": round(v["usd"], 2), "n_apps": len(v["apps"])} for g, v in top if v["usd"] >= 1.0},
    "groups_under_1usd": {"n_groups": sum(v["usd"] < 1.0 for _, v in top), "usd": round(sum(v["usd"] for _, v in top if v["usd"] < 1.0), 2)},
    "ours_by_hour_pt": {h: round(v, 2) for h, v in sorted(by_hour.items())},
    "other_workspace_apps_usd_not_ours": round(other_ws, 2),
}

# ---------- Simulator API ----------
launches, per_box = {}, {}
for p in sorted(glob.glob(os.path.join(RAW, "sim_spend_*.jsonl"))):
    box = os.path.basename(p)[len("sim_spend_"):-len(".jsonl")]
    per_box[box] = {"file_sha256": sha(p), "n_launch_json": 0}
    for line in open(p):
        d = json.loads(line)
        assert not d.get("read_error"), d["f"]
        if box == "volcopy":  # a relayed copy is final only if its launch had finished when it was copied
            assert d["finished"], d["f"]
        per_box[box]["n_launch_json"] += 1
        name = d["f"].rstrip("/").split("/")[-2]
        series = d["series"]
        assert all(t.endswith("-07:00") for t, _ in series)
        vals = [v for _, v in series if v is not None]
        last = vals[-1] if vals else None
        base = [v for t, v in series if t < DAY0 and v is not None]
        rec = {"path": d["f"], "box": box, "n_snapshots": len(series), "first": series[0][0] if series else None,
               "last_time": series[-1][0] if series else None, "last_actual_usd": last,
               "before_today_usd": base[-1] if base else 0.0,
               "today_usd": round((last or 0.0) - (base[-1] if base else 0.0), 4),
               "monotonic": all(b >= a for a, b in zip(vals, vals[1:])), "attempts_live_usd": d["attempts_live_usd"]}
        if name in launches:
            prev = launches[name]
            same = (prev["last_actual_usd"], prev["n_snapshots"], prev["last_time"]) == (rec["last_actual_usd"], rec["n_snapshots"], rec["last_time"])
            prev.setdefault("copies", []).append({"box": box, "identical": same})
            if not same:  # a launch.json copied before its launch ended: keep the record with the latest snapshot
                if (rec["last_time"] or "") > (prev["last_time"] or ""):
                    rec["copies"] = prev["copies"]
                    launches[name] = rec
        else:
            launches[name] = rec
today = {k: v for k, v in launches.items() if v["today_usd"] != 0}
assert all(v["monotonic"] for v in today.values())
q2 = load("q2vol/live.json")
q2_ls = load("q2vol/replay_ls.json")
y = load("modal_billing_yesterday_h_pt.json")
# the replay counts as today's only if it demonstrably started today: every /replay entry on the volume was created today
# and none of the replay's Modal apps billed anything yesterday PT
q2_today = (all(e["created_modified"].startswith("2026-10-15 ") for e in q2_ls)
            and not any("sub-q2-replay" in r["description"] for r in y))
assert q2_today, "q2 replay spend may predate today"
q2_buckets = q2["rate_limit_errors_by_10min"]
sim_total = sum(v["today_usd"] for v in today.values()) + q2["spend_logged_usd"]
sim = {
    "source": "launch.json spend[].actual_usd series on each running box (jq over ssh, read "
              + open(os.path.join(RAW, "sim_read_at.txt")).read().strip() + "); q2 replay live.json on Modal volume "
              "probe-shift_aswin-surya_sub-q2-onpolicysim_data_kill-2026-11-14 (read " + open(os.path.join(RAW, "q2_read_at.txt")).read().strip() + ")",
    "kind": "readable logged snapshot delta: actual_usd logged by the launchers (in-progress launches include live attempts), last snapshot minus the last snapshot before 00:00 PT; snapshots straddling midnight leave a small amount unresolved by day; stopped boxes missing",
    "boxes_read": per_box,
    "boxes_not_readable_stopped": ["sims-astra (i-05b1d910fd70016e6; hosted simh_q3 and simh_nm_s5 today)",
                                   "q3-scaleup2 / box2 (i-0c02a38abcde32bfb)", "q3-ep2-nm (i-07f91f4dbd4192faa)"],
    "stopped_boxes_covered_by": ("the chain volume's relayed launch.json copies (probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14 "
                                 "evals_v1/<run>/launch.json, read " + open(os.path.join(RAW, "vol_read_at.txt")).read().strip()
                                 + "); only the spend series was extracted; every copy is of a finished launch; every 10-14/10-15 launch "
                                 "named in the chain's routing files is in a box read or a volume copy")
                                if os.path.exists(os.path.join(RAW, "sim_spend_volcopy.jsonl")) else None,
    "per_launch_today": {k: {f: v[f] for f in ("box", "today_usd", "before_today_usd", "last_actual_usd", "last_time")} | ({"copies": v["copies"]} if "copies" in v else {})
                         for k, v in sorted(today.items())},
    "n_launches_with_spend_today": len(today),
    "launches_usd": round(sum(v["today_usd"] for v in today.values()), 2),
    "q2_replay": {"spend_logged_usd": q2["spend_logged_usd"], "runs_total": q2["runs_total"], "runs_done": q2["runs_done"],
                  "status_written_pt": q2["status_written_pt"],
                  "runs_ended_by_10min_all_today": all(k.startswith("2026-10-15T") for k in q2_buckets),
                  "runs_ended_sum": sum(v["runs_ended"] for v in q2_buckets.values()),
                  "started_today_evidence": {"replay_dir_entries_created": sorted(e["created_modified"] for e in q2_ls),
                                             "replay_apps_billed_yesterday_pt": 0, "all_today": q2_today}},
    "total_usd_readable": round(sim_total, 2),
}

# ---------- EC2 ----------
inst = load("ec2_instances.json")
rates = load("ec2_rates.json")
ct = load("ct_events.json")
now = read_at("ec2_read_at.txt")
ec2_rows = []
for i in inst:
    ev = sorted(((e["t"], e["e"]) for e in (ct[i["id"]]["events"] or [])), key=lambda x: x[0])
    ev = [(dt.datetime.fromisoformat(t), e) for t, e in ev]
    last_start = dt.datetime.fromisoformat(i["launch"])
    before = [e for t, e in ev if t < T0]
    on_at_t0 = (before[-1] in ("StartInstances", "RunInstances")) if before else (last_start < T0)
    starts_today = [t for t, e in ev if t >= T0 and e in ("StartInstances", "RunInstances")]
    if i["state"] == "running":
        lo_h = (now - max(T0, last_start)).total_seconds() / 3600
    else:
        lo_h = 0.0
    if on_at_t0:
        up_from = T0
    elif starts_today:
        up_from = starts_today[0]
    elif i["state"] == "running":
        up_from = max(T0, last_start)
    else:
        up_from = None
    up_h = (now - up_from).total_seconds() / 3600 if up_from else 0.0
    r = rates[i["type"]]
    assert len(r) == 1
    ec2_rows.append({"name": i["name"].replace(OURS, ""), "id": i["id"], "type": i["type"], "state": i["state"], "usd_per_h": r[0],
                     "hours_low": round(lo_h, 2), "hours_high": round(up_h, 2),
                     "usd_low": round(lo_h * r[0], 2), "usd_high": round(up_h * r[0], 2), "_low": lo_h * r[0], "_high": up_h * r[0]})
ce = load("ce_today.json")
ec2 = {
    "kind": "ESTIMATE (instance-hours x on-demand rate), a range; not billed",
    "cost_explorer": {"reachable": True, "attribution": "not possible: tag groups show only untagged (rr_owner_id$) and "
                      "GetCostAndUsageWithResources is denied (resource-level data not enabled)",
                      "account_total_utc_day_2026_10_15_estimated_not_ours": round(float(ce["ResultsByTime"][0]["Groups"][0]["Metrics"]["UnblendedCost"]["Amount"]), 2)},
    "rates_source": "aws pricing get-products (us-east-1 endpoint), on-demand Linux, us-west-2, shared tenancy",
    "window_pt": [T0.isoformat(), now.isoformat()],
    "instances": [{k: v for k, v in x.items() if not k.startswith("_")} for x in ec2_rows],
    "usd_low": round(sum(x["_low"] for x in ec2_rows), 2),
    "usd_high": round(sum(x["_high"] for x in ec2_rows), 2),
}

out = {
    "what": "running spend today, 2026-10-15 PT (00:00 to the read times), by source; every number from a saved read",
    "raw_inputs_sha256": {os.path.relpath(p, RAW): sha(p) for p in sorted(glob.glob(os.path.join(RAW, "**", "*"), recursive=True))
                          if os.path.isfile(p) and not p.endswith(".err")},
    "modal": modal, "sim_api": sim, "ec2": ec2,
    "table": {"modal_billed_usd": modal["ours_usd"], "sim_api_actual_usd": sim["total_usd_readable"],
              "ec2_estimate_usd": [ec2["usd_low"], ec2["usd_high"]],
              "total_usd": [round(modal["ours_usd"] + sim["total_usd_readable"] + ec2["usd_low"], 2),
                            round(modal["ours_usd"] + sim["total_usd_readable"] + ec2["usd_high"], 2)],
              "missing": ("EC2 non-compute; Modal's current partial hour; sim API launches on stopped boxes that were never relayed to the chain volume (none among the routed launches)"
                          if os.path.exists(os.path.join(RAW, "sim_spend_volcopy.jsonl")) else
                          "sim API on the 3 stopped boxes (incl. simh_q3, simh_nm_s5); EC2 non-compute; Modal's current partial hour")},
}
with open(OUT, "w") as f:
    json.dump(out, f, indent=1)
    f.write("\n")
print(json.dumps(out["table"]))
print(json.dumps({"modal_groups": modal["groups"], "modal_hours": modal["hours_covered_pt"]}))
print(json.dumps({"sim_launches": {k: v["today_usd"] for k, v in sim["per_launch_today"].items()}, "q2": q2["spend_logged_usd"]}))
print(json.dumps({"ec2": [(x["name"], x["usd_low"], x["usd_high"]) for x in ec2_rows]}))
