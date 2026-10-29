"""Read-only rel9+ cap report: per-app bound (launcher rule), billed cost posted so far by app ID, and a serial projection.
Usage: python3 -I cap_report.py <n_more_simh_launches>"""
import collections, datetime, json, subprocess, sys
P2 = "/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/823cc0a0-23fc-47f8-b84f-0087b7bffcbd/scratchpad/p2"
MODAL, UTC, HOUR, FINAL = "/Users/aswinsurya/.local/bin/modal", datetime.timezone.utc, datetime.timedelta(hours=1), datetime.timedelta(hours=3)
led = [json.loads(l) for l in open(f"{P2}/dt4f/rel9plus_ledger.jsonl") if l.strip()]
res = {x["reserve"]: x for x in led if "reserve" in x and "app" not in x and "reconciled" not in x}
rec = {x["reconciled"]: x for x in led if "reconciled" in x}
obs = {x["observe"]: x for x in led if "observe" in x}
apps = {a: rid for rid, x in rec.items() for a in x.get("apps", [])}
t = datetime.datetime.now(UTC)
start = (min(datetime.datetime.fromisoformat(x["at"]).astimezone(UTC) for x in res.values()) - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
bill = json.loads(subprocess.run([MODAL, "billing", "report", "--start", start, "--resolution", "h", "--json"], capture_output=True, text=True, check=True).stdout)
rows = collections.defaultdict(list)
for x in bill:
    if x["object_id"] in apps:
        rows[x["object_id"]].append(x | {"_t": datetime.datetime.fromisoformat(x["interval_start"]).replace(tzinfo=UTC)})
out, tot_bound, tot_billed, n_final, walls, simh_walls = [], 0.0, 0.0, 0, [], []
for rid, r in sorted(res.items(), key=lambda kv: kv[1]["at"]):
    rc = rec.get(rid)
    if rc is None:
        out.append({"reserve": rid, "state": "UNRECONCILED"}); continue
    if rc.get("no_app_created") is True:
        out.append({"reserve": rid, "bound": 0.0, "state": "no app"}); continue
    for a in rc["apps"]:
        o = obs.get(a)
        billed = round(sum(float(x["cost"]) for x in rows[a]), 4)
        tot_billed += billed
        if o is None:
            out.append({"reserve": rid, "app": a, "state": "NOT OBSERVED STOPPED", "billed_posted": billed}); continue
        t0, t1 = (datetime.datetime.fromisoformat(o[k]).astimezone(UTC) for k in ("created_at", "stopped_at"))
        wall = (t1 - t0).total_seconds()
        fin = bool(rows[a]) and t1.replace(minute=0, second=0, microsecond=0) + HOUR + FINAL <= t and all(x["_t"] + HOUR + FINAL <= t for x in rows[a])
        b = billed if fin else (wall + 60) * r["rate_usd_s"]
        n_final += fin; tot_bound += b; walls.append(wall)
        if r["kind"] == "simh_arm": simh_walls.append(((wall + 60) * r["rate_usd_s"], billed, fin))
        out.append({"reserve": rid, "kind": r["kind"], "app": a, "wall_s": round(wall), "billed_posted": billed, "final": fin, "bound": round(b, 4)})
ceil_simh = max(r["ceiling_usd"] for r in res.values() if r["kind"] == "simh_arm")
n_more = int(sys.argv[1]) if len(sys.argv) > 1 else 0
wb = sorted(x[0] for x in simh_walls)
med, mx = wb[len(wb) // 2], wb[-1]
proj = [round(tot_bound + k * mx + ceil_simh, 2) for k in range(n_more)]
summ = {"at_pt": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "reservations": len(res), "apps": len(apps),
        "apps_final": n_final, "bound_now_usd": round(tot_bound, 2), "billed_posted_usd": round(tot_billed, 2),
        "apps_with_any_billing_row": sum(1 for a in apps if rows[a]), "simh_ceiling_usd": ceil_simh,
        "simh_wall_bound_median_usd": round(med, 4), "simh_wall_bound_max_usd": round(mx, 4),
        "serial_projection_bound_at_each_next_launch_usd (earlier walls at the max simh wall bound, nothing new final)": proj, "cap_usd": 15.0}
json.dump({"summary": summ, "apps": out}, open(f"{P2}/dt4f/cap_report.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
