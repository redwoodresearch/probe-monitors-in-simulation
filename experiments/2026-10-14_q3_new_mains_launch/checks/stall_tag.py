"""Tag every stall ALARM in one census JSON (sim_error_census_v2.py's per-launch "stall": ALARM = not finished, 0 running
(all arms), > 0 pending, idle > 15 min): "paused (PAUSE_SIMS present, expected)" if its run dir holds PAUSE_SIMS or a
PAUSED_*.txt marker (the census's "markers"), else "UNEXPLAINED (no PAUSE_SIMS)". Keys and counts only.
    python3 stall_tag.py <census json>   -> one line per ALARM; exit 3 if any is unexplained."""
import json
import sys

d = json.load(open(sys.argv[1]))
bad = 0
for box, launches in sorted(d["per_box"].items()):
    if not isinstance(launches, dict):
        continue
    for name, e in sorted(launches.items()):
        s = e.get("stall") if isinstance(e, dict) else None
        if not s or not s.get("ALARM"):
            continue
        m = s.get("markers") or []
        held = any(str(x).startswith("PAUSED:") for x in m)
        paused = "PAUSE_SIMS" in m
        bad += not (paused or held)
        tag = ("paused (PAUSE_SIMS present, expected)" if paused else "held (PAUSED_*.txt marker, expected)" if held
               else "UNEXPLAINED (no PAUSE_SIMS)")
        print(f"{box} {name}: {tag}; running {s.get('running_all_arms')}, pending {s.get('pending_all_arms')}, "
              f"idle {s.get('idle_minutes')} min, markers {m}")
print(f"{d.get('census_time_pt')}: {bad} unexplained")
sys.exit(3 if bad else 0)
