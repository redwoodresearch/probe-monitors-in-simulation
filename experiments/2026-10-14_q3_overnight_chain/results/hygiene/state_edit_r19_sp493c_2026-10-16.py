# Restart-19 state edit (d9 10-16 ~00:55): sp493-c was gate-stopped at commit_targets at 00:00:58 because its L3 donor lookup (t887 -> sp493-b's
# honest_a t805) ran before sp493-b's commit_targets (00:00:59). sp493-b's manifests are committed now, and add_index_donors waits instead of stopping.
# Re-open sp493-c at commit_targets; its try counters and abandoned tries are kept (next names: c_main / c_run2 _try3).
import json, os, sys, time
p = os.path.join(sys.argv[1], "state.json")
d = json.load(open(p))
c, b = d["batches"]["sp493-c"], d["batches"]["sp493-b"]
assert c["status"] == "STOPPED" and c.get("stopped_at") == "commit_targets" and c["done"] == ["ended", "boxfiles", "relay"], (c["status"], c.get("stopped_at"), c["done"])
assert b.get("commit_targets"), "sp493-b has not committed its manifests"
assert c["stage_tries"] == {"sp493_c_main": 2, "sp493_c_run2": 2} and c["stage_abandoned"] == c["stage_tries"], (c["stage_tries"], c.get("stage_abandoned"))
c["recovered_stop"] = {"stopped_at": c.pop("stopped_at"), "reason": c.pop("reason")[-300:], "at": time.strftime("%Y-%m-%d %H:%M:%S")}
c["status"] = "waiting"
open(p + ".tmp", "w").write(json.dumps(d, indent=1)); os.replace(p + ".tmp", p)
print("sp493-c re-opened at commit_targets", time.strftime("%H:%M:%S"))
