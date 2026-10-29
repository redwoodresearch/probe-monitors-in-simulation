"""While the chain is STOPPED: re-queue an IL batch (argv[2]) from commit_targets with rev _v2 (09 via d9 10-15: re-stage from the v2 manifests into
_try2). Backs up state.json first; touches only that batch. argv[3]: the key check commit named in the reason."""
import json, shutil, sys, time
from pathlib import Path
st = Path(sys.argv[1]) / "state.json"
BID, KC = sys.argv[2], sys.argv[3]
bak = st.with_name(f"state.json.bak_{BID}_v2_{time.strftime('%H%M%S')}")
shutil.copy2(st, bak)
s = json.loads(st.read_text())
b = s["batches"][BID]
T = list(b["targets"])
assert b["status"] == "attest" and b["done"][-1] == "attest" and b.get("stage_tries") == {t: 1 for t in T}, (b["status"], b.get("stage_tries"))
assert all(b["stage"][t]["epochs"] == {} and b["stage"][t]["out"] == f"{t}_v1" for t in T), "first staging is not the all-excluded _v1"
first = {k: b.get(k) for k in ("stage", "commit_targets", "record_commit", "hygiene_commit", "provenance_commit", "attest_commit", "extract", "compare")}
for k in ("targets", "commit_targets", "stage", "format", "shards", "repro", "extract", "cfg_committed", "extract_job", "extract_attempt", "compare",
          "labels", "labels_sim2", "record_commit", "hygiene_commit", "provenance", "provenance_commit", "attest_commit", "stats", "stage_unknown"):
    b.pop(k, None)
b["rev"] = "_v2"
b["stage_abandoned"] = {t: 1 for t in T}  # try 1 has its record (the all-excluded _v1): never re-attach; the next try is _try2
b["first_staging_v1"] = first | {"authoritative": False, "reason": f"re-stage from v2 manifests: donor rows missing on the family's box; key check {KC}"}
b["done"] = ["ended", "boxfiles", "relay"]
b["status"] = "relay"
s.setdefault("final_combined", {}).pop("il", None)
tmp = st.with_suffix(".tmp"); tmp.write_text(json.dumps(s, indent=1)); tmp.replace(st)
print(json.dumps({"backup": bak.name, BID: {k: b[k] for k in ("rev", "done", "status", "stage_tries", "stage_abandoned")}}))
