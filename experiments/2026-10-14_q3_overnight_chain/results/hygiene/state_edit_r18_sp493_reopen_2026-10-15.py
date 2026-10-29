# Restart-18 state edit, part 2 (d9 GO 10-15 ~23:50; 09's naming ruling): re-open the S_paper493 batches STOPPED for the donor bug so the fixed
# commit_targets (index donors; donor_launches never staged) rebuilds their manifests and stages them under the NEXT FREE write-once name of each
# target (the try counters are kept; each earlier try is marked abandoned, so nothing re-attaches to a SUPERSEDED set). b_run2 is HELD (09: t615).
# g1/g2 snapshots are re-opened likewise; the g1/g2 families stay final_only (no new _s<j>), re-opened to wait for their launches. The old 0-staged
# routing files of g1_s1 / g1_s3 move aside in the routing dir (renamed *.superseded_try1.json). The sp493 pools' combined finals are recomputed.
import json, os, sys, time
st_dir = sys.argv[1]
p = os.path.join(st_dir, "state.json")
d = json.load(open(p))
B = d["batches"]
STEPS = ["ended", "boxfiles", "relay", "commit_targets", "stage", "format", "repro", "extract", "compare", "labels", "record", "ready", "hygiene",
         "provenance", "attest"]
DROP = ("targets", "commit_targets", "stage", "format", "repro", "extract", "extract_job", "extract_attempt", "shards", "stats", "cfg_committed",
        "compare", "labels", "labels_sim2", "record_commit", "hygiene_commit", "provenance", "provenance_commit", "attest_commit", "stage_unknown",
        "stage_donor_launches", "retries")  # every field a step from commit_targets on writes (Codex 10-15 ~23:55: cfg_committed)
snaps = sorted(k for k, b in B.items() if k.startswith(("sp493-g1_s", "sp493-g2_s")) and b["status"] != "EMPTY")
reopen = ["sp493-a", "sp493-b", "sp493-c", *snaps]
names = {}
for bid in reopen:
    b = B[bid]
    assert b["status"] in ("STOPPED",) or "attest" in b["done"], (bid, b["status"])
    assert "boxfiles" in b["done"] and "relay" in b["done"], (bid, b["done"])
    tries = b.get("stage_tries") or {}
    for t, n in tries.items():
        b.setdefault("stage_abandoned", {})[t] = n  # never re-attach to an earlier (SUPERSEDED) try: the next try is n + 1
        names[t] = f"next try {n + 1}"
    b["reopened"] = {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "from_status": b["status"], "stopped_at": b.get("stopped_at"),
                     "reason": "d9 GO ~23:50: re-stage with the donor fix under the next free write-once names (09)"}
    for k in DROP:
        b.pop(k, None)
    for k in ("stopped_at", "reason"):
        b.pop(k, None)
    b["done"] = ["ended", "boxfiles", "relay"]
    b["status"] = "waiting"
    if bid == "sp493-b":
        b["hold_targets"] = ["sp493_b_run2"]  # 09 rules on t615_media_processing_r2 (labeler rule gap) first
for fid in ("sp493-g1", "sp493-g2"):
    f = B[fid]
    assert f["status"] == "STOPPED" and f.get("final_only") is True and set(f["done"]) <= {"ended", "boxfiles"} and not f.get("commit_targets"), (fid, f["status"], f["done"])
    for k in ("stopped_at", "reason"):
        f.pop(k, None)
    f["status"] = "waiting"  # its final remainder (cells no snapshot took) when the launch ends; no new snapshots (final_only)
R = os.path.join(st_dir, "routing")
for bid, t in (("sp493-g1_s1", "sp493_g1_sim_s1"), ("sp493-g1_s3", "sp493_g1_sim_s3")):
    for suf in (f"__{t}", "_attest", "_hygiene", "_provenance"):
        src = os.path.join(R, f"{bid}{suf}.json")
        if os.path.exists(src):
            os.rename(src, os.path.join(R, f"{bid}{suf}.superseded_try1.json"))
for pool in ("sp493", "sp493_g1_sim", "sp493_g2_sim"):
    d.setdefault("final_combined", {}).pop(pool, None)
open(p + ".tmp", "w").write(json.dumps(d, indent=1)); os.replace(p + ".tmp", p)
print("re-opened", reopen, "| held sp493_b_run2 | next names:", names, time.strftime("%H:%M:%S"))
