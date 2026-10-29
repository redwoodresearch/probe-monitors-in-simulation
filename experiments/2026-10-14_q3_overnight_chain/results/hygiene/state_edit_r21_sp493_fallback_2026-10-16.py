# Restart-21 state edit (09 via d9 10-16 ~01:05 / ~01:40): re-open sp493-b and sp493-c at commit_targets so the index-donor FALLBACK (a cell whose
# launch donors all carry a sample error gets its pinned index copy) rebuilds their manifests and re-stages b_main (t619), c_main (t699) and c_run2
# (t699_r2) under the NEXT FREE write-once name (try counters kept; each earlier try marked abandoned, never re-attached). b_run2's hold is
# released (d9 ~01:45): it re-stages at try 3 with the scoped t615 exclusion and the fallback for t619_r2.
# sp493-a is untouched (no fallback cell). The routed routing files of b/c are COPIED (not moved: they stay valid until the re-stage's record step
# would rewrite them; route_pending sends the re-stage's routing files to routing/pending/ until the diff gate passes) to
# routing/superseded/<name>.superseded_try<n>.json, exclusive; the sp493 pool's combined finals are recomputed.
import json, os, shutil, sys, time
st_dir = sys.argv[1]
p = os.path.join(st_dir, "state.json")
d = json.load(open(p))
B = d["batches"]
DROP = ("targets", "commit_targets", "stage", "format", "repro", "extract", "extract_job", "extract_attempt", "shards", "stats", "cfg_committed",
        "compare", "labels", "labels_sim2", "record_commit", "hygiene_commit", "provenance", "provenance_commit", "attest_commit", "stage_unknown",
        "stage_donor_launches", "retries")  # every field a step from commit_targets on writes (as restart 18)
want = {"sp493-b": {"sp493_b_main": 4, "sp493_b_run2": 2}, "sp493-c": {"sp493_c_main": 3, "sp493_c_run2": 3}}
R = os.path.join(st_dir, "routing")
copies = []
for bid, tries in want.items():
    b = B[bid]
    assert "attest" in b["done"] and "boxfiles" in b["done"] and "relay" in b["done"], (bid, b["done"])
    assert b.get("stage_tries") == tries, (bid, b.get("stage_tries"))
    for t, n in tries.items():
        b.setdefault("stage_abandoned", {})[t] = n  # the next try is n + 1
    for f in sorted(os.listdir(R)):
        if f.startswith((f"{bid}__", f"{bid}_attest", f"{bid}_hygiene", f"{bid}_provenance")) and f.endswith(".json") and ".superseded" not in f and ".pre_" not in f:
            t = f.split("__", 1)[1][:-5] if "__" in f else None
            n = tries[t] if t else max(tries[x] for x in tries if x in b.get("targets", []))
            dst = os.path.join(R, "superseded", f"{f[:-5]}.superseded_try{n}.json")
            assert not os.path.exists(dst), dst
            shutil.copy2(os.path.join(R, f), dst)
            copies.append(os.path.basename(dst))
    b["reopened"] = {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "from_status": b["status"],
                     "reason": "09 via d9 ~01:05/~01:40: re-stage with the index-donor fallback under the next free write-once names"}
    for k in DROP:
        b.pop(k, None)
    for k in ("stopped_at", "reason"):
        b.pop(k, None)
    b["done"] = ["ended", "boxfiles", "relay"]
    b["status"] = "waiting"
    b["route_pending"] = True  # routing files go to routing/pending/ until the by-key byte diff passes (09); the live routed files stay
assert B["sp493-b"].get("hold_targets") == ["sp493_b_run2"], B["sp493-b"].get("hold_targets")
B["sp493-b"].pop("hold_targets")  # d9 10-16 ~01:45 (Aswin wants every box-B cell): b_run2 re-stages too, with the scoped t615 exclusion (config) and the fallback (t619_r2)
d.setdefault("final_combined", {}).pop("sp493", None)
open(p + ".tmp", "w").write(json.dumps(d, indent=1)); os.replace(p + ".tmp", p)
print("re-opened sp493-b (b_main -> try 5; b_run2 -> try 3, hold released) and sp493-c (c_main, c_run2 -> try 4); copied", copies, time.strftime("%H:%M:%S"))
