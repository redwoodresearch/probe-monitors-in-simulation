"""Extraction-writer audit (2e's finding on ilnm_s5_sim2, 09 via d9 10-15 ~07:40): for every extraction root on the activations
Volume, does every chunk the root holds come from a RECORDED shard that ran on one H200? A chunk is covered when a committed
extraction record (results/extract_chain_<root>_a<n>_*.json) has a shard without error, gpu_name "NVIDIA H200", gpu_count 1, that
lists it. Also reported per root: shards that failed with "already has a writer", skip-only shards (done_before == epochs, no
chunks), and the attempt complete.json names. A root with a chunk no such shard covers, or with no record, is FLAGGED (held until
redone). Lists file names and reads the small complete.json markers only; never reads activations.

  python audit_extract_writers.py <out.json>
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
VOL = "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14"
H200 = "NVIDIA H200"
REPO = HERE.parents[1]
OTHER_BRANCHES = ["origin/aswin/q3-ha25f"]


def main():
    vol = modal.Volume.from_name(VOL, version=2)
    roots = sorted(e.path.strip("/") for e in vol.listdir("/") if "_extract_v" in e.path)
    recs = {}
    srcs = [(p.name, p.read_text, "this branch") for p in sorted((HERE / "results").glob("extract_chain_*_a*_20*.json"))]
    for br in OTHER_BRANCHES:  # roots extracted from another branch's chain worktree (ha25f): their records live there
        names = subprocess.run(["git", "ls-tree", "-r", "--name-only", "--full-name", br, "--", "results/"], capture_output=True, text=True, check=True, cwd=HERE).stdout.split()
        for n in names:
            b = n.split("/")[-1]
            if b.startswith("extract_chain_") and b not in {x[0] for x in srcs}:
                srcs.append((b, (lambda n=n: subprocess.run(["git", "show", f"{br}:{n}"], capture_output=True, text=True, check=True, cwd=REPO).stdout), br))
    for name, read, where in srcs:
        m = re.match(r"extract_chain_(.+)_a(\d+)_(20\d\d-\d\d-\d\d_\d{6})\.json$", name)
        if m:
            recs.setdefault(m.group(1), []).append((int(m.group(2)), name, read, where))
    rows = []
    for root in roots:
        try:
            vol_chunks = sorted(re.sub(r"\.json$", "", e.path.split("/")[-1]) for e in vol.listdir(f"{root}/chunks"))
        except Exception as e:  # noqa: BLE001 (a root without a chunks folder is reported, not skipped)
            vol_chunks, err = [], type(e).__name__
        else:
            err = None
        try:
            complete = json.loads(b"".join(vol.read_file(f"{root}/complete.json")))
        except Exception:  # noqa: BLE001
            complete = None
        covered, writer_errors, skip_only, other_gpu, files = {}, [], [], [], []
        for att, name, read, where in sorted(recs.get(root, []), key=lambda x: (x[0], x[1])):
            d = json.loads(read())
            files.append(name if where == "this branch" else f"{name} ({where})")
            for s in d["shards"]:
                if "error" in s:
                    if "already has a writer" in s["error"]:
                        writer_errors.append(f"a{att}")
                    continue
                if s.get("gpu_name") != H200 or s.get("gpu_count") != 1:
                    other_gpu.append(f"a{att}/shard{s.get('shard')}")
                    continue
                if not s.get("chunks") and s.get("done_before") == s.get("epochs"):
                    skip_only.append(f"a{att}/shard{s['shard']}")
                for c in s.get("chunks") or []:
                    covered[c["chunk"]] = name
        unrecorded = [c for c in vol_chunks if c not in covered]
        missing = sorted(set(covered) - set(vol_chunks))
        chunk_atts = sorted({re.search(r"_a(\d+)_", c).group(1) for c in vol_chunks if re.search(r"_a(\d+)_", c)})
        flag = bool(unrecorded) or not files or bool(other_gpu)
        test = bool(re.search(r"TEST|orchtest|_test$|killtest", root))  # chain tests (10-14), never routed or scored
        rows.append({"root": root, "records": files, "volume_chunks": len(vol_chunks), "chunks_listing_error": err, "chunk_attempts": chunk_atts,
                     "covered_by_recorded_h200_shards": len(vol_chunks) - len(unrecorded), "unrecorded_chunks": unrecorded,
                     "recorded_chunks_missing_on_volume": missing, "already_has_a_writer": sorted(set(writer_errors)),
                     "skip_only_shards": len(skip_only), "non_h200_shards": other_gpu,
                     "complete_json_attempt": (complete or {}).get("attempt"), "flagged": flag, "test_root": test,
                     "why": [w for w, c in (("unrecorded chunks", unrecorded), ("no committed record", not files), ("non-H200 shard", other_gpu)) if c]})
    doc = {"rule": "09 via d9 10-15 ~07:40: every chunk of a root must come from a recorded H200 shard (the writer's own record); flagged roots are held",
           "volume": VOL, "records_dir": "experiments/2026-10-14_q3_overnight_chain/results",
           "counts": {"roots_checked": len(rows), "full_gpu_record": sum(not r["flagged"] for r in rows), "flagged": sum(r["flagged"] and not r["test_root"] for r in rows),
                      "flagged_test_roots": sum(r["flagged"] and r["test_root"] for r in rows),
                      "with_already_has_a_writer": sum(bool(r["already_has_a_writer"]) for r in rows), "with_skip_only_shards": sum(r["skip_only_shards"] > 0 for r in rows)},
           "flagged": [r["root"] for r in rows if r["flagged"] and not r["test_root"]], "flagged_test_roots": [r["root"] for r in rows if r["flagged"] and r["test_root"]],
           "roots": rows}
    Path(sys.argv[1]).open("x").write(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"counts": doc["counts"], "flagged": [(r["root"], r["why"], len(r["unrecorded_chunks"]), r["already_has_a_writer"]) for r in rows if r["flagged"] and not r["test_root"]],
                      "flagged_test_roots": doc["flagged_test_roots"]}))


if __name__ == "__main__":
    main()
