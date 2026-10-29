"""Append-only manifest revision carrying sample_error (09's admission rule via d9 10-16 ~00:05) for a set staged before the chain wrote the field
(g4, g5): every row of the committed manifest, unchanged, plus sample_error (null | class) from sample_errors_chain.py's scan of the Volume copies
(the B fork's class rule). Asserts by key that every STAGED row (the target's own stage rows minus its staging record's exclusions; donor rows are reported, not staged) has sample_error == null
(else it stops: such a row needs 09's ruling). Writes config/<manifest stem>.sample_error_v1.json (exclusive) naming the manifest it extends.

  python make_sample_error_revision.py <manifest "commit:path"> <staging record path> <scan json (sample_errors_chain.py --out)> <target>
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def main():
    ref, rec_p, scan_p, target = sys.argv[1:5]
    tv = json.loads((HERE / "config" / "chain_targets.json").read_text())[target]
    commit, path = ref.split(":", 1)
    raw = subprocess.run(["git", "show", ref], capture_output=True, check=True, cwd=REPO).stdout
    man = json.loads(raw)
    scan = json.loads(Path(scan_p).read_text())["rows"]
    assert {(f["path"], f["sha256"]) for f in man["files"]} == {(p, v["sha256"]) for p, v in scan.items()}, "the scan does not match the manifest's (path, sha256) rows"
    errs = {p: v["sample_error"] for p, v in scan.items()}
    rec_all = json.loads(Path(rec_p).read_text())
    rec = rec_all[next(k for k in rec_all if k not in ("git", "stamp"))]
    assert Path(rec_p).name.startswith(f"chain_stage_{target}_20") and rec["out"].startswith(tv["out"]), "the staging record is not this target's"
    at_stage = subprocess.run(["git", "show", f"{rec_all['git']}:{path}"], capture_output=True, check=True, cwd=REPO).stdout
    assert at_stage == raw, "the manifest at the staging record's commit differs from this manifest (staged from another version)"
    excl = {k.split("#", 1)[0]: v for k, v in rec["c_gate"]["excluded"].items()}
    paths = {f["path"] for f in man["files"]}
    assert set(excl) <= paths and rec["c_gate"]["staged"] + len(excl) == rec["c_gate"]["manifest_keys"], "the staging record does not belong to this manifest"
    own = set(tv["launch"] if isinstance(tv["launch"], list) else [tv["launch"]]) if tv.get("donor_launches") else None
    keys = [f for f in man["files"] if f["arm_dir"] in tv["stage_arms"] and f.get("attempt_kind") != "reference" and (own is None or f["launch"] in own)]
    assert path.endswith(tv["manifest"]) and rec["c_gate"]["manifest_keys"] == len(keys), "the staging record / target does not belong to this manifest"
    rows = [f | {"sample_error": errs[f["path"]]} for f in man["files"]]
    stage_keys = {f["path"] for f in keys}  # the target's own stage rows (its c_gate keys); the rest are donor rows, read but never staged
    staged_bad = [f"{f['arm_dir']}|{f['task']} ({f['sample_error']})" for f in rows if f["path"] in stage_keys and f["path"] not in excl and f["sample_error"] is not None]
    if staged_bad:
        raise SystemExit(f"staged rows with a sample error (09 must rule): {staged_bad}")
    donor_err = [{"key": f"{f['arm_dir']}|{f['task']}", "sample_error": f["sample_error"], "path": f["path"]} for f in rows
                 if f["path"] not in stage_keys and f["sample_error"] is not None]  # reported for 09: a donor's own run carries a sample error
    out = {k: v for k, v in man.items() if k != "files"} | {
        "files": rows, "revision": "sample_error_v1",
        "extends": {"manifest": f"{path} @ {commit}", "sha256": hashlib.sha256(raw).hexdigest(), "rows_unchanged": True},
        "sample_error_rule": "09 via d9 10-16 ~00:05: sample_error from the .eval's sample metadata (the B fork's class rule, run_hsim_eval_scan.py @ d1c147767); "
                             "a non-null class refuses the row; every staged row here is null",
        "staging_record": Path(rec_p).name,
        "excluded_rows": [{"key": f"{f['arm_dir']}|{f['task']}", "staging_reason": excl[f["path"]], "sample_error": f["sample_error"]} for f in rows if f["path"] in excl],
        "donor_rows_with_sample_error": donor_err}
    dst = HERE / "config" / (Path(path).stem + ".sample_error_v1.json")
    dst.open("x").write(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"revision": dst.name, "rows": len(rows), "sample_errors": sum(f["sample_error"] is not None for f in rows),
                      "staged_with_error": 0, "excluded": out["excluded_rows"], "donor_rows_with_sample_error": donor_err}))


if __name__ == "__main__":
    main()
