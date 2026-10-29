"""Routing revision (sample_error_v1) for a routed set (09 via d9 10-16 ~00:35): the chain's routing file, unchanged, except that `manifest` points at
the append-only manifest revision carrying sample_error (config/<manifest stem>.sample_error_v1.json @ its commit), with the admission assert's
result: every admitted row (the target's own stage rows not staging-excluded) has an explicit sample_error == null (counts). The revision's
bytes are read at its pinned commit and checked against the pointer's sha256; it must extend exactly this routing file's plain manifest (path,
commit, sha256). Writes results/hygiene/routing_revisions/<batch>__<target>.sample_error_v1.json (exclusive, the committed copy) and, with --inplace,
replaces the routing-dir file (the old one kept as <batch>__<target>.pre_sample_error.json; d9 10-16 ~00:40). Keys and counts only.

  python make_routing_revision.py [--inplace] <routing dir> <batch>__<target> ...
"""
import ast
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def lit(x):
    return x if isinstance(x, (dict, list)) else ast.literal_eval(x)


def main():
    import hashlib
    import subprocess

    args = sys.argv[1:]
    inplace = args[0] == "--inplace"
    rdir, keys = Path(args[1 if inplace else 0]), args[2 if inplace else 1:]
    repo = HERE.parents[1]
    show = lambda ref: subprocess.run(["git", "show", ref], capture_output=True, check=True, cwd=repo).stdout  # noqa: E731
    ct = json.loads((HERE / "config" / "chain_targets.json").read_text())
    (HERE / "results" / "hygiene" / "routing_revisions").mkdir(exist_ok=True)
    for k in keys:
        src = rdir / f"{k}.pre_sample_error.json" if (rdir / f"{k}.pre_sample_error.json").exists() else rdir / f"{k}.json"
        r = json.loads(src.read_text())
        assert "routing_revision" not in r, f"{k}: {src.name} is already a revision"
        ptr = json.loads((rdir / "sample_error" / f"{k}.json").read_text())
        rev_path, rev_commit = ptr["manifest_revision"].split(" @ ")
        rev_raw = show(f"{rev_commit}:{rev_path}")  # the pinned bytes, never the working tree
        assert hashlib.sha256(rev_raw).hexdigest() == ptr["sha256"], f"{k}: revision bytes at {rev_commit} differ from the pointer's sha256"
        rev = json.loads(rev_raw)
        plain_path, plain_commit = [x.strip() for x in r["manifest"].split("@")]
        ext_path, ext_commit = rev["extends"]["manifest"].split(" @ ")
        plain_raw = show(f"{plain_commit}:{plain_path}")
        assert ext_path == plain_path and show(f"{ext_commit}:{ext_path}") == plain_raw and rev["extends"]["sha256"] == hashlib.sha256(plain_raw).hexdigest(), \
            f"{k}: the revision does not extend this routing file's plain manifest"
        tv = ct[r["target"]]
        own = set(tv["launch"] if isinstance(tv["launch"], list) else [tv["launch"]]) if tv.get("donor_launches") else None
        excl = {kk.split("#", 1)[0] for kk in lit(r["staging_exclusions"])}
        admitted = [f for f in rev["files"] if f["arm_dir"] in tv["stage_arms"] and f.get("attempt_kind") != "reference"
                    and (own is None or f["launch"] in own) and f["path"] not in excl]
        bad = [f"{f['arm_dir']}|{f['task']}" for f in admitted if "sample_error" not in f or f["sample_error"] is not None]  # explicit nulls only
        assert not bad, (k, bad)
        out = r | {"manifest": f"{rev_path} @ {rev_commit}", "manifest_sha256": ptr["sha256"], "manifest_plain": r["manifest"],
                   "routing_revision": "sample_error_v1 (09 via d9 10-16 ~00:35): manifest = the sample_error revision; the plain manifest is superseded for admission",
                   "sample_error_assert": {"admitted_rows": len(admitted), "admitted_null": len(admitted) - len(bad), "passes": True,
                                           "rule": "every admitted row has sample_error == null (B fork's class rule)"}}
        dst = HERE / "results" / "hygiene" / "routing_revisions" / f"{k}.sample_error_v1.json"
        dst.open("x").write(json.dumps(out, indent=1) + "\n")
        if inplace:  # the routing dir's own file becomes the revision; the old one is kept beside it
            if src.name == f"{k}.json":
                (rdir / f"{k}.json").rename(rdir / f"{k}.pre_sample_error.json")
            (rdir / f"{k}.json").write_text(json.dumps(out, indent=1))
        print(k, out["sample_error_assert"]["admitted_rows"], "admitted, all null")


if __name__ == "__main__":
    main()
