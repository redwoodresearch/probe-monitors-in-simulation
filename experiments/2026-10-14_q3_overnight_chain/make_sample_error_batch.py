"""sample_error revisions for already-attested sets, in one scan (09 via d9 10-16 ~00:10). Step "spec": from the named routing files (chain routing
dir), the union of their committed manifests' (path, sha256) rows -> one spec for sample_errors_chain.py. Step "revise": per routing file, the scan's
rows for exactly that manifest -> make_sample_error_revision.py (target, its staging record from the chain state). Keys and classes only.

  python make_sample_error_batch.py spec <routing dir> <chain state dir> <out spec.json> <batch>__<target> ...
  python make_sample_error_batch.py revise <routing dir> <chain state dir> <scan.json> <batch>__<target> ...
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def manifest_of(rdir: Path, key: str):
    r = json.loads((rdir / f"{key}.json").read_text())
    path, commit = [x.strip() for x in r["manifest"].split("@")]
    man = json.loads(subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, check=True, cwd=REPO).stdout)
    return r, f"{commit}:{path}", man


def main():
    mode, rdir, st_dir, io_p, keys = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5:]
    if mode == "spec":
        rows = sorted({(f["path"], f["sha256"]) for k in keys for f in manifest_of(rdir, k)[2]["files"]})
        Path(io_p).open("x").write(json.dumps([{"path": p, "sha256": h} for p, h in rows]))
        print(json.dumps({"sets": len(keys), "rows": len(rows)}))
        return
    scan = json.loads(Path(io_p).read_text())
    st = json.loads((st_dir / "state.json").read_text())["batches"]
    for k in keys:
        r, ref, man = manifest_of(rdir, k)
        mine = {f["path"] for f in man["files"]}
        part = Path(io_p).with_name(f"{Path(io_p).stem}__{k}.json")
        part.open("x").write(json.dumps({"rule": scan["rule"], "rows": {p: v for p, v in scan["rows"].items() if p in mine}}))
        rec = HERE / "results" / st[r["batch"]]["stage"][r["target"]]["record"]
        subprocess.run([sys.executable, str(HERE / "make_sample_error_revision.py"), ref, str(rec), str(part), r["target"]], check=True, cwd=HERE)


if __name__ == "__main__":
    main()
