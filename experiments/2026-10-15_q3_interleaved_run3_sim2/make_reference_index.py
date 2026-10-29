"""Merge pinned honest_a references from several launches into one `--references-from` folder (RULE.md, step 2).

run_pilot.py's --references-from reads only `<folder>/launch.json["references"]` (key -> {copy, sha256, ...}), copies
each `copy` into the new run and re-checks its sha256. A box whose cells' references come from more than one launch
needs one merged index. This script writes it, failing loudly on: a wanted key in no source, a key in two sources, a
missing `copy`/`sha256`, or (with --check-files, on the box) a file whose sha256 differs from its record.

  make_reference_index.py --source <launch.json>[=<relay dir>] ... --cells <task file> ... --out <dir> [--check-files]
                          [--allow-missing <key>,...]

`=<relay dir>`: that source's reference files were relayed (hashes only) to <relay dir>; each `copy` becomes
<relay dir>/<file name>, and the original path is kept as `relayed_from`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def merge(
    sources: list[tuple[Path, Path | None]],
    wanted: list[str],
    allow_missing: set[str],
    check_files: bool,
) -> dict:
    found, origin = {}, {}
    for src, relay in sources:
        refs = json.loads(src.read_text())["references"]
        for key, ref in refs.items():
            if key not in wanted:
                continue
            assert key not in found, (
                f"{key}: in two sources ({origin[key]} and {src}); refusing to pick one"
            )
            assert ref.get("copy") and ref.get("sha256"), (
                f"{key} in {src}: no copy or sha256"
            )
            ref = dict(ref)
            if relay is not None:
                ref["relayed_from"], ref["copy"] = (
                    ref["copy"],
                    str(relay / Path(ref["copy"]).name),
                )
            if check_files:
                got = sha256(Path(ref["copy"]))
                assert got == ref["sha256"], (
                    f"{key}: {ref['copy']} sha256 {got} != recorded {ref['sha256']}"
                )
            found[key], origin[key] = ref | {"index_source": str(src)}, str(src)
    missing = sorted(set(wanted) - set(found))
    unexpected = sorted(set(missing) - allow_missing)
    assert not unexpected, (
        f"{len(unexpected)} wanted keys have no reference in any source: {unexpected}"
    )
    stale = sorted(allow_missing - set(missing))
    assert not stale, (
        f"--allow-missing keys that do have a reference (stale list): {stale}"
    )
    return {
        "references": found,
        "missing_allowed": missing,
        "sources": [
            {
                "launch_json": str(s),
                "sha256": sha256(s),
                "relay_dir": str(r) if r else None,
            }
            for s, r in sources
        ],
        "files_checked": check_files,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--source", action="append", required=True, help="<launch.json>[=<relay dir>]"
    )
    ap.add_argument(
        "--cells",
        action="append",
        required=True,
        type=Path,
        help="task file(s): wanted keys = their cells",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="new folder; its launch.json is the --references-from index",
    )
    ap.add_argument(
        "--allow-missing",
        default="",
        help="comma-separated keys expected to have no reference",
    )
    ap.add_argument(
        "--check-files",
        action="store_true",
        help="re-hash every reference file (on the box)",
    )
    args = ap.parse_args()
    sources = [
        (Path(a), Path(b)) if b else (Path(a), None)
        for a, _, b in (s.partition("=") for s in args.source)
    ]
    wanted = [c["key"] for f in args.cells for c in json.loads(f.read_text())["cells"]]
    assert len(wanted) == len(set(wanted)), "a cell key appears twice across --cells"
    index = merge(
        sources,
        wanted,
        {k for k in args.allow_missing.split(",") if k},
        args.check_files,
    )
    args.out.mkdir(
        parents=True, exist_ok=False
    )  # never overwrite an index a launch may have read
    (args.out / "launch.json").write_text(
        json.dumps(index, indent=1, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "out": str(args.out / "launch.json"),
                "references": len(index["references"]),
                "missing_allowed": index["missing_allowed"],
                "files_checked": args.check_files,
            }
        )
    )


if __name__ == "__main__":
    main()
