"""Build copies.py's index (0d's /tmp fix, 10-17) from the backups' sha lists, without opening any copy: for each of the ten folders,
<folder>_src_shas.txt (the /private/tmp originals) and <folder>_copy_shas.txt (the durable copies), both `shasum -a 256` output over
`./<file>`, must be byte-identical; every *.json entry becomes "<folder>/<file>": sha256. Non-JSON helpers (get.sh, jobs*.txt, fetch*.log)
are left out: no code reads them. Class K (names and shas). Writes exclusive-create and prints the index's sha256 for copies.INDEX_SHA.

  uv run --no-project --python 3.12 python -I make_copies_index.py --lists /Users/aswinsurya/probe-shift-durable-2026-10-17 \\
      --out config/fpr507_copies_index_2026-10-17.json
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
FOLDERS = ("simh", "ownfam", "ownall_man", "ownall_loo_man", "ownall_rs_man", "v2man", "nmman", "man455", "man710", "mandt")
VOLUME = "probe-shift_aswin-surya_sub-fpr507copies_kill-2026-11-14"
LINE = re.compile(r"([0-9a-f]{64})  \./([A-Za-z0-9._@+-]+)")


def build(lists: Path) -> dict:
    files, src = {}, {}
    for f in FOLDERS:
        a, b = (lists / f"{f}_src_shas.txt").read_bytes(), (lists / f"{f}_copy_shas.txt").read_bytes()
        if a != b:
            raise SystemExit(f"STOP: {f}: the source and copy sha lists differ")
        rows = [LINE.fullmatch(x) for x in a.decode().splitlines()]
        if not rows or not all(rows):
            raise SystemExit(f"STOP: {f}: a sha-list line is not '<64 hex>  ./<name>'")
        files |= {f"{f}/{m.group(2)}": m.group(1) for m in rows if m.group(2).endswith(".json")}
        src[f] = {"list_sha256": hashlib.sha256(a).hexdigest(), "lines": len(rows), "json": sum(m.group(2).endswith(".json") for m in rows)}
    return {"what": "fpr507 manifest copies index (copies.py; 0d's /tmp fix, 10-17): <folder>/<file> -> sha256, from the backups' sha lists",
            "volume": VOLUME, "layout": "fpr507/<folder>/<file> on the Volume; <folder>/<file> under $PROBE_SHIFT_COPIES",
            "folders": list(FOLDERS), "sha_lists": src, "n": len(files), "files": dict(sorted(files.items()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lists", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = HERE / a.out
    if out.exists():
        raise SystemExit(f"STOP: {a.out} exists")
    b = json.dumps(build(Path(a.lists)), indent=1).encode()
    with out.open("xb") as fh:
        fh.write(b)
    print(json.dumps({"written": a.out, "sha256": hashlib.sha256(b).hexdigest(), "n": json.loads(b)["n"]}))


if __name__ == "__main__":
    main()
