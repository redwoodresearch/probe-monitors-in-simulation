"""Corrected counted manifests for an interleaved family whose box read had no donor rows (09 via d9 10-15, option (a); il-q3c):
the committed originals' rows unchanged plus exactly the main's counted honest_a/honest_b rows for the family's cells, by key, from
a pinned main manifest (commit + file sha256) — the same rows chain.donor_supply (option b) would have supplied, each naming its
source. New files manifest_<target>_v2.json (open 'x', `supersedes` the original); each original stays, with a SUPERSEDED note.

  python3 make_il_v2_from_main.py <chain state dir> <batch id> <main manifest path> <commit> <file sha256> <key check record>
"""
import copy
import hashlib
import json
import sys
from pathlib import Path

import chain as C


class St:  # donor_supply's logging only
    def log(self, m):
        print(m)


def main():
    st_dir, bid, mpath, commit, msha, keycheck = sys.argv[1:7]
    b = copy.deepcopy(json.loads((Path(st_dir) / "state.json").read_text())["batches"][bid])
    b["donor_manifests"] = [{"path": mpath, "commit": commit, "sha256": msha}]
    reason = f"re-stage from v2 manifests: donor rows missing on {b['box'].split('@')[1]} (its copy of the main holds only launch.json and references/); key check {keycheck}; registered 09 @ 65ef3a9ff section 1"
    out = {}
    for t, tv in C.targets_for(b | {"rev": ""}).items():
        src = C.HERE / "config" / tv["manifest"]
        orig = json.loads(src.read_text())
        man = copy.deepcopy(orig)
        C.donor_supply(b, t, man, St())  # raises on a mixed state or a pinned-sha mismatch
        assert all(f in man["files"] for f in orig["files"])  # every original row unchanged
        doc = {"note": orig["note"] + " (v2: honest donors from the main's counted rows, which this box read could not see)", "counts": man["counts"],
               "counted_attempts_without_eval": orig["counted_attempts_without_eval"], "files": man["files"], "donors_supplied": man["donors_supplied"],
               "supersedes": src.name, "supersedes_reason": reason}
        dst = src.with_name(src.stem + "_v2.json")
        with dst.open("x") as fh:
            fh.write(json.dumps(doc, indent=1) + "\n")
        sha = hashlib.sha256(dst.read_bytes()).hexdigest()
        with src.with_name(src.stem + ".SUPERSEDED.md").open("x") as fh:
            fh.write(f"# SUPERSEDED\n\n`{src.name}` is superseded by `{dst.name}` (sha256 {sha}): {reason}. The v2 file keeps every row of this one unchanged "
                     f"and adds {man['donors_supplied']['rows']} donor rows ({mpath} @ {commit}). This file stays unchanged; its all-excluded first staging is a "
                     "non-authoritative audit artifact, never routed or scored.\n")
        out[dst.name] = {"sha256": sha, "counts": doc["counts"], "donors": man["donors_supplied"]["rows"], "cells_without_a_donor_row": man["donors_supplied"]["cells_without_a_donor_row"]}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
