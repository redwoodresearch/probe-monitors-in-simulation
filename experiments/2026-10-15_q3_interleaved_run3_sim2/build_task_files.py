"""Build one task file per interleaved launch (RULE.md, "Task files"): a source follow-up task file's cells, seeded shuffle.

Each launch's cells are its source's cells (the main's r1 entries, as the follow-up used), copied verbatim: no field
added or changed, no `real_arms` (so each cell gets the launch's real attack and its simulated attack), the source's
file-level `construction` kept. The order is random.Random(int(sha256("<seed>:<name>"), 16)).shuffle: the launcher starts
tasks in file order, so this is the launch order, and it spreads environments against the 90 s per-environment stagger.
Sources are `<git rev>:<path>` (read with git show; the commit is recorded) or a plain path.

  build_task_files.py --seed <text> --out-dir <dir> <name>=<source> ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from collections import Counter
from pathlib import Path


def read(src: str) -> tuple[dict, dict]:
    if ":" in src and not Path(src).exists():
        rev, path = src.split(":", 1)
        commit = subprocess.check_output(
            ["git", "rev-parse", f"{rev}^{{commit}}"], text=True
        ).strip()
        raw = subprocess.check_output(["git", "show", f"{commit}:{path}"])
        prov = {"source": src, "commit": commit, "path": path}
    else:
        raw, prov = Path(src).read_bytes(), {"source": src}
    return json.loads(raw), prov | {"sha256": hashlib.sha256(raw).hexdigest()}


def shuffled(seed: str, name: str, cells: list[dict]) -> list[dict]:
    keys = [c["key"] for c in cells]
    assert len(keys) == len(set(keys)), f"{name}: duplicate cell keys"
    assert not any("real_arms" in c or c["key"].endswith("_r2") for c in cells), (
        f"{name}: an r2 entry; base cells only"
    )
    out = list(cells)
    random.Random(
        int(hashlib.sha256(f"{seed}:{name}".encode()).hexdigest(), 16)
    ).shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seed", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "sources", nargs="+", help="<name>=<source task file>, e.g. q1=<rev>:<path>"
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "tasks_il_manifest.json"
    assert not manifest_path.exists(), (
        f"{manifest_path} exists; task files are never overwritten"
    )
    manifest, seen = (
        {
            "seed": args.seed,
            "rule": __doc__.split("\n\n")[1].replace("\n", " "),
            "launches": {},
        },
        set(),
    )
    for spec in args.sources:
        name, _, src = spec.partition("=")
        doc, prov = read(src)
        cells = shuffled(args.seed, name, doc["cells"])
        assert not seen & {c["key"] for c in cells}, (
            f"{name}: a key already in another launch"
        )
        seen |= {c["key"] for c in cells}
        out = {
            "rule": "experiments/2026-10-15_q3_interleaved_run3_sim2/RULE.md",
            "seed": args.seed,
            "launch": name,
            **({"construction": doc["construction"]} if "construction" in doc else {}),
            "cells": cells,
        }
        path = args.out_dir / f"tasks_il_{name}.json"
        assert not path.exists(), f"{path} exists; task files are never overwritten"
        path.write_text(json.dumps(out, indent=1) + "\n")
        manifest["launches"][name] = {
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "cells": len(cells),
            "construction": doc.get("construction", "legacy_sol_hash"),
            "source": prov,
            "by_env": dict(sorted(Counter(c["env"] for c in cells).items())),
            "keys_in_order": [c["key"] for c in cells],
        }
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    print(
        json.dumps(
            {
                k: {x: v[x] for x in ("cells", "construction", "sha256")}
                for k, v in manifest["launches"].items()
            }
        )
    )


if __name__ == "__main__":
    main()
