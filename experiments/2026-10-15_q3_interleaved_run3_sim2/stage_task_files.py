"""Split each box's interleaved cells into ordered stages (RULE.md, "Stages"): pilot, rest of estimation, selection.

Input: the per-launch task files from build_task_files.py (already seeded-shuffled), one per box (q1..q4), plus 09's key
lists. Stages, each its own launch on that box, run in this order by the on-box chain:
  pilot  - a stratified seeded sample of --pilot-n estimation cells over ALL given boxes: strata = environment x stratum
           (S_old / S_new), allocated in proportion to stratum size by largest remainder (ties: sha256(seed:stratum)),
           cells within a stratum taken in sha256(seed:key) order;
  est    - the other estimation cells: headline cells (--headline-keys) first, each block in the input file's order;
  sel    - the selection-half cells (--selection-keys), in the input file's order.
Without --estimation-keys every cell is estimation (sel is empty). Every input cell lands in exactly one stage; the
estimation and selection lists must be disjoint and cover every given cell. Writes tasks_il_<box>_<stage>.json and
tasks_il_stages_manifest.json.

  stage_task_files.py --seed <text> --pilot-n 150 --stratum <box>=S_new ... [--estimation-keys F --selection-keys F]
                      [--headline-keys F] --out-dir <dir> <box>=<tasks_il_<box>.json> ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

H = lambda s: hashlib.sha256(s.encode()).hexdigest()


def keys_file(path: str | None) -> set[str] | None:
    return None if path is None else set(json.loads(Path(path).read_text()))


def stage(
    seed: str,
    pilot_n: int,
    boxes: dict[str, list[dict]],
    stratum: dict[str, str],
    est: set[str] | None,
    sel: set[str] | None,
    headline: set[str],
) -> dict[str, dict[str, list[dict]]]:
    all_keys = [c["key"] for b in boxes.values() for c in b]
    if est is None:
        assert sel is None, "--selection-keys without --estimation-keys"
        est, sel = set(all_keys), set()
    assert not est & sel, "estimation and selection overlap"
    assert set(all_keys) <= est | sel, (
        f"cells in neither list: {sorted(set(all_keys) - est - sel)[:5]}"
    )
    assert headline <= est, "a headline key outside the estimation set"
    strata: dict[tuple[str, str], list[dict]] = {}
    for box, cells in boxes.items():
        for c in cells:
            if c["key"] in est:
                strata.setdefault((c["env"], stratum[box]), []).append(c)
    n_est = sum(len(v) for v in strata.values())
    assert 0 < pilot_n <= n_est, (pilot_n, n_est)
    quota = {s: pilot_n * len(v) / n_est for s, v in strata.items()}
    take = {s: int(q) for s, q in quota.items()}
    for s in sorted(quota, key=lambda s: (-(quota[s] - take[s]), H(f"{seed}:{s}")))[
        : pilot_n - sum(take.values())
    ]:
        take[s] += 1
    pilot = {
        c["key"]
        for s, v in strata.items()
        for c in sorted(v, key=lambda c: H(f"{seed}:{c['key']}"))[: take[s]]
    }
    out = {}
    for box, cells in boxes.items():
        rest = [c for c in cells if c["key"] in est and c["key"] not in pilot]
        out[box] = {
            "pilot": [c for c in cells if c["key"] in pilot],
            "est": [c for c in rest if c["key"] in headline]
            + [c for c in rest if c["key"] not in headline],
            "sel": [c for c in cells if c["key"] in sel],
        }
        assert sorted(c["key"] for s in out[box].values() for c in s) == sorted(
            c["key"] for c in cells
        ), box
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seed", required=True)
    ap.add_argument("--pilot-n", type=int, required=True)
    ap.add_argument(
        "--stratum", action="append", required=True, help="<box>=S_old|S_new"
    )
    ap.add_argument("--estimation-keys")
    ap.add_argument("--selection-keys")
    ap.add_argument("--headline-keys")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("inputs", nargs="+", help="<box>=<per-launch task file>")
    args = ap.parse_args()
    docs = {
        b: json.loads(Path(p).read_text())
        for b, _, p in (s.partition("=") for s in args.inputs)
    }
    stratum = dict(s.split("=", 1) for s in args.stratum)
    assert set(stratum) == set(docs), "one --stratum per input box"
    staged = stage(
        args.seed,
        args.pilot_n,
        {b: d["cells"] for b, d in docs.items()},
        stratum,
        keys_file(args.estimation_keys),
        keys_file(args.selection_keys),
        keys_file(args.headline_keys) or set(),
    )
    manifest = {
        "seed": args.seed,
        "pilot_n": args.pilot_n,
        "lists": {
            k: getattr(args, k)
            for k in ("estimation_keys", "selection_keys", "headline_keys")
        },
        "stages": {},
    }
    for box, stages in staged.items():
        for name, cells in stages.items():
            path = args.out_dir / f"tasks_il_{box}_{name}.json"
            assert not path.exists(), f"{path} exists"
            doc = {k: v for k, v in docs[box].items() if k != "cells"} | {
                "stage": name,
                "cells": cells,
            }
            path.write_text(json.dumps(doc, indent=1) + "\n")
            manifest["stages"][f"{box}_{name}"] = {
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "cells": len(cells),
                "by_env": dict(Counter(c["env"] for c in cells)),
            }
    (args.out_dir / "tasks_il_stages_manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n"
    )
    print(
        json.dumps(
            {k: (v["cells"], v["sha256"][:12]) for k, v in manifest["stages"].items()}
        )
    )


if __name__ == "__main__":
    main()
