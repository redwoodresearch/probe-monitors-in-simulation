"""The 217 repair-rule cells (PLAN.md here): swap100 @ 907f5467 plus ext117, from 09's audit file, sha256 asserted."""
import hashlib
import json
from pathlib import Path

SWAP_SHA = "1e4f5b3b296beacd9a239d10d95ee4f946c84751fbbd0786bd9ba825f59af0a2"
SUBSETS = ("all", "out217", "in217")


def cells217(swap_path):
    b = Path(swap_path).read_bytes()
    if hashlib.sha256(b).hexdigest() != SWAP_SHA:
        raise SystemExit("swap_sets.json differs from its pinned sha256")
    s = json.loads(b)
    a, e = set(s["swap100@907f5467"]["cells"]), set(s["ext117 (S3 sha256 24f6c132)"]["cells"])
    if len(a) != 100 or len(e) != 117 or a & e:
        raise SystemExit("not 100 + 117 disjoint cells")
    return a | e


def keeper(subset, swap_path):
    if subset not in SUBSETS:
        raise SystemExit(f"subset must be one of {SUBSETS}")
    if subset == "all":
        return lambda c: True
    c217 = cells217(swap_path)
    return (lambda c: c not in c217) if subset == "out217" else (lambda c: c in c217)
