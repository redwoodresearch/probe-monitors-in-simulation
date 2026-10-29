"""The fpr507 manifest copies (0d's /tmp fix, 10-17): ONE root and ONE committed index for every local manifest copy the runner and its
input checks read (simh, ownfam, ownall_man, ownall_loo_man, ownall_rs_man, v2man, nmman, man455, man710, mandt). Before this, each module
read its own absolute /private/tmp/…/scratchpad/fpr507 path, and four loads skipped silently on a missing file.

- root(): the folder that holds <folder>/<file>. 0d's ruling B (10-17): set_root(<path>) from the runner's explicit `copies_root` flag
  (route C's step spec passes the records wrapper's staged prefix, /private/tmp/…/scratchpad/fpr507), else $PROBE_SHIFT_COPIES, but the
  env fallback is LAPTOP-ONLY: inside the records wrapper (its runner env carries RECSTEP_EXC, set for every runner it starts:
  run_records_step_in_container.py @ a54926d3e L360-362, @ c51fae10b L466-468) a missing flag STOPs. No default: an unset or non-directory
  root STOPs at the first READ (not at import, so tools and tests that import the runner still load). On the laptop the env is the durable
  copy (/Users/aswinsurya/probe-shift-durable-2026-10-17/fpr507).
- read(rel): the bytes of root()/rel, read ONCE, hashed and returned (the same bytes); STOP if the file is missing, if rel is not in the
  index, or if its sha256 is not the index's.
- names(folder, pattern): the index's entries in a folder matching a glob, sorted (a listing never depends on what is on disk).
- check(rel): None, or the problem string (for the verify functions that collect problems instead of stopping).
The index (INDEX_PATH, sha256 INDEX_SHA, class K: names and shas only) is built by make_copies_index.py from the backups' sha lists.
"""

import fnmatch
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
INDEX_PATH = "config/fpr507_copies_index_2026-10-17.json"
INDEX_SHA = "047e633d073c2fd5f49e53a035d16cbf0c5848030186dad3226fd60e196917e1"  # make_copies_index.py output, 10-17
ENV = "PROBE_SHIFT_COPIES"
WRAPPER_MARK = "RECSTEP_EXC"  # the records wrapper sets it in every runner's environment (see the docstring)
_IDX, _ROOT = {}, {}


def set_root(path: str) -> None:
    if not path or not Path(path).is_dir():
        raise SystemExit(f"copies: copies_root {path!r} is not a directory (STOP)")
    _ROOT["root"] = Path(path)


def index() -> dict:
    if "files" not in _IDX:
        b = (HERE / INDEX_PATH).read_bytes()
        if INDEX_SHA is None or hashlib.sha256(b).hexdigest() != INDEX_SHA:
            raise SystemExit(f"copies: {INDEX_PATH} is not the pinned index (sha256; STOP)")
        _IDX["files"] = json.loads(b)["files"]
    return _IDX["files"]


def root() -> Path:
    if "root" in _ROOT:
        return _ROOT["root"]
    if WRAPPER_MARK in os.environ:
        raise SystemExit("copies: inside the records wrapper the root comes only from the runner's copies_root flag (no env fallback; STOP)")
    v = os.environ.get(ENV)
    if not v or not Path(v).is_dir():
        raise SystemExit(f"copies: ${ENV} is unset or not a directory (the fpr507 manifest copies' root; STOP)")
    return Path(v)


def _load(rel: str):
    """-> (bytes, None) or (None, problem). The file is read once; the bytes returned are the bytes hashed (O1)."""
    if rel not in index():
        return None, f"copies: {rel} is not in the index"
    p = root() / rel
    if not p.is_file():
        return None, f"copies: {rel} is missing under the copies root"
    b = p.read_bytes()
    if hashlib.sha256(b).hexdigest() != index()[rel]:
        return None, f"copies: {rel} sha256 differs from the index"
    return b, None


def check(rel: str):
    return _load(rel)[1]


def read(rel: str) -> bytes:
    b, prob = _load(rel)
    if prob:
        raise SystemExit(prob + " (STOP)")
    return b


def exists(rel: str) -> bool:
    return rel in index()


def names(folder: str, pattern: str) -> list:
    return sorted(k for k in index() if k.startswith(folder + "/") and fnmatch.fnmatchcase(k[len(folder) + 1:], pattern))
