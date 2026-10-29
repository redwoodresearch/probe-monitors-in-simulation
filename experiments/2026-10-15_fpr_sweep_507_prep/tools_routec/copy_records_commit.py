"""Route C COVB records: one copy commit on the branch COVB runs on (09 accepts option (b), via 2e, 10-18). The builder hashes every
covered record at HEAD, and some cited originals live on commits that are not ancestors of B. This script copies them byte for byte to
the SAME paths, under 09's conditions:
1. each copy is written from `git show <commit>:<path>` with no parse, and its git BLOB ID must equal the original's (checked on the
   written file, in the index and in the new commit), and its sha256 must equal the listed one;
2. the commit adds ONLY the copied paths, and its message lists each path, its original commit and its sha256;
3. every cited commit is on a pushed origin ref (after `git fetch origin`; a remote-tracking ref under refs/remotes/origin contains it);
4. it lands only when nothing on B is in flight (the operator's check; this script never pushes).
A listed path already at HEAD with the original's blob id is reported PRESENT and not copied; at HEAD with another blob id it STOPs. All
checks except the blob ids of the written files run before anything is written. The list is checked before any write, and so is every
path's absence from the working tree. Nothing is ever deleted or overwritten.

  uv run --offline --no-project --python 3.12 python -I tools_routec/copy_records_commit.py --list <file> [--dry-run]

The list file has one item per line: `<40-hex commit> <repo-root path> <64-hex sha256>`; blank lines and lines starting with # are skipped.
Run it from the repo root of a worktree on a branch, with a clean tracked tree and an empty index. It prints only paths, commits, blob
ids, sha256s and counts.
"""
import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

HEX40, HEX64 = re.compile(r"[0-9a-f]{40}"), re.compile(r"[0-9a-f]{64}")


def canonical(p):
    return isinstance(p, str) and p != "" and not p.startswith("/") and all(s not in ("", ".", "..") for s in p.split("/"))


def parse_list(text):
    """Pure. -> (items [(commit, path, sha256)], problems)."""
    items, bad = [], []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split()
        if len(f) != 3 or not HEX40.fullmatch(f[0]) or not canonical(f[1]) or not HEX64.fullmatch(f[2]):
            bad.append(f"list line {n}: want '<40-hex commit> <canonical repo-root path> <64-hex sha256>'")
            continue
        items.append(tuple(f))
    paths = [p for _, p, _ in items]
    if len(set(paths)) != len(paths):
        bad.append("a path is listed twice")
    if not items and not bad:
        bad.append("the list is empty")
    return items, bad


def git(*a, check=False):
    r = subprocess.run(["git", *a], capture_output=True)
    if check and r.returncode != 0:
        sys.exit(f"STOP git {' '.join(a)} failed")
    return r


def out(*a):
    return git(*a).stdout.decode().strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not out("symbolic-ref", "--quiet", "HEAD") or out("status", "--porcelain", "--untracked-files=no") or out("diff", "--cached", "--name-only"):
        sys.exit("STOP: HEAD must be on a branch, with a clean tracked tree and an empty index")
    items, bad = parse_list(Path(a.list).read_text())
    if bad:
        print("\n".join("STOP " + b for b in bad))
        sys.exit(1)
    git("fetch", "-q", "origin", check=True)
    head = out("rev-parse", "HEAD")
    todo, present = [], []
    for c, p, s in items:
        if out("cat-file", "-t", c) != "commit":
            bad.append(f"not a commit: {c}")
            continue
        if not out("for-each-ref", "--contains", c, "--format=%(refname)", "refs/remotes/origin"):
            bad.append(f"commit on no pushed origin ref: {c}")
        oid = out("rev-parse", "--verify", "-q", f"{c}:{p}")
        if not HEX40.fullmatch(oid) or out("cat-file", "-t", oid) != "blob":
            bad.append(f"not a blob at its commit: {c}:{p}")
            continue
        if hashlib.sha256(git("cat-file", "blob", oid).stdout).hexdigest() != s:
            bad.append(f"sha256 differs from the listed one: {c}:{p}")
            continue
        at_head = out("rev-parse", "--verify", "-q", f"HEAD:{p}")
        if at_head == oid:
            present.append((c, p, s, oid))
        elif at_head:
            bad.append(f"already at HEAD with another blob id: {p}")
        elif Path(p).exists() or Path(p).is_symlink():
            bad.append(f"exists in the working tree but not at HEAD: {p}")
        else:
            todo.append((c, p, s, oid))
    if bad:
        print("\n".join("STOP " + b for b in bad))
        sys.exit(1)
    for c, p, s, oid in present:
        print(f"PRESENT {p} blob {oid} sha256 {s} (at HEAD as at {c})")
    if not todo:
        print(f"NOTHING_TO_COPY present {len(present)}")
        return
    if a.dry_run:
        for c, p, s, oid in todo:
            print(f"WOULD_COPY {p} blob {oid} sha256 {s} from {c}")
        print(f"DRY_RUN_OK copy {len(todo)} present {len(present)} head {head}")
        return
    for c, p, s, oid in todo:  # 09 (1): written from `git show <commit>:<path>`, no parse
        b = git("show", f"{c}:{p}", check=True).stdout
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        with open(p, "xb") as fh:
            fh.write(b)
        if out("hash-object", "--", p) != oid or hashlib.sha256(Path(p).read_bytes()).hexdigest() != s:
            sys.exit(f"STOP the written copy's blob id or sha256 differs from the original: {p} (copies are left in place, unstaged)")
    paths = sorted(p for _, p, _, _ in todo)
    git("add", "--", *paths, check=True)
    if sorted(out("diff", "--cached", "--name-only").splitlines()) != paths:
        sys.exit("STOP the index holds other paths than the copies (nothing committed)")
    if any(out("rev-parse", f":{p}") != oid for _, p, _, oid in todo):
        sys.exit("STOP a staged blob id differs from the original (nothing committed)")
    msg = ["Copy route C records for COVB to their original paths (09 option (b), byte-exact)", "",
           "Each file is `git show <commit>:<path>` with no parse; its blob id equals the original's (09's conditions):"]
    msg += [f"- {p} from {c} sha256 {s} blob {oid}" for c, p, s, oid in sorted(todo, key=lambda t: t[1])]
    r = subprocess.run(["git", "commit", "-q", "-F", "-"], input="\n".join(msg).encode(), capture_output=True)
    if r.returncode != 0:
        sys.exit("STOP git commit failed (the copies stay staged)")
    new = out("rev-parse", "HEAD")
    if out("rev-parse", f"{new}~1") != head or sorted(out("diff", "--name-only", head, new).splitlines()) != paths:
        sys.exit("STOP the new commit's parent or path set is not the expected one")
    if any(out("rev-parse", f"{new}:{p}") != oid for _, p, _, oid in todo) or out("status", "--porcelain", "--untracked-files=no"):
        sys.exit("STOP a committed blob id differs from the original, or the tracked tree is not clean")
    for c, p, s, oid in sorted(todo, key=lambda t: t[1]):
        print(f"COPIED {p} blob {oid} sha256 {s} from {c}")
    print(f"COPY_COMMIT {new} parent {head} copied {len(todo)} present {len(present)} (not pushed)")


if __name__ == "__main__":
    main()
