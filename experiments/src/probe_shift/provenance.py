"""Provenance captured when a run is launched, before any code can change: the commit, whether tracked files differ
from it, and a hash of the code that runs, which also covers uncommitted and untracked scripts (as q2's).

With `commit`, every file must be byte-identical to that commit (a launch from a working tree whose code is committed
on a PR branch), or the launch fails; the commit is then recorded as the code that ran."""

import hashlib
import subprocess
from pathlib import Path


def capture(root: Path, files: list[Path], commit: str = "") -> dict:
    git = lambda *args: subprocess.run(["git", *args], capture_output=True, cwd=root).stdout
    rel = [str(Path(f).resolve().relative_to(Path(root).resolve())) for f in files]
    out = {"git": git("rev-parse", "HEAD").decode().strip(),
           "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no").strip()),
           "source_sha": hashlib.sha256(b"".join(Path(f).read_bytes() for f in files)).hexdigest()[:16], "files": rel}
    if commit:
        full = git("rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}").decode().strip()
        if not full:
            raise ValueError(f"unknown commit {commit}")
        differ = [r for r, f in zip(rel, files) if git("show", f"{full}:{r}") != Path(f).read_bytes()]
        if differ:
            raise ValueError(f"not as committed in {full[:10]}: {differ}")
        out["source_matches_commit"] = full
    return out
