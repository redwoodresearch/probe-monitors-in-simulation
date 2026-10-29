"""Which LinuxArena environment release did each side run, and did the main-task scorers change between them? (local,
read-only, $0). For each environment repo (cloned from GitHub into ~/.cache/probe-shift/linuxarena-envs), every
release tag's `codebase/` is hashed with the runner's own reference-contract digest (simulated-control-evals
runner/arenas/reference_contract.py, imported from the sibling checkout) and matched to the codebase digests recorded
in each side's runs (analyze_scorer_versions.py's result). Then, for every main task on both sides, the files under
`main_tasks/<task>/` are compared between the matched tags.

  python3 experiments/2026-10-09_q2_scorer_check/analyze_env_versions.py results/scorer_versions_<stamp>.json
"""
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
CACHE = Path.home() / ".cache" / "probe-shift" / "linuxarena-envs"
SEPT = Path.home() / "simulated-control-evals-sept" / ".settings" / "linuxarena"
SCE = Path.home() / "simulated-control-evals" / "src" / "runner" / "arenas" / "reference_contract.py"
spec = importlib.util.spec_from_file_location("reference_contract", SCE)
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)


def git(repo: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True).stdout


def repo(env: str) -> Path:
    path = CACHE / env
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--bare", "--quiet", f"https://github.com/linuxarena/{env}.git", str(path)], check=True)
    git(path, "fetch", "--quiet", "--tags", "origin")
    return path


def tag_digest(path: Path, tag: str) -> str | None:
    """The reference-contract digest of `codebase/` at a tag, from a clean extraction (git's file modes)."""
    try:
        data = git(path, "archive", "--format=tar", tag, "codebase")
    except subprocess.CalledProcessError:
        return None
    with tempfile.TemporaryDirectory() as tmp, tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(tmp, filter="tar")
        return rc._tree_digest((("environment/0", Path(tmp) / "codebase"),))


def task_files(path: Path, tag: str, task: str) -> dict[str, str]:
    """sha256 of every file under main_tasks/<task>/ at a tag."""
    names = git(path, "ls-tree", "-r", "--name-only", tag, f"main_tasks/{task}").decode().split()
    return {n.removeprefix(f"main_tasks/{task}/"): hashlib.sha256(git(path, "show", f"{tag}:{n}")).hexdigest()[:16] for n in names}


def main(versions_json: Path):
    v = json.loads(versions_json.read_text())
    runs = v["runs"]
    envs = sorted({r["environment"] for side in runs.values() for r in side})
    out = {"source": versions_json.name, "environments": {}, "tasks": {}}
    for env in envs:
        path = repo(env)
        tags = [t for t in git(path, "tag", "--list", "v*").decode().split() if t]
        digests = {t: tag_digest(path, t) for t in tags}
        match = {side: {d[:12]: sorted(t for t, g in digests.items() if g == d) for d in sorted({r["contents_sha256"] for r in rows if r["environment"] == env})}
                 for side, rows in runs.items()}
        local = rc._tree_digest((("environment/0", SEPT / env / "codebase"),))  # the September checkout as it sits on disk
        out["environments"][env] = {"tags": tags, "matched tags per recorded digest": match,
                                    "September checkout digest matches": {side: any(r["contents_sha256"] == local for r in rows if r["environment"] == env)
                                                                          for side, rows in runs.items()}}
        tasks = sorted({r["main_task"] for r in runs["sol"] if r["environment"] == env} & {r["main_task"] for r in runs["qwen3.5"] if r["environment"] == env})
        for task in tasks:
            sides = {}
            for side, rows in runs.items():
                ds = sorted({r["contents_sha256"][:12] for r in rows if r["main_task"] == task})
                sides[side] = sorted({t for d in ds for t in match[side].get(d, [])})
            row = {"tags": sides}
            if sides["sol"] and sides["qwen3.5"]:
                a, b = task_files(path, sides["sol"][0], task), task_files(path, sides["qwen3.5"][0], task)
                row["changed files"] = sorted(f for f in set(a) | set(b) if a.get(f) != b.get(f))
                row["compared"] = [sides["sol"][0], sides["qwen3.5"][0]]
            out["tasks"][task] = row
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    p = HERE / "results" / f"env_versions_{stamp}.json"
    p.write_text(json.dumps(out, indent=1) + "\n")
    for env, e in out["environments"].items():
        print(env, {side: m for side, m in e["matched tags per recorded digest"].items()}, "September checkout matches:", e["September checkout digest matches"])
    for task, row in out["tasks"].items():
        print(task, row["tags"], "changed:", row.get("changed files", "not comparable (a side's digest matches no tag)"))
    print(p)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
