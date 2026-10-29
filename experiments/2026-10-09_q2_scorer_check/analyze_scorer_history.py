"""Did any matched task's main-task scorer change between the September environment releases and the newest code? (local,
read-only, $0). For each environment, the September checkout in ../simulated-control-evals-sept/.settings/linuxarena/<env>
(whose codebase digest matches one of Sol's recorded digests in 8 of 10 environments) is compared with the newest commit
of the environment repo (cloned by analyze_env_versions.py into ~/.cache/probe-shift/linuxarena-envs). For every main
task with runs on both sides: the commits touching main_tasks/<task>/ in between, and the changed lines, split into
registration-only lines (the scorer function's name, the @linux_scorer / @scorer wrapper, imports, blank lines) and the
rest. Shared scorer helpers and environment code outside main_tasks/<task>/ are not covered.

  python3 experiments/2026-10-09_q2_scorer_check/analyze_scorer_history.py results/scorer_versions_<stamp>.json
"""
import difflib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
CACHE = Path.home() / ".cache" / "probe-shift" / "linuxarena-envs"
SEPT = Path.home() / "simulated-control-evals-sept" / ".settings" / "linuxarena"
# The refactor renames the module-level `scorer` to `score` (applied to the old file before diffing, so a changed
# right-hand side still shows) and wraps it: "@scorer(...)", "def <task>_scorer() -> Scorer:", "return <fn>".
RENAME = [(re.compile(r"^scorer = ", re.MULTILINE), "score = "), (re.compile(r"^async def scorer\(", re.MULTILINE), "async def score(")]
REGISTRATION = re.compile(r"linux_scorer|^[+-]\s*$|^[+-]\s*(from|import) |^[+-]@scorer\(|^[+-]def \w+_scorer\(\) -> Scorer:|^[+-]\s+return \w+$")


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


def main(versions_json: Path):
    runs = json.loads(versions_json.read_text())["runs"]
    tasks = {}
    for r in runs["sol"]:
        if any(q["main_task"] == r["main_task"] for q in runs["qwen3.5"]):
            tasks.setdefault(r["environment"], set()).add(r["main_task"])
    out = {"source": versions_json.name, "environments": {}, "tasks": {}}
    for env in sorted(tasks):
        repo = CACHE / env
        base, head = git(SEPT / env, "rev-parse", "HEAD").strip(), git(repo, "rev-parse", "HEAD").strip()
        out["environments"][env] = {"september": f"{base[:8]} ({git(SEPT / env, 'describe', '--tags').strip()})", "newest": head[:8]}
        for t in sorted(tasks[env]):
            commits = git(repo, "log", "--format=%h %s", f"{base}..{head}", "--", f"main_tasks/{t}/").strip().splitlines()
            lines = []
            for name in git(repo, "diff", "--name-only", f"{base}..{head}", "--", f"main_tasks/{t}/").split():
                old = git(repo, "show", f"{base}:{name}") if name in git(repo, "ls-tree", "-r", "--name-only", base, "--", name) else ""
                new = git(repo, "show", f"{head}:{name}") if name in git(repo, "ls-tree", "-r", "--name-only", head, "--", name) else ""
                for pattern, repl in RENAME:
                    old = pattern.sub(repl, old)
                lines += [ln for ln in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=0)
                          if ln[:1] in "+-" and not ln.startswith(("+++", "---"))]
            other = [ln for ln in lines if not REGISTRATION.search(ln)]
            out["tasks"][t] = {"environment": env, "commits": commits, "changed lines": len(lines), "other changed lines": other}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    p = HERE / "results" / f"scorer_history_{stamp}.json"
    p.write_text(json.dumps(out, indent=1) + "\n")
    for t, row in out["tasks"].items():
        print(f"{row['environment']}/{t}: {len(row['commits'])} commit(s) {[c.split(' ', 1)[1][:60] for c in row['commits']]}; "
              f"{row['changed lines']} changed lines, {len(row['other changed lines'])} beyond registration")
    print("tasks:", len(out["tasks"]), "| with any change beyond registration:", sorted(t for t, r in out["tasks"].items() if r["other changed lines"]))
    print(p)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
