"""Item 3, review follow-up (POST HOC, 2026-10-14): hashes of every --assets document the scanners read, and where each
came from. Answers #1675's review threads (Codex: provenance of the side-task documents; Bugbot: whether they loaded).

The scanners (run_probe_sees*.py, read_assets) read <assets>/side_tasks/<env>/<side task>/{goal.md,scoring_details.md}
and <assets>/attacker_2026_08_28.md. That assets folder was a flattened copy of each environment repo's
side_tasks/<side task>/docs/ under the September sims snapshot's .settings/linuxarena/. This script hashes each assets
file, checks it byte-for-byte against the environment repo's docs/ file, and records each environment repo's commit.
Hashes and names only: no document text is written.

  uv run --no-project python experiments/2026-10-13_q2_item3_probe_sees/make_assets_manifest.py \
      --assets <assets folder> --settings <sims snapshot>/.settings/linuxarena --out <results json>
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731


def main(a):
    assets, settings = Path(a.assets), Path(a.settings)
    files, mismatch, envs, dirty = {}, [], {}, 0
    for f in sorted((assets / "side_tasks").glob("*/*/*")):
        env, st, name = f.relative_to(assets / "side_tasks").parts
        src = settings / env / "side_tasks" / st / "docs" / name
        files[f"{env}/{st}/{name}"] = sha(f)
        if not src.is_file() or src.read_bytes() != f.read_bytes():
            mismatch.append(f"{env}/{st}/{name}")
        if env not in envs:
            def git(*c, env=env):
                return subprocess.run(["git", "-C", str(settings / env), *c], capture_output=True, text=True,
                                      check=True).stdout.strip()

            envs[env] = git("rev-parse", "HEAD")
            dirty += len(git("status", "--porcelain", "--", "side_tasks").splitlines())
    side_tasks = {k.rsplit("/", 1)[0] for k in files}
    out = {"POST HOC": True, "what": __doc__.split("\n\n")[0],
           "side task folders": len(side_tasks), "with goal.md": sum(k.endswith("/goal.md") for k in files),
           "with scoring_details.md": sum(k.endswith("/scoring_details.md") for k in files),
           "files equal to the environment repo's docs/ file": len(files) - len(mismatch),
           "files that differ or are missing": mismatch,
           "attacker prompt sha256": sha(assets / "attacker_2026_08_28.md"),
           "environment repo commits": envs, "uncommitted side_tasks changes in those repos": dirty,
           "sha256 by file": files}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("sha256 by file", "environment repo commits")}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--assets", required=True)
    p.add_argument("--settings", required=True)
    p.add_argument("--out", required=True)
    main(p.parse_args())
