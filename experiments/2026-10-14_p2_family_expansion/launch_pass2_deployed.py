"""Pass 2 through the DEPLOYED app (4f's option (b), used only on 4f's word): every check of run_famexp_pass2.main, then the
deployed orchestrate2, so no ephemeral app is created. The deployed app keeps its canonical name (APP, with its kill date).

  PYTHONPATH=src modal deploy experiments/2026-10-14_p2_family_expansion/run_famexp_pass2.py   # from a clean, pushed commit
  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" python <this file> <run_famexp_pass2.main's arguments> --deployed-git <sha>
  modal app stop <deployed app id>                                                              # when done

Refuses unless: the tree is clean; --deployed-git is a commit on origin; run_famexp_pass2.py is byte-identical between that
commit and HEAD; and the deployed code_version() returns that file's sha256 (stale-code-after-redeploy guard).
"""

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_famexp_pass2 as rfp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    for a in ("--spec", "--fit-run", "--message", "--message-sha", "--r36", "--open-manifest", "--open-sha", "--deployed-git"):
        ap.add_argument(a, required=True)
    for a in ("--elig-commit", "--elig-path", "--elig-sha256"):
        ap.add_argument(a, default="")
    ap.add_argument("--out-prefix", default="scaleup")
    a = ap.parse_args()
    git = lambda *x: subprocess.run(["git", "-C", str(rfp.REPO), *x], capture_output=True, text=True, check=True).stdout.strip()  # noqa: E731
    me = str((HERE / "run_famexp_pass2.py").relative_to(rfp.REPO))
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("uncommitted changes: commit and push first")
    if git("cat-file", "-t", a.deployed_git) != "commit" or not any(b.strip().startswith("origin/") for b in git("branch", "-r", "--contains", a.deployed_git).splitlines()):
        raise SystemExit("--deployed-git is not a commit on origin")
    if subprocess.run(["git", "-C", str(rfp.REPO), "diff", "--quiet", a.deployed_git, "HEAD", "--", me]).returncode != 0:
        raise SystemExit("run_famexp_pass2.py differs between the deployed commit and HEAD: redeploy from HEAD")
    want = {"script_sha256": hashlib.sha256((HERE / "run_famexp_pass2.py").read_bytes()).hexdigest(), "code_fingerprint": rfp._fingerprint()}
    got = modal.Function.from_name(rfp.APP, "code_version").remote()
    if got != want:  # an early refusal; the binding check runs inside every worker (head["code_fingerprint"], Codex round 37)
        raise SystemExit(f"the deployed code is not this commit's run_famexp_pass2.py + helper bundle ({got} vs {want}): redeploy")
    orch = modal.Function.from_name(rfp.APP, "orchestrate2")
    rfp._launch(a.spec, a.fit_run, a.message, a.message_sha, a.r36, a.open_manifest, a.open_sha, a.elig_commit, a.elig_path, a.elig_sha256,
                a.out_prefix, call=orch.remote, extra_head={"deployed": {"app": rfp.APP, "git": a.deployed_git} | want})


if __name__ == "__main__":
    main()
