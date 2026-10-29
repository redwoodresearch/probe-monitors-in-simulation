"""The suffix copy (0d, 10-17). Checks, each with its exact outcome:
- run_records_step_in_container_sfx.py is the reviewed wrapper (blob 70a13a24, read from git) with exactly the pinned hunks applied,
  each anchored on a unique original passage: so run_step, preflight, the allowlists, the reducers, the payload and every Modal function
  spec (block_network, timeouts, retries, volumes, images) are byte-unchanged. The docstring paragraph is inert text (no quote or
  backslash can end the string).
- app names: the planned post-§1 steps get distinct names, each at most 63 characters (Modal cuts descriptions at 64), none equal to
  the shared APP_NAME; the longest legal suffix (7 characters) gives exactly 63. A missing, empty, non-string, 8-character, hyphen,
  upper-case, underscore, slash, space, non-ASCII or newline-ended suffix STOPs. The two name guards STOP on their own (the suffix check
  loosened in the test): a name over 63 characters, and a name equal to APP_NAME. Positive control: the old base plus a planned suffix is over 63.
- the lock path: in a synthetic repo, a linked worktree and the main checkout resolve the same `--git-common-dir`, and a
  refs/recstep/* ref made in one is seen in the other (the shared state the lock serialises).
Code only; no Modal call, no record read.

  TMPDIR=<a scratchpad dir> uv run --no-project --python 3.12 python -I experiments/2026-10-17_records_step_container/test_records_step_suffix.py [--copy-file <mutant>]
"""
import argparse
import importlib.util
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = "70a13a249e1dfc9eff29dcb18e17f2624e48f679"  # the reviewed wrapper; S13, PS and CAT pin it unchanged
STEPS = {"§2 verify": "s2v", "PS spec-only": "psso", "CAT spec-only": "catso", "§3 A": "a", "§3 B": "b", "§3 R37": "r37",
         "§6 A pinned": "apin", "§6 B pinned": "bpin", "§6 R37 pinned": "r37pin"}  # 2e's planned post-§1 steps -> app_suffix
HUNKS = [  # (unique original passage, its replacement), applied in order to the base blob
    ("import argparse\n", "import argparse\nimport fcntl\n"),
    ('APP_NAME = "probe-shift_aswin-surya_sub-q3cutfull_recstep_kill-2026-11-14"  # 61 chars\n',
     'APP_NAME = "probe-shift_aswin-surya_sub-q3cutfull_recstep_kill-2026-11-14"  # 61 chars\n'
     'RUN_NAME = "probe-shift_aswin-surya_sub-q3cutfull_rs{}_kill-2026-11-14"  # the suffix copy: the step\'s app description, 56 + suffix <= 63 chars\n'
     'SUFFIX = re.compile(r"[a-z0-9]{1,7}")  # the step spec\'s app_suffix (fullmatch)\n'),
    ("\n\ndef client(spec_path: str, commit: str) -> None:\n",
     '\n\ndef run_name(spec: dict) -> str:\n'
     '    """The suffix copy: the step\'s own app description (each step\'s watcher matches it exactly), from the spec\'s app_suffix."""\n'
     '    s = spec.get("app_suffix")\n'
     '    if type(s) is not str or not SUFFIX.fullmatch(s):\n'
     '        raise Stop("the step spec\'s app_suffix must match [a-z0-9]{1,7}")\n'
     '    n = RUN_NAME.format(s)\n'
     '    if len(n) > 63 or n == APP_NAME:  # Modal cuts a description at 64: a longer one matches no exact-description check\n'
     '        raise Stop("the step\'s app name is over 63 characters or equals the shared APP_NAME")\n'
     '    return n\n'
     '\n\ndef client(spec_path: str, commit: str) -> None:\n'),
    ('    payload, spec = preflight(repo, spec_path, commit)\n',
     '    payload, spec = preflight(repo, spec_path, commit)\n'
     '    name = run_name(spec)  # before any Modal call\n'),
    ('"commit", commit)  # O-S3: the log names its spec\n',
     '"commit", commit)  # O-S3: the log names its spec\n'
     '    print("app", name)\n'),
    ('    tmp = Path(tempfile.mkdtemp(prefix="recstep_"))\n',
     '    lock = open(Path(git_l(repo, "rev-parse", "--path-format=absolute", "--git-common-dir").strip()) / "recstep_bundle.lock", "a")\n'
     '    tmp = Path(tempfile.mkdtemp(prefix="recstep_"))\n'),
    ('    try:\n        f, bsha = make_bundle(repo, payload["commits"], tmp)\n',
     '    try:\n'
     '        fcntl.flock(lock, fcntl.LOCK_EX)  # concurrent clients: refs/recstep/* and bundles/ are shared; released by close() below\n'
     '        f, bsha = make_bundle(repo, payload["commits"], tmp)\n'),
    ('    finally:\n        shutil.rmtree(tmp, ignore_errors=True)\n    with app.run():',
     '    finally:\n        lock.close()\n        shutil.rmtree(tmp, ignore_errors=True)\n    with app.run(name=name):'),
]
DOC_OLD, DOC_HEAD, DOC_END = "a fixed phrase.\n\n  PY=", "a fixed phrase.\n\n", "\n  PY="  # the paragraph goes between head and end


def git(repo, *a):
    return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True).stdout.strip()


def main(copy_file=None):
    res = {}
    real = git(HERE, "rev-parse", "--show-toplevel")
    base = subprocess.run(["git", "-C", real, "cat-file", "-p", BASE], capture_output=True, text=True, check=True).stdout
    copy = Path(copy_file or HERE / "run_records_step_in_container_sfx.py").read_text()  # copy_file: a mutant (the test must FAIL)

    want, unique = base, True
    for old, new in HUNKS:
        unique &= want.count(old) == 1
        want = want.replace(old, new)
    i = copy.find(DOC_HEAD) + len(DOC_HEAD)
    para = copy[i:copy.find(DOC_END, i)]
    unique &= want.count(DOC_OLD) == 1 and copy.count(DOC_HEAD) == 1
    want = want.replace(DOC_OLD, DOC_HEAD + para + DOC_END)
    res["the copy is 70a13a24 with exactly the pinned hunks (each anchor unique); everything else byte-unchanged"] = unique and want == copy
    res["the docstring paragraph is inert: non-empty, no quote run or backslash, one paragraph ending in a full stop"] = \
        para.startswith("The suffix copy (0d, 10-17)") and para.endswith(".\n") and '"""' not in para and "\\" not in para

    d = Path(tempfile.mkdtemp(prefix="recstep_sfx_mod_"))
    (d / "w.py").write_text(copy)
    spec = importlib.util.spec_from_file_location("recstep_sfx_under_test", d / "w.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    names = {k: m.run_name({"app_suffix": s}) for k, s in STEPS.items()}
    pat = re.compile(r"probe-shift_aswin-surya_sub-q3cutfull_rs[a-z0-9]{1,7}_kill-2026-11-14")
    res["the 9 planned steps: distinct names, each <= 63 characters, the pattern whole, none the shared APP_NAME"] = \
        len(set(names.values())) == 9 and all(len(n) <= 63 and pat.fullmatch(n) for n in names.values()) and m.APP_NAME not in names.values()
    res["the longest legal suffix (7 characters) gives exactly 63 characters"] = len(m.run_name({"app_suffix": "a" * 7})) == 63

    def stops(sp, why="the step spec's app_suffix must match [a-z0-9]{1,7}"):
        try:
            m.run_name(sp)
        except m.Stop as e:
            return str(e) == why
        return False
    bad = [{}, {"app_suffix": None}, {"app_suffix": 5}, {"app_suffix": ""}, {"app_suffix": "a" * 8}, {"app_suffix": "a-so"}, {"app_suffix": "Aso"},
           {"app_suffix": "a_so"}, {"app_suffix": "a/so"}, {"app_suffix": "a so"}, {"app_suffix": "é"}, {"app_suffix": "aso\n"}]
    res["NEGATIVE a missing, empty, non-string, 8-character, hyphen, upper-case, _ / space, non-ASCII or newline-ended suffix: STOP"] = \
        all(stops(b) for b in bad)
    long_why = "the step's app name is over 63 characters or equals the shared APP_NAME"
    m.SUFFIX = re.compile(r"[a-z0-9]{1,20}")  # the length guard alone: the suffix check loosened
    long_stops = stops({"app_suffix": "a" * 8}, long_why) and len(m.RUN_NAME.format("a" * 8)) == 64
    m.RUN_NAME = "probe-shift_aswin-surya_sub-q3cutfull_re{}_kill-2026-11-14"  # the base guard alone: a stem that can rebuild APP_NAME
    base_stops = stops({"app_suffix": "cstep"}, long_why) and m.RUN_NAME.format("cstep") == m.APP_NAME
    res["NEGATIVE the name guards on their own: a 64-character name STOPs; a name equal to APP_NAME STOPs"] = long_stops and base_stops
    res["positive control: the old 61-character base with a planned suffix (_r37pin) is over 63 characters"] = \
        len(m.APP_NAME.replace("_kill-", "_r37pin_kill-")) > 63 and len(m.APP_NAME) == 61

    tmp = Path(tempfile.mkdtemp(prefix="recstep_sfx_repo_"))
    repo, wt = tmp / "repo", tmp / "wt"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "f").write_text("x\n")
    git(repo, "add", "f")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "c1")
    c1 = git(repo, "rev-parse", "HEAD")
    git(repo, "worktree", "add", "--quiet", str(wt), c1)
    cd = lambda r: git(r, "rev-parse", "--path-format=absolute", "--git-common-dir")  # noqa: E731
    git(wt, "update-ref", f"refs/recstep/{c1}", c1)
    seen = git(repo, "for-each-ref", "--format=%(refname)", "refs/recstep/")
    res["the lock path: a linked worktree and the main checkout share one git common dir, and refs/recstep/* with it"] = \
        cd(repo) == cd(wt) == str((repo / ".git").resolve()) and seen == f"refs/recstep/{c1}"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(d, ignore_errors=True)

    for k, v in res.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("ALL PASS" if all(res.values()) else "FAILED")
    if not all(res.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy-file", default=None, help="test this file instead (the mutation runs: the test must FAIL)")
    main(ap.parse_args().copy_file)
