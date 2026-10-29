"""Synthetic test of the wrapper's route C features (no Modal; run_step runs locally on a real bundle of a throwaway repo; no real
record, manifest or Volume is read). Cases, each with its exact outcome:
- raw_f spec-only mode: a fabricated runner whose `main.info.raw_f(**flags)` writes ONE pickle at the "@out:" path; it also runs
  `git status` (after touching a tracked file's mtime) and `git ls-files` through the wrapper, reads a staged manifest at its absolute
  path and a working-tree config, and starts a CHILD process (no -I, cwd = its folder) whose git call goes through the wrapper.
  Result: OK; the one pickle comes back as metadata (sha256, size, vol_path) with its bytes on the outputs Volume, and the client
  fetches and writes it without unpickling; the reducer relays only
  whether the stdout JSON line's keys match the pinned expect_keys, plus counts (no value line, no value, no key text); .git is
  byte-identical (no stray).
- `git status` and the knob: with GIT_OPTIONAL_LOCKS=0 (forced by the wrapper) a status after an mtime change leaves .git/index
  byte-identical; the control without the knob rewrites it (why the knob is required).
- staging (the INDEX schema of 33f1ceaf2: n_files, files[] with original / repo_path / sha256 / bytes / status / sources): every
  entry is staged by sha256 whatever its status; a sha mismatch, a missing file, an extra file on the Volume, a count != n_files, an
  INDEX that differs from its pin, an entry outside the prefix: each STOPs before the runner starts; a runner that changes a staged
  file STOPs; an "@out:" flag that is not an allowlisted output STOPs.
- a spec-pinned script blob (SCRIPTS blob None): the right blob passes preflight; a wrong or missing script_blob STOPs.
- the reply-size fix: the pickle comes back as metadata, its bytes on the fake outputs Volume (runs/<run_id>/...), and the client
  fetches, sha-checks and writes them; the VolumeFS version pins: (a) every OUT_VOL / RCMAN_VOL from_name passes version=2, the
  bundle and stage ones unchanged; (b) modal 1.6.1's own _validate_volume_version raises for v2 on a v1 Volume and passes on v2
  (the pinned modal python, no Modal call).
- O19: the step spec pins expect_keys; only keys_match and counts return; keys that differ STOP; a json_line_keys spec without
  expect_keys STOPs in preflight. O20: a staging prefix that overlaps the work dir STOPs before staging.
- 09's F-out2 and (iii) (10-17, the census side a STOP): the runner opens its @out path "xb" WITHOUT making its folder (like
  run_peraction311 L5347), in a folder with no tracked file. The wrapper as it ran (ed31995, read from git) STOPs with
  FileNotFoundError; this wrapper makes the parent and the same payload is OK. The guard: an escape by "..", an absolute path, a path
  outside experiments/<folder>/results/, a symlinked parent, an existing FILE where a parent would go and an output that already exists
  each STOP before the runner starts. A runner raise returns exc_type and code lines only ("fab/runner.py:<line>", "<outside>"): its
  message text reaches neither the reply nor the client's STOP line, and the stop reason is the first failing condition.
- a0b05's review of 47d297455 (10-17): an OUT_PREFIX-valid but unlisted @out (G4), non-canonical @out paths ("./", "//", "/./", a
  trailing "/"; O2) and a symlinked parent resolving outside the checkout (O3) STOP before the runner; a runner that forges the
  exception file relays ["?", "?"] (T1); the client re-filters a forged reply (T2); "a/../b.py:1" is refused (T7); SystemExit(3)
  keeps its code lines (T8). --wrapper-file runs the test on a mutant (each of G4, T1, T2, T7, T8 and O2 makes it FAIL).

  uv run --no-project --python 3.12 python experiments/2026-10-17_records_step_container/test_records_step_routec.py [--worktree]
"""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
REAL = HERE.parents[1]
MODAL_PY = "/Users/aswinsurya/.cache/probe-shift-modal161/bin/python"  # the pinned modal 1.6.1 env (the version-pin library check only)
REL = "experiments/2026-10-17_records_step_container"
RUNNER = r'''
import json, os, pickle, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = HERE.parent


class _Info:
    def __init__(self, f):
        self.raw_f = f


class _Fn:
    def __init__(self, f):
        self.info = _Info(f)


def _main(spec_only=None, alltasks=False, staged=None, mutate_staged=False, boom=False, forge_tb=False, exit3=False):
    if boom:  # a raise whose message must never leave the container (09's (iii): code lines only)
        raise FileNotFoundError("SENTINEL_MESSAGE_TEXT /Users/someone/laptop/path.json")
    if forge_tb:  # a runner that writes the wrapper's exception file itself, then exits before the boot can (a0b05's T1)
        with open(os.environ["RECSTEP_EXC"], "w") as fh:
            fh.write(json.dumps({"code": 1, "type": "FileNotFoundError", "tb": ["SENTINEL text", "a/../b.py:1"]}))
        os._exit(1)
    if exit3:  # a SystemExit with a non-zero code keeps its code lines (a0b05's T8)
        raise SystemExit(3)
    os.utime(REPO / "config" / "cfg.json", (1, 1))  # an mtime change: a plain status would refresh .git/index
    st = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    if st.strip():
        raise SystemExit("the tracked tree is not clean")
    subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, check=True)
    man = Path(staged).read_text()
    cfg = json.loads((REPO / "config" / "cfg.json").read_text())
    if mutate_staged:
        Path(staged).write_text("changed by the runner")
    child = subprocess.run([sys.executable, str(HERE / "child.py"), "check"], cwd=HERE, capture_output=True)
    if child.returncode:
        raise SystemExit("the child failed")
    with open(spec_only, "xb") as fh:  # like run_peraction311 L5347 (dispatch, --spec-only): no mkdir of the parent (the census STOP)
        pickle.dump({"n": 3, "alltasks": alltasks, "man_len": len(man), "cfg": cfg["k"]}, fh)
    print("auroc 0.97 (a value line)")
    print(json.dumps({"spec_only": spec_only, "n_units": 3, "score": 0.97}))


main = _Fn(_main)
if __name__ == "__main__":
    raise SystemExit("the runner must not run as __main__ here")
'''
CHILD = r'''
import subprocess, sys
subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True)
'''


def git(repo, *a, env=None):
    return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True, env=env).stdout.strip()


def wrapper_src(worktree, wrapper_file=None):
    if wrapper_file:  # a mutant for the mutation runs (a0b05's REQUIRED, 10-17): the test must FAIL on it
        return Path(wrapper_file).read_bytes()
    return (HERE / "run_records_step_in_container.py").read_bytes() if worktree else \
        subprocess.run(["git", "-C", str(REAL), "show", f"HEAD:{REL}/run_records_step_in_container.py"], capture_output=True, check=True).stdout


def load(worktree, wrapper_file=None):
    src = wrapper_src(worktree, wrapper_file)
    d = Path(tempfile.mkdtemp(prefix="recstep_rc_mod_"))
    (d / "w.py").write_bytes(src)
    spec = importlib.util.spec_from_file_location("recstep_rc_under_test", d / "w.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def tree(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


def main(worktree=False, wrapper_file=None):
    m = load(worktree, wrapper_file)
    tmp = Path(tempfile.mkdtemp(prefix="recstep_rc_test_"))
    res = {}
    repo = tmp / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    for k, v in (("user.email", "test@example.invalid"), ("user.name", "test"), ("commit.gpgsign", "false")):
        git(repo, "config", k, v)
    (repo / "fab").mkdir()
    (repo / "fab" / "runner.py").write_text(RUNNER)
    (repo / "fab" / "child.py").write_text(CHILD)
    (repo / "config").mkdir()
    (repo / "config" / "cfg.json").write_text(json.dumps({"k": "v"}))
    # the staged manifests: an absolute prefix outside the repo; a fake read-only Volume holding them at their original paths
    prefix = tmp / "abs" / "scratchpad" / "fpr507"
    rcman = tmp / "rcman"
    files = {f"{prefix}/simh/a.json": b'{"a": 1}', f"{prefix}/v2man/b.json": b'{"b": 2}', f"{prefix}/man710/c.json": b'{"c": 3}'}
    statuses = ["match", "no pin", "not verified locally: verified in the container"]
    for (o, b) in files.items():
        (rcman / o.lstrip("/")).parent.mkdir(parents=True, exist_ok=True)
        (rcman / o.lstrip("/")).write_bytes(b)
    index = {"what": "fabricated", "source_root": str(prefix), "mirror_root": "x", "n_files": 3, "per_folder": {}, "excluded_non_json": [],
             "excluded_note": "", "files": [{"original": o, "repo_path": f"m/{i}.json", "sha256": m.sha(b), "bytes": len(b), "status": statuses[i],
                                             "sources": []} for i, (o, b) in enumerate(files.items())], "backup_check": None}
    (repo / "mirror").mkdir()
    (repo / "mirror" / "INDEX.json").write_text(json.dumps(index, indent=1))
    # 09's F-out2 guard fixtures (10-17): a symlinked parent, a FILE where a parent directory would go, an output that already exists.
    # The main case's output folder experiments/fab/results/specs/ holds no tracked file, like the census's results/routec_census_specs/.
    (repo / "experiments" / "lnk" / "results").mkdir(parents=True)
    os.symlink("../../../elsewhere", repo / "experiments" / "lnk" / "results" / "out")  # resolves INSIDE the checkout (repo/elsewhere)
    (repo / "experiments" / "esc" / "results").mkdir(parents=True)
    os.symlink("../../../../outside", repo / "experiments" / "esc" / "results" / "out")  # resolves OUTSIDE the checkout (O3)
    (repo / "experiments" / "filep" / "results").mkdir(parents=True)
    (repo / "experiments" / "filep" / "results" / "specs").write_text("a file, not a directory")
    (repo / "experiments" / "exist" / "results").mkdir(parents=True)
    (repo / "experiments" / "exist" / "results" / "spec.pkl").write_bytes(b"already here")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "c1")
    c1 = git(repo, "rev-parse", "HEAD")
    blob = git(repo, "rev-parse", f"{c1}:fab/runner.py")
    staged_a = f"{prefix}/simh/a.json"
    spec = {"script": "fab/runner.py", "script_blob": blob, "reducer": "json_line_keys", "expect_keys": ["n_units", "score", "spec_only"], "image": "rc",
            "memory_mib": 4096, "n_outputs": 1,
            "sys_path": ["fab"], "commits": [c1], "outputs": [{"dir": "experiments/fab/results/specs", "regex": r"spec\.pkl"}],
            "entry": {"attr": "main", "flags": {"spec_only": "@out:experiments/fab/results/specs/spec.pkl", "alltasks": True, "staged": staged_a}},
            "staging": {"index": {"commit": c1, "path": "mirror/INDEX.json", "sha256": m.sha((repo / "mirror" / "INDEX.json").read_bytes())},
                        "count": 3, "prefix": str(prefix)}}
    (repo / "steps").mkdir()
    (repo / "steps" / "rc.json").write_text(json.dumps(spec))
    (repo / "steps" / "rc_noblob.json").write_text(json.dumps({k: v for k, v in spec.items() if k != "script_blob"}))
    (repo / "steps" / "rc_badblob.json").write_text(json.dumps(spec | {"script_blob": "0" * 40}))
    (repo / "steps" / "rc_nokeys.json").write_text(json.dumps({k: v for k, v in spec.items() if k != "expect_keys"}))
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "step specs")
    C = git(repo, "rev-parse", "HEAD")
    m.SCRIPTS = {"fab/runner.py": (None, "json_line_keys")}
    payload, _ = m.preflight(repo, "steps/rc.json", C)
    res["spec-pinned blob: the right script_blob passes preflight"] = payload["blob"] == blob and payload["image"] == "rc"
    for name, sp, why in (("missing", "steps/rc_noblob.json", "the script's blob at C (or its reducer) is not the allowlisted one"),
                          ("wrong", "steps/rc_badblob.json", "the script's blob at C (or its reducer) is not the allowlisted one")):
        try:
            m.preflight(repo, sp, C)
            e = None
        except m.Stop as x:
            e = str(x)
        res[f"NEGATIVE spec-pinned blob: a {name} script_blob STOPs"] = e == why
    try:
        m.preflight(repo, "steps/rc_nokeys.json", C)
        e = None
    except m.Stop as x:
        e = str(x)
    res["NEGATIVE O19 a json_line_keys step without expect_keys: STOP in preflight"] = \
        e == "expect_keys: a json_line_keys step pins a sorted, unique, non-empty list of identifier keys; no other step has one"
    broot = tmp / "bundlevol"
    (broot / "bundles").mkdir(parents=True)
    bt = tmp / "bt"
    bt.mkdir()
    f, bsha = m.make_bundle(repo, payload["commits"], bt)
    shutil.copy(f, broot / "bundles" / f"{bsha}.bundle")
    real_git = shutil.which("git")

    def run(fresh_prefix=True, vol=rcman, **kw):
        if fresh_prefix and prefix.exists():
            shutil.rmtree(prefix)
        p = dict(payload, bundle=f"bundles/{bsha}.bundle", bundle_sha256=bsha, run_id=uuid.uuid4().hex) | kw
        w = Path(tempfile.mkdtemp(prefix="recstep_rc_work_", dir=tmp))
        return m.run_step(p, w, broot, tmp / "stagevol", sys.executable, real_git, outv, lambda: None, rcman_root=vol)

    (tmp / "stagevol").mkdir()
    outv = tmp / "outvol"  # the fake outputs Volume (the reply-size fix)
    outv.mkdir()
    OUTP = "experiments/fab/results/specs/spec.pkl"
    # 09's F-out2 (10-17): the census side a STOP reproduced on the wrapper as it ran (ed31995: no @out parent made, the runner's
    # open("xb") raises FileNotFoundError), then fixed by this wrapper (the same payload passes; next case)
    old = subprocess.run(["git", "-C", str(REAL), "cat-file", "-p", "ed31995269bc78b7fc14c37623516e5c05a3e7ef"], capture_output=True, check=True).stdout
    od = Path(tempfile.mkdtemp(prefix="recstep_rc_old_", dir=tmp))
    (od / "w_old.py").write_bytes(old)
    ospec = importlib.util.spec_from_file_location("recstep_rc_old", od / "w_old.py")
    mo = importlib.util.module_from_spec(ospec)
    ospec.loader.exec_module(mo)
    if prefix.exists():
        shutil.rmtree(prefix)
    ro = mo.run_step(dict(payload, bundle=f"bundles/{bsha}.bundle", bundle_sha256=bsha, run_id=uuid.uuid4().hex),
                     Path(tempfile.mkdtemp(prefix="recstep_rc_work_", dir=tmp)), broot, tmp / "stagevol", sys.executable, real_git, outv,
                     lambda: None, rcman_root=rcman)
    res["F-out2 reproduced: the wrapper as it ran (ed31995) on an @out whose parent folder is untracked: STOP, FileNotFoundError, no output"] = \
        ro["status"] == "STOP" and ro["exc_type"] == "FileNotFoundError" and ro["outputs"] == [] and ro["rc"] == 1
    r = run()
    blob_reply = json.dumps(r)  # no bytes anywhere: json.dumps would raise on a bytes value
    on_vol = outv / (r["outputs"] or [{}])[0].get("vol_path", "missing")
    res["F-out2 fixed + raw_f spec-only: OK (the @out parent made in the checkout), exactly one pickle output (metadata; the bytes on the "
        "outputs Volume), status / ls-files / a child's git through the wrapper, .git unchanged"] = r["status"] == "OK" \
        and [o["path"] for o in r["outputs"]] == [OUTP] and r["stray"] == [] and r["exc_tb"] == [] and r["stop_reason"] is None \
        and r["refused_git"] == [] and on_vol.is_file() and m.sha(on_vol.read_bytes()) == r["outputs"][0]["sha256"]
    res["raw_f: the reducer relays only keys_match against the pinned keys and counts (O19); no value line, no JSON value, no key text"] = \
        r["reduced"] == {"json_lines": 1, "keys_match": True, "n_keys": 3} and "0.97" not in blob_reply and "auroc" not in blob_reply \
        and "n_units" not in blob_reply
    res["staging: all 3 entries staged by sha256 whatever their status (match / no pin / verified in the container)"] = \
        all(Path(o).read_bytes() == b for o, b in files.items())
    dest = tmp / "dest"
    dest.mkdir()
    o0 = (r["outputs"] or [{"vol_path": "runs/missing/x", "sha256": None}])[0]  # O1: an empty reply fails the case, not the test run
    try:
        done = m.write_outputs(dest, dict(payload, run_id=o0["vol_path"].split("/")[1]), r, lambda q: (outv / q).read_bytes())
    except m.Stop:
        done = None
    res["client: the pickle is fetched from the outputs Volume, written with open('xb') and re-hashed, never unpickled (bytes only)"] = \
        done == [(OUTP, o0["sha256"])] and (dest / OUTP).is_file() and (dest / OUTP).read_bytes() == on_vol.read_bytes()
    # 09's guard on F-out2 (10-17): parents only inside the checkout, under experiments/<folder>/results/, through real directories
    def out_stop(name, q, odir, why):
        r_ = run(outputs=[{"dir": odir, "regex": r"spec\.pkl"}], entry={"attr": "main", "flags": dict(payload["entry"]["flags"], spec_only=q)})
        ok = r_["status"] == "STOP" and r_["stop_reason"] == why and r_["outputs"] == [] and r_["rc"] is None  # before the runner starts
        res[f"NEGATIVE F-out2 guard, {name}: STOP before the runner"] = ok
        if not ok:
            print("   got:", r_.get("stop_reason"), r_.get("exc_type"), r_.get("rc"))

    nal = "an @out flag is not an allowlisted output path"
    out_stop("an escape by '..'", "@out:experiments/fab/results/../../../../x/results/spec.pkl", "experiments/fab/results/../../../../x/results", nal)
    out_stop("an absolute path", f"@out:{tmp}/experiments/fab/results/specs/spec.pkl", f"{tmp}/experiments/fab/results/specs", nal)
    out_stop("a path outside experiments/<folder>/results/", "@out:config/spec.pkl", "config", nal)
    out_stop("a symlinked parent resolving INSIDE the checkout", "@out:experiments/lnk/results/out/spec.pkl", "experiments/lnk/results/out",
             "an @out parent is a symlink or an existing file")
    out_stop("a symlinked parent resolving OUTSIDE the checkout (O3)", "@out:experiments/esc/results/out/spec.pkl", "experiments/esc/results/out",
             "an @out parent is a symlink or an existing file")
    out_stop("an existing FILE where a parent directory would go", "@out:experiments/filep/results/specs/spec.pkl", "experiments/filep/results/specs",
             "an @out parent is a symlink or an existing file")
    out_stop("an output that already exists", "@out:experiments/exist/results/spec.pkl", "experiments/exist/results",
             "an @out output resolves outside the checkout or already exists")
    out_stop("an OUT_PREFIX-valid @out that is NOT allowlisted (the name misses the regex; G4)", "@out:experiments/fab/results/specs/other.pkl",
             "experiments/fab/results/specs", nal)
    for nm, q in (("a './' prefix", "./experiments/fab/results/specs/spec.pkl"), ("a '//'", "experiments/fab/results//specs/spec.pkl"),
                  ("a '/./' segment", "experiments/fab/results/./specs/spec.pkl"), ("a trailing '/'", "experiments/fab/results/specs/spec.pkl/")):
        out_stop(f"a non-canonical @out path ({nm}; O2)", f"@out:{q}", "experiments/fab/results/specs", nal)
    # 09's (iii): a raise returns its code lines only (no message text), and the stop reason names the first failing condition
    r = run(entry={"attr": "main", "flags": dict(payload["entry"]["flags"], boom=True)})
    n_raise = next(i for i, x in enumerate(RUNNER.split("\n"), 1) if "raise FileNotFoundError(" in x)
    tb_ok = r["exc_tb"][-1:] == [f"fab/runner.py:{n_raise}"] and all(x == "<outside>" or x.startswith("fab/runner.py:") for x in r["exc_tb"])
    try:
        m.write_outputs(tmp / "dest_boom", payload, r, lambda q: b"")
        cmsg = ""
    except m.Stop as x:
        cmsg = str(x)
    res["traceback: a builder raise returns exc_type and code lines only (fab/runner.py:<line of the raise>, <outside> frames); no message "
        "text in the reply or the client's STOP line; the reason is the first failing condition"] = r["status"] == "STOP" \
        and r["exc_type"] == "FileNotFoundError" and tb_ok and "SENTINEL" not in json.dumps(r) and "laptop" not in json.dumps(r) \
        and r["stop_reason"] == "the builder exited non-zero or raised (see the exception type and the traceback)" \
        and f"fab/runner.py:{n_raise}" in cmsg and "SENTINEL" not in cmsg
    res["tb_frames: a forged frame (message text, '..' leading or inside, no .py, extra colon) becomes '?'; only the innermost MAX_TB frames"] = \
        m.tb_frames(["fab/runner.py:3", "<outside>", "SENTINEL text", "../x.py:1", "a/../b.py:1", "x.txt:2", "a.py:1:2"]) == \
        ["fab/runner.py:3", "<outside>", "?", "?", "?", "?", "?"] and len(m.tb_frames(["<outside>"] * 100)) == m.MAX_TB and m.tb_frames("x") == []
    # a0b05's mutants (10-17): the container re-filters a forged exception file (T1); the client re-filters a forged reply (T2); a
    # SystemExit with a non-zero code keeps its code lines (T8)
    r = run(entry={"attr": "main", "flags": dict(payload["entry"]["flags"], forge_tb=True)})
    res["T1: a runner that writes the exception file itself (tb ['SENTINEL text', 'a/../b.py:1']) and exits: the reply's tb is ['?', '?']"] = \
        r["status"] == "STOP" and r["exc_tb"] == ["?", "?"] and "SENTINEL" not in json.dumps(r)
    forged = {"status": "STOP", "exc_type": "FileNotFoundError", "exc_tb": ["SENTINEL text"], "stop_reason": "x", "refused_git": [], "stray": [],
              "outputs": []}
    try:
        m.write_outputs(tmp / "dest_forged", payload, forged, lambda q: b"")
        cmsg2 = ""
    except m.Stop as x:
        cmsg2 = str(x)
    res["T2: the client re-filters a forged reply's tb (['SENTINEL text']): its STOP line shows ['?'], never the text"] = \
        "traceback ['?']" in cmsg2 and "SENTINEL" not in cmsg2
    r = run(entry={"attr": "main", "flags": dict(payload["entry"]["flags"], exit3=True)})
    n_exit = next(i for i, x in enumerate(RUNNER.split("\n"), 1) if "raise SystemExit(3)" in x)
    res["T8: a runner's SystemExit(3): exc_type SystemExit, rc 3, its tb ending at the raise line (fab/runner.py:<line>)"] = \
        r["status"] == "STOP" and r["exc_type"] == "SystemExit" and r["rc"] == 3 and r["exc_tb"][-1:] == [f"fab/runner.py:{n_exit}"]
    # git status and the knob
    sc = tmp / "statuscheck"
    git(tmp, "clone", "-q", str(repo), str(sc))
    os.utime(sc / "config" / "cfg.json", (2, 2))
    before = tree(sc / ".git")
    env0 = {**os.environ, **m.GIT_ENV}
    subprocess.run(["git", "status", "--porcelain"], cwd=sc, capture_output=True, check=True, env=env0)
    res["git status with the O14 knobs (GIT_OPTIONAL_LOCKS=0) after an mtime change: .git byte-identical"] = tree(sc / ".git") == before
    envc = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    subprocess.run(["git", "status", "--porcelain"], cwd=sc, capture_output=True, check=True, env=envc)
    res["control: the same status WITHOUT the knob rewrites .git/index (why the knob is required)"] = tree(sc / ".git") != before
    # staging negatives (each before the runner starts: no output, the runner never ran)
    def stop(name, why, **kw):
        r_ = run(**kw)
        ok = r_["status"] == "STOP" and r_.get("stop_reason") == why and r_["outputs"] == []
        res[f"NEGATIVE staging, {name}: STOP"] = ok
        if not ok:
            print("   got:", r_.get("stop_reason"), r_.get("exc_type"))

    bad = tmp / "rcman_bad"
    shutil.copytree(rcman, bad)
    (bad / staged_a.lstrip("/")).write_bytes(b'{"a": 9}')
    stop("a sha mismatch on the Volume", "a staged file differs from its INDEX sha256 or byte count", vol=bad)
    miss = tmp / "rcman_miss"
    shutil.copytree(rcman, miss)
    (miss / staged_a.lstrip("/")).unlink()
    stop("a file missing on the Volume", "a staged file is missing on the manifests Volume", vol=miss)
    ext = tmp / "rcman_extra"
    shutil.copytree(rcman, ext)
    (ext / f"{prefix}/simh/extra.json".lstrip("/")).write_text("{}")
    stop("an extra file on the Volume", "1 files on the manifests Volume are not in the INDEX", vol=ext)
    stop("a count != the pinned count", "the staging INDEX does not hold exactly n_files == the pinned count of files, or the prefix is not a plain absolute path",
         staging=dict(payload["staging"], count=4))
    stop("an INDEX that differs from its pin", "the staging INDEX differs from its pin", staging=dict(payload["staging"], index=dict(payload["staging"]["index"], sha256="0" * 64)))
    stop("an entry outside the prefix", "a staging INDEX entry is malformed, duplicated, or outside the allowed prefix",
         staging=dict(payload["staging"], prefix=str(tmp / "elsewhere")))
    r = run(entry={"attr": "main", "flags": dict(payload["entry"]["flags"], mutate_staged=True)})
    res["NEGATIVE the runner changes a staged manifest: STOP"] = r["status"] == "STOP" and r["stop_reason"] == "the runner changed the staged manifests" \
        and r["outputs"] == []
    r = run(expect_keys=["n_units", "spec_only"])
    res["NEGATIVE O19 the runner's JSON keys differ from the pinned ones: STOP, no output"] = r["status"] == "STOP" and r["rc"] == 0 \
        and r["stop_reason"] == m.CONFIRM and r["outputs"] == []
    stop("O20 a prefix that contains the work dir", "the staging prefix and the work dir overlap", staging=dict(payload["staging"], prefix=str(tmp)))
    r = run(entry={"attr": "main", "flags": dict(payload["entry"]["flags"], spec_only="@out:other/spec.pkl")})
    res["NEGATIVE an @out flag that is not an allowlisted output: STOP"] = r["status"] == "STOP" and r["stop_reason"] == "an @out flag is not an allowlisted output path"
    # the VolumeFS version pins (2e): (a) the source: OUT_VOL and RCMAN_VOL are opened with version=2, the bundle and stage Volumes
    # unchanged; (b) the library: modal 1.6.1's own check raises on a v1 Volume when v2 is requested, and passes on v2 (no Modal call)
    src = wrapper_src(worktree, wrapper_file).decode()
    calls = sorted(x.split(")")[0] for x in src.split("modal.Volume.from_name(")[1:])
    res["version pin (a): every OUT_VOL and RCMAN_VOL from_name passes version=2; BUNDLE_VOL and STAGE_VOL unchanged"] = calls == sorted(
        ["BUNDLE_VOL", "BUNDLE_VOL", "OUT_VOL, version=2", "OUT_VOL, version=2", "RCMAN_VOL, version=2", "STAGE_VOL"])
    lib = subprocess.run([MODAL_PY, "-I", "-B", "-c", (
        "import modal\nfrom modal.volume import _validate_volume_version as v\nfrom modal.exception import InvalidError\n"
        "from modal_proto import api_pb2\nE = api_pb2.VolumeFsVersion\nassert modal.__version__ == '1.6.1'\n"
        "try:\n    v(2, E.VOLUME_FS_VERSION_V1, 'x')\n    print('NO_RAISE')\nexcept InvalidError:\n    print('RAISED')\n"
        "v(2, E.VOLUME_FS_VERSION_V2, 'x')\nprint('V2_OK')")], capture_output=True, text=True)
    res["version pin (b): modal 1.6.1's _validate_volume_version raises InvalidError for v2 requested on a v1 Volume, and passes on v2"] = \
        lib.returncode == 0 and lib.stdout.split() == ["RAISED", "V2_OK"]
    for k, v in res.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("ALL PASS" if all(res.values()) else "FAILED")
    shutil.rmtree(tmp, ignore_errors=True)
    if not all(res.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worktree", action="store_true", help="test the working-tree wrapper instead of the HEAD blob")
    ap.add_argument("--wrapper-file", default=None, help="test this file instead (the mutation runs: the test must FAIL)")
    a = ap.parse_args()
    main(a.worktree, a.wrapper_file)
