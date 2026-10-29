"""Synthetic test of run_records_step_in_container.py: a throwaway git repo with a fabricated builder (fab/builder.py), its sibling
module, a config file and a fabricated record; the container body run_step() runs locally on a real bundle of that repo (no Modal).
No real record is read. Cases (each asserts its exact outcome):
- good: OK; the allowlisted output comes back with its sha; a child process's read-only git call works through the wrapper; the
  builder's stdout value line never reaches the reply (only the reducer's lines);
- a write verb from the builder (git update-ref), and from a CHILD process (git commit): refused, argv logged, STOP;
- an extra output file in results/, a file outside results/, and a file written straight into .git (no git verb): STOP, paths only;
- a missing sibling module (sys_path not given): STOP, ModuleNotFoundError only;
- a missing working-tree file read: STOP, FileNotFoundError only;
- a bundle sha mismatch; a bundle with an EXTRA commit, or MISSING a listed one (its heads are not exactly the listed set); C missing
  from the bundle with its heads matching the listing (argv-only reason); a wrong tree; a wrong script blob; an @volsha path outside
  the Volume: each STOP;
- client side: an unlisted script, a wrong blob, HEAD != C, a dirty tree, a memory_mib outside the allowed sizes: preflight STOPs
  (before any Modal call);
- write_outputs: exclusive create (an existing file), a sha mismatch, a non-allowlisted path: each STOPs;
- reducers: make_pilot_records keeps only publish lines; fix_v2 returns booleans / ints / shas only;
- refresh2's R1-R4 / O1-O5: a HEAD move (.git/HEAD written) and a nested .git name STOP; an existing output name STOPs in preflight;
  the strict whitelist refuses float / tuple / set / bytes / complex / int subclasses / number-like reducer strings (a digit-only
  sha passes); a reducer line naming a path or sha that is not a returned output STOPs; a symlinked output STOPs; fetched refs that
  differ from the listing STOP; the client checks the output count and every destination before its first write (all or nothing).
- refresh2's route C R1 / R2 and 2e's O18 / O19 / O23 (10-17): every blob pinned in the REAL allowlist exists in the repo (and the
  old typo does not); a verify_only step (inputs_full's indented JSON) passes with 0 outputs, and STOPs when verify exits 1, exits 0
  with problems, or writes a file; the client re-checks the pass; preflight refuses verify_only with outputs and n_outputs 0 without
  it; the inputs_full and json_line_keys reducers return counts / booleans / a bound sha only; a builder's GIT_CONFIG_PARAMETERS
  hook never runs (control: real git runs it); a child python without -B writes no __pycache__ (control: it does without the env).
- the reply-size fix (the fake outputs Volume is a temp dir; commit is a counter): OK publishes runs/<run_id>/<path> (sha, size), ONE
  commit, no "bytes" anywhere in the reply; a STOP step publishes nothing and never commits; a builder writing into the outputs Volume,
  an existing run dir and a malformed run_id STOP; refresh2's R1: the completion marker runs/<run_id>/_COMPLETE.json is written
  last (present at commit) with the outputs' shas, and an otherwise-OK step whose reply is too large STOPs before publishing
  anything; the client fetches by vol_path and a size / sha / vol_path mismatch on the second output STOPs before any write; a STOP reply over the limit becomes the minimal STOP; a 256 KiB payload STOPs in client() before any Modal call
  (a stub modal module is never touched); fix_v2 confirmation (fab/fixv2.py, reducer fix_v2): unparsed stdout or a missing part STOPs
  with the confirmation reason, an unbound sha STOPs at bind_reduced, a good one passes.
- the route C port (O24): an inputs_full BUILD step (not verify_only) passes only with its build JSON parsed and its sha256 bound to a
  returned output: an extra stdout line STOPs with the confirmation reason and publishes nothing.

  uv run --no-project --python 3.12 python experiments/2026-10-17_records_step_container/test_run_records_step.py [--worktree]
"""
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
REAL = HERE.parents[1]
REL = "experiments/2026-10-17_records_step_container"
FAB = "fab/builder.py"
BUILDER = r'''
import json, subprocess, sys, hashlib
from pathlib import Path
import helper  # a sibling module (needs sys_path)
REPO = Path(__file__).resolve().parents[1]
mode, c1 = sys.argv[1], sys.argv[2]
import os
if os.environ.get("GIT_OPTIONAL_LOCKS") != "0" or os.environ.get("GIT_CONFIG_KEY_0") != "gc.auto":  # O14: the hardened git env is set
    raise SystemExit(5)
rec = subprocess.run(["git", "show", f"{c1}:data/rec.json"], cwd=REPO, capture_output=True, check=True).stdout
cfg = json.loads((REPO / ("config/missing.json" if mode == "missing_read" else "config/cfg.json")).read_text())
child = "import subprocess, sys; subprocess.run(['git', *sys.argv[1:]], check=True, capture_output=True)"
argv = ["commit", "--allow-empty", "-m", "x"] if mode == "child_write" else ["rev-parse", c1]
subprocess.run([sys.executable, "-I", "-c", child, *argv], cwd=REPO, check=True)
if mode == "write_verb":
    subprocess.run(["git", "update-ref", "refs/heads/x", c1], cwd=REPO, check=True)
if mode == "fsmonitor_param":  # O18: a hook injected through GIT_CONFIG_PARAMETERS would write .git/fsm_ran
    hook = Path(os.environ["HOME"]) / "fsm.sh"
    hook.write_text(f"#!/bin/sh\ntouch {REPO / '.git' / 'fsm_ran'}\n")
    hook.chmod(0o755)
    subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO, check=True, capture_output=True,
                   env={**os.environ, "GIT_CONFIG_PARAMETERS": f"'core.fsmonitor'='{hook}'"})
if mode == "child_import":  # O23: a child python WITHOUT -B importing a repo module (cwd fab/ is on its sys.path)
    subprocess.run([sys.executable, "-c", "import helper"], cwd=REPO / "fab", check=True)
if mode == "vol_write":
    (Path(sys.argv[3]) / "planted.txt").write_text("x")
out = REPO / "results" / "out_fab.json"
out.parent.mkdir(exist_ok=True)
body = json.dumps({"n": helper.N, "cfg": cfg["k"], "rec_sha256": hashlib.sha256(rec).hexdigest()}).encode()
out.write_bytes(body)
if mode == "extra":
    (REPO / "results" / "extra.txt").write_text("x")
if mode == "outside":
    (REPO / "stray.txt").write_text("x")
if mode == "git_dir":
    (REPO / ".git" / "refs" / "heads" / "planted").write_text(c1 + "\n")
if mode == "head_move":
    (REPO / ".git" / "HEAD").write_text(c1 + "\n")
if mode == "nested_git":
    (REPO / "results" / ".git").mkdir()
    (REPO / "results" / ".git" / "x").write_text("x")
if mode == "symlink_out":
    out.unlink()
    (REPO / "config" / "target.json").write_bytes(body)
    out.symlink_to(REPO / "config" / "target.json")
if mode == "many_stray":
    for i in range(60):
        (REPO / "results" / f"s{i:02d}.txt").write_text("x")
    (REPO / "results" / ("L" * 220 + ".txt")).write_text("x")
if mode == "exc_forge":
    open(os.environ["RECSTEP_EXC"], "w").write(json.dumps({"type": "auroc 0.97 secret"}))
    os._exit(3)
if mode == "lying_line":
    print(f"results/out_fab.json written; sha256 {'0' * 64}")
print("auroc 0.97 (a measured value line that must never reach the reply)")
print(f"results/out_fab.json written; sha256 {hashlib.sha256(body).hexdigest()}")
'''
VERIFIER = r'''
import json, sys
from pathlib import Path
mode = sys.argv[1]
if mode == "writes":
    (Path(__file__).resolve().parents[1] / "results").mkdir(exist_ok=True)
    (Path(__file__).resolve().parents[1] / "results" / "v.json").write_text("{}")
if mode.startswith("build"):  # inputs_full build's form (L211): one written file, then its indented JSON
    import hashlib
    body = b'{"fabricated": "inputs_full"}'
    (Path(__file__).resolve().parents[1] / "results").mkdir(exist_ok=True)
    (Path(__file__).resolve().parents[1] / "results" / "IF.json").write_bytes(body)
    if mode == "build_extra":
        print("an extra stdout line")
    print(json.dumps({"file": "results/IF.json", "sha256": hashlib.sha256(body).hexdigest(), "pending": [], "problems": []}, indent=1))
    raise SystemExit(0)
pr = [] if mode in ("pass", "writes") else ["a problem text quoting auroc 0.97"]
print(json.dumps({"file": "config/cfg.json", "problems": pr, "pass": not pr}, indent=1))  # inputs_full verify's form (L944)
if pr and mode != "fail_rc0":
    raise SystemExit(1)
'''
FIX = "fab/fixv2.py"  # a fabricated fix_v2-like builder: two outputs and a final JSON for reduce_fix_v2
FIXB = r'''
import hashlib, json, sys
from pathlib import Path
REPO, mode, h = Path(__file__).resolve().parents[1], sys.argv[1], {}
(REPO / "results").mkdir(exist_ok=True)
for k, body in (("pilot_rows", b'{"rows": "fabricated"}'), ("fix_v2", b'{"rows": "fabricated too"}')):
    (REPO / "results" / f"fx_{k}.json").write_bytes(body)
    h[k] = {"file": f"fx_{k}.json", "sha256": hashlib.sha256(body).hexdigest(), "n": 1}
if mode == "unbound":
    h["fix_v2"]["sha256"] = "0" * 64
if mode == "missing":
    del h["fix_v2"]
print("not json" if mode == "unparsed" else json.dumps({"status": "COMPLETE", "dry": False, **h, "stops": [], "post_write_checks": []}))
'''


def git(repo, *a):
    return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True).stdout.strip()


def load(worktree):
    src = (HERE / "run_records_step_in_container.py").read_bytes() if worktree else \
        subprocess.run(["git", "-C", str(REAL), "show", f"HEAD:{REL}/run_records_step_in_container.py"], capture_output=True, check=True).stdout
    d = Path(tempfile.mkdtemp(prefix="recstep_mod_"))
    (d / "w.py").write_bytes(src)
    spec = importlib.util.spec_from_file_location("recstep_under_test", d / "w.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main(worktree=False):
    m = load(worktree)

    def stop_of(f, *a):  # -> the Stop text f(*a) raises, or None
        try:
            f(*a)
        except m.Stop as x:
            return str(x)
        return None

    kind = lambda b: subprocess.run(["git", "-C", str(REAL), "cat-file", "-t", b], capture_output=True, text=True).stdout.strip()  # noqa: E731
    pinned = {p: kind(b) for p, (b, _) in m.SCRIPTS.items() if b is not None}  # R1: the REAL allowlist, before the test replaces it
    typo = kind("cf66c533d09d8c2e514a0619bf06feb159e50eb8")  # a603bc874's mistyped inputs_full pin (the control)
    tmp = Path(tempfile.mkdtemp(prefix="recstep_test_"))
    repo = tmp / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    for k, v in (("user.email", "test@example.invalid"), ("user.name", "test"), ("commit.gpgsign", "false")):
        git(repo, "config", k, v)
    (repo / "fab").mkdir()
    (repo / FAB).write_text(BUILDER)
    (repo / "fab" / "verifier.py").write_text(VERIFIER)
    (repo / FIX).write_text(FIXB)
    (repo / "fab" / "helper.py").write_text("N = 20\n")
    (repo / "config").mkdir()
    (repo / "config" / "cfg.json").write_text(json.dumps({"k": "v"}))
    (repo / "data").mkdir()
    (repo / "data" / "rec.json").write_text(json.dumps({"fabricated": True}))
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "c1")
    c1 = git(repo, "rev-parse", "HEAD")
    spec = {"script": FAB, "reducer": "make_pilot_records", "args": ["good", c1], "sys_path": ["fab"], "commits": [c1], "memory_mib": 1024, "n_outputs": 1,
            "outputs": [{"dir": "results", "regex": r"out_fab\.json"}]}
    vspec = {"script": "fab/verifier.py", "reducer": "inputs_full", "args": ["pass"], "commits": [c1], "memory_mib": 4096, "n_outputs": 0,
             "outputs": [], "verify_only": True}
    (repo / "steps").mkdir()
    (repo / "steps" / "fab.json").write_text(json.dumps(spec))
    (repo / "steps" / "verify.json").write_text(json.dumps(vspec))
    (repo / "steps" / "verify_out.json").write_text(json.dumps(vspec | {"outputs": [{"dir": "results", "regex": r"v\.json"}]}))
    (repo / "steps" / "zero_noverify.json").write_text(json.dumps(spec | {"n_outputs": 0}))
    (repo / "steps" / "build.json").write_text(json.dumps(vspec | {"args": ["build"], "verify_only": False, "n_outputs": 1,
                                                                   "outputs": [{"dir": "results", "regex": r"IF\.json"}]}))
    (repo / "steps" / "fix.json").write_text(json.dumps(spec | {"script": FIX, "reducer": "fix_v2", "args": ["good"], "sys_path": [], "memory_mib": 4096,
                                                                "n_outputs": 2, "outputs": [{"dir": "results", "regex": r"fx_(pilot_rows|fix_v2)\.json"}]}))
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "step spec")
    C = git(repo, "rev-parse", "HEAD")
    m.SCRIPTS = {FAB: (git(repo, "rev-parse", f"{C}:{FAB}"), "make_pilot_records"), FIX: (git(repo, "rev-parse", f"{C}:{FIX}"), "fix_v2"),
                 "fab/verifier.py": (git(repo, "rev-parse", f"{C}:fab/verifier.py"), "inputs_full")}
    payload, _ = m.preflight(repo, "steps/fab.json", C)
    payload_v, _ = m.preflight(repo, "steps/verify.json", C)
    payload_b, _ = m.preflight(repo, "steps/build.json", C)
    fpay, _ = m.preflight(repo, "steps/fix.json", C)
    broot = tmp / "bundlevol"
    (broot / "bundles").mkdir(parents=True)
    bt = tmp / "bt"
    bt.mkdir()
    f, bsha = m.make_bundle(repo, payload["commits"], bt)
    shutil.copy(f, broot / "bundles" / f"{bsha}.bundle")
    vol = tmp / "stagevol"
    (vol / "d").mkdir(parents=True)
    (vol / "d" / "stage.json").write_text("{}")
    (tmp / "outside.json").write_text("{}")
    real_git = shutil.which("git")
    outv = tmp / "outvol"  # the fake outputs Volume
    outv.mkdir()
    commits = []  # commit() calls of the LAST run
    marked = []  # R1: whether runs/<run_id>/_COMPLETE.json already existed at each commit() call
    res = {}

    def run(out_root=outv, base=None, **kw):
        p = dict(base or payload, bundle=f"bundles/{bsha}.bundle", bundle_sha256=bsha, run_id=uuid.uuid4().hex) | kw
        w = Path(tempfile.mkdtemp(prefix="recstep_work_", dir=tmp))
        commits.clear()
        marked.clear()
        return m.run_step(p, w, broot, vol, sys.executable, real_git, out_root, lambda: commits.append(p["run_id"])
                          or marked.append((out_root / "runs" / str(p["run_id"]) / m.MARKER).is_file()))

    # R1: every blob pinned in the real SCRIPTS exists in the repo as a blob; the old typo is no object (so this check catches one)
    res["R1 every pinned SCRIPTS blob exists in the repo (git cat-file -t == blob)"] = len(pinned) == 3 and set(pinned.values()) == {"blob"}
    res["R1 control: a603bc874's mistyped inputs_full pin is not an object"] = typo == ""
    if set(pinned.values()) != {"blob"}:
        print("   missing:", sorted(p for p, k in pinned.items() if k != "blob"))
    # R2: a verify-only step (inputs_full's indented JSON form)
    r = run(base=payload_v)
    res["R2 verify_only, a passing verify: OK with 0 outputs; the reducer relays only {pass, n_problems}"] = \
        r["status"] == "OK" and r["outputs"] == [] and r["reduced"] == {"pass": True, "n_problems": 0}
    res["R2 the client writes nothing for a passing verify (0 == n_outputs 0)"] = m.write_outputs(tmp / "dest_verify", payload_v, r, lambda p: b"") == [] \
        and not (tmp / "dest_verify").exists()
    r = run(base=payload_v, args=["fail"])
    res["NEGATIVE R2 a failing verify (exit 1): STOP; n_problems only, no problem text"] = r["status"] == "STOP" and r["rc"] == 1 \
        and r["reduced"] == {"pass": False, "n_problems": 1} and r["outputs"] == [] and "0.97" not in json.dumps(r)
    r = run(base=payload_v, args=["fail_rc0"])
    res["NEGATIVE R2 a verify that exits 0 but reports problems: STOP (the reducer's pass gates the step)"] = r["status"] == "STOP" and r["rc"] == 0 \
        and r["stop_reason"] == m.CONFIRM
    r = run(base=payload_v, args=["writes"])
    res["NEGATIVE R2 a verify that writes a file: STOP, the path only"] = r["status"] == "STOP" and r["stray"] == ["results/v.json"]
    forged = {"status": "OK", "refused_git": [], "stray": [], "outputs": [], "reduced": {"pass": False, "n_problems": 1}}
    res["NEGATIVE R2 the client re-checks the pass: a forged OK reply without pass STOPs"] = \
        stop_of(m.write_outputs, tmp / "dest_forged", payload_v, forged, lambda p: b"") == "a verify_only step did not report pass"
    out_stop = "the step's commit set must be full 40-hex shas, its outputs a list of {dir, regex}, and n_outputs an int >= 1 " \
               "(R2: verify_only true needs outputs [], n_outputs 0 and a reducer that reports pass)"
    res["NEGATIVE R2 preflight: verify_only with an output pattern STOPs"] = stop_of(m.preflight, repo, "steps/verify_out.json", C) == out_stop
    res["NEGATIVE R2 preflight: n_outputs 0 without verify_only STOPs"] = stop_of(m.preflight, repo, "steps/zero_noverify.json", C) == out_stop
    # O24: an inputs_full BUILD (not verify_only) is OK only with its build JSON parsed and its sha bound to a returned output
    rid = uuid.uuid4().hex
    r = run(base=payload_b, run_id=rid)
    res["O24 an inputs_full build: OK, its parsed sha256 bound to the one output, published with ONE commit"] = r["status"] == "OK" \
        and [o["path"] for o in r["outputs"]] == ["results/IF.json"] and r["reduced"]["sha256"] == r["outputs"][0]["sha256"] and commits == [rid]
    rid = uuid.uuid4().hex
    r = run(base=payload_b, args=["build_extra"], run_id=rid)
    res["NEGATIVE O24 an inputs_full build with an extra stdout line (unparsed): STOP with the confirmation reason, nothing published"] = \
        r["status"] == "STOP" and r["rc"] == 0 and r["reduced"] == {"parsed": False} and r["stop_reason"] == m.CONFIRM and r["outputs"] == [] \
        and commits == [] and not (outv / "runs" / rid).exists()
    # O18 / O23
    r = run(args=["fsmonitor_param", c1])
    res["O18 a builder's GIT_CONFIG_PARAMETERS hook never runs (inherited GIT_* dropped): OK, .git unchanged"] = \
        r["status"] == "OK" and r["stray"] == []
    ctl = tmp / "fsmctl"
    git(tmp, "clone", "-q", str(repo), str(ctl))
    (tmp / "fsm.sh").write_text(f"#!/bin/sh\ntouch {ctl / '.git' / 'fsm_ran'}\n")
    (tmp / "fsm.sh").chmod(0o755)
    subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ctl, capture_output=True,
                   env={**os.environ, **m.GIT_ENV, "GIT_CONFIG_PARAMETERS": f"'core.fsmonitor'='{tmp / 'fsm.sh'}'"})
    res["O18 control: real git with the forced knobs PLUS that GIT_CONFIG_PARAMETERS runs the hook (why the drop is needed)"] = \
        (ctl / ".git" / "fsm_ran").exists()
    r = run(args=["child_import", c1])
    res["O23 a child python without -B importing a repo module writes no __pycache__: OK"] = r["status"] == "OK" and r["stray"] == []
    subprocess.run([sys.executable, "-c", "import helper"], cwd=ctl / "fab", check=True,
                   env={k: v for k, v in os.environ.items() if k != "PYTHONDONTWRITEBYTECODE"})
    res["O23 control: the same child import without the env writes fab/__pycache__ (why the env is needed)"] = (ctl / "fab" / "__pycache__").is_dir()

    rid = uuid.uuid4().hex
    r = run(run_id=rid)
    good = r["status"] == "OK" and [o["path"] for o in r["outputs"]] == ["results/out_fab.json"] and r["refused_git"] == [] and r["stray"] == [] \
        and r["reduced"]["written"] == [{"path": "results/out_fab.json", "sha256": r["outputs"][0]["sha256"]}]
    res["good: OK, the allowlisted output with its sha, a child process's read-only git call through the wrapper, the reducer's line only"] = good
    blob = json.dumps(r)  # no bytes anywhere: json.dumps would raise on a bytes value
    res["good: the builder's stdout value line never reaches the reply"] = "auroc" not in blob and "0.97" not in blob
    o, landed = (r["outputs"] or [{}])[0], outv / "runs" / rid / "results" / "out_fab.json"
    res["good: the output lands at runs/<run_id>/<path> on the outputs Volume (sha, size), ONE commit, no 'bytes' key anywhere in the reply"] = \
        set(o) == {"path", "sha256", "size", "vol_path"} and o["vol_path"] == f"runs/{rid}/results/out_fab.json" and landed.is_file() \
        and m.sha(landed.read_bytes()) == o["sha256"] and landed.stat().st_size == o["size"] and commits == [rid] and '"bytes"' not in blob
    mk = outv / "runs" / rid / m.MARKER
    res["R1 the completion marker runs/<run_id>/_COMPLETE.json: written LAST (present at commit, newest mtime), its run_id and the output's "
        "path / sha256 / size, and not in the reply's outputs"] = mk.is_file() and marked == [True] \
        and json.loads(mk.read_bytes()) == {"run_id": rid, "outputs": [{"path": o.get("path"), "sha256": o.get("sha256"), "size": o.get("size")}]} \
        and mk.stat().st_mtime_ns >= landed.stat().st_mtime_ns and m.MARKER not in blob
    rid = uuid.uuid4().hex
    r = run(args=["extra", c1], run_id=rid)
    res["NEGATIVE a STOP step (a stray file) publishes nothing: no run dir, no commit, outputs []"] = \
        r["status"] == "STOP" and not (outv / "runs" / rid).exists() and commits == [] and r["outputs"] == []
    ow = tmp / "outvol_w"
    ow.mkdir()
    r = run(out_root=ow, args=["vol_write", c1, str(ow)])
    res["NEGATIVE the builder writes into the outputs Volume: STOP, no run dir, no commit"] = r["status"] == "STOP" and r["rc"] == 0 \
        and r["stop_reason"] == "the builder changed the outputs Volume" and r["outputs"] == [] and commits == [] and not (ow / "runs").exists()
    rid = uuid.uuid4().hex
    (outv / "runs" / rid).mkdir()
    r = run(run_id=rid)
    res["NEGATIVE an existing run dir: STOP (exception TYPE only), no commit, nothing written into it"] = r["status"] == "STOP" and r["rc"] == 0 \
        and r["exc_type"] == "FileExistsError" and r["outputs"] == [] and commits == [] and list((outv / "runs" / rid).iterdir()) == []
    bad_ids = True
    for x in ("A" * 32, "0" * 31, "../" + "0" * 29, None):
        r = run(run_id=x)
        bad_ids &= r["status"] == "STOP" and r["stop_reason"] == "the run_id is not 32 lowercase hex" and r["outputs"] == [] and commits == []
    res["NEGATIVE a malformed run_id (uppercase, 31 hex, '../', None): STOP, no commit"] = bad_ids
    saved = m.MAX_INLINE
    m.MAX_INLINE = 200  # route C relays a refused argv as [verb, n, sha] (O13), so no builder makes a 256 KiB reply: lower the limit
    try:
        r = run(args=["extra", c1])  # a STOP step whose reply is now "too large"
    finally:
        m.MAX_INLINE = saved
    res["NEGATIVE a STOP reply at or over the limit (the post-try guard): the minimal STOP reply, no commit"] = commits == [] and r == {
        "status": "STOP", "exc_type": None, "stop_reason": "the reply is too large to return inline", "refused_git": [], "stray": [], "outputs": [],
        "reduced": None, "resolved": {}, "rc": None}
    rid, saved = uuid.uuid4().hex, m.MAX_INLINE
    m.MAX_INLINE = m.ENVELOPE + 64  # any OK reply is now "too large": the guard must fire BEFORE anything is published
    try:
        r = run(run_id=rid)
    finally:
        m.MAX_INLINE = saved
    res["NEGATIVE R1 an otherwise-OK step whose reply is too large: STOP before publishing (no commit, no runs/<run_id>/, outputs [])"] = \
        r["status"] == "STOP" and r["stop_reason"] == "the reply is too large to return inline" and commits == [] and r["outputs"] == [] \
        and not (outv / "runs" / rid).exists()
    conf = m.CONFIRM
    rid = uuid.uuid4().hex
    r = run(base=fpay, run_id=rid)
    res["fix_v2 confirmation: parsed, both shas bound: OK, both outputs published, ONE commit"] = r["status"] == "OK" and r["stop_reason"] is None \
        and sorted(o["vol_path"] for o in r["outputs"]) == [f"runs/{rid}/results/fx_{k}.json" for k in ("fix_v2", "pilot_rows")] and commits == [rid] \
        and r["reduced"]["parsed"] is True and {r["reduced"][k]["sha256"] for k in ("pilot_rows", "fix_v2")} == {o["sha256"] for o in r["outputs"]} \
        and marked == [True] and sorted(x["path"] for x in json.loads((outv / "runs" / rid / m.MARKER).read_bytes())["outputs"]) \
        == [f"results/fx_{k}.json" for k in ("fix_v2", "pilot_rows")]
    for mode in ("unparsed", "missing"):
        rid = uuid.uuid4().hex
        r = run(base=fpay, args=[mode], run_id=rid)
        res[f"NEGATIVE fix_v2 confirmation, {mode} ({'stdout not JSON' if mode == 'unparsed' else 'parsed, no fix_v2 part'}): STOP, rc / exc_type kept, "
            "nothing published"] = r["status"] == "STOP" and r["stop_reason"] == conf and r["rc"] == 0 and r["exc_type"] is None and r["outputs"] == [] \
            and commits == [] and not (outv / "runs" / rid).exists()
    r = run(base=fpay, args=["unbound"])
    res["NEGATIVE fix_v2 confirmation, a parsed sha bound to no output: STOP (at bind_reduced, which runs first), nothing published"] = \
        r["status"] == "STOP" and r["exc_type"] == "Stop" and r["stop_reason"] == "a reducer sha is not one of the returned outputs" and r["outputs"] == [] \
        and commits == []
    r = run(args=["write_verb", c1])
    res["NEGATIVE a write verb from the builder (git update-ref): refused, argv logged, STOP"] = \
        r["status"] == "STOP" and r["refused_git"] == [["update-ref", 3, m.sha(json.dumps(["update-ref", "refs/heads/x", c1]).encode())]] and r["outputs"] == []
    r = run(args=["child_write", c1])
    res["NEGATIVE a write verb from a CHILD process (git commit): refused, argv logged, STOP"] = \
        r["status"] == "STOP" and r["refused_git"] == [["commit", 4, m.sha(json.dumps(["commit", "--allow-empty", "-m", "x"]).encode())]]
    r = run(args=["extra", c1])
    res["NEGATIVE an extra output file in results/: STOP, the path only; O16 no output bytes in a STOP reply"] = \
        r["status"] == "STOP" and r["stray"] == ["results/extra.txt"] and r["outputs"] == []
    r = run(args=["outside", c1])
    res["NEGATIVE a file outside results/: STOP, the path only"] = r["status"] == "STOP" and r["stray"] == ["stray.txt"]
    r = run(args=["git_dir", c1])
    res["NEGATIVE a file written straight into .git (refs, no git verb): STOP, the path only"] = \
        r["status"] == "STOP" and r["stray"] == [".git/refs/heads/planted"]
    r = run(args=["head_move", c1])
    res["NEGATIVE R1 a HEAD move (.git/HEAD written directly): STOP, the path only"] = r["status"] == "STOP" and r["stray"] == [".git/HEAD"]
    r = run(args=["nested_git", c1])
    res["NEGATIVE R1 a nested .git name (results/.git/x): STOP, the path only"] = r["status"] == "STOP" and r["stray"] == ["results/.git/x"]
    r = run(args=["symlink_out", c1])
    res["NEGATIVE O3 a symlinked output: STOP"] = r["status"] == "STOP" and r["stop_reason"] == "an output is a symlink" and r["outputs"] == []
    r = run(args=["lying_line", c1])
    res["NEGATIVE O2 a reducer line whose sha is not the returned output's: STOP"] = \
        r["status"] == "STOP" and r["stop_reason"] == "a reducer path is too long or not one of the returned outputs (by path and sha)" and r["outputs"] == []
    stub = tmp / "gitstub"
    stub.write_text(f"#!/bin/bash\nif [ \"$1\" = for-each-ref ]; then {real_git} \"$@\"; echo \"{'0' * 40} refs/recstep/extra\"; else exec {real_git} \"$@\"; fi\n")
    stub.chmod(0o755)
    w = Path(tempfile.mkdtemp(prefix="recstep_work_", dir=tmp))
    r = m.run_step(dict(payload, bundle=f"bundles/{bsha}.bundle", bundle_sha256=bsha, run_id=uuid.uuid4().hex), w, broot, vol, sys.executable, str(stub),
                   outv, lambda: commits.append("stub"))
    res["NEGATIVE O4 fetched refs that differ from the listing: STOP"] = r["status"] == "STOP" and r["stop_reason"] == "the fetched refs are not exactly the listed commits"
    r = run(args=["many_stray", c1])
    res["O13 stray paths capped: 50 paths of at most 200 characters, then a count line"] = r["status"] == "STOP" and len(r["stray"]) == 51 \
        and r["stray"][-1] == "… 11 more" and all(len(x) <= 201 for x in r["stray"])
    r = run(args=["exc_forge", c1])
    res["O13 a forged exception text becomes 'unknown' (a type name only ever returns)"] = r["status"] == "STOP" and r["exc_type"] == "unknown" \
        and "0.97" not in json.dumps(r)
    r = run(sys_path=[])
    res["NEGATIVE a missing sibling module: STOP, ModuleNotFoundError only"] = r["status"] == "STOP" and r["exc_type"] == "ModuleNotFoundError"
    r = run(args=["missing_read", c1])
    res["NEGATIVE a missing working-tree file read: STOP, FileNotFoundError only"] = r["status"] == "STOP" and r["exc_type"] == "FileNotFoundError"
    r = run(bundle_sha256="0" * 64)
    res["NEGATIVE a bundle sha mismatch: STOP"] = r["status"] == "STOP" and r["stop_reason"] == "the bundle is missing, outside its Volume, or differs from its sha256"
    heads_stop = "the bundle's heads are not exactly the listed commits"
    extra = git(repo, "commit-tree", f"{C}^{{tree}}", "-p", C, "-m", "extra")  # an object only; HEAD stays C
    xb = tmp / "xb"
    xb.mkdir()
    fx, xsha = m.make_bundle(repo, payload["commits"] + [extra], xb)
    shutil.copy(fx, broot / "bundles" / f"{xsha}.bundle")
    r = run(bundle=f"bundles/{xsha}.bundle", bundle_sha256=xsha)
    res["NEGATIVE a bundle with an EXTRA commit beyond the listed set: STOP"] = r["status"] == "STOP" and r["stop_reason"] == heads_stop
    r = run(commits=payload["commits"] + [extra])
    res["NEGATIVE a bundle MISSING a listed commit: STOP"] = r["status"] == "STOP" and r["stop_reason"] == heads_stop
    missing = "1" * 40
    mb = tmp / "mb"
    mb.mkdir()
    # C itself absent although the heads match the listing: list a commit set without C and check out C
    fm, msha = m.make_bundle(repo, [c1], mb)
    shutil.copy(fm, broot / "bundles" / f"{msha}.bundle")
    r = run(bundle=f"bundles/{msha}.bundle", bundle_sha256=msha, commits=[c1], commit=missing)
    res["NEGATIVE C missing from the bundle (heads match the listing): STOP with the argv only"] = \
        r["status"] == "STOP" and r["stop_reason"] == f"git checkout -q --detach {missing}: exit 128"
    r = run(tree="0" * 40)
    res["NEGATIVE a wrong tree: STOP"] = r["status"] == "STOP" and r["stop_reason"] == "C, its tree, or the script blob differs from the pins"
    r = run(blob="0" * 40)
    res["NEGATIVE a wrong script blob: STOP"] = r["status"] == "STOP" and r["stop_reason"] == "C, its tree, or the script blob differs from the pins"
    r = run(args=["good", c1, "@volsha:../outside.json"])
    res["NEGATIVE an @volsha path outside the Volume: STOP"] = r["status"] == "STOP" and r["stop_reason"] == "an @volsha path is outside the mounted Volume or missing"
    r = run(args=["good", c1, "@volsha:d/stage.json"])
    res["@volsha inside the Volume resolves to the file's sha256 (the builder ignores the extra arg)"] = \
        r["status"] == "OK" and r["resolved"] == {"@volsha:d/stage.json": m.sha(b"{}")}
    # client side
    for name, kw, why in (("an unlisted script", {"SCRIPTS": {}}, "the step's script is not on the allowlist"),
                          ("a wrong blob", {"SCRIPTS": {FAB: ("0" * 40, "make_pilot_records")}}, "the script's blob at C (or its reducer) is not the allowlisted one")):
        saved = m.SCRIPTS
        m.SCRIPTS = kw["SCRIPTS"]
        try:
            m.preflight(repo, "steps/fab.json", C)
            e = None
        except m.Stop as x:
            e = str(x)
        m.SCRIPTS = saved
        res[f"NEGATIVE client preflight, {name}: STOP before any Modal call"] = e == why
    try:
        m.preflight(repo, "steps/fab.json", c1)
        e = None
    except m.Stop as x:
        e = str(x)
    res["NEGATIVE client preflight, HEAD != C: STOP"] = e == "HEAD is not C, or the tracked tree is dirty"
    (repo / "steps" / "fab_mem.json").write_text(json.dumps(spec | {"memory_mib": 2048}))
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "a spec with a bad memory size")
    Cm = git(repo, "rev-parse", "HEAD")
    try:
        m.preflight(repo, "steps/fab_mem.json", Cm)
        e = None
    except m.Stop as x:
        e = str(x)
    res["NEGATIVE client preflight, a memory_mib outside (1024, 4096, 8192): STOP"] = e == "the step's memory_mib must be one of (1024, 4096, 8192)"
    git(repo, "reset", "-q", "--hard", C)
    (repo / "steps" / "fab_big.json").write_text(json.dumps(spec | {"args": ["good", c1, "x" * 300_000]}))
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "a spec whose payload is 256 KiB or more")
    Cb = git(repo, "rev-parse", "HEAD")
    touched = []

    class StubModal(types.ModuleType):  # client() imports modal; any use of it (the first Modal call) is recorded
        def __getattr__(self, k):
            touched.append(k)
            raise AttributeError(k)

    saved_modal, cwd = sys.modules.get("modal"), os.getcwd()
    sys.modules["modal"] = StubModal("modal")
    os.chdir(repo)
    try:
        m.client("steps/fab_big.json", Cb)
        e = None
    except m.Stop as x:
        e = str(x)
    finally:
        os.chdir(cwd)
        sys.modules.pop("modal")
        if saved_modal is not None:
            sys.modules["modal"] = saved_modal
    git(repo, "reset", "-q", "--hard", C)
    res["NEGATIVE client: a payload of 256 KiB or more STOPs before any Modal call (the stub modal module is never touched)"] = \
        e == "the payload is too large to send inline" and touched == []
    (repo / "results").mkdir(exist_ok=True)
    (repo / "results" / "out_fab.json").write_text("{}")  # untracked, with an output name
    try:
        m.preflight(repo, "steps/fab.json", C)
        e = None
    except m.Stop as x:
        e = str(x)
    (repo / "results" / "out_fab.json").unlink()
    res["NEGATIVE R4 an existing file with an output name (untracked): STOP in preflight, before any Modal call"] = \
        e == "results: a file matching the step's output pattern already exists"
    (repo / "config" / "cfg.json").write_text("dirty")
    try:
        m.preflight(repo, "steps/fab.json", C)
        e = None
    except m.Stop as x:
        e = str(x)
    git(repo, "checkout", "-q", "--", "config/cfg.json")
    res["NEGATIVE client preflight, a dirty tracked tree: STOP"] = e == "HEAD is not C, or the tracked tree is dirty"
    grid = uuid.uuid4().hex
    good_reply, gp = run(run_id=grid), dict(payload, run_id=grid)
    fetched = []

    def fetch(p):  # the fake outputs Volume: bytes only
        fetched.append(p)
        return (outv / p).read_bytes()

    dest_repo = tmp / "dest"
    dest_repo.mkdir()
    done = m.write_outputs(dest_repo, gp, good_reply, fetch)
    res["client write_outputs: fetched by vol_path from the outputs Volume, exclusive create at the same path, re-hashed"] = \
        done == [("results/out_fab.json", good_reply["outputs"][0]["sha256"])] and fetched == [f"runs/{grid}/results/out_fab.json"] \
        and (dest_repo / "results" / "out_fab.json").read_bytes() == (outv / fetched[0]).read_bytes()
    try:
        m.write_outputs(dest_repo, gp, good_reply, fetch)
        e = None
    except m.Stop as x:
        e = str(x)
    res["NEGATIVE client write_outputs: an existing destination STOPs before writing (open('xb') stays the last guard)"] = \
        e == "results/out_fab.json: the destination already exists; nothing written"
    bad = dict(good_reply, outputs=[good_reply["outputs"][0] | {"path": "results/other.json"}])
    try:
        m.write_outputs(tmp / "dest_other", gp, bad, fetch)
        e = None
    except m.Stop as x:
        e = str(x)
    res["NEGATIVE client write_outputs, a non-allowlisted path: STOP"] = e == "results/other.json: not an allowlisted output, or its vol_path, size or sha256 is malformed"
    two = dict(gp, n_outputs=2, outputs=[{"dir": "results", "regex": r"out_fab2?\.json"}])
    o1 = good_reply["outputs"][0]
    (outv / "runs" / grid / "results" / "out_fab2.json").write_bytes(b"two")
    o2 = {"path": "results/out_fab2.json", "sha256": m.sha(b"two"), "size": 3, "vol_path": f"runs/{grid}/results/out_fab2.json"}
    malformed = "results/out_fab2.json: not an allowlisted output, or its vol_path, size or sha256 is malformed"
    differ = "results/out_fab2.json: the fetched bytes differ from the returned size or sha256; nothing written"
    for name, mut, why, n_fetched in (("size", {"size": 4}, differ, 2), ("sha", {"sha256": "0" * 64}, differ, 2),
                                      ("vol_path", {"vol_path": f"runs/{'0' * 32}/results/out_fab2.json"}, malformed, 0)):
        d = tmp / f"dest_{name}"
        fetched.clear()
        try:
            m.write_outputs(d, two, dict(good_reply, outputs=[o1, o2 | mut]), fetch)
            e = None
        except m.Stop as x:
            e = str(x)
        res[f"NEGATIVE client fetch, a {name} mismatch on the SECOND output: STOP before any write ({n_fetched} fetched)"] = \
            e == why and not d.exists() and len(fetched) == n_fetched
    d2 = tmp / "dest_two"
    (d2 / "results").mkdir(parents=True)
    (d2 / "results" / "out_fab2.json").write_text("exists")
    try:
        m.write_outputs(d2, two, dict(good_reply, outputs=[o1, o2]), fetch)
        e = None
    except m.Stop as x:
        e = str(x)
    res["NEGATIVE O5 the second destination exists: STOP before the FIRST write (nothing written)"] = \
        e == "results/out_fab2.json: the destination already exists; nothing written" and not (d2 / "results" / "out_fab.json").exists()
    try:
        m.write_outputs(tmp / "dest_cnt", two, good_reply, fetch)
        e = None
    except m.Stop as x:
        e = str(x)
    res["NEGATIVE O5 fewer outputs than n_outputs: STOP before any write"] = e == "1 outputs returned, the step expects exactly 2" and not (tmp / "dest_cnt").exists()
    # reducers
    res["reducer make_pilot_records: only its publish lines"] = m.reduce_make_pilot_records(
        "x 0.5\nexperiments/a/b.json written; sha256 " + "a" * 64 + "\nanything else\n") == {"written": [{"path": "experiments/a/b.json", "sha256": "a" * 64}]}
    fx = m.reduce_fix_v2(json.dumps({"status": "COMPLETE", "dry": False, "pilot_rows": {"file": "p.json", "sha256": "b" * 64, "n": 20},
                                     "fix_v2": {"file": "f.json", "sha256": "c" * 64, "n": 4353}, "stops": ["quotes FINAL 0.123"], "post_write_checks": []}))
    res["reducer fix_v2: booleans, ints and shas only (no names, no stop text)"] = fx == {
        "parsed": True, "complete": True, "dry": False, "pilot_rows": {"sha256": "b" * 64, "n": 20}, "fix_v2": {"sha256": "c" * 64, "n": 4353},
        "n_stops": 1, "n_post_write_checks": 0} and "FINAL" not in json.dumps(fx)
    ib = m.reduce_inputs_full(json.dumps({"file": "config/INPUTS_FULL_x.json", "sha256": "e" * 64, "pending": ["p quoting 0.5"], "problems": []}, indent=1))
    res["reducer inputs_full, build form (L211): {sha256, n_pending, n_problems} only"] = ib == {"sha256": "e" * 64, "n_pending": 1, "n_problems": 0}
    res["NEGATIVE reducer inputs_full: an extra stdout line leaves it unparsed (a verify_only step then STOPs)"] = \
        m.reduce_inputs_full("x\n" + json.dumps({"file": "f", "problems": [], "pass": True}, indent=1)) == {"parsed": False}
    res["NEGATIVE reducer inputs_full, build: a printed sha that is not a returned output STOPs"] = \
        stop_of(m.bind_reduced, ib, []) == "a reducer sha is not one of the returned outputs"
    res["reducer json_line_keys (O19): keys_match against the pinned list, counts only"] = \
        m.reduce_json_line_keys('a 0.5\n{"b": 1, "a": 2}\n', ["a", "b"]) == {"json_lines": 1, "keys_match": True, "n_keys": 2} \
        and m.reduce_json_line_keys('{"a": 1}\n', ["a", "b"]) == {"json_lines": 1, "keys_match": False, "n_keys": 1}
    class NPInt(int):
        pass

    for name, bad, why in (("a float", {"a": [1, {"b": 0.5}]}, "a relayed field has a disallowed type (float)"),
                           ("a tuple", {"a": (1, 2)}, "a relayed field has a disallowed type (tuple)"),
                           ("a set", {"a": {1}}, "a relayed field has a disallowed type (set)"),
                           ("bytes", {"a": b"x"}, "a relayed field has a disallowed type (bytes)"),
                           ("a complex", {"a": 1j}, "a relayed field has a disallowed type (complex)"),
                           ("an int subclass (a numpy-like scalar)", {"a": NPInt(3)}, "a relayed field has a disallowed type (NPInt)"),
                           ("a non-str dict key", {1: "a"}, "a relayed field has a disallowed type (dict)")):
        try:
            m._no_float(bad)
            e = None
        except m.Stop as x:
            e = str(x)
        res[f"NEGATIVE O1 the whitelist refuses {name}"] = e == why
    for v in ("0.123", "1e-3", "-7", "nan"):
        try:
            m._no_float({"v": v}, values=True)
            e = None
        except m.Stop as x:
            e = str(x)
        res[f"NEGATIVE O1 a number-like reducer string {v!r}: STOP"] = e == "a reducer value is a number-like string"
    res["O1 a digit-only 64-hex sha, a path and a verdict pass as reducer values"] = \
        m._no_float({"s": "1" * 64, "p": "results/out_fab.json", "v": "COMPLETE", "n": 20, "b": True, "z": None}, values=True) is not None
    for k, v in res.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("ALL PASS" if all(res.values()) else "FAILED")
    shutil.rmtree(tmp, ignore_errors=True)
    if not all(res.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worktree", action="store_true", help="test the working-tree wrapper instead of the HEAD blob")
    main(ap.parse_args().worktree)
