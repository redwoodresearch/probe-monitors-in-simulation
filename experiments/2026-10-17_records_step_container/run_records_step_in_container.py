"""Run ONE allowlisted, already-reviewed records builder UNCHANGED inside a Modal CPU container, from a git bundle of its pinned commits
(2e's design, 10-17). The laptop never parses a record: it checks the allowlist, makes and hashes the bundle, uploads it once, and
writes back only the step's allowlisted output files (exclusive create, re-hashed). Records holding measured values are read only
inside the container.

Laptop (client):
1. HEAD == C (40 hex, from the step spec's go) and a clean tracked tree.
2. The step spec is read from C (git show C:<spec>); its script must be in SCRIPTS with the pinned blob at C; its args, commit set,
   output allowlist and reducer come from the spec.
3. The bundle: a temporary ref refs/recstep/<sha> per commit of the step's set (C included), `git bundle create` of those refs, the refs
   deleted in finally. The bundle file lives only under a 0700 mktemp dir, removed in finally. Its sha256 is printed.
4. Upload once to BUNDLE_VOL at bundles/<sha256>.bundle with batch_upload(force=False): an existing name is reused only because the
   container re-checks the sha.
5. Before any Modal call: no file matching a step output (dir + regex) may already exist in the laptop's working tree (R4).
6. A fresh run_id (uuid4 hex) per launch goes into the payload; a payload of 256 KiB or more STOPs before any Modal call.
7. One remote call; the reply carries METADATA only (no bytes: a reply over Modal's 2 MiB inline limit is uploaded to its blob store
   from inside the container, which block_network forbids): status, stop reason, the exit code, the exception TYPE, refused git argvs,
   stray paths, and per allowlisted output its repo path, sha256, size and vol_path (runs/<run_id>/<repo path> on OUT_VOL), plus the
   reducer's values. Every relayed field passes a strict type whitelist (str, int, bool, None, list, dict with str keys; reducer values
   also refuse number-like strings). The client checks ALL outputs first (exact count n_outputs, allowlist, vol_path, size, sha, no
   existing destination), then fetches ALL bytes from OUT_VOL and checks each size and sha, then writes each with open("xb") and
   re-hashes it: all or nothing, bytes only (never parsed). Only keys and shas are printed.
8. A verify_only step (R2) writes no file: outputs [], n_outputs 0, and it passes only when rc is 0, nothing is refused or stray, and
   its reducer reports pass (checked in the container and again by the client). A json_line_keys step pins its expect_keys (O19).

Container (run_step): the bundle sha is checked, and its heads must be EXACTLY refs/recstep/<sha> for the listed commits (no more, no
fewer; `git bundle list-heads`); `git init` + `git fetch <bundle> refs/recstep/*` (no remote, network blocked), and the fetched
refs must again be exactly that set (`git for-each-ref`); checkout --detach C; C^{tree} and the script blob are checked. A `git` wrapper first on PATH (inherited by child processes) allows only
read verbs and execs the real git; anything else is refused (argv logged, exit 97). The script runs as `python -I -B` (sys.path from
the spec) with cwd = the repo root; its stdout goes to the per-script reducer only, stderr is dropped. A before/after diff of the
WHOLE checkout with NOTHING excluded: every file under the top-level .git (HEAD, refs, objects, config), any nested .git name, every
other file; symlinks are recorded by their target, never followed. The container runs as root, so file modes cannot protect .git, and
read verbs never write it. The diff must match the step's output allowlist exactly (regular files only: a symlinked output STOPs);
anything else STOPs. The reducer's paths and shas must be among the returned outputs; a fix_v2 step is OK only if its reducer parsed
the final JSON and both its shas are bound to returned outputs, and an inputs_full BUILD step only if its printed sha256 parsed and is
bound to a returned output (O24). The run_id must be 32 lowercase hex (never in the builder's env). The
outputs Volume (OUT_VOL at OUT_MNT, the ONLY read-write mount) is hashed before and after the builder: any change STOPs. ONLY an
otherwise-OK step publishes, and only after its reply has passed the whitelist and the size guard: exclusive mkdir runs/<run_id>,
each output open("xb") + fsync + re-hash, then the completion marker runs/<run_id>/_COMPLETE.json LAST (the same way), then ONE
commit(). A reply of 256 KiB or more becomes a minimal STOP. Modal also commits the read-write Volume by itself (in the background
and at exit), so OUT_VOL is TRANSPORT ONLY: a run without _COMPLETE.json is incomplete, and the sha-checked git copies are the truth.

After the census side a STOP (10-17; 09's (iii) and F-out2): an "@out:" flag's parent directories are created in the checkout before
the builder starts (out_path: relative, no "..", allowlisted, under experiments/<folder>/results/, real directories only, inside the
checkout, the output not yet present; the builder still opens it exclusive-create). A builder that raises (or exits non-zero through
SystemExit) returns its traceback as code lines only: "<repo path>.py:<line>" per frame inside the checkout, "<outside>" otherwise,
"?" for anything else (tb_frames); no message, source line or local. A step that ran but did not pass carries the stop reason of its
FIRST failing condition (first_failure: exit code / exception, refused git, stray, reducer), a fixed phrase.

  PY=/Users/aswinsurya/.cache/probe-shift-modal161/bin/python   # modal 1.6.1
  "$PY" -I <this file> client --spec <repo path of the step spec> --commit <C, 40 hex>
"""
import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

APP_NAME = "probe-shift_aswin-surya_sub-q3cutfull_recstep_kill-2026-11-14"  # 61 chars
BUNDLE_VOL = "probe-shift_aswin-surya_sub-q3cutfull_recbundle_kill-2026-11-14"  # 63 chars, created once by `"$M" volume create`
STAGE_VOL = "probe-shift_aswin-surya_sub-q3onpol_work_kill-2026-11-13"  # Cp's Volume stage record (read-only)
OUT_VOL = "probe-shift_aswin-surya_sub-q3cutfull_recout_kill-2026-11-14"  # 60 chars, created once by `"$M" volume create --version 2`; READ-WRITE
OUT_MNT = "/recout"  # the outputs go to OUT_VOL at runs/<run_id>/<repo path>; the reply carries metadata only
BUNDLE_MNT, VOL_MNT = "/bundles", "/vol"
RCMAN_VOL, RCMAN_MNT = "probe-shift_aswin-surya_sub-routec_manifests_kill-2026-11-14", "/rcman"  # route C: the 862 manifest copies (read-only)
OUT_ARG = "@out:"  # a raw_f flag value "@out:<repo path>" becomes that path in the checkout (it must be an allowlisted output)
OUT_PREFIX = re.compile(r"experiments/[A-Za-z0-9._-]+/results/[A-Za-z0-9._/-]+")  # 09 (10-17): an @out path lives under experiments/<folder>/results/
# the allowlist: repo path -> (git blob at C, reducer). Anything else STOPs before any Modal call.
SCRIPTS = {  # path -> (blob, reducer); blob None = the STEP SPEC pins it (script_blob), for a script whose route C head is not fixed yet
    # each blob re-derived 10-17 by `git rev-parse <commit>:<path>` at full length (09f4e2815, 5ba2b21a8, bf90327ae); the main test
    # checks that every pinned blob exists (`git cat-file -t` == blob), so a typo fails in review
    "experiments/2026-10-14_q3_overnight_chain/cutfull/make_pilot_records.py": ("9cbd7f2171b3277e5b186feca2bb6dee2768f483", "make_pilot_records"),
    "experiments/2026-10-15_cut_sensitivity/make_inventory_b_fix_v2.py": ("080ac97b0df53d872608b50bb0f150da4641409d", "fix_v2"),
    "experiments/2026-10-15_fpr_sweep_507_prep/inputs_full.py": ("093bc2bb1e032646903ca8d753dea2a9e1a23b39", "inputs_full"),  # route C steps 1, 2
    # F2 (0d, 10-17): cf66c533 (as reviewed @ bf90327ae) plus --copies-root only, on aswin/routec-launch
    "experiments/2026-10-15_fpr_sweep_507_prep/run_peraction311.py": (None, "json_line_keys"),  # route C steps 3, 6 (raw_f, spec_only)
}
VERIFY_REDUCERS = {"inputs_full", "dryrun_plan"}  # a verify_only step (no output file) needs a reducer that reports "pass"
ALT_REDUCERS = {  # route C §0.3 (10-17): a second allowed reducer for an allowlisted script, by step mode
    "experiments/2026-10-15_fpr_sweep_507_prep/run_peraction311.py": ("dryrun_plan",),  # the full-length dry run on a TEST INPUTS_FULL
}
MAX_PLAN_REFUSED = 200  # the dry-run reducer lists at most this many refused key paths (a longer list STOPs the step: not a pass)
MEMORY_MIB = (1024, 4096, 8192)  # the step spec's memory_mib picks one function per size (cap 8 GiB); Cp / Ca 1024, fix_v2 / inputs_full 4096
READ_VERBS = {"show", "rev-parse", "ls-tree", "cat-file", "merge-base", "status", "ls-files"}  # merge-base only with --is-ancestor;
# status / ls-files run only with GIT_OPTIONAL_LOCKS=0 (the wrapper forces GIT_ENV on every call): status then never refreshes .git/index
HEX40, HEX64, RUN_ID = re.compile(r"[0-9a-f]{40}"), re.compile(r"[0-9a-f]{64}"), re.compile(r"[0-9a-f]{32}")
MAX_INLINE = 256 * 1024  # payload and reply guard, far below Modal's 2 MiB inline limit (modal/_utils/blob_utils.py L36)
ENVELOPE = 1024  # headroom for what _remote adds to the reply (git_version) when the guard runs before commit()
MARKER = "_COMPLETE.json"  # runs/<run_id>/_COMPLETE.json: {run_id, outputs: [{path, sha256, size}]}, written last, just before commit()
VOLSHA = "@volsha:"  # an arg "@volsha:<path under the stage Volume>" becomes that file's sha256, inside the container


class Stop(Exception):
    pass


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


NUMLIKE = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?|[+-]?(nan|inf|infinity)", re.I)
# R-S1 (second review, 10-17): a relayed string passes only if PINNED (the step spec's expect_strings), or if it is a plain identifier /
# path (STRCHARS) that is not wholly number-like and holds no EMBEDDED number token (a decimal, an exponent, a percent, nan or inf) once
# its 40- and 64-hex shas are set aside. Integers inside a name (t309, S_paper493, a date) are allowed; "fpr_0.0123", "catch-0.41",
# "t309:0.0123", "1e-3" and "12.5%" are not. Blinding (507-prep REQUIRED-1, ruling (a), 10-17): a refused VALUE is relayed as None and a
# refused dict KEY or path segment as the index placeholder refused_key_<i>; no hash of the text in any form (an unsalted hash of a
# low-entropy string can be reversed by guessing). R-S1b (second review, 10-17): GLUED numbers count too ("fpr0.01", "0.97auc", "lr1e-5",
# "32.5k"): any digit.digit, any digit e[+-]digit and any '%' anywhere; word boundaries only around nan / inf; a 40- or 64-hex run is set
# aside only when no '.' or hex character touches it (so "x_0." + 40 digits is still a decimal). The real root "rh1e4_stage_v1__honest_b"
# carries "1e4": it relays only when pinned, and PIN_ALLOW is the only way such a string may be pinned (O2-1, adapted).
# Leading-dot decimals count too (".41", "catch_.41", "a/.5/b"; 507-prep, 10-17), and a hex run touching ".<digit>" is not set aside
# (second review, sidecar3: sha-named files like "x/<sha256>.json" pass). Accepted contrived passes: "nan5", "5nan", "0x1p-3", and an
# exponent glued to a 40- or 64-hex run ("5E" + 40 zeros + "."; see the corpus diff in the delta's report).
STRCHARS = re.compile(r"[A-Za-z0-9._/@|:-]{1,400}")
HEXTOK = re.compile(r"(?<![0-9a-f.])(?:[0-9a-f]{64}|[0-9a-f]{40})(?![0-9a-f])(?!\.\d)")  # 2nd review (sidecar3): sha-named files pass
EMBEDNUM = re.compile(r"\.\d|\d[eE][+-]?\d|%|(?<![A-Za-z0-9])(?:nan|inf(?:inity)?)(?![A-Za-z0-9])", re.I)  # 507-prep: leading-dot decimals
PIN_ALLOW = ("rh1e4_stage_v1__honest_b",)  # O2-1: the committed allowlist of pinnable strings that carry an embedded number


def _has_number(s: str) -> bool:
    """A wholly number-like string (not a 40- / 64-hex sha), or one with an embedded or glued number token (R-S1b)."""
    return bool((NUMLIKE.fullmatch(s) and not (HEX40.fullmatch(s) or HEX64.fullmatch(s))) or EMBEDNUM.search(HEXTOK.sub("x", s)))


def _str_ok(s: str, pinned=frozenset()) -> bool:
    if s in pinned:
        return True
    return bool(STRCHARS.fullmatch(s)) and not _has_number(s)


def _no_float(o, values=False, pinned=frozenset()):
    """STRICT type whitelist for everything relayed: str, int, bool, None, list, dict with str keys. Anything else (float, tuple, set,
    bytes, complex, numpy scalars, ...) STOPs. With values=True (reducer output), every string AND every dict key must pass _str_ok
    (pinned strings, or the identifier / no-embedded-number rule), else STOP."""
    if o is None or type(o) in (bool, int):
        return o
    if type(o) is str:
        if values and not _str_ok(o, pinned):
            if NUMLIKE.fullmatch(o):  # the O1 message, unchanged for a wholly number-like string
                raise Stop("a reducer value is a number-like string")
            raise Stop("a reducer value is a string that is neither pinned nor a plain identifier without an embedded number")
        return o
    if type(o) is list:
        for v in o:
            _no_float(v, values, pinned)
        return o
    if type(o) is dict and all(type(k) is str for k in o):
        for k, v in o.items():
            if values and not _str_ok(k, pinned):
                raise Stop("a reducer dict key is neither pinned nor a plain identifier without an embedded number")
            _no_float(v, values, pinned)
        return o
    raise Stop(f"a relayed field has a disallowed type ({type(o).__name__})")


def reduce_make_pilot_records(stdout: str) -> dict:
    """Only make_pilot_records' own publish lines (write_once L131): '<repo path> written; sha256 <64 hex>'."""
    w = [m.groups() for m in (re.fullmatch(r"([A-Za-z0-9_./@-]+\.json) written; sha256 ([0-9a-f]{64})", x) for x in stdout.splitlines()) if m]
    return {"written": [{"path": p, "sha256": h} for p, h in w]}


def reduce_fix_v2(stdout: str) -> dict:
    """make_inventory_b_fix_v2.py's final JSON (L245-246) -> booleans, ints and shas only (its stop / check texts could quote FINAL)."""
    try:
        d = json.loads(stdout)
    except ValueError:
        return {"parsed": False}
    part = lambda x: {"sha256": x.get("sha256") if HEX64.fullmatch(str(x.get("sha256"))) else None,  # noqa: E731
                      "n": x.get("n") if type(x.get("n")) is int else None} if isinstance(x, dict) else None
    return {"parsed": isinstance(d, dict), "complete": isinstance(d, dict) and d.get("status") == "COMPLETE", "dry": isinstance(d, dict) and d.get("dry") is True,
            "pilot_rows": part(d.get("pilot_rows")) if isinstance(d, dict) else None, "fix_v2": part(d.get("fix_v2")) if isinstance(d, dict) else None,
            "n_stops": len(d["stops"]) if isinstance(d, dict) and isinstance(d.get("stops"), list) else None,
            "n_post_write_checks": len(d["post_write_checks"]) if isinstance(d, dict) and isinstance(d.get("post_write_checks"), list) else None}


def reduce_json_line_keys(stdout: str, expect: list) -> dict:
    """Route C runner (spec-only): exactly ONE stdout line that is a JSON object; its keys are compared with the step spec's pinned
    expect_keys (O19). Only a boolean and counts return: no key text, no value."""
    objs = []
    for x in stdout.splitlines():
        try:
            o = json.loads(x)
        except ValueError:
            continue
        if isinstance(o, dict):
            objs.append(o)
    one = objs[0] if len(objs) == 1 else None
    return {"json_lines": len(objs), "keys_match": one is not None and sorted(one) == expect, "n_keys": len(one) if one is not None else None}


def reduce_inputs_full(stdout: str) -> dict:
    """inputs_full.py @ bf90327ae prints ONE indented JSON object (indent=1) as its whole stdout: verify (L944) {file, problems, pass},
    build (L211) {file, sha256, pending, problems}. -> verify {pass, n_problems}; build {sha256, n_pending, n_problems}; anything else
    {parsed: False}. No problem or pending text returns."""
    try:
        d = json.loads(stdout)
    except ValueError:
        return {"parsed": False}
    n = lambda k: len(d[k]) if isinstance(d[k], list) else None  # noqa: E731
    if isinstance(d, dict) and set(d) == {"file", "problems", "pass"}:
        return {"pass": d["pass"] is True and n("problems") == 0, "n_problems": n("problems")}
    if isinstance(d, dict) and set(d) == {"file", "sha256", "pending", "problems"}:
        return {"sha256": d["sha256"] if HEX64.fullmatch(str(d["sha256"])) else None, "n_pending": n("pending"), "n_problems": n("problems")}
    return {"parsed": False}


def _plan_view(o, path, refused, pinned=frozenset()):
    """The dry-run plan, keys and counts only. int, bool and None pass; a string passes only under _str_ok (pinned, or a plain identifier
    without an embedded or glued number), else it is dropped (None); a float or any other type is dropped (None); a dict KEY that fails
    becomes refused_key_<i> (i = its position in its dict). Each refused item's path is listed as a LIST of segments (O2-3: never a dotted
    string, which the digit.digit rule would refuse): list items by their int index, keys by the name they relay under (refused_key_<i>
    when refused). No raw unpinned text, and no hash of it, reaches the output (R-S1; ruling (a))."""
    if o is None or type(o) in (bool, int):
        return o
    if type(o) is str:
        if _str_ok(o, pinned):
            return o
        refused.append(list(path))
        return None
    if type(o) is list:
        return [_plan_view(v, path + [i], refused, pinned) for i, v in enumerate(o)]
    if type(o) is dict and all(type(k) is str for k in o):
        out = {}
        j = 0  # O3-1: refused keys are numbered among the REFUSED siblings only (the index reveals no sort position)
        for k, v in o.items():
            if _str_ok(k, pinned):
                k2 = k
            else:
                k2, j = f"refused_key_{j}", j + 1
            if k2 in out:  # a placeholder equal to a real key of the same dict: the whole dict is dropped
                refused.append(list(path))
                return None
            if k2 != k:
                refused.append(path + [k2])
            out[k2] = _plan_view(v, path + [k2], refused, pinned)
        return out
    refused.append(list(path))
    return None


def reduce_dryrun_plan(stdout: str, expect_strings=(), expect_plan_keys=None) -> dict:
    """Route C §0.3 (10-17): run_peraction311.py --dryrun (L5678-5715 at the relock-pilot head) prints ONE line {read_started_pt, rows} and
    then the plan as ONE indent=1 JSON object (a "{" line ... a "}" line). -> {pass, plan (keys and counts only: _plan_view), refused_paths,
    n_refused, rows_n, read_started_pt}. pass needs exactly one plan object with dryrun true and a full_length block, exactly one
    {read_started_pt, rows} line, at most MAX_PLAN_REFUSED refused paths, the plan's top-level keys equal to the spec's expect_plan_keys
    (O-S1) and full_length.inputs_full.test true (O-S2: the TEST flag from INPUTS_FULL itself, not only the file name). No value-typed
    field and no unpinned free text returns (R-S1: _plan_view with the spec's expect_strings)."""
    lines = stdout.splitlines()
    heads = []
    for x in lines:
        try:
            o = json.loads(x)
        except ValueError:
            continue
        if isinstance(o, dict) and set(o) == {"read_started_pt", "rows"}:
            heads.append(o)
    starts = [i for i, x in enumerate(lines) if x == "{"]
    plans = []
    for i in starts:
        ends = [j for j in range(i + 1, len(lines)) if lines[j] == "}"]
        for j in ends:
            try:
                o = json.loads("\n".join(lines[i:j + 1]))
            except ValueError:
                continue
            if isinstance(o, dict):
                plans.append(o)
            break
    refused = []
    one = plans[0] if len(plans) == 1 else None
    pinned = frozenset(expect_strings)
    view = _plan_view(one, ["plan"], refused, pinned) if one is not None else None
    head = heads[0] if len(heads) == 1 else None
    ifu = (one.get("full_length") or {}).get("inputs_full") if one is not None and isinstance(one.get("full_length"), dict) else None
    ok = (one is not None and one.get("dryrun") is True and isinstance(one.get("full_length"), dict) and head is not None
          and type(head.get("read_started_pt")) is str and isinstance(head.get("rows"), list) and len(refused) <= MAX_PLAN_REFUSED
          and expect_plan_keys is not None and sorted(one) == list(expect_plan_keys) and isinstance(ifu, dict) and ifu.get("test") is True)
    stamp = head.get("read_started_pt") if head is not None and type(head.get("read_started_pt")) is str else None
    return {"pass": bool(ok), "plan": view if ok else None, "refused_paths": refused[:MAX_PLAN_REFUSED] if ok else [],
            "n_refused": len(refused), "n_plans": len(plans), "n_heads": len(heads),
            "rows_n": len(head["rows"]) if head is not None and isinstance(head.get("rows"), list) else None,
            "read_started_pt": stamp if stamp is not None and _str_ok(stamp) else None}


REDUCERS = {"make_pilot_records": reduce_make_pilot_records, "fix_v2": reduce_fix_v2, "json_line_keys": reduce_json_line_keys,
            "inputs_full": reduce_inputs_full, "dryrun_plan": reduce_dryrun_plan}
CONFIRM = "the reducer did not confirm the step (pinned JSON keys, a verify pass, a bound inputs_full build sha, or fix_v2 parsed with both shas bound)"
MAX_PATH = 200  # O2: a reducer path is a repo path of at most this length, and must be one of the returned outputs


def bind_reduced(reduced: dict, outputs: list) -> None:
    """O2: every path / sha the reducer names must be one of the returned outputs (by path and sha)."""
    got = {o["path"]: o["sha256"] for o in outputs}
    for w in reduced.get("written", []):
        if len(w["path"]) > MAX_PATH or got.get(w["path"]) != w["sha256"]:
            raise Stop("a reducer path is too long or not one of the returned outputs (by path and sha)")
    for k in ("pilot_rows", "fix_v2"):
        if isinstance(reduced.get(k), dict) and reduced[k].get("sha256") not in set(got.values()):
            raise Stop("a reducer sha is not one of the returned outputs")
    if "sha256" in reduced and reduced["sha256"] not in set(got.values()):  # inputs_full build: its printed sha is its written file
        raise Stop("a reducer sha is not one of the returned outputs")

GIT_WRAPPER = r'''#!{python} -I
import json, os, sys
a, i = sys.argv[1:], 0
while i + 1 < len(a) and a[i] == "-C":
    i += 2
verb, rest = (a[i] if i < len(a) else ""), a[i + 1:]
ok = verb in {verbs} and not any(x.startswith(("--output", "--exec", "--upload-pack", "--receive-pack")) for x in rest) \
    and (verb != "merge-base" or "--is-ancestor" in rest)
if not ok:
    with open({log!r}, "a") as fh:
        fh.write(json.dumps(sys.argv[1:]) + "\n")
    sys.exit(97)
os.execve({real!r}, [{real!r}, *sys.argv[1:]], {{**{{k: v for k, v in os.environ.items() if not k.startswith("GIT_")}}, **{genv!r}}})
'''  # O18: every inherited GIT_* (GIT_CONFIG_PARAMETERS, GIT_DIR, GIT_INDEX_FILE, ...) is dropped, then GIT_ENV is forced

TB_DEF = r'''
import json, os, runpy, sys, traceback
ROOT = os.path.realpath(os.getcwd()) + os.sep  # the checkout (run_step starts the builder with cwd = the repo root)


def _tb(e):  # 09's (iii), 10-17: each frame's code line, "<repo path>:<line>" inside the checkout, else "<outside>"; no message text
    out = []
    for fr, ln in traceback.walk_tb(e.__traceback__):  # frames and line numbers only: no source line, no message, no local is read
        f = fr.f_code.co_filename
        p = os.path.realpath(f) if os.path.isabs(f) else ""
        out.append(f"{p[len(ROOT):]}:{ln}" if p.startswith(ROOT) and os.path.isfile(p) else "<outside>")
    return out
'''

RAW_F_BOOT = TB_DEF + r'''
for d in reversed(json.loads(os.environ["RECSTEP_SYS_PATH"])):
    sys.path.insert(0, d)
entry = json.loads(os.environ["RECSTEP_ENTRY"])
code, et, tb = 0, None, []
try:  # run_path with a non-main name: the module's `if __name__ == "__main__"` block does not run; then main.info.raw_f(**flags)
    getattr(runpy.run_path(entry["module"], run_name="recstep_entry")[entry["attr"]], "info").raw_f(**entry["flags"])
except SystemExit as e:
    code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    et, tb = (None, []) if code == 0 else ("SystemExit", _tb(e))
except BaseException as e:
    code, et, tb = 1, type(e).__name__, _tb(e)
with open(os.environ["RECSTEP_EXC"], "w") as fh:
    fh.write(json.dumps({"code": code, "type": et, "tb": tb}))
sys.exit(code)
'''

BOOT = TB_DEF + r'''
for d in reversed(json.loads(os.environ["RECSTEP_SYS_PATH"])):
    sys.path.insert(0, d)
script, sys.argv = sys.argv[1], sys.argv[1:]
code, et, tb = 0, None, []
try:
    runpy.run_path(script, run_name="__main__")
except SystemExit as e:
    code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    et, tb = (None, []) if code == 0 else ("SystemExit", _tb(e))
except BaseException as e:
    code, et, tb = 1, type(e).__name__, _tb(e)
with open(os.environ["RECSTEP_EXC"], "w") as fh:
    fh.write(json.dumps({"code": code, "type": et, "tb": tb}))
sys.exit(code)
'''


GIT_ENV = {"GIT_OPTIONAL_LOCKS": "0", "GIT_CONFIG_COUNT": "3", "GIT_CONFIG_KEY_0": "gc.auto", "GIT_CONFIG_VALUE_0": "0",
           "GIT_CONFIG_KEY_1": "maintenance.auto", "GIT_CONFIG_VALUE_1": "false", "GIT_CONFIG_KEY_2": "core.fsmonitor",
           "GIT_CONFIG_VALUE_2": "false"}  # O14: no background writes to .git (git >= 2.31), so a clean step cannot false-STOP
EXC_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")  # O13: an exception TYPE name only
MAX_STRAY, MAX_STRAY_LEN = 50, 200  # O13: stray paths are capped in count and length
TB_FRAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._/-]{0,199}\.py:[0-9]{1,7}")  # 09's (iii): a repo .py path and a line number, nothing else
MAX_TB = 40  # the innermost frames kept


def tb_frames(tb) -> list:
    """The builder's traceback as relayed: each frame "<repo path>.py:<line>" (a relative path inside the checkout, no "..") or
    "<outside>"; anything else becomes "?". The innermost MAX_TB frames; no message text, no source line, no local ever."""
    if not isinstance(tb, list):
        return []
    return [x if x == "<outside>" or (type(x) is str and TB_FRAME.fullmatch(x) and ".." not in x.split("/")) else "?" for x in tb[-MAX_TB:]]


def first_failure(rc, exc_type, refused, stray, red_ok) -> str:
    """09's (iii), 10-17: the stop reason of a step that ran but did not pass is the FIRST failing condition, in the order the publish
    check reads them (exit code / exception, refused git, stray paths, the reducer); a fixed phrase, never runner text."""
    if rc != 0 or exc_type is not None:
        return "the builder exited non-zero or raised (see the exception type and the traceback)"
    if refused:
        return "the builder made a git call outside the read verbs (refused)"
    if stray:
        return "the builder changed a path outside its output allowlist (stray)"
    return CONFIRM if not red_ok else "the step did not pass"


def out_path(repo: Path, q: str, okout) -> Path:
    """An "@out:<repo path>" flag -> that path in the checkout (09, 10-17, after the census side a STOP: the runner opens it with
    open("xb") and creates no directory). The path must be relative, canonical, without "..", allowlisted (dir + regex) and under
    experiments/<folder>/results/; each missing parent is created ONLY by walking real directories inside the checkout (a symlink or an
    existing FILE on the way STOPs), the created parent must resolve inside the checkout, and the output itself must not exist yet."""
    p = Path(q)
    if p.is_absolute() or ".." in p.parts or str(p) != q or not OUT_PREFIX.fullmatch(q) or not okout(q):  # O2: canonical only ("./",
        raise Stop("an @out flag is not an allowlisted output path")  # "//", "/./" and a trailing "/" change under Path(); a0b05, 10-17)
    d = repo
    for part in p.parts[:-1]:
        d = d / part
        if d.is_symlink() or (d.exists() and not d.is_dir()):
            raise Stop("an @out parent is a symlink or an existing file")
        d.mkdir(exist_ok=True)
    dest = repo / p
    if not dest.parent.resolve().is_relative_to(repo.resolve()) or dest.exists() or dest.is_symlink():
        raise Stop("an @out output resolves outside the checkout or already exists")
    return dest


def stage_manifests(st: dict, repo: Path, rcman: Path, g) -> Path:
    """Route C: place the manifest copies at their ORIGINAL absolute paths, before the runner starts. The INDEX comes from the bundle
    (commit + path + sha256): top-level n_files and files[]; each entry has original (the absolute path, which is also the file's path
    on the manifests Volume), repo_path, sha256, bytes, status and sources. ALL entries are staged by sha256, whatever their status.
    A count != n_files (or != the spec's count), a missing, extra, duplicate or mismatched file, or a path outside the allowed prefix:
    STOP, before the runner starts. -> the prefix root."""
    ib = g("show", f"{st['index']['commit']}:{st['index']['path']}", raw=True)
    if sha(ib) != st["index"]["sha256"]:
        raise Stop("the staging INDEX differs from its pin")
    doc = json.loads(ib)
    rows = doc.get("files") if isinstance(doc, dict) else None
    prefix = Path(st["prefix"])
    if not isinstance(rows, list) or type(doc.get("n_files")) is not int or len(rows) != doc["n_files"] or len(rows) != st["count"] \
            or not prefix.is_absolute() or ".." in prefix.parts:
        raise Stop("the staging INDEX does not hold exactly n_files == the pinned count of files, or the prefix is not a plain absolute path")
    seen = set()
    for e in rows:
        o, h, n = (e.get(k) if isinstance(e, dict) else None for k in ("original", "sha256", "bytes"))
        op = Path(str(o))
        if not (isinstance(o, str) and HEX64.fullmatch(str(h)) and type(n) is int) or not op.is_absolute() or ".." in op.parts \
                or not str(op).startswith(str(prefix) + "/") or o in seen:
            raise Stop("a staging INDEX entry is malformed, duplicated, or outside the allowed prefix")
        seen.add(o)
        f = rcman / o.lstrip("/")  # the Volume path is the original absolute path
        if f.is_symlink() or not f.is_file():
            raise Stop("a staged file is missing on the manifests Volume")
        b = f.read_bytes()
        if len(b) != n or sha(b) != h:
            raise Stop("a staged file differs from its INDEX sha256 or byte count")
        op.parent.mkdir(parents=True, exist_ok=True)
        if op.exists() or op.is_symlink():
            raise Stop("a staged path already exists in the container")
        op.write_bytes(b)
    on_vol = {"/" + str(q.relative_to(rcman)) for q in rcman.rglob("*") if q.is_file() or q.is_symlink()}
    if on_vol != seen:
        raise Stop(f"{len(on_vol - seen)} files on the manifests Volume are not in the INDEX")
    return prefix


def tree_shas(root: Path) -> dict:
    """Every entry of the checkout, NOTHING excluded: all of the top-level .git (HEAD, refs, objects, config), any nested .git name,
    every other file. A symlink is recorded by its target (never followed); a regular file by its bytes."""
    out = {}
    for p in sorted(root.rglob("*")):
        rel = str(p.relative_to(root))
        if p.is_symlink():
            out[rel] = "symlink:" + sha(os.readlink(p).encode())
        elif p.is_file():
            out[rel] = sha(p.read_bytes())
    return out


def run_step(payload: dict, work: Path, bundle_root: Path, vol_root: Path, python: str, real_git: str, out_root: Path, commit,
             rcman_root: Path = Path(RCMAN_MNT)) -> dict:
    """The container body (also run locally by the test, with a temp dir as out_root and a counter as commit). Returns only: status,
    stop reason, rc, exception type, refused git argvs, stray paths, the allowlisted outputs' METADATA (path, sha256, size, vol_path;
    no bytes), the reducer's output and resolved @volsha shas."""
    def g(*a, raw=False):  # the container's own git calls (setup only): argv only on failure
        r = subprocess.run([real_git, *a], cwd=repo, capture_output=True, env={**{k: v for k, v in os.environ.items() if not k.startswith("GIT_")}, **GIT_ENV})
        if r.returncode:
            raise Stop(f"git {' '.join(a)}: exit {r.returncode}")
        return r.stdout if raw else r.stdout.decode().strip()

    out = {"status": "STOP", "rc": None, "exc_type": None, "exc_tb": [], "stop_reason": None, "refused_git": [], "stray": [], "outputs": [],
           "reduced": None, "resolved": {}}
    try:
        if type(payload.get("run_id")) is not str or not RUN_ID.fullmatch(payload["run_id"]):  # the outputs dir name: never in the builder's env
            raise Stop("the run_id is not 32 lowercase hex")
        b = (bundle_root / payload["bundle"]).resolve()
        if not str(b).startswith(str(bundle_root.resolve()) + "/") or sha(b.read_bytes()) != payload["bundle_sha256"]:
            raise Stop("the bundle is missing, outside its Volume, or differs from its sha256")
        repo = work / "repo"
        repo.mkdir(parents=True)
        g("init", "-q")
        heads = {}
        for line in g("bundle", "list-heads", str(b)).splitlines():  # the bundle's CONTENT: exactly the listed commits, by ref and sha
            h, ref = line.split(" ", 1)
            heads[ref] = h
        if heads != {f"refs/recstep/{c}": c for c in payload["commits"]}:
            raise Stop("the bundle's heads are not exactly the listed commits")
        g("fetch", "--no-auto-maintenance", "-q", str(b), "refs/recstep/*:refs/recstep/*")  # O14
        fetched = dict(reversed(x.split(" ", 1)) for x in g("for-each-ref", "--format=%(objectname) %(refname)").splitlines())
        if fetched != {f"refs/recstep/{c}": c for c in payload["commits"]}:
            raise Stop("the fetched refs are not exactly the listed commits")
        g("checkout", "-q", "--detach", payload["commit"])
        if g("rev-parse", "HEAD") != payload["commit"] or g("rev-parse", f"{payload['commit']}^{{tree}}") != payload["tree"] \
                or g("rev-parse", f"{payload['commit']}:{payload['script']}") != payload["blob"] or g("remote") != "":
            raise Stop("C, its tree, or the script blob differs from the pins")
        resolved = {}
        args = []
        for x in payload["args"]:
            if x.startswith(VOLSHA):
                f = (vol_root / x[len(VOLSHA):]).resolve()
                if not str(f).startswith(str(vol_root.resolve()) + "/") or not f.is_file():
                    raise Stop("an @volsha path is outside the mounted Volume or missing")
                resolved[x] = sha(f.read_bytes())
                args.append(resolved[x])
            else:
                args.append(x)
        out["resolved"] = resolved
        binp, log, exc = work / "bin", work / "git_refused.jsonl", work / "exc.json"
        binp.mkdir()
        (binp / "git").write_text(GIT_WRAPPER.format(python=python, verbs=sorted(READ_VERBS), log=str(log), real=real_git, genv=GIT_ENV))
        (binp / "git").chmod(0o755)
        (work / "home").mkdir()
        if payload.get("staging"):  # O20: the staged tree and the work dir (repo included) are disjoint, so neither re-hash sees the other
            pre, w = Path(payload["staging"]["prefix"]).resolve(), work.resolve()
            if pre == w or pre.is_relative_to(w) or w.is_relative_to(pre):
                raise Stop("the staging prefix and the work dir overlap")
        staged_root = stage_manifests(payload["staging"], repo, rcman_root, g) if payload.get("staging") else None
        staged_before = tree_shas(staged_root) if staged_root else {}
        vol_before = tree_shas(out_root)  # the outputs Volume, symlinks by target: the builder (root, same container) must not touch it
        before = tree_shas(repo)
        env = {"PATH": f"{binp}:{os.environ.get('PATH', '/usr/bin:/bin')}", "HOME": str(work / "home"), "LANG": "C.UTF-8",
               "PYTHONDONTWRITEBYTECODE": "1",  # O23: a child python without -B writes no __pycache__ (a stray path, a false STOP)
               "RECSTEP_SYS_PATH": json.dumps([str(repo / d) for d in payload.get("sys_path", [])]), "RECSTEP_EXC": str(exc), **GIT_ENV}
        entry = payload.get("entry")
        if entry:  # route C: module.attr.info.raw_f(**flags); an "@out:<repo path>" flag becomes that path in the checkout
            okout = lambda q: any(str(Path(q).parent) == o["dir"] and re.fullmatch(o["regex"], Path(q).name) for o in payload["outputs"])  # noqa: E731
            flags = {}
            for k, v in entry["flags"].items():
                if isinstance(v, str) and v.startswith(OUT_ARG):  # 09 (10-17): checked, its parent made inside the checkout (out_path)
                    flags[k] = str(out_path(repo, v[len(OUT_ARG):], okout))
                else:
                    flags[k] = v
            env["RECSTEP_ENTRY"] = json.dumps({"module": str(repo / payload["script"]), "attr": entry["attr"], "flags": flags})
            cmd = [python, "-I", "-B", "-c", RAW_F_BOOT]
        else:
            cmd = [python, "-I", "-B", "-c", BOOT, str(repo / payload["script"]), *args]
        r = subprocess.run(cmd, cwd=repo, env=env, capture_output=True, timeout=payload.get("timeout_s", 1200))
        out["rc"] = r.returncode
        ej = json.loads(exc.read_text()) if exc.exists() else {"type": "unknown", "tb": []}
        et = ej.get("type")
        out["exc_type"] = et if et is None or EXC_RE.fullmatch(str(et)) else "unknown"  # O13: a type name or "unknown", never text
        out["exc_tb"] = tb_frames(ej.get("tb"))  # 09's (iii): code lines only
        refused = [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
        out["refused_git"] = [[str(a[0]) if a and EXC_RE.fullmatch(str(a[0]).replace("-", "_")) else "?", len(a), sha(json.dumps(a).encode())]
                              for a in refused]  # O13: [verb, number of args, sha256 of the argv]; the args themselves never return
        after = tree_shas(repo)
        if staged_root and tree_shas(staged_root) != staged_before:
            raise Stop("the runner changed the staged manifests")
        if tree_shas(out_root) != vol_before:
            raise Stop("the builder changed the outputs Volume")
        changed = sorted(p for p, h in after.items() if before.get(p) != h)
        allow = lambda p: any(str(Path(p).parent) == o["dir"] and re.fullmatch(o["regex"], Path(p).name) for o in payload["outputs"])  # noqa: E731
        stray = [p for p in changed if not allow(p)] + sorted(p for p in before if p not in after)
        out["stray"] = [p if len(p) <= MAX_STRAY_LEN else p[:MAX_STRAY_LEN] + "…" for p in stray[:MAX_STRAY]] + \
            ([f"… {len(stray) - MAX_STRAY} more"] if len(stray) > MAX_STRAY else [])  # O13: capped
        if any((repo / p).is_symlink() for p in changed if allow(p)):
            raise Stop("an output is a symlink")
        out["outputs"] = [{"path": p, "sha256": after[p], "size": (repo / p).stat().st_size} for p in changed if allow(p)]  # metadata only
        text = r.stdout.decode(errors="replace")
        red = (reduce_json_line_keys(text, payload["expect_keys"]) if payload["reducer"] == "json_line_keys"
               else reduce_dryrun_plan(text, payload["expect_strings"], payload["expect_plan_keys"]) if payload["reducer"] == "dryrun_plan"
               else REDUCERS[payload["reducer"]](text))
        out["reduced"] = _no_float(red, values=True, pinned=frozenset(payload.get("expect_strings") or ()))
        bind_reduced(out["reduced"], out["outputs"])
        _no_float(out)
        got = {o["sha256"] for o in out["outputs"]}
        red_ok = (payload["reducer"] != "json_line_keys" or red.get("keys_match") is True) \
            and (not payload.get("verify_only") or red.get("pass") is True) \
            and (payload["reducer"] != "inputs_full" or payload.get("verify_only") or (HEX64.fullmatch(str(red.get("sha256"))) and red["sha256"] in got)) \
            and (payload["reducer"] != "fix_v2" or (red.get("parsed") is True and all(
                isinstance(red.get(k), dict) and HEX64.fullmatch(str(red[k].get("sha256"))) and red[k]["sha256"] in got for k in ("pilot_rows", "fix_v2"))))
        # O19 pinned keys; R2 a verify step passes only on pass; O24 an inputs_full BUILD only with its parsed sha bound to an output; fix_v2
        # only parsed with both shas bound
        if r.returncode == 0 and not out["refused_git"] and not out["stray"] and out["exc_type"] is None and red_ok:
            # publish ONLY an otherwise-OK step: runs/<run_id>/<repo path>, exclusive dir and files, fsynced, re-read; then ONE commit
            for o in out["outputs"]:
                o["vol_path"] = f"runs/{payload['run_id']}/{o['path']}"
            _no_float(out | {"status": "OK"})  # R1: the final whitelist and the reply-size guard run BEFORE anything is written or
            if len(json.dumps(out | {"status": "OK"})) >= MAX_INLINE - ENVELOPE:  # committed, so a reply that would STOP never publishes
                raise Stop("the reply is too large to return inline")
            rd = out_root / "runs" / payload["run_id"]
            (out_root / "runs").mkdir(exist_ok=True)
            rd.mkdir()  # exclusive: an existing run dir STOPs (FileExistsError)
            mb = json.dumps({"run_id": payload["run_id"], "outputs": [{k: o[k] for k in ("path", "sha256", "size")} for o in out["outputs"]]},
                            indent=1).encode()  # the completion marker: written LAST, just before commit() (R1); never in the reply
            for q, size, want in [(o["path"], o["size"], o["sha256"]) for o in out["outputs"]] + [(MARKER, len(mb), sha(mb))]:
                (rd / q).parent.mkdir(parents=True, exist_ok=True)
                with (rd / q).open("xb") as fh:
                    fh.write(mb if q == MARKER else (repo / q).read_bytes())
                    fh.flush()
                    os.fsync(fh.fileno())
                back = (rd / q).read_bytes()
                if len(back) != size or sha(back) != want:
                    raise Stop("a file on the outputs Volume differs from its sha256 or size")
            commit()
            out["status"] = "OK"
        else:
            out["outputs"] = []  # O16: a reply that is not OK carries no outputs (and nothing is published)
            out["stop_reason"] = first_failure(r.returncode, out["exc_type"], out["refused_git"], out["stray"], red_ok)  # rc / exc_type kept
    except Stop as e:
        out |= {"status": "STOP", "exc_type": "Stop", "stop_reason": str(e), "outputs": []}  # our own reasons only (setup argv or a fixed phrase)
    except Exception as e:  # noqa: BLE001  (the publish block included: exception TYPE only)
        out |= {"status": "STOP", "exc_type": type(e).__name__, "outputs": []}
    try:
        _no_float(out)
    except Stop:
        out = {"status": "STOP", "exc_type": "Stop", "stop_reason": "a relayed field has a disallowed type", "refused_git": [], "stray": [], "outputs": [],
               "reduced": None, "resolved": {}, "rc": None}
    if len(json.dumps(out)) >= MAX_INLINE:  # Modal would upload a big reply to its blob store from inside the container: STOP instead
        out = {"status": "STOP", "exc_type": None, "stop_reason": "the reply is too large to return inline", "refused_git": [], "stray": [],
               "outputs": [], "reduced": None, "resolved": {}, "rc": None}
    return out


# ------------------------------------------------------------------ laptop side
def git_l(repo: Path, *a, text=True):
    return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=text, check=True).stdout


def preflight(repo: Path, spec_path: str, commit: str) -> tuple[dict, dict]:
    """-> (payload, spec). No Modal call, no record parsed: HEAD == C, a clean tracked tree, the spec from C, the allowlist and blob."""
    if not HEX40.fullmatch(commit) or git_l(repo, "rev-parse", "HEAD").strip() != commit or git_l(repo, "status", "--porcelain", "--untracked-files=no").strip():
        raise Stop("HEAD is not C, or the tracked tree is dirty")
    spec = json.loads(git_l(repo, "show", f"{commit}:{spec_path}"))  # a config file
    s = spec.get("script")
    if s not in SCRIPTS:
        raise Stop("the step's script is not on the allowlist")
    blob = git_l(repo, "rev-parse", f"{commit}:{s}").strip()
    want = SCRIPTS[s][0] if SCRIPTS[s][0] is not None else spec.get("script_blob")  # None: the reviewed step spec pins the blob
    if not HEX40.fullmatch(str(want)) or blob != want or spec.get("reducer") not in (SCRIPTS[s][1], *ALT_REDUCERS.get(s, ())):
        raise Stop("the script's blob at C (or its reducer) is not the allowlisted one")
    commits = spec.get("commits") or []
    if spec.get("memory_mib") not in MEMORY_MIB:
        raise Stop(f"the step's memory_mib must be one of {MEMORY_MIB}")
    vo, outs = spec.get("verify_only", False), spec.get("outputs")
    if not all(HEX40.fullmatch(str(c)) for c in commits) or type(vo) is not bool or not isinstance(outs, list) \
            or any(not isinstance(o, dict) or set(o) != {"dir", "regex"} for o in outs) or type(spec.get("n_outputs")) is not int \
            or (not vo and (not outs or spec["n_outputs"] < 1)) or (vo and (outs or spec["n_outputs"] != 0 or spec.get("reducer") not in VERIFY_REDUCERS)):
        raise Stop("the step's commit set must be full 40-hex shas, its outputs a list of {dir, regex}, and n_outputs an int >= 1 "
                   "(R2: verify_only true needs outputs [], n_outputs 0 and a reducer that reports pass)")
    ek = spec.get("expect_keys")  # O19: the runner's one stdout JSON line must have exactly these keys
    if (spec["reducer"] == "json_line_keys") != (ek is not None) or (ek is not None and (not isinstance(ek, list) or not ek
                                                                                       or not all(type(k) is str and EXC_RE.fullmatch(k) for k in ek) or ek != sorted(set(ek)))):
        raise Stop("expect_keys: a json_line_keys step pins a sorted, unique, non-empty list of identifier keys; no other step has one")
    if spec.get("image", "base") not in ("base", "rc") or (spec.get("staging") and spec.get("image") != "rc") \
            or (spec.get("image") == "rc" and spec["memory_mib"] == 1024):
        raise Stop("the step's image must be base or rc; staging needs rc; rc runs at 4096 or 8192 MiB")
    st = spec.get("staging")
    if st and (not isinstance(st.get("index"), dict) or not HEX40.fullmatch(str(st["index"].get("commit"))) or not HEX64.fullmatch(str(st["index"].get("sha256")))
               or st["index"]["commit"] not in commits or type(st.get("count")) is not int or not str(st.get("prefix", "")).startswith("/")):
        raise Stop("the step's staging needs index {commit (in the commit set), path, sha256}, an int count and an absolute prefix")
    en = spec.get("entry")
    if en and (set(en) != {"attr", "flags"} or not isinstance(en["flags"], dict)):
        raise Stop("the step's entry must be {attr, flags}")
    if spec["reducer"] == "dryrun_plan" and not (en and en["attr"] == "main" and en["flags"].get("dryrun") is True and not en["flags"].get("spec_only")
                                                 and vo and str(en["flags"].get("inputs_full", "")).startswith("config/INPUTS_FULL_TEST_")):
        raise Stop("a dryrun_plan step is verify_only, runs main with dryrun true (no spec_only) on a TEST INPUTS_FULL (route C §0.3)")
    es, ek2 = spec.get("expect_strings"), spec.get("expect_plan_keys")  # R-S1 / O-S1: the pinned strings (labels, override keys from config
    if (spec["reducer"] == "dryrun_plan") != (es is not None) or (spec["reducer"] == "dryrun_plan") != (ek2 is not None) or (es is not None and (
            not isinstance(es, list) or not es or not all(type(x) is str and 0 < len(x) <= 400 and x.isprintable() for x in es) or es != sorted(set(es))
            or not isinstance(ek2, list) or not ek2 or not all(type(k) is str and EXC_RE.fullmatch(k) for k in ek2) or ek2 != sorted(set(ek2)))):
        raise Stop("a dryrun_plan step pins expect_strings (sorted, unique, printable, non-empty) and expect_plan_keys (sorted, unique "
                   "identifiers); no other step has either")  # and code at C) and the plan's top-level key set
    if es is not None and [x for x in es if _has_number(x) and x not in PIN_ALLOW]:  # O2-1: no membership oracle through the pins
        raise Stop("a dryrun_plan step may not pin a string with an embedded number, except the wrapper's PIN_ALLOW")
    for o in spec["outputs"]:  # R4: an output name that already exists in the working tree (tracked or not) STOPs before any Modal call
        d = repo / o["dir"]
        if d.is_dir() and any(re.fullmatch(o["regex"], q.name) for q in d.iterdir()):
            raise Stop(f"{o['dir']}: a file matching the step's output pattern already exists")
    return {"commit": commit, "tree": git_l(repo, "rev-parse", f"{commit}^{{tree}}").strip(), "script": s, "blob": blob, "args": spec.get("args", []),
            "sys_path": spec.get("sys_path", []), "outputs": spec["outputs"], "reducer": spec["reducer"], "timeout_s": spec.get("timeout_s", 1200),
            "memory_mib": spec["memory_mib"], "n_outputs": spec["n_outputs"], "image": spec.get("image", "base"),
            "entry": spec.get("entry"), "staging": spec.get("staging"), "verify_only": vo, "expect_keys": ek, "expect_strings": es, "expect_plan_keys": ek2,
            "commits": sorted(set(commits) | {commit})}, spec


def make_bundle(repo: Path, commits: list, tmp: Path) -> tuple[Path, str]:
    """Temporary local refs refs/recstep/<sha> (never pushed; deleted in finally) -> one bundle under tmp; returns (path, sha256)."""
    refs = [f"refs/recstep/{c}" for c in commits]
    try:
        for r, c in zip(refs, commits):
            git_l(repo, "update-ref", r, c)
        f = tmp / "recstep.bundle"
        git_l(repo, "bundle", "create", str(f), *refs)
        return f, sha(f.read_bytes())
    finally:
        for r in refs:
            subprocess.run(["git", "update-ref", "-d", r], cwd=repo, capture_output=True)


def write_outputs(repo: Path, payload: dict, res: dict, fetch) -> list:
    """Exclusive create at the same repo paths, each re-hashed; STOP on anything but an OK reply with allowlisted outputs. fetch(vol_path)
    -> bytes from the outputs Volume (bytes only, never parsed); ALL are fetched and checked before the first write."""
    if res.get("status") != "OK" or res.get("refused_git") or res.get("stray"):
        raise Stop(f"the step did not pass: status {res.get('status')}, exception type {res.get('exc_type')}, reason {res.get('stop_reason')}, "
                   f"refused git {res.get('refused_git')}, "
                   f"stray {res.get('stray')}, traceback {tb_frames(res.get('exc_tb'))}")  # 09's (iii): code lines only, re-filtered here
    if payload.get("verify_only") and (res.get("reduced") or {}).get("pass") is not True:  # R2: the client re-checks the verify pass
        raise Stop("a verify_only step did not report pass")
    outs = res.get("outputs") or []
    if len(outs) != payload["n_outputs"]:  # O5: everything is checked BEFORE the first write (all or nothing)
        raise Stop(f"{len(outs)} outputs returned, the step expects exactly {payload['n_outputs']}")
    for o in outs:
        p = o["path"]
        if not any(str(Path(p).parent) == x["dir"] and re.fullmatch(x["regex"], Path(p).name) for x in payload["outputs"]) or ".." in Path(p).parts \
                or o.get("vol_path") != f"runs/{payload['run_id']}/{p}" or type(o.get("size")) is not int or not HEX64.fullmatch(str(o.get("sha256"))):
            raise Stop(f"{p}: not an allowlisted output, or its vol_path, size or sha256 is malformed")
        if (repo / p).exists() or (repo / p).is_symlink():
            raise Stop(f"{p}: the destination already exists; nothing written")
    blobs = [fetch(o["vol_path"]) for o in outs]
    for o, b in zip(outs, blobs):
        if len(b) != o["size"] or sha(b) != o["sha256"]:
            raise Stop(f"{o['path']}: the fetched bytes differ from the returned size or sha256; nothing written")
    done = []
    for o, b in zip(outs, blobs):
        dest = repo / o["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as fh:
            fh.write(b)
        if sha(dest.read_bytes()) != o["sha256"]:
            raise Stop(f"{o['path']}: written bytes differ from the returned sha256")
        done.append((o["path"], o["sha256"]))
    return done


def client(spec_path: str, commit: str) -> None:
    import modal

    repo = Path(git_l(Path.cwd(), "rev-parse", "--show-toplevel").strip())
    payload, spec = preflight(repo, spec_path, commit)
    print("spec", spec_path, "sha256", sha(git_l(repo, "show", f"{commit}:{spec_path}", text=False)), "commit", commit)  # O-S3: the log names its spec
    payload["run_id"] = uuid.uuid4().hex  # this launch's outputs dir on OUT_VOL: runs/<run_id>/
    tmp = Path(tempfile.mkdtemp(prefix="recstep_"))
    os.chmod(tmp, 0o700)
    try:
        f, bsha = make_bundle(repo, payload["commits"], tmp)
        if spec.get("bundle_sha256") not in (None, bsha):
            raise Stop("the bundle's sha256 differs from the step spec's")
        payload |= {"bundle": f"bundles/{bsha}.bundle", "bundle_sha256": bsha}
        if len(json.dumps(payload)) >= MAX_INLINE:  # before any Modal call (the bundle upload below is the first)
            raise Stop("the payload is too large to send inline")
        print("bundle sha256", bsha, "commits", len(payload["commits"]), "bytes", f.stat().st_size, "run_id", payload["run_id"])
        vol = modal.Volume.from_name(BUNDLE_VOL)
        names = {e.path for e in vol.listdir("bundles/")} if any(e.path == "bundles" for e in vol.listdir("/")) else set()
        if payload["bundle"] not in names:
            with vol.batch_upload(force=False) as up:
                up.put_file(str(f), payload["bundle"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    with app.run():  # no enable_output: the terminal shows only this file's key / sha lines
        res = REMOTE[(payload["image"], payload["memory_mib"])].remote(payload)
    done = write_outputs(repo, payload, res, lambda p: b"".join(modal.Volume.from_name(OUT_VOL, version=2).read_file(p)))
    for p, h in done:
        print(p, "written; sha256", h)
    print("reduced", json.dumps(res["reduced"], sort_keys=True), "git_version", json.dumps(res.get("git_version")))
    print("RECSTEP_OK")


try:  # the Modal app (absent in the test environment, which runs run_step locally)
    import modal

    image = modal.Image.debian_slim(python_version="3.12").apt_install("git").pip_install("tzdata==2025.2")
    app = modal.App(APP_NAME)

    OUTV = modal.Volume.from_name(OUT_VOL, version=2)  # READ-WRITE, no create_if_missing; VolumeFS v2 pinned (a mismatch raises); others read-only
    VOLS = {BUNDLE_MNT: modal.Volume.from_name(BUNDLE_VOL).read_only(), VOL_MNT: modal.Volume.from_name(STAGE_VOL).read_only(), OUT_MNT: OUTV}

    def _remote(payload: dict) -> dict:
        real = shutil.which("git")
        if not real:
            return {"status": "STOP", "exc_type": "Stop", "stop_reason": "no git in the image"}
        gv = subprocess.run([real, "--version"], capture_output=True, text=True).stdout.strip()  # Debian's git (unpinned apt package): recorded
        with tempfile.TemporaryDirectory() as w:
            return run_step(payload, Path(w), Path(BUNDLE_MNT), Path(VOL_MNT), sys.executable, real, Path(OUT_MNT), OUTV.commit) | {"git_version": gv}

    @app.function(image=image, cpu=1.0, memory=1024, timeout=1800, retries=0, block_network=True, volumes=VOLS)
    def remote_step_1g(payload: dict) -> dict:
        return _remote(payload)

    @app.function(image=image, cpu=1.0, memory=4096, timeout=1800, retries=0, block_network=True, volumes=VOLS)
    def remote_step_4g(payload: dict) -> dict:
        return _remote(payload)

    @app.function(image=image, cpu=1.0, memory=8192, timeout=1800, retries=0, block_network=True, volumes=VOLS)
    def remote_step_8g(payload: dict) -> dict:
        return _remote(payload)

    # route C: modal itself importable in the image (the runner module imports it; no Modal call is made under spec_only), plus the
    # read-only manifests Volume. The runner's spec path imports only the stdlib, its fpr507 siblings and modal (CUT's code read @ 5781ee1af).
    image_rc = modal.Image.debian_slim(python_version="3.12").apt_install("git").pip_install("tzdata==2025.2", "modal==1.6.1")  # no numpy on the spec path
    VOLS_RC = VOLS | {RCMAN_MNT: modal.Volume.from_name(RCMAN_VOL, version=2).read_only()}  # VolumeFS v2 pinned (the upload's --version 2)

    @app.function(image=image_rc, cpu=1.0, memory=4096, timeout=1800, retries=0, block_network=True, volumes=VOLS_RC)
    def remote_step_rc_4g(payload: dict) -> dict:
        return _remote(payload)

    @app.function(image=image_rc, cpu=1.0, memory=8192, timeout=1800, retries=0, block_network=True, volumes=VOLS_RC)
    def remote_step_rc_8g(payload: dict) -> dict:
        return _remote(payload)

    REMOTE = {("base", 1024): remote_step_1g, ("base", 4096): remote_step_4g, ("base", 8192): remote_step_8g,
              ("rc", 4096): remote_step_rc_4g, ("rc", 8192): remote_step_rc_8g}
except ImportError:
    modal = None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("client")
    c.add_argument("--spec", required=True)
    c.add_argument("--commit", required=True)
    a = ap.parse_args()
    try:
        client(a.spec, a.commit)
    except Stop as e:
        print("STOP:", e)
        sys.exit(1)
