"""O-P1 (10-17): drive() re-hashes the spec pickle it read, against the sha its launcher pinned, BEFORE unpickling; every spawn path
passes that sha. Synthetic, no Modal, nothing real unpickled.

  uv run --no-project --python 3.12 python -I test_drive_spec_sha.py
"""

import ast
import hashlib
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import full_override as FO  # noqa: E402


def stops(f, *a):
    try:
        f(*a)
    except SystemExit:
        return True
    return False


B = pickle.dumps({"meta": {"git": "x"}, "dest": "d.json"})
H = hashlib.sha256(B).hexdigest()
calls = []
real_loads = pickle.loads
pickle.loads = lambda b, *a, **k: calls.append(len(b)) or real_loads(b, *a, **k)  # counts every unpickle in the checks below
cases = {"load_spec_checked: the pinned sha of these bytes -> the object (one unpickle)": FO.load_spec_checked(B, H) == {"meta": {"git": "x"}, "dest": "d.json"}
         and calls == [len(B)]}
calls.clear()
cases["load_spec_checked: other bytes than the pinned sha (a swapped pickle) STOP, nothing unpickled"] = stops(FO.load_spec_checked, B + b"x", H) and calls == []
cases["load_spec_checked: a short, non-hex, upper-case or None sha STOPs, nothing unpickled"] = all(
    stops(FO.load_spec_checked, B, w) for w in (H[:63], "z" * 64, H.upper(), None, "")) and calls == []
pickle.loads = real_loads
cases["resume_spec_pin: <path>@<sha256> -> (path, sha)"] = FO.resume_spec_pin(f"xfer/fpr507prep/drivers/a.pkl@{H}") == ("xfer/fpr507prep/drivers/a.pkl", H)
cases["resume_spec_pin: no sha, a short sha, or an empty flag STOPs"] = all(
    stops(FO.resume_spec_pin, s) for s in ("xfer/fpr507prep/drivers/a.pkl", f"xfer/fpr507prep/drivers/a.pkl@{H[:40]}", ""))

# by syntax tree: drive's signature, its only unpickle, and every spawn of drive
T = ast.parse((HERE / "run_peraction311.py").read_text())
drive = next(n for n in T.body if isinstance(n, ast.FunctionDef) and n.name == "drive")
a_ = drive.args
cases["drive(spec_path, spec_sha256, ...): the sha is the REQUIRED second parameter"] = (
    [x.arg for x in a_.args[:2]] == ["spec_path", "spec_sha256"] and len(a_.defaults) == len(a_.args) - 2)
loads_ = [n for n in ast.walk(drive) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "loads"
          and isinstance(n.func.value, ast.Name) and n.func.value.id == "pickle"]
chk_ = [n for n in ast.walk(drive) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "load_spec_checked"]
cases["drive: no pickle.loads of its own; exactly one FO.load_spec_checked(<bytes read>, spec_sha256)"] = (
    not loads_ and len(chk_) == 1 and isinstance(chk_[0].args[1], ast.Name) and chk_[0].args[1].id == "spec_sha256")


def is_drive(v):
    return (isinstance(v, ast.Name) and v.id == "drive") or (isinstance(v, ast.IfExp) and is_drive(v.body) and is_drive(v.orelse)) or (
        isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr == "with_options" and is_drive(v.func.value))


spawns = [n for n in ast.walk(T) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "spawn" and is_drive(n.func.value)]
funcs = {id(s): f.name for f in ast.walk(T) if isinstance(f, ast.FunctionDef) for s in ast.walk(f) if s in spawns}
cases["every drive spawn passes a sha as its 2nd argument, not a constant: dispatch, launch_from, the resume branch of main"] = (
    len(spawns) == 3 and all(len(s.args) >= 2 and not isinstance(s.args[1], ast.Constant) for s in spawns)
    and sorted(funcs[id(s)] for s in spawns) == ["dispatch", "launch_from", "main"])
cases["the shas are the right ones: C.sha_b(blob) (dispatch), lh_ (launch_from), resume_sha_ from resume_spec_pin (resume)"] = sorted(
    ast.unparse(s.args[1]) for s in spawns) == ["C.sha_b(blob)", "lh_", "resume_sha_"]

for name, ok in cases.items():
    print(("PASS " if ok else "FAIL ") + name)
print(f"{sum(cases.values())} of {len(cases)} checks pass")
sys.exit(0 if all(cases.values()) else 1)
