"""0d's /tmp fix (10-17): copies.py reads every fpr507 manifest copy through ONE root ($PROBE_SHIFT_COPIES) and the committed index, and
nothing skips silently. Synthetic trees for the behaviour; the real index for its shape (class K: names and shas). No Modal.

  uv run --no-project --python 3.12 python -I test_copies.py
"""

import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import copies as CP  # noqa: E402

sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731


def stops(f, *a):
    try:
        f(*a)
    except SystemExit:
        return True
    return False


real_path, real_sha = CP.INDEX_PATH, CP.INDEX_SHA
d = Path(tempfile.mkdtemp(prefix="copies_t_"))
(d / "root" / "man455").mkdir(parents=True)
(d / "root" / "v2man").mkdir()
files = {"man455/a__sol.json": b'{"files": {}}', "man455/a__p2.json": b"{}", "v2man/simh311_x.json": b"{}", "v2man/gone.json": b"{}"}
for k, v in files.items():
    if k != "v2man/gone.json":
        (d / "root" / k).write_bytes(v)
(d / "root" / "v2man" / "simh311_not_indexed.json").write_bytes(b"{}")
idx = json.dumps({"files": {k: sha(v) for k, v in files.items()}}).encode()
(d / "index.json").write_bytes(idx)


def use(index_bytes=idx, root=d / "root"):
    (d / "index.json").write_bytes(index_bytes)
    CP.INDEX_PATH, CP.INDEX_SHA = str(d / "index.json"), sha(idx)
    CP._IDX.clear()
    if root is None:
        os.environ.pop(CP.ENV, None)
    else:
        os.environ[CP.ENV] = str(root)


use()
cases = {"read: an indexed file at its sha -> its bytes": CP.read("man455/a__sol.json") == files["man455/a__sol.json"],
         "read: an indexed file missing on disk STOPs": stops(CP.read, "v2man/gone.json"),
         "read: a file on disk but not in the index STOPs": stops(CP.read, "v2man/simh311_not_indexed.json"),
         "names: from the index, sorted; a file on disk outside the index is not listed": CP.names("v2man", "simh311_*.json") == ["v2man/simh311_x.json"]
         and CP.names("man455", "*__sol.json") == ["man455/a__sol.json"] and CP.names("nmman", "*") == [],
         "check: None when good; a problem string when missing or outside the index": CP.check("man455/a__p2.json") is None
         and "missing" in CP.check("v2man/gone.json") and "not in the index" in CP.check("v2man/nope.json")}
(d / "root" / "man455" / "a__p2.json").write_bytes(b'{"changed": 1}')
cases["read: a copy whose bytes changed (sha differs from the index) STOPs"] = stops(CP.read, "man455/a__p2.json")
(d / "root" / "man455" / "a__p2.json").write_bytes(files["man455/a__p2.json"])
use(root=None)
cases["root: $PROBE_SHIFT_COPIES unset STOPs at the first read"] = stops(CP.read, "man455/a__sol.json")
use(root=d / "no_such_dir")
cases["root: $PROBE_SHIFT_COPIES not a directory STOPs"] = stops(CP.read, "man455/a__sol.json")
use(index_bytes=idx + b" ")
cases["index: an index whose bytes are not the pinned sha STOPs"] = stops(CP.index)
use()
cases["positive control: after the negatives, the good tree reads again"] = CP.read("man455/a__sol.json") == files["man455/a__sol.json"]

# O1: one read per CP.read, and the bytes returned are the bytes hashed
_rb, n_reads = Path.read_bytes, []
Path.read_bytes = lambda self: n_reads.append(str(self)) or _rb(self)
CP._IDX.clear()
CP.index()
n_reads.clear()
got_ = CP.read("man455/a__sol.json")
Path.read_bytes = _rb
cases["O1: CP.read reads the copy ONCE and returns exactly the bytes it hashed"] = (
    n_reads == [str(d / "root" / "man455" / "a__sol.json")] and got_ == files["man455/a__sol.json"] and sha(got_) == json.loads(idx)["files"]["man455/a__sol.json"])

# 0d's B: an explicit copies_root (set_root) overrides the env; inside the records wrapper (RECSTEP_EXC) the env is never used
(d / "other").mkdir()
use(root=d / "other")  # the env points at a tree WITHOUT the copies
cases["B: without set_root, the env root is used (here: a tree without the file, so STOP)"] = stops(CP.read, "man455/a__sol.json")
CP.set_root(str(d / "root"))
cases["B: set_root (the runner's copies_root flag) overrides $PROBE_SHIFT_COPIES"] = CP.read("man455/a__sol.json") == files["man455/a__sol.json"]
CP._ROOT.clear()
cases["B: set_root to a missing directory STOPs"] = stops(CP.set_root, str(d / "nope")) and stops(CP.set_root, "")
os.environ[CP.WRAPPER_MARK] = "x"
use()  # the env points at the GOOD tree, but we are "inside the wrapper"
cases["B: inside the records wrapper (RECSTEP_EXC set) without the flag: STOP, even with a good $PROBE_SHIFT_COPIES"] = stops(CP.read, "man455/a__sol.json")
CP.set_root(str(d / "root"))
cases["B: inside the records wrapper WITH the flag: reads"] = CP.read("man455/a__sol.json") == files["man455/a__sol.json"]
CP._ROOT.clear()
os.environ.pop(CP.WRAPPER_MARK)
use()

# the real committed index (K): its pin, shape and the counts the runner's checks rely on
CP.INDEX_PATH, CP.INDEX_SHA = real_path, real_sha
CP._IDX.clear()
rb = (HERE / real_path).read_bytes()
ri = json.loads(rb)
cases["real index: its sha256 is copies.INDEX_SHA; 10 folders; every entry <folder>/<name>.json -> 64 hex"] = (
    sha(rb) == real_sha and ri["folders"] == ["simh", "ownfam", "ownall_man", "ownall_loo_man", "ownall_rs_man", "v2man", "nmman", "man455", "man710", "mandt"]
    and ri["n"] == len(ri["files"]) and all(re.fullmatch(r"[a-z0-9_]+/[A-Za-z0-9._@+-]+\.json", k) and re.fullmatch(r"[0-9a-f]{64}", v) for k, v in ri["files"].items())
    and {k.split("/")[0] for k in ri["files"]} == set(ri["folders"]))
cases["real index: man455 holds the 24 S_new arm sol manifests IF's coverage check requires; ownfam holds packs311.json"] = (
    len(CP.names("man455", "*__sol.json")) == 24 and CP.exists("ownfam/packs311.json"))

# wiring: no module on the launch path keeps a /private/tmp path or a silent fallback
src = {f: (HERE / f).read_text() for f in ("run_peraction311.py", "run_common455.py", "run_common710.py", "inputs_final.py", "inputs_final_v2.py")}
cases["wired: no /private/tmp path in the runner, run_common455 / 710, inputs_final(_v2)"] = all("/private/tmp" not in s for s in src.values())
rp = src["run_peraction311.py"]
cases["wired: OWN_PACKS (a silent {} at import) is gone; own_packs() reads ownfam/packs311.json through copies"] = (
    "OWN_PACKS" not in rp and 'return json.loads(CP.read(f"{OWNF}/packs311.json"))' in rp and "own_packs()" in rp)
cases["wired: the v2 manifest-copy shas come from the index (CP.names) and an empty listing STOPs"] = (
    'CP.names(V2MAN, "simh311_*.json")' in rp and 'raise SystemExit("v2 roots: no simh311_* manifest copy in the copies index (STOP)")' in rp
    and "V2MAN.glob(" not in rp)
cases["wired: the --decisive man455 reads STOP when missing (no `if exists() else []`)"] = "if mf.exists() else" not in rp and "if mf_.exists() else" not in rp
cases["wired: IF's man455 coverage comes from CP.names and needs exactly 24"] = (
    'CP.names("man455", "*__sol.json")' in src["inputs_final.py"] and "if len(sol455) != 24:" in src["inputs_final.py"])
cases["wired (B): main takes copies_root and sets it right after the launch-from-pickle branch; make_inputs_final reads copies via CP"] = (
    'pw_calib_res: str = "", copies_root: str = "", routec_zero_unit' in rp and "    if copies_root:" in rp and "CP.set_root(copies_root)" in rp
    and rp.index("return launch_from(launch_from_pickle, launch_diff, dict(locals()))") < rp.index("CP.set_root(copies_root)")
    and "IF.CP.read(IF.copy_path(r, k))" in (HERE / "make_inputs_final.py").read_text() and ".read_bytes()" not in
    (HERE / "make_inputs_final.py").read_text().split("manifest_copies")[1].split("\n")[0])
cases["wired: the runner's image ships copies.py and PW_CODE_MODULES lists it"] = (
    '.add_local_file(HERE / "copies.py", "/root/copies.py")' in rp and '"inputs_final_v2.py", "copies.py")' in rp)

for name, ok in cases.items():
    print(("PASS " if ok else "FAIL ") + name)
print(f"{sum(cases.values())} of {len(cases)} checks pass")
sys.exit(0 if all(cases.values()) else 1)
