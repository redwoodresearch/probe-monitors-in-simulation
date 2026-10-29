"""Synthetic test of the D1 calibration files (no project data, no Modal): calib_npz_bytes gives identical bytes for equal arrays written at
different times; calib_arrays_sha equals across zip writers, differs on a last-bit change and on a 0-d vs 1-d shape; stored_calib picks the file
stored member rows name, else the first by name, records the rest, ignores partial files, and STOPs on mixed names or on a name with no complete
file. Usage: python test_calib_files.py <experiment dir> <scratch dir>"""
import io
import sys
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np

sys.path.insert(0, sys.argv[1])
import run_peraction311 as M  # noqa: E402

ok = []
sha = M.C.sha_b
arr = {"joint__B__mu": np.arange(12, dtype=np.float64).reshape(4, 3), "joint__0.01__B__T": np.array([0.5, np.nan, -0.0, 1.0]), "meta_json": np.array('{"a": 1}')}
b1 = M.calib_npz_bytes(arr)
time.sleep(2.1)
ok.append(("fixed bytes across time", M.calib_npz_bytes(arr) == b1))
buf = io.BytesIO()
np.savez(buf, **arr)
ok.append(("np.savez and fixed bytes: same decoded sha", M.calib_arrays_sha(buf.getvalue()) == M.calib_arrays_sha(b1)))
with np.load(io.BytesIO(b1)) as z:
    ok.append(("np.load reads it", sorted(z.files) == sorted(arr) and np.array_equal(z["joint__B__mu"], arr["joint__B__mu"])))
b3 = M.calib_npz_bytes(arr | {"joint__B__mu": np.nextafter(arr["joint__B__mu"], np.inf)})
ok.append(("last-bit difference: different decoded sha", M.calib_arrays_sha(b3) != M.calib_arrays_sha(b1)))
ok.append(("0-d vs 1-d: different decoded sha", M.calib_arrays_sha(M.calib_npz_bytes({"x": np.array(1.0)})) != M.calib_arrays_sha(M.calib_npz_bytes({"x": np.array([1.0])}))))
d = Path(tempfile.mkdtemp(dir=Path(sys.argv[2]).resolve())).resolve()  # absolute (stored_calib joins rel to /sol; Astra r7 #2)
rel = str(d / "lab__combo")
for b in (b1, b3):
    (d / f"lab__combo_{sha(b)[:16]}_{uuid.uuid4().hex[:8]}.npz").write_bytes(b)
(d / f"lab__combo_{'0' * 16}_{uuid.uuid4().hex[:8]}.npz").write_bytes(b1[:100])  # a partial file: ignored
first, other = sorted([sha(b1), sha(b3)], key=lambda s: s[:16])
pick, dup = M.stored_calib(rel)
ok.append(("no names: first by name, the other recorded, partial ignored", pick["sha256"] == first and [x["sha256"] for x in dup] == [other]))
ok.append(("stored rows' name wins", M.stored_calib(rel, {other})[0]["sha256"] == other))
for names, what in (({sha(b1), sha(b3)}, "mixed names STOP"), ({"f" * 64}, "a name with no complete file STOPs")):
    try:
        M.stored_calib(rel, names)
        ok.append((what, False))
    except SystemExit:
        ok.append((what, True))
(d / f"lab__combo_{first[:16]}_{uuid.uuid4().hex[:8]}.npz").write_bytes(b1 if sha(b1) == first else b3)  # a second complete copy of the pick
pick3, dup3 = M.stored_calib(rel)
ok.append(("identical copies: one calibration, the copy recorded as same_arrays", pick3["sha256"] == first and sum(x["same_arrays"] for x in dup3) == 1))
e = Path(tempfile.mkdtemp(dir=Path(sys.argv[2]).resolve())).resolve()
(e / f"lab__combo_{'0' * 16}_{uuid.uuid4().hex[:8]}.npz").write_bytes(b1[:100])  # only a partial file
try:
    M.stored_calib(str(e / "lab__combo"), {sha(b1)})
    ok.append(("named, only a partial file: STOP", False))
except SystemExit:
    ok.append(("named, only a partial file: STOP", True))
ok.append(("no names, no complete file: a fresh calibration", M.stored_calib(str(e / "lab__combo")) == (None, [])))
for n, v in ok:
    print("PASS" if v else "FAIL", n)
print(f"{sum(v for _, v in ok)}/{len(ok)} PASS" + ("" if all(v for _, v in ok) else "  FAILURES"))
sys.exit(0 if all(v for _, v in ok) else 1)
