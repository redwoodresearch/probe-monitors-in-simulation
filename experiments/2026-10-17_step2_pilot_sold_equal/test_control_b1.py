"""Synthetic test of control_b1.py's 09 rebind (10-17): fabricated bytes only, served by a stub of its git() (no real record is read,
no worktree is made). The driver's bytes are this branch's HEAD blob (--worktree: the working-tree file, to test before a commit).
Each case asserts the exact STOP; the good case must pass every rebind gate and reach the worktree step (a sentinel).
- good: the diff record's pins equal the literals; the adapter's file sha and its inventory pin match; the INVENTORY pin and bytes match;
- (iii) the adapter file's sha256 differs; (i) the adapter's inventory pin sha256 differs; (ii) the INVENTORY bytes differ, or its pin;
- the diff record differs from its pinned sha256; the diff record's pins differ from the literals; its scope_verdict is not IDENTICAL;
- the FIX_V1 or ADAPTER literal mutated (so the diff record's pins no longer equal it);
- past the worktree step, with a fabricated pilot-only check copy that prints a value line: EQUAL, and the copy's print never reaches
  the driver's stdout (the redirect); a copy whose committed REQUIRED is not exactly its PILOT_RECORD STOPs.

  uv run --no-project --python 3.12 python experiments/2026-10-17_step2_pilot_sold_equal/test_control_b1.py [--worktree]
"""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REAL = HERE.parents[1]
REL = "experiments/2026-10-17_step2_pilot_sold_equal"
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731


class Reached(Exception):
    pass


def main(worktree=False):
    src = (HERE / "control_b1.py").read_bytes() if worktree else \
        subprocess.run(["git", "-C", str(REAL), "show", f"HEAD:{REL}/control_b1.py"], capture_output=True, check=True).stdout
    d = Path(tempfile.mkdtemp(prefix="control_b1_test_"))
    (d / "control_b1.py").write_bytes(src)
    spec = importlib.util.spec_from_file_location("control_b1_under_test", d / "control_b1.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    note = json.dumps({"sold": [{"record": f"{m.DIR}/{m.B1['record']} @ x", "message": "m", "epochs": 3, "checks": {"a": True}}]}).encode()
    inv = json.dumps({"runs": [{"key": "fabricated"}]}).encode()
    v2 = sha(b"fabricated inventory_b_v2")
    adapter = json.dumps({"pins": {"inventory": {"path": "inventory_b_v2.json", "sha256": v2}}}).encode()

    copy_ok = ('''import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
INVENTORY = None
PILOT_RECORD = "sold__simh_q4f_v1_pilot_cutfull.json"
REQUIRED = {PILOT_RECORD: "x"}
def main():
    print("SECRET 0.97 (a value line the copy prints)")
    (HERE / "results").mkdir(exist_ok=True)
    row = {"record": "%s @ x", "message": "m", "epochs": 3, "checks": {"a": True}}
    (HERE / "results" / "cutfull_sold_equality_note_2026-01-01_000000.json").write_text(json.dumps({"sold": [row], "pass": True, "mismatches": []}))
''' % f"{m.DIR}/{m.B1['record']}").encode()

    def setup(adapter_b=adapter, inv_b=inv, inv_pin=None, diff_pins=None, diff_sha=None, verdict="IDENTICAL", copy_src=None):
        pins = diff_pins or {"v2": {"sha256": v2}, "fix_v1": {"sha256": sha(inv)}, "b1_adapter": {"sha256": sha(adapter)}}
        diff = json.dumps({"pins": pins, "scope_verdict": verdict}).encode()
        m.NOTE = (m.NOTE[0], sha(note))
        m.DIFF = (m.DIFF[0], m.DIFF[1], diff_sha or sha(diff))
        m.V2_SHA, m.FIX_V1_SHA, m.ADAPTER_SHA = v2, sha(inv), sha(adapter)
        m.B1 = dict(m.B1, inventory=(m.B1["inventory"][0], m.B1["inventory"][1], inv_pin or sha(inv)))
        served = {f"{m.S3[0]}:{m.NOTE[0]}": note, f"{m.DIFF[0]}:{m.DIFF[1]}": diff, f"{m.S3[0]}:{m.B1_ADAPTER}": adapter_b,
                  f"{m.B1['inventory'][0]}:{m.B1['inventory'][1]}": inv_b}

        def git(*a, cwd=None, text=False):
            if a[0] == "show" and a[1] in served:
                return served[a[1]]
            if a[0] == "show" and a[1].startswith("HEAD:"):
                return copy_src if (copy_src is not None and a[1].endswith("check_sold_equality.py")) else b"# not used before the worktree step\n"
            if a[0] == "worktree" and copy_src is None:
                raise Reached()
            if a[0] == "worktree" and a[1] == "add":
                Path(a[3]).mkdir(parents=True)
                return b""
            if a[0] == "worktree" and a[1] == "remove":
                shutil.rmtree(a[3])
                return b""
            if a[0] == "rev-parse":
                return "0" * 40 if text else b"0" * 40
            raise AssertionError(f"unexpected git call {a}")
        m.git = git

    def run(literal=None, **kw):
        setup(**kw)
        if literal:
            setattr(m, literal, "f" * 64)
        try:
            m.main(str(d))
        except Reached:
            return "REACHED"
        except SystemExit as e:
            return str(e)
        return "RETURNED"

    other = json.dumps({"pins": {"inventory": {"path": "inventory_b_v2.json", "sha256": "0" * 64}}}).encode()
    cases = {
        "good: every rebind gate passes and the worktree step is reached": (run(), "REACHED"),
        "(iii) the adapter file's sha256 differs": (run(adapter_b=adapter + b" "),
                                                    "STOP (09 rebind iii): b1's adapter file sha256 is not the diff record's pins.b1_adapter.sha256"),
        "(ii) the INVENTORY bytes differ": (run(inv_b=inv + b" "),
                                            "STOP (09 rebind ii): the in-process INVENTORY's sha256 is not the diff record's pins.fix_v1.sha256"),
        "(ii) the INVENTORY pin differs": (run(inv_pin="0" * 64),
                                           "STOP (09 rebind ii): the in-process INVENTORY's sha256 is not the diff record's pins.fix_v1.sha256"),
        "the diff record differs from its pinned sha256": (run(diff_sha="0" * 64),
                                                           "STOP (09 rebind): the inventory diff record differs from its pin, or its pins.v2/fix_v1/b1_adapter sha256 are not the literals"),
        "the diff record's v2 pin is not the literal": (run(diff_pins={"v2": {"sha256": "0" * 64}, "fix_v1": {"sha256": sha(inv)}, "b1_adapter": {"sha256": sha(adapter)}}),
                                                        "STOP (09 rebind): the inventory diff record differs from its pin, or its pins.v2/fix_v1/b1_adapter sha256 are not the literals"),
        "the diff record's scope_verdict is DIFFERENT": (run(verdict="DIFFERENT"), "STOP (09 rebind): the inventory diff record's scope_verdict is not IDENTICAL"),
        "the FIX_V1_SHA literal mutated": (run(literal="FIX_V1_SHA"),
                                           "STOP (09 rebind): the inventory diff record differs from its pin, or its pins.v2/fix_v1/b1_adapter sha256 are not the literals"),
        "the ADAPTER_SHA literal mutated": (run(literal="ADAPTER_SHA"),
                                            "STOP (09 rebind): the inventory diff record differs from its pin, or its pins.v2/fix_v1/b1_adapter sha256 are not the literals"),
    }
    cap = io.StringIO()
    with contextlib.redirect_stdout(cap):
        got_full = run(copy_src=copy_ok)
    out_ = cap.getvalue()
    cases["past the worktree step: EQUAL (no exit), and the copy's value line never reaches the driver's stdout"] = (
        "EQUAL" if got_full == "RETURNED" and out_.splitlines()[-1:] == ["EQUAL"] and "SECRET" not in out_ else f"{got_full} / {out_[-200:]!r}", "EQUAL")
    bad_req = copy_ok.replace(b'REQUIRED = {PILOT_RECORD: "x"}', b'REQUIRED = {PILOT_RECORD: "x", "sold__other.json": "y"}')
    cases["a check copy whose committed REQUIRED is not exactly its PILOT_RECORD"] = (
        run(copy_src=bad_req), "STOP: the check copy's committed REQUIRED is not exactly its PILOT_RECORD")
    # (i): an adapter whose inventory pin differs; its file sha is then the literal (so gate (iii) passes and gate (i) must STOP)
    setup()
    m.ADAPTER_SHA = sha(other)
    pins = {"v2": {"sha256": v2}, "fix_v1": {"sha256": sha(inv)}, "b1_adapter": {"sha256": sha(other)}}
    diff = json.dumps({"pins": pins, "scope_verdict": "IDENTICAL"}).encode()
    m.DIFF = (m.DIFF[0], m.DIFF[1], sha(diff))
    served_git = m.git

    def git_i(*a, cwd=None, text=False):
        if a[0] == "show" and a[1] == f"{m.DIFF[0]}:{m.DIFF[1]}":
            return diff
        if a[0] == "show" and a[1] == f"{m.S3[0]}:{m.B1_ADAPTER}":
            return other
        return served_git(*a, cwd=cwd, text=text)
    m.git = git_i
    try:
        m.main(str(d))
        got = "RETURNED"
    except Reached:
        got = "REACHED"
    except SystemExit as e:
        got = str(e)
    cases["(i) the adapter's inventory pin sha256 differs (its file sha matching)"] = (
        got, "STOP (09 rebind i): b1's adapter's inventory pin sha256 is not the diff record's pins.v2.sha256")
    for k, (got_, want) in cases.items():
        ok = got_ == want
        print(f"  {'PASS' if ok else 'FAIL'}  {k}: {want}")
        if not ok:
            print("     got:", got_)
    print("ALL PASS" if all(g == w for g, w in cases.values()) else "FAILED")
    if not all(g == w for g, w in cases.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worktree", action="store_true", help="test the working-tree control_b1.py instead of the HEAD blob")
    main(ap.parse_args().worktree)
