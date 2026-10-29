"""Control (09 R-a, required) for the pilot-only check copy: pointed at ONE existing S_old record (A25 b1) with b1's pins set
in-process, the copy must reproduce the ORIGINAL check's per-record verdict for b1, i.e. its row in step3's committed NOTE
results/cutfull_sold_equality_note_2026-10-16_031017.json @ 969ee2e35. It compares keys and booleans only, and reads committed
records only. The copy's file does not change.

How it works:
- A throwaway detached worktree of origin/aswin/step3-retrain at 969ee2e35 holds b1's adapter and message as the original read them.
- This branch's committed copy (HEAD blob) is written there, at this folder's path, so REPO and HERE resolve as in use.
- b1's adapter (its bytes) goes into the copy's results/cutfull_b_msgs/.
- The copy is loaded with its two pins set to b1's:
  - INVENTORY = step3's INVENTORY_B (inventory_b_fix_v1 @ 09b070617);
  - REQUIRED = {b1's record: b1's original}.
- main() then runs with --records-commit 703ccbe0f3fb, the original's.
09's rebind (10-17), before anything else: (S_old path) nothing in the copy binds b1's adapter's own inventory pin, so three shas are
checked against the inventory diff record (DIFF @ e7ee4d6b6, its pins.*.sha256, and its scope_verdict == "IDENTICAL"), each a STOP
with an exact reason:
  (i) b1's adapter's inventory pin sha256 == pins.v2.sha256; (ii) the in-process INVENTORY (its pin and the bytes at its commit) ==
  pins.fix_v1.sha256; (iii) b1's adapter file's sha256 == pins.b1_adapter.sha256. The diff record's three pins must equal the literals.
The copy's own summary print is discarded (stdout redirected while it runs), so the log holds only this file's prints: the two b1
`checks` dicts (check name -> boolean) and EQUAL / DIFFERENT.
The verdicts and the comparison go to results/control_b1_<stamp>.json (write-once). The worktree is removed afterwards.

  uv run --no-project --python 3.12 python experiments/2026-10-17_step2_pilot_sold_equal/control_b1.py [--tmp <dir>]
"""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
REL = "experiments/2026-10-17_step2_pilot_sold_equal"
S3 = ("969ee2e35", "experiments/2026-10-15_step3_retrain")  # step3's commit holding the original NOTE, b1's adapter and message
NOTE = (f"{S3[1]}/results/cutfull_sold_equality_note_2026-10-16_031017.json", "b4b7f685630186b76d345153e687fd501b3eb3f31e3516236d59a00d6cd38aef")
B1_ADAPTER = f"{S3[1]}/results/cutfull_b_msgs/adapter_simh_q4f_v1_full_b1_cutfull_r2_2026-10-16_023636.json"
B1 = {"record": "sold__simh_q4f_v1_full_b1_cutfull_r2.json", "original": "simh_q4f_v1_full_b1__sim_honest", "records_commit": "703ccbe0f3fb",
      "inventory": ("09b070617", "experiments/2026-10-15_cut_sensitivity/results/inventory_b_fix_v1_2026-10-16_025058.json",
                    "4a5d0fb3cad32740f849df5def3c7a03a40a18aa4c42bc360709f0c740973d03")}  # = step3's INVENTORY_B, the pin the original check read
DIR = "experiments/2026-10-14_q3_overnight_chain/results/cutfull/routing"
# 09's rebind (10-17): the full shas, from the inventory diff record (read keys-only: pins.v2.sha256, pins.fix_v1.sha256, pins.b1_adapter.sha256)
DIFF = ("e7ee4d6b6", "experiments/2026-10-14_q3_overnight_chain/results/cutfull/inventory_v2_vs_fix_v1_b1_scope_2026-10-17_033715.json",
        "9c080a16be4138522ca47fb7ba84c1f7defb80566af9cfd946729d152684d598")
V2_SHA = "fc0e84ea5260193f7e5f249f022b2f4b04de544794e06a7137d777c222d5bbf8"  # DIFF pins.v2.sha256 (inventory_b_v2 @ a0bf963a7)
FIX_V1_SHA = "4a5d0fb3cad32740f849df5def3c7a03a40a18aa4c42bc360709f0c740973d03"  # DIFF pins.fix_v1.sha256 (inventory_b_fix_v1 @ 09b070617)
ADAPTER_SHA = "dd6c0506d0ec04b7ab0ce3495104678818353914643326e623383cf6e199c415"  # DIFF pins.b1_adapter.sha256
APRIME = ("fe4e44306", "experiments/2026-10-14_q3_overnight_chain/results/cutfull/aprime_b1_unit_counts_2026-10-17_052052.json")  # 09-verified; cited, never read


def git(*a, cwd=REPO, text=False):
    return subprocess.run(["git", "-C", str(cwd), *a], capture_output=True, text=text, check=True).stdout


def main(tmp):
    note = git("show", f"{S3[0]}:{NOTE[0]}")
    if hashlib.sha256(note).hexdigest() != NOTE[1]:
        raise SystemExit("the original NOTE differs from its pin")
    orig = [r for r in json.loads(note)["sold"] if r["record"].split(" @ ")[0] == f"{DIR}/{B1['record']}"]
    diff = git("show", f"{DIFF[0]}:{DIFF[1]}")  # a check record of keys and shas: only its three pins.*.sha256 are read
    dp = json.loads(diff).get("pins") or {} if hashlib.sha256(diff).hexdigest() == DIFF[2] else None
    if dp is None or [(dp.get(k) or {}).get("sha256") for k in ("v2", "fix_v1", "b1_adapter")] != [V2_SHA, FIX_V1_SHA, ADAPTER_SHA]:
        raise SystemExit("STOP (09 rebind): the inventory diff record differs from its pin, or its pins.v2/fix_v1/b1_adapter sha256 are not the literals")
    if json.loads(diff).get("scope_verdict") != "IDENTICAL":  # keys-only: the diff's own verdict on b1's scope (e7ee4d6b6)
        raise SystemExit("STOP (09 rebind): the inventory diff record's scope_verdict is not IDENTICAL")
    ab = git("show", f"{S3[0]}:{B1_ADAPTER}")
    if hashlib.sha256(ab).hexdigest() != ADAPTER_SHA:
        raise SystemExit("STOP (09 rebind iii): b1's adapter file sha256 is not the diff record's pins.b1_adapter.sha256")
    if ((json.loads(ab).get("pins") or {}).get("inventory") or {}).get("sha256") != V2_SHA:
        raise SystemExit("STOP (09 rebind i): b1's adapter's inventory pin sha256 is not the diff record's pins.v2.sha256")
    if B1["inventory"][2] != FIX_V1_SHA or hashlib.sha256(git("show", f"{B1['inventory'][0]}:{B1['inventory'][1]}")).hexdigest() != FIX_V1_SHA:
        raise SystemExit("STOP (09 rebind ii): the in-process INVENTORY's sha256 is not the diff record's pins.fix_v1.sha256")
    copy_src = git("show", f"HEAD:{REL}/check_sold_equality.py")
    build_src = git("show", f"HEAD:{REL}/build_cutfull_message.py")  # the check compares its pin literal with the build copy's (both committed as None)
    wt = Path(tempfile.mkdtemp(prefix="control_b1_", dir=tmp)) / "wt"
    git("worktree", "add", "--detach", str(wt), S3[0])
    try:
        d = wt / REL
        (d / "results" / "cutfull_b_msgs").mkdir(parents=True)
        (d / "check_sold_equality.py").write_bytes(copy_src)
        (d / "build_cutfull_message.py").write_bytes(build_src)
        (d / "results" / "cutfull_b_msgs" / Path(B1_ADAPTER).name).write_bytes(ab)  # the sha-checked bytes
        spec = importlib.util.spec_from_file_location("check_sold_equality_copy", d / "check_sold_equality.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        if set(getattr(m, "REQUIRED", {})) != {getattr(m, "PILOT_RECORD", None)}:  # the committed copy is the pilot-only one
            raise SystemExit("STOP: the check copy's committed REQUIRED is not exactly its PILOT_RECORD")
        m.INVENTORY, m.REQUIRED = B1["inventory"], {B1["record"]: B1["original"]}  # b1's pins, in-process
        sys.argv = ["check_sold_equality.py", "--records-commit", B1["records_commit"]]
        exit_ = None
        try:
            with contextlib.redirect_stdout(io.StringIO()):  # the copy's summary (counts, mismatch strings) stays out of the log
                m.main()
        except SystemExit as e:
            exit_ = str(e)
        notes = sorted((d / "results").glob("cutfull_sold_equality_note_*.json"))
        out = json.loads(notes[-1].read_text()) if len(notes) == 1 else {}
    finally:
        git("worktree", "remove", "--force", str(wt))
    copy = [r for r in out.get("sold", []) if r["record"].split(" @ ")[0] == f"{DIR}/{B1['record']}"]
    view = lambda r: {"record": r["record"], "message": r["message"], "epochs": r["epochs"], "checks": r["checks"]}  # noqa: E731
    res = {"what": "control: the pilot-only check copy on b1 vs the original check's b1 row (keys and booleans only)",
           "rebind_09": {"diff_record": f"{DIFF[1]} @ {DIFF[0]}", "diff_record_sha256": DIFF[2], "aprime_record": f"{APRIME[1]} @ {APRIME[0]}",
                         "checked": {"b1_adapter_sha256": ADAPTER_SHA, "b1_adapter_inventory_pin_sha256": V2_SHA, "inventory_sha256": FIX_V1_SHA}},
           "copy_blob": git("rev-parse", f"HEAD:{REL}/check_sold_equality.py", text=True).strip(), "git": git("rev-parse", "HEAD", text=True).strip(),
           "original": {"note": f"{NOTE[0]} @ {S3[0]}", "note_sha256": NOTE[1], "rows_for_b1": len(orig), "row": view(orig[0]) if len(orig) == 1 else None},
           "copy": {"rows_for_b1": len(copy), "row": view(copy[0]) if len(copy) == 1 else None, "pass": out.get("pass"), "mismatches": out.get("mismatches"),
                    "exit": exit_, "pins_set_in_process": {"INVENTORY": list(B1["inventory"]), "REQUIRED": {B1["record"]: B1["original"]}}}}
    res["equal"] = len(orig) == len(copy) == 1 and view(orig[0]) == view(copy[0])
    res["copy_clean"] = out.get("pass") is True and exit_ is None and out.get("mismatches") == []  # the copy itself passed: a broken one-to-one filter cannot print EQUAL
    res["equal"] = res["equal"] and res["copy_clean"]
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dest = HERE / "results" / f"control_b1_{stamp}.json"
    dest.parent.mkdir(exist_ok=True)
    dest.open("x").write(json.dumps(res, indent=1))
    print(dest.name, "original b1 checks", res["original"]["row"] and res["original"]["row"]["checks"])
    print(dest.name, "copy b1 checks    ", res["copy"]["row"] and res["copy"]["row"]["checks"], "copy pass", res["copy"]["pass"])
    print("EQUAL" if res["equal"] else "DIFFERENT")
    if not res["equal"]:
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmp", default=None)
    main(ap.parse_args().tmp)
