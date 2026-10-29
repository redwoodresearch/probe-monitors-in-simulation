"""Lean test of s2_gate.jq (09's S2 marker gate, GATES Amendment 15) on SYNTHETIC pointers and markers: it runs jq -f on the committed
file, as rc1a / rc1b do, and never reads a NOTE. Cover mode: the entries, their order, the .json.gz sha rule and bad pointers. Gate mode:
the positive case, then one change at a time (09's list: wrong signoff, signed false, by not 09, a missing entry, an extra entry, a
duplicate entry; plus a COVER list that is not this run's inputs, an unsorted COVER list, and malformed markers).

  uv run --no-project --python 3.12 python -I test_s2_gate.py   (from this folder; needs jq on PATH)
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GATE = str(HERE / "s2_gate.jq")
bad = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        bad.append(name)


def jq(mode, ptrs, marker=None, cover=None):
    """-> (exit code, stdout stripped)."""
    cmd = ["jq", "-e", "-c", "-f", GATE, "--arg", "mode", mode] + (["--argjson", "cover", json.dumps(cover)] if cover is not None else [])
    cmd = (cmd[:1] + ["-n"] + cmd[1:] if marker is None else cmd) + ["--args", *ptrs]
    r = subprocess.run(cmd, input=None if marker is None else json.dumps(marker), capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


C = ["%040x" % (i + 1) for i in range(4)]
S = ["%064x" % (i + 1) for i in range(8)]
R = "experiments/2026-10-15_fpr_sweep_507_prep/results"
PTRS = [f"git:{C[0]}:xfer/routec/records/A_pooled.json.gz@{S[0]}",  # A's committed copy: decompressed sha
        f"git:{C[1]}:xfer/routec/records/A_pooled_LABEL.json@{S[1]}",
        f"git:{C[2]}:{R}/group_excess_2026-10-18_120000_{'a' * 32}.json@{S[2]}",
        f"git:{C[2]}:{R}/group_excess_2026-10-18_120000_{'a' * 32}_LABEL.json@{S[3]}",
        f"git:{C[3]}:{R}/group_catch_2026-10-18_120500.json@{S[4]}",
        f"git:{C[3]}:{R}/group_catch_2026-10-18_120500_LABEL.json@{S[5]}",
        f"git:{C[0]}:{R}/LABELS_171pool_2026-10-17.json@{S[6]}"]
WANT = sorted(p.split(":", 2)[2] for p in PTRS)

# cover mode
rc, out = jq("cover", list(reversed(PTRS)))
check("cover: the 7 pointers -> 7 entries '<path>@<sha256>', sorted, unique, whatever the argument order",
      rc == 0 and json.loads(out) == WANT and len(WANT) == 7)
check("cover: the .json.gz entry keeps the pointer's (decompressed) sha", f"xfer/routec/records/A_pooled.json.gz@{S[0]}" in json.loads(out))
for nm, ptrs in (("a vol: pointer", PTRS[:-1] + [f"vol:sol:xfer/x.json@{S[7]}"]), ("a 40-hex sha", PTRS[:-1] + [f"git:{C[0]}:a.json@{C[1]}"]),
                 ("a quote in the path", PTRS[:-1] + [f"git:{C[0]}:a'b.json@{S[7]}"]), ("a non-JSON path", PTRS[:-1] + [f"git:{C[0]}:a.txt@{S[7]}"]),
                 ("the same entry twice", PTRS + [f"git:{C[3]}:{PTRS[0].split(':', 2)[2]}"]), ("no pointer", [])):
    rc, out = jq("cover", ptrs)
    check(f"cover: {nm} is an error (exit 5, nothing printed)", rc == 5 and out == "")

# gate mode
MK = {"signoff": "S2", "signed": True, "by": "09", "covered": list(WANT), "time": "2026-10-18T12:00:00-07:00", "notes": "n", "for": "rc1a",
      "items": [1]}


def gate(marker, cover=WANT, ptrs=PTRS):
    return jq("gate", ptrs, marker, cover)


check("gate: the positive marker (with time, notes, for, items) -> true, exit 0", gate(MK) == (0, "true"))
check("gate: covered in another order -> true (set equality)", gate(MK | {"covered": list(reversed(WANT))}) == (0, "true"))
check("gate: the pointers in another order -> true", gate(MK, ptrs=list(reversed(PTRS))) == (0, "true"))
NEG = {"wrong signoff (S1)": MK | {"signoff": "S1"}, "signed false": MK | {"signed": False}, "signed the string 'true'": MK | {"signed": "true"},
       "by 0d": MK | {"by": "0d"}, "a missing entry": MK | {"covered": WANT[1:]}, "an extra entry": MK | {"covered": WANT + [f"x.json@{S[7]}"]},
       "a duplicate entry": MK | {"covered": WANT + [WANT[0]]}, "no covered": {k: v for k, v in MK.items() if k != "covered"},
       "covered not a list": MK | {"covered": ",".join(WANT)}, "no signed": {k: v for k, v in MK.items() if k != "signed"}}
for nm, mk in NEG.items():
    check(f"gate: {nm} -> false, exit 1", gate(mk) == (1, "false"))
check("gate: a marker that is a list -> false, exit 1", gate([MK]) == (1, "false"))
check("gate: the COVER list misses one of this run's inputs (marker matches the COVER) -> false",
      gate(MK | {"covered": WANT[1:]}, cover=WANT[1:]) == (1, "false"))
check("gate: the COVER list has an entry this run does not send (marker matches the COVER) -> false",
      gate(MK | {"covered": WANT + [f"x.json@{S[7]}"]}, cover=sorted(WANT + [f"x.json@{S[7]}"])) == (1, "false"))
check("gate: an unsorted COVER list -> false", gate(MK, cover=list(reversed(WANT))) == (1, "false"))
check("gate: a COVER entry with a 40-hex sha -> false", gate(MK | {"covered": WANT[:-1] + [f"x.json@{C[0]}"]},
                                                              cover=sorted(WANT[:-1] + [f"x.json@{C[0]}"])) == (1, "false"))
rc, out = jq("gate", PTRS, MK)
check("gate: no --argjson cover -> false (exit 1)", (rc, out) == (1, "false"))
rc, out = jq("other", PTRS, MK, WANT)
check("an unknown --arg mode is an error (exit 5)", rc == 5 and out == "")

print(f"{len(bad)} failed" if bad else "ALL PASS")
sys.exit(1 if bad else 0)
