"""Load-bearing checks for the two builders on fake fixtures: python3 test_interleave.py (no box, no network)."""

import hashlib
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from build_task_files import shuffled
from make_reference_index import merge
from stage_task_files import stage


def fails(fn, *a) -> bool:
    try:
        fn(*a)
    except AssertionError:
        return True
    return False


with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    files = {}
    for k in ("t1", "t2", "t3"):
        (d / f"{k}.eval").write_bytes(k.encode())
        files[k] = {
            "copy": str(d / f"{k}.eval"),
            "sha256": hashlib.sha256(k.encode()).hexdigest(),
        }
    a, b = d / "a.json", d / "b.json"
    a.write_text(json.dumps({"references": {"t1": files["t1"], "t2": files["t2"]}}))
    b.write_text(json.dumps({"references": {"t3": files["t3"]}}))
    idx = merge([(a, None), (b, None)], ["t1", "t2", "t3"], set(), True)
    assert sorted(idx["references"]) == ["t1", "t2", "t3"]
    assert fails(
        merge, [(a, None), (b, None)], ["t1", "t4"], set(), False
    )  # a wanted key in no source
    assert not fails(
        merge, [(a, None), (b, None)], ["t1", "t4"], {"t4"}, False
    )  # unless listed as expected
    assert fails(
        merge, [(a, None), (a, None)], ["t1"], set(), False
    )  # a key in two sources
    (d / "t2.eval").write_bytes(b"changed")
    assert fails(merge, [(a, None)], ["t2"], set(), True)  # sha256 mismatch on the box
    relay = d / "relay"
    relay.mkdir()
    (relay / "t3.eval").write_bytes(b"t3")
    r = merge([(b, relay)], ["t3"], set(), True)["references"]["t3"]
    assert (
        r["copy"] == str(relay / "t3.eval") and r["relayed_from"] == files["t3"]["copy"]
    )

cells = [{"key": f"c{i}", "env": "e1" if i < 6 else "e2"} for i in range(10)]
a, b = shuffled("s", "x", cells), shuffled("s", "y", cells)
assert (
    a == shuffled("s", "x", cells)
    and a != b
    and sorted(c["key"] for c in a) == sorted(c["key"] for c in cells)
)
assert fails(
    shuffled, "s", "x", [*cells, {"key": "c1_r2", "env": "e1", "real_arms": ["attack"]}]
)  # no r2 entries
boxes = {"q1": cells[:5], "old": cells[5:]}
st = stage(
    "s",
    4,
    boxes,
    {"q1": "S_new", "old": "S_old"},
    {f"c{i}" for i in range(8)},
    {"c8", "c9"},
    {"c7"},
)
assert sum(len(v["pilot"]) for v in st.values()) == 4 and {
    c["key"] for c in st["old"]["sel"]
} == {"c8", "c9"}
assert st["old"]["est"][:1] == [c for c in cells if c["key"] == "c7"] or "c7" in {
    c["key"] for c in st["old"]["pilot"]
}
assert fails(
    stage,
    "s",
    4,
    boxes,
    {"q1": "S_new", "old": "S_old"},
    {f"c{i}" for i in range(8)},
    {"c7"},
    set(),
)  # overlap
assert fails(
    stage, "s", 4, boxes, {"q1": "S_new", "old": "S_old"}, {"c1"}, {"c2"}, set()
)  # a cell in neither list
print("ok")
