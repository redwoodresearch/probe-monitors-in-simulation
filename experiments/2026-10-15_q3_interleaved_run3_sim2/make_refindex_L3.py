"""refindex_L3 (d9 ~21:55; 09's pin ruling ~21:58): the reference index for S_paper493 L3 (the 38 cells whose main has no pinned
reference in L1/L2: 37 new-main cells + t699_model_registry) and for the sim-honest fork's g5 (its keys are L3 cell keys).
Keys, paths and sha256s only; no outcome, approach or .eval content read.

Rule (09, unchanged from L1/L2): a cell's FORK PIN is its main's pinned honest_a reference in the origin's registered launch, i.e.
the S_paper493 v2 real launch's own launch.json "references" entry for the main's honest key, which the launcher selects by its
own usable-call count (an unusable counted attempt is replaced by the next usable reference rerun in the launcher's order). A main
whose launch recorded no usable reference gets no pin and is listed under "no_usable_reference".
Inputs:
- cells: facts v2 (aswin/q3-sp493 @ d800edc6a) minus the 455 cells pinned in results/sp493_sim_refpins_check_2026-10-15.json;
- main per cell: the committed task entries @ 27c327b03 (tasks_sp493_box{A,B,C}.json);
- honest key per main: the as-run honest entries @ 24e7dc722 (the 37 keys, each with its box);
- references: results/sp493_l3_refreads_2026-10-15/refs_{A,B,C}.txt (read on scaleup4 / xbox1 / realhonest ~21:55: launch.json
  sha256, then per key: copy, recorded sha256, the copy's sha256 on the box, reference_state).
  python make_refindex_L3.py
"""

import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "config/sp493sim/refindex_L3.json"
READS = HERE / "results/sp493_l3_refreads_2026-10-15"
B = "experiments/2026-10-15_q3_sp493"
RUNS = {"A": "/home/aswin/phase2/runs_sp493/2026-10-15_181326_sp493_real_A", "B": "/home/aswin/phase2/runs_sp493/2026-10-15_181403_sp493_real_B",
        "C": "/home/aswin/phase2/runs_sp493/2026-10-15_181249_sp493_real_C"}
BOXES = {"A": "scaleup4 i-0ad4140e97b11d30f", "B": "xbox1 i-04331f272d6778df0", "C": "realhonest i-0abea93b8ad99ee41"}


def git_json(commit, path):
    return json.loads(subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, check=True, cwd=HERE).stdout)


def main():
    assert not OUT.exists(), f"{OUT} exists"
    facts = {r["key"]: r for r in git_json("d800edc6a", f"{B}/results/sp493_facts_2026-10-15_165730.json")["rows"]}
    pinned = {r["key"] for r in json.loads((HERE / "results/sp493_sim_refpins_check_2026-10-15.json").read_text())["rows"] if r["status"] == "pinned"}
    cells = sorted(set(facts) - pinned)
    assert len(facts) == 493 and len(pinned) == 455 and len(cells) == 38, (len(facts), len(pinned), len(cells))
    entries = {c["key"]: c for b in "ABC" for c in git_json("27c327b03", f"{B}/config/tasks_sp493_box{b}.json")["cells"]}
    asrun = git_json("24e7dc722", f"{B}/results/asrun_honest_entries_2026-10-15_192615.json")["entries"]
    hk_of_main = {}
    for k, e in asrun.items():
        assert e["main"] == entries[k]["main"] and e["main"] not in hk_of_main, k
        hk_of_main[e["main"]] = (k, e["_box"])
    refs, launch_sha = {}, {}
    for b in "ABC":
        lines = (READS / f"refs_{b}.txt").read_text().splitlines()
        hdr = lines[0].split()
        assert hdr[0] == "HDR" and hdr[2] != "None", hdr  # each launch finished before the read
        launch_sha[b] = hdr[1]
        for ln in lines[1:]:
            _, k, copy, rec, onbox, state = ln.split()[:6]
            assert k not in refs and copy.startswith(RUNS[b] + "/references/"), ln
            refs[k] = {"box": b, "copy": copy, "recorded": rec, "onbox": onbox, "state": state}
    out_refs, none = {}, {}
    for c in cells:
        hk, b = hk_of_main[entries[c]["main"]]
        r = refs.get(hk)
        if r is None or r["state"] != "usable":
            none[c] = {"honest_key": hk, "box": b, "reason": "the launch recorded no usable reference for this main" if r is None else f"reference_state {r['state']}"}
            continue
        assert r["box"] == b and r["recorded"] == r["onbox"] and len(r["recorded"]) == 64, (c, r)
        out_refs[c] = {"copy": r["copy"], "sha256": r["recorded"], "origin_key": hk, "source_launch": RUNS[b], "source_box": BOXES[b],
                       "source_launch_json_sha256": launch_sha[b]}
    assert len(out_refs) + len(none) == 38
    OUT.write_text(json.dumps({"launch": "sp493_sim_refindex_L3", "finished": "2026-10-15T21:55:00-07:00",
                               "what": "reference index for --references-from (S_paper493 L3 + g5): per cell key, its main's pinned honest_a "
                                       "reference in the S_paper493 v2 real launch (that launch's own usable reference); make_refindex_L3.py",
                               "references": out_refs, "no_usable_reference": none}, indent=1) + "\n")
    print(json.dumps({"cells": len(cells), "pinned": len(out_refs), "no_usable_reference": none,
                      "launch_json_sha256": launch_sha, "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()}, indent=1))


if __name__ == "__main__":
    main()
