"""ug B5/B6 inputs, index side (09's Append 32 format note §2, render donors with identity binding; d9 10-15 ~13:30): for
every registered ug cell, the il_refindex_ug entry, its copy's recomputed sha256, and the canonical main's own pinned
reference record, i.e. gate (a1)'s inputs. ON q3-xbox3, read-only, any python3; stdout captured on the laptop:
  DEVBOX=aswin@i-0f2299f29dd296259 bash devbox.sh 'python3 - ' < ug_b5_index_pin.py > <out.json>
Keys, paths, hashes, try numbers and attempt kinds only (no .eval is opened beyond hashing its bytes).

Per cell (the 149 of the ug task file, config/tasks_ug_pilot.json in the harness copy):
  - index: il_refindex_ug/launch.json references[cell] (copy, sha256, source, try, index_source, relayed_from);
  - copy: exists, its recomputed sha256, and whether it equals the index's;
  - main_pinned: the index_source launch.json's references[cell] (sha256, source, try). For an S_old cell the index_source
    is il_refindex_old (pinned 79acbfc1..., built with --check-files): its entry IS the main's pin, as census_il_gate_a.py
    took it for il_old; its own index_source (the T196 launch, a path relative to xbox1's ~/phase2, not relayed here) is
    recorded with that launch.json's sha256 from il_refindex_old's own `sources` list (no file opened);
  - source attempt kind from the source path (first try / redraw / reference rerun; check_il_gate_a.py's rule);
  - a1_index_equals_main: the ug index sha256 == the main's pinned sha256 (== the copy's recomputed sha256).
Fails (exit 1) on: a missing index, task file, main launch.json or reference entry (one fixed line); any input file whose
sha256 is not its frozen pin (PINS); a cell missing, duplicated or extra; a copy missing, unreadable or not equal to the
index sha256; (a1) failing. Rows are still printed for inspection; `problems` lists each failure by cell.
"""

import hashlib
import json
import os
import sys

HOME = "/home/aswin/phase2"
UG_INDEX = f"{HOME}/il_refindex_ug/launch.json"
TASKS = f"{HOME}/attack_check_ug/config/tasks_ug_pilot.json"
# the frozen pins (Append 34 manifest @ 71c308096; build_index_ug.sh; ug_box.sh TSHA): enforced, not just recorded (Codex r28)
PINS = {UG_INDEX: "36c2b2736908d7d6a028e101fe95cd1ff17da35343c0678d62b6b69ac5f6ad8f",
        TASKS: "cf89e595a37b85375ba1a84796718211d07b48a6e6f697c69a5faad72f76b80b",
        f"{HOME}/il_refindex_old/launch.json": "79acbfc1ef4523f9a57cc19dc876252159c6c1d62521001f1ac71e2644036940",
        f"{HOME}/refs_ug/q1_launch.json": "7e8227cd859502e60ad718fc60215410ece9d4b86eea04040bd7bee0a82c53ac",
        f"{HOME}/refs_ug/q2_launch.json": "fa489bf06d2be9c4a2b5c432ded7fb6065df49763618fb19e96a5f5d01cf03fb",
        f"{HOME}/refs_ug/q3_launch.json": "289ffb1030625f24666b1c9fb86a2ad577cf66a55407b9f18db125f0ed121701",
        f"{HOME}/refs_ug/q4_launch.json": "f12ed4cd86ed516036bcc47857c7a4d90ba95e21595a8c8f484653b9055a9ad6"}


def sha_file(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def kind(src):
    s = src or ""
    return "reference rerun" if "_rerun" in s else "redraw" if "_redraw" in s else "first try" if s else None


def entry(e):
    return {k: e.get(k) for k in ("copy", "sha256", "source", "try", "index_source", "relayed_from") if k in e}


def main():
    ug = json.load(open(UG_INDEX))
    refs = ug["references"]
    cells = [c["key"] for c in json.load(open(TASKS))["cells"]]
    problems = []
    if len(cells) != len(set(cells)) or len(cells) != 149:
        problems.append("task file: not 149 distinct cells")
    if set(refs) != set(cells):
        problems.append("il_refindex_ug references are not exactly the task file's cells")
    launches = {}

    def lj(p):
        if p not in launches:
            launches[p] = json.load(open(p))
        return launches[p]

    rows = []
    for cell in cells:
        e = refs.get(cell)
        if e is None:
            problems.append(f"{cell}: not in il_refindex_ug")
            continue
        copy = e.get("copy")
        try:
            csha = sha_file(copy) if copy and os.path.exists(copy) else None
            cstate = "ok" if csha else "missing"
        except OSError:
            csha, cstate = None, "unreadable"
        src_index = e.get("index_source")
        m = (lj(src_index).get("references") or {}).get(cell)
        if m is None:
            problems.append(f"{cell}: not in its index_source")
            continue
        row = {"cell": cell, "index": entry(e), "copy_state": cstate, "copy_sha256": csha,
               "copy_equals_index": csha is not None and csha == e.get("sha256"),
               "main_pinned": {"via": src_index, **{k: m.get(k) for k in ("sha256", "source", "try")}},
               "source_kind": kind(e.get("source"))}
        if m.get("index_source"):  # S_old: record the T196 launch the S_old index took it from, by its pinned sha256
            srcs = [x for x in lj(src_index).get("sources") or [] if x.get("launch_json") == m["index_source"]]
            if len(srcs) != 1:
                problems.append(f"{cell}: its S_old source launch is not exactly one entry of the index's sources")
                continue
            row["main_pinned_origin"] = {"launch_json": m["index_source"], "launch_json_sha256": srcs[0].get("sha256"),
                                         "source": m.get("source"), "try": m.get("try")}
        pinned = row["main_pinned"]["sha256"]
        row["a1_index_equals_main"] = bool(pinned) and e.get("sha256") == pinned and csha == pinned
        if cstate != "ok" or not row["copy_equals_index"]:
            problems.append(f"{cell}: copy {cstate}, or not equal to the index sha256")
        if not row["a1_index_equals_main"]:
            problems.append(f"{cell}: index sha256, main pin and copy disagree")
        rows.append(row)
    for path, pin in PINS.items():
        if (path in launches or path in (UG_INDEX, TASKS)) and sha_file(path) != pin:
            problems.append(f"{path}: sha256 is not its frozen pin")
    for path in launches:
        if path not in PINS:
            problems.append(f"{path}: an index source with no frozen pin")
    out = {"ug_index": {"path": UG_INDEX, "sha256": sha_file(UG_INDEX), "n_references": len(refs)},
           "tasks_file": {"path": TASKS, "sha256": sha_file(TASKS), "n_cells": len(cells)},
           "launch_jsons": {p: sha_file(p) for p in sorted(launches)}, "pins_enforced": PINS, "problems": problems,
           "rows": rows}
    print(json.dumps(out, indent=1))
    if problems or len(rows) != len(cells):
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:  # noqa: BLE001 - one fixed line, the class only
        print(json.dumps({"failed": type(ex).__name__}))
        sys.exit(1)
