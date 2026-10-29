"""A corrected IL final run table in the snapshot tables' layout (4f via d9, 10-15 ~03:20): the rows chain.py's il_final_rows writes
for a final batch, from the chain's own records for that batch (state, boxfiles record, committed manifests), as a NEW file beside the
old one (append-only). Keys and timestamps only.
Run: python make_il_final_table.py --state <chain_run_state dir> --batch il-s2 --out-name run_table_ilnm_s2_v2.json --old run_table_ilnm_s2.json"""
import argparse
import json
from pathlib import Path

import chain as C


class _St:
    def __init__(self, d: Path):
        self.d = d
        self.s = json.loads((d / "state.json").read_text())


def main():
    ap = argparse.ArgumentParser()
    for a in ("--state", "--batch", "--out-name", "--old"):
        ap.add_argument(a, required=True)
    a = ap.parse_args()
    st = _St(Path(a.state))
    b = st.s["batches"][a.batch]
    fam_b = [b, *C.snaps_of(st, b)]
    box_ = json.loads((st.d / f"boxfiles_{b['id']}.json").read_text())
    pauses = [(x[:19], x) for x in box_.get("pause_events", [])]
    starts = {}
    for x in fam_b:
        if not x.get("commit_targets"):
            continue
        for t, tv in C.targets_for(x).items():
            for f in json.loads((C.HERE / "config" / tv["manifest"]).read_text())["files"]:
                if f["world"] in ("run3", "sim2") and f.get("first_start"):
                    starts[(f["world"], f["task"])] = f["first_start"]

    def pause_at(t0):  # as chain.py step_hygiene's
        if not t0:
            return None
        t0 = t0.replace("T", " ")[:19]
        before = [line for ts, line in pauses if ts <= t0]
        return before[-1] if before else "none"

    common = {"launch_id": C.bare_id(b["run"]), "main_launch": b["main_run"], "interleaved_launch": True, "sim_attempt_timeout_s": 300}
    old_p = C.HERE / "results" / "hygiene" / a.old
    old = json.loads(old_p.read_text())
    rows = C.il_final_rows(b, fam_b, box_, starts, pause_at, common, C.il_no_eval(st, fam_b))
    doc = {"launch": b["run"], "pool": b["pool"], "staged_sha": old["staged_sha"], "layout": "world-keyed (as the snapshot tables)", "cells": rows,
           "staged_sets": old.get("staged_sets"), "supersedes": a.old}
    out = C.HERE / "results" / "hygiene" / a.out_name
    out.open("x").write(json.dumps(doc, indent=1) + "\n")
    (C.HERE / "results" / "hygiene" / (Path(a.old).stem + ".SUPERSEDED.md")).open("x").write(
        f"`{a.old}` is SUPERSEDED by `{a.out_name}`: the final table uses the snapshot tables' layout (world-keyed rows `run3|<cell>` / "
        "`sim2|<cell>`, each run's own first_start and pause_gate_at_start) and marks a counted attempt without an .eval as \"counted, no .eval (<launcher state>)\", "
        "never \"ran\". The old file stays unchanged (append-only).\n")
    print(json.dumps({"file": out.name, "rows": len(rows), "by_status": {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})},
                      "with_start": sum(bool(r["first_start"]) for r in rows), "pause_lines": len(pauses)}))


if __name__ == "__main__":
    main()
