"""Per-cell VALIDATION and per-arm supersession revision for one host's rlr3 (09's registration of Addendum 1 @
7bd9d5e89, items 3 and 4), after make_rerun3_files.py wrote results/il_rlr3_selection_<h>.json. For every selected cell,
across ALL launch names: (a) the input launch is the cell's CURRENTLY designated replacement launch (the rlr unless an
append-only supersession revision moved the cell, e.g. to rlr2); (b) both arms had 0 attempts there (from the
rlr_final_read.py record); (c) no replacement run exists (the rlr3 run dir is absent on the box: --rlr3-absent, read on
the box just before); (d) no earlier extra designation was used (the cell is in no supersession revision and no other
rlr3 selection). Any failure is an error: nothing is written. Writes results/il_rlr3_validation_<h>.json and the
append-only results/il_rlr3_supersession_<h>.json (retires the cell's rlr designation, names the rlr3 pair; both arms
authoritative even if never started, failed or unusable; originals and retired records stay in accounting).
  python3 validate_rlr3.py <h> --rlr3-absent yes|no
"""

import argparse
import glob
import json
from pathlib import Path

HERE = Path(__file__).parent
R = HERE / "results"
C = HERE / "config"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("h")
    ap.add_argument("--rlr3-absent", choices=("yes", "no"), required=True)
    a = ap.parse_args()
    h = a.h
    sel = json.loads((R / f"il_rlr3_selection_{h}.json").read_text())
    fin = json.loads((R / f"il_rlr_final_{Path(sel['rlr_launch']).name}.json").read_text())
    assert fin["launch_json_sha256"] == sel["rlr_launch_json_sha256"] == fin["sha_before"], "final read / selection mismatch"
    # every append-only designation record, globally by cell (Codex r24): rlr2 revisions, other hosts' rlr3 revisions and
    # selections, and the completion per-arm maps
    recs = sorted(glob.glob(str(R / "il_rlr*_supersession_*.json"))) + sorted(glob.glob(str(C / "completion" / "ilc_supersession_perarm_*.json")))
    designations = {}
    for p in recs:
        if p.endswith(f"il_rlr3_supersession_{h}.json"):
            continue
        for r in json.loads(Path(p).read_text())["rows"]:
            designations.setdefault(r["cell"], set()).add(r["authoritative"]["launch"])
    moved = {c: sorted(v)[0] for c, v in designations.items()}
    other_rlr3 = {c for p in glob.glob(str(R / "il_rlr3_selection_*.json")) if not p.endswith(f"_{h}.json")
                  for c in json.loads(Path(p).read_text())["selected"]}
    rows, bad = [], []
    for cell in sel["selected"]:
        designated = moved.get(cell, f"2026-10-15_il_{h}_rlr")
        st = fin["cells"][cell]
        checks = {"input_is_designated": Path(fin["run"]).name == designated,
                  "both_arms_zero_attempts": st["real"]["attempts"] == 0 and st["sim"]["attempts"] == 0
                  and st["real"]["state"] == st["sim"]["state"] == "not run",
                  "no_replacement_run": a.rlr3_absent == "yes",
                  "no_earlier_extra_designation": cell not in designations and cell not in other_rlr3}
        rows.append({"cell": cell, "designated_launch": designated, "input_launch": Path(fin["run"]).name, **checks})
        if not all(checks.values()):
            bad.append(cell)
    assert not bad, f"validation failed for {bad}: nothing written"
    v, s = R / f"il_rlr3_validation_{h}.json", R / f"il_rlr3_supersession_{h}.json"
    assert not v.exists() and not s.exists()
    v.write_text(json.dumps({"host": h, "rlr_launch": fin["run"], "rlr_launch_json_sha256": fin["launch_json_sha256"],
                             "rlr3_run_dir_absent_on_box": a.rlr3_absent == "yes", "rows": rows}, indent=1) + "\n")
    s.write_text(json.dumps({"revision": f"append-only supersession revision for {h}'s rlr3 (Addendum 1 @ 7bd9d5e89)",
                             "rows": [{"cell": c, "arm": arm, "key": f"{arm}|{c}", "host": h,
                                       "retired_designation": {"launch": Path(fin["run"]).name, "run_key": rk, "state": "not run, 0 attempts (spend cap)"},
                                       "authoritative": {"launch": f"2026-10-15_il_{h}_rlr3", "run_key": rk},
                                       "authoritative_even_if": "never started, failed or unusable"}
                                      for c in sel["selected"] for arm, rk in (("run3", f"attack/{c}"), ("sim2", f"sim_attack@september/{c}"))]},
                            indent=1) + "\n")
    print(json.dumps({"host": h, "validated": len(rows), "supersession_rows": 2 * len(rows)}))


if __name__ == "__main__":
    main()
