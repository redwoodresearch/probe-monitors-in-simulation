"""Third paired rerun (rlr3) for ONE host, after its paired rerun launch (rlr or rlr2) ended (0d 04:08; addendum to
e46ba05d0; 09's never-started exception, d9 04:12): a registered rerun cell may be re-designated ONLY if BOTH its arms had
0 attempts in the designated launch (admission state alone: both "not run", sim2's why "spend cap: ..."), before any
replacement run exists, and at most ONE extra designation per cell across all launch names (so not s1, s2 or q4, whose
cells were already re-designated to rlr2). A cell whose run3 ran but whose sim2 was never admitted is NOT re-designated
(listed as such). Selection by RUN STATE ONLY, from that launch's launch.json at the sha256 recorded by rlr_final_read.py
(results/il_rlr_final_<launch>.json, committed by key before rlr3 starts). Same cells' ORIGINAL entries
(config/tasks_ilrr_<h>.json order), the same pinned failed-original bindings (config/ilrr_rerun_of_<h>.json @ f368ebab);
cap = round(cells x $14.6079 + min(cells, 5) x $75). Writes config/tasks_ilrr3_<h>.json, config/ilrr3_rerun_of_<h>.json,
config/pilot_ilrr3_<h>.yaml and ilrr3_<h>_box.sh (ilrr_box.sh @ 307fc2633 with
mechanical renames: run dir 2026-10-15_il_<h>_rlr3, session ilrlr3<h>, rlr3 file names; one case). Never overwrites;
with 0 selected cells it writes only the selection record and no launch files.
  python3 make_rerun3_files.py <case key> <rlr final json> <prevs on the box, comma-separated>
"""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
C = HERE / "config"
PRICE, RESERVE, BOX_COMMIT = 14.6079, 75.0, "307fc2633f36da743d6b1218ddeda87e0e8c231a"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731


def main() -> None:
    key, final, prevs = sys.argv[1], Path(sys.argv[2]), sys.argv[3].split(",")
    fin = json.loads(final.read_text())
    assert fin["finished"] and len(fin["launch_json_sha256"]) == 64 and fin["sha_before"] == fin["launch_json_sha256"], "final read"
    box = subprocess.check_output(["git", "show", f"{BOX_COMMIT}:experiments/2026-10-15_q3_interleaved_run3_sim2/ilrr_box.sh"], text=True)
    line = next(x for x in box.splitlines() if x.startswith(f"  {key}) "))
    h = re.search(r"HF=(\w+);", line).group(1) if "HF=" in line else key
    assert h not in ("s1", "s2", "q4"), f"{h}: its cells already have their one extra designation (rlr2)"
    rr = json.loads((C / f"ilrr_rerun_of_{h}.json").read_text())
    reg = [b["cell"] for b in rr["rerun_of"]]
    sel, excluded = [], {}
    for cell in reg:
        c = fin["cells"].get(cell)
        assert c is not None, (cell, "a registered rerun cell missing from the rlr launch")
        s, r = c["sim"], c["real"]
        both_zero = s["state"] == r["state"] == "not run" and s["attempts"] == r["attempts"] == 0
        if both_zero and str(s.get("why", "")).startswith("spend cap"):
            sel.append(cell)
        elif s["state"] == "not run":
            excluded[cell] = ("sim2 not admitted but run3 ran: not re-designatable (09)" if not both_zero
                              else f"both arms not run for another reason: {str(s.get('why', ''))[:60]}")
    record = {"host": h, "case": key, "rlr_launch": fin["run"], "rlr_launch_json_sha256": fin["launch_json_sha256"],
              "rule": "registered rerun cells with BOTH arms 'not run', 0 attempts, sim2 why 'spend cap...' (run state only; one extra designation per cell)",
              "selected": sel, "excluded_not_run_other": excluded}
    rec = HERE / "results" / f"il_rlr3_selection_{h}.json"
    assert not rec.exists(), rec
    rec.write_text(json.dumps(record, indent=1) + "\n")
    if not sel:
        print(json.dumps({"host": h, "selected": 0}))
        return
    doc = json.loads((C / f"tasks_ilrr_{h}.json").read_text())
    cells = [c for c in doc["cells"] if c["key"] in set(sel)]
    assert {c["key"] for c in cells} == set(sel)
    cap = round(len(cells) * PRICE + min(len(cells), 5) * RESERVE)
    tp, mp, cp = C / f"tasks_ilrr3_{h}.json", C / f"ilrr3_rerun_of_{h}.json", C / f"pilot_ilrr3_{h}.yaml"
    for p in (tp, mp, cp):
        assert not p.exists(), p
    tp.write_text(json.dumps(doc | {"rerun3_of_launch": fin["run"], "cells": cells}, indent=1) + "\n")
    mp.write_text(json.dumps(rr | {"rerun_of": [b for b in rr["rerun_of"] if b["cell"] in set(sel)]}, indent=1) + "\n")
    text = (C / f"pilot_ilrr_{h}.yaml").read_text()
    n = 0
    for pat, rep in ((r"(?m)^tasks: .*$", f"tasks: config/tasks_ilrr3_{h}.json   # third paired rerun: {len(cells)} cells"),
                     (r"(?m)^  cap_usd: .*$", f"  cap_usd: {cap}   # {len(cells)} cells x ${PRICE} + {min(len(cells), 5)} x ${RESERVE:.0f} admission reserve"),
                     (r"(?m)^  worst_case_limit_usd: .*$", f"  worst_case_limit_usd: {cap}   # = cap")):
        text, k = re.subn(pat, rep, text)
        n += k
    assert n == 3
    cp.write_text(f"# THIRD PAIRED RATE-LIMIT RERUN {h} (rlr3): pilot_ilrr_{h}.yaml with only tasks and caps changed.\n" + text)
    new = re.sub(r"CAP=\d+;", f"CAP={cap};", line)
    for k, v in (("TSHA", sha(tp)), ("CSHA", sha(cp)), ("MSHA", sha(mp))):
        new, m = re.subn(rf"\b{k}=[0-9a-f]{{64}};", f"{k}={v};", new)
        assert m == 1, k
    new = re.sub(r'PREVS="[^"]*";', f'PREVS="{" ".join(prevs)}";', new)
    i0, i1 = box.index("case $H in\n") + len("case $H in\n"), box.index("  *) echo \"host $H: no rerun prepared\"")
    box = box[:i0] + new + "\n" + box[i1:]
    # the freeze exclusion first (it must still exclude the first rerun's files), then the mechanical renames
    excl = "! -path './config/pilot_ilrr_*' ! -path './config/tasks_ilrr_*' ! -path './config/ilrr_*'"
    assert box.count(excl) == 1
    box = box.replace(excl, "@@EXCL@@")
    for a, b, k in [
        ("# GENERATED by make_rerun_files.py (the case table); do not edit by hand.",
         f"# rlr3 for {h} (make_rerun3_files.py from ilrr_box.sh @ {BOX_COMMIT[:12]}; cells from {fin['run'].split('/')[-1]} @ "
         f"{fin['launch_json_sha256'][:16]}): mechanical renames only. Do not edit.", 1),
        ("#   bash ~/phase2/il_stage/ilrr_box.sh <host>", f"#   bash ~/phase2/il_stage/ilrr3_{h}_box.sh <host>", 1),
        ("2026-10-15_il_${HF}_rlr;", "2026-10-15_il_${HF}_rlr3;", 2),
        ("SESSION=ilrlr$HF; LOG=/home/aswin/phase2/il_${HF}_rlr.log;", "SESSION=ilrlr3$HF; LOG=/home/aswin/phase2/il_${HF}_rlr3.log;", 1),
        ("[il ${HF}_rlr @ $H]", "[il ${HF}_rlr3 @ $H]", 1),
        ("pilot_ilrr_", "pilot_ilrr3_", 4), ("tasks_ilrr_", "tasks_ilrr3_", 3), ("ilrr_rerun_of_", "ilrr3_rerun_of_", 4),
        ("ilrr_original_pairs_", "ilrr3_original_pairs_", 2), ("gate_settings_ilrr_", "gate_settings_ilrr3_", 1),
        ("\"launch\": \"il_%s_rlr\"", "\"launch\": \"il_%s_rlr3\"", 1),
        ("@@EXCL@@", excl + " ! -path './config/pilot_ilrr3_*' ! -path './config/tasks_ilrr3_*' ! -path './config/ilrr3_*'"
         " ! -path './config/pilot_ilrr2_*' ! -path './config/pilot_ilc_*' ! -path './config/tasks_ilc_*' ! -path './config/ilc_*'", 1),  # Codex r21
    ]:
        assert box.count(a) == k, (a, box.count(a))
        box = box.replace(a, b)
    assert "_rlr;" not in box and "_rlr.log" not in box
    out = HERE / f"ilrr3_{h}_box.sh"
    assert not out.exists()
    out.write_text(box)
    print(json.dumps({"host": h, "selected": len(sel), "cap": cap, "tasks_sha256": sha(tp)}))


if __name__ == "__main__":
    main()
