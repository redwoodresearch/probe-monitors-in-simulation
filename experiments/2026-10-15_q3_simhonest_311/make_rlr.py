"""make_rlr.py (sim-honest replacements; 09 @ 13be06982, 91485ce80, c5420f2f5, c10a14948; d9's plan ~13:45): per replacement
launch, a task file holding only its cells (each byte-identical to its parent's task-file entry, priority cells first) and
a config (the parent's simulated-honest config with tasks pointed at it; cap = expected + cells x $106 so admission fits
every cell at the $106 reserve; sim_cap_stop pause at 2x expected), plus results/simh_rlr_plan_2026-10-15.json with the
per-key class ("infrastructure" for the 7 RateLimitError keys, "unknown" for the 9 terminal-error keys) and rulings.
  python3 make_rlr.py <dir holding the frozen tasks_311_q<k>_fu.json>
"""
import hashlib
import json
import re
import sys
from pathlib import Path

HERE, H311 = Path(__file__).parent, Path(sys.argv[1])
FU311 = {"q1": "b5b02a57e6e4f02eeed7b7dc38ec7f48a1e8ab7669a4e9a97b98c6f23f860818", "q2": "31ee96143aa74b35eb898edb64d9c7b33b79ff779d6b8b84b360b754207bd1cf",
         "q3": "a477bb8ee83d0e29943caae41f8d43c5c7892aa620240f6095076461d0631305", "q4": "dab4bf2d00b890b2add9e7f9746f66dbf12457f11d66b5004f00599dd344122a"}
RL = {"t462", "t242", "t271", "t410", "t298", "t310", "nm103", "t240", "t275", "t457"}  # 09 @ 13be06982 / c5420f2f5 / 81f75b763: infrastructure
RAISED_STOP = {"q2_rlr2", "nm_s4d_rlr2", "nm_s4r_rlr", "q1_rlr2", "q3_rlr2"}
LAUNCHES = {  # launch: (parent, cells in launch order, parent task file, parent config)
    "q1_rlr": ("q1", ["t462", "t242", "t237", "t300", "t389"], H311 / "tasks_311_q1_fu.json", "config/pilot_simh_q1.yaml"),
    "q2_rlr": ("q2", ["t271", "t410", "t240", "t275", "t457"], H311 / "tasks_311_q2_fu.json", "config/pilot_simh_q2.yaml"),  # v2: t293 held (hold record v3 @ ea1f83581); t240/t275/t457: 09 @ 81f75b763
    "q3_rlr": ("q3", ["t298", "t310"], H311 / "tasks_311_q3_fu.json", "config/pilot_simh_q3.yaml"),
    "q4_rlr": ("q4", ["t361", "t411", "t413"], H311 / "tasks_311_q4_fu.json", "config/pilot_simh_q4.yaml"),  # v3: t361 (test of the :313 hypothesis, 0d 15:5x) + t411/t413 released (0d rule 15:4x; t237 completed OK 16:19:39)
    "nm_s3_rlr": ("nm_s3", ["nm134", "nm103"], HERE / "config_nm/tasks_simh_nm_s3.json", "config_nm/pilot_simh_nm_s3.yaml"),  # 0d: nm134 first
    "nm_s4d_rlr": ("nm_s4d", ["nm258"], HERE / "config_nm/tasks_simh_nm_s4d.json", "config_nm/pilot_simh_nm_s4d.yaml"),  # v2: nm107 held (hold record v3)
    "q2_rlr2": ("q2", ["t293"], H311 / "tasks_311_q2_fu.json", "config/pilot_simh_q2.yaml"),  # v3: t293 released (0d rule 15:4x; t237 completed OK 16:19:39); second Q2-parent launch
    "nm_s4d_rlr2": ("nm_s4d", ["nm107"], HERE / "config_nm/tasks_simh_nm_s4d.json", "config_nm/pilot_simh_nm_s4d.yaml"),  # v3: nm107 released (same); second S4d-parent launch, box4 only (09 @ df46b9be1)
    "q3_rlr2": ("q3", ["t298", "t310"], H311 / "tasks_311_q3_fu.json", "config/pilot_simh_q3.yaml"),  # v4: Q3_rlr cut by astra's EC2 stop; 09: root simh311_q3_sim_rlr2_v1__sim_honest; t310 un-excluded (0d: approach present on box3, fixed-path check 16:44:53)
    "nm_s4r_rlr": ("nm_s4r", ["nm353"], HERE / "config_nm/tasks_simh_nm_s4r.json", "config_nm/pilot_simh_nm_s4r.yaml"),  # step_limit.py:72, 0d 15:4x on census @ 391fa659b
    "q1_rlr2": ("q1", ["t320"], H311 / "tasks_311_q1_fu.json", "config/pilot_simh_q1.yaml"),  # second Q1-parent launch (q1_rlr is taken); step_limit.py:72, 0d 15:4x on census @ 391fa659b
    "nm_s2_rlr": ("nm_s2", ["nm150"], HERE / "config_nm/tasks_simh_nm_s2.json", "config_nm/pilot_simh_nm_s2.yaml"),  # 09 @ dd93a41cd §2
}
out = {"rulings": {"RateLimitError (infrastructure)": "09 @ 13be06982, c5420f2f5; grammar @ 91485ce80",
                   "terminal error (class unknown)": "09 @ c10a14948 (+ t293 @ dd93a41cd); admission root checks @ 98ffa9e40 (RL), 2b66c9dbe (terminal)",
                   "v4 (d9 ~16:45)": "q3_rlr2 = t298 + t310 (Q3_rlr cut by astra's EC2 stop, read 16:31:12; 09 root simh311_q3_sim_rlr2_v1__sim_honest; t310 briefly excluded on a buggy presence read, un-excluded by 0d after the fixed-path check at 16:44:53); queued launches pause at max(4 x expected, $100) (0d)",
                   "v3 (hold record v5)": "t293, t411, t413, nm107 released: 0d rule 15:4x; t237 replacement completed OK 16:19:39 (exit 0, submit, 162 steps)",
                   "v2 (hold record v3 @ ea1f83581)": "t293, t411, t413, nm107 held for the base rate; t361 released as a test of the :313 hypothesis (0d 15:5x); t320 and nm353 GO at step_limit.py:72 (0d 15:4x on census @ 391fa659b)"},
       "cap_rule": "expected (cells x $19.6) + cells x $106; sim_cap_stop pauses at 2x expected", "launches": {}}
for name, (parent, ids, src, pcfg) in LAUNCHES.items():
    raw = src.read_bytes()
    if parent in FU311:
        assert hashlib.sha256(raw).hexdigest() == FU311[parent], (name, "parent task file is not the frozen one")
    d = json.loads(raw)
    by_id = {c["key"].split("_")[0]: c for c in d["cells"]}
    cells = [by_id[i] for i in ids]
    tasks_name = f"tasks_simh_{name}.json"
    tpath = HERE / "config_rlr" / tasks_name
    tpath.write_text(json.dumps(dict(d, cells=cells), indent=1) + "\n")
    n = len(cells); exp = round(n * 19.6); cap = exp + n * 106
    stop = max(4 * exp, 100) if name in RAISED_STOP else 2 * exp  # 0d via d9 ~16:41: queued launches pause at max(4 x expected, $100)
    cfg = (HERE / pcfg).read_text()
    cfg, c1 = re.subn(r"^tasks: .*$", f"tasks: config/{tasks_name}   # make_rlr.py: the replacement cells only ({', '.join(c['key'] for c in cells)})", cfg, flags=re.M)
    cfg, c2 = re.subn(r"^  cap_usd: .*$", f"  cap_usd: {cap}   # {n} cells: expected ${exp} + {n} x $106 (one admission per cell at the $106 reserve)", cfg, flags=re.M)
    cfg, c3 = re.subn(r"^  worst_case_limit_usd: .*$", f"  worst_case_limit_usd: {cap}   # = cap: the box-side admission bound", cfg, flags=re.M)
    assert c1 == c2 == c3 == 1, name
    cfg = f"# SIM-HONEST REPLACEMENT launch simh_{name} (RULE.md; 09's rulings in results/simh_rlr_plan_2026-10-15.json), from {pcfg}:\n" + cfg
    cpath = HERE / "config_rlr" / f"pilot_simh_{name}.yaml"
    cpath.write_text(cfg)
    out["launches"][name] = {"parent": parent, "cells": [{"key": c["key"], "class": "infrastructure" if c["key"].split("_")[0] in RL else "unknown"} for c in cells],
                             "tasks_file": tasks_name, "tasks_sha256": hashlib.sha256(tpath.read_bytes()).hexdigest(),
                             "config": cpath.name, "config_sha256": hashlib.sha256(cpath.read_bytes()).hexdigest(),
                             "expected_usd": exp, "cap_usd": cap, "sim_cap_stop_pause_at": stop}
    print(name, n, "cells", [c["key"] for c in cells], "cap", cap, "stop", stop)
# v4: results are append-only, so the v1-v3 plans stay and this writes a new file
(HERE / "results" / "simh_rlr_plan_v4_2026-10-15.json").write_text(json.dumps(out, indent=1) + "\n")
