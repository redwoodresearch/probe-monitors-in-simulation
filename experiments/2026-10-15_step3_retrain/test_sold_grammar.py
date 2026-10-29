"""09's ACKed S_old grammar and per-folder world map (shared by p2 / famexp / step3): positive and negative name tests, and every S_old row of
inventory_b_v1 (@ 755d7c9cc) checked against the map by key. Strings and JSON only. Exits nonzero on any failure.

  python3 -I test_sold_grammar.py
"""

import importlib.util
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
s = importlib.util.spec_from_file_location("build_cutfull_message", HERE / "build_cutfull_message.py")
A = importlib.util.module_from_spec(s)
s.loader.exec_module(A)
YES = {"q4f_main_v1__attack": "run1", "q4f_main_v1__honest_a": "honest_a", "q4f_main_v1__honest_b": "honest_b", "q4f_main_v1__sim_attack": "sim",
       "q4f_run2_v1__attack": "run2", "q4f_c_v1__honest_b": "honest_c", "rh1e4_stage_v1__honest_b": "relaunch",
       "simh_q4f_v1_full_b1__sim_honest": "sim_honest", "simh_q4f_v1_full_b2__sim_honest": "sim_honest"}
NO = ["q4f_main_v1__sim_honest", "q4f_d_v1__attack", "q4f_main_v1_try2__attack", "rh1e4_stage_v2__honest_b", "q4b_main_v1__honest_b", "q4f_main_cutfull_v1__attack",
      "simh_q4f_v1_full__sim_honest", "simh_q4f_v1_full_b1__sim_attack", " q4f_c_v1__honest_b", "q4f_c_v1__honest_b@september", "q4f_run2_v1__honest_a",
      "rh1e4_stage_v1__honest_a", "q4f_c_v1__attack", "simh_q4f_v1_full_b1_cutfull_v1__sim_honest", "q4f_main_v1__run1"]
fails = [n for n, w in YES.items() if A.sold_world(n) != w] + [n for n in NO if A.sold_world(n) is not None]
fails += [k for k in A.SOLD_WORLD if not A.SOLD_RE.fullmatch(k)]
b = subprocess.run(["git", "-C", str(HERE.parents[1]), "show", "755d7c9cc:experiments/2026-10-15_cut_sensitivity/results/inventory_b_v1_2026-10-15.json"],
                   capture_output=True, check=True).stdout
rows = [r for r in json.loads(b)["runs"] if r["task_set"] == "T196"]
fails += sorted({f"{r['original_folder']}->{r['world']}" for r in rows if A.sold_world(r["original_folder"]) != r["world"]})
print({"positives": len(YES), "negatives": len(NO), "inventory_b_v1_S_old_rows": len(rows), "failures": fails})
if fails:
    raise SystemExit(f"FAIL: {fails}")
print("PASS")
