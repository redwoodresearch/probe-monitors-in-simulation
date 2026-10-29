"""Amendment 4's example check: 09's canonical FOLDER regex plus SET_ARMS and POOL_STRATA, read from run_dt_q4b.py (no Modal import),
on names d9's chain emits (via the job's folder() transform) and on names that must fail; then the two pinned new-mains files agree.
  python3 -I experiments/2026-10-14_p2_dt_scaleup/check_folder_names.py   (from the repo root; reads git only)"""

import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path

ns = {}  # literals only: FOLDER's pattern string (the re.compile argument) and the three dicts
for node in ast.parse((Path(__file__).parent / "run_dt_q4b.py").read_text()).body:
    if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") in ("FOLDER", "SET_ARMS", "POOL_STRATA", "NM_PINS", "FOLDER_IL", "IL_POOL_STRATA", "FOLDER_NM", "IL_TOK"):
        v = node.value.args[0] if isinstance(node.value, ast.Call) else node.value
        ns[node.targets[0].id] = ast.literal_eval(v)
FOLDER, SET_ARMS, POOL = re.compile(ns["FOLDER"]), ns["SET_ARMS"], ns["POOL_STRATA"]
FOLDER_IL, IL_POOL = re.compile(ns["FOLDER_IL"]), ns["IL_POOL_STRATA"]
FOLDER_NM, IL_TOK = re.compile(ns["FOLDER_NM"]), re.compile(ns["IL_TOK"])


def ok(g: str, stratum: str) -> bool:  # the job's folder test (arm): FOLDER, the set's meta-arms, a stratum the pool allows
    m = FOLDER.fullmatch(g)
    if FOLDER_NM.fullmatch(g):  # Amendment 14: 09's nm360 follow-up/relaunch names (pairings built in); S_newmains
        return stratum == "S_newmains"
    if FOLDER_IL.fullmatch(g):  # Amendment 11: 09's IL regex (pairings built in); the prefix gives the stratum
        return stratum in IL_POOL[g.split("_")[0]]
    return bool(m) and m[6] in SET_ARMS[m[3]] and not (m[2].endswith("r") and m[3] != "sim") and stratum in POOL[m[1]]


def folder(run: str, arm_dir: str) -> str:  # the job's transform: extraction run without "_extract", "__", meta-arm without "@..."
    return run.replace("_extract", "", 1) + "__" + arm_dir.split("@")[0]


emitted = [  # nm360_s3_main_s1 and nm360_s3_run2_s2 carry _s<k> on main/run2 per 09's correction (example nm360_s3_main_s1_extract_v1);
    # d9's committed targets_for @ 9bcfc519c puts _s<k> on sim only, so those two are expected names, not yet emitted ones
("su311_q1_main_extract_v1", "honest_a"), ("su311_q1_main_extract_v1", "honest_b"), ("su311_q1_main_extract_v1", "attack"),
           ("su311_q1_run2_extract_v1", "attack"), ("su311_q4_sim_extract_v1", "sim_attack@september"),
           ("su311_q2_sim_s3_extract_v1", "sim_attack@september"), ("su311_q1_main_extract_v1_try2", "honest_b"),
           ("nm360_s3_main_s1_extract_v1", "attack"), ("nm360_s3_run2_s2_extract_v1", "attack"), ("nm360_s1_main_extract_v1", "honest_a"),
           ("nm360_s5_sim_s4_extract_v1_try2", "sim_attack@september"),
           ("su311_q2r_sim_extract_v1", "sim_attack@september"),
           ("il311_q1_run3_extract_v1", "attack"), ("ilnm_s4d_sim2_s1_extract_v1", "sim_attack@september"), ("ilold_old_run3_extract_v1_try2", "attack"),  # 4f's IL accepts
           ("nm360_s5r_sim_extract_v1", "sim_attack@september"), ("nm360_s4d_sim_s2_extract_v1", "sim_attack@september"), ("nm360_s4r_main_extract_v1_try2", "honest_b"),
           ("nm360_s4r_run2_extract_v1", "attack"), ("nm360_s4r_sim_extract_v1", "sim_attack@september"),  # 09 @ bffe22e3e nm360 names (Amendment 14)
           ("il311_q3r_run3_extract_v1", "attack"), ("ilold_oldr_sim2_extract_v1", "sim_attack@september"), ("ilnm_s4dr_sim2_s1_extract_v1", "sim_attack@september"),  # 09's r (Amendment 13: grammar only; the job blocks r)
           ("il311_q1c_run3_extract_v1", "attack"), ("ilold_oldc_sim2_extract_v1_try2", "sim_attack@september"), ("ilnm_s2r2_run3_extract_v1", "attack"),
           ("ilnm_s2r2_sim2_extract_v1", "sim_attack@september"), ("il311_q4r2_sim2_extract_v1", "sim_attack@september"), ("il311_q2r3_run3_extract_v1", "attack"),
           ("ilnm_s4dr3_sim2_s1_extract_v1", "sim_attack@september"), ("ilold_oldr3_run3_extract_v1", "attack")]  # 09 @ 7bd9d5e89 (Amendment 15; records gate the job)
bad = ["../su311_q1_main_v1__attack", "su311_q1_main_v1__attack/..", "su311_q1_main_v1__attack/x", "su311_q1_main_v1__attack\n",
       "su311_q1_main_extract_v1__attack", "su311_q1_sim_v1__sim_attack@september", "su311_q1_main_v1__honest_c", "su311_q1t_main_v1__attack",
       "su311_q1_main_E2ETEST_v1__attack", "rh_c_restage_v1__honest_b", "su311_q1_main_v2__attack", "SU311_q1_main_v1__attack",
       "su311_q1_run2_v1__sim_attack", "nm360_s1_sim_s2_v1__honest_a", "su311_q1_main_v1__sim_attack",
       "su311_q2r_main_v1__attack", "su311_q2r_run2_v1__attack", "nm360_s2r_sim_v1__sim_attack", "su311_q2rr_sim_v1__sim_attack",
       "il311_q1_sim2_v1__attack", "il311_q5_run3_v1__attack",  # 4f's IL rejects
       "il311_q3rr_run3_v1__attack", "nm360_s5r_sim_s1_v1__sim_attack", "nm360_s5r_main_v1__attack", "nm360_s4d_main_v1__attack", "nm360_s3r_sim_v1__sim_attack",
       "nm360_s4r_run2_v1__honest_a", "nm360_s4r_sim_v1__attack", "nm360_s5rr_sim_v1__sim_attack", "ilold_oldrr_run3_v1__attack", "il311_q3r_sim2_v1__attack", "ilnm_s4rd_run3_v1__attack", "il311_q3_rlr_run3_v1__attack",
       "il311_q1_run3_v1__sim_attack", "ilnm_s6_run3_v1__attack", "ilold_q1_run3_v1__attack", "il311_q1_run2_v1__attack", "ilnm_s4d_sim2_s1_v1__sim_attack/..",
       "ilnm_s1c_run3_v1__attack", "ilold_oldr2_run3_v1__attack", "il311_q1r4_run3_v1__attack", "ilnm_s2r2r_run3_v1__attack", "il311_q1cr_run3_v1__attack",
       "il311_q1c2_run3_v1__attack", "ilnm_s2rr2_sim2_v1__sim_attack", "il311_q1r2c_run3_v1__attack", "il311_q1c_sim2_v1__attack"]  # Amendment 15 rejects
acc = [folder(r, a) for r, a in emitted]


def st(g: str) -> str:  # the stratum d9's chain gives a folder's rows
    return {"nm360": "S_newmains", "ilnm": "S_newmains", "ilold": "S_old"}.get(g.split("_")[0], "S_new")


res = {"accepted (should all be true)": {g: ok(g, st(g)) for g in acc},
       "rejected (should all be false)": {repr(g): ok(g, st(g)) for g in bad}
       | {f"{g} with {x}": ok(g, x) for g, x in [("su311_q1_main_v1__attack", "S_newmains"), ("nm360_s1_main_v1__attack", "S_new"),
                                                 ("nm360_s1_main_v1__attack", "S_old"), ("su311_q1_main_v1__attack", None),
                                                 ("il311_q1_run3_v1__attack", "S_old"), ("ilold_old_run3_v1__attack", "S_new"), ("ilnm_s1_run3_v1__attack", "S_new"),
                                                 ("su311_q1_main_v1__attack", "S_old")]}}  # 09: S_old is T196 only
nm = {}
for k, (rev, path, pin) in ns["NM_PINS"].items():
    raw = subprocess.run(["git", "show", f"{rev}:{path}"], capture_output=True, check=True).stdout
    nm[k] = {"sha256_matches_pin": hashlib.sha256(raw).hexdigest() == pin, "map": {c["key"]: c["main"] for c in json.loads(raw)["cells"]}}
res["new mains"] = {"cells_v1 sha ok": nm["cells_v1"]["sha256_matches_pin"], "built sha ok": nm["built"]["sha256_matches_pin"],
                    "keys": [len(nm["cells_v1"]["map"]), len(nm["built"]["map"])], "maps equal": nm["cells_v1"]["map"] == nm["built"]["map"]}
want = {"ilnm_s2r2_run3_extract_v1": ("ilnm", "s2", "r2"), "ilnm_s4dr3_sim2_s1_v1": ("ilnm", "s4d", "r3"), "ilnm_s4d_sim2_s1_v1": ("ilnm", "s4d", ""),
        "il311_q1c_run3_v1__attack": ("il311", "q1", "c"), "ilold_oldr_sim2_v1": ("ilold", "old", "r"), "nm360_s5r_sim_v1": None}  # Amendment 15: il_kind
res["IL tokens parsed"] = {g: (lambda m: (m[1], m[2], m[3] or "") if m else None)(IL_TOK.match(g)) == t for g, t in want.items()}
res["PASS"] = all(res["accepted (should all be true)"].values()) and not any(res["rejected (should all be false)"].values()) and all(res["IL tokens parsed"].values()) \
    and res["new mains"]["cells_v1 sha ok"] and res["new mains"]["built sha ok"] and res["new mains"]["maps equal"]
print(json.dumps(res, indent=1))
raise SystemExit(0 if res["PASS"] else 1)
