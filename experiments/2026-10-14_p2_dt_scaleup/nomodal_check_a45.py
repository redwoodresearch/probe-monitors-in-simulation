"""Amendment 45 no-Modal check (names only): the job's four sp493 regexes are 09's (@ 9cb3d0291) byte for byte; sp493_fam accepts the corrected
names with the right family and pool and refuses superseded / pending / replacement / outside names; no existing DT folder name or d9-routed
folder name is an sp493 name, and no sp493 positive matches an existing DT regex; NON_IL covers sp493 run and staged-set names; out_of puts sp493
at /scaleup/<folder>/dt (real, sim attack) and /simh/sp493_g<k>/dt/<folder> (sim-honest), 09 Q1; SP493_SUPERSEDED equals d9's names-record union @ 32226c415 and every listed sp493 set is refused (45c). Usage: python3 -I nomodal_check_a45.py <repo> <run_dt_q4b.py> <names file> <routing dir> <out json>"""
import ast
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO, JOB, NAMES, RT, OUT = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5])
src = JOB.read_text()
tree, g = ast.parse(src), {"re": re, "Path": Path}
want = {"FOLDER", "FOLDER_IL", "FOLDER_NM", "FOLDER_OLDSEL", "S4RGAP", "HA25F", "NON_IL", "UG", "SIMH", "SIMH_BATCH", "T196FILL_BATCH", "SET_WORLD", "SP493", "SP493_SET",
        "SP493_POOLS", "SP493_MIN_TRY", "SP493_STRATUM", "SP493_WORLDS", "SP493_SIMH_WORLD", "SP493_SUPERSEDED", "SP493_WORLD_MAP", "SP493_NAMES", "SP493_NAMES_ALL", "SP493_G12_REMAINDER_OK", "SP493_TAIL_POOLS", "SP493_GROUP", "sp493_fam", "simh_of", "simh_batch", "out_of"}
keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in want) or (isinstance(n, ast.Assign) and any(getattr(x, "id", "") in want for x in n.targets))]
exec(compile(ast.Module(body=keep, type_ignores=[]), "job", "exec"), g)
rec = json.loads(subprocess.run(["git", "-C", REPO, "show", "9cb3d0291:experiments/2026-10-15_q3_sp493/results/sp493_regex_test_2026-10-15_171735.json"],
                                capture_output=True, check=True).stdout)
V2 = r"^sp493_g[0-9]+_sim(?:_[xy])?(?:_s[0-9]+)?_v1(?:_try[0-9]+)?__sim_attack$"  # 09 @ b9e0b6eda (replaces @ 9cb3d0291's sim_attack pattern)
assert V2.encode() in subprocess.run(["git", "-C", REPO, "show", "b9e0b6eda:experiments/2026-10-14_priority1_spec_09/RULE.md"], capture_output=True, check=True).stdout
verbatim = {k: g["SP493"][k].pattern == (V2 if k == "sim_attack" else v) for k, v in rec["regex"].items()} | {"same_keys": set(rec["regex"]) == set(g["SP493"])}
v2_cases = {"sp493_g1_sim_s3_v1_try2__sim_attack": 1, "sp493_g1_sim_v1__sim_attack": 1, "sp493_g3_sim_x_s1_v1__sim_attack": 1, "sp493_g6_sim__x_v1__sim_attack": 0,
            "sp493_g6_sim_s2_x_v1__sim_attack": 0, "sp493_g6_sim_x_v1__sim_attack": 1, "sp493_g6_sim_x_v1__sim_honest": 0, "sp493_g6_sim_x_v1_try__sim_attack": 0,
            "sp493_g6_sim_x_v2__sim_attack": 0, "sp493_g6_sim_xy_v1__sim_attack": 0, "sp493_g6_sim_y_s2_v1_try2__sim_attack": 1, "sp493_g6_sim_z_v1__sim_attack": 0,
            "sp493_g7_sim_y_v1_try3__sim_attack": 1, "sp493_gx_sim_x_v1__sim_attack": 0}  # 09's 14 test cases @ b9e0b6eda
v2_bad = {k: e for k, e in v2_cases.items() if bool(g["SP493"]["sim_attack"].fullmatch(k)) != bool(e)}


def call(x):
    try:
        return g["sp493_fam"](x)
    except ValueError as e:
        return "REFUSED: " + str(e).split(": ", 1)[1][:60]


pos = {"sp493_a_main_v1_try3__attack": ("real_main", None), "sp493_b_main_v1_try5__honest_a": ("real_main", None), "sp493_c_main_v1_try4__honest_b": ("real_main", None),
       "sp493_c_run2_v1_try4__attack": ("real_run2", None), "sp493_a_main_v1_try4__attack": ("real_main", None), "sp493_g3_sim_v1__sim_attack": ("sim_attack", "g3"),
       "sp493_g6_sim_s2_v1__sim_attack": ("sim_attack", "g6"), "sp493_g7_sim_s1_v1_try2__sim_attack": ("sim_attack", "g7"), "sp493_g4_simh_v1_try2__sim_honest": ("sim_honest", "g4"),
       "sp493_g5_simh_s1_v1__sim_honest": ("sim_honest", "g5"), "sp493_g5_simh_v1__sim_honest": ("sim_honest", "g5"), "sp493_b_main_v1_try5__honest_b": ("real_main", None),
       "sp493_g1_sim_v1__sim_attack": ("sim_attack", "g1"), "sp493_g2_sim_v1__sim_attack": ("sim_attack", "g2"), "sp493_g1_sim_s4_v1_try2__sim_attack": ("sim_attack", "g1"), "sp493_a_main_v1_try3": ("real_main", None), "sp493_g3_sim_s1_v1": ("sim_attack", "g3"),
       "sp493_g4_simh_v1_try2": ("sim_honest", "g4"), "sp493_g1_sim_s1_v1_try2__sim_attack": ("sim_attack", "g1"), "sp493_g2_sim_v1_try3__sim_attack": ("sim_attack", "g2"),
       "sp493_g1_sim_s3_v1_try2": ("sim_attack", "g1"), "sp493_g6_sim_x_v1__sim_attack": ("sim_attack", "g6"), "sp493_g6_sim_y_s2_v1_try2__sim_attack": ("sim_attack", "g6"),
       "sp493_g3_sim_x_s1_v1__sim_attack": ("sim_attack", "g3"), "sp493_g7_sim_y_v1_try3__sim_attack": ("sim_attack", "g7")}
refuse = ["sp493_a_main_v1__attack", "sp493_a_main_v1_try1__attack", "sp493_a_main_v1_try2__honest_a", "sp493_b_run2_v1_try2__attack", "sp493_a_main_v1_try03__attack",
          "sp493_a_main_v1_try2", "sp493_g1_sim_s1_v1__sim_attack", "sp493_g1_sim_s3_v1", "sp493_g2_sim_s1_v1_try1__sim_attack", "sp493_g2_sim_v1_try02__sim_attack", "sp493_g8_sim_v1__sim_attack",
          "sp493_g4_sim_v1__sim_attack", "sp493_g3_simh_v1__sim_honest", "sp493_g4_simh_rlr_v1__sim_honest", "sp493_g5_simh_rlr2_v1__sim_honest", "sp493_d_main_v1_try3__attack",
          "sp493_a_main_v1_try3__sim_attack", "sp493_a_run2_v1_try3__honest_a", "sp493_g4_simh_rlr1_v1__sim_honest", "sp493_a_main_cutfull_v1_try3__attack",
          "sp493_b_main_v1_try3__attack", "sp493_b_main_v1_try4__attack", "sp493_c_main_v1_try3__honest_a", "sp493_c_run2_v1_try3__attack", "sp493_g1_sim_x_v1__sim_attack", "sp493_g2_sim_y_s2_v1_try2__sim_attack", "sp493_g6_sim_z_v1__sim_attack", "sp493_g4_simh_v1__sim_honest", "sp493_g1_sim_s4_v1__sim_attack", "sp493_g2_sim_s5_v1", "sp493_g1_sim_v1_dry", "sp493_c_run2_v1_try2"]
got_pos = {x: call(x) for x in pos}
got_ref = {x: call(x) for x in refuse}
names = set(NAMES.read_text().split())
for f in RT.glob("*__*.json"):
    try:
        r = json.loads(f.read_text())
    except ValueError:
        continue
    for a in r.get("arms") or []:
        names.add(f"{r['staged_set']}__{a['meta_arm'].split('@')[0]}")
    names.add(r.get("staged_set", ""))
existing = sorted(n for n in names if n and not n.startswith("sp493_"))
exist_rx = {k: g[k] for k in ("FOLDER", "FOLDER_IL", "FOLDER_NM", "FOLDER_OLDSEL", "S4RGAP", "HA25F")} | {"UG": g["UG"]["folder"]} | {f"SIMH {k}": v["folder"] for k, v in g["SIMH"].items()}
acc = [n for n in existing if g["sp493_fam"](n) is not None or any(r.fullmatch(n) for r in g["SP493"].values())]
cross = {x: [k for k, r in exist_rx.items() if r.fullmatch(x)] for x in pos if any(r.fullmatch(x) for r in exist_rx.values())}
non_il = {x: bool(g["NON_IL"].fullmatch(x)) for x in ("sp493_a_main_extract_v1_try3", "sp493_a_main_v1_try3", "sp493_c_run2_extract_v1_try3", "sp493_g3_sim_s1_extract_v1",
                                                     "sp493_g6_sim_v1", "sp493_g4_simh_extract_v1", "sp493_g5_simh_s2_v1")}
outs = {x: [str(p) for p in g["out_of"](x)] for x in ("sp493_a_main_v1_try3__attack", "sp493_g3_sim_v1__sim_attack", "sp493_g4_simh_v1_try2__sim_honest")}
union, names_ok = set(), True
for pin_ in g.get("SP493_NAMES_ALL", (g["SP493_NAMES"],)):  # 45h: every pinned names record
    nraw = subprocess.run(["git", "-C", REPO, "show", f"{pin_[0]}:{pin_[1]}"], capture_output=True, check=True).stdout
    nrec = json.loads(nraw)
    names_ok &= hashlib.sha256(nraw).hexdigest() == pin_[2]
    union |= {y["staged_set"] for k in ("superseded", "burned_or_taken", "superseded_v1_launches") for y in nrec.get(k, [])} | set((nrec.get("non_routable") or {}).get("names", []))
names_ok &= union == set(g["SP493_SUPERSEDED"])
listed_refused = {x: call(x) for x in sorted(union) if x.startswith("sp493_")}
ok = all(verbatim.values()) and not v2_bad and got_pos == pos and all(str(v).startswith("REFUSED") for v in got_ref.values()) and not acc and not cross and all(non_il.values()) \
    and outs == {"sp493_a_main_v1_try3__attack": ["/work/scaleup/sp493_a_main_v1_try3__attack", "dt"], "sp493_g3_sim_v1__sim_attack": ["/work/scaleup/sp493_g3_sim_v1__sim_attack", "dt"],
                 "sp493_g4_simh_v1_try2__sim_honest": ["/work/simh/sp493_g4/dt/sp493_g4_simh_v1_try2__sim_honest", ""]} and g["SP493_WORLD_MAP"] == {"sp493_sim": "sim"} \
    and names_ok and all(str(v).startswith("REFUSED") for v in listed_refused.values()) and g["SP493_G12_REMAINDER_OK"] is True
res = {"what": "Amendment 45 no-Modal check: sp493 grammar, refusals, overlap with every existing DT / routed name (names only)",
       "written_pt": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S"), "regex_record": "9cb3d0291:experiments/2026-10-15_q3_sp493/results/sp493_regex_test_2026-10-15_171735.json",
       "verbatim": verbatim, "positives": {k: list(v) if isinstance(v, tuple) else v for k, v in got_pos.items()}, "refusals": got_ref, "existing_names": len(existing),
       "existing_names_accepted_as_sp493": acc, "sp493_positives_accepted_by_existing_regexes": cross, "non_il": non_il, "out_of": outs, "v2_cases_bad": v2_bad, "names_record_union_equals_pin": names_ok, "listed_sp493_sets": listed_refused, "existing_name_list": existing, "pass": ok}
OUT.open("x").write(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k != "existing_name_list"}, indent=0)[:3500])
print("ALL OK" if ok else "FAIL")
