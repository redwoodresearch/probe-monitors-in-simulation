"""Amendment 43 regex test (names only): 09's s4rgap folder regex accepts its own names and rejects ordinary ilnm_s4* names; the existing
IL folder grammars (FOLDER_IL, FOLDER_OLDSEL) reject s4rgap names; t196fill's regex accepts its names and rejects other sim-honest names,
and no existing scorer regex accepts t196fill names. Usage: python3 -I regex_test_a43.py <run_dt_q4b.py>"""
import re, sys
src = open(sys.argv[1]).read()
def rx(name):
    m = re.search(rf"^{name} = re\.compile\((r\"[^\"]*\")\)", src, re.M)
    return re.compile(eval(m[1]))
S4RGAP, FOLDER_IL, FOLDER_OLDSEL, FOLDER = rx("S4RGAP"), rx("FOLDER_IL"), rx("FOLDER_OLDSEL"), rx("FOLDER")
T196 = re.compile(r"^simh_t196fill(?:_s[0-9]+)?_v1(?:_try[0-9]+)?__sim_honest$")
SIMH311 = re.compile(r"^simh311_q[1-4]_sim(?:_s[0-9]+|_rlr(?:[2-9]|[1-9][0-9]+)?)?_v1(?:_try[0-9]+)?__sim_honest$")
SIMHNM = re.compile(r"^simh_nm360_(s1|s2|s3|s4d|s4r|s5)_sim(?:_s[0-9]+|_rlr(?:[2-9]|[1-9][0-9]+)?)?_v1(?:_try[0-9]+)?__sim_honest$")
pos_gap = ["ilnm_s4rgap_sim2_v1__sim_attack", "ilnm_s4rgap_sim2_s2_v1__sim_attack", "ilnm_s4rgap_sim2_v1_try2__sim_attack"]
neg_gap = ["ilnm_s4r_sim2_v1__sim_attack", "ilnm_s4r2_sim2_v1__sim_attack", "ilnm_s4r3_sim2_v1__sim_attack", "ilnm_s4_sim2_s1_v1__sim_attack",
           "ilnm_s4d_sim2_v1__sim_attack", "ilnm_s4rgap_run3_v1__attack", "ilnm_s4rgap_sim2_v1__attack", "ilnm_s4rgap_v1__sim_attack", "ilnm_s4rgapx_sim2_v1__sim_attack"]
pos_t = ["simh_t196fill_v1__sim_honest", "simh_t196fill_s1_v1__sim_honest", "simh_t196fill_v1_try2__sim_honest"]
neg_t = ["simh_t196fill_v2__sim_honest", "simh_t196fill_v1__sim_attack", "simh311_q1_sim_v1__sim_honest", "simh_nm360_s1_sim_v1__sim_honest", "simh_t196_v1__sim_honest"]
ok = all(S4RGAP.fullmatch(n) for n in pos_gap) and not any(S4RGAP.fullmatch(n) for n in neg_gap) \
    and not any(FOLDER_IL.fullmatch(n) or FOLDER_OLDSEL.fullmatch(n) or FOLDER.fullmatch(n) for n in pos_gap) \
    and all(T196.fullmatch(n) for n in pos_t) and not any(T196.fullmatch(n) for n in neg_t) \
    and not any(SIMH311.fullmatch(n) or SIMHNM.fullmatch(n) or FOLDER.fullmatch(n) or FOLDER_IL.fullmatch(n) for n in pos_t)
print({"s4rgap_pos": [bool(S4RGAP.fullmatch(n)) for n in pos_gap], "s4rgap_neg": [bool(S4RGAP.fullmatch(n)) for n in neg_gap],
       "il_grammars_on_s4rgap": [bool(FOLDER_IL.fullmatch(n) or FOLDER_OLDSEL.fullmatch(n)) for n in pos_gap],
       "t196_pos": [bool(T196.fullmatch(n)) for n in pos_t], "t196_neg": [bool(T196.fullmatch(n)) for n in neg_t]})
print("ALL OK" if ok else "FAIL")
