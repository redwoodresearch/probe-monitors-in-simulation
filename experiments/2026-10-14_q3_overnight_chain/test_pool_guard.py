"""pool_files (chain.py combine_final's guard): a shorter pool's glob skips the files of a longer configured pool name; and every
S_paper493 folder the chain names (real arms _try2, sim g1-g3, sim-honest g4/g5, the X/Y group families g{3,6,7}_sim_{x,y}) matches exactly
one of 09's four regexes (sim_attack v2: 09 via d9 10-16 ~01:00); and a group family's box read keeps only its cells, index donors included. Names only.
  python test_pool_guard.py   (the chain's python)"""
import json
import re
from pathlib import Path

from chain import ambiguous_names, only_cells_view, pool_files, targets_for

D = Path("/x")
POOLS = {"su311", "nm360", "il", "simh311", "ug", "simh_nm360", "ep2", "sp493", "simh_t196fill", "ilnm", "ilold",
         "sp493_g1_sim", "sp493_g2_sim", "sp493_g3_sim", "sp493_g4_simh", "sp493_g5_simh",  # g1-g3 / g5 as they would be configured
         *{f"sp493_{g}_sim_{x}" for g in ("g3", "g6", "g7") for x in "xy"}}  # the X/Y group families (09's ruling (B))
FILES = {"run_table": ["run_table_sp493_a.json", "run_table_sp493_b.json", "run_table_sp493_g4_simh.json", "run_table_sp493_g1_sim.json",
                       "run_table_sp493_g1_sim_s1.json", "run_table_sp493_g5_simh_x.json", "run_table_simh_nm360_s1.json", "run_table_simh_t196fill_x.json"],
         "labels": ["labels_sp493_sp493-a.json", "labels_sp493_g4_simh_sp493-g4.json", "labels_sp493_g2_sim_sp493-g2_s1.json",
                    "labels_sp493_g6_sim_x_sp493-g6x_s1.json", "labels_sp493_g6_sim_y_sp493-g6y.json"],
         "exclusions": ["exclusions_sp493_sp493-b.json", "exclusions_sp493_g3_sim_sp493-g3.json"]}


def glob(kind, pf):  # what d.glob(f"{kind}_{pf}_*.json") returns
    return sorted(n for n in FILES[kind] if n.startswith(f"{kind}_{pf}_") and n.endswith(".json"))


def names(kind, pf, own):
    return [p.name for p in pool_files(glob(kind, pf), kind, pf, POOLS - own, D)]


cases = [  # (kind, pool, own names, expected)
    ("run_table", "sp493", {"sp493"}, ["run_table_sp493_a.json", "run_table_sp493_b.json"]),
    ("labels", "sp493", {"sp493"}, ["labels_sp493_sp493-a.json"]),
    ("exclusions", "sp493", {"sp493"}, ["exclusions_sp493_sp493-b.json"]),
    ("run_table", "sp493_g1_sim", {"sp493_g1_sim"}, ["run_table_sp493_g1_sim_s1.json"]),
    ("labels", "sp493_g4_simh", {"sp493_g4_simh"}, ["labels_sp493_g4_simh_sp493-g4.json"]),
    ("labels", "sp493_g2_sim", {"sp493_g2_sim"}, ["labels_sp493_g2_sim_sp493-g2_s1.json"]),
    ("exclusions", "sp493_g3_sim", {"sp493_g3_sim"}, ["exclusions_sp493_g3_sim_sp493-g3.json"]),
    ("run_table", "simh_nm360", {"simh_nm360"}, ["run_table_simh_nm360_s1.json"]),  # unchanged pools keep every file
    ("run_table", "simh_t196fill", {"simh_t196fill"}, ["run_table_simh_t196fill_x.json"]),
    ("labels", "sp493_g6_sim_x", {"sp493_g6_sim_x"}, ["labels_sp493_g6_sim_x_sp493-g6x_s1.json"]),
    ("labels", "sp493_g6_sim_y", {"sp493_g6_sim_y"}, ["labels_sp493_g6_sim_y_sp493-g6y.json"]),
]
bad = [(k, pf, got, want) for k, pf, own, want in cases if (got := names(k, pf, own)) != want]
assert not bad, bad
B = [{"id": "sp493-a", "q": "a", "pool": "sp493"}, {"id": "sp493-g4", "q": "g4", "qtok": "", "pool": "sp493_g4_simh"},
     {"id": "il-s4rgap", "q": "s4rgap", "pool": "il", "name_prefix": "ilnm"},
     *[{"id": f"sp493-{g}{x}", "q": f"{g}{x}", "qtok": "", "pool": f"sp493_{g}_sim_{x}"} for g in ("g3", "g6", "g7") for x in "xy"]]
assert ambiguous_names(B, POOLS) == []
assert ambiguous_names(B + [{"id": "g4_simh", "q": "x", "pool": "sp493"}], POOLS) == ["sp493_g4_simh ~ sp493_g4_simh"]  # Codex's case
assert ambiguous_names(B + [{"id": "y", "q": "g1_sim_z", "pool": "sp493"}], POOLS) == ["sp493_g1_sim_z ~ sp493_g1_sim"]

RX = {  # 09's S_paper493 folder regexes, verbatim from experiments/2026-10-15_q3_sp493/results/sp493_regex_test_2026-10-15_171735.json @ 9cb3d0291
    "real_main": r"^sp493_(?:a|b|c)_main_v1(?:_try[0-9]+)?__(?:attack|honest_a|honest_b)$",
    "real_run2": r"^sp493_(?:a|b|c)_run2_v1(?:_try[0-9]+)?__attack$",
    "sim_attack": r"^sp493_g[0-9]+_sim(?:_[xy])?(?:_s[0-9]+)?_v1(?:_try[0-9]+)?__sim_attack$",  # v2 (09 via d9 10-16 ~01:00): the launch token
    "sim_honest": r"^sp493_g[0-9]+_simh(?:_s[0-9]+|_rlr(?:[2-9]|[1-9][0-9]+)?)?_v1(?:_try[0-9]+)?__sim_honest$"}
folders = {}
for q in "abc":  # the real arms (main kind, first_try 2: the v2 relaunch folders _try2)
    for t, tv in targets_for({"kind": "main", "q": q, "pool": "sp493", "run": "r"}).items():
        for arm in tv["stage_arms"]:
            folders[f"{tv['out']}_try2__{arm}"] = "real_run2" if tv["set"] == "run2" else "real_main"
for g, arm, kind, snap in (("g1", "sim_attack@september", "fu", None), ("g2", "sim_attack@september", "fus", 3), ("g3", "sim_attack@september", "fu", None),
                           ("g4", "sim_honest@september", "fu", None), ("g5", "sim_honest@september", "fu", None)):
    pool = f"sp493_{g}_{'simh' if 'honest' in arm else 'sim'}"
    b = {"kind": kind, "q": g, "pool": pool, "qtok": "", "sim_tag": "", "main_run": "@index:x", "donor_launches": [], "run": "r", "sim_arm": arm, **({"snap": snap} if snap else {})}
    (tv,) = targets_for(b).values()
    folders[f"{tv['out']}__{arm.split('@')[0]}"] = "sim_honest" if "honest" in arm else "sim_attack"
for g in ("g3", "g6", "g7"):  # the X/Y group families: final remainder and a snapshot (+ a _try retry) each
    for x in "xy":
        for kind, snap in (("fu", None), ("fus", 2)):
            b = {"kind": kind, "q": f"{g}{x}", "pool": f"sp493_{g}_sim_{x}", "qtok": "", "sim_tag": "", "main_run": "@index:x", "donor_launches": [], "run": "r",
                 "sim_arm": "sim_attack@september", **({"snap": snap} if snap else {})}
            (tv,) = targets_for(b).values()
            folders[f"{tv['out']}__sim_attack"] = folders[f"{tv['out']}_try3__sim_attack"] = "sim_attack"
assert "sp493_g6_sim_x_s2_v1__sim_attack" in folders and "sp493_g3_sim_y_v1__sim_attack" in folders
assert not [n for n in ("sp493_g6_sim_z_v1__sim_attack", "sp493_g6_sim_xy_v1__sim_attack", "sp493_g6_simh_x_v1__sim_honest", "sp493_g6_sim_s1_x_v1__sim_attack")
            if any(re.fullmatch(r, n) for r in RX.values())]
wrong = {f: (want, [k for k, r in RX.items() if re.fullmatch(r, f)]) for f, want in folders.items() if [k for k, r in RX.items() if re.fullmatch(r, f)] != [want]}
assert not wrong, wrong
assert not any(re.fullmatch(RX["sim_honest"], n) for n in ("simh_t196fill_v1__sim_honest", "ilnm_s4rgap_sim2_v1__sim_attack"))
DRY = ("sp493_a_main_v1_dry__attack", "sp493_b_run2_v1_dry__attack", "sp493_c_main_v1_dry__honest_a", "sp493_g1_sim_v1_dry__sim_attack",
       "sp493_g2_sim_s3_v1_dry__sim_attack", "sp493_g4_simh_v1_dry__sim_honest",
       "simh_t196fill_v1_regress1__sim_honest")  # dry-run / regression sets (d9 10-15 ~23:15 / ~23:45): no regex may accept them
assert not [n for n in DRY for r in RX.values() if re.fullmatch(r, n)], "a dry-run name matches a routing regex"
T196FILL = r"^simh_t196fill(?:_s[0-9]+)?_v1(?:_try[0-9]+)?__sim_honest$"  # 09's t196fill regex: the regression set is refused too
assert not re.fullmatch(T196FILL, "simh_t196fill_v1_regress1__sim_honest") and re.fullmatch(T196FILL, "simh_t196fill_v1__sim_honest")
V = {"facts": {"roster": ["c1", "c2", "c3"], "not_counted": []}, "members": ["c1", "c2", "c3"], "references": ["c1.eval", "c2.eval", "c3.eval"],
     "counted_without_eval": [{"path": "r/sim_attack@september/c2", "arm": "sim_attack@september", "task": "c2"},
                              {"path": "r/sim_attack@september/c3", "arm": "sim_attack@september", "task": "c3"}],
     "launch_subset": '{"runs": [{"task": "c1"}, {"task": "c3"}]}', "files": [{"path": "launch.json"}, {"path": "r/c1.eval"}, {"path": "o/c1.eval"}, {"path": "r/c3.eval"}, {"path": "o/c3.eval"}],
     "manifests": {"main": {"counted_attempts_without_eval": {"main": ["r/sim_attack@september/c2", "r/sim_attack@september/c3"]},
                            "files": [{"path": "r/c1.eval", "task": "c1", "world": "sp493_sim", "arm_dir": "sim_attack@september"},
                                      {"path": "o/c1.eval", "task": "c1", "world": "honest_a", "arm_dir": "honest_a"},
                                      {"path": "r/c3.eval", "task": "c3", "world": "sp493_sim", "arm_dir": "sim_attack@september"},
                                      {"path": "o/c3.eval", "task": "c3", "world": "honest_a", "arm_dir": "honest_a"}]}}}
gv = only_cells_view(json.loads(json.dumps(V)), {"c1", "c2"}, il=False)
assert gv["members"] == ["c1", "c2"] and gv["facts"]["roster"] == ["c1", "c2"] and gv["references"] == ["c1.eval", "c2.eval"]
assert gv["manifests"]["main"]["counted_attempts_without_eval"] == {"main": ["r/sim_attack@september/c2"]} and [x["task"] for x in gv["counted_without_eval"]] == ["c2"]  # c3's (another group) dropped
assert [f["path"] for f in gv["manifests"]["main"]["files"]] == ["r/c1.eval", "o/c1.eval"] and [f["path"] for f in gv["files"]] == ["launch.json", "r/c1.eval", "o/c1.eval"]
assert [f["path"] for f in only_cells_view(json.loads(json.dumps(V)), {"c1"})["manifests"]["main"]["files"]] == []  # the IL view: run3/sim2 rows only
print(f"PASS: {len(cases)} guard cases; 3 ambiguous-name cases; group view (index donors kept); {len(DRY)} dry-run names refused; {len(folders)} S_paper493 folders each match exactly one of 09's regexes: {sorted(folders)}")
