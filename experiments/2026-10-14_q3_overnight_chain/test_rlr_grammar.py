"""Grammar test for _rlr<k> (09 @ 502c1b05e): every existing simh root name and the 3 new ones parse under 09's regex; the chain's sim_tag
and run-table patterns accept rlr / rlr<k>=2.. and reject rlr0 / rlr1; the chain's stem rule yields the registered folders."""
import glob, json, os, re, sys
sys.path.insert(0, sys.argv[1])
src = open(os.path.join(sys.argv[1], "chain.py")).read()
RLR_TAG = re.compile(re.search(r'RLR_TAG = re.compile\(r"([^"]+)"\)', src).group(1))
RLR_TABLE = re.compile(re.search(r'RLR_TABLE = re.compile\(r"([^"]+)"\)', src).group(1))
G = re.compile(r"^(?:simh311_q[1-4]|simh_nm360_(?:s1|s2|s3|s4d|s4r|s5))_sim(?:_s[0-9]+|_rlr(?:[2-9]|[1-9][0-9]+)?)?_v1(?:_try[0-9]+)?__sim_honest$")
routed = sorted({json.load(open(p))["staged_set"] + "__sim_honest" for p in glob.glob(os.path.join(sys.argv[2], "simh-*__*.json")) if json.load(open(p)).get("staged_set")})
bad = [r for r in routed if not G.match(r)]
new = ["simh311_q1_sim_rlr2_v1__sim_honest", "simh311_q2_sim_rlr2_v1__sim_honest", "simh_nm360_s4d_sim_rlr2_v1__sim_honest"]
neg = ["simh311_q1_sim_rlr1_v1__sim_honest", "simh311_q1_sim_rlr0_v1__sim_honest", "simh311_q1_sim_rlr02_v1__sim_honest"]
stem = lambda pool, q, tag: "_".join(x for x in (pool, q, tag) if x) + "_v1__sim_honest"  # chain targets_for (fu, final) + staged set + folder
out = {"existing_routed_roots": len(routed), "existing_not_matching": bad, "new_match": {n: bool(G.match(n)) for n in new},
       "negatives_rejected": {n: not G.match(n) for n in neg},
       "tag": {t: bool(RLR_TAG.fullmatch(t)) for t in ("sim_rlr", "sim_rlr2", "sim_rlr10", "sim_rlr1", "sim_rlr0", "sim_rlr02", "sim", "sim_rlr_2")},
       "table": {t: bool(RLR_TABLE.fullmatch(t)) for t in ("run_table_rlr", "run_table_rlr2", "run_table_rlr1", "run_table_rlr0", "run_table_snap")},
       "stems": {s: s in new for s in (stem("simh311", "q1", "sim_rlr2"), stem("simh311", "q2", "sim_rlr2"), stem("simh_nm360", "s4d", "sim_rlr2"))},
       "k1_unchanged": stem("simh_nm360", "s3", "sim_rlr") == "simh_nm360_s3_sim_rlr_v1__sim_honest"}
ok = (not bad and all(out["new_match"].values()) and all(out["negatives_rejected"].values()) and out["tag"] == {"sim_rlr": True, "sim_rlr2": True, "sim_rlr10": True,
      "sim_rlr1": False, "sim_rlr0": False, "sim_rlr02": False, "sim": False, "sim_rlr_2": False} and out["table"] == {"run_table_rlr": True, "run_table_rlr2": True,
      "run_table_rlr1": False, "run_table_rlr0": False, "run_table_snap": False} and all(out["stems"].values()) and out["k1_unchanged"])
out["pass"] = ok
print(json.dumps(out, indent=1))
sys.exit(0 if ok else 1)
