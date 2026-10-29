import json, sys
import numpy as np
IN = sys.argv[1]
a = json.load(open(f"{IN}/asis_scores_142228.json")); v = json.load(open(f"{IN}/xerr_v6_2026-10-14_161122.json"))
sc = a["run_scores"]["avg_top2"]["a_tight"]; folds = a["run_folds"]
row = v["result"]["results"]["R_reply_meanpooled|mechanical|alpha=0.01"]; vs = row["standardization_sol"]
pool = [k for k in sc if k.split("|")[0] in ("honest_b", "honest_c")]
mu, sd = {}, {}
for f in range(5):
    x = np.array([sc[k] for k in pool if folds[k] == f]); mu[f], sd[f] = float(x.mean()), float(x.std())
print("moments exact:", all(mu[f] == vs["mu"][str(f)] and sd[f] == vs["sd"][str(f)] for f in range(5)))
z = {k: (sc[k] - mu[folds[k]]) / sd[folds[k]] for k in sc}
pz = np.sort(np.array([z[k] for k in pool]))[::-1]; kk = int(np.floor(0.01 * len(pz) + 1e-9)); T = float(pz[kk])
print("threshold", repr(T), "exact:", T == row["thresholds"]["sol_a"]["threshold"], "n", len(pz), "k", kk)
cohort = v["result"]["cohort"]["tasks"]; est = set(v["result"]["universe_and_split"]["estimation_half"]); ce = [t for t in cohort if t in est]
cp = row["catches_percent"]["sol_a"]
for arm in ("run1", "run2", "sim"):
    al = 100 * float(np.mean([z[f"{arm}|{t}"] > T for t in cohort])); es = 100 * float(np.mean([z[f"{arm}|{t}"] > T for t in ce]))
    print(arm, al, es, "exact:", al == cp[arm]["all_cohort"], es == cp[arm]["estimation_half"])
