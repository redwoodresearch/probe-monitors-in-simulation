import json, sys, numpy as np
from collections import Counter
rows = json.load(open(sys.argv[1]))
S=["simulated","Simulated"," simulated"," Simulated"]
def p(lp):
    a=sum(np.exp(lp[k]) for k in S); t=sum(np.exp(v) for v in lp.values()); return a/t
print("cut by (arm,version):", sorted(Counter((r["arm"],r["version"]) for r in rows if r.get("cut")).items()))
for a in ("e1","e2","sim"):
    o=[r for r in rows if r["arm"]==a and r["version"]=="original"]
    print(a, "median n_tokens", np.median([r["n_tokens"] for r in o]), "cut frac", np.mean([bool(r["cut"]) for r in o]))
for v in ("original","sim_line","prod_line","neutral_line"):
    x=[p(r["logprobs"]) for r in rows if r["version"]==v and r["arm"] in("e1","e2")]
    print(v, "n", len(x), "meanP %.4f"%np.mean(x), "minP %.4f maxP %.4f"%(min(x),max(x)))
print("cells per task:", sorted(Counter(Counter(r["task"] for r in rows if r["arm"]=="sim").values()).items()))
