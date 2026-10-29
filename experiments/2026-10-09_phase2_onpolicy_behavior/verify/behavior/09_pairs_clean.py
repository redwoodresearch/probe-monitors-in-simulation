import json, pandas as pd, numpy as np
D = json.load(open("records.json"))
df = pd.DataFrame(D["records"]).set_index("key")
pp = pd.DataFrame(D["pairs"])
pp["s"] = pp.sim.map(df.n_steps); pp["r"] = pp.real.map(df.n_steps); pp["d"] = pp.s - pp.r
pp["sim_err"] = pp.sim.map(df.sample_error)
print("pairs with errored (0-step) sim run:", pp[pp.sim_err][["sim", "s", "real", "r"]].values.tolist())
c = pp[~pp.sim_err]
def summ(x): return f"n={len(x)} sim shorter {int((x.d<0).sum())} ({(x.d<0).mean():.0%}), ties {int((x.d==0).sum())}, longer {int((x.d>0).sum())}; median diff {x.d.median():+.1f} [Q1 {x.d.quantile(.25):+.1f}, Q3 {x.d.quantile(.75):+.1f}]"
print("ALL excl. errored:", summ(c))
for a, g in c.groupby("arm"): print(f"  {a} excl. errored:", summ(g))
# sign test p-value (two-sided) for all 80 and clean, ignoring ties
from math import comb
for name, x in (("all80", pp), ("clean78", c)):
    k, n = int((x.d < 0).sum()), int((x.d != 0).sum())
    p = sum(comb(n, i) for i in range(n + 1) if abs(i - n / 2) >= abs(k - n / 2)) / 2 ** n
    print(f"sign test {name}: {k}/{n} sim shorter, two-sided p={p:.2f}")
# Inspect-flagged tool errors per group
d2 = pd.DataFrame(D["records"]); d2["grp"] = np.where(d2.world == "sim", d2.arm + "@" + d2.simulator.fillna(""), d2.arm)
print(d2.groupby("grp").agg(errfield_total=("n_errfield", "sum"), runs_with_errfield=("n_errfield", lambda s: int((s > 0).sum())), tool_outputs=("n_out", "sum")).to_string())
