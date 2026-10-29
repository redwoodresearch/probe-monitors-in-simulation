import json, random, collections
import pandas as pd, numpy as np
D = json.load(open("/private/tmp/claude-501/-Users-aswinsurya-probe-shift-viewer/b3b0fe45-ac17-4911-908e-445ff32d8b31/scratchpad/verify/behavior/records.json"))
df = pd.DataFrame(D["records"])
df["grp"] = np.where(df.world == "sim", df.arm + "@" + df.simulator.fillna(""), df.arm)
df["sh_bash"] = df.n_bash / df.n_calls
df["sh_view"] = df.n_view / df.n_calls
df["sh_edit"] = (df.n_create + df.n_str_replace + df.n_insert) / df.n_calls
df["sh_ro"] = df.n_ro / df.n_calls
df["sh_sc"] = df.n_sc / df.n_calls
df["sh_errfield"] = df.n_errfield / df.n_out
df["sh_failtext"] = df.n_failtext / df.n_out
df["reread_rate"] = df.rereads / df.edits.replace(0, np.nan)
print("runs per group:", df.grp.value_counts().sort_index().to_dict())
print("attempt != of:", df[df.attempt != df["of"]][["key","attempt","of"]].values.tolist())
print("states:", df.state.value_counts().to_dict())
print("other functions:", collections.Counter(f for fs in df.other_fns for f in fs))
def q(s):
    s = s.dropna()
    return f"{s.median():.3g} [{s.quantile(.25):.3g}, {s.quantile(.75):.3g}] (min {s.min():.3g}, max {s.max():.3g})"
def auc(a, b):  # P(a > b) + 0.5 P(tie); 0.5 = no separation
    a, b = np.asarray(a.dropna()), np.asarray(b.dropna())
    return ((a[:, None] > b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum()) / (len(a) * len(b))
measures = ["n_steps", "n_calls", "sh_bash", "sh_view", "sh_edit", "sh_ro", "sh_sc", "n_errfield", "sh_errfield", "sh_failtext", "edits", "rereads", "reread_rate"]
order = {"attack": ["attack", "sim_attack@september", "sim_attack@today"], "honest": ["honest_b", "sim_honest@september", "sim_honest@today"]}
print("\n=== CLAIM 1: per-run distributions, median [Q1, Q3] (min, max); AUC = P(sim > real), 0.5 = full overlap ===")
for kind, gs in order.items():
    print(f"\n--- {kind} ---")
    for m in measures:
        line = f"{m:12s} " + " | ".join(f"{g}: {q(df[df.grp == g][m])}" for g in gs)
        real = df[df.grp == gs[0]][m]
        line += " | AUC sept %.2f today %.2f" % (auc(df[df.grp == gs[1]][m], real), auc(df[df.grp == gs[2]][m], real))
        print(line)
print("\nreference honest_a for context:", {m: q(df[df.grp == "honest_a"][m]) for m in ["n_steps", "sh_bash", "sh_failtext", "reread_rate"]})
print("\ntool mix totals (sum over runs):")
print(df.groupby("grp")[["n_calls", "n_bash", "n_view", "n_create", "n_str_replace", "n_insert", "n_submit", "n_other", "n_ro", "n_sc", "n_unc"]].sum().to_string())
print("\nending (how runs end):")
print(pd.crosstab(df.grp, df.end).to_string())
print("sample errors:", df[df.sample_error][["key"]].values.ravel().tolist(), "limits:", df[df.limit.notna()][["key", "limit"]].values.tolist())
print("runs with >1 submit:", df[df.n_submit > 1][["key", "n_submit"]].values.tolist())

print("\n=== CLAIM 2: paired steps ===")
steps = df.set_index("key").n_steps
P = D["pairs"]
assert P == sorted(P, key=lambda p: (p["task"], p["arm"], p["simulator"] or "")), "pairs not sorted"
rng = random.Random(20261009)
idx = rng.sample(range(len(P)), 5)
print("seed 20261009, rng.sample(range(80), 5) ->", idx)
for i in idx:
    p = P[i]; print(f"  {p['sim']}: sim {steps[p['sim']]} vs real {p['real']} {steps[p['real']]} -> sim shorter: {steps[p['sim']] < steps[p['real']]}")
pp = pd.DataFrame([dict(p, s=steps[p["sim"]], r=steps[p["real"]]) for p in P]); pp["d"] = pp.s - pp.r
def summ(x): return f"n={len(x)} sim shorter {int((x.d < 0).sum())} ({(x.d < 0).mean():.0%}), ties {int((x.d == 0).sum())}, sim longer {int((x.d > 0).sum())}; median diff (sim-real) {x.d.median():+.1f} steps [Q1 {x.d.quantile(.25):+.1f}, Q3 {x.d.quantile(.75):+.1f}]; median ratio sim/real {(x.s / x.r).median():.2f}"
print("ALL:", summ(pp))
for (a, s), g in pp.groupby(["arm", "simulator"]): print(f"{a}@{s}:", summ(g))
for a, g in pp.groupby("arm"): print(f"{a} (both sims):", summ(g))

print("\n=== CLAIM 3: live-env problem strings (tool outputs) ===")
for name, col in (("password authentication failed", "pw_hits"), ("ModuleNotFoundError|No module named", "mod_hits")):
    print(f"[{name}]")
    for g, x in df.groupby("grp"):
        print(f"  {g:22s} runs with >=1: {int((x[col] > 0).sum())}/{len(x)}  outputs: {int(x[col].sum())}  runs: {x[x[col] > 0].key.tolist()}")
print("quotes (key, step, line):")
for k, v in D["quotes"].items():
    for row in v: print(" ", k, row)

print("\n=== CLAIM 4: curl vs pytest in honest runs (bash invocations) ===")
for g in ["honest_b", "honest_a", "sim_honest@september", "sim_honest@today", "attack", "sim_attack@september", "sim_attack@today"]:
    x = df[df.grp == g]
    print(f"  {g:22s} " + "  ".join(f"{c}: total {int(x[c].sum())}, runs>=1 {int((x[c] > 0).sum())}/{len(x)}, median/run {x[c].median():.0f}" for c in ["curl", "pytest", "pyscript", "shscript"]))

print("\n=== CLAIM 5: scores per group ===")
for g, x in df.groupby("grp"):
    keys = collections.Counter(tuple(k) for k in x.score_keys)
    vals = collections.Counter((k, v) for d in x.score_vals for k, v in d.items())
    print(f"  {g:22s} scorer keys {dict(keys)}  values {dict(sorted(vals.items(), key=str))}")
df.drop(columns=["score_vals"]).to_csv("/private/tmp/claude-501/-Users-aswinsurya-probe-shift-viewer/b3b0fe45-ac17-4911-908e-445ff32d8b31/scratchpad/verify/behavior/per_run.csv", index=False)
