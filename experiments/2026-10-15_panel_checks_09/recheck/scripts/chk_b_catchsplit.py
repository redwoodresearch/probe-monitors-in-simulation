"""Independent check of panel check (b): membership from v6 catches, exact gate vs v6, low-group values."""
import json, math, sys
cs = json.load(open(sys.argv[1])); v6 = json.load(open(sys.argv[2])); out_path = sys.argv[3]
C, V = cs["result"]["results"], v6["result"]["results"]

def diff(a, b, path, acc):
    """structural exact comparison; bools never equal numbers; NaN==NaN"""
    if type(a) is bool or type(b) is bool:
        if not (type(a) is bool and type(b) is bool and a == b): acc["mism"].append(path)
        return
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if math.isnan(a) and math.isnan(b): return
        if math.isnan(a) or math.isnan(b): acc["mism"].append(path); return
        acc["maxd"] = max(acc["maxd"], abs(a - b)); acc["nnum"] += 1
        if a != b: acc["mism"].append(path)
        return
    if type(a) != type(b): acc["mism"].append(path + f" type {type(a).__name__}/{type(b).__name__}"); return
    if isinstance(a, dict):
        if set(a) != set(b): acc["mism"].append(path + f" keys {sorted(set(a) ^ set(b))}")
        for k in set(a) & set(b): diff(a[k], b[k], f"{path}.{k}", acc)
    elif isinstance(a, list):
        if len(a) != len(b): acc["mism"].append(path + " len"); return
        for i, (x, y) in enumerate(zip(a, b)): diff(x, y, f"{path}[{i}]", acc)
    elif a != b: acc["mism"].append(path)

res = {"rows_cs": len(C), "rows_v6": len(V), "same_row_keys": sorted(C) == sorted(V)}
# gate: primary block, plus every other v6 key in the row
gate = {"maxd": 0.0, "nnum": 0, "mism": []}
other = {"maxd": 0.0, "nnum": 0, "mism": []}
for row in V:
    diff(C[row]["primary_eligible_estimation_half"], V[row]["primary_eligible_estimation_half"], row, gate)
    for k in V[row]:
        if k != "primary_eligible_estimation_half":
            diff(C[row].get(k), V[row][k], f"{row}.{k}", other)
extra = {row: sorted(set(C[row]) - set(V[row])) for row in V}
res["gate_primary"] = {"rows": len(V), "numeric_leaves": gate["nnum"], "max_abs_diff": gate["maxd"], "mismatches": gate["mism"][:10], "n_mism": len(gate["mism"])}
res["other_v6_keys"] = {"numeric_leaves": other["nnum"], "max_abs_diff": other["maxd"], "n_mism": len(other["mism"]), "first": other["mism"][:5]}
res["extra_keys_in_cs"] = sorted({k for ks in extra.values() for k in ks})
res["cs_gate_field"] = cs["result"]["gates"]["v6_primary_equal"]

# membership from v6's own catches (estimation half), independent of the split block
mem = {}; lines = []
for row in V:
    elig = V[row]["eligibility_append13_4"]["eligible"]
    cp = V[row]["catches_percent"]
    rc = {p: (cp[p]["run1"]["estimation_half"] + cp[p]["run2"]["estimation_half"]) / 2 for p in elig}
    hi = [p for p in elig if rc[p] >= 50]; lo = [p for p in elig if rc[p] < 50]
    s = C[row]["catch_level_split_posthoc"]
    cs_hi = s["groups"].get("high_real_catch_ge_50", {}).get("members", [])
    cs_lo = s["groups"].get("low_real_catch_lt_50", {}).get("members", [])
    rc_cs = s["real_catch_estimation_half"]
    rcd = max([abs(rc[p] - rc_cs[p]) for p in elig], default=0.0)
    ok = (hi == cs_hi and lo == cs_lo and sorted(rc_cs) == sorted(elig))
    near = [f"{p}:{rc[p]:.1f}" for p in elig if 40 <= rc[p] < 60]
    mem[row] = {"eligible": elig, "real_catch": rc, "high": hi, "low": lo, "cs_high": cs_hi, "cs_low": cs_lo, "match": ok, "max_rc_diff": rcd, "near_50": near}
    lines.append(f"{row}: elig={elig} high={hi} low={lo} match={ok} maxdiff={rcd:.1e} near50={near}")
res["membership"] = mem

# low/high group X_err and U where >=2 members; per-probe deltas vs v6 full-family per-probe deltas
grp = {}
for row in V:
    for g, x in C[row]["catch_level_split_posthoc"]["groups"].items():
        if not x.get("members"): continue
        for contrast in ("sim_minus_mean_real", "reexecution_run2_minus_run1"):
            for st in ("pooled", "with_strategy", "without_strategy"):
                b = x[contrast][st]
                pp_dev = None
                vpp = V[row]["primary_eligible_estimation_half"][contrast][st].get("per_probe", {})
                if "per_probe" in b:
                    pp_dev = max(abs(b["per_probe"][p]["delta"] - vpp[p]["delta"]) for p in b["per_probe"])
                grp[f"{row}|{g}|{contrast}|{st}"] = {"members": x["members"], "X_err": b.get("X_err"), "U": b.get("U_approx_upper_95"), "n_tasks": b.get("n_tasks"), "per_probe_delta_maxdiff_vs_v6": pp_dev}
res["groups"] = grp
json.dump(res, open(out_path, "w"), indent=1, default=str)
print("rows", res["rows_cs"], res["rows_v6"], "same keys", res["same_row_keys"])
print("GATE primary_eligible_estimation_half:", res["gate_primary"])
print("other v6 keys in each row:", res["other_v6_keys"], "extra keys in cs rows:", res["extra_keys_in_cs"])
print("record gate field:", res["cs_gate_field"])
print("\n".join(lines))
print("membership all match:", all(m["match"] for m in mem.values()))
for k in ["R_reply_meanpooled|mechanical|alpha=0.05", "R_reply_meanpooled|mechanical|alpha=0.01"]:
    for st in ("pooled", "with_strategy"):
        g = grp[f"{k}|low_real_catch_lt_50|sim_minus_mean_real|{st}"]
        print(k, "low", st, g)
print("all low-group sim_minus_mean_real X_err (pooled/with_strategy) where >=2 members:")
for k, g in grp.items():
    if "low_real" in k and "sim_minus_mean_real" in k and k.split("|")[-1] in ("pooled", "with_strategy") and g["X_err"] is not None:
        print("  ", k, "X", g["X_err"], "U", round(g["U"], 2), "ppdiff", g["per_probe_delta_maxdiff_vs_v6"])
