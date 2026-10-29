"""Follow-up: TEST records hold no U_rt numbers; stratum labels; certification on rt blocks. Usage: python -I check_thr2.py in/"""
import json, sys, glob
d = sys.argv[1]
for f in sorted(glob.glob(d + "test_*.json")):
    t = json.load(open(f)); res = t.get("result", t)
    print(f.split("/")[-1], "keys", sorted(res.keys()), "rows" in res, "script_sha", (res.get("provenance") or t.get("provenance") or {}).get("script_sha256_in_container", "")[:16], "pass", res["test_checks"]["pass"])
r = json.load(open(d + "real.json"))
print("real script_sha", r["result"]["provenance"]["script_sha256_in_container"][:16])
for rk, row in r["result"]["rows"].items():
    for cn, cb in row["comparisons"].items():
        for sc, b in cb.items():
            labs = {k: b[k].get("certification") for k in ("fixed_threshold_reproduced", "rt_joint_U_rt", "rt_independent_sensitivity")}
            unst = {k: b[k].get("unstable_over_1pct_nan") for k in ("rt_joint_U_rt", "rt_independent_sensitivity")}
            if sc == "without_strategy" or any(v not in (None, "ok (>= 95% complete draws)") for v in labs.values()) or any(unst.values()):
                print(rk[:22], rk[-4:], cn[:6], sc, "label:", b.get("stratum_label"), "empty_eval:", b["empty_evaluation_weight_draws"], "cert:", set(labs.values()), "unstable:", unst)
