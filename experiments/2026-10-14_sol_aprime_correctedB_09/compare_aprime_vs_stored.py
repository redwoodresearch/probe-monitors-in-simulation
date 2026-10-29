"""Pre-stated decision quantities (RULE.md here): (a') vs a_stored, like-for-like, through X_err v6's machinery. Prints and
writes the moves; a move >= 5 points in any of them triggers the Sol re-stage quote (POSTHOC Amendment 22)."""
import json
import sys

ap, st = (json.load(open(f))["result"] for f in sys.argv[1:3])
assert ap["gates"]["pass"] and st["gates"]["pass"] and ap["variant"]["sol_fit"] == "aprime" and st["variant"]["sol_fit"] == "stored"
rows, worst = {}, 0.0
for ro in ("R_reply_meanpooled", "C_anycommand_max"):
    for a in ("0.01", "0.05"):
        k = f"{ro}|mechanical|alpha={a}"
        pa = ap["results"][k]["primary_eligible_estimation_half"]["sim_minus_mean_real"]
        ps = st["results"][k]["primary_eligible_estimation_half"]["sim_minus_mean_real"]
        cap, cst = ap["results"][k]["catches_percent"]["sol_a"], st["results"][k]["catches_percent"]["sol_a"]
        d_ap = cap["sim"]["estimation_half"] - (cap["run1"]["estimation_half"] + cap["run2"]["estimation_half"]) / 2
        d_st = cst["sim"]["estimation_half"] - (cst["run1"]["estimation_half"] + cst["run2"]["estimation_half"]) / 2
        r = {"sol_delta": [d_st, d_ap, d_ap - d_st],
             "eligible": [st["results"][k]["eligibility_append13_4"]["eligible"], ap["results"][k]["eligibility_append13_4"]["eligible"]]}
        for q in ("X_err", "U_approx_upper_95"):
            xs, xa = (ps.get("pooled") or {}).get(q), (pa.get("pooled") or {}).get(q)
            r[q] = [xs, xa, None if xs is None or xa is None else xa - xs]
        rows[k] = r
        worst = max([worst, abs(r["sol_delta"][2])] + [abs(r[q][2]) for q in ("X_err", "U_approx_upper_95") if r[q][2] is not None])
out = {"label": "(a') vs a_stored, like-for-like; points; [stored, aprime, move]", "rows": rows, "max_abs_move": worst,
       "decision": "Sol re-stage quote to Aswin (a move >= 5 points)" if worst >= 5 else f"Sol-side leak checked: moves the headline by at most {worst:.2f} points"}
print(json.dumps(out, indent=1))
