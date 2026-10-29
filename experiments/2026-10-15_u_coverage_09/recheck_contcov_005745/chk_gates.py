# Check G0/G0c reported values against the pinned reference records (independent read).
import json, sys
d = sys.argv[1]
rec = json.load(open(f"{d}/record.json"))["result"]["setup"]
v6 = json.load(open(f"{d}/xerr_v6.json"))["result"]["results"]
vc = json.load(open(f"{d}/xerr_cont_v6.json"))["result"]["results"]
w0 = 0.0; wref0 = 0.0
for k, v in rec["G0"]["rows"].items():
    rk, sc = k.rsplit("|", 1); ro, al = rk.split("|alpha=")
    ref = v6[f"{ro}|mechanical|alpha={al}"]["primary_eligible_estimation_half"]["sim_minus_mean_real"][sc]
    if isinstance(v, str):
        print("G0", k, v, "ref X_err", ref.get("X_err")); continue
    wref0 = max(wref0, abs(v["v6_X_err"] - ref["X_err"]), abs(v["v6_U"] - ref["U_approx_upper_95"]))
    w0 = max(w0, abs(v["X_err"] - ref["X_err"]), abs(v["U"] - ref["U_approx_upper_95"]))
    print("G0", k, "X", v["X_err"], "U", v["U"])
print("G0 pass", rec["G0"]["pass"], "recorded max", rec["G0"]["max_abs_diff"], "| my max |mine-v6file|", w0, "| ref-copy vs file", wref0)
wc = 0.0; wrefc = 0.0
for k, v in rec["G0c"]["rows"].items():
    rk, st = k.rsplit("|", 1); ro, al = rk.split("|alpha=")
    ref = vc[f"{ro}|mechanical|alpha={al}"]["continuous"][st]["family_mapped_points"]
    wrefc = max(wrefc, abs(v["ref_X"] - ref["X_err"]), abs(v["ref_U"] - ref["U_approx_upper_95"]))
    wc = max(wc, abs(v["X_map"] - ref["X_err"]), abs(v["U_map"] - ref["U_approx_upper_95"]))
    print("G0c", k, "Xmap", v["X_map"], "Umap", v["U_map"], "fileX", ref["X_err"], "fileU", ref["U_approx_upper_95"])
print("G0c pass", rec["G0c"]["pass"], "recorded max", rec["G0c"]["max_abs_diff"], "| my max |mine-file|", wc, "| ref-copy vs file", wrefc)
print("cp_gate_max_abs_diff", rec["cp_gate_max_abs_diff"])
