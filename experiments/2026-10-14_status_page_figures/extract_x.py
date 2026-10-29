# Pull every number for the two top figures by key from committed records (git show). No compute beyond selection.
import gzip, io, json, subprocess
R = "/Users/aswinsurya/probe-distribution-shift-from-sims"
def raw(sha, path): return subprocess.check_output(["git", "-C", R, "show", f"{sha}:{path}"])
def gs(sha, path): return json.loads(raw(sha, path))
OUT = {}
# v6 (registered headline), estimation half, mechanical pool
v6 = gs("0ed287c95", "experiments/2026-10-14_xerr_prelim_09/results/xerr_v6_2026-10-14_161122.json")["result"]["results"]
CELLS = [("R_reply_meanpooled", "0.01", "Replies, 1%"), ("R_reply_meanpooled", "0.05", "Replies, 5%"), ("C_anycommand_max", "0.01", "Commands, 1%"), ("C_anycommand_max", "0.05", "Commands, 5%")]
def fam(b): return None if b is None or b.get("X_err") is None else {"X": b["X_err"], "U": b["U_approx_upper_95"], "bias": b.get("bias_mean"), "bias_ci": b.get("bias_ci95")}
for row, a, lab in CELLS:
    e = v6[f"{row}|mechanical|alpha={a}"]["primary_eligible_estimation_half"]
    d = OUT.setdefault(lab, {})
    for st in ("with_strategy", "pooled", "without_strategy"):
        d[f"v6_{st}"] = fam(e["sim_minus_mean_real"].get(st))
        d[f"v6_rerun_{st}"] = fam(e["reexecution_run2_minus_run1"].get(st))
    pp = {}
    for st in ("with_strategy", "pooled"):
        for comp in ("sim_minus_mean_real", "reexecution_run2_minus_run1"):
            for p, v in (e[comp].get(st, {}).get("per_probe") or {}).items():
                if "delta" in v: pp.setdefault(p, {})[f"{comp}|{st}"] = [v["delta"], v["ci95"][0], v["ci95"][1]]
    d["v6_per_probe"] = pp
# v7 expanded family (primary)
v7 = gs("e109b2a9", "experiments/2026-10-14_xerr_prelim_09/results/xerr_v7_phase2_2026-10-14_181336.json")["result"]["results"]
for (row, a, lab), k in zip(CELLS, ("R|0.01", "R|0.05", "C|0.01", "C|0.05")):
    p = v7[k]["primary"]
    for st in ("with_strategy", "pooled"):
        OUT[lab][f"v7_{st}"] = fam(p["sim_minus_mean_real"].get(st))
        OUT[lab][f"v7_rerun_{st}"] = fam(p["reexecution_run2_minus_run1"].get(st))
# A25 fully simulated (alarm line on simulated honest), plus the matched real yardstick's upper limit
a25 = gs("1f69e2e67", "experiments/2026-10-14_simhonest_a25_09/results/a25_full_2026-10-14_232415.json")["result"]
for row, a, lab in CELLS:
    r = a25["rows"].get(f"{row}|mechanical|alpha={a}", {})
    p = r.get("primary", {})
    for st in ("with_strategy", "pooled"):
        OUT[lab][f"a25_{st}"] = fam(p.get("full_minus_mean_real", {}).get(st))
    rd = a25["append25c_reading"]["rows"].get(f"{row}|mechanical|alpha={a}", {}).get("X_err_full", {})
    OUT[lab]["a25_yardstick_U"] = rd.get("ref_U"); OUT[lab]["a25_reading"] = rd.get("reading")
# served-context sensitivity (layout A), pooled 90 tasks
sv = gs("baae12ac1", "experiments/2026-10-14_served_check_09/results/served_read_real_2026-10-14_212330.json")["result"]["scenarios"]["real"]["x_err"]
for row, a, lab in CELLS:
    x = sv.get(f"{row}|alpha={a}", {})
    OUT[lab]["served"] = {lay: fam(x[lay]) for lay in ("reader", "A0", "A", "B") if isinstance(x.get(lay), dict)}
# FPR sweep (p1_main), reply readout rows as drawn in M4
rows = [json.loads(l) for l in gzip.open(io.BytesIO(raw("389c813ff", "experiments/2026-10-14_q35_sol_avg_POSTHOC/results/p1_main_2026-10-14_162810.jsonl.gz")), "rt")]
sweep = {r["probe_id"]: r for r in rows if r["row_type"] == "sweep"}
REPLY = {"GPT-5.6-Sol-trained": "sol_a|avg_top2|corrected|none|tight", "Published deception (FIBS)": "fibs|reply_mean|corrected|none",
         "Harmfulness direction": "harmfulness|reply_mean|corrected|none", "Refusal direction": "refusal|reply_mean|corrected|none",
         "High-stakes": "highstakes|reply_mean|corrected|none", "Positive control": "c_asis|avg_top2|corrected|none"}
OUT["sweep"] = {}
for nm, pid in REPLY.items():
    s = sweep[pid]
    OUT["sweep"][nm] = {"probe_id": pid, "calibration": s.get("calibration"), "n_honest_cal": s.get("n_honest_cal"), "n_tasks": s.get("n_tasks"),
        "points": [{"fpr": p["fpr_target"], "k": p["k"], "real": [p["tpr"]["real_mean"]["point"]] + p["tpr"]["real_mean"]["ci95"],
                    "sim": [p["tpr"]["sim"]["point"]] + p["tpr"]["sim"]["ci95"], "d": [p["delta"]["point"]] + (p["delta"]["ci95"] or [None, None]),
                    "rr": [p["control_run2_minus_run1"]["point"]] + (p["control_run2_minus_run1"]["ci95"] or [None, None])} for p in s["points"]]}
json.dump(OUT, open("/private/tmp/claude-501/-Users-aswinsurya-probe-shift-viewer/b3b0fe45-ac17-4911-908e-445ff32d8b31/scratchpad/status/figx/data.json", "w"), indent=1)
for lab in [c[2] for c in CELLS]:
    d = OUT[lab]; f = lambda x: None if not x else (round(x["X"], 2), round(x["U"], 2))
    print(lab, "v6ws", f(d["v6_with_strategy"]), "rr", f(d["v6_rerun_with_strategy"]), "v7ws", f(d["v7_with_strategy"]), "v7pool", f(d["v7_pooled"]), "a25ws", f(d["a25_with_strategy"]), "a25pool", f(d["a25_pooled"]), "yard", d["a25_yardstick_U"] and round(d["a25_yardstick_U"], 2), "servedA", f(d["served"].get("A")))
for nm, s in OUT["sweep"].items(): print(nm, s["calibration"], s["n_honest_cal"], len(s["points"]), "fpr", round(s["points"][0]["fpr"], 5), "-", round(s["points"][-1]["fpr"], 3))
