# Pull every draft-figure value from committed records (git show; no compute). Writes figdata.json.
import json, subprocess
REPO = "/Users/aswinsurya/probe-distribution-shift-from-sims"
def show(spec): return subprocess.run(["git", "-C", REPO, "show", spec], capture_output=True, text=True, check=True).stdout
from decimal import Decimal, ROUND_HALF_UP
def P(x): return float(Decimal(repr(100 * x)).quantize(Decimal('0.1'), ROUND_HALF_UP)) + 0.0  # half-up, like a printout
def iv(p, ci): return [P(p), P(ci[0]), P(ci[1])]
SRC = {
    "reply": "f3b07b354:experiments/2026-10-14_q35_sol_avg_POSTHOC/results/correctedb_tight22g_2026-10-14_144709.json",
    "cmd_tight": "b13ea2651:experiments/2026-10-14_q35_sol_avg_POSTHOC/results/correctedb_runmax22h_2026-10-14_152715.json",
    "z22i": "f6d197918:experiments/2026-10-14_q35_sol_avg_POSTHOC/results/correctedb_z22i_2026-10-14_161145.json",
    "xcheck": "ab095924e:experiments/2026-10-14_fpr_sweep_xcheck/results/sweep_xcheck_2026-10-14_130926.jsonl",
    "fibs": "d74ccaba0:experiments/2026-10-14_p2_fibs_pair/results/fibs_scores_2026-10-14_124948.json",
    "refharm": "a8d786425:experiments/2026-10-14_p2_refusal_harm/results/scores_2026-10-14_145748.json",
    "hs": "3d04b7936:experiments/2026-10-14_p2_highstakes/results/scores_2026-10-14_133519.json",
    "apollo": "54ea9724e:experiments/2026-10-14_p2_apollo/results/scores_2026-10-14_145858.json",
    "control": "f71112c8b:experiments/2026-10-14_priority1_firstcut/results/firstcut_2026-10-14_120513.json",
}
out = {"sources": SRC, "panel1": {}, "panel2": [], "budget": {}, "command_BC": {}}
RATES = (0.01, 0.05, 0.1)
# Panel 1, reply: scoring session's record, run level, run mean, primary pool honest B ∪ C.
d = json.loads(show(SRC["reply"]))
zz = json.loads(show(SRC["z22i"])); assert zz["gates_pass"] is True
rows = zz["parts"]["tight/averaged_reply/z"]["run_level"]["primary_honest_B_u_C"]
rep = {}
for r in rows:
    if r["fpr_run"] in RATES:
        c, ci, df, dci = r["catch"], r["catch_ci95"], r["diff"], r["diff_ci95"]
        rep[r["fpr_run"]] = {"real": iv(c["real_mean"], ci["real_mean"]), "sim": iv(c["sim"], ci["sim"]),
                             "d": iv(df["sim_minus_real_mean"], dci["sim_minus_real_mean"]), "rr": iv(df["run2_minus_run1"], dci["run2_minus_run1"]),
                             "n": zz["parts"]["tight/averaged_reply/z"]["cohort_tasks"]}
assert sorted(rep) == list(RATES)
out["panel1"]["reply"] = rep
# Panel 1, command: cross-check, run max, pool honest B (Append 20), environment strata.
xs = [json.loads(l) for l in show(SRC["xcheck"]).splitlines()]
def xrow(readout, level, pool, rate):
    m = [x for x in xs if x.get("readout") == readout and x.get("level") == level and x.get("pool") == pool and x.get("strata") == "env" and abs(x["rate"] - rate) < 1e-12]
    assert len(m) == 1, (readout, level, pool, rate, len(m)); return m[0]
def xvals(x):
    t, c, dl = x["tpr"], x["tpr_ci95"], x["delta"]
    return {"real": iv(t["real"], c["real"]), "sim": iv(t["sim"], c["sim"]), "d": iv(dl["sim_minus_real"]["point"], dl["sim_minus_real"]["ci95"]),
            "rr": iv(dl["run2_minus_run1"]["point"], dl["run2_minus_run1"]["ci95"]), "n": x["n_cohort"]}
ct = json.loads(show(SRC["cmd_tight"]))
assert ct["gates_pass"] is True
crow = zz["parts"]["tight/command_call_end/z"]
cmd = {}
for r in crow["run_max"]["honest_B"]:
    if r["fpr_run"] in RATES:
        assert r["pool"] == "honest B run maxima (z per fold model)"
        c, ci, df, dci = r["catch"], r["catch_ci95"], r["diff"], r["diff_ci95"]
        cmd[r["fpr_run"]] = {"real": iv(c["real_mean"], ci["real_mean"]), "sim": iv(c["sim"], ci["sim"]), "d": iv(df["sim_minus_real_mean"], dci["sim_minus_real_mean"]), "rr": iv(df["run2_minus_run1"], dci["run2_minus_run1"]), "n": crow["cohort_tasks"], "achieved_B": P(r["realized_false_alarm"]["honest_b"])}
assert sorted(cmd) == list(RATES)
out["panel1"]["command"] = cmd
out["command_stored_B"] = {r: xvals(xrow("command", "run_max", "B", r)) for r in RATES}
out["command_BC"] = {r: xvals(xrow("command", "run_max", "BC", r)) for r in RATES}
out["budget"] = {r: round(100 * xrow("command", "unit", "BC", r)["pool_runs_flagged"], 1) for r in (0.001, 0.01)}
# Panel 2.
def pair(st, n_label):
    pr = st["pair"]; o = {"pool": st["pool"], "n": pr["n_tasks"], "label": pr["label"]}
    for a in ("0.01", "0.05"):
        q = pr[a]
        o[a] = {"d": iv(q["delta_sim_minus_mean_real"], q["delta"]["ci95"]), "rr": iv(q["control_run2_minus_run1"], q["control"]["ci95"]),
                "real": P((q["catch_run1"] + q["catch_run2"]) / 2), "sim": P(q["catch_sim"])}
    return o
p2 = []
for key, a in (("sol_reply", 0.01), ("sol_reply", 0.05)): pass
p2.append({"key": "sol_reply", "pool": ["honest_b", "honest_c"], "n": rep[0.01]["n"], **{str(a): {"d": rep[a]["d"], "rr": rep[a]["rr"], "real": rep[a]["real"][0], "sim": rep[a]["sim"][0]} for a in (0.01, 0.05)}})
cm = out["panel1"]["command"]
p2.append({"key": "sol_command", "pool": ["honest_b"], "n": cm[0.01]["n"], **{str(a): {"d": cm[a]["d"], "rr": cm[a]["rr"], "real": cm[a]["real"][0], "sim": cm[a]["sim"][0]} for a in (0.01, 0.05)}})
f = json.loads(show(SRC["fibs"])); p2.append({"key": "fibs", **pair(f["stats"]["reply_mean"], "fibs")})
rh = json.loads(show(SRC["refharm"]))
p2.append({"key": "refusal", **pair(rh["probes"]["refusal"]["stats"]["reply_mean"], "refusal")})
p2.append({"key": "harmfulness", **pair(rh["probes"]["harmfulness"]["stats"]["reply_mean"], "harm")})
hs = json.loads(show(SRC["hs"])); p2.append({"key": "highstakes", **pair(hs["probes"]["highstakes"]["stats"]["reply_mean"], "hs")})
ap = json.loads(show(SRC["apollo"])); p2.append({"key": "apollo", **pair(ap["probes"]["apollo_d2_tight"]["stats"]["reply_mean"], "apollo")})
fc = json.loads(show(SRC["control"]))
ctl = {"key": "control", "pool": ["honest_a", "honest_b"], "interval": "paired-t"}
for a in (0.01, 0.05):
    m = [r for r in fc["rows"] if r["probe"] == "item1_c" and r["readout"] == "last token L63 (replay)" and r["base"] == "all runs (unfixed render)" and r["alpha"] == a]
    assert len(m) == 1; r = m[0]; ctl["n"] = r["n"]
    b, rr = r["sim_minus_run2"]["binary"], r["run2_minus_run1"]["binary"]
    ctl[str(a)] = {"d": iv(b["mean"], b["ci95"]), "rr": iv(rr["mean"], rr["ci95"]), "real": P(r["catch"]["run2"]), "sim": P(r["catch"]["sim"])}
p2.append(ctl)
out["panel2"] = p2
json.dump(out, open("/private/tmp/claude-501/-Users-aswinsurya-probe-shift-viewer/b3b0fe45-ac17-4911-908e-445ff32d8b31/scratchpad/status/draftfig/figdata.json", "w"), indent=1)
for k, v in out["panel1"].items():
    for r, x in v.items(): print("P1", k, r, x)
for x in p2: print("P2", x["key"], x.get("pool"), x.get("n"), x["0.01"], x["0.05"])
print("budget", out["budget"]); print("command BC", out["command_BC"])
