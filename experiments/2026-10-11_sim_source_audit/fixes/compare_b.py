"""Fix (b) results next to their originals, numbers only (keys from the result JSONs; no transcripts). Originals are
read from git at the commits PLAN_b cites; swapped results from the worktree. Run from the repo root:

  python3 -I experiments/2026-10-11_sim_source_audit/fixes/compare_b.py [stripped D3 result name]

The swapped result files over 300 lines are on S3 (fixes/results_manifest_v1.json); restore them to results/ (or
check out 4fb13f56) before rerunning.
"""
import json, subprocess, sys
R = "experiments/2026-10-08_q3_onpolicy_figures/results"
git = lambda rev, name: json.loads(subprocess.run(["git", "show", f"{rev}:{R}/{name}"], capture_output=True, text=True, check=True).stdout)
load = lambda f: json.load(open(f"{R}/{f}")) if not isinstance(f, tuple) else git(*f)
pct = lambda x: None if x is None else round(100 * x, 1)
iv = lambda d: None if d is None else [pct(v) for v in d["ci95"]] + (["INVALID"] if d.get("invalid") else [])

def claimstyle(x):
    o = x["observed"]
    return {"gap": pct(o["gap"]), "gap_ci": iv(x["gap_interval"]), "verdicts": x["verdicts"],
            "fa_honest_b": pct(o["false_alarm_honest_b"]), "fa_ci": iv(x.get("false_alarm_interval")),
            "control": pct(o.get("control")), "cohort": x.get("cohort_tasks")}

print("== claim C (part 1)")
for tag, f in (("orig 225232@ab6facfe", ("ab6facfe", "claim_c_POSTHOC_2026-10-10_225232.json")), ("swap 211605", "claim_c_POSTHOC_2026-10-11_211605.json")):
    d = load(f)
    for ro in ("command_end", "reply_end"):
        print(tag, ro, claimstyle(d["selections"][ro]))
print("== D2 side5")
for tag, f in (("orig 103358@01a92967", ("01a92967", "d2_POSTHOC_2026-10-11_103358.json")), ("swap 225636", "d2_POSTHOC_2026-10-11_225636.json")):
    d = load(f)["results"]
    for ro in ("command_end_content_only", "reply_end"):
        for sc in ("seen_environments", "all_environments"):
            print(tag, ro, sc, claimstyle(d["side5"][ro][sc]))
    for ro, g in d["generalization"].items():
        print(tag, "generalization", ro, "INVALID for the swap (PLAN_b Addendum 1)" if tag.startswith("swap") else "",
              {k: (pct(v["difference"]), iv(v["interval"]), v["reading"]) for k, v in g.items()})
    e = d["env10"]["command_end_content_only"]
    print(tag, "env10 command_end_content_only per env gap:", {k: (pct(v.get("gap")), iv(v.get("interval")) if v.get("interval") else None, v["tasks"]) for k, v in e["per_environment"].items()})
print("== k26 matched length")
for tag, f in (("orig 102216@588e2ca6", ("588e2ca6", "claim_c_k26_POSTHOC_2026-10-11_102216.json")), ("swap 225643", "claim_c_k26_POSTHOC_2026-10-11_225643.json")):
    d = load(f)
    print(tag, "ref", d["reference_tag"])
    for ro, x in d["results"].items():
        m = x["matched_length"]
        print(tag, ro, "gate", x["gate"]["passed"], {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if not isinstance(v, (dict, list))})
        print("   ", {k: v for k, v in m.items() if isinstance(v, dict) and "ci95" in v})
D3 = [("D3 claim C detector", "orig 112843@ef966b86", ("ef966b86", "d3_position_POSTHOC_2026-10-11_112843.json"), "swap 225756", "d3_position_POSTHOC_2026-10-11_225756.json")]
if len(sys.argv) > 1:
    D3.append(("D3 stripped detector", "orig 120159@874cfdd8", ("874cfdd8", "d3_position_claimcnad3_POSTHOC_2026-10-11_120159.json"), "swap", sys.argv[1]))
for title, t0, f0, t1, f1 in D3:
  print(f"== {title}, fixed cohort")
  for tag, f in ((t0, f0), (t1, f1)):
      d = load(f)
      fam = d["family"]
      print(tag, "family:", {k: fam[k] for k in ("gap_exists", "parity_holds_at_all_positions", "all_plants_detected")})
      for ro in ("command_end", "reply_end"):
          r = d["readouts"][ro]
          print(tag, ro, "cohort", r["fixed_cohort_tasks"], "unconverged", round(r["unconverged_share"], 4))
          for j in ("1", "2", "3"):
              x = r["fixed"][j]
              print("   j", j, "G", pct(x["observed"]["gap"]), iv(x["gap_interval"]), x["verdicts"], "fa_b", pct(x["observed"]["false_alarm_honest_b"]),
                  "control", pct(x["observed"]["control"]), iv(x["control_interval"]), "passes", x["control_passes"])
          p = r["pre_c_star"]
          print("   pre-c*", {k: p[k] for k in ("eligible", "caught") if k in p})
