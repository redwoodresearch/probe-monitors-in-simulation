"""Are A25's PILOT sim-honest runs (t21-t40) usable as fill-in? Per key, metadata only (no outcomes): generator identity
vs the continuation (simh_full / primary_v3), pinned reference vs merged_refs and il_refindex_old, completion per gate12,
and why the pilot is not in primary_v3. Usage: python3 -I make_pilot_check.py <inputs_pilot dir> <inputs dir> <partition json> <out json>"""
import hashlib, json, os, sys

IP, I, PART, OUT = sys.argv[1:5]
L = lambda d, n: json.load(open(os.path.join(d, n)))
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
SRC = {
    "simh_counted_keys_pilot_v3.json": "experiments/2026-10-14_q3_q35_simhonest_purchase/results/simh_counted_keys_pilot_v3.json @ 344829af6",
    "gate12_pilot_2026-10-14_140858.json": "…/results/gate12_pilot_2026-10-14_140858.json @ 4af62db74",
    "sim_settings_pilot_final_2026-10-14_140858.json": "…/results/sim_settings_pilot_final_2026-10-14_140858.json @ 4af62db74",
    "sim_settings_cont_final_2026-10-14_160932_simh_cont.json": "…/results/sim_settings_cont_final_2026-10-14_160932_simh_cont.json @ d8d816c89",
    "pilot_simhonest.yaml": "…/config/pilot_simhonest.yaml @ f3a067b26", "cont_simhonest.yaml": "…/config/cont_simhonest.yaml @ 3d9c9b168",
    "RULE.md": "experiments/2026-10-14_q3_q35_simhonest_purchase/RULE.md @ 6baad6864",
    "il_refindex_old_t21_t40.json": "xbox1 /home/aswin/phase2/il_refindex_old/launch.json references[t21..t40] (sha256, try), jq read 2026-10-15 ~18:2x PT",
}
G4_13 = ["t22", "t24", "t25", "t26", "t27", "t29", "t34", "t35", "t36", "t37", "t38", "t39", "t40"]
pk = L(IP, "simh_counted_keys_pilot_v3.json")
g12 = L(IP, "gate12_pilot_2026-10-14_140858.json")
sp, sc = L(IP, "sim_settings_pilot_final_2026-10-14_140858.json"), L(IP, "sim_settings_cont_final_2026-10-14_160932_simh_cont.json")
refs = L(I, "merged_refs_launch_2026-10-14.json")["references"]
ilr = L(IP, "il_refindex_old_t21_t40.json")
pv3 = L(I, "primary_v3.json")
part = {r["task"]: r["partition"] for r in json.load(open(PART))["rows"]}
pilot_keys = [k for k in open(os.path.join(I, "pilot_keys.txt")).read().strip().split(",") if k]
assert len(pilot_keys) == 20 and g12["pass"] and g12["usable"] == 20 and g12["launcher_finished"]

def ident(s):
    return {"given_sha256": sorted({r["given_sha256"] for r in s["new_runs"]}),
            "tree": sorted({json.dumps(r["given"]["tree"], sort_keys=True) for r in s["new_runs"]}),
            "flags": sorted({json.dumps(r["given"]["flags"]) for r in s["new_runs"]}),
            "recorded_set_sha256": hashlib.sha256(json.dumps(sorted({json.dumps(r["recorded"], sort_keys=True) for r in s["new_runs"]})).encode()).hexdigest(),
            "mismatches": sum(len(r["mismatches"]) for r in s["new_runs"]), "states": sorted({r["state"] for r in s["new_runs"]})}
ip, ic = ident(sp), ident(sc)
same_gen = {k: ip[k] == ic[k] for k in ("given_sha256", "tree", "flags", "recorded_set_sha256")}
assert ip["mismatches"] == ic["mismatches"] == 0
yp = open(os.path.join(IP, "pilot_simhonest.yaml")).read().splitlines()
yc = open(os.path.join(IP, "cont_simhonest.yaml")).read().splitlines()
yaml_diff_lines = [i + 1 for i, (a, b) in enumerate(zip(yp, yc)) if a != b] + ([] if len(yp) == len(yc) else ["length differs"])
refs_from = all("--references-from ~/phase2/runs/2026-10-14_simh_refs_merged" in y[0] for y in (yp, yc))

g12_runs = {r["task"]: r for r in g12["runs"]}
pk_keys = {k["task"]: k for k in pk["keys"]}
rows = []
for t in sorted(pilot_keys, key=lambda k: int(k.split("_")[0][1:])):
    r, k = g12_runs[t], pk_keys[t]
    m, o = refs.get(t, {}), ilr.get(t, {})
    rows.append({"task": t, "partition_46": part.get(t), "in_g4_13": t.split("_")[0] in G4_13,
                 "pilot_run": {"run_key": k["run_key"], "launch": k["launch_id"], "world": k["world"], "attempt_kind": k["attempt_kind"],
                               "eval_sha256": k["eval_sha256"], "volume_path": k["volume_path"]},
                 "completion_gate12": {"state": r["state"], "attempts": r["attempts"], "usable": r["usable"], "policy_called": r["policy_called"],
                                       "unpriced": r["unpriced"], "sample_errors": r["sample_errors"], "sample_limits": len(r["sample_limits"])},
                 "pin": {"merged_refs_sha256": m.get("sha256"), "merged_refs_source_launch": (m.get("source") or "").split("/")[5] if m.get("source") else None,
                         "il_refindex_old_sha256": o.get("sha256"), "merged_refs_vs_il_refindex_old_equal": m.get("sha256") is not None and m.get("sha256") == o.get("sha256") and m.get("try") == o.get("try"),
                         "pilot_launch_pin": "documented/inferred, not read: see pin_note"}})
assert {t.split("_")[0] for t in pilot_keys if part.get(t) == "b"} == set(G4_13)
out = {
    "what": "per key, can A25's PILOT sim-honest runs (t21-t40) serve as fill-in: generator identity, pin, completion, and why not in primary_v3",
    "sources": SRC, "inputs_sha256": [{"file": n, "sha256": sha(os.path.join(IP, n))} for n in sorted(SRC)]
    + [{"file": n, "sha256": sha(os.path.join(I, n))} for n in ("merged_refs_launch_2026-10-14.json", "primary_v3.json", "pilot_keys.txt")]
    + [{"file": os.path.basename(PART), "sha256": sha(PART), "path": "experiments/2026-10-15_t196_46_partition/results/t196_46_partition_2026-10-15.json @ cadb3c1f6"}],
    "generator_identity_pilot_vs_continuation": {"pilot": ip, "continuation": ic, "equal": same_gen,
        "yaml_lines_that_differ": yaml_diff_lines,
        "yaml_note": "the two launch configs differ only in comment lines, the task file (pilot: scaleup176; cont: all196, every cell equal per check_launch_tasks.py, RULE.md table lines 72-76), server count and spend caps; both start with --references-from the same merged refs",
        "both_yaml_headers_name_merged_refs": refs_from},
    "pin_note": "DOCUMENTED/INFERRED; the pilot's actual launch pins are UNVERIFIED here. Both launches copy references from ~/phase2/runs/2026-10-14_simh_refs_merged (run_pilot.py sha256-checks each copy, RULE.md amendment 7), whose entries are merged_refs_launch_2026-10-14.json (file sha256 26487d40..., as RULE.md records); the pilot's own launch.json is on a stopped box with no off-box copy found, so its per-key copy is not read directly",
    "why_not_in_primary_v3": {"answer": "design, not a defect: the pilot is its own world, simh_pilot (09's world names, RULE.md amendment 15, lines 559-563 @ 6baad6864), with its own counted-key manifest (simh_counted_keys_pilot_v3.json, 20 keys); primary_v3 is the continuation's manifest (world simh_full, derived_from simh_counted_keys_cont_v2.json) and so holds only continuation keys",
                              "primary_v3_derived_from": pv3["derived_from"]["file"], "primary_v3_launch": pv3["launch"],
                              "pilot_world": sorted({k["world"] for k in pk["keys"]})},
    "gate12_pilot": {k: g12[k] for k in ("pass", "gate1_completion", "gate1_pass", "gate2_pass", "usable", "keys_ok", "launcher_finished")},
    "rows": rows,
    "summary": {"n": len(rows), "usable": sum(r["completion_gate12"]["usable"] for r in rows),
                "merged_refs_pins_equal_il_refindex_old": sum(r["pin"]["merged_refs_vs_il_refindex_old_equal"] for r in rows),
                "in_g4_13": [r["task"] for r in rows if r["in_g4_13"]],
                "in_fill_in_a": [r["task"] for r in rows if r["partition_46"] == "a"]},
}
with open(OUT, "w") as f:
    json.dump(out, f, indent=1)
    f.write("\n")
print(json.dumps({"same_gen": same_gen, "yaml_diff_lines": yaml_diff_lines, "refs_from": refs_from, **out["summary"]}))
