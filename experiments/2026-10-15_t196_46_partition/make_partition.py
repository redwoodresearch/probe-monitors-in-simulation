"""By-key partition of the 46 T196 tasks that S_old's H_B has but A25's sim-honest pool (primary_v3, 149 keys) lacks:
(a) filled by 09's A25 fill-in (the 15 keys relayed by d9), (b) covered by S_paper493's g4 sim-honest generation (d9's
B-job file), (c) excluded with reason and evidence. Plus whether each key was in A25's purchase list and its launch key
lists, and why it isn't counted in primary_v3. Keys only.
Usage: python3 -I make_partition.py <inputs dir> <out json>"""
import hashlib, json, os, sys

I, OUT = sys.argv[1:3]
L = lambda n: json.load(open(os.path.join(I, n)))
keys_txt = lambda n: [k for k in open(os.path.join(I, n)).read().strip().split(",") if k]
sha = lambda n: hashlib.sha256(open(os.path.join(I, n), "rb").read()).hexdigest()

SRC = {
    "removed_HB.json": "aswin/runlength-v2-2e @ d5fbb9573 experiments/2026-10-15_runlength_check/results/calibmatch_v2_table_2026-10-15_172012.json removed['Sol (a) C']['S_old']['H_B']",
    "primary_v3.json": "experiments/2026-10-14_q3_q35_simhonest_purchase/results/simh_counted_keys_primary_v3.json @ 572f67dbb",
    "tasks_simhonest_170.json": "experiments/2026-10-14_q3_q35_simhonest_purchase/config/tasks_simhonest_170.json @ f2702a90b (aswin/q3-q35-simhonest-purchase)",
    "merged_refs_launch_2026-10-14.json": "experiments/2026-10-14_q3_q35_simhonest_purchase/results/merged_refs_launch_2026-10-14.json @ 015bea313",
    "pilot_keys.txt": "experiments/2026-10-14_q3_q35_simhonest_purchase/config/pilot_keys.txt (aswin/q3-q35-simhonest-purchase @ c7fcab333)",
    "cont_keys.txt": "experiments/2026-10-14_q3_q35_simhonest_purchase/config/cont_keys.txt (@ c7fcab333)",
    "a_keys.txt": "experiments/2026-10-14_q3_q35_simhonest_purchase/config/a_keys.txt (@ c7fcab333); equals the 20 keys of config/tasks_simattack2_20.json, i.e. the sim_attack2 list, not a sim-honest purchase",
    "t43_keys.txt": "experiments/2026-10-14_q3_q35_simhonest_purchase/config/t43_keys.txt (@ c7fcab333)",
    "t196_46_vs_b28_2026-10-15.json": "d9 scratchpad/d9_audit/t196_46_vs_b28_2026-10-15.json (B job: 5237ca017 sp493_honest_coverage_2026-10-15_170958.json)",
    "il_old_t88_t169.json": "xbox1 /home/aswin/phase2/runs/2026-10-14_il_old/launch.json, jq read of runs[] and reference_state for t88/t169/t139/t151",
}
FILL_IN = ["t04", "t07", "t08", "t11", "t12", "t13", "t17", "t21", "t23", "t28", "t30", "t31", "t32", "t33", "t48"]
EXCL = {"t88": "September preflight failed in il_old (reference usable; preflight state failed; sim_attack not run, 0 attempts); also excluded from A25's 170 purchase list by its rule (attack construction fails)",
        "t169": "September preflight failed in il_old (reference usable; preflight state failed; sim_attack not run, 0 attempts); also excluded from A25's 170 purchase list by its rule (attack construction fails)",
        "t139": "no pinned reference (il_old reference_state: none in the prior run)",
        "t151": "no pinned reference (il_old reference_state: none in the prior run)"}

removed = [r["task"] for r in L("removed_HB.json")["removed_tasks"]]
assert len(removed) == len(set(removed)) == 46
short = {t.split("_", 1)[0]: t for t in removed}
assert len(short) == 46
pool = set(L("primary_v3.json")["counted_tasks"])
assert len(pool) == 149 and not pool & set(removed)
purchase = {t["key"] for t in L("tasks_simhonest_170.json")["tasks"]}
assert len(purchase) == 170
refs = L("merged_refs_launch_2026-10-14.json")["references"]
lists = {n[:-4]: set(keys_txt(n)) for n in ("pilot_keys.txt", "cont_keys.txt", "a_keys.txt", "t43_keys.txt")}
b = L("t196_46_vs_b28_2026-10-15.json")
covered = set(b["covered_by_b"])
assert covered | set(b["not_covered"]) == set(removed) and not covered & set(b["not_covered"])
ilo = L("il_old_t88_t169.json")
p_excl = {r["run_key"].split("/")[1]: r["classification"] for r in L("primary_v3.json")["exclusions"]}

fill = {short[s] for s in FILL_IN}
excl = {short[s]: why for s, why in EXCL.items()}
assert not fill & covered and not set(excl) & (fill | covered)
rows = []
for t in sorted(removed, key=lambda k: int(k.split("_")[0][1:])):
    part = "a" if t in fill else "b" if t in covered else "c" if t in excl else None
    in_lists = sorted(n for n, s in lists.items() if t in s)
    simh_lists = [n for n in in_lists if n != "a_keys"]  # a_keys = the sim_attack2 tasks
    if t in p_excl:
        why_not = f"purchased (cont launch) but excluded from primary_v3: {p_excl[t]}"
    elif simh_lists == ["pilot_keys"]:
        why_not = "purchased in the PILOT launch (pilot_keys), not the cont launch whose counted keys form primary_v3"
    elif simh_lists:
        why_not = f"in sim-honest launch key lists {simh_lists} but not in primary_v3's counted tasks; reason not resolved in v1"
    elif t in purchase:
        why_not = "in the 170 purchase list but in no launch key list; reason not resolved in v1"
    else:
        why_not = ("only in a_keys (the sim_attack2 tasks); " if in_lists == ["a_keys"] else "") + "not purchased: not in the 170 purchase list (its rule needs real honest A and B, real attack run 1 and sim_attack@september present, no sim_honest; t88/t169 excluded by name); per-criterion reason not resolved in v1"
    rows.append({"task": t, "partition": part,
                 "partition_reason": {"a": "filled by 09's A25 fill-in (15 keys relayed by d9)", "b": "covered by S_paper493's g4 sim-honest generation (d9 B-job file)"}.get(part) or excl.get(t),
                 "covered_by_b_job": t in covered, "in_a25_purchase_list_170": t in purchase, "in_launch_key_lists": in_lists,
                 "pinned_reference_in_merged_refs": t in refs, "in_primary_v3_counted": t in pool, "why_not_counted_in_primary_v3": why_not})
assert all(r["partition"] for r in rows), [r["task"] for r in rows if not r["partition"]]
counts = {p: sum(r["partition"] == p for r in rows) for p in "abc"}
out = {"what": "by-key partition of the 46 T196 tasks in S_old H_B but not in A25's primary_v3 sim-honest pool",
       "sources": SRC, "inputs_sha256": [{"file": n, "sha256": sha(n)} for n in sorted(SRC)],
       "fill_in_source": "09's registered A25 fill-in, 15 keys as relayed by d9 (2026-10-15 ~17:50 PT); no committed file read",
       "counts": counts, "il_old_evidence": ilo, "rows": rows,
       "cross_checks": {"b_job_not_covered_equals_a_plus_c": set(b["not_covered"]) == fill | set(excl),
                        "c_keys_in_purchase_list": {t: t in purchase for t in excl}, "c_keys_pinned_reference_in_merged_refs": {t: t in refs for t in excl}}}
with open(OUT, "w") as f:
    json.dump(out, f, indent=1)
    f.write("\n")
print(json.dumps({"counts": counts, **out["cross_checks"]}))
for r in rows:
    print(r["task"], r["partition"], int(r["in_a25_purchase_list_170"]), ",".join(r["in_launch_key_lists"]) or "-", int(r["pinned_reference_in_merged_refs"]), r["why_not_counted_in_primary_v3"][:60])
