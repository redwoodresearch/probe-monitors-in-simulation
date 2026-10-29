"""S_paper493 sim-attack approach-provenance map, per cell key (493). Keys and shas only; no approach text is read.
Per key: (1) the registered q2 approach sha256 (facts v2), (2) the sim-attack launch's on-box built approach sha256
(tasks_sp493_sim_L1/L2.json, built on the box), (3) the real arm's on-box built approach sha256 (runlist_box{A,B,C}_launch_v2.json,
r1 and its _r2 copy), (4) the real-arm roots that will hold r1 and r2. A cell is ADMITTED only if (1) = (2) = (3), all
present; otherwise it is a status row with the reason. L3 (g3) has no committed task file yet, so its cells are missing.
Usage: python3 -I make_sp493_approach_map.py <inputs dir> <out json>"""
import hashlib, json, os, re, sys

I, OUT = sys.argv[1:3]
L = lambda n: json.load(open(os.path.join(I, n)))
sha = lambda n: hashlib.sha256(open(os.path.join(I, n), "rb").read()).hexdigest()
EMPTY = hashlib.sha256(b"").hexdigest()
SHA = re.compile(r"[0-9a-f]{64}")
ok = lambda v: isinstance(v, str) and SHA.fullmatch(v) is not None
SRC = {
    "facts_v2.json": "aswin/q3-sp493 experiments/2026-10-15_q3_sp493/results/sp493_facts_2026-10-15_165730.json @ d800edc6a (rows[].approach_sha256 = the registered q2 sha)",
    "tasks_sp493_sim_L1.json": "aswin/q3-interleaved-run3-sim2 experiments/2026-10-15_q3_interleaved_run3_sim2/config/sp493sim/tasks_sp493_sim_L1.json @ 590a2bdbc (built by on-box prep; launch 2026-10-15_sp493_sim_L1, box3)",
    "tasks_sp493_sim_L2.json": "… tasks_sp493_sim_L2.json @ 590a2bdbc (built by on-box prep; launch 2026-10-15_sp493_sim_L2, box4)",
    "runlist_boxA_launch_v2.json": "aswin/q3-sp493 experiments/2026-10-15_q3_sp493/config/runlist_boxA_launch_v2.json @ 548c75d30 (fetched from box A after prep)",
    "runlist_boxB_launch_v2.json": "… runlist_boxB_launch_v2.json @ 548c75d30", "runlist_boxC_launch_v2.json": "… runlist_boxC_launch_v2.json @ 548c75d30",
    "approach_check_boxA.json": "… results/approach_check_boxA.json @ df8383daf", "approach_check_boxB.json": "… results/approach_check_boxB.json @ 5c0417f88",
    "approach_check_boxC.json": "… results/approach_check_boxC.json @ 9065ecbe5",
    "sp493_group_map_v2.json": "… config/sp493_group_map_v2.json @ c8ff270da",
    "chain_state_snapshot.json": "chain fork run state (chain_run_state/state.json, not in git): batches sp493-{a,b,c} stage_tries/stage_abandoned",
}
frows = L("facts_v2.json")["rows"]
facts = {r["key"]: r for r in frows}
assert len(frows) == len(facts) == 493, "duplicate or missing facts keys"
sim = {}
for lname in ("L1", "L2"):
    t = L(f"tasks_sp493_sim_{lname}.json")
    assert t["dropped_at_preflight"] == []
    for c in t["cells"]:
        assert c["key"] not in sim
        sim[c["key"]] = {"launch": t["launch"], "box": t["box"], "approach_sha256": c["approach_sha256"], "q2_in_file": c["q2_approach_sha256"]}
real = {}
st = L("chain_state_snapshot.json")["batches"]
for x in "ABC":
    r = L(f"runlist_box{x}_launch_v2.json")
    chk = L(f"approach_check_box{x}.json")
    assert chk["bad_keys"] == [] and chk["approach_files_hash_equal_name_and_q2"] == chk["r1_cells"]
    cells = {c["key"]: c for c in r["cells"]}
    assert len(cells) == len(r["cells"]), f"duplicate keys in box {x}"
    assert all(k[:-3] in cells for k in cells if k.endswith("_r2")), f"orphan _r2 in box {x}"
    b = st[f"sp493-{x.lower()}"]
    tries = b["stage_tries"]
    for k, c in cells.items():
        if k.endswith("_r2"):
            continue
        assert k not in real, f"key {k} in more than one box"
        r2 = cells.get(k + "_r2")
        nxt = {t: tries.get(t, 0) + 1 for t in (f"sp493_{x.lower()}_main", f"sp493_{x.lower()}_run2")}
        real[k] = {"box": x, "launch": b["run"], "approach_sha256": c["approach_sha256"], "q2_in_runlist": c["q2_approach_sha256"],
                   "r2_present": r2 is not None, "r2_approach_sha256_equal": r2 is not None and r2["approach_sha256"] == c["approach_sha256"],
                   "root_r1": f"sp493_{x.lower()}_main_v1_try{nxt[f'sp493_{x.lower()}_main']}__attack",
                   "root_r2": f"sp493_{x.lower()}_run2_v1_try{nxt[f'sp493_{x.lower()}_run2']}__attack"}
assert set(real) == set(facts), (len(real), len(set(real) ^ set(facts)))
assert not (set(sim) - set(facts))

rows, counts = [], {}
for k in sorted(facts, key=lambda s: int(s.split("_")[0][1:])):
    q2, s, rr = facts[k]["approach_sha256"], sim.get(k), real.get(k)
    launch = s["launch"] if s else "2026-10-15_sp493_sim_L3 (g3; no committed task file yet)"
    reasons = []
    for name, v in (("q2", q2), ("sim", s and s["approach_sha256"]), ("real", rr and rr["approach_sha256"])):
        if v is not None and v is not False and not ok(v):
            reasons.append(f"invalid sha: {name} is not a 64-hex sha256")
    if not ok(q2):
        reasons.append("missing/invalid: registered q2 sha")
    if not s:
        reasons.append("missing: no sim-attack launch record (L3 not built/committed)")
    elif s["approach_sha256"] != q2:
        reasons.append("mismatch: sim on-box sha != registered q2 sha")
    if not rr:
        reasons.append("missing: no real-arm record")
    else:
        if rr["approach_sha256"] != q2:
            reasons.append("mismatch: real on-box sha != registered q2 sha")
        if not rr["r2_present"] or not rr["r2_approach_sha256_equal"]:
            reasons.append("mismatch/missing: real r2 entry")
    if rr and rr["r2_present"] and not ok(rr.get("approach_sha256")):
        reasons.append("invalid sha: real r1")
    status = "equal" if not reasons else ("missing" if all(x.startswith("missing") for x in reasons) else "mismatch")
    counts.setdefault(launch.split(" ")[0], {"equal": 0, "mismatch": 0, "missing": 0})[status] += 1
    rows.append({"chain_key": f"sp493_sim|{k}", "key": k, "sim_launch": launch, "q2_approach_sha256": q2, "sim_onbox_approach_sha256": s["approach_sha256"] if s else None,
                 "real_box": rr["box"] if rr else None, "real_onbox_approach_sha256": rr["approach_sha256"] if rr else None,
                 "real_r1_root": rr["root_r1"] if rr else None, "real_r2_root": rr["root_r2"] if rr else None,
                 "approach_is_empty_string_sha": q2 == EMPTY, "status": status, "admitted": status == "equal", "reasons": reasons})
out = {"what": "S_paper493 sim-attack approach provenance, per cell key: registered q2 sha = sim on-box sha = real on-box sha; keys and shas only",
       "sources": SRC, "inputs_sha256": [{"file": n, "sha256": sha(n)} for n in sorted(SRC)],
       "roots_note": "real-arm roots are PLANNED names (sp493_{a,b,c}_main_v1_try2__attack / sp493_{a,b,c}_run2_v1_try2__attack, the chain fork's naming): chain state has stage try 1 abandoned for sp493_{a,b,c}_main/_run2 and the real launches still final-only/waiting; no staged root exists yet",
       "per_box_approach_check": {x: L(f"approach_check_box{x}.json") for x in "ABC"},
       "counts_by_sim_launch": counts,
       "counts_total": {s_: sum(c[s_] for c in counts.values()) for s_ in ("equal", "mismatch", "missing")},
       "n_registered_empty_string_approach": sum(r["approach_is_empty_string_sha"] for r in rows),
       "n_admitted_with_empty_string_approach_in_all_three": sum(r["approach_is_empty_string_sha"] and r["admitted"] for r in rows),
       "empty_note": "the empty-string sha256 (e3b0c442...) is an intended empty approach: prep kept empty approaches by design (prep_summary_box{A,B,C}.json empty_approach_kept 92/90/84 = 266)",
       "keying": "map is keyed by the chain's row key sp493_sim|<cell> (cell = the launcher task name), one entry per cell (chain fork ad74c4801cc69e41c)",
       "map": {r["chain_key"]: {k2: v for k2, v in r.items() if k2 != "chain_key"} for r in rows}}
assert len(out["map"]) == 493
with open(OUT, "w") as f:
    json.dump(out, f, indent=1)
    f.write("\n")
print(json.dumps({"counts": counts, "total": out["counts_total"], "empty_admitted": out["n_admitted_with_empty_string_approach_in_all_three"], "n": len(rows)}))
