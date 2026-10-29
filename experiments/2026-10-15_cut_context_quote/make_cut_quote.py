"""Quote for 2e's CUT fork ("retrain without the context limit?"): epoch counts by root, tokens before the 32,768 cut where
recorded, epochs over 32,768, and the longest-epoch H200 extraction timing. Counts and keys only; no scores are read.
Inputs (saved by git show into <in dir>, each with its commit in the output): ss/ staged_stats_simh*, att/ attestations_simh*,
xc/ extract_chain_*.json (aswin/q3-overnight-chain tip), a25/ A25 staging records (aswin/q3-q35-simhonest-purchase),
sp493/ S_paper493 launch files (aswin/q3-sp493), sol/ the Sol avg POSTHOC RULE lines and repfull staging (aswin/p1-correctedb).
Usage: python3 -I make_cut_quote.py <in dir> <out json>"""
import glob, hashlib, json, os, sys

I, OUT = sys.argv[1:3]
L = lambda p: json.load(open(p))
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
CHAIN = open(os.path.join(I, "chain_sha.txt")).read().strip()
commits = lambda d: dict(l.split() for l in open(os.path.join(I, d, "commits.txt")) if l.strip())

# ---- 1. S_paper493 planned epochs per arm (launch files) ----
sp = {}
for x in "ABC":
    r = L(os.path.join(I, "sp493", f"runlist_box{x}_launch_v2.json"))
    r1 = [c for c in r["cells"] if not c["key"].endswith("_r2")]
    r2 = [c for c in r["cells"] if c["key"].endswith("_r2")]
    arms = {"attack_run1": 0, "attack_run2": len(r2), "honest_a": 0, "honest_b": 0}
    for c in r1:
        for a in c.get("real_arms") or ["honest_a", "attack", "honest_b"]:  # default per honest_keys_sp493.json's rule
            arms["attack_run1" if a == "attack" else a] += 1
    assert all((c.get("real_arms") or ["attack"]) == ["attack"] for c in r2)
    sp[x] = {"r1_cells": len(r1), "r2_cells": len(r2), **arms, "dropped_at_preflight": len(r["dropped_at_preflight"])}
tot = {k: sum(v[k] for v in sp.values()) for k in ("attack_run1", "attack_run2", "honest_a", "honest_b")}
gm = L(os.path.join(I, "sp493", "sp493_group_map_v2.json"))["groups"]
c_sp = commits("sp493")
s1 = {
    "staged": "RELAYED, not verified here: nothing staged yet; the simulated arms are held (d9). The aswin/q3-sp493 tip commit 5753e554c is titled 'S_paper493 v2 LAUNCHED ... C 18:12:49'. tokens_before not available",
    "real_arms_planned_epochs": tot, "per_box": sp,
    "real_arms_source": {f: f"experiments/2026-10-15_q3_sp493/config/{f} @ {c_sp[f]}" for f in ("runlist_boxA_launch_v2.json", "runlist_boxB_launch_v2.json", "runlist_boxC_launch_v2.json", "honest_keys_sp493.json")},
    "readme": f"experiments/2026-10-15_q3_sp493/README.md @ {c_sp['README.md']} lines 15-25: '1,060 real runs: 493 + 493 attacks, 37 + 37 honest'; simulated arms are separate follow-ups",
    "t619": "honest_b's 37 include t619 (box B honest key); 'the t619 extra, not pooled' is d9's note, not checked here",
    "sim_attack_HELD": {g: gm[g]["cells"] for g in ("g1", "g2", "g3")} | {"total": sum(gm[g]["cells"] for g in ("g1", "g2", "g3"))},
    "sim_honest_HELD": {"group_map_v2": {g: gm[g]["mains"] for g in ("g4", "g5")},
                        "current_per_d9": {"g4": 14, "g5": 36, "note": "g4 28 - 13 covered by the accepted pilot runs - t98 (moved to g5); g5 35 + t98; relayed by d9, not in a committed file read here"}},
    "a25_fill_in": {"n": 8, "keys": ["t04", "t07", "t08", "t11", "t12", "t13", "t17", "t48"], "note": "15 minus the 7 covered by the accepted pilot runs (pilot_t21_t40_fillin_check @ 477b3d030); relayed plan"},
    "group_map_source": f"experiments/2026-10-15_q3_sp493/config/sp493_group_map_v2.json @ {c_sp['sp493_group_map_v2.json']}",
}

# ---- 2. every sim-honest root routed so far ----
roots = {}
for p in sorted(glob.glob(os.path.join(I, "ss", "staged_stats_simh*.json"))):
    d = L(p)
    for s in d["sets"]:
        key = s["target"]
        e = roots.setdefault(key, {"arms": {}, "sources": []})
        e["arms"][s["arm"]] = {"epochs": s["epochs"], "max_staged_tokens": s["max_staged_tokens"], "n_epochs_over_32768": s["n_epochs_over_32768"]}
        e["sources"].append(f"experiments/2026-10-14_q3_overnight_chain/results/{os.path.basename(p)} @ {CHAIN}")
att_only = {}
for p in sorted(glob.glob(os.path.join(I, "att", "attestations_simh*.json"))):
    d = L(p)
    for lst in (d.get("attestations") or {}).values():
        for a in lst:
            if a["check"] != "longest_epoch_h200_gate":
                continue
            t, arm, det = a["covers"]["target"], a["covers"]["arm"], a["detail"]
            if t in roots and arm in roots[t]["arms"]:
                roots[t]["arms"][arm]["attested_rows"] = det.get("rows")
                roots[t]["arms"][arm]["attested_n_epochs_over_32768"] = det.get("n_epochs_over_32768")
            else:
                att_only.setdefault(t, {})[arm] = {"rows": det.get("rows"), "max_staged_tokens": det.get("max_staged_tokens"),
                                                    "n_epochs_over_32768": det.get("n_epochs_over_32768"),
                                                    "source": f"results/hygiene/{os.path.basename(p)} @ {CHAIN}"}
batches_no_attestation = sorted(L(p)["batch"] for p in glob.glob(os.path.join(I, "att", "*.json")) if not L(p).get("attestations"))
a25c = commits("a25")
a25 = {}
for f, c in a25c.items():
    d = L(os.path.join(I, "a25", f))["simh"]
    cov = d["assembled"]["coverage"]["sim_honest@september"]
    a25[d["out"]] = {"epochs": d["assembled"]["epochs"]["sim_honest@september"], "tokens_before_total": cov["totals"]["tokens_before"],
                     "tokens_after_total": cov["totals"]["tokens_after"], "truncated_epochs": cov["totals"]["truncated"],
                     "source": f"experiments/2026-10-14_q3_q35_simhonest_purchase/results/{f} @ {c}"}
for r in roots.values():
    r["max_staged_tokens_at_cut"] = max(a["max_staged_tokens"] for a in r["arms"].values()) == 32768
s2 = {"read_this_first": "n_epochs_over_32768 counts STAGED epochs longer than 32,768 tokens, i.e. after the cut, so it is 0 by construction and does NOT say how many runs the cut truncated. The chain records no per-epoch pre-cut length; max_staged_tokens_at_cut marks roots with at least one staged epoch AT exactly 32,768 (consistent with a cut, but post-cut lengths alone do not establish truncation). Truncation counts exist only for the A25 roots ('truncated_epochs').",
      "tokens_before": "NOT recorded in the chain's staged_stats or attestations (they record epochs, max_staged_tokens and n_epochs_over_32768); recorded only in the A25 staging records below",
      "chain_roots": roots, "chain_roots_from_attestation_only": att_only, "chain_batches_with_no_attestation_yet": batches_no_attestation,
      "a25_roots": a25, "a25_note": "primary_v3 (149) = full_b1 145 + full_b2 4; t43rerun is the sensitivity rerun; the pilot (20) is world simh_pilot; 'truncated' = epochs cut at 32,768",
      "totals": {"note": "sum over roots, not deduplicated by key (snapshot roots s1..sN, the final root and rlr roots look like disjoint increments; not verified here)",
                 "roots_with_max_at_cut": sum(r["max_staged_tokens_at_cut"] for r in roots.values()), "n_roots": len(roots),
                 "chain_epochs": sum(a["epochs"] for r in roots.values() for a in r["arms"].values()),
                 "chain_over_32768": sum(a["n_epochs_over_32768"] for r in roots.values() for a in r["arms"].values())}}

# ---- 3. Sol (a) / c_asis training data ----
c_sol = commits("sol")
rf = L(os.path.join(I, "sol", "stage_repfull_2026-10-14_021642.json"))["repfull"]
rcov = rf["assembled"]["coverage"]["sim_attack@september"]
s3 = {"rule_quote": open(os.path.join(I, "sol", "rule_lines.txt")).read(),
      "rule_source": f"experiments/2026-10-14_q35_sol_avg_POSTHOC/RULE.md @ {c_sol['RULE.md']} (aswin/p1-correctedb), lines 13-16 and 548-550",
      "sol_a_training": "per RULE: Sol (a) trains on #36's real honest (990) vs first real attack (939) from the v4 five-arm staged inputs (stage_sol_avg.py, byte-identical rows); (a') drops 41 changed runs (5 honest, 36 attack)",
      "tokens_before_for_sol_training": "NOT FOUND in committed records within the time box",
      "stage_q35_repaired_py_output": {"root": rf["out"], "epochs": rf["assembled"]["epochs"], "tokens_before_total": rcov["totals"]["tokens_before"],
                                       "tokens_after_total": rcov["totals"]["tokens_after"], "truncated": rcov["totals"]["truncated"], "runs": rcov["runs"],
                                       "note": "stage_q35_repaired.py stages B's repaired simulated-attack set (Arm S, repfull), the TEST side, not Sol (a)'s training data. 'runs' (191 source runs) is what the token totals and the 132 truncations cover; only the 102 runs whose token ids changed are emitted as epochs (89 unchanged runs are not staged)",
                                       "source": f"experiments/2026-10-14_q35_sol_avg_POSTHOC/results/stage_repfull_2026-10-14_021642.json @ {c_sol['stage_repfull_2026-10-14_021642.json']}"},
      "c_asis": "named in RULE.md line 912 ('b_asis and c_asis are not scored here'); its staging root was not located in the time box"}

# ---- 4. longest-epoch H200 timing ----
shards = []  # shard-ATTEMPT records (a resumed shard can appear more than once)
for p in sorted(glob.glob(os.path.join(I, "xc", "extract_chain_*.json"))):
    d = L(p)
    for s in d.get("shards") or []:
        if s.get("gpu_name") == "NVIDIA H200" and s.get("seconds") is not None:
            shards.append({"record": os.path.basename(p), "test": d["config"].get("test"), "shard": s["shard"], "epochs": s["epochs"],
                           "seconds": s["seconds"], "max_peak_gib": s.get("max_peak_gib"), "load_s": s.get("load_s")})
assert shards
peak = max(shards, key=lambda s: (s["max_peak_gib"] or 0))
slow = max(shards, key=lambda s: s["seconds"])
s4 = {"single_longest_epoch_test_records": [s for s in shards if s["test"]],
      "note": "no --test (single longest epoch) record on the chain branch; every attested shard carries the 32,768-token longest staged epoch (repro-c: 'batch C's 32,768-token longest staged epoch + one seeded, per shard', longest_epoch_h200_gate reference), so a shard's peak GiB includes it; seconds are per shard (several epochs), not per epoch",
      "n_h200_shard_attempt_records": len(shards), "n_distinct_record_shards": len({(s["record"], s["shard"]) for s in shards}),
      "caveat": "the longest epoch is assigned to every shard, but a resumed attempt can skip already-done epochs, so not every recorded peak necessarily includes it",
      "max_peak_gib": peak, "max_shard_seconds": slow,
      "peak_gib_values": sorted({s["max_peak_gib"] for s in shards if s["max_peak_gib"] is not None}), "n_shards_without_peak": sum(s["max_peak_gib"] is None for s in shards), "token_count": 32768,
      "source": f"experiments/2026-10-14_q3_overnight_chain/results/extract_chain_*.json @ {CHAIN} ({len(glob.glob(os.path.join(I, 'xc', '*.json')))} records)"}

out = {"what": "CUT-fork quote: epochs by root, tokens before the 32,768 cut where recorded, epochs over 32,768, longest-epoch H200 timing; counts only, no scores",
       "chain_commit": CHAIN, "1_s_paper493": s1, "2_sim_honest_roots": s2, "3_sol_a_training": s3, "4_longest_epoch_h200": s4,
       "inputs_sha256": [{"file": os.path.relpath(p, I), "sha256": sha(p)} for p in sorted(glob.glob(os.path.join(I, "**", "*"), recursive=True)) if os.path.isfile(p)]}
with open(OUT, "w") as f:
    json.dump(out, f, indent=1)
    f.write("\n")
print(json.dumps({"sp493": tot, "simh_chain_roots": len(roots), "att_only": len(att_only), "no_att": batches_no_attestation, **s2["totals"],
                  "a25": {k: (v["epochs"], v["truncated_epochs"], v["tokens_before_total"]) for k, v in a25.items()},
                  "repfull": (rf["out"], rf["assembled"]["epochs"], rcov["totals"]["truncated"]), "peak": peak, "slow": slow, "n_shards": len(shards)}))
