"""Epoch-2 pool-key artifact for 09's Append 38 §4 note (@ 71dc715db, Codex fixes @ d9013b2e7): which epoch-2 real honest runs enter which
epoch-1 pool, by key. Keys, manifests and .eval sha256s only: no epoch-2 (or any) score file is opened, and no Modal call is made.

Rules (the note): an epoch-2 run of task t and arm X is admitted to a pool only if the epoch-1 run X|t is already a member of that pool
(replicate only; task membership never changes); minus the launcher-unusable status rows in d9's key files and the stager's exclusions
(recorded per launch as available or "pending"); the T196 1e-4 honest_b replicates never enter (they are not in these launches).
- H_B: epoch-2 honest_b (T196: e2b) on B members. Where a T196 row calibrates on B u C or uses C as its yardstick, C is unchanged and e2a
  never stands in for C.
- H_A (25h §3 yardstick, held-out false-alarm checks): epoch-2 honest_a on honest_a members. Where honest_a is not registered (T196 rows
  whose yardstick is C), nothing is added.
- H_sim: no epoch 2.
Epoch-1 pools come from the sources the final rebuild reads: FINAL's calibration records (S_old, S_new; the configurations BuC, B and AuB that
FINAL registers per probe x readout), INPUTS_FINAL's 25f honest-A roots (S_new honest_a), and membership710 folder_keys (S_newmains).

  python3 experiments/2026-10-15_epoch2_poolkeys/make_epoch2_poolkeys.py [--draft PATH]
An origin with any launch whose staging exclusions are still pending reads "not in this build" (no keys). --draft keeps its keys, labels the
file DRAFT and writes it to PATH (outside results/), for checking the rules before the staging records land.
Rulings applied (09 via 2e, 10-15 ~18:10): A u B pools are labelled "AuB (registered sensitivity)", never H_B; "without S4" removes S4 from
attack estimation only (RULE.md @ 51a612b29, lines 2858 and 2981), so S_newmains has one calibration pool for the with- and without-S4 rows;
S_new groups are named by their new pool's arms (FINAL's BuC records calibrate on B in S_new).
"""
import argparse

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
HYG = "experiments/2026-10-14_q3_overnight_chain/results/hygiene"
KEYLISTS = {"T196": [("b51f5c52c", f"{HYG}/ep2_keylist_t196_2026-10-15.json")],
            "+311": [("15b79df39", f"{HYG}/ep2_keylist_311_l1_2026-10-15.json"), ("b51f5c52c", f"{HYG}/ep2_keylist_311_l2_2026-10-15.json")],
            "nm360": [("5197f3759", f"{HYG}/ep2_keylist_nm_2026-10-15.json")]}
FINAL = ("3ce6ea55e", "experiments/2026-10-14_xerr_scaleup_09/results/xerr_scaleup_final_2026-10-15_013255.json",
         "7dceb8bb25752aa3619cfde64add702fd4f7d13a1309a56dbc8880a6b27af9f5")
MEMBERSHIP = ("0f9d96158", "experiments/2026-10-15_fpr_sweep_507_prep/config/membership710.json", "6516d15fab77d7f344497e72db10b60c5f709b35c3a1859a2c9e016d2fc6e4ad")
CELLS311 = ("d75f997f3", "experiments/2026-10-14_q35_sol_avg_POSTHOC/config/scaleup_tasks_plus311_2026-10-14_125135.json",
            "69f1f75313c9610f7d16b18225c1166268c0b6e084859c641d0213f293896a8b")
INPUTS_FINAL = ("32abe4120", "experiments/2026-10-15_fpr_sweep_507_prep/config/INPUTS_FINAL_2026-10-15_203213.json",
                "d8be69e672acaca11be3f76807929660c61db30a3cf50b083f8fef8c4d6dee9a")  # FROZEN 20:18 PT (pools.stage_a and 25f equal draft 191701's)
DT_MANIFESTS = ("fc2ee229d", "experiments/2026-10-14_p2_dt_scaleup/results/dt_unseen_check_2026-10-15_111101.json",
                "ac3c84fe9ddde626bf54777f44fb6f1a76cbe1ea906e7b9fe439b8429eeccd30")  # 4f's DT manifests (run_common710.DT_UNSEEN)
COVERAGE_LABEL = "H_B restricted to folders this probe's files cover (coverage pool)"
BCHAIN = "origin/aswin/q3-overnight-chain"
SPEC = {"note": "experiments/2026-10-14_priority1_spec_09/RULE.md @ 71dc715db (Append 38 §4 note)",
        "codex_fixes": "experiments/2026-10-14_priority1_spec_09/RULE.md @ d9013b2e7 (release gate; T196 B u C / C yardstick preserved; actual counts)"}


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout


def sha(b):
    return hashlib.sha256(b).hexdigest()


def pinned(commit, path, want=None):
    b = git("show", f"{commit}:{path}")
    assert want is None or sha(b) == want, (path, sha(b))
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", type=Path, default=None)
    ap.add_argument("--man710", type=Path, required=True, help="membership710's manifest copies (<folder>__<kind>.json), each checked against its sha in membership710")
    args = ap.parse_args()
    out = {"label": "epoch-2 pool keys for the 'epoch 1 + 2 calibration' rows (Append 38 §4 note); keys, launch ids and .eval sha256s only; PRELIMINARY until 09 verifies by key",
           "spec": SPEC, "inputs": {}, "epoch2_candidates": {}, "staging_exclusions": {}, "origins": {}}
    # ---- epoch-2 candidates: d9's key lists, minus their status rows; +311 = launch 1 (index < 265) u launch 2 (index >= 265), each key once
    cand, prefix_of = {}, {}
    for origin, lists in KEYLISTS.items():
        rows, status, launches = {}, [], []
        for c_, p_ in lists:
            b = pinned(c_, p_)
            d = json.loads(b)
            out["inputs"][f"keylist {origin} {d['launch']}"] = f"{p_} @ {c_} (sha256 {sha(b)})"
            launches.append(d["launch"])
            prefix_of[next(iter(d["folders"])).split("__")[0]] = (d["launch"], origin, sum(len(v) for v in d["folders"].values()))
            for folder, ents in d["folders"].items():
                arm = folder.split("__")[1]
                for e in ents:
                    assert e["key"].split("|", 1)[0] == arm, (folder, e["key"])
                    assert e["key"] not in rows, ("a key in two lists", e["key"])
                    rows[e["key"]] = e | {"launch_id": d["launch"], "keylist": f"{p_} @ {c_}"}
            for s_ in d["status_rows"]:
                status.append(s_ | {"launch_id": d["launch"], "source": f"{p_} @ {c_}"})
            out["staging_exclusions"][d["launch"]] = d["staging_exclusions"]
        st_keys = {(s_["key"], s_["launch_id"]) for s_ in status}
        assert not any((k, v["launch_id"]) in st_keys for k, v in rows.items()), (origin, "a status-row key is also listed as counted")
        cand[origin] = rows
        out["epoch2_candidates"][origin] = {"launches": launches, "counted_keys": {a: sum(k.startswith(a + "|") for k in rows) for a in ("honest_a", "honest_b")},
                                            "status_rows_removed": [{k_: s_.get(k_) for k_ in ("key", "launch_id", "eval_sha256", "class", "source")} for s_ in status]}
    # staging exclusions: the stager's c_gate in each launch's chain_stage_ep2_* record (its "out" folder prefix names the key list); a launch without
    # one is "pending"; an excluded epoch is removed by its .eval path
    chain_files = git("ls-tree", "--full-tree", "-r", "--name-only", BCHAIN, "experiments/2026-10-14_q3_overnight_chain/results/").decode().split()
    head_chain = git("rev-parse", BCHAIN).decode().strip()
    stage_recs = sorted(f for f in chain_files if Path(f).name.startswith("chain_stage_ep2_") and f.endswith(".json"))
    staged_info = {}
    for f in stage_recs:
        c_ = git("log", "--format=%h", "-1", BCHAIN, "--", f":(top){f}").decode().strip()
        assert c_, ("no commit found for", f)
        b = git("show", f"{c_}:{f}")
        sr = json.loads(b)
        for part in sr.values():
            if not (isinstance(part, dict) and "c_gate" in part):
                continue
            assert part["out"] in prefix_of, ("staging record names no key list's folder prefix", f, part["out"])
            launch, origin, n_counted = prefix_of[part["out"]]
            g = part["c_gate"]
            assert g["manifest_keys"] == n_counted == g["staged"] + len(g["excluded"]), (f, g["manifest_keys"], n_counted, g["staged"], len(g["excluded"]))
            assert launch not in staged_info, ("two staging records for one launch", launch)
            staged_info[launch] = {"status": "available", "source": f"{f} @ {c_} (sha256 {sha(b)})", "manifest_keys": g["manifest_keys"], "staged": g["staged"],
                                   "excluded": g["excluded"], "origin": origin}
    for launch, v in list(out["staging_exclusions"].items()):
        out["staging_exclusions"][launch] = staged_info.get(launch) or {"status": "pending", "key_file_says": v,
                                                                       "searched": f"{BCHAIN} @ {head_chain[:9]}: no chain_stage_ep2_* record for this launch"}
    for launch, si in staged_info.items():
        rows = cand[si["origin"]]
        for path_e, why in si["excluded"].items():
            lid, rel = path_e.split("#", 1)[0].split("/", 1)
            assert lid == launch, (path_e, launch)
            hit = [k for k, e in rows.items() if e["launch_id"] == launch and e["eval"] == rel]
            assert len(hit) == 1, ("staging exclusion matches no single counted key", path_e)
            out["epoch2_candidates"][si["origin"]]["status_rows_removed"].append({"key": hit[0], "launch_id": launch, "eval_sha256": rows[hit[0]]["eval_sha256"],
                                                                                 "class": f"staging exclusion: {why}", "source": si["source"]})
            del rows[hit[0]]
    for origin, rows in cand.items():
        out["epoch2_candidates"][origin]["after_staging_exclusions"] = {a: sum(k.startswith(a + "|") for k in rows) for a in ("honest_a", "honest_b")}

    # ---- epoch-1 pools
    fb = pinned(FINAL[0], FINAL[1], FINAL[2])
    fin = json.loads(fb)["result"]
    out["inputs"]["FINAL"] = f"{FINAL[1]} @ {FINAL[0]} (sha256 {FINAL[2]})"
    configs = {}
    for k_, r_ in fin["calibration_records"]["primary"].items():
        s_ = r_.get("schemes", {}).get(r_.get("primary"))
        if s_ is None:
            continue
        key = (tuple(sorted(s_["pool_keys_old"])), tuple(sorted(s_["pool_keys_new"])))
        configs.setdefault(key, []).append(k_)
    name = {}
    for (o_, n_), recs in configs.items():
        wo = {x.split("|")[0] for x in o_}
        name[(o_, n_)] = "AuB" if "honest_a" in wo else "BuC" if "honest_c" in wo else "B"
    assert sorted(name.values()) == ["AuB", "B", "BuC"], name.values()
    mb = pinned(MEMBERSHIP[0], MEMBERSHIP[1], MEMBERSHIP[2])
    mem = json.loads(mb)
    out["inputs"]["membership710"] = f"{MEMBERSHIP[1]} @ {MEMBERSHIP[0]} (sha256 {MEMBERSHIP[2]})"
    c311 = json.loads(pinned(*CELLS311))
    cells311 = sorted(c["key"] for c in c311["cells"])
    out["inputs"]["plus311_cells"] = f"{CELLS311[1]} @ {CELLS311[0]} (sha256 {CELLS311[2]})"
    IN = json.loads(pinned(*INPUTS_FINAL))
    assert IN["status"].startswith("FROZEN"), IN["status"]
    out["inputs"]["INPUTS_FINAL (25f honest-A roots; pools.stage_a equality)"] = f"{INPUTS_FINAL[1]} @ {INPUTS_FINAL[0]} (sha256 {INPUTS_FINAL[2]}; status {IN['status']})"
    sa = IN["pools"]["stage_a"]

    def admit(pool_keys, origin, arms, src):
        """pool_keys: epoch-1 member keys. Admit the epoch-2 candidate with the same key for every member whose arm is in arms."""
        adm, miss = [], []
        for k in sorted(pool_keys):
            if k.split("|", 1)[0] not in arms:
                continue
            e = cand[origin].get(k)
            if e is None:
                miss.append(k)
                continue
            adm.append({"key": k, "launch_id": e["launch_id"], "launch_task": e["launch_task"], "eval": e["eval"], "eval_sha256": e["eval_sha256"],
                        "attempt_index": e["attempt_index"], "attempt_kind": e["attempt_kind"], "keylist": e["keylist"],
                        "replicates_epoch1_member": k, "epoch1_source": src})
        return adm, miss

    def pool_entry(label, members, origin, arms, src, notes):
        adm, miss = admit(members, origin, arms, src)
        st = {}
        for s_ in out["epoch2_candidates"][origin]["status_rows_removed"]:
            st.setdefault(s_["key"], []).append(f"{s_['launch_id']}: {s_['class']} ({s_['source']})")
        by_arm = lambda xs: {a: sum(x.split("|", 1)[0] == a for x in xs) for a in sorted({x.split("|", 1)[0] for x in members})}  # noqa: E731
        return {"pool": label, "epoch1_source": src, "epoch2_arms_admitted": sorted(arms), "notes": notes,
                "counts": {"epoch1_members": len(members), "epoch1_by_arm": by_arm(members), "epoch2_admitted": len(adm), "epoch2_by_arm": by_arm([a_["key"] for a_ in adm]),
                           "members_without_epoch2": len(miss)},
                "epoch1_member_keys": sorted(members),
                "admitted_epoch2": adm,
                "members_without_epoch2": [{"key": k, "reason": "; ".join(st.get(k, ["no counted epoch-2 key (not launched, or not counted)"]))}
                                           for k in miss]}

    def nonmember_candidates(origin, all_members):
        return sorted(k for k in cand[origin] if k not in all_members)

    # ---- S_old (T196): FINAL pool_keys_old per configuration
    so = {}
    for (o_, n_), recs in configs.items():
        nm_ = name[(o_, n_)]
        src = f"FINAL calibration_records.primary[*].schemes[primary].pool_keys_old (configuration {nm_}) @ {FINAL[0]}"
        pools = []
        if nm_ == "BuC":
            pools.append(pool_entry("H_B real line (B u C)", set(o_), "T196", {"honest_b"}, src, "e2b added only where epoch-1 B is a member; C unchanged; e2a never stands in for C"))
            pools.append(pool_entry("B alone (yardstick run 1 at tau_B)", {k for k in o_ if k.startswith("honest_b|")}, "T196", {"honest_b"}, src, "e2b replicates of B members"))
            pools.append(pool_entry("C (yardstick run 2 at tau_C)", {k for k in o_ if k.startswith("honest_c|")}, "T196", set(), src, "C unchanged: no epoch 2 (e2a never stands in for C)"))
        elif nm_ == "B":
            pools.append(pool_entry("H_B (B)", set(o_), "T196", {"honest_b"}, src, "e2b replicates of B members"))
        else:
            pools.append(pool_entry("AuB (registered sensitivity)", set(o_), "T196", {"honest_a", "honest_b"}, src,
                                    "honest_a is registered in this pool, so e2a replicates its A members and e2b its B members (never a primary pool)"))
        so[nm_] = {"probe_readout_records": sorted(recs), "pools": pools}
    allold = set().union(*[set(o_) for (o_, n_) in configs])
    so["epoch2_candidates_not_admitted_anywhere"] = nonmember_candidates("T196", allold)
    out["origins"]["S_old"] = so

    # ---- S_new (+311): FINAL pool_keys_new per configuration; H_A = honest_a members (FINAL AuB honest_a) plus INPUTS_FINAL's 25f honest-A roots
    sn, by_new = {}, {}
    for (o_, n_), recs in configs.items():
        by_new.setdefault(n_, []).append((name[(o_, n_)], recs))
    for n_, grp in by_new.items():
        wn = {x.split("|")[0] for x in n_}
        assert "honest_c" not in wn, "an S_new pool holds honest_c"
        nm_ = "AuB" if "honest_a" in wn else "B"
        assert nm_ not in sn, ("two S_new pools with the same arms", nm_)
        old_cfgs = sorted(g[0] for g in grp)
        src = f"FINAL calibration_records.primary[*].schemes[primary].pool_keys_new (S_old configuration {' + '.join(old_cfgs)}) @ {FINAL[0]}"
        arms = {"honest_a", "honest_b"} if nm_ == "AuB" else {"honest_b"}
        sn[nm_] = {"s_old_configurations": old_cfgs, "probe_readout_records": sorted(r for g in grp for r in g[1]),
                   "pools": [pool_entry("AuB (registered sensitivity)" if nm_ == "AuB" else "H_B (B)", set(n_), "+311", arms, src,
                                        "epoch-2 honest_b replicates of B members" + ("; epoch-2 honest_a replicates of A members (never a primary pool)" if nm_ == "AuB" else ""))]}
    a_new = {k for (o_, n_) in configs for k in n_ if k.startswith("honest_a|")}
    ha25 = {}
    for e in IN["sets"]["S_new"]["honest_a_25f"]:
        (cell,) = [c for c in cells311 if c.startswith(e["cell_prefix"])]
        ha25.setdefault(e["kind"], set()).add(f"honest_a|{cell}")
    fam_kind = {"public probes (FIBS, harmfulness, refusal, high-stakes)": "p2", "Sol (a), c_asis": "sol", "own family (own_frac10/25/100_L48)": "famexp",
                "DT (no 25f DT honest-A roots scored)": None}
    assert set(ha25) == {k_ for k_ in fam_kind.values() if k_}, ("INPUTS_FINAL's 25f honest-A kinds differ from the families", sorted(ha25))
    sn["H_A (25h §3 yardstick, held-out false alarm)"] = {
        "rule": "honest_a members = FINAL's registered S_new honest_a keys (AuB pool_keys_new honest_a) plus INPUTS_FINAL's 25f honest-A roots of the probe's family; "
                "Stage A restricts H_A to each probe x readout's H_sim cells, which removes a task's epoch-1 and epoch-2 replicates together (replicate only)",
        "families": {fam: pool_entry(f"H_A, {fam}", a_new | (ha25[k_] if k_ else set()), "+311", {"honest_a"},
                                     f"FINAL AuB pool_keys_new honest_a @ {FINAL[0]} + INPUTS_FINAL honest_a_25f kind {k_}", "epoch-2 honest_a replicates of honest_a members")
                     for fam, k_ in fam_kind.items()}}
    alln = {k for (o_, n_) in configs for k in n_} | set().union(*ha25.values())
    sn["epoch2_candidates_not_admitted_anywhere"] = nonmember_candidates("+311", alln)
    out["origins"]["S_new"] = sn

    # ---- S_newmains (nm360): membership710; H_B = pool_honest_b (+ 4r lists), H_A = honest_a keys owned by a membership folder; with and without S4
    owner = {k: f for f, ks in mem["folder_keys"].items() for k in ks}
    broken = {c for c, v in mem["status_rows"].items() if str(v).startswith("setup failure")}
    hb_all = set(mem["pool_honest_b"]) | set(mem["pool_honest_b_4r_sol_only"])
    excl_runs = set(mem["excluded_runs"])  # membership710's exclusion records by key: never epoch-1 members
    ha_owned = {k for k, f in owner.items() if k.startswith("honest_a|") and k not in excl_runs}
    ha_listed = set(mem["held_out_honest_a"]) | set(mem["held_out_honest_a_4r_sol_only"])
    snm = {"note": "the 4r lists join every probe whose files cover all five 4r folders (membership710's s4_rule); setup-failure cells and membership710's "
                   "excluded_runs are out",
           "excluded_runs_applied": sorted(k for k in excl_runs if k.startswith(("honest_a|", "honest_b|"))),
           "H_A_owned_vs_held_out_lists": {"owned_honest_a_keys": len(ha_owned), "held_out_honest_a_lists": len(ha_listed), "equal": ha_owned == ha_listed,
                                           "owned_not_listed": sorted(ha_owned - ha_listed), "listed_not_owned": sorted(ha_listed - ha_owned)}}
    def keep(k):
        return k not in excl_runs and k.split("|", 1)[1] not in broken
    src = f"membership710 @ {MEMBERSHIP[0]}"
    snm["with S4 and without S4 (one calibration pool; S4r lists joined)"] = {
        "rule": "'without S4' removes S4 from attack estimation only and keeps the parent's calibration pools, thresholds and shared multiplicities "
                "(RULE.md @ 51a612b29, lines 2858 and 2981), so both rows calibrate on these pools; an RMS consumer on S1-S3 + S5 is a different, separately "
                "labelled analysis and is not covered here",
        "pools": [pool_entry("H_B (honest_b pool)", {k for k in hb_all if keep(k)}, "nm360", {"honest_b"}, src + ": pool_honest_b u pool_honest_b_4r_sol_only",
                             "epoch-2 honest_b replicates of B members"),
                  pool_entry("H_A (honest_a owned by a membership folder; Stage A restricts to H_sim cells)", {k for k in ha_owned if keep(k)}, "nm360", {"honest_a"},
                             src + ": folder_keys of honest_a folders", "epoch-2 honest_a replicates of honest_a members"),
                  pool_entry(COVERAGE_LABEL, {k for k in mem["pool_honest_b"] if keep(k)}, "nm360", {"honest_b"}, src + ": pool_honest_b",
                             "never the without-S4 calibration (that is the H_B pool above, identical to with-S4); epoch-2 honest_b replicates of B members; "
                             "H_A needs no variant (Stage A restricts it to H_sim cells)")]}
    # which probes use the coverage pool: a probe uses it when one of the five S4r folders lacks its file (507-prep run_peraction311.py's rule)
    s4r = sorted(mem["s4r_landing"]["arms"])
    have = {}
    for folder, fi in mem["folders"].items():
        for kind, h in fi["manifests"].items():
            mb_ = (args.man710 / f"{folder}__{kind}.json").read_bytes()
            assert sha(mb_) == h, (folder, kind, "manifest copy differs from membership710's sha")
            for fn in json.loads(mb_)["files"]:
                have.setdefault(f"{kind}/{Path(fn).name}", set()).add(folder)
    dtm = json.loads(pinned(*DT_MANIFESTS))
    dt_folders = {Path(k).name.removesuffix(".json") for k in dtm["manifests_dt"]}
    miss_ = {f_: sorted(set(s4r) - fs) for f_, fs in sorted(have.items()) if set(s4r) - fs}
    cov_pool = snm["with S4 and without S4 (one calibration pool; S4r lists joined)"]["pools"][2]
    st_nm = {(s_["key"], s_["launch_id"]): s_ for s_ in out["epoch2_candidates"]["nm360"]["status_rows_removed"]}
    snm["coverage_pool"] = {
        "label": COVERAGE_LABEL,
        "rule": "a probe uses the coverage pool only when one of the five S4r folders lacks its file; otherwise it uses the H_B pool with the 4r lists. "
                "It is never the without-S4 calibration, which equals the with-S4 pool (RULE.md @ 51a612b29)",
        "S4r_folders": s4r,
        "probe_files_checked": {"source": f"membership710 folder manifests (sha-checked copies in {args.man710.name}/)", "files": len(have),
                                "by_family": {k_: sum(f_.startswith(k_ + "/") for f_ in have) for k_ in sorted({f_.split('/')[0] for f_ in have})},
                                "files_missing_from_an_S4r_folder": miss_},
        "DT": {"source": f"{DT_MANIFESTS[1]} @ {DT_MANIFESTS[0]} (sha256 {DT_MANIFESTS[2]})", "S4r_folders_with_DT_manifests": sorted(set(s4r) & dt_folders),
               "S4r_folders_without": sorted(set(s4r) - dt_folders)},
        "probes_using_the_coverage_pool": sorted({f_ for f_ in miss_}) + (["DT (all designs)"] if set(s4r) - dt_folders else []),
        "not_checked": "own_all (its files come from its own score shas, not membership710's manifests); 507-prep's rebuild records list the probes it ran in "
                       "meta.coverage_pool_probes (run_peraction311.py on aswin/fpr-sweep-507-prep)",
        "coverage_gaps_by_key": [{"key": m_["key"], "pool": COVERAGE_LABEL,
                                  "also_missing_in": [q_["pool"] for q_ in snm["with S4 and without S4 (one calibration pool; S4r lists joined)"]["pools"][:2]
                                                      if m_["key"] in {x_["key"] for x_ in q_["members_without_epoch2"]}],
                                  "gap": "epoch-1 member with no usable epoch-2 replicate",
                                  "removed_epoch2_run": {k2: st_nm[(m_["key"], lid)][k2] for k2 in ("launch_id", "eval_sha256", "class", "source")}}
                                 for m_ in cov_pool["members_without_epoch2"] for lid in out["epoch2_candidates"]["nm360"]["launches"] if (m_["key"], lid) in st_nm]}
    assert len(snm["coverage_pool"]["coverage_gaps_by_key"]) == len(cov_pool["members_without_epoch2"]), "a coverage-pool gap without a status row"
    snm["epoch2_candidates_not_admitted_anywhere"] = nonmember_candidates("nm360", hb_all | ha_owned)
    out["origins"]["S_newmains"] = snm

    # an origin with a pending launch reads "not in this build" (no keys) unless this is a --draft build
    for o, co in (("S_old", "T196"), ("S_new", "+311"), ("S_newmains", "nm360")):
        pend = sorted(lch for lch in out["epoch2_candidates"][co]["launches"] if out["staging_exclusions"][lch]["status"] == "pending")
        if pend and args.draft is None:
            out["origins"][o] = {"status": "not in this build", "reason": f"staging exclusions pending for {pend} (no chain_stage_ep2_* record at build)"}
        else:
            out["origins"][o] = {"status": f"DRAFT: staging exclusions pending for {pend}" if pend else "complete"} | out["origins"][o]
    # ---- epoch-1 members equal Stage A's own resolution (INPUTS_FINAL pools.stage_a); H_sim-restricted H_A is Stage A's, not checked here
    rk = sa["S_old"]["H_B"]["key"].split('"')[1]
    assert sa["S_old"]["H_B"]["key"] == f'result.calibration_records.primary["{rk}"].schemes[<primary>].pool_keys_old' and (sa["S_old"]["H_B"]["commit"], sa["S_old"]["H_B"]["path"],
                                                                                                                       sa["S_old"]["H_B"]["sha256"]) == FINAL, sa["S_old"]["H_B"]
    assert sa["S_new"]["H_B"]["key"].replace("pool_keys_new", "pool_keys_old") == sa["S_old"]["H_B"]["key"] and sa["S_new"]["H_B"]["path"] == FINAL[1], sa["S_new"]["H_B"]
    r_ = fin["calibration_records"]["primary"][rk]
    sch = r_["schemes"][r_["primary"]]
    so_b = so["BuC"]
    assert rk in so_b["probe_readout_records"] and set(so_b["pools"][0]["epoch1_member_keys"]) == set(sch["pool_keys_old"]) and len(sch["pool_keys_old"]) == sa["S_old"]["H_B"]["n"]
    assert set(so_b["pools"][2]["epoch1_member_keys"]) == {k for k in sch["pool_keys_old"] if k.startswith("honest_c|")} and len(so_b["pools"][2]["epoch1_member_keys"]) == sa["S_old"]["H_A"]["n"]
    assert rk in sn["B"]["probe_readout_records"] and set(sn["B"]["pools"][0]["epoch1_member_keys"]) == set(sch["pool_keys_new"]) and len(sch["pool_keys_new"]) == sa["S_new"]["H_B"]["n"]
    assert {next(x for x in e["folder"].split("/") if x.endswith("__honest_a")) for e in IN["sets"]["S_new"]["honest_a_25f"]} == set(sa["S_new"]["H_A"]["honest_a_25f"]), "25f honest-A roots differ from pools.stage_a"
    hbm = sa["S_newmains"]["H_B"]
    assert hbm["sha256"] == MEMBERSHIP[2] and hbm["keys"] == ["pool_honest_b", "pool_honest_b_4r_sol_only"], hbm
    snm_p = snm["with S4 and without S4 (one calibration pool; S4r lists joined)"]["pools"]
    assert snm_p[2]["pool"] == COVERAGE_LABEL and len(snm_p[0]["epoch1_member_keys"]) == hbm["n"] and len(snm_p[2]["epoch1_member_keys"]) == hbm["n_without_4r_lists"], (
        hbm, [len(x["epoch1_member_keys"]) for x in snm_p])
    out["stage_a_equality"] = {"source": out["inputs"]["INPUTS_FINAL (25f honest-A roots; pools.stage_a equality)"], "checked_record": rk,
                               "S_old H_B (BuC real line) = pool_keys_old": True, "S_old C = honest_c of H_B (Stage A's S_old H_A)": True,
                               "S_new H_B (B) = pool_keys_new": True, "S_new 25f honest-A root folders": True,
                               "S_newmains H_B with 4r lists (n)": hbm["n"], "S_newmains H_B without 4r lists (n)": hbm["n_without_4r_lists"],
                               "not checked": "H_A restricted to H_sim cells (Stage A applies it per probe; this artifact lists the unrestricted honest_a members)"}
    summ = {}
    for o, d in out["origins"].items():
        if d["status"] == "not in this build":
            summ[o] = "not in this build"
            continue
        for cfg, v in d.items():
            if isinstance(v, dict) and "pools" in v:
                for p_ in v["pools"]:
                    summ[f"{o} | {cfg} | {p_['pool']}"] = p_["counts"]
            elif isinstance(v, dict) and "families" in v:
                for fam, x in v["families"].items():
                    summ[f"{o} | {cfg} | {fam}"] = x["counts"]
        summ[f"{o} | epoch-2 candidates admitted to no pool"] = len(d["epoch2_candidates_not_admitted_anywhere"])
    out["summary"] = summ
    out["meta"] = {"git": git("rev-parse", "HEAD").decode().strip(), "script_sha256": sha(Path(__file__).read_bytes()),
                   "stamp_pt": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S"),
                   "reads": "git only: key lists, FINAL calibration keys, membership710, INPUTS_FINAL pins, the +311 cell list; no score file, no Modal"}
    out["meta"]["draft"] = args.draft is not None
    if args.draft is not None:
        out["label"] = "DRAFT (not for use): " + out["label"]
    dest = args.draft or HERE / "results" / f"epoch2_poolkeys_{out['meta']['stamp_pt']}.json"
    dest.parent.mkdir(exist_ok=True)
    dest.open("x").write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["summary"], indent=1))
    print(dest, sha(dest.read_bytes()))


if __name__ == "__main__":
    main()
