"""The epoch 1 + 2 calibration block (09 @ 64f00b09e; "e12"): every input run_peraction311.py --calib e12 reads beyond the frozen INPUTS_FINAL, by key and
sha, git-only (plus the sha-checked local manifest copies fetched by fetch_slot_manifests.py). Keys, shas and counts only; no score file is opened.

  python make_e12_block.py [--test] --sp493 <INPUTS_FINAL v2 path>@<sha256>   # writes config/E12_BLOCK[_TEST]_<stamp>.json (exclusive create)

What it pins (schema e12_block/v1):
  rule, rulings      09's RULE @ 64f00b09e (sha of RULE.md there); Q1 (namespace honest_x_e2|cell; identity = physical key + .eval sha256 + the replicated
                     epoch-1 member) and Q4 (S_old H_A = honest_c gets no epoch 2; S_old e2b enter B u C) as relayed
  artifact           the epoch-2 pool-key artifact @ 5b19c08d8 (coverage-pool relabel), and the 19:20 build @ f303cb98d it was verified against: the
                     admitted epoch-2 keys of every pool present in both must be equal by key (else STOP)
  pools              per origin and pool: the artifact's epoch-1 member count and its admitted epoch-2 entries, renamed to honest_x_e2|cell
  folders            the epoch-2 score folders the primary pools need (S_old: ep2_t196 honest_b only; S_new: ep2_311_l1/l2 a + b; S_newmains: ep2_nm a + b),
                     per family file maps {"<family>/<file>": {path, sha256, vol}} from the sealed score records (Sol, p2 inline; famexp, own_all_rs via
                     their manifest copies, each sha-checked against its record)
  fill_in            S_old tau_sim = primary_v3 u A25 fill-in (simh_t196fill, 8 admitted) u g4 (14 admitted, world sp493_t196), files per family
  coverage           per family x folder: present or missing (a missing headline family STOPs the job; others are withheld, labelled)
  asserts            (i) every admitted epoch-2 entry's .eval sha256 equals d9's stager manifest entry for its eval path; (ii) no epoch-2 sha equals any
                     sha in the epoch-1 stager manifests (every chain manifest except ep2), and the epoch-2 launch ids are disjoint from the epoch-1 ones
"""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
COPIES = Path("/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/a683364c-3a60-44cd-bb15-e08d87dd4f9a/scratchpad/fpr507")
CF = "experiments/2026-10-14_q3_overnight_chain/config"
SCH = "experiments/2026-10-14_p2_simh_scores/results"
SOL = "experiments/2026-10-14_q35_sol_avg_POSTHOC/results"
FAM = "experiments/2026-10-14_p2_family_expansion/results"
RULE = ("64f00b09e", "experiments/2026-10-14_priority1_spec_09/RULE.md")
ART = ("5b19c08d8", "experiments/2026-10-15_epoch2_poolkeys/results/epoch2_poolkeys_2026-10-15_214512.json")
VER = ("f303cb98d", "experiments/2026-10-15_epoch2_poolkeys/results/epoch2_poolkeys_2026-10-15_192028.json")
CAVEAT = ("real pools use epoch 1 + 2 (about twice the runs of τ_sim's pool): calibration precision is asymmetric between sim and real thresholds; "
          "S_old τ_sim includes the A25 fill-in")  # 09 @ 64f00b09e §4, verbatim
RULINGS = {"Q1": ("honest_a_e2|<cell> / honest_b_e2|<cell> is the namespace; identity = PHYSICAL run key + .eval sha256 + the epoch-1 member it replicates "
                  "(Append 38 §4, §6), verified as the relabel @ 5b19c08d8 against the 19:20 build @ f303cb98d; asserts (i) and (ii) in their own call"),
           "Q4": ("S_old's H_A (honest_c) gets NO epoch 2 ('S_old | BuC | C … epoch2_admitted 0'); the S_old epoch-2 replicates are e2b and enter the B∪C "
                  "τ_B pool (191); T196's e2a has no member to replicate in S_old's registered pools; AuB is the registered labelled sensitivity, not primary"),
           "source": "09's rulings relayed by 2e via 507-prep (2026-10-16 ~02:10 PT)"}
HEADLINE_FAMILIES = ("sol", "p2", "own_all_rs")  # Sol (a), the four external probes, own_all_rs_L48 (c_asis rides in Sol's files)
# primary pools by origin: (pool name in the artifact, where in the artifact) -> the runner's pool
POOLS = {"S_old": {"H_B": ("BuC", "H_B real line (B u C)"), "H_B2": ("BuC", "B alone (yardstick run 1 at tau_B)"), "H_A": ("BuC", "C (yardstick run 2 at tau_C)")},
         "S_new": {"H_B": ("B", "H_B (B)")},
         "S_newmains": {"H_B": ("with S4 and without S4 (one calibration pool; S4r lists joined)", "H_B (honest_b pool)"),
                        "H_B_coverage": ("with S4 and without S4 (one calibration pool; S4r lists joined)", "H_B restricted to folders this probe's files cover (coverage pool)"),
                        "H_A": ("with S4 and without S4 (one calibration pool; S4r lists joined)", "H_A (honest_a owned by a membership folder; Stage A restricts to H_sim cells)")}}
HA_NEW_GROUP = {"p2": "public probes (FIBS, harmfulness, refusal, high-stakes)", "sol": "Sol (a), c_asis", "famexp": "own family (own_frac10/25/100_L48)",
                "dt": "DT (no 25f DT honest-A roots scored)"}
FOLDERS = {  # folder -> origin, world, and the sealed records per family
    "ep2_t196_v1__honest_b": ("S_old", "honest_b"), "ep2_311_l1_v1__honest_a": ("S_new", "honest_a"), "ep2_311_l1_v1__honest_b": ("S_new", "honest_b"),
    "ep2_311_l2_v1__honest_a": ("S_new", "honest_a"), "ep2_311_l2_v1__honest_b": ("S_new", "honest_b"), "ep2_nm_v1__honest_a": ("S_newmains", "honest_a"),
    "ep2_nm_v1__honest_b": ("S_newmains", "honest_b")}
SOL_RECS = {"ep2_t196_v1__honest_b": ("82e6ba337", f"{SOL}/scaleup507_ep2_t196_v1__honest_b_2026-10-15_184345.json"),
            "ep2_311_l1_v1__honest_a": ("553960bcd", f"{SOL}/scaleup507_ep2_311_l1_v1__honest_a_2026-10-15_191812.json"),
            "ep2_311_l1_v1__honest_b": ("553960bcd", f"{SOL}/scaleup507_ep2_311_l1_v1__honest_b_2026-10-15_191816.json"),
            "ep2_311_l2_v1__honest_a": ("747f5ae07", f"{SOL}/scaleup507_ep2_311_l2_v1__honest_a_2026-10-15_183520.json"),
            "ep2_311_l2_v1__honest_b": ("747f5ae07", f"{SOL}/scaleup507_ep2_311_l2_v1__honest_b_2026-10-15_183523.json"),
            "ep2_nm_v1__honest_a": ("1482d08e8", f"{SOL}/scaleup507_ep2_nm_v1__honest_a_2026-10-15_190643.json"),
            "ep2_nm_v1__honest_b": ("1482d08e8", f"{SOL}/scaleup507_ep2_nm_v1__honest_b_2026-10-15_190647.json"),
            "simh_t196fill_v1__sim_honest": ("1b35852d6", f"{SOL}/scaleup507_simh_t196fill_v1__sim_honest_2026-10-15_193122.json")}
P2_RECS = [("21130122b", f"{SCH}/scaleup_manifest_2026-10-15_184343.json"), ("3fdc383cd", f"{SCH}/scaleup_manifest_2026-10-15_185715.json"),
           ("a9f5a6b54", f"{SCH}/scaleup_manifest_2026-10-15_182306.json"), ("31af55b5a", f"{SCH}/scaleup_manifest_2026-10-15_184738.json"),
           ("11b1d0758", f"{SCH}/simh_manifest_2026-10-15_234133.json")]
FAMEXP_RECS = [("246f0d3c4", f"{FAM}/pass2_ep2_t196_2026-10-15_184736.json"), ("68b620b3d", f"{FAM}/pass2_ep2_311_l1_2026-10-15_234339.json"),
               ("ba3b3e5ee", f"{FAM}/pass2_ep2_311_l2_2026-10-15_182017.json"), ("246f0d3c4", f"{FAM}/pass2_ep2_nm_2026-10-15_184738.json")]
RS_RECS = [("57161b203", f"{FAM}/own_all_rs_score_2026-10-16_014050.json"), ("1f6dfa141", f"{FAM}/own_all_rs_score_2026-10-16_022424.json")]
STAGER = {"2026-10-15_115153_ep2_t196": ("91abd7639", f"{CF}/manifest_ep2_t196.json"), "2026-10-15_122516_ep2_311": ("237732c75", f"{CF}/manifest_ep2_311_l1.json"),
          "2026-10-15_144034_ep2_311b": ("2fa788597", f"{CF}/manifest_ep2_311_l2.json"), "2026-10-15_123630_ep2_nm": ("e5ba3b226", f"{CF}/manifest_ep2_nm.json")}
UNCUT_AT = ("16609a1b5", "bea00298c")  # the honest pools' uncut (cutfull) routing r1 + attestations, and their pointers (aswin/q3-cut-option)
UNCUT_DIR = "experiments/2026-10-14_q3_overnight_chain/results/cutfull/routing"
UNCUT = {"ep2-t196": ("ep2-t196__ep2_t196_cutfull", "2026-10-15_115153_ep2_t196"), "ep2-311-l1": ("ep2-311-l1__ep2_311_l1_cutfull", "2026-10-15_122516_ep2_311"),
         "ep2-311-l2": ("ep2-311-l2__ep2_311_l2_cutfull", "2026-10-15_144034_ep2_311b"), "ep2-nm": ("ep2-nm__ep2_nm_cutfull", "2026-10-15_123630_ep2_nm"),
         "simh-t196fill": ("simh-t196fill__simh_t196fill_cutfull", None)}
KEYLISTS = {"2026-10-15_115153_ep2_t196": ("b51f5c52c", "experiments/2026-10-14_q3_overnight_chain/results/hygiene/ep2_keylist_t196_2026-10-15.json"),
            "2026-10-15_122516_ep2_311": ("15b79df39", "experiments/2026-10-14_q3_overnight_chain/results/hygiene/ep2_keylist_311_l1_2026-10-15.json"),
            "2026-10-15_144034_ep2_311b": ("b51f5c52c", "experiments/2026-10-14_q3_overnight_chain/results/hygiene/ep2_keylist_311_l2_2026-10-15.json"),
            "2026-10-15_123630_ep2_nm": ("5197f3759", "experiments/2026-10-14_q3_overnight_chain/results/hygiene/ep2_keylist_nm_2026-10-15.json")}
INVENTORY = ("06077a8a8", "experiments/2026-10-15_cut_sensitivity/results/inventory_ep2_v1_2026-10-16_025227.json",
             "experiments/2026-10-15_cut_sensitivity/results/inventory_fill_v1_2026-10-16_025228.json")  # CUT's uncut inventories (aswin/cut-sensitivity-2e)
EPOCH1_MANIFESTS_AT = "463e944f9"  # every chain stager manifest at this commit except the ep2 ones (assert ii's epoch-1 side)
FILL = {"A25": {"root": "simh_t196fill_v1__sim_honest", "batch": "simh_t196fill", "world": "t196fill", "cells": ("t04", "t07", "t08", "t11", "t12", "t13", "t17", "t48"),
                "routing": "simh-t196fill__simh_t196fill.json", "manifest": ("61482c1c0", f"{CF}/manifest_simh_t196fill.json"),
                "attestation": ("1549de878", "experiments/2026-10-14_q3_overnight_chain/results/hygiene/attestations_simh_t196fill_simh-t196fill.json"),
                "ack": "09 ACK READ_LOG l.641/800", "label": "A25 fill-in"},
        "g4": {"root": "sp493_g4_simh_v1_try2__sim_honest", "batch": "sp493_g4", "world": "sp493_t196",
               "cells": ("t01", "t02", "t03", "t05", "t06", "t09", "t14", "t15", "t16", "t18", "t19", "t20", "t43", "t115"),
               "routing": "sp493-g4__sp493_g4_simh.json (1aa0110a…, release r1)", "manifest": ("1f3f914e9", f"{CF}/manifest_sp493_g4_simh.sample_error_v1.json"),
               "label": "g4 (S_paper493 sim-honest on T196 mains)"}}


def task_cell(task):
    """The cell of a launch task or eval task dir: the cell itself, or plus T196's _e2a/_e2b tag and/or _redraw<n> (CUT 03:4x); never a raw join."""
    import re

    return re.sub(r"(_e2[ab]|_redraw\d+)+$", "", task)  # both tags, in either order


def git_show(c, p):
    return subprocess.run(["git", "show", f"{c}:{p}"], capture_output=True, check=True, cwd=REPO).stdout


def sha(b):
    return hashlib.sha256(b).hexdigest()


def pin(c, p):
    return {"commit": c, "path": p, "sha256": sha(git_show(c, p))}


def load(c, p):
    return json.loads(git_show(c, p))


def pool_of(art, org, where, name):
    src = art["origins"][org][where]
    return next(x for x in src["pools"] if x["pool"] == name)


def e2_entries(pool, folder_of_launch):
    out = []
    for x in pool["admitted_epoch2"]:
        w, c = x["key"].split("|", 1)
        if x["replicates_epoch1_member"] != x["key"]:
            raise SystemExit(f"{x['key']}: replicates {x['replicates_epoch1_member']} (expected the same physical key)")
        if x["eval"].split("/")[0] != w or task_cell(x["eval"].split("/")[1]) != c:  # the eval's task dir names this cell (redraw / e2 tags only)
            raise SystemExit(f"{x['key']}: eval task dir {x['eval'].split('/')[1]} is not this cell")
        out.append({"key_e2": f"{w}_e2|{c}", "key": x["key"], "replicates": x["replicates_epoch1_member"], "eval": x["eval"], "eval_sha256": x["eval_sha256"],
                    "launch_id": x["launch_id"], "folder": folder_of_launch(x["launch_id"], w), "inventory_key": f"{x['launch_id']}/{x['eval']}#e0"})
    return sorted(out, key=lambda e: e["key_e2"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sp493", required=True, help="INPUTS_FINAL v2 <path>@<sha256> (the g4 sim-honest files and keys)")
    ap.add_argument("--test", action="store_true", help="TEST block: missing headline files are listed, not fatal")
    a = ap.parse_args()
    art_b, ver_b = git_show(*ART), git_show(*VER)
    art, ver = json.loads(art_b), json.loads(ver_b)
    if art["meta"].get("draft") or ver["meta"].get("draft"):
        raise SystemExit("a draft pool-key artifact")

    def folder_of_launch(lid, w):
        f = {"2026-10-15_115153_ep2_t196": "ep2_t196_v1", "2026-10-15_122516_ep2_311": "ep2_311_l1_v1", "2026-10-15_144034_ep2_311b": "ep2_311_l2_v1",
             "2026-10-15_123630_ep2_nm": "ep2_nm_v1"}[lid]
        return f"{f}__{w}"

    pools, vs = {}, {}
    for org, d in POOLS.items():
        for nm, (where, name) in d.items():
            p_a = pool_of(art, org, where, name)
            ents = e2_entries(p_a, folder_of_launch)
            try:
                p_v = pool_of(ver, org, where, name)
                ka, kv = {e["key"] for e in ents}, {x["key"] for x in p_v["admitted_epoch2"]}
                vs[f"{org}|{nm}"] = {"equal": ka == kv, "only_artifact": sorted(ka - kv), "only_verified_build": sorted(kv - ka)}
                if ka != kv:
                    raise SystemExit(f"{org} {nm}: admitted epoch-2 keys differ between the relabel and the 19:20 build")
            except StopIteration:
                vs[f"{org}|{nm}"] = {"equal": "not in the 19:20 build (added by the relabel)"}
            if org == "S_old" and nm == "H_A" and ents:
                raise SystemExit("S_old H_A (honest_c) has epoch-2 entries (Q4: none)")
            pools.setdefault(org, {})[nm] = {"artifact_pool": f"{where} / {name}", "epoch1_members": len(p_a["epoch1_member_keys"]),
                                             "epoch1_members_sha256": sha(json.dumps(sorted(p_a["epoch1_member_keys"])).encode()), "e2": ents}
    ha_new = art["origins"]["S_new"]["H_A (25h §3 yardstick, held-out false alarm)"]["families"]
    ha_ver = ver["origins"]["S_new"]["H_A (25h §3 yardstick, held-out false alarm)"]["families"]
    for g, v in ha_new.items():  # the H_A family groups, by key, against the 19:20 build too
        ka, kv = {x["key"] for x in v["admitted_epoch2"]}, {x["key"] for x in ha_ver.get(g, {}).get("admitted_epoch2", [])}
        vs[f"S_new|H_A|{g}"] = {"equal": ka == kv, "only_artifact": sorted(ka - kv), "only_verified_build": sorted(kv - ka)}
        if ka != kv:
            raise SystemExit(f"S_new H_A {g}: admitted epoch-2 keys differ between the relabel and the 19:20 build")
    pools["S_new"]["H_A"] = {g: {"artifact_pool": f"H_A / {g}", "epoch1_members": len(v["epoch1_member_keys"]), "e2": e2_entries(v, folder_of_launch)}
                             for g, v in ha_new.items()}
    pools["S_new"]["H_A_group_of_family"] = HA_NEW_GROUP | {"own_a": None, "own_all_rs": None, "_rule": (
        "own_a / own_all_rs (not in the artifact's H_A families): e2a entries of any group whose replicated epoch-1 member is in the row's H_A, by key")}
    for org, d in pools.items():  # every e2 entry's folder is one this block pins (or the S_old AuB-only honest_a folder, never primary)
        for nm, v in d.items():
            for e in (v["e2"] if "e2" in v else [x for g in v.values() if isinstance(g, dict) and "e2" in g for x in g["e2"]]):
                if e["folder"] not in FOLDERS or FOLDERS[e["folder"]][0] != org:
                    raise SystemExit(f"{org} {nm}: {e['key_e2']} in folder {e['folder']} (not this origin's pinned folder)")

    # ---- asserts (i) and (ii), in their own call
    def asserts():
        stg = {lid: load(c, p) for lid, (c, p) in STAGER.items()}
        by_rel = {lid: {x["rel"]: x["sha256"] for x in m["files"]} for lid, m in stg.items()}
        all_e2 = {e["key_e2"]: e for d in pools.values() for v in d.values() if isinstance(v, dict)
                  for e in (v.get("e2") or [x for g in v.values() if isinstance(g, dict) for x in g.get("e2", [])])}
        bad_i = [k for k, e in all_e2.items() if by_rel[e["launch_id"]].get(e["eval"]) != e["eval_sha256"]]
        ls = subprocess.run(["git", "ls-tree", "--name-only", EPOCH1_MANIFESTS_AT, CF + "/"], capture_output=True, text=True, check=True, cwd=REPO).stdout.split()
        e1m = [p for p in ls if Path(p).name.startswith("manifest_") and p.endswith(".json") and "manifest_ep2_" not in p]
        e1_sha, e1_launch = set(), set()
        for p in e1m:
            m = json.loads(git_show(EPOCH1_MANIFESTS_AT, p))
            for x in m.get("files", []) if isinstance(m, dict) else []:
                if isinstance(x, dict) and "sha256" in x:
                    e1_sha.add(x["sha256"])
                    e1_launch.add(x.get("launch"))
        e2_sha = {e["eval_sha256"] for e in all_e2.values()}
        e2_launch = {e["launch_id"] for e in all_e2.values()}
        bad_ii = sorted(e2_sha & e1_sha)
        res = {"i": {"rule": "each admitted epoch-2 entry's .eval sha256 = d9's stager manifest entry for its eval path (by launch)",
                     "stager_manifests": {lid: pin(c, p) for lid, (c, p) in STAGER.items()}, "checked": len(all_e2), "mismatch": sorted(bad_i), "pass": not bad_i},
               "ii": {"rule": "no epoch-2 .eval sha256 equals any sha in the epoch-1 chain stager manifests; epoch-2 launch ids disjoint from theirs",
                      "epoch1_manifests": {"commit": EPOCH1_MANIFESTS_AT, "n_files": len(e1m), "n_eval_shas": len(e1_sha)}, "e2_shas": len(e2_sha),
                      "collisions": bad_ii, "launch_overlap": sorted(e2_launch & e1_launch), "pass": not bad_ii and not (e2_launch & e1_launch),
                      "coverage_note": ("the epoch-1 side is every chain stager manifest at the commit (incl. T196's honest_b via manifest_q4b_eval_upload_v1 "
                                        "and honest_c via manifest_q4f_c_eval_upload_v1, +311 via su311_q*_main*, nm360 via nm360_s*_main*); a pool member whose eval "
                                        "is in none of them is not covered by (ii)")}}
        if not (res["i"]["pass"] and res["ii"]["pass"]):
            raise SystemExit(f"identity asserts failed: (i) {bad_i[:5]} (ii) {bad_ii[:5]} {sorted(e2_launch & e1_launch)}")
        res["_e1_sha"] = sorted(e1_sha)
        return res

    ident = asserts()
    e1_sha_list = ident.pop("_e1_sha")

    def uncut_inventory():  # 507-prep 02:5x: pool membership uses the ADMITTED keys only (never the routed-not-admitted extras), by key
        inv = load(INVENTORY[0], INVENTORY[1])
        keys = {x["key"] for x in inv["runs"]}
        extra = {x["key"] for x in inv["epoch2_admitted_subset_check"]["routed_not_admitted"]}
        mine = {f"{e['launch_id']}/{e['eval']}#e0": e["key_e2"] for d in pools.values() for v in d.values() if isinstance(v, dict)
                for e in (v.get("e2") or [x for g in v.values() if isinstance(g, dict) for x in g.get("e2", [])])}
        bad_r, bad_x = sorted(k for k in mine if k not in keys), sorted(mine[k] for k in mine if k in extra)
        if inv["status"] != "COMPLETE" or bad_r or bad_x:
            raise SystemExit(f"uncut inventory: status {inv['status']}, block keys not routed {bad_r[:3]}, block keys among the routed-not-admitted {bad_x[:3]}")
        fill = load(INVENTORY[0], INVENTORY[2])
        return {"ep2": pin(INVENTORY[0], INVENTORY[1]), "fill": pin(INVENTORY[0], INVENTORY[2]), "fill_status": fill.get("status"),
                "assert": {"block_e2_keys": len(mine), "all_routed": True, "none_routed_not_admitted": True, "routed_not_admitted": len(extra),
                           "inventory_admitted_keys": inv["epoch2_admitted_subset_check"]["admitted_keys"],
                           "note": "the inventory's 1,665 admitted keys include S_old's AuB-only e2a (registered sensitivity, never primary): not in this block"}}

    def uncut_routing():  # 507-prep 02:5x: pin the honest pools' uncut routing; routed epochs per folder = the artifact's keys by key (honest_a / honest_b)
        out, bad = {}, []
        cand = {lid: {} for lid in KEYLISTS}  # artifact: admitted-or-not candidates by launch and world = the keylist keys after staging exclusions
        for lid, (c, p) in KEYLISTS.items():
            kl = load(c, p)
            for fo, ks in kl["folders"].items():
                cand[lid][fo.rsplit("__", 1)[1]] = {x["key"] for x in ks}
        grp = {"2026-10-15_115153_ep2_t196": "T196", "2026-10-15_122516_ep2_311": "+311", "2026-10-15_144034_ep2_311b": "+311", "2026-10-15_123630_ep2_nm": "nm360"}
        for g in ("T196", "+311", "nm360"):  # the artifact's counted keys per launch group equal the keylists' keys (by count; keylists are its inputs)
            for w in ("honest_a", "honest_b"):
                n_kl = sum(len(cand[lid].get(w, ())) for lid in cand if grp[lid] == g)
                n_art = art["epoch2_candidates"][g]["after_staging_exclusions"][w]
                if n_kl != n_art:
                    bad.append(f"{g} {w}: keylists {n_kl} != artifact {n_art}")
        for b, (fn, lid) in UNCUT.items():
            r = load(UNCUT_AT[0], f"{UNCUT_DIR}/{fn}.json")
            arms = {x["meta_arm"]: {"epochs": x["epochs"], "staged_sha": x["staged_sha"], "format_check": x["format_check"]} for x in r["arms"]}
            ent = {"routing": pin(UNCUT_AT[0], f"{UNCUT_DIR}/{fn}.json"), "pointer": pin(UNCUT_AT[1], f"{UNCUT_DIR}/{fn}_attest.json"),
                   "original_folders": r["original_folders"], "arms": arms, "cut": r["cut"], "revision": r["revision"]}
            if lid:
                for w, v in arms.items():
                    want = cand[lid].get(w, set())
                    ok = v["epochs"] == len(want) and {task_cell(m) for m in r["members"]} >= {k.split("|", 1)[1] for k in want}
                    ent.setdefault("assert_epochs_eq_artifact_keys", {})[w] = {"routed": v["epochs"], "artifact_keys": len(want), "pass": ok}
                    if not ok:
                        bad.append(f"{b} {w}: routed {v['epochs']} vs artifact keys {len(want)}")
            else:
                ent["assert_epochs_eq_admitted"] = {w: {"routed": v["epochs"], "admitted": 8, "pass": v["epochs"] == 8} for w, v in arms.items()}
                if any(v["epochs"] != 8 for v in arms.values()):
                    bad.append(f"{b}: routed {[v['epochs'] for v in arms.values()]} vs 8 admitted")
            out[b] = ent
        if bad:
            raise SystemExit(f"uncut routing vs the pool-key artifact: {bad}")
        return out

    # ---- per-folder file maps, per family
    folders = {fo: {"arm": fo, "origin": o, "world": w, "world_e2": f"{w}_e2", "files": {}, "sources": {}} for fo, (o, w) in FOLDERS.items()}
    fill = {k: dict(v) for k, v in FILL.items()}
    for k, v in fill.items():
        v["manifest"] = pin(*v["manifest"])
        if "attestation" in v:
            v["attestation"] = pin(*v["attestation"])
        v["files"], v["sources"] = {}, {}
    fa = fill["A25"]
    stg_fill = json.loads(git_show(FILL["A25"]["manifest"][0], FILL["A25"]["manifest"][1]))
    tasks_ = sorted({x["task"] for x in stg_fill["files"]})
    fa["admitted"] = sorted({f"{fa['world']}|{task_cell(t)}" for t in tasks_ if task_cell(t).split("_", 1)[0] in FILL["A25"]["cells"]})
    if len(fa["admitted"]) != 8 or {k.split("|")[1].split("_", 1)[0] for k in fa["admitted"]} != set(FILL["A25"]["cells"]):
        raise SystemExit(f"A25 fill-in admitted list {fa['admitted']} != the 8 admitted cells")
    sp_path, sp_sha = a.sp493.rsplit("@", 1)
    spb = (HERE / sp_path).read_bytes()
    if sha(spb) != sp_sha:
        raise SystemExit("--sp493 sha256")
    g4e = [e for e in json.loads(spb)["extensions"]["S_paper493"]["simh"] if e["root"] == FILL["g4"]["root"]]
    if len(g4e) != 1:
        raise SystemExit("g4 root not exactly once in the v2 manifest")
    fg = fill["g4"]
    fg["admitted"] = sorted(g4e[0]["keys"])
    if {k.split("|")[1].split("_", 1)[0] for k in fg["admitted"]} != set(FILL["g4"]["cells"]) or len(fg["admitted"]) != 14 or any(
            not k.startswith("sp493_t196|") for k in fg["admitted"]):
        raise SystemExit(f"g4 admitted list {fg['admitted']} != the 14 admitted (world sp493_t196)")
    fg["files"] = dict(g4e[0]["files"])
    fg["sources"]["p2+own_all_rs"] = {"path": sp_path, "sha256": sp_sha, "what": "INPUTS_FINAL v2 extensions.S_paper493.simh[g4].files"}

    for fo, (c, p) in SOL_RECS.items():  # Sol: files inline in the sealed record
        r = load(c, p)
        vol, d = r["outputs"]["path"].split(":", 1)
        if r["arm"] != fo or "convaudit_work" not in vol or not r["outputs"]["published"]:
            raise SystemExit(f"Sol record {p}: arm/volume/published")
        tgt = folders[fo] if fo in folders else fa
        for fn, h in r["outputs"]["files"].items():
            if fn.endswith(".npz"):
                tgt["files"][f"sol/{fn}"] = {"path": f"{d.strip('/')}/{fn}", "sha256": h, "vol": "conv"}
        tgt["sources"]["sol"] = pin(c, p)
    for c, p in P2_RECS:  # p2: per folder, files inline
        r = load(c, p)
        for d, v in r["folders"].items():
            fo = v.get("name") or d.rstrip("/").split("/")[-1]
            tgt = folders.get(fo) if fo in folders else fa if fo == fa["root"] else None
            if tgt is None:
                continue
            for fn, h in v["files"].items():
                if fn.endswith(".npz"):
                    tgt["files"][f"p2/{fn}"] = {"path": f"{d.strip('/')}/{fn}", "sha256": h, "vol": "conv"}
            tgt["sources"]["p2"] = pin(c, p)
    for c, p in FAMEXP_RECS:  # famexp: the manifest copy, sha-checked against the record
        r = load(c, p)
        for fo, v in r["result"]["manifests"].items():
            if fo not in folders:
                continue
            b = (COPIES / "e12man" / f"{fo}__famexp.json").read_bytes()
            if sha(b) != v["sha256"] or not v["complete"]:
                raise SystemExit(f"famexp {fo}: manifest copy sha256 differs from {p}")
            d = v["manifest"].replace("/work/", "").rsplit("/", 1)[0]
            for fn, h in json.loads(b)["files"].items():
                folders[fo]["files"][f"famexp/{Path(fn).name}"] = {"path": f"{d}/{Path(fn).name}", "sha256": h, "vol": "conv"}
            folders[fo]["sources"]["famexp"] = pin(c, p) | {"manifest_sha256": v["sha256"]}
    for c, p in RS_RECS:  # own_all_rs: manifest copies (ownall_rs_man), sha-checked against the record
        r = load(c, p)
        for v_ in r["results"].values():
            for fo, fi in v_["folders"].items():
                tgt = folders.get(fo) if fo in folders else fa if fo == fa["root"] else None
                if tgt is None:
                    continue
                b = (COPIES / "ownall_rs_man" / f"{fo}__own_all.json").read_bytes()
                if sha(b) != fi["manifest_sha256"] or json.loads(b).get("folder") != fo:
                    raise SystemExit(f"own_all_rs {fo}: manifest copy sha256/folder differs from {p}")
                base = f"scaleup/{fo}/own_all_rs" if fi["kind"] == "scaleup" else f"simh/{fa['batch']}/own_all_rs/{fo}"
                for fn, h in json.loads(b)["files"].items():
                    if fn.endswith(".npz"):
                        tgt["files"][f"own_all_rs/{fn}"] = {"path": f"{base}/{fn}", "sha256": h, "vol": "conv"}
                tgt["sources"]["own_all_rs"] = pin(c, p) | {"manifest_sha256": fi["manifest_sha256"], "kind": fi["kind"]}

    fams = ("sol", "p2", "famexp", "own_all_rs")
    cov = {f: {"folders_missing": sorted(fo for fo, v in folders.items() if f not in v["sources"]),
               "fill_in_missing": sorted(k for k, v in fill.items() if not any(x.startswith(f + "/") for x in v["files"]))} for f in fams}
    cov["own_a"] = {"folders_missing": sorted(folders), "fill_in_missing": sorted(fill), "note": "own_a is not wired for e12 (S_old only; withheld, labelled)"}
    cov["dt"] = {"folders_missing": sorted(folders), "fill_in_missing": sorted(fill), "note": "DT is not a headline member; withheld under e12, labelled"}
    miss_head = {f: cov[f] for f in HEADLINE_FAMILIES if cov[f]["folders_missing"] or cov[f]["fill_in_missing"]}
    if miss_head and not a.test:
        raise SystemExit(f"headline families without epoch-2 / fill-in files: {miss_head}")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    blk = {"schema": "e12_block/v1", "status": "TEST (missing headline files listed, values never read)" if a.test else "FINAL",
           "rule": pin(*RULE), "rulings": RULINGS, "caveat": CAVEAT, "artifact": pin(*ART), "verified_against": pin(*VER),
           "artifact_vs_verified_build": vs, "pools": pools, "folders": folders, "fill_in": fill,
           "coverage": {"by_family": cov, "headline_families": list(HEADLINE_FAMILIES), "missing_headline": miss_head,
                        "rule": "a missing headline family STOPs the job (no epoch-1 fallback: 09 withdrew it; 0d decides); others are withheld, labelled"},
           "identity_asserts": ident, "uncut_routing": uncut_routing(), "uncut_inventory": uncut_inventory(),
           "identity_sets": {"rule": ("507-prep 04:4x (Sol launcher via 2e): epoch-2 score files carry no epoch column and share world|cell keys with epoch 1; "
                                      "the epoch comes from metadata only (this block's ep2 folders); every epoch-2 run a reader uses must have its physical key "
                                      "(launch/eval#e0) in inventory_ep2_v1 and its .eval sha256 outside the epoch-1 stager-manifest shas"),
                             "inventory_ep2_keys": sorted({x["key"] for x in load(INVENTORY[0], INVENTORY[1])["runs"]}),
                             "inventory_pin": pin(INVENTORY[0], INVENTORY[1]), "epoch1_eval_sha256": e1_sha_list,
                             "epoch1_manifests_at": EPOCH1_MANIFESTS_AT},
           "counts": {o: {nm: (len(v["e2"]) if "e2" in v else {g: len(x["e2"]) for g, x in v.items() if isinstance(x, dict) and "e2" in x})
                          for nm, v in d.items() if isinstance(v, dict)} for o, d in pools.items()},
           "meta": {"stamp_pt": stamp, "git": git_head, "script_sha256": sha(Path(__file__).read_bytes()), "reads": "git and sha-checked manifest copies only; no score file"}}
    out = HERE / "config" / f"E12_BLOCK{'_TEST' if a.test else ''}_{stamp}.json"
    b = json.dumps(blk, indent=1, ensure_ascii=False).encode()
    with out.open("xb") as fh:
        fh.write(b)
    print(json.dumps({"file": str(out.relative_to(REPO)), "sha256": sha(b), "counts": blk["counts"], "missing_headline": miss_head,
                      "coverage": {f: {k: len(v) for k, v in x.items() if isinstance(v, list)} for f, x in cov.items()},
                      "identity": {k: v["pass"] for k, v in ident.items()}, "artifact_vs_verified": {k: v["equal"] for k, v in vs.items()}}, indent=1))


if __name__ == "__main__":
    main()
