"""LOO uncut ELIGIBILITY FREEZE (09 READ_LOG 10-16 04:51 "LOO4 eligibility freeze"; loo3u, plan approved by 4f 10-16 14:3x). Keys only.

Decides, by key and before any LOO value is read, whether own_all_LOO4_uncut_L48 is available (all four sets pass), else
own_all_LOO3_uncut_L48 (its three sets), else own_all_rs (availability order; never by value). Criteria (09):
- amendment 2 item 3: main-disjoint (the held-out set's mains are in none of model o's training mains), the dropped-main counts,
  in_fold_violations = 0;
- item 4: registered arms (run1, run2, sim of the TEST cohort); both honest pools fail closed; >= 30 eligible mains per set; no
  training launch of model o in set o's arms or pools (ep2 and fill-in included). A violation fails the probe CLOSED and is stated
  by key (the record says which launches and keys), never a silent withhold;
- l.4291: sim-arm launch disjointness;
- item 19: separate manifests (training-key identity: the sorted planned keys of every model hash to the fit record's
  training_keys_sha256), cut none, refit-match, certified, verify, and S_paper493 only via LOO4's model 3 (the o3 coverage NOTE).
Coverage vs the cohort of the FROZEN INPUTS_FULL 110447 (its exclusions pin: 1,313 cells; its inventories) per set x primary arm x
pool, with a reason per uncovered cell. NO batch filter (09 ruling, 10-16 ~16:1x): an S4 / S4r cell is decided by the registered
criteria like any other; an uncovered cell is withheld BY KEY with its real reason. Completeness: the 110447 coverage check
(analyze_loo3_coverage_110447.py, extended) must be audit_valid and coverage_complete for the chosen probe. E12 fill-in roots in
S_paper493 folders (g4) are S_old tau_sim pool members with their own item-4 result by key; 09's pre-ruling: PASS = the full S_old
tau_sim pool on both sides; FAIL = S_old tau_sim WITHOUT g4 on both sides, labelled (the probe stays available).

Pools are WHOLE and checked as a SUPERSET (every inventory run of the set's honest worlds, ep2 and fill-in included; for S_paper493
every honest run on its mains, in any folder): a superset that passes implies the registered pools pass; any member sharing a
training launch fails the pool closed. The rh1e4 relaunch is not a lead pool (09: alternate calibration only).

The laptop only reads committed files from git by commit:path (sha-checked) and launches; the logic runs in ONE Modal CPU function,
which also reads the score packs' manifests (keys and counts only; no score). Writes results/loo_uncut_eligibility_freeze_<stamp>.json.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run <this file>   (script committed and pushed first)
"""

import collections
import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE  # in the Modal container the file sits at /root
FAM = "experiments/2026-10-14_p2_family_expansion/results"
CUT = "experiments/2026-10-15_cut_sensitivity/results"
APP = "probe-shift_aswin-surya_sub-ownallloo3uncut_kill-2026-11-14"
SRC = {  # name: (commit, path, sha256 or None = recorded): every input, read from git and sha-checked inside the Modal function too
    "own_all_inputs": ("924ccd43b", f"{FAM}/own_all_inputs_2026-10-15.json", "26f9d8d1fe08dd2ef430a1bef8af50f336da47234fe364e1a94879bcb54491d1"),
    "t196_cells": ("4d1eec572", "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json", "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb"),
    "plus311_cells": ("0caadbfb5", "experiments/2026-10-14_q2_scaleup_cohort/results/scaleup_tasks_plus311_2026-10-14_125135.json", "69f1f75313c9610f7d16b18225c1166268c0b6e084859c641d0213f293896a8b"),
    "nm360_cells": ("ef171c813", "experiments/2026-10-14_q3_new_mains_mode/config/new_mains_cells_v1.json", "98bfb97e26e7ec580b08643a3a38a5e66a0f6b63333f4013aa57559383402e8e"),
    "sp493_facts": ("d800edc6a", "experiments/2026-10-15_q3_sp493/results/sp493_facts_2026-10-15_165730.json", "07caa527db89ec519314486d8c529c37430ca41d773a8bce26c041eb141b0a57"),
    "test": ("927fabb0d", "experiments/2026-10-15_fpr_sweep_507_prep/config/INPUTS_FULL_2026-10-16_110447.json", "d6923a75aa8f914cd3b2e92d8403f125dd095219997aebea8b1d6e4e2b67f77c"),  # the FROZEN 110447 (was TEST)
    "loo3_32k_eligibility": ("71fc9f650", "experiments/2026-10-15_own_all_loo/results/own_all_loo_eligibility_2026-10-15_232451.json", "9fde1eedad5ed0fb186fa1123f9e432ca44b394a6ba9eb3395a7491d89cd092b"),
    "loo4_fit": ("65d735320", f"{FAM}/own_all_loo4_uncut_fit_2026-10-16_040431.json", "8536985f3e347536a08967500a332c956d9558ba42ea52a333cdea8bd3ea1f2e"),
    "loo3_fit": ("42c8f6925", f"{FAM}/own_all_loo3_uncut_fit_2026-10-16_033003.json", "698df68ff4b235cf1f22edd871271d9d02fe1810ec3cc1f43c267ae15ec316d6"),
    "loo4_verify": ("7e3ac7df8", f"{FAM}/own_all_loo4_uncut_verify_2026-10-16_045504.json", None),
    "loo4_o3pool_verify": ("e422298cf", f"{FAM}/own_all_loo4_uncut_o3pool_verify_2026-10-16_101357.json", "8630c6d70998cd9c232311df8bf22e8aa215d5f9bb57e89a1a1f13733b679f49"),
    "loo3_verify": ("7e3ac7df8", f"{FAM}/own_all_loo3_uncut_verify_2026-10-16_045434.json", None),
    "coverage_check": ("fdab7980f", "experiments/2026-10-16_own_all_loo_uncut/results/loo3_coverage_110447_2026-10-16_165107.json",
                       "d37272f2d4544d3fc725ab506489e711e333b66b56ed3de939c68e4884221e22"),  # the post-fill 110447 coverage check (LOO3): completeness
    "o3_coverage": ("e422298cf", f"{FAM}/own_all_loo4_uncut_o3_coverage_NOTE_2026-10-16_101211.json", "de09b96137f1ffe074f64384a93c4b82ec0b9ec76aad3a497a001966eaf1bd9c"),
}
SCORES = {  # probe -> [(tree, git commit, record path)]: every committed score record of the probe (its packs' manifests are read in Modal)
    "own_all_LOO4_uncut_L48": [("own_all_loo4_uncut", "e422298cf", f"{FAM}/own_all_loo4_uncut_cutfull_score_2026-10-16_{s}.json") for s in ("041727", "044049", "044250", "044120", "044150")]
                              + [("own_all_loo4_uncut_o3pool", "e422298cf", f"{FAM}/own_all_loo4_uncut_o3pool_cutfull_score_2026-10-16_{s}.json") for s in ("095407", "095559", "101008")],
    "own_all_LOO3_uncut_L48": [("own_all_loo3_uncut", "e422298cf", f"{FAM}/own_all_loo3_uncut_cutfull_score_2026-10-16_{s}.json") for s in ("034101", "044020", "044219")]
                              + [("own_all_loo3_uncut", "fbc9fa13a", f"{FAM}/own_all_loo3_uncut_cutfull_score_2026-10-16_151910.json")]  # the coverage completion (re-freeze)
                              + [("own_all_loo3_uncut", "102157dc9", f"{FAM}/own_all_loo3_uncut_cutfull_score_2026-10-16_164700.json"),  # rh1e4 (coverage completion, 110447)
                                 ("own_all_loo3_uncut", "186105acf", f"{FAM}/own_all_loo3_uncut_cutfull_score_2026-10-16_164906.json")],  # g4 (E12 fill-in, o0)
}
FIRST_FREEZE = ("57b6671e7", "experiments/2026-10-16_own_all_loo_uncut/results/loo_uncut_eligibility_freeze_2026-10-16_152215.json",
                "0525d4f16f7275ec561b5422397c4cbc4083e6ccdafde7b9df0816fd2d85f3f7")  # this re-freeze supersedes the 15:22 re-freeze (which superseded 993b0a995's)
REASON_FIX = {"correction_to": "57b6671e7 (loo_uncut_eligibility_freeze_2026-10-16_152215.json)",
              "replaced": "S4 / S4r: out of the attack scope by rule (not in the deciding eligibility file @ 71fc9f650)",
              "by": "no batch filter (09 ruling, 10-16 ~16:1x); an uncovered cell's reason is the failing criterion, e.g. '<arm> run not scored by LOO3 (outside its scoring inventory v3)'"}
TEST_PINS = ("exclusions", "cohort", "e12")  # 110447's own pins: final cohort keys, the no-filter record (S_newmains batch_of_cell), the E12 block
TEST_INVENTORIES = ("S_old", "S_new", "S_newmains", "S_paper493", "S_paper493_w23", "pools_ep2", "pools_fill", "ha25f")

app = modal.App(APP)
WORK = modal.Volume.from_name("probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14", version=2)
img = modal.Image.debian_slim(python_version="3.12")


def _base(c):
    return re.sub(r"(_redraw[0-9]+|_r[0-9]+|_e2[a-z])$", "", c)


def _launch(k):
    return k.removeprefix("runs/").split("/", 1)[0]


@app.function(image=img, cpu=(4, 4), memory=(16384, 16384), timeout=1800, retries=0, scaledown_window=2, volumes={"/work": WORK})
def freeze(bundle_path: str, bundle_sha: str) -> dict:
    """All the logic (keys only). The bundle (uploaded once to the Volume; large function inputs stall Modal): files: name -> text;
    pins: name -> {commit, path, sha256}; scores: probe -> [(tree, name)]."""
    WORK.reload()
    bb = Path(bundle_path).read_bytes()
    if hashlib.sha256(bb).hexdigest() != bundle_sha:
        raise SystemExit("the uploaded bundle differs from its sha256")
    bundle = json.loads(bb)
    files, pins, scores = bundle["files"], bundle["pins"], bundle["scores"]
    sha = lambda t: hashlib.sha256(t.encode()).hexdigest()  # noqa: E731
    for n, p in pins.items():
        if sha(files[n]) != p["sha256"]:
            raise SystemExit(f"{n}: sha256 differs from its pin")
    J = {n: json.loads(t) for n, t in files.items()}
    sha_l = lambda xs: hashlib.sha256(json.dumps(xs).encode()).hexdigest()  # noqa: E731
    # ---- cells -> main / origin
    main_of, origin = {}, {}
    for n, s in (("t196_cells", "S_old"), ("plus311_cells", "S_new"), ("nm360_cells", "S_newmains")):
        for c in J[n]["cells"]:
            main_of[c["key"]], origin[c["key"]] = c["main"], s
    sp_cells = {r["key"]: r["main"] for r in J["sp493_facts"]["rows"] if r["key"] != "t987_model_registry"}
    if set(sp_cells) & set(main_of):
        raise SystemExit("S_paper493 cell keys collide with another set's")
    mainc = lambda c: main_of.get(c) or main_of.get(_base(c)) or sp_cells.get(c) or sp_cells.get(_base(c))  # noqa: E731
    P = set(sp_cells.values())
    # ---- TEST cohort (final cohort keys) and S_newmains attack scope
    ex = J["exclusions"]["final_cohort_keys"]
    cohort = {s: list(v["cells"]) for s, v in ex.items()}
    for s, v in ex.items():  # the exclusions record hashes the sorted list as compact JSON
        if hashlib.sha256(json.dumps(sorted(v["cells"]), separators=(",", ":")).encode()).hexdigest() != v["sha256"]:
            raise SystemExit(f"cohort {s}: cells do not hash to the recorded sha256")
    if sum(len(v) for v in cohort.values()) != 1313:
        raise SystemExit("the TEST cohort is not 1,313 cells")
    batch = J["cohort"]["origins"]["S_newmains"]["batch_of_cell"]
    fill_b = {v["batch"]: k for k, v in J["e12"]["fill_in"].items()}  # E12 fill-in roots, by batch (A25 simh_t196fill, g4 sp493_g4)
    # ---- runs by set and world, from the TEST file's inventories (keys only)
    runs = collections.defaultdict(list)  # (set, world) -> [(key, launch, cell, main, folder)]
    seen = {}
    for n in ("S_old", "S_new", "S_newmains", "S_paper493", "S_paper493_w23", "pools_ep2", "pools_fill", "ha25f"):
        for r in J[f"inv_{n}"]["runs"]:
            fo = r["uncut_planned"]["folder"]
            fk = next((k for b, k in fill_b.items() if fo.startswith(b + "_")), None) if fo.startswith("sp493_") else None
            fs = None
            if fk:  # an E12 fill-in root in an S_paper493 folder (g4): stays in the S_paper493 bucket (LOO4 unchanged); [8] = its cells' set, for LOO3 only
                fs = origin.get(_base(r["task"]))
                if fs not in ("S_old", "S_new", "S_newmains") or r["world"] != "sim_honest":
                    raise SystemExit(f"{n} {r['key']}: E12 fill-in {fk} run is not a sim-honest run on a three-set cell")
            st = ("S_paper493" if fo.startswith("sp493_") else "S_old" if n == "S_old" and r.get("task_set") == "T196" else
                  "S_newmains" if n == "S_newmains" and r.get("task_set") == "new mains" else "S_new" if n in ("S_new", "ha25f") else
                  {"pools_ep2": {"ep2_t196": "S_old", "ep2_311": "S_new", "ep2_nm": "S_newmains"}.get(re.sub(r"_l[0-9]+$", "", fo.split("_cutfull")[0]), "?"),
                   "pools_fill": "S_old"}.get(n))
            if st is None:
                continue
            if st == "?":
                raise SystemExit(f"{n} {fo}: no set")
            m = mainc(r["task"])
            if m is None:
                raise SystemExit(f"{n} {r['key']}: cell {r['task']} has no main")
            row = (r["key"], _launch(r["key"]), r["task"], m, fo, r["world"], st, fk, fs)  # [7], [8]: an E12 fill-in root in an S_paper493 folder (g4) and its cells' set
            if r["key"] in seen:
                if seen[r["key"]] != row:
                    raise SystemExit(f"{r['key']}: two different inventory rows")
                continue
            seen[r["key"]] = row
            runs[(st, r["world"])].append(row)
    # ---- shared gates (item 19 / item 3)
    gates = {"loo4_o3_coverage_pass": J["o3_coverage"].get("pass") is True, "loo4_verify": J["loo4_verify"].get("pass") is True,
             "loo4_o3pool_verify": J["loo4_o3pool_verify"].get("pass") is True, "loo3_verify": J["loo3_verify"].get("pass") is True}
    # ---- the score packs' run coverage (manifests; counts only)
    present, present2, completion, ifv = {}, {}, {}, {}
    for probe, recs in scores.items():
        pr = set()
        for tree, n in recs:
            cc = J[n].get("skipped_routing_records", {}).get("coverage_completion")
            if cc:  # 0d: keys scored after the first freeze pass, as completion of the frozen pinned pools
                completion.setdefault(probe, []).append({"record": pins[n], "keys": cc["keys"], "label": "scored after the first freeze pass (coverage completion)",
                                                         **{k_: cc[k_] for k_ in ("in_test_pinned_ha25f_inventory", "folders_in_companion", "in_110447_pinned_inventories",
                                                                                  "folders_in_110447_lead_overrides", "pins") if k_ in cc}})
            ifv[probe] = ifv.get(probe, 0) + sum(x["in_fold_violations"] for r_ in J[n]["results"].values() for x in r_["folders"].values())
            for r_ in J[n]["results"].values():
                for fo, x in r_["folders"].items():
                    mp = Path("/work/cutfull") / x["batch"] / tree / fo / "manifest_own_all.json"
                    mb = mp.read_bytes()
                    if hashlib.sha256(mb).hexdigest() != x["manifest_sha256"]:
                        raise SystemExit(f"{tree}/{fo}: manifest sha256 differs from its score record")
                    man = json.loads(mb)
                    for rd in ("R", "C"):
                        pr |= {(fo, u.split("|", 1)[1], u.split("|", 1)[0], tree) for u in list(man["counts"][rd]["per_run"]) + list(man["counts"][rd]["runs_without_rows"])}
        present[probe] = pr
        present2[probe] = {t: {(fo, c) for fo, c, _, t_ in pr if t_ == t} for t in {x[3] for x in pr}}
    out = {"probes": {}}
    sets_all = ("S_old", "S_new", "S_newmains", "S_paper493")
    for probe, fitn in (("own_all_LOO4_uncut_L48", "loo4_fit"), ("own_all_LOO3_uncut_L48", "loo3_fit")):
        fr = J[fitn]
        lu, fit = fr["loo_uncut"], fr["fit"]
        K, drop = lu["models"], lu["drop_mains"]
        k5 = J["own_all_inputs"]["folds"]["by_main"]
        nopol = set(lu["no_policy_call_refs"])
        rows_all = [(e["key"], main_of[e["cell"]], ("S_old", "S_new", "S_newmains").index(origin[e["cell"]])) for e in J["own_all_inputs"]["training_epochs"]
                    if e["world"] in ("run1", "honest_a") and main_of[e["cell"]] in k5 and e["key"] not in nopol]
        if K == 4:
            rows_all += [(k, sp_cells[_base(k.removeprefix("runs/").split("/")[2])], 3) for k in lu["sp493"]["training_keys_eval_sha256"]]
        T, tm, ident = {}, {}, {}
        for o in range(K):
            ds = set(drop[str(o)])
            keys = sorted(k for k, m, og in rows_all if og != o and m not in ds)
            mains = sorted({m for k, m, og in rows_all if og != o and m not in ds})
            ident[str(o)] = {"keys": len(keys), "keys_sha256_equal": sha_l(keys) == lu["training_keys_sha256"][str(o)], "mains": len(mains),
                             "mains_sha256_equal": sha_l(mains) == lu["training_mains_sha256"][str(o)]}
            if not (ident[str(o)]["keys_sha256_equal"] and ident[str(o)]["mains_sha256_equal"]):
                raise SystemExit(f"STOP: {probe} model o{o}: the rebuilt training keys / mains differ from the fit record's identity")
            T[o], tm[o] = {_launch(k) for k in keys}, set(mains)
        g = fit["gate"]
        pgates = {"fit_pass": fr.get("pass") is True, "refit_match_pass": g["refit_match_pass"], "all_certified": g["all_certified"], "cut_none": fit.get("cut") == "none",
                  "in_fold_violations_0": fit.get("in_fold_violations") == 0, "dropped_mains": fit["dropped_mains"],
                  "verify": gates["loo4_verify"] and gates["loo4_o3pool_verify"] if K == 4 else gates["loo3_verify"],
                  "s_paper493_only_via_model3": gates["loo4_o3_coverage_pass"] if K == 4 else "n/a (S_paper493 excluded by design)", "training_identity": ident,
                  "score_in_fold_violations_0": ifv.get(probe, -1) == 0}
        if K == 3:  # 0d / 09 (10-16): completeness against the 110447 pins, from the post-fill coverage check (LOO3's)
            ck = J["coverage_check"]
            pgates["coverage_110447_complete"] = (ck.get("audit_valid") is True and ck.get("coverage_complete") is True and ck.get("stop_unexpected_gap") is False
                                                  and ck["sources"]["full"]["sha256"] == pins["test"]["sha256"])
        gates_ok = all(v is True for k, v in pgates.items() if k not in ("dropped_mains", "training_identity", "s_paper493_only_via_model3")) and \
            (pgates["s_paper493_only_via_model3"] is True or K == 3)
        per_set, ok_all = {}, gates_ok
        for o, s in enumerate(sets_all[:K]):
            arm = {w: [r for r in runs[(s, w)]] for w in ("run1", "run2", "sim")}
            if s == "S_paper493":
                pool_rows = [r for (s_, w), rs in runs.items() for r in rs if w in ("honest_a", "honest_b", "sim_honest") and r[3] in P]
            else:
                pool_rows = [r for w in ("honest_a", "honest_b", "honest_c", "sim_honest") for r in runs[(s, w)]]
            fills = {}
            fill_rows = [r for rs in runs.values() for r in rs if r[7] and r[8] == s] if K == 3 else []  # Codex r37: LOO3 only (its o0 scored g4); LOO4 unchanged
            for fk_ in sorted({r[7] for r in fill_rows}):  # 09 pre-ruling (10-16): g4's item 4 by key; PASS = full S_old tau_sim, FAIL = without g4 on both sides
                fm = [r for r in fill_rows if r[7] == fk_]
                sh = sorted(r[0] for r in fm if r[1] in T[o])
                un = sorted(r[0] for r in fm if not any((r[4], r[2]) in present2[probe][t] for t in present2[probe]))
                fills[fk_] = {"members": len(fm), "keys": sorted(r[0] for r in fm), "launches": sorted({r[1] for r in fm}), "training_launches_of_model": f"o{o}",
                              "item4_sharing_training_launch": sh, "item4_pass": not sh, "members_not_in_the_probe_packs": un,
                              "branch": ("PASS: the full S_old tau_sim pool (with this fill-in) on both sides" if not sh else
                                         "FAIL: S_old tau_sim WITHOUT this fill-in on BOTH sides, labelled, with the lead's full-pool value printed beside it (09 pre-ruling); the probe stays available")}
            pools = {}
            for w in ("honest_a", "honest_b", "honest_c", "sim_honest"):
                mem = [r for r in pool_rows if r[5] == w]
                if not mem:
                    continue
                bad = sorted(r[0] for r in mem if r[1] in T[o])
                unscored = sorted(r[0] for r in mem if not any((r[4], r[2]) in present2[probe][t] for t in present2[probe]))  # Codex r18: every member scored
                pools[w] = {"members": len(mem), "launches": sorted({r[1] for r in mem}), "sharing_training_launch": len(bad), "examples": bad[:5],
                            "members_not_in_the_probe_packs": unscored, "pass": not bad and not unscored}
            hit = collections.defaultdict(set)
            for w, rs in list(arm.items()) + [(f"pool:{w}", [r for r in pool_rows if r[5] == w]) for w in ("honest_a", "honest_b", "honest_c", "sim_honest")]:
                for r in rs:
                    if r[1] in T[o]:
                        hit[w].add(r[1])
            item4 = {"training_launches_of_model": sorted(T[o]), "arm_and_pool_launches_in_training": {w: sorted(v) for w, v in hit.items()},
                     "set_mains_in_training_mains": len(set(m for m in tm[o]) & {mainc(c) for c in cohort[s]}), "pass": not hit}
            item4["pass"] = item4["pass"] and item4["set_mains_in_training_mains"] == 0
            byc = {w: collections.defaultdict(list) for w in arm}
            for w, rs in arm.items():
                for r in rs:
                    byc[w][_base(r[2])].append(r)  # run-2 / redraw cells join their base cell
            cov, reasons, elig = collections.Counter(), {}, []
            treeset = {"own_all_loo4_uncut", "own_all_loo3_uncut"}
            for c in cohort[s]:  # 09 ruling (10-16): no batch filter
                cov["in_scope"] += 1
                why = []
                for w in ("run1", "run2", "sim"):
                    rs = byc[w].get(c, [])
                    if not rs:
                        why.append(f"no {w} run in the TEST inventories")
                        continue
                    cov[f"{w}_run"] += 1
                    if any(r[1] in T[o] for r in rs):
                        why.append(f"{w} shares a training launch of model o{o}")
                        continue
                    cov[f"{w}_launch_disjoint"] += 1
                    if not all(any((r[4], r[2], w, t) in present[probe] for t in treeset) for r in rs):
                        why.append(f"{w} run not scored by {probe.split('_')[2]} (outside its scoring inventory v3)")  # 09's reason string (correction to 57b6671e7)
                        continue
                    cov[f"{w}_scored"] += 1
                if mainc(c) in tm[o]:
                    why.append(f"main in model o{o}'s training mains")
                if why:
                    reasons[c] = "; ".join(why)
                else:
                    elig.append(c)
            n_mains = len({mainc(c) for c in elig})
            pools_ok = bool(pools) and all(v["pass"] for v in pools.values())
            fills_ok = not any(v["members_not_in_the_probe_packs"] for v in fills.values())  # a fill-in must be scored; its item 4 only picks the branch
            set_ok = item4["pass"] and pools_ok and fills_ok and n_mains >= 30
            pending = item4["pass"] and n_mains >= 30 and not (pools_ok and fills_ok) and all(v["sharing_training_launch"] == 0 for v in pools.values())  # only unscored members
            ok_all = ok_all and set_ok
            per_set[s] = {"scoring_model": f"o{o} (held out: {s})", "cohort_cells": len(cohort[s]), "coverage": dict(cov), "eligible_cells": sorted(elig),
                          "eligible_mains": n_mains, "pools_superset": pools, "pools_pass": pools_ok, "e12_fill_in_pools": fills, "item4_assert": item4,
                          "uncovered_cells": reasons, "pass": set_ok, "status": "pass" if set_ok else "INCOMPLETE: pool members not yet scored (listed by key)" if pending else "FAIL (closed)",
                          "fail_reasons": [x for x, b in (("item-4 assert: a training launch of the model in the set's arms or pools", not item4["pass"]),
                                                          ("a pool shares a training launch, or has members the probe has not scored (fail closed)", not pools_ok),
                                                          ("an E12 fill-in pool has members the probe has not scored", not fills_ok),
                                                          ("fewer than 30 eligible mains", n_mains < 30)) if b]}
        structural_fail = any(v["status"] == "FAIL (closed)" for v in per_set.values()) or not gates_ok
        out["probes"][probe] = {"gates": pgates, "gates_pass": gates_ok, "sets": per_set, "available": ok_all,
                                "status": "AVAILABLE" if ok_all else "UNAVAILABLE (a criterion fails closed)" if structural_fail else "INCOMPLETE (only unscored pool members; re-freeze after they are scored)"}
    # ---- the non-S4 S_newmains cells outside the 32k LOO3 deciding eligibility file
    e32 = set(J["loo3_32k_eligibility"]["sets"]["own_all_LOO_L48"]["S_newmains"]["eligible_and_scored_cells"])
    extra = sorted(c for c in cohort["S_newmains"] if batch.get(c) in ("S1", "S2", "S3", "S5") and c not in e32)  # informational (the 32k file's non-S4 gap)
    out["s_newmains_non_s4_outside_71fc9f650"] = {"cells": extra, "n": len(extra),
                                                  "status_under": {p: {c: ("eligible" if c in v["sets"]["S_newmains"]["eligible_cells"] else v["sets"]["S_newmains"]["uncovered_cells"].get(c, "?"))
                                                                       for c in extra} for p, v in out["probes"].items()}}
    a4, a3 = out["probes"]["own_all_LOO4_uncut_L48"]["available"], out["probes"]["own_all_LOO3_uncut_L48"]["available"]
    st4, st3 = out["probes"]["own_all_LOO4_uncut_L48"]["status"], out["probes"]["own_all_LOO3_uncut_L48"]["status"]
    ch = "own_all_LOO4_uncut_L48" if a4 else "own_all_LOO3_uncut_L48" if a3 else None
    out["comparison_cells"] = ({"probe": ch, "cells": {s: v["eligible_cells"] for s, v in out["probes"][ch]["sets"].items()},
                                "n": {s: len(v["eligible_cells"]) for s, v in out["probes"][ch]["sets"].items()},
                                "rule": "09 (10-16 ~15:1x): any LOO-vs-own_all_rs comparison uses this IDENTICAL cell list on both sides; 09 ruling (10-16 ~16:1x): no batch filter",
                                "e12_fill_in_tau_sim": {s: {fk: {"item4_pass": f["item4_pass"], "branch": f["branch"]} for fk, f in v["e12_fill_in_pools"].items()}
                                                        for s, v in out["probes"][ch]["sets"].items() if v.get("e12_fill_in_pools")}}
                               if ch else None)
    out["coverage_completion"] = completion
    out["reason_string_correction"] = REASON_FIX
    ck = J["coverage_check"]  # 09's checklist (10-16): the completeness citation and g4's branch, stated at the top level
    out["completeness_110447"] = {"record": pins["coverage_check"], "audit_valid": ck.get("audit_valid"), "coverage_complete": ck.get("coverage_complete"),
                                  "stop_unexpected_gap": ck.get("stop_unexpected_gap"), "read_the_110447_pins": ck["sources"]["full"]["sha256"] == pins["test"]["sha256"],
                                  "asserted_for": "own_all_LOO3_uncut_L48 (gate coverage_110447_complete)"}
    g4s = out["probes"]["own_all_LOO3_uncut_L48"]["sets"]["S_old"]["e12_fill_in_pools"].get("g4")
    out["g4_branch"] = ({"item4_pass": g4s["item4_pass"], "item4_sharing_training_launch": g4s["item4_sharing_training_launch"], "g4_launches": g4s["launches"],
                         "against": "model o0's training launches (LOO3, S_old held out)", "branch": g4s["branch"]} if g4s else "STOP: no g4 fill-in pool under LOO3's S_old")
    out["decision"] = {"chosen": "own_all_LOO4_uncut_L48" if a4 else ("PENDING: LOO4 incomplete" if st4.startswith("INCOMPLETE") else
                                                                     "own_all_LOO3_uncut_L48" if a3 else "PENDING: LOO3 incomplete" if st3.startswith("INCOMPLETE") else
                                                                     "own_all_rs (neither LOO probe is available)"),
                       "loo4_status": st4, "loo3_status": st3,
                       "order": "LOO4 iff all four sets pass; else LOO3 (its three sets); else own_all_rs (09 READ_LOG 04:51); by availability, never by value",
                       "loo4_available": a4, "loo3_available": a3}
    return out


@app.local_entrypoint()
def main():
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, check=True).stdout  # noqa: E731
    me = str(Path(__file__).resolve().relative_to(REPO))
    if git("status", "--porcelain", "--untracked-files=no").strip() or git("rev-parse", "HEAD").strip() != git("rev-parse", "origin/aswin/own-all-loo3-uncut-4f").strip():
        raise SystemExit("commit and push the script first (HEAD must be on origin, tracked tree clean)")
    src = dict(SRC)
    test = json.loads(git("show", f"{SRC['test'][0]}:{SRC['test'][1]}"))
    for n in TEST_PINS:
        src[n] = (test[n]["commit"], test[n]["path"], test[n]["sha256"])
    for n in TEST_INVENTORIES:
        src[f"inv_{n}"] = (test["inventories"][n]["commit"], test["inventories"][n]["path"], test["inventories"][n]["sha256"])
    if hashlib.sha256(git("show", f"{FIRST_FREEZE[0]}:{FIRST_FREEZE[1]}").encode()).hexdigest() != FIRST_FREEZE[2]:
        raise SystemExit("the first freeze record differs from its pin")
    files, pins = {}, {}
    for n, (c, p, h) in src.items():
        t = git("show", f"{c}:{p}")
        hh = hashlib.sha256(t.encode()).hexdigest()
        if h and hh != h:
            raise SystemExit(f"{n}: sha256 {hh} != pinned {h}")
        files[n], pins[n] = t, {"commit": c, "path": p, "sha256": hh}
    scores = {}
    for probe, recs in SCORES.items():
        scores[probe] = []
        for i, (tree, c, p) in enumerate(recs):
            n = f"score_{probe}_{i}"
            t = git("show", f"{c}:{p}")
            files[n], pins[n] = t, {"commit": c, "path": p, "sha256": hashlib.sha256(t.encode()).hexdigest()}
            scores[probe].append((tree, n))
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    bb = json.dumps({"files": files, "pins": pins, "scores": scores}, sort_keys=True).encode()
    bsha, rp = hashlib.sha256(bb).hexdigest(), f"/loo3u_freeze/{stamp}/bundle.json"
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json") as tf:
        tf.write(bb)
        tf.flush()
        with WORK.batch_upload() as up:  # create-only (a fresh stamped path)
            up.put_file(tf.name, rp)
    print(json.dumps({"files": len(files), "MB": round(len(bb) / 1e6, 1), "bundle": rp, "sha256": bsha}))
    res = freeze.remote("/work" + rp, bsha)
    rec = {"label": "LOO uncut ELIGIBILITY RE-FREEZE against the FROZEN INPUTS_FULL 110447 (09 READ_LOG 10-16 04:51 criteria; 09's no-batch-filter ruling; keys only; "
                    "committed before any LOO value is read; PROVISIONAL until 09 signs)",
           "stamp (PT)": stamp, "generator": {"path": me, "commit": git("rev-parse", "HEAD").strip(), "app": APP, "app_id": app.app_id or ""},
           "criteria": __doc__.split("\n\n")[1], "sources": pins, "bundle": {"volume_path": rp, "sha256": bsha},
           "supersedes": {"commit": FIRST_FREEZE[0], "path": FIRST_FREEZE[1], "sha256": FIRST_FREEZE[2]}} | res
    p = HERE / "results" / f"loo_uncut_eligibility_freeze_{stamp}.json"
    p.parent.mkdir(exist_ok=True)
    p.open("x").write(json.dumps(rec, indent=1))
    print(p)
    print(json.dumps({"decision": res["decision"], "sets": {pr: {s: (v["pass"], v["eligible_mains"], v["fail_reasons"]) for s, v in x["sets"].items()} for pr, x in res["probes"].items()},
                      "gates": {pr: x["gates_pass"] for pr, x in res["probes"].items()}, "s_newmains_extra": res["s_newmains_non_s4_outside_71fc9f650"]["n"]}))
