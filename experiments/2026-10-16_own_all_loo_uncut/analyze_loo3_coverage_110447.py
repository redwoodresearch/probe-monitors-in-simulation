"""LOO3 uncut COVERAGE CHECK against the FROZEN INPUTS_FULL 110447 pins (4f / 0d / 09 / 2e, 10-16 ~15:3x). Keys only: the npz
files' run_key / unit / call_index columns and the manifests; no score column is read.

Lead = own_all_rs_uncut_L48: every file the locked 110447 overrides list for it (S_old, S_new, S_newmains, pools_ep2, S_new+25f
and S_paper493; R and C). Twin = the same Volume folder under own_all_loo3_uncut/, file own_all_LOO3_uncut_L48__<R|C>.npz.
EXTENDED (0d / 09 via 4f, 10-16 ~16:2x): a lead folder of the S_paper493 override is REQUIRED (it feeds an S_old / S_new / S_newmains
row) when it is an E12 fill-in root (110447's E12 block, matched by batch) or its rows' cells are cohort cells of those sets; the
other S_paper493 folders are listed as S_paper493-only (LOO3 excluded). The scope is listed BY KEY per set: every lead folder with
its role, override and file path@sha; the E12 pools and fill-in roots. Member checks: every E12 e2 entry (by inventory key) and
every E12 fill-in admitted world|cell has lead units, and LOO3 units equal to the lead's.
1. FOLDER (2e): each lead file has a LOO3 twin whose manifest is in a committed LOO3 score record and whose sha256 matches it.
2. RUN (2e): per folder and readout, LOO3's unit identities (run_key, unit, call_index) equal the lead's, run by run.
3. PINNED RUNS (09): each run of the 110447 pinned inventories routed (uncut_planned.folder) to a lead folder has units in the
   lead file, or its world|cell is in the lead manifest's runs_without_rows. POSITIVE CONTROL: 0 runs fail this for the lead.
List (a): runs with lead units but no equal LOO3 units, by set x pool x arm, by key. List (b): the cells the re-freeze @ 57b6671e7
withheld (its uncovered_cells), each with its reason and whether every arm run (run1, run2, sim) is LOO3-scored. Also: the
skipped / REFUSED routing records of the LOO3 score records against the lead folders, and the rh1e4 runs against the 958
relaunch members. Problems (a lead file or manifest that differs from its pin, a twin outside the score records) fail the check.

The laptop reads committed files from git by commit:path (sha-checked against the 110447 pins) and launches; the logic runs in
ONE Modal CPU function. Writes results/loo3_coverage_110447_<stamp>.json.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run <this file>   (script committed and pushed first)
"""

import collections
import hashlib
import io
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE  # in the Modal container the file sits at /root
FAM = "experiments/2026-10-14_p2_family_expansion/results"
APP = "probe-shift_aswin-surya_sub-ownallloo3uncut_kill-2026-11-14"
FULL = ("927fabb0d", "experiments/2026-10-15_fpr_sweep_507_prep/config/INPUTS_FULL_2026-10-16_110447.json",
        "d6923a75aa8f914cd3b2e92d8403f125dd095219997aebea8b1d6e4e2b67f77c")  # the FROZEN INPUTS_FULL
OVERRIDES = ("S_old", "S_new", "S_newmains", "pools_ep2", "S_new+25f", "S_paper493")  # x (R, C); S_paper493's folders: REQUIRED or S_paper493-only
INVENTORIES = ("S_old", "S_new", "S_newmains", "pools_ep2", "pools_fill", "ha25f", "il", "S_paper493", "S_paper493_w23")  # all of 110447's pinned inventories
THREE = ("S_old", "S_new", "S_newmains")
APPROVED_GAPS = ("rh1e4_stage_cutfull_v1__honest_b", "sp493_g4_simh_cutfull_v1__sim_honest", "su311_q1_main_ha25f_cutfull_v1__honest_a",
                 "su311_q2_main_ha25f_cutfull_v1__honest_a", "su311_q4_main_ha25f_cutfull_v1__honest_a")  # 0d (via 4f, 10-16): rh1e4 + g4 + the known ha25f; else STOP
LOO3_RECORDS = [("e422298cf", f"{FAM}/own_all_loo3_uncut_cutfull_score_2026-10-16_{s}.json") for s in ("034101", "044020", "044219")] + \
               [("fbc9fa13a", f"{FAM}/own_all_loo3_uncut_cutfull_score_2026-10-16_151910.json")] + \
               [("102157dc9", f"{FAM}/own_all_loo3_uncut_cutfull_score_2026-10-16_164700.json"), ("186105acf", f"{FAM}/own_all_loo3_uncut_cutfull_score_2026-10-16_164906.json")]  # the rh1e4 and g4 fills
REFREEZE = ("57b6671e7", "experiments/2026-10-16_own_all_loo_uncut/results/loo_uncut_eligibility_freeze_2026-10-16_152215.json",
            "0525d4f16f7275ec561b5422397c4cbc4083e6ccdafde7b9df0816fd2d85f3f7")
LEAD, TWIN, LEAD_VAR, TWIN_VAR = "own_all_rs_uncut", "own_all_loo3_uncut", "own_all_rs_uncut_L48", "own_all_LOO3_uncut_L48"
PROBE = "own_all_LOO3_uncut_L48"

app = modal.App(APP)
WORK = modal.Volume.from_name("probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14", version=2)
img = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _base(c):
    while (b := re.sub(r"(_redraw[0-9]+|_r[0-9]+|_e2[a-z])$", "", c)) != c:
        c = b
    return c


def _setof(fo):
    return ("S_old" if re.match(r"^(q4f_|rh1e4_|simh_q4f_|ep2_t196_|simh_t196fill_)", fo) else "S_new" if re.match(r"^(su311_|simh311_|ep2_311_)", fo)
            else "S_newmains" if re.match(r"^(nm360_|simh_nm360_|ep2_nm_)", fo) else None)


def _ident(r):
    return r["uncut_planned"]["folder"], r["world"], r["task"], r.get("eval_sha256")


def _root(fo):
    return fo.split("_v1__", 1)[0]


@app.function(image=img, cpu=(8, 8), memory=(32768, 32768), timeout=2400, retries=0, scaledown_window=2, volumes={"/work": WORK})
def check(bundle_path: str, bundle_sha: str) -> dict:
    """All the logic (keys only). The bundle (uploaded once to the Volume): files: name -> text; pins: name -> {commit, path, sha256}."""
    import numpy as np

    WORK.reload()
    bb = Path(bundle_path).read_bytes()
    if _sha(bb) != bundle_sha:
        raise SystemExit("the uploaded bundle differs from its sha256")
    B = json.loads(bb)
    for n, p in B["pins"].items():
        if _sha(B["files"][n].encode()) != p["sha256"]:
            raise SystemExit(f"{n}: sha256 differs from its pin")
    J = {n: json.loads(t) for n, t in B["files"].items()}
    problems = []
    # ---- 1. the lead's files, from the locked overrides
    jobs, lists_ov = {}, collections.defaultdict(set)  # (batch, folder, readout) -> (lead sha256, lead manifest sha256, override, kind, registered name, path)
    for ov in OVERRIDES:
        for rd in ("R", "C"):
            o, nm = J[f"ov_{ov}|{rd}"], f"{LEAD}/{LEAD_VAR}__{rd}.npz"
            if o["readout"] != rd:
                raise SystemExit(f"override {ov}|{rd}: readout is {o['readout']}")
            ents = [("arm", a, d[nm]) for a, d in o["arms"].items() if nm in d] + [("simh", "", e) for e in (lambda x: x if isinstance(x, list) else [x] if x else [])(o["simh"].get(nm))]
            problems += [f"override {ov}|{rd}: arm {a} lists no {nm}" for a, d in o["arms"].items() if nm not in d]
            for kind, reg, e in ents:
                pt = e["path"].split("/")
                if len(pt) != 5 or pt[0] != "cutfull" or pt[2] != LEAD or pt[4] != f"{LEAD_VAR}__{rd}.npz" or e["manifest"]["path"] != "/".join(pt[:4] + ["manifest_own_all.json"]):
                    raise SystemExit(f"override {ov}|{rd}: unexpected lead path {e['path']}")
                k = (pt[1], pt[3], rd)
                if k in jobs and jobs[k][:2] != (e["sha256"], e["manifest"]["sha256"]):
                    raise SystemExit(f"{k}: listed twice with different pins")
                jobs[k] = (e["sha256"], e["manifest"]["sha256"], ov, kind, reg, e["path"])
                lists_ov[k[:2]].add(ov)
    if any(_setof(k[1]) is None for k in jobs if lists_ov[k[:2]] != {"S_paper493"}):
        raise SystemExit("lead folders outside the S_paper493 override with no set")
    cohort = {s: set(v["cells"]) for s, v in J["exclusions"]["final_cohort_keys"].items()}
    cell2set = {c: s_ for s_ in THREE for c in cohort[s_]}
    # ---- the LOO3 score records: folder -> manifest sha256
    recm = {}
    for i in range(len(LOO3_RECORDS)):
        for r_ in J[f"loo3_{i}"]["results"].values():
            for fo, x in r_["folders"].items():
                if (x["batch"], fo) in recm:
                    problems.append(f"{x['batch']}/{fo}: in two LOO3 score records")
                recm[(x["batch"], fo)] = (x["manifest_sha256"], B["pins"][f"loo3_{i}"]["path"])

    def units(b):
        z = np.load(io.BytesIO(b))
        if not {"key", "run_key", "unit", "call_index"} <= set(z.files):
            raise ValueError("no key / run_key / unit / call_index columns")
        u, n, km = collections.defaultdict(set), 0, collections.defaultdict(set)
        for a, b_, c, k_ in zip(z["run_key"].tolist(), z["unit"].tolist(), z["call_index"].tolist(), z["key"].tolist(), strict=True):
            u[str(a)].add((int(b_), int(c)))
            km[str(k_)].add(str(a))
            n += 1
        return u, {x.split("|", 1)[0] for x in km}, n - sum(map(len, u.values())), km  # units, world labels, duplicate identities, world|cell -> run keys

    def one(k):
        batch, fo, rd = k
        lsha, lmsha = jobs[k][:2]
        out, prob = {"lead_rows": 0, "loo3_rows": 0, "twin": "present", "lead_runs_without_rows": []}, []
        lp, tp = Path("/work/cutfull") / batch / LEAD / fo, Path("/work/cutfull") / batch / TWIN / fo
        lb, lm = (lp / f"{LEAD_VAR}__{rd}.npz").read_bytes(), (lp / "manifest_own_all.json").read_bytes()
        if _sha(lb) != lsha or _sha(lm) != lmsha or json.loads(lm)["files"].get(f"{LEAD_VAR}__{rd}.npz") != lsha:
            return k, out, [f"{batch}/{fo} {rd}: the lead file or manifest differs from its 110447 pin"], {}, {}, {}
        lead, lw, out["lead_duplicate_identities"], km = units(lb)
        out["lead_three_set_cells"] = dict(collections.Counter(cell2set[_base(x.split("|", 1)[1])] for x in km if _base(x.split("|", 1)[1]) in cell2set))
        out["lead_runs_without_rows"] = json.loads(lm)["counts"][rd]["runs_without_rows"]
        out["lead_world_labels"] = sorted(lw | {x.split("|", 1)[0] for x in out["lead_runs_without_rows"]})
        loo = {}
        tf = tp / f"{TWIN_VAR}__{rd}.npz"
        if not tf.exists():
            out["twin"] = "missing"
        else:
            tm = (tp / "manifest_own_all.json").read_bytes()
            rec = recm.get((batch, fo))
            tb = tf.read_bytes()
            if rec is None or _sha(tm) != rec[0]:
                prob.append(f"{batch}/{fo}: a LOO3 twin whose manifest is in no committed LOO3 score record")
                out["twin"] = "unrecorded"
            elif _sha(tb) != json.loads(tm)["files"].get(f"{TWIN_VAR}__{rd}.npz"):
                prob.append(f"{batch}/{fo} {rd}: the LOO3 file differs from its manifest")
                out["twin"] = "sha mismatch"
            else:
                loo, _, out["loo3_duplicate_identities"], tkm = units(tb)
                lk_, tk_ = {x: frozenset(v) for x, v in km.items()}, {x: frozenset(v) for x, v in tkm.items()}
                out["key_column_equal"] = lk_ == tk_  # the world|cell -> run keys map; a runner matching by world|cell needs it equal (a world-label alias shows here)
                if not out["key_column_equal"]:
                    out["key_column_world_labels"] = {"lead": sorted({x.split("|", 1)[0] for x in lk_}), "loo3": sorted({x.split("|", 1)[0] for x in tk_})}
                out["loo3_runs_without_rows"], out["loo3_score_record"] = json.loads(tm)["counts"][rd]["runs_without_rows"], rec[1]
        out["lead_rows"], out["loo3_rows"] = sum(map(len, lead.values())), sum(map(len, loo.values()))
        return k, out, prob, lead, loo, km

    folders, lead_rk, ok_rk, loo_rk, why, lead_km = {}, {}, {}, {}, {}, {}
    with ThreadPoolExecutor(16) as ex:
        for k, out, prob, lead, loo, km in ex.map(one, sorted(jobs)):
            lead_km[k] = km
            problems += prob
            ok = {r for r in lead if loo.get(r) == lead[r]} if out["twin"] == "present" else set()
            lead_rk[k], ok_rk[k], loo_rk[k] = set(lead), ok, set(loo)
            for r in set(lead) - ok:
                why[(k, r)] = ("folder: no LOO3 file" if out["twin"] == "missing" else f"folder: LOO3 twin {out['twin']}" if out["twin"] != "present"
                               else "run: no LOO3 unit" if r not in loo else "run: LOO3 units differ from the lead's")
            out |= {"lead_runs": len(lead), "loo3_runs": len(loo), "runs_not_equal": len(set(lead) - ok), "loo3_extra_runs": sorted(set(loo) - set(lead))[:20],
                    "n_loo3_extra_runs": len(set(loo) - set(lead)), "override": jobs[k][2], "kind": jobs[k][3], "registered": jobs[k][4]}
            folders[f"{k[0]}/{k[1]}|{k[2]}"] = out
    # ---- EXTENDED: which lead folders feed an S_old / S_new / S_newmains row (E12 fill-in roots by batch; S_paper493-override folders by cohort cells)
    e12 = J["e12"]
    lead_fo = {(b, fo) for b, fo, _ in jobs}
    fill = {}
    for fk, fv in e12["fill_in"].items():
        hits = sorted((b, fo) for b, fo in lead_fo if b.replace("-", "_") == fv["batch"])
        if len(hits) != 1:
            problems.append(f"E12 fill-in {fk} (batch {fv['batch']}): {len(hits)} lead folders, need exactly one")
        fill |= dict.fromkeys(hits, fk)
    fset = {}
    for b, fo in sorted(lead_fo):
        ov_ = lists_ov[(b, fo)]
        sets = {s_ for rd in ("R", "C") for s_ in folders.get(f"{b}/{fo}|{rd}", {}).get("lead_three_set_cells", {})}
        if ov_ != {"S_paper493"}:
            fset[(b, fo)] = _setof(fo)
            if "S_paper493" in ov_ or (sets and sets != {_setof(fo)}):
                problems.append(f"{b}/{fo}: listed by {sorted(ov_)}, cells of {sorted(sets)}")
        elif (b, fo) in fill or sets:
            if len(sets) != 1:
                problems.append(f"{b}/{fo}: an S_paper493-override folder feeding {sorted(sets)} (need exactly one set)")
            fset[(b, fo)] = next(iter(sorted(sets)), "?")
        else:
            fset[(b, fo)] = "S_paper493"
    req = {bf for bf, s_ in fset.items() if s_ in THREE}
    for k in list(why):
        if k[0][:2] not in req:
            del why[k]  # S_paper493-only folders: LOO3 excluded; listed below, never in list (a)
    # ---- 3. the pinned inventories (positive control on the lead, then LOO3)
    inv, seen, hs = {}, {}, collections.defaultdict(set)
    for n in INVENTORIES:
        src = B["inv_alias"][n]
        if src in seen:
            continue
        seen[src] = n
        for r in J[src]["runs"]:
            if r.get("eval_sha256"):
                hs[r["key"]].add(r["eval_sha256"])
            if r["key"] in inv:  # Codex r28/r29: a key in two inventories must route identically (folder, world, task; every eval sha256 seen agrees)
                if _ident(inv[r["key"]][1])[:3] != _ident(r)[:3]:
                    problems.append(f"run {r['key']}: inventories {inv[r['key']][0]} and {n} route it differently")
                continue
            inv[r["key"]] = (n, r)
    problems += [f"run {k_}: inventories give {len(v)} eval sha256s" for k_, v in hs.items() if len(v) > 1]
    fo2b = collections.defaultdict(set)
    for b, fo in lead_fo:
        fo2b[fo].add(b)
    if any(len(v) > 1 for v in fo2b.values()):
        raise SystemExit("a lead folder name sits in two batches")
    outside, ctl_bad, ctl, routed = collections.Counter(), [], collections.Counter(), 0
    for key, (n, r) in inv.items():
        fo = r["uncut_planned"]["folder"]
        if fo not in fo2b:
            outside[f"{n}|{fo}"] += 1
            continue
        routed += 1
        b = next(iter(fo2b[fo]))
        for rd in ("R", "C"):
            k = (b, fo, rd)
            if key in lead_rk[k]:
                ctl["lead_scored"] += 1
            elif (lambda f: f"{f['lead_world_labels'][0] if len(f['lead_world_labels']) == 1 else r['world']}|{r['task']}" in f["lead_runs_without_rows"])(folders[f"{b}/{fo}|{rd}"]):
                # Codex r28: the full world|cell, with the folder's single scorer world label (else the inventory world)
                ctl["lead_no_rows (runs_without_rows)"] += 1
            else:
                ctl_bad.append({"key": key, "folder": fo, "readout": rd, "inventory": n})
    own = collections.defaultdict(set)  # Codex r29: a run key in two lead folders (any readout, matching or not) is a problem
    for (b, fo, _rd), s_ in lead_rk.items():
        for r in s_:
            own[r].add(f"{b}/{fo}")
    problems += [f"run {r}: in {len(v)} lead folders {sorted(v)}" for r, v in own.items() if len(v) > 1]
    lead_not_inv = collections.Counter(f"{b}/{fo}" for (b, fo, rd), s in lead_rk.items() for r in s if r not in inv)
    # ---- list (a): every lead run without equal LOO3 units, both readouts
    members = {m["eval_sha256"] for m in J["relaunch_members"]["members"]}
    cohort_eq = {s: set(v["headline_cells"]) == cohort.get(s) for s, v in J["cohort"]["origins"].items()}  # f154cad6's headline cells = 110447's final cohort keys?
    a = {}
    for (k, r), w in sorted(why.items()):
        b, fo, rd = k
        n, ir = inv.get(r, (None, {}))
        e = a.setdefault(r, {"key": r, "set": fset[(b, fo)], "e12_fill_in": fill.get((b, fo)), "pool": fo.rsplit("__", 1)[1], "arm": fo, "batch": b, "readouts": [], "reasons": [],
                             "in_110447_inventory": n, "inventory_world": ir.get("world"), "task": ir.get("task"),
                             "cell_in_cohort": (_base(ir["task"]) in cohort.get(fset[(b, fo)], ())) if ir.get("task") else None,
                             **({"rh1e4_member_958": ir.get("eval_sha256") in members} if fo.startswith("rh1e4_") else {})})
        e["readouts"].append(rd)
        if w not in e["reasons"]:
            e["reasons"].append(w)
    by_spa = collections.Counter(f"{e['set']} x {e['pool']} x {e['arm']}" for e in a.values())
    # ---- list (b): the re-freeze's withheld cells, with LOO3 coverage of their arm runs
    rz = J["refreeze"]["probes"][PROBE]["sets"]
    batch_of = J["cohort"]["origins"]["S_newmains"]["batch_of_cell"]
    arm_runs = collections.defaultdict(list)
    for key, (_n, r) in inv.items():
        fo = r["uncut_planned"]["folder"]
        if r["world"] in ("run1", "run2", "sim") and _setof(fo):
            arm_runs[(_setof(fo), _base(r["task"]))].append((key, r["world"], fo))
    bl, OK1, OK2 = [], "LOO3-scored (R and C equal the lead)", "no rows in the lead or LOO3 (twin present)"
    for s, v in rz.items():
        if s not in cohort:
            continue
        for c, reason in sorted(v["uncovered_cells"].items()):
            st = collections.Counter()
            for key, w, fo in arm_runs.get((s, c), []):
                if fo not in fo2b:
                    st[f"{w}: run routed outside the lead folders"] += 1
                    continue
                ks = [(next(iter(fo2b[fo])), fo, rd) for rd in ("R", "C")]
                tw = all(folders[f"{k[0]}/{k[1]}|{k[2]}"]["twin"] == "present" for k in ks)
                st[f"{w}: " + (OK1 if all(key in ok_rk[k] for k in ks) else
                               OK2 if tw and not any(key in lead_rk[k] or key in loo_rk[k] for k in ks) else
                               "no rows in the lead; LOO3 twin not present" if not any(key in lead_rk[k] for k in ks) else "NOT LOO3-scored")] += 1
            worlds = {w for _, w, _ in arm_runs.get((s, c), [])}
            bl.append({"set": s, "cell": c, "batch": batch_of.get(c) if s == "S_newmains" else None, "freeze_reason": reason, "arm_runs": dict(st),
                       "all_arm_runs_scored": {"run1", "run2", "sim"} <= worlds and all(x.endswith((OK1, OK2)) for x in st),  # Codex r28: all three arms, verified
                       "has_no_row_runs": any(x.endswith(OK2) for x in st), "worlds_present": sorted(worlds)})
    cc = J["refreeze"]["comparison_cells"]["cells"]
    dropped = {s: sorted(cohort[s] - set(cc.get(s, []))) for s in cc}
    # ---- the LOO3 score records' skipped / REFUSED routing records, against the lead folders
    lead_roots = collections.defaultdict(list)
    for b, fo in lead_fo:
        lead_roots[(b, _root(fo))].append(fo)
    loo3_roots = {(b, _root(fo)) for b, fo in recm}
    sk = collections.defaultdict(lambda: {"reasons": collections.defaultdict(list)})
    for i in range(len(LOO3_RECORDS)):
        for nm, v in J[f"loo3_{i}"]["skipped_routing_records"].items():
            if not isinstance(v, str):
                continue
            b, rt = (re.split(r"__| ", nm, maxsplit=1) + [""])[:2]
            sk[(b, rt)]["reasons"][v].append(B["pins"][f"loo3_{i}"]["path"].rsplit("_", 1)[1].removesuffix(".json"))
    skipped = []
    for (b, rt), x in sorted(sk.items()):
        lf = {fo: sorted({folders[f"{b}/{fo}|{rd}"]["twin"] for rd in ("R", "C") if f"{b}/{fo}|{rd}" in folders}) for fo in sorted(lead_roots.get((b, rt), []))}
        skipped.append({"batch": b, "root": rt, "reasons": dict(x["reasons"]), "a_folder_at_this_root_scored_by_LOO3": (b, rt) in loo3_roots,
                        "lead_folders_in_110447": lf})  # each lead folder's LOO3 twin status (R, C)
    gap = [x for x in skipped if any(v != ["present"] for fo, v in x["lead_folders_in_110447"].items() if (b_ := next(iter(fo2b[fo]))) and (b_, fo) in req)]
    rh = [r for r in inv.values() if r[1]["uncut_planned"]["folder"].startswith("rh1e4_")]
    # ---- EXTENDED member checks: E12 e2 entries (by inventory key) and E12 fill-in admitted world|cell keys
    run_fo = {r: k for k, s_ in lead_rk.items() for r in s_}  # run key -> one (batch, folder, readout) holding it
    def mem(keys_runs):  # [(member, [run keys])] -> counts; a member is LOO3-covered when every run is LOO3-equal in R and C
        st, bad = collections.Counter(), []
        for m_, rs_ in keys_runs:
            if not rs_ or any(r not in run_fo for r in rs_):
                st["no lead units"] += 1
                bad.append(m_)
                continue
            ok_ = all(r in ok_rk[run_fo[r][:2] + (rd,)] for r in rs_ for rd in ("R", "C"))
            st["LOO3-equal" if ok_ else "NOT LOO3-equal"] += 1
            if not ok_:
                bad.append(m_)
        return {"counts": dict(st), "not_covered": sorted(bad)}
    e2 = {}
    for s_, pv in e12["pools"].items():
        ents = {x["inventory_key"]: x for v in pv.values() if isinstance(v, dict) for x in ([y for w in v.values() if isinstance(w, dict) for y in w.get("e2", [])] + v.get("e2", []))}
        e2[s_] = mem([(k_, [k_]) for k_ in sorted(ents)])
    fills = {}
    for fk, fv in e12["fill_in"].items():  # 09 RULING (via 4f, 10-16): match each admitted member to ONE run by PHYSICAL identity
        bf = next((h for h, f_ in fill.items() if f_ == fk), None)
        kms = [lead_km.get(bf + (rd,), {}) for rd in ("R", "C")] if bf else [{}, {}]
        ew = sorted({m_.split("|", 1)[0] for m_ in fv["admitted"]})
        lw = sorted({k_.split("|", 1)[0] for km_ in kms for k_ in km_} | {x.split("|", 1)[0] for rd in ("R", "C") for x in folders.get(f"{bf[0]}/{bf[1]}|{rd}", {}).get("lead_runs_without_rows", [])} if bf else set())
        alias = {ew[0]: lw[0]} if len(ew) == 1 and len(lw) == 1 else None  # the ONE-entry world-label alias map (E12 key world -> the lead folder's scorer world label)
        if alias is None:
            problems.append(f"E12 fill-in {fk}: no one-entry world alias (E12 worlds {ew}, lead folder worlds {lw})")
        man = J.get(f"e12_manifest_{fk}", {}).get("files", [])  # the E12-pinned manifest: task -> .eval path + sha256 (non-reference, no sample error)
        by_task = collections.defaultdict(list)
        for f_ in man:
            if f_.get("attempt_kind") != "reference" and f_.get("sample_error") is None:
                by_task[f_["task"]].append(f_)
        matched, how, used = {}, collections.Counter(), collections.Counter()
        for m_ in fv["admitted"]:
            w_, c_ = m_.split("|", 1)
            runs_k = set().union(*(km_.get(f"{alias[w_]}|{c_}", set()) for km_ in kms)) if alias else set()
            mf = by_task.get(c_, [])
            phys = {f_["path"] + "#e0" for f_ in mf} | {f_["path"] for f_ in mf}
            hit = runs_k & phys
            if len(mf) == 1 and len(hit) == 1 and runs_k == hit:  # 1. physical identity: the manifest's .eval path (#e0) IS the lead's run key under the aliased world|cell
                r_ = next(iter(hit))
                ish = hs.get(r_, set())  # Codex r34: every eval sha256 any pinned inventory records for this run
                if ish and ish != {mf[0]["sha256"]}:
                    problems.append(f"E12 fill-in {fk} member {m_}: .eval sha256 differs between the E12 manifest and the inventory")
                    continue
                matched[m_] = {"run": r_, "method": "physical identity (.eval path#epoch, world alias)", "eval_sha256_manifest": mf[0]["sha256"],
                               "eval_sha256_check": "equal to the inventory's" if ish else "not recorded on the inventory / lead side"}
            else:  # 2. by-cell FALLBACK only with a unique candidate whose .eval sha256 equals the admitted member's; else STOP
                cand = {k_ for km_ in kms for k_ in km_ if k_.split("|", 1)[1] == c_}
                cr = set().union(*(km_[k_] for km_ in kms for k_ in cand if k_ in km_))
                ish = hs.get(next(iter(cr)), set()) if len(cr) == 1 else set()
                if len(cand) == 1 and len(cr) == 1 and len(mf) == 1 and ish == {mf[0]["sha256"]}:
                    matched[m_] = {"run": next(iter(cr)), "method": "by-cell fallback (unique candidate, .eval sha256 equal)", "eval_sha256_manifest": mf[0]["sha256"]}
                else:
                    problems.append(f"E12 fill-in {fk} member {m_}: no physical-identity match and the by-cell fallback is not allowed ({len(cand)} candidate keys, "
                                    f"{len(cr)} runs, {len(mf)} manifest files, sha256 {'equal' if mf and ish == {mf[0]['sha256']} else 'not shown equal'})")
                    continue
            how[matched[m_]["method"]] += 1
            used[matched[m_]["run"]] += 1
        twice = sorted(r_ for r_, n_ in used.items() if n_ > 1)
        if len(matched) != len(fv["admitted"]) or twice:  # 3. 1:1
            problems.append(f"E12 fill-in {fk}: {len(matched)} of {len(fv['admitted'])} admitted members matched; runs matched twice: {twice}")
        fills[fk] = {"root": fv["root"], "batch": fv["batch"], "world": fv["world"], "label": fv["label"], "lead_folder": "/".join(bf) if bf else None,
                     "set": fset.get(bf), "admitted": len(fv["admitted"]), "manifest_pin": fv.get("manifest"), "world_alias_map": alias,
                     "matched": len(matched), "matched_by_method": dict(how), "runs_matched_twice": twice, "members_matched": matched,
                     "members": mem([(m_, [matched[m_]["run"]] if m_ in matched else []) for m_ in fv["admitted"]])}
    # ---- the scope BY KEY, per set (09): every lead folder feeding the set's rows, and the E12 pools / fill-in roots
    scope = {s_: {"folders": {}, "e12_pools": {}, "e12_fill_in": {fk: v for fk, v in fills.items() if v["set"] == s_}} for s_ in THREE}
    for (b, fo), s_ in sorted(fset.items()):
        if s_ not in THREE:
            continue
        ks = [k for k in jobs if k[:2] == (b, fo)]
        scope[s_]["folders"][f"{b}/{fo}"] = {"pool_or_arm": fo.rsplit("__", 1)[1], "kind": jobs[ks[0]][3], "override": sorted(lists_ov[(b, fo)]),
                                             "registered": jobs[ks[0]][4], "e12_fill_in": fill.get((b, fo)), "role_S_old": J["full"].get("roles_old", {}).get(jobs[ks[0]][4]),
                                             "lead_files": {k[2]: {"path": jobs[k][5], "sha256": jobs[k][0]} for k in ks},
                                             "loo3_twin": {k[2]: folders[f"{b}/{fo}|{k[2]}"]["twin"] for k in ks}}
    for s_, pv in e12["pools"].items():
        for pn, v in pv.items():
            scope[s_]["e12_pools"][pn] = ({"artifact_pool": v.get("artifact_pool"), "epoch1_members": v.get("epoch1_members"), "epoch1_members_sha256": v.get("epoch1_members_sha256"),
                                           "e2_entries": len(v.get("e2", []))} if "epoch1_members" in v else
                                          {fam: {"artifact_pool": w.get("artifact_pool"), "epoch1_members": w.get("epoch1_members"), "e2_entries": len(w.get("e2", []))}
                                           for fam, w in v.items() if isinstance(w, dict)} if pn != "H_A_group_of_family" else v)
    e12_not_covered = sorted({m_ for v in e2.values() for m_ in v["not_covered"]} | {m_ for v in fills.values() for m_ in v["members"]["not_covered"]})
    lead_missing_members = sum(v["counts"].get("no lead units", 0) for v in e2.values()) + sum(v["members"]["counts"].get("no lead units", 0) for v in fills.values())
    gap_fo = {e["arm"] for e in a.values()} | {fo for (b, fo, rd), x in ((k, folders[f"{k[0]}/{k[1]}|{k[2]}"]) for k in jobs) if (b, fo) in req and x["twin"] != "present"}
    unexpected = {"list_a_or_twin_folders_outside_the_approved": sorted(gap_fo - set(APPROVED_GAPS)),
                  "e2_members_not_covered": {s_: [m_ for m_ in v["not_covered"] if m_ not in run_fo or run_fo[m_][1] not in APPROVED_GAPS] for s_, v in e2.items()},
                  "fill_in_members_not_covered_outside_g4": {fk: v["members"]["not_covered"] for fk, v in fills.items()
                                                             if v["members"]["not_covered"] and (v["lead_folder"] or "").split("/")[-1] not in APPROVED_GAPS},
                  "lead_missing_e12_members": lead_missing_members}
    unexpected = {k_: v for k_, v in unexpected.items() if v and (not isinstance(v, dict) or any(v.values()))}
    sp_only = {f"{b}/{fo}": sorted({folders[f"{b}/{fo}|{rd}"]["twin"] for rd in ("R", "C") if f"{b}/{fo}|{rd}" in folders}) for (b, fo), s_ in sorted(fset.items()) if s_ == "S_paper493"}
    return {
        "control_own_all_rs_uncut": {"lead_files": len(jobs), "lead_folders": len(lead_fo), "pinned_runs_routed_to_lead_folders": routed,
                                     "checks_by_readout": dict(ctl), "unexplained": ctl_bad, "n_unexplained": len(ctl_bad),
                                     "lead_runs_outside_the_pinned_inventories": dict(lead_not_inv),
                                     "pass": not ctl_bad and not any("lead file" in p for p in problems)},
        "folder_level": {"lead_folders": len(lead_fo), "required_folders": len(req), "required_by_set": dict(collections.Counter(fset[x] for x in req)),
                         "loo3_twin_missing_required": sorted({f"{b}/{fo}" for (b, fo, rd), x in ((k, folders[f"{k[0]}/{k[1]}|{k[2]}"]) for k in jobs) if x["twin"] != "present" and (b, fo) in req}),
                         "by_twin_status": dict(collections.Counter(x["twin"] for x in folders.values()))},
        "list_a": {"rule": "runs with lead (own_all_rs_uncut_L48, 110447-pinned) units and no equal LOO3 units in R or C; by key", "n_runs": len(a),
                   "by_set_pool_arm": dict(sorted(by_spa.items())), "by_reason": dict(collections.Counter(w for e in a.values() for w in e["reasons"])),
                   "not_in_a_110447_inventory": sum(e["in_110447_inventory"] is None for e in a.values()),
                   "runs": sorted(a.values(), key=lambda e: (e["set"], e["pool"], e["arm"], e["key"]))},
        "list_b": {"rule": "cells the re-freeze @ 57b6671e7 withheld (uncovered_cells), with the LOO3 coverage of their arm runs (run1, run2, sim) in the lead folders",
                   "n_cells": len(bl), "n_all_arm_runs_scored": sum(x["all_arm_runs_scored"] for x in bl),
                   "by_reason": dict(collections.Counter(x["freeze_reason"] for x in bl)),
                   "cohort_minus_comparison_equals_withheld": all(set(dropped[s]) == set(rz[s]["uncovered_cells"]) for s in dropped),
                   "cells": bl},
        "skipped_routing_records": {"records": skipped, "in_the_lead_overrides_and_never_scored_by_LOO3": gap},
        "scope_by_set": scope,
        "e12_member_checks": {"e2_by_set": e2, "fill_in": fills, "paired_pool_exclusions_110447": J["full"].get("paired_pool_exclusions")},
        "s_paper493_only_folders": {"n": len(sp_only), "loo3_twin_status": sp_only, "rule": "S_paper493-override folders with no S_old / S_new / S_newmains cell and no E12 fill-in role: LOO3 excluded, not in list (a)"},
        "key_column_differs": {f: x.get("key_column_world_labels") for f, x in sorted(folders.items()) if x.get("key_column_equal") is False},
        "rh1e4": {"inventory_runs": len(rh), "members_958": sum(r[1].get("eval_sha256") in members for r in rh), "members_in_file": len(members),
                  "in_list_a": sum(1 for e in a.values() if e["arm"].startswith("rh1e4_")),
                  "in_list_a_members": sum(1 for e in a.values() if e.get("rh1e4_member_958"))},
        "pinned_runs_routed_outside_the_lead_folders": dict(sorted(outside.items())),
        "cohort_f154cad6_headline_equals_110447_final_cohort_keys": cohort_eq,
        "folders": folders, "problems": problems,
        "unexpected_gaps": unexpected,  # Codex r32: anything beyond the approved rh1e4 + g4 + known ha25f folders is a STOP (0d)
        "stop_unexpected_gap": bool(unexpected),
        "audit_valid": not problems and not ctl_bad and not lead_missing_members,  # Codex r28/r32: inputs as pinned, control passes, every E12 member has lead units ...
        "coverage_complete": not problems and not ctl_bad and not a and all(folders[f"{k[0]}/{k[1]}|{k[2]}"]["twin"] == "present" and not folders[f"{k[0]}/{k[1]}|{k[2]}"]["n_loo3_extra_runs"]
                                                                            for k in jobs if k[:2] in req) and not e12_not_covered,  # ... vs LOO3 complete (REQUIRED folders, E12 members)
    }


@app.local_entrypoint()
def main():
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, check=True).stdout  # noqa: E731
    me = str(Path(__file__).resolve().relative_to(REPO))
    if git("status", "--porcelain", "--untracked-files=no").strip() or git("rev-parse", "HEAD").strip() != git("rev-parse", "origin/aswin/own-all-loo3-uncut-4f").strip():
        raise SystemExit("commit and push the script first (HEAD must be on origin, tracked tree clean)")
    files, pins = {}, {}

    def put(n, c, p, h):
        t = git("show", f"{c}:{p}")
        if h and _sha(t.encode()) != h:
            raise SystemExit(f"{n}: sha256 differs from its pin")
        files[n], pins[n] = t, {"commit": c, "path": p, "sha256": _sha(t.encode())}

    put("full", *FULL)
    full = json.loads(files["full"])
    for ov in OVERRIDES:
        for rd in ("R", "C"):
            x = full["overrides"][f"{ov}|{rd}"]
            put(f"ov_{ov}|{rd}", x["commit"], x["path"], x["sha256"])
    alias, by_src = {}, {}
    for n in INVENTORIES:
        x = full["inventories"][n]
        if (x["commit"], x["path"]) not in by_src:
            by_src[(x["commit"], x["path"])] = f"inv_{n}"
            put(f"inv_{n}", x["commit"], x["path"], x["sha256"])
        elif pins[by_src[(x["commit"], x["path"])]]["sha256"] != x["sha256"]:
            raise SystemExit(f"inventory {n}: same file, another sha256")
        alias[n] = by_src[(x["commit"], x["path"])]
    for n in ("exclusions", "cohort", "relaunch_members", "e12"):
        put(n, full[n]["commit"], full[n]["path"], full[n]["sha256"])
    for fk, fv in json.loads(files["e12"])["fill_in"].items():  # the E12-pinned manifests of the fill-in roots (09: physical identity of each admitted member)
        put(f"e12_manifest_{fk}", fv["manifest"]["commit"], fv["manifest"]["path"], fv["manifest"]["sha256"])
    for i, (c, p) in enumerate(LOO3_RECORDS):
        put(f"loo3_{i}", c, p, None)
    put("refreeze", *REFREEZE)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    bb = json.dumps({"files": files, "pins": pins, "inv_alias": alias}, sort_keys=True).encode()
    bsha, rp = _sha(bb), f"/loo3u_cov110447/{stamp}/bundle.json"
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json") as tf:
        tf.write(bb)
        tf.flush()
        with WORK.batch_upload() as up:  # create-only (a fresh stamped path)
            up.put_file(tf.name, rp)
    print(json.dumps({"files": len(files), "MB": round(len(bb) / 1e6, 1), "bundle": rp, "sha256": bsha}))
    res = check.remote("/work" + rp, bsha)
    rec = {"label": "LOO3 uncut coverage check against the FROZEN INPUTS_FULL 110447 pins (4f / 0d / 09 / 2e, 10-16; keys only; no score read)",
           "stamp (PT)": stamp, "generator": {"path": me, "commit": git("rev-parse", "HEAD").strip(), "app": APP, "app_id": app.app_id or ""},
           "method": __doc__.split("\n\n")[1], "sources": pins, "inventory_files": alias, "bundle": {"volume_path": rp, "sha256": bsha}} | res
    p = HERE / "results" / f"loo3_coverage_110447_{stamp}.json"
    p.open("x").write(json.dumps(rec, indent=1))
    print(p)
    print(json.dumps({"audit_valid": res["audit_valid"], "coverage_complete": res["coverage_complete"], "stop_unexpected_gap": res["stop_unexpected_gap"],
                      "unexpected_gaps": res["unexpected_gaps"], "key_column_differs": res["key_column_differs"], "e12": {"e2": {s_: v["counts"] for s_, v in res["e12_member_checks"]["e2_by_set"].items()},
                                                                         "fill_in": {k_: v["members"]["counts"] | {"by_method": v["matched_by_method"], "alias": v["world_alias_map"]}
                                                                                     for k_, v in res["e12_member_checks"]["fill_in"].items()}}, "problems": res["problems"][:10], "control": {k: v for k, v in res["control_own_all_rs_uncut"].items() if k != "unexplained"},
                      "folder_level": res["folder_level"], "list_a": {k: v for k, v in res["list_a"].items() if k != "runs"},
                      "list_b": {k: v for k, v in res["list_b"].items() if k != "cells"}, "rh1e4": res["rh1e4"],
                      "skipped_gap": res["skipped_routing_records"]["in_the_lead_overrides_and_never_scored_by_LOO3"]}, indent=1))
