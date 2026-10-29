"""4f's by-key pin checks for IL token roots (IL rule item 6; 09 Append 32 @ 92d21b8f6 and Addendum 1 @ 7bd9d5e89): r, r2, r3, c.
Reads a built pass-2 message and the records it cites, at their commits. Keys and shas only; never builds or launches anything.

  python verify_il_tokens.py <message.json> [...]

Every token folder: every staged and excluded (arm, cell) has a row keyed "<arm>|<cell>" in the cited supersession record, and a
staged run comes from that row's authoritative launch and run key (and, where the record lists them, its replacement .eval sha256);
both arms of every cell resolve to the same authoritative launch (no mixed pair). d9's check records ("records", authoritative
"replacement" or a "moved ... retired" string) are read as: only "replacement" rows are authoritative here. c folders also check each cell against its OWN tier (one
completion root mixes tiers 2 and 3): in the manifest's tier list for the host; tier 2 = both original arms "not run" with 0
attempts; tier 3 = original run3 ran, sim2 "not run" with 0 attempts, and the cell is in the run3-only tier 3 record; a cell
outside the tier lists or in the blocker status rows is an error. r3 folders (Addendum 1 item 2) also check, per staged cell: it is
exactly the rlr3 selection; the retired rlr launch's final read shows both arms "not run" with 0 attempts; its validation row passes;
the three records name the same rlr launch.json sha256 (unchanged before/after); and, if the per-arm rlr3 revision is cited as
supersession_revision_rlr3, each staged run is its authoritative run.
"""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
D = "experiments/2026-10-15_q3_interleaved_run3_sim2/config/completion/"
TOKEN = re.compile(r"^(?:il311_q([1-4])(r|r2|r3|c)|ilnm_s(?:[1-5]|4d)(r|r2|r3)|ilold_old(r|r3|c))_")
C_MANIFEST = ("e46ba05d0", D + "completion_manifest.json", "afd7af9593510b82")  # 09 @ 92d21b8f6: sha256 afd7af95...
C_PERARM = ("54823ddbf", D + "ilc_supersession_perarm_{h}.json")
C_TIER3 = ("e46ba05d0", D + "ilc_supersession_tier3_{h}.json")
C_BLOCKERS = ("54823ddbf", D + "il_blocker_status_rows.json")
ARM_DIR = {"run3": "attack", "sim2": "sim_attack@september"}
PINS = {"r": ("supersession", "freeze_addendum"), "c": ("registration", "supersession"), "r2": ("registration", "supersession"),
        "r3": ("registration", "supersession", "rlr_final", "rlr3_selection", "rlr3_validation")}  # as TOKEN_PINS in run_famexp_pass2.py


def show(c, p):
    return subprocess.run(["git", "-C", str(REPO), "show", f"{c}:{p}"], capture_output=True, check=True).stdout


def parse(key):  # "[runs/]<launch>/<arm dir>/<cell>/<eval>#e0"
    parts = [x for x in key.split("/") if x != "runs"]
    return parts[0], parts[1], parts[-2]


def check(path):
    msg = json.loads(Path(path).read_text())
    out, fails = {"message": path, "folders": {}}, []
    def canon(key, cell):  # the cell is d9's manifest task; the key's directory is the cell or its counted redraw (<cell>_redraw<n>)
        launch, arm_dir, dcell = parse(key)
        if not cell or not (dcell == cell or re.fullmatch(re.escape(cell) + r"_redraw[0-9]+", dcell)):
            fails.append(f"{key.rsplit('/', 1)[0]}: the key's directory does not name the manifest cell {cell}")
        return launch, arm_dir, cell or dcell

    rows = [(e["folder"], e["world"], *canon(e["key"], e.get("cell")), e["run"], "staged", e.get("eval_sha256")) for e in msg["epochs"]]
    rows += [(x["folder"], next(iter(p_["by_world"])) if len(p_.get("by_world") or {}) == 1 else None, *canon(x["key"], x.get("cell")), p_["run"], "excluded", None)
             for p_ in msg["provenance"] for x in p_["excluded_by_staging"]]
    for fo in sorted({r[0] for r in rows}):
        m_ = TOKEN.match(fo)
        if not m_:
            continue
        tok = next(g for g in m_.groups()[1:] if g)
        runs = {r[5] for r in rows if r[0] == fo}
        parts = [p_ for p_ in msg["provenance"] if p_["run"] in runs]
        pinsets = [{k: (p_.get("citations") or {}).get(k) for k in PINS[tok]} for p_ in parts]
        if not parts or {p_["run"] for p_ in parts} != runs or any(ps != pinsets[0] for ps in pinsets):
            fails.append(f"{fo}: a run feeding it has no message part, or its parts cite different pins")
            continue
        cites = pinsets[0]  # every part feeding this folder cites the same pins (checked above)
        bad = [k for k, v in cites.items() if not v or not v.get("sha256") or hashlib.sha256(show(v["commit"], v["path"])).hexdigest() != v["sha256"]]
        if bad:
            fails.append(f"{fo}: pins missing, or their sha256 differs from git: {bad}")
            continue
        sup = cites["supersession"]
        doc = json.loads(show(sup["commit"], sup["path"]))
        if isinstance(doc.get("records"), list):  # d9's check record: authoritative here only if "replacement"
            srows = {r["key"]: {**r, "authoritative": {"launch": r["replacement"]["launch_id"], "run_key": r["replacement"]["run_key"]} if r["authoritative"] == "replacement" else None,
                                "evals": r["replacement"].get("eval_sha256")} for r in doc["records"]}
        else:
            srows = {r["key"]: r for r in doc["rows"]}
        res = {"token": tok, "supersession": f"{sup['path']} @ {sup['commit']}", "rows": 0}
        if tok == "c":
            h = f"q{m_.group(1)}" if m_.group(1) else "old"
            reg = cites["registration"]
            mraw = show(*C_MANIFEST[:2])
            if (reg["commit"], reg["path"]) != C_MANIFEST[:2] or not reg["sha256"].startswith(C_MANIFEST[2]) or hashlib.sha256(mraw).hexdigest() != reg["sha256"]:
                fails.append(f"{fo}: the registration pin is not completion_manifest.json @ {C_MANIFEST[0]} (sha256 {C_MANIFEST[2]}...)")
            if (sup["commit"], sup["path"]) != (C_PERARM[0], C_PERARM[1].format(h=h)):
                fails.append(f"{fo}: the supersession pin is not {C_PERARM[1].format(h=h)} @ {C_PERARM[0]}")
            host = json.loads(mraw)["hosts"][h]
            tier = {c["cell"]: t for t in ("tier 2", "tier 3") for c in host[t]}
            t3raw = show(C_TIER3[0], C_TIER3[1].format(h=h))
            if hashlib.sha256(t3raw).hexdigest() != host["supersession_sha256"]:
                fails.append(f"{fo}: the tier 3 record's sha256 differs from the manifest's")
            t3 = {r["cell"] for r in json.loads(t3raw)["rows"] if r["arm"] == "run3"}
            blocked = {r["cell"] for r in json.loads(show(*C_BLOCKERS))["rows"]}
            res["tiers"] = {}
        for c_ in sorted({r[4] for r in rows if r[0] == fo}):  # no mixed pair: both arms resolve to one authoritative launch
            au = [(srows.get(f"{a}|{c_}") or {}).get("authoritative") for a in ("run3", "sim2")]
            if not all(au) or au[0]["launch"] != au[1]["launch"]:
                fails.append(f"{fo}: {c_} does not resolve both arms to one authoritative launch ({au})")
        for fo_, world, launch, arm_dir, cell, _, kind, ev in (r for r in rows if r[0] == fo):
            res["rows"] += 1
            sr = srows.get(f"{world}|{cell}")
            if sr is None:
                fails.append(f"{fo}: {world}|{cell} has no row in the supersession record")
                continue
            if kind == "staged" and (not sr["authoritative"] or sr["authoritative"]["launch"] != launch or sr["authoritative"]["run_key"] != f"{ARM_DIR[world]}/{cell}"
                                     or arm_dir != ARM_DIR[world] or (sr.get("evals") is not None and ev not in sr["evals"])):
                fails.append(f"{fo}: {world}|{cell} is not the authoritative run ({sr['authoritative']})")
            if tok == "c":
                t = tier.get(cell)
                res["tiers"][t] = res["tiers"].get(t, 0) + 1
                orig = {a: srows.get(f"{a}|{cell}", {}).get("original", {}) for a in ("run3", "sim2")}
                not_run = lambda o: o.get("state") == "not run" and o.get("attempts") == 0
                if t is None or cell in blocked:
                    fails.append(f"{fo}: {cell} is outside the tier lists or a blocker status row")
                elif sr.get("tier") != t:
                    fails.append(f"{fo}: {world}|{cell} tier differs between the manifest ({t}) and the per-arm map ({sr.get('tier')})")
                elif t == "tier 2" and not (not_run(orig["run3"]) and not_run(orig["sim2"])):
                    fails.append(f"{fo}: tier 2 {cell} does not have both original arms 'not run' with 0 attempts")
                elif t == "tier 3" and not (orig["run3"].get("state") not in (None, "not run") and not_run(orig["sim2"]) and cell in t3):
                    fails.append(f"{fo}: tier 3 {cell} is not (original run3 ran, sim2 'not run' 0 attempts, in the run3-only tier 3 record)")
        cit = {k: v for p_ in parts for k, v in (p_.get("citations") or {}).items()}
        chk = cit.get("supersession_check")  # optional: d9's check record, for replacement .eval sha256s the pinned record lacks
        cev = {}
        if chk and hashlib.sha256(show(chk["commit"], chk["path"])).hexdigest() == chk.get("sha256"):
            cdoc = json.loads(show(chk["commit"], chk["path"]))
            cev = {r["key"]: r["replacement"].get("eval_sha256") for r in cdoc.get("records", []) if r.get("authoritative") == "replacement"}
        elif chk:
            fails.append(f"{fo}: the supersession_check citation's sha256 differs from git")
        evs_map = {k_: (srows.get(k_) or {}).get("evals") or cev.get(k_) for k_ in set(srows) | set(cev)}
        for fo_, world, launch, arm_dir, cell, _, kind, ev in (r for r in rows if r[0] == fo and r[6] == "staged"):
            if evs_map.get(f"{world}|{cell}") is not None and ev not in evs_map.get(f"{world}|{cell}"):
                fails.append(f"{fo}: {world}|{cell} staged .eval is not the record's replacement")
        xk = [(p_, x["key"]) for p_ in parts for x in p_["excluded_by_staging"] if x["folder"] == fo]
        if len({parse(k_)[2] for _, k_ in xk}) != len(xk):
            fails.append(f"{fo}: duplicate exclusions for one cell")
        for p_, xkey in xk:  # every token folder (4f's OK for r3, extended to c): each exclusion by its OWN key, class cited, bound to the authoritative run
            launch, arm_dir, cell = parse(xkey)
            world = next(w for w, d in ARM_DIR.items() if d == arm_dir)
            lab = [k for k in cit if k.startswith(f"class_ruling_{cell.split('_')[0]}_")]
            if not lab or any(hashlib.sha256(show(cit[k]["commit"], cit[k]["path"])).hexdigest() != cit[k]["sha256"] for k in lab):
                fails.append(f"{fo}: excluded {world}|{cell} has no class_ruling_{cell.split('_')[0]}_* citation, or its sha256 differs")
            a_ = (srows.get(f"{world}|{cell}") or {}).get("authoritative") or {}
            mf = p_["manifest"]
            mrow = [f for f in json.loads(show(mf["commit"], mf["path"]))["files"] if f"{f['launch']}/{f['rel']}#e0" == xkey]
            evs = evs_map.get(f"{world}|{cell}")
            if (a_.get("launch"), a_.get("run_key")) != (launch, f"{ARM_DIR[world]}/{cell}") or len(mrow) != 1 or (evs is not None and mrow[0]["sha256"] not in evs):
                fails.append(f"{fo}: excluded {world}|{cell} key {xkey} is not the authoritative replacement run (record {a_}, manifest rows {len(mrow)})")
        if tok == "r3":
            cs = sorted({r[4] for r in rows if r[0] == fo and r[6] == "staged"})  # staged cells
            xr = [r for r in rows if r[0] == fo and r[6] == "excluded"]
            xs = sorted({r[4] for r in xr})  # selected cells whose replacement arm was excluded at staging (4f's OK, 10-15 ~08:0x)
            fin, sel, val = (json.loads(show(cites[k]["commit"], cites[k]["path"])) for k in ("rlr_final", "rlr3_selection", "rlr3_validation"))
            vrows = {r["cell"]: r for r in val["rows"]}
            if set(cs) & set(xs) or sorted(set(cs) | set(xs)) != sorted(sel["selected"]):
                fails.append(f"{fo}: staged {cs} and excluded {xs} are not a partition of the rlr3 selection {sorted(sel['selected'])}")
            if not (fin["sha_before"] == fin["launch_json_sha256"] == sel["rlr_launch_json_sha256"] == val["rlr_launch_json_sha256"]):
                fails.append(f"{fo}: the rlr final read, selection and validation do not name one unchanged rlr launch.json sha256")
            for c_ in sorted(set(cs) | set(xs)):  # every selected cell, excluded ones included
                st = fin["cells"].get(c_, {})
                if not all(st.get(a, {}).get("state") == "not run" and st.get(a, {}).get("attempts") == 0 for a in ("real", "sim")):
                    fails.append(f"{fo}: {c_}: the retired rlr arms are not both 'not run' with 0 attempts ({st})")
                v_ = vrows.get(c_, {})
                if not all(v_.get(k) is True for k in ("input_is_designated", "both_arms_zero_attempts", "no_replacement_run", "no_earlier_extra_designation")):
                    fails.append(f"{fo}: {c_}: no passing rlr3 validation row ({v_})")
            revs = [(p_.get("citations") or {}).get("supersession_revision_rlr3") for p_ in parts]
            rev = revs[0]
            if any(r_ != rev for r_ in revs) or (rev and hashlib.sha256(show(rev["commit"], rev["path"])).hexdigest() != rev.get("sha256")):
                fails.append(f"{fo}: the parts cite different rlr3 revisions, or its sha256 differs from git")
            elif rev:
                rrows = {r["key"]: r for r in json.loads(show(rev["commit"], rev["path"]))["rows"]}
                for fo_, world, launch, arm_dir, cell, _, kind, ev in (r for r in rows if r[0] == fo):  # staged and excluded rows alike
                    a_ = (rrows.get(f"{world}|{cell}") or {}).get("authoritative") or {}
                    if (a_.get("launch"), a_.get("run_key")) != (launch, f"{ARM_DIR[world]}/{cell}"):
                        fails.append(f"{fo}: {world}|{cell} is not the rlr3 revision's authoritative run ({a_})")
            res["r3"] = {"cells": cs, "excluded": xs, "selection": f"{cites['rlr3_selection']['path']} @ {cites['rlr3_selection']['commit']}"}
        out["folders"][fo] = res
    out["pass"], out["fails"] = not fails, fails
    return out


if __name__ == "__main__":
    res = [check(a) for a in sys.argv[1:]]
    print(json.dumps(res, indent=1))
    sys.exit(0 if res and all(r["pass"] for r in res) else 1)  # non-zero on any failure, so a shell gate stops
