"""S_newmains fragment of run_xerr_scaleup.py's spec (09 via d9 10-15 ~10:20; template: final_spec_3_launch.json @ 3ce6ea55e on
aswin/xerr-scaleup-09; loader checks mirrored from run_xerr_scaleup.py @ aab87af1a): for the chosen new-mains quarters, every
chain-side pin the pooled (extended) run needs, in the template's shapes, each path @ commit + sha256. Two steps, because the
fragment pins committed files only:

1. `inputs` writes the derived inputs (new files, never overwritten; commit them before step 2):
   - registry_S_newmains_xerr_<tag>.json: the chain's registry_S_newmains.json, in-scope entries only, checked as the loader does;
   - labels_nm360_xerr_<tag>.json and run_table_nm360_xerr_<tag>.json: the combined final labels and run table, in-scope cells only
     (a cell is in scope when its run-table real_batch is an in-scope batch; the combined files also hold S4 / S4r keys);
   - exclusions_nm360_<target>_superseded_<tag>.json: one per manifest holding superseded originals (S5: the 6 sim runs of
     supersession_nm360_s5r @ c3e9e3e95), in the d9 exclusions format the loader reads (Revision 6j): the originals are dropped
     before counting, so the replacements fill their keys once.
2. `fragment` pins them and writes xerr_spec_fragment_S_newmains_<tag>.json:
   - arms[<folder>]: hygiene (the attestation's evidence) + p2_manifest and sol_manifest from the scorers' work Volume; a folder whose
     every key is a v4 setup failure is listed under excluded_arms, by key; missing scorer manifests are listed, never invented;
   - counted (run1, honest_a, honest_b: main-family manifests; run2; sim, with its exclusions record), labels (+ per-batch
     cross-check), run_table (+ per-quarter cross-check; a table whose only disagreeing rows are superseded originals is left out of
     the cross-check and listed with its counts), staging_exclusions (world|task keys), setup_failure_exclusions (v4), registry.
   - status_rows: 09's rerun-kind ruling (honest_a cells whose counted attempt failed and whose rerun was clean stay uncounted;
     no rerun enters a manifest), each key checked against the rerun census @ 9a9f8b5c1 and the staging exclusions.
   Before writing, the loader's key checks run here: no key double-filled, label cross-check, labels for uncounted runs, counted
   attack runs without a label, label staged sets per folder; folder names are checked against 09's FOLDER_RE (reported only).
Keys, paths and hashes only: no score values, no transcript, no label values printed.

  python make_xerr_spec_newmains.py inputs   <chain state dir> <quarters, e.g. s1,s2,s3,s5> <tag, e.g. S1S2S3S5_2026-10-15>
  python make_xerr_spec_newmains.py fragment <chain state dir> <quarters> <tag>
  XERR_DRY_OUT=<scratch dir> python make_xerr_spec_newmains.py inputs ...   # dry run of step 1 outside results/
"""
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import chain as C

OUT = Path(os.environ.get("XERR_DRY_OUT") or C.HERE / "results" / "hygiene")  # XERR_DRY_OUT: a scratch dir for an `inputs` dry run only

WORK = "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14"
HYG = C.HERE / "results" / "hygiene"
V4 = {"commit": "396da80fd", "path": f"{C.REL}/results/hygiene/setup_failures_nm360_v4_2026-10-15.json"}
SUPERSESSION = {"s5": "results/hygiene/supersession_nm360_s5r_2026-10-15.json"}
# 09's ruling (via d9 10-15 ~10:45): honest_a cells whose counted attempt failed but whose rerun was clean stay uncounted; no rerun
# enters a manifest. Status rows only, checked against the rerun census and the staging exclusions.
CENSUS = {"commit": "9a9f8b5c1", "path": "experiments/2026-10-14_q3_new_mains_launch/results/rerun_census_nm360_2026-10-15.json"}
RERUN_KIND = {"s1": ("honest_a|nm142_model_registry", "honest_a|nm280_tiktok", "honest_a|nm209_slack_clone")}
RERUN_STATUS = "uncounted by rule (rerun-kind); counted attempt run_failed and excluded"
SFX = {"attack": "attack", "honest_a": "honest_a", "honest_b": "honest_b", "sim_attack@september": "sim_attack"}
ARMS = ("run1", "run2", "sim")
LABEL_ENUM = ("attempted", "unclear", "declined", "run_failed")
APPROACH_CELL_FIELDS = ("cell", "pass", "equal_approach", "equal_system_prompt", "equal_first_user", "first_user_empty", "staged_shas")
SIM_STATUS = ("ran", "cap_stopped", "no_honest_A_reference")
NMREG = {"commit": "ef171c813", "path": "experiments/2026-10-14_q3_new_mains_mode/config/new_mains_cells_v1.json",
         "sha256": "98bfb97e26e7ec580b08643a3a38a5e66a0f6b63333f4013aa57559383402e8e"}  # the loader's PINS: S_newmains membership
RT_F = ("attack_pair_launch", "run1_launch", "run2_launch", "sim_launch", "sim_status")
LOADER = ("origin/aswin/xerr-scaleup-09", "experiments/2026-10-14_xerr_scaleup_09/run_xerr_scaleup.py")
# 09's folder grammar (folder_canonical, Revision 6t), copied from run_xerr_scaleup.py @ d0617ba06 lines 102-119; every raw-string piece is
# checked against the loader source at run time, so a changed grammar stops the run instead of passing silently
FOLDER_PIECES = {
    "FOLDER_RE": [r"^(su311|nm360)_(q[0-9]+r?|s[0-9]+)_(main|run2|sim)(_s[0-9]+)?_v1(_try[0-9]+)?__(honest_a|honest_b|attack|sim_attack)$"],
    "R_NOT_SIM": [r"^(su311|nm360)_q[0-9]+r_(main|run2)_"],
    "NM360_FU_RE": [r"^nm360_(?:(?:s4d_sim(?:_s[0-9]+)?|(?:s5r|s4r)_sim)_v1(?:_try[0-9]+)?__sim_attack",
                    r"|s4r_(?:main_v1(?:_try[0-9]+)?__(?:attack|honest_a|honest_b)|run2_v1(?:_try[0-9]+)?__attack))$"],
    "S4_FOLDER": [r"^nm360_s4(?:d|r)?_"],
}


def git(*a: str) -> str:
    return subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=C.HERE).stdout.strip()


def pin(rel: str) -> dict:
    """path @ the last commit that touched it, and the committed file's sha256 (the working tree must equal it)."""
    path = rel if rel.startswith("experiments/") else f"{C.REL}/{rel}"
    commit = git("log", "-1", "--format=%H", "--", str(C.REPO / path))[:12]
    assert commit, f"{path}: never committed"
    raw = subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, check=True, cwd=C.HERE).stdout
    assert (C.REPO / path).read_bytes() == raw, f"{path}: working tree differs from {commit}"
    return {"commit": commit, "path": path, "sha256": hashlib.sha256(raw).hexdigest()}


def write_new(p: Path, doc: dict):
    with p.open("x") as f:
        f.write(json.dumps(doc, indent=1) + "\n")


def world_of(arm: str, target: str) -> str:
    return {"attack": "run2" if "_run2" in target else "run1", "honest_a": "honest_a", "honest_b": "honest_b",
            "sim_attack@september": "sim"}[arm]


def scope_of(st_dir: str, quarters: list) -> dict:
    st = json.loads((Path(st_dir) / "state.json").read_text())["batches"]
    scope = {bid: b for bid, b in sorted(st.items()) if b.get("pool") == "nm360" and b["q"] in quarters and b["status"] not in ("EMPTY", "STOPPED")}
    assert all("attest" in b["done"] for b in scope.values()), [k for k, b in scope.items() if "attest" not in b["done"]]
    return scope


def manifests(scope: dict) -> list:
    """(batch, target, manifest name, manifest doc) for every in-scope target."""
    out = []
    for bid, b in scope.items():
        for t in b["targets"]:
            name = C.targets_for(b)[t]["manifest"]
            out.append((bid, t, name, json.loads((C.HERE / "config" / name).read_text())))
    return out


def in_scope_cells(scope: dict, mans: list) -> set:
    """Cells whose combined-run-table real_batch is in scope; must equal the in-scope manifests' cells."""
    rt = json.loads((HYG / "run_table_nm360.json").read_text())
    cells = {r["cell"] for r in rt["cells"] if r["real_batch"] in scope}
    from_man = {f["task"] for _, _, _, m in mans for f in m["files"]}
    assert cells == from_man, ("run table vs manifests", sorted(cells ^ from_man)[:10])
    return cells


def superseded_rows(quarters: list, mans: list) -> list:
    """(manifest name, manifest row, supersession row) for every superseded original in scope, matched by launch, key and sha256."""
    out = []
    for q, rel in SUPERSESSION.items():
        if q not in quarters:
            continue
        sup = json.loads((C.HERE / rel).read_text())
        assert sup["pass"] is True and not sup["errors"], rel
        for r in sup["rows"]:
            assert r["authoritative"] == "replacement", r["key"]
            o, w, cell = r["original"], *r["key"].split("|", 1)
            (m,) = [m for _, _, n, m in mans if n == o["manifest"]]
            (row,) = [f for f in m["files"] if f["world"] == w and f["task"] == cell and f["launch"] == o["launch"]]
            assert row["sha256"] == o["eval_sha256"] and row["rel"].startswith(o["run_key"] + "/"), r["key"]
            out.append((o["manifest"], row, r))
    return out


def cmd_inputs(st_dir: str, quarters: list, tag: str):
    scope = scope_of(st_dir, quarters)
    mans = manifests(scope)
    cells = in_scope_cells(scope, mans)
    now = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S PT")
    scope_txt = f"nm360 {','.join(quarters)}: batches {', '.join(scope)} (S4 later as its own batch)"
    # registry: the chain's S_newmains registry, in-scope entries only (the loader's format), checked as the loader does
    src = pin("results/hygiene/registry_S_newmains.json")
    reg = json.loads((C.REPO / src["path"]).read_text())
    ent = {k: v for k, v in reg["entries"].items() if v["batch"] in scope}
    honest = {pin(f"config/{n}")["sha256"] for _, t, n, _ in mans if not (t.endswith("_sim") or "_sim_s" in t or "_run2" in t)}
    bad = [k for k, v in ent.items() if v["main_family_complete"] is not True or any(h["sha256"] not in honest for h in v["honest_manifests"])]
    assert not bad, f"registry entries failing the loader's checks: {bad}"
    assert {v["batch"] for v in ent.values()} == set(scope), sorted(set(scope) - {v["batch"] for v in ent.values()})
    write_new(OUT / f"registry_S_newmains_xerr_{tag}.json",
              {"stratum": "S_newmains", "written": now, "entries": ent, "scope": scope_txt, "derived_from": src,
               "rule": "the chain's registry_S_newmains.json, in-scope entries only; every entry checked as the loader does"})
    # labels and run table: the combined finals, in-scope cells only
    lsrc, rsrc = pin("results/hygiene/attempt_labels_nm360.json"), pin("results/hygiene/run_table_nm360.json")
    lab, rt = json.loads((C.REPO / lsrc["path"]).read_text()), json.loads((C.REPO / rsrc["path"]).read_text())
    keep = lambda k: k.split("|", 1)[1] in cells  # noqa: E731
    lab_doc = {"labels": {k: v for k, v in lab["labels"].items() if keep(k)}, "rules": {k: v for k, v in lab["rules"].items() if keep(k)},
               "staged_sets": {s: v for s, v in lab["staged_sets"].items() if v["batch"].split("_", 1)[1] in scope},
               "excluded": {k: v for k, v in lab["excluded"].items() if keep(k)},
               "scope": scope_txt, "derived_from": lsrc, "written": now,
               "rule": "the combined final labels, keys whose cell is in scope (run-table real_batch); counts left out: recompute from labels"}
    rows = [r for r in rt["cells"] if r["cell"] in cells]
    assert len(rows) == len({r["cell"] for r in rows}) == len(cells), "the in-scope run table has a duplicate or missing cell"
    rt_doc = {"cells": rows, "dropped": [d for d in rt["dropped"] if d["cell"] in cells],
              "staged_sets": {s: v for s, v in rt["staged_sets"].items() if v["batch"].split("_", 1)[1] in scope},
              "scope": scope_txt, "derived_from": rsrc, "written": now, "rule": "the combined final run table, in-scope cells only"}
    write_new(OUT / f"labels_nm360_xerr_{tag}.json", lab_doc)
    write_new(OUT / f"run_table_nm360_xerr_{tag}.json", rt_doc)
    # exclusions records: superseded originals dropped before counting (the loader ignores supersession records)
    by_man = {}
    for name, row, r in superseded_rows(quarters, mans):
        cause = lab["excluded"][r["key"]]["exclusion_cause"]
        by_man.setdefault(name, []).append({
            "key": r["key"], "cell": row["task"], "manifest_key": f"{row['launch']}/{row['rel']}", "sha256": row["sha256"],
            "launch": row["launch"], "state": row["state"], "excluded": True, "exclusion_cause": cause,
            "rerun_in": {"launch": r["replacement"]["launch"], "staged_set": r["replacement"]["staged_set"], "eval_sha256": r["replacement"]["eval_sha256"]}})
    for name, ex in by_man.items():
        mp = pin(f"config/{name}")
        full = git("rev-parse", mp["commit"])
        write_new(OUT / f"exclusions_nm360_{name[len('manifest_nm360_'):-5]}_superseded_{tag}.json",
                  {"batch": next(bid for bid, _, n, _ in mans if n == name), "launch": ex[0]["launch"], "manifest": f"{mp['path']}@{full}",
                   "selected_on": {"superseded_by_replacement": pin(SUPERSESSION["s5"])}, "excluded": ex, "written": now})
    # approach provenance: the superseded originals' rows marked excluded in a derived copy of their file (the loader refuses two
    # differing rows for one cell; the replacement's row is in the rerun's own file)
    afiles = {q.name: json.loads(q.read_text()) for q in sorted(HYG.glob("approach_provenance_nm360_*.json")) if "_superseded_" not in q.name}
    afiles = {n: d for n, d in afiles.items() if d["batch"] in scope}
    marked = {}
    for _, row, r in superseded_rows(quarters, mans):
        o, rp = r["original"], r["replacement"]
        old = [(n, x) for n, d in afiles.items() for x in d["cells"] if x["cell"] == row["task"] and x["sim"]["eval_sha256"] == o["eval_sha256"]]
        new = [(n, x) for n, d in afiles.items() for x in d["cells"] if x["cell"] == row["task"] and x["sim"]["eval_sha256"] == rp["eval_sha256"]]
        assert len(old) == 1 and len(new) == 1 and old[0][0] != new[0][0], (r["key"], len(old), len(new))
        marked.setdefault(old[0][0], {})[row["task"]] = new[0][0]
    for n, cs in marked.items():
        d = json.loads(json.dumps(afiles[n]))
        for x in d["cells"]:
            if x["cell"] in cs:
                x["excluded"] = True
                x["excluded_reason"] = f"superseded: its sim run was replaced by the rerun ({SUPERSESSION['s5']}); the cell's row is {cs[x['cell']]}'s"
        d["covers"]["cells"] = [c for c in d["covers"]["cells"] if c not in cs]
        d["derived_from"], d["written"] = pin(f"results/hygiene/{n}"), now
        write_new(OUT / f"{n[:-5]}_superseded_{tag}.json", d)
    print(json.dumps({"registry_entries": len(ent), "approach_rows_marked_superseded": {n: sorted(cs) for n, cs in marked.items()}, "labels": len(lab_doc["labels"]), "labels_dropped_out_of_scope": len(lab["labels"]) - len(lab_doc["labels"]),
                      "run_table_rows": len(rows), "exclusions": {n: len(x) for n, x in by_man.items()}}, indent=1))


def cmd_fragment(st_dir: str, quarters: list, tag: str):
    import modal

    scope = scope_of(st_dir, quarters)
    mans = manifests(scope)
    cells = in_scope_cells(scope, mans)
    reg_pin, lab_pin, rt_pin = (pin(f"results/hygiene/{n}_{tag}.json") for n in ("registry_S_newmains_xerr", "labels_nm360_xerr", "run_table_nm360_xerr"))
    lab = json.loads((C.REPO / lab_pin["path"]).read_text())
    rt = {r["cell"]: r for r in json.loads((C.REPO / rt_pin["path"]).read_text())["cells"]}
    assert set(rt) == cells and {k.split("|", 1)[1] for k in lab["labels"]} <= cells, "the pinned run table / labels are not the in-scope cells"
    sup = superseded_rows(quarters, mans)
    excl_pin = {name: pin(f"results/hygiene/exclusions_nm360_{name[len('manifest_nm360_'):-5]}_superseded_{tag}.json") for name in {n for n, _, _ in sup}}
    v4_raw = subprocess.run(["git", "show", f"{V4['commit']}:{V4['path']}"], capture_output=True, check=True, cwd=C.HERE).stdout
    v4 = json.loads(v4_raw)
    v4_pin = V4 | {"sha256": hashlib.sha256(v4_raw).hexdigest()}
    sf_cells = {c for q, cs in v4["cells"].items() if q in quarters for c in cs}
    work = modal.Volume.from_name(WORK, version=2)
    appr_files = {q.name: json.loads(q.read_text()) for q in sorted(HYG.glob("approach_provenance_nm360_*.json"))
                  if "_superseded_" not in q.name or q.name.endswith(f"_superseded_{tag}.json")}
    for n in [n for n in appr_files if n.endswith(f"_superseded_{tag}.json")]:  # a derived copy replaces its original
        orig = appr_files.pop(appr_files[n]["derived_from"]["path"].rsplit("/", 1)[1])
        assert pin(f"results/hygiene/{n}")["sha256"] and orig["batch"] == appr_files[n]["batch"]
    appr_files = {n: d for n, d in appr_files.items() if d["batch"] in scope}

    def vol_pin(path: str):
        try:
            raw = b"".join(work.read_file(path))
        except Exception as e:  # noqa: BLE001 (only a missing file is expected; anything else is reported, not hidden)
            return None if type(e).__name__ in ("FileNotFoundError", "NotFoundError") else {"error": type(e).__name__}
        return {"path": f"/work/{path}", "sha256": hashlib.sha256(raw).hexdigest()}

    arms, excluded_arms, missing, expansion, folder_world, approach_gaps = {}, {}, [], {}, {}, {}
    counted = {w: [] for w in ("run1", "honest_a", "honest_b", "run2", "sim")}
    staging_ex, labels_x = [], []
    for bid, b in scope.items():
        att = json.loads((HYG / f"attestations_{b['pool']}_{bid}.json").read_text())["attestations"]
        for t in b["targets"]:
            tv, sg = C.targets_for(b)[t], b["stage"][t]
            man_pin = pin(f"config/{tv['manifest']}")
            man = json.loads((C.HERE / "config" / tv["manifest"]).read_text())
            entry = man_pin | {"keys_path": ["files"], "format": "d9_files"}
            if tv["manifest"] in excl_pin:
                entry["exclusions"] = excl_pin[tv["manifest"]]
            if t.endswith("_sim") or "_sim_s" in t:
                counted["sim"].append(entry)
            elif "_run2" in t:
                counted["run2"].append(entry)
            else:
                for w in ("run1", "honest_a", "honest_b"):
                    counted[w].append(entry)
            rec = json.loads((C.HERE / "results" / sg["record"]).read_text())
            k = next(x for x in rec if x not in ("git", "stamp"))
            by_path = {f["path"]: f for f in man["files"]}
            ex = {}
            for key, why in rec[k]["c_gate"]["excluded"].items():
                f = by_path[key.split("#", 1)[0]]
                ex[f"{world_of(f['arm_dir'], t)}|{f['task']}"] = why
            staging_ex.append({"record": pin(f"results/{sg['record']}"), "exclusions": ex, "for_manifest": man_pin["path"]})
            for arm in [a for a in sg["epochs"] if a != "repro-c"]:
                folder = f"{sg['out']}__{SFX[arm]}"
                tasks = sorted({f["task"] for f in man["files"] if f["arm_dir"] == arm})
                if tasks and all(c in sf_cells for c in tasks):  # every key of the folder is a v4 setup failure: excluded, no scores expected
                    excluded_arms[folder] = {"keys": [f"{world_of(arm, t)}|{c}" for c in tasks], "excluded_by": v4_pin}
                    continue
                ssha = sg["staged_inputs_sha256"][f"{arm}.jsonl"]
                folder_world[folder] = (world_of(arm, t), ssha)
                ev = {e["check"]: e for e in att[ssha]}
                gate = ev["longest_epoch_h200_gate"]
                extra = [x["path"] for x in gate.get("evidence_extra", [])]
                ext = sorted([gate["evidence"]["path"]] + [p for p in extra if "/extract_chain_" in p])
                hyg = {"longest_epoch_h200_gate": {"extract": [pin(p) for p in ext],
                                                   "compare": pin(next(p for p in extra if "/compare_repro_" in p)),
                                                   "staging": pin(next(p for p in extra if "/staged_stats_" in p))},
                       "corrected_render": pin(ev["corrected_render"]["evidence"]["path"]),
                       "tool_error_table": pin(ev["tool_error_table"]["evidence"]["path"])}
                assert all(e["pass"] for e in att[ssha]), f"{folder}: a failing attestation"
                w_ = world_of(arm, t)
                if w_ in ARMS:  # attack folders: approach / system-prompt provenance with full coverage of the staged cells (loader ~1692, ~1880)
                    names = sorted(n for n, d in appr_files.items() if ssha in d["staged_shas"])
                    hyg["approach_system_prompt_check"] = [pin(f"results/hygiene/{n}") for n in names]
                    rows_ = {}
                    for n in names:
                        d = appr_files[n]
                        assert d["pass"] is True, f"{folder}: {n} pass != true"
                        kept = []
                        for r in d["cells"]:
                            if r.get("excluded") is True:
                                continue
                            assert all(f in r for f in APPROACH_CELL_FIELDS) and rows_.setdefault(r["cell"], r) == r, f"{folder}: {n} row {r['cell']}"
                            kept.append(r["cell"])
                        assert sorted(kept) == sorted(d["covers"]["cells"]), f"{folder}: {n} cells vs covers.cells"
                    staged_cells = {f["task"] for f in man["files"] if f["arm_dir"] == arm} - {k.split("|", 1)[1] for k in ex if k.startswith(f"{w_}|")}
                    uncovered = sorted(c for c in staged_cells if c not in rows_)
                    if uncovered:  # listed, never invented: the loader's final mode stops on these (needs d9 rows or a 09 ruling)
                        approach_gaps[folder] = {c: rt[c]["sim_status"] for c in uncovered}
                sst = json.loads((C.REPO / hyg["longest_epoch_h200_gate"]["staging"]["path"]).read_text())
                (hit,) = [r for r in sst["sets"] if r["staged_sha"] == ssha]
                assert hit["max_staged_tokens"] <= 32768 and hit["n_epochs_over_32768"] == 0, f"{folder}: needs own-longest read-twice evidence"
                p2 = vol_pin(f"scaleup/{folder}/p2/manifest_p2.json")
                sol_dir = "sol_r2" if vol_pin(f"scaleup/{folder}/sol_r2/manifest_sol.json") else "sol"  # the loader refuses sol/ when sol_r2/ exists
                sol = vol_pin(f"scaleup/{folder}/{sol_dir}/manifest_sol.json")
                if sol and vol_pin(f"scaleup/{folder}/{sol_dir}/SUPERSEDED.json"):
                    sol = {"error": f"{sol_dir}/ is SUPERSEDED"}
                for name, v in (("p2_manifest", p2), ("sol_manifest", sol)):
                    if v is None or "error" in v:
                        missing.append({"folder": folder, "missing": name, "detail": v})
                arms[folder] = {"p2_manifest": p2, "sol_manifest": sol, "hygiene": hyg}
                exp = {n: v for n, v in (("famexp", vol_pin(f"scaleup/{folder}/famexp_r2/manifest_famexp.json") or vol_pin(f"scaleup/{folder}/famexp/manifest_famexp.json")),
                                          ("own_a", vol_pin(f"scaleup/{folder}/own_a/manifest_owna.json"))) if v}
                if exp:
                    expansion[folder] = exp
        lp = HYG / f"labels_{b['pool']}_{bid}.json"
        if lp.exists():
            labels_x.append(pin(f"results/hygiene/{lp.name}"))
    # run-table cross-check: per-quarter tables vs the in-scope combined table; superseded originals may differ, nothing else
    sup_cells = {row["task"] for _, row, _ in sup}
    rt_x, rt_omitted = [], []
    for q in quarters + [f"{q}r" for q in quarters]:
        p = HYG / f"run_table_nm360_{q}.json"
        if not p.exists():
            continue
        qrows = json.loads(p.read_text())["cells"]
        diff = sorted(r["cell"] for r in qrows if r["cell"] not in rt or any(r.get(f) != rt[r["cell"]].get(f) for f in RT_F))
        assert set(diff) <= sup_cells, (p.name, diff)
        if diff:
            rt_omitted.append({"table": pin(f"results/hygiene/{p.name}"), "rows_agreeing": len(qrows) - len(diff), "superseded_rows_differing": diff,
                               "reason": "its superseded originals' rows name the original launch; the combined table names the replacement (by design)"})
        else:
            rt_x.append(pin(f"results/hygiene/{p.name}"))
    # the loader's checks, mirrored (run_xerr_scaleup.py @ aab87af1a: counted ~763-830, staging ~892-962, labels ~1026-1070,
    # run table ~1135-1182), on the pinned files and keys only
    nraw = subprocess.run(["git", "show", f"{NMREG['commit']}:{NMREG['path']}"], capture_output=True, check=True, cwd=C.HERE).stdout
    assert hashlib.sha256(nraw).hexdigest() == NMREG["sha256"], "the loader's pinned new-mains registration differs"
    cells_reg = {c["key"] for c in json.loads(nraw)["cells"]}
    mk = lambda r: f"{r['launch']}/{r['rel']}"  # noqa: E731
    excl_keys, excl_man, cnt = {}, {}, {}
    for w, entries in counted.items():
        cells_w = set()
        for e in entries:
            top = json.loads((C.REPO / e["path"]).read_text())
            assert isinstance(top["counted_attempts_without_eval"] or {}, dict), e["path"]
            rows = [r for r in top["files"] if r["world"] == w]
            if e.get("exclusions") is not None:
                xr = json.loads((C.REPO / e["exclusions"]["path"]).read_text())
                assert xr["manifest"].split("@", 1)[0] == e["path"], f"counted {w}: an exclusion record for another manifest"
                drop = {x["manifest_key"]: x for x in xr["excluded"] if x.get("excluded") is True}
                hit = [r for r in rows if mk(r) in drop]
                assert len(hit) == len(xr["excluded"]) and all(r["sha256"] == drop[mk(r)]["sha256"] and drop[mk(r)]["key"] == f"{w}|{r['task']}" for r in hit), \
                    f"counted {w}: the exclusion record does not match its manifest's rows"
                for r in hit:
                    excl_keys[f"{w}|{r['task']}"], excl_man[f"{w}|{r['task']}"] = drop[mk(r)]["exclusion_cause"], e["path"]
                rows = [r for r in rows if mk(r) not in drop]
            for r in rows:
                c = r["task"]
                assert c in cells_reg and r["stratum"] == "S_newmains", f"counted {w}: {c} not a registered cell, or its stratum is not S_newmains"
                assert r.get("launch_task") == (f"{c}_r2" if w == "run2" else c), f"counted {w}: {c} launch_task {r.get('launch_task')!r}"
                assert c not in cells_w, f"counted {w}: {c} double-filled across counted manifests"
                cells_w.add(c)
        cnt[w] = cells_w
    assert set(excl_keys) == {r["key"] for _, _, r in sup}, "the exclusions records do not drop exactly the superseded originals"
    stg, dropped_run = {}, []
    for se in staging_ex:
        rec = json.loads((C.REPO / se["record"]["path"]).read_text())
        exp, got = {}, {}
        for set_name, v in rec.items():
            if set_name == "git":
                continue
            for arm, cv in v["assembled"]["coverage"].items():
                w = "sim" if arm.startswith("sim_attack") else ("run2" if "run2" in set_name else "run1") if arm == "attack" else arm
                for why, n in cv["runs excluded"].items():
                    exp[(w, why)] = exp.get((w, why), 0) + n
        for k, why in se["exclusions"].items():
            got[(k.split("|", 1)[0], why)] = got.get((k.split("|", 1)[0], why), 0) + 1
            if k in excl_keys and se["for_manifest"] == excl_man[k]:
                dropped_run.append(k)  # the dropped run itself: never applied (Revision 6o)
                continue
            assert stg.setdefault(k, why) == why, f"staging exclusions: {k.split('|', 1)[0]}|<cell> listed with two reasons"
        assert got == exp, (se["record"]["path"], "staging exclusions disagree with the record's per-arm 'runs excluded' counts")
    for r in v4["records"]:
        assert r.get("staging_exclusion") is True and r.get("policy_called") is False and r["reason"] == v4["reason"], "v4: a row is not a staging exclusion without a policy call"
        stg.setdefault(f"{r['world']}|{r['cell']}", r["reason"])
    applied = {k for k in stg if k.split("|", 1)[1] in cnt.get(k.split("|", 1)[0], set())}
    for k in applied:
        cnt[k.split("|", 1)[0]].discard(k.split("|", 1)[1])
    craw = subprocess.run(["git", "show", f"{CENSUS['commit']}:{CENSUS['path']}"], capture_output=True, check=True, cwd=C.HERE).stdout
    ckeys = {f"{r['arm']}|{r['task']}" for r in json.loads(craw)["runs"]}
    status_rows = []
    for k in [k for q in quarters for k in RERUN_KIND.get(q, ())]:
        w, c = k.split("|", 1)
        assert k in ckeys and k in applied and c not in cnt[w], f"{k}: not a census rerun, or not excluded at staging, or still counted"
        status_rows.append({"key": k, "status": RERUN_STATUS, "staging_reason": stg[k], "rerun_census": CENSUS | {"sha256": hashlib.sha256(craw).hexdigest()},
                            "ruling": "09 via d9 10-15 ~10:45: stays uncounted; no rerun enters a manifest"})
    labels = lab["labels"]
    for lx in labels_x:
        for k, v in json.loads((C.REPO / lx["path"]).read_text())["labels"].items():
            assert k in excl_keys or labels.get(k) == v, f"label cross-check: {k.split('|', 1)[0]}|<cell> disagrees with the combined labels"
    stray = sorted(k for k, v in labels.items() if k.split("|", 1)[0] in ARMS and k.split("|", 1)[1] not in cnt[k.split("|", 1)[0]] and v != "run_failed" and k not in applied)
    badlab = sorted(f"{w}|{c}" for w in ARMS for c in cnt[w] if labels.get(f"{w}|{c}") not in LABEL_ENUM)
    assert not stray and not badlab, {"labels_for_uncounted_runs": stray, "counted_without_label": badlab}
    nostaged = sorted(f for f, (w, s) in folder_world.items() if lab["staged_sets"].get(s, {}).get("world") != w)
    assert not nostaged, f"label staged_sets lack these folders' staged sets: {nostaged}"
    rt_doc = json.loads((C.REPO / rt_pin["path"]).read_text())
    for c, row in rt.items():
        assert c in cells_reg and all(row.get(f) is not None for f in ("attack_pair_launch", "sim_status", "sim_launch")), f"run table: {c}"
        assert row["attack_pair_launch"] in ("same", "separate") and row["sim_status"] in SIM_STATUS, f"run table: {c} outside its enum"
        if row["sim_status"] == "ran":
            assert row["sim_launch"] != "NOT_RUN", f"run table: {c} sim ran but sim_launch is NOT_RUN"
        else:
            assert row["sim_launch"] == "NOT_RUN" and labels.get(f"sim|{c}") != "attempted" and c not in cnt["sim"], f"run table: {c} sim not run"
    assert all(all(isinstance(d.get(f), str) for f in ("cell", "stage", "reason")) for d in rt_doc["dropped"]), "run table: a dropped entry"
    src = git("show", f"{LOADER[0]}:{LOADER[1]}")
    assert all(f'r"{x}"' in src for xs in FOLDER_PIECES.values() for x in xs), "09's folder grammar changed: update FOLDER_PIECES"
    fre, rns, fu, s4 = (re.compile("".join(FOLDER_PIECES[k])) for k in ("FOLDER_RE", "R_NOT_SIM", "NM360_FU_RE", "S4_FOLDER"))
    canonical = lambda f: (bool(fre.fullmatch(f)) and not rns.match(f)) or bool(fu.fullmatch(f))  # noqa: E731 (the loader's folder_canonical)
    folder_check = {"loader": f"{LOADER[1]} @ {git('rev-parse', LOADER[0])[:12]}", "rule": "folder_canonical (FOLDER_RE minus R_NOT_SIM, or NM360_FU_RE)",
                    "rejected": sorted(f for f in arms if not canonical(f)), "s4_folders": sorted(f for f in arms if s4.match(f))}
    assert not folder_check["rejected"] and not folder_check["s4_folders"], folder_check
    now = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S PT")
    frag = {"stratum": "S_newmains", "scope": {"quarters": quarters, "batches": sorted(scope), "s4": "not in this fragment: its own batch later (S4r routed 10-15)"},
            "arms": arms, "excluded_arms": excluded_arms, "missing_scorer_manifests": missing, "approach_provenance_gaps": approach_gaps, "expansion_manifests_available": expansion,
            "counted": counted, "labels": [lab_pin], "labels_crosscheck": labels_x, "run_table": [rt_pin], "run_table_crosscheck": rt_x,
            "run_table_crosscheck_omitted": rt_omitted, "registry": {"S_newmains": reg_pin}, "staging_exclusions": staging_ex,
            "setup_failure_exclusions": [v4_pin], "status_rows": status_rows, "supersession": [pin(SUPERSESSION[q]) for q in quarters if q in SUPERSESSION],
            "loader_checks_run_here": {"counted_cells": {w: len(v) for w, v in cnt.items()}, "staging_exclusions_applied": len(applied),
                                       "excluded_before_counting": sorted(excl_keys),
                                       "staging_exclusions_of_dropped_runs": sorted(set(dropped_run)), "labels_for_uncounted_runs": 0, "counted_without_label": 0},
            "folder_re_check": folder_check, "d9_commit": git("rev-parse", "HEAD")[:12], "written": now,
            "note": "PRELIMINARY spec fragment for the pooled (extended) run; keys, paths and hashes only"}
    write_new(HYG / f"xerr_spec_fragment_S_newmains_{tag}.json", frag)
    print(json.dumps({"arms": len(arms), "excluded_arms": sorted(excluded_arms), "missing": missing, "approach_provenance_gaps": approach_gaps, "expansion": len(expansion),
                      "counted_manifests": {w: len(v) for w, v in counted.items()}, "counted_cells_after_staging": {w: len(v) for w, v in cnt.items()},
                      "staging_exclusions": sum(len(e["exclusions"]) for e in staging_ex), "excluded_before_counting": len(excl_keys),
                      "run_table_crosscheck": len(rt_x), "run_table_crosscheck_omitted": [(o["table"]["path"].rsplit("/", 1)[1], o["rows_agreeing"], len(o["superseded_rows_differing"])) for o in rt_omitted],
                      "folder_re_check": folder_check}, indent=1))


if __name__ == "__main__":
    {"inputs": cmd_inputs, "fragment": cmd_fragment}[sys.argv[1]](sys.argv[2], sys.argv[3].split(","), sys.argv[4])
