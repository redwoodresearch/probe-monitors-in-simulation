"""Build INPUTS_FINAL (schema inputs_final/v1, see inputs_final.py) from a pin file, then verify it (fail closed).

The pin file (config/inputs_final_pins.json) lists, by commit, every FINAL mapping, the label overrides, the NOTEs and coverage records that make
the ledger, the relayed ledger rows that have no NOTE yet, the 25f roots, the DT records and the stage-B (RMS) pins. The ledger is built by key:
a NOTE row's status decides admitted / excluded / pending; its m_r is the row's explicit integer "m_r" (checked against 09's max rule over
m_r_provenance) or, without one, the single value its text states ("m_r = N", "m_r = max(a, b) = N"; several different values fail); its class is
the explicit "class X" in the status text, else the status wording (RL / infrastructure / standing rule); no m_r or class comes from a constant.

  <modal env python> experiments/2026-10-15_fpr_sweep_507_prep/make_inputs_final.py [--pins config/inputs_final_pins.json] [--status "FROZEN 20:30"]
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import inputs_final as IF  # noqa: E402

TEMPLATE = ("+311 Q1–Q4 and nm360 S1–S5 sim-honest FINAL as of <cut-off PT>. Replacements admitted by key: <list, with class>. Excluded cells by key: "
            "<list, with class and reason>. Pending at cut-off: <list or 'none'>. Epoch-2 honest pools in per Append 38 §4 @ <sha> (or 'epoch-2 not in this build (second refresh)'). S4 is a labelled batch; every pooled headline is also shown without S4. nm360 Sol (a) / c_asis / own_frac rows: held-out gate "
            "<verified @ sha | pending>. PRELIMINARY; POST HOC where marked.")
DT_REC = "experiments/2026-10-14_p2_dt_scaleup/results"


def pin(commit, path):
    return {"commit": commit, "path": path, "sha256": IF.sha(IF.git_show(commit, path))}


CLASSES = ("terminal", "standing", "RL", "infrastructure", "unknown")


def attempt_value(key, att, v, recorded=False):
    """One failed attempt's tool-event count (09's rule, 2e 17:2x): tool_events_final where present, else events_by_type.tool; both present and
    different = error; recorded=True also requires the entry's own 'value' to equal it."""
    t, e = v.get("tool_events_final"), v.get("events_by_type_tool")
    if t is not None and e is not None and t != e:
        raise SystemExit(f"{key} {att}: tool_events_final {t} != events_by_type.tool {e} (09's rule: error)")
    val = t if t is not None else e
    if val is None or (recorded and val != v.get("value")):
        raise SystemExit(f"{key} {att}: no attempt value, or it differs from the recorded value")
    return int(val)


def mr_inputs(key, x):
    """A NOTE's 'm_r_inputs' entry (09's rule, 2e 17:2x): per failed attempt, tool_events_final where present, else events_by_type.tool; both
    present and different = error; m_r = max over failed attempts."""
    vals = [attempt_value(key, att, v, recorded=True) for att, v in x["failed_attempts"].items()]
    if not isinstance(x["m_r"], int) or x["m_r"] != max(vals):
        raise SystemExit(f"{key}: m_r_inputs m_r {x['m_r']} is not the max over failed attempts {vals}")
    return x["m_r"]


def note_mr(key, x, mr_in=None):
    """The row's m_r: explicit integer (== max over m_r_provenance when present), else the one value the text states, else the NOTE's m_r_inputs
    value; when the NOTE has m_r_inputs for the key, every value found must equal it."""
    mr = _note_mr(key, x, mr_in is not None)
    if mr is None:
        return mr_in
    if mr_in is not None and mr != mr_in:
        raise SystemExit(f"{key}: row m_r {mr} differs from the supplied value {mr_in} (the NOTE's m_r_inputs or the relayed ruling)")
    return mr


def _note_mr(key, x, optional):
    if isinstance(x, dict) and "m_r" in x:
        mr = x["m_r"]
        if not isinstance(mr, int) or isinstance(mr, bool):
            raise SystemExit(f"{key}: m_r field is not an integer")
        prov = x.get("m_r_provenance")
        if prov is not None and mr != max(attempt_value(key, att, v) for att, v in prov.items()):
            raise SystemExit(f"{key}: m_r {mr} is not the max over m_r_provenance (09's rule)")
        return mr
    t = json.dumps(x)
    vals = set()
    for a_, b_, c_ in re.findall(r"\bm_r ?= ?max\((\d+), ?(\d+)\) ?= ?(\d+)", t):
        if int(c_) != max(int(a_), int(b_)):
            raise SystemExit(f"{key}: 'm_r = max({a_}, {b_}) = {c_}' breaks the max rule")
        vals.add(int(c_))
    for c_, a_, b_ in re.findall(r"\bm_r ?= ?(\d+) ?= ?max\((\d+), ?(\d+)\)", t):
        if int(c_) != max(int(a_), int(b_)):
            raise SystemExit(f"{key}: 'm_r = {c_} = max({a_}, {b_})' breaks the max rule")
        vals.add(int(c_))
    vals |= {int(v) for v in re.findall(r"\bm_r ?= ?(\d+)", t)}
    if not vals and optional:
        return None
    if len(vals) != 1:
        raise SystemExit(f"{key}: m_r ambiguous or missing in the NOTE text: {sorted(vals)}")
    return vals.pop()


def note_class(key, st):
    """Explicit 'class X' in the status text first (all mentions must agree); else the status wording; else fail."""
    ex = set(re.findall(r"\bclass (unknown|terminal|standing|RL|infrastructure)\b", st))
    if len(ex) > 1:
        raise SystemExit(f"{key}: several classes in the status text {sorted(ex)}")
    if ex:
        return ex.pop()
    if "infrastructure" in st or "RL" in st:
        return "infrastructure"
    if "standing rule" in st:
        return "standing"
    raise SystemExit(f"{key}: no class in the status text: {st[:120]}")


def ledger_from_notes(notes, batch_of, extra):
    """NOTE rows by key -> ledger rows. Section 'pending' -> pending; status 'replacement in' -> admitted (the original excluded); 'excluded' /
    'staging-excluded' -> excluded. Relayed rows (pin file 'ledger_extra') fill keys that have no NOTE yet; status pending | excluded | admitted."""
    rows = {"excluded": [], "replacements_admitted": [], "pending_at_cutoff": [], "structural_no_rows": []}
    seen = set()
    rel_mr = {(r["set"], r["cell_prefix"]): r["m_r"] for r in extra}
    for n in notes:
        d = json.loads(IF.git_show(n["commit"], n["path"]))
        mri = {k: mr_inputs(k, x) for sec, body in d.items() if sec.startswith("m_r_inputs") for k, x in body.items()}
        for sec, body in d.items():
            if not isinstance(body, dict) or sec in ("mapping", "supersedes_note", "replacement_accounting") or sec.startswith("m_r_inputs"):
                continue
            if sec == "structural_no_row":  # 09's class @ 51a612b29 (e.g. nm014): listed by key with the NOTE's verbatim text; never a ledger row
                rows["structural_no_rows"] += [{"key": k, "text": x, "note": f"{n['path']} @ {n['commit']}"} for k, x in body.items()]
                continue
            for key, x in body.items():
                if "|" not in key:
                    continue
                st = x.get("status", "") if isinstance(x, dict) else str(x)  # a pending row may be a bare status string
                c0_ = key.split("|", 1)[1].split("_", 1)[0]
                rel = rel_mr.get(("S_newmains" if key.startswith("simh_nm") else "S_new", c0_))
                mr = note_mr(key, x, mri.get(key) if key in mri else rel)  # a NOTE row without m_r takes the relayed ruling's (listed in the pin file)
                if rel is not None and rel != mr:  # every supplied value must agree (row, m_r_inputs, relayed)
                    raise SystemExit(f"{key}: relayed m_r {rel} differs from the NOTE's {mr}")
                cls = note_class(key, st)
                if isinstance(x, dict) and x.get("class"):  # an explicit class field must agree
                    cf = str(x["class"]).split()[0]
                    if cf not in CLASSES or cf != cls:
                        raise SystemExit(f"{key}: class field {x['class']!r} disagrees with the status text ({cls})")
                c0 = key.split("|", 1)[1].split("_", 1)[0]
                setn = "S_newmains" if key.startswith("simh_nm") else "S_new"
                bt = batch_of[(setn, c0)]
                row = {"set": setn, "batch": bt, "cell_prefix": c0, "key": key, "class": cls, "m_r": mr, "reason": st[:300], "ruling": f"NOTE {n['path']} @ {n['commit']}"}
                if sec == "pending":
                    rows["pending_at_cutoff"].append(row)
                elif st.startswith("replacement in") or (sec.startswith("sensitivities") and re.search(r"\badmitted\b", st) and "not admitted" not in st
                                                         and "excluded" not in st.lower()):  # 09's 18:5x wording: "...; admitted, class unknown ..."
                    rows["excluded"].append(row)  # the original is excluded (superseded status row); its replacement is admitted
                    rows["replacements_admitted"].append({k: row[k] for k in ("set", "batch", "cell_prefix", "key", "class", "ruling")} | {"root": x.get("root") or ""})
                elif "excluded" in st.lower() or sec == "excluded":
                    rows["excluded"].append(row)
                else:
                    raise SystemExit(f"NOTE {n['path']}: cannot classify {key}: {st[:120]}")
                seen.add((setn, c0))
    for r in extra:  # relayed rulings for keys without a NOTE yet (listed in the pin file with their source)
        if (r["set"], r["cell_prefix"]) in seen:
            continue
        if r["status"] not in ("pending", "excluded", "admitted") or r["class"] not in CLASSES or not isinstance(r["m_r"], int):
            raise SystemExit(f"ledger_extra {r['cell_prefix']}: status, class or m_r invalid")
        row = {k: r[k] for k in ("set", "batch", "cell_prefix", "class", "m_r", "reason", "ruling")}
        rows["pending_at_cutoff" if r["status"] == "pending" else "excluded"].append(row)
        if r["status"] == "admitted":  # the original is excluded; the replacement root comes from the pinned mapping (main())
            rows["replacements_admitted"].append({k: r[k] for k in ("set", "batch", "cell_prefix", "class", "ruling")} | {"root": ""})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pins", default="config/inputs_final_pins.json")
    ap.add_argument("--status", default="DRAFT (unfrozen)")
    ap.add_argument("--name", default="", help="file tag instead of DRAFT/FROZEN (e.g. REPRO_e0b803833)")
    a = ap.parse_args()
    P = json.loads((HERE / a.pins).read_text())
    sets, batch_of = {}, {}
    for st in ("S_new", "S_newmains"):
        sm, roots = {}, {}
        for bt, e in P["mappings"][st].items():
            b = IF.git_show(e["commit"], f"{IF.MAPDIR}/{e['file']}.json")
            d = json.loads(b)
            sm[bt] = {"commit": e["commit"], "file": e["file"], "sha256": IF.sha(b), "label": d.get("label")}
            if e.get("label_override"):
                sm[bt]["label_override"] = pin(e["label_override"]["commit"], e["label_override"]["path"])
            roots[bt] = [r["arm"] for r in d["roots"]]
            for k in (d.get("rl_replacement") or {}).get("rl_ruling_keys", []) if isinstance(d.get("rl_replacement"), dict) else []:
                batch_of[(st, k.split("|", 1)[1].split("_", 1)[0])] = bt
            for c0 in e.get("cells", []):
                batch_of[(st, c0)] = bt
        sets[st] = {"sol_mappings": sm, "roots": roots, "file_maps": {"commit": P["file_maps_commit"]},
                    "manifest_copies": {r: {k: (IF.sha(IF.CP.read(IF.copy_path(r, k))) if IF.copy_path(r, k) and k not in P.get("families_pending", {}).get(st, {}).get(r, [])
                                                else None) for k in ("p2", "famexp", "own_a")}
                                        for rs in roots.values() for r in rs}}
    for st in ("S_new", "S_newmains"):  # a scorer that missed the cut-off for a root: listed per family (famexp | own_a | own_all | dt), by root
        fp_ = P.get("families_pending", {}).get(st, {})
        sets[st]["families_pending"] = fp_
        sets[st]["dt_roots_pending"] = sorted(set(P.get("dt_roots_pending", {}).get(st, [])) | {r for r, v in fp_.items() if "dt" in v})
    sets["S_new"] |= {"honest_a_25f": P["honest_a_25f"], "dt_sim_honest": [pin(c, f"{DT_REC}/{n}.json") | {"record": n} for c, n in P["dt"]["S_new"]]}
    sets["S_newmains"] |= {"honest_a_25f": [], "dt_sim_honest": [pin(c, f"{DT_REC}/{n}.json") | {"record": n} for c, n in P["dt"]["S_newmains"]],
                           "membership710": {"path": "config/membership710.json", "sha256": IF.sha((HERE / "config" / "membership710.json").read_bytes())},
                           "s4_source_map": pin(P["s4_source_map"]["commit"], P["s4_source_map"]["path"])}
    sets["S_old"] = {"e2e_config": pin(P["s_old_e2e_config"]["commit"], P["s_old_e2e_config"]["path"]), "dt_sim_honest": [pin(c, f"{DT_REC}/{n}.json") | {"record": n} for c, n in P["dt"]["S_old"]]}
    for x in sets["S_new"]["dt_sim_honest"] + sets["S_newmains"]["dt_sim_honest"] + sets["S_old"]["dt_sim_honest"]:
        x.pop("path")
    notes = [pin(n["commit"], n["path"]) for n in P["notes"]]
    led = ledger_from_notes(notes, batch_of, P["ledger_extra"])
    reps = {(x["set"], x["cell_prefix"]): x for x in led["replacements_admitted"]}
    ovr = P.get("replacement_root_by_key", {})  # "<set>|<batch>|<cell>": root, for a batch with more than one replacement root (e.g. _rlr and _rlr2)
    claimed = {(k.split("|")[0], k.split("|")[1]): set() for k in ovr}
    for k, r in ovr.items():
        claimed[(k.split("|")[0], k.split("|")[1])].add(r)
    for (st, c0), x in reps.items():  # the replacement root: the pinned override, else the batch's one remaining _rlr root in the pinned mapping
        k = f"{st}|{x['batch']}|{c0}"
        if k in ovr:
            if ovr[k] not in sets[st]["roots"][x["batch"]]:
                raise SystemExit(f"{k}: override root {ovr[k]} is not in the pinned {x['batch']} mapping")
            x["root"] = ovr[k]
            continue
        cands = [r for r in sets[st]["roots"][x["batch"]] if "_rlr" in r and r not in claimed.get((st, x["batch"]), set())]
        if len(cands) > 1:
            raise SystemExit(f"{c0}: more than one replacement root in {x['batch']} ({cands}); pin replacement_root_by_key")
        x["root"] = cands[0] if cands else f"awaiting the {x['batch']} FINAL mapping with its replacement root"  # verify() fails until it is pinned
    m = {"schema": "inputs_final/v1", "status": a.status, "cutoff_pt": P["cutoff_pt"], "freeze_pt": P["freeze_pt"], "label": {"template": TEMPLATE, "filled": None},
         "sets": sets, "ledger": {"source": "NOTEs by key (m_r from each row's text) + relayed rulings for keys without a NOTE", "notes": notes,
                                  "coverage_records": [pin(c["commit"], c["path"]) for c in P["coverage_records"]], **led,
                                  "empty_replacement_roots": [x | {"evidence": [pin(e["commit"], e["path"]) for e in x["evidence"]]} for x in P.get("empty_replacement_roots", [])]},
         "epoch2_honest": P["epoch2_honest"] | ({"artifact": pin(P["epoch2_honest"]["artifact"]["commit"], P["epoch2_honest"]["artifact"]["path"])}
                                                 if P["epoch2_honest"].get("artifact") else {}), "nm360_fold_extension": P["nm360_fold_extension"], "gapfill_sim2": P["gapfill_sim2"],
         "sel97": P["sel97"] | {k: pin(P["sel97"][k]["commit"], P["sel97"][k]["path"]) for k in ("cells", "cells_vendored", "prep_config") if P["sel97"].get(k)},
         "extensions": P.get("extensions", {"S_paper493": "not in this build; second refresh", "own_all_L48": {"status": "not in this build", "pointer": None}}),
         # own_all pointer pins re-hashed into the manifest (git-side sha256 of each record)
         "rms": {k: (pin(v["commit"], v["path"]) | {kk: vv for kk, vv in v.items() if kk in ("key", "n", "lists")} if isinstance(v, dict) and "commit" in v else v)
                 for k, v in P["rms"].items() if k not in ("arm_pin_sources", "H_B")}}
    m["rms"]["arm_pin_sources"] = [pin(v["commit"], v["path"]) | {kk: vv for kk, vv in v.items() if kk in ("kind", "key")} for v in P["rms"]["arm_pin_sources"]]
    m["rms"]["H_B"] = {st: pin(v["commit"], v["path"]) | {kk: vv for kk, vv in v.items() if kk in ("key", "n")} for st, v in P["rms"]["H_B"].items()}
    m["rms"]["H_sim_roots"] = {"S_new": sorted({x for rs in sets["S_new"]["roots"].values() for x in rs}),
                               "S_newmains": sorted({x for bt, rs in sets["S_newmains"]["roots"].items() if not bt.endswith(("_s4d", "_s4r")) for x in rs}),
                               "rule": "RMS consumer (09 17:4x, option (a)): S_new = every pinned Q1-Q4 FINAL root; S_newmains = the S1-S3 + S5 FINAL roots only"}
    m["rms"]["scope"] = {"S_newmains": IF.RMS_SCOPE_NM}
    m["pools"] = IF.stage_a_pools(m)  # per-consumer pools: Stage A's own resolution (the RMS consumer's entries are rms.H_B / rms.H_sim_roots)
    oa = m["extensions"].get("own_all_L48") or {}
    if oa.get("status") == "in" and isinstance(oa.get("pointer"), dict):  # slot OFF: the pins pass through untouched
        m["extensions"]["own_all_L48"] = oa | {"pointer": {k: pin(v["commit"], v["path"]) for k, v in oa["pointer"].items()}}
    m["label"]["filled"] = IF.fill_label(m)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dest = HERE / "config" / (f"INPUTS_FINAL_{stamp}.json" if a.status.startswith("FROZEN") and not a.name else
                              f"INPUTS_FINAL_{a.name or 'DRAFT'}_{stamp}.json")  # the frozen file: INPUTS_FINAL_<date>_<time>.json (exclusive create)
    blob = json.dumps(m, indent=1).encode()
    dest.open("xb").write(blob)
    probs = IF.verify(m)
    print(json.dumps({"file": str(dest.relative_to(HERE)), "sha256": IF.sha(blob), "problems": probs, "pass": not probs,
                      "ledger": {k: [x.get("cell_prefix", x.get("key")) for x in v] for k, v in led.items()}}, indent=1))


if __name__ == "__main__":
    main()
