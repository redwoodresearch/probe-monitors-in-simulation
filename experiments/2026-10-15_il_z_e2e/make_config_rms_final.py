"""Stage B (RMS only; Z removed 17:1x): build the RMS export configs from INPUTS_FINAL (inputs_final/v1). Fields INPUTS_FINAL does not carry yet are read
from a side-pins file (--side), every one flagged "side pin, to move into INPUTS_FINAL"; the real run uses only the frozen file (no --side).

  python3 experiments/2026-10-15_il_z_e2e/make_config_rms_final.py --inputs=<INPUTS_FINAL json> --inputs-sha256=<its sha256> [--side=config/rms_side_pins.json] --tag=<dryrun*|final>

--tag=final requires a FROZEN file, no side pins and a filled label. Every input carries a sha256 pin, checked on read.
"""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout  # noqa: E731
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
MAPDIR = "experiments/2026-10-14_q35_sol_avg_POSTHOC/results/simh_mapping"
CELLS311 = ("d75f997f3:experiments/2026-10-14_q35_sol_avg_POSTHOC/config/scaleup_tasks_plus311_2026-10-14_125135.json",
            "69f1f75313c9610f7d16b18225c1166268c0b6e084859c641d0213f293896a8b")
ARM_OF = {"sim": "sim1", "sim2": "sim2", "run1": "run1", "run2": "run2", "run3": "run3"}
ROWS = [("Sol (a)", "R", "sol", "a_tight__avg_top2.npz", 5), ("Sol (a)", "C", "sol", "a_tight__cmd_A.npz", 5),
        ("c_asis", "R", "sol", "c_asis_tight__avg_top2.npz", 5), ("c_asis", "C", "sol", "c_asis_tight__cmd_A.npz", 5)]
for fam, nm in (("FIBS", "fibs"), ("harmfulness", "harmfulness"), ("refusal", "refusal"), ("high-stakes", "highstakes")):
    ROWS += [(fam, "R", "p2", f"{nm}__reply_mean.npz", 1), (fam, "C", "p2", f"{nm}__command.npz", 1)]
S4_IN_S_NEWMAINS = False  # 09's choice (a) for tonight (via 2e): S1–S3 + S5 on every pool and arm. True = option (b), S4 everywhere (second refresh)
S4_BATCHES = ("simh_nm360_s4d", "simh_nm360_s4r")
SCOPE_A, SCOPE_B = "S_newmains here = S1–S3 + S5 (S4d / S4r not included)", "S_newmains here = S1–S5 (S4d / S4r included)"
EXT_LABEL = "fold −1-only pool: five-model mean with all-unit per-model moments (25h §1 extension; POST HOC)"  # 09, gate c6023b805 passed (16:58)

args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
inp_b = Path(args["--inputs"]).read_bytes()
if sha(inp_b) != args["--inputs-sha256"]:
    raise SystemExit(f"INPUTS_FINAL sha256 {sha(inp_b)} differs from --inputs-sha256")
inp = json.loads(inp_b)
assert inp["schema"] == "inputs_final/v1", inp["schema"]
side = json.loads((HERE / args["--side"]).read_text()) if "--side" in args else {}
rms = inp.get("rms") or side.get("rms")
if rms is None:
    raise SystemExit("no rms block in INPUTS_FINAL and no side pins")
flag = {} if inp.get("rms") else {"side_pins": "rms block read from the side-pins file; to move into INPUTS_FINAL"}
TAG = args["--tag"]
if TAG == "final" and (not inp["status"].startswith("FROZEN") or "--side" in args or not inp["label"]["filled"]):
    raise SystemExit(f"--tag=final needs a FROZEN INPUTS_FINAL, no --side and a filled label (status {inp['status']!r})")


def blob(ent):
    if not re.fullmatch("[0-9a-f]{64}", str(ent.get("sha256"))):
        raise SystemExit(f"{ent.get('path')}: no sha256 pin")
    b = git("show", f"{ent['commit']}:{ent['path']}")
    if sha(b) != ent["sha256"]:
        raise SystemExit(f"{ent['path']}: sha256 mismatch")
    return b


SEL = re.compile(r'\.?([A-Za-z_]\w*)|\["([^"]+)"\]|\[<(primary|any)>\]')


def select(obj, key):  # a selector such as a.b["x|y"].schemes[<primary>].keys; <primary> = the parent's "primary" field, <any> = the only entry
    parent, pos = None, 0
    while pos < len(key):
        m = SEL.match(key, pos)
        if not m:
            raise SystemExit(f"unparsable selector at {pos}: {key}")
        name, lit, special = m.groups()
        if special == "any" and len(obj) != 1:
            raise SystemExit(f"<any> needs exactly one entry, found {len(obj)}: {key}")
        k = parent["primary"] if special == "primary" else next(iter(obj)) if special == "any" else name or lit
        parent, obj, pos = obj, obj[k], m.end()
    return obj


v2 = json.loads(blob(rms["five_arm_cells"]))
pins = {}
for src in rms["arm_pin_sources"]:
    for fol, e in select(json.loads(blob(src)), src["key"]).items():
        pins[fol] = {"sol": e["sol_manifest"], "p2": e["p2_manifest"]}
r36 = json.loads(blob(rms["r36"]))
fold_of_main = {}
for r in r36:
    assert fold_of_main.setdefault(r["main_task"], r["fold"]) == r["fold"]
cb = git("show", CELLS311[0])
assert sha(cb) == CELLS311[1]
cell_main = {c["key"]: c["main"] for c in json.loads(cb)["cells"]}
# 09 (17:3x): d9's S4 source map (4r part) is authoritative for every arm and pool on its cells; its 115 outside_source_map rows are orphans, never used
smap_ent = rms.get("s4_source_map") or side.get("s4_source_map")
if smap_ent is None:
    raise SystemExit("no s4_source_map pin in INPUTS_FINAL rms (or side pins)")
if not rms.get("s4_source_map"):
    flag = flag | {"side_pin_s4_source_map": "S4 source map read from the side-pins file; to move into INPUTS_FINAL"}
smap = json.loads(blob(smap_ent))
ORPH = {(r["root"], r["key"]) for r in smap["outside_source_map"]}
O_ROOTS, O_SHAS, FOUR_R = {r["root"] for r in smap["outside_source_map"]}, {r["sha256"] for r in smap["outside_source_map"]}, {c["cell"] for c in smap["cells"]}
root_of = lambda folder: folder.strip("/").split("/")[-1].split("__")[0]  # noqa: E731
S4_CELLS = FOUR_R | {r["cell"] for r in json.loads(blob(smap["s4d_part"]))["rows"]}  # 4r (61) and s4d (19), disjoint by d9's checks
assert len(S4_CELLS) == len(FOUR_R) + smap["s4d_part"]["cells"], len(S4_CELLS)


def roots_of(setname):
    S = inp["sets"][setname]
    out = []
    for batch, roots in S["roots"].items():
        m = S["sol_mappings"][batch]
        mb = git("show", f"{m['commit']}:{MAPDIR}/{m['file']}.json")
        assert sha(mb) == m["sha256"], (batch, "Sol mapping sha256")
        ents = {r["routed_folder"]: r for r in json.loads(mb)["roots"]}
        for rt in roots:
            assert rt in ents and ents[rt]["arm"] == rt, (rt, "not in the Sol mapping under its arm name")
            out.append({"root": rt, "batch": batch, "replacement_root": "_rlr_" in rt,
                        "sol": {"path": ents[rt]["path_written"], "files": ents[rt]["files"], "mapping": [f"{m['commit']}:{MAPDIR}/{m['file']}.json", m["sha256"]]},
                        "p2": {"path": f"simh/{batch}/p2/{rt}", "manifest": {"name": "manifest_p2.json", "sha256": S["manifest_copies"][rt]["p2"]}}})
    return out


def hb_keys(ent):
    keys = select(json.loads(blob(ent)), ent["key"])
    assert isinstance(keys, list) and len(keys) == len(set(keys)) == ent["n"], (len(keys), ent["n"])
    return sorted(keys)


def build(setname, listname, hb_folders_pred, extra):
    cells = v2[listname]["cells"]
    rs = v2["resolved_sources"]["cells"]
    folders, arm_rows = {}, {a: {} for a in ARM_OF.values()}
    assert len(set(cells)) == len(cells), "a cell listed twice"
    for c in cells:
        for w, a in ARM_OF.items():
            e = rs[c][w]
            assert "scorer_folder" in e and e["row_key"] == f"{w}|{c}" and c not in arm_rows[a], (c, w)  # exactly one row per cell per arm
            arm_rows[a][c] = [e["scorer_folder"], e["row_key"]]
            folders[e["scorer_folder"]] = pins[e["scorer_folder"]]
    hb = sorted(f for f in pins if hb_folders_pred(f))
    for f in hb:
        folders[f] = pins[f]
    hb_keys_ = hb_keys(rms["H_B"][setname])
    hs_all = roots_of(setname)
    excl = S4_BATCHES if setname == "S_newmains" and not S4_IN_S_NEWMAINS else ()
    hs_roots = [r for r in hs_all if r["batch"] not in excl]
    # 507-prep's per-consumer rms entry (draft 9+): the resolved pools must equal it by key; --tag=final requires it
    eq = {}
    want_roots, want_scope = (rms.get("H_sim_roots") or {}).get(setname), (rms.get("scope") or {}).get(setname)
    if want_roots is None and TAG == "final":
        raise SystemExit(f"INPUTS_FINAL rms.H_sim_roots[{setname}] missing")
    if want_roots is not None:
        if len(set(want_roots)) != len(want_roots) or sorted(r["root"] for r in hs_roots) != sorted(want_roots):
            raise SystemExit(f"{setname}: resolved H_sim roots differ from rms.H_sim_roots: {sorted({r['root'] for r in hs_roots} ^ set(want_roots))[:5]}")
        eq["H_sim_roots"] = f"equal to rms.H_sim_roots[{setname}] ({len(want_roots)} roots)"
    lists = (rms["five_arm_cells"].get("lists") or {}).get(setname)
    if lists is not None:
        if select(v2, lists["key"]) != cells or len(cells) != lists["n"]:
            raise SystemExit(f"{setname}: five-arm cells differ from rms.five_arm_cells.lists")
        eq["cells"] = f"equal to rms.five_arm_cells.lists[{setname}] ({lists['n']} cells)"
    assert len({r["root"] for r in hs_all}) == len(hs_all), "an H_sim root listed twice"
    scope = None
    if setname == "S_newmains":
        s4_arms, s4_hb = sorted(set(cells) & S4_CELLS), sorted({k.split("|", 1)[1] for k in hb_keys_} & S4_CELLS)
        s4_hs = sorted(r["root"] for r in hs_roots if r["batch"] in S4_BATCHES or "nm360_s4" in r["root"])
        if S4_IN_S_NEWMAINS:  # Codex r3: S4 needs 4r-ownership resolution through the source map (per cell and arm: 4r rows only), not written yet
            raise SystemExit("option (b) needs the 4r-ownership resolution through d9's source map (second refresh); not implemented")
        if S4_IN_S_NEWMAINS and not (s4_arms and s4_hb and s4_hs):
            raise SystemExit("option (b) needs S4 in every pool and arm (S4-inclusive real_pools); arms/H_B/H_sim S4 cells or roots: "
                             f"{len(s4_arms)}/{len(s4_hb)}/{len(s4_hs)}")
        if not S4_IN_S_NEWMAINS and (s4_arms or s4_hb or s4_hs):
            raise SystemExit(f"option (a): S4 found in arms {s4_arms[:3]}, H_B {s4_hb[:3]} or H_sim {s4_hs[:3]}")
        if (want_scope is None and TAG == "final") or (want_scope is not None and want_scope != (SCOPE_B if S4_IN_S_NEWMAINS else SCOPE_A)):
            raise SystemExit(f"rms.scope[S_newmains] {want_scope!r} differs from the builder's scope")
        scope = {"scope_label": SCOPE_B if S4_IN_S_NEWMAINS else SCOPE_A,
                 "s4_status": {"arms": f"included ({len(s4_arms)} S4 cells)" if s4_arms else "not included",
                               "H_B": f"included ({len(s4_hb)} S4 cells)" if s4_hb else "not included",
                               "H_sim": f"included ({len(s4_hs)} S4 roots)" if s4_hs else "not included"},
                 "H_sim_roots_used": [r["root"] for r in hs_roots], "H_sim_roots_excluded": [r["root"] for r in hs_all if r["batch"] in excl],
                 "s4_cells_excluded": [] if S4_IN_S_NEWMAINS else sorted(S4_CELLS)}  # the runner stops on any H_sim unit of these cells
    s4 = {"arm_rows_orphan": sorted(f"{a}:{f}:{k}" for a, r in arm_rows.items() for f, k in r.values() if (root_of(f), k) in ORPH),
          "H_B_orphan": sorted(f"{f}:{k}" for f in hb for k in hb_keys_ if (root_of(f), k) in ORPH),
          "H_sim_roots_at_orphan_roots": sorted(r["root"] for r in hs_roots if root_of(r["root"]) in O_ROOTS)}
    if any(s4.values()):
        raise SystemExit(f"{setname}: S4 orphan rows resolved: {json.dumps(s4)[:600]}")
    s4 |= {"source_map": smap_ent, "outside_rows": len(ORPH), "four_r_cells_in_arms": sorted(set(cells) & FOUR_R),
           "four_r_cells_in_H_B_keys": sorted({k.split("|", 1)[1] for k in hb_keys_} & FOUR_R)}
    eq["H_B"] = f"equal to rms.H_B[{setname}] by key ({len(hb_keys_)} keys, selector resolved from the pinned record)"
    out = {"set": setname, "pool_equality": eq, "points": [0.001, 0.01], "cells": cells, "arm_rows": arm_rows, "folders": folders,
           "H_B": {"folders": hb, "keys": hb_keys_}, "s4_orphan_check": s4,
           "H_sim": {"label": inp["label"]["filled"] or f"(label not filled; template) {inp['label']['template']}", "roots": hs_roots, "exclude_keys": []},
           "sources": {"inputs_final": {"path": args["--inputs"], "sha256": sha(inp_b), "status": inp["status"]}, "rms": rms} | flag,
           "rows": [{"label": f"{f} {ro}", "sub": s, "file": fn, "folds": nf} for f, ro, s, fn, nf in ROWS]} | extra | (scope or {})
    if scope:
        out["set_label"] = f"{out['set_label']}; {scope['scope_label']}"
    text = json.dumps(out, indent=1) + "\n"
    if any(x in text for x in O_SHAS):
        raise SystemExit(f"{setname}: an orphan .eval sha256 appears in the config")
    (HERE / "config" / f"inputs_rms_{setname}_{TAG}.json").write_text(text)
    print(setname, "cells", len(cells), "folders", len(folders), "H_B keys", len(out["H_B"]["keys"]), "roots", len(out["H_sim"]["roots"]),
          "replacement roots", sum(r["replacement_root"] for r in out["H_sim"]["roots"]), "S4 orphans 0; 4r cells in arms",
          len(s4["four_r_cells_in_arms"]), "in H_B", len(s4["four_r_cells_in_H_B_keys"]), "|", json.dumps(scope["s4_status"]) if scope else "", "H_sim roots excluded", len(scope["H_sim_roots_excluded"]) if scope else 0)


build("S_new", "S_new_on_M", lambda f: f.startswith("su311_q") and f.endswith("_main_v1__honest_b"),
      {"cell_fold": {c: fold_of_main.get(m, -1) for c, m in cell_main.items()}, "fold_minus1_rule": "s1",
       "set_label": "S_new on Append 36's M (257), five-arm v2; POST HOC; H_sim = INPUTS_FINAL S_new; Sol (a)/c_asis under 25h §1"})
build("S_newmains", "S_newmains_five_arm", lambda f: f.startswith("nm360_") and f.endswith("__honest_b"),
      {"cell_fold": None, "withhold_5fold": True, "sol_extension_label": EXT_LABEL,
       "set_label": "S_newmains five-arm (239), v2; POST HOC; H_sim = INPUTS_FINAL S_newmains sim-honest roots of the batches in scope (differs from the 19-root 14:55 export)"})
