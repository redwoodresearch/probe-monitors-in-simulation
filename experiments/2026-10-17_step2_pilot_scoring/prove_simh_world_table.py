"""STEP-2 (09 via 2e, 10-17): proof that own_all's EXPLICIT sim-honest folder -> world table reproduces, folder by folder, the label the old rule gave.
Keys only: folder names, world labels, commits, paths and sha256s; no score, no count of rows is read (score records are read for folder names and
the KEYS of counts.by_world only).

For each scorer version (run_own_all.py at 82f6e77ba / 460acc84d / e2003d463):
  - the old rule is read from that commit's file by AST (SIMH_WORLD's literal, or e2003d463's inline expression, evaluated on the folder name),
    and the (a'') path's CUT_WORLD["sim_honest"];
  - the folder universe U = every sim_honest row's uncut_planned.folder in every committed inventory (all refs; (b), S_old and (a'') jobs are
    built only from an inventory's planned folders), cross-checked against d9's S_old attestations (highest revision) at every routing commit
    the scorers' records and fits name, and against every sim-honest folder in the version's committed cutfull score records;
  - with --edited <commit> per version: the edited file's SIMH_WORLD_TABLE (+ PILOT_FOLDER -> PILOT_WORLD) must equal {fo: old label} on U plus
    exactly the pilot entry; every (a'') inventory-v3 folder's table label must equal CUT_WORLD["sim_honest"]; every scored folder's by_world
    label keys must equal its table label.
  uv run --no-project --python 3.12 python -I prove_simh_world_table.py <repo> [--edited 82f6e77ba=<commit or file:path>,460acc84d=...,e2003d463=...] [--out <json>]
"""

import ast
import hashlib
import json
import re
import subprocess
import sys

SCORER = "experiments/2026-10-14_p2_family_expansion/run_own_all.py"
RES = "experiments/2026-10-14_p2_family_expansion/results"
ATT = "experiments/2026-10-14_q3_overnight_chain/results/cutfull/attest"
VERSIONS = {"82f6e77ba": ("origin/aswin/own-all-loo", r"^own_all_rs(_uncut)?_cutfull_score_"),
            "460acc84d": ("origin/aswin/own-all-simtrained-2e", r"^own_all_rs(_uncut|_simtrained_uncut)?_cutfull_score_"),
            "e2003d463": ("origin/aswin/own-all-loo3-uncut-4f", r"^own_all_loo[34]_uncut(_o3pool)?_cutfull_score_")}
ROUTING_COMMITS = ["9bdeddcf571b", "4b1f38d236c1", "d38fb15ec21b", "e9c9ebf964154921df735feb1fc866b4ca196e7b", "5bd21f37c32f"]  # score records' + fits'
INV_V3 = ("b8a39f329", "experiments/2026-10-15_cut_sensitivity/results/inventory_v3_2026-10-15.json", "19306ffa7c42ab35a62715c1a18822a43c6273492e5ede9b2f62cd9e428ad9a3")
PILOT_EXPECTED = "simh_q4f_v1_pilot_cutfull_v1__sim_honest"  # the pilot's planned folder (2e rename, 10-17): NEW, so outside the existing universe


def inventory_versions(git):
    """Every committed version of every *inventory*.json (case-insensitive, any path) in any commit reachable from any ref (Astra R2): merge diffs
    against each parent (--full-history --diff-merges=separate), so a file only a merge resolution adds is seen, and no rename detection
    (--no-renames), so a renamed-and-edited file shows as added; added / modified paths (R / C cannot occur without rename or copy detection; 507-prep, 10-17). Returns {(path, commit)}."""
    versions, cur = set(), None
    for ln in git("log", "--all", "--full-history", "--diff-merges=separate", "--no-renames", "--format=C %H", "--name-only", "--diff-filter=AM", "--",
                  ":(icase)*inventory*.json").decode().splitlines():
        if ln.startswith("C "):
            cur = ln[2:]
        elif ln.strip().lower().endswith(".json") and "inventory" in ln.strip().rsplit("/", 1)[-1].lower():
            versions.add((ln.strip(), cur))
    return versions


def record_labels_ok(by_world_keys, expected):
    """Astra R3: a scored sim-honest folder's UNFILTERED counts.by_world keys must be exactly [its table label]; extra labels (e.g. run1 beside
    simh_full), a wrong label and an empty set (no labelled rows) all FAIL, never a skip."""
    return sorted(set(by_world_keys)) == [expected]


def main(repo, edited, out):
    git = lambda *a: subprocess.run(["git", "-C", repo, *a], capture_output=True, check=True).stdout  # noqa: E731
    sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
    # (1) the universe: sim_honest planned folders of every committed inventory (all refs), by the commit that added each file
    versions = inventory_versions(git)  # every committed version of every *inventory*.json, case-insensitive, anywhere, merge-only and renamed files included
    loaded, not_json, no_runs = {}, [], 0
    for p, c in sorted(versions):
        b = git("show", f"{c}:{p}")
        try:
            d = json.loads(b)
        except json.JSONDecodeError:
            not_json.append(f"{p} @ {c[:12]}")  # listed, never silently dropped
            continue
        if isinstance(d, dict) and isinstance(d.get("runs"), list):  # any file with a runs list counts, whatever its name pattern
            loaded[(p, c)] = (c, b, d)
        else:
            no_runs += 1
    sup = [str(d.get("supersedes", "")) for _, _, d in loaded.values()]  # an inventory named first in another's "supersedes" is superseded (v1 -> v2 -> v3)
    U, invs, superseded_outside = {}, [], {}
    for (p, _c), (c, b, d) in loaded.items():
        old_ = any(s_.startswith(p.rsplit("/", 1)[1] + " ") for s_ in sup)
        n, fos = 0, set()
        for r in d["runs"]:
            up = r.get("uncut_planned") if isinstance(r, dict) else None
            if isinstance(up, dict) and up.get("folder") and r.get("world") == "sim_honest":
                n += 1
                fos.add(up["folder"])
            elif isinstance(up, dict) and str(up.get("folder", "")).endswith("__sim_honest"):
                raise SystemExit(f"{p}: a non-sim_honest row in a __sim_honest planned folder")
        if not old_:
            for fo in fos - {PILOT_EXPECTED}:  # the pilot folder is the one new entry, not an existing folder (an inventory may already plan it)
                U.setdefault(fo, set()).add(p.rsplit("/", 1)[1])
        invs.append({"commit": c, "path": p, "sha256": sha(b), "sim_honest_rows": n, "sim_honest_folders": len(fos), "superseded": old_, "plans_the_pilot_folder": PILOT_EXPECTED in fos})
        if old_:
            superseded_outside[p] = sorted(set(superseded_outside.get(p, [])) | fos)
    superseded_outside = {p: sorted(set(v) - set(U)) for p, v in superseded_outside.items()}
    if any(not fo.endswith("__sim_honest") for fo in U):
        raise SystemExit("a sim_honest row planned outside a __sim_honest folder (non-superseded inventory)")
    v3b = git("show", f"{INV_V3[0]}:{INV_V3[1]}")
    if sha(v3b) != INV_V3[2]:
        raise SystemExit("inventory v3 differs from cutfull.INVENTORY")
    v3_simh = sorted({r["uncut_planned"]["folder"] for r in json.loads(v3b)["runs"] if r["world"] == "sim_honest"})
    # (2) cross-check: d9's S_old attestations (highest revision per stem, as the scorers read them) at every routing commit
    att = {}
    for c in ROUTING_COMMITS:
        best = {}
        for p in git("ls-tree", "-r", "--name-only", c, ATT).decode().split():
            m = re.fullmatch(r".*/attestations_sold__(.+?_cutfull)(?:_r([0-9]+))?\.json", p)
            if m and int(m[2] or 1) > best.get(m[1], (0, ""))[0]:
                best[m[1]] = (int(m[2] or 1), p)
        for _, p in best.values():
            for tok, fo in json.loads(git("show", f"{c}:{p}"))["cutfull"]["folders"].items():
                if tok == "sim_honest":
                    att.setdefault(fo, set()).add(c[:12])
    res = {"label": "STEP-2 own_all explicit sim-honest world table: proof vs the old rule, folder by folder (keys only)", "universe": {
        "rule": "every sim_honest row's uncut_planned.folder in every committed version (added, modified, renamed or copied; merge diffs against each "
                "parent; no rename detection; all refs) of every *inventory*.json (case-insensitive, any path) that has a runs list; superseded inventories "
                "listed apart", "inventory_versions_scanned": len(versions), "folders": len(U), "inventories": invs,
        "inventory_json_versions_without_runs_list": no_runs, "inventory_files_not_json": not_json,
        "inventory_v3_sim_honest_folders": len(v3_simh), "superseded_inventories_folders_outside_universe": superseded_outside, "sold_attestation_sim_honest_folders": sorted(att),
        "sold_attestation_folders_outside_universe": sorted(set(att) - set(U))}, "versions": {}}
    if set(att) - set(U):
        raise SystemExit(f"S_old attestation folders outside the universe: {sorted(set(att) - set(U))}")
    for v, (br, rec_re) in VERSIONS.items():
        src = git("show", f"{v}:{SCORER}").decode()
        tree = ast.parse(src)
        consts = {t.id: n.value for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
        cut_world = ast.literal_eval(consts["CUT_WORLD"])["sim_honest"]
        if "SIMH_WORLD" in consts:
            rules = ast.literal_eval(consts["SIMH_WORLD"])
            old = lambda fo, rules=rules: next(lb for rx, lb in rules if re.search(rx, fo))  # noqa: E731
            rule_src = ast.get_source_segment(src, consts["SIMH_WORLD"])
        else:  # e2003d463: the inline expression in _cut_jobs_b's _job, evaluated on the folder name with r = {"world": "sim_honest"}
            hits = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "get"
                    and isinstance(n.func.value, ast.Dict) and any(isinstance(k, ast.Constant) and k.value == "sim_honest" for k in n.func.value.keys)]
            if len(hits) != 1:
                raise SystemExit(f"{v}: expected exactly one inline sim_honest world expression, found {len(hits)}")
            rule_src = ast.get_source_segment(src, hits[0])
            code = compile(ast.Expression(hits[0]), f"{v}:{SCORER}", "eval")
            old = lambda fo, code=code: eval(code, {"__builtins__": {}}, {"fo": fo, "r": {"world": "sim_honest"}})  # noqa: E731, S307
        table = {fo: old(fo) for fo in sorted(U)}
        a2_bad = sorted(fo for fo in v3_simh if table[fo] != cut_world)
        # scored folders in this version's committed cutfull score records: folder name + the KEYS of counts.by_world
        scored, rec_bad, recs = {}, [], []
        sup_names = {fo for v_ in superseded_outside.values() for fo in v_}
        for n in sorted(git("ls-tree", "--name-only", f"{br}:{RES}").decode().split()):
            if not re.match(rec_re, n) or "NOTE" in n:
                continue
            rb = git("show", f"{br}:{RES}/{n}")
            recs.append({"path": f"{RES}/{n}", "ref": br, "sha256": sha(rb)})
            for x in json.loads(rb)["results"].values():
                for fo, y in x["folders"].items():
                    if not fo.endswith("__sim_honest"):
                        continue
                    labs = sorted({w for c in y["counts"].values() for w in c.get("by_world", {})})  # UNFILTERED (Astra R3)
                    scored.setdefault(fo, set()).update(labs)
                    if fo not in table or not record_labels_ok(labs, table[fo]):  # extra, wrong or no labels: FAIL
                        rec_bad.append([n, fo, labs, table.get(fo)])
        r_ = {"commit": v, "old_rule_source": rule_src, "old_rule_source_sha256": sha(rule_src.encode()), "a2_cut_world_sim_honest": cut_world,
              "table": {lb: sorted(fo for fo, x in table.items() if x == lb) for lb in sorted(set(table.values()))},
              "table_sha256": sha(json.dumps(table, sort_keys=True).encode()), "a2_inventory_v3_folders_checked": len(v3_simh), "a2_label_mismatches": a2_bad,
              "score_records": recs, "scored_sim_honest_folders": len(scored), "scored_folders_outside_universe": sorted(set(scored) - set(table)),
              "record_label_mismatches": rec_bad, "scored_superseded_names": sorted(set(scored) & sup_names), "pilot_folder_scored_in_a_record": PILOT_EXPECTED in scored,
              "old_rule_label_of_the_pilot_folder": old(PILOT_EXPECTED),
              "pass_old_side": not a2_bad and not rec_bad and set(scored) <= set(table) and not set(scored) & sup_names and set(v3_simh) <= set(table)
                               and PILOT_EXPECTED not in scored and PILOT_EXPECTED not in table}
        if v in edited:  # the edited file's explicit table must be exactly the old labels on U plus the one pilot entry
            ebytes = open(edited[v][5:], "rb").read() if edited[v].startswith("file:") else git("show", f"{edited[v]}:{SCORER}")  # file:<path> = the edit being committed
            esrc = ebytes.decode()
            et = ast.parse(esrc)
            ec = {t.id: n.value for n in et.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
            ew = ast.literal_eval(ec["SIMH_WORLD_TABLE"])
            pf, pw = ast.literal_eval(ec["PILOT_FOLDER"]), ast.literal_eval(ec["PILOT_WORLD"])
            new = [(fo, lb) for lb, fos in ew for fo in fos]
            lit = dict(new)  # SIMH_WORLD_TABLE's entries; the code adds {PILOT_FOLDER: PILOT_WORLD} (SIMH_FOLDER_WORLD)
            nd = lit | {pf: pw}
            r_["edited"] = {"source": edited[v] if edited[v].startswith("file:") else git("rev-parse", edited[v]).decode().strip(), "blob_sha256": sha(ebytes),
                            "table_entries": len(new), "duplicate_folders": len(new) - len(lit),
                            "pilot": [pf, pw], "pilot_in_table_literal": pf in lit, "pilot_in_old_universe": pf in table, "simh_world_regex_left": "SIMH_WORLD" in ec,
                            "table_sha256_without_pilot": sha(json.dumps(lit, sort_keys=True).encode()),
                            "folders_added": sorted(set(nd) - set(table)), "folders_dropped": sorted(set(table) - set(nd)),
                            "label_changes": sorted([fo, table[fo], nd[fo]] for fo in set(table) & set(nd) if table[fo] != nd[fo])}
            e = r_["edited"]
            e["pass"] = (len(new) == len(lit) and not e["pilot_in_table_literal"] and e["folders_added"] == [pf] and not e["folders_dropped"] and not e["label_changes"] and pw == "simh_pilot"
                         and pf == PILOT_EXPECTED and not e["pilot_in_old_universe"] and not e["simh_world_regex_left"]
                         and e["table_sha256_without_pilot"] == r_["table_sha256"])
        res["versions"][v] = r_
    res["pass"] = all(x["pass_old_side"] and x.get("edited", {}).get("pass", not edited) for x in res["versions"].values())
    js = json.dumps(res, indent=1, sort_keys=True)
    if out:
        open(out, "x").write(js + "\n")
    print(json.dumps({"pass": res["pass"], "universe_folders": len(U), "versions": {v: {"pass_old_side": x["pass_old_side"], "edited_pass": x.get("edited", {}).get("pass"),
                      "labels": {lb: len(f) for lb, f in x["table"].items()}, "scored_folders": x["scored_sim_honest_folders"],
                      "record_label_mismatches": len(x["record_label_mismatches"]), "a2_label_mismatches": len(x["a2_label_mismatches"])}
                      for v, x in res["versions"].items()}}, indent=1))
    return res


if __name__ == "__main__":
    a = sys.argv[1:]
    ed = dict(x.split("=", 1) for x in a[a.index("--edited") + 1].split(",")) if "--edited" in a else {}
    sys.exit(0 if main(a[0], ed, a[a.index("--out") + 1] if "--out" in a else "")["pass"] else 1)  # a failed proof exits non-zero
