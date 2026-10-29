"""d9's CUT fork (Append 41): the `_cutfull_v1` naming rule's collision test. Reads names only (no transcripts): every routed folder and
staged-set name in the chain's routing directory, the routing-authority file and the chain targets, and checks that (1) the
cutfull regexes match none of them, and (2) every example name in cutfull_naming_v1.json follows the rule and is not an
existing name. Writes results/cutfull_names_test_<stamp>.json. Local, names only.

  python3 experiments/2026-10-14_q3_overnight_chain/cutfull/cutfull_names_test.py <routing_dir> [<addendum json>]
"""
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
CHAIN = HERE.parent
rule = json.loads((HERE / "cutfull_naming_v1.json").read_text())
FOLDER = re.compile(rule["regex"]["routed_folder"])
STAGED = re.compile(rule["regex"]["staged_set"])
TOKEN = re.compile(r"[A-Za-z0-9_@.\-]*_v[0-9]+(?:_try[0-9]+)?(?:__[A-Za-z0-9_@.\-]+)?")

existing = set()
routing = Path(sys.argv[1])
for f in sorted(routing.glob("*.json")):
    existing |= set(TOKEN.findall(f.read_text(errors="replace")))
    existing.add(f.stem)
    try:
        r = json.loads(f.read_text())
    except ValueError:
        continue
    if isinstance(r, dict) and r.get("staged_set"):  # a routing record: its folders are <staged_set>__<meta_arm> (and the arm without "@...")
        existing.add(r["staged_set"])
        for a in r.get("arms") or []:
            if isinstance(a, dict) and a.get("meta_arm"):
                existing |= {f"{r['staged_set']}__{a['meta_arm']}", f"{r['staged_set']}__{a['meta_arm'].split('@')[0]}"}
existing |= set(TOKEN.findall((CHAIN / "results/hygiene/routing_authority_2026-10-15.json").read_text()))
targets = json.loads((CHAIN / "config/chain_targets.json").read_text())
existing |= set(targets) | {t.get("out", "") for t in targets.values()}
existing.discard("")

hits_folder = sorted(n for n in existing if FOLDER.fullmatch(n))
hits_staged = sorted(n for n in existing if STAGED.fullmatch(n))
ex_rows = []
extra = json.loads(Path(sys.argv[2]).read_text())["examples"] if len(sys.argv) > 2 else []  # an addendum's examples, checked the same way
for e in rule["examples"] + extra:
    ok_rule = e["cutfull_staged_set"] == e["original_target"] + "_cutfull_v1" and FOLDER.fullmatch(e["cutfull_folder"]) is not None \
        and e["cutfull_folder"].startswith(e["cutfull_staged_set"] + "__")
    ok_gate = FOLDER.fullmatch(e["gate_folder"]) is not None and e["gate_folder"].startswith(e["original_target"] + "_cutfull_gate_v1__")
    ex_rows.append({"origin": e["origin"], "folder": e["cutfull_folder"], "follows_rule": ok_rule, "gate_follows_rule": ok_gate,
                    "folder_is_existing": e["cutfull_folder"] in existing, "original_folder_is_existing": e["original_folder"] in existing})
out = {"what": "cutfull naming collision test (names only)", "rule_file": "cutfull/cutfull_naming_v1.json", "addendum": sys.argv[2] if len(sys.argv) > 2 else None,
       "addendum_sha256": __import__("hashlib").sha256(Path(sys.argv[2]).read_bytes()).hexdigest() if len(sys.argv) > 2 else None,
       "routing_dir_files": len(list(routing.glob("*.json"))), "existing_names_scanned": len(existing),
       "regex_matches_existing_folder_names": hits_folder, "regex_matches_existing_staged_names": hits_staged,
       "examples": ex_rows,
       "passes": not hits_folder and not hits_staged and all(r["follows_rule"] and r["gate_follows_rule"] and not r["folder_is_existing"] for r in ex_rows)}
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
p = CHAIN / "results" / f"cutfull_names_test_{stamp}.json"
p.open("x").write(json.dumps(out, indent=1))
print(p.name, "passes:", out["passes"], "| scanned", len(existing), "| folder hits", len(hits_folder), "| staged hits", len(hits_staged))
print([ (r["origin"], r["original_folder_is_existing"]) for r in ex_rows])
