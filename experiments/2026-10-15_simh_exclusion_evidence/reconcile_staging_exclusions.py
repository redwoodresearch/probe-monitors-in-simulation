"""Reconcile the chain's sim-honest staging exclusions against our terminal-error evidence, by run key.

Keys, classes and counts only. Exclusion reasons are printed only when they equal the stager's fixed literal; anything
else is reported as sha256 + length.
- Every JSON in the routing dir is inventoried. A sim-honest .eval key anywhere except as a direct key of the top-level
  staging_exclusions dict (any nesting, any value), or a staging_exclusions that is not a dict, stops the run.
- Each simh routing file's exclusion count is checked against its hygiene file's sim_status.excluded, where that field
  exists (the replacement-launch run tables; batch snapshot run tables have no such field).
- Evidence = contract census runs + RL runs + replacement rows. Replacement status comes from the evidence metadata
  (census is_replacement, a row label starting "replacement"), and must agree with the launch name (_rlr).
- equal = no other-reason entries, no run key excluded twice, and both set differences empty. Exits 1 when not equal
  (after writing the output).

Usage: python3 -I reconcile_staging_exclusions.py <routing dir> <contract census json> <rl runs json>
         <out json> <provenance json> <replacement row json>...
"""

import datetime
import glob
import hashlib
import json
import os
import re
import sys

MISALIGNED = "errored run: events and messages misaligned"
KEY_RE = re.compile(
    r"^(?P<launch>2026-10-15_simh_[a-z0-9_]+)/(?P<arm>sim_honest@september)/"
    r"(?P<cell>[A-Za-z0-9_]+)/(?P<file>[A-Za-z0-9_.:+-]+\.eval)#e(?P<epoch>[0-9]+)$"
)
SIMH_EVAL_RE = re.compile(r"2026-10-15_simh_[a-z0-9_]+/sim_honest@september/[A-Za-z0-9_]+/[^\"]*\.eval")
RUN_ID_RE = re.compile(r"^(?P<launch>2026-10-15_simh_[a-z0-9_]+)\|sim_honest@september/[A-Za-z0-9_]+$")


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load(path):
    with open(path) as f:
        return json.load(f)


def walk_strings(obj, path=()):
    """Yield (path, string) for every dict key and string value."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield path, k
            yield from walk_strings(v, path + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk_strings(v, path + (i,))
    elif isinstance(obj, str):
        yield path, obj


def routing_exclusions(routing_dir):
    all_json = sorted(glob.glob(os.path.join(routing_dir, "*.json")))
    rows, files, hygiene_checks = [], {}, []
    for path in all_json:
        name = os.path.basename(path)
        doc = load(path)
        # a sim-honest .eval key is allowed ONLY as a direct key of the top-level staging_exclusions dict (as of 17:3x
        # none appears anywhere else, keys or values, in any routing file); anything else stops
        for p, s in walk_strings(doc):
            if SIMH_EVAL_RE.search(s) and p != ("staging_exclusions",):
                sys.exit(f"STOP: sim-honest .eval key under an unexpected field in {name}")
        if "staging_exclusions" not in doc:
            continue
        excl = doc["staging_exclusions"]
        if not isinstance(excl, dict):
            sys.exit(f"STOP: staging_exclusions in {name} is {type(excl).__name__}, not dict")
        if not name.startswith("simh-"):
            if any(SIMH_EVAL_RE.search(k) for k in excl):
                sys.exit(f"STOP: sim-honest exclusion key in a non-simh routing file {name}")
            continue
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")  # local clock (PT)
        files[name] = {"sha256": sha256_file(path), "n_exclusions": len(excl), "mtime_pt": mtime}
        hyg = os.path.join(routing_dir, name.split("__", 1)[0] + "_hygiene.json")
        if "__" in name and os.path.isfile(hyg):
            st = [f.get("sim_status") for f in (load(hyg).get("files") or {}).values() if isinstance(f, dict) and f.get("sim_status")]
            if st:
                # only the replacement-launch run tables carry an "excluded" field; batch snapshots count every run as "ran"
                has = any("excluded" in x for x in st)
                n_h = sum(x.get("excluded", 0) for x in st) if has else None
                hygiene_checks.append({"routing_file": name, "hygiene_has_excluded_field": has, "hygiene_excluded": n_h,
                                       "routing_exclusions": len(excl), "equal": (n_h == len(excl)) if has else None})
        for key, reason in excl.items():
            m = KEY_RE.match(key)
            if not m:
                sys.exit(f"STOP: unparsed exclusion key in {name} (len {len(key)})")
            if not isinstance(reason, str):
                sys.exit(f"STOP: non-string reason in {name}")
            rows.append({
                "routing_file": name,
                "run_id": f"{m['launch']}|{m['arm']}/{m['cell']}",
                "eval_file": m["file"],
                "epoch": int(m["epoch"]),
                "reason": reason if reason == MISALIGNED else {"sha256": hashlib.sha256(reason.encode()).hexdigest(), "len": len(reason)},
                "is_misaligned": reason == MISALIGNED,
            })
    return rows, files, hygiene_checks, len(all_json)


def evidence_set(census_path, rl_path, replacement_paths):
    ours = {}

    def add(run_id, cls, site, source, is_repl):
        m = RUN_ID_RE.match(run_id)
        if not m:
            sys.exit(f"STOP: bad run_id from {source}")
        if is_repl != ("_rlr" in m["launch"]):
            sys.exit(f"STOP: replacement flag disagrees with the launch name for {run_id} ({source})")
        prev = ours.get(run_id)
        if prev and (prev["error_class"] != cls or prev["is_replacement"] != is_repl):
            sys.exit(f"STOP: class/replacement disagreement for {run_id}")
        if prev:
            prev["sources"].append(source)
        else:
            ours[run_id] = {"error_class": cls, "raise_site": site, "is_replacement": is_repl, "sources": [source]}

    for r in load(census_path)["runs"]:
        add(r["run_id"], r["error_class"], r["raise_site"], "contract_census", bool(r["is_replacement"]))
    for r in load(rl_path)["runs"]:
        add(r["run_id"], r["wrapped_class"], r["raise_site"], "rl_runs_by_key", False)
    for p in replacement_paths:
        row = load(p)
        run_id = row.get("run_id") or f"{row['launch']}|{row['key']}"
        tc = row["terminal_class"]
        if not tc.get("concordant"):
            sys.exit(f"STOP: replacement row {p} not concordant")
        add(run_id, tc["eval_sample"], row.get("raise_site"), f"replacement_row:{os.path.basename(p)}",
            row["label"].lower().startswith("replacement"))
    return ours


def main():
    routing_dir, census_path, rl_path, out_path, prov_path, *replacement_paths = sys.argv[1:]
    rows, files, hygiene_checks, n_json = routing_exclusions(routing_dir)
    ours = evidence_set(census_path, rl_path, replacement_paths)

    mis = [r for r in rows if r["is_misaligned"]]
    other = [r for r in rows if not r["is_misaligned"]]
    routing_ids = {}
    for r in mis:
        routing_ids.setdefault(r["run_id"], []).append(r["routing_file"])
    dup = {k: v for k, v in routing_ids.items() if len(v) > 1}

    only_routing = sorted(set(routing_ids) - set(ours))
    only_ours = sorted(set(ours) - set(routing_ids))
    both = sorted(set(routing_ids) & set(ours))
    by_class = {}
    for k in both:
        key = ("replacement" if ours[k]["is_replacement"] else "original") + " | " + ours[k]["error_class"]
        by_class[key] = by_class.get(key, 0) + 1
    equal = not only_routing and not only_ours and not other and not dup

    out = {
        "what": "simh staging exclusions in the chain routing dir vs terminal-error evidence (census + RL + replacement rows), by run key",
        "provenance": load(prov_path),
        # a list of {file, sha256}, not {file: sha256}: gitleaks reads "<name with 'key'>": "<hex>" as a secret
        "inputs": [{"file": os.path.basename(p), "sha256": sha256_file(p)} for p in [census_path, rl_path, *replacement_paths]],
        "routing": {
            "n_json_files_inventoried": n_json,
            "n_simh_files_with_staging_exclusions": len(files),
            "files": files,
            "n_exclusion_entries": len(rows),
            "n_misaligned_entries": len(mis),
            "n_other_reason_entries": len(other),
            "other_reason_entries": other,
            "n_unique_misaligned_run_ids": len(routing_ids),
            "run_ids_in_more_than_one_file": dup,
            "hygiene_cross_check": hygiene_checks,
            "n_misaligned_in_files_written_by_1659": sum(1 for r in mis if files[r["routing_file"]]["mtime_pt"] <= "2026-10-15 16:59:59"),
            "misaligned_in_files_written_after_1659": [r["run_id"] for r in mis if files[r["routing_file"]]["mtime_pt"] > "2026-10-15 16:59:59"],
            "hygiene_with_excluded_field_all_equal": all(h["equal"] for h in hygiene_checks if h["hygiene_has_excluded_field"]),
            "n_hygiene_with_excluded_field": sum(h["hygiene_has_excluded_field"] for h in hygiene_checks),
        },
        "evidence": {"n_run_ids": len(ours), "runs": ours},
        "match": {
            "equal": equal,
            "in_routing_not_evidence": [{"run_id": k, "routing_files": routing_ids[k]} for k in only_routing],
            "in_evidence_not_routing": [{"run_id": k, "error_class": ours[k]["error_class"]} for k in only_ours],
            "matched_by_kind_and_class": by_class,
            "n_matched": len(both),
        },
        "exclusions": mis,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write("\n")
    print(json.dumps({k: out["routing"][k] for k in out["routing"] if k not in ("files", "other_reason_entries", "hygiene_cross_check")}))
    print(json.dumps({"evidence_n": len(ours), **out["match"]}))
    sys.exit(0 if equal else 1)


if __name__ == "__main__":
    main()
