"""Finding 7 of the FPR507 recheck, by key: cohort attack keys that also appear as keys in FINAL's exclusion records.

Git-only scan (no Modal, no scores): reads pinned blobs with `git show`, checks their sha256, and writes
results/cohort_exclusion_keys_<stamp>.json with keys, record paths @ commits, key paths and stored label strings only.

  uv run --no-project --python 3.12 python experiments/2026-10-15_fpr507_recheck/analyze_cohort_exclusion_keys.py
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
FINAL = ("3ce6ea55e", "experiments/2026-10-14_xerr_scaleup_09/results/xerr_scaleup_final_2026-10-15_013255.json",
         "7dceb8bb25752aa3619cfde64add702fd4f7d13a1309a56dbc8880a6b27af9f5")
TARGET = ("404b6d241", "experiments/2026-10-15_fpr_sweep_507_prep/results/registered507_2026-10-15_103452.json")
FIELDS = ("applied", "setup_failures", "excluded_before_counting")  # the dict-valued exclusion records (key -> reason)
LISTS = ("staging_exclusions_of_dropped_runs", "not_counted_here")  # list-valued exclusion records
LABEL_FIELDS = ("exclusion_cause", "state", "excluded", "launch", "rerun_in", "manifest_key")


def show(commit, path):
    return subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, check=True, cwd=HERE).stdout


def paths_mentioning(x, needle, p=""):
    """Key paths in a JSON value whose key or string value contains needle."""
    out = []
    if isinstance(x, dict):
        for k, v in x.items():
            q = f"{p}[{json.dumps(k)}]"
            if needle in k:
                out.append(q)
            out += paths_mentioning(v, needle, q)
    elif isinstance(x, list):
        for i, v in enumerate(x):
            out += paths_mentioning(v, needle, f"{p}[{i}]")
    elif isinstance(x, str) and needle in x:
        out.append(p)
    return out


def main():
    b = show(FINAL[0], FINAL[1])
    if hashlib.sha256(b).hexdigest() != FINAL[2]:
        raise SystemExit("FINAL sha256 differs")
    fin = json.loads(b)
    r, sp = fin["result"], fin["spec"]
    se = r["staging_exclusions"]
    coh = r["cohort"]["cohort_cells"]
    tgt_rows = json.loads(show(*TARGET))["rows"]
    fref = f"{FINAL[1]} @ {FINAL[0]}"
    keys = {f"{w}|{c}": (w, c) for c in coh for w in ("run1", "run2", "sim")}
    hits = sorted({k for f in FIELDS for k in se[f] if k in keys} | {k for f in LISTS for k in se[f] if k in keys})
    blobs = {}

    def blob(commit, path):
        if (commit, path) not in blobs:
            blobs[(commit, path)] = json.loads(show(commit, path))
        return blobs[(commit, path)]

    rows = []
    for k in hits:
        w, c = keys[k]
        app = []
        for f in FIELDS:
            if k in se[f]:
                app.append({"record": fref, "key_path": f"result.staging_exclusions.{f}[{json.dumps(k)}]", "stored_reason": se[f][k]})
        for f in LISTS:
            if k in se[f]:
                app.append({"record": fref, "key_path": f"result.staging_exclusions.{f}[{se[f].index(k)}]", "stored_reason": None})
        for i, e in enumerate(sp["staging_exclusions"]):
            if k in e["exclusions"]:
                rec = e["record"]
                app.append({"record": fref, "key_path": f"spec.staging_exclusions[{i}].exclusions[{json.dumps(k)}]",
                            "stored_reason": e["exclusions"][k], "cites_record": f"{rec['path']} @ {rec['commit']}",
                            "cites_record_paths_mentioning_cell": paths_mentioning(blob(rec["commit"], rec["path"]), c),
                            "for_manifest": e["for_manifest"]})
        # the pre-count exclusion records attached to the counted manifests (d9), with their stored labels
        counted = []
        for j, m in enumerate(sp["counted"][w]):
            md = blob(m["commit"], m["path"])
            ents = [f_ for f_ in md["files"] if f_.get("task") == c and not str(f_.get("arm_dir", "")).startswith("honest")]
            if ents:
                counted.append({"manifest": f"{m['path']} @ {m['commit']}", "spec_key_path": f"spec.counted.{w}[{j}]",
                                "attack_entries": [{x: f_.get(x) for x in ("arm_dir", "launch", "attempt_kind", "attempt_index",
                                                                         "policy_called")} for f_ in ents]})
            if m.get("exclusions"):
                ex = blob(m["exclusions"]["commit"], m["exclusions"]["path"])
                for i, x in enumerate(ex.get("excluded", [])):
                    if x.get("key") == k:
                        app.append({"record": f"{m['exclusions']['path']} @ {m['exclusions']['commit']}",
                                    "key_path": f"excluded[{i}]", "stored_labels": {f_: x.get(f_) for f_ in LABEL_FIELDS},
                                    "record_batch": ex.get("batch"), "record_launch": ex.get("launch"),
                                    "record_selected_on": ex.get("selected_on"),
                                    "attached_to": f"spec.counted.{w}[{j}].exclusions ({m['path']} @ {m['commit']})"})
        rk = blob(sp["q2_rerun_run_keys"]["commit"], sp["q2_rerun_run_keys"]["path"])
        arms = [a for a, e in r["scale_up_inputs"].items() if k in e.get("sol_folds", {})]
        rows.append({
            "key": k,
            "exclusion_appearances": app,
            "counted_manifests_listing_the_cell": counted,
            "q2_rerun_run_keys_record": {"record": f"{sp['q2_rerun_run_keys']['path']} @ {sp['q2_rerun_run_keys']['commit']}",
                                         "paths_mentioning_cell": paths_mentioning(rk, c)},
            "target_use": {
                "in_FINAL_cohort_cells": c in coh,
                "scored_in_arms_per_FINAL_scale_up_inputs_sol_folds": arms,
                "staged_run_of_those_arms": {a: r["scale_up_inputs"][a]["staged"].get("run") for a in arms},
                "target_rows_list_it_as_a_missing_cohort_run": sorted(rk_ for rk_, tr in tgt_rows.items()
                                                                       if k in tr.get("coverage", {}).get("cohort_runs_missing", [])),
            },
        })
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, check=True, cwd=HERE).stdout.decode().strip()
    out = {"label": "FPR507 recheck finding 7 by key: cohort attack keys that also appear as keys in FINAL's exclusion records "
                    "(keys, record paths, key paths and stored labels only; no scores)",
           "git": git, "stamp_PT": stamp, "final_record": {"path": FINAL[1], "commit": FINAL[0], "sha256": FINAL[2]},
           "target": f"{TARGET[1]} @ {TARGET[0]}",
           "scanned": {"cohort_attack_keys": len(keys), "dict_records": [f"result.staging_exclusions.{f}" for f in FIELDS],
                       "list_records": [f"result.staging_exclusions.{f}" for f in LISTS]},
           "n_keys": len(rows), "rows": rows}
    dest = HERE / "results" / f"cohort_exclusion_keys_{stamp}.json"
    with dest.open("x") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({"record": dest.name, "n_keys": len(rows), "keys": [x["key"] for x in rows]}))


if __name__ == "__main__":
    main()
