"""By-key coverage of one sim-honest replacement root (_rlr / _rlr<k>), from COMMITTED files only (git show HEAD:<path>): the replacement
run table's cells == the family's ruled keys (config rlr_keys); the manifest's sim-arm tasks == the ruled keys minus the rows marked
replacement_not_counted; each row's key is <sim_world>|<cell>; the staging record is the run table's own staged set (same name and staged
sha), its counts agree, and each row's ran / excluded status agrees with its exclusions; each superseded original sits exactly once in the
parent root's committed manifest. Keys only; fails loudly on any mismatch.

  python3 make_rlr_coverage.py <family id e.g. simh-q1-rlr> <out.json>
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REL = "experiments/2026-10-14_q3_overnight_chain"
ARM = "sim_honest@september"


def git(*a) -> str:
    return subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=HERE).stdout


def committed(rel: str) -> tuple[dict, str]:
    """The file's content at HEAD and the last commit that touched it."""
    return json.loads(git("show", f"HEAD:{REL}/{rel}")), git("log", "-1", "--format=%h", "--", rel).strip()


def main():
    fid, out = sys.argv[1:3]
    fam = next(f for f in json.loads(git("show", f"HEAD:{REL}/config/chain_batches.json"))["followups"] if f["id"] == fid)
    tag, pool, q, world = fam["sim_tag"], fam["pool"], fam["qtok"], fam["sim_world"]
    target = f"{pool}_{q}_{tag}"
    rt_rel, man_rel = f"results/hygiene/run_table_{tag.removeprefix('sim_')}_{pool}_{q}.json", f"config/manifest_{target}.json"
    rt, rt_c = committed(rt_rel)
    man_all, man_c = committed(man_rel)
    problems = []
    rows = {c["cell"]: c for c in rt["cells"]}
    if len(rows) != len(rt["cells"]) or rt["pool"] != pool or rt["rlr_of"] != fam["rlr_of"]:
        problems.append("run table identity (duplicate cells, pool or rlr_of)")
    man = [f for f in man_all["files"] if f["arm_dir"] == ARM]
    tasks = [f["task"] for f in man]
    not_counted = {c for c, r in rows.items() if r["sim_status"] == "replacement_not_counted"}
    if len(tasks) != len(set(tasks)) or set(rows) != set(fam["rlr_keys"]) or set(tasks) != set(fam["rlr_keys"]) - not_counted:
        problems.append({"table": sorted(rows), "manifest": sorted(tasks), "ruled": sorted(fam["rlr_keys"]), "not_counted": sorted(not_counted)})
    if len(rt["staged_sets"]) > 1:
        problems.append(f"{len(rt['staged_sets'])} staged sets in the run table")
    empty = not rt["staged_sets"]  # every replacement run excluded at staging: no staged set, no extraction, nothing to score
    if empty:
        ss_sha, staged_set = None, None
        if any(r["sim_status"] == "ran" for r in rows.values()):
            problems.append("a row says ran but the run table has no staged set")
    else:
        (ss_sha, ss), = rt["staged_sets"].items()  # the run table's own staged set
        staged_set = ss["staged_set"].split("/")[0]
    recs = sorted(p for p in git("ls-files", "results").split() if p.startswith(f"results/chain_stage_{target}_2026-"))
    matching = []
    for p in recs:
        r = json.loads(git("show", f"HEAD:{REL}/{p}"))
        r = r[next(k for k in r if k != "git")]
        if (r.get("out") == staged_set) if not empty else (r["c_gate"].get("staged") == 0 and r.get("out", "").startswith(f"{target}_v1")):
            matching.append((p, r))
    if len(matching) != 1:
        problems.append(f"staging records for {staged_set}: {len(matching)}")
        rec_p, rec = (matching[-1] if matching else (None, {"c_gate": {"excluded": {}}, "assembled": {"staged_inputs_sha256": {}}}))
    else:
        rec_p, rec = matching[0]
    if not empty and rec["assembled"]["staged_inputs_sha256"].get(f"{ARM}.jsonl") != ss_sha:
        problems.append("staging record's staged sha != the run table's staged set")
    att, att_c = committed(f"results/hygiene/attestations_{pool}_{fid}.json")  # binds the staged sha to the manifest it was staged from
    covers = [e["covers"] for e in att["attestations"].get(ss_sha, []) if e["check"] == "longest_epoch_h200_gate"]
    if empty:
        if att["attestations"]:
            problems.append("attestation holds sets but the run table has none")
    elif len(covers) != 1 or "@" not in covers[0].get("manifest", ""):
        problems.append("no single attestation entry covering the staged sha with its manifest")
    else:
        mpath, mcommit = covers[0]["manifest"].rsplit("@", 1)
        raw = lambda ref: subprocess.run(["git", "show", ref], capture_output=True, check=True, cwd=HERE).stdout  # bytes, no newline translation
        if covers[0].get("staged_sha") != ss_sha or mpath != f"{REL}/{man_rel}" or raw(f"{mcommit}:{mpath}") != raw(f"HEAD:{REL}/{man_rel}"):
            problems.append({"attested_manifest": covers[0]["manifest"], "differs_from": f"{man_rel} @ HEAD"})
    excl, by_path = {}, {f["path"]: f["task"] for f in man}
    for k in rec["c_gate"]["excluded"]:  # each exclusion is a full run path: it must BE a manifest row's path (same launch and .eval)
        path = k.split("#", 1)[0]
        if path not in by_path:
            problems.append({"exclusion_not_a_manifest_path": path})
            continue
        excl[by_path[path]] = path
    mk, stg = rec["c_gate"].get("manifest_keys"), rec["c_gate"].get("staged")
    if mk is None or stg is None or mk != len(man) or stg != len(man) - len(excl):
        problems.append({"manifest_keys": mk, "staged": stg, "manifest_rows": len(man), "excluded": len(excl)})
    cells = {}
    for c, r in sorted(rows.items()):
        if r["key"] != f"{world}|{c}":
            problems.append({"cell": c, "key": r["key"]})
        st = r["sim_status"]
        want = "replacement_not_counted" if c not in tasks else ("excluded" if c in excl else "ran")
        if st != want:
            problems.append({"cell": c, "status": st, "expected": want})
        root = r["supersedes"]["root"]  # the parent family's batch id: <family> (final) or <family>_s<k> (snapshot k)
        k = root.rsplit("_s", 1)[1] if "_s" in root and root.rsplit("_s", 1)[1].isdigit() else None
        if root != fam["rlr_of"] + (f"_s{k}" if k else ""):
            problems.append({"cell": c, "supersedes_root": root})
        pm_rel = f"config/manifest_{pool}_{q}_sim{f'_s{k}' if k else ''}.json"
        try:
            pm, pm_c = committed(pm_rel)
        except subprocess.CalledProcessError:
            pm, pm_c = {"files": []}, None
        orig = [f for f in pm["files"] if f["task"] == c and f["arm_dir"] == ARM]
        if len(orig) != 1:
            problems.append({"cell": c, "original_rows_in_parent_root": len(orig), "parent_manifest": pm_rel})
        cells[f"{world}|{c}"] = {"replacement_status": st, "class": r["class"], "ruling": r["ruling"], "root_check": r["root_check"],
                                 "supersedes_root": root, "parent_manifest": f"{pm_rel} @ {pm_c}",
                                 "original_eval_sha256": orig[0]["sha256"] if len(orig) == 1 else None}
    doc = {"what": f"by-key coverage of the replacement root {staged_set} ({fid}), from committed files", "run_table": f"{rt_rel} @ {rt_c}",
           "manifest": f"{man_rel} @ {man_c}", "staging_record": f"{rec_p} @ {git('log', '-1', '--format=%h', '--', rec_p).strip() if rec_p else None}",
           "staged_set": staged_set, "staged_sha256": ss_sha, "attestation": f"results/hygiene/attestations_{pool}_{fid}.json @ {att_c}",
           "counts": {"ruled": len(fam["rlr_keys"]), "rows": len(rows), "staged": sum(r["sim_status"] == "ran" for r in rows.values()),
                      "replacement_excluded": sum(r["sim_status"] == "excluded" for r in rows.values()), "replacement_not_counted": len(not_counted),
                      "problems": len(problems)},
           "problems": problems, "cells": cells}
    Path(out).open("x").write(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc["counts"]))
    if problems:
        raise SystemExit("coverage FAILED")


if __name__ == "__main__":
    main()
