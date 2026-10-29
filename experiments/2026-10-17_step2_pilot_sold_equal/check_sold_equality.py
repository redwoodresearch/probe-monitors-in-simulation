"""Append-only NOTE (RULE F.12): by-key equality between what step3's --sold mode read for the 6 S_old (b) messages (built from d9's S_old
attestations) and d9's standard S_old routing records (0d option (a)): staged set, extraction root, routed and original folders, the
attestation file (path and sha256), and per meta arm the staged sha, the epoch count and the epoch KEYS. 09 (03:1x) item 1 for A25: the
original is the record's original_folders.sim_honest AS IS and must equal inventory_b_fix_v1's registered original folder by key. Also
lists, per folder, the runs without rows with their pack manifests (all zero-call epochs: 09 item 11 structural zeros). Git and JSON only.
Writes results/cutfull_sold_equality_note_<stamp>.json; exits nonzero (STOP) on any mismatch.

  python3 -I check_sold_equality.py --records-commit <sha>
"""

import argparse
import ast
import fnmatch
import glob
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DIR = "experiments/2026-10-14_q3_overnight_chain/results/cutfull/routing"
# STEP-2 pilot-only copy (09 R-a, option (i)): (1) the pilot inventory pin, by sha256; (3) exactly ONE required record (the pilot)
INVENTORY = ("98575cc3471a224c87401111fbb8caee28ff5323", "experiments/2026-10-15_cut_sensitivity/results/inventory_b_fix_v2_2026-10-17_081456.json", "f8dfc018d44c5a5bd373f88d9f7974dd9774c25a325d3ceceb385fcb9c82719f")  # (1) the pilot inventory (commit, path, sha256) = fix_v2 (09 via 2e, 10-17); unset = STOP
REQUIRED = {"sold__simh_q4f_v1_pilot_cutfull.json": "simh_q4f_v1_pilot__sim_honest"}  # (3) the one routing record -> its A25 original, as is
PILOT_RECORD, S_OLD_ADAPTERS = "sold__simh_q4f_v1_pilot_cutfull.json", "adapter_*_2026-10-16_*.json"  # 09 (i): the exact set per path (the S_old one = the 10-16 glob)
COUNTED = ("344829af6", "experiments/2026-10-14_q3_q35_simhonest_purchase/results/simh_counted_keys_pilot_v3.json",
           "2149c5cde368aef9e5012c8d9cda5b12fe5a93db3d27ad2d525225bb9f511b27")  # 09 (i): the pilot adapter's epochs = exactly these 20 run keys
git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, check=True).stdout  # noqa: E731


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records-commit", required=True)
    a = ap.parse_args()
    lit = {n: [ast.literal_eval(x.value) for x in ast.parse((HERE / f).read_text()).body if isinstance(x, ast.Assign) and getattr(x.targets[0], "id", "") == n]
           for f, n in (("build_cutfull_message.py", "INVENTORY_B"), ("check_sold_equality.py", "INVENTORY"))}
    ok_lit = lambda v: v is None or (isinstance(v, tuple) and len(v) == 3 and all(isinstance(x, str) for x in v))  # noqa: E731
    if len(lit["INVENTORY_B"]) != 1 or len(lit["INVENTORY"]) != 1 or not all(ok_lit(v[0]) for v in lit.values()) or lit["INVENTORY_B"] != lit["INVENTORY"]:
        # one tuple fills both copies' pins: two identical literals, each None (unset) or a (commit, path, sha256) tuple of strings
        raise SystemExit("STOP: the build and check copies do not pin the same pilot inventory")
    if INVENTORY is None:  # (1)
        raise SystemExit("STOP: the pilot inventory pin is unset")
    try:
        ib = git("show", f"{INVENTORY[0]}:{INVENTORY[1]}")
    except subprocess.CalledProcessError:
        raise SystemExit("STOP: the pilot inventory is not in git at its pin") from None
    if hashlib.sha256(ib).hexdigest() != INVENTORY[2]:
        raise SystemExit("STOP: the pilot inventory differs from its pin (sha256)")
    inv = {r["key"]: r for r in json.loads(ib)["runs"]}
    rows, bad, seen, msgs = [], [], {}, {}
    for p in sorted(glob.glob(str(HERE / "results" / "cutfull_b_msgs" / "adapter_*.json"))):
        ad = json.loads(Path(p).read_text())
        if "attestations_sold__" not in ad["pins"]["routing_record"]["path"]:  # 507-prep R1: an adapter of another kind STOPs, never skipped silently
            bad.append(f"{Path(p).name}: not an S_old/pilot (sold) adapter")
            continue
        msg = json.loads((REPO / ad["message_path"]).read_bytes())
        att_path, att_sha = ad["pins"]["routing_record"]["path"], ad["pins"]["routing_record"]["sha256"]
        name = "sold__" + Path(att_path).name.split("attestations_sold__", 1)[1]
        seen.setdefault(name, []).append(Path(p).name)
        msgs[name] = msg
        if name == PILOT_RECORD and ((ad["pins"].get("inventory") or {}).get("path"), (ad["pins"].get("inventory") or {}).get("sha256")) != (INVENTORY[1], INVENTORY[2]):
            # the pilot path only (507-prep, 10-17): S_old's 10-16 adapters were built on inventory_b_v2 @ a0bf963a7, while the original
            # check read inventory_b_fix_v1; the two agree on b1's scope (inventory_v2_vs_fix_v1_b1_scope_2026-10-17_033715.json @
            # e7ee4d6b6: IDENTICAL 145/145), and the per-key original_folder checks below read this check's INVENTORY on both paths
            bad.append(f"{name}: the adapter's inventory pin is not this check's INVENTORY")
        try:
            rec = json.loads(git("show", f"{a.records_commit}:{DIR}/{name}"))
        except subprocess.CalledProcessError:  # an absent routing record: a reasoned STOP, not a crash
            bad.append(f"{name}: no routing record {DIR}/{name} at the records commit")
            continue
        arms = rec["arms"] if isinstance(rec["arms"], list) else json.loads(rec["arms"])
        ratt = rec["attestation"] if isinstance(rec["attestation"], dict) else json.loads(rec["attestation"])
        mine = {}
        for e in msg["epochs"]:
            mine.setdefault(e["arm"], {"keys": set(), "sha": set(), "folder": set(), "original": set()})
            for k_, v in (("keys", e["key"]), ("sha", e["staged_sha256"]), ("folder", e["folder"]), ("original", e["original_folder"])):
                mine[e["arm"]][k_].add(v)
        chk = {"staged_set": rec["staged_set"] == msg["provenance"][0]["staged_set"], "extraction_root": rec["extraction_root"] == msg["provenance"][0]["run"],
               "attestation": ratt["path"].split(" @ ")[0] == att_path and ratt["sha256"] == att_sha, "batch": rec["batch"] == msg["batch"] == "sold",
               "arms": set(mine) == {x["meta_arm"] for x in arms}}
        for x in arms:
            m, t = mine.get(x["meta_arm"], {}), x["meta_arm"].split("@")[0]
            n_msg = sum(e["arm"] == x["meta_arm"] for e in msg["epochs"])  # duplicates in the message would make n_msg exceed the key set
            chk[f"{t}: keys"] = m.get("keys") == set(x["keys"]) and len(x["keys"]) == len(set(x["keys"])) == x["epochs"] == n_msg
            chk[f"{t}: staged_sha"] = m.get("sha") == {x["staged_sha"]}
            chk[f"{t}: folder"] = m.get("folder") == {rec["folders"][t]}
            chk[f"{t}: original_folder"] = m.get("original") == {rec["original_folders"][t]} and all(inv[k]["original_folder"] == rec["original_folders"][t]
                                                                                                    and inv[k]["original_folder_registered"] is True for k in m.get("keys", ()))
        if "sim_honest" in rec["original_folders"]:  # 09 item 1: A25 names by exact equality, as is
            o = rec["original_folders"]["sim_honest"]
            chk["A25: original as is"] = o == REQUIRED.get(name) and mine.get("sim_honest@september", {}).get("original") == {o}
        rows.append({"record": f"{DIR}/{name} @ {a.records_commit}", "message": ad["message_path"], "epochs": len(msg["epochs"]), "checks": chk})
        bad += [f"{name}: {c}" for c, ok in chk.items() if not ok]
    try:
        cb = git("show", f"{COUNTED[0]}:{COUNTED[1]}")
    except subprocess.CalledProcessError:
        raise SystemExit("STOP: the counted pilot keys are not in git at their pin") from None
    if hashlib.sha256(cb).hexdigest() != COUNTED[2]:
        raise SystemExit("STOP: the counted pilot keys differ from their pin (sha256)")
    counted = sorted(k["volume_path"].removeprefix("simh_evals_v1/") + "#e0" for k in json.loads(cb)["keys"])
    for rn in REQUIRED:  # 09 (i): the widened glob, with an EXACT-SET filter per path; missing, extra or duplicate STOPs
        if len(seen.get(rn, [])) != 1:
            bad.append(f"{rn}: {len(seen.get(rn, []))} adapters (exactly one required)")
        elif rn == PILOT_RECORD and sorted(e["key"] for e in msgs[rn]["epochs"]) != counted:
            bad.append(f"{rn}: the adapter's epochs are not exactly the {len(counted)} counted pilot run keys")
        elif rn != PILOT_RECORD and not fnmatch.fnmatch(seen[rn][0], S_OLD_ADAPTERS):
            bad.append(f"{rn}: the adapter is not in the S_old 10-16 adapter set")
    bad += [f"{n}: an adapter for a record outside REQUIRED" for n in sorted(set(seen) - set(REQUIRED))]
    rowless = []
    for p in sorted(glob.glob(str(HERE / "results" / "cutfull_*_reverify_*.json"))):
        r = json.loads(Path(p).read_text())
        for fo, x in r["result"].items():
            for k in x.get("rowless_epochs", []):
                rowless.append({"folder": fo, "key": k, "manifest": x["manifest"], "manifest_sha256": x["manifest_sha256"],
                                "class": "structural zero: the extraction meta has 0 calls, so 0 eligible units in R and C (09 item 11): kept as not caught"})
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    targets = [r["record"].split(" @ ")[0] for r in rows]
    if len(set(targets)) != len(targets) or set(targets) != {f"{DIR}/{n}" for n in git("ls-tree", "--name-only", f"{a.records_commit}:{DIR}").decode().split()
                                                               if n in REQUIRED}:
        bad.append(f"the adapter records do not map one-to-one onto d9's standard S_old records: {sorted(targets)}")
    out = {"what": "append-only NOTE: --sold read vs d9's standard S_old records, by key; and the (b) runs without rows (RULE F.12)", "git": git("rev-parse", "HEAD").decode().strip(),
           "stamp (PT)": stamp, "records_commit": a.records_commit, "inventory": INVENTORY, "sold": rows, "mismatches": bad, "rowless_epochs": rowless,
           "pass": not bad and len(rows) == len(REQUIRED) == 1}
    q = HERE / "results" / f"cutfull_sold_equality_note_{stamp}.json"
    q.open("x").write(json.dumps(out, indent=1))
    print(q.name, {"records": len(rows), "epochs": sum(r["epochs"] for r in rows), "mismatches": bad[:5], "rowless": len(rowless), "pass": out["pass"]})
    if not out["pass"]:
        raise SystemExit("STOP: a mismatch between the --sold read and d9's standard S_old records")


if __name__ == "__main__":
    main()
