"""Append-only NOTE (RULE F.12): by-key equality between what step3's --sold mode read for the 6 S_old (b) messages (built from d9's S_old
attestations) and d9's standard S_old routing records (0d option (a)): staged set, extraction root, routed and original folders, the
attestation file (path and sha256), and per meta arm the staged sha, the epoch count and the epoch KEYS. 09 (03:1x) item 1 for A25: the
original is the record's original_folders.sim_honest AS IS and must equal inventory_b_fix_v1's registered original folder by key. Also
lists, per folder, the runs without rows with their pack manifests (all zero-call epochs: 09 item 11 structural zeros). Git and JSON only.
Writes results/cutfull_sold_equality_note_<stamp>.json; exits nonzero (STOP) on any mismatch.

  python3 -I check_sold_equality.py --records-commit <sha>
"""

import argparse
import glob
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DIR = "experiments/2026-10-14_q3_overnight_chain/results/cutfull/routing"
git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, check=True).stdout  # noqa: E731


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records-commit", required=True)
    a = ap.parse_args()
    import importlib.util
    s = importlib.util.spec_from_file_location("build_cutfull_message", HERE / "build_cutfull_message.py")
    A = importlib.util.module_from_spec(s)
    s.loader.exec_module(A)
    inv = {r["key"]: r for r in json.loads(git("show", f"{A.INVENTORY_B[0]}:{A.INVENTORY_B[1]}"))["runs"]}
    rows, bad = [], []
    for p in sorted(glob.glob(str(HERE / "results" / "cutfull_b_msgs" / "adapter_*_2026-10-16_*.json"))):
        ad = json.loads(Path(p).read_text())
        if "attestations_sold__" not in ad["pins"]["routing_record"]["path"]:
            continue
        msg = json.loads((REPO / ad["message_path"]).read_bytes())
        att_path, att_sha = ad["pins"]["routing_record"]["path"], ad["pins"]["routing_record"]["sha256"]
        name = "sold__" + Path(att_path).name.split("attestations_sold__", 1)[1]
        rec = json.loads(git("show", f"{a.records_commit}:{DIR}/{name}"))
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
            chk["A25: original as is"] = o in ("simh_q4f_v1_full_b1__sim_honest", "simh_q4f_v1_full_b2__sim_honest") and mine.get("sim_honest@september", {}).get("original") == {o}
        rows.append({"record": f"{DIR}/{name} @ {a.records_commit}", "message": ad["message_path"], "epochs": len(msg["epochs"]), "checks": chk})
        bad += [f"{name}: {c}" for c, ok in chk.items() if not ok]
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
                                                               if n.startswith("sold__") and not n.endswith("_attest.json")}:
        bad.append(f"the adapter records do not map one-to-one onto d9's standard S_old records: {sorted(targets)}")
    out = {"what": "append-only NOTE: --sold read vs d9's standard S_old records, by key; and the (b) runs without rows (RULE F.12)", "git": git("rev-parse", "HEAD").decode().strip(),
           "stamp (PT)": stamp, "records_commit": a.records_commit, "inventory": A.INVENTORY_B, "sold": rows, "mismatches": bad, "rowless_epochs": rowless,
           "pass": not bad and len(rows) == 6}
    q = HERE / "results" / f"cutfull_sold_equality_note_{stamp}.json"
    q.open("x").write(json.dumps(out, indent=1))
    print(q.name, {"records": len(rows), "epochs": sum(r["epochs"] for r in rows), "mismatches": bad[:5], "rowless": len(rowless), "pass": out["pass"]})
    if not out["pass"]:
        raise SystemExit("STOP: a mismatch between the --sold read and d9's standard S_old records")


if __name__ == "__main__":
    main()
