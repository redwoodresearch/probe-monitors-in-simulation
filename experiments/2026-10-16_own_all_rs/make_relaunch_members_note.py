"""09's ruling (04:5x via 4f, 10-16) for rh1e4 / honest_b_relaunch: the FROZEN MEMBERS rule governs; the pool and its false-alarm denominator are
exactly the 958 keys of the members file @ 051899844; the 5 staged rows outside it are EXCLUDED by key. APPEND-ONLY NOTE (counts / keys only, no
rescore): for each given own_all score record, its rh1e4 message is rebuilt from its routing commit + inventory_b_fix_v1 with run_own_all's own code and
must equal the scored message's sha256; then, by key, its 963 scored epochs (= the manifest's runs with rows + runs without rows) minus the 5 equal the
958 members (eval path and .eval sha256), and the 5 are listed with the members file's reasons and the sample-error scan's class."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MEMBERS = ("051899844", "experiments/2026-10-15_fpr_sweep_507_prep/results/honest_b_relaunch_members_2026-10-16_032731.json")
SCAN = "experiments/2026-10-14_p2_family_expansion/results/rh1e4_sample_error_scan_2026-10-16_044138.json"


def main(records, man_dirs, inv_pin, out_rel):
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, check=True).stdout.strip()  # noqa: E731
    spec = importlib.util.spec_from_file_location("oa", REPO / "experiments/2026-10-14_p2_family_expansion/run_own_all.py")
    oa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oa)
    mb = subprocess.run(["git", "-C", str(REPO), "show", f"{MEMBERS[0]}:{MEMBERS[1]}"], capture_output=True, check=True).stdout
    mem = json.loads(mb)
    by_path = {m["eval_path"]: m for m in mem["members"]}
    outside = {x["staged_key"]: x for x in mem["staged_outside_membership"]}
    sb = (REPO / SCAN).read_bytes()
    scan = json.loads(sb)["by_key"]
    out, ok = {}, True
    for rel, md in zip(records.split(","), man_dirs.split(","), strict=True):
        b = (REPO / rel).read_bytes()
        if subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{rel}"], capture_output=True).stdout != b:
            raise SystemExit(f"{rel}: not committed as read")
        rec = json.loads(b)
        jobs, _ = oa._cut_jobs_b(git, rec["routing_commit"], tuple(inv_pin.split(":", 2)), ("only=rh1e4_stage",))
        ((m, h, k, _),) = [j for j in jobs if j[0]["routing_pin"].get("kind") == "S_old attestation" and "rh1e4_stage" in j[0]["routing_pin"]["path"]]
        if f"{k}|{h}" not in rec["results"] or len(rec["results"]) != 1:
            raise SystemExit(f"{rel}: the rebuilt rh1e4 message differs from the scored one")
        (fo, x), = rec["results"][f"{k}|{h}"]["folders"].items()
        man_b = (Path(md) / f"{fo}.json").read_bytes()
        if hashlib.sha256(man_b).hexdigest() != x["manifest_sha256"]:
            raise SystemExit(f"{fo}: manifest differs from the score record")
        man = json.loads(man_b)
        units = {e["key"]: f"{e['world']}|{e['cell']}" for e in m["epochs"]}
        if len(set(units.values())) != len(units):
            raise SystemExit("a unit maps to two epochs")
        scored = {"R": set(man["counts"]["R"]["per_run"]) | set(man["counts"]["R"]["runs_without_rows"]),
                  "C": set(man["counts"]["C"]["per_run"]) | set(man["counts"]["C"]["runs_without_rows"])}
        eps = {e["key"]: e for e in m["epochs"]}
        excl = sorted(k_ for k_ in eps if k_.removesuffix("#e0") not in by_path)
        kept = sorted(set(eps) - set(excl))
        inv = {r["key"]: r for r in json.loads(subprocess.run(["git", "-C", str(REPO), "show", f"{inv_pin.split(':', 2)[0]}:{inv_pin.split(':', 2)[1]}"],
                                                                     capture_output=True, check=True).stdout)["runs"] if r["world"] == "relaunch"}
        res = {"scored_epochs": len(eps), "manifest_units_R": len(scored["R"]), "manifest_units_C": len(scored["C"]),
               "units_equal_message": scored["R"] == scored["C"] == set(units.values()), "excluded": len(excl), "pool_keys_after_exclusion": len(kept),
               "pool_equals_members_by_eval_path": {k_.removesuffix("#e0") for k_ in kept} == set(by_path),
               "pool_eval_sha_equal_members": all(inv[k_]["eval_sha256"] == by_path[k_.removesuffix("#e0")]["eval_sha256"] for k_ in kept),
               "excluded_equal_members_staged_outside": set(excl) == set(outside),
               "excluded_rows": {k_: {"R": man["counts"]["R"]["per_run"].get(units[k_], [0, 0]), "C": man["counts"]["C"]["per_run"].get(units[k_], [0, 0])} for k_ in excl},
               "manifest_sha256": x["manifest_sha256"], "message_sha256": h}
        res["pass"] = (res["units_equal_message"] and res["pool_keys_after_exclusion"] == 958 == len(by_path) and res["pool_equals_members_by_eval_path"]
                       and res["pool_eval_sha_equal_members"] and res["excluded_equal_members_staged_outside"] and len(excl) == 5
                       and all(sum(v["R"]) == 0 and sum(v["C"]) == 0 for v in res["excluded_rows"].values()))
        ok &= res["pass"]
        out[rel] = res
    note = {"label": "honest_b_relaunch pool = the frozen members (09 ruling 04:5x, items 15 / 23; FPR sweep Part (a) §1): own_all rh1e4 records by key; APPEND-ONLY; counts / keys only",
            "members": {"commit": MEMBERS[0], "path": MEMBERS[1], "sha256": hashlib.sha256(mb).hexdigest(), "members": len(by_path)},
            "sample_error_scan": {"path": SCAN, "sha256": hashlib.sha256(sb).hexdigest()}, "inventory": inv_pin,
            "excluded_by_key": [{"key": k_, "reason": outside[k_].get("reason"), "sample_error": scan.get(k_, {}).get("sample_error")} for k_ in sorted(outside)],
            "records": out, "pass": ok,
            "rule": "the pool and its false-alarm denominator are exactly the 958 members; the 5 staged rows outside are excluded by key; all 5 carry 0 scored rows, so no score changes"}
    (REPO / out_rel).open("x").write(json.dumps(note, indent=1))
    print(json.dumps({"pass": ok, "records": {r: {k_: v for k_, v in x.items() if k_ not in ("excluded_rows",)} for r, x in out.items()}, "excluded": [(e["key"].split("/")[-2], e["sample_error"]) for e in note["excluded_by_key"]]}))
    if not ok:
        raise SystemExit("STOP")


if __name__ == "__main__":
    main(*sys.argv[1:5])
