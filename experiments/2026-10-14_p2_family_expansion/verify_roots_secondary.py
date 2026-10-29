"""4f's SECONDARY pre-launch checks by key (the checks the scratch verifier ran beyond verify_roots.py), committed so they can be
re-run from git: attestation coverage per staged sha (with the H200 gate's detail), extraction records on the chain branches, repro-c
coverage, format checks, the labels file's sha256, the stage record (from the attestation's corrected_render evidence) against the
routing arms and exclusions, the manifest's staged counts, and cell membership (pinned list for the pool; the 410-cell keylist for
il311 / ilold; nm_built for ilnm / nm360). Keys, shas and counts only; never builds or launches anything.

  python verify_roots_secondary.py <name>=<routing.json>:<attest pointer.json>:<records commit> [...] > out.json   (exit 1 if any root fails)
"""

import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RES = "experiments/2026-10-14_q3_overnight_chain/results/"


def show(c, p):
    return subprocess.run(["git", "-C", str(REPO), "show", f"{c}:{p}"], capture_output=True, check=True).stdout


def js(c, p):
    return json.loads(show(c, p))


def at(s):  # "path @ commit"
    p, c = (x.strip() for x in s.rsplit(" @ ", 1))
    return p, c


C311 = {c["key"] for c in js("0caadbfb5", "experiments/2026-10-14_q2_scaleup_cohort/results/scaleup_tasks_plus311_2026-10-14_125135.json")["cells"]}
NMV1 = {c["key"]: c["main"] for c in js("ef171c813", "experiments/2026-10-14_q3_new_mains_mode/config/new_mains_cells_v1.json")["cells"]}
NMB = {c["key"]: c["main"] for c in js("6feeee4b7", "experiments/2026-10-14_q3_new_mains_launch/config/nm_built.json")["cells"]}
T196 = {c["key"] for c in js("4d1eec572", "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json")["cells"]}
KL = js("2e75fa8c5", "experiments/2026-10-14_il_launch_09/results/il_keylists_2026-10-14.json")["estimation_set"]
K410 = set(KL["t196_tasks"] + KL["plus311_cells"])


def check(routing, attest, rec):
    r, ap = json.loads(Path(routing).read_text()), json.loads(Path(attest).read_text())
    res = {"root": r["extraction_root"], "staged_set": r["staged_set"], "stratum": r["stratum"], "routing_sha256": hashlib.sha256(Path(routing).read_bytes()).hexdigest()}
    att = js(ap["commit"], ap["path"])
    arms = {a["meta_arm"]: a for a in r["arms"]}
    cov = {}
    for a in arms.values():
        checks = att["attestations"].get(a["staged_sha"], [])
        h2 = [c for c in checks if c["check"] == "longest_epoch_h200_gate"]
        names = [c["check"] for c in checks]
        cov[a["meta_arm"]] = {"checks": [[c["check"], c["pass"]] for c in checks], "all_true": bool(checks) and all(c["pass"] is True for c in checks) and len(names) == len(set(names)),
                              "h200": len(h2) == 1 and h2[0]["pass"] is True and h2[0]["detail"].get("every_shard_h200") is True and h2[0]["detail"].get("last_attempt_no_errors") is True}
    res["attestation"] = {"commit": ap["commit"][:10], "pointer_failing": ap.get("failing"), "file_failing": att.get("failing", "absent"), "covers": cov,
                          "pass": ap.get("failing") == {} and att.get("failing", {}) == {} and all(v["h200"] and v["all_true"] for v in cov.values())}
    ex = {}
    for f in subprocess.run(["git", "-C", str(REPO), "ls-tree", "-r", "--name-only", rec, RES], capture_output=True, text=True, check=True).stdout.split():  # at the pinned records commit
            n = f.split("/")[-1]
            if n.startswith(f"extract_chain_{r['extraction_root']}_a"):
                d = js(rec, f)
                ex[n] = {"attempt": d["config"].get("attempt"), "gpu": dict(Counter(str(x.get("gpu_name")) for x in d["shards"])), "errors": sum("error" in x for x in d["shards"]),
                         "done_before": sum(x.get("done_before", 0) for x in d["shards"] if "error" not in x)}
    res["extraction_records"] = ex
    last = [v for v in ex.values() if str(v["attempt"]) == str(max((int(x["attempt"]) for x in ex.values()), default=-1))]
    res["extraction_ok"] = bool(last) and all(v["errors"] == 0 and set(v["gpu"]) == {"NVIDIA H200"} for v in last)  # the last attempt: every shard H200, no errors
    p, c = at(r["repro_gate"])
    x = js(c, p)
    res["repro"] = {"epochs": x["epochs"], "bit_exact": x["bit_exact"], "failing_shards": x["failing_shards"], "one_of_each": x["coverage"]["every_shard_one_of_each"],
                    "ok": x["epochs"] == x["passing"] == x["bit_exact"] and not x["failing_shards"] and x["coverage"]["every_shard_one_of_each"]}
    res["format"] = {}
    for a in arms.values():
        p, c = at(a["format_check"])
        x = js(c, p)
        res["format"][a["meta_arm"]] = x["passes"] is True and x["staged_sha"] == a["staged_sha"]
    if str(r["labels"]).startswith("not applicable") and all(a.startswith("sim_honest") or a in ("honest_a", "honest_b", "e2a", "e2b") for a in arms):  # honest-only arms (sim-honest; epoch-2 real honest) carry no attack labels
        res["labels"] = {"not_applicable": r["labels"], "match": True}
    else:
        lp = r["labels"].split(" (sha256 ")[0]
        lsha = r["labels"].split("(sha256 ")[1].split(")")[0]
        lc = r["labels"].rsplit(" @ ", 1)[1].strip() if " @ " in r["labels"] else rec
        res["labels"] = {"sha256": lsha[:12], "match": hashlib.sha256(show(lc, lp)).hexdigest() == lsha}
    decl = {}
    srp = sorted({e["path"] for a in arms.values() for ck in att["attestations"].get(a["staged_sha"], []) for e in ck.get("evidence_extra", [])
                  if e["path"].split("/")[-1].startswith(f"chain_stage_{r['target']}_")})
    if len(srp) != 1:
        res["stage"] = {"error": f"attestation names {len(srp)} stage records: {srp}"}
    else:
        st = js(rec, srp[0])
        sets = [v for v in st.values() if isinstance(v, dict) and "c_gate" in v and v.get("out") == r["staged_set"]]
        if len(sets) != 1:
            res["stage"] = {"error": f"{len(sets)} sets staging {r['staged_set']} in {srp[0]}"}
        else:
            sv = sets[0]
            res["stage"] = {"record": srp[0].split("/")[-1], "epochs_match": all(sv["assembled"]["epochs"].get(a) == arms[a]["epochs"] for a in arms) and set(sv["assembled"]["epochs"]) == set(arms),
                            "shas_match": all(sv["assembled"]["staged_inputs_sha256"].get(f"{a}.jsonl") == arms[a]["staged_sha"] for a in arms),
                            "exclusions_match_routing": sv["c_gate"].get("excluded", {}) == r.get("staging_exclusions", {}), "excluded": len(sv["c_gate"].get("excluded", {}))}
            decl = sv["c_gate"].get("excluded", {})
    p, c = at(r["manifest"])
    mraw = show(c, p)
    fs = json.loads(mraw)["files"]
    keys = {f"{f['launch']}/{f['rel']}#e0": f for f in fs if f["arm_dir"] in arms}
    pk = "simh_nm360" if r["staged_set"].startswith("simh_nm360_") else ("ep2_" + r["staged_set"].split("_")[1]) if r["staged_set"].startswith("ep2_") else r["staged_set"].split("_")[0]
    cells = {"nm360": set(NMV1), "ilnm": set(NMV1), "ilold": T196, "il311": C311, "simh311": C311, "simh_nm360": set(NMV1),  # simh: 25f / 25g
             "ep2_t196": T196, "ep2_311": C311, "ep2_nm": set(NMV1)}[pk]  # epoch-2 real honest (Append 38): the label's origin list
    tasks = {f["task"] for f in keys.values()}
    m = {"path": p.split("/")[-1], "commit": c, "sha256": hashlib.sha256(mraw).hexdigest(), "rows": len(fs), "staged_arm_rows": len(keys),
         "declared_in_manifest": sum(k in keys for k in decl), "declared": len(decl), "expected_staged": dict(Counter(f["arm_dir"] for k, f in keys.items() if k not in decl)),
         "tasks": len(tasks), "tasks_not_in_pinned_list": sorted(tasks - cells)}
    m["counts_match_routing"] = m["expected_staged"] == {a: arms[a]["epochs"] for a in arms}
    if pk in ("il311", "ilold"):
        m["tasks_not_in_410_keylist"] = sorted(tasks - K410)
    if pk in ("nm360", "ilnm", "simh_nm360"):
        m["nm_built_agrees"] = all(NMB.get(t) == NMV1[t] for t in tasks if t in NMV1)
    res["manifest"] = m
    res["pool_key"] = pk
    st_ = res["stage"]
    res["pass"] = bool(res["attestation"]["pass"] and res["extraction_ok"] and res["repro"]["ok"] and all(res["format"].values())
                       and res["labels"]["match"] and "error" not in st_ and st_["epochs_match"] and st_["shas_match"] and st_["exclusions_match_routing"]
                       and m["counts_match_routing"] and m["declared_in_manifest"] == m["declared"] and not m["tasks_not_in_pinned_list"]
                       and not m.get("tasks_not_in_410_keylist") and m.get("nm_built_agrees", True))
    return res


if __name__ == "__main__":
    out = {"nm_pins_agree": NMV1 == NMB, "roots": {}}
    for a in sys.argv[1:]:
        name, spec = a.split("=", 1)
        routing, attest, rec = spec.rsplit(":", 2)
        if not re.fullmatch(r"[0-9a-f]{7,40}", rec):
            raise SystemExit(f"{name}: the records commit must be a hex sha, not {rec!r}")
        out["roots"][name] = check(routing, attest, rec)
    out["pass"] = out["nm_pins_agree"] and all(v["pass"] for v in out["roots"].values())
    print(json.dumps(out, indent=1))
    sys.exit(0 if out["pass"] else 1)
