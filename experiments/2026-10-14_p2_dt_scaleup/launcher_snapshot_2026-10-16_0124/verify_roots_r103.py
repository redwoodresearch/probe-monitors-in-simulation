"""Pre-launch verification by key for d9 roots (4f, 10-14 ~23:30). Reads d9's routing files (read-only, another session's scratchpad)
and committed records from git. Keys, shas and counts only. Usage: python3 -I verify_roots.py <routing dir> <repo> <out json> <spec json>"""
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

RT, REPO, OUT, SPEC = Path(sys.argv[1]), sys.argv[2], sys.argv[3], json.loads(sys.argv[4])
show = lambda s: subprocess.run(["git", "-C", REPO, "show", s], capture_output=True, check=True).stdout  # noqa: E731
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
at = lambda s: tuple(x.strip() for x in s.split(" @ "))  # noqa: E731  "path @ commit"
C311 = {c["key"]: c for c in json.loads(show("0caadbfb5:experiments/2026-10-14_q2_scaleup_cohort/results/scaleup_tasks_plus311_2026-10-14_125135.json"))["cells"]}
NMV1 = {c["key"]: c for c in json.loads(show("ef171c813:experiments/2026-10-14_q3_new_mains_mode/config/new_mains_cells_v1.json"))["cells"]}
NMB = {c["key"]: c["main"] for c in json.loads(show("6feeee4b7:experiments/2026-10-14_q3_new_mains_launch/config/nm_built.json"))["cells"]}
out = {"nm_pins_agree": {k: v["main"] for k, v in NMV1.items()} == NMB, "roots": {}}


def pointer(name: str) -> tuple:  # 4f 10-15 07:3x: follow d9's pointer revisions; never use a superseded or non-authoritative pointer
    hexc = lambda c: isinstance(c, str) and re.fullmatch(r"[0-9a-f]{7,40}", c) is not None  # noqa: E731  Codex round 52: no empty or odd commits
    full = lambda c: subprocess.run(["git", "-C", REPO, "rev-parse", "--verify", "--quiet", f"{c}^{{commit}}"], capture_output=True, text=True).stdout.strip()  # noqa: E731
    chain = []
    while True:
        ap = json.loads((RT / name).read_bytes())
        chain.append(name)
        if ap.get("authoritative") is False:  # Codex round 52: checked before any revision is followed, on every pointer in the chain
            raise SystemExit(f"{name}: marked authoritative: false")
        if ap.get("superseded") is True:
            rev = (ap.get("revisions") or [None])[-1]
            nxt = rev.get("pointer") if isinstance(rev, dict) else None
            if not nxt or nxt in chain:
                raise SystemExit(f"{name}: superseded, with no usable revisions[-1] pointer")
            ap2 = json.loads((RT / nxt).read_bytes())
            if not (hexc(rev.get("commit")) and hexc(ap2.get("commit")) and full(rev["commit"]) and full(rev["commit"]) == full(ap2["commit"])
                    and rev.get("path") and rev.get("path") == ap2.get("path")):
                raise SystemExit(f"{name}: revisions[-1] and {nxt} disagree on the attestation commit or path, or one is empty or unresolvable")
            name = nxt
            continue
        if not (hexc(ap.get("commit")) and full(ap["commit"]) and ap.get("path")):
            raise SystemExit(f"{name}: the final pointer needs a resolvable commit and a path")
        return ap, chain


INDEX = json.loads(show("976603962:experiments/2026-10-14_q3_overnight_chain/results/hygiene/input_index_retries_2026-10-15_v5.json"))  # DT Amendment 36: v5
if sha(json.dumps(INDEX).encode()) and sha(show("976603962:experiments/2026-10-14_q3_overnight_chain/results/hygiene/input_index_retries_2026-10-15_v5.json")) != \
        "17fd765b86323896fffbd6ef245ad021f9cca39f1427e71d1f777467b400d95d":
    raise SystemExit("the input index differs from its pin")
for name, s in SPEC.items():
    rraw = (RT / s["routing"]).read_bytes()
    r = json.loads(rraw)
    if r.get("superseded") is True or r.get("authoritative") is False:
        raise SystemExit(f"{s['routing']}: a superseded or non-authoritative routing file")
    ap, ptr_chain = pointer(s["attest"])
    ent = [e for e in INDEX["entries"] if e["target"] == r.get("target")]  # 4f 10-15 08:2x: resolve a retried root's identity from d9's input index
    if len(ent) > 1:
        raise SystemExit(f"{s['routing']}: two input-index entries for {r.get('target')}")
    if ent:
        a = ent[0]["authoritative"]
        want = a["staged_sha"] if isinstance(a["staged_sha"], dict) else None
        got = {f"{x['meta_arm']}.jsonl": x["staged_sha"] for x in r["arms"]}
        if a["staged_set"] != r["staged_set"] or (want is not None and want != got) or (want is None and [x["staged_sha"] for x in r["arms"]] != [a["staged_sha"]]):
            raise SystemExit(f"{s['routing']}: the routing's staged set or sha differs from d9's input index authoritative entry")
        if not a.get("extraction_root"):  # DT Amendment 33 (Codex 78): an index entry without its extraction root is held
            raise SystemExit(f"{s['routing']}: d9's input index names no extraction root for {r.get('target')} yet; held")
        r["extraction_root"] = a["extraction_root"]
    res = {"routing_sha256": sha(rraw), "index_resolved": bool(ent), "root": r["extraction_root"], "staged_set": r["staged_set"], "stratum": r["stratum"], "pool": r["pool"]}
    # attestation at its commit: failing {} and every staged sha covered with all checks passing, H200 gate included
    att = json.loads(show(f"{ap['commit']}:{ap['path']}"))
    arms = {a["meta_arm"]: a for a in r["arms"]}
    cov = {}
    for a in arms.values():
        checks = att["attestations"].get(a["staged_sha"], [])
        h2 = [c for c in checks if c["check"] == "longest_epoch_h200_gate"]
        cov[a["meta_arm"]] = {"checks": {c["check"]: c["pass"] for c in checks},
                              "h200": bool(h2) and h2[0]["pass"] and h2[0]["detail"].get("every_shard_h200") is True and h2[0]["detail"].get("last_attempt_no_errors") is True,
                              "h200_evidence": h2[0]["evidence"]["path"].split("/")[-1] if h2 else None}
    res["attestation"] = {"pointer_chain": ptr_chain, "pointer_failing": ap.get("failing"), "file_failing": att.get("failing", "absent"), "commit": ap["commit"][:10],
                          "covers": cov, "pass": ap.get("failing") == {} and att.get("failing", {}) == {} and all(v["h200"] and v["checks"] and all(v["checks"].values()) for v in cov.values())}
    # extraction records for this root, every attempt on the chain branches (21g applies if attempt > 1 or done_before > 0)
    ex = {}
    for br in dict.fromkeys(("origin/aswin/q3-overnight-chain", "origin/aswin/q3-overnight-chain-next", s["records"])):  # DT 39: also the root's records commit (ha25f: aswin/q3-ha25f)
        for f in subprocess.run(["git", "-C", REPO, "ls-tree", "-r", "--name-only", br, "experiments/2026-10-14_q3_overnight_chain/results/"], capture_output=True, text=True).stdout.split():
            if f.split("/")[-1].startswith(f"extract_chain_{r['extraction_root']}_a"):
                d = json.loads(show(f"{br}:{f}"))
                ex[f.split("/")[-1]] = {"attempt": d["config"].get("attempt"), "gpu": dict(Counter(x.get("gpu_name") for x in d["shards"])),
                                        "errors": sum("error" in x for x in d["shards"]), "done_before": sum(x.get("done_before", 0) for x in d["shards"] if "error" not in x),
                                        "epochs": sum(x.get("epochs", 0) for x in d["shards"] if "error" not in x)}
    res["extraction_records"] = ex
    res["needs_21g"] = any(v["attempt"] != 1 or v["done_before"] for v in ex.values()) or len(ex) != 1
    # repro, format, labels
    p, c = at(r["repro_gate"]); x = json.loads(show(f"{c}:{p}"))
    res["repro"] = {"epochs": x["epochs"], "passing": x["passing"], "bit_exact": x["bit_exact"], "failing_shards": x["failing_shards"], "one_of_each": x["coverage"]["every_shard_one_of_each"],
                    "ok": x["epochs"] == x["passing"] == x["bit_exact"] and not x["failing_shards"] and x["coverage"]["every_shard_one_of_each"]}
    res["format"] = {}
    for a in arms.values():
        p, c = at(a["format_check"]); x = json.loads(show(f"{c}:{p}"))
        res["format"][a["meta_arm"]] = x["passes"] is True and x["staged_sha"] == a["staged_sha"]
    if r.get("labels") and not str(r["labels"]).startswith("not applicable"):  # d9: "not applicable: sim_honest@september is an honest arm"
        lp, lc = r["labels"].split(" (sha256 ")[0], r["labels"].split(" @ ")[-1] if " @ " in r["labels"] else s["records"]
        lsha = r["labels"].split("(sha256 ")[1].split(")")[0]
        res["labels"] = {"sha256": lsha[:12], "match": sha(show(f"{lc}:{lp}")) == lsha}
    else:  # DT Amendment 31: an honest-type (sim-honest) root has no attack labels; the launcher accepts this only for sim-honest roots
        res["labels"] = {"absent": True, "match": None, "routing_says": r.get("labels")}
    # stage record (from the attestation's corrected_render evidence_extra), against the routing arms and exclusions
    srp = next((e["path"] for v in att["attestations"].values() for ck in v for e in ck.get("evidence_extra", []) if e["path"].split("/")[-1].startswith(f"chain_stage_{r['target']}_")), None)
    st = json.loads(show(f"{s['records']}:{srp}"))
    sets = {k: v for k, v in st.items() if isinstance(v, dict) and "c_gate" in v}
    (sk, sv), = sets.items()
    res["stage"] = {"record": srp.split("/")[-1], "epochs_match": all(sv["assembled"]["epochs"].get(a) == arms[a]["epochs"] for a in arms) and set(sv["assembled"]["epochs"]) == set(arms),
                    "shas_match": all(sv["assembled"]["staged_inputs_sha256"].get(f"{a}.jsonl") == arms[a]["staged_sha"] for a in arms),
                    "exclusions_match_routing": sv["c_gate"].get("excluded", {}) == r.get("staging_exclusions", {}), "excluded": len(sv["c_gate"].get("excluded", {}))}
    # manifest at its commit: exclusions are manifest rows; staged counts; cells in the pinned list for the pool
    p, c = at(r["manifest"]); mraw = show(f"{c}:{p}"); fs = json.loads(mraw)["files"]
    keys = {f"{f['launch']}/{f['rel']}#e0": f for f in fs if f["arm_dir"] in arms}
    decl = sv["c_gate"].get("excluded", {})
    T196C = {c["key"]: c for c in json.loads(show("HEAD:experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"))["cells"]}
    pk = r["staged_set"].split("_")[0] if r["pool"] == "il" else r["pool"]  # IL routing says pool "il"; the staged-set prefix decides
    res["pool_key"] = pk
    simh = "simh_nm360" if r["staged_set"].startswith("simh_nm360_") else "simh311" if r["staged_set"].startswith("simh311_") else None  # DT Amendment 31
    pk = simh or pk
    res["pool_key"] = pk
    cells = NMV1 if pk in ("nm360", "ilnm", "simh_nm360") else T196C if pk in ("ilold", "simh_t196fill") else {**T196C, **C311} if pk == "ug" else C311  # DT 38: ug; 43: t196fill
    tasks = {f["task"] for f in keys.values()}
    if pk.startswith("sp493"):  # DT Amendment 45: the sp493 cell list at the launched HEAD; a staged t987 row is not in it (09 Q2 (a): refused)
        SP = {c["key"]: c for c in json.loads(show("HEAD:experiments/2026-10-14_p2_dt_scaleup/config/sp493_cells_v1.json"))["cells"]}
        sp_t196 = {(c["env"], c["main"]) for c in SP.values() if c["main_origin"] == "t196"}
        cells = SP | ({k: c for k, c in T196C.items() if (c["env"], c["main"]) in sp_t196} if pk == "sp493_g4_simh" else {})  # 09 Q2; g4 rule (~23:5x)
        # DT 45e/45f (d9 NOTE @ 1d624a961: one origin .eval donates to several cells, so donor rows can share a path): count and drop them from the
        # row LIST before keying by path, so a donor row can never overwrite or hide a staged row
        res["sp493_reference_rows"] = sum(f["arm_dir"] in arms and f.get("attempt_kind") == "reference" for f in fs)
        keys = {f"{f['launch']}/{f['rel']}#e0": f for f in fs if f["arm_dir"] in arms and f.get("attempt_kind") != "reference"}
        if len(keys) != sum(f["arm_dir"] in arms and f.get("attempt_kind") != "reference" for f in fs):
            raise SystemExit(f"{s['routing']}: two non-reference rows share a path; stop")
        res["sp493_declared_t987_rows"] = sum(f["task"] == "t987_model_registry" for k, f in keys.items() if k in decl)
        tasks = {f["task"] for k, f in keys.items() if k not in decl}  # staged rows only: d9 may list t987 as a staged-out row (09: unscored and listed)
    res["manifest"] = {"path": p.split("/")[-1], "commit": c, "sha256": sha(mraw), "rows": len(fs), "staged_arm_rows": len(keys),
                       "by_arm_state": dict(Counter(f"{f['arm_dir']}/{f['world']}/{f.get('state')}" for f in fs)), "strata": dict(Counter(f.get("stratum") for f in fs)),
                       "declared_in_manifest": sum(k in keys for k in decl), "declared": len(decl),
                       "expected_staged": dict(Counter(f["arm_dir"] for k, f in keys.items() if k not in decl)),
                       "tasks_not_in_pinned_list": sorted(tasks - set(cells)), "tasks": len(tasks)}
    res["manifest"]["counts_match_routing"] = res["manifest"]["expected_staged"] == {a: arms[a]["epochs"] for a in arms}
    if pk in ("il311", "ilold"):  # Append 32 section 8: coverage against the frozen 410-cell keylist
        kl = json.loads(show("2e75fa8c5:experiments/2026-10-14_il_launch_09/results/il_keylists_2026-10-14.json"))["estimation_set"]
        res["manifest"]["tasks_not_in_410_keylist"] = sorted(tasks - set(kl["t196_tasks"] + kl["plus311_cells"]))
    if pk in ("nm360", "ilnm", "simh_nm360"):
        res["manifest"]["nm_built_agrees"] = all(NMB.get(t) == NMV1[t]["main"] for t in tasks if t in NMV1)
    out["roots"][name] = res
Path(OUT).write_text(json.dumps(out, indent=1))
for n, v in out["roots"].items():
    m = v["manifest"]
    print(f"{n}: root {v['root']} set {v['staged_set']} {v['stratum']} | attest pass {v['attestation']['pass']} (ptr failing {v['attestation']['pointer_failing']}, file {v['attestation']['file_failing']})"
          f" | extraction {[(k.split('_a')[-1][:1], e['attempt'], e['gpu'], e['errors'], e['done_before']) for k, e in v['extraction_records'].items()]} 21g {v['needs_21g']}")
    print(f"   repro {v['repro']['ok']} ({v['repro']['bit_exact']}/{v['repro']['epochs']}) | format {v['format']} | labels {v['labels']} | stage {v['stage']}")
    print(f"   manifest {m['path']}@{m['commit']} sha {m['sha256'][:16]} rows {m['rows']} {m['by_arm_state']} strata {m['strata']} | declared {m['declared_in_manifest']}/{m['declared']}"
          f" | expected {m['expected_staged']} match {m['counts_match_routing']} | tasks {m['tasks']} not-in-pinned {m['tasks_not_in_pinned_list'][:4]}" + (f" nm_built {m.get('nm_built_agrees')}" if "nm_built_agrees" in m else "") + (f" not-in-410 {m['tasks_not_in_410_keylist']}" if "tasks_not_in_410_keylist" in m else ""))
print("nm pins agree:", out["nm_pins_agree"])
