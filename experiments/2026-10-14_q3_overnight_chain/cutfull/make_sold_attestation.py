"""d9's CUT fork (Append 41): the S_old (b) full-length attestations, bound to 2e's stager provenance (the AUTHORITY, in place of a
routing record; 09 and 2e, 10-15 ~22:40) and each set's registered stage record. Committed records only; names, keys, counts, hashes.
Every row is "(b) descriptive". Written once (open "x"); a correction is a new _r<k>. Nothing is written if any input is malformed.

Per S_old set (config/cutfull/sold_sets_2026-10-15.json):
1. authority: the provenance file at its commit names the registered stage record (commit:path); the record's staged sha256 per arm
   (its named field) EQUALS the registered staging's file sha256 that the prefix check read (prefix cut_sha); any mismatch fails.
2. one-to-one by key, per arm: the prefix check's keys_equal (cutfull staged keys == registered staged keys) and its per-run checks;
   excluded sets equal (the registered record's c_gate excluded when it has one, else by complement: same manifest, same staged keys).
3. full consumption (as v2): the prefix check (len == the registered tokens_before, every run) and attest_metas (every epoch).
4. membership (09 condition 3): staged keys not in the set's membership and membership keys with no staged epoch, listed by key
   (membership files per set in the config; "pending" when not yet given, which the attestation states).
5. template class per epoch: "render changed with the cut" for keys whose fixed render differs from the original (the cutfull c_gate's
   render_changed, and the registered record's changed keys), else "fixed, render unchanged".
6. dropped arms (q4f_main's sim_honest pilot) stated.

  uv run --no-project python experiments/2026-10-14_q3_overnight_chain/cutfull/make_sold_attestation.py --sets <sold_sets json> --prefix <rec>[,<rec>] --attest <rec>[,<rec>]
"""
import argparse
import glob
import hashlib
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHAIN = HERE.parent
REPO = CHAIN.parents[1]
REL = "experiments/2026-10-14_q3_overnight_chain"
OUT = CHAIN / "results/cutfull/attest"
NAME = re.compile(r"[A-Za-z0-9_.@-]+")


def git(*a, text=True):
    return subprocess.run(["git", *a], capture_output=True, text=text, check=True, cwd=REPO).stdout


def committed(path: Path) -> bytes:
    rel = str(path.resolve().relative_to(REPO))
    blob = subprocess.run(["git", "show", f"HEAD:{rel}"], capture_output=True, cwd=REPO).stdout
    if not blob or blob != path.read_bytes():
        raise SystemExit(f"{rel}: not committed as read")
    return blob


def at(path: Path) -> str:
    rel = str(path.resolve().relative_to(REPO))
    return f"{rel} @ {git('log', '-1', '--format=%H', '--', rel).strip()[:12]}"


def dig(d, dotted):
    for k in dotted.split("."):
        d = d[k]
    return d


HEX64 = re.compile(r"[0-9a-f]{64}")


def isint(v):
    return isinstance(v, int) and not isinstance(v, bool)


def token(arm):
    return arm.split("@")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--attest", required=True)
    ap.add_argument("--revision", type=int, default=1, help="k; 0 = the next revision of each set's attestation")
    ap.add_argument("--format", default="", help="comma-separated every-epoch format-check records (required for revision >= 2 by 0d/09)")
    ap.add_argument("--only", default="", help="comma-separated original targets (a corrected subset)")
    a = ap.parse_args()
    cfg = json.loads(committed(Path(a.sets)))
    global MEM, GROUPS
    GROUPS = {}
    MEM = json.loads(committed(CHAIN / "config" / cfg["membership_file"]))
    if len(set(MEM["attack_universe_195"]["tasks"])) != 195 or len(set(MEM["pool_old_386"]["keys"])) != 386 or len(set(MEM["a25_simh_149"]["keys"])) != 149 \
            or any(len(set(v)) != len(v) for v in (MEM["attack_universe_195"]["tasks"], MEM["pool_old_386"]["keys"], MEM["a25_simh_149"]["keys"])):
        raise SystemExit("membership file counts are not 195 / 386 / 149 unique; nothing written")
    prov_raw = git("show", cfg["provenance"], text=False)
    prov_txt = prov_raw.decode()
    fmt_files = [f for f in a.format.split(",") if f]
    fmt = {}
    for f in fmt_files:
        r = json.loads(committed(Path(f)))
        if (r.get("target"), r.get("arm")) in fmt:
            raise SystemExit("two format-check records for one (target, arm); nothing written")
        fmt[(r.get("target"), r.get("arm"))] = (f, r)
    if a.revision < 0:
        raise SystemExit("revision >= 1")
    pre = [p for f in a.prefix.split(",") for p in json.loads(committed(Path(f)))["pairs"]]
    att = [r for f in a.attest.split(",") for r in json.loads(committed(Path(f)))["roots"]]

    writes, n_fail = [], 0
    only = {z for z in a.only.split(",") if z}
    if only - {s_["original_target"] for s_ in cfg["sets"]}:
        raise SystemExit("--only names unknown sets")
    for s in [s_ for s_ in cfg["sets"] if not only or s_["original_target"] in only]:
        failing = {}

        def bad(reason, pairs):
            failing.setdefault(reason, []).extend(pairs or [["", "(no detail)"]])

        rc, rp = s["registered_record"].split(":", 1)
        section = json.dumps(json.loads(prov_txt)[s["provenance_section"]])  # the AUTHORITY's own entry for this set
        if f"results/{Path(rp).name} @ {rc}" not in section:
            raise SystemExit(f"{s['original_target']}: results/{Path(rp).name} @ {rc} is not named in provenance section {s['provenance_section']}; nothing written")
        reg = json.loads(git("show", s["registered_record"]))
        reg_set = reg[s["record_key"]]
        if reg_set.get("out") != s["original_staged_set"]:
            raise SystemExit(f"{s['original_target']}: the registered record's out is {reg_set.get('out')}; nothing written")
        reg_sha = dig(reg_set, s["sha_field"])
        if not isinstance(reg_sha, dict) or not all(isinstance(v, str) and HEX64.fullmatch(v) for v in reg_sha.values()):
            raise SystemExit(f"{s['original_target']}: malformed registered staged hashes; nothing written")
        def strings(o):
            return [o] if isinstance(o, str) else [z for v in (o.values() if isinstance(o, dict) else o if isinstance(o, list) else []) for z in strings(v)]
        entry = [z for z in strings(json.loads(prov_txt)[s["provenance_section"]]) if f"{Path(rp).name} @ {rc}" in z]
        if len(entry) != 1:
            raise SystemExit(f"{s['original_target']}: {len(entry)} provenance entries name {Path(rp).name}; nothing written")
        prov_listed = {f: f"{f} {v}" in entry[0] for f, v in reg_sha.items()}  # arm file -> THIS record's entry lists "<arm file> <sha256>"
        p = [x for x in pre if x.get("uncut_set") == s["uncut_set"] and x.get("cut_set") == s["original_staged_set"]]
        x = [r for r in att if r.get("uncut_set") == s["uncut_set"] and r.get("root") == s["root"]]
        if len(p) != 1 or len(x) != 1:
            raise SystemExit(f"{s['uncut_set']}: {len(p)} prefix and {len(x)} attestation entries; nothing written")
        p, x = p[0], x[0]
        if s.get("sha_source") == "registered_volume_stage_record":  # q4f_main: its committed record was "not re-assembled"; the provenance
            vr = p.get("cut_stage_record") or {}                        # names the Volume stage record that carries its staged hashes
            if vr.get("name") != s["volume_stage_record"] or s["volume_stage_record"] not in section \
                    or not isinstance(vr.get("staged_inputs_sha256"), dict) or not all(isinstance(z, str) and HEX64.fullmatch(z) for z in vr["staged_inputs_sha256"].values()):
                raise SystemExit(f"{s['original_target']}: the registered Volume stage record is not the provenance's; nothing written")
            reg_sha = vr["staged_inputs_sha256"]
        for m, v in (p.get("arms_checked") or {}).items():
            if not (isinstance(v, dict) and isinstance(v.get("bad"), list) and isinstance(v.get("keys_equal"), bool) and isint(v.get("runs"))
                    and all(isinstance(z, str) and HEX64.fullmatch(z) for z in (v.get("cut_sha"), v.get("uncut_sha")))):
                raise SystemExit(f"{s['uncut_set']}: malformed prefix arm {m}; nothing written")
        if p.get("passes") is True and (p.get("fail") or any(v["bad"] or not v["keys_equal"] for v in p["arms_checked"].values())):
            raise SystemExit(f"{s['uncut_set']}: prefix verdict contradicts its arms; nothing written")
        for tk, v in (x.get("arms") or {}).items():
            ks = v.get("keys") if isinstance(v, dict) else None
            if not (isinstance(ks, list) and all(isinstance(k, str) for k in ks) and len(set(ks)) == len(ks) and isint(v.get("epochs")) and len(ks) == v["epochs"]
                    and isinstance(v.get("staged_sha256"), str) and HEX64.fullmatch(v["staged_sha256"]) and v.get("meta_arm") and token(v["meta_arm"]) == tk):
                raise SystemExit(f"{s['root']}: malformed attestation arm {tk}; nothing written")
        if p.get("passes") is not True:
            bad("prefix check", [[token(m), k] for m, v in p.get("arms_checked", {}).items() for k, _ in v.get("bad", [])] or [["", str(p.get("fail"))[:200]]])
        for reason, pairs in (x.get("failing") or {}).items():
            bad(f"extraction: {reason}", pairs)
        if not isinstance(x.get("failing"), dict):
            raise SystemExit(f"{s['root']}: malformed attestation entry; nothing written")
        arms = sorted(p.get("arms_checked", {}))
        if arms != sorted(s["arms"]) or sorted(v["meta_arm"] for v in x.get("arms", {}).values()) != arms:
            bad("arms", [["", f"prefix {arms}, attest {sorted(x.get('arms', {}))}, config {sorted(s['arms'])}"]])
        for m in arms:
            v = p["arms_checked"][m]
            if reg_sha.get(f"{m}.jsonl") != v.get("cut_sha"):
                bad("authority: registered staged sha256", [[token(m), f"record {reg_sha.get(f'{m}.jsonl')}, Volume {v.get('cut_sha')}"]])
            if v.get("keys_equal") is not True:
                bad("one_to_one keys", [[token(m), ""]])
            xa = x["arms"].get(token(m), {})
            if xa.get("staged_sha256") != v.get("uncut_sha") or xa.get("epochs") != v.get("runs"):
                bad("binding prefix/extraction", [[token(m), f"epochs {xa.get('epochs')} vs {v.get('runs')}"]])
        st = [(f, json.loads(committed(Path(f))).get(s["set"])) for f in sorted(glob.glob(str(CHAIN / f"results/chain_stage_{s['cutfull_target']}_2*.json")))]
        st = [(f, r) for f, r in st if isinstance(r, dict) and r.get("out") == s["uncut_set"]]
        if len(st) != 1:
            raise SystemExit(f"{s['uncut_set']}: {len(st)} cutfull stage records; nothing written")
        st_path, srec = st[0]
        for m in s["arms"]:
            if srec["assembled"]["staged_inputs_sha256"].get(f"{m}.jsonl") != (p.get("arms_checked", {}).get(m) or {}).get("uncut_sha"):
                bad("binding stage record/prefix", [[token(m), ""]])
            if s.get("provenance_lists_sha", True) and not prov_listed.get(f"{m}.jsonl"):
                bad("authority: staged sha256 not the provenance's", [[token(m), reg_sha[f"{m}.jsonl"]]])
        cg = srec.get("c_gate") if isinstance(srec.get("c_gate"), dict) else {}
        if not isinstance(cg.get("excluded"), dict) or not isinstance(cg.get("render_changed"), list) or not all(isinstance(k, str) for k in cg["render_changed"]):
            raise SystemExit(f"{s['uncut_set']}: malformed cutfull c_gate; nothing written")
        reg_ex = reg_set.get("c_gate", {}).get("excluded") if isinstance(reg_set.get("c_gate"), dict) else None
        excl_rule = "registered c_gate excluded keys"
        if reg_ex is None:  # by complement: OUR manifest bytes == the registered manifest, and the staged keys are equal (prefix keys_equal)
            mine, theirs = committed(CHAIN / "config" / s["manifest"]), git("show", s["registered_manifest"], text=False)
            excl_rule = f"by complement: manifest {s['manifest']} == {s['registered_manifest']} (sha256 {hashlib.sha256(mine).hexdigest()[:16]}), staged keys equal"
            if mine != theirs:
                bad("staging exclusions: manifest differs", [["", s["registered_manifest"]]])
        if reg_ex is not None and set(reg_ex) != set(cg.get("excluded", {})):
            bad("staging exclusions", [["", k] for k in sorted(set(reg_ex) ^ set(cg.get("excluded", {})))])
        rch = dig(reg_set, s["changed_field"])  # q4f: arm -> [keys]; c_gate: [keys] (single-arm sets)
        if isinstance(rch, list):
            if len(s["arms"]) != 1:
                raise SystemExit(f"{s['original_target']}: a flat changed-key list for a multi-arm set; nothing written")
            rch = {s["arms"][0]: rch}
        if not isinstance(rch, dict) or not all(isinstance(v, list) and all(isinstance(k, str) for k in v) for v in rch.values()):
            raise SystemExit(f"{s['original_target']}: malformed registered changed keys; nothing written")
        changed = {token(m): (set(cg["render_changed"]) | set(rch.get(m, []))) for m in s["arms"]}  # per arm: its OWN changed keys
        keys = {tk: v.get("keys", []) for tk, v in x["arms"].items()}
        # 09 condition 3: membership by key, per arm (tasks via the committed manifest's task field, never the folder name)
        man = json.loads(committed(CHAIN / "config" / s["manifest"]))
        task_of = {f"{f['launch']}/{f['rel']}#e0": f["task"] for f in man["files"]}
        if len(task_of) != len(man["files"]):
            raise SystemExit(f"{s['manifest']}: duplicate manifest keys; nothing written")
        member = {}
        for m in s["arms"]:
            spec = s["membership"].get(m, "pending")
            ks = keys.get(token(m), [])
            if spec == "pending":
                member[token(m)] = {"status": "pending (09 has not named this arm's membership)"}
                bad("membership pending", [[token(m), "09 condition 3: membership not yet named"]])
                continue
            if spec["kind"] == "listed_no_pool":  # 09: not a registered pool; checked against its own registering record (keys, exclusions)
                member[token(m)] = {"rule": spec, "label": "listed; no pool membership", "staged": len(ks),
                                    "own_record": s["registered_record"], "staged_keys_equal_record": p["arms_checked"][m]["keys_equal"],
                                    "excluded_equal_record": "staging exclusions" not in failing}
                continue
            if spec["kind"] == "keys":  # 09: A25 = the 149 pinned keys; per set, and the group union in the last set of the group
                want = set(MEM[spec["ref"]]["keys"])
                group = GROUPS.setdefault(spec["ref"], set())
                group |= set(ks)
                member[token(m)] = {"rule": spec, "source": MEM[spec["ref"]]["source"], "membership_size": len(want), "staged": len(ks),
                                    "staged_not_in_membership": sorted(set(ks) - want), "membership_not_staged_in_this_set": sorted(want - set(ks))}
                if spec.get("group_last"):
                    member[token(m)] |= {"group_union_equals_membership": group == want, "group_union_minus_membership": sorted(group - want),
                                         "membership_minus_group_union": sorted(want - group)}
                    if group != want:
                        bad("membership: group union", [["", k] for k in sorted(group ^ want)])
                continue
            if spec["kind"] == "tasks":
                want = set(MEM["attack_universe_195"]["tasks"])
            elif spec["kind"] == "pool":
                want = {k.split("|", 1)[1] for k in MEM["pool_old_386"]["keys"] if k.split("|", 1)[0] == spec["world"]}
                if not want:
                    raise SystemExit(f"{s['original_target']}: pool world {spec['world']} has no keys; nothing written")
            else:
                raise SystemExit(f"{s['original_target']}: unknown membership kind; nothing written")
            if any(k not in task_of for k in ks):
                raise SystemExit(f"{s['original_target']}: a staged key is not in the manifest; nothing written")
            staged_tasks = {task_of[k] for k in ks}
            member[token(m)] = {"rule": spec, "source": MEM["attack_universe_195" if spec["kind"] == "tasks" else "pool_old_386"]["source"],
                                "membership_size": len(want), "staged": len(ks),
                                "staged_not_in_membership": sorted(k for k in ks if task_of[k] not in want),
                                "membership_without_staged_epoch": sorted(want - staged_tasks)}  # listed by key / task, never dropped or added
        have = {int(m_.group(1) or 1): f for f in OUT.glob(f"attestations_sold__{s['original_target']}_cutfull*.json")
                if (m_ := re.fullmatch(rf"attestations_sold__{re.escape(s['original_target'])}_cutfull(?:_r(\d+))?\.json", f.name))}
        k = a.revision if a.revision >= 1 else max(have, default=0) + 1
        if have and k <= max(have):
            raise SystemExit(f"{s['original_target']}: revision {k} is not after the latest ({max(have)}); nothing written")
        rk = "" if k == 1 else f"_r{k}"
        base = f"sold__{s['original_target']}_cutfull{rk}"
        fmt_cite, unchanged = {}, None
        if True:  # 0d/09: every epoch's staged ids == the whole render (+ Amendment 12) on EVERY revision; a missing or failing record fails the set
            for m in s["arms"]:
                fr = fmt.get((s["cutfull_target"], m))
                xa = x["arms"][token(m)]
                if fr is None:
                    bad("format check missing", [[token(m), ""]])
                    continue
                fmt_cite[token(m)] = at(Path(fr[0]))
                r_ = fr[1]
                if not isinstance(r_.get("failing_keys"), list) or r_["failing_keys"] or r_.get("passes") is not True \
                        or r_.get("epochs_checked") != xa["epochs"] or r_.get("staged_sha") != [xa["staged_sha256"]]:
                    bad("format check", [[token(m), str(z)] for z in (r_.get("failing_keys") or [f"epochs {r_.get('epochs_checked')} vs {xa['epochs']}, sha bound {r_.get('staged_sha') == [xa['staged_sha256']]}"])])
        if have:  # 4f: staged set, root and per-arm staged shas UNCHANGED from the latest revision (its own check; STOP if not)
            prev = json.loads(committed(have[max(have)]))
            pa = [r for f in prev.get("extraction_attestation", []) for r in json.loads(git("show", f"{f.split(' @ ')[1]}:{f.split(' @ ')[0]}"))["roots"]
                  if r.get("uncut_set") == s["uncut_set"] and r.get("root") == s["root"]]
            prev_sha = {tk: v.get("staged_sha256") for tk, v in pa[0]["arms"].items()} if len(pa) == 1 else None
            now_sha = {tk: v["staged_sha256"] for tk, v in x["arms"].items()}
            if (prev.get("cutfull", {}).get("staged_set"), prev.get("cutfull", {}).get("extraction_root"), prev_sha) != (s["uncut_set"], s["root"], now_sha):
                raise SystemExit(f"STOP: {base}: staged set/root/shas differ from revision {prev.get('revision')}; nothing written")
            unchanged = {"revision": prev.get("revision"), "file": at(have[max(have)]), "staged_set": True, "extraction_root": True, "staged_sha_per_arm": True}
        rec = {"what": "S_old (b) full-length attestation (provenance binding; 09 + 2e, 10-15)", "label": "(b) descriptive", "revision": k,
               "authority": {"provenance": cfg["provenance"], "provenance_sha256": hashlib.sha256(prov_raw).hexdigest(),
                             "registered_record": s["registered_record"], "registered_staged_sha256": reg_sha,
                             "registered_volume_stage_record": p.get("cut_stage_record")},
               "original_target": s["original_target"], "original_staged_set": s["original_staged_set"], "original_volume": s["original_volume"],
               "original_folders": {token(m): f"{s['original_staged_set']}__{token(m)}" for m in s["arms"]},
               "cutfull": {"volume": "probe-shift_aswin-surya_sub-q3cutfull_acts_kill-2026-11-14", "staged_set": s["uncut_set"], "extraction_root": s["root"],
                           "folders": {token(m): f"{s['original_target']}_cutfull_v1__{token(m)}" for m in s["arms"]},  # addendum 2: no _try
                           "stage_record": at(Path(st_path))},
               "dropped_arms": s.get("dropped_arms", []), "dropped_note": s.get("dropped_note"),
               "repro_gate": f"{REL}/results/cutfull_step0_compare_2026-10-15_214206.json @ {git('rev-parse', 'e6896f33b').strip()[:12]} (step-0 keys cf5c4f5e...)",
               "prefix_check": [at(Path(f)) for f in a.prefix.split(",")], "extraction_attestation": [at(Path(f)) for f in a.attest.split(",")],
               "exclusions_rule": excl_rule, "membership": member,
               "template_class": {tk: {k: ("render changed with the cut" if k in changed.get(tk, set()) else "fixed, render unchanged") for k in ks} for tk, ks in keys.items()},
               "membership_file": at(CHAIN / "config" / cfg["membership_file"]) + f" (sha256 {hashlib.sha256(committed(CHAIN / 'config' / cfg['membership_file'])).hexdigest()})",
               "units": {tk: {k2: v.get(k2) for k2 in ("epochs", "units", "units_past_32768")} for tk, v in x["arms"].items()},
               "format_check": fmt_cite, **({"unchanged_from_revision": unchanged} if unchanged else {}),
               "failing": failing}
        writes.append((OUT / f"attestations_{base}.json", json.dumps(rec, indent=1)))
        n_fail += bool(failing)
        print(base, "FAILING" if failing else "ok", {k: len(v) for k, v in failing.items()}, "render-changed", {k: len(v) for k, v in changed.items()})
    if any(d.exists() or not NAME.fullmatch(d.name) for d, _ in writes) or len({d for d, _ in writes}) != len(writes):
        raise SystemExit("existing, duplicate or unsafe output names; nothing written")
    OUT.mkdir(parents=True, exist_ok=True)
    for d, body in writes:
        d.open("x").write(body)
    print(len(writes), "written,", n_fail, "failing")
    if n_fail:
        raise SystemExit(f"{n_fail} failing")


if __name__ == "__main__":
    main()
