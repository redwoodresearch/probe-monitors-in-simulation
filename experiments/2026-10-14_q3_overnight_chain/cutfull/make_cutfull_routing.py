"""d9's CUT fork (Append 41): write the cutfull routing records and their attestations (cutfull/cutfull_routing_v2.json), one per
ORIGINAL routing record of each cutfull staged set, from committed records only (names, keys, counts, hashes; no transcripts, no
Volume reads). Revision 1 files are written once (open "x"); a correction is a new _r<k>.

Every output is built in memory first; malformed input raises and NOTHING is written (loud failure); destinations must be new
and unique. Inputs: the private targets file, the chain's routing directory (the original records) and its committed routing authority, the
committed records of: the step-0 gate, the prefix check (staging side), attest_metas (extraction side), the cutfull stage and
extraction records. Each record must say passes, else its keys go to `failing` and the revision fails.

  python3 experiments/2026-10-14_q3_overnight_chain/cutfull/make_cutfull_routing.py --targets <file> --routing-dir <dir> \
      --prefix <results/cutfull_prefix_check_*.json> --attest <results/cutfull_attest_metas_*.json>
  then commit, and run again with --pointers <that commit> to write the attestation pointers.
"""
import argparse
import re
import glob
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHAIN = HERE.parent
REPO = CHAIN.parents[1]
REL = "experiments/2026-10-14_q3_overnight_chain"
VOLUME = "probe-shift_aswin-surya_sub-q3cutfull_acts_kill-2026-11-14"
GATE = f"{REL}/results/cutfull_step0_compare_2026-10-15_214206.json @ e6896f33b825"
KEYS = "experiments/2026-10-15_cut_sensitivity/results/step0_keys_v2_2026-10-15.json @ f3bfe92080bb (sha256 cf5c4f5e3e0e0664...)"
AUTH = CHAIN / "results/hygiene/routing_authority_2026-10-15.json"
OUT_R, OUT_A = CHAIN / "results/cutfull/routing", CHAIN / "results/cutfull/attest"


NAME = re.compile(r"[A-Za-z0-9_.@-]+")
HEX64 = re.compile(r"[0-9a-f]{64}")


def pairs_ok(f) -> bool:
    """A failing dict: {reason: [[arm token, detail], ...]} with strings throughout."""
    return isinstance(f, dict) and all(isinstance(k, str) and isinstance(v, list) and all(
        isinstance(q, list) and len(q) == 2 and all(isinstance(z, str) for z in q) for q in v) for k, v in f.items())


def isint(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def token(arm: str) -> str:
    return arm.split("@")[0]


def git(*a) -> str:
    return subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()


def committed(path: Path) -> bytes:
    """The file's bytes, required to equal its blob at HEAD (committed evidence only)."""
    rel = str(path.resolve().relative_to(REPO))
    blob = subprocess.run(["git", "show", f"HEAD:{rel}"], capture_output=True, cwd=REPO).stdout
    if not blob or blob != path.read_bytes():
        raise SystemExit(f"{rel}: not committed as read")
    return blob


def at(path: Path) -> str:
    rel = str(path.resolve().relative_to(REPO))
    return f"{rel} @ {git('log', '-1', '--format=%H', '--', rel)[:12]}"


def authoritative(recs: list, auth: dict) -> list:
    """When a staged set has several original routing records (a re-extraction revision, e.g. il-s5__ilnm_s5_sim2 and its _try2), keep
    the ones the routing authority names; the others are superseded originals and get no cutfull record."""
    if len(recs) <= 1:
        return recs
    ss = {r[1].get("staged_set") for r in recs}
    named = {v.get("routing_file") for v in auth.values() if v.get("staged_set") in ss}
    keep = [r for r in recs if r[0].name in named]
    if not keep:
        raise SystemExit(f"{sorted(ss)}: several original routing records and none named by the authority; nothing written")
    return keep


def original_records(routing_dir: Path, staged_set: str) -> list:
    """The ORIGINAL routing records of a staged set: every file in the chain's routing directory naming it (superseded files aside)."""
    out = []
    for f in sorted(routing_dir.glob("*__*.json")):
        raw = f.read_bytes()  # the chain's routing directory is not in git: parse and hash the SAME bytes, bound by the authority entry
        try:
            d = json.loads(raw)
        except ValueError:  # an unreadable routing file could be an original: abort, never skip
            raise SystemExit(f"{f.name}: unparsable routing file; nothing written") from None
        if isinstance(d, dict) and d.get("staged_set") == staged_set and "superseded" not in f.name:
            out.append((f, d, hashlib.sha256(raw).hexdigest()))
    return out


def one(rows: list, what: str):
    return rows[0] if len(rows) == 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--routing-dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--attest", required=True)
    ap.add_argument("--pointers", default="", help="the commit holding the records: write the attestation pointers")
    ap.add_argument("--authority", default=str(AUTH), help="the committed routing authority file (append-only versions, e.g. _v2)")
    ap.add_argument("--only", default="", help="comma-separated original targets to write (a correction for a subset); default all")
    ap.add_argument("--format", default="", help="comma-separated every-epoch format-check records (results/format_full_*.json)")
    ap.add_argument("--format-policy", required=True, choices=["required", "gate_only"],
                    help="required: every arm needs a passing every-epoch record ((a''), 09); gate_only: format_check not run, format_gate only ((b), 09)")
    ap.add_argument("--revision", type=int, default=1, help="k: files <base>_r<k> for k >= 2 (a correction; the highest k is authoritative)")
    a = ap.parse_args()
    if a.revision < 0:
        raise SystemExit("revision >= 1, or 0 for the next revision of each record")
    fmt = [json.loads(committed(Path(f))) for f in a.format.split(",") if f]  # every-epoch format-check records (09), per (target, arm)
    fmt_path = {(r.get("target"), r.get("arm")): f for f, r in zip([f for f in a.format.split(",") if f], fmt)}
    if len(fmt_path) != len(fmt):
        raise SystemExit("two format-check records for one (target, arm); nothing written")

    def existing(batch, target):  # the committed revisions of one routing record: {k: path}
        out = {}
        for f in OUT_R.glob(f"{batch}__{target}_cutfull*.json"):
            m = re.fullmatch(rf"{re.escape(batch)}__{re.escape(target)}_cutfull(?:_r(\d+))?\.json", f.name)
            if m:
                out[int(m.group(1) or 1)] = f
        return out
    T = json.loads(committed(Path(a.targets)))
    prefix, attest = json.loads(committed(Path(a.prefix))), json.loads(committed(Path(a.attest)))
    gate_raw = committed(REPO / GATE.split(" @ ")[0])
    if gate_raw != subprocess.run(["git", "show", f"{GATE.split(' @ ')[1]}:{GATE.split(' @ ')[0]}"], capture_output=True, cwd=REPO).stdout:
        raise SystemExit("the gate record at HEAD differs from the cited commit")
    gate = json.loads(gate_raw)
    auth_path = Path(a.authority)
    auth = json.loads(committed(auth_path))["batches"]
    only = {x for x in a.only.split(",") if x}
    if only - {n_[: -len("_cutfull")] for n_ in T}:
        raise SystemExit(f"--only names unknown targets: {sorted(only - {n_[: -len('_cutfull')] for n_ in T})}")
    T = {n_: t_ for n_, t_ in T.items() if not only or n_[: -len("_cutfull")] in only}
    OUT_R.mkdir(parents=True, exist_ok=True)
    OUT_A.mkdir(parents=True, exist_ok=True)
    n, n_fail, writes = 0, 0, []  # every output is built first; ANY exception (malformed input) aborts with nothing written
    for name, t in sorted(T.items()):  # before ANY write: every original record exists and has the fields used below
        found = authoritative(original_records(Path(a.routing_dir), t["arm_list"]["original_staged_set"]), auth)
        want = {v.get("routing_file") for v in auth.values() if v.get("staged_set") == t["arm_list"]["original_staged_set"]}
        if not want <= {f.name for f, _, _ in found}:
            raise SystemExit(f"{t['arm_list']['original_staged_set']}: authority routing files {sorted(want - {f.name for f, _, _ in found})} not found; nothing written")
        for f, o, _ in found or [(None, None, None)]:
            if o is None or not isinstance(o.get("batch"), str) or not isinstance(o.get("extraction_root"), str) or not isinstance(o.get("arms"), list) \
                    or not all(isinstance(ar, dict) and isinstance(ar.get("meta_arm"), str) and isinstance(ar.get("epochs"), int) and not isinstance(ar.get("epochs"), bool) for ar in o["arms"]) \
                    or not isinstance(o.get("staging_exclusions", {}), dict) or not NAME.fullmatch(o["batch"]) \
                    or len({ar["meta_arm"] for ar in o["arms"]}) != len(o["arms"]):
                raise SystemExit(f"{t['arm_list']['original_staged_set']}: original routing record missing or malformed ({f}); nothing written")
    for name, t in sorted(T.items()):
        target, orig_set, out = name[: -len("_cutfull")], t["arm_list"]["original_staged_set"], t["out"]
        root = re.sub(r"_v1(_try[0-9]+)?$", r"_extract_v1\1", out)  # the TRAILING _v1[_try<n>] (A25 sets carry _v1 mid-name)
        for f, o, o_sha in authoritative(original_records(Path(a.routing_dir), orig_set), auth) or [(None, None, None)]:
            if o is None:
                raise SystemExit(f"{orig_set}: no original routing record")
            batch = o["batch"]
            have = existing(batch, target)
            k = a.revision if a.revision >= 1 else (max(have, default=0) if a.pointers else max(have, default=0) + 1)  # pointers: the latest record
            rk = "" if k == 1 else f"_r{k}"
            base = f"{batch}__{target}_cutfull{rk}"
            if have and not a.pointers and k <= max(have):
                raise SystemExit(f"{base}: revision {k} is not after the latest ({max(have)}); nothing written")
            prev = json.loads(committed(have[max(have)])) if have and not a.pointers else None  # always the LATEST existing revision
            if a.pointers:
                c = git("rev-parse", "--verify", f"{a.pointers}^{{commit}}")
                att_rel, rec_rel = f"{REL}/results/cutfull/attest/attestations_{base}.json", f"{REL}/results/cutfull/routing/{base}.json"
                ap_ = json.loads(git("show", f"{c}:{att_rel}"))
                git("cat-file", "-e", f"{c}:{rec_rel}")
                if not pairs_ok(ap_.get("failing")) or ap_.get("batch") != batch or ap_.get("original_target") != target \
                        or not isint(ap_.get("revision")) or ap_.get("revision") != k or ap_.get("routing_record") != rec_rel:
                    raise SystemExit(f"{att_rel} @ {c[:12]}: malformed or unbound attestation; no pointers written")
                n_fail += bool(ap_["failing"])
                writes.append((OUT_R / f"{base}_attest.json", json.dumps({
                    "batch": batch, "original_target": target, "revision": k, "routing_record": f"{rec_rel} @ {c[:12]}", "commit": c[:12],
                    "path": att_rel, "sets": 1, "failing": ap_["failing"]}, indent=1)))
                n += 1
                continue
            failing = {}

            def bad(reason, pairs):
                failing.setdefault(reason, []).extend(pairs or [["", "(no detail)"]])

            if gate.get("passes") is not True or gate.get("n_pass") != 26:
                bad("repro_gate", [["", GATE]])
            st = sorted(glob.glob(str(CHAIN / f"results/chain_stage_{name}_2*.json")))
            ex = sorted(glob.glob(str(CHAIN / f"results/extract_chain_{root}_a*_2*.json")), key=lambda e: (int(e.split(f"{root}_a")[1].split("_")[0]), e))
            srec = json.loads(committed(Path(st[0])))[t.get("set", "c")] if len(st) == 1 else None
            if srec is None:
                bad("stage record", [["", f"{len(st)} stage records"]])
                srec = {"assembled": {"epochs": {}, "staged_inputs_sha256": {}, "format_gate": {}}}
            A = srec.get("assembled") if isinstance(srec.get("assembled"), dict) else {}
            c_epochs = A.get("epochs") if isinstance(A.get("epochs"), dict) else {}
            c_sha = A.get("staged_inputs_sha256") if isinstance(A.get("staged_inputs_sha256"), dict) else {}
            fg = A.get("format_gate") if isinstance(A.get("format_gate"), dict) else {}
            if A.get("STOP") or not c_epochs or set(fg) != set(c_epochs) or set(c_sha) != {f"{m}.jsonl" for m in c_epochs} \
                    or not all(isinstance(v, str) and HEX64.fullmatch(v) for v in c_sha.values()) \
                    or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in c_epochs.values()):
                bad("stage arms", [["", f"STOP {A.get('STOP')}, epochs {c_epochs}, gate arms {sorted(fg)}, sha arms {sorted(c_sha)}"]])
                c_epochs = {m: v for m, v in c_epochs.items() if isinstance(v, int) and not isinstance(v, bool) and v > 0}  # invalid data not used further
            for m, g in fg.items():
                if not isinstance(g, dict) or g.get("passes") is not True or g.get("think_open_truncated_at_cut_runs") != 0:
                    bad("stage format gate", [[token(m), f"passes {g.get('passes') if isinstance(g, dict) else g}, think_open_truncated_at_cut_runs {g.get('think_open_truncated_at_cut_runs') if isinstance(g, dict) else None}"]])
            cg = srec.get("c_gate") if isinstance(srec.get("c_gate"), dict) else {}
            if not cg or not isinstance(cg.get("excluded"), dict) or cg.get("staged") != sum(c_epochs.values()) or cg.get("manifest_keys") != cg.get("staged", 0) + len(cg["excluded"]):
                bad("c_gate", [["", f"c_gate {({k: cg.get(k) for k in ('manifest_keys', 'staged')}) if cg else 'missing'}"]])
                cg = {"excluded": cg.get("excluded") if isinstance(cg.get("excluded"), dict) else {}}
            xrecs = [json.loads(committed(Path(e))) for e in ex]
            # the LAST attempt is the one that completed the root (a resumed attempt counts every epoch of its shards, done ones included);
            # earlier attempts' shard errors are what the resume fixed and stay listed in extraction_record
            shards_ok = bool(xrecs) and isinstance(xrecs[-1], dict) and isinstance(xrecs[-1].get("shards"), list) and xrecs[-1]["shards"] \
                and all(isinstance(s_, dict) and "error" not in s_ and isint(s_.get("epochs")) for s_ in xrecs[-1]["shards"])
            if not xrecs or not shards_ok or sum(s_["epochs"] for s_ in xrecs[-1]["shards"]) != sum(c_epochs.values()):
                bad("extraction record", [["", f"{len(xrecs)} records"]])
            p = one([x for x in prefix["pairs"] if x.get("uncut_set") == out], "prefix")
            if p is not None and (not isinstance(p.get("arms_checked"), dict) or not all(
                    isinstance(v, dict) and isinstance(v.get("bad"), list) and all(isinstance(q, list) and len(q) == 2 and all(isinstance(z, str) for z in q) for q in v["bad"]) for v in p["arms_checked"].values())):
                raise SystemExit(f"{out}: malformed prefix-check entry; nothing written")
            if p is None or p.get("passes") is not True or p.get("cut_set") != orig_set or not p.get("arms_checked") \
                    or set(p["arms_checked"]) != set(c_epochs):
                bad("prefix check", [[token(m), k] for m, v in (p or {}).get("arms_checked", {}).items() for k, _ in v.get("bad", [])]
                    or [["", f"prefix entry for {out}: {'missing' if p is None else p.get('fail')}"]])
            else:
                for m, v in p["arms_checked"].items():
                    if v.get("keys_equal") is not True or v.get("bad") != [] or not isint(v.get("runs")) or v.get("runs") != c_epochs[m] \
                            or not (isinstance(v.get("uncut_sha"), str) and HEX64.fullmatch(v["uncut_sha"])) or v.get("uncut_sha") != c_sha.get(f"{m}.jsonl"):
                        bad("prefix check binding", [[token(m), f"keys_equal {v.get('keys_equal')}, runs {v.get('runs')}, sha bound {v.get('uncut_sha') == c_sha.get(f'{m}.jsonl')}"]])
            x = one([r for r in attest["roots"] if r.get("uncut_set") == out], "attest")
            if x is not None and (not pairs_ok(x.get("failing")) or not isinstance(x.get("arms"), dict) or not all(
                    isinstance(v, dict) and isinstance(v.get("keys"), list) and all(isinstance(k, str) for k in v["keys"]) for v in x["arms"].values())):
                raise SystemExit(f"{root}: malformed extraction-attestation entry; nothing written")
            if x is None or x.get("root") != root:
                bad("extraction attestation", [["", f"entry for {root}: {'missing' if x is None else 'malformed'}"]])
                x = {"failing": {}, "arms": {}}
            for reason, pairs in x["failing"].items():
                ok_shape = isinstance(pairs, list) and all(isinstance(q, list) and len(q) == 2 for q in pairs)
                bad(f"extraction: {reason}", pairs if ok_shape else [["", f"malformed failure entry {str(pairs)[:120]}"]])
            xa = x.get("arms") if isinstance(x.get("arms"), dict) else {}
            if not all(isinstance(v, dict) and isinstance(v.get("keys"), list) and all(isinstance(k, str) for k in v["keys"]) for v in xa.values()):
                bad("extraction arms shape", [["", root]])
                xa = {}
            elif sorted(xa) != sorted(token(m) for m in c_epochs) or any(tk != token(str(v.get("meta_arm"))) or v.get("meta_arm") not in c_epochs for tk, v in xa.items()):
                bad("extraction arms", [["", f"{sorted(xa)} vs {sorted(c_epochs)}"]])
                xa = {}
            for tk, v in xa.items():
                m = v.get("meta_arm")
                keys = v.get("keys") if isinstance(v.get("keys"), list) else []
                if not isint(v.get("epochs")) or v.get("epochs") != c_epochs.get(m) or not (isinstance(v.get("staged_sha256"), str) and HEX64.fullmatch(v["staged_sha256"])) \
                        or v.get("staged_sha256") != c_sha.get(f"{m}.jsonl") or len(keys) != c_epochs.get(m) or len(set(keys)) != len(keys):
                    bad("extraction binding", [[tk, f"epochs {v.get('epochs')} vs {c_epochs.get(m)}, sha bound {v.get('staged_sha256') == c_sha.get(f'{m}.jsonl')}"]])
            tmpl = x.get("chat_template_sha256")
            if not (isinstance(tmpl, dict) and tmpl.get("fixed_staged") and tmpl.get("original")):
                bad("chat template pin", [["", root]])
            if a.format_policy == "gate_only" and fmt:
                raise SystemExit("format records given with --format-policy gate_only; nothing written")
            if a.format_policy == "required":  # 09: EVERY epoch's staged ids == the whole render, plus Amendment 12; failing epochs withheld by key
                for m in c_epochs:
                    fr = next((r for r in fmt if r.get("target") == name and r.get("arm") == m), None)
                    if fr is None:
                        bad("format check missing", [[token(m), ""]])
                    elif fr.get("passes") is not True or fr.get("epochs_checked") != c_epochs[m] or fr.get("staged_sha") != [c_sha.get(f"{m}.jsonl")]:
                        bad("format check", [[token(m), str(z)] for z in (fr.get("failing_keys") or [f"epochs {fr.get('epochs_checked')} vs {c_epochs[m]}, sha bound {fr.get('staged_sha') == [c_sha.get(f'{m}.jsonl')]}"])])
            o_arms = {ar["meta_arm"]: ar for ar in o["arms"]}
            if set(o_arms) != set(c_epochs) or any(o_arms[m]["epochs"] != c_epochs[m] for m in o_arms):
                bad("one_to_one epochs", [[token(m), f"original {o_arms.get(m, {}).get('epochs')}, cutfull {c_epochs.get(m)}"] for m in sorted(set(o_arms) | set(c_epochs))])
            excl = cg.get("excluded", {})
            # the original's exclusions: its routing record's field when recorded, else (null in the chain's records) the ORIGINAL
            # staging's own c_gate (its chain stage record whose out is the original staged set, committed)
            o_excl = o.get("staging_exclusions")
            if o_excl is None:
                ost = []
                for fs in sorted(glob.glob(str(CHAIN / f"results/chain_stage_{target}_2*.json"))):
                    r_ = json.loads(committed(Path(fs)))
                    v_ = r_.get(t.get("set", "c")) if isinstance(r_, dict) else None
                    if isinstance(v_, dict) and v_.get("out") == orig_set:
                        ost.append(v_)
                o_excl = ost[0].get("c_gate", {}).get("excluded") if len(ost) == 1 and isinstance(ost[0].get("c_gate"), dict) else None
                if not isinstance(o_excl, dict):
                    bad("original exclusions", [["", f"{orig_set}: {len(ost)} original stage records with a c_gate"]])
                    o_excl = {}
            if set(excl) != set(o_excl):
                bad("staging exclusions", [["", k] for k in sorted(set(excl) ^ set(o_excl))])
            if len({token(m) for m in o_arms}) != len(o_arms):
                bad("arm tokens not unique", [["", orig_set]])
            au = [v for v in auth.values() if v.get("routing_file") == f.name]
            if len(au) != 1 or au[0].get("staged_set") != orig_set or au[0].get("extraction_root") != o["extraction_root"]:
                bad("routing authority", [["", f"{f.name}: {len(au)} authority entries in {auth_path.name}"]])
            arms = [{"meta_arm": m, "world": o_arms[m].get("world"), "sim_launch_ne_run1_launch": o_arms[m].get("sim_launch_ne_run1_launch"),
                     "staged_inputs_file": f"{out}/inputs/{m}.jsonl", "epochs": c_epochs.get(m), "staged_sha": c_sha.get(f"{m}.jsonl"),
                     "think_open_truncated_at_cut_runs": fg[m].get("think_open_truncated_at_cut_runs") if isinstance(fg.get(m), dict) else None,
                     "format_check": (at(Path(fmt_path[(name, m)])) if (name, m) in fmt_path else None) if a.format_policy == "required"
                     else "not run; format_gate only (09)",
                     **({"stage_counts": o_arms[m]["stage_counts"]} if "stage_counts" in o_arms[m] else {})} for m in sorted(o_arms)]
            rec = {k: o[k] for k in ("batch", "stratum", "tag", "pool", "members", "runs", "manifest", "labels", "sealed") if k in o} | {
                "target": f"{target}_cutfull", "volume": VOLUME, "staged_set": out, "extraction_root": root,
                "fmt_gate_sha256": A.get("fmt_gate_sha256"), "repro_gate": GATE, "step0_key_list": KEYS,
                "staging_exclusions": excl, "arms": arms, "original_folders": {token(m): f"{orig_set}__{token(m)}" for m in o_arms},
                "cut": "full", "revision": k,
                "original_routing": {"routing_file": f.name, "sha256": o_sha, "authority": at(auth_path) if len(au) == 1 else None},
                "original_staged_set": orig_set, "original_extraction_root": o["extraction_root"], "stage_record": at(Path(st[0])) if st else None,
                "extraction_record": [{"path": at(Path(e)).split(" @ ")[0], "commit": at(Path(e)).split(" @ ")[1]} for e in ex],
                "chat_template_sha256": tmpl, "routing_spec": f"{REL}/cutfull/cutfull_routing_v2.json @ c75141175cb9"}
            attn = {"what": "cutfull attestation (cutfull_routing_v2.json)", "batch": batch, "original_target": target, "revision": k,
                    "routing_record": f"{REL}/results/cutfull/routing/{base}.json", "repro_gate": GATE, "prefix_check": at(Path(a.prefix)),
                    "extraction_attestation": at(Path(a.attest)),
                    "template_class": {tk: {k: "fixed" for k in v.get("keys", [])} for tk, v in xa.items()},
                    "template_evidence": "the original 32k staging was made by the chain's stage_chain.py, which applies the Amendment 9 fixed template to every epoch; per-run prefix identity in the prefix check",
                    "units": {tk: {k2: v.get(k2) for k2 in ("epochs", "units", "units_past_32768")} for tk, v in xa.items()},
                    "failing": failing}
            if prev is not None:  # 4f: the staged sets and staged shas are UNCHANGED from the current revision (its own check; STOP if not)
                same = (prev.get("staged_set"), prev.get("extraction_root"), {x["meta_arm"]: x["staged_sha"] for x in prev.get("arms", [])}) == \
                       (rec["staged_set"], rec["extraction_root"], {x["meta_arm"]: x["staged_sha"] for x in rec["arms"]})
                if not same:
                    raise SystemExit(f"STOP: {base}: staged set/root/shas differ from revision {prev.get('revision')}; nothing written")
                rec["unchanged_from_revision"] = {"revision": prev.get("revision"), "file": at(have[max(have)]), "staged_set": True, "extraction_root": True, "staged_sha_per_arm": True}
            writes += [(OUT_R / f"{base}.json", json.dumps(rec, indent=1)), (OUT_A / f"attestations_{base}.json", json.dumps(attn, indent=1))]
            n += 1
            n_fail += bool(failing)
            print(base, "FAILING" if failing else "ok", {k: len(v) for k, v in failing.items()})
    dests = [d for d, _ in writes]
    if any(d.parent.resolve() not in (OUT_R.resolve(), OUT_A.resolve()) or not NAME.fullmatch(d.name) for d in dests):
        raise SystemExit("an output path leaves the routing/attest directories; nothing written")
    if len(set(dests)) != len(dests) or any(d.exists() for d in dests):
        raise SystemExit("duplicate or existing output paths; nothing written (a correction is a new _r<k>)")
    for d, body in writes:
        d.open("x").write(body)
    print(n, "written,", n_fail, "failing")
    if n_fail:
        raise SystemExit(f"{n_fail} attestations failing (written; they do not route)")


if __name__ == "__main__":
    main()
