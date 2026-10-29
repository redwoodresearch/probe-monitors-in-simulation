"""d9's CUT fork: record-only r2 of the two il-oldr cutfull routing records (run3, sim2), naming the v2 manifests @ 240d27bb0 (bare
launch ids) that the registered packs were scored on (d9 10-16 ~03:20, for 2e's Sol IL scorer). From committed records only. Gates:
r1 passes (attestation failing {}, bound to its routing record; every arm's cited format record passes on its staged sha and keys);
v1 and v2 differ ONLY in files[].launch_id 'runs/<id>' -> '<id>' plus v2's supersedes fields; the staged keys (bare) are exactly
v2's done files of each staged arm. The r2 routing record is r1 in every field except manifest, manifest_sha256, revision,
unchanged_from_revision and revision_reason (the attestation: revision, routing_record, unchanged_from_revision, revision_reason). Nothing is written if any gate fails. With --pointer <commit>: write each r2's attestation pointer.

  python3 experiments/2026-10-14_q3_overnight_chain/cutfull/make_oldr_v2_revision.py [--pointer <commit>]
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHAIN = HERE.parent
REPO = CHAIN.parents[1]
REL = "experiments/2026-10-14_q3_overnight_chain"
OUT_R, OUT_A = CHAIN / "results/cutfull/routing", CHAIN / "results/cutfull/attest"
V1, V2 = "bbeba22f53", "240d27bb0"
V2_SHA = {"run3": "56d997896fb5c1d823110f53e12652d69973ede3ad1f1d03d91024518a381f5c", "sim2": None}  # run3's sha from d9 / the SUPERSEDED note
LAUNCH = "2026-10-15_il_old_rlr"
REC_CHANGED = {"manifest", "manifest_sha256", "revision", "unchanged_from_revision", "revision_reason"}  # the only fields r2 changes
ATT_CHANGED = {"revision", "routing_record", "unchanged_from_revision", "revision_reason"}


def git(*a) -> bytes:
    return subprocess.run(["git", *a], capture_output=True, check=True, cwd=REPO).stdout


def committed(path: Path) -> bytes:
    rel = str(path.resolve().relative_to(REPO))
    blob = subprocess.run(["git", "show", f"HEAD:{rel}"], capture_output=True, cwd=REPO).stdout
    if not blob or blob != path.read_bytes():
        raise SystemExit(f"{rel}: not committed as read")
    return blob


def at(path: Path) -> str:
    rel = str(path.resolve().relative_to(REPO))
    return f"{rel} @ {git('log', '-1', '--format=%H', '--', rel).decode().strip()[:12]}"


def bare(k: str) -> str:
    return k[5:] if k.startswith("runs/") else k


def diffs(a, b, p=""):
    if isinstance(a, dict) and isinstance(b, dict):
        return [d for k in sorted(set(a) | set(b)) for d in (diffs(a.get(k), b.get(k), f"{p}.{k}") if k in a and k in b else [(f"{p}.{k}", a.get(k), b.get(k))])]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in diffs(x, y, f"{p}[{i}]")]
    return [] if a == b else [(p, a, b)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pointer", default="")
    a = ap.parse_args()
    writes = []
    for w in ("run3", "sim2"):
        base = f"il-oldr__ilold_oldr_{w}_cutfull"
        r1, r2 = OUT_R / f"{base}.json", OUT_R / f"{base}_r2.json"
        a1, a2 = OUT_A / f"attestations_{base}.json", OUT_A / f"attestations_{base}_r2.json"
        if a.pointer:
            c = git("rev-parse", "--verify", f"{a.pointer}^{{commit}}").decode().strip()
            rel = lambda f: str(f.resolve().relative_to(REPO))  # noqa: E731
            ap_, rp_, a1_, r1_ = (json.loads(git("show", f"{c}:{rel(f)}")) for f in (a2, r2, a1, r1))
            m2p = f"{REL}/config/manifest_ilold_oldr_{w}_v2.json"
            drop = lambda d, ks: {k: v for k, v in d.items() if k not in ks}  # noqa: E731
            if (ap_.get("failing") != {} or ap_.get("revision") != 2 or rp_.get("revision") != 2 or ap_.get("routing_record") != rel(r2)
                    or drop(rp_, REC_CHANGED) != drop(r1_, REC_CHANGED) or drop(ap_, ATT_CHANGED) != drop(a1_, ATT_CHANGED)
                    or rp_.get("manifest") != f"{m2p} @ {V2}" or rp_.get("manifest_sha256") != hashlib.sha256(git("show", f"{V2}:{m2p}")).hexdigest()
                    or ap_.get("batch") != rp_.get("batch") or f"{ap_.get('original_target')}_cutfull" != rp_.get("target")):
                raise SystemExit(f"{base}: r2 at {c[:12]} is not the record-only revision of r1 (fields, manifest, binding); no pointer written")
            writes.append((OUT_R / f"{base}_r2_attest.json", json.dumps({"batch": ap_["batch"], "original_target": ap_["original_target"], "revision": 2,
                                                                           "routing_record": f"{rel(r2)} @ {c[:12]}", "commit": c[:12], "path": rel(a2), "sets": 1,
                                                                           "failing": ap_["failing"]}, indent=1)))
            continue
        if any(f.exists() for f in OUT_R.glob(f"{base}_r*.json")):
            raise SystemExit(f"{base}: a later revision exists; nothing written")
        rec, att = json.loads(committed(r1)), json.loads(committed(a1))
        rel1 = str(r1.resolve().relative_to(REPO))
        if rec.get("revision") != 1 or att.get("revision") != 1 or att.get("failing") != {} or att.get("routing_record") != rel1 or att.get("batch") != rec.get("batch"):
            raise SystemExit(f"{base}: r1 does not pass or its attestation is not bound to it; nothing written")
        keys = {}
        for arm in rec["arms"]:
            tk = arm["meta_arm"].split("@")[0]
            fp, fc = arm["format_check"].split(" @ ")
            fr = json.loads(git("show", f"{fc}:{fp}"))
            ks = sorted(r_["key"] for r_ in fr.get("rows", []))
            if fr.get("passes") is not True or fr.get("failing_keys") != [] or fr.get("staged_sha") != [arm["staged_sha"]] or fr.get("epochs_checked") != arm["epochs"] \
                    or len(ks) != arm["epochs"] or set(ks) != set(att["template_class"].get(tk, {})):
                raise SystemExit(f"{base}/{tk}: the cited format record does not pass or does not bind to this arm; nothing written")
            keys[arm["meta_arm"]] = ks
        m1p = f"{REL}/config/manifest_ilold_oldr_{w}.json"
        m2p = f"{REL}/config/manifest_ilold_oldr_{w}_v2.json"
        if rec["manifest"] != f"{m1p} @ {V1}":
            raise SystemExit(f"{base}: r1 names {rec['manifest']}, not the v1 manifest @ {V1}; nothing written")
        b2 = git("show", f"{V2}:{m2p}")
        if git("show", f"HEAD:{m2p}") != b2 or (V2_SHA[w] and hashlib.sha256(b2).hexdigest() != V2_SHA[w]):
            raise SystemExit(f"{base}: the v2 manifest at HEAD is not the one @ {V2}, or its sha256 is not the registered one; nothing written")
        m1, m2 = json.loads(git("show", f"{V1}:{m1p}")), json.loads(b2)
        extra = {"supersedes", "supersedes_reason"}
        d = diffs(m1, {k: v for k, v in m2.items() if k not in extra})
        bad = [x for x in d if not (x[0].endswith(".launch_id") and x[0].startswith(".files[") and x[1] == f"runs/{LAUNCH}" and x[2] == LAUNCH)]
        if bad or not d or m2.get("supersedes") != f"manifest_ilold_oldr_{w}.json":
            raise SystemExit(f"{base}: v1 and v2 differ beyond the launch-id form: {bad[:3]}; STOP (tell d9); nothing written")
        for ma, ks in keys.items():  # the staged keys are exactly v2's done files of this arm (bare)
            want = sorted(f"{bare(f['path'])}#e0" for f in m2["files"] if f["arm_dir"] == ma and f.get("state") == "done" and f.get("attempt_kind") != "reference")
            if sorted(bare(k) for k in ks) != want:
                raise SystemExit(f"{base}/{ma}: staged keys differ from v2's files; STOP (tell d9); nothing written")
        reason = (f"d9 10-16 ~03:20: the registered packs were scored on the v2 manifest (bare launch ids, {V2}); r2 names it. v1 -> v2 differs only in "
                  f"{len(d)} files[].launch_id values runs/{LAUNCH} -> {LAUNCH} (asserted); staged set, extraction root and keys unchanged")
        prev = {"revision": 1, "file": at(r1), "staged_set": True, "extraction_root": True, "staged_sha_per_arm": True, "keys": True, "changed_fields": ["manifest", "manifest_sha256"]}
        rec2 = rec | {"manifest": f"{m2p} @ {V2}", "manifest_sha256": hashlib.sha256(b2).hexdigest(), "revision": 2, "unchanged_from_revision": prev, "revision_reason": reason}
        att2 = att | {"revision": 2, "routing_record": str(r2.relative_to(REPO)), "unchanged_from_revision": prev, "revision_reason": reason}
        writes += [(r2, json.dumps(rec2, indent=1)), (a2, json.dumps(att2, indent=1))]
        print(base, "keys", {k: len(v) for k, v in keys.items()}, "launch-id diffs", len(d), "v2 sha", hashlib.sha256(b2).hexdigest()[:12])
    if any(p.exists() for p, _ in writes):
        raise SystemExit("output exists; nothing written")
    for p, body in writes:
        p.open("x").write(body)
    print("written", [p.name for p, _ in writes])


if __name__ == "__main__":
    main()
