"""d9's CUT fork: record-only r2 of the two s4rgap cutfull routing records (09 via d9 10-16 ~03:38), adding the per-key authority that
mirrors the 32k originals: nm145_model_registry from the donor-fix retry ilnm_s4rgap_sim2_try2_cutfull_v1__sim_attack (the 32k
il-s4rgap-t2 staging retry, nm145 only), every other key from ilnm_s4rgap_sim2_cutfull_v1__sim_attack. From committed records only.
Gates: both r1 records pass (attestation failing {}, bound); the retry's keys are exactly the nm145 key(s), each a v1 row (superseded) or a v1 staging exclusion the retry restaged;
exactly ONE authoritative row per key across both folders (superseded rows listed by key); each extraction root is in exactly one
active routing file (highest revision per record). The routing r2 is r1 plus key_authority, revision, unchanged_from_revision and
revision_reason (the attestation: revision, routing_record, unchanged_from_revision, revision_reason). Folder names are matched
EXACTLY, never by glob: ilnm_s4rgap_sim2_cutfull and ilnm_s4rgap_sim2_try2_cutfull share a prefix. With --pointer <commit>: pointers.

  python3 experiments/2026-10-14_q3_overnight_chain/cutfull/make_s4rgap_key_authority.py [--pointer <commit>]
"""
import argparse
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHAIN = HERE.parent
REPO = CHAIN.parents[1]
OUT_R, OUT_A = CHAIN / "results/cutfull/routing", CHAIN / "results/cutfull/attest"
V1 = ("il-s4rgap__ilnm_s4rgap_sim2_cutfull", "ilnm_s4rgap_sim2_cutfull_v1", "il-s4rgap__ilnm_s4rgap_sim2.json", "il-s4rgap", "ilnm_s4rgap_sim2")
T2 = ("il-s4rgap-t2__ilnm_s4rgap_sim2_try2_cutfull", "ilnm_s4rgap_sim2_try2_cutfull_v1", "il-s4rgap-t2__ilnm_s4rgap_sim2_try2.json", "il-s4rgap-t2",
      "ilnm_s4rgap_sim2_try2")
TOKEN = "sim_attack"
REC_CHANGED = {"key_authority", "revision", "unchanged_from_revision", "revision_reason"}
ATT_CHANGED = {"revision", "routing_record", "unchanged_from_revision", "revision_reason"}
RULE = ("09 via d9 10-16 ~03:38: per-key authority mirrors the 32k originals exactly (nm145_model_registry from the donor-fix retry, "
        "il-s4rgap-t2; every other key from il-s4rgap); exactly one authoritative row per key across both folders; exact-name matching only")
REASON = "per-key authority across the s4rgap pair (09's conditions via d9 ~03:38): one authoritative row per key; exact names only"


def git(*a) -> bytes:
    return subprocess.run(["git", *a], capture_output=True, check=True, cwd=REPO).stdout


def rel(f: Path) -> str:
    return str(f.resolve().relative_to(REPO))


def reader(commit: str):
    """read(path) -> the committed bytes at `commit`; at HEAD also asserted equal to the working tree (committed records only)."""
    def read(path: Path) -> bytes:
        blob = git("show", f"{commit}:{rel(path)}")
        if commit == "HEAD" and blob != path.read_bytes():
            raise SystemExit(f"{rel(path)}: not committed as read")
        return blob
    return read


def active_roots(commit: str) -> Counter:
    """extraction root -> number of ACTIVE routing records naming it at `commit` (the highest revision of each record base)."""
    top = {}
    for name in git("ls-tree", "--name-only", f"{commit}:{rel(OUT_R)}").decode().split():
        m = re.fullmatch(r"(.+?)(?:_r(\d+))?\.json", name)
        if name.endswith("_attest.json") or not m:
            continue
        k = int(m.group(2) or 1)
        if k >= top.get(m.group(1), (0, None))[0]:
            top[m.group(1)] = (k, name)
    c = Counter()
    for _, name in top.values():
        d = json.loads(git("show", f"{commit}:{rel(OUT_R)}/{name}"))
        if isinstance(d, dict) and d.get("extraction_root"):
            c[d["extraction_root"]] += 1
    return c


def authority(read) -> tuple:
    """From the r1 records (via `read`): the gates, and per record the key_authority block, r1 and its attestation."""
    recs, atts, keys = {}, {}, {}
    for b, ss, orig, batch, ot in (V1, T2):
        rec, att = json.loads(read(OUT_R / f"{b}.json")), json.loads(read(OUT_A / f"attestations_{b}.json"))
        if (rec.get("revision") != 1 or att.get("revision") != 1 or att.get("failing") != {} or att.get("routing_record") != rel(OUT_R / f"{b}.json")
                or rec.get("staged_set") != ss or rec.get("original_routing", {}).get("routing_file") != orig or set(att["template_class"]) != {TOKEN}
                or rec.get("batch") != batch or att.get("batch") != batch or att.get("original_target") != ot or rec.get("target") != f"{ot}_cutfull"):
            raise SystemExit(f"{b}: r1 does not pass, is not bound, or is not {ss} ({batch}, {ot}) from {orig}; nothing written")
        recs[b], atts[b], keys[b] = rec, att, sorted(att["template_class"][TOKEN])
        if len(keys[b]) != sum(x["epochs"] for x in rec["arms"]) or len(set(keys[b])) != len(keys[b]):
            raise SystemExit(f"{b}: attested keys and routed epochs differ; nothing written")
    k1, k2 = keys[V1[0]], keys[T2[0]]
    n1, ex1 = {k for k in k1 if "/nm145_model_registry/" in k}, recs[V1[0]].get("staging_exclusions") or {}
    if not k2 or any("/nm145_model_registry/" not in k for k in k2) or (n1 and n1 != set(k2)) or (not n1 and not all(k in ex1 for k in k2)):
        raise SystemExit(f"the retry's keys {k2} are neither v1's nm145 rows nor v1 staging exclusions; nothing written")
    resolved = {k: ex1[k] for k in k2 if k not in n1}  # no v1 row: v1's staging excluded the key (e.g. no usable donor), the retry restaged it
    f1, f2 = f"{V1[1]}__{TOKEN}", f"{T2[1]}__{TOKEN}"
    auth = {f1: [k for k in k1 if k not in k2], f2: list(k2)}
    rows = Counter(k for ks in auth.values() for k in ks)
    if set(rows) != set(k1) | set(k2) or any(n != 1 for n in rows.values()):
        raise SystemExit("not exactly one authoritative row per key across both folders; nothing written")
    out = {}
    for b, folder, other in ((V1[0], f1, f2), (T2[0], f2, f1)):
        out[b] = ({"rule": RULE, "folder": folder, "sibling_folder": other, "exact_name_matching_only": True,
                   "authoritative_keys": auth[folder], "n_authoritative": len(auth[folder]),
                   "superseded_keys": [{"key": k, "authoritative_in": other} for k in keys[b] if k not in auth[folder]],
                   "keys_across_both_folders": len(rows), "v1_staging_exclusions_restaged_by_the_retry": resolved}, recs[b], atts[b])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pointer", default="")
    a = ap.parse_args()
    c = git("rev-parse", "--verify", f"{a.pointer}^{{commit}}").decode().strip() if a.pointer else "HEAD"
    ka = authority(reader(c))  # pointer mode: recomputed from r1 AT the commit, then compared with the committed r2
    live = active_roots(c)
    if any(live.get(r["extraction_root"]) != 1 for _, r, _ in ka.values()):
        raise SystemExit(f"an extraction root is not in exactly one active routing file at {c[:12]}: {[(r['extraction_root'], live.get(r['extraction_root'])) for _, r, _ in ka.values()]}")
    writes = []
    for b, (block, rec, att) in ka.items():
        r1, r2, a2 = OUT_R / f"{b}.json", OUT_R / f"{b}_r2.json", OUT_A / f"attestations_{b}_r2.json"
        if a.pointer:
            drop = lambda d, ks: {k: v for k, v in d.items() if k not in ks}  # noqa: E731
            rp_, ap_ = json.loads(git("show", f"{c}:{rel(r2)}")), json.loads(git("show", f"{c}:{rel(a2)}"))
            if (ap_.get("failing") != {} or ap_.get("revision") != 2 or rp_.get("revision") != 2 or ap_.get("routing_record") != rel(r2)
                    or drop(rp_, REC_CHANGED) != drop(rec, REC_CHANGED) or drop(ap_, ATT_CHANGED) != drop(att, ATT_CHANGED)
                    or rp_.get("key_authority") != block or rp_.get("revision_reason") != REASON or ap_.get("revision_reason") != REASON
                    or ap_.get("unchanged_from_revision") != rp_.get("unchanged_from_revision") or rp_["unchanged_from_revision"].get("changed_fields") != ["key_authority"]
                    or ap_.get("batch") != rec["batch"]):
                raise SystemExit(f"{b}: r2 at {c[:12]} is not the record-only key-authority revision of r1; no pointer written")
            writes.append((OUT_R / f"{b}_r2_attest.json", json.dumps({"batch": ap_["batch"], "original_target": ap_["original_target"], "revision": 2,
                                                                        "routing_record": f"{rel(r2)} @ {c[:12]}", "commit": c[:12], "path": rel(a2),
                                                                        "sets": 1, "failing": ap_["failing"]}, indent=1)))
            continue
        if [f.name for f in OUT_R.iterdir() if re.fullmatch(rf"{re.escape(b)}_r\d+(_attest)?\.json", f.name)] or a2.exists():
            raise SystemExit(f"{b}: a later revision exists; nothing written")
        prev = {"revision": 1, "file": f"{rel(r1)} @ {git('log', '-1', '--format=%H', '--', rel(r1)).decode().strip()[:12]}", "staged_set": True,
                "extraction_root": True, "staged_sha_per_arm": True, "keys": True, "changed_fields": ["key_authority"]}
        writes += [(r2, json.dumps(rec | {"key_authority": block, "revision": 2, "unchanged_from_revision": prev, "revision_reason": REASON}, indent=1)),
                   (a2, json.dumps(att | {"revision": 2, "routing_record": rel(r2), "unchanged_from_revision": prev, "revision_reason": REASON}, indent=1))]
        print(b, "authoritative", block["n_authoritative"], "superseded", len(block["superseded_keys"]))
    if any(p.exists() for p, _ in writes):
        raise SystemExit("output exists; nothing written")
    for p, body in writes:
        p.open("x").write(body)
    print("written", [p.name for p, _ in writes])


if __name__ == "__main__":
    main()
