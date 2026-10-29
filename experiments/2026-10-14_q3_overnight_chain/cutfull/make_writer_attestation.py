"""d9's CUT fork: the writer attestation of one cutfull root (09 via d9 10-16, the Q4 precedent @ 7c86d8397), from committed records only
(keys, counts, hashes). If the fresh full re-extraction B is bit-identical to the routed root A on every epoch, A stands: the routing record
and attestation are re-issued at the NEXT revision, unchanged in every field except revision, writer_attestation and
unchanged_from_revision, so p2/famexp packs bound to the previous revision confirm by sha. If not identical, nothing is written here
(B replaces A through the routing writer at a new revision).

  python3 experiments/2026-10-14_q3_overnight_chain/cutfull/make_writer_attestation.py --base <batch>__<target>_cutfull --compare <compare_roots record> \
      --fresh-record <extract_chain record of B>
"""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHAIN = HERE.parent
REPO = CHAIN.parents[1]
OUT_R, OUT_A = CHAIN / "results/cutfull/routing", CHAIN / "results/cutfull/attest"


def committed(path: Path) -> bytes:
    rel = str(path.resolve().relative_to(REPO))
    blob = subprocess.run(["git", "show", f"HEAD:{rel}"], capture_output=True, cwd=REPO).stdout
    if not blob or blob != path.read_bytes():
        raise SystemExit(f"{rel}: not committed as read")
    return blob


def at(path: Path) -> str:
    rel = str(path.resolve().relative_to(REPO))
    c = subprocess.run(["git", "log", "-1", "--format=%H", "--", rel], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()
    return f"{rel} @ {c[:12]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--compare", required=True)
    ap.add_argument("--fresh-record", required=True)
    ap.add_argument("--pointer", default="", help="the commit holding the new revision: write its attestation pointer")
    ap.add_argument("--replace-root", default="", help="the fresh root's attest_metas record: the next revision points at the FRESH root (0d/09 10-16)")
    a = ap.parse_args()
    have = {}
    for f in OUT_R.glob(f"{a.base}*.json"):
        m = re.fullmatch(rf"{re.escape(a.base)}(?:_r(\d+))?\.json", f.name)
        if m:
            have[int(m.group(1) or 1)] = f
    k0 = max(have)
    if a.pointer:  # the latest revision is the one this script wrote; its pointer, from the committed files at that commit
        c = subprocess.run(["git", "rev-parse", "--verify", f"{a.pointer}^{{commit}}"], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()
        rel = lambda f: str(f.resolve().relative_to(REPO))  # noqa: E731
        att_p = OUT_A / f"attestations_{have[k0].name}"
        ap_ = json.loads(subprocess.run(["git", "show", f"{c}:{rel(att_p)}"], capture_output=True, check=True, cwd=REPO).stdout)
        rp_ = json.loads(subprocess.run(["git", "show", f"{c}:{rel(have[k0])}"], capture_output=True, check=True, cwd=REPO).stdout)
        if ("writer_attestation" not in ap_ or ap_.get("failing") != {} or ap_.get("revision") != k0 or rp_.get("revision") != k0
                or ap_.get("routing_record") != rel(have[k0]) or rp_.get("writer_attestation") != ap_.get("writer_attestation")):
            raise SystemExit("the latest revision is not a passing writer-attestation revision; no pointer written")
        dest = OUT_R / f"{a.base}_r{k0}_attest.json"
        dest.open("x").write(json.dumps({"batch": ap_["batch"], "original_target": ap_["original_target"], "revision": k0,
                                         "routing_record": f"{rel(have[k0])} @ {c[:12]}", "commit": c[:12], "path": rel(att_p), "sets": 1,
                                         "failing": ap_["failing"]}, indent=1))
        print("pointer", dest.name)
        return
    rec = json.loads(committed(have[k0]))
    att_f = OUT_A / f"attestations_{have[k0].name}"
    att = json.loads(committed(att_f))
    cmp_ = json.loads(committed(Path(a.compare)))
    fresh = json.loads(committed(Path(a.fresh_record)))
    rel_rec = str(have[k0].resolve().relative_to(REPO))
    if att.get("failing") != {} or att.get("revision") is None or att.get("revision") != rec.get("revision") or att.get("routing_record") != rel_rec \
            or att.get("batch") != rec.get("batch") or f"{att.get('batch')}__{att.get('original_target')}_cutfull" != a.base:
        raise SystemExit("the latest revision does not pass, or its attestation is not bound to its routing record; nothing written")
    n_ep = sum(x["epochs"] for x in rec["arms"])
    cfg_b = fresh.get("config", {})
    if (not fresh.get("shards") or len(fresh["shards"]) != cfg_b.get("shards") or sum(s_.get("epochs", 0) for s_ in fresh["shards"]) != n_ep
            or cfg_b.get("inputs") != rec["staged_set"] or {f"{x['meta_arm']}.jsonl": x["staged_sha"] for x in rec["arms"]} != cfg_b.get("inputs_sha256")):
        raise SystemExit("the fresh extraction record is incomplete or not of this staged set; nothing written")
    ok = (cmp_.get("identical") is True and cmp_.get("a") == rec["extraction_root"] and cmp_.get("staged_set") == rec["staged_set"]
          and cmp_.get("n_epochs") == sum(x["epochs"] for x in rec["arms"]) == cmp_.get("n_bit_identical")
          and fresh["config"]["run"] == cmp_.get("b") and all("error" not in s and s.get("done_before") == 0 and s.get("gpu_name") == "NVIDIA H200"
                                                              for s in fresh["shards"]))
    if not ok:
        raise SystemExit("not identical, or the records do not bind: nothing written (the fresh root replaces it via the routing writer)")
    wa = {"rule": "09 via d9 10-16 (the Q4 precedent @ 7c86d8397): a full fresh H200 re-extraction of the same staged inputs, bit for bit on every epoch x 64 layers; identical -> the routed root stands",
          "routed_root": rec["extraction_root"], "fresh_root": cmp_["b"], "compare": at(Path(a.compare)), "fresh_extraction_record": at(Path(a.fresh_record)),
          "epochs_bit_identical": cmp_["n_bit_identical"], "epochs": cmp_["n_epochs"], "fresh_gpus": cmp_["gpus"]["b"], "verdict": "the routed root stands"}
    k = k0 + 1
    prev = {"revision": k0, "file": at(have[k0]), "staged_set": True, "extraction_root": True, "staged_sha_per_arm": True}
    rec2 = rec | {"revision": k, "writer_attestation": wa, "unchanged_from_revision": prev}
    att2 = att | {"revision": k, "writer_attestation": wa, "unchanged_from_revision": prev,
                  "routing_record": str((OUT_R / f"{a.base}_r{k}.json").relative_to(REPO))}  # the only changed fields
    if a.replace_root:  # the FRESH root replaces the routed one (its bytes are identical; its records carry no error entry)
        fa = json.loads(committed(Path(a.replace_root)))
        xr = [r for r in fa.get("roots", []) if r.get("root") == cmp_["b"] and r.get("uncut_set") == rec["staged_set"]]
        if len(xr) != 1 or xr[0].get("failing") != {} or fa.get("passes") is not True or {v.get("meta_arm"): v.get("staged_sha256") for v in xr[0]["arms"].values()} \
                != {x["meta_arm"]: x["staged_sha"] for x in rec["arms"]}:
            raise SystemExit("the fresh root's attestation does not pass or does not bind to this staged set; nothing written")
        wa |= {"verdict": "the FRESH root replaces the routed root (bit-identical bytes; the routed root's a1 shard-2 entry is error-only)"}
        prev = prev | {"extraction_root": False, "extraction_root_replaced": {"from": rec["extraction_root"], "to": cmp_["b"]}}
        rec2 = rec | {"revision": k, "writer_attestation": wa, "unchanged_from_revision": prev, "extraction_root": cmp_["b"],
                      "extraction_record": [{"path": at(Path(a.fresh_record)).split(" @ ")[0], "commit": at(Path(a.fresh_record)).split(" @ ")[1]}]}
        att2 = att | {"revision": k, "writer_attestation": wa, "unchanged_from_revision": prev, "extraction_attestation": at(Path(a.replace_root)),
                      "units": {tk: {k2: v.get(k2) for k2 in ("epochs", "units", "units_past_32768")} for tk, v in xr[0]["arms"].items()},
                      "routing_record": str((OUT_R / f"{a.base}_r{k}.json").relative_to(REPO))}
    dests = [OUT_R / f"{a.base}_r{k}.json", OUT_A / f"attestations_{a.base}_r{k}.json"]
    if any(d.exists() for d in dests):
        raise SystemExit("output exists; nothing written")
    dests[0].open("x").write(json.dumps(rec2, indent=1))
    dests[1].open("x").write(json.dumps(att2, indent=1))
    print("written", [d.name for d in dests], "verdict:", wa["verdict"], hashlib.sha256(dests[0].read_bytes()).hexdigest()[:12])


if __name__ == "__main__":
    main()
