"""Routing authority, next version (append-only; the earlier file stays): an earlier authority file's batches UNCHANGED plus exactly the named
logical batches, each read from its routing file (<batch>__<target>.json) and checked by key against committed records: its coverage (a
replacement root's rlr coverage file: staged set and staged sha256; any other root: the one committed simh root coverage file listing its staged
set and staging record, with nothing missing, doubled or not ran), its staging record and a committed extraction record of its root. Fails loudly on a key already present, a
shared extraction root, an unattested or failing batch, or any mismatch. Keys and hashes only.

  python3 make_routing_authority_append.py <earlier authority json> <routing dir> <out.json> <batch>__<target> ...
"""
import ast
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def tracked(pattern: str) -> list:
    return subprocess.run(["git", "ls-files", pattern], capture_output=True, text=True, check=True, cwd=HERE).stdout.split()


def main():
    prev_p, rdir, out, keys = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4:]
    prev = json.loads(prev_p.read_text())
    batches = dict(prev["batches"])
    added = {}
    for key in keys:
        assert key not in batches, f"{key} is already in {prev_p.name}"
        bid, target = key.split("__", 1)
        r = json.loads((rdir / f"{key}.json").read_text())
        assert (r["batch"], r["target"]) == (bid, target) and not r.get("superseded"), key
        att = json.loads((rdir / f"{bid}_attest.json").read_text())
        assert att["failing"] == {} and len(att["commit"]) == 40, f"{bid}: attestation {att['failing']}"
        arms = r["arms"] if isinstance(r["arms"], list) else ast.literal_eval(r["arms"])
        (arm,) = arms
        assert r["extraction_root"] == r["staged_set"].replace("_v1", "_extract_v1"), key
        ext = tracked(f"results/extract_chain_{r['extraction_root']}_a*.json")
        stg = tracked(f"results/chain_stage_{target}_2026-*.json")
        assert ext and len(stg) == 1, f"{key}: committed extraction records {ext}, staging records {stg}"
        rlr_cov = HERE / "results" / "hygiene" / f"simh_rlr_coverage_{bid}_2026-10-15.json"
        if rlr_cov.exists():  # a replacement root (_rlr / _rlr<k>)
            cov_path = f"results/hygiene/{rlr_cov.name}"
            cov = json.loads(rlr_cov.read_text())
            assert cov["problems"] == [] and cov["staged_set"] == r["staged_set"] and cov["staged_sha256"] == arm["staged_sha"], f"{key}: coverage disagrees"
        else:  # a quarter's snapshot or remainder: the committed root coverage that lists this staged set
            hits = [(p, x) for p in tracked("results/hygiene/simh_root_coverage_*_2026-10-15.json")
                    for x in (lambda d: d["roots"] if isinstance(d["roots"], list) else ast.literal_eval(d["roots"]))(json.loads((HERE / p).read_text()))
                    if x["staged_set"] == r["staged_set"]]
            assert len(hits) == 1, f"{key}: {len(hits)} root coverage entries list {r['staged_set']}"
            cov_path, x = hits[0]
            cov = json.loads((HERE / cov_path).read_text())
            counts = cov["counts"] if isinstance(cov["counts"], dict) else ast.literal_eval(cov["counts"])
            assert x["target"] == target and x["staging_record"].split(" @ ")[0] == stg[0].rsplit("/", 1)[1], f"{key}: root coverage entry {x}"
            assert counts["missing"] == counts["in_several_roots"] == counts["in_roots_not_ran"] == 0, f"{key}: root coverage counts {counts}"
        added[key] = {"routing_file": f"{key}.json", "extraction_root": r["extraction_root"], "staged_set": r["staged_set"]}
        batches[key] = added[key] | {"staged_sha256": arm["staged_sha"], "epochs": arm["epochs"], "attestation": f"{att['path']} @ {att['commit'][:12]}",
                                     "coverage": cov_path, "staging_record": stg[0], "extraction_records": ext}
    roots = [v["extraction_root"] for v in batches.values()]
    assert len(roots) == len(set(roots)), "two logical batches share an extraction root"
    doc = {"what": prev["what"] + "; next version: the earlier file's batches unchanged plus the appended batches (append-only)",
           "supersedes_file": prev_p.name, "appended": sorted(added), "earlier_batches": len(prev["batches"]), "logical_batches": len(batches),
           **{k: prev[k] for k in ("revision_files", "superseded_resolved", "superseded_renamed", "input_index")}, "batches": dict(sorted(batches.items()))}
    Path(out).open("x").write(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"earlier": len(prev["batches"]), "appended": len(added), "logical_batches": len(batches),
                      "epochs_appended": sum(batches[k]["epochs"][next(iter(batches[k]["epochs"]))] if isinstance(batches[k]["epochs"], dict) else batches[k]["epochs"] for k in added)}))


if __name__ == "__main__":
    main()
