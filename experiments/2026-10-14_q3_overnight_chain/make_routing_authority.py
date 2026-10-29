"""The authoritative root of every routed logical batch in the chain's routing directory (keys only): each <batch>__<target>.json, with a
superseded file resolved through its own `revisions` pointer to the revision file, and the earlier staging/extraction tries named by the
input index (non-authoritative). Fails unless every logical batch has exactly one authoritative extraction root, no two batches share
one, and every input-index entry agrees with the routing directory.

  python3 make_routing_authority.py <routing dir> <input index json> <out.json>
"""
import ast
import json
import sys
from pathlib import Path


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def main():
    rdir, idx, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    revision_files, rows = set(), {}
    files = sorted(p for p in rdir.glob("*__*.json"))
    renamed = {}  # re-stage convention (09 @ 65ef3a9ff section 1): the earlier file renamed <base>_v1.json, superseded_by the base-name file
    for p in files:  # revision files are named by their base file's `revisions` pointer
        d = load(p)
        if d.get("superseded") and "superseded_by" in d:
            renamed[p.name] = d["extraction_root"]
        elif d.get("superseded"):
            revs = d["revisions"] if isinstance(d["revisions"], list) else ast.literal_eval(d["revisions"])
            revision_files |= {r["routing_file"] for r in revs}
    for p in files:
        d = load(p)
        if p.name in revision_files or p.name in renamed:
            continue
        key = (d["batch"], d["target"])
        assert key not in rows, key
        if d.get("superseded"):
            revs = d["revisions"] if isinstance(d["revisions"], list) else ast.literal_eval(d["revisions"])
            last = revs[-1]
            r = load(rdir / last["routing_file"])
            assert (r["batch"], r["target"]) == key and r.get("supersedes"), (p.name, last)
            rows[key] = {"routing_file": last["routing_file"], "extraction_root": r["extraction_root"], "staged_set": r.get("staged_set"),
                         "scorer_folder": last.get("scorer_folder"), "superseded": {"routing_file": p.name, "extraction_root": d["extraction_root"]}}
        else:
            rows[key] = {"routing_file": p.name, "extraction_root": d["extraction_root"], "staged_set": d.get("staged_set")}
    for name, root in renamed.items():  # each renamed earlier file has its base-name file, which names a different (later) root
        base = name.removesuffix("_v1.json") + ".json"
        assert base in {v["routing_file"] for v in rows.values()} and load(rdir / base)["extraction_root"] != root, name
    roots = [v["extraction_root"] for v in rows.values()]
    assert len(roots) == len(set(roots)), "two logical batches share an extraction root"
    disagree = []
    for e in load(idx)["entries"]:
        k = (e["batch"], e["target"])
        if k not in rows or rows[k]["extraction_root"] != e["authoritative"]["extraction_root"]:
            disagree.append({"batch": k[0], "target": k[1], "index": e["authoritative"]["extraction_root"], "routing": rows.get(k, {}).get("extraction_root")})
    doc = {"what": "authoritative root per routed logical batch (chain routing directory, revisions resolved); keys only",
           "routing_dir_files": len(files), "revision_files": sorted(revision_files), "logical_batches": len(rows),
           "superseded_resolved": sorted(f"{b}__{t}" for (b, t), v in rows.items() if "superseded" in v),
           "superseded_renamed": renamed,
           "input_index": {"path": str(idx.name), "entries": len(load(idx)["entries"]), "disagree": disagree},
           "batches": {f"{b}__{t}": v for (b, t), v in sorted(rows.items())}}
    Path(out).open("x").write(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: doc[k] for k in ("routing_dir_files", "logical_batches", "superseded_resolved")} | {"index_disagree": len(disagree)}))
    if disagree:
        raise SystemExit(f"input index and routing directory disagree: {disagree}")


if __name__ == "__main__":
    main()
