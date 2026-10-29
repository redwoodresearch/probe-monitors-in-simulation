"""Shared-extraction scan (0d / 09 via 2e 17:3x; NOT RUN until 09's go, the pair step after the nm172 hash check): do two different run keys share
one source extraction (a pipeline alias)? KEYS, IDENTIFIERS AND COUNTS ONLY: no transcript, message or tool text is printed or written.

On the activation Volume, each staging root <name>_v1*/inputs/<arm>.jsonl holds one staged run per line, and each extraction root
<name>_extract_v1*/meta/<arm>/<id>.json holds one extracted run's metadata. For every line and meta file, the scan keeps only: the run key, the
NAMES of the top-level fields, and the values of provenance-like fields (names matching PROV) when the value is a short identifier (a path, sha, uuid
or id: no whitespace, at most 512 characters). It then groups by each (field, value) and by the meta file's own id, and reports every group that
holds more than one distinct run key: such a group is a shared source extraction. --keys restricts the report to groups that hold a named key (the
nm172 pair step) and lists those keys' own provenance. Usage: modal run scan_shared_extraction.py --out <new local json> [--keys k1,k2] [--roots r1,r2]"""
import json
import re

import modal

VOL = "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14"
app = modal.App("probe-shift_aswin-surya_sub-e2esim_aliasscan_kill-2026-11-16")
image = modal.Image.debian_slim(python_version="3.12")
vol = modal.Volume.from_name(VOL)
PROV = re.compile(r"eval|path|sha|source|uuid|sample|file|launch|run_id|task_id|epoch|origin|src|transcript_id", re.I)
KEYF = ("key", "run_key", "rid", "run")  # the first present field whose value looks like "<world>|<cell>" is the run key
KEYV = re.compile(r"^[^|\s]+\|[^|\s]+$")
IDV = re.compile(r"^\S{1,512}$")
# grouped only when the value identifies ONE extraction source: a file (.eval / .json / .jsonl / .npz), a hex digest, or a long opaque id (uuid,
# base62); directories, epochs, origins, launch names and task ids are shared by many runs legitimately and are reported, never grouped
GRP = re.compile(r"(\.(eval|json|jsonl|npz)$)|(^[0-9a-f]{16,64}$)|(^[A-Za-z0-9_-]{16,64}$)")


def short_id(v):
    return isinstance(v, str) and bool(IDV.match(v)) or isinstance(v, (int, float)) and not isinstance(v, bool)


def one(d):  # -> (run key or None, field names, {field: identifier}) of one staged line or meta record; values of other fields are never kept
    if not isinstance(d, dict):
        return None, [], {}
    k = next((d[f] for f in KEYF if isinstance(d.get(f), str) and KEYV.match(d[f])), None)
    return k, sorted(d), {f: v for f, v in d.items() if PROV.search(f) and short_id(v)}


@app.function(image=image, cpu=2, memory=8192, timeout=3600, retries=0, volumes={"/acts": vol.read_only()})
def scan_root(root: str) -> dict:
    from pathlib import Path

    base, out = Path("/acts") / root, {"root": root, "staged": [], "meta": [], "fields": {}, "errors": []}
    for f in sorted(base.glob("inputs/*.jsonl")):
        with f.open() as fh:
            for i, line in enumerate(fh):
                try:
                    k, names, pv = one(json.loads(line))
                except json.JSONDecodeError:
                    out["errors"].append(f"{f.name}:{i}: not JSON")
                    continue
                out["staged"].append({"file": f"{root}/inputs/{f.name}", "line": i, "key": k, "prov": pv})
                for n in names:
                    out["fields"][f"staged:{n}"] = out["fields"].get(f"staged:{n}", 0) + 1
    for f in sorted(base.glob("meta/*/*.json")):
        try:
            k, names, pv = one(json.loads(f.read_text()))
        except json.JSONDecodeError:
            out["errors"].append(f"{f.parent.name}/{f.name}: not JSON")
            continue
        out["meta"].append({"file": f"{root}/meta/{f.parent.name}/{f.name}", "extraction_id": f.stem, "key": k, "prov": pv})
        for n in names:
            out["fields"][f"meta:{n}"] = out["fields"].get(f"meta:{n}", 0) + 1
    return out


@app.function(image=image, cpu=1, memory=2048, timeout=7200, retries=0, volumes={"/acts": vol.read_only()})
def drive(roots: list, keys: list) -> dict:
    """The one spawned call: lists the roots (all staging and extraction roots unless named), scans them in parallel, groups, reports."""
    from pathlib import Path

    if not roots:
        roots = sorted(p.name for p in Path("/acts").iterdir() if p.is_dir() and p.name != "evals_v1" and re.search(r"_v1", p.name))
    res = list(scan_root.map(roots))
    groups = {}
    for r in res:
        for kind in ("staged", "meta"):
            for e in r[kind]:
                if e["key"] is None:
                    continue
                for f, v in e["prov"].items():
                    if isinstance(v, str) and GRP.search(v):
                        groups.setdefault((kind, f, v), set()).add(e["key"])
                if kind == "meta":
                    groups.setdefault(("meta", "extraction_id", e["extraction_id"]), set()).add(e["key"])
    shared = [{"kind": k, "field": f, "value": v, "keys": sorted(ks)} for (k, f, v), ks in sorted(groups.items()) if len(ks) > 1]
    if keys:
        shared = [g for g in shared if set(g["keys"]) & set(keys)]
    own = {k: [{"root": r["root"], "kind": kind, "file": e["file"], "prov": e["prov"]} for r in res for kind in ("staged", "meta") for e in r[kind] if e["key"] == k]
           for k in keys}
    return {"roots_scanned": len(res), "staged_records": sum(len(r["staged"]) for r in res), "meta_records": sum(len(r["meta"]) for r in res),
            "records_without_key": sum(1 for r in res for kind in ("staged", "meta") for e in r[kind] if e["key"] is None),
            "field_names": {r["root"]: r["fields"] for r in res}, "errors": {r["root"]: r["errors"][:20] for r in res if r["errors"]},
            "shared_groups": shared, "n_shared_groups": len(shared), "keys_provenance": own}


@app.local_entrypoint()
def main(out: str, keys: str = "", roots: str = ""):
    r = drive.remote([x for x in roots.split(",") if x], [x for x in keys.split(",") if x])
    with open(out, "x") as fh:
        json.dump(r, fh, indent=1, default=str)
    print(json.dumps({k: r[k] for k in ("roots_scanned", "staged_records", "meta_records", "records_without_key", "n_shared_groups")}))
