"""One routing + by-key coverage record for attested S_paper493 / il-s4rgap-t2 sets (d9 10-16 ~00:30): per routing file, its staged set, extraction
root, attestation (failing {}), the sample_error manifest revision (pointer in routing/sample_error/), staged rows by key, exclusions by key with the
staging reason and sample_error, and its cell list; and for the sim groups g1 / g2 / g3 / g6 / g7 the disjointness assert 09 / 4f require (each cell
in exactly ONE admitted set of its group: no overlap between snapshots and the remainder, and for the X/Y tail across x AND y; 09 via d9 10-16 ~01:00).
With --tail <split commit:path@sha256> <L3 index commit:path@sha256>: the X/Y universe by key (the six families' pinned task files' groups maps):
g6 / g7 = the split list's moved keys exactly, g3 = 37 cells of refindex_L3, every admitted cell inside its group's universe, and what is not yet
admitted. Keys only; fails loudly on an overlap, a failing attestation, a missing revision or a universe mismatch.

  python make_sp493_routing_record.py <routing dir> <out.json> [--tail <split> <L3>] <batch>__<target> ...
"""
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GROUP = re.compile(r"sp493-(g1|g2|g3|g6|g7)([xy])?(?:_s[0-9]+)?")  # a sim-attack family's batch id (g4 / g5 are sim-honest: not grouped)


def pinned(ref: str) -> dict:
    """<commit>:<path>@<sha256>: the committed bytes, sha-asserted."""
    loc, sha = ref.rsplit("@", 1)
    raw = subprocess.run(["git", "show", loc], capture_output=True, check=True, cwd=HERE.parents[1]).stdout
    assert hashlib.sha256(raw).hexdigest() == sha, f"{loc}: sha256 differs from the pin"
    return json.loads(raw)


def tail_universe(split_ref: str, l3_ref: str) -> dict:
    split, l3 = pinned(split_ref), pinned(l3_ref)
    keys = [r["key"] for r in split["rows"]]
    assert len(keys) == len(set(keys)), "the split list repeats a key"
    uni = {}
    for F in json.loads((HERE / "config" / "chain_batches.json").read_text())["followups"]:
        g = F.get("group_cells")
        if g:
            m = pinned(f"{g['commit']}:{g['path']}@{g['sha256']}")["groups"]
            uni.setdefault(g["group"], set()).update(c for c, x in m.items() if x == g["group"])
    for g in ("g6", "g7"):
        assert uni[g] == {r["key"] for r in split["rows"] if r["group"] == g}, f"{g}: the task files' cells differ from the split list's moved keys"
    assert uni["g3"] <= set(l3["references"]) and len(uni["g3"]) == 37, "g3: not 37 cells of refindex_L3"
    return {"universe": uni, "l3_not_in_g3": sorted(set(l3["references"]) - uni["g3"]), "split": split_ref, "l3": l3_ref}


def lit(x):
    return x if isinstance(x, (dict, list)) else ast.literal_eval(x)


def main():
    rdir, out, keys = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    tail = None
    if keys[:1] == ["--tail"]:
        tail, keys = tail_universe(keys[1], keys[2]), keys[3:]
    sets, groups = [], {}
    for k in keys:
        r = json.loads((rdir / f"{k}.json").read_text())
        att = json.loads((rdir / f"{r['batch']}_attest.json").read_text())
        assert att["failing"] == {}, (k, att["failing"])
        ptr = json.loads((rdir / "sample_error" / f"{k}.json").read_text())
        rev = json.loads((HERE / "config" / ptr["manifest_revision"].split(" @ ")[0].split("/config/", 1)[1]).read_text())
        assert all("sample_error" in f for f in rev["files"]), k
        excl = {kk.split("#", 1)[0]: v for kk, v in lit(r["staging_exclusions"]).items()}
        arms = [a["meta_arm"] for a in lit(r["arms"])]
        own = [f for f in rev["files"] if f["arm_dir"] in arms and f.get("attempt_kind") != "reference"]
        staged = sorted(f"{f['arm_dir']}|{f['task']}" for f in own if f["path"] not in excl)
        cells = sorted({f["task"] for f in own if f["path"] not in excl})
        sets.append({"routing_file": f"{k}.json", "batch": r["batch"], "staged_set": r["staged_set"], "extraction_root": r["extraction_root"],
                     "attestation": f"{att['path']} @ {att['commit'][:12]}", "manifest_revision": ptr["manifest_revision"], "revision_sha256": ptr["sha256"],
                     "epochs": {a["meta_arm"]: a["epochs"] for a in lit(r["arms"])}, "staged_keys": len(staged), "staged": staged, "cells": cells,
                     "excluded": [{"key": f"{f['arm_dir']}|{f['task']}", "staging_reason": excl[f["path"]], "sample_error": f["sample_error"]} for f in own if f["path"] in excl],
                     "donor_rows_with_sample_error": ptr["donor_rows_with_sample_error"]})
        gm = GROUP.fullmatch(r["batch"])
        g = gm.group(1) if gm else None
        if g:
            for c in cells:
                groups.setdefault(g, {}).setdefault(c, []).append(r["staged_set"])
    overlap = {g: {c: v for c, v in m.items() if len(v) > 1} for g, m in groups.items()}
    assert not any(overlap.values()), f"a sim-group cell is in more than one admitted set: {overlap}"
    doc = {"what": "S_paper493 / il-s4rgap-t2 routing + by-key coverage (SEALED, PRELIMINARY); every manifest revision carries sample_error",
           "sets": sets, "group_disjoint": {g: {"cells": len(m), "sets": len({s for v in m.values() for s in v})} for g, m in groups.items()}}
    if tail:
        out_of = {g: sorted(set(m) - tail["universe"].get(g, set())) for g, m in groups.items() if g in ("g3", "g6", "g7")}
        assert not any(out_of.values()), f"admitted cells outside their group's X/Y universe: {out_of}"
        doc["tail_universe"] = {"split_list": tail["split"], "refindex_L3": tail["l3"], "l3_cells_not_in_g3": tail["l3_not_in_g3"],
                                "groups": {g: {"universe": len(u), "admitted_here": len(set(groups.get(g, {})) & u),
                                               "not_admitted_here": sorted(u - set(groups.get(g, {})))} for g, u in sorted(tail["universe"].items())}}
    Path(out).open("x").write(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"sets": len(sets), "group_disjoint": doc["group_disjoint"], "staged": {s["staged_set"]: s["staged_keys"] for s in sets},
                      **({"tail": {g: (v["universe"], v["admitted_here"]) for g, v in doc["tail_universe"]["groups"].items()}} if tail else {})}))


if __name__ == "__main__":
    main()
