"""Corrected supersession records (2e, under 0d's RULE Amendment 27 @ b9838215b condition (c), via d9 10-15 ~08:20): an authoritative
replacement row whose replacement run staging EXCLUDED must say `"staged": false`. For every IL supersession record in
results/supersession/ (rerun and completion families), find each row's replacement run in the replacement family's committed
manifests and its staging records' exclusion lists; if any row is staged out, write <record>_v2.json (open 'x') that adds
`"staged": false` to exactly those rows and changes nothing else (asserted), plus `supersedes`; the original stays. Prints every
record checked and the rows fixed. Keys and hashes only.

  python3 make_supersession_staged_v2.py <chain state dir>
"""
import copy
import json
import sys
from pathlib import Path

import chain as C

ARM = {"run3": "attack", "sim2": "sim_attack@september"}


def main():
    batches = json.loads((Path(sys.argv[1]) / "state.json").read_text())["batches"]
    by_prefix = {f"{b['name_prefix']}_{b['q']}": b for b in batches.values() if b.get("kind") == "il"}
    out = []
    for p in sorted((C.HERE / "results" / "supersession").glob("supersession_*_2026-10-15.json")):
        doc = json.loads(p.read_text())
        fam = doc.get("rerun_family") or doc.get("completion_family")
        if fam is None or fam not in by_prefix:
            out.append({"record": p.name, "checked": False, "why": "not an IL rerun/completion record"})
            continue
        b = by_prefix[fam]
        if "stage" not in b["done"] or any(not (C.HERE / "results" / b["stage"][t]["record"]).exists() for t in b.get("targets", [])):
            out.append({"record": p.name, "family": fam, "batch": b["id"], "checked": False, "why": "not staged yet (re-checked before its routing)"})
            continue
        excluded, path_of = {}, {}
        for t, tv in C.targets_for(b).items():
            if t not in b.get("targets", []):
                continue
            rec = json.loads((C.HERE / "results" / b["stage"][t]["record"]).read_text())
            k = next(x for x in rec if x not in ("git", "stamp"))
            excluded |= {key.split("#", 1)[0]: why for key, why in rec[k]["c_gate"]["excluded"].items()}
            for f in json.loads((C.HERE / "config" / tv["manifest"]).read_text())["files"]:
                if f["arm_dir"] in ARM.values():
                    path_of[(f["arm_dir"], f["task"])] = f["path"]
        fixed = []
        new = copy.deepcopy(doc)
        for r in new["records"]:
            path = path_of.get((ARM[r["arm"]], r["cell"]))
            if r.get("authoritative") == "replacement" and path in excluded and r.get("staged") is not False:
                r["staged"] = False
                fixed.append({"key": r["key"], "run": path, "staging_reason": excluded[path]})
        out.append({"record": p.name, "family": fam, "batch": b["id"], "rows": len(doc["records"]), "staged_out_rows": fixed})
        dst = p.with_name(p.stem + "_v2.json")
        if not fixed:
            assert not dst.exists(), f"{dst.name} exists but this run finds no staged-out row"  # a stale correction never passes silently
            continue
        diff = [(a["key"], set(a) ^ set(c)) for a, c in zip(doc["records"], new["records"]) if a != c]
        assert all(d == {"staged"} for _, d in diff) and len(diff) == len(fixed), diff  # nothing but the new field
        new |= {"supersedes": p.name, "correction": "rows whose replacement run staging excluded say staged: false (2e; RULE Amendment 27 @ b9838215b (c))",
                "staged_out_rows": fixed}
        text = json.dumps(new, indent=1, sort_keys=True) + "\n"
        if dst.exists():  # corrected earlier: the same rows must be fixed, or it stops (a further correction is a new version)
            old = json.loads(dst.read_text())
            assert old["records"] == new["records"] and old["staged_out_rows"] == fixed, f"{dst.name} differs from this run's correction"
            out[-1]["already_corrected"] = dst.name
            continue
        with dst.open("x") as f:
            f.write(text)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
