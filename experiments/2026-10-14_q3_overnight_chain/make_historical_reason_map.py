"""Corrected record for the historical stager's reason strings (09 @ 527337ee3 section 3): every IL epoch the historical stager
excluded, mapped by EXACT epoch identity (path#e0, .eval sha256, world, cell) to the amended stager's section 1 / 6 string, with the
original string kept beside it. Not a re-staging: the strings come from the replay reports, which re-ran the historical and the
amended stager on the same pinned inputs and found identical decisions and donors for every epoch.
Run: python make_historical_reason_map.py --reports <replay report> [<replay report> ...] --out <json>"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows, src, errors = {}, [], []
    for r in a.reports:
        raw = Path(r).read_bytes()
        d = json.loads(raw)
        if not d["pass"]:
            errors.append(f"{r}: the replay did not pass")
        src.append({"path": r, "sha256": hashlib.sha256(raw).hexdigest(), "commit_of_code": d["pins"]["commit"], "targets": d["targets"], "counts": d["counts"]})
        for c in d["changed_reasons"]:
            target, arm, cell, launch, run, sha, epoch = c["key"]
            k = f"{run}#{epoch}"
            if k in rows:
                errors.append(f"{k}: in two reports")
            rows[k] = {"epoch_key": k, "eval_sha256": sha, "world": "run3" if arm == "attack" else "sim2", "cell": cell, "launch": launch, "staged_set_target": target,
                       "original_string": c["hist_reason"], "corrected_string": c["amended_reason"],
                       "class_pointer": (c.get("ruling_09") or {}).get("class") or "see exclusion_rulings_09_* (tables branch)"}
    doc = {"rule": "09 @ 527337ee3 section 3: the historical-stager stagings stand; their reason strings map to the section 1 / 6 strings",
           "sources": src, "rows": sorted(rows.values(), key=lambda x: x["epoch_key"]), "count": len(rows), "errors": errors, "pass": not errors}
    Path(a.out).open("x").write(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"count": len(rows), "by_corrected": {s: sum(x["corrected_string"] == s for x in rows.values()) for s in sorted({x["corrected_string"] for x in rows.values()})},
                      "errors": errors}, indent=1))


if __name__ == "__main__":
    main()
