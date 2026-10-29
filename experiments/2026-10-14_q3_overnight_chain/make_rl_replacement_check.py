"""For each RateLimitError row of d9's corrected IL snapshot-1 exclusion record: is a replacement epoch of the same cell and world STAGED (in its
manifest and not staging-excluded) and ROUTED (an attested routing file in the chain's routing dir) in one of the rerun roots stage B reads (its spec's
il.arms + il.arms_extended folders whose staged set is a rerun family: q-token ending r / r<k>)? Keys only; fails loudly on an unreadable record.

  python3 make_rl_replacement_check.py <exclusions "commit:path"> <stage B spec "commit:path"> <routing dir> <out.json>
"""
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RERUN = re.compile(r"^(il311_q[1-4]r[0-9]*|ilold_oldr[0-9]*|ilnm_s[0-9]+r[0-9]*)_(run3|sim2)_v1(_try[0-9]+)?$")


def show(ref: str) -> bytes:
    return subprocess.run(["git", "show", ref], capture_output=True, check=True, cwd=REPO).stdout


def lit(x):
    return x if isinstance(x, (dict, list)) else ast.literal_eval(x)


def main():
    ex_ref, spec_ref, rdir, out = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4]
    rows = [r for r in json.loads(show(ex_ref))["rows"] if str(r.get("proposed_class", "")).startswith("RateLimitError")]
    il = json.loads(show(spec_ref))["il"]
    folders = sorted(f for f in set(il["arms"]) | set(il["arms_extended"]) if RERUN.match(f.split("__")[0]))
    by_set = {}
    for p in rdir.glob("*__*.json"):
        r = json.loads(p.read_text())
        if r.get("staged_set"):
            by_set.setdefault(r["staged_set"], []).append((p, r))
    roots = {}  # (world, cell) -> [{folder, status, reason}]
    audit = []
    for f in folders:
        sset = f.split("__")[0]
        world = RERUN.match(sset).group(2)
        hits = [(p, r) for p, r in by_set.get(sset, []) if not r.get("superseded")]
        assert len(hits) == 1, f"{f}: {len(hits)} routing files"
        p, r = hits[0]
        bid = r["batch"]
        att = rdir / f"{bid}_attest.json"
        attested = att.exists() and json.loads(att.read_text())["failing"] == {}
        man_path, man_sha = [x.strip() for x in r["manifest"].split("@")]
        man = json.loads(show(f"{man_sha}:{man_path}"))
        excl = {k.split("#", 1)[0]: v for k, v in lit(r["staging_exclusions"]).items()}
        n = 0
        for row in man["files"]:
            if row.get("world") != world:
                continue
            n += 1
            st = ("excluded" if row["path"] in excl else "staged")
            roots.setdefault((world, row["task"]), []).append({"folder": f, "status": st, **({"reason": excl[row["path"]]} if st == "excluded" else {}),
                                                            "routed_attested": attested, "routing_file": p.name})
        audit.append({"folder": f, "routing_file": p.name, "world": world, "rows": n, "attested": attested, "manifest": r["manifest"]})
    res = []
    for r in rows:
        world, cell = r["key"].split("|", 1)
        cand = roots.get((world, cell), [])
        good = [c for c in cand if c["status"] == "staged" and c["routed_attested"]]
        why = None if good else ("no rerun epoch of this cell and world in any rerun root" if not cand
                                 else "rerun epoch(s) also excluded: " + "; ".join(f"{c['folder']}: {c.get('reason')}" for c in cand if c["status"] == "excluded")
                                 if all(c["status"] == "excluded" for c in cand) else "rerun epoch staged but its root is not attested")
        res.append({"key": r["key"], "batch": r.get("batch"), "launch": r.get("launch"), "covered": bool(good), "by": [c["folder"] for c in good],
                    **({"why_not": why} if why else {}), "candidates": cand})
    doc = {"what": "RateLimitError rows of the corrected IL snap1 exclusion record: replacement staged + routed in a stage-B rerun root (keys only)",
           "exclusions": ex_ref, "stage_b_spec": spec_ref, "rerun_roots": audit, "rows": len(res), "covered": sum(x["covered"] for x in res),
           "not_covered": [x["key"] for x in res if not x["covered"]], "results": res}
    Path(out).open("x").write(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"rows": len(res), "covered": doc["covered"], "not_covered": len(doc["not_covered"]), "rerun_roots": len(audit)}))


if __name__ == "__main__":
    main()
