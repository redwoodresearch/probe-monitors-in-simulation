"""The constants-with-sources file for 09's approval (item 6, 10-20): writing's constants rows joined, BY ROW INDEX, to the class and the
registering source(s) decided per row (an overrides JSON: 09's rulings plus the owner's re-sourced lines). For every cited source line
(path:L or path:L1-L2 @ commit, read with git show) it records two booleans: the line holds the literal exactly, and it holds every
number of the literal (thousands separators ignored). No literal is printed or written (the literal's sha256 binds the row).

  uv run --no-project --python 3.12 python -I tools/build_constants_sources.py <constants.jsonl> <overrides.json> <proposal.jsonl> <out.jsonl>
  <proposal.jsonl>: propose_constant_sources.py's output on an earlier constants file; a row whose overrides say "proposal_exact" takes
  that proposal's exact sources, only when the proposal row's text_sha256 equals this row's literal sha (else STOP).
Prints counts only.
"""

import hashlib
import json
import re
import subprocess
import sys

REPO = "/Users/aswinsurya/probe-distribution-shift-from-sims"


def lines_at(ref):
    m = re.fullmatch(r"(?P<path>[^:@ ]+):(?P<a>\d+)(?:-(?P<b>\d+))?@(?P<c>[0-9a-f]{7,40})", ref.split(" ")[0])
    if not m:
        raise SystemExit("STOP: a source ref is not path:L[-L2]@commit")
    txt = subprocess.run(["git", "-C", REPO, "show", f"{m['c']}:{m['path']}"], check=True, capture_output=True).stdout.decode("utf-8", "replace")
    ls = txt.split("\n")
    a, b = int(m["a"]), int(m["b"] or m["a"])
    return "\n".join(ls[a - 1:b])


def main(cpath, opath, ppath, outpath):
    rows = {r["row"]: r for r in (json.loads(x) for x in open(cpath, encoding="utf-8") if x.strip())}
    ov = json.load(open(opath, encoding="utf-8"))["rows"]
    prop = {r["row"]: r for r in (json.loads(x) for x in open(ppath, encoding="utf-8") if x.strip())}
    if set(map(int, ov)) != set(rows):
        raise SystemExit("STOP: the overrides do not cover exactly the constants rows")
    out, by_cls, checks = [], {}, {"sources": 0, "literal": 0, "value": 0}
    for i in sorted(rows):
        r, o = rows[i], ov[str(i)]
        lit = r["text"]
        tsha = hashlib.sha256(lit.encode()).hexdigest()
        srcs = o.get("sources", [])
        if srcs == "proposal_exact":
            p = prop.get(i)
            if not p or p["text_sha256"] != tsha or p["match"] != "exact":
                raise SystemExit("STOP: a proposal_exact row's proposal is not the same literal, or not exact")
            srcs = p["proposed_sources"]
        nums = [n.replace(",", "") for n in re.findall(r"\d[\d,]*(?:\.\d+)?", lit)]
        res = []
        for ref in srcs:
            body = lines_at(ref)
            flat = body.replace(",", "")
            has_lit = lit in body
            has_val = bool(nums) and all(re.search(r"(?<![\d.])" + re.escape(n) + r"(?![\d])", flat) for n in nums)
            res.append({"ref": ref, "contains_literal": has_lit, "contains_value": has_val})
            checks["sources"] += 1
            checks["literal"] += has_lit
            checks["value"] += has_val
        cls = o["class"]
        by_cls[cls] = by_cls.get(cls, 0) + 1
        out.append({"row": i, "writing_kind": r.get("kind"), "class": cls, "status": o["status"], "text_sha256": tsha, "sources": res,
                    "note": o.get("note", "")})
    with open(outpath, "x", encoding="utf-8") as f:
        for x in out:
            f.write(json.dumps(x, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(out), "by_class": by_cls, "source_lines": checks}))


if __name__ == "__main__":
    main(*sys.argv[1:5])
