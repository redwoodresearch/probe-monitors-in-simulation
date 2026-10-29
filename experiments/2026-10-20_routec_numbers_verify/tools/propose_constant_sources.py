"""Propose a registering source (path:line@commit) for each design constant writing listed (09's item 6, 10-20), WITHOUT printing any
literal: the constants file is writing's (post text), so the output names rows by index and the literal's sha256 only.

For each row: an exact (case-sensitive) occurrence of the literal in the source files is a proposal ("exact"); failing that, a line that
holds every number of the literal (thousands separators optional) and at least one of its words of 4+ letters is a weaker proposal
("loose"); else none ("not a constant" until someone names a source; writing then needs a provenance row). 09 checks every proposal by
key and approves the final list by {sha256, count}.

  uv run --no-project --python 3.12 python -I tools/propose_constant_sources.py <constants.jsonl> <sources.tsv> <out.jsonl>
  sources.tsv: one "repo_dir<TAB>commit<TAB>path" per line; files are read with git show (never the working tree).
Prints counts only.
"""

import hashlib
import json
import re
import subprocess
import sys


def git_show(repo, commit, path):
    return subprocess.run(["git", "-C", repo, "show", f"{commit}:{path}"], check=True, capture_output=True).stdout.decode("utf-8", "replace")


def main(cpath, spath, opath):
    rows = [json.loads(x) for x in open(cpath, encoding="utf-8") if x.strip()]
    srcs = []
    for ln in open(spath, encoding="utf-8"):
        if ln.strip() and not ln.startswith("#"):
            repo, commit, path = ln.rstrip("\n").split("\t")
            srcs.append((repo, commit, path, git_show(repo, commit, path).split("\n")))
    out, kinds = [], {"exact": 0, "loose": 0, "none": 0}
    for r in rows:
        lit = r["text"]
        nums = [n.replace(",", "") for n in re.findall(r"\d[\d,]*(?:\.\d+)?", lit)]
        words = [w.lower() for w in re.findall(r"[A-Za-z]{4,}", lit)]
        exact, loose = [], []
        for _repo, commit, path, lines in srcs:
            for i, line in enumerate(lines, 1):
                ref = f"{path}:{i}@{commit[:9]}"
                if lit in line:
                    exact.append(ref)
                elif nums and all(re.search(r"(?<![\d.])" + re.escape(n) + r"(?![\d])", line.replace(",", "")) for n in nums) and \
                        (not words or any(w in line.lower() for w in words)):
                    loose.append(ref)
        kind = "exact" if exact else ("loose" if loose else "none")
        kinds[kind] += 1
        out.append({"row": r.get("row"), "kind": r.get("kind", "constant"), "v84_sha256": r.get("v84_sha256"), "text_sha256": hashlib.sha256(lit.encode()).hexdigest(),
                    "match": kind, "proposed_sources": (exact or loose)[:5], "n_candidates": len(exact or loose),
                    "writing_source_field_present": bool(r.get("source")) and "to be supplied" not in str(r.get("source"))})
    with open(opath, "x", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(out), "by_match": kinds, "sources_read": len(srcs)}))


if __name__ == "__main__":
    main(*sys.argv[1:4])
