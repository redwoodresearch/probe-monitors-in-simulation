"""Keys-only comparison of two Stage 1 results files' rules (c8887f6e7's vs route C's): rule keys (row, rule text), violations counts, and a
sha256 digest of each entry; prints counts by rule family only (no values). Usage: compare_s1_rules.py <repo> <old commit:path> <new path>"""
import hashlib
import json
import re
import subprocess
import sys

repo, old_spec, new_path = sys.argv[1:4]
c, p = old_spec.split(":", 1)
old = json.loads(subprocess.run(["git", "-C", repo, "show", f"{c}:{p}"], capture_output=True, check=True).stdout)
new = json.load(open(new_path))
UNTOUCHED = ("cfg0", "not covered")  # pre-stated: cfg0 regression (12) and the stated not-covered rules (4)


def fam(rule):
    r = re.sub(r"\(.*", "", rule).strip().lower()
    for k in ("manifest module pins", "canonical main", "pooled identity gates", "pooled pools file", "cfg0", "planted", "not covered"):
        if k in r:
            return k
    m = re.match(r"fold-model rules on (\S+)", r)
    return f"fold-model rules on {m.group(1)}" if m else r[:60]


def table(res):
    out = {}
    for e in res["rules"]:
        k = (e["row"], e["rule"])
        assert k not in out, ("duplicate rule key", k)
        v = e.get("violations")
        out[k] = (len(v) if isinstance(v, (list, dict)) else (0 if not v else 1),
                  hashlib.sha256(json.dumps(e, sort_keys=True).encode()).hexdigest())
    return out


to, tn = table(old), table(new)
print("rules old", len(to), "new", len(tn), "same keys", sorted(to) == sorted(tn))
print("keys only old", len(set(to) - set(tn)), "keys only new", len(set(tn) - set(to)))
print("violating rules new", sum(1 for v in tn.values() if v[0]), "old", sum(1 for v in to.values() if v[0]))
print("n_compared new", new.get("n_compared"), "(> 0:", isinstance(new.get("n_compared"), int) and new["n_compared"] > 0, ")")
fams = {}
for k in sorted(set(to) & set(tn)):
    f = fam(k[1])
    d = fams.setdefault(f, {"n": 0, "equal_digest": 0, "pass_new": 0})
    d["n"] += 1
    d["equal_digest"] += to[k][1] == tn[k][1]
    d["pass_new"] += tn[k][0] == 0
for f, d in sorted(fams.items()):
    print(f"{f}: n {d['n']}, equal entry digest {d['equal_digest']}, PASS (0 violations) {d['pass_new']}", "[UNTOUCHED]" if f in UNTOUCHED else "")
viol_ids = sorted(f"{k[0]} | {fam(k[1])}" for k, v in tn.items() if v[0])
print("violating rule ids (row | family):", viol_ids[:20])
