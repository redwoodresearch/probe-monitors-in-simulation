"""Amendment 45: the S_paper493 cell list DT scores against (keys, envs, mains, main origins only). From the B fork's facts v2 @ d800edc6a
(09 Append 39 amendment 1), checked by key against plus804 cells[311:] @ 8a3af7b1e; t987 dropped (09's ruling, recheck record @ a380be4eb:
"t987 leaves the S_paper493 attack-estimation cohort for every row"). Usage: python3 -I write_sp493_cells.py <repo> <out json>"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO, OUT = sys.argv[1], Path(sys.argv[2])
SRC = {"facts": ("d800edc6a", "experiments/2026-10-15_q3_sp493/results/sp493_facts_2026-10-15_165730.json", "07caa527db89ec519314486d8c529c37430ca41d773a8bce26c041eb141b0a57"),
       "plus804": ("8a3af7b1e", "experiments/2026-10-14_q2_scaleup_cohort/results/scaleup_tasks_plus804_2026-10-14_125135.json", "5fd2de4497141c4fab051f073ec70476d90abbccd93305898984c28fe012ae51"),
       "t987_ruling": ("a380be4eb", "experiments/2026-10-15_q3_sp493/results/unusable_honest_recheck_record_2026-10-15_220125.json", None)}
raw = {k: subprocess.run(["git", "-C", REPO, "show", f"{c}:{p}"], capture_output=True, check=True).stdout for k, (c, p, _) in SRC.items()}
assert all(h is None or hashlib.sha256(raw[k]).hexdigest() == h for k, (_, _, h) in SRC.items()), "a source differs from its pin"
facts, p804 = json.loads(raw["facts"])["rows"], json.loads(raw["plus804"])["cells"][311:]
assert [(r["key"], r["env"], r["main"]) for r in facts] == [(c["key"], c["env"], c["main"]) for c in p804] and len(facts) == 493, "facts rows != plus804 cells[311:]"
assert any("t987 leaves the S_paper493" in n for n in json.loads(raw["t987_ruling"])["notes"]), "t987 ruling not in its record"
drop = {"t987_model_registry": "09's ruling (recheck record @ a380be4eb notes): t987 leaves the S_paper493 attack-estimation cohort for every row"}
cells = [{"key": r["key"], "env": r["env"], "main": r["main"], "main_origin": r["main_origin"]} for r in facts if r["key"] not in drop]
ident = hashlib.sha256(json.dumps(sorted([c["key"], c["env"], c["main"]] for c in cells), separators=(",", ":")).encode()).hexdigest()
n = lambda o: (sum(c["main_origin"] == o for c in cells), len({(c["env"], c["main"]) for c in cells if c["main_origin"] == o}))  # noqa: E731
out = {"what": "S_paper493 cells DT scores against (Amendment 45): facts v2 rows minus 09's dropped keys; keys, envs, mains and main origins only",
       "sources": {k: {"commit": c, "path": p, "sha256": h} for k, (c, p, h) in SRC.items()}, "dropped": drop, "identity_sha256_sorted_triples": ident,
       "counts": {"cells": len(cells), "mains": len({(c["env"], c["main"]) for c in cells}), "by_origin_cells_mains": {o: n(o) for o in ("t196", "p311", "new")}},
       "cells": cells}
OUT.open("x").write(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ("identity_sha256_sorted_triples", "counts")}))
