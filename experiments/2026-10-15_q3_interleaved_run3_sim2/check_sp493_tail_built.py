"""After the on-box prep (prep_sp493_tail.sh): each built X/Y entry must equal, by key, its L1/L2 built entry (g6/g7: the same cell
as launched in L1/L2, approach and template hashes included) or its B-fork as-run entry less attack_pair_launch, _box and (t887/t962)
real_arms (g3); same order and groups as the input; nothing dropped; the top-level template_sha256 = L1/L2's. Keys and hashes only.
  python check_sp493_tail_built.py          # prints one line per launch; exit 1 on any failure
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
C = HERE / "config/sp493sim"
B = "experiments/2026-10-15_q3_sp493"


def git_json(commit, path):
    return json.loads(subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, check=True, cwd=HERE).stdout)


built_L = {L: json.loads((C / f"tasks_sp493_sim_{L}.json").read_text()) for L in ("L1", "L2")}
lb = {c["key"]: c for L in ("L1", "L2") for c in built_L[L]["cells"]}
asrun = git_json("24e7dc722", f"{B}/results/asrun_honest_entries_2026-10-15_192615.json")["entries"]
asrun |= git_json("af0c27a06", f"{B}/results/asrun_newmain_attackonly_2026-10-15_194135.json")["entries"]
tpl = {json.dumps(built_L[L].get("template_sha256"), sort_keys=True) for L in ("L1", "L2")}  # a dict (per mode)
ok = True
for L in "XY":
    inp = json.loads((C / f"tasks_sp493_tail_{L}_in.json").read_text())
    out = json.loads((C / f"tasks_sp493_tail_{L}.json").read_text())
    bad = []
    if [c["key"] for c in out["cells"]] != [c["key"] for c in inp["cells"]] or out.get("dropped_at_preflight"):
        bad.append("order or drops")
    if out.get("groups") != inp["groups"]:
        bad.append("groups")
    if len(tpl) != 1 or json.dumps(out.get("template_sha256"), sort_keys=True) not in tpl:
        bad.append("template_sha256")
    for c in out["cells"]:
        k, g = c["key"], inp["groups"][c["key"]]
        want = lb[k] if g in ("g6", "g7") else {f: v for f, v in asrun[k].items() if f not in ("attack_pair_launch", "_box", "real_arms")}
        if c != want:
            bad.append(f"{k} ({g}): fields {sorted(f for f in set(c) | set(want) if c.get(f) != want.get(f))}")
    print(f"{L}: {len(out['cells'])} built cells, {sum(g == 'g3' for g in inp['groups'].values())} g3; "
          + ("PASS: every entry equals its L1/L2 built entry (g6/g7) or its as-run entry (g3)" if not bad else f"FAIL {bad[:5]}"))
    ok &= not bad
sys.exit(0 if ok else 1)
