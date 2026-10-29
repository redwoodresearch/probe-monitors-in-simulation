"""Pins for the pre-freeze reproduction check (2e GO): exactly the inputs alltasks_CRD_2026-10-15_161306 read, as an INPUTS_FINAL pin file.
Source of truth: the record's own meta for the Sol mappings (meta.v2.sol_mappings incl. the --q2map override, meta.alltasks.nm360_mappings), and
the module constants of run_peraction311.py at the commit that wrote it (meta.git e0b803833; read from git, executed in a scratch folder) for the rest; no ledger NOTEs, no replacements, today's exclusions (m_r, class) as they were; 4f's file maps = packs_v2.json then
(53 packs, equal by pack to the maps @ fb48e6a36); fold-extension gates pending (Sol/c_asis withheld on nm360, as then). DT roots the then-pinned
records did not cover are listed in dt_roots_pending (computed by verify's rule).

  <modal env python> make_repro_pins.py --code-commit e0b803833 --scratch <dir>   # writes config/inputs_final_pins_repro_<commit>.json (exclusive)
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import inputs_final as IF  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--code-commit", default="e0b803833")
ap.add_argument("--scratch", required=True)
ap.add_argument("--file-maps-commit", default="fb48e6a36")
ap.add_argument("--record", default="results/alltasks_CRD_2026-10-15_161306.json")
a = ap.parse_args()
sc = Path(a.scratch)
sc.mkdir(parents=True, exist_ok=True)
for f in ("run_peraction311.py", "run_common455.py", "run_common710.py", "run_fprsweep_507.py"):
    (sc / f).write_bytes(IF.git_show(a.code_commit, f"experiments/2026-10-15_fpr_sweep_507_prep/{f}"))
sys.path.insert(0, str(sc))
spec = importlib.util.spec_from_file_location("old311", sc / "run_peraction311.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
P = json.loads((HERE / "config" / "inputs_final_pins.json").read_text())  # rms, epoch-2, sel97, gapfill, extensions: not read by Stage A
rm = json.loads((HERE / a.record).read_text())["meta"]
if rm["git"][:9] != a.code_commit[:9]:
    raise SystemExit(f"record git {rm['git'][:9]} is not --code-commit")
parse = lambda s_: (s_.split(" @ ")[1].split(" ")[0], s_.split(" @ ")[0], s_.split("(")[1].rstrip(")"))  # noqa: E731  "file @ commit (sha)"
P["mappings"] = {}
for st, src in (("S_new", rm["v2"]["sol_mappings"]), ("S_newmains", rm["alltasks"]["nm360_mappings"])):
    P["mappings"][st] = {}
    for bt, s_ in src.items():
        c, f, h = parse(s_)
        if IF.sha(IF.git_show(c, f"{IF.MAPDIR}/{f}.json")) != h:
            raise SystemExit(f"{bt}: mapping sha256 differs from the record's meta")
        P["mappings"][st][bt] = {"commit": c, "file": f}
if {k: (v[0], v[1]) for k, v in (R.NM_MAPS | R.NM_MAPS_AT).items()} != {k: (v["commit"], v["file"]) for k, v in P["mappings"]["S_newmains"].items()}:
    raise SystemExit("nm360 mappings: record meta and code constants disagree")
if {k: v for k, v in R.EXCL_V2.items()} != {k: {c0: tuple(x) for c0, x in d.items()} for k, d in rm["v2"]["exclusions"].items()}:
    raise SystemExit("S_new exclusions: record meta and code constants disagree")
# 25f honest-A roots and the S_old E2E config: today's pins must equal what the code at --code-commit read (fail otherwise)
want25 = {("p2", c, r, h) for c, r, h, _, _ in R.HA25F_ALL} | {("famexp", t[0], t[1], t[2]) for t in (R.HA25F_FAM, R.HA25F_FAM_Q2)} | \
         {("sol", t[0], t[1], t[2]) for t in (R.HA25F_SOL, R.HA25F_SOL_Q2)}
have25 = {(e["kind"], e["commit"], e["record"], e["sha256"]) for e in P["honest_a_25f"]}
if want25 != have25:
    raise SystemExit(f"25f honest-A pins differ from the {a.code_commit} constants: {sorted(want25 ^ have25)}")
src_ = IF.git_show(a.code_commit, "experiments/2026-10-15_fpr_sweep_507_prep/run_peraction311.py").decode()
if f'{P["s_old_e2e_config"]["commit"]}:{P["s_old_e2e_config"]["path"]}' not in src_:
    raise SystemExit(f"S_old E2E config pin is not the one the {a.code_commit} code read")
P["file_maps_commit"] = a.file_maps_commit
P["notes"], P["coverage_records"], P["empty_replacement_roots"] = [], [], []
P["ledger_extra"] = [{"set": st, "batch": bt, "cell_prefix": c0, "class": cls, "m_r": int(mr), "status": "excluded",
                      "reason": f"exclusion constant at {a.code_commit} (as the 16:13 record read it)", "ruling": f"run_peraction311.py @ {a.code_commit}"}
                     for st, ex in (("S_new", R.EXCL_V2), ("S_newmains", R.EXCL_NM)) for bt, d in ex.items() for c0, (mr, cls) in d.items()]
P["dt"] = {"S_new": [[c, n] for c, n, _ in R.DT_SIMH_V2], "S_newmains": [[c, n] for c, n, _ in R.DT_SIMH_NM], "S_old": [list(R.DT_SIMH_OLD[:2])]}
P["nm360_fold_extension"] = {"gates": {k: "held-out gate pending (c6023b805)" for k in P["nm360_fold_extension"]["gates"]}, "files": None}
P["cutoff_pt"] = P["freeze_pt"] = f"reproduction of the {a.code_commit} pins"
# DT gap: mapped roots without a then-pinned DT record (listed so verify accepts them; the runner labels them)
pend = {}
for st in ("S_new", "S_newmains"):
    mapped = {r["arm"] for e in P["mappings"][st].values() for r in json.loads(IF.git_show(e["commit"], f"{IF.MAPDIR}/{e['file']}.json"))["roots"]}
    have = {f for c, n in P["dt"][st] for f in json.loads(IF.git_show(c, f"experiments/2026-10-14_p2_dt_scaleup/results/{n}.json"))["folders"]}
    pend[st] = sorted(mapped - have)
P["dt_roots_pending"] = pend
dest = HERE / "config" / f"inputs_final_pins_repro_{a.code_commit}.json"
dest.open("x").write(json.dumps(P, indent=1, ensure_ascii=False) + "\n")
print(json.dumps({"pins": str(dest.relative_to(HERE)), "ledger_excluded": len(P["ledger_extra"]), "dt_roots_pending": pend,
                  "maps": {st: {bt: e["file"] for bt, e in v.items()} for st, v in P["mappings"].items()}}, indent=1))
