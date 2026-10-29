"""Stage B posthoc spec (Z fork for 09, 10-15 19:1x): 09's latest posthoc spec (base, sha-checked) with its posthoc_hits replaced by the
stage B RMS exports, S_newmains' 25h §1 extension (deviation) export attached, pooling on (09 @ 2bb4f5146 item 3, ACK on f1435604b), and
09's numerics note verbatim. The hit entries are built exactly as make_il_spec.py builds them (gate fields only; thresholds T dropped).

  python3 -I experiments/2026-10-15_il_analysis_09/make_stageB_posthoc_spec.py --base specs/il_posthoc_spec_r7.json --base-sha256 <sha> \\
      --hits <manifest path>@<commit> (x3) --extension <deviation manifest path>@<commit> --out specs/<name>.json
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ACK = "f1435604b"  # 09's ACK of the pooled-RMS patch (via 2e, 19:0x); the posthoc code must be byte-identical to this commit's
R7_SHA = "118118abb50d2139e79656eb7858f359ec1828f6bc2adee9793576f5f083720f"  # 09: specs/il_posthoc_spec_r7.json (run 6's spec @ 40ab6db4e)
V2 = ("668a3eb91", "experiments/2026-10-15_il_analysis_09/results/il_five_arm_cells_2026-10-15_v2.json", "5ca4574c8d627125f0a02f87e3a1e8f91d94dd307793c9b6f72e5fb629c6db14")
V2_KEY = {"S_old": "S_old_five_arm", "S_new": "S_new_on_M", "S_new_on_M": "S_new_on_M", "S_newmains": "S_newmains_five_arm"}
SCOPE_A = "S_newmains here = S1–S3 + S5 (S4d / S4r not included)"
ALLOWED = (("il", "posthoc_hits"), ("il", "posthoc_pooled_rms"), ("stage_b_numerics_note",), ("stage_b_meta",))  # 09: any other difference is a build error
NOTE = ("score-level RMS sums show ~1e-14 relative run-to-run variation across containers (non-dyadic floating-point accumulation); "
        "catch-level and per-action rows are bitwise stable")


def show(commit, path):
    return subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, check=True, cwd=REPO).stdout


def gate_pools(pools):  # as make_il_spec.py: every field except the thresholds T; a withheld probe keeps its status string
    return {pl: {sd: ({k_: ({f_: x_ for f_, x_ in v_.items() if f_ != "T"} if isinstance(v_, dict) else v_) for k_, v_ in sv_.items()}
                      if isinstance(sv_, dict) else sv_) for sd, sv_ in pv_.items()} for pl, pv_ in pools.items()}


ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--base-sha256", required=True)
ap.add_argument("--hits", action="append", required=True)
ap.add_argument("--extension", required=True)
ap.add_argument("--inputs-final", required=True, help="<path>@<commit> of the FROZEN INPUTS_FINAL the exports were built from")
ap.add_argument("--out", required=True)
a = ap.parse_args()
bb = (HERE / a.base).read_bytes()
if hashlib.sha256(bb).hexdigest() != a.base_sha256 or a.base_sha256 != R7_SHA:
    raise SystemExit("base spec sha256 differs from --base-sha256 or from 09's r7")
code_ = (HERE / "run_il_posthoc.py").read_bytes()
if code_ != show(ACK, "experiments/2026-10-15_il_analysis_09/run_il_posthoc.py"):
    raise SystemExit(f"run_il_posthoc.py differs from the ACKed commit {ACK}")
v2b = show(V2[0], V2[1])
if hashlib.sha256(v2b).hexdigest() != V2[2]:
    raise SystemExit("five-arm v2 sha256 differs")
v2 = json.loads(v2b)
ifp_, ifc_ = a.inputs_final.rsplit("@", 1)
ifb_ = show(ifc_, ifp_)
if_sha = hashlib.sha256(ifb_).hexdigest()
if not json.loads(ifb_)["status"].startswith("FROZEN"):
    raise SystemExit("--inputs-final is not FROZEN")
base_ = json.loads(bb)
spec = json.loads(bb)
il = spec["il"]
hits = []
for h_ in a.hits:
    mp_, mc_ = h_.rsplit("@", 1)
    mb_ = show(mc_, mp_)
    m_ = json.loads(mb_)
    if m_.get("numerics_note") != NOTE and (m_.get("meta") or {}).get("numerics_note") != NOTE:
        raise SystemExit(f"{mp_}: the export lacks 09's numerics note")
    if m_["cells"] != v2[V2_KEY[m_["set"]]]["cells"] or m_["cells"] != spec["il"]["posthoc_cells"][V2_KEY[m_["set"]].replace("_five_arm", "")]:
        raise SystemExit(f"{mp_}: cells differ from the five-arm v2 list / r7's posthoc_cells")
    if m_["set"] in ("S_new", "S_newmains"):  # 09: tonight's FINAL sim-honest roots, through a --tag=final config built from the FROZEN INPUTS_FINAL
        cb_ = show(m_["meta"]["git"], f"experiments/2026-10-15_il_z_e2e/{m_['meta']['config']}")
        cf_ = json.loads(cb_)
        if (hashlib.sha256(cb_).hexdigest() != m_["meta"]["config_sha256"] or not m_["meta"]["config"].endswith("_final.json")
                or not cf_["sources"]["inputs_final"]["status"].startswith("FROZEN") or cf_["sources"]["inputs_final"]["sha256"] != if_sha
                or "H_sim_roots" not in cf_.get("pool_equality", {})):
            raise SystemExit(f"{mp_}: not built from a --tag=final config on this FROZEN INPUTS_FINAL ({if_sha[:12]}…) with pool equality")
        if m_["set"] == "S_newmains" and (m_["meta"].get("scope_label") != SCOPE_A or cf_.get("scope_label") != SCOPE_A):
            raise SystemExit(f"{mp_}: the S_newmains export lacks 09's scope label")
    e_ = {"set": m_["set"], "manifest": {"commit": mc_, "path": mp_, "sha256": hashlib.sha256(mb_).hexdigest()}, "cells": m_["cells"], "arms": m_["arms"],
          "probes": m_["probes"], "points": m_["points"], "lines": m_["lines"], "pools": gate_pools(m_["pools"]),
          "npz": {"path": f"/sol/xfer/il_z/rms_hits/{m_['npz']}", "sha256": m_["npz_sha256"]}}
    lab_ = {"set_label_verbatim": m_.get("set_label"), "H_sim_label_verbatim": m_.get("H_sim_label"),  # every label verbatim from the export
            "scope_label": (m_.get("meta") or {}).get("scope_label"), "numerics_note": NOTE}
    e_["labels"] = {k_: v_ for k_, v_ in lab_.items() if v_}
    hits.append(e_)
if sorted(h["set"] for h in hits) != ["S_new", "S_newmains", "S_old"]:  # Codex: r7 carries only S_new; stage B needs all three
    raise SystemExit(f"the stage B exports' sets {sorted(h['set'] for h in hits)} are not S_old, S_new and S_newmains")
dp_, dc_ = a.extension.rsplit("@", 1)
db_ = show(dc_, dp_)
d_ = json.loads(db_)
nm = [h for h in hits if h["set"] == "S_newmains"][0]
if d_["set"] != "S_newmains" or d_["cells"] != nm["cells"]:
    raise SystemExit("the extension export's set or cells differ from S_newmains'")
nm_meta = json.loads(show(nm["manifest"]["commit"], nm["manifest"]["path"]))["meta"]
if (d_["meta"].get("config_sha256") != nm_meta["config_sha256"] or d_["meta"].get("config") != nm_meta["config"]
        or d_["meta"].get("scope_label") != SCOPE_A or d_["meta"].get("numerics_note") != NOTE):  # Codex: same frozen final config as S_newmains' export
    raise SystemExit("the extension export was not built from the validated final S_newmains config, or lacks the scope label / numerics note")
nm["extension"] = {"manifest": {"commit": dc_, "path": dp_, "sha256": hashlib.sha256(db_).hexdigest()},
                   "npz": {"path": f"/sol/xfer/il_z/rms_hits/{d_['npz']}", "sha256": d_["npz_sha256"]},
                   "label": d_["label"], "cells": d_["cells"], "arms": d_["arms"], "probes": d_["probes"], "points": d_["points"], "pools": gate_pools(d_["pools"])}
il["posthoc_hits"] = hits
il["posthoc_pooled_rms"] = True
spec["stage_b_numerics_note"] = NOTE
spec["stage_b_meta"] = {"base_spec": {"path": f"experiments/2026-10-15_il_analysis_09/{a.base}", "sha256": a.base_sha256}, "acked_code_commit": ACK,
                        "acked_code_sha256": hashlib.sha256(code_).hexdigest(),
                        "exports": {h["set"]: f"{h['manifest']['path']}@{h['manifest']['commit']} sha256 {h['manifest']['sha256']}" for h in hits},
                        "extension": f"{dp_}@{dc_} sha256 {nm['extension']['manifest']['sha256']}",
                        "five_arm_v2": {"commit": V2[0], "path": V2[1], "sha256": V2[2]},
                        "inputs_final": f"{ifp_}@{ifc_} sha256 {if_sha}"}


def diffs(x, y, path=()):  # 09: by-key difference between r7 and the stage B spec
    if isinstance(x, dict) and isinstance(y, dict):
        out_ = []
        for k_ in set(x) | set(y):
            out_ += [path + (k_,)] if (k_ in x) != (k_ in y) else diffs(x[k_], y[k_], path + (k_,))
        return out_
    return [] if json.dumps(x, sort_keys=True) == json.dumps(y, sort_keys=True) else [path]


bad_ = [d for d in diffs(base_, spec) if not any(d[: len(al)] == al for al in ALLOWED)]
if bad_:
    raise SystemExit(f"BUILD ERROR: the stage B spec differs from r7 outside the allowed keys: {bad_[:5]}")
Path(HERE / a.out).open("x").write(json.dumps(spec, indent=1) + "\n")
print(a.out, hashlib.sha256((HERE / a.out).read_bytes()).hexdigest())
