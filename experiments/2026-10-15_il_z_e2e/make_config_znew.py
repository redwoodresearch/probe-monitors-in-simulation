"""Build config/inputs_znew.json (Z on S_new) from committed sources only:
- 09's IL interim record @ c3d904c0a: resolution index v4 (il311 rows), IL key list (+311 estimation cells), Append 36's not-in-M keys, IL folder manifest pins;
- 09's FINAL record @ 3ce6ea55e: the S_new honest-B pool keys (calibration_records.primary["sol_a|R|0.01|primary|BuC|B"], primary scheme, pool_keys_new);
- 09's FINAL spec @ 3ce6ea55e: the 24 S_new arm folders' Sol / p2 manifest pins;
- the +311 cell map @ d75f997f3 (cell -> main, env); the Sol sim-honest mapping file for simh311_q3 @ 8f792ae8e (per-file sha256s);
- 4f's p2 sim-honest record for simh311_q3 s1 @ f155abeaa (folder manifest).
Only simh311 Q3 s1 is used for tau_sim (09's read gate); Q1 / Q4 stay unread.

  python3 experiments/2026-10-15_il_z_e2e/make_config_znew.py
"""

import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout  # noqa: E731
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
IL_REC = "c3d904c0a:experiments/2026-10-15_il_analysis_09/results/il_interim_s1_result_il_2026-10-15_115759.json"
FINAL = "3ce6ea55e:experiments/2026-10-14_xerr_scaleup_09/results/xerr_scaleup_final_2026-10-15_013255.json"
SPEC = "3ce6ea55e:experiments/2026-10-14_xerr_scaleup_09/specs/final_spec_3_launch.json"
CELLS = "d75f997f3:experiments/2026-10-14_q35_sol_avg_POSTHOC/config/scaleup_tasks_plus311_2026-10-14_125135.json"
SOL_MAP = "8f792ae8e:experiments/2026-10-14_q35_sol_avg_POSTHOC/results/simh_mapping/simh311_q3__2026-10-15_124917.json"
P2_SIMH = "f155abeaa:experiments/2026-10-14_p2_simh_scores/results/simh_manifest_2026-10-15_122917.json"
# second pass: roots given on the command line; Sol pins from the latest mapping file per quarter (@ 106fea88e), p2 pins from 4f's committed records
SOL_MAPS = {q: f"106fea88e:experiments/2026-10-14_q35_sol_avg_POSTHOC/results/simh_mapping/simh311_{q}__2026-10-15_{s}.json"
            for q, s in (("q1", "132803"), ("q3", "132805"), ("q4", "132807"))}
P2_RECS = {"simh311_q1_sim_s1_v1__sim_honest": "ddc8e19da:experiments/2026-10-14_p2_simh_scores/results/simh_manifest_2026-10-15_122300.json",
           "simh311_q4_sim_s1_v1__sim_honest": "ddc8e19da:experiments/2026-10-14_p2_simh_scores/results/simh_manifest_2026-10-15_122415.json",
           "simh311_q3_sim_s1_v1__sim_honest": "f155abeaa:experiments/2026-10-14_p2_simh_scores/results/simh_manifest_2026-10-15_122917.json",
           "simh311_q1_sim_s2_v1__sim_honest": "2b1060531:experiments/2026-10-14_p2_simh_scores/results/simh_manifest_2026-10-15_125455.json",
           "simh311_q3_sim_s2_v1__sim_honest": "2b1060531:experiments/2026-10-14_p2_simh_scores/results/simh_manifest_2026-10-15_125456.json",
           "simh311_q4_sim_s2_v1__sim_honest": "e152df599:experiments/2026-10-14_p2_simh_scores/results/simh_manifest_2026-10-15_130010.json",
           "simh311_q1_sim_s3_v1__sim_honest": "ddaf4856d:experiments/2026-10-14_p2_simh_scores/results/simh_manifest_2026-10-15_131232.json",
           "simh311_q4_sim_s3_v1__sim_honest": "ddaf4856d:experiments/2026-10-14_p2_simh_scores/results/simh_manifest_2026-10-15_131237.json",
           # Q3 s3: the client lost its local record; 4f committed the Volume's own manifest_p2.json byte for byte (client_lost_note @ ddaf4856d)
           "simh311_q3_sim_s3_v1__sim_honest": "ddaf4856d:experiments/2026-10-14_p2_simh_scores/results/simh_manifest_2026-10-15_131236.from_volume.json"}
# 09's exclusion note @ ed7932987 (+ 690d79f50) and RL ruling @ 13be06982; m_r from READ_LOG @ 06f736fa8 (relayed verbatim by 2e)
EXCLUDED = {"q1": [("t237", 22), ("t300", 73), ("t389", 40)], "q4": [("t361", 124), ("t411", 17), ("t413", 27)]}
RL_EXCLUDED = {"q3": [("t298", 193)]}
ROOTS = [a for a in sys.argv[1:] if not a.startswith("--")]
OUT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")), "inputs_znew.json")
R36 = {"path": "/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/a683364c-3a60-44cd-bb15-e08d87dd4f9a/scratchpad/cb_inputs/r36_runs.json",
       "sha256": "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96"}
ROWS = [("Sol (a)", "R", "sol", "a_tight__avg_top2.npz", 5), ("Sol (a)", "C", "sol", "a_tight__cmd_A.npz", 5),
        ("c_asis", "R", "sol", "c_asis_tight__avg_top2.npz", 5), ("c_asis", "C", "sol", "c_asis_tight__cmd_A.npz", 5)]
for fam, nm in (("FIBS", "fibs"), ("harmfulness", "harmfulness"), ("refusal", "refusal"), ("high-stakes", "highstakes")):
    ROWS += [(fam, "R", "p2", f"{nm}__reply_mean.npz", 1), (fam, "C", "p2", f"{nm}__command.npz", 1)]
RECIPE = {"Sol (a)": "other-model-real", "c_asis": "stress test", "FIBS": "public direction", "harmfulness": "public direction", "refusal": "public direction",
          "high-stakes": "public direction"}

ilb, finb, specb, cellb, smapb, p2b = (git("show", x) for x in (IL_REC, FINAL, SPEC, CELLS, SOL_MAP, P2_SIMH))
il = json.loads(ilb)["result_il"]
idx_b = base64.b64decode(il["record_inputs"]["index"]["content_b64"])
kl_b = base64.b64decode(il["record_inputs"]["keylist"]["content_b64"])
assert sha(idx_b) == il["record_inputs"]["index"]["sha256"] and sha(kl_b) == il["record_inputs"]["keylist"]["sha256"]
idx, kl = json.loads(idx_b), json.loads(kl_b)
est_new = set(kl["estimation_set"]["plus311_cells"])
by = {}
for r in idx["rows"]:
    if r["launch"] in ("q1", "q2", "q3", "q4"):
        assert (r["cell"], r["world"]) not in by
        by[(r["cell"], r["world"])] = r
ok = lambda c, w: by.get((c, w), {}).get("status") == "scored" and by[(c, w)].get("attempt_label") == "attempted"  # noqa: E731
co = sorted(c for c in est_new if ok(c, "run3") and ok(c, "sim2"))
n09 = il["results"]["R|0.05"]["co_primary"]["sim2_minus_run3"]["S_new"]["cells"]
assert len(co) == n09, (len(co), n09)
a36 = il["append36"]
not_in_m = set(a36["not_in_M_by_key"])
M = sorted(c for c in kl["estimation_set"]["plus311_cells"] if c not in not_in_m)
assert len(M) == a36["M_cells"], (len(M), a36["M_cells"])
rows_il = {w: {c: [by[(c, w)]["scorer_folder"], by[(c, w)]["key"]] for c in co} for w in ("run3", "sim2")}
il_folders = sorted({v[0] for w in rows_il for v in rows_il[w].values()})
il_man = {f: {"sol": il["inputs"][f]["sol_manifest"], "p2": il["inputs"][f]["p2_manifest"]} for f in il_folders}
fin = json.loads(finb)["result"]
ref = fin["calibration_records"]["primary"]["sol_a|R|0.01|primary|BuC|B"]
pool_new = sorted(ref["schemes"][ref["primary"]]["pool_keys_new"])
spec = json.loads(specb)
arms = {a: {"sol": e["sol_manifest"], "p2": e["p2_manifest"]} for a, e in spec["arms"].items()
        if a.endswith(("_main_v1__attack", "_main_v1__honest_a", "_main_v1__honest_b", "_run2_v1__attack"))}
assert len(arms) == 16, len(arms)
cells = json.loads(cellb)["cells"]
cell_main = {c["key"]: c["main"] for c in cells}
cell_env = {c["key"]: c["env"] for c in cells}
assert len(cell_main) == 311
smap = json.loads(smapb)
root = [r for r in smap["roots"] if r["routed_folder"] == "simh311_q3_sim_s1_v1__sim_honest"]
assert len(root) == 1
sol_simh = {"folder": root[0]["path_written"], "files": root[0]["files"]}
roots_cfg = []
for rt in ROOTS:
    q = rt.split("_")[1]
    mb = git("show", SOL_MAPS[q])
    ent = [r for r in json.loads(mb)["roots"] if r["routed_folder"] == rt]
    assert len(ent) == 1, rt
    pb_ = git("show", P2_RECS[rt])
    pj = json.loads(pb_)
    pf = pj["folders"][f"/simh/simh311_{q}/p2/{rt}"]["files"] if pj.get("folders") else pj["files"]
    assert pj.get("folders") or pj.get("folder", "").rstrip("/").endswith(rt), rt
    roots_cfg.append({"root": rt, "batch": f"simh311_{q}", "sol": {"path": ent[0]["path_written"], "files": ent[0]["files"], "mapping": [SOL_MAPS[q], sha(mb)]},
                      "p2": {"path": f"simh/simh311_{q}/p2/{rt}", "files": pf, "record": [P2_RECS[rt], sha(pb_)]}})
qs = sorted({r["batch"].split("_")[1] for r in roots_cfg})
cid = {c.split("_", 1)[0]: c for c in [x["key"] for x in json.loads(cellb)["cells"]]}
imput = {"terminal_error_exclusions": [{"batch": f"simh311_{q}", "cell": cid[t_], "m_r": m} for q in qs for t_, m in EXCLUDED.get(q, [])],
         "rl_replacement_sensitivity": [{"batch": f"simh311_{q}", "cell": cid[t_], "m_r": m} for q in qs for t_, m in RL_EXCLUDED.get(q, [])]}
p2s = json.loads(p2b)
p2_files = p2s["folders"]["/simh/simh311_q3/p2/simh311_q3_sim_s1_v1__sim_honest"]["files"]
r36_b = Path(R36["path"]).read_bytes()
assert sha(r36_b) == R36["sha256"]
fold_of_main = {}
for r in json.loads(r36_b):
    assert fold_of_main.setdefault(r["main_task"], r["fold"]) == r["fold"]
out = {"label": "Z on S_new (IL il311 co-primary cells; Append 36 M for the matched yardstick); tau_sim from simh311 Q3 s1 ONLY (snapshot, not final); POST HOC wrt 09's registered machinery",
       "fprs": [0.01, 0.001, 0.05],
       "sources": {"il_record": [IL_REC, sha(ilb)], "final_record": [FINAL, sha(finb)], "final_spec": [SPEC, sha(specb)], "cells311": [CELLS, sha(cellb)],
                   "sol_simh_mapping": [SOL_MAP, sha(smapb)], "p2_simh_record": [P2_SIMH, sha(p2b)], "r36": R36},
       "co_primary_cells": co, "M_cells": M, "il_rows": rows_il, "il_manifests": il_man, "H_B_pool_keys": pool_new, "arms": arms,
       "cell_main": cell_main, "cell_env": cell_env, "universe": sorted(cell_main), "fold_of_main": {m: fold_of_main[m] for m in sorted(set(cell_main.values())) if m in fold_of_main},
       "simh": {"batch": "simh311_q3", "routed_folder": "simh311_q3_sim_s1_v1__sim_honest", "sol": sol_simh,
                "p2_folder": "simh/simh311_q3/p2/simh311_q3_sim_s1_v1__sim_honest", "p2_record_sha256": sha(p2b), "p2_files": p2_files,
                "label": "snapshot s1 of Q3, not final"},
       "simh_roots": roots_cfg, "imputation": imput if ROOTS else None,
       "ha25f_folders": ["su311_q1_main_ha25f_v1__honest_a", "su311_q2_main_ha25f_v1__honest_a", "su311_q4_main_ha25f_v1__honest_a"] if ROOTS else [],
       "batch_labels": {"simh311_q4": "FINAL (all Q4 roots routed)", "simh311_q1": "interim (routed roots so far; not the registered row)",
                        "simh311_q3": "interim (routed roots so far; not the registered row); timing amendment: Q3 s1 was read 13:03:38 before the t298 ruling (@ 13be06982)"}
       if ROOTS else {},
       "rows": [{"label": f"{f} {ro}", "family": f, "recipe": RECIPE[f], "readout": ro, "sub": s, "file": fn, "folds": nf} for f, ro, s, fn, nf in ROWS]}
if ROOTS:
    out["label"] = ("Z on S_new, second pass: tau_sim from simh311 " + ", ".join(ROOTS) + " (snapshots, not final), with 09's exclusion scenarios; POST HOC wrt 09's "
                    "registered machinery")
    out["simh"]["label"] = "snapshots: " + ", ".join(ROOTS) + "; not final"
(HERE / "config" / OUT).write_text(json.dumps(out, indent=1) + "\n")
print("co-primary", len(co), "M", len(M), "H_B keys", len(pool_new), "arms", len(arms), "IL folders", len(il_folders), "mapped mains", len(out["fold_of_main"]), "of", len(set(cell_main.values())))
