"""Are famexp's own_a R / C fits (correctedb_scores @ cc53afc3a: cb_tight R, cb_cmd C) the SAME fits as 09 Append 12 item 3's "own_a's command fits"?
By key from committed records and code only; no score values are read. Usage (from fx): python3 -I <this>"""
import hashlib
import json
import subprocess

git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True).stdout  # noqa: E731
X = "experiments/2026-10-14_q35_sol_avg_POSTHOC"
REC = {"append12_cmd_readout (2e Amendment 12)": ("737e6a92a", f"{X}/results/cmd_readout_2026-10-14_033117.json"),
       "append12_cmd_readout_cueex (2e Amendment 20b/20c stage 2)": ("af0c0ec41", f"{X}/results/cmd_readout_cueex_2026-10-14_104112.json"),
       "famexp_own_a = correctedb_scores (2e Amendments 22b/22c)": ("cc53afc3a", f"{X}/results/correctedb_scores_2026-10-14_164934.json")}
MAPS = {"v1 (before the correction)": ("581ffb229", "experiments/2026-10-13_q4_why_weaker_POSTHOC/results/q4b_epoch_map_v1.json"),
        "correctedb_v1": ("4d1eec572", f"{X}/config/q4b_epoch_map_correctedb_v1.json")}
r9 = git("show", "origin/aswin/convergence-audit-09:experiments/2026-10-14_priority1_spec_09/RULE.md").decode().splitlines()
out = {"append_12_item_3": {"where": "experiments/2026-10-14_priority1_spec_09/RULE.md on origin/aswin/convergence-audit-09, line "
                            + str([i + 1 for i, x in enumerate(r9) if "own_a's command fits" in x and x.startswith("3.")]),
                            "dated": [x for x in r9 if x.startswith("## Append 12 ")][0]}}
recs = {}
for k, (c, p) in REC.items():
    b = git("show", f"{c}:{p}")
    d = json.loads(b)
    recs[k] = d
    out[k] = {"path@commit": f"{p} @ {c}", "sha256": hashlib.sha256(b).hexdigest(), "written_by_git": d["git"][:9], "label": d["label"][:110],
              "saved_weights": "none (the record holds run scores, no saved weights)" if "saved" not in d else
              {"C own_a (outer)": d["saved"]["cmd"]["cmdA"]["own_a_weights"]["sha256"], "C own_a_tight": d["saved"]["cmd"]["cmdA"]["own_a_tight_weights"]["sha256"],
               "R own_a_tight (avg) fold models": len(d["saved"]["tight"]["avg"])}}
cb = recs["famexp_own_a = correctedb_scores (2e Amendments 22b/22c)"]
mp = {k: json.loads(git("show", f"{c}:{p}")) for k, (c, p) in MAPS.items()}
inv = {k: {tuple(v): e for e, v in m["map"].items() if v[0] in ("run1", "honest_a")} for k, m in mp.items()}
a, b_ = inv["v1 (before the correction)"], inv["correctedb_v1"]
out["training epochs (run1 + honest_a cells) v1 map vs corrected map"] = {
    "map shas": {k: hashlib.sha256(git("show", f"{c}:{p}")).hexdigest()[:12] + "…" for k, (c, p) in MAPS.items()},
    "correctedb record's corrected_map_sha256": cb["inputs"]["corrected_map_sha256"][:12] + "…",
    "cells in v1 / corrected": [len(a), len(b_)], "cells whose epoch differs": sum(a.get(k) != v for k, v in b_.items()),
    "cells only in one map": len(set(a) ^ set(b_))}
CANDS = ("append12_cmd_readout (2e Amendment 12)", "append12_cmd_readout_cueex (2e Amendment 20b/20c stage 2)")
rc_all = set(cb["per_run_folds"])
rc = {u for u in rc_all if not u.startswith("honest_c|")}  # honest_c (batch C) is correctedb's addition: scored only, never trained
for k in CANDS:
    ra = set(recs[k]["run_scores"]["A_content_only_call_end"]["own_a"])
    tr = lambda s_: {u for u in s_ if u.split("|", 1)[0] in ("run1", "honest_a")}  # noqa: E731
    out[f"own_a command-A runs: {k} vs correctedb (correctedb: {len(rc_all)} runs = {len(rc)} + {len(rc_all) - len(rc)} honest_c, compared without honest_c)"] = {
        "runs": [len(ra), len(rc)], "only in the Append-12 record": len(ra - rc), "only in correctedb": len(rc - ra),
        "training runs (run1 + honest_a) only in correctedb, by world": {w: sum(u.startswith(w + "|") for u in tr(rc) - tr(ra)) for w in ("run1", "honest_a")},
        "training runs only in the Append-12 record": len(tr(ra) - tr(rc))}
rat = git("show", f"{recs['append12_cmd_readout (2e Amendment 12)']['git']}:{X}/run_avg_transfer.py").decode().splitlines()
rcb = git("show", f"{cb['git']}:{X}/run_correctedb.py").decode().splitlines()
ln = lambda src, s: [i + 1 for i, x in enumerate(src) if s in x]  # noqa: E731
LBL = 'y = np.where(rworld == "run1", 1.0'
out["fit code"] = {
    "Append-12 command fits": f"run_avg_transfer.py @ {recs['append12_cmd_readout (2e Amendment 12)']['git'][:9]} cmd_score l.{ln(rat, 'def cmd_score(')}: label rule l.{ln(rat, LBL)}, train rfold != f l.{ln(rat, '(rfold != f))))')}, score rfold == f l.{ln(rat, 'sc = np.flatnonzero(rfold[rid] == f)')}; as-run fit, no tight refit",
    "famexp own_a C": f"run_correctedb.py @ {cb['git'][:9]} cb_cmd l.{ln(rcb, 'def cb_cmd(')} with the 22c/22f tight refit (own_a_tight) on the same rows and standardization; R: cb_tight l.{ln(rcb, 'def cb_tight(')}"}
tm = out["training epochs (run1 + honest_a cells) v1 map vs corrected map"]
ver = {"fitted weights": "UNVERIFIED: the Append-12 records saved no weights, so numerical identity cannot be shown either way",
       "fit procedure": "different: the Append-12 command fits are as-run fits (other code commits); famexp's own_a C is correctedb's 22c/22f TIGHT refit (own_a_tight) and R is cb_tight",
       "training epochs (run1 + honest_a) in the two maps": f"{tm['cells whose epoch differs']} of {tm['cells in v1 / corrected'][1]} cells differ",
       "recipe and fold rule": "same: run1 vs honest_a; train rfold != f, score rfold == f, r36 folds by main (code lines above); every candidate and famexp's fits are cross-fitted by main task"}
for k in CANDS:
    x = next(v for kk, v in out.items() if kk.startswith(f"own_a command-A runs: {k}"))
    same_tr = sum(x["training runs (run1 + honest_a) only in correctedb, by world"].values()) == 0 and x["training runs only in the Append-12 record"] == 0
    ver[f"training run set vs {k}"] = "same" if same_tr else f"different: correctedb trains on {x['training runs (run1 + honest_a) only in correctedb, by world']} more"
out["verdict (by key)"] = ver
print(json.dumps(out, indent=1))
