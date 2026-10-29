"""RULE.md (this folder), "Pooled Z, S_old + S_new": POST HOC equal-main pooled Z over both origins' mains, each origin on its own lines, from the
stored per-draw arrays: S_old ilz_2026-10-15_130917 (seed [20261014, 1]) and an S_new export with seed [20261015, 11] (same code, config and roots as
ilznew_2026-10-15_135712). Pooled_i = (A_i Z_old,i + B_i Z_new,i) / (A_i + B_i), with A_i, B_i the summed main multiplicities of each origin's cells.

  PYTHONPATH=src <modal 1.6.1> run experiments/2026-10-15_il_z_e2e/run_zpool.py --new-name <ilznew_... export run name>
"""

import hashlib
import io
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-ilz_kill-2026-11-14"
OLD_NAME, OLD_SEED, D = "ilz_2026-10-15_130917", [20261014, 1], 10000
OLD_REC = ("results/ilz_2026-10-15_130917.json", "10194f115dbc")
NEW_RECS = {"interim239": ("results/ilznew_2026-10-15_135712.json", "03c389c7823d4a50c4a6382413acc7c6315af627093427f6296c3bb8355ce0ac", "config/inputs_znew3_pool.json"),
            "final277": ("results/ilznew_2026-10-15_143726.json", "4de02ecc1e9543af0a27151b29377f1ba648297cdc05c02ec74020333a3d2de1", "config/inputs_znew_final_pool.json"),
            "final277_pass4": ("results/ilznew_2026-10-15_150553.json", "00950ea448d23bb08e428231927380058bef64a16f8a0edfe2d4daa315b2b5f8", "config/inputs_znew_pass4_pool.json"),
            "final277_pass5_s1c": ("results/ilznew_2026-10-15_154306.json", "beb30668bd009d8c81f9949ae648658a072f017f30b5a35b4bfe555f67a07fc1", "config/inputs_znew_pass5_pool.json"),
            "final277_pass5_s1b": ("results/ilznew_2026-10-15_153748.json", "3287a031f73f28162d8b115370e9fe4cb57f8ee0cc61f990e71cb7a1cf5ac62d", "config/inputs_znew_pass5_pool.json"),
            "final277_pass5_s1": ("results/ilznew_2026-10-15_152926.json", "89e15e7ee26f2b2b381b0861a655ceb7aa8e2854f38d03c49bff1ac24f63e33e", "config/inputs_znew_pass5_pool.json")}
ROWS = [f"{f} {ro}" for ro in ("R", "C") for f in ("Sol (a)", "FIBS", "harmfulness", "refusal", "high-stakes", "c_asis")]
MAIN5 = ["Sol (a)", "FIBS", "harmfulness", "refusal", "high-stakes"]
PUB4 = ["FIBS", "harmfulness", "refusal", "high-stakes"]
PTS = (0.01, 0.05)
LABEL = ("POST HOC; not an authorized registered row (25h §5 authorizes only S_new ∪ S_newmains); S_old's sim-honest is A25 T196, read earlier; S_new is "
         "interim (routed roots so far; Q4 FINAL, the other batches interim); origin strata not shown equivalent (21j: inconclusive at this n)")
app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "matplotlib==3.10.6").add_local_python_source("probe_shift")
solavg = modal.Volume.from_name("probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13", version=2)


def fname(lab):
    return lab.replace(" ", "_").replace("(", "").replace(")", "")


@app.function(image=image, cpu=(4, 4), memory=(16384, 16384), timeout=1800, retries=0, volumes={"/solavg": solavg})
def pool(new_name: str, blobs: dict, recs: dict, cfg_old: dict, cfg_new: dict, meta: dict) -> dict:  # noqa: C901
    import numpy as np

    from probe_shift import fpr_core as fc

    def stop(msg):
        raise SystemExit(f"gate failed: {msg}")

    shas = {}
    t196 = json.loads(blobs["t196"])
    v6 = json.loads(blobs["v6"])["result"]["universe_and_split"]
    U_old = sorted(v6["universe"])
    ui_old = {t: i for i, t in enumerate(U_old)}
    env_old = {c["key"]: c["env"] for c in t196["cells"]}
    main_old = {c["key"]: c["main"] for c in t196["cells"]}
    W_old = fc.bootstrap_weights(U_old, env_old, D, OLD_SEED)
    old_cells = cfg_old["il"]["cells"]
    if len(old_cells) != 91 or any(c not in ui_old for c in old_cells):
        stop("S_old cells are not the 91 IL cells in the universe")
    new_main = cfg_new["cell_main"]
    mo, mn = {main_old[c] for c in old_cells}, {new_main[c] for c in cfg_new["co_primary_cells"]}
    if len(mo) != len(old_cells) or len(mn) != len(cfg_new["co_primary_cells"]) or mo & mn:
        stop(f"cells are not one per main, or the origins share mains: {sorted(mo & mn)[:3]}")
    rold, rnew = json.loads(recs["old"]), json.loads(recs["new"])

    def arr(path):
        b = (Path("/solavg") / path).read_bytes()
        shas[path] = hashlib.sha256(b).hexdigest()
        return dict(np.load(io.BytesIO(b)))

    out, perdraw = {"rows": {}}, {}
    for lab in ROWS:
        ao = arr(f"xfer/il_z/{OLD_NAME}/{fname(lab)}/arrays.npz")
        an = arr(f"xfer/il_z/{new_name}/arrays/{fname(lab)}.npz")
        ro_, rn_ = rold["rows"][lab], rnew["rows"][lab]
        if ro_["cells"]["n"] != 91:
            stop(f"{lab}: S_old Z cells {ro_['cells']['n']}")
        A = W_old[:, [ui_old[c] for c in old_cells]].sum(1)
        m1 = [c for c in old_cells if c not in set(ro_["comparator_subsets"]["m1"]["dropped"])]
        Ay = W_old[:, [ui_old[c] for c in m1]].sum(1)
        B, By = an["Wsum_Z"], an["Wsum_M"]
        exp_B = len(cfg_new["co_primary_cells"]) - rnew["diag"][lab]["fold_minus1_exclusion"]["co_primary_cells_excluded"]
        exp_By = len(cfg_new["M_cells"]) - rnew["diag"][lab]["fold_minus1_exclusion"]["M_cells_excluded"]
        if (A[0], Ay[0], B[0], By[0]) != (len(old_cells), len(m1), exp_B, exp_By):
            stop(f"{lab}: draw-0 weights {(A[0], Ay[0], B[0], By[0])} differ from the cell counts {(len(old_cells), len(m1), exp_B, exp_By)}")
        row = {"cells_draw0": {"S_old_Z": len(old_cells), "S_new_Z": int(round(B[0])), "S_old_yardstick_m1": len(m1), "S_new_yardstick_M": int(round(By[0]))},
               "points": {}}
        for a in PTS:
            k = f"unit{a:g}"
            zo, zn = ao[f"Z|{k}"], an[f"Z|{k}"]
            yo, yn = ao[f"yard_cross_launch_run3_at_Alt_minus_run1_at_B|{k}"], an[f"Delta_rr|{k}"]
            # gates: stored arrays reproduce each committed record's draw-0 point; S_old also its interval (same draws)
            so_, sn_ = fc.summ(zo), fc.summ(zn)
            eo, en = ro_["points"][k], rn_["points"][k]
            if abs(so_["point"] - eo["gap"]["Z"]["point"]) > 1e-12 or (so_.get("ci95") or [0, 0]) != (eo["gap"]["Z"].get("ci95") or [0, 0]):
                stop(f"{lab} {k}: S_old arrays differ from the record")
            if abs(sn_["point"] - en["gap"]["Z"]["point"]) > 1e-12:
                stop(f"{lab} {k}: S_new export's point differs from the record")
            yo_r = eo["comparators"]["yard_cross_launch_run3_at_Alt_minus_run1_at_B"]
            if abs(fc.summ(yo)["point"] - yo_r["point"]) > 1e-12 or (fc.summ(yo).get("ci95") or [0, 0]) != (yo_r.get("ci95") or [0, 0]):
                stop(f"{lab} {k}: S_old yardstick arrays differ from the record")
            if abs(fc.summ(yn)["point"] - en["yardstick_on_M"]["Delta_rr_on_M_run2_at_A_minus_run1_at_B"]["point"]) > 1e-12:
                stop(f"{lab} {k}: S_new export's yardstick point differs from the record")
            for sk_ in [x for x in an if x.startswith("Zscen|") and x.endswith(f"|{k}")]:
                nm_ = sk_.split("|", 1)[1].rsplit("|", 1)[0]
                pr_ = (en["replacement_removal"]["Z"]["point"] if nm_ == "replacement_removal" else en["exclusion_scenarios"]["scenarios"][nm_]["Z"]["point"])
                if not (abs(an[sk_][0] - pr_) <= 1e-12 or (np.isnan(an[sk_][0]) and np.isnan(pr_))):
                    stop(f"{lab} {k}: S_new export's {nm_} point differs from the record")
            pz = (A * zo + B * zn) / (A + B)
            py = (Ay * yo + By * yn) / (Ay + By)
            sz, sy = fc.summ(pz), fc.summ(py)
            rd_old = eo["readable_both_sides"]
            rd_new = en["readable_both_sides"]
            ry_old = (eo["comparators"]["yard_cross_launch_run3_at_Alt_minus_run1_at_B"].get("readable"))
            ry_new = en["readable_yardstick_pools"]
            ent = {"pooled_Z": sz, "pooled_yardstick": sy, "readable": bool(rd_old and rd_new), "yardstick_readable": bool(ry_old and ry_new),
                   "weights_draw0": {"S_old": float(A[0]), "S_new": float(B[0])},
                   "per_set": {"S_old_record": eo["gap"]["Z"], "S_new_record": en["gap"]["Z"], "S_new_export_seed_20261015_11": sn_,
                               "S_old_yardstick_record": {kk: eo["comparators"]["yard_cross_launch_run3_at_Alt_minus_run1_at_B"].get(kk) for kk in ("point", "ci95")},
                               "S_new_yardstick_record": en["yardstick_on_M"]["Delta_rr_on_M_run2_at_A_minus_run1_at_B"]},
                   "scenarios": {}}
            ent["point_inside_pooled_yardstick_95"] = (bool(sy["ci95"][0] <= sz["point"] <= sy["ci95"][1]) if sy.get("ci95") and sz.get("ci95") and ent["readable"]
                                                       and ent["yardstick_readable"] else None)
            for sk in sorted(x for x in an if x.startswith("Zscen|") and x.endswith(f"|{k}")):
                nm = sk.split("|", 1)[1].rsplit("|", 1)[0]
                if "rl_t298" in nm:
                    continue  # 09: t298 is not in the roots read (S_new reports it as a labelled extra only)
                ps = fc.summ((A * zo + B * an[sk]) / (A + B))
                sc_new = en.get("replacement_removal", {}) if nm == "replacement_removal" else en.get("exclusion_scenarios", {}).get("scenarios", {}).get(nm, {})
                ent["scenarios"][nm] = {"pooled_Z": ps, "S_new_status": sc_new.get("status"), "label": sc_new.get("label")}
                yk_ = f"Delta_rr_scen|{nm}|{k}"
                if yk_ in an:  # Codex r7: a scenario with its own refitted S_new yardstick (replacement removal) pools it the same way
                    psy = fc.summ((Ay * yo + By * an[yk_]) / (Ay + By))
                    rdz_ = bool(sc_new.get("readable_both_sides")) and bool(rd_old)
                    rdy_ = bool(sc_new.get("readable_yardstick_pools")) and bool(ry_old)
                    ent["scenarios"][nm] |= {"pooled_yardstick": psy, "readable": rdz_, "yardstick_readable": rdy_,
                                             "point_inside_pooled_yardstick_95": (bool(psy["ci95"][0] <= ps["point"] <= psy["ci95"][1])
                                                                                  if rdz_ and rdy_ and psy.get("ci95") and ps.get("ci95") else None)}
            row["points"][k] = ent
            perdraw[(lab, k)] = pz
        out["rows"][lab] = row
    fam = {}
    for ro in ("R", "C"):
        for fn, mem in (("all 5 main probes", MAIN5), ("4 publics", PUB4)):
            for a in PTS:
                k = f"unit{a:g}"
                labs = [f"{m} {ro}" for m in mem]
                st = np.vstack([perdraw[(l_, k)] for l_ in labs])
                fam[f"{fn}|{ro}|{k}"] = {"members": labs, "readable": {l_: out["rows"][l_]["points"][k]["readable"] for l_ in labs},
                                         "mean_pooled_Z": fc.summ(st.mean(0)), "mean_abs_pooled_Z": fc.summ(np.abs(st).mean(0))}
    out["families"] = fam
    out["containment"] = {f"unit{a:g}": {"inside": [l_ for l_ in ROWS if l_.split(" ")[0] != "c_asis" and out["rows"][l_]["points"][f"unit{a:g}"]["point_inside_pooled_yardstick_95"]],
                                         "outside": [l_ for l_ in ROWS if l_.split(" ")[0] != "c_asis" and out["rows"][l_]["points"][f"unit{a:g}"]["point_inside_pooled_yardstick_95"] is False],
                                         "not_readable": [l_ for l_ in ROWS if l_.split(" ")[0] != "c_asis" and out["rows"][l_]["points"][f"unit{a:g}"]["point_inside_pooled_yardstick_95"] is None]}
                          for a in PTS}
    rec = {"label": "Pooled Z (S_old + S_new), combined: " + LABEL + "; PRELIMINARY; from raw data, by script", "meta": meta | {"arrays_sha256": shas},
           "design": {"pooled": "equal-main: (A_i Z_old,i + B_i Z_new,i) / (A_i + B_i); A_i, B_i = summed main multiplicities of each origin's Z cells in draw i "
                                "(cells are mains; the origins' main sets are disjoint, gated)",
                      "draws": "S_old: A24 generator over v6's universe, seed [20261014, 1] (the record's draws); S_new: A24 generator over the +311 universe, "
                               "seed [20261015, 11] (an export rerun; the record used [20261014, 1]); independent streams on disjoint task sets, paired by "
                               "draw index (a valid joint bootstrap)",
                      "yardstick": "S_old cross-launch run3 at tau_A - run1 at tau_B on m1 (record's per-row dropped list); S_new Delta_rr on its record's M ("
                                   f"{out['rows']['FIBS R']['cells_draw0']['S_new_yardstick_M']}; Sol (a)/c_asis {out['rows']['Sol (a) R']['cells_draw0']['S_new_yardstick_M']}; "
                                   f"cohort {meta.get('cohort')}), combined the same way",
                      "scenarios": "S_new's exclusion scenarios pooled with S_old's primary (S_old has none); a scenario S_new labels an extra (the interim cohort's t298 arm) is left out",
                      "order": "per-set records keep their own ordering; only this POST HOC record and figure lead with the pooled row"},
           **out}
    return {"bytes": json.dumps(rec, indent=1, default=float).encode()}


@app.function(image=image, cpu=(2, 2), memory=(4096, 4096), timeout=600, retries=0)
def render(rec_b: bytes, stem: str) -> dict:
    import textwrap

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rec = json.loads(rec_b)
    fams = MAIN5 + ["c_asis"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), sharex=True)
    cols = {"pooled": "#d95f02", "S_old": "#7570b3", "S_new": "#1b9e77"}
    for i, ro in enumerate(("R", "C")):
        for j, a in enumerate(PTS):
            ax, k = axes[i, j], f"unit{a:g}"
            yt, yl = [], []
            for n_, fam in enumerate(fams):
                e = rec["rows"][f"{fam} {ro}"]["points"][k]
                y0 = -n_ * 4.0 - (0.8 if fam == "c_asis" else 0)
                for dy_, yk_ in ((0, e["per_set"]["S_new_yardstick_record"]), (-1.0, e["per_set"]["S_old_yardstick_record"]), (-2.0, e["pooled_yardstick"])):
                    if yk_ and yk_.get("ci95") and (dy_ != -2.0 or e["yardstick_readable"]):
                        ax.fill_betweenx([y0 + dy_ - 0.35, y0 + dy_ + 0.35], 100 * yk_["ci95"][0], 100 * yk_["ci95"][1], color="#cccccc", zorder=1)
                for dy, nm, z in ((0, "S_new", e["per_set"]["S_new_record"]), (-1.0, "S_old", e["per_set"]["S_old_record"]), (-2.0, "pooled", e["pooled_Z"])):  # 09: per-set rows lead
                    yy = y0 + dy
                    if z.get("ci95"):
                        ax.hlines(yy, 100 * z["ci95"][0], 100 * z["ci95"][1], color=cols[nm], lw=2.2 if nm == "pooled" else 1.3, zorder=3)
                        ax.plot(100 * z["point"], yy, "o", color=cols[nm], ms=6 if nm == "pooled" else 4, zorder=4)
                        ax.text(44, yy, f"{100 * z['point']:+.1f} [{100 * z['ci95'][0]:+.1f}, {100 * z['ci95'][1]:+.1f}]", fontsize=7, va="center", ha="right",
                                color=cols[nm])
                    yt.append(yy)
                    yl.append(f"{'stress test: c_asis' if fam == 'c_asis' else fam}: {'combined (POST HOC, labelled)' if nm == 'pooled' else nm}")
            ax.set_yticks(yt)
            ax.set_yticklabels(yl, fontsize=7.5)
            ax.axvline(0, color="black", lw=0.8)
            ax.set_xlim(-45, 45)
            ax.set_title(f"{'reply' if ro == 'R' else 'command'} readout, per-action FPR {a:g}", fontsize=10)
            if i == 1:
                ax.set_xlabel("Z = sim2 catch at tau_sim - run3 catch at tau_real (points)\n< 0: the simulated eval under-states catch", fontsize=9)
    c1 = rec["containment"]["unit0.01"]
    w0 = rec["rows"]["FIBS R"]["points"]["unit0.01"]["weights_draw0"]
    ws = rec["rows"]["Sol (a) R"]["points"]["unit0.01"]["weights_draw0"]
    coh = {"interim239": "IL interim cohort", "final277": "IL FINAL cohort", "final277_pass4": "IL FINAL cohort, fourth pass", "final277_pass5_s1": "IL FINAL cohort, fifth pass, 25h §1 fold -1", "final277_pass5_s1b": "IL FINAL cohort, fifth pass, 25h §1 fold -1", "final277_pass5_s1c": "IL FINAL cohort, fifth pass, 25h §1 fold -1"}.get(rec["meta"].get("cohort", "interim239"))
    def inside(z, y):
        return bool(z.get("ci95") is not None and y and y.get("ci95") and y["ci95"][0] <= z["point"] <= y["ci95"][1])
    mains_ = [f"{f_} {r_}" for r_ in ("R", "C") for f_ in MAIN5]
    pts_ = [rec["rows"][l_]["points"]["unit0.01"] for l_ in mains_]
    n_new = sum(inside(e_["per_set"]["S_new_record"], e_["per_set"]["S_new_yardstick_record"]) for e_ in pts_)
    n_old = sum(inside(e_["per_set"]["S_old_record"], e_["per_set"]["S_old_yardstick_record"]) for e_ in pts_)
    title = (f"POST HOC. Z per origin at per action 1e-2 (S_new {coh}; S_old): Z points inside each origin's real-vs-real yardstick 95%: S_new {n_new} of 10, "
             f"S_old {n_old} of 10 main rows; combined row labelled (equal-main over {int(w0['S_old'])} + {int(w0['S_new'])} mains; {int(ws['S_old'])} + {int(ws['S_new'])} for "
             f"Sol (a)/c_asis): combined Z points lie inside the combined "
             f"real-vs-real yardstick's 95% interval for {len(c1['inside'])} of {len(c1['inside']) + len(c1['outside'])} readable main rows at per action 1e-2")
    fig.suptitle("\n".join(textwrap.wrap(title, 130)), fontsize=12)
    yc = rec["rows"]["FIBS R"]["cells_draw0"]
    yc_s = rec["rows"]["Sol (a) R"]["cells_draw0"]
    cap = (LABEL + f". Per-set rows (S_old record ilz_2026-10-15_130917; S_new record {rec['meta']['records']['S_new'].split('/')[-1]}) sit beside each pooled row; this figure "
           "reorders them (pooled first) only here; the per-set records keep their own ordering. Pooled = equal-main mean over both origins' mains, each origin "
           "on its own lines (tau_sim from its sim-honest units, tau_real from its honest B units); 95% intervals pair draw i of independent per-origin task "
           "bootstraps (S_old seed [20261014, 1]; S_new export seed [20261015, 11], so the pooled S_new component's interval differs slightly from the "
           f"record's). Grey: pooled yardstick = S_old cross-launch run3 at tau_A - run1 at tau_B (m1) with S_new Delta_rr on its M ({yc['S_new_yardstick_M']}; Sol (a)/c_asis "
           f"{yc_s['S_new_yardstick_M']}), combined the same way. Sol (a)/c_asis S_new cells: {yc_s['S_new_Z']}. 25h §5: (a) intervals resample TASKS (mains with "
           "all runs and units together), never units independently (each origin's code: run_z.py / run_znew.py unit pipelines, W[draw, task] multiplicities); "
           "(b) per-pool per-run alarm shares and units per run are in each origin's record and pools table; the share of runs at the 32,768-token limit is not "
           "available in these score files; (c) tau_sim and tau_real are each re-estimated inside every draw. PRELIMINARY.")
    fig.text(0.01, 0.005, "\n".join(textwrap.wrap(cap, 210)), fontsize=7.5, va="bottom")
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=120)
    return {f"{stem}.png": b.getvalue(), f"{stem}_title.json": json.dumps({"title": title, "caption": cap}).encode()}


@app.local_entrypoint()
def main(new_name: str = "", mode: str = "pool", record: str = "", sha: str = "", cohort: str = "interim239"):
    NEW_REC = NEW_RECS[cohort][:2]
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout  # noqa: E731
    R = HERE / "results"
    if mode == "render":
        rb = (R / record).read_bytes()
        if hashlib.sha256(rb).hexdigest() != sha:
            raise SystemExit("record sha256 mismatch")
        for fn, b in render.remote(rb, record.removesuffix(".json") + "_fig_r3").items():
            (R / fn).open("xb").write(b)
            print(fn, hashlib.sha256(b).hexdigest())
        return
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("uncommitted tracked changes")
    cfg_old = json.loads((HERE / "config/inputs_z.json").read_text())
    cfg_new = json.loads((HERE / NEW_RECS[cohort][2]).read_text())
    blobs = {}
    for k in ("t196", "v6"):
        spec = cfg_old["local"][k]
        b = git("show", spec["git"])
        if hashlib.sha256(b).hexdigest() != spec["sha256"]:
            raise SystemExit(f"{k} sha256 mismatch")
        blobs[k] = b
    exp_ = json.loads((R / f"{new_name}.json").read_bytes())  # the export's own record: its config must be the pooling config (independent seed)
    if exp_["meta"]["config_sha256"] != hashlib.sha256((HERE / NEW_RECS[cohort][2]).read_bytes()).hexdigest() or cfg_new.get("draw_seed") != [20261015, 11]:
        raise SystemExit(f"the S_new export was not run on {NEW_RECS[cohort][2]} with draw_seed [20261015, 11]")
    recs = {}
    for k, (p, s) in (("old", OLD_REC), ("new", NEW_REC)):
        b = (HERE / p).read_bytes()
        if not hashlib.sha256(b).hexdigest().startswith(s):
            raise SystemExit(f"{p}: sha256 mismatch")
        recs[k] = b
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    meta = {"git": git("rev-parse", "HEAD").decode().strip(), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "stamp_pt": stamp,
            "S_new_export_run": new_name, "records": {"S_old": OLD_REC[0], "S_new": NEW_REC[0]}, "cohort": cohort}
    out = pool.remote(new_name, blobs, recs, cfg_old, cfg_new, meta | {"NEW_REC": list(NEW_REC)})
    fn = f"zpool_{stamp}.json"
    (R / fn).open("xb").write(out["bytes"])
    print(json.dumps({"record": f"experiments/2026-10-15_il_z_e2e/results/{fn}", "sha256": hashlib.sha256(out["bytes"]).hexdigest()}, indent=1))
