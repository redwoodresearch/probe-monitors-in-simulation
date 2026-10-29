"""+311 per-action e2e lead (the 455 fork's records, rendered here; no recomputation): reply left, command right, one row per probe, at per
action 1e-2 (lead) and 5e-2 (companion), plus 1e-3 when asked (--levels); 1e-3 readability is stated by script from the records. Delta_full (vs the real mean) with its joint interval (tau re-estimated
per draw) and its fixed-tau interval, the Delta_rr yardstick (tau_A -> run 2 vs tau_B -> run 1) in grey, per-run alarm shares, and the exclusion
scenarios (HIGH, LOW and 09's hot-deck, the last from the 455 fork's hot-deck records, whose shared gaps are asserted equal to the main records').
Titles by script. Modal CPU.

  PYTHONPATH=src <modal 1.6.1> run experiments/2026-10-15_e2e_sim_pipeline/plot_p311.py --c-commit 871a1269d --c-record ... --r-commit 4f9b57700 --r-record ...
"""

import hashlib
import io
import json
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
app = modal.App("probe-shift_aswin-surya_sub-e2esim_kill-2026-11-14")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "matplotlib==3.10.6")
D507 = "experiments/2026-10-15_fpr_sweep_507_prep/results"


@app.function(image=image, cpu=2, memory=4096, timeout=600, retries=0)
def render(recs: dict, names: dict, stem: str, dl: dict, hds: dict, held: list, held_note: str, checks_note: str, own_note: str, levels: list,
           label_note: str = "") -> dict:  # noqa: C901
    import textwrap

    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    R = {ro: json.loads(b) for ro, b in recs.items()}
    HD = {ro: json.loads(b) for ro, b in hds.items()}
    assert set(HD) == set(R), (sorted(HD), sorted(R))
    for k_ in R:  # the hot-deck records reproduce the main ones: every shared gap equal at every level (new fields only differ)
        assert set(HD[k_]["rows"]) == set(R[k_]["rows"]), k_
        for lab_, r_ in R[k_]["rows"].items():
            for kind in ("joint", "fixed"):
                for f_, pt_ in r_["points"][kind].items():
                    for g_, v_ in pt_["gap"].items():
                        h_ = HD[k_]["rows"][lab_]["points"][kind][f_]["gap"][g_]
                        same = (v_["point"] is None and h_["point"] is None) or (v_["point"] != v_["point"] and h_["point"] != h_["point"]) or abs(v_["point"] - h_["point"]) <= 1e-12
                        assert same and v_.get("ci95") == h_.get("ci95"), (k_, lab_, kind, f_, g_)
    HEAD = ["Sol (a)", "FIBS", "harmfulness", "refusal", "high-stakes"]
    OWN = [f for f in ("own_a", "own_frac10_L48", "own_frac25_L48", "own_frac100_L48") if "O" in R and f"{f} R" in R["O"]["rows"]]
    XS = ["c_asis"] if "X" in R else []
    FAM = HEAD + OWN + XS
    NICE = {"Sol (a)": "Sol-trained probe (a)", "FIBS": "FIBS (public)", "harmfulness": "harmfulness direction", "refusal": "refusal direction", "high-stakes": "high-stakes probe",
            "own_a": "own-real (own_a)", "own_frac10_L48": "own-real 10%", "own_frac25_L48": "own-real 25%", "own_frac100_L48": "own-real 100%", "c_asis": "stress test: c_asis"}
    src = lambda fam, ro: R["O"] if fam in OWN else R["X"] if fam in XS else R[ro]  # noqa: E731
    ypos = {fm: i + (0.6 if fm in OWN else 0) + (1.2 if fm in XS else 0) for i, fm in enumerate(FAM)}
    CE = "#e6550d"
    pc = lambda s: (100 * s["point"], 100 * s["ci95"][0], 100 * s["ci95"][1]) if s.get("ci95") else (100 * s["point"], np.nan, np.nan)  # noqa: E731
    out, nums = {}, {}
    hd_f = [x for x in HEAD if x not in held]
    ex0 = R["C"]["rows"]["FIBS C"]["exclusions"]
    assert all(r_["exclusions"]["m_r"] == ex0["m_r"] for k_ in R for r_ in R[k_]["rows"].values()), "rows disagree on the exclusions"
    cls0 = ex0.get("classes", {})
    exT = [m_ for d_ in ex0["m_r"].values() for c0, m_ in d_.items() if cls0.get(c0, "terminal") != "RL"]
    exR = [m_ for d_ in ex0["m_r"].values() for c0, m_ in d_.items() if cls0.get(c0, "terminal") == "RL"]
    ex_txt = f"{len(exT)} terminal/standing-rule runs, M = {sum(exT)} imputed units" + (f"; {len(exR)} RL-class runs, M = {sum(exR)}, are in the records' rl/all scenarios" if exR else "")
    held_head = [x for x in held if x in HEAD]
    hd_n = "five headline probes" if not held_head else f"{len(hd_f)} public headline probes ({', '.join(held_head)} held)"
    cov = {}  # H_A cells (runs) on H_sim's cells, per probe group, from every row's coverage block
    for fam in FAM:
        for ro in ("R", "C"):
            c_ = src(fam, ro)["rows"][f"{fam} {ro}"]["coverage"]
            grp = "public" if fam in HEAD[1:] else "own_a (withheld)" if fam == "own_a" else "own-real" if fam in OWN else fam
            cov.setdefault(grp, set()).add((c_["H_A_runs"], c_["H_sim_runs"]))
    ha_txt = "; ".join(f"{g_} " + "/".join(f"{a_} of {b_}" for a_, b_ in sorted(v)) + (f" ({own_note})" if g_ == "own-real" and own_note and
                                                                                         any(a_ < b_ for a_, b_ in v) else "") for g_, v in cov.items())
    if "O" in HD:  # 2e relayed 09/455: own_frac fold coverage confirmed (every fold 0-4 non-empty on all three pools); own_a has no mapped unit anywhere
        for lab_, r_ in HD["O"]["rows"].items():
            mu_ = r_["mapped_units_per_fold"]
            if lab_.startswith("own_a "):
                assert all(mu_[pl][str(g)] == 0 for pl in ("H_B", "H_A", "H_sim") for g in range(5)), lab_
            else:
                assert all(mu_[pl][str(g)] > 0 for pl in ("H_B", "H_A", "H_sim") for g in range(5)), lab_
    rd3 = {lab_: r_["readable"]["0.001"] for k_ in R for lab_, r_ in R[k_]["rows"].items()}
    un3 = {pl: [lab_ for lab_ in sorted(rd3) if not rd3[lab_][pl]] for pl in ("A", "S", "B")}  # 1e-3 readability by script (it depends on the pool sizes)
    rd3_txt = ("At 1e-3 every pool is readable on every row (k ≥ 1 and at least 10 units above)" if not any(un3.values()) else
               "At 1e-3, unreadable (fewer than 10 units above): " + "; ".join(f"{pl} on " + ("every row" if len(v) == len(rd3) else ", ".join(v)) for pl, v in un3.items() if v))
    hd0 = HD["C"]["rows"]["Sol (a) C"]["hotdeck"]
    for f, lv in [(x_, {"0.01": "1e-2", "0.05": "5e-2", "0.001": "1e-3"}[x_]) for x_ in levels]:
        fig, axs = plt.subplots(1, 2, figsize=(27, 8.6 + 1.25 * (len(OWN) + len(XS))), sharey=True)
        inside, nread, excl0, shares, scen, hb = {}, {}, [], {"sim": [], "real": []}, {"high": [], "low": [], "hotdeck": []}, {}
        inside_own, nown = {}, {}
        rows_out = []
        for ax, ro in zip(axs, ("R", "C")):
            inside[ro], nread[ro], inside_own[ro], nown[ro] = 0, 0, 0, 0
            for fam in FAM:
                i = ypos[fam]
                if fam in held:  # 2e: rows whose records are being rerun are drawn without values
                    ax.text(1.02, i, "HELD: " + held_note, transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=7.4, color="#888888")
                    ax.text(0, i, "held (no values shown)", ha="center", va="center", fontsize=7.5, color="#999999")
                    rows_out.append({"row": f"{fam} {ro}", "held": held_note})
                    continue
                r = src(fam, ro)["rows"][f"{fam} {ro}"]
                jt, fx = r["points"]["joint"][f], r["points"]["fixed"][f]
                gk = "sim_minus_run2" if fam in OWN else "delta_full"  # own-real: the registered comparator (run 1 is their training launch)
                e, ef, y = pc(jt["gap"][gk]), pc(fx["gap"][gk]), pc(jt["gap"]["delta_rr"])
                rd = r["readable"][f]
                withheld = fam == "own_a"  # 09 (25h §1): no mapped-main units on either side on +311
                ok = rd["S"] and rd["B"] and not withheld
                pz = r["pools_at_draw0"][f]
                if np.isfinite(y[1]):
                    ax.barh(i, y[2] - y[1], left=y[1], height=0.55, color="#d9d9d9", zorder=1)
                ax.errorbar([e[0]], [i - 0.12], xerr=[[e[0] - e[1]], [e[2] - e[0]]], fmt="D" if fam in OWN else "o", color=CE if ok else "#bbbbbb",
                            mfc="white" if withheld else (CE if ok else "#bbbbbb"), ms=6, lw=1.8, capsize=3, zorder=3)
                ax.errorbar([e[0]], [i + 0.14], xerr=[[e[0] - ef[1]], [ef[2] - e[0]]], fmt="none", ecolor="#555555", lw=1.0, capsize=2, zorder=3)
                hi = jt["gap"]["delta_full_high"]["point"]
                lo = jt["gap"]["delta_full_low"]["point"]
                d0 = jt["gap"]["delta_full"]["point"]
                hi_t = "unavailable" if hi is None or hi != hi else f"{100 * hi:+.1f} (move {100 * (hi - d0):+.1f})"
                lo_t = "unavailable" if lo is None or lo != lo else f"{100 * lo:+.1f} (move {100 * (lo - d0):+.1f})"
                hdg = HD["O" if fam in OWN else "X" if fam in XS else ro]["rows"][f"{fam} {ro}"]["points"]["joint"][f]["gap"]["delta_full_hotdeck"]
                hdv = hdg["point"]
                hd_t = ("unavailable" if hdv is None or hdv != hdv else f"{100 * hdv:+.1f}" + (f" [{100 * hdg['ci95'][0]:+.1f}, {100 * hdg['ci95'][1]:+.1f}]" if hdg.get("ci95") else "")
                        + f" (move {100 * (hdv - d0):+.1f})")
                dlab = dl.get(ro, {}).get("labels", {}).get(f"{fam} {ro}", {}).get(f, {}).get("label", "") if fam in HEAD else ""
                hi_pt = hi is not None and hi == hi
                dshort = (("(d) same; HIGH point available, joint CI withheld (<95% complete draws)" if hi_pt else "(d) same; HIGH unavailable")
                          if dlab.startswith("same under the available") else "(d) DIFFERS under a scenario" if dlab.startswith("differs")
                          else f"(d) {dlab}" if dlab else "")
                tag = ("WITHHELD (no mapped units on +311); hollow = the 455 fork's all-units sensitivity, deviation from 25h §1\n" if withheld else
                       "vs run 2 (registered); fold coverage confirmed (every fold non-empty, all three pools)\n" if fam in OWN else "")
                ax.text(1.02, i, tag + f"{e[0]:+.1f} [{e[1]:+.1f}, {e[2]:+.1f}] joint; [{ef[1]:+.1f}, {ef[2]:+.1f}] fixed-τ\n"
                        f"yardstick {y[0]:+.1f} [{y[1]:+.1f}, {y[2]:+.1f}]; runs alarmed sim {100 * pz['S']['per_run_alarm_share']:.0f}% / real {100 * pz['B']['per_run_alarm_share']:.0f}%\n"
                        + ("Δ_full scenarios (vs the real mean, not the plotted sim − run 2): " if fam in OWN else "scenarios: ") + f"HIGH {hi_t}; LOW {lo_t}; {dshort}\n"
                        f"hot-deck {hd_t}" + ("" if ok or withheld else "  UNREADABLE"),
                        transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=7.4, color="#222222" if ok else "#888888")
                if ok and fam in HEAD:
                    nread[ro] += 1
                    inside[ro] += bool(np.isfinite(y[1]) and y[1] <= e[0] <= y[2])
                    if not (e[1] <= 0 <= e[2]):
                        excl0.append(f"{fam} {'reply' if ro == 'R' else 'command'} {e[0]:+.1f} [{e[1]:+.1f}, {e[2]:+.1f}]")
                    shares["sim"].append(pz["S"]["per_run_alarm_share"])
                    shares["real"].append(pz["B"]["per_run_alarm_share"])
                elif ok and fam in OWN:
                    nown[ro] += 1
                    inside_own[ro] += bool(np.isfinite(y[1]) and y[1] <= e[0] <= y[2])
                for sc, v in (("high", hi), ("low", lo), ("hotdeck", hdv)):
                    if fam in HEAD and v is not None and v == v:
                        scen[sc].append(100 * (v - jt["gap"]["delta_full"]["point"]))
                hb[f"{fam} {ro}"] = r["n_per_batch"]["H_sim_runs"]
                rows_out.append({"row": f"{fam} {ro}", "d_label": dlab, "gap_key": gk, "withheld": withheld, "delta_full_joint": e, "delta_full_fixed": ef, "delta_rr_joint": y, "readable": rd, "high": hi, "low": lo, "hotdeck": hdg,
                                 "per_run_alarm_share": {"sim": pz["S"]["per_run_alarm_share"], "real_B": pz["B"]["per_run_alarm_share"]},
                                 "H_sim_runs_by_batch": r["n_per_batch"]["H_sim_runs"], "cohort_cells": r["coverage"]["cohort_cells"]})
            for g0 in (OWN[:1] + XS[:1]):
                ax.axhline(ypos[g0] - 0.8, color="#bbbbbb", lw=0.6)
            ax.axvline(0, color="black", lw=0.8)
            for xv in (-5, 5):
                ax.axvline(xv, color="#777777", ls="--", lw=0.7)
            ax.set_xlim(-45, 45)
            ax.set_ylim(min(ypos.values()) - 0.8, max(ypos.values()) + 0.8)  # held rows carry no data: keep every row inside the axes
            ax.set_title(f"{'reply readout (per honest tool output)' if ro == 'R' else 'command readout (per honest tool call)'}", loc="left", fontsize=10.5)
            ax.set_xlabel(f"Δ_full = sim catch at τ_sim − real catch (mean of run 1, run 2) at τ_B, per-action FPR {lv} (points)", fontsize=9.5)
        axs[0].set_yticks([ypos[x] for x in FAM])
        axs[0].set_yticklabels([NICE[x] for x in FAM], fontsize=10)
        axs[0].invert_yaxis()
        med = lambda v: float(np.median(v)) if v else float("nan")  # noqa: E731
        sc_txt = ("HIGH unavailable (the line falls among imputed units)" if not scen["high"] else f"HIGH moves Δ_full by {min(scen['high']):+.1f} to {max(scen['high']):+.1f}")
        sc_txt += "" if not scen["low"] else f"; LOW by {min(scen['low']):+.1f} to {max(scen['low']):+.1f}"
        sc_txt += "" if not scen["hotdeck"] else f"; hot-deck by {min(scen['hotdeck']):+.1f} to {max(scen['hotdeck']):+.1f}"
        lab0 = R["C"]["meta"]["row_label"] + (f". HELD rows ({', '.join(held)}): {held_note}" if held else "")
        ttl = (f"+311 (S_new), per action {lv}, each side sets its own threshold: the Δ_full point lies inside the 95% joint interval of the two-real-pipeline yardstick for "
               f"{inside['R']} of {nread['R']} readable {'headline probes' if not held_head else hd_n.split(' ', 1)[1]} (reply) and {inside['C']} of {nread['C']} (command); Δ_full joint intervals excluding 0 ({hd_n}): "
               f"{'; '.join(excl0) if excl0 else 'none'}; honest runs with ≥ 1 alarm: sim {100 * med(shares['sim']):.0f}% / real {100 * med(shares['real']):.0f}% (median across the "
               f"{hd_n}' {2 * len(hd_f)} probe/readout rows). Exclusion scenarios, {hd_n} ({ex_txt}; moves in percentage points): "
               f"{sc_txt}" + (f"; own-real (vs run 2; own_a {'withheld' if 'own_a' in OWN else 'not in these records'}): inside for {inside_own['R']} of {nown['R']} (reply) and {inside_own['C']} of {nown['C']} (command)" if OWN else "")
               + f". {lab0}")
        fig.suptitle(textwrap.fill(ttl, 170), x=0.01, ha="left", y=0.995, fontsize=11.5)
        hand = [Line2D([], [], color=CE, marker="o", lw=1.8, label="Δ_full vs the real mean (the e2e lead), 95% joint interval (τ re-estimated per draw)"),
                Line2D([], [], color="#555555", lw=1.0, label="the same point's 95% fixed-τ interval (attack tasks resampled only)"),
                Line2D([], [], color=CE, marker="D", lw=1.8, label="own-real: sim − run 2 at each side's own line (registered; run 1 is their training launch)"),
                Patch(color="#d9d9d9", label="yardstick Δ_rr = run 2 at τ_A − run 1 at τ_B, 95% joint")]
        fig.legend(handles=hand, loc="lower left", ncol=3, frameon=False, fontsize=8.8, bbox_to_anchor=(0.01, 0.135))
        sol_held = "Sol (a)" in held
        r0 = R["C"]["rows"]["FIBS C" if sol_held else "Sol (a) C"]
        r0n = "FIBS C" if sol_held else "Sol C"
        hb_txt = str(hb.get("FIBS C")) if sol_held else f"Sol {hb.get('Sol (a) C')}, others {hb.get('FIBS C')}"
        cap = (f"FPR type: per action (share of honest tool calls flagged at the command readout, of honest tool outputs at the reply readout); a run is caught if any unit is "
               f"above the line. Threshold sources: sim = simh311 sim-honest units (H_sim runs per batch: {hb_txt}); real = S_new "
               f"honest B units (the FINAL pool_new; {r0['coverage']['H_B_runs']} runs for {r0n}); yardstick τ_A = S_new honest A units on H_sim's cells (honest A runs of H_sim cells: "
               f"{ha_txt}). Attack cohort: S_new cells "
               + (f"({r0['coverage']['cohort_cells']} of {r0['coverage']['of']} for the public probes), " if sol_held else
                  f"({r0['coverage']['cohort_cells']} of {r0['coverage']['of']} for every probe; Sol (a) and c_asis: 25h §1 fold −1 handling, each pool's fold −1 units scored by "
                  "the mean of the five per-model z-scores, re-standardized on that pool's own fold −1 units), " if str(r0.get("fold_minus1_rule", "")).startswith("weighted 21b") else
                  f"({r0['coverage']['cohort_cells']} of {r0['coverage']['of']} for Sol after its fold −1 drop; 272 for the public probes), ") + "equal-main weighting. Intervals: joint "
               f"(main-level task draws, each pool's moments and τ re-estimated per draw) and fixed-τ. Exclusion scenarios per 09 (@ ed7932987 + 690d79f50): the {len(exT)} terminal/standing-rule excluded runs' "
               "imputed units placed above (HIGH) or below (LOW) every sim-honest unit; hot-deck per " + hd0["rule"] + " (" + hd0["label"] + f"; seed {hd0['stream_seed']}). "
               f"{rd3_txt}. The registered "
               "per-row comparator (sim − run 1 or run 2) is in the records; own-real rows plot it (sim − run 2). Records: " + "; ".join(names[k_] for k_ in sorted(names)) +
               ". " + checks_note + " The hot-deck values are the 455 fork's, not re-checked here. " + (label_note + " " if label_note else "")
               + ("own_a is withheld (09, 25h §1). " if "own_a" in OWN else "own_a is not in these records. ") + "POST HOC; PRELIMINARY.")
        fig.text(0.01, 0.005, textwrap.fill(cap, 230), fontsize=7.8, color="#333333", va="bottom", ha="left")
        fig.subplots_adjust(top=0.83, bottom=0.27, left=0.08, right=0.77, wspace=1.15)
        bio = io.BytesIO()
        fig.savefig(bio, format="png", dpi=115)
        plt.close(fig)
        out[f"{stem}_P311_{lv}.png"] = bio.getvalue()
        nums[lv] = {"title": ttl, "caption": cap, "rows": rows_out, "scenario_shifts_points": scen, "H_A_cells": {k_: sorted(v) for k_, v in cov.items()}, "held": held}
    out[f"{stem}_P311_figure_numbers.json"] = json.dumps(nums, indent=1, default=float).encode()
    return out


def nonha_diffs(new, old, path=(), extra=frozenset()):
    """Leaf-by-leaf, over the OLD record's keys: every field that does not use the honest A pool must be equal (floats to 1e-12, NaN = NaN). Excluded: the
    tau_A catches and gaps (run1@A, run2@A, delta_rr, same_attack), pools_at_draw0 / readable A, coverage H_A_*, ha25f, and the stated fold -1 rule string.
    A per-run alarm share may move by exactly one run when the same pool's threshold moves by <= 1e-12 (host-BLAS near-tie; listed in the diff check)."""
    out, allowed = [], []
    excl = {"delta_rr", "same_attack", "run2@A", "run1@A", "ha25f", "fold_minus1_rule"} | set(extra)
    if any(str(q) in excl or str(q).startswith("H_A") for q in path) or (any(q in ("pools_at_draw0", "readable") for q in path) and "A" in path):
        return out, allowed
    if isinstance(old, dict):
        if not isinstance(new, dict):
            return [(path, "type")], allowed
        for k in old:
            if k not in new:
                out.append((path + (k,), "missing"))
                continue
            if k == "per_run_alarm_share" and isinstance(old[k], float) and abs(new[k] - old[k]) > 1e-12:
                one = abs(abs(new[k] - old[k]) * old["runs"] - 1) < 1e-9 and abs(new["T"] - old["T"]) <= 1e-12
                (allowed if one else out).append((path + (k,), new[k], old[k]))
                continue
            o_, a_ = nonha_diffs(new[k], old[k], path + (k,), extra)
            out += o_
            allowed += a_
        return out, allowed
    if isinstance(old, list):
        if not isinstance(new, list) or len(new) != len(old):
            return [(path, "list")], allowed
        for i_, (x_, y_) in enumerate(zip(new, old)):
            o_, a_ = nonha_diffs(x_, y_, path + (i_,), extra)
            out += o_
            allowed += a_
        return out, allowed
    if isinstance(old, float) and isinstance(new, (int, float)) and not isinstance(new, bool):
        return ([] if abs(new - old) <= 1e-12 or (new != new and old != old) else [(path, new, old)]), allowed
    return ([] if new == old else [(path, new, old)]), allowed


@app.local_entrypoint()
def main(c_commit: str, c_record: str, r_commit: str, r_record: str, dl_commit: str = "9d9b43b78", o_commit: str = "", o_record: str = "", x_commit: str = "",
         x_record: str = "", hd: str = "", base: str = "", dl_base: str = "", held: str = "", held_note: str = "", checks_note: str = "", stem: str = "",
         base_rows: str = "", base_skip: str = "", labels_from_record: bool = False, own_note: str = "", origin: str = "", row_label: str = "", levels: str = "0.01,0.05"):
    """hd: per record key, 'self' (the record carries the hot-deck fields) or <commit>:<file>. base: per key, the older hot-deck record whose every non-H_A
    field the new record must equal. dl_base: per key, the plain record the d-labels were written for (its non-H_A fields must also equal the new record's)."""
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout  # noqa: E731
    label_note = ""
    recs = {"C": git("show", f"{c_commit}:{D507}/{c_record}"), "R": git("show", f"{r_commit}:{D507}/{r_record}")}
    srcs = {"C": (c_commit, c_record), "R": (r_commit, r_record)}
    for key_, (cm, rn) in (("O", (o_commit, o_record)), ("X", (x_commit, x_record))):
        if rn:
            recs[key_] = git("show", f"{cm}:{D507}/{rn}")
            srcs[key_] = (cm, rn)
    names = {k_: f"{rn} @ {cm} (sha256 {hashlib.sha256(recs[k_]).hexdigest()[:16]})" for k_, (cm, rn) in srcs.items()}
    if origin:  # thin adapter for pooled records: lift rows[<label>][<origin>] (the same row() structure) into one record per group, verbatim
        vC, vR = json.loads(recs["C"]), json.loads(recs["R"])
        heads = ["Sol (a)", "FIBS", "harmfulness", "refusal", "high-stakes"]
        grp = {"C": [f"{f} C" for f in heads], "R": [f"{f} R" for f in heads], "X": ["c_asis C", "c_asis R"],
               "O": [f"{f} {ro}" for f in ("own_a", "own_frac10_L48", "own_frac25_L48", "own_frac100_L48") for ro in ("C", "R") if f"{f} {ro}" in {**vC["rows"], **vR["rows"]}]}
        src_of = lambda lab_: vC if lab_.endswith(" C") else vR  # noqa: E731
        recs = {}
        for k_, labs_ in grp.items():
            rows_ = {lab_: src_of(lab_)["rows"][lab_][origin] for lab_ in labs_}
            for lab_ in labs_:  # equality on every field the renderer can read: the lifted row is the source row, byte for byte as JSON
                assert json.dumps(rows_[lab_], sort_keys=True) == json.dumps(src_of(lab_)["rows"][lab_][origin], sort_keys=True), lab_
            recs[k_] = json.dumps({"meta": {"row_label": row_label or vC["meta"]["row_label"]}, "rows": rows_}).encode()
        if row_label and "Q2 interim" in vC["meta"]["row_label"]:  # Codex r9: reconcile the title with the records' stale metadata, by script
            roots_ = sorted(vC["rows"]["FIBS C"][origin]["n_per_batch"]["H_sim_runs_by_root"])
            label_note = (f"Note: the records' meta.row_label still reads 'Q2 interim (s1–s4)'; their {origin} rows hold {len(roots_)} sim-honest roots, "
                          f"Q2 {', '.join(r_.split('_sim_')[1].split('_v1')[0] for r_ in roots_ if r_.startswith('simh311_q2'))}.")
        names = {f"pooled {k_}": f"{rn} @ {cm} (sha256 {hashlib.sha256(git('show', f'{cm}:{D507}/{rn}')).hexdigest()[:16]}), rows[*]['{origin}']"
                 for k_, (cm, rn) in srcs.items()}
    spec = lambda x_: dict(e_.split("=", 1) for e_ in x_.split(",")) if x_ else {}  # noqa: E731
    hds = {}
    for k_, v_ in spec(hd).items():
        if v_ == "self":
            hds[k_] = recs[k_]
        else:
            cm, rn = v_.split(":", 1)
            hds[k_] = git("show", f"{cm}:{D507}/{rn}")
            names[f"{k_} hot-deck"] = f"{rn} @ {cm} (sha256 {hashlib.sha256(hds[k_]).hexdigest()[:16]})"
    assert set(hds) == set(recs), (sorted(hds), sorted(recs))
    eq_log = {}
    for what, sp in (("base", base), ("dl_base", dl_base)):
        for k_, v_ in spec(sp).items():
            cm, rn = v_.split(":", 1)
            old = json.loads(git("show", f"{cm}:{D507}/{rn}"))
            new = json.loads(recs[k_])
            assert set(new["rows"]) == set(old["rows"]), (what, k_)
            d_, a_ = [], []
            for lab_ in [x_ for x_ in old["rows"] if what != "base" or not base_rows or x_ in base_rows.split(",")]:
                # the plain records' exclusion_label strings were relabelled later (the d-labels file carries the corrected ones), so skip them there
                x_, y_ = nonha_diffs(new["rows"][lab_], old["rows"][lab_], (lab_,), {"exclusion_label"} if what == "dl_base" else frozenset(x_ for x_ in base_skip.split(",") if x_))
                d_ += x_
                a_ += y_
            assert not d_, (what, k_, d_[:5])
            eq_log[f"{what} {k_}"] = {"record": f"{rn} @ {cm}", "allowed_one_run_ties": [[list(map(str, p_)), n_, o_] for p_, n_, o_ in a_]}
    dl = {}
    dlb = spec(dl_base)
    for ro in (() if labels_from_record else ("C", "R")):
        cm, rn = dlb[ro].split(":", 1) if ro in dlb else (None, (c_record if ro == "C" else r_record))
        b = git("show", f"{dl_commit}:{D507}/{rn.removesuffix('.json')}_dlabels.json")
        dl[ro] = json.loads(b)
        bound = git("show", f"{cm}:{D507}/{rn}") if cm else recs[ro]
        assert dl[ro]["record_sha256"] == hashlib.sha256(bound).hexdigest(), f"{ro}: d-labels are for another record"
    if labels_from_record:  # newer records carry the corrected exclusion labels themselves (the d-labels file relabelled the first records only)
        for ro in ("C", "R"):
            rr_ = json.loads(recs[ro])["rows"]
            dl[ro] = {"labels": {lab_: {f_: {"label": v_["exclusion_label"]} for f_, v_ in r_["points"]["joint"].items()} for lab_, r_ in rr_.items()}}
    out = render.remote(recs, names, stem or "p311_" + c_record.removesuffix(".json").replace("peraction311_all_C_", ""), dl, hds,
                        [x_ for x_ in held.split(",") if x_], held_note, checks_note or "Independent draw-0 checks: see the E2E results folder.", own_note, levels.split(","), label_note)
    out[f"{stem}_P311_equality_log.json"] = json.dumps(eq_log, indent=1).encode()
    for fn, b in out.items():
        (HERE / "results" / fn).open("xb").write(b)
    print(json.dumps({fn: hashlib.sha256(b).hexdigest() for fn, b in out.items()}, indent=1))
