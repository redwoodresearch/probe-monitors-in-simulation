"""Combined appendix figure (0d GO via 2e, 17:2x): left, the same-launch Z per origin (fifth-pass records; predates the final inputs); right, the
registered Append 36 Delta_same (IL FINAL record @ d851974f5), read by key. Render only (Modal CPU); records sha-checked before upload; a figure-numbers JSON
lists the key path of every number drawn. Replaces z_appendix_fifthpass_RC_1e-2_2026-10-15_170040.png (kept; superseded by NOTE).

  PYTHONPATH=src <modal 1.6.1> run experiments/2026-10-15_il_z_e2e/plot_appendix_combined.py
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
RECS = {"S_new": ("results/ilznew_2026-10-15_154306.json", "beb30668bd009d8c81f9949ae648658a072f017f30b5a35b4bfe555f67a07fc1"),
        "S_old": ("results/ilz_2026-10-15_130917.json", "10194f115dbcbb2866c5")}
IL = ("d851974f5", "experiments/2026-10-15_il_analysis_09/results/il_final_result_il_2026-10-15_130350.json")
app = modal.App("probe-shift_aswin-surya_sub-ilz_kill-2026-11-14")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("matplotlib==3.10.6")
PROBES = ["Sol (a)", "FIBS", "harmfulness", "refusal", "high-stakes", "c_asis"]


@app.function(image=image, cpu=(1, 1), memory=(2048, 2048), timeout=600, retries=0)
def render(new_b: bytes, old_b: bytes, il_b: bytes, stem: str, il_sha: str) -> dict:  # noqa: C901
    import textwrap

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    new, old, il = json.loads(new_b), json.loads(old_b), json.loads(il_b)["result_il"]
    nums = {}
    k = "unit0.01"
    fig = plt.figure(figsize=(21, 9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.8])
    cols = {"S_new": "#d95f02", "S_old": "#7570b3"}
    for i_, ro in enumerate(("R", "C")):
        ax = fig.add_subplot(gs[0, i_])
        yt, yl = [], []
        for n_, fam in enumerate(PROBES):
            y0 = -n_ * 3.0 - (0.8 if fam == "c_asis" else 0)
            for dy, origin, rec in ((0, "S_new", new), (-1.0, "S_old", old)):
                lab = f"{fam} {ro}"
                e = rec["rows"][lab]["points"][k]
                if origin == "S_new":
                    ykey = f"rows['{lab}'].points.{k}.yardstick_on_M.Delta_rr_on_M_run2_at_A_minus_run1_at_B"
                    y, ryd = e["yardstick_on_M"]["Delta_rr_on_M_run2_at_A_minus_run1_at_B"], e["readable_yardstick_pools"]
                else:
                    ykey = f"rows['{lab}'].points.{k}.comparators.yard_cross_launch_run3_at_Alt_minus_run1_at_B"
                    y = e["comparators"]["yard_cross_launch_run3_at_Alt_minus_run1_at_B"]
                    ryd = y.get("readable")
                z, rd, P = e["gap"]["Z"], e["readable_both_sides"], e["pools"]
                nums[f"Z|{origin}|{lab}|1e-2"] = {"record": RECS[origin][0], "record_field": f"rows['{lab}'].points.{k}.gap.Z", "point": z.get("point"), "ci95": z.get("ci95")}
                nums[f"yardstick|{origin}|{lab}|1e-2"] = {"record": RECS[origin][0], "record_field": ykey, "point": y.get("point"), "ci95": y.get("ci95")}
                yy = y0 + dy
                if ryd and y.get("ci95"):
                    ax.fill_betweenx([yy - 0.35, yy + 0.35], 100 * y["ci95"][0], 100 * y["ci95"][1], color="#cccccc", zorder=1)
                if not rd:
                    ax.text(0, yy, f"not readable (sim n={P['H_sim']['n_units']}, k={P['H_sim']['k']}; real n={P['H_B']['n_units']}, k={P['H_B']['k']})",
                            ha="center", va="center", fontsize=6.5, color="#666666")
                elif z.get("ci95"):
                    ax.hlines(yy, 100 * z["ci95"][0], 100 * z["ci95"][1], color=cols[origin], lw=1.8, zorder=3)
                    ax.plot(100 * z["point"], yy, "o", color=cols[origin], ms=5, zorder=4)
                    ax.text(44, yy, f"{100 * z['point']:+.1f} [{100 * z['ci95'][0]:+.1f}, {100 * z['ci95'][1]:+.1f}]", fontsize=7, va="center", ha="right",
                            color=cols[origin])
                yt.append(yy)
                yl.append(f"{'stress test: c_asis' if fam == 'c_asis' else fam}: {origin}{' (IL co-primary cohort (277))' if origin == 'S_new' else ''}")
        ax.set_yticks(yt)
        ax.set_yticklabels(yl, fontsize=7)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlim(-45, 45)
        ax.set_title(f"Z, {'reply' if ro == 'R' else 'command'} readout, per-action FPR 1e-2\n(fifth pass; predates the final inputs)", fontsize=9.5)
        ax.set_xlabel("Z = sim2 catch at tau_sim - run3 catch at tau_real (points)", fontsize=8.5)
    # right: Append 36, read by key
    a = il["append36"]
    q = a["quantities_pp"]
    rd_ = a["reading"]
    ct = a["claim_title_append36_addendum"]
    ax = fig.add_subplot(gs[0, 2])
    items = [("Delta_sep", "Delta_sep (sim - mean real, original launches)"), ("Delta_same", "Delta_same (sim2 - run3, interleaved)"), ("L", "L = Delta_sep - Delta_same")]
    for j, (key, name) in enumerate(items):
        v = q[key]
        yy = -j
        ax.hlines(yy, v["ci95"][0], v["ci95"][1], color="#1b9e77", lw=2, zorder=3)
        ax.plot(v["point"], yy, "o", color="#1b9e77", zorder=4)
        ax.text(v["ci95"][1] + 0.5, yy, f"{v['point']:+.1f} [{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}]", fontsize=8, va="center")
        nums[f"append36|{key}"] = {"record": f"{IL[1]} @ {IL[0]}", "record_field": f"result_il.append36.quantities_pp.{key}", "point": v["point"], "ci95": v["ci95"]}
    ax.axvline(0, color="black", lw=0.8)
    for x_ in (-5, 5):
        ax.axvline(x_, color="#888888", lw=0.6, ls=":")
    ax.set_yticks([-j for j in range(len(items))])
    ax.set_yticklabels([n for _, n in items], fontsize=8)
    ax.set_xlim(-15, 25)
    ax.set_xlabel("catch gap (percentage points)", fontsize=8.5)
    ax.set_title("Same-launch (interleaved) gap, per run,\nreal-line calibration (Append 36; registered)\nSol (a), reply, run-level FPR 5%, S_new on M "
                 f"({a['M_cells']} cells)", fontsize=9.5)
    gate0 = f"(0) availability: {rd_['jointly_finite_draws']} of {rd_['draws']} draws jointly finite (rule: at least 9,500) -> available"
    gate1 = (f"(1) baseline: Delta_sep 95% [{q['Delta_sep']['ci95'][0]:+.1f}, {q['Delta_sep']['ci95'][1]:+.1f}] "
             f"{'lies entirely above 0 -> established' if q['Delta_sep']['ci95'][0] > 0 else 'does not lie entirely above 0 -> no reading'}")
    ax.text(0.0, -0.16, "\n".join(textwrap.wrap(gate0, 62) + textwrap.wrap(gate1, 62) + [f"reading (record): {rd_['reading']}",
                                                                                         f"title-rule case (record field): {ct['case']}"]),
            transform=ax.transAxes, fontsize=7.5, va="top")
    nums["append36|gates"] = {"record": f"{IL[1]} @ {IL[0]}", "record_field": "result_il.append36.reading.jointly_finite_draws / .draws and quantities_pp.Delta_sep.ci95",
                              "note": "gate outcomes derived from the stored fields by 09's RULE Append 36 §6 (0)-(1); the record stores no separate gate fields",
                              "jointly_finite_draws": rd_["jointly_finite_draws"], "draws": rd_["draws"], "Delta_sep_ci95": q["Delta_sep"]["ci95"]}
    nums["append36|reading"] = {"record_field": "result_il.append36.reading.reading", "value": rd_["reading"]}
    nums["append36|title_rule"] = {"record_field": "result_il.append36.claim_title_append36_addendum", "case": ct["case"], "title_text": ct["title"]}
    title = "APPENDIX (POST HOC): the same-launch Z per origin (left) and the registered Append 36 same-launch gap (right)"
    fig.suptitle(title, fontsize=12)
    cap = ("Registered headline Z moved to the appendix by the author's choice; all-tasks separate-line Delta_full leads. LEFT, Z: fifth pass; predates the final "
           "inputs. S_new = ilznew_2026-10-15_154306 (POST HOC; sim-honest Q1-Q4 FINAL with 09's labels; IL co-primary cohort (277); Delta_rr on Append 36's M (257) in "
           "grey; Sol (a)/c_asis under 25h §1 fold -1 handling); S_old = ilz_2026-10-15_130917 (A25 T196 sim-honest; cross-launch yardstick, grey). harmfulness R "
           "(S_new) at 1e-2 is +6.9 [0.0, +14.8]: its lower bound touches 0 and does not exclude it. RIGHT, Append 36 (IL FINAL record " + IL[0] + ", sha256 "
           + il_sha[:12] + "...): Delta_sep, Delta_same and L with 95% intervals; dotted lines: +/-5 (21j margin). Ordered gates (derived from the stored fields): "
           + gate0 + "; " + gate1 + ". Reading (record): " + rd_["reading"] + ". Title-rule outcome (record field, quoted): case " + ct["case"] + ", text: \""
           + ct["title"] + "\". 25h §5: (a) intervals resample TASKS (mains with all runs and units together), never units; (b) per-pool per-run alarm shares "
           "and units per run are in the Z records; the share of runs at the 32,768-token limit is not available here; (c) tau_sim and tau_real are each re-estimated "
           "inside every draw (Z); Append 36 uses the registered real-line calibration. PRELIMINARY.")
    fig.text(0.01, 0.005, "\n".join(textwrap.wrap(cap, 250)), fontsize=7.2, va="bottom")
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=125)
    return {f"{stem}.png": b.getvalue(), f"{stem}_title.json": json.dumps({"title": title, "caption": cap}).encode(),
            f"{stem}_figure_numbers.json": json.dumps(nums, indent=1).encode()}


@app.local_entrypoint()
def main():
    bs = {}
    for k_, (p, s) in RECS.items():
        b = (HERE / p).read_bytes()
        if not hashlib.sha256(b).hexdigest().startswith(s):
            raise SystemExit(f"{p}: sha256 mismatch")
        bs[k_] = b
    ilb = subprocess.run(["git", "show", f"{IL[0]}:{IL[1]}"], capture_output=True, check=True, cwd=HERE).stdout
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    stem = f"appendix_Z_and_Append36_1e-2_{stamp}"
    for fn, b in render.remote(bs["S_new"], bs["S_old"], ilb, stem, hashlib.sha256(ilb).hexdigest()).items():
        (HERE / "results" / fn).open("xb").write(b)
        print(fn, hashlib.sha256(b).hexdigest())
