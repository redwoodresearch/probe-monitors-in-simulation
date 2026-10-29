"""Appendix Z panel (2e / 0d, 17:1x: Z leaves the main figures; no recompute). Renders the EXISTING fifth-pass records only: S_new
ilznew_2026-10-15_154306 (IL co-primary cohort (277); Sol (a)/c_asis under 25h §1; label sidecar ilznew_2026-10-15_154306.NOTE.md) and S_old
ilz_2026-10-15_130917. Per action 1e-2, both readouts, Delta_rr beside each Z. Modal CPU render; the record bytes are sha-checked before upload.

  PYTHONPATH=src <modal 1.6.1> run experiments/2026-10-15_il_z_e2e/plot_z_appendix.py
"""

import hashlib
import io
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
RECS = {"S_new": ("results/ilznew_2026-10-15_154306.json", "beb30668bd009d8c81f9949ae648658a072f017f30b5a35b4bfe555f67a07fc1"),
        "S_old": ("results/ilz_2026-10-15_130917.json", "10194f115dbcbb2866c5")}
app = modal.App("probe-shift_aswin-surya_sub-ilz_kill-2026-11-14")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("matplotlib==3.10.6")
PROBES = ["Sol (a)", "FIBS", "harmfulness", "refusal", "high-stakes", "c_asis"]


@app.function(image=image, cpu=(1, 1), memory=(2048, 2048), timeout=600, retries=0)
def render(new_b: bytes, old_b: bytes, stem: str) -> dict:
    import textwrap

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    new, old = json.loads(new_b), json.loads(old_b)
    k = "unit0.01"

    def zy(rec, lab, origin):
        r = rec["rows"][lab]
        e = r["points"][k]
        if origin == "S_new":
            return e["gap"]["Z"], e["yardstick_on_M"]["Delta_rr_on_M_run2_at_A_minus_run1_at_B"], e["readable_both_sides"], e["readable_yardstick_pools"], e["pools"]
        y = e["comparators"]["yard_cross_launch_run3_at_Alt_minus_run1_at_B"]
        return e["gap"]["Z"], y, e["readable_both_sides"], y.get("readable"), e["pools"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5), sharex=True)
    cols = {"S_new": "#d95f02", "S_old": "#7570b3"}
    for ax, ro in zip(axes, ("R", "C")):
        yt, yl = [], []
        for n_, fam in enumerate(PROBES):
            y0 = -n_ * 3.0 - (0.8 if fam == "c_asis" else 0)
            for dy, origin, rec in ((0, "S_new", new), (-1.0, "S_old", old)):
                z, y, rd, ryd, P = zy(rec, f"{fam} {ro}", origin)
                yy = y0 + dy
                if ryd and y and y.get("ci95"):
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
        ax.set_yticklabels(yl, fontsize=7.5)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlim(-45, 45)
        ax.set_title(f"{'reply' if ro == 'R' else 'command'} readout, per-action FPR 1e-2", fontsize=10)
        ax.set_xlabel("Z = sim2 catch at tau_sim - run3 catch at tau_real (points)\n< 0: the simulated eval under-states catch", fontsize=9)
    title = "APPENDIX (POST HOC): the same-launch Z per origin at per action 1e-2, with each origin's real-vs-real yardstick (grey); fifth pass; predates the final inputs"
    fig.suptitle("\n".join(textwrap.wrap(title, 140)), fontsize=11.5)
    cap = ("Registered headline Z moved to the appendix by the author's choice; all-tasks separate-line Delta_full leads. Fifth pass; predates the final inputs. "
           "S_new: ilznew_2026-10-15_154306 (POST HOC; sim-honest Q1-Q4 FINAL with 09's labels and pending replacements; IL co-primary cohort (277); "
           "Delta_rr on Append 36's M (257); Sol (a)/c_asis under 25h §1 fold -1 handling, 42 of 10,000 draws non-finite under 7310476ae). "
           "S_old: ilz_2026-10-15_130917 (A25 T196 sim-honest; cross-launch yardstick run3 at tau_A - run1 at tau_B, m1). "
           "harmfulness R (S_new) at 1e-2 is +6.9 [0.0, +14.8]: its lower bound touches 0 and does not exclude it. "
           "25h §5: (a) intervals resample TASKS (mains with all runs and units together), never units; (b) per-pool per-run alarm shares and units per run "
           "are in the records; the share of runs at the 32,768-token limit is not available here; (c) tau_sim and tau_real are each re-estimated inside every "
           "draw. PRELIMINARY.")
    fig.text(0.01, 0.005, "\n".join(textwrap.wrap(cap, 210)), fontsize=7.5, va="bottom")
    fig.tight_layout(rect=(0, 0.12, 1, 0.92))
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=130)
    return {f"{stem}.png": b.getvalue(), f"{stem}_title.json": json.dumps({"title": title, "caption": cap}).encode()}


@app.local_entrypoint()
def main():
    bs = {}
    for k, (p, s) in RECS.items():
        b = (HERE / p).read_bytes()
        if not hashlib.sha256(b).hexdigest().startswith(s):
            raise SystemExit(f"{p}: sha256 mismatch")
        bs[k] = b
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    stem = f"z_appendix_fifthpass_RC_1e-2_{stamp}"
    for fn, b in render.remote(bs["S_new"], bs["S_old"], stem).items():
        (HERE / "results" / fn).open("xb").write(b)
        print(fn, hashlib.sha256(b).hexdigest())
