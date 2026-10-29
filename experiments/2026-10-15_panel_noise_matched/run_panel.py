"""Noise-matched gap vs the two-real-pipeline yardstick (judge panel; RULE.md @ 40b9db23d). POST HOC; PRELIMINARY until reviewed.

Per row p (probe x readout) and per-action point f, on each draw of the input's joint bootstrap (tau re-estimated per draw):
  G_p = 0.5 |sim@tau_sim - run1@tau_B| + 0.5 |sim@tau_sim - run2@tau_B|,   Y_p = |yardstick| = |run2@tau_C(or A) - run1@tau_B|.
Group means of G and Y, their paired difference and the readout contrast are computed per draw on the same draws; point = draw 0,
95% interval = percentiles over draws 1..D (fpr_core.summ). A group's draw is NaN if any member row is NaN in it.

  PYTHONPATH=src uv run modal run experiments/2026-10-15_panel_noise_matched/run_panel.py --cfg s_old_183_matched
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-panelnm_kill-2026-11-14"
VOL = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13"
LABEL = "POST HOC (judge-panel request, not registered); PRELIMINARY until Codex, 09 or Aswin review; from raw data, by script"
ON = ["own_frac10_L48", "own_frac25_L48", "own_frac100_L48", "own_a"]
OFF = ["Sol (a)", "FIBS", "harmfulness", "refusal", "high-stakes"]
DT = ["DT_destructive_primary", "DT_destructive_secondary", "DT_transfer_primary", "DT_transfer_secondary"]
GROUPS = {"on_policy_own_family": ON, "off_policy": OFF, "DT": DT, "overall": ON + OFF + DT, "overall_without_DT": ON + OFF}
STRESS = ["c_asis"]
app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "matplotlib==3.10.6").add_local_python_source("probe_shift")


def split(lab):
    fam, ro = lab.rsplit(" ", 1)
    return fam, ro


def load_e2e183(cfg, rec):
    """The 183 e2e per-draw npz (matched design: sim-honest, honest B and honest C on M). Gate: draw 0 of every arm used equals the record."""
    import numpy as np

    p = Path("/sol") / cfg["npz"]["path"]
    b = p.read_bytes()
    assert hashlib.sha256(b).hexdigest() == cfg["npz"]["sha256"], "npz sha256"
    z = np.load(p)
    cat, arms, labs, pts = z["catches_e2e"], list(z["arms_e2e"]), list(z["row_labels"]), list(z["points"])
    ai = {k: arms.index(v) for k, v in cfg["arms"].items()}
    out, gates = {}, []
    for pt in cfg["points"]:
        k = pts.index(pt)
        for i, lab in enumerate(labs):
            m = rec["rows"][lab]["unit"][pt]["matched"]
            c = {a: cat[i, k, ai[a]] for a in ai}
            for a, rk in cfg["record_catch_keys"].items():
                ok = c[a][0] == m["catch"][rk]["point"]
                gates.append({"row": lab, "point": pt, "arm": a, "record_key": f"rows/{lab}/unit/{pt}/matched/catch/{rk}/point", "pass": bool(ok)})
            rd = bool(m["readable_contrast_e2e"]) and bool(m["readable_contrast_yardstick"])
            out.setdefault(pt, {})[lab] = {"d1": c["sim"] - c["run1"], "d2": c["sim"] - c["run2"], "y": c["yard"] - c["run1"],
                                           "draw0_catch": {a: float(c[a][0]) for a in c}, "readable": rd,
                                           "readable_npz": bool(z["readable_e2e_both_sides"][i, k]),
                                           "saturated": bool(max(float(c[a][0]) for a in c) >= cfg["saturation"])}
    return out, gates, {"npz_sha256_checked": cfg["npz"]["sha256"], "draws": int(cat.shape[-1]) - 1}


def stats(rows, members, sat_ok=True):
    """members: row labels; returns per-draw mean G, mean Y, difference summaries (points x 100)."""
    import numpy as np

    from probe_shift.fpr_core import summ

    use = [lb for lb in members if rows[lb]["readable"] and (sat_ok or not rows[lb]["saturated"])]
    if not use:
        return {"rows": [], "unavailable": "no readable rows"}
    G = np.mean([0.5 * np.abs(rows[lb]["d1"]) + 0.5 * np.abs(rows[lb]["d2"]) for lb in use], 0)
    Y = np.mean([np.abs(rows[lb]["y"]) for lb in use], 0)
    pc = lambda s: {**s, "point": 100 * s["point"], **({"ci95": [100 * s["ci95"][0], 100 * s["ci95"][1]]} if s.get("ci95") else {})}
    return {"rows": use, "n": len(use), "mean_G": pc(summ(G)), "mean_Y": pc(summ(Y)), "mean_G_minus_mean_Y": pc(summ(G - Y)), "_G": G, "_Y": Y}


def readout_contrast(rows, fams, sat_ok=True):
    """(mean G_R - mean Y_R) - (mean G_C - mean Y_C) over the families readable at both readouts."""
    from probe_shift.fpr_core import summ

    ok = [f for f in fams if all(f"{f} {ro}" in rows and rows[f"{f} {ro}"]["readable"] and (sat_ok or not rows[f"{f} {ro}"]["saturated"]) for ro in "RC")]
    if not ok:
        return {"families": [], "unavailable": "no family readable at both readouts"}
    sR, sC = stats(rows, [f"{f} R" for f in ok]), stats(rows, [f"{f} C" for f in ok])
    d = (sR["_G"] - sR["_Y"]) - (sC["_G"] - sC["_Y"])
    s = summ(d)
    return {"families": ok, "n_families": len(ok), "reply_excess_minus_command_excess": {**s, "point": 100 * s["point"],
            **({"ci95": [100 * s["ci95"][0], 100 * s["ci95"][1]]} if s.get("ci95") else {})}}


def strip(o):
    return {k: v for k, v in o.items() if not k.startswith("_")}


@app.function(image=image, cpu=(8, 8), memory=(32768, 32768), timeout=1800, retries=0, scaledown_window=2, volumes={"/sol": modal.Volume.from_name(VOL, version=2)})
def panel(cfg: dict, rec_bytes: bytes) -> dict:
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from probe_shift.fpr_core import summ

    rec = json.loads(rec_bytes)
    data, gates, src = {"e2e183": load_e2e183}[cfg["source"]](cfg, rec)
    bad = [g for g in gates if not g["pass"]]
    if bad:
        return {"GATE_FAIL": bad[:20], "n_fail": len(bad)}
    res = {}
    for pt, rows in data.items():
        o = {"rows": {}, "groups": {}, "stress": {}, "readout_contrast": {}}
        for lb, r in rows.items():
            G, Y = 0.5 * np.abs(r["d1"]) + 0.5 * np.abs(r["d2"]), np.abs(r["y"])
            pc = lambda s: {**s, "point": 100 * s["point"], **({"ci95": [100 * s["ci95"][0], 100 * s["ci95"][1]]} if s.get("ci95") else {})}
            o["rows"][lb] = {"G": pc(summ(G)), "Y": pc(summ(Y)), "G_minus_Y": pc(summ(G - Y)), "sim_minus_run1": pc(summ(r["d1"])), "sim_minus_run2": pc(summ(r["d2"])),
                             "yardstick_signed": pc(summ(r["y"])), "draw0_catch": r["draw0_catch"], "readable": r["readable"], "readable_npz": r.get("readable_npz"),
                             "saturated": r["saturated"]}
        for gname, fams in GROUPS.items():
            for sel, ros in (("R", "R"), ("C", "C"), ("R+C", "RC")):
                mem = [lb for lb in rows if split(lb)[0] in fams and split(lb)[1] in ros]
                if not mem:
                    continue
                o["groups"][f"{gname}|{sel}"] = {"primary": strip(stats(rows, mem)), "without_saturated": strip(stats(rows, mem, sat_ok=False)),
                                                  "unreadable_rows": [lb for lb in mem if not rows[lb]["readable"]], "saturated_rows": [lb for lb in mem if rows[lb]["saturated"]]}
        for sel, ros in (("R", "R"), ("C", "C"), ("R+C", "RC")):
            mem = [lb for lb in rows if split(lb)[0] in STRESS and split(lb)[1] in ros]
            if mem:
                o["stress"][f"c_asis|{sel}"] = strip(stats(rows, mem))
        for gname in ("overall_without_DT", "on_policy_own_family", "off_policy"):
            o["readout_contrast"][gname] = {"primary": readout_contrast(rows, GROUPS[gname]), "without_saturated": readout_contrast(rows, GROUPS[gname], sat_ok=False)}
        res[pt] = o
    # ---- figure: lead point; bars of mean G vs mean Y by group x readout with 95% intervals, plus the paired-difference panel ----
    lead = res[cfg["points"][0]]
    cats = [(g, s) for g in ("on_policy_own_family", "off_policy", "DT", "overall") for s in ("R", "C", "R+C") if f"{g}|{s}" in lead["groups"] and not (g == "DT" and s == "R+C")]
    nice = {"on_policy_own_family": "on-policy\nown family", "off_policy": "off-policy\nSol (a)+public", "DT": "DT", "overall": "overall"}
    rdn = {"R": "reply", "C": "command", "R+C": "both"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(17, 6.6), gridspec_kw={"width_ratios": [1.5, 1]})
    x = np.arange(len(cats))
    for j, (key, col, nm) in enumerate((("mean_G", "#3b6fb6", "mean noise-matched sim-vs-real gap G"), ("mean_Y", "#e08a2c", "mean two-real-pipeline yardstick |Y|"))):
        v = [lead["groups"][f"{g}|{s}"]["primary"][key] for g, s in cats]
        pt_ = np.array([q["point"] for q in v])
        lo = np.array([q["point"] - q["ci95"][0] if q.get("ci95") else 0 for q in v])
        hi = np.array([q["ci95"][1] - q["point"] if q.get("ci95") else 0 for q in v])
        a1.bar(x + (j - 0.5) * 0.38, pt_, 0.38, yerr=[lo, hi], color=col, label=nm, capsize=3)
        for xi, p_, h_ in zip(x + (j - 0.5) * 0.38, pt_, hi):
            a1.text(xi, p_ + h_ + 0.2, f"{p_:.1f}", ha="center", va="bottom", fontsize=7.5)
    a1.set_xticks(x, [f"{nice[g]}\n{rdn[s]}\nn={lead['groups'][f'{g}|{s}']['primary']['n']}" for g, s in cats], fontsize=7)
    a1.set_ylabel("catch-rate gap, percentage points (lower = closer)")
    a1.legend(fontsize=8, loc="upper left")
    dv = [lead["groups"][f"{g}|{s}"]["primary"]["mean_G_minus_mean_Y"] for g, s in cats]
    dp = np.array([q["point"] for q in dv])
    a2.errorbar(dp, x, xerr=[[q["point"] - q["ci95"][0] if q.get("ci95") else 0 for q in dv], [q["ci95"][1] - q["point"] if q.get("ci95") else 0 for q in dv]],
                fmt="o", color="#333333", capsize=3)
    for xi, q in zip(x, dv):
        a2.text(1.02, xi, f"{q['point']:+.1f} [{q['ci95'][0]:+.1f}, {q['ci95'][1]:+.1f}]" if q.get("ci95") else f"{q['point']:+.1f} (interval withheld)",
                va="center", fontsize=7.5, transform=a2.get_yaxis_transform())
    a2.axvline(0, color="#888888", lw=1)
    a2.set_yticks(x, [f"{nice[g].replace(chr(10), ' ')} | {rdn[s]}" for g, s in cats], fontsize=7.5)
    a2.invert_yaxis()
    a2.set_xlabel("mean G - mean Y, points (> 0: sim-vs-real gaps bigger than real-vs-real)")
    ov = lead["groups"]["overall|R+C"]["primary"]["mean_G_minus_mean_Y"]
    title = (f"{cfg['set_name']}, per-action FPR {100 * cfg['point_values'][0]:g}%: per-probe sim-vs-real gap vs two-real-pipeline gap. "
             f"Overall excess {ov['point']:+.1f} points [{ov['ci95'][0]:+.1f}, {ov['ci95'][1]:+.1f}]"
             + (" (interval includes 0)" if ov["ci95"][0] <= 0 <= ov["ci95"][1] else " (interval excludes 0)"))
    fig.suptitle(title + "\nPOST HOC; PRELIMINARY. Bars: 95% joint-bootstrap intervals (tasks resampled, thresholds re-estimated per draw); launch variance not included.\nResampled |gaps| sit above their draw-0 value (absolute values push draws away from 0), so the G and Y intervals skew up; read the paired difference (right)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    return {"res": res, "gates": {"n": len(gates), "n_fail": 0, "all_pass": True, "checked": "draw 0 of sim, run1, run2, yardstick arm equals the record, exactly"},
            "source": src, "title": title, "png": buf.getvalue()}


def captions(rec):
    """Captions by key, every number read from the record."""
    o = {}
    for pt, r in rec["results"].items():
        for gk, g in r["groups"].items():
            p = g["primary"]
            if "mean_G" not in p:
                continue
            d = p["mean_G_minus_mean_Y"]
            ci = f"[{d['ci95'][0]:+.1f}, {d['ci95'][1]:+.1f}]" if d.get("ci95") else "(interval withheld)"
            o[f"results/{pt}/groups/{gk}/primary"] = (f"{gk.replace('|', ', ')} at {pt}: mean G {p['mean_G']['point']:.1f} points, mean |Y| {p['mean_Y']['point']:.1f}, "
                                                      f"difference {d['point']:+.1f} {ci} over {p['n']} rows ({', '.join(p['rows'])})")
        for gk, g in r["readout_contrast"].items():
            q = g["primary"]
            if "reply_excess_minus_command_excess" in q:
                d = q["reply_excess_minus_command_excess"]
                ci = f"[{d['ci95'][0]:+.1f}, {d['ci95'][1]:+.1f}]" if d.get("ci95") else "(interval withheld)"
                o[f"results/{pt}/readout_contrast/{gk}/primary"] = f"{gk} at {pt}: reply excess minus command excess {d['point']:+.1f} {ci} over {q['n_families']} families"
    return o


@app.local_entrypoint()
def main(cfg: str = "s_old_183_matched"):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("uncommitted tracked changes")
    c = json.loads((HERE / "config" / f"{cfg}.json").read_text())
    rb = git("show", c["record"]["git"])
    if hashlib.sha256(rb).hexdigest() != c["record"]["sha256"]:
        raise SystemExit("record sha256")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = panel.remote(c, rb)
    if "GATE_FAIL" in out:
        print(json.dumps(out)[:4000])
        raise SystemExit("draw-0 gate failed")
    rec = {"label": LABEL, "set_label": c["set_label"],
           "meta": {"git": git("rev-parse", "HEAD").decode().strip(), "stamp_pt": stamp, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "config": c, "record_sha256_checked": c["record"]["sha256"], "rule": "experiments/2026-10-15_panel_noise_matched/RULE.md @ 40b9db23d"},
           "definitions": {"G": "0.5|sim@tau_sim - run1@tau_B| + 0.5|sim@tau_sim - run2@tau_B| (points)", "Y": "|yardstick| = |run2@tau_C - run1@tau_B| (points)",
                           "group": "per-draw mean over readable member rows; NaN if any member is NaN in a draw; summ: point = draw 0, 95% percentile interval over draws 1..D",
                           "saturated": f"any arm >= {c['saturation']} at draw 0"},
           "gates": out["gates"], "source": out["source"], "title": out["title"], "results": out["res"]}
    rec["captions"] = captions(rec)
    d = HERE / "results"
    blob = json.dumps(rec, default=float, indent=1).encode()
    (d / f"panel_{cfg}_{stamp}.json").open("xb").write(blob)
    (d / f"panel_{cfg}_{stamp}.png").open("xb").write(out["png"])
    (d / f"panel_{cfg}_{stamp}_captions.json").open("xb").write(json.dumps({"record": f"panel_{cfg}_{stamp}.json", "record_sha256": hashlib.sha256(blob).hexdigest(),
                                                                            "title": out["title"], "captions": rec["captions"]}, indent=1).encode())
    print(json.dumps({"record": f"panel_{cfg}_{stamp}.json", "sha256": hashlib.sha256(blob).hexdigest(), "title": out["title"]}))
