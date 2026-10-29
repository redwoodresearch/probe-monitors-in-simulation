"""Read-only lookup, on route C's final runs, of simulated catch against the REAL threshold (real_line) next to the separate-threshold
gap (delta_full), and of the two thresholds, for the six headline probes and the App D extra probe (c_asis_uncut), C and R, 1% and 5%.

Question (Writing for Aswin, 10-25): is the slightly lower simulated catch caused by the simulated threshold sitting higher, rather than
by simulated attacks scoring lower? No recompute of any catch, threshold or interval: every one is a stored value.
  - Record A (route C pooled record, in v2.7's in-scope list; committed, sha-pinned): pooled[label][pooled_all_origins][joint][lvl]
    .gap.{real_line, delta_full} (point + stored ci95) and .catch.{sim, real_mean}, .readable_real_line.
  - A18 clean-room values (cleanroom Volume, sha-pinned): per label and level tau_B, tau_S, tpr, tpr_sim_at_tauB, real_mean,
    real_line, delta_full, and the per-fold moments of each side's honest pool.
Derived here (arithmetic on stored values only): six-probe means of the stored points; real_line - delta_full; tau_S - tau_B in raw
score units for the raw-score probes; for the fold-standardized probes, tau_S mapped through each fold's moments into real-honest SD
units minus tau_B (folds 0-4; fold -1 not mapped).
Mapping control (STOP if it fails): c_asis_uncut at joint 1% must give App D's printed points and intervals (draft v88 L260, FINAL).
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

APP = "probe-shift_aswin-surya_sub-realline_finalread_kill-2026-11-25"
CLEANROOM_VOL = "probe-shift_aswin-surya_sub-repro_cleanroom_kill-2026-11-16"
A18_FILE = "cleanroom_values_A18_2026-10-20_040452.json"
A18_SHA = "9a6c50f41e1e394f13f1cf0feda7ac5d2b2a8bcea6a0d0c2154df0f12de1bbe2"  # cleanroom_A18_compute_2026-10-20_040452.json @ 376586003
REC_A = ("experiments/2026-10-15_fpr_sweep_507_prep/results/routec_records/"
         "routec_A_alltasks_CRD_FULL_r2_allorigins_pooledworld_scored_e12_2026-10-19_180327.json.gz")
REC_A_SHA = "191c467ab095b0f9ddfe2271822655bc5b49ee02a1a2c5a5b0e11bc855a00865"  # v2.7 in_scope_records A .sha256
REC_A_DSHA = "54afad04422afd288db9b77043713cfdb708adf5c0eaa74499ea3e5dd600082a"  # v2.7 in_scope_records A .decompressed_sha256
SIX = ["sol_a_uncut", "FIBS|uncut", "harmfulness|uncut", "refusal|uncut", "high-stakes|uncut", "own_all_rs_uncut_L48"]
EXTRA = "c_asis_uncut"
RAW = {"FIBS|uncut", "harmfulness|uncut", "refusal|uncut", "high-stakes|uncut"}  # the four p2 probes: raw scores (SPEC_READING §2)
LEVELS = ["0.01", "0.05"]
COMBO, MODE = "pooled_all_origins", "joint"
# App D (draft v88 L260), points and 95% intervals in points, 1 dp: (gap, readout) -> (point, lo, hi)
APP_D = {("real_line", "R"): (5.6, 3.9, 7.6), ("delta_full", "R"): (-0.4, -3.3, 2.4),
         ("real_line", "C"): (3.3, 0.4, 7.3), ("delta_full", "C"): (-2.3, -9.2, 7.2)}

HERE = Path(__file__).resolve()
# The container imports this file as /root/run_realline_read.py, which has no parents[2]; the local file is only read at launch.
REPO = HERE.parents[2] if len(HERE.parents) > 2 else HERE.parent

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").add_local_file(str(REPO / REC_A), "/in/routec_A.json.gz")
vol = modal.Volume.from_name(CLEANROOM_VOL, version=2)


def lvl_key(d, lvl):
    """The stored level key equal to lvl as a number (fail loudly if absent or ambiguous)."""
    ks = [k for k in d if float(k) == float(lvl)]
    assert len(ks) == 1, f"level {lvl}: keys {sorted(d)}"
    return ks[0]


@app.function(image=image, volumes={"/o": vol.read_only()}, cpu=(1.0, 1.0), memory=(2048, 2048), timeout=300,
              startup_timeout=300, scaledown_window=10, retries=0)
def read() -> dict:
    import gzip

    raw = Path("/o", A18_FILE).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == A18_SHA, "A18 values file sha256 differs from the pin"
    a18 = json.loads(raw)["values"]
    gz = Path("/in/routec_A.json.gz").read_bytes()
    assert hashlib.sha256(gz).hexdigest() == REC_A_SHA, "record A sha256 differs from the v2.7 pin"
    dec = gzip.decompress(gz)
    assert hashlib.sha256(dec).hexdigest() == REC_A_DSHA, "record A decompressed sha256 differs from the v2.7 pin"
    rec = json.loads(dec)["pooled"]

    def summ(b, what, readable):
        """A stored summary in points. The producer's rule for a level that is not readable: print 'not readable', never the
        interval (run_peraction311.py @ a936a5d97 L3381), so its interval is not returned and the row is flagged."""
        assert isinstance(b, dict) and isinstance(b.get("point"), (int, float)) and isinstance(b.get("ci95"), list), what
        assert "withheld" not in b, f"{what}: withheld"
        assert isinstance(readable, bool), f"{what}: readability flag {readable!r}"
        return {"point": 100 * b["point"], "ci95": [100 * x for x in b["ci95"]] if readable else "not readable",
                "readable": readable, "finite_draws": b.get("finite_draws"), "draws": b.get("draws")}

    rows = {}
    for name in SIX + [EXTRA]:
        for ro in ("C", "R"):
            lab = f"{name} {ro}"
            va, ra = a18[lab], rec[lab][COMBO][MODE]
            for lvl in LEVELS:
                q, d = ra[lvl_key(ra, lvl)], va["levels"][lvl_key(va["levels"], lvl)]
                r = {"A": {"real_line": summ(q["gap"]["real_line"], f"{lab} {lvl} real_line", q.get("readable_real_line")),
                           "delta_full": summ(q["gap"]["delta_full"], f"{lab} {lvl} delta_full", q["readable"].get("all")),
                           "catch_sim": 100 * q["catch"]["sim"]["point"], "catch_real_mean": 100 * q["catch"]["real_mean"]["point"]},
                     "A18": {"real_line": 100 * d["real_line"], "delta_full": 100 * d["delta_full"],
                             "tpr_sim_at_tauS": 100 * d["tpr"]["sim"], "tpr_sim_at_tauB": 100 * d["tpr_sim_at_tauB"],
                             "real_mean": 100 * d["real_mean"], "tau_B": d["tau_B"], "tau_S": d["tau_S"],
                             "n_runs_B": va["n_runs_B"], "n_runs_S": va["n_runs_S"]}}
                r["A18"]["real_line_minus_delta_full"] = r["A18"]["real_line"] - r["A18"]["delta_full"]
                r["A_real_line_minus_delta_full_point"] = r["A"]["real_line"]["point"] - r["A"]["delta_full"]["point"]
                r["abs_diff_real_line_A_vs_A18"] = abs(r["A"]["real_line"]["point"] - r["A18"]["real_line"])
                r["abs_diff_delta_full_A_vs_A18"] = abs(r["A"]["delta_full"]["point"] - r["A18"]["delta_full"])
                tB, tS = d["tau_B"]["T"], d["tau_S"]["T"]
                if name in RAW:
                    assert va.get("moments") is None, f"{lab}: raw-score probe has moments"
                    r["threshold"] = {"units": "raw score", "tau_S_minus_tau_B": tS - tB}
                else:
                    m = va["moments"]
                    per = {}
                    for k in range(5):
                        mb, ms = m["B"]["fold"].get(str(k)), m["S"]["fold"].get(str(k))
                        if mb is None or ms is None:
                            continue
                        raw_s = ms[0] + tS * ms[1]  # the sim threshold in this fold's raw score units
                        per[str(k)] = (raw_s - mb[0]) / mb[1] - tB  # in SDs of this fold's real honest units, minus tau_B
                    r["threshold"] = {"units": "real-honest SD of each fold (folds 0-4 where both sides store moments; fold -1 not mapped)",
                                      "tau_S_minus_tau_B_by_fold": per, "n_folds_mapped": len(per),
                                      "min": min(per.values()) if per else None, "max": max(per.values()) if per else None}
                rows[f"{lab}|{lvl}"] = r

    # mapping control: App D's printed numbers from record A at joint 1%
    ctl = {}
    for (gap, ro), (p, lo, hi) in APP_D.items():
        b = rows[f"{EXTRA} {ro}|0.01"]["A"][gap]
        got = (round(b["point"], 1), round(b["ci95"][0], 1), round(b["ci95"][1], 1)) if b["readable"] else "not readable"
        ctl[f"{gap} {ro}"] = {"expected": [p, lo, hi], "got": list(got) if b["readable"] else got,
                              "ok": b["readable"] and got == (p, lo, hi)}
    if not all(v["ok"] for v in ctl.values()):
        raise SystemExit("REALLINE_READ_STOP mapping control failed: " + json.dumps({k: v["ok"] for k, v in ctl.items()}))

    six = {}
    for ro in ("C", "R"):
        for lvl in LEVELS:
            rs = [rows[f"{n} {ro}|{lvl}"] for n in SIX]
            six[f"{ro}|{lvl}"] = {
                "mean_A_real_line_point": sum(x["A"]["real_line"]["point"] for x in rs) / 6,
                "mean_A_delta_full_point": sum(x["A"]["delta_full"]["point"] for x in rs) / 6,
                "mean_A18_real_line": sum(x["A18"]["real_line"] for x in rs) / 6,
                "mean_A18_delta_full": sum(x["A18"]["delta_full"] for x in rs) / 6,
                "n_probes_tau_S_above_tau_B_raw": sum(1 for n, x in zip(SIX, rs) if n in RAW and x["threshold"]["tau_S_minus_tau_B"] > 0),
                "n_probes_tau_S_above_tau_B_every_mapped_fold": sum(1 for n, x in zip(SIX, rs) if n not in RAW and x["threshold"]["min"] is not None
                                                                    and x["threshold"]["min"] > 0),
                "n_probes_readable": {g: sum(1 for x in rs if x["A"][g]["readable"]) for g in ("real_line", "delta_full")},
                "n_probes_real_line_ci_includes_0_of_readable": sum(1 for x in rs if x["A"]["real_line"]["readable"]
                                                                    and x["A"]["real_line"]["ci95"][0] <= 0 <= x["A"]["real_line"]["ci95"][1]),
                "interval_for_six_probe_mean": "not stored (needs the per-draw arrays)"}
    return {"rows": rows, "six_probe": six, "mapping_control": ctl, "units": "catch and gaps in points (x100)",
            "pins": {"A18": [A18_FILE, A18_SHA], "A": [REC_A, REC_A_SHA, REC_A_DSHA]}, "combo": COMBO, "mode": MODE}


@app.local_entrypoint()
def main(out_dir: str):
    res = read.remote()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = Path(out_dir) / f"realline_final_read_{stamp}.json"
    with out.open("x") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print("REALLINE_READ_OK", out, hashlib.sha256(out.read_bytes()).hexdigest(),
          "control_ok", all(v["ok"] for v in res["mapping_control"].values()), "rows", len(res["rows"]))
