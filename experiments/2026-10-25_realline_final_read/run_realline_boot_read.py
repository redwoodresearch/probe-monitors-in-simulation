"""Six-probe intervals for the real-threshold gap (real_line) and the threshold's contribution (delta_full - real_line), from route C
record A's STORED per-draw arrays. No new resampling: the record's npz holds every draw (draw 0 = the point estimate, draws 1.. = the
main-task resamples, joint mode, shared by all probes), so these use the same draws and seeds as the post's [^ci] intervals.

Method (as run_group_catch.py / record A's co-headline S): per probe, the pooled per-draw arrays gap_delta_full and gap_real_line
(npzkey(label, pooled_all_origins, joint, level, "pooled", "gap_<k>")); the six-probe mean = np.mean over the probes draw by draw, in the
record's co-headline row order; summaries with the record's summ (point = draw 0; 2.5 / 97.5 np.percentile of the finite resampled
draws; withheld below 95% complete).

Gates (any failure STOPs before a new value is written):
  G1  record A gz / decompressed sha256 = v2.7 pins; the npz sha256 = the record's npz.sha256.
  G2  per probe x readout x level: summ(stored array) == the record's gap node (point, ci95, draws, finite_draws, withheld), exactly,
      for BOTH delta_full and real_line.
  G3  the six-probe mean of the delta_full arrays, summarised, == the record's coheadline_signed[...]["headline_six_full|<ro>"]["S"]
      exactly, at each level; and at 1% it rounds (1 dp, points) to the post's six-probe separate intervals: commands -1.0 [-3.0, +1.0],
      tool outputs -1.3 [-3.8, +1.1].
New outputs (computed only after every gate passes): the six-probe real_line mean, and the contribution (delta_full - real_line) per
probe and six-probe mean, at 1% and 5%, C and R. A contribution is "not readable" unless both gaps are readable (producer rule: an
unreadable level prints 'not readable', never the interval).
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

APP = "probe-shift_aswin-surya_sub-realline_bootread_kill-2026-11-25"
SOLAVG_VOL = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13"  # record A's npz (record["npz"]["volume"]), read only
REC_A = ("experiments/2026-10-15_fpr_sweep_507_prep/results/routec_records/"
         "routec_A_alltasks_CRD_FULL_r2_allorigins_pooledworld_scored_e12_2026-10-19_180327.json.gz")
REC_A_SHA = "191c467ab095b0f9ddfe2271822655bc5b49ee02a1a2c5a5b0e11bc855a00865"  # v2.7 in_scope_records A .sha256
REC_A_DSHA = "54afad04422afd288db9b77043713cfdb708adf5c0eaa74499ea3e5dd600082a"  # v2.7 in_scope_records A .decompressed_sha256
SIX = ["sol_a_uncut", "FIBS|uncut", "harmfulness|uncut", "refusal|uncut", "high-stakes|uncut", "own_all_rs_uncut_L48"]
LEVELS = ["0.01", "0.05"]
LEAD, MODE, GROUP = "pooled_all_origins", "joint", "headline_six_full"
POST_SIX_SEP_1PCT = {"C": (-1.0, -3.0, 1.0), "R": (-1.3, -3.8, 1.1)}  # draft v88, six-probe separate-threshold mean and 95% interval
KEYS = ("point", "ci95", "draws", "finite_draws", "withheld")

HERE = Path(__file__).resolve()
# The container imports this file as /root/run_realline_boot_read.py, which has no parents[2]; the local file is only read at launch.
REPO = HERE.parents[2] if len(HERE.parents) > 2 else HERE.parent

app = modal.App(APP)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")  # record A's producer pin
         .add_local_file(str(REPO / REC_A), "/in/routec_A.json.gz"))
vol = modal.Volume.from_name(SOLAVG_VOL, version=2)


def npzkey(*parts):  # run_peraction311.npzkey @ 20c6fbd20, verbatim rule (as run_group_catch.py)
    out = []
    for p_ in parts:
        q = str(p_).replace("@", "_at_").replace("|", "_").replace(" ", "_").replace("(", "").replace(")", "")
        if "__" in q or not q:
            raise SystemExit(f"npz key part {p_!r} would be ambiguous")
        out.append(q)
    return "__".join(out)


def lvl_key(d, lvl):
    ks = [k for k in d if float(k) == float(lvl)]
    assert len(ks) == 1, f"level {lvl}: keys {sorted(d)}"
    return ks[0]


@app.function(image=image, volumes={"/sol": vol.read_only()}, cpu=(2.0, 2.0), memory=(8192, 8192), timeout=900,
              startup_timeout=300, scaledown_window=10, retries=0)
def read() -> dict:
    import gzip

    import numpy as np

    def summ(v, D):  # run_fprsweep_507.summ @ a936a5d97 (D = the record's draw count)
        f = v[1:][np.isfinite(v[1:])]
        o = {"point": float(v[0]), "finite_draws": int(len(f)), "draws": D}
        if len(f) < 0.95 * D:
            return o | {"ci95": None, "withheld": "fewer than 95% complete draws"}
        return o | {"ci95": [float(np.percentile(f, 2.5)), float(np.percentile(f, 97.5))], "degenerate": bool(np.ptp(f) == 0)}

    def same(a, b):
        return all(a.get(k) == b.get(k) for k in KEYS)

    def pts(s):  # a summary in points, for the output
        return {"point": 100 * s["point"], "ci95": [100 * x for x in s["ci95"]] if s.get("ci95") else s.get("withheld"),
                "draws": s["draws"], "finite_draws": s["finite_draws"]}

    # G1
    gz = Path("/in/routec_A.json.gz").read_bytes()
    assert hashlib.sha256(gz).hexdigest() == REC_A_SHA, "STOP G1: record A sha256 differs from the v2.7 pin"
    dec = gzip.decompress(gz)
    assert hashlib.sha256(dec).hexdigest() == REC_A_DSHA, "STOP G1: record A decompressed sha256 differs from the v2.7 pin"
    rec = json.loads(dec)
    npz = rec["npz"]
    assert npz["volume"] == SOLAVG_VOL, f"STOP G1: record A's npz is on {npz['volume']}"
    p_ = npz["path"]
    assert isinstance(p_, str) and not p_.startswith("/") and ".." not in Path(p_).parts, "STOP G1: npz path"
    h = hashlib.sha256()
    with open(Path("/sol", p_), "rb") as fh:  # streamed: only the selected arrays are held in memory
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    assert h.hexdigest() == npz["sha256"], "STOP G1: npz sha256 differs from record A's npz.sha256"
    Z = np.load(Path("/sol", p_))  # lazy NpzFile: members are read on access

    gates, arr, rd, D_ = {}, {}, {}, {}
    for ro in ("C", "R"):
        for lvl in LEVELS:
            for name in SIX:  # G2
                lab = f"{name} {ro}"
                node = rec["pooled"][lab][LEAD][MODE]
                F = lvl_key(node, lvl)
                q = node[F]
                D = q["gap"]["delta_full"]["draws"]
                D_.setdefault((ro, lvl), D)
                assert D_[(ro, lvl)] == D, f"STOP: {lab} {lvl} draws {D} differ from the other probes'"
                for gname in ("delta_full", "real_line"):
                    a = np.asarray(Z[npzkey(lab, LEAD, MODE, F, "pooled", "gap_" + gname)], dtype=np.float64)
                    assert q["gap"][gname]["draws"] == D and a.shape == (D + 1,), f"STOP: {lab} {lvl} {gname}: shape {a.shape}, draws {D}"
                    arr[(lab, lvl, gname)] = a
                    gates[f"G2|{lab}|{lvl}|{gname}"] = same(summ(a, D), q["gap"][gname])
                rd[(lab, lvl)] = {"delta_full": q["readable"]["all"] is True, "real_line": q.get("readable_real_line") is True}
            # G3
            cn = rec["coheadline_signed"][LEAD][MODE]
            g = cn[lvl_key(cn, lvl)][f"{GROUP}|{ro}"]
            rows = g["rows"]
            assert sorted(rows) == sorted(f"{n} {ro}" for n in SIX), f"STOP G3: co-headline rows {rows}"
            sep6 = summ(np.mean([arr[(lb, lvl, "delta_full")] for lb in rows], 0), D_[(ro, lvl)])
            gates[f"G3|{ro}|{lvl}|six_sep_equals_record_S"] = same(sep6, g["S"])
            if lvl == "0.01":
                p, lo, hi = POST_SIX_SEP_1PCT[ro]
                got = (round(100 * sep6["point"], 1), round(100 * sep6["ci95"][0], 1), round(100 * sep6["ci95"][1], 1))
                gates[f"G3|{ro}|{lvl}|six_sep_equals_post"] = got == (p, lo, hi)
    if not all(gates.values()):
        raise SystemExit("REALLINE_BOOT_STOP gates failed: " + json.dumps(sorted(k for k, v in gates.items() if not v)))

    out = {"per_probe": {}, "six_probe": {}}
    for ro in ("C", "R"):
        for lvl in LEVELS:
            D = D_[(ro, lvl)]
            rows = rec["coheadline_signed"][LEAD][MODE][lvl_key(rec["coheadline_signed"][LEAD][MODE], lvl)][f"{GROUP}|{ro}"]["rows"]
            for lab in rows:
                c = arr[(lab, lvl, "delta_full")] - arr[(lab, lvl, "real_line")]
                ok = rd[(lab, lvl)]["delta_full"] and rd[(lab, lvl)]["real_line"]
                s = pts(summ(c, D))
                out["per_probe"][f"{lab}|{lvl}"] = {"contribution_sep_minus_real": s if ok else {"point": s["point"], "ci95": "not readable"},
                                                    "readable": ok}
            all_rd = all(rd[(lb, lvl)]["delta_full"] and rd[(lb, lvl)]["real_line"] for lb in rows)
            rl6 = pts(summ(np.mean([arr[(lb, lvl, "real_line")] for lb in rows], 0), D))
            sep6 = pts(summ(np.mean([arr[(lb, lvl, "delta_full")] for lb in rows], 0), D))
            con6 = pts(summ(np.mean([arr[(lb, lvl, "delta_full")] - arr[(lb, lvl, "real_line")] for lb in rows], 0), D))
            out["six_probe"][f"{ro}|{lvl}"] = {"rows": rows, "all_readable": all_rd, "sep_mean": sep6,
                                               "real_line_mean": rl6 if all_rd else {"point": rl6["point"], "ci95": "not readable"},
                                               "contribution_sep_minus_real_mean": con6 if all_rd else {"point": con6["point"], "ci95": "not readable"}}
    return {"gates": gates, "n_gates": len(gates), "results": out, "units": "points (x100)", "combo": LEAD, "mode": MODE,
            "method": "stored per-draw arrays of record A; np.mean over probes draw by draw; record's summ",
            "pins": {"A": [REC_A, REC_A_SHA, REC_A_DSHA], "npz": [SOLAVG_VOL, p_, npz["sha256"]]}}


@app.local_entrypoint()
def main(out_dir: str):
    res = read.remote()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = Path(out_dir) / f"realline_boot_read_{stamp}.json"
    with out.open("x") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print("REALLINE_BOOT_OK", out, hashlib.sha256(out.read_bytes()).hexdigest(), "gates", res["n_gates"], "all_pass", all(res["gates"].values()))
