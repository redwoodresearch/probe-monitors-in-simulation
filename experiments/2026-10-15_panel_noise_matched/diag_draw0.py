"""Gate diagnostic (2e 10-16, GO on (b)): the per-set full-length PANEL failed its per-origin draw-0 catch gate (npz per-origin array[0] == the record's
rows[L][o].points.<mode>.<f>.catch.<arm>.point, EXACT). This repeats that comparison for EVERY gate it makes, exactly as run_pooled.py does (the label's
combos from the record's members, headline combo first; each origin read once from the first combo holding it; arms sim@S|primary, run1@B, run2@B,
run2@A, plus run1@B2 where the npz has it) and reports, per failing gate, row / f / origin / arm and |npz draw0 - record point|, then per row the max and
min of that magnitude, the failing-row set, and whether any failing row is outside the fold -1 families. Magnitudes only (a gate diagnostic, not a
result); no catch value is printed or written. Modal CPU; the record and config by path@sha from git, the npz from the Volume by sha.

  PYTHONPATH=src modal run --detach experiments/2026-10-15_panel_noise_matched/diag_draw0.py --cfg FINAL_full_2026-10-16_1434
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
VOL = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13"
APP = "probe-shift_aswin-surya_sub-panelnm_kill-2026-11-14"
ARMS = ("sim@S|primary", "run1@B", "run2@B", "run2@A")
app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3").add_local_python_source("probe_shift")
SOL = modal.Volume.from_name(VOL, version=2)


def san2(x):
    return x.replace("@", "_at_").replace("|", "_").replace(" ", "_").replace("(", "").replace(")", "")


def key(lab, cb, mode, f, o, c):  # run_pooled.keys_for, key format v2c (per-origin catch / D)
    return f"{san2(lab)}__{san2(cb)}__{mode}__{f}__{san2(o)}__" + ("D" if c == "D" else f"catch_{san2(c)}")


def dg(d, *path):  # null-safe nested get: None as soon as a level is missing or not a dict
    for p_ in path:
        if not isinstance(d, dict):
            return None
        d = d.get(p_)
    return d


def num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def gkey(lab, cb, mode, f, g):  # v2c pooled gap, every part sanitised as Stage A's npzkey does (a no-op for delta_full etc.; 'delta_full|B' -> 'delta_full_B')
    return f"{san2(lab)}__{san2(cb)}__{mode}__{f}__pooled__gap_{san2(g)}"


@app.function(image=image, cpu=(4, 4), memory=(32768, 32768), timeout=900, retries=0, scaledown_window=2, volumes={"/sol": SOL})
def diag(cfg: dict, rec_blob: bytes, ctx: dict) -> dict:
    import numpy as np

    from probe_shift.panel_out import save_out

    rec = json.loads(rec_blob)
    src = cfg["npz"][0]
    p = Path("/sol") / src["path"]
    if hashlib.sha256(p.read_bytes()).hexdigest() != src["sha256"]:
        raise SystemExit("STOP: npz sha256 != config")
    z = np.load(p)
    files = set(z.files)
    mode, hc, combos = cfg.get("mode", "joint"), cfg["headline_combo"], cfg["combos"]
    fails, n_cmp, n_missing, cmp_by_row = [], 0, 0, {}  # cmp_by_row: comparisons made per row (0d: a row with comparisons and no failure matches EXACTLY)
    for lab in src["labels"]:
        for f in cfg["f_keys"]:
            mem = {}
            for cb in sorted(combos, key=lambda c_: c_ != hc):  # as run_pooled: the record's members per combo, headline first
                e_ = (rec.get("pooled") or {}).get(lab, {}).get(cb) or {}
                if e_.get("members"):
                    mem[cb] = list(e_["members"])
            if not mem or gkey(lab, next(iter(mem)), mode, f, "delta_rr") not in files:  # run_pooled's guard: no origin comparison for this row and level
                fails.append({"row": lab, "f": f, "origin": None, "arm": None, "absdiff": None, "why": "no available combo, or pooled arrays absent (run_pooled skips the row)"})
                continue
            for o in [o_ for o_ in cfg["origins"] if any(o_ in m for m in mem.values())]:
                cb0 = next(cb for cb, m in mem.items() if o in m)
                arms = ARMS + (("run1@B2",) if key(lab, cb0, mode, f, o, "run1@B2") in files else ())
                for c in arms:
                    k_ = key(lab, cb0, mode, f, o, c)
                    try:
                        ref = rec["rows"][lab][o]["points"][mode][f]["catch"][c]["point"]
                    except (KeyError, TypeError):
                        ref = None
                    if k_ not in files or ref is None:
                        n_missing += 1
                        fails.append({"row": lab, "f": f, "origin": o, "arm": c, "combo_read": cb0, "absdiff": None, "why": "npz key or record point missing"})
                        continue
                    n_cmp += 1
                    cmp_by_row[lab] = cmp_by_row.get(lab, 0) + 1
                    a0 = np.asarray(z[k_])[0]  # compared as the gate does (the numpy scalar); converted only for the magnitude
                    if not bool(a0 == ref):  # the gate's exact equality (NaN on either side fails, as in run_pooled)
                        fails.append({"row": lab, "f": f, "origin": o, "arm": c, "combo_read": cb0,
                                      "absdiff": abs(float(a0) - ref) if np.isfinite(float(a0)) and np.isfinite(ref) else "non-finite"})
    fam = lambda lb: lb.rsplit(" ", 1)[0]
    fold_fams = set(cfg["drop_evidence_from_record"]["required_families"])
    agg_ = lambda v: {"n_fail": len(v), "max_absdiff": max((d for d in v if isinstance(d, float)), default=None),
                      "min_absdiff": min((d for d in v if isinstance(d, float)), default=None), "non_float": sum(not isinstance(d, float) for d in v)}
    by_rl, by_r = {}, {}
    for x in fails:
        by_rl.setdefault(f"{x['row']} | {x['f']}", []).append(x["absdiff"])  # row = '<family> <readout>': family x readout x level
        by_r.setdefault(x["row"], []).append(x["absdiff"])
    fail_rows = sorted(by_r)
    exp_rows = sorted(f"{fa} {ro}" for fa in fold_fams for ro in ("C", "R"))
    # 0d (iii): the headline aggregates at draw 0, npz vs record (rate units): per probe the union row's saved gaps vs the record's pooled gap points, and the
    # six-probe S, Y and excess (registered Y = |delta_rr|) from the npz draw 0 vs coheadline_signed S / Y and panel_groups' excess, per level and readout
    six, hg = list(cfg["coheadline"]["probes"]), "headline_six_full"
    head_ = {}
    for f in cfg["f_keys"]:
        for ro in ("C", "R"):
            labs = [f"{p_} {ro}" for p_ in six]
            g0 = {(lb, k): float(np.asarray(z[gkey(lb, hc, mode, f, k)])[0]) for lb in labs for k in ("delta_full", "sim_minus_run1", "sim_minus_run2", "delta_rr")}
            per_probe = {f"{lb} {k}": abs(g0[(lb, k)] - rec["pooled"][lb][hc][mode][f]["gap"][k]["point"]) for lb in labs for k in ("delta_full", "sim_minus_run1", "sim_minus_run2", "delta_rr")}
            S0 = sum(g0[(lb, "delta_full")] for lb in labs) / 6
            Y0 = sum(g0[(lb, "sim_minus_run1")] - g0[(lb, "sim_minus_run2")] for lb in labs) / 6
            X0 = sum(0.5 * abs(g0[(lb, "sim_minus_run1")]) + 0.5 * abs(g0[(lb, "sim_minus_run2")]) - abs(g0[(lb, "delta_rr")]) for lb in labs) / 6
            cs_ = rec["coheadline_signed"][hc][mode][f][f"{hg}|{ro}"]
            pg_ = rec["panel_groups"][hc]["pooled"][mode][f][f"{hg}|{ro}"]["primary"]["mean_G_minus_mean_Y"]
            head_[f"{f} {ro}"] = {"S_absdiff": abs(S0 - cs_["S"]["point"]), "Y_absdiff": abs(Y0 - cs_["Y"]["point"]), "excess_absdiff": abs(X0 - pg_["point"]),
                                  "per_probe_gap_max_absdiff": max(per_probe.values()), "per_probe_gap_absdiff": per_probe}
    # 0d (10-16) B: EVERY pooled gap key (each label x the record's combos with members x level x the record's gap names) draw 0 vs the record's pooled gap point
    pg_scan, pg_bad, pg_miss = {}, [], []
    for lab in src["labels"]:
        n_, bad_, mx_, mi_ = 0, 0, 0.0, 0
        for cb, e_ in sorted(((rec.get("pooled") or {}).get(lab) or {}).items()):
            if not isinstance(e_, dict) or not e_.get("members"):
                continue
            for f in cfg["f_keys"]:
                for k, s_ in (dg(e_, mode, f, "gap") or {}).items():
                    gk_ = gkey(lab, cb, mode, f, k)
                    p_ = s_.get("point") if isinstance(s_, dict) else None
                    if gk_ not in files or not num(p_):  # Codex diag r4: a missing key or point is counted and named, never silently skipped
                        mi_ += 1
                        pg_miss.append({"row": lab, "combo": cb, "f": f, "gap": k, "why": "npz key absent" if gk_ not in files else "record point missing or not a number"})
                        continue
                    n_ += 1
                    a0 = np.asarray(z[gk_])[0]
                    if not bool(a0 == p_):
                        bad_ += 1
                        d_ = abs(float(a0) - p_) if np.isfinite(float(a0)) and np.isfinite(p_) else float("inf")
                        mx_ = max(mx_, d_)
                        pg_bad.append({"row": lab, "combo": cb, "f": f, "gap": k, "absdiff": d_ if np.isfinite(d_) else "non-finite"})
        pg_scan[lab] = {"n_compared": n_, "n_mismatch": bad_, "n_missing": mi_, "max_absdiff": mx_ if bad_ else 0.0}
    # 0d (10-16) C: every pair of labels whose npz keys collide (same san2): for EACH shared key, which label's record point its draw 0 equals
    coll_ = {}
    for i_, a_ in enumerate(src["labels"]):
        for b_ in src["labels"][i_ + 1:]:
            if san2(a_) != san2(b_):
                continue
            cls_ = {"per_origin": {}, "pooled_gap": {}}
            pa_, pb_ = (rec.get("pooled") or {}).get(a_) or {}, (rec.get("pooled") or {}).get(b_) or {}
            def classify(a0, ea, eb, a_=a_, b_=b_):  # which label's record point the shared key's draw 0 equals (exact); a missing / non-number point never matches
                ma = num(ea) and bool(a0 == ea)
                mb = num(eb) and bool(a0 == eb)
                return "both" if ma and mb else a_ if ma else b_ if mb else "neither"

            for cb in sorted(set(pa_) | set(pb_)):
                ma_, mb_ = dg(pa_, cb, "members") or [], dg(pb_, cb, "members") or []
                for f in cfg["f_keys"]:
                    for o in sorted(set(ma_) | set(mb_)):
                        ca_, cb_ = (dg(rec, "rows", lb, o, "points", mode, f, "catch") or {} for lb in (a_, b_))
                        for c in sorted(set(ca_) | set(cb_)):  # Codex diag r4: every catch name either record row states (real_mean@B, run1@A, ... included)
                            k_ = key(a_, cb, mode, f, o, c)
                            if k_ not in files:
                                cls_["per_origin"].setdefault(f"{cb} | {f}", {}).setdefault("npz key absent", 0)
                                cls_["per_origin"][f"{cb} | {f}"]["npz key absent"] += 1
                                continue
                            tag = classify(np.asarray(z[k_])[0], dg(ca_, c, "point"), dg(cb_, c, "point"))
                            cls_["per_origin"].setdefault(f"{cb} | {f}", {}).setdefault(tag, 0)
                            cls_["per_origin"][f"{cb} | {f}"][tag] += 1
                    ga_, gb_ = dg(pa_, cb, mode, f, "gap") or {}, dg(pb_, cb, mode, f, "gap") or {}
                    for k in sorted(set(ga_) | set(gb_)):
                        gk_ = gkey(a_, cb, mode, f, k)
                        if gk_ not in files:
                            cls_["pooled_gap"].setdefault(f"{cb} | {f}", {}).setdefault("npz key absent", 0)
                            cls_["pooled_gap"][f"{cb} | {f}"]["npz key absent"] += 1
                            continue
                        tag = classify(np.asarray(z[gk_])[0], dg(ga_, k, "point"), dg(gb_, k, "point"))
                        cls_["pooled_gap"].setdefault(f"{cb} | {f}", {}).setdefault(tag, 0)
                        cls_["pooled_gap"][f"{cb} | {f}"][tag] += 1
            coll_[f"{a_} <-> {b_}"] = cls_
    sol_ = {lb: {"per_origin_compared": cmp_by_row.get(lb, 0), "per_origin_failures": len(by_r.get(lb, [])), "pooled_gap": pg_scan.get(lb)}
            for lb in src["labels"] if lb.rsplit(" ", 1)[0] in ("sol_a_uncut", "Sol (a)|uncut")}
    out = {"label": "gate diagnostic: |npz per-origin draw0 - record rows[L][o] catch point| (magnitudes only, not a result)", "n_compared": n_cmp,
           "pooled_gap_scan_by_label": pg_scan, "pooled_gap_mismatches": pg_bad, "pooled_gap_missing": pg_miss, "colliding_label_pairs": coll_, "sol_a_exact": sol_,
           "compared_by_row": dict(sorted(cmp_by_row.items())), "exact_rows": sorted(r for r in cmp_by_row if r not in {x["row"] for x in fails}),
           "n_missing": n_missing, "n_fail": len(fails), "failing_rows": fail_rows, "fold_minus1_rows_CR": exp_rows,
           "failing_rows_equal_fold_minus1_rows": fail_rows == exp_rows, "failing_rows_outside_fold_families": [r for r in fail_rows if fam(r) not in fold_fams],
           "failing_levels": sorted({x["f"] for x in fails}), "by_row": {r: agg_(v) for r, v in sorted(by_r.items())},
           "by_family_readout_level": {k: agg_(v) for k, v in sorted(by_rl.items())}, "headline_draw0_npz_vs_record_rate_units": head_, "gates": fails}
    return out | {"volume_copy": save_out(SOL, "/sol", ctx["out_name"], {"context": ctx, "out": out})}


@app.local_entrypoint()
def main(cfg: str):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("uncommitted tracked changes")
    c = json.loads((HERE / "config" / f"{cfg}.json").read_text())
    r_ = c["npz"][0]["record"]
    b = git("show", r_["git"])
    if hashlib.sha256(b).hexdigest() != r_["sha256"]:
        raise SystemExit("STOP: record sha256 != config")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    if not app.app_id:
        raise SystemExit("STOP: no app id at launch (the Volume copy needs one)")
    ctx = {"kind": "diag_draw0", "cfg_name": cfg, "stamp": stamp, "git": git("rev-parse", "HEAD").decode().strip(), "app_id": app.app_id,
           "out_name": f"diag_draw0_{cfg}_{stamp}_{app.app_id}"}
    print(json.dumps({"app_id": ctx["app_id"], "volume_copy_name": ctx["out_name"]}))
    out = diag.remote(c, b, ctx)
    from probe_shift.panel_out import write_once

    q = HERE / "results" / f"diag_draw0_{cfg}_{stamp}.json"
    write_once(q, json.dumps(out | {"context": ctx}, indent=1).encode())
    print(json.dumps({"file": q.name, "sha256": hashlib.sha256(q.read_bytes()).hexdigest(), "n_compared": out["n_compared"], "n_fail": out["n_fail"],
                      "n_missing": out["n_missing"], "failing_rows_equal_fold_minus1_rows": out["failing_rows_equal_fold_minus1_rows"],
                      "failing_rows_outside_fold_families": out["failing_rows_outside_fold_families"]}))
