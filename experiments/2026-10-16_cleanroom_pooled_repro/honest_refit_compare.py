"""Clean-room 4(b) comparison (addendum 23 step 4, a1, 2026-10-20): the refit records (fit_cell, on the output Volume) against the
producer's record honest_all_routec (sha-pinned), by key, inside Modal. PRESTATEMENT_DRAFT §1 / §4 govern:
  PRINTED (P1 reply|L48, P2 command|L48; simh_vs_b; run-level AUROC): PASS iff |refit - record| <= 0.002 (unrounded record value);
    plus the boundary-case flag (record within 0.002 of an x.xx5 rounding boundary) and whether the 2-dp prints agree.
  SECONDARY (reported; never in the verdict): per-action AUROC |d| <= 0.005; run-AUROC bootstrap ci95 ends |d| <= 0.010 each;
    run-level catch at 1% / 5% |d rate| <= 0.01 (d caught count reported).
  PRECONDITION (data identity, exact; OQ20): per cell, the problem's runs (by meta key, both classes), rows per run, and the
    zero-row runs equal the record's; else NOT LIKE-FOR-LIKE (not FAIL).
The laptop gets booleans, counts and shas only; values go to the output Volume.

  HM_CFG=<folder with honest_all_routec.json> uv run --no-project --python 3.12 --with modal==1.6.1 modal run honest_refit_compare.py
"""

import json
import os
import time
from pathlib import Path

import modal

OUT = "probe-shift_aswin-surya_sub-repro_cleanroom_kill-2026-11-16"
STAMP = "2026-10-20_140924"
RECORD = ("honest_all_routec.json", "10ed5645e0a8357150e07ac6416a06c6196e995d01eb5088b94de1d554d73c46")  # as pinned in 4(a)
CELLS = (("reply", 48), ("command", 48))  # P1, P2
PROB, CLS1, CLS0 = "simh_vs_b", "sim", "B"
PRINTED_TOL, ACTION_TOL, CI_TOL, CATCH_TOL = 0.002, 0.005, 0.010, 0.01
ALPHAS = ("0.01", "0.05")
RUN_TABLE_SHA = "a006e7bb169bcb53732227dc41cbbf46c80d1be025b2130833d37081f9bd426d"  # dry record (2064e73e1)
ROWS_INDEX_SHA = "9e63cf9473227de9d1329b8176812a1a08688806d0f439e21a288ac903fe127b"  # skip record (9b5d4af13)
FIT_SUMMARY_SHA = "a8da93216484325b47264aa796f514d38ef8b182c33032fef04d727994858344"  # committed fit summary (5fd730e68); 4d (R2 P2)
EPS = 1e-12  # 4d (Codex P2): round-off allowance on the inclusive tolerance tests

app = modal.App(OUT)
OUT_VOL = modal.Volume.from_name(OUT, version=2)
image = modal.Image.debian_slim(python_version="3.12")
if modal.is_local():
    image = image.add_local_file(str(Path(os.environ["HM_CFG"]) / RECORD[0]), f"/hm/{RECORD[0]}")


@app.function(image=image, cpu=(1, 1), memory=(4096, 4096), timeout=600, retries=0, startup_timeout=300, scaledown_window=60,
              volumes={"/o": OUT_VOL})
def compare(stamp: str) -> dict:
    import hashlib
    import math

    def num(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)

    out, err = None, None
    try:  # nothing but a type name leaves on failure; raised after the except block (as 24d)
        raw = Path("/hm", RECORD[0]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != RECORD[1]:
            err = "record sha"
        else:
            H = json.loads(raw)
            d = Path("/o", stamp)
            sraw = (d / "fit_summary.json").read_bytes()
            if hashlib.sha256(sraw).hexdigest() != FIT_SUMMARY_SHA:  # 4d: the root of the cell-record sha chain
                err = "fit_summary sha"
                raise ValueError
            summ = json.loads(sraw)
            rt_raw, ri_raw = (d / "run_table.json").read_bytes(), (d / "rows_index.json").read_bytes()
            if hashlib.sha256(rt_raw).hexdigest() != RUN_TABLE_SHA:  # 4c (R2 P3): provenance of the reviewed inputs
                err = "run_table sha"
                raise ValueError
            if hashlib.sha256(ri_raw).hexdigest() != ROWS_INDEX_SHA:
                err = "rows_index sha"
                raise ValueError
            T, idx = json.loads(rt_raw), json.loads(ri_raw)["index"]
            cell_sha = {c["cell"]: c["record_sha256"] for c in summ["cells"]}
            # my run -> (key, key + "#e0", class, set); the producer's run_keys map its run names to meta keys (OQ11 join)
            meta_of = {r["name"]: (r["key"], r["key"] + "#e0", r["cls"], r["set"]) for r in T["runs"]}
            prod_meta, prod_set = H["run_keys"], H["run_set"]
            inv = {v: p for p, v in prod_meta.items()}
            if len(inv) != len(prod_meta):
                err = "record run_keys not one-to-one"
                raise ValueError
            rows, ident, values = [], {}, {}
            for ro, ly in CELLS:
                cell = f"{ro}|L{ly}"
                craw = (d / f"cell_{ro}_L{ly}.json").read_bytes()
                if hashlib.sha256(craw).hexdigest() != cell_sha.get(cell):  # 4c (R2 P3): the cell record fit_summary names
                    err = "cell record sha"
                    raise ValueError
                mine = json.loads(craw)
                if mine.get("run_table_sha256") != RUN_TABLE_SHA:  # 4d (Codex P1): the cell was fitted on this run table
                    err = "cell run_table sha"
                    raise ValueError
                hc = H["cells"][cell]
                hp = hc["problems"][PROB]
                mp = mine["problems"][PROB]
                # ---- data identity (§4 / OQ20): runs by class AND set, rows per run, zero-row runs (4c: R2 P1, R1 / R2 P2s)
                my_runs = {n: v for n, v in meta_of.items() if v[2] in (CLS1, CLS0)}
                my_rows = {n: len(idx[n][ro]) for n in my_runs}
                unmatched = class_diff = set_diff = 0
                pname = {}
                for n, (a, b, c, st) in my_runs.items():  # every run of this problem, used or zero-row, must match exactly one
                    cands = [inv[m] for m in (a, b) if m in inv]
                    if len(cands) != 1:
                        unmatched += 1
                        continue
                    pn = cands[0]
                    pname[n] = pn
                    class_diff += pn.split("|", 1)[0] != c
                    set_diff += prod_set.get(pn) != st
                my_used = {pname[n] for n in pname if my_rows[n] > 0}
                my_zero = {pname[n] for n in pname if my_rows[n] == 0}
                n_used_mine = sum(1 for n in pname if my_rows[n] > 0)
                pm_used = set(hp["run_scores"])  # producer run names scored in this problem
                rows_equal = rows_diff = 0
                for n in pname:
                    if my_rows[n] == 0:
                        continue
                    pr = (hc.get("rows_per_run") or {}).get(pname[n])
                    if num(pr) and pr == my_rows[n]:  # the record's rows_per_run values are JSON numbers
                        rows_equal += 1
                    else:
                        rows_diff += 1
                prod_zero = hc.get("runs_without_rows")  # producer run names with no rows in this cell (array of strings)
                if not (isinstance(prod_zero, list) and all(isinstance(p, str) and p in prod_meta for p in prod_zero)):
                    err = "record runs_without_rows layout"
                    raise ValueError
                prod_zero = {p for p in prod_zero if p.split("|", 1)[0] in (CLS1, CLS0)}  # this problem's classes only
                # 4d (Codex P1): the fit record's own membership must equal what this join derives
                n_cls = {c: sum(1 for n in pname if my_rows[n] > 0 and my_runs[n][2] == c) for c in (CLS1, CLS0)}
                fit_runs_ok = mp.get("n_runs") == n_cls
                fit_zero_ok = sorted(mp.get("zero_row_runs") or []) == sorted(n for n in my_runs if my_rows[n] == 0)
                one_to_one = len(set(pname.values())) == len(pname)  # 4e (Codex P1): no two runs of mine (used or zero-row) share a producer run
                like = (one_to_one and unmatched == 0 and class_diff == 0 and set_diff == 0 and n_used_mine == len(my_used) == len(pm_used)
                        and my_used == pm_used and rows_diff == 0 and my_zero == prod_zero and fit_runs_ok and fit_zero_ok)
                ident[cell] = {"like_for_like": bool(like), "n_runs_mine": n_used_mine, "n_runs_record": len(pm_used),
                               "n_unmatched": unmatched, "n_class_diff": class_diff, "n_set_diff": set_diff,
                               "runs_set_equal": my_used == pm_used, "n_rows_per_run_equal": rows_equal, "n_rows_per_run_diff": rows_diff,
                               "n_zero_mine": len(my_zero), "n_zero_record": len(prod_zero), "zero_runs_equal": my_zero == prod_zero,
                               "fit_record_n_runs_equal": fit_runs_ok, "fit_record_zero_runs_equal": fit_zero_ok, "join_one_to_one": one_to_one}
                # ---- printed gate
                a_m, a_r = mp["observed"]["run_level"]["auroc"], hp["run_level"]["auroc"]
                ok = num(a_m) and num(a_r) and abs(a_m - a_r) <= PRINTED_TOL + EPS
                boundary = num(a_r) and abs(a_r - (math.floor(a_r * 100) + 0.5) / 100) <= PRINTED_TOL + EPS  # within 0.002 of x.xx5
                rows.append({"key": f"{cell}|printed_run_auroc", "verdict": True, "match": bool(ok),
                             "boundary_case": bool(boundary), "print_2dp_equal": bool(num(a_m) and num(a_r) and round(a_m, 2) == round(a_r, 2))})
                values[f"{cell}|run_auroc"] = {"mine": a_m, "record": a_r}
                # ---- secondary
                pa_m, pa_r = mp["observed"]["per_action"]["auroc"], (hp.get("call_level") or {}).get("auroc")
                rows.append({"key": f"{cell}|per_action_auroc", "verdict": False,
                             "match": bool(num(pa_m) and num(pa_r) and abs(pa_m - pa_r) <= ACTION_TOL + EPS)})
                values[f"{cell}|per_action_auroc"] = {"mine": pa_m, "record": pa_r}
                ci_m, ci_r = mp["bootstrap"]["run_auroc"], (hp.get("run_auroc_boot") or {}).get("ci95")
                cm = [ci_m.get("lo"), ci_m.get("hi")] if isinstance(ci_m, dict) else None  # fit_cell ci() = {lo, hi, n_noncomputable}
                for i in (0, 1):
                    ok_i = (isinstance(cm, list) and isinstance(ci_r, list) and len(cm) == 2 and len(ci_r) == 2
                            and num(cm[i]) and num(ci_r[i]) and abs(cm[i] - ci_r[i]) <= CI_TOL + EPS)
                    rows.append({"key": f"{cell}|run_auroc_ci95[{i}]", "verdict": False, "match": bool(ok_i)})
                values[f"{cell}|run_auroc_ci95"] = {"mine": cm, "record": ci_r}
                for a in ALPHAS:
                    c_m = mp["observed"]["run_level"].get(f"catch_{a}") or {}
                    c_r = hp["run_level"].get(f"catch_{a}") or {}
                    ok_c = num(c_m.get("rate")) and num(c_r.get("tpr")) and abs(c_m["rate"] - c_r["tpr"]) <= CATCH_TOL + EPS
                    rows.append({"key": f"{cell}|run_catch_{a}", "verdict": False, "match": bool(ok_c),
                                 "d_caught": (c_m.get("hits") - c_r.get("k_caught")) if isinstance(c_m.get("hits"), int)
                                 and isinstance(c_r.get("k_caught"), int) else None})
                    values[f"{cell}|run_catch_{a}"] = {"mine": c_m.get("rate"), "record": c_r.get("tpr")}
            like_all = all(v["like_for_like"] for v in ident.values())
            printed = [r for r in rows if r["verdict"]]
            failing = [r["key"] for r in printed if not r["match"]]
            if not like_all:
                verdict = "NOT LIKE-FOR-LIKE"
            elif not failing:
                verdict = "PRINTED REPRODUCED (P1, P2 within 0.002)"
            else:
                verdict = "PRINTED NOT REPRODUCED: " + ", ".join(failing)  # 4c (R2 P2): keys, not values
            uncert = bool(summ.get("exit_nonzero_required")) or bool(summ.get("n_uncertified_total"))
            label = ("clean-room re-fit (PRELIMINARY until reviewed); comparison by key against the producer record"
                     + ("; uncertified fits present" if uncert else "; all fits certified"))
            gate = {"n_cells_done": summ.get("n_cells_done"), "exit_nonzero_required": summ.get("exit_nonzero_required"),
                    "n_uncertified_total": summ.get("n_uncertified_total")}
            vb = json.dumps({"rows": rows, "identity": ident, "values": values, "verdict": verdict, "label": label,
                             "fit_summary_gate": gate}, default=float).encode()  # 4d (Codex P1): the qualification travels with the values
            vpath = d / f"refit_compare_values_{time.strftime('%Y%m%d_%H%M%S')}.json"
            with open(vpath, "xb") as fh:  # create-only, on the Volume; values never leave the container
                fh.write(vb)
            OUT_VOL.commit()
            out = {"stamp": stamp, "verdict": verdict, "like_for_like": like_all, "identity": ident,
                   "printed": {"n": len(printed), "n_match": sum(r["match"] for r in printed),
                               "boundary_case": [r["key"] for r in printed if r["boundary_case"]],
                               "boundary_warnings": [f"{r['key']}: printed last digit not robust (boundary case)" for r in printed
                                                     if r["boundary_case"] and not r["print_2dp_equal"]],  # 4e (Codex P2): §4 wording
                               "print_2dp_equal": {r["key"]: r["print_2dp_equal"] for r in printed}},
                   "secondary": {"n": sum(not r["verdict"] for r in rows), "n_match": sum(r["match"] for r in rows if not r["verdict"]),
                                 "breaches": [r["key"] for r in rows if not r["verdict"] and not r["match"]],
                                 "d_caught": {r["key"]: r["d_caught"] for r in rows if "d_caught" in r}},
                   "fit_summary_gate": gate,
                   "values_file": {"volume": OUT, "path": f"/{stamp}/{vpath.name}", "sha256": hashlib.sha256(vb).hexdigest()},
                   "label": label}
    except Exception as e:
        err = err or type(e).__name__
    if err:
        raise RuntimeError(f"STOP: {err}")
    return out


@app.local_entrypoint()
def main():
    out = compare.remote(STAMP)
    path = Path(__file__).parent / "results" / f"honest_refit_compare_{time.strftime('%Y-%m-%d_%H%M%S')}_{out['values_file']['sha256'][:8]}.json"
    with open(path, "x") as fh:  # 4e (Codex P2): exclusive create (append-only results)
        fh.write(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps({k: out[k] for k in ("verdict", "label", "like_for_like", "printed", "secondary", "fit_summary_gate")}))
    print("wrote", path)
