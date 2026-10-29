"""Clean-room Table 7 row 5 comparison (addendum 26, a1, 2026-10-20): the a_vs_b refit records (fit_cell under <stamp>/a_vs_b/ on the
output Volume) against the printed source, simpool items.5_table7_row5_control.run_level_auroc.routec[cell] (release rows 95-98), by key,
inside Modal. PRESTATEMENT_ROW5.md §5 governs; the logic is honest_refit_compare.py (23 step 4, blob 070aca50) with these changes:
  PRINTED (P1 reply|L48 rows 95/97, P2 command|L48 rows 96/98; a_vs_b; run-level AUROC): PASS iff |refit - simpool| <= 0.002;
    plus the boundary-case flag and whether the 2-dp prints agree.
  SOURCE CONSISTENCY (precondition): simpool routec[cell] == honest_all a_vs_b run_level.auroc within 1e-12 (as 4(a) found).
  SECONDARY (reported; never in the verdict): vs honest_all a_vs_b, as 23 step 4.
  PRECONDITION (data identity, exact; OQ20): as 23 step 4, classes A and B.
  BINDING: the RUN_ME hashes the committed fit-summary record (git show) and passes only its sha (--fit-summary-sha).
  FAIL CLOSED (26d, 09): every key the comparison reads is indexed directly (a missing key raises, so nothing compares None == None);
    before any verdict, the container removes each key in FIXTURES in turn and requires core() to STOP on every one.
The laptop gets booleans, counts and shas only; values go to the output Volume.

  HM_CFG=<folder with honest_all_routec.json and simpool_record.json> uv run --no-project --python 3.12 --with modal==1.6.1 \
    modal run honest_refit_compare_row5.py --fit-summary-sha <sha256>
"""

import json
import math
import os
import time
from pathlib import Path

import modal

OUT = "probe-shift_aswin-surya_sub-repro_cleanroom_kill-2026-11-16"
STAMP, SUB = "2026-10-20_140924", "a_vs_b"
RECORD = ("honest_all_routec.json", "10ed5645e0a8357150e07ac6416a06c6196e995d01eb5088b94de1d554d73c46")  # @ 596bc767e, as in 4(a)
SIMPOOL = ("simpool_record.json", "2e17de0c852612d8a2b1d83468997c0cb6f787dcfee9a3c93eef6edee711fd11")  # @ 104087089, the printed source
ITEM = "5_table7_row5_control"
CELLS = (("reply", 48), ("command", 48))  # P1, P2
PROB, CLS1, CLS0 = "a_vs_b", "A", "B"
PRINTED_TOL, ACTION_TOL, CI_TOL, CATCH_TOL, SRC_TOL = 0.002, 0.005, 0.010, 0.01, 1e-12
ALPHAS = ("0.01", "0.05")
RUN_TABLE_SHA = "a006e7bb169bcb53732227dc41cbbf46c80d1be025b2130833d37081f9bd426d"  # dry record (2064e73e1)
ROWS_INDEX_SHA = "9e63cf9473227de9d1329b8176812a1a08688806d0f439e21a288ac903fe127b"  # skip record (9b5d4af13)
EPS = 1e-12  # round-off allowance on the inclusive tolerance tests (4d)
_HC = ("cells", "reply|L48")  # fixture prefixes: the first cell
_HP = _HC + ("problems", PROB)
_MP = ("reply|L48", "problems", PROB)
FIXTURES = (  # 26d / 26e (09; Codex r2 P2): (root, key path), one per key core() reads, parents included; each removal must raise
    ("T", ("runs",)), *(("T", ("runs", 0, k)) for k in ("name", "key", "cls", "set")),
    ("H", ("run_keys",)), ("H", ("run_set",)), ("H", ("cells",)), ("H", _HC), ("H", _HC + ("problems",)), ("H", _HP),
    ("H", _HC + ("rows_per_run",)), ("H", _HC + ("runs_without_rows",)), ("H", _HP + ("run_scores",)),
    ("H", _HP + ("run_level",)), ("H", _HP + ("run_level", "auroc")), ("H", _HP + ("call_level",)),
    ("H", _HP + ("call_level", "auroc")), ("H", _HP + ("run_auroc_boot",)), ("H", _HP + ("run_auroc_boot", "ci95")),
    *(("H", _HP + ("run_level", f"catch_{a}")) for a in ALPHAS),
    *(("H", _HP + ("run_level", f"catch_{a}", k)) for a in ALPHAS for k in ("tpr", "k_caught")),
    ("SP", ("reply|L48",)),
    ("mine", _MP[:1]), ("mine", _MP[:2]), ("mine", _MP),
    ("mine", _MP + ("n_runs",)), ("mine", _MP + ("zero_row_runs",)), ("mine", _MP + ("observed",)),
    ("mine", _MP + ("observed", "run_level")), ("mine", _MP + ("observed", "run_level", "auroc")),
    ("mine", _MP + ("observed", "per_action")), ("mine", _MP + ("observed", "per_action", "auroc")),
    ("mine", _MP + ("bootstrap",)), ("mine", _MP + ("bootstrap", "run_auroc")),
    ("mine", _MP + ("bootstrap", "run_auroc", "lo")), ("mine", _MP + ("bootstrap", "run_auroc", "hi")),
    *(("mine", _MP + ("observed", "run_level", f"catch_{a}")) for a in ALPHAS),
    *(("mine", _MP + ("observed", "run_level", f"catch_{a}", k)) for a in ALPHAS for k in ("rate", "hits")),
    ("summ", ("n_cells_done",)), ("summ", ("exit_nonzero_required",)), ("summ", ("n_uncertified_total",)),
)  # plus 2 built at run time from the data: idx[<first A / B run>] and idx[<first A / B run>]["reply"] (run_fixtures)
N_FIXTURES = len(FIXTURES) + 2

app = modal.App(OUT)
OUT_VOL = modal.Volume.from_name(OUT, version=2)
image = modal.Image.debian_slim(python_version="3.12")
if modal.is_local():
    for f in (RECORD[0], SIMPOOL[0]):
        image = image.add_local_file(str(Path(os.environ["HM_CFG"]) / f), f"/hm/{f}")


def num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def req(x, typ, what):  # fail closed: a required field of the wrong type (incl. null) raises
    if not isinstance(x, typ):
        raise ValueError(what)
    return x


def core(H, SP, mine, summ, T, idx):
    """The comparison, no I/O. H = producer record, SP = simpool routec map, mine = {cell: fit_cell record}, summ = fit summary,
    T / idx = run table / rows index. Every read is a direct index: a missing key raises (26d)."""
    meta_of = {r["name"]: (r["key"], r["key"] + "#e0", r["cls"], r["set"]) for r in T["runs"]}
    prod_meta, prod_set = req(H["run_keys"], dict, "run_keys"), req(H["run_set"], dict, "run_set")
    inv = {v: p for p, v in prod_meta.items()}
    if len(inv) != len(prod_meta):
        raise ValueError("record run_keys not one-to-one")
    rows, ident, values = [], {}, {}
    for ro, ly in CELLS:
        cell = f"{ro}|L{ly}"
        hc = H["cells"][cell]
        hp = hc["problems"][PROB]
        mp = mine[cell]["problems"][PROB]
        # ---- source consistency (precondition): the printed value is the record's a_vs_b run AUROC
        s_v, h_v = SP[cell], hp["run_level"]["auroc"]
        src_ok = num(s_v) and num(h_v) and abs(s_v - h_v) <= SRC_TOL * max(1.0, abs(h_v))
        # ---- data identity (OQ20): runs by class AND set, rows per run, zero-row runs (as 23 step 4c-4e)
        rpr = req(hc["rows_per_run"], dict, "rows_per_run")  # cell level (09's key check)
        prod_zero = req(hc["runs_without_rows"], list, "runs_without_rows")  # cell level
        if not all(isinstance(p, str) and p in prod_meta for p in prod_zero):
            raise ValueError("record runs_without_rows layout")
        pm_used = set(req(hp["run_scores"], dict, "run_scores"))  # producer run names scored in this problem
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
            set_diff += prod_set.get(pn) != st  # a run missing from run_set counts as a difference
        my_used = {pname[n] for n in pname if my_rows[n] > 0}
        my_zero = {pname[n] for n in pname if my_rows[n] == 0}
        n_used_mine = sum(1 for n in pname if my_rows[n] > 0)
        rows_equal = rows_diff = 0
        for n in pname:
            if my_rows[n] == 0:
                continue
            pr = rpr.get(pname[n])  # a scored run absent from rows_per_run counts as a difference
            if num(pr) and pr == my_rows[n]:
                rows_equal += 1
            else:
                rows_diff += 1
        prod_zero = {p for p in prod_zero if p.split("|", 1)[0] in (CLS1, CLS0)}  # this problem's classes only
        n_cls = {c: sum(1 for n in pname if my_rows[n] > 0 and my_runs[n][2] == c) for c in (CLS1, CLS0)}
        fit_runs_ok = req(mp["n_runs"], dict, "fit n_runs") == n_cls
        fit_zero_ok = sorted(req(mp["zero_row_runs"], list, "fit zero_row_runs")) == sorted(n for n in my_runs if my_rows[n] == 0)
        one_to_one = len(set(pname.values())) == len(pname)
        like = (bool(src_ok) and one_to_one and unmatched == 0 and class_diff == 0 and set_diff == 0
                and n_used_mine == len(my_used) == len(pm_used) and my_used == pm_used and rows_diff == 0
                and my_zero == prod_zero and fit_runs_ok and fit_zero_ok)
        ident[cell] = {"like_for_like": bool(like), "source_consistent": bool(src_ok), "n_runs_mine": n_used_mine,
                       "n_runs_record": len(pm_used), "n_unmatched": unmatched, "n_class_diff": class_diff, "n_set_diff": set_diff,
                       "runs_set_equal": my_used == pm_used, "n_rows_per_run_equal": rows_equal, "n_rows_per_run_diff": rows_diff,
                       "n_zero_mine": len(my_zero), "n_zero_record": len(prod_zero), "zero_runs_equal": my_zero == prod_zero,
                       "fit_record_n_runs_equal": fit_runs_ok, "fit_record_zero_runs_equal": fit_zero_ok, "join_one_to_one": one_to_one}
        # ---- printed gate, against the printed source (simpool)
        a_m = mp["observed"]["run_level"]["auroc"]
        ok = num(a_m) and num(s_v) and abs(a_m - s_v) <= PRINTED_TOL + EPS
        boundary = num(s_v) and abs(s_v - (math.floor(s_v * 100) + 0.5) / 100) <= PRINTED_TOL + EPS  # within 0.002 of x.xx5
        rows.append({"key": f"{cell}|printed_run_auroc", "verdict": True, "match": bool(ok),
                     "boundary_case": bool(boundary), "print_2dp_equal": bool(num(a_m) and num(s_v) and round(a_m, 2) == round(s_v, 2))})
        values[f"{cell}|run_auroc"] = {"mine": a_m, "simpool": s_v, "record": h_v}
        # ---- secondary (vs honest_all a_vs_b)
        pa_m, pa_r = mp["observed"]["per_action"]["auroc"], hp["call_level"]["auroc"]
        rows.append({"key": f"{cell}|per_action_auroc", "verdict": False,
                     "match": bool(num(pa_m) and num(pa_r) and abs(pa_m - pa_r) <= ACTION_TOL + EPS)})
        values[f"{cell}|per_action_auroc"] = {"mine": pa_m, "record": pa_r}
        cm = [mp["bootstrap"]["run_auroc"]["lo"], mp["bootstrap"]["run_auroc"]["hi"]]  # fit_cell ci() = {lo, hi, n_noncomputable}
        ci_r = req(hp["run_auroc_boot"]["ci95"], list, "ci95")
        for i in (0, 1):
            ok_i = len(ci_r) == 2 and num(cm[i]) and num(ci_r[i]) and abs(cm[i] - ci_r[i]) <= CI_TOL + EPS
            rows.append({"key": f"{cell}|run_auroc_ci95[{i}]", "verdict": False, "match": bool(ok_i)})
        values[f"{cell}|run_auroc_ci95"] = {"mine": cm, "record": ci_r}
        for a in ALPHAS:
            c_m, c_r = mp["observed"]["run_level"][f"catch_{a}"], hp["run_level"][f"catch_{a}"]
            rate, hits, tpr, kc = c_m["rate"], c_m["hits"], c_r["tpr"], c_r["k_caught"]
            ok_c = num(rate) and num(tpr) and abs(rate - tpr) <= CATCH_TOL + EPS
            rows.append({"key": f"{cell}|run_catch_{a}", "verdict": False, "match": bool(ok_c),
                         "d_caught": (hits - kc) if num(hits) and num(kc) else None})
            values[f"{cell}|run_catch_{a}"] = {"mine": rate, "record": tpr}
    like_all = all(v["like_for_like"] for v in ident.values())
    printed = [r for r in rows if r["verdict"]]
    failing = [r["key"] for r in printed if not r["match"]]
    if not like_all:  # §5 wording; the per-key reasons are in identity[cell] (26e, R1 P3)
        verdict = "NOT LIKE-FOR-LIKE" + (" (source)" if not all(v["source_consistent"] for v in ident.values()) else "")
    elif not failing:
        verdict = "PRINTED REPRODUCED (P1, P2 within 0.002)"
    else:
        verdict = "PRINTED NOT REPRODUCED: " + ", ".join(failing)  # keys, not values
    gate = {"n_cells_done": summ["n_cells_done"], "exit_nonzero_required": summ["exit_nonzero_required"],
            "n_uncertified_total": summ["n_uncertified_total"]}
    uncert = bool(gate["exit_nonzero_required"]) or bool(gate["n_uncertified_total"])
    label = ("clean-room re-fit (PRELIMINARY until reviewed); Table 7 row 5 (a_vs_b) against the printed source by key"
             + ("; uncertified fits present" if uncert else "; all fits certified"))
    return rows, ident, values, verdict, label, gate, like_all, printed


def run_fixtures(roots: dict) -> dict:
    """26d (09): remove each FIXTURES key in turn (restored after) and require core() to raise. A fixture whose key is absent
    would be a no-op, so that STOPs the job instead of counting as a pass."""
    stopped = {}
    first_ab = next(r["name"] for r in roots["T"]["runs"] if r["cls"] in (CLS1, CLS0))  # a run whose index entry core() reads
    for root, path in FIXTURES + (("idx", (first_ab,)), ("idx", (first_ab, "reply"))):
        parent = roots[root]
        for k in path[:-1]:
            parent = parent[k]
        if not isinstance(parent, dict) or path[-1] not in parent:
            raise ValueError("fixture key absent")
        saved = parent.pop(path[-1])
        name = f"{root}." + ".".join(map(str, path))
        try:
            core(**roots)
            stopped[name] = False
        except Exception:
            stopped[name] = True
        finally:
            parent[path[-1]] = saved
    if len(stopped) != N_FIXTURES:  # a duplicate path would collapse two fixtures into one
        raise ValueError("fixture count")
    return stopped


@app.function(image=image, cpu=(1, 1), memory=(4096, 4096), timeout=300, retries=0, startup_timeout=120, scaledown_window=60,
              volumes={"/o": OUT_VOL})
def compare(stamp: str, fit_summary_sha: str) -> dict:
    import hashlib

    out, err = None, None
    try:  # nothing but a type name or a fixed string leaves on failure; raised after the except block (as 24d)
        raw, sraw_p = Path("/hm", RECORD[0]).read_bytes(), Path("/hm", SIMPOOL[0]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != RECORD[1]:
            err = "record sha"
            raise ValueError
        if hashlib.sha256(sraw_p).hexdigest() != SIMPOOL[1]:
            err = "simpool sha"
            raise ValueError
        H = json.loads(raw)
        SP = json.loads(sraw_p)["items"][ITEM]["run_level_auroc"]["routec"]
        d = Path("/o", stamp)
        ds = d / SUB
        sraw = (ds / "fit_summary.json").read_bytes()
        if hashlib.sha256(sraw).hexdigest() != fit_summary_sha:  # bound to the committed fit-summary record (RUN_ME: git show + sha)
            err = "fit_summary sha"
            raise ValueError
        summ = json.loads(sraw)
        if summ["problems"] != [PROB] or summ["sub"] != SUB:
            err = "fit_summary problem"
            raise ValueError
        rt_raw, ri_raw = (d / "run_table.json").read_bytes(), (d / "rows_index.json").read_bytes()
        if hashlib.sha256(rt_raw).hexdigest() != RUN_TABLE_SHA:
            err = "run_table sha"
            raise ValueError
        if hashlib.sha256(ri_raw).hexdigest() != ROWS_INDEX_SHA:
            err = "rows_index sha"
            raise ValueError
        T, idx = json.loads(rt_raw), json.loads(ri_raw)["index"]
        cell_sha = {c["cell"]: c["record_sha256"] for c in summ["cells"]}
        mine = {}
        for ro, ly in CELLS:
            cell = f"{ro}|L{ly}"
            craw = (ds / f"cell_{ro}_L{ly}.json").read_bytes()
            if hashlib.sha256(craw).hexdigest() != cell_sha.get(cell):  # the cell record fit_summary names
                err = "cell record sha"
                raise ValueError
            mine[cell] = json.loads(craw)
            if mine[cell]["run_table_sha256"] != RUN_TABLE_SHA or mine[cell]["rows_index_sha256"] != ROWS_INDEX_SHA:
                err = "cell input sha"
                raise ValueError
        roots = {"H": H, "SP": SP, "mine": mine, "summ": summ, "T": T, "idx": idx}
        try:
            fx = run_fixtures(roots)
        except Exception:
            err = "fixture key absent"
            raise
        if not all(fx.values()):  # a removed key that did not STOP means a vacuous check: no verdict
            err = "fixture did not stop"
            raise ValueError
        rows, ident, values, verdict, label, gate, like_all, printed = core(**roots)
        vb = json.dumps({"rows": rows, "identity": ident, "values": values, "verdict": verdict, "label": label,
                         "fit_summary_gate": gate}, default=float).encode()
        vpath = ds / f"refit_compare_row5_values_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(vpath, "xb") as fh:  # create-only, on the Volume; values never leave the container
            fh.write(vb)
        OUT_VOL.commit()
        out = {"stamp": stamp, "sub": SUB, "verdict": verdict, "like_for_like": like_all, "identity": ident,
               "fixtures": {"n": len(fx), "n_stopped": sum(fx.values())},
               "printed": {"n": len(printed), "n_match": sum(r["match"] for r in printed),
                           "boundary_case": [r["key"] for r in printed if r["boundary_case"]],
                           "boundary_warnings": [f"{r['key']}: printed last digit not robust (boundary case)" for r in printed
                                                 if r["boundary_case"] and not r["print_2dp_equal"]],
                           "print_2dp_equal": {r["key"]: r["print_2dp_equal"] for r in printed}},
               "secondary": {"n": sum(not r["verdict"] for r in rows), "n_match": sum(r["match"] for r in rows if not r["verdict"]),
                             "breaches": [r["key"] for r in rows if not r["verdict"] and not r["match"]],
                             "d_caught": {r["key"]: r["d_caught"] for r in rows if "d_caught" in r}},
               "fit_summary_gate": gate, "fit_summary_sha256": fit_summary_sha,
               "values_file": {"volume": OUT, "path": f"/{stamp}/{SUB}/{vpath.name}", "sha256": hashlib.sha256(vb).hexdigest()},
               "label": label}
    except Exception as e:
        err = err or type(e).__name__
    if err:
        raise RuntimeError(f"STOP: {err}")
    return out


@app.local_entrypoint()
def main(fit_summary_sha: str):
    if len(fit_summary_sha) != 64:
        raise SystemExit("STOP: --fit-summary-sha must be a sha256")
    out = compare.remote(STAMP, fit_summary_sha)
    path = Path(__file__).parent / "results" / f"honest_refit_compare_row5_{time.strftime('%Y-%m-%d_%H%M%S')}_{out['values_file']['sha256'][:8]}.json"
    with open(path, "x") as fh:  # exclusive create (append-only results)
        fh.write(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps({k: out[k] for k in ("verdict", "label", "like_for_like", "fixtures", "printed", "secondary", "fit_summary_gate")}))
    print("wrote", path)
