"""Does a probe's per-task drop track how obviously simulated each task is? (Aswin-approved via the writing session, relayed by 2e 10-16 ~15:0x;
POST HOC; per-set calibration; analysis only.) One Modal CPU container; the client ships sha-checked bytes and never parses a record.

Per cohort cell of the four per-set headline rows (S_old, S_new, S_newmains, S_paper493; 1,313 cells):
  x = the certified attack detector's held-out per-run score of the cell's SIMULATED run (refit3: cells[<readout>].problems.c_simatk_vs_run1
      .run_scores["sim|<cell>"]; higher = more sim-like), for commands L48 and replies L48 separately (10-20 rerun: commands moved from L63 to
      L48, the detector of Fig 2 / App E, refit3 cells["command|L48"]);
  y = the probe's per-task change sim - mean(run1, run2) of max_minus_tau (Stage D record: rows["<probe> C"][<set>].per_task, f = 1e-2,
      draw 0, per-set tau).
(1) Spearman(x, y) for sol_a_uncut C, (2) the same for the other five lead probes: points and main-clustered 95% intervals (10,000 draws; one
    default_rng([20261015, 41]) multinomial over the sorted canonical mains (refit3 run_main) of the 1,313 cells, shared by every analysis, so
    the analyses are paired); (3) for sol_a_uncut: catch-flip counts (run1 caught & sim missed; sim caught & run1 missed; and the same with run2)
    by quartile of x; (4) the null: Spearman(the detector's held-out run-2 score heldout_run2.run_scores["run2|<cell>"], run2 - run1 of the
    probe's max_minus_tau), every probe. Cells without a score (refit3 zero_row_runs) or with a non-finite y are excluded and counted.

  uv run --no-project --with modal==1.6.1 modal run --detach experiments/2026-10-15_fpr_sweep_507_prep/run_detector_vs_drop.py

POOLED-WORLD option (refresh2 for 2e, 10-17; default off: the per-set path's rows and analysis are unchanged, and since 10-17 both paths
carry meta.pins.folds and x's held-out guard, which can STOP): --stage-d <commit>:<path>@<sha256> names a
pooled Stage D record (run_peraction311 --pertask on the pooled-world A) and --pooled-combo <combo> (the lead: pooled_all_origins) reads each
set's rows under the pooled row key "<set> (all-origins draws) [pooled: <combo>]" (run_peraction311 l.1272 vtag); a missing key STOPs;
--pool-label is recorded in the label (e.g. "171-pool; superseded when the fix lands"). y then uses the pooled world's tau.
  modal run --detach experiments/2026-10-15_fpr_sweep_507_prep/run_detector_vs_drop.py --stage-d <c>:<path>@<sha> \
      --pooled-combo pooled_all_origins --pool-label "<label>"
ROUTE C (09 @ READ_LOG 8a4e2bf84 L2213; 2e 10-17): no typed label (--pool-label is refused). --a-sidecar git:<40-hex>:<path>.json@<sha256>
(A's label sidecar), --a-final <40-hex>:<path>@<sha256> (route C's pooled A, as Stage D's --pertask names it), --sd-git and
--sd-script-sha256 (the Stage D record's launch head and runner), --approval-spend "SPEND.md L<n> @ <40-hex>" (the route C
SPEND row, checked by git: spend_pointer); the container re-asserts the Stage D record by key and composes 09's
via-Stage-D label (routec_label, the same function as run_pertask_anchor's), and writes <record stem>_LABEL.json {record, record_sha256,
label} beside the record; the client keeps a results copy of both.
  modal run --detach experiments/2026-10-15_fpr_sweep_507_prep/run_detector_vs_drop.py --stage-d <40-hex>:<path>@<sha> \
      --pooled-combo pooled_all_origins --a-sidecar <...> --a-final <...> --sd-git <...> --sd-script-sha256 <...> --approval-spend <...>
"""

import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
app = modal.App("probe-shift_aswin-surya_sub-rerun_detl48_kill-2026-11-19")  # 10-20 rerun name (COMMON_RULES L34)
VOL = modal.Volume.from_name("probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13", version=2)
REFIT3 = ("80d75b27e", "experiments/2026-10-16_attack_refit3/results/attack_all_refit3_2026-10-16_141813.json",
          "df7171a98d82d28ae86653480be922d978fc36104c63f622b3a69fb43a074eac")
STAGE_D = ("695afc3bc", "experiments/2026-10-15_fpr_sweep_507_prep/results/pertask_full_2026-10-16_135838.json",
           "bce67f943f8ee8f912e51352f37c714f405a9a1798b5f89af7148f20245b6534")
RECORD_A = "ae6b09ada:experiments/2026-10-15_fpr_sweep_507_prep/results/alltasks_CRD_FULL_r2_allorigins_e12_2026-10-16_113538.json.gz (95dc8602…)"
SETS = ("S_old", "S_new", "S_newmains", "S_paper493")
PROBES = ("sol_a_uncut", "FIBS|uncut", "harmfulness|uncut", "refusal|uncut", "high-stakes|uncut", "own_all_rs_uncut_L48")
DETECTOR = {"commands L48": "command|L48", "replies L48": "reply|L48"}
FOLDS = ("ec56720ad", "experiments/2026-10-16_own_all_rs/results/own_all_rs_folds_2026-10-16.json",  # the fold list refit3 used (its FOLDS pin)
         "c2fc5d40eb84a34a64cbb9ce7155572d1414f12ea1fbc2ec05708bd73061e299")
APPROVAL = ("per-set check: Aswin's ask via the writing session, relayed by 2e 10-16, SPEND.md l.510 @ 060f4c8db; "  # 09 10-17: the approval
            "this pooled run: Aswin via 0d, READ_LOG @ e4dfdf63e")  # phrase points at the records that hold it
HELDOUT_BASIS = ("run_attack_all.py L181 (train = fold != f, test = fold == f) and L192-223 @ 80d75b27e (blob a73f64d0b, the same blob as the "
                 "10-16 fits' run at 8ea4a6a5d); neither L48 cell's c_simatk_vs_run1 fit was refit by refit3 (its only L48 refit: reply|L48 "
                 "d_run2_vs_run1 fold 4); folds by main from the fold list @ ec56720ad")
HELDOUT_LIMIT = "fold-map consistency plus the code's out-of-fold assignment; not a training-keys record"
APPROVAL_ROUTEC = ("the pooled analysis: Aswin via 0d, READ_LOG @ e4dfdf63e L1923; route C: GATES_routec_2026-10-17.md @ ccc5a32db L3 (09; Aswin: "
                   "\"run things in parallel, get me numbers asap\"); launch approval: SPEND row {spend} (0d GO)")  # 2e: pointers only, each line
# verified by git (the test greps both files at those commits, fixed strings); {spend} = --approval-spend, checked by spend_pointer
SPEND_ROW = "| ROUTE C DETECTOR_VS_DROP (GO"  # the GO row --approval-spend must point at (its line starts with this; af820177 O5)


def spend_pointer(ptr: str, git) -> str:
    """--approval-spend "SPEND.md L<n> @ <40-hex commit>": the commit is an ancestor of HEAD and SPEND.md's line n there is a route C
    detector_vs_drop row (starts with SPEND_ROW) holding no '<' (no placeholder: the row appended at the GO with its approval pointer,
    af820177 R1); git(*args) -> stdout bytes. Else STOP. Returns ptr."""
    m = re.fullmatch(r"SPEND\.md L([1-9][0-9]*) @ ([0-9a-f]{40})", ptr or "")
    if not m:
        raise SystemExit("route C: --approval-spend must be 'SPEND.md L<n> @ <40-hex commit>' (STOP)")
    try:
        git("merge-base", "--is-ancestor", m[2], "HEAD")
    except subprocess.CalledProcessError:
        raise SystemExit("route C: --approval-spend's commit is not an ancestor of HEAD (STOP)") from None
    lines = git("show", f"{m[2]}:SPEND.md").decode("utf-8").splitlines()
    if int(m[1]) > len(lines) or not lines[int(m[1]) - 1].startswith(SPEND_ROW) or "<" in lines[int(m[1]) - 1]:  # af820177 R1: no placeholder
        raise SystemExit("route C: --approval-spend does not point at a filled route C detector_vs_drop SPEND row (STOP)")
    return ptr


REL = "experiments/2026-10-15_fpr_sweep_507_prep"  # the results copy's repo folder (the label sidecar names the record by this path)
D1_TAIL = "(read, never recomputed; identity-guarded on the bytes read)"  # 09's two templates' shared tail (READ_LOG @ 8a4e2bf84 L2213)


def run_label(pooled_combo: str, pool_label: str, approval: str = APPROVAL) -> str:
    """The record's meta.label (the client prints it; the launch line compares it exactly). The per-set label is unchanged."""
    return (f"POST HOC ({approval}); POOLED-world calibration (combo {pooled_combo}); {pool_label}" if pooled_combo
            else "POST HOC (Aswin-approved via the writing session, 2e 10-16); per-set calibration") + "; analysis only; f = 1e-2, draw 0, readout C"


def routec_label(rec: dict, side: tuple, args: dict) -> str:
    """Route C (09 @ READ_LOG 8a4e2bf84 L2213; 2e 10-17): no typed label. side = (A's label sidecar bytes, its pinned sha256), checked
    before the parse; it must be exactly {record, record_sha256, label}, name the A that --final pins and route C, and be PRELIMINARY (no
    FINAL-release path exists). The Stage D source record is re-asserted by key (09 GATES: record sha == pin is checked by prep): meta.git
    == --sd-git, meta.script_sha256 == --sd-script-sha256, meta.pertask.reference_record == --final's A, and its D1 label equal to 09's
    template on A's sidecar label. A commits are full 40-hex (2e). Returns 09's via-Stage-D label:
    "<A sidecar label>; D1 read from route C pooled A @ <A commit> via Stage D @ <Stage D record's commit> (read, never recomputed; ...)"."""
    b_, want_ = side
    if hashlib.sha256(b_).hexdigest() != want_:
        raise SystemExit("route C: A's label sidecar sha256 differs from its pin (STOP)")
    sc = json.loads(b_)
    if not isinstance(sc, dict) or set(sc) != {"record", "record_sha256", "label"} or not all(isinstance(v, str) and v.strip() for v in sc.values()):
        raise SystemExit("route C: A's label sidecar is not {record, record_sha256, label} as non-empty strings (STOP)")
    a_ref, a_sha = (args.get("final") or "").rsplit("@", 1) if "@" in (args.get("final") or "") else ("", "")
    a_commit, sd_commit = a_ref.split(":", 1)[0], args["src"].split(":", 1)[0]
    if not (re.fullmatch(r"[0-9a-f]{40}", a_commit) and re.fullmatch(r"[0-9a-f]{40}", sd_commit)):
        raise SystemExit("route C: --final and --src must cite full 40-hex commits (STOP)")
    lab = sc["label"]
    if sc["record_sha256"] != a_sha or "route C" not in lab or "PRELIMINARY" not in lab or "171-pool" in lab:
        raise SystemExit("route C: A's label sidecar does not name --final's A, or its label is not route C and PRELIMINARY (STOP)")
    m_ = rec.get("meta") or {}
    if m_.get("git") != args["sd_git"] or m_.get("script_sha256") != args["sd_script_sha256"]:
        raise SystemExit("route C: the Stage D record's meta.git or meta.script_sha256 differs from its pin (STOP)")
    if (m_.get("pertask") or {}).get("reference_record") != f"{a_ref} ({a_sha})":
        raise SystemExit("route C: the Stage D record's reference record is not --final's A (STOP)")
    d1_ = rec.get("d1") or {}
    if d1_.get("mode") != "read" or d1_.get("labels") != [f"{lab}; D1 read from route C pooled A @ {a_commit} {D1_TAIL}"]:
        raise SystemExit("route C: the Stage D record's D1 label is not 09's template on A's sidecar label (STOP)")
    return f"{lab}; D1 read from route C pooled A @ {a_commit} via Stage D @ {sd_commit} {D1_TAIL}"


def label_sidecar(nm: str, sha: str, lab):
    """Route C: the record's label sidecar {record, record_sha256, label} (bytes) for the slot readout; None off route C."""
    return (json.dumps({"record": f"{REL}/results/{nm}", "record_sha256": sha, "label": lab}, indent=1) + "\n").encode() if lab else None


def fold_guard(run_fold: dict, run_main: dict, fold_of_main: dict, cellruns: dict, x_keys: list) -> list:
    """09 / 2e 10-17 (x's held-out guard): over ALL runs of each used refit3 cell (those with rows AND the zero-row runs, together equal to
    run_folds' keys), every run's fold is an int in 0..4 and has a main, every main sits in exactly ONE fold, and that fold equals the fold
    list's; every x key has a run score or is one of the cell's zero-row runs (a sim run with neither STOPs). cellruns: {cell: (all run names,
    scored names, zero-row names)}. Returns the violations (keys only, no values)."""
    v = []
    for k, (names, scored, zero) in cellruns.items():
        if set(names) != set(run_fold):
            v.append(f"{k}: its runs (with rows + zero-row) differ from run_folds' keys")
        by_ = {}
        for n in names:
            f_, m_ = run_fold.get(n), run_main.get(n)
            if type(f_) is not int or not 0 <= f_ <= 4 or m_ is None:
                v.append(f"{k}|{n}: fold not an int in 0..4, or no main")
                continue
            by_.setdefault(m_, set()).add(f_)
        for m_, fs_ in sorted(by_.items()):
            if len(fs_) != 1:
                v.append(f"{k}|{m_}: main in {len(fs_)} folds")
            elif fold_of_main.get(m_) != next(iter(fs_)):
                v.append(f"{k}|{m_}: fold differs from the fold list @ ec56720ad")
        v += [f"{k}|{x}: no x score and not a zero-row run" for x in x_keys if x not in scored and x not in zero]
    return v


def heldout_guard(refit: dict, folds: dict, x_keys: list) -> dict:
    """Runs fold_guard on refit3, then three POSITIVE CONTROLS that must each fire (fold -1 planted; a main given two folds; one scored x
    key dropped). STOPs on any real violation or any control that does not fire."""
    fom = {m: v[0] for m, v in folds["by_main"].items()}
    cr = {}
    for k in DETECTOR.values():
        c = refit["cells"][k]
        names_ = sorted(set(c["rows_per_run"]) | set(c.get("runs_without_rows", [])))  # rows_per_run holds only runs with rows (507-prep R1)
        cr[k] = (names_, set(c["problems"]["c_simatk_vs_run1"]["run_scores"]), set(c.get("runs_without_rows", [])))
    rf, rm = refit["run_folds"], refit["run_main"]
    v = fold_guard(rf, rm, fom, cr, x_keys)
    if v:
        raise SystemExit(f"held-out guard: {len(v)} fold-map violations, e.g. {v[:3]} (STOP)")
    k0 = next(iter(cr))
    names0 = cr[k0][0]
    runs_by_main = {}
    for n in names0:
        runs_by_main.setdefault(rm[n], []).append(n)
    two_ = next((ns for _, ns in sorted(runs_by_main.items()) if len(ns) >= 2), None)
    x0 = next((x for x in x_keys if x in cr[k0][1] and x not in cr[k0][2]), None)
    if two_ is None or x0 is None:
        raise SystemExit("held-out guard: a positive control has no unit to plant on (STOP)")
    m0_ = rm[two_[0]]
    fired = {"fold_minus_1": bool(fold_guard(rf | {names0[0]: -1}, rm, fom, {k0: cr[k0]}, [])),
             "fold_differs_from_list": bool(fold_guard(rf, rm, fom | {m0_: (fom[m0_] + 1) % 5}, {k0: cr[k0]}, [])),
             "main_two_folds": bool(fold_guard(rf | {two_[1]: (rf[two_[1]] + 1) % 5}, rm, fom, {k0: cr[k0]}, [])),
             "x_key_dropped": bool(fold_guard(rf, rm, fom, {k0: (cr[k0][0], cr[k0][1] - {x0}, cr[k0][2])}, [x0]))}
    if not all(fired.values()):
        raise SystemExit(f"held-out guard: a positive control did not fire: {sorted(k for k, f in fired.items() if not f)} (STOP)")
    return {"fold_map_violations": 0, "basis": HELDOUT_BASIS, "limit": HELDOUT_LIMIT, "positive_controls_fired": fired,
            "cells": sorted(cr), "runs_checked": sum(len(c[0]) for c in cr.values()), "x_keys_checked": len(x_keys),
            "fold_list": f"{FOLDS[0]}:{FOLDS[1]} ({FOLDS[2]})"}


SEED, D = [20261015, 41], 10000


def analyse(refit, stage_d, combo=""):
    import numpy as np

    def rows_of(pr, og):  # the set's Stage D row: per-set key og, or (pooled) "<og> (all-origins draws) [pooled: <combo>]"; missing STOPs
        k_ = f"{og} (all-origins draws) [pooled: {combo}]" if combo else og
        r_ = (stage_d["rows"].get(f"{pr} C") or {}).get(k_)
        if r_ is None:
            raise SystemExit(f"{pr} C: no Stage D row {k_!r}")
        return r_["per_task"]

    def rank_avg(a):  # average ranks for ties (pertask_reduce's)
        o = np.argsort(a, kind="mergesort")
        s_ = a[o]
        st = np.flatnonzero(np.r_[True, s_[1:] != s_[:-1]])
        cnt = np.diff(np.r_[st, len(a)])
        r = np.empty(len(a))
        r[o] = np.repeat(st + (cnt - 1) / 2.0, cnt) + 1.0
        return r

    def spear(x, y):
        if len(x) < 3:
            return np.nan
        rx, ry = rank_avg(x), rank_avg(y)
        rx, ry = rx - rx.mean(), ry - ry.mean()
        den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
        return float((rx * ry).sum() / den) if den > 0 else np.nan

    cells, cell_set = [], {}
    for og in SETS:  # the cohort cells of the per-set headline rows (the same for every probe; asserted below)
        for r in rows_of(PROBES[0], og):
            if r["arm"] == "sim":
                if r["cell"] in cell_set:  # a3fd1612 O1 (10-18): a second sim row for a cell, in another set or the same one, STOPs (no double count)
                    raise SystemExit(f"cell {r['cell']}: two sim rows ({cell_set[r['cell']]}, {og})")
                cell_set[r["cell"]] = og
                cells.append(r["cell"])
    main = {}
    for c in cells:
        ms = {refit["run_main"].get(f"{w}|{c}") for w in ("sim", "run1", "run2")} - {None}
        if len(ms) != 1:
            raise SystemExit(f"cell {c}: no single canonical main in refit3 run_main")
        main[c] = ms.pop()
    mains = sorted(set(main.values()))
    mcol = {m: i for i, m in enumerate(mains)}
    W = np.random.default_rng(SEED).multinomial(len(mains), np.full(len(mains), 1 / len(mains)), size=D)  # draws 1..D, shared by every analysis

    def boot(x, y, cs):
        cols = np.array([mcol[main[c]] for c in cs], int)
        bt = np.empty(D)
        for b in range(D):
            k = W[b, cols]
            bt[b] = spear(np.repeat(x, k), np.repeat(y, k))
        fin = bt[np.isfinite(bt)]
        return {"point": spear(x, y), "n_cells": len(cs), "n_mains": len(set(cols.tolist())), "draws": D, "finite_draws": int(len(fin)),
                "ci95": [float(np.percentile(fin, 2.5)), float(np.percentile(fin, 97.5))] if len(fin) >= 0.95 * D else None}

    per_probe = {}
    for pr in PROBES:
        mt, cg = {}, {}
        for og in SETS:
            for r in rows_of(pr, og):
                if r["arm"] in mt.get(r["cell"], {}):  # a3fd1612 O1: one row per (cell, arm) for every probe (never a silent overwrite)
                    raise SystemExit(f"{pr} C: cell {r['cell']} has two {r['arm']} rows")
                mt.setdefault(r["cell"], {})[r["arm"]] = r["max_minus_tau"]
                cg.setdefault(r["cell"], {})[r["arm"]] = r["caught"]
        if set(mt) != set(cells):
            raise SystemExit(f"{pr} C: its Stage D cells differ from the cohort")
        fin_ = lambda v: v is not None and np.isfinite(v)  # noqa: E731
        y_drop = {c: v["sim"] - (v["run1"] + v["run2"]) / 2 for c, v in mt.items() if all(fin_(v.get(a)) for a in ("sim", "run1", "run2"))}
        y_null = {c: v["run2"] - v["run1"] for c, v in mt.items() if all(fin_(v.get(a)) for a in ("run1", "run2"))}
        res = {}
        for dn, ck in DETECTOR.items():
            xs = refit["cells"][ck]["problems"]["c_simatk_vs_run1"]["run_scores"]
            x2 = refit["cells"][ck]["heldout_run2"]["run_scores"]
            out = {}
            for nm, xmap, ymap in (("sim_score_vs_drop", {c: xs.get(f"sim|{c}") for c in cells}, y_drop),
                                   ("null_run2_score_vs_run2_minus_run1", {c: x2.get(f"run2|{c}") for c in cells}, y_null)):
                cs = [c for c in cells if fin_(xmap[c]) and c in ymap]
                out[nm] = boot(np.array([xmap[c] for c in cs]), np.array([ymap[c] for c in cs]), cs) | {
                    "excluded": {"no_detector_score": sum(1 for c in cells if not fin_(xmap[c])),
                                 "no_finite_y": sum(1 for c in cells if fin_(xmap[c]) and c not in ymap)}}
            if pr == "sol_a_uncut":  # (3) catch flips by detector-score quartile
                cs = [c for c in cells if fin_(xs.get(f"sim|{c}")) and all(cg[c].get(a) is not None for a in ("sim", "run1", "run2"))]
                x = np.array([xs[f"sim|{c}"] for c in cs])
                edges = np.quantile(x, [0.25, 0.5, 0.75])
                q = np.searchsorted(edges, x, side="right")
                fl = {}
                for i, c in enumerate(cs):
                    d_ = fl.setdefault(f"Q{q[i] + 1}", {"n_cells": 0, "run1_caught_sim_missed": 0, "sim_caught_run1_missed": 0,
                                                         "run2_caught_sim_missed": 0, "sim_caught_run2_missed": 0})
                    d_["n_cells"] += 1
                    for real in ("run1", "run2"):
                        d_[f"{real}_caught_sim_missed"] += int(cg[c][real] and not cg[c]["sim"])
                        d_[f"sim_caught_{real}_missed"] += int(cg[c]["sim"] and not cg[c][real])
                out["catch_flips_by_score_quartile"] = {"quartile_edges": [float(e) for e in edges], "by_quartile": dict(sorted(fl.items())),
                                                        "excluded_cells": len(cells) - len(cs)}
            res[dn] = out
        per_probe[f"{pr} C"] = res
    return {"_x_keys": [f"sim|{c}" for c in cells],  # for the held-out guard (popped before the record is written)
            "cells": len(cells), "cells_by_set": {og: sum(1 for c in cells if cell_set[c] == og) for og in SETS}, "mains": len(mains),
            "zero_row_runs_refit3": {c: len(v.get("runs", [])) for c, v in (refit.get("zero_row_runs") or {}).items()}, "probes": per_probe}


@app.function(image=modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3"), cpu=(2, 2), memory=(8192, 16384), timeout=3600,
              retries=0, volumes={"/sol": VOL})
def run(blobs: dict, meta: dict) -> dict:
    for k_, (b_, want_) in blobs.items():
        if hashlib.sha256(b_).hexdigest() != want_:
            raise SystemExit(f"{k_}: sha256 (container)")
    sd = json.loads(blobs["stage_d"][0])
    if sd.get("gate", {}).get("pass") is not True:
        raise SystemExit("the Stage D record's gate did not pass")
    lab_c = routec_label(sd, blobs["a_sidecar"], meta["routec"]) if meta.get("routec") else None  # route C: before any compute
    if lab_c:  # 09 (READ_LOG @ 8a4e2bf84 L2213): composed here from A's sidecar, never typed
        meta = meta | {"label": run_label(meta["pooled_combo"], lab_c, APPROVAL_ROUTEC.format(spend=meta["routec"]["approval_spend"])), "routec_label": {
            "label": lab_c, "a_sidecar": meta["routec"]["a_sidecar"], "rule": "09 @ READ_LOG 8a4e2bf84 L2213 (via Stage D template)",
            "stage_d_reasserted": {"git": meta["routec"]["sd_git"], "script_sha256": meta["routec"]["sd_script_sha256"], "d1_label": "09's template, exact"}}}
    refit = json.loads(blobs["refit3"][0])
    res_ = analyse(refit, sd, meta.get("pooled_combo", ""))
    out = {"meta": meta, "heldout_guard": heldout_guard(refit, json.loads(blobs["folds"][0]), res_.pop("_x_keys")), "result": res_}
    b = json.dumps(out, indent=1).encode()
    p = Path("/sol/xfer/pertask") / f"{meta['name']}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("xb") as fh:
        fh.write(b)
    sha_ = hashlib.sha256(b).hexdigest()
    sc_ = label_sidecar(p.name, sha_, lab_c)
    if sc_:  # route C: written after the record, exclusive
        with (p.parent / f"{p.stem}_LABEL.json").open("xb") as fh:
            fh.write(sc_)
    VOL.commit()
    return {"name": p.name, "volume_path": str(p).removeprefix("/sol/"), "sha256": sha_, "blob": b, "label": meta["label"], "sidecar": sc_}


@app.local_entrypoint()
def main(stage_d: str = "", pooled_combo: str = "", pool_label: str = "", a_sidecar: str = "", a_final: str = "", sd_git: str = "",
         sd_script_sha256: str = "", approval_spend: str = ""):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("uncommitted tracked changes")
    if git("show", f"HEAD:./{Path(__file__).name}") != Path(__file__).read_bytes():  # the recorded git must contain the code that runs (Codex r1)
        raise SystemExit("this script is not committed at HEAD as it is on disk")
    routec = any((a_sidecar, a_final, sd_git, sd_script_sha256, approval_spend))  # route C (09 @ READ_LOG 8a4e2bf84 L2213): the label is composed in the container
    m_ = re.fullmatch(r"git:([0-9a-f]{40}):([^ @]+\.json)@([0-9a-f]{64})", a_sidecar)
    if routec and (not (m_ and stage_d and pooled_combo and re.fullmatch(r"[0-9a-f]{40}:[^ @]+@[0-9a-f]{64}", a_final)
                        and re.fullmatch(r"[0-9a-f]{40}", sd_git) and re.fullmatch(r"[0-9a-f]{64}", sd_script_sha256)) or pool_label):
        raise SystemExit("route C: --stage-d, --pooled-combo, --a-sidecar git:<40-hex>:<path>.json@<sha256>, --a-final <40-hex>:<path>@<sha256>, "
                         "--sd-git <40-hex>, --sd-script-sha256 <64-hex> and --approval-spend go together, without --pool-label (no typed label) (STOP)")
    if not routec and any((stage_d, pooled_combo, pool_label)) and not (stage_d and pooled_combo and pool_label.strip()):  # all three or none (Astra r1)
        raise SystemExit("--stage-d, --pooled-combo and --pool-label go together (the pooled-world option); give all three or none")
    sd_pin = STAGE_D
    if stage_d:
        cp_, h_ = stage_d.rsplit("@", 1)
        sd_pin = (cp_.split(":", 1)[0], cp_.split(":", 1)[1], h_)
    blobs = {}
    for k_, (c_, p_, h_) in (("refit3", REFIT3), ("stage_d", sd_pin), ("folds", FOLDS)) + ((("a_sidecar", (m_[1], m_[2], m_[3])),) if routec else ()):
        b = git("show", f"{c_}:{p_}")  # bytes only (A's sidecar too: parsed in the container, routec_label)
        if hashlib.sha256(b).hexdigest() != h_:
            raise SystemExit(f"{k_}: sha256")
        blobs[k_] = (b, h_)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    meta = {"git": git("rev-parse", "HEAD").decode().strip(), "stamp": stamp, "app_id": app.app_id, "name": f"detector_vs_drop_{stamp}_{app.app_id}",
            "label": None if routec else run_label(pooled_combo, pool_label),  # route C: set in the container (run)
            "pins": {"refit3": f"{REFIT3[0]}:{REFIT3[1]} ({REFIT3[2]})", "stage_d": f"{sd_pin[0]}:{sd_pin[1]} ({sd_pin[2]})",
                     "folds": f"{FOLDS[0]}:{FOLDS[1]} ({FOLDS[2]})"}
            | ({} if pooled_combo else {"record_A": RECORD_A}),
            "definitions": {"x": "refit3 cells[<readout>].problems.c_simatk_vs_run1.run_scores['sim|<cell>'] (certified held-out; higher = more sim-like)",
                            "y": "Stage D per_task max_minus_tau: sim - mean(run1, run2)", "null": "heldout_run2 'run2|<cell>' vs run2 - run1",
                            "interval": "main-clustered: one default_rng([20261015, 41]) multinomial over the sorted canonical mains (refit3 run_main), "
                                        "10,000 draws shared by every analysis; 2.5/97.5 percentiles of finite draws (>= 95% finite)"}}
    if pooled_combo:  # no pooled_combo key when the option is off (the per-set meta gains only pins.folds, 10-17)
        meta["pooled_combo"] = pooled_combo
    if routec:  # the pins routec_label re-asserts in the container (the same keys as run_pertask_anchor's args)
        meta["routec"] = {"final": a_final, "src": f"{sd_pin[0]}:{sd_pin[1]}", "sd_git": sd_git, "sd_script_sha256": sd_script_sha256, "a_sidecar": a_sidecar,
                         "approval_spend": spend_pointer(approval_spend, git)}
    r = run.remote(blobs, meta)
    if hashlib.sha256(r["blob"]).hexdigest() != r["sha256"]:
        raise SystemExit("result sha256")
    (HERE / "results" / r["name"]).open("xb").write(r["blob"])
    if r.get("sidecar"):  # route C: the record's label sidecar as the container wrote it beside the record (bytes as returned; never parsed here)
        sn_ = f"{Path(r['name']).stem}_LABEL.json"
        (HERE / "results" / sn_).open("xb").write(r["sidecar"])
        print(json.dumps({"label_sidecar": sn_, "sha256": hashlib.sha256(r["sidecar"]).hexdigest(), "record_sha256": r["sha256"]}), flush=True)
    elif routec:
        raise SystemExit("route C: the container returned no label sidecar (STOP)")
    print(json.dumps({"record": r["name"], "volume_path": r["volume_path"], "sha256": r["sha256"], "label": r["label"]}), flush=True)  # 2e 10-17: the launch line checks the label
