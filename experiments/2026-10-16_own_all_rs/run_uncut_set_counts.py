"""Per training SET counts for own_all_rs{,25,10}_uncut_L48 (4f / 2e's OOD r7 builder, 10-16), APPEND-ONLY NOTE bound to the fit (git record + its
sha256, Volume fit record): for each variant x readout x fold model, per set (S_old, S_new incl. the three 25f H_A roots, S_newmains): training rows,
rows past token 32768, max training token position, runs, runs with a row past 32768, and the set's uncut roots by key. The set of a row is its cell's
origin in the fit inputs' pinned cell lists (T196 -> S_old, +311 -> S_new, nm360 -> S_newmains), never a root name. Each fold model's rows are rebuilt
with fit_uncut's own rule (universe mask, fold, run1 / honest_a, the seeded per-fold subsample) and asserted against the fit record (rows, rows past
32768, runs) and the sha-checked training manifest (its mains); per-set totals must sum to them, the per-root rows to rows_uncut's, the training-unit
summary to finalize's. Also the ROW-ORDER check behind the descriptive cut-32768 equality: the uncut rows below token 32768, in fit_uncut's order, vs
own_all's registered rows (T196 then scale-up), by (unit, call_index) and bitwise on X. Counts and positions only, no scores. One CPU job (~cents)."""

import hashlib
import io
import json
import math
import random
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE  # in the container the script sits at /root
APP = "probe-shift_aswin-surya_sub-p2probes_ownall-setcounts_kill-2026-11-14"
FIT_GIT = ("experiments/2026-10-14_p2_family_expansion/results/own_all_rs_uncut_fit_2026-10-16_030614.json", "f62f4d660109fff5d3d207a024299f3c254fbfb2c603ddc7fc62b40b38760442")
INPUTS = "experiments/2026-10-14_p2_family_expansion/results/own_all_inputs_2026-10-15.json"
RS_FOLDS = "experiments/2026-10-16_own_all_rs/results/own_all_rs_folds_2026-10-16.json"
ROWS_STAMP = "2026-10-15_173309"  # own_all v1's registered rows (run_own_all.py ROWS_SHA, verbatim)
ROWS_SHA = {("T196", "R"): "8b375c6174a12740165cd483a6cfa1be58c9cec6caed95387bdc82c48ba53fcb", ("T196", "C"): "26edc3300421ee915f08d9f0d8f3cff0488d5c66c596286c016df2e6aff478f7",
            ("scaleup", "R"): "ec50594b39d6f63e83df0918c9773e25c9c60921975248610fcf11bb1a1ebd24", ("scaleup", "C"): "09cbe4995372aa100ca682d53d8ac3c481756a3bf42c311f9001f63b9a56e308"}
UNCUT_VARS = {"own_all_rs_uncut_L48": 1.0, "own_all_rs25_uncut_L48": 0.25, "own_all_rs10_uncut_L48": 0.10}
SETS = ("S_old", "S_new", "S_newmains")
OUT = modal.Volume.from_name("probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13", version=2)
img = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")
app = modal.App(APP)


def _subsample(tasks, k, v):
    """run_own_all._subsample (= run_famexp._train_tasks @ b5a0586fa), verbatim."""
    t = sorted(tasks)
    random.Random(f"famexp|own_a|f{k}").shuffle(t)
    return set(t[: math.ceil(v * len(t))])


@app.function(image=img, cpu=(8, 8), memory=(65536, 65536), timeout=1800, retries=0, scaledown_window=2, volumes={"/out": OUT.read_only()})
def counts(stamp: str, fit_sha: str, rows_sha: dict, main_of: dict, origin: dict, fold: dict, k1: list) -> dict:
    import numpy as np

    od = Path("/out/own_all_rs_uncut") / stamp
    fb = (od / "fit_record.json").read_bytes()
    if hashlib.sha256(fb).hexdigest() != fit_sha:
        raise SystemExit("the Volume fit record differs from the git record's fit_record_sha256")
    fr = json.loads(fb)
    fits = {(f["variant"], f["readout"], f["fold"]): f for f in fr["fits"]}
    k1 = set(k1)
    table, per_root, summary, order, zero = [], {}, {}, {}, []
    for rd in ("R", "C"):
        zs, roots = [], []
        for nm, h in rows_sha[rd].items():  # fit_uncut's concatenation order (the fit record's dict order)
            b = (od / "rows" / f"{nm}__{rd}.npz").read_bytes()
            if hashlib.sha256(b).hexdigest() != h:
                raise SystemExit(f"{nm}__{rd}: rows differ from the fit record")
            z = np.load(io.BytesIO(b))
            zs.append(z)
            roots += [nm] * len(z["names"])
            per_root[f"{nm}|{rd}"] = int(len(z["names"]))
        names = np.concatenate([z["names"] for z in zs])
        ci = np.concatenate([z["call_index"] for z in zs])
        tok = np.concatenate([z["token"] for z in zs])
        rk = np.concatenate([z["run_key"] for z in zs])
        roots = np.array(roots)
        cells = np.array([u.split("|", 1)[1] for u in names])
        mains = np.array([main_of[c] for c in cells])
        world = np.array([u.split("|", 1)[0] for u in names])
        y = np.where(world == "run1", 1.0, np.where(world == "honest_a", 0.0, np.nan))
        uni = np.array([m in k1 for m in mains])
        rf = np.array([fold[m] for m in mains])
        st = np.array([origin[c] for c in cells])
        trn = uni & ~np.isnan(y)
        summary[rd] = {"training_rows": int(trn.sum()), "training_rows_past_32768": int((trn & (tok >= 32768)).sum()), "training_runs": len(set(rk[trn].tolist())),
                       "training_runs_with_rows_past_32768": len(set(rk[trn & (tok >= 32768)].tolist()))}
        for k in range(5):
            full = uni & (rf != k) & ~np.isnan(y)
            for var, v in UNCUT_VARS.items():
                tr = full & np.isin(cells, sorted(_subsample(set(cells[full].tolist()), k, v)))
                f = fits[(var, rd, k)]
                mb = (od / "training" / f"{var}__{rd}__f{k}.json").read_bytes()
                if hashlib.sha256(mb).hexdigest() != f["training_manifest_sha256"] or json.loads(mb) != sorted(set(mains[tr].tolist())):
                    raise SystemExit(f"{var} {rd} f{k}: the training manifest differs from its sha or from the rebuilt rows' mains")
                if (int(tr.sum()), int((tr & (tok >= 32768)).sum()), len(set(names[tr].tolist()))) != (f["rows"], f["rows_past_32768"], f["runs"]):
                    raise SystemExit(f"{var} {rd} f{k}: the rebuilt rows differ from the fit record")
                sets = {}
                for s in SETS:
                    m = tr & (st == s)
                    p = m & (tok >= 32768)
                    sets[s] = {"training_rows": int(m.sum()), "training_rows_past_32768": int(p.sum()), "max_training_token_position": int(tok[m].max()) if m.any() else None,
                               "runs": len(set(rk[m].tolist())), "runs_with_rows_past_32768": len(set(rk[p].tolist())),
                               "roots": {str(r): int(n) for r, n in zip(*np.unique(roots[m], return_counts=True), strict=True)}}
                    if sets[s]["training_rows_past_32768"] == 0:
                        zero.append(f"{var}|{rd}|f{k}|{s}")
                if sum(x["training_rows"] for x in sets.values()) != f["rows"] or sum(x["training_rows_past_32768"] for x in sets.values()) != f["rows_past_32768"]:
                    raise SystemExit(f"{var} {rd} f{k}: per-set totals do not sum to the fit's rows")
                table.append({"variant": var, "readout": rd, "fold": k, "training_manifest_sha256": f["training_manifest_sha256"], "weights_sha256": f["weights_sha256"],
                              "rows": f["rows"], "rows_past_32768": f["rows_past_32768"], "sets": sets})
        keys_by_set = {s: sorted(set(rk[trn & (st == s)].tolist())) for s in SETS}
        roots_by_set = {s: sorted(set(roots[trn & (st == s)].tolist())) for s in SETS}
        # the row-order check: uncut rows below 32768 (fit_uncut's order) vs own_all's registered rows (T196 then scale-up)
        rg = []
        for p_ in ("T196", "scaleup"):
            bb = (Path("/out/own_all") / ROWS_STAMP / "rows" / f"{p_}__{rd}.npz").read_bytes()
            if hashlib.sha256(bb).hexdigest() != ROWS_SHA[(p_, rd)]:
                raise SystemExit(f"own_all's rows {p_}__{rd}: sha256 differs from the pin")
            rg.append(np.load(io.BytesIO(bb)))
        rn = np.concatenate([z["names"] for z in rg])
        rci = np.concatenate([z["call_index"] for z in rg])
        below = np.flatnonzero(tok < 32768)
        un, uci = names[below], ci[below]
        seq_r, seq_u = list(zip(rn.tolist(), rci.tolist(), strict=True)), list(zip(un.tolist(), uci.tolist(), strict=True))
        in_order = seq_r == seq_u
        first = next((i for i, (a, b_) in enumerate(zip(seq_r, seq_u, strict=False)) if a != b_), None)
        ms = sorted(seq_r) == sorted(seq_u)
        xb = None
        if ms and len(set(seq_r)) == len(seq_r):
            Xr = np.concatenate([z["X"] for z in rg])
            Xu = np.concatenate([z["X"] for z in zs])[below]
            if Xr.dtype != np.float32 or Xu.dtype != np.float32 or Xr.shape[1] != Xu.shape[1]:
                raise SystemExit("X dtype / width differs between the registered and uncut rows")
            pos = {kk: i for i, kk in enumerate(seq_u)}
            idx = np.array([pos[kk] for kk in seq_r])
            xb = int((Xr.view(np.uint32) == Xu[idx].view(np.uint32)).all(axis=1).sum())
            del Xr, Xu
        order[rd] = {"registered_rows": len(seq_r), "uncut_rows_below_32768": len(seq_u), "multiset_equal_by_unit_and_call_index": ms, "in_order_equal": in_order,
                     "first_order_mismatch_index": first, "unique_unit_call_keys": len(set(seq_r)) == len(seq_r), "X_bitwise_equal_rows": xb,
                     "registered_rows_sha256": {p_: ROWS_SHA[(p_, rd)] for p_ in ("T196", "scaleup")}}
        summary[rd] |= {"keys_by_set": {s: len(v) for s, v in keys_by_set.items()}, "roots_by_set": roots_by_set, "run_keys_by_set": keys_by_set}
    return json.loads(json.dumps({"table": table, "per_root_rows": per_root, "summary": summary, "zero_cells": zero, "row_order": order}))  # plain types (no numpy on the client)


@app.local_entrypoint()
def main():
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, check=True).stdout  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no").strip() or git("rev-parse", "HEAD") != git("rev-parse", "origin/aswin/own-all-loo"):
        raise SystemExit("uncommitted tracked changes, or HEAD is not on origin")
    fb = git("show", f"HEAD:{FIT_GIT[0]}")
    if hashlib.sha256(fb).hexdigest() != FIT_GIT[1]:
        raise SystemExit("the git fit record differs from its pin")
    fr = json.loads(fb)
    inp = json.loads(git("show", f"HEAD:{INPUTS}"))
    if hashlib.sha256(git("show", f"HEAD:{INPUTS}")).hexdigest() != fr["inputs"]["sha256"]:
        raise SystemExit("the inputs record differs from the fit's")
    main_of, origin = {}, {}
    for (c_, p_, s_), og in ((inp["pins"]["t196"], "S_old"), (inp["pins"]["plus311"], "S_new"), (inp["pins"]["nm360"], "S_newmains")):
        b = git("show", f"{c_}:{p_}")
        if hashlib.sha256(b).hexdigest() != s_:
            raise SystemExit(f"{p_}: sha256 differs")
        for c in json.loads(b)["cells"]:
            main_of[c["key"]] = c["main"]
            if origin.setdefault(c["key"], og) != og:
                raise SystemExit(f"{c['key']}: a cell in two sets")
    fl = git("show", f"HEAD:{RS_FOLDS}")
    if hashlib.sha256(fl).hexdigest() != fr["rs"]["fold_list"]["sha256"]:
        raise SystemExit("the fold list differs from the fit's")
    fold = {m: v[0] for m, v in json.loads(fl)["by_main"].items()}
    fit_sha = fr["fit"]["fit_record_sha256"]
    res = counts.remote(fr["stamp (PT)"], fit_sha, fr["rs_uncut"]["rows"], main_of, origin, fold, sorted(inp["folds"]["by_main"]))
    pr = {f"{nm}|{rd}": r[rd]["rows"] for nm, r in fr["rows"].items() for rd in ("R", "C")}
    sm = {rd: {k: v for k, v in res["summary"][rd].items() if k.startswith("training_")} for rd in ("R", "C")}
    fs = {rd: {k: v for k, v in fr["fit"]["training_units_past_32768"][rd].items() if k != "cut"} for rd in ("R", "C")}
    ha = {"su311_q1_main_ha25f_cutfull_extract_v1", "su311_q2_main_ha25f_cutfull_extract_v1", "su311_q4_main_ha25f_cutfull_extract_v1"}
    checks = {"per_root_rows_equal_rows_uncut": res["per_root_rows"] == pr, "training_unit_summary_equal_finalize": sm == fs,
              "ha25f_roots_in_S_new": all(ha <= set(res["summary"][rd]["roots_by_set"]["S_new"]) for rd in ("R", "C")),
              "roots_in_one_set": all(not (set(a) & set(b)) for rd in ("R", "C") for i, a in enumerate(res["summary"][rd]["roots_by_set"].values())
                                      for j, b in enumerate(res["summary"][rd]["roots_by_set"].values()) if i < j)}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    note = {"label": "own_all_rs{,25,10}_uncut_L48 per-training-set counts (APPEND-ONLY NOTE; 4f / 2e OOD r7, 10-16): counts and token positions only, no scores; PRELIMINARY",
            "stamp (PT)": stamp, "writer": {"commit": git("rev-parse", "HEAD").decode().strip(), "app": APP},
            "fit": {"git_record": {"path": FIT_GIT[0], "sha256": FIT_GIT[1]}, "fit_record_sha256": fit_sha, "cut": fr["fit"]["cut"]},
            "set_rule": "a row's set = its cell's origin in the fit inputs' pinned cell lists (T196 -> S_old, +311 -> S_new, nm360 -> S_newmains), by key; never a root name",
            "pass": all(checks.values()), "pass_scope": "the counts asserts only (checks; the in-job STOPs on rebuilt rows, manifests and per-set sums); the row-order "
                                                        "check is DESCRIPTIVE and reported separately (row_order, row_order_all_equal), never part of pass (Codex r20)",
            "checks": checks, "zero_cells": res["zero_cells"], "table": res["table"], "summary": res["summary"], "row_order": res["row_order"],
            "row_order_all_equal": all(o["multiset_equal_by_unit_and_call_index"] and o["in_order_equal"] and o["X_bitwise_equal_rows"] == o["registered_rows"] == o["uncut_rows_below_32768"]
                                       for o in res["row_order"].values())}
    out = HERE.parents[0] / "2026-10-14_p2_family_expansion" / "results" / f"own_all_rs_uncut_set_counts_NOTE_{stamp}.json"
    out.open("x").write(json.dumps(note, indent=1))
    print(json.dumps({"pass": note["pass"], "row_order_all_equal": note["row_order_all_equal"], "checks": checks, "zero_cells": len(res["zero_cells"]), "row_order": {rd: {k: v for k, v in o.items() if k != "registered_rows_sha256"}
                                                                                                              for rd, o in res["row_order"].items()}, "path": str(out.relative_to(REPO))}))
