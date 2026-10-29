"""Gate for adds (a), the planted-shift arms on the all-tasks job (09 @ 002c1824b; on Modal, the npz never leave the Volume): the plant npz against
the final all-tasks npz, by key, over the plant labels (the headline six, C and R). Every reference array of those labels must be present and EQUAL
(shape, dtype, NaN == NaN); the only new keys are plant arms ("...__<origin>__catch_<arm>_plant_<+-d>", "...__pooled__catch_<arm>_plant_<+-d>",
"...__pooled__gap_delta_full[_<arm>]_plant_<+-d>"), of plant labels only, each with the shape of its base array; every delta-0 arm equals its base
(catch_sim_at_S_primary / pooled gap_delta_full); every joint (row, point) with a sim@S|primary array outside the registered rows carries a delta-0
arm unless the plant record lists that row, point and arm as unavailable (draw-0 tau_sim not finite), and no other row is planted; every pooled
planted catch equals the members' per-origin planted catches combined as pool_reduce_n combines them (1e-12), and every pooled planted gap equals
that catch minus the members' combined real mean (1e-12). No array is empty. Exit 1 on any failure (the
report is still written).

  modal run plant_gate_alltasks.py --ref xfer/fpr507prep/<final all-tasks npz>@<sha> --new xfer/fpr507prep/<plant npz>@<sha> \
      --record results/<plant record>.json@<sha> --out results/checks/<name>.json
"""

import hashlib
import io
import json
import re
from pathlib import Path

import modal

app = modal.App("probe-shift_aswin-surya_sub-fpr507prep_plantgateat_kill-2026-11-14")
vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13", version=2)  # run_common455.VOLS["solavg"]
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")
SIX = ("Sol_a", "FIBS", "harmfulness", "refusal", "high-stakes", "own_all_rs_L48")  # npzkey form of the plant labels' families
PLANT_LABELS = {f"{x}_{r}" for x in SIX for r in "CR"}
# full-length records: every record label (full_override's names after the rename, npzkey form) is a plant label (Codex r12)
ALLO = {"pooled_all_origins", "pooled_all_origins_without_S4", "paper_cells", "pooled_all_origins_registered_cohort"}  # run_peraction311.COMBOS_ALLO
ARM_CATCH = re.compile(r"^catch_(?P<arm>sim2?)_plant_(?P<d>[+-][0-9]+)$")  # per origin and pooled
POOL_GAP = re.compile(r"^gap_delta_full(?:_(?P<arm>sim2))?_plant_(?P<d>[+-][0-9]+)$")  # pooled only


def npz_part(s):  # run_peraction311.npzkey's in-part rule
    return str(s).replace("@", "_at_").replace("|", "_").replace(" ", "_").replace("(", "").replace(")", "")


@app.function(image=image, volumes={"/sol": vol}, cpu=2, memory=16384, timeout=1800)
def check(ref, new, unavailable, full_labels=None):
    import numpy as np

    global PLANT_LABELS
    if full_labels:  # the full-length job's record labels (renamed after the row), npzkey form
        PLANT_LABELS = set(full_labels)

    z = {}
    for tag, r in (("ref", ref), ("new", new)):
        p_, h_ = r.rsplit("@", 1)
        b = (Path("/sol") / p_).read_bytes()
        if hashlib.sha256(b).hexdigest() != h_:
            raise SystemExit(f"{tag} npz sha256 differs")
        z[tag] = np.load(io.BytesIO(b))
    lab = lambda k: k.split("__", 1)[0]  # noqa: E731
    ref_keys = [k for k in z["ref"].files if lab(k) in PLANT_LABELS]
    new_keys = set(z["new"].files)
    empty = [f"{t}:{k}" for t in z for k in (ref_keys if t == "ref" else z[t].files) if z[t][k].size == 0]
    missing = sorted(set(ref_keys) - new_keys)
    bad = [k for k in ref_keys if k in new_keys and not (z["ref"][k].shape == z["new"][k].shape and z["ref"][k].dtype == z["new"][k].dtype
                                                          and np.array_equal(z["ref"][k], z["new"][k], equal_nan=True))]
    extra = sorted(new_keys - set(ref_keys))
    other_label = [k for k in extra if lab(k) not in PLANT_LABELS]
    not_plant, shape_bad, d0_bad, by_d, d0_stems, row_stems, pool_bad = [], [], [], {}, set(), set(), []
    F = z["new"]

    mem_ = {}  # <label>__<combo>__joint__<f> -> the combo's member origins (those with a D array)
    for k_ in F.files:
        q_ = k_.split("__")
        if len(q_) == 6 and q_[5] == "D" and q_[4] != "pooled":
            mem_.setdefault("__".join(q_[:4]), set()).add(q_[4])

    def members(stem5):  # per-origin keys of a pooled stem's combo: <label>__<combo>__joint__<f>__<origin>
        return sorted(mem_.get("__".join(stem5.split("__")[:4]), ()))

    def comb(stem5, name):  # pool_reduce_n's arithmetic for this combo: the all-origins combos drop an origin where its D is 0
        os_ = members(stem5)
        pre = "__".join(stem5.split("__")[:4])
        D = [F[f"{pre}__{o_}__D"] for o_ in os_]
        X = [F[f"{pre}__{o_}__{name}"] for o_ in os_]
        Dd = sum(D)
        with np.errstate(invalid="ignore", divide="ignore"):
            if stem5.split("__")[1] in ALLO:
                return sum(np.where(d > 0, x * d, 0.0) for x, d in zip(X, D)) / Dd
            return sum(x * d for x, d in zip(X, D)) / Dd

    def close(a, b):
        return a.shape == b.shape and a.dtype == b.dtype and np.allclose(a, b, rtol=0, atol=1e-12, equal_nan=True)

    for k in extra:  # keys: <label>__<combo>__<mode>__<f>__<origin | pooled>__<name>
        pt = k.split("__")
        if len(pt) != 6 or pt[2] != "joint":
            not_plant.append(k)
            continue
        stem5 = "__".join(pt[:5])
        m = ARM_CATCH.match(pt[5])
        if m is None and pt[4] == "pooled":
            m = POOL_GAP.match(pt[5])
            if m is not None:  # pooled planted gap = pooled planted catch - pooled real mean (recomputed from the members)
                arm_ = m.group("arm") or "sim"
                ck = f"{stem5}__catch_{arm_}_plant_{m.group('d')}"
                try:
                    rm_ = comb(stem5, "catch_real_mean_at_B")
                    ok_ = ck in F.files and close(F[k], F[ck] - rm_)
                except KeyError as e:
                    ok_ = False
                    pool_bad.append(f"{k}: {e!r}")
                if not ok_:
                    pool_bad.append(k)
                if m.group("d") == "+0" and arm_ == "sim":
                    base = stem5 + "__gap_delta_full"
                    if base not in F.files or not np.array_equal(F[k], F[base], equal_nan=True):
                        d0_bad.append(k)
        if m is None:
            not_plant.append(k)
            continue
        by_d[m.group("d")] = by_d.get(m.group("d"), 0) + 1
        if pt[4] == "pooled" and ARM_CATCH.match(pt[5]):  # pooled planted catch = the members' per-origin planted catches, combined
            try:
                if not close(F[k], comb(stem5, pt[5])):
                    pool_bad.append(k)
            except KeyError as e:
                pool_bad.append(f"{k}: {e!r}")
        elif pt[4] != "pooled":
            row_stems.add((stem5, m.group("arm")))
            if m.group("arm") == "sim":
                base = stem5 + "__catch_sim_at_S_primary"
                if base not in F.files or F[k].shape != F[base].shape or F[k].dtype != F[base].dtype:
                    shape_bad.append(k)
                elif m.group("d") == "+0":
                    d0_stems.add(stem5)
                    if not np.array_equal(F[k], F[base], equal_nan=True):
                        d0_bad.append(k)
    want = set()
    for k in ref_keys:
        pt = k.split("__")
        if len(pt) == 6 and pt[2] == "joint" and pt[3] in ("0.01", "0.05") and pt[4] != "pooled" and pt[5] == "catch_sim_at_S_primary" and "registered_cohort" not in pt[4]:
            want.add("__".join(pt[:5]))
    unexpected = sorted(st for st, arm_ in row_stems if arm_ == "sim" and st not in want)  # plants on registered rows or other levels
    unav = {(a, b, c) for a, b, c, arm_ in unavailable if arm_ == "sim"}
    miss_d0 = sorted(s_ for s_ in want - d0_stems if (s_.split("__")[0], s_.split("__")[4], s_.split("__")[3]) not in unav)
    return {"ref": ref, "new": new, "ref_keys_plant_labels": len(ref_keys), "new_keys": len(new_keys), "compared_exact": len(ref_keys) - len(missing),
            "mismatch": bad[:40], "n_mismatch": len(bad), "missing_in_new": missing[:40], "n_missing_in_new": len(missing),
            "new_keys_of_other_labels": other_label[:20], "new_non_plant_keys": not_plant[:40], "plant_arrays_by_delta": by_d,
            "expected_planted_rows": len(want), "rows_with_delta0": len(d0_stems), "unavailable_rows_listed": len(unav),
            "expected_without_delta0": miss_d0[:20], "n_expected_without_delta0": len(miss_d0), "delta0_not_equal_base": d0_bad[:20],
            "shape_or_dtype_differs_from_base": shape_bad[:20], "empty_arrays": empty[:20], "unexpected_planted_rows": unexpected[:20],
            "pooled_plant_arrays_not_reproduced": pool_bad[:20],
            "pass": not (bad or missing or other_label or not_plant or d0_bad or shape_bad or empty or miss_d0 or unexpected or pool_bad)
                    and len(want) > 0 and len(d0_stems) > 0}


@app.local_entrypoint()
def main(ref: str, new: str, record: str, out: str):
    p_, h_ = record.rsplit("@", 1)
    b = (Path(__file__).resolve().parent / p_).read_bytes()
    if hashlib.sha256(b).hexdigest() != h_:
        raise SystemExit("plant record sha256 differs")
    rec = json.loads(b)
    unav = sorted({(npz_part(lb), npz_part(og), f, arm) for lb, oo in rec["rows"].items() for og, o in oo.items()  # solve_draw0[<f>][<arm>]
                   for f, byarm in ((o.get("plant") or {}).get("solve_draw0") or {}).items() for arm, sv in byarm.items()
                   if str(sv.get("status", "")).startswith("unavailable")})
    fl_ = sorted({npz_part(lb) for lb in rec["rows"]}) if (rec.get("meta") or {}).get("full_length") else None
    res = check.remote(ref, new, unav, fl_)
    res["record"] = record
    blob = json.dumps(res, indent=1).encode()
    (Path(__file__).resolve().parent / out).open("xb").write(blob)
    print(json.dumps({k: v for k, v in res.items() if k not in ("mismatch", "missing_in_new")} | {"file": out, "sha256": hashlib.sha256(blob).hexdigest()}, indent=1))
    if not res["pass"]:
        raise SystemExit(1)
