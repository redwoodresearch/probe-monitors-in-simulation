"""The width-only all-runs selection (09 RULE @ 3aaf425ef §1, amended @ 36b9b47df §3): HW(A) and HW(B) of the six-probe signed S at C, 1e-2,
joint 95%, from the full-length Stage A all-tasks job's per-draw npz (never leaves the Volume). Lead = B iff HW(B) <= 0.95 x HW(A).
The record holds ONLY the two half-widths, their ratio, the verdict and the coverage counts (per probe x set: tasks with sim2, with run3, with
neither, read from the record's rows[<label>][<origin>].estimator_B counts). No point estimate and no interval endpoint is computed into the output,
printed or stored. Commit the selection record BEFORE any point estimate is printed or rendered; READ_LOG records its sha first.

  modal run select_estimator.py --record results/alltasks_CRD_<stamp>.json@<sha> [--combo pooled_all_origins] [--group headline_six_full] [--test]
  # writes results/estimator_selection_<stamp>.json (exclusive create)
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-fpr507prep_estsel_kill-2026-11-14"
vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13", version=2)
app = modal.App(APP, image=modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3"))
RULE = "09 RULE @ 3aaf425ef §1 (width-only adopt rule), amended @ 36b9b47df §3 (both computed in the full-length Stage A; selection record first)"
THRESHOLD = 0.95
B_NOT_EVALUABLE = "(B) not evaluable at the freeze: IL extra-arm overrides not pinned (build omission)"  # 09 RULE item 40, verbatim


def npzkey(*parts):  # run_peraction311.npzkey (v2)
    out = []
    for p_ in parts:
        q = str(p_).replace("@", "_at_").replace("|", "_").replace(" ", "_").replace("(", "").replace(")", "")
        if "__" in q or not q:
            raise SystemExit(f"npz key part {p_!r} would be ambiguous")
        out.append(q)
    return "__".join(out)


@app.function(volumes={"/sol": vol.read_only()}, cpu=1, memory=4096, timeout=900, retries=0)
def widths(npz_path: str, npz_sha: str, keys_a: list, keys_b: list) -> dict:
    import io

    import numpy as np

    b = (Path("/sol") / npz_path).read_bytes()
    if hashlib.sha256(b).hexdigest() != npz_sha:
        raise SystemExit("npz sha256 differs from the record")
    z = np.load(io.BytesIO(b))
    miss = [k for k in keys_a + keys_b if k not in z.files]
    if miss:
        raise SystemExit(f"npz lacks {miss[:4]}")

    def hw(keys):  # the records' summ(): percentiles over the finite draws 1..D; a withheld interval (< 95% finite) STOPs
        arrs_ = [np.asarray(z[k], dtype=np.float64) for k in keys]
        if len({a_.shape for a_ in arrs_}) != 1 or arrs_[0].ndim != 1 or arrs_[0].shape[0] < 2:
            raise SystemExit("the six arrays differ in shape or are not per-draw vectors (draw 0 + draws 1..D)")
        S = np.mean(arrs_, 0)
        draws = S.shape[0] - 1
        f = S[1:][np.isfinite(S[1:])]
        if len(f) < 0.95 * draws:
            raise SystemExit("fewer than 95% complete draws: the interval is withheld")
        return float((np.percentile(f, 97.5) - np.percentile(f, 2.5)) / 2), int(len(f))  # only the half-width leaves this function

    (ha, fa) = hw(keys_a)
    if not keys_b:  # 09 RULE item 40: (B) not evaluable at the freeze; only HW(A) is computed
        return {"HW_A": ha, "HW_B": None, "finite_draws": {"A": fa, "B": None}}
    (hb, fb) = hw(keys_b)
    return {"HW_A": ha, "HW_B": hb, "finite_draws": {"A": fa, "B": fb}}


@app.local_entrypoint()
def main(record: str, combo: str = "pooled_all_origins", group: str = "headline_six_full", test: bool = False, b_not_evaluable: bool = False):
    """b_not_evaluable (09 RULE item 40): the freeze pinned no IL extra-arm overrides, so (B) fell back to (A) on every task; the record reads (A),
    computes HW(A) only and carries B_NOT_EVALUABLE verbatim. Refused unless the record's coverage counts show 0 tasks on extra arms everywhere."""
    f, mode = "0.01", "joint"  # fixed by the rule: the six-probe S at C, 1e-2, joint 95%
    import subprocess

    rp, rh = record.rsplit("@", 1)
    rb = (HERE / rp).read_bytes()
    if hashlib.sha256(rb).hexdigest() != rh:
        raise SystemExit("record sha256 differs")
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rp], capture_output=True, cwd=HERE).returncode == 0
    if not tracked:
        raise SystemExit("the Stage A record must be committed before the selection")
    rec = json.loads(rb)
    fl_ = (rec["meta"].get("full_length") or {}).get("inputs_full") or {}
    if not test and (not fl_ or fl_.get("test")):  # the lead decision is made on the frozen full-length build only (--test: a TEST record, labelled)
        raise SystemExit("not a frozen full-length record (meta.full_length.inputs_full)")
    try:  # the six members as the record's reducer grouped them (labels only; the group's values are not read)
        labs = list(rec["coheadline_signed"][combo][mode][f][f"{group}|C"]["rows"])
    except (KeyError, TypeError):
        raise SystemExit(f"the record has no coheadline group {group}|C on {combo} {mode} {f}")
    if len(labs) != 6 or len(set(labs)) != 6 or not all(lb.endswith(" C") for lb in labs):
        raise SystemExit(f"six distinct C-readout labels expected, the group lists {labs}")
    est = rec["meta"].get("estimator") or {}
    if "B" not in est.get("computed", []):
        raise SystemExit("the record did not compute (B)")
    cov = {}
    for lb in labs:  # coverage counts only, per probe and set: tasks on extra arms vs (A) fallback; extra-arm keys admitted / routed / scored / not built
        for o_, row in (rec["rows"].get(lb) or {}).items():
            e_ = row.get("estimator_B") if isinstance(row, dict) else None
            if e_:
                c_ = {k: e_[k] for k in ("tasks_with_sim2", "tasks_with_run3", "tasks_with_neither", "cohort_cells")} | (
                    {"label": e_["label"]} if "label" in e_ else {})
                c_["tasks_B_used_extra_arms"] = e_["cohort_cells"] - e_["tasks_with_neither"]
                c_["tasks_B_fell_back_to_A"] = e_["tasks_with_neither"]
                if isinstance(row.get("il_arms"), dict) and "counts" in row["il_arms"]:
                    c_["extra_arm_keys"] = {k: row["il_arms"]["counts"][k] for k in ("admitted", "routed", "scored", "in_scope_not_built")}
                cov.setdefault(lb, {})[o_] = c_
    ila_ = rec["meta"].get("il_arms") or {}
    if not ila_.get("block"):
        raise SystemExit("the record names no ilarms block (meta.il_arms.block)")
    ka = [npzkey(lb, combo, mode, f, "pooled", "gap_delta_full") for lb in labs]
    kb = [npzkey(lb, combo, mode, f, "pooled", "gap_delta_full|B") for lb in labs]
    if b_not_evaluable:  # every task fell back to (A) by the record's own counts, else this mode is refused
        used_ = {f"{lb} | {o_}": c_["tasks_B_used_extra_arms"] for lb, d_ in cov.items() for o_, c_ in d_.items()}
        if not used_ or any(used_.values()):
            raise SystemExit(f"--b-not-evaluable: the record shows tasks on extra arms (or no coverage counts): { {k: v for k, v in used_.items() if v} }")
        kb = []
    w = widths.remote(rec["npz"]["path"], rec["npz"]["sha256"], ka, kb)
    ratio = (w["HW_B"] / w["HW_A"] if w["HW_A"] > 0 else None) if w["HW_B"] is not None else None  # a degenerate (A) interval: ratio undefined
    out = {"label": "all-runs selection record (width only; no point estimate)" + ("; TEST" if test else ""), "rule": RULE, "threshold": THRESHOLD,
           "group": group,
           "record": {"path": rp, "sha256": rh}, "npz": rec["npz"], "combo": combo, "mode": mode, "f": f, "labels": labs,
           "HW_A": w["HW_A"], "HW_B": w["HW_B"], "ratio": ratio,
           "lead": "A" if b_not_evaluable else "B" if w["HW_B"] <= THRESHOLD * w["HW_A"] else "A",
           **({"note": B_NOT_EVALUABLE, "rule_b_not_evaluable": "09 RULE item 40: a build failure (not item 20's missing-arm fallback); the (B) + RMS "
                                                                   "addendum is a labelled update on the same frozen draws and union stream; no headline "
                                                                   "switch after (A)'s points are read"} if b_not_evaluable else {}),
           "finite_draws": w["finite_draws"], "coverage_counts": cov,
           "freeze_moment": {"ilarms_block": ila_["block"], "block_built_pt": ila_.get("built_pt"),
                             "rule": "inclusion by availability at the freeze, never by value; nothing added after this record; later landings = a labelled update"},
           "stamp_pt": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S"),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    blob = json.dumps(out, indent=1).encode()
    dest = HERE / "results" / f"estimator_selection{'_TEST' if test else ''}_{out['stamp_pt']}.json"
    dest.open("xb").write(blob)
    print(json.dumps({"file": dest.name, "sha256": hashlib.sha256(blob).hexdigest(), "HW_A": out["HW_A"], "HW_B": out["HW_B"], "ratio": ratio,
                      "lead": out["lead"]}, indent=1))
