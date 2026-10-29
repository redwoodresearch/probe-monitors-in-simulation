"""PANEL's drop evidence (fold -1): per fold-routed probe label and origin, the attack cells that MAY be dropped by row()'s fold -1 rule. KEYS ONLY.

Source: a committed keys-only census record (run_peraction311.py --keys-only on the same INPUTS manifest; nofilter_keys_*.json) whose rows carry
attack_fold_minus1_cells_by_source_world (the census row's U["fold"] == -1 attack units, after the loaders and the DT unmapped rule; fold is
routing metadata, not a score). row() drops a cohort cell only when it has an attack fold -1 unit AND the 21b test fails (25h §1; per-pool 21b
for Sol (a)/c_asis), so the list is the may-drop SUPERSET: whether a cell actually drops depends on the pools' fold -1 unit counts, which the
job decides; the job's --fold-drops hook STOPs if an actual drop is not listed here. Origins: the census source (S_old, S_new, S_newmains,
S_paper493; "relaunch" folded into S_old); a row's origin is the first word of its tag (all-origins / registered / without-S4 rows included).
Covered origins per label = the census sources with attack rows (an empty list = nothing may drop there); the job checks covered pairs only.

  python make_fold_drops.py results/nofilter_keys_<stamp>.json [--window cut32768|full]   # writes results/fold_minus1_drops_<stamp>.json
"""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RULE = ("row() (run_peraction311.py): a cohort cell is dropped from every pool and the cohort when one of its attack runs (run1, run2, sim) has a "
        "fold -1 unit and the weighted 21b does not apply (25h §1 @ 0f9927f3e; per-pool 21b for Sol (a)/c_asis and DT; 21b zero-weight "
        "clarification @ 7310476ae; without-S4 rows take the with-S4 cohort's fold -1 columns, 09 @ 51a612b29). Listed here: every cell with an "
        "attack fold -1 unit (the may-drop superset; keys only).")
REQUIRED = ("Sol (a)", "c_asis")  # PANEL's dedup-combo members that state fold -1 drops (sol_a_uncut in the full-length build)
# full length (census labels in full_override's namespace): the registered Sol (a) / c_asis scored uncut and their retrains
REQUIRED_FULL = ("Sol (a)|uncut", "c_asis|uncut", "sol_a_uncut", "sol_a_uncut_pinned", "c_asis_uncut")
SOURCE_ORIGIN = {"S_old": "S_old", "relaunch": "S_old", "S_new": "S_new", "S_newmains": "S_newmains", "S_paper493": "S_paper493"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("census")
    ap.add_argument("--window", default="cut32768", choices=("cut32768", "full"))
    ap.add_argument("--require-s-paper493", action=argparse.BooleanOptionalAction, default=True)
    a = ap.parse_args()
    cp = (HERE / a.census).resolve()
    rel = str(cp.relative_to(REPO))
    b = cp.read_bytes()
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, check=True, cwd=REPO).stdout.decode().strip()
    if subprocess.run(["git", "show", f"HEAD:{rel}"], capture_output=True, check=True, cwd=REPO).stdout != b:
        raise SystemExit("census: not committed at HEAD as read")
    cen = json.loads(b)
    inp = cen["meta"].get("inputs_final") or {}
    if not inp.get("sha256"):
        raise SystemExit("census: meta.inputs_final has no sha256")
    drops, counts, single, missing = {}, {}, [], []
    for lb, by_o in sorted(cen["rows"].items()):
        for o, r in by_o.items():
            f1 = r.get("attack_fold_minus1_cells_by_source_world")
            if f1 is None:
                missing.append(f"{lb} / {o}")
                continue
            if isinstance(f1, str):  # single-model probe: no fold routing, nothing can drop
                single.append(lb)
                continue
            for s_ in r.get("attack_rows_by_source_world_cell", {}):  # covered origins: the census read attack rows there (an empty list = none may drop)
                if s_ not in SOURCE_ORIGIN:
                    raise SystemExit(f"{lb}: unknown census source {s_}")
                drops.setdefault(lb, {}).setdefault(SOURCE_ORIGIN[s_], set())
            for s_, ws in f1.items():
                og = SOURCE_ORIGIN.get(s_)
                if og is None:
                    raise SystemExit(f"{lb}: unknown census source {s_}")
                d = drops.setdefault(lb, {}).setdefault(og, set())
                for w_, cells in ws.items():
                    d |= set(cells)
    if missing:
        raise SystemExit(f"census rows without the fold -1 field (census predates the drop-evidence patch?): {missing[:5]}")
    need = {"S_old", "S_new", "S_newmains"} | ({"S_paper493"} if a.require_s_paper493 else set())
    fl_ = (cen["meta"].get("full_length") or {})
    if (a.window == "full") != bool(fl_):  # the full-length drops come from the UNCUT census (--inputs-full --keys-only), and only from it
        raise SystemExit(f"census: window {a.window} but the census is {'full-length' if fl_ else '32k'}")
    req_f = REQUIRED_FULL if a.window == "full" else REQUIRED
    req = sorted(lb for lb in cen["rows"] if lb.rsplit(" ", 1)[0] in req_f)  # every census label of a required family, by name (Codex r2)
    if not req:
        raise SystemExit("no Sol (a) / c_asis label in the census: the drop evidence would be vacuous")
    short = {lb: sorted(need - set(drops.get(lb, {}))) for lb in req if need - set(drops.get(lb, {}))}
    if short:  # PANEL's dedup combos need Sol (a) / c_asis evidence on every origin the combos use
        raise SystemExit(f"Sol (a) / c_asis labels without census coverage on an origin: {short}")
    out = {"what": "fold -1 drop evidence (PANEL): cells each fold-routed probe label may drop, by origin; keys only",
           "rule": RULE, "generator": "make_fold_drops.py",
           "stamp_pt": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S"),
           "census": {"commit": head, "path": rel, "sha256": hashlib.sha256(b).hexdigest()},
           "inputs": {"path": inp.get("path"), "sha256": inp["sha256"]},
           "fold_sources": "the probes' score files as the census row read them (fold column; the INPUTS manifest pins every file by sha256)",
           "window": a.window,
           "single_model_labels": sorted(set(single)),
           "required_families": list(req_f), "census_full_length_overrides": fl_.get("overrides"),
           "check_rule": ("the job STOPs when a row's actual fold -1 drops are not in drops[<label>][<origin>]; for the required families a missing "
                          "label x origin lists nothing (fail closed); other labels are checked on the label x origin pairs the census covered"),
           "drops": {lb: {og: sorted(v) for og, v in sorted(d.items())} for lb, d in sorted(drops.items())},
           "counts": {lb: {og: len(v) for og, v in sorted(d.items())} for lb, d in sorted(drops.items())}}
    dest = HERE / "results" / f"fold_minus1_drops{'_TEST' if 'TEST' in rel else ''}_{out['stamp_pt']}.json"
    blob = json.dumps(out, indent=1).encode()
    dest.open("xb").write(blob)
    print(json.dumps({"file": str(dest.relative_to(HERE)), "sha256": hashlib.sha256(blob).hexdigest(), "counts": out["counts"],
                      "single_model_labels": out["single_model_labels"]}, indent=1))


if __name__ == "__main__":
    main()
