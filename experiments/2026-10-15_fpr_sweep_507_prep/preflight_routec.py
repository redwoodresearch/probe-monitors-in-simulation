"""Route C pre-flight (09 / 2e, 10-17; keys only, cents on Modal CPU): for the drive spec a paid route C run would send (run_peraction311.py
--spec-only with the route C INPUTS_FULL), every row-time STOP risk and every filter drop of the 20 pilot sim-honest keys, BEFORE any paid run.
It reads key arrays, fold arrays (model indices), cell maps and fold lists; never a score. Output: keys and counts only.

Per job carrying the S_old-only (route C) entry, i.e. every S_old job (per-set, merged and pooled variants alike):
- extra keys in the pilot file (the row STOPs: FO.s_old_extra_check); admitted keys without rows: zero-unit members (09: listed, NOT a STOP);
- pilot cells not in the row's cell -> main map (rp:1744 STOPs): the merge map for an all-origins job, else S_old's T196 cells;
- folds, by the row's own path:
  * a fold-model file with a fold array: its fold -1 pilot units are dropped by 25h §1 (09 Append 25h @ 0f9927f3e §1; rp "fold -1 dropped
    (stated deviation)") unless the row applies 21b; in a pooled job of a NO_FOLD_DROPS family the run STOPs instead (#6);
  * Sol / c_asis files without a fold array (the A25 path, rp:1727): each pilot task needs a real-side fold >= 0 in the S_old real files the job
    reads (else the row STOPs);
  * own_all_rs: each pilot main must be in the job's own_all_rs_fold_of_main and every unit on that fold (rs_fold_check STOPs otherwise);
- the cap (30 / taskmin) keeps <= 30 units per run, so it drops no run; S_old's exclusion map is empty (rp:1359), so no exclusion drops a key.
A tagged entry on any non-S_old job is a DEFECT (the build drops it). Non-own tagged entries (09's ruling (b), 10-18): another label's entry
inside an S_old job's simh_old, whose file no job reads as its own (a pooled spec), is read by key too; its zero-unit list joins
zero_unit_by_file (the union the gate reads) and is annotated in zero_unit_by_file_nonown (omitted when empty; labels, n_admitted, n_with_rows, n_zero_unit,
the zero-unit keys by name); a file own for one job and non-own for another must have the identical list (else a DEFECT); a non-own entry
gives no drop and no stop risk (a row reads only its own label's list), but keys outside its admitted ones are a DEFECT naming them (09's
extra-key rule: a data-identity defect, whichever label reads the file). Plus, locally: each pilot task's T196 main is in the fold map
@ ec56720ad (901 mains; 09 asserts coverage by key), with its fold and basis.
Any STOP risk, any drop without a citable registered filter, or a pilot main outside the fold map is a DEFECT: the record is written first
(exclusive create), then the script exits 1.

  modal run experiments/2026-10-15_fpr_sweep_507_prep/preflight_routec.py --spec <route C spec-only .pkl> --out results/routec_preflight/<name>.json
"""

import hashlib
import json
import pickle
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
TAG = "s_old_only"  # = full_override.S_OLD_ONLY (the test asserts it)
VOL_CONV = "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14"  # = run_common455.VOLS["conv"] (the test asserts it)
FOLDS = ("ec56720ad", "experiments/2026-10-16_own_all_rs/results/own_all_rs_folds_2026-10-16.json",
         "c2fc5d40eb84a34a64cbb9ce7155572d1414f12ea1fbc2ec05708bd73061e299")
T196 = ("debef0af2", "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json",
        "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb")
FOLD_RULE = "25h §1 fold -1 drop (09 Append 25h amendment @ 0f9927f3e §1; the row's fold_minus1_rule 'fold -1 dropped (stated deviation)')"
app = modal.App("probe-shift_aswin-surya_sub-fpr507prep_rcpf_kill-2026-11-14")  # <= 63 chars (0d, 10-17: Modal cuts a description at 64)
vol = modal.Volume.from_name(VOL_CONV, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")


def analyze(spec, read_kf, main_t, no_fold_drops):
    """Pure: spec {jobs: [(label, cfg, blobs)]}; read_kf(vol, path, sha256) -> (keys, folds or None) of one per-row file (sha-checked by the
    reader); main_t: S_old's T196 cell -> main; no_fold_drops: FO.NO_FOLD_DROPS. -> {jobs, defects, drops, zero_unit_by_job}."""
    out = {"jobs": [], "defects": [], "drops": [], "zero_unit_by_job": {}, "zero_unit_by_file": {}}
    for i, (lb, cfg, _) in enumerate(spec["jobs"]):
        ents = [e for es in (cfg.get("simh_old") or {}).values() for e in es if e.get(TAG)]
        if not ents:
            continue
        orig, fam = cfg.get("origin"), cfg.get("full_as", lb).rsplit(" ", 1)[0]  # the runner's no-fold-drop check reads full_as (rp:1931; Claude r1 R1)
        jid = f"jobs/{i} {cfg.get('full_as', lb)} | {orig} | {cfg.get('pw_combo') or cfg.get('cohort') or ''}{' merge' if cfg.get('merge') else ''}"
        lc = (f"{cfg.get('full_as', lb)} | {cfg['pw_combo']}" if cfg.get("pw_combo")  # gate v7's label x combo name
              else f"{cfg.get('full_as', lb)} | {orig} | {cfg.get('cohort') or ''}{' merge' if cfg.get('merge') else ''}")
        if orig != "S_old":
            out["defects"].append(f"{jid}: an S_old-only entry on a non-S_old job (the build should have dropped it)")
            continue
        own = [e for e in (cfg.get("simh_old") or {}).get(lb, []) if e.get(TAG)]
        p = next((q for q in cfg.get("probes") or [] if q.get("label") == lb), None)
        if p is None or len(own) != 1:
            out["defects"].append(f"{jid}: no probe for the label, or not exactly one S_old-only entry for it")
            continue
        e = own[0]
        adm = set(e["admitted"])
        if e.get("vol", "conv") != "conv":
            out["defects"].append(f"{jid}: the pilot entry is not on the conv Volume the row reads")
            continue
        keys, folds = read_kf("conv", e["path"], e["sha256"])
        got = set(keys)
        c2m = dict((cfg.get("merge") or {}).get("main_of_cell") or {}) if cfg.get("merge") else {c: c for c in main_t}  # the row's moa (rp:1678 / 1701)
        moa = set(c2m)
        j = {"job": jid, "label_combo": lc, "root": e["root"], "pilot_keys_with_rows": len(got & adm), "extra_keys": sorted(got - adm), "zero_unit": sorted(adm - got),
             "cells_not_in_map": sorted({k.split("|", 1)[1] for k in got} - moa), "fold_minus1_units": {}, "stop_risks": []}
        if j["extra_keys"]:
            j["stop_risks"].append("extra keys in the pilot file (FO.s_old_extra_check STOPs)")
        if j["cells_not_in_map"]:
            j["stop_risks"].append("pilot cells outside the row's cell -> main map (rp:1744 STOPs)")
        fold_model = p.get("folds") == 5
        if fold_model and p.get("family") == "own_all_rs":
            fom = cfg.get("own_all_rs_fold_of_main") or {}
            mn = {k: main_t.get(k.split("|", 1)[1]) for k in got}  # rs_fold_check: each unit on its main's fold in the committed list
            miss = sorted({k.split("|", 1)[1] for k in got if mn[k] not in fom})
            off = sorted({k for k, f in zip(keys, folds or []) if mn.get(k) in fom and int(f) != int(fom[mn[k]])})
            if miss or off or folds is None:
                j["stop_risks"].append(f"own_all_rs: pilot mains outside the fold list {miss[:3]} or units off their main's fold {off[:3]} (rs_fold_check)")
        elif fold_model and folds is not None:
            f1, nu, rk = {}, {}, {}
            capped = cfg.get("cap") in (30, "taskmin")  # Astra r1 R1: the row caps BEFORE the fold -1 drop (rp: kH = rank_in_run(...) < 30)
            for k, f in zip(keys, folds):  # file order = the row's order (the pilot's keys are its root's alone, so ranks are within this file)
                r = rk.get(k, 0)
                rk[k] = r + 1
                if capped and r >= 30:
                    continue
                nu[k] = nu.get(k, 0) + 1
                if int(f) == -1 and k in adm:
                    f1[k] = f1.get(k, 0) + 1
            j["fold_minus1_units"] = {k: {"fold_minus1": n, "of": nu[k], "run_dropped": n == nu[k]} for k, n in sorted(f1.items())}
            c21 = lb in (cfg.get("use21b") or []) or lb in (cfg.get("per_pool_21b") or [])  # the row may apply 21b (rp:1902 / 1905: by its pool counts)
            if f1 and cfg.get("pw_combo") and fam in no_fold_drops:
                j["stop_risks"].append(f"fold -1 pilot units in a pooled NO_FOLD_DROPS family ({len(f1)} keys; #6 STOPs the run"
                                       + (", unless the row applies 21b)" if c21 else ")"))
            elif f1:
                out["drops"] += [{"job": jid, "label_combo": lc, "key": k, "fold_minus1_units": n, "of_units": nu[k], "run_dropped": n == nu[k],
                                  "filter": FOLD_RULE, "conditional_on_21b": c21,
                                  "unless": "the row applies 21b (label in use21b / per_pool_21b, >= 2 fold -1 units in every pool)"} for k, n in sorted(f1.items())]
        elif fold_model:  # the A25 path: the task's real-side fold from the S_old real files this job reads
            real = {}
            for f_ in ((p.get("old") or {}).get("files") or {}).values():
                rk, rf = read_kf(f_.get("vol") or "conv", f_["path"], f_["sha256"])
                for k, f in zip(rk, rf or []):
                    if int(f) >= 0:
                        real.setdefault(k.split("|", 1)[1], set()).add(int(f))
            nof = sorted({k.split("|", 1)[1] for k in got} - set(real))
            j["a25_tasks_without_real_fold"] = nof
            if nof or not (p.get("old") or {}).get("files"):
                j["stop_risks"].append(f"A25 path: pilot tasks without a real-side fold >= 0 {nof[:3]} (rp:1727 STOPs)")
        out["zero_unit_by_job"][jid] = j["zero_unit"]
        if out["zero_unit_by_file"].setdefault(e["sha256"], j["zero_unit"]) != j["zero_unit"]:  # the list the runner pins (--routec-zero-unit)
            out["defects"].append(f"{jid}: one pilot file, two zero-unit lists")
        out["defects"] += [f"{jid}: {r}" for r in j["stop_risks"]]
        out["jobs"].append(j)
    # 09's ruling (b), 10-18: every S_old job carries every label's S_old list (full_override.full_jobs L474, simh_old_for), but a pooled
    # spec has S_old jobs only for the PW_FAMS C / R labels (rp L6748), so some tagged files are no job's own. The gate checks every tagged
    # entry against zero_unit_by_file (gate_spec_diff_routec L61), so that map is the UNION of the own files above and these non-own files;
    # zero_unit_by_file_nonown annotates the non-own ones separately. Each non-own file is read here by key (read_kf, sha-checked against its
    # pin), never inferred from another file; "admitted" is per pilot root, the same for every label (full_override.s_old_simh L256:
    # ext_[e["root"]]). A non-own entry gives no drop and no stop risk: a row reads only its own label's list (rp L1751-1752). A file that is
    # own for one job and non-own for another must have the IDENTICAL zero-unit set on both sides (09), else a DEFECT. A non-own file
    # with keys outside its admitted ones is a DEFECT that names them (09's extra-key rule).
    own_files, nonown = set(out["zero_unit_by_file"]), {}
    for i, (lb, cfg, _) in enumerate(spec["jobs"]):
        if cfg.get("origin") != "S_old":
            continue
        for lb2, es in (cfg.get("simh_old") or {}).items():
            if lb2 == lb:
                continue
            for e in es:
                if not e.get(TAG):
                    continue
                at = f"jobs/{i} non-own {lb2}"
                if e.get("vol", "conv") != "conv":
                    out["defects"].append(f"{at}: a tagged entry not on the conv Volume (its keys cannot be read)")
                    continue
                adm, got = set(e["admitted"]), set(read_kf("conv", e["path"], e["sha256"])[0])
                zu = sorted(adm - got)
                if got - adm:  # 09's extra-key rule, 10-18: a pinned pilot-root file with keys outside its admitted ones is a data-identity defect
                    out["defects"].append(f"{at}: a non-own pilot file with {len(got - adm)} keys outside the {len(adm)} admitted: {sorted(got - adm)}")
                if e["sha256"] in own_files:
                    if out["zero_unit_by_file"][e["sha256"]] != zu:
                        out["defects"].append(f"{at}: a file own for one job and non-own for another, with different zero-unit sets")
                    continue
                x = nonown.setdefault(e["sha256"], {"labels": set(), "n_admitted": len(adm), "n_with_rows": len(adm & got), "n_zero_unit": len(zu),
                                                    "zero_unit": zu})
                if (x["n_admitted"], x["n_with_rows"], x["zero_unit"]) != (len(adm), len(adm & got), zu):
                    out["defects"].append(f"{at}: one pilot file, two zero-unit lists among its non-own entries")
                x["labels"].add(lb2)
    for h, x in sorted(nonown.items()):
        out["zero_unit_by_file"][h] = x["zero_unit"]
    if nonown:  # omitted when empty, so a spec without non-own files (PS) gets the reviewed record byte for byte (09 (iii))
        out["zero_unit_by_file_nonown"] = {h: x | {"labels": sorted(x["labels"])} for h, x in sorted(nonown.items())}
    if not out["jobs"]:
        out["defects"].append("no S_old job carries the S_old-only entry (is this the route C spec?)")
    # 507-prep (gate (iii)): the zero_unit_ok paths a pinned paid spec adds, counted the way spec_diff walks them: every S_old job's cfg carries
    # every label's S_old list, so each tagged entry in it contributes its own file's zero-unit keys once per job
    out["expected_zero_unit_ok_paths"] = sum(len(out["zero_unit_by_file"].get(e["sha256"], [])) for _, cfg, _ in spec["jobs"] if cfg.get("origin") == "S_old"
                                             for es in (cfg.get("simh_old") or {}).values() for e in es if e.get(TAG))
    # gate v7's schema (refresh2, 10-17): run-level drops only (a fold -1 drop of some of a run's units keeps the run), per label x combo
    by_run, all_lc = {}, sorted({j["label_combo"] for j in out["jobs"]})
    for d in out["drops"]:
        if d["run_dropped"]:
            by_run.setdefault((d["key"], d["conditional_on_21b"]), set()).add(d["label_combo"])
    out["gate_view"] = {"drops": [{"run": k, "filter": "fold -1 drop (25h §1)", "citation": "09 Append 25h amendment @ 0f9927f3e §1",
                                   "conditional_on_21b": c, "label_combos": "all" if sorted(lcs) == all_lc else sorted(lcs)}
                                  for (k, c), lcs in sorted(by_run.items())],
                        "zero_unit_pilot_runs": sorted({k for v in out["zero_unit_by_job"].values() for k in v})}
    return out


@app.function(image=image, cpu=(2, 4), memory=(4096, 32768), timeout=1800, retries=0, volumes={"/conv": vol.read_only()})
def preflight_remote(spec_b: bytes, main_t: dict, no_fold_drops: list) -> dict:
    import io

    import numpy as np

    cache = {}

    def read_kf(v, path, sha):
        if v != "conv":
            raise SystemExit(f"read outside the conv Volume: {v}")
        if (path, sha) not in cache:
            b = (Path("/conv") / path.lstrip("/")).read_bytes()
            if hashlib.sha256(b).hexdigest() != sha:
                raise SystemExit(f"{path}: sha256 differs")
            z = np.load(io.BytesIO(b))  # only the key and fold members are read
            cache[(path, sha)] = (z["key"].astype(str).tolist(), z["fold"].astype(np.int64).tolist() if "fold" in z.files else None)
        return cache[(path, sha)]

    return analyze(pickle.loads(spec_b), read_kf, main_t, tuple(no_fold_drops))


@app.local_entrypoint()
def main(spec: str, out: str):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import full_override as FO

    git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout  # noqa: E731
    dest = HERE / out
    if dest.exists() or not out.startswith("results/routec_preflight/"):
        raise SystemExit("--out results/routec_preflight/<new name>.json")
    spec_b = Path(spec).read_bytes()
    pins = {}
    for nm, (c, p, h) in {"fold_map": FOLDS, "t196": T196}.items():
        b = git("show", f"{c}:{p}")
        if hashlib.sha256(b).hexdigest() != h:
            raise SystemExit(f"{nm}: sha256 differs")
        pins[nm] = json.loads(b)
    main_t = {c["key"]: c["main"] for c in pins["t196"]["cells"]}
    by_main = pins["fold_map"]["by_main"]
    xpin = next(iter(FO.S_OLD_EXTRA.values()))
    xb = git("show", f"{xpin['commit']}:{xpin['path']}")
    if hashlib.sha256(xb).hexdigest() != xpin["sha256"]:  # Claude r1 NB: the counted keys by sha
        raise SystemExit("counted keys: sha256 differs from the approved pin")
    pilot = sorted({k.split("|", 1)[1] for k in FO.s_old_extra_keys(json.loads(xb))})
    fm = {t: {"main": main_t.get(t), "fold": (by_main.get(main_t.get(t)) or [None, None])[0], "basis": (by_main.get(main_t.get(t)) or [None, None])[1]}
          for t in pilot}
    try:
        res = preflight_remote.remote(spec_b, main_t, list(FO.NO_FOLD_DROPS))
    except Exception as x:  # noqa: BLE001  Claude r1 NB: a crash still leaves a record (then exits 1); nothing is scored as clear
        res = {"jobs": [], "drops": [], "zero_unit_by_job": {}, "zero_unit_by_file": {}, "gate_view": None,
               "defects": [f"CRASH in the container: {type(x).__name__}: {str(x)[:300]}"]}
    res["defects"] += [f"pilot task {t}: its T196 main {v['main']} is not in the fold map @ {FOLDS[0]} (report to 09)" for t, v in fm.items() if v["fold"] is None]
    rec = {"what": FO.ROUTEC_PREFLIGHT_WHAT,
           "spec": {"local": spec, "sha256": hashlib.sha256(spec_b).hexdigest()}, "git": git("rev-parse", "HEAD").decode().strip(),
           "pins": {"fold_map": dict(zip(("commit", "path", "sha256"), FOLDS)), "t196": dict(zip(("commit", "path", "sha256"), T196)),
                    "allowlist": FO.S_OLD_EXTRA},
           "made_pt": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S %Z"),
           "fold_map_pilot_mains": fm, "status": "DEFECT" if res["defects"] else "CLEAR",
           "gate_expectation": "S_old H_sim + exactly (20 - the listed drops) pilot keys per S_old row; zero-unit runs as members with 0 units"} | res
    blob = json.dumps(rec, indent=1).encode()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("xb") as fh:  # written before any non-zero exit
        fh.write(blob)
    print(json.dumps({"record": out, "sha256": hashlib.sha256(blob).hexdigest(), "status": rec["status"], "jobs": len(res["jobs"]),
                      "defects": len(res["defects"]), "drops": len(res["drops"]),
                      "zero_unit_keys": len({k for v in res["zero_unit_by_job"].values() for k in v})}, indent=1))
    if res["defects"]:
        raise SystemExit(1)
