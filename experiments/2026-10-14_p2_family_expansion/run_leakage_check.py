"""0d's by-key leakage check (after the judge panel): for own_frac{10,25,100} x L16/32/48 x R/C x fold f (fit run 2026-10-14_175636),
training mains / run keys vs the mains / run keys model f scored in pass 1, attack runs' honest donors vs training run keys, and
row fold == fold_of_main in every pass-1 score file. KEYS AND COUNTS ONLY: score arrays are never loaded.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run --detach <this file>
"""

import hashlib
import json
import math
import random
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE  # inside the container the module sits at /root
APP = "probe-shift_aswin-surya_sub-p2probes_leakcheck_kill-2026-11-13"
SPEC, FIT = "2026-10-14_163027", "2026-10-14_175636"
PINS = {"spec": "5e2409a75edd6a92a1eae6c222e032911a17140bcefac80de28fdc4780b70b61", "fit": "024fbb0ac04067a08c6dabc49e524a1b2841ab090b2c7f4acdeaf41345034745"}
T196 = ("4d1eec572", "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json", "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb")
app = modal.App(APP)
VOLS = {"/out": modal.Volume.from_name("probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13", version=2).read_only(),
        "/b": modal.Volume.from_name("probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13", version=2).read_only()}
img = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")


@app.function(image=img, cpu=(4, 4), memory=(8192, 8192), timeout=600, retries=0, scaledown_window=2, volumes=VOLS)
def check(main_of: dict) -> dict:
    import numpy as np

    base = Path("/out/famexp") / SPEC
    sp, fp = base / f"spec_{SPEC}.json", base / f"fit_{FIT}.json"
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    if sha(sp) != PINS["spec"] or sha(fp) != PINS["fit"]:
        raise SystemExit("spec or fit manifest differs from its pin")
    spec, fit = json.loads(sp.read_text()), json.loads(fp.read_text())
    fot = spec["fot"]
    runkey = {k: f"{w}|{t}" for k, (w, t) in spec["map"].items()}
    ri = {"R": 0, "C": 1}
    rows_runs = {r: {w: {f"{w}|{t}" for k, (w_, t) in spec["map"].items() if w_ == w and spec["n_rows"][k][ri[r]] > 0} for w in ("run1", "honest_a")} for r in ri}
    # honest donors of the attack epochs, from the reader staging inputs (every field whose name contains "donor")
    donor_of, donor_fields, cfg = {}, set(), {}
    for k, (w, t) in spec["map"].items():
        if w not in ("run1", "run2", "sim"):
            continue
        run = spec["run_of"][k]
        if run not in cfg:
            c = json.loads((Path("/b") / run / "config.json").read_text())
            cfg[run] = {}
            for arm in c["inputs_sha256"]:
                for line in (Path("/b") / c["inputs"] / "inputs" / arm).read_text().splitlines():
                    if line:
                        e = json.loads(line)
                        cfg[run][e["key"]] = e
        e = cfg[run].get(k, {})
        refs = []
        for fk, fv in e.items():
            if "donor" in fk.lower():
                donor_fields.add(fk)
                for v in (fv if isinstance(fv, list) else [fv]):
                    if isinstance(v, dict):
                        v = v.get("key") or v.get("run") or v.get("path")
                    if isinstance(v, str):
                        refs.append(v)
        donor_of[runkey[k]] = refs
    # a donor reference resolves to a run key when it is (or names) an epoch of the map
    by_path = {}
    for k in spec["map"]:
        by_path[k] = runkey[k]
        by_path[k.removesuffix("#e0")] = runkey[k]
    res, totals = [], {"train_eval_main_overlap": 0, "train_eval_run_overlap": 0, "donor_in_training": 0, "fold_mismatch_rows": 0, "train_count_mismatch": 0,
                       "donor_refs_unresolved": 0}
    fits = [x for g in fit["groups"] for x in g["fits"] if x["variant"].startswith("own_frac")]
    for x in sorted(fits, key=lambda x: (x["variant"], x["readout"], x["fold"])):
        v, r, f, L = x["variant"], x["readout"], x["fold"], x["layer"]
        t = sorted({u.split("|", 1)[1] for w in ("run1", "honest_a") for u in rows_runs[r][w] if fot[u.split("|", 1)[1]] != f})
        random.Random(f"famexp|own_a|f{f}").shuffle(t)
        tt = set(t[: math.ceil(int(v.split("frac")[1].split("_")[0]) / 100 * len(t))])
        train_runs = {u for w in ("run1", "honest_a") for u in rows_runs[r][w] if u.split("|", 1)[1] in tt}
        train_mains = sorted({main_of[c] for c in tt})
        keys, mism = set(), 0
        for part in ("open", "sim_sealed"):
            z = np.load(base / f"run_{FIT}" / part / f"{v}__{r}__L{L}__f{f}.npz")
            kk, ff = z["key"].astype(str), z["fold"]  # key and fold arrays only; the score array is never read
            keys |= set(kk.tolist())
            mism += int(sum(int(fo) != f or fot[k_.split("|", 1)[1]] != f for k_, fo in zip(kk.tolist(), ff.tolist(), strict=True)))
        eval_mains = {main_of[k_.split("|", 1)[1]] for k_ in keys}
        atk = [k_ for k_ in keys if k_.split("|", 1)[0] in ("run1", "run2", "sim")]
        donors = [d for k_ in atk for d in donor_of.get(k_, [])]
        no_ev = sum(not any(by_path.get(d) for d in donor_of.get(k_, [])) for k_ in atk)  # attack runs with no resolved donor evidence
        dres = [by_path.get(d) for d in donors]
        row = {"variant": v, "readout": r, "layer": L, "fold": f, "train_tasks": len(tt), "train_tasks_recorded": x.get("train_tasks"),
               "train_runs": len(train_runs), "train_mains": len(train_mains),
               "train_mains_sha256": hashlib.sha256(json.dumps(train_mains).encode()).hexdigest(),
               "eval_runs": len(keys), "eval_mains": len(eval_mains),
               "main_overlap": sorted(set(train_mains) & eval_mains), "run_overlap": sorted(train_runs & keys),
               "donor_refs": len(donors), "donor_refs_resolved": sum(d is not None for d in dres), "attack_runs_scored": len(atk), "attack_runs_without_donor_evidence": no_ev,
               "donor_in_training": sorted({d for d in dres if d and d in train_runs}),
               "donor_main_in_training": sorted({main_of[d.split("|", 1)[1]] for d in dres if d} & set(train_mains)),
               "fold_mismatch_rows": mism}
        res.append(row)
        totals["train_eval_main_overlap"] += len(row["main_overlap"])
        totals["train_eval_run_overlap"] += len(row["run_overlap"])
        totals["donor_in_training"] += len(row["donor_in_training"]) + len(row["donor_main_in_training"])
        totals["fold_mismatch_rows"] += mism
        totals["train_count_mismatch"] += int(row["train_tasks"] != row["train_tasks_recorded"])
        totals["donor_refs_unresolved"] += row["donor_refs"] - row["donor_refs_resolved"]
        totals["attack_runs_without_donor_evidence"] = totals.get("attack_runs_without_donor_evidence", 0) + no_ev
    return {"rows": res, "totals": totals, "donor_fields_found": sorted(donor_fields), "attack_epochs": len(donor_of),
            "attack_epochs_with_donor_ref": sum(bool(v_) for v_ in donor_of.values()),
            "no_leakage_main_run_fold": all(totals[k] == 0 for k in ("train_eval_main_overlap", "train_eval_run_overlap", "fold_mismatch_rows", "train_count_mismatch")),
            "donor_check": "pass" if totals["donor_in_training"] == 0 and totals.get("attack_runs_without_donor_evidence", 0) == 0 and totals["donor_refs_unresolved"] == 0
            else ("FAIL" if totals["donor_in_training"] else "inconclusive: donor evidence missing or unresolved for some attack runs (fail-closed)"),
            "no_leakage": all(totals[k] == 0 for k in ("train_eval_main_overlap", "train_eval_run_overlap", "donor_in_training", "fold_mismatch_rows", "train_count_mismatch",
                                                         "donor_refs_unresolved")) and totals.get("attack_runs_without_donor_evidence", 0) == 0}


@app.local_entrypoint()
def main():
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, check=True).stdout.strip()  # noqa: E731
    if git("status", "--porcelain", "--", str(Path(__file__).relative_to(REPO))):
        raise SystemExit("commit this script first")
    raw = subprocess.run(["git", "-C", str(REPO), "show", f"{T196[0]}:{T196[1]}"], capture_output=True, check=True).stdout
    if hashlib.sha256(raw).hexdigest() != T196[2]:
        raise SystemExit("T196 differs from its pin")
    main_of = {c["key"]: c["main"] for c in json.loads(raw)["cells"]}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    res = check.remote(main_of)
    rec = {"git": git("rev-parse", "HEAD"), "stamp (PT)": stamp, "app": APP, "spec": [SPEC, PINS["spec"]], "fit": [FIT, PINS["fit"]], "t196": list(T196),
           "label": "keys and counts only; no score array was read", "result": res}
    p = HERE / "results" / f"leakage_check_{stamp}.json"
    with p.open("x") as fh:
        fh.write(json.dumps(rec, indent=1))
    print(json.dumps({"path": str(p.relative_to(REPO)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "totals": res["totals"], "no_leakage": res["no_leakage"], "no_leakage_main_run_fold": res["no_leakage_main_run_fold"], "donor_check": res["donor_check"],
                      "donor_fields_found": res["donor_fields_found"], "attack_epochs_with_donor_ref": res["attack_epochs_with_donor_ref"], "attack_epochs": res["attack_epochs"]}, indent=1))
