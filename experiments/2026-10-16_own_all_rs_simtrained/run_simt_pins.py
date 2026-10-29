"""Keys-only addendum pins for own_all_rs_simtrained_uncut_L48 (09 RULE item 48 @ 1ca6ad673), in 4f's LOO3 pins format (@ 25a98b1cb).

Every folder the LOCKED INPUTS_FULL 110447's overrides list for own_all_rs_uncut_L48 (each arm, H_sim root, the pools_ep2 / E12 folders and
the S_old fill-in root; both readouts) maps to its own_all_rs_simtrained_uncut counterpart, which must sit in exactly one committed sim-trained
cutfull score record under the same batch. A Modal CPU job reads only those folders' manifests (sha-checked against the score record) for the
score files' sha256s. The cell list is the frozen Stage A's final cohort (INPUTS_FULL's exclusions pin). No score is read.

uv run modal run experiments/2026-10-16_own_all_rs_simtrained/run_simt_pins.py --score-records <first record>,<follow-up record>
"""

import collections
import datetime
import hashlib
import json
import re
import subprocess
import zoneinfo
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if modal.is_local() else HERE  # the container imports this file from /root; only the local entrypoint uses the repo
RES = REPO / "experiments/2026-10-14_p2_family_expansion/results"
VAR, TREE, UNCUT = "own_all_rs_simtrained_uncut_L48", "own_all_rs_simtrained_uncut", "own_all_rs_uncut/own_all_rs_uncut_L48"
INPUTS_FULL = ("927fabb0d", "experiments/2026-10-15_fpr_sweep_507_prep/config/INPUTS_FULL_2026-10-16_110447.json",
               "d6923a75aa8f914cd3b2e92d8403f125dd095219997aebea8b1d6e4e2b67f77c")
FIT = RES / "own_all_rs_simtrained_uncut_fit_2026-10-16_152335.json"
SCOPE = ("POST HOC: on-policy probe trained on SIMULATED runs (sim attack vs sim honest)", "09 RULE item 48 @ 1ca6ad673, Presentation (verbatim)")
NOTES = [("H_sim is not an independent calibration set for this probe (09 RULE item 48): its training sim_honest runs overlap H_sim; the overlap is "
          "listed by key in the committed training key list (fit.training_keys)"),
         "NEVER in S, Y, the excess or the four-probe subset; ONE separated Fig 1 pair (C at 1/100) with its AUROC beside it (09 RULE item 48)"]
APP = "probe-shift_aswin-surya_sub-p2probes_simtpins_kill-2026-11-14"
WORK_NAME = "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14"

app = modal.App(APP)
WORK = modal.Volume.from_name(WORK_NAME, version=2)


@app.function(image=modal.Image.debian_slim(python_version="3.12"), cpu=1, memory=1024, timeout=600, retries=0, volumes={"/work": WORK.read_only()})
def manifests(folders: list) -> dict:
    """Keys and shas only: each folder's manifest, sha-checked against its score record -> this probe's two score files' sha256s."""
    out = {}
    for batch, fo, man_sha in folders:
        mb = (Path("/work/cutfull") / batch / TREE / fo / "manifest_own_all.json").read_bytes()
        if hashlib.sha256(mb).hexdigest() != man_sha:
            raise SystemExit(f"{batch}/{fo}: manifest sha256 differs from the score record")
        fs = json.loads(mb)["files"]
        out[f"{batch}/{fo}"] = {rd: fs[f"{VAR}__{rd}.npz"] for rd in "RC"}
        if not all(re.fullmatch(r"[0-9a-f]{64}", s) for s in out[f"{batch}/{fo}"].values()):
            raise SystemExit(f"{batch}/{fo}: a score file's manifest entry is not a sha256")
    return out


def git(*a, text=True):
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, check=True, text=text).stdout


def pinned(commit, path, sha):
    """A JSON file read from git at its pinned commit, sha-checked."""
    b = git("show", f"{commit}:{path}", text=False)
    if hashlib.sha256(b).hexdigest() != sha:
        raise SystemExit(f"{path} @ {commit}: sha256 differs from its pin")
    return json.loads(b)


def committed(p):
    """A results file that must equal its HEAD blob -> (path, last commit, sha256, JSON)."""
    b, rel = p.read_bytes(), str(p.resolve().relative_to(REPO))
    if git("show", f"HEAD:{rel}", text=False) != b:
        raise SystemExit(f"{rel}: not committed as read")
    return rel, git("log", "-1", "--format=%H", "--", rel).strip(), hashlib.sha256(b).hexdigest(), json.loads(b)


def setof(fo):  # 4f's set grammar (run_own_all.py @ 12156db74, pins mode)
    return ("S_paper493" if fo.startswith("sp493_") else "S_old" if re.match(r"^(q4f_|rh1e4_|simh_q4f_|ep2_t196_|simh_t196fill_)", fo) else
            "S_new" if re.match(r"^(su311_|simh311_|ep2_311_)", fo) else "S_newmains" if re.match(r"^(nm360_|simh_nm360_|ep2_nm_)", fo) else "?")


@app.local_entrypoint()
def main(score_records: str):
    stamp = datetime.datetime.now(zoneinfo.ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    inp = pinned(*INPUTS_FULL)
    if not inp["status"].startswith("FROZEN"):
        raise SystemExit("INPUTS_FULL is not FROZEN")
    listed = collections.defaultdict(list)  # (batch, folder) -> its locked override listings, one per readout
    for key, pin in inp["overrides"].items():
        row, rd = key.rsplit("|", 1)
        ov, fn = pinned(pin["commit"], pin["path"], pin["sha256"]), f"{UNCUT}__{rd}.npz"
        ents = [("arms", a, v[fn]) for a, v in ov["arms"].items() if fn in v] + [("simh", e["root"], e) for e in ov["simh"].get(fn, [])]
        if not ents:
            raise SystemExit(f"override {key}: lists no {fn}")
        for sec, ent, e in ents:
            m = re.fullmatch(rf"cutfull/([^/]+)/own_all_rs_uncut/([^/]+)/own_all_rs_uncut_L48__{rd}\.npz", e["path"])
            if not m:
                raise SystemExit(f"override {key}: unexpected path {e['path']}")
            listed[m.groups()].append({"row": row, "readout": rd, "override": key, "section": sec, "entry": ent, "uncut": {"path": e["path"], "sha256": e["sha256"]}})
    for bf, ls in listed.items():
        if sorted(x["readout"] for x in ls) != ["C", "R"] or len({x["row"] for x in ls}) != 1:
            raise SystemExit(f"{bf}: not listed exactly once per readout under one override row")
    frl, fcm, fsh, fr = committed(FIT)
    f_ = fr["fit"]
    if not (sorted(g["readout"] for g in fr["gates"]) == ["C", "R"] and all(g["pass"] for g in fr["gates"]) and len(f_["fits"]) == 10
            and all(x["certified"] for x in f_["fits"])):
        raise SystemExit("the fit record's refit-match gates or certification do not pass")
    recs, scored = [], {}
    for sp in score_records.split(","):
        rel, cm, sha, sr = committed(Path(sp))
        if sr["fit_record_sha256"] != f_["fit_record_sha256"] or sr.get("line") != "cutfull" or VAR not in sr["cutfull_label"]:
            raise SystemExit(f"{rel}: another fit, not the cutfull line, or not this label")
        recs.append({"path": rel, "commit": cm, "sha256": sha, "parent": (sr.get("parent_score_record") or {}).get("path"), "tree": TREE})
        for r in sr["results"].values():
            if sorted(r["misrouted_negative"]) != [f"{VAR}|C", f"{VAR}|R"] or not all(v["gate_flags_violation"] for v in r["misrouted_negative"].values()):
                raise SystemExit(f"{rel}: a misrouted-negative control is not flagged")
            for fo, x in r["folders"].items():
                if x["path"] != f"/work/cutfull/{x['batch']}/{TREE}/{fo}" or (x["batch"], fo) in scored or x["in_fold_violations"] != 0:
                    raise SystemExit(f"{fo}: not this tree's path, scored twice across the records, or in-fold violations")
                scored[(x["batch"], fo)] = (x["manifest_sha256"], rel)
    miss = sorted(set(listed) - set(scored))
    if miss:
        raise SystemExit(f"STOP: {len(miss)} listed folders are in no score record: {miss[:5]}")
    if any(setof(fo) == "?" for _, fo in listed) or len({fo for _, fo in listed}) != len(listed):
        raise SystemExit("STOP: a listed folder has no set, or one folder name sits under two batches (the output key is <tree>/<folder>)")
    shas = manifests.remote([(b, fo, scored[(b, fo)][0]) for b, fo in sorted(listed)])
    ex, coh = (pinned(inp[k]["commit"], inp[k]["path"], inp[k]["sha256"]) for k in ("exclusions", "cohort"))
    cells = {og: v["cells"] for og, v in ex["final_cohort_keys"].items()}
    for og, v in ex["final_cohort_keys"].items():
        if hashlib.sha256(json.dumps(sorted(v["cells"]), separators=(",", ":")).encode()).hexdigest() != v["sha256"] or len(set(v["cells"])) != len(v["cells"]):
            raise SystemExit(f"final_cohort_keys {og}: its sha256 or uniqueness does not hold")
    if any(sorted(v["headline_cells"]) != sorted(cells[og]) for og, v in coh["origins"].items()):
        raise SystemExit("the exclusions file's final cohort differs from INPUTS_FULL's cohort pin by key")
    prev = sorted(RES.glob(f"{TREE}_addendum_pins_*.json"))
    rowof = {bf: ls[0]["row"] for bf, ls in listed.items()}
    rec = {"label": f"{VAR} addendum pins for 2e's full-length Stage A (keys and shas only; no values)", "variant": VAR,
           "labels": [f"{VAR} R", f"{VAR} C"], "stamp (PT)": stamp,
           "writer": {"commit": git("rev-parse", "HEAD").strip(), "app": APP, "app_id": app.app_id or "", "stamp": stamp},
           "rule": "09 RULE item 48 @ 1ca6ad673, integration option (a): a label addendum beside the LOCKED INPUTS_FULL (unchanged), then a ONE-LABEL Stage A run",
           "inputs_full": dict(zip(("commit", "path", "sha256"), INPUTS_FULL)), "overrides": inp["overrides"],
           "fit": {"git_record": {"path": frl, "commit": fcm, "sha256": fsh}, "volume_fit_record_sha256": f_["fit_record_sha256"],
                   "refit_match": {"pass": True, "control": "own_all_rs_uncut_L48 refit from its own rows equals its registered weights (R and C x 5 folds)",
                                   "gate_files_sha256": f_["gate_files_sha256"]},
                   "pass": True, "score_files": {"C": f"{VAR}__C.npz", "R": f"{VAR}__R.npz"}, "training_keys": f_["training_keys"]},
           "score_records": recs,
           "score_files": {f"{TREE}/{fo}": {"row": rowof[(b, fo)], "set": setof(fo), "tree": TREE, "batch": b,
                                             "manifest": {"path": f"/cutfull/{b}/{TREE}/{fo}/manifest_own_all.json", "sha256": scored[(b, fo)][0]},
                                             "score_record": scored[(b, fo)][1],
                                             "files": {rd: {"path": f"/cutfull/{b}/{TREE}/{fo}/{VAR}__{rd}.npz", "sha256": shas[f"{b}/{fo}"][rd]} for rd in "RC"},
                                             "locked_override": {x["readout"]: {k: x[k] for k in ("override", "section", "entry", "uncut")} for x in listed[(b, fo)]}}
                           for b, fo in sorted(listed)},
           "supersedes": ({"path": str(prev[-1].relative_to(REPO)), "sha256": hashlib.sha256(prev[-1].read_bytes()).hexdigest()} if prev else None),
           "volumes": {"scores": f"{WORK_NAME} (/cutfull/...)", "fit": "probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13"},
           "eligibility": {"status": "no eligibility freeze for this probe: 09 RULE item 48 scores every main by its DESIGNATED held-out fold model "
                                     "(S_paper493 and mapped-but-untrained mains included; no fold -1)", "this_probe": "S_paper493 INCLUDED (designated fold model)"},
           "coverage": {"listed_folders": len(listed), "scored_folders_by_row": dict(collections.Counter(rowof.values())),
                        "score_record_folders_not_listed": sorted(f"{b}/{fo}" for b, fo in set(scored) - set(listed)),
                        "by_key": "NOT in this file: item 48's by-key verification (coverage per cell x arm x pool, fold routing, 0 in-sample) is the addendum's next step"},
           "comparison_cells": {"cells": cells, "n": {k: len(v) for k, v in cells.items()}, "total": sum(len(v) for v in cells.values()),
                                "rule": "the frozen Stage A's final cohort (INPUTS_FULL 'exclusions' pin, final_cohort_keys, each origin's sha256 re-checked); S_old / S_new / "
                                        "S_newmains equal INPUTS_FULL's 'cohort' pin (headline_cells) by key. Item 48 states no probe-specific cell restriction",
                                "source": inp["exclusions"], "cohort_pin": inp["cohort"]},
           "scope_label": {"text": SCOPE[0], "source": SCOPE[1]}, "label_notes": NOTES,
           "status": "keys-only pins; the addendum's by-key verification and the one-label Stage A run are not started"}
    out = RES / f"{TREE}_addendum_pins_{stamp}.json"
    out.open("x").write(json.dumps(rec, indent=1))
    print(json.dumps({"path": str(out.relative_to(REPO)), "folders": len(listed), "by_row": rec["coverage"]["scored_folders_by_row"],
                      "not_listed": len(rec["coverage"]["score_record_folders_not_listed"]), "cells": rec["comparison_cells"]["n"], "app_id": app.app_id}))
