"""09's revised Append 21b (4f, 10-14 about 20:35): a +311 run's fold -1 ("unseen") must be VERIFIED from each DT fold model's own
training rows, not assumed. Per design and fold model k, the training rows are world "real", in_design, fold not in {k, (k+1)%5}
(run_dt_attack_fits.py l.13 and 143; run_score_onpolicy.one_draw l.242-244 and 276: k is held out, (k+1)%5 calibrates; rows with
zero weight are also left out there, so these sets are supersets). rows.json stores tasks as indices into sorted(cohort task keys)
(run_score_onpolicy.load l.478-481); each index's frozen fold is checked (a consistency check: a swap of two keys in one fold would
pass, but it cannot change a fold's training mains). Also every main anywhere in the fit rows (any world, fold or design flag), a
superset that does not depend on the CV rule or the mapping. Keys and counts only: main.json (it holds scores) is hashed, never
parsed. Modal CPU, read-only on the fits' Volume; the record is written by the local entrypoint.
  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run \\
      experiments/2026-10-14_p2_dt_scaleup/check_dt_unseen.py --manifests <manifest_dt.json,...> --cells <+311 list>[,<new-mains list>]
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-p2probes_dtq4b_kill-2026-11-13"  # the DT job's app, so its billing stays in one place
SVOL = "probe-shift_aswin-surya_sub-q3onpol_score_kill-2026-11-08"  # the DT fits and xcache_scaleup196 (run_dt_q4b.py SVOL)
DT, XC = "dt_attack_fits_00f4994d6c1e", "xcache_scaleup196"
DESIGNS = ("destructive_primary", "transfer_primary", "destructive_secondary", "transfer_secondary")
TASKS = "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"
PINS = {"tasks": "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb",
        "cells": {"69f1f75313c9610f7d16b18225c1166268c0b6e084859c641d0213f293896a8b": "plus311",  # q2's +311 list @ 0caadbfb
                  "98bfb97e26e7ec580b08643a3a38a5e66a0f6b63333f4013aa57559383402e8e": "new_mains",  # new_mains_cells_v1.json @ ef171c813
                  "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb": "t196"},  # Amendment 11: the T196 tasks file (ilold, S_old)
        "cache_rows_sha256": "ddc42dc64200481923fd14eff15b8c5fb9fbd440481269b11fc15352d5f0908c"}

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.13").add_local_file(HERE / "dt_q4b_lib.py", "/root/dt_q4b_lib.py")


@app.function(image=image, cpu=(2, 2), memory=(4096, 4096), timeout=600, retries=0, scaledown_window=10, startup_timeout=120,
              volumes={"/score": modal.Volume.from_name(SVOL, version=2).read_only()})
def check(t196: dict, scored: list, held: list) -> dict:
    sys.path.insert(0, "/root")
    import dt_q4b_lib as L

    sha = lambda b: hashlib.sha256(b).hexdigest()
    frozen = L.scaleup_folds(t196, t196["cells"])  # T196 key -> its frozen fold (the rule the arm job applies to +311 mains)
    main_of = {c["key"]: (c["env"], c["main"]) for c in t196["cells"]}
    s311 = {tuple(m) for m in scored}  # every fold -1 scored (env, main): the +311 and the new mains
    bad_fold = [h for h in held if frozen.get(h[1]) != h[4]]  # S_old runs (world|cell, cell, env, main, fold): fold must be the main's frozen fold
    cj = json.loads((Path("/score") / XC / "rows.json").read_text())
    out = {"xcache_rows_sha256": cj["rows_sha256"], "xcache_rows": cj["rows"], "designs": {}, "held_runs": len(held), "held_fold_mismatches": sorted(map(list, bad_fold))}
    for des in DESIGNS:
        mraw, rraw = (Path("/score") / DT / des / "main.json").read_bytes(), (Path("/score") / DT / des / "rows.json").read_bytes()
        r = json.loads(rraw)  # main.json is only hashed
        names = sorted(set(main_of) - set(r["summary"]["excluded_tasks"]))
        if sorted(set(r["task"])) != list(range(len(names))) or {len(r[a]) for a in ("task", "world", "in_design", "fold")} != {cj["rows"]}:
            raise ValueError(f"{des}: task indices are not 0..{len(names) - 1} over the cohort, or an array's length differs from the cache's rows")
        by = {}
        for t, f in zip(r["task"], r["fold"]):
            by.setdefault(t, set()).add(f)
        mism = sum(by[i] != {frozen[names[i]]} for i in by)
        rows = list(zip(r["task"], r["world"], r["in_design"], r["fold"], strict=True))
        anywhere = {main_of[names[t]] for t in r["task"]}
        models, train = {}, {}
        for k in range(5):
            tr = [t for t, w, d, f in rows if w == "real" and d and f not in (k, (k + 1) % 5)]
            mk = train[k] = {main_of[names[t]] for t in tr}
            models[str(k)] = {"training_rows": len(tr), "training_tasks": len(set(tr)), "training_mains": len(mk),
                              "scored_mains_in_training": sorted(map(list, mk & s311)),
                              "held_out_runs_with_main_in_own_model_training": sorted(h[0] for h in held if h[4] == k and (h[2], h[3]) in mk),
                              "held_out_runs_routed_here": sum(h[4] == k for h in held)}
        held_trained_elsewhere = sum(any((h[2], h[3]) in train[j] for j in range(5) if j != h[4]) for h in held)  # sanity: a T196 main is trained on elsewhere
        out["designs"][des] = {"main_json_sha256": sha(mraw), "rows_json_sha256": sha(rraw), "cohort_tasks": len(names),
                               "excluded_tasks": sorted(r["summary"]["excluded_tasks"]), "index_fold_mismatches": mism,
                               "mains_anywhere_in_rows": len(anywhere), "scored_mains_anywhere_in_rows": sorted(map(list, anywhere & s311)),
                               "fold_models": models, "held_out_mains_in_some_other_model_training": held_trained_elsewhere}
    return out


@app.local_entrypoint()
def main(manifests: str, cells: str):
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, check=True).stdout.strip()
    me = "experiments/2026-10-14_p2_dt_scaleup/check_dt_unseen.py"
    if git("status", "--porcelain", "--", me) or not git("branch", "-r", "--contains", git("rev-parse", "HEAD")):
        raise SystemExit("commit and push this script first")
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if sha(REPO / TASKS) != PINS["tasks"] or any(sha(c) not in PINS["cells"] for c in cells.split(",")):
        raise SystemExit("the T196 tasks file or a cells list differs from its pin")
    t196, c311, lists = json.loads((REPO / TASKS).read_text()), {}, {}
    for c in cells.split(","):  # the +311 list and the new-mains list; keys are disjoint (checked)
        for x in json.loads(Path(c).read_text())["cells"]:
            if x["key"] in c311:
                raise SystemExit(f"{x['key']} is in two cells lists")
            c311[x["key"]], lists[x["key"]] = x, PINS["cells"][sha(c)]
    used, mans, held = {}, {}, {}
    for p in manifests.split(","):  # the mains the arms scored, from each folder's manifest_dt.json
        m = json.loads(Path(p).read_text())
        for wt, v in m["runs"].items():
            key = wt.split("|", 1)[1]
            f = v["fold"]
            if key not in c311 or c311[key]["main"] != v["main"] or not isinstance(f, int) or isinstance(f, bool) \
                    or (f not in range(5) if lists[key] == "t196" else f != -1):
                raise SystemExit(f"{p}: {wt} is in no pinned cells list, its main differs, or its fold is wrong for its list (T196: 0..4; else -1)")
            if lists[key] == "t196":  # Amendment 11: ilold (S_old) runs route to their held-out fold, verified remotely by key, per (world, cell)
                if wt in held and held[wt] != (wt, key, c311[key]["env"], v["main"], f):
                    raise SystemExit(f"{p}: {wt} appears twice with different records")
                held[wt] = (wt, key, c311[key]["env"], v["main"], f)
            else:
                used[key] = (c311[key]["env"], v["main"])
        mans[str(p)] = {"sha256": sha(p), "weights_main_json_sha256": m["weights_main_json_sha256"],
                              "cache_rows_sha256": m["gate_a"]["cache_rows_sha256"], "runs": len(m["runs"])}
    if not used and not held:
        raise SystemExit("no scored runs in these manifests")
    r = check.remote(t196, sorted(set(used.values())), sorted(set(held.values())))
    t196_pairs, t196_mains = {(c["env"], c["main"]) for c in t196["cells"]}, {c["main"] for c in t196["cells"]}
    rec = {"label": "PRELIMINARY; keys and counts only (09's revised Append 21b)", "git": git("rev-parse", "HEAD"),
           "stamp (PT)": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S"), "fits": DT, "volume": SVOL,
           "cells_scored": len(used), "mains_scored": len(set(used.values())), "held_out_cells_scored": len(held),
           "by_list": {n: {"cells": sum(lists[k] == n for k in used), "mains": len({used[k] for k in used if lists[k] == n}),
                           "mains_in_T196_by_env_and_main": len({used[k] for k in used if lists[k] == n} & t196_pairs),
                           "mains_in_T196_by_main_alone": len({used[k][1] for k in used if lists[k] == n} & t196_mains)} for n in sorted(set(lists.values()))},
           "mains_in_T196_by_env_and_main": len(set(used.values()) & t196_pairs),
           "mains_in_T196_by_main_alone": len({m for _, m in used.values()} & t196_mains), "manifests_dt": mans,
           "recomputed_vs_manifest": {"cache_rows_sha256": all(v["cache_rows_sha256"] == r["xcache_rows_sha256"] == PINS["cache_rows_sha256"] for v in mans.values()),
                                      "weights_main_json_sha256": all(v["weights_main_json_sha256"] == {d: r["designs"][d]["main_json_sha256"] for d in DESIGNS}
                                                                      for v in mans.values())},
           "check": r}
    rec["unseen_verified"] = rec["recomputed_vs_manifest"]["cache_rows_sha256"] and rec["recomputed_vs_manifest"]["weights_main_json_sha256"] \
        and not r["held_fold_mismatches"] and all(d["held_out_mains_in_some_other_model_training"] == len(held) for d in r["designs"].values()) \
        and all(d["index_fold_mismatches"] == 0 and not d["scored_mains_anywhere_in_rows"] and all(not k["scored_mains_in_training"] and not k["held_out_runs_with_main_in_own_model_training"] and k["training_rows"] > 0
                                                                                                    for k in d["fold_models"].values()) for d in r["designs"].values())
    dest = HERE / "results" / f"dt_unseen_check_{rec['stamp (PT)']}.json"
    dest.open("x").write(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("unseen_verified", "mains_scored", "held_out_cells_scored", "by_list", "mains_in_T196_by_env_and_main",
                                          "mains_in_T196_by_main_alone", "recomputed_vs_manifest")}, indent=1), dest.name)
