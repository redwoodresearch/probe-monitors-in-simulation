"""POST HOC (after the 507 read): priority-1 RULE.md Append 37 §2 @ afd4eed43 (aswin/convergence-audit-09), Sol (a)'s
training coverage of the S_old (90) and S_new (272) estimation mains of the FINAL R|0.05 cohort. Metadata only: no probe
score, no transcript text (identifiers, folds and counts).

- **Training mains per fold model k:** the `training_mains` of `training_mains__a_tight__f{k}.json`, whose sha256 is the
  Sol manifest's `training_manifest_sha256_by_fold[k]`, for every S_new run1/run2 attack arm (FINAL spec pins of
  manifest_sol.json on the work Volume); all eight arms must name the same five. S_old's scores come from the same five
  fits (each Sol manifest's honest_b gate is bound to asis_scores_142228.json with 0 non-identical runs); the fit
  identities are printed side by side. Environment and side tasks of a training main come from its runs36 rows (the
  manifests' own source, "#36 run table (r36), arms real-attack-e1, real-honest, folds != k").
- **Scoring model:** S_new: the Sol manifest's `folds["<world>|<cell>"]`; S_old: asis_scores_142228 `run_folds`. A
  mapped main with fold f uses model f; a fold -1 main uses all five and its count is the mean of the five per-model counts.
- **Counts per estimation main (distinct training mains):** same environment; same side task (null side tasks never
  match); same (environment, side task); the same three restricted to training mains with >= 1 attack-arm (real-attack-e1)
  row; and runs36 rows by arm.
- **Run-level overlap:** each scored run1/run2 attack run against its own scoring model's training manifest. The manifests
  list mains only, so the matched key field is main_task; a run-key match against runs36 rows (.eval stem and its trailing
  22-character run id) is reported separately as "proxy: r36 routing map".
- **09's addition (a), "proxy: r36 routing map":** per main, runs36 rows by arm; per origin, the share of mains with >= 1
  real attack row (e1 or e2) and the e1+e2 rows-per-main distribution; per cell, runs36 attack rows (e1; e1+e2) sharing
  its environment and its (environment, side task), in all folds and in the scoring model's training folds.
- **Per origin:** share of mains in r36; median, quartiles (numpy linear) and share >= 1 for each count. No conclusion.
Modal CPU, read-only Volume. PRELIMINARY.

  SD_IN=<folder with the PINS and COUNTED files> $MODAL run --detach \\
      experiments/2026-10-15_sol_diagnostics_09/run_train_similarity.py
"""

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REC = HERE.parent / "2026-10-14_xerr_scaleup_09" / "results" / "xerr_scaleup_final_2026-10-15_013255.json"
IN = Path(os.environ.get("SD_IN", "/nonexistent"))
APP = "probe-shift_aswin-surya_sub-convaudit_soldiag_kill-2026-11-14"
PINS = {
    "runs36.json": "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96",
    "t196.json": "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb",
    "scaleup_tasks_plus311_2026-10-14_125135.json": "69f1f75313c9610f7d16b18225c1166268c0b6e084859c641d0213f293896a8b",
    "final.json": "7dceb8bb25752aa3619cfde64add702fd4f7d13a1309a56dbc8880a6b27af9f5",
    "correctedb_manifest_v1.json": "bf847d425b80d95ebb75a9ed1d524fc68c9efcd014710ddc18e445ed9d373ebc",
    "asis_scores_142228.json": "59c051a4ff9cc40811745a119251224f031cdaf479b8d3fb909c1059ab94da31",
}
# S_new counted manifests: the FINAL record's spec.counted.run1/run2 entries carry (commit, path, sha256); checked in-job
COUNTED = tuple(f"manifest_su311_q{q}_{w}.json" for q in (1, 2, 3, 4) for w in ("main", "run2"))
ARM_WORLD = {f"su311_q{q}_{a}_v1__attack": w for q in (1, 2, 3, 4) for a, w in (("main", "run1"), ("run2", "run2"))}
TRAIN_ARMS = ("real-attack-e1", "real-honest")  # recipe (a): (E1, H); the scorer's train_arms["a_tight"]
SOL_TRAIN_RUNS = 1929  # "41 of 1,929 training runs" (xerr_prelim_09 RULE.md:385): a check, reported, not a gate
RUNID = re.compile(r"[A-Za-z0-9]{22}")
ATTACK_ARMS = ("real-attack-e1", "real-attack-e2")
ROWCOUNTS = (
    "e1_rows_same_env_all_folds",
    "e1_rows_same_env_side_all_folds",
    "e1e2_rows_same_env_all_folds",
    "e1e2_rows_same_env_side_all_folds",
    "e1_rows_same_env_scoring_training",
    "e1_rows_same_env_side_scoring_training",
    "e1e2_rows_same_env_scoring_training",
    "e1e2_rows_same_env_side_scoring_training",
)
COUNTS = ("same_env", "same_side", "same_env_side", "same_env_attack", "same_side_attack", "same_env_side_attack")

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")
if IN.exists():
    for n in (*PINS, *COUNTED):
        if n != "final.json":
            image = image.add_local_file(IN / n, f"/root/in/{n}")
    image = image.add_local_file(REC, "/root/in/final.json")
work_vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14", version=2).read_only()


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def read_pinned(p: str, digest: str):
    raw = Path(p).read_bytes()
    if sha(raw) != digest:
        raise SystemExit(f"pin mismatch: {p}")
    return json.loads(raw)


def summary(xs: list) -> dict:
    import numpy as np

    a = np.asarray(xs, dtype=float)
    q1, med, q3 = np.percentile(a, [25, 50, 75])  # numpy default (linear interpolation)
    return {
        "n": len(a),
        "median": float(med),
        "q1": float(q1),
        "q3": float(q3),
        "mean": float(a.mean()),
        "share_ge_1": float((a >= 1).mean()),
    }


def stem_id(name: str):
    """.eval file stem (basename without .eval) and its trailing 22-character run id (None if absent)."""
    s = name.rsplit("/", 1)[-1].removesuffix(".eval")
    t = s.rsplit("_", 1)[-1]
    return s, (t if RUNID.fullmatch(t) else None)


@app.function(image=image, volumes={"/work": work_vol}, cpu=2, memory=4096, timeout=600, retries=0)
def compute() -> dict:
    ld = {n: read_pinned(f"/root/in/{n}", d) for n, d in PINS.items()}
    runs, t196, p311, fin = (
        ld["runs36.json"],
        ld["t196.json"],
        ld["scaleup_tasks_plus311_2026-10-14_125135.json"],
        ld["final.json"],
    )
    cb, asis = ld["correctedb_manifest_v1.json"], ld["asis_scores_142228.json"]
    r = fin["result"]
    pinned = {Path(e["path"]).name: e["sha256"] for w in ("run1", "run2") for e in fin["spec"]["counted"][w]}
    if set(pinned) != set(COUNTED):
        raise SystemExit("the FINAL record's counted run1/run2 manifests differ from COUNTED")
    counted = {n: read_pinned(f"/root/in/{n}", pinned[n]) for n in COUNTED}
    # runs36: one fold and one environment per main (the scale-up job's FOLDMAP["sol"])
    fold, env_r = {}, {}
    for row in runs:
        f = row["fold"]
        if not isinstance(f, int) or not 0 <= f <= 4 or fold.setdefault(row["main_task"], f) != f:
            raise SystemExit("runs36: a main with an invalid fold or two folds")
        if env_r.setdefault(row["main_task"], row["environment"]) != row["environment"]:
            raise SystemExit("runs36: a main with two environments")
    arms = sorted({row["arm"] for row in runs})
    train_rows = [row for row in runs if row["arm"] in TRAIN_ARMS]
    sides_of = {}  # training main -> side tasks of its attack-arm rows (null never matches)
    for row in train_rows:
        if row["arm"] == "real-attack-e1" and row["side_task"] is not None:
            sides_of.setdefault(row["main_task"], set()).add(row["side_task"])
    # the five training manifests, from each S_new run1/run2 arm's pinned Sol manifest on the work Volume
    # (Append 37 §2: an unresolvable manifest makes actual coverage and overlap "unavailable"; a hash or identity
    # contradiction stays fatal)
    tm_ids, folds_by_arm, fits, missing = {}, {}, {}, []
    a0 = next(iter(ARM_WORLD))
    for arm, w in ARM_WORLD.items():  # each file on its own: a missing one is recorded, every other check still runs
        sm_pin = fin["spec"]["arms"][arm]["sol_manifest"]
        try:
            sman = read_pinned(sm_pin["path"], sm_pin["sha256"])
        except FileNotFoundError:
            missing.append(Path(sm_pin["path"]).parent.parent.name + "/" + Path(sm_pin["path"]).name)
            continue
        tm_ids[arm] = {k: sman["training_manifest_sha256_by_fold"][k] for k in "01234"}
        fits[arm] = sman.get("fit_sha256")
        folds_by_arm[arm] = (w, sman["folds"], str(Path(sm_pin["path"]).parent))
    if len({json.dumps(v, sort_keys=True) for v in tm_ids.values()}) > 1:
        raise SystemExit("the loaded Sol manifests name different training manifests")
    tl = {}
    if tm_ids:
        ids = next(iter(tm_ids.values()))
        for k in range(5):
            got = None
            for arm in tm_ids:  # the five files beside any loaded manifest (identical sha256 by the check above)
                try:
                    got = read_pinned(f"{folds_by_arm[arm][2]}/training_mains__a_tight__f{k}.json", ids[str(k)])
                    break
                except FileNotFoundError:
                    continue
            if got is None:
                missing.append(f"training_mains__a_tight__f{k}.json")
                continue
            if got["fold_model"] != k or got["probe"] != "a_tight":
                raise SystemExit(f"training manifest f{k}: wrong probe or fold")
            tl[k] = set(got["training_mains"])
    tmains = [tl[k] for k in range(5)] if len(tm_ids) == len(ARM_WORLD) and len(tl) == 5 else None
    missing = ", ".join(missing) if missing else ("no Sol manifest loaded" if not tm_ids else None)
    avail = tmains is not None
    r36_tm = [{x["main_task"] for x in train_rows if x["fold"] != k} for k in range(5)]
    unknown_tm = sorted(set().union(*(tmains or r36_tm)) - set(env_r))
    if unknown_tm:
        raise SystemExit(f"{len(unknown_tm)} training mains without runs36 rows (no environment): stop")
    # estimation cells at R|0.05 (primary): S_new = cells_used.cells; S_old = old cohort cells minus this row's exclusions
    prim = r["results"]["R|0.05"]["primary"]
    cu = prim["cells_used"]
    cells_meta = {c["key"]: c for c in t196["cells"]} | {c["key"]: c for c in p311["cells"]}
    old_keys = {c["key"] for c in t196["cells"]}
    excl = set(cu["excluded_cohort_cells_this_row"])
    s_old = sorted(c for c in r["cohort"]["cohort_cells"] if c in old_keys and c not in excl)
    s_new = sorted(cu["cells"])
    want = {o: prim["sim_minus_mean_real"][o]["cells"] for o in ("S_old", "S_new")}
    if len(s_old) != want["S_old"] or len(s_new) != want["S_new"] or set(s_new) & old_keys:
        raise SystemExit(f"estimation cells {len(s_old)}/{len(s_new)} differ from the record's {want}")
    if any(r["maps"]["main_of"][c] != cells_meta[c]["main"] for c in s_old + s_new):
        raise SystemExit("cell -> main differs between the record's map and the task files")
    # scored real attack runs and their scoring folds
    scored = {}  # cell -> [(world, eval name)], all listed attempts
    for e in cb["epochs"]:
        if e["world"] in ("run1", "run2") and e["arm"] == "attack":
            scored.setdefault(e["task"], []).append((e["world"], e["key"].split("#", 1)[0]))
    for m_ in counted.values():
        for e in m_["files"]:
            if e["world"] in ("run1", "run2") and e["arm_dir"] == "attack":
                scored.setdefault(e["task"], []).append((e["world"], e["path"]))
    sfold = {}  # "<world>|<cell>" -> scoring fold (S_new from the Sol manifests; if unavailable, S_new is proxy-only)
    for w, fo, _ in folds_by_arm.values():
        for k, f in fo.items():
            if k.split("|", 1)[0] == w and sfold.setdefault(k, f) != f:
                raise SystemExit(f"two scoring folds for {k}")
    for k, f in asis["run_folds"].items():
        if k.split("|", 1)[0] in ("run1", "run2") and k.split("|", 1)[1] in old_keys:
            sfold.setdefault(k, f)
    r36_stems, r36_ids, id_by_arm = set(), set(), {}
    for row in runs:
        s, i = stem_id(row["key"].split("__", 1)[0])
        r36_stems.add(s)
        if i:
            r36_ids.add(i)
            id_by_arm[row["arm"]] = id_by_arm.get(row["arm"], 0) + 1
    att_rows = [x for x in runs if x["arm"] in ATTACK_ARMS]

    def rowcounts_for(k, env: str, side) -> dict:
        """proxy: r36 attack rows (e1; e1+e2) with the main's environment / (environment, side task); folds != k (k None: all)."""
        rs = [x for x in att_rows if k is None or x["fold"] != k]
        out = {}
        for nm, arms_ in (("e1", ("real-attack-e1",)), ("e1e2", ATTACK_ARMS)):
            ra = [x for x in rs if x["arm"] in arms_ and x["environment"] == env]
            out[f"{nm}_rows_same_env"] = len(ra)
            out[f"{nm}_rows_same_env_side"] = sum(side is not None and x["side_task"] == side for x in ra)
        return out

    def counts_for(lists, k: int, env: str, side) -> dict:
        tm = lists[k]
        ta = {t for t in tm if t in sides_of}  # training mains with >= 1 attack-arm row
        se = {t for t in tm if env_r[t] == env}
        ss = {t for t in tm if side is not None and side in sides_of.get(t, ())}
        sp = se & ss
        return {
            "same_env": len(se),
            "same_side": len(ss),
            "same_env_side": len(sp),
            "same_env_attack": len(se & ta),
            "same_side_attack": len(ss & ta),
            "same_env_side_attack": len(sp & ta),
        }

    per_cell, by_origin = {}, {}
    for origin, keys in (("S_old", s_old), ("S_new", s_new)):
        rows = []
        for c in keys:
            m, env, side = cells_meta[c]["main"], cells_meta[c]["env"], cells_meta[c]["side"]
            f36 = fold.get(m, -1)
            sf = {w: sfold.get(f"{w}|{c}") for w in ("run1", "run2")}
            if None not in sf.values() and sf["run1"] != sf["run2"]:
                raise SystemExit(f"{c}: scoring folds {sf} differ between run1 and run2")
            if (avail or origin == "S_old") and None in sf.values():
                raise SystemExit(f"{c}: scoring folds {sf} (both worlds required)")
            f = sf["run1"] if sf["run1"] is not None else f36  # f36 only when S_new manifests are unavailable
            models = [f] if f >= 0 else list(range(5))
            cnt = {}
            for tag, lists in (("", tmains), ("proxy_r36_", r36_tm)):
                if lists is None:
                    cnt |= {q: None for q in COUNTS}
                    continue
                pm = [counts_for(lists, k, env, side) for k in models]
                cnt |= {f"{tag}{q}": sum(x[q] for x in pm) / len(models) for q in COUNTS}
            rc_all = rowcounts_for(None, env, side)
            rc_tr = [rowcounts_for(k, env, side) for k in models]
            rcnt = {f"{q}_all_folds": v for q, v in rc_all.items()} | {
                f"{q}_scoring_training": sum(x[q] for x in rc_tr) / len(models) for q in rc_all
            }
            runs_c = scored.get(c, [])
            in_own_tm = [n for _, n in runs_c if avail and any(m in tmains[k] for k in models)]
            d = {
                "scoring_fold": f,
                "scoring_fold_equals_r36": f == f36,
                "main_in_r36": m in fold,
                **cnt,
                **rcnt,
                "r36_rows_by_arm": {a: sum(x["main_task"] == m and x["arm"] == a for x in runs) for a in arms},
                "scored_attack_runs_listed": len(runs_c),
                "scored_runs_main_in_own_scoring_training_manifest": len(in_own_tm) if avail else None,
                "proxy_r36_stem_matches": sum(stem_id(n)[0] in r36_stems for _, n in runs_c),
                "proxy_r36_run_id_matches": sum(bool(stem_id(n)[1]) and stem_id(n)[1] in r36_ids for _, n in runs_c),
                "runs_without_run_id": sum(stem_id(n)[1] is None for _, n in runs_c),
            }
            per_cell[c] = {"origin": origin, "env": env, "side": side} | d
            rows.append(d)
        n = len(rows)
        by_origin[origin] = {
            "mains": n,
            "share_mains_in_r36": sum(d["main_in_r36"] for d in rows) / n,
            "scoring_fold_minus_1": sum(d["scoring_fold"] == -1 for d in rows),
            "scoring_fold_equals_r36 (mains)": sum(d["scoring_fold_equals_r36"] for d in rows),
            "counts_distinct_training_mains": {q: summary([d[q] for d in rows]) for q in COUNTS}
            if avail
            else f"unavailable ({missing})",
            "proxy: r36 routing map, counts_distinct_training_mains": {
                q: summary([d[f"proxy_r36_{q}"] for d in rows]) for q in COUNTS
            },
            "proxy: r36 routing map, arm composition": {
                "share_mains_with_ge1_r36_real_attack_row (e1 or e2)": sum(
                    sum(d["r36_rows_by_arm"].get(a, 0) for a in ATTACK_ARMS) >= 1 for d in rows
                )
                / n,
                "share_mains_with_ge1_r36_e1_row": sum(d["r36_rows_by_arm"].get("real-attack-e1", 0) >= 1 for d in rows)
                / n,
                "r36_real_attack_rows_per_main (e1+e2)": summary(
                    [sum(d["r36_rows_by_arm"].get(a, 0) for a in ATTACK_ARMS) for d in rows]
                ),
                "r36_rows_per_main_by_arm": {a: summary([d["r36_rows_by_arm"][a] for d in rows]) for a in arms},
                "attack_rows_sharing_env_or_env_side (fold -1: mean of five models)": {
                    q: summary([d[q] for d in rows]) for q in ROWCOUNTS
                },
            },
            "run_overlap": {
                "matched_key_field": "main_task (the training manifests list training mains only, no run keys)",
                "scored_attack_runs_listed (all listed counted attempts, run1 and run2)": sum(
                    d["scored_attack_runs_listed"] for d in rows
                ),
                "runs_whose_main_is_in_own_scoring_training_manifest": sum(
                    d["scored_runs_main_in_own_scoring_training_manifest"] for d in rows
                )
                if avail
                else f"unavailable ({missing})",
            },
            "proxy: r36 routing map": {
                "matched_key_fields": ".eval stem vs the runs36 key's part before '__'; the stem's trailing 22-char run id",
                "runs_matching_r36_stem": sum(d["proxy_r36_stem_matches"] for d in rows),
                "runs_matching_r36_run_id": sum(d["proxy_r36_run_id_matches"] for d in rows),
                "runs_without_run_id": sum(d["runs_without_run_id"] for d in rows),
                "caveat": "a zero match does not prove disjointness: r36 keys without a 22-char run id (see r36 ids by arm) "
                "cannot be matched after a rename; a shared .eval names file-level, not sample-level, overlap",
            },
        }
    return {
        "label": "POST HOC (after the 507 read): Append 37 §2 @ afd4eed43, Sol (a) training coverage; metadata only",
        "inputs_sha256": PINS | {n: pinned[n] for n in COUNTED},
        "sol_manifests": {a: fin["spec"]["arms"][a]["sol_manifest"] for a in ARM_WORLD},
        "training_manifests": {
            "sha256_by_fold": tm_ids[a0],
            "same_in_all_eight_arms": True,
            "training_mains_per_fold": [len(t) for t in tmains],
            "equal_to_r36_e1_h_mains_folds_ne_k": [tmains[k] == r36_tm[k] for k in range(5)],
            "distinct_training_mains": len(set().union(*tmains)),
            "with_attack_arm_row": len(set().union(*tmains) & set(sides_of)),
        }
        if avail
        else f"unavailable ({missing})",
        "fits": {
            "sol_manifest_fit_sha256": fits.get(a0),
            "same_in_all_eight_arms": len({json.dumps(v, sort_keys=True) for v in fits.values()}) == 1
            if len(fits) == len(ARM_WORLD)
            else f"unavailable ({len(fits)} of {len(ARM_WORLD)} Sol manifests loaded)",
            "asis_fit_record": asis.get("fit_record"),
        },
        "r36": {
            "rows": len(runs),
            "rows_by_arm": {a: sum(x["arm"] == a for x in runs) for a in arms},
            "training_rows (E1 + H)": len(train_rows),
            "training_rows_match_1929": len(train_rows) == SOL_TRAIN_RUNS,
            "rows_with_22char_run_id_by_arm": id_by_arm,
            "fold_map_mains": len(fold),
        },
        "by_origin": by_origin,
        "per_cell": per_cell,
    }


@app.local_entrypoint()
def main():
    def git(*a):
        return subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()

    if git("status", "--porcelain", "--", str(HERE)):
        raise SystemExit("refusing to run: commit this folder first")
    head = git("rev-parse", "HEAD")
    (HERE / "results").mkdir(exist_ok=True)
    res = compute.remote()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dest = HERE / "results" / f"train_similarity_{stamp}.json"
    dest.open("x").write(json.dumps({"git": head, "stamp": stamp, **res}, indent=1))
    keys = ("label", "training_manifests", "fits", "r36", "by_origin")
    print(json.dumps({k: res[k] for k in keys}, indent=1))
    print("wrote", dest.name)
