"""Draw-0 hit indicators for 09's RMS-bias analysis (note @ c10a14948, addendum @ 4ff49b1be; POST HOC; requested by 2e). For each IL five-arm cell
of a set and each probe x readout: is each arm's run (sim1, sim2, run1, run2, run3) hit at per-action 1e-3 and 1e-2?
  - line "real": tau_B on the set's H_B units (g_B), applied to all five arms;
  - line "separate": sim arms on g_sim at tau_sim from the set's sim-honest units; real arms at tau_B. Only where H_sim is given (admitted).
Fixed calibration: full-sample fold moments (ddof 0) of each pool (5-fold probes) or raw scores (single-model probes); the 507 unit rule
(k = floor(a n + 1e-9), T = sorted-descending[min(k, n - 1)]); a run is hit iff any unit is strictly above T. Per pool x point: k, units above,
estimable (k >= 1) and shaded (fewer than 10 units above). Cells on a main outside r36 are -1 for 5-fold probes (2e's fold -1 rule). Missing rows
stop the job: every (arm, cell) row named in the config must exist.

  PYTHONPATH=src <modal 1.6.1> run --detach experiments/2026-10-15_il_z_e2e/run_rms_hits.py --config config/inputs_rms_<set>.json
"""

import hashlib
import io
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-ilz_kill-2026-11-14"
ARMS = ("sim1", "sim2", "run1", "run2", "run3")
SIM_ARMS = ("sim1", "sim2")
app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3").add_local_python_source("probe_shift")
conv = modal.Volume.from_name("probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14")
outv = modal.Volume.from_name("probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13", version=2)


NUMERICS_NOTE = "score-level RMS sums show ~1e-14 relative run-to-run variation across containers (non-dyadic floating-point accumulation); catch-level and per-action rows are bitwise stable"  # 09's ACK (via 2e, 19:0x): verbatim in every stage B record


def scope_meta(cfg):  # 09 (a): every record states its S4 status and the H_sim roots it used
    return {"scope_label": cfg.get("scope_label"), "s4_status": cfg.get("s4_status"), "H_sim_roots_used": [r["root"] for r in cfg["H_sim"]["roots"]],
            "H_sim_roots_excluded": cfg.get("H_sim_roots_excluded"), "numerics_note": NUMERICS_NOTE, "app_id": app.app_id}


WORLD_OF = {"sim1": "sim", "sim2": "sim2", "run1": "run1", "run2": "run2", "run3": "run3"}  # arm -> the world in its row selector (five-arm v2)
SCOPE_A = "S_newmains here = S1–S3 + S5 (S4d / S4r not included)"


def s4_recheck(cfg, git):
    """09 (17:3x): recompute the S4 checks from the pinned source maps (Codex r3: never trust the config's own report). Orphans (d9's outside_source_map
    rows) never enter an arm or pool; every arm has exactly one row per cell; with 09's choice (a) no S4 cell or root enters any pool or arm."""
    if "s4_orphan_check" not in cfg:
        raise SystemExit("config lacks an S4 orphan check (build it with make_config_rms_final.py)")

    def pinned(ent):
        b = git("show", f"{ent['commit']}:{ent['path']}")
        if hashlib.sha256(b).hexdigest() != ent["sha256"]:
            raise SystemExit(f"{ent['path']}: sha256 differs from the pin")
        return json.loads(b)

    sm = pinned(cfg["s4_orphan_check"]["source_map"])
    orph, o_roots = {(r["root"], r["key"]) for r in sm["outside_source_map"]}, {r["root"] for r in sm["outside_source_map"]}
    s4cells = {c["cell"] for c in sm["cells"]} | {r["cell"] for r in pinned(sm["s4d_part"])["rows"]}
    root_of = lambda f: f.strip("/").split("/")[-1].split("__")[0]  # noqa: E731
    roots = [r["root"] for r in cfg["H_sim"]["roots"]]
    bad = [f"{a}:{f}:{k}" for a, r in cfg["arm_rows"].items() for f, k in r.values() if (root_of(f), k) in orph]
    bad += [f"H_B {f}:{k}" for f in cfg["H_B"]["folders"] for k in cfg["H_B"]["keys"] if (root_of(f), k) in orph]
    bad += [f"H_sim {r}" for r in roots if root_of(r) in o_roots] + (["an H_sim root listed twice"] if len(set(roots)) != len(roots) else [])
    bad += [f"arm {a}: cells differ from the cell list" for a, r in cfg["arm_rows"].items() if sorted(r) != sorted(cfg["cells"]) or a not in WORLD_OF]
    bad += [f"{a}:{c}: selector {k} is not {WORLD_OF.get(a)}|{c}" for a, r in cfg["arm_rows"].items() for c, (_, k) in r.items() if k != f"{WORLD_OF.get(a)}|{c}"]
    if len(set(cfg["cells"])) != len(cfg["cells"]) or sorted(cfg["arm_rows"]) != sorted(WORLD_OF):
        bad.append("a cell listed twice, or the arms differ from sim1/sim2/run1/run2/run3")
    # 09's option (a) is the only supported S4 policy (Codex r4): no S4 cell or root in any arm or pool, for every set; S_newmains must carry its label
    if cfg["set"] == "S_newmains" and (cfg.get("scope_label") != SCOPE_A or sorted(cfg.get("s4_cells_excluded") or ()) != sorted(s4cells)):
        bad.append(f"S_newmains config is not option (a) (scope_label {cfg.get('scope_label')!r}, or s4_cells_excluded differs from the maps' 80 S4 cells)")
    bad += [f"S4 cell {c}" for c in sorted(set(cfg["cells"]) & s4cells)] + [f"S4 H_sim root {r}" for r in roots if "nm360_s4" in r]
    bad += [f"S4 H_B key {k}" for k in cfg["H_B"]["keys"] if k.split("|", 1)[1] in s4cells]
    if bad:
        raise SystemExit(f"S4 recheck failed: {bad[:5]} ({len(bad)})")
    cfg["s4_cells_excluded"] = sorted(s4cells)  # the run-time H_sim guards use the verified list, for every set


def root_bytes(load, ent, fname):
    """A sim-honest root's file: pinned by the entry's own file map (Sol mappings, 4f's records), or (INPUTS_FINAL) by the root's pinned manifest, whose
    files entry then pins the file."""
    d = ent["path"].strip("/")
    if "files" in ent:
        return load(f"{d}/{fname}", ent["files"][fname])
    man = json.loads(load(f"{d}/{ent['manifest']['name']}", ent["manifest"]["sha256"]))["files"]
    e = man[fname]
    return load(f"{d}/{fname}", e if isinstance(e, str) else e["sha256"])


@app.function(image=image, cpu=(4, 4), memory=(16384, 16384), timeout=1800, retries=0, volumes={"/conv": conv.read_only()})
def compute(cfg: dict, row: dict) -> dict:  # noqa: C901
    import numpy as np

    from probe_shift import fpr_core as fc

    lab, single = row["label"], row["folds"] == 1
    diag = {"files": {}, "pinned_files": 0, "read_started_pt": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S")}

    def stop(msg):
        raise SystemExit(f"{cfg['set']} {lab}: gate failed: {msg}")

    def load(path, pin):
        b = (Path("/conv") / path.lstrip("/").removeprefix("work/")).read_bytes()
        s_ = hashlib.sha256(b).hexdigest()
        if not re.fullmatch("[0-9a-f]{64}", str(pin)) or s_ != pin:
            stop(f"{path}: sha256 {s_} differs from the pin {pin}")
        diag["files"][path] = s_
        diag["pinned_files"] += pin is not None
        return b

    def folder_rows(folder):  # the folder's pinned sub-manifest -> the row file at its manifest entry
        man = cfg["folders"][folder][row["sub"]]
        fe = json.loads(load(man["path"], man["sha256"]))["files"]
        ent = fe[row["file"]]
        d = man["path"].replace("/work/", "").rsplit("/", 1)[0]
        return dict(np.load(io.BytesIO(load(f"{d}/{row['file']}", ent if isinstance(ent, str) else ent["sha256"])), allow_pickle=False))

    fold_cell = cfg["cell_fold"]
    filefold = fold_cell is None  # S_newmains: no r36 map for nm360 mains; 5-fold rows use the scorer's designated fold per row (fold -1 / non-finite rows leave)
    cache = {}

    def rows_of(folder):
        if folder not in cache:
            z = folder_rows(folder)
            cache[folder] = (z["key"].astype(str), z["score"].astype(np.float64), None if single else z["fold"].astype(np.int64))
        return cache[folder]

    def pool(spec):  # -> (scores, folds) over the spec's keys from its folders, plus run count
        s_, f_, runs = [], [], set()
        keys = set(spec["keys"]) if single or filefold else {k for k in spec["keys"] if fold_cell.get(k.split("|", 1)[1], -1) >= 0}  # fold -1 runs leave 5-fold pools
        diag["H_B_fold_minus1_keys_dropped"] = len(spec["keys"]) - len(keys)
        for fol in spec["folders"]:
            k, s, f = rows_of(fol)
            m = np.isin(k, sorted(keys))
            if not single and filefold:
                drop_ = m & ((f < 0) | ~np.isfinite(s))
                diag["H_B_rows_fold_minus1_or_nonfinite"] = diag.get("H_B_rows_fold_minus1_or_nonfinite", 0) + int(drop_.sum())
                keys -= set(k[drop_].tolist()) - set(k[m & ~drop_].tolist())
                m &= ~drop_
            elif not single and not np.array_equal(f[m], np.array([fold_cell[x.split("|", 1)[1]] for x in k[m]], np.int64)):
                stop(f"H_B {fol}: unit folds differ from r36's fold of the cell's main")
            dupk = runs & set(k[m].tolist())
            if dupk:
                stop(f"H_B keys in two folders: {sorted(dupk)[:3]}")  # one owner per key
            s_.append(s[m])
            f_.append(np.zeros(m.sum(), np.int64) if single else f[m])
            runs |= set(k[m].tolist())
        if runs != keys:
            stop(f"pool keys without rows: {sorted(keys - runs)[:5]} ({len(keys - runs)})")
        return np.concatenate(s_), np.concatenate(f_), len(runs)

    def sim_pool(spec):
        s_, f_, runs = [], [], set()
        for rt in spec["roots"]:
            ent = rt[row["sub"]]
            z = dict(np.load(io.BytesIO(root_bytes(load, ent, row["file"])), allow_pickle=False))
            k = z["key"].astype(str)
            if {x.split("|", 1)[1] for x in k} & set(cfg.get("s4_cells_excluded") or ()):
                stop(f"H_sim {rt['root']}: S4 cells in H_sim under scope (a)")
            m = ~np.isin(k, sorted(spec.get("exclude_keys", [])))
            if not single and filefold:
                m &= (z["fold"].astype(np.int64) >= 0) & np.isfinite(z["score"].astype(np.float64))
            elif not single:
                cl = np.array([x.split("|", 1)[1] for x in k])
                m &= np.array([fold_cell.get(c, -1) >= 0 for c in cl])  # fold -1 runs leave the 5-fold pool (2e's rule)
            if not single and not filefold and not np.array_equal(z["fold"].astype(np.int64)[m], np.array([fold_cell[x.split("|", 1)[1]] for x in k[m]], np.int64)):
                stop(f"H_sim {rt['root']}: unit folds differ from r36's fold of the cell's main")
            dup = runs & set(k[m].tolist())
            if dup:
                stop(f"a sim-honest run appears in two roots: {sorted(dup)[:3]}")
            runs |= set(k[m].tolist())
            s_.append(z["score"].astype(np.float64)[m])
            f_.append(np.zeros(m.sum(), np.int64) if single else z["fold"].astype(np.int64)[m])
        return np.concatenate(s_), np.concatenate(f_), len(runs)

    def line(s, f, n_runs):  # draw-0 calibration of one pool
        if not np.isfinite(s).all():
            stop("non-finite pool scores")
        F = int(f.max()) + 1
        mu, sd = np.zeros(F), np.ones(F)
        if not single:
            for g in range(F):
                x = s[f == g]
                if not (len(x) >= 2 and x.std() > 0):
                    stop(f"pool fold {g} empty or constant")
                mu[g], sd[g] = x.mean(), x.std()
        z = (s - mu[f]) / sd[f]
        zs = np.sort(z)[::-1]
        out = {"n_units": int(len(s)), "n_runs": n_runs}
        for a in cfg["points"]:
            k = int(np.floor(a * len(s) + 1e-9))
            T = float(zs[min(k, len(s) - 1)])
            if k >= 1 and T != fc.thr(z, np.ones((1, len(s))), a)[0]:
                stop("draw-0 threshold differs from fc.thr")
            above = int((z > T).sum())
            out[f"{a:g}"] = {"k": k, "T": T, "units_above": above, "estimable": k >= 1, "shaded": above < 10}
        return mu, sd, out

    cells = cfg["cells"]
    arm_max = {}  # arm -> (max raw unit score per cell, fold per cell); -1 fold = excluded (fold -1 on 5-fold probes)
    for arm in ARMS:
        mx = np.full(len(cells), np.nan)
        fo = np.full(len(cells), -1, np.int64)
        for i, c in enumerate(cells):
            fol, key = cfg["arm_rows"][arm][c]
            k, s, f = rows_of(fol)
            ix = k == key
            if not ix.any():
                stop(f"{arm} {key} in {fol}: no rows")
            if not single and filefold:
                fs_ = set(f[ix].tolist())
                if len(fs_) != 1:
                    stop(f"{arm} {key}: units carry several folds {sorted(fs_)}")
                if min(fs_) < 0 or not np.isfinite(s[ix]).all():
                    continue  # fold -1 / no designated score: excluded (-1)
                fo[i], mx[i] = min(fs_), float(s[ix].max())
                continue
            if not np.isfinite(s[ix]).all():
                if not single and fold_cell.get(c, -1) < 0:
                    continue  # fold -1 cell: no designated score, excluded below
                stop(f"{arm} {key}: non-finite unit scores")
            if not single:
                fc_ = fold_cell.get(c, -1)
                if fc_ < 0:
                    continue
                if set(f[ix].tolist()) != {fc_}:
                    stop(f"{arm} {key}: unit folds differ from r36's fold of the cell's main")
                fo[i] = fc_
            else:
                fo[i] = 0
            mx[i] = float(s[ix].max())
        arm_max[arm] = (mx, fo)

    lines = {}
    dev = bool(cfg.get("withhold_5fold")) and not single  # 09: S_newmains Sol (a)/c_asis rows are withheld; values only as a labelled deviation sensitivity
    pre = "DEVIATION_SENSITIVITY|" if dev else ""
    sB, fB, nB = pool(cfg["H_B"])
    sS, fS, nS = sim_pool(cfg["H_sim"]) if cfg.get("H_sim") else (None, None, 0)
    empty = [nm for nm, x in (("H_B", sB), ("H_sim", sS)) if x is not None and len(x) == 0]
    if empty:
        if not dev:
            stop(f"empty pool(s) {empty}")
        buf = io.BytesIO()
        np.savez_compressed(buf)
        return {"label": lab, "npz": buf.getvalue(), "lines": {"status": f"WITHHELD (0 mapped units on the nm360 pools; 25h amendment §1); the designated-fold rule leaves {empty} empty"}, "diag": diag}
    muB, sdB, infoB = line(sB, fB, nB)
    lines["H_B"] = infoB
    if cfg.get("H_sim"):
        muS, sdS, infoS = line(sS, fS, nS)
        lines["H_sim"] = infoS
    if dev:
        lines = {"status": "WITHHELD (0 mapped units on the nm360 pools; 25h amendment §1)", "deviation_sensitivity": lines}
    hits = {}
    for a in cfg["points"]:
        for ln in ("real", "separate") if cfg.get("H_sim") else ("real",):
            H = np.full((len(cells), len(ARMS)), -1, np.int8)
            for j, arm in enumerate(ARMS):
                mx, fo = arm_max[arm]
                ok = fo >= 0
                use_sim = ln == "separate" and arm in SIM_ARMS
                mu, sd, T = (muS, sdS, infoS[f"{a:g}"]["T"]) if use_sim else (muB, sdB, infoB[f"{a:g}"]["T"])
                H[ok, j] = ((mx[ok] - mu[fo[ok]]) / sd[fo[ok]] > T).astype(np.int8)
            hits[f"{pre}{cfg['set']}|{ln}|{a:g}|{row['label']}"] = H
    diag["cells_excluded_fold_minus1"] = [c for i, c in enumerate(cells) if arm_max["run3"][1][i] < 0] if not single else []
    diag["arm_cells_excluded"] = {a: int((arm_max[a][1] < 0).sum()) for a in ARMS}
    buf = io.BytesIO()
    np.savez_compressed(buf, **hits)
    return {"label": lab, "npz": buf.getvalue(), "lines": lines, "diag": diag}


@app.function(image=image, cpu=(1, 1), memory=(2048, 2048), timeout=600, retries=0, volumes={"/solavg": outv})
def merge(parts: list, name: str) -> bytes:
    import numpy as np

    arrs = {}
    for b in parts:
        z = np.load(io.BytesIO(b))
        for k in z.files:
            if k in arrs:
                raise SystemExit(f"duplicate array {k}")
            arrs[k] = z[k]
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrs)
    od = Path("/solavg") / "xfer" / "il_z" / "rms_hits"
    od.mkdir(parents=True, exist_ok=True)
    (od / f"{name}.npz").open("xb").write(buf.getvalue())  # the repo ignores *.npz: the npz lives here, its sha256 in the committed manifest
    outv.commit()
    return buf.getvalue()


@app.local_entrypoint()
def main(config: str = "", check: str = "", census_only: bool = False, deviation_mode: bool = False, fields: str = "", leakage_addendum_mode: bool = False):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout  # noqa: E731
    if leakage_addendum_mode:  # 09's gate c6023b805: the leakage check with its +311 positive control and the selection provenance, by key
        cfg_nm = json.loads((HERE / "config/inputs_rms_S_newmains.json").read_text())
        s4_recheck(cfg_nm, git)  # Codex r3: the old 14:55 config has no S4 check, so this retired mode refuses until pointed at a checked config
        z5 = json.loads((HERE / "config/inputs_znew_pass5.json").read_text())
        srcs = {"nm360_cells": "31d7329ef:experiments/2026-10-14_q35_sol_avg_POSTHOC/config/nm360_cells_nm_built_2026-10-14.json",
                "t196": "HEAD:experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"}
        blobs = {k: git("show", v) for k, v in srcs.items()}
        r36p = json.loads((HERE / "config/inputs_z.json").read_text())["local"]["r36"]
        r36b = Path(r36p["path"]).read_bytes()
        if hashlib.sha256(r36b).hexdigest() != r36p["sha256"] or hashlib.sha256(blobs["t196"]).hexdigest() != "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb":
            raise SystemExit("input sha mismatch")
        inp = {"nm360_cells": json.loads(blobs["nm360_cells"])["cells"], "t196_cells": json.loads(blobs["t196"])["cells"], "r36": json.loads(r36b),
               "cell_main_311": z5["cell_main"]}
        res = leakage_addendum.remote(cfg_nm, ["su311_q1_main_v1__honest_b", "su311_q3_main_v1__honest_b", "su311_q4_main_v1__honest_b"], inp)
        stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
        rec = {"label": "Leakage addendum (09 gate c6023b805): the deviation()'s overlap rule rerun with its +311 positive control, plus the selection "
                        "and preprocessing provenance, by key; POST HOC; PRELIMINARY",
               "sources": srcs | {"r36": r36p, "rms_S_newmains_config_sha256": hashlib.sha256((HERE / "config/inputs_rms_S_newmains.json").read_bytes()).hexdigest(),
                                 "selection_record": "aa41408f6:experiments/2026-10-14_q35_sol_avg_POSTHOC/results/avg_transfer_avg_v1_2026-10-14_011347.json (sha256 cc90ed91…)",
                                 "selection_code": "62f252885:experiments/2026-10-14_q35_sol_avg_POSTHOC/run_avg_transfer.py (_select: lines 72-175; per outer fold f every fit "
                                                   "and the inner-CV AUROC use runs with fold != f; fit_on fits the mean/std preprocessing on those rows)",
                                 "fit_record": "04f4e4acb:experiments/2026-10-14_q35_sol_avg_POSTHOC/results/asis_fit_2026-10-14_135701.json (sha256 9c5479ed…; layer_choice_a "
                                               "from the selection record)"},
               "meta": {"git": git("rev-parse", "HEAD").decode().strip(), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "stamp_pt": stamp},
               **res}
        b = json.dumps(rec, indent=1).encode()
        fn = f"leakage_addendum_{stamp}.json"
        (HERE / "results" / fn).open("xb").write(b)
        print(fn, hashlib.sha256(b).hexdigest())
        return
    if fields:  # metadata: array names in the listed npz files (comma-separated Volume paths)
        print(json.dumps(npz_fields.remote(fields.split(",")), indent=0))
        return
    if check:  # cross-check mode: mean hit rates per array and arm from the Volume npz (no recompute)
        print(json.dumps(hit_means.remote(check), indent=0))
        return
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("uncommitted tracked changes")
    cfg = json.loads((HERE / config).read_text())
    s4_recheck(cfg, git)
    if deviation_mode:  # 2e/0d GO: the all-units deviation sensitivity for the 5-fold rows (never leads)
        cells_b = git("show", "31d7329ef:experiments/2026-10-14_q35_sol_avg_POSTHOC/config/nm360_cells_nm_built_2026-10-14.json")
        if hashlib.sha256(cells_b).hexdigest() != "93d9b538710b6bd930e7df01b0f5296d417550a0f0a35bcb6eb696166cf39a04":
            raise SystemExit("nm360 cells (main-task mapping): sha256 differs from the pin")
        main_of = {c_["key"]: c_["main"] for c_ in json.loads(cells_b)["cells"]}
        rows5 = [r_ for r_ in cfg["rows"] if r_["folds"] == 5]
        outs = list(deviation.map([cfg] * len(rows5), rows5, [main_of] * len(rows5)))
        stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
        name = f"rmshits_{cfg['set']}_deviation_{stamp}_{app.app_id}"  # 09: names carry the app id (exclusive create below)
        npz = merge.remote([o["npz"] for o in outs], name)
        man = {"label": cfg.get("sol_extension_label", DEV_LABEL), "beside": "the registered row reads WITHHELD (0 mapped units on the nm360 pools; 25h amendment §1); this "
               "extension row sits beside it",
               "set": cfg["set"], "npz": f"{name}.npz", "npz_sha256": hashlib.sha256(npz).hexdigest(),
               "npz_location": f"Modal Volume probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13: xfer/il_z/rms_hits/{name}.npz",
               "array_key_format": "DEVIATION|<set>|<line>|<point>|<probe label>; int8 [cell, arm]: 1 hit, 0 not hit", "cells": cfg["cells"], "arms": list(ARMS),
               "points": cfg["points"], "probes": [o["label"] for o in outs], "pools": {o["label"]: o["lines"] for o in outs},
               "leakage_check": {o["label"]: o["leakage"] for o in outs}, "main_map": "31d7329ef:experiments/2026-10-14_q35_sol_avg_POSTHOC/config/nm360_cells_nm_built_2026-10-14.json",
               "diag": {o["label"]: o["diag"] for o in outs},
               "meta": {"git": git("rev-parse", "HEAD").decode().strip(), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "config": config,
                        "config_sha256": hashlib.sha256((HERE / config).read_bytes()).hexdigest(), "stamp_pt": stamp}}
        man["meta"] |= scope_meta(cfg)
        mb = json.dumps(man, indent=1, default=float).encode()
        (HERE / "results" / f"{name}.json").open("xb").write(mb)
        print(json.dumps({"manifest": f"experiments/2026-10-15_il_z_e2e/results/{name}.json", "manifest_sha256": hashlib.sha256(mb).hexdigest(),
                          "npz_sha256": man["npz_sha256"]}, indent=1))
        return
    if census_only:  # 09's (a): mapped vs fold -1 counts for the 5-fold rows
        rows5 = [r_ for r_ in cfg["rows"] if r_["folds"] == 5]
        res = list(census.map([cfg] * len(rows5), rows5))
        b = json.dumps({"label": "Sol fold census v2 (09's request a; split per 2e 14:57): MAPPED = fold >= 0 with a finite score; fold -1 and non-finite counted separately; by key", "config": config,
                        "config_sha256": hashlib.sha256((HERE / config).read_bytes()).hexdigest(), "rows": {r_["label"]: r_["census"] for r_ in res}}, indent=1).encode()
        fn = f"fold_census_{cfg['set']}_{datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d_%H%M%S')}.json"
        (HERE / "results" / fn).open("xb").write(b)
        print(fn, hashlib.sha256(b).hexdigest())
        return
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    name = f"rmshits_{cfg['set']}_{stamp}_{app.app_id}"  # 09: names carry the app id (exclusive create below)
    print(f"{datetime.now(ZoneInfo('America/Los_Angeles')):%H:%M:%S} {cfg['set']}: {len(cfg['rows'])} rows, {len(cfg['cells'])} cells", flush=True)
    if cfg.get("fold_minus1_rule") == "s1":  # 5-fold rows under 25h §1; single-model rows unchanged
        r5 = [r_ for r_ in cfg["rows"] if r_["folds"] == 5]
        r1 = [r_ for r_ in cfg["rows"] if r_["folds"] != 5]
        outs = list(compute.map([cfg] * len(r1), r1, order_outputs=True)) + list(compute_s1.map([cfg] * len(r5), r5, order_outputs=True))
    else:
        outs = list(compute.map([cfg] * len(cfg["rows"]), cfg["rows"], order_outputs=True))
    npz = merge.remote([o["npz"] for o in outs], name)
    R = HERE / "results"
    (R / f"{name}.npz").open("xb").write(npz)
    man = {"label": "POST HOC hit indicators for 09's RMS-bias analysis (note @ c10a14948, addendum @ 4ff49b1be); draw 0, fixed calibration; PRELIMINARY",
           "set": cfg["set"], "set_label": cfg.get("set_label"), "npz": f"{name}.npz",
           "npz_location": f"Modal Volume probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13: xfer/il_z/rms_hits/{name}.npz (the repo ignores *.npz)", "npz_sha256": hashlib.sha256(npz).hexdigest(),
           "array_key_format": "<set>|<line>|<point>|<probe label>; value int8 [cell, arm]: 1 hit, 0 not hit, -1 excluded (fold -1 cell on a 5-fold probe)",
           "cells": cfg["cells"], "arms": list(ARMS), "points": cfg["points"], "probes": [r_["label"] for r_ in cfg["rows"]],
           "lines": {"real": "tau_B on H_B units (g_B), all five arms",
                     "separate": "sim1/sim2 on g_sim at tau_sim from H_sim units; run1/run2/run3 at tau_B (only where H_sim is given)"},
           "pools": {o["label"]: o["lines"] for o in outs}, "H_sim_label": (cfg.get("H_sim") or {}).get("label"),
           "diag": {o["label"]: o["diag"] for o in outs},
           "meta": {"git": git("rev-parse", "HEAD").decode().strip(), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "config": config, "config_sha256": hashlib.sha256((HERE / config).read_bytes()).hexdigest(), "stamp_pt": stamp}}
    man["meta"] |= scope_meta(cfg)
    mb = json.dumps(man, indent=1, default=float).encode()
    (R / f"{name}.json").open("xb").write(mb)
    print(json.dumps({"npz": f"experiments/2026-10-15_il_z_e2e/results/{name}.npz", "npz_sha256": man["npz_sha256"],
                      "manifest": f"experiments/2026-10-15_il_z_e2e/results/{name}.json", "manifest_sha256": hashlib.sha256(mb).hexdigest()}, indent=1))


@app.function(image=image, cpu=(1, 1), memory=(2048, 2048), timeout=600, retries=0, volumes={"/solavg": outv})
def hit_means(name: str) -> dict:
    """Cross-check helper: per array, the mean hit rate per arm over cells that are not -1 (equal weight per cell = per main at draw 0)."""
    import numpy as np

    z = np.load(Path("/solavg") / "xfer" / "il_z" / "rms_hits" / f"{name}.npz")
    out = {}
    for k in z.files:
        H = z[k]
        out[k] = {a: (float(H[H[:, j] >= 0, j].mean()) if (H[:, j] >= 0).any() else None) for j, a in enumerate(ARMS)} | {"n_cells": int((H[:, 0] >= 0).sum())}
    return out


@app.function(image=image, cpu=(4, 4), memory=(16384, 16384), timeout=1800, retries=0, volumes={"/conv": conv.read_only()})
def census(cfg: dict, row: dict) -> dict:
    """09's request (a): per pool and arm, Sol units and runs that are MAPPED (fold >= 0, finite score) vs fold -1 / non-finite, for a 5-fold row."""
    import numpy as np

    def load(path, pin):
        b = (Path("/conv") / path.lstrip("/").removeprefix("work/")).read_bytes()
        if not re.fullmatch("[0-9a-f]{64}", str(pin)) or hashlib.sha256(b).hexdigest() != pin:
            raise SystemExit(f"{path}: sha256 differs from the pin {pin}")
        return b

    def folder_rows(folder):
        man = cfg["folders"][folder][row["sub"]]
        fe = json.loads(load(man["path"], man["sha256"]))["files"]
        ent = fe[row["file"]]
        d = man["path"].replace("/work/", "").rsplit("/", 1)[0]
        return dict(np.load(io.BytesIO(load(f"{d}/{row['file']}", ent if isinstance(ent, str) else ent["sha256"])), allow_pickle=False))

    def count(k, s, f):  # split (2e 14:57): fold -1 vs non-finite score, by unit and by run
        fin, neg = np.isfinite(s), f < 0
        ok = ~neg & fin
        cls = np.where(ok, "mapped", np.where(neg & ~fin, "fold-1_and_nonfinite", np.where(neg, "fold-1_finite", "fold>=0_nonfinite")))
        runs = {}
        for kk, c_ in zip(k.tolist(), cls.tolist()):
            runs.setdefault(kk, set()).add(c_)
        rc = {}
        for v in runs.values():
            nm = next(iter(v)) if len(v) == 1 else "mixed"
            rc[nm] = rc.get(nm, 0) + 1
        return {"units_mapped": int(ok.sum()), "units_fold_minus1_and_nonfinite": int((neg & ~fin).sum()), "units_fold_minus1_finite": int((neg & fin).sum()),
                "units_fold_ge0_nonfinite": int((~neg & ~fin).sum()), "runs_by_class": rc, "fold_values": sorted(set(f.tolist()))}

    out = {"npz_fields": {}}
    keys = set(cfg["H_B"]["keys"])
    parts = [folder_rows(fol) for fol in cfg["H_B"]["folders"]]
    out["npz_fields"]["H_B"] = sorted({x for z in parts for x in z})
    k = np.concatenate([z["key"].astype(str) for z in parts])
    m = np.isin(k, sorted(keys))
    out["H_B"] = count(k[m], np.concatenate([z["score"].astype(np.float64) for z in parts])[m], np.concatenate([z["fold"].astype(np.int64) for z in parts])[m])
    ks, ss, fs = [], [], []
    for rt in cfg["H_sim"]["roots"]:
        ent = rt[row["sub"]]
        z = dict(np.load(io.BytesIO(root_bytes(load, ent, row["file"])), allow_pickle=False))
        out["npz_fields"].setdefault("H_sim", sorted(z))
        ks.append(z["key"].astype(str))
        ss.append(z["score"].astype(np.float64))
        fs.append(z["fold"].astype(np.int64))
    out["H_sim"] = count(np.concatenate(ks), np.concatenate(ss), np.concatenate(fs))
    cache = {}
    for arm in ARMS:
        kk_, s_, f_ = [], [], []
        for c in cfg["cells"]:
            fol, key = cfg["arm_rows"][arm][c]
            if fol not in cache:
                cache[fol] = folder_rows(fol)
            z = cache[fol]
            ix = z["key"].astype(str) == key
            kk_.append(z["key"].astype(str)[ix])
            s_.append(z["score"].astype(np.float64)[ix])
            f_.append(z["fold"].astype(np.int64)[ix])
        out[f"arm {arm}"] = count(np.concatenate(kk_), np.concatenate(s_), np.concatenate(f_))
    return {"label": row["label"], "census": out}


DEV_LABEL = "deviation from 25h §1: per-model moments from all pool units (no mapped units on nm360)"


@app.function(image=image, cpu=(4, 4), memory=(16384, 16384), timeout=1800, retries=0, volumes={"/conv": conv.read_only()})
def deviation(cfg: dict, row: dict, main_of: dict) -> dict:  # noqa: C901
    """2e/0d GO (14:5x): Sol (a)/c_asis on S_newmains, never leading. Per pool: per-model moments (ddof 0) over ALL the pool's units from score_by_model,
    standardize each model, take the five-model mean, re-standardize that mean on the same pool; no borrowing across pools. Attack units use the pool's
    parameters. Draw 0, the 507 unit rule (gated against fc.thr). Plus the leakage check: pool/arm mains vs each fold model's training mains."""
    import numpy as np

    from probe_shift import fpr_core as fc

    lab = row["label"]
    diag = {"files": {}, "read_started_pt": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S")}

    def stop(msg):
        raise SystemExit(f"{cfg['set']} {lab} (deviation): gate failed: {msg}")

    def load(path, pin):
        b = (Path("/conv") / path.lstrip("/").removeprefix("work/")).read_bytes()
        s_ = hashlib.sha256(b).hexdigest()
        if not re.fullmatch("[0-9a-f]{64}", str(pin)) or s_ != pin:
            stop(f"{path}: sha256 differs from the pin {pin}")
        diag["files"][path] = s_
        return b

    cache = {}

    def folder(fol):
        if fol not in cache:
            man = cfg["folders"][fol][row["sub"]]
            mj = json.loads(load(man["path"], man["sha256"]))
            ent = mj["files"][row["file"]]
            d = man["path"].replace("/work/", "").rsplit("/", 1)[0]
            z = dict(np.load(io.BytesIO(load(f"{d}/{row['file']}", ent if isinstance(ent, str) else ent["sha256"])), allow_pickle=False))
            tpin = mj["training_manifest_sha256_by_fold" if probe == "a_tight" else "training_manifest_sha256_by_fold_c_asis"]
            cache[fol] = (z["key"].astype(str), z["score_by_model"].astype(np.float64), d, tpin)
        return cache[fol]

    probe = row["file"].split("__")[0]  # a_tight / c_asis_tight
    train = {}
    k_, x_ = [], []
    for fol in cfg["H_B"]["folders"]:
        k, x, d, tpin = folder(fol)
        m = np.isin(k, sorted(cfg["H_B"]["keys"]))
        if set(k[m].tolist()) & {kk for ka in k_ for kk in ka.tolist()}:
            stop("H_B keys in two folders")  # one owner per key
        k_.append(k[m])
        x_.append(x[m])
        for f_ in range(5):
            tm = sorted(json.loads(load(f"{d}/training_mains__{probe}__f{f_}.json", tpin[str(f_)]))["training_mains"])
            if train.setdefault(f_, tm) != tm:
                stop(f"training mains of fold model {f_} differ across folders")
    kB, xB = np.concatenate(k_), np.concatenate(x_)
    if set(kB.tolist()) != set(cfg["H_B"]["keys"]) or len(set(kB.tolist())) != len(cfg["H_B"]["keys"]):
        stop("H_B keys missing")
    k_, x_, seen = [], [], set()
    for rt in cfg["H_sim"]["roots"]:
        ent = rt[row["sub"]]
        z = dict(np.load(io.BytesIO(root_bytes(load, ent, row["file"])), allow_pickle=False))
        ks_ = set(z["key"].astype(str).tolist())
        if {x.split("|", 1)[1] for x in ks_} & set(cfg.get("s4_cells_excluded") or ()):
            stop(f"H_sim {rt['root']}: S4 cells in H_sim under scope (a)")
        if ks_ & seen:
            stop("a sim-honest run appears in two roots")  # Codex r3: same duplicate check as the standard path
        seen |= ks_
        k_.append(z["key"].astype(str))
        x_.append(z["score_by_model"].astype(np.float64))
    kS, xS = np.concatenate(k_), np.concatenate(x_)
    for nm, x in (("H_B", xB), ("H_sim", xS)):
        if x.ndim != 2 or x.shape[1] != 5 or not np.isfinite(x).all():
            stop(f"{nm}: score_by_model is not a finite (n, 5) array")

    def fit(x):
        mu, sd = x.mean(0), x.std(0)
        if not (sd > 0).all():
            stop("a model's scores are constant on the pool")
        m = ((x - mu) / sd).mean(1)
        return mu, sd, m.mean(), m.std()

    def apply(p, x):
        mu, sd, M, S = p
        return (((x - mu) / sd).mean(1) - M) / S

    pars, lines = {}, {}
    for nm, x, k in (("H_B", xB, kB), ("H_sim", xS, kS)):
        pars[nm] = fit(x)
        z = apply(pars[nm], x)
        zs = np.sort(z)[::-1]
        inf = {"n_units": int(len(z)), "n_runs": int(len(set(k.tolist())))}
        for a in cfg["points"]:
            kk = int(np.floor(a * len(z) + 1e-9))
            T = float(zs[min(kk, len(z) - 1)])
            if kk >= 1 and T != fc.thr(z, np.ones((1, len(z))), a)[0]:
                stop("draw-0 threshold differs from fc.thr")
            ab = int((z > T).sum())
            inf[f"{a:g}"] = {"k": kk, "T": T, "units_above": ab, "estimable": kk >= 1, "shaded": ab < 10}
        lines[nm] = inf
    cells = cfg["cells"]
    runmax = {}
    for arm in ARMS:
        for c in cells:
            fol, key = cfg["arm_rows"][arm][c]
            k, x, _, _ = folder(fol)
            ix = k == key
            if not ix.any() or not np.isfinite(x[ix]).all():
                stop(f"{arm} {key}: no rows or non-finite score_by_model")
            runmax[(arm, c)] = {nm: float(apply(pars[nm], x[ix]).max()) for nm in pars}
    hits = {}
    for a in cfg["points"]:
        for ln in ("real", "separate"):
            H = np.zeros((len(cells), len(ARMS)), np.int8)
            for j, arm in enumerate(ARMS):
                nm = "H_sim" if (ln == "separate" and arm in SIM_ARMS) else "H_B"
                T = lines[nm][f"{a:g}"]["T"]
                H[:, j] = np.array([runmax[(arm, c)][nm] > T for c in cells], np.int8)
            hits[f"DEVIATION|{cfg['set']}|{ln}|{a:g}|{lab}"] = H
    # leakage check (2e): mains of every unit's cell vs each fold model's training mains
    cell = lambda key: key.split("|", 1)[1]  # noqa: E731
    mains = {"H_B": sorted({main_of[cell(k)] for k in set(kB.tolist())}), "H_sim": sorted({main_of[cell(k)] for k in set(kS.tolist())}),
             "attack arms": sorted({main_of[c] for c in cells})}
    leak = {nm: {f"fold_model_{f_}": len(set(ms) & set(train[f_])) for f_ in range(5)} | {"n_mains": len(ms)} for nm, ms in mains.items()}
    diag["training_mains_n"] = {f"fold_model_{f_}": len(train[f_]) for f_ in range(5)}
    buf = io.BytesIO()
    np.savez_compressed(buf, **hits)
    return {"label": lab, "npz": buf.getvalue(), "lines": lines, "leakage": leak, "diag": diag}


@app.function(image=image, cpu=(1, 1), memory=(4096, 4096), timeout=600, retries=0, volumes={"/conv": conv.read_only()})
def npz_fields(paths: list) -> dict:
    """Metadata only: the array names (and shapes) in each npz, to check whether per-model Sol scores (score_by_model) exist."""
    import numpy as np

    out = {}
    for p_ in paths:
        f_ = Path("/conv") / p_
        if not f_.exists():
            out[p_] = "missing"
            continue
        z = np.load(f_)
        out[p_] = {k: list(z[k].shape) for k in z.files}
    return out


@app.function(image=image, cpu=(4, 4), memory=(16384, 16384), timeout=1800, retries=0, volumes={"/conv": conv.read_only()})
def compute_s1(cfg: dict, row: dict) -> dict:  # noqa: C901
    """5-fold rows under 25h §1 (draw 0, fixed calibration): per pool, mapped units z = (s - mu_f)/sd_f from the pool's mapped fold-f units; fold -1 units
    m = mean_f (s_f - mu_f)/sd_f from score_by_model, z0 = (m - mu_0)/sd_0 with mu_0, sd_0 from the pool's own fold -1 units. The 507 rule (gated against
    fc.thr); a run is hit iff any unit is strictly above T."""
    import numpy as np

    from probe_shift import fpr_core as fc

    lab = row["label"]
    diag = {"files": {}, "read_started_pt": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S")}

    def stop(msg):
        raise SystemExit(f"{cfg['set']} {lab} (§1): gate failed: {msg}")

    def load(path, pin):
        b = (Path("/conv") / path.lstrip("/").removeprefix("work/")).read_bytes()
        s_ = hashlib.sha256(b).hexdigest()
        if not re.fullmatch("[0-9a-f]{64}", str(pin)) or s_ != pin:
            stop(f"{path}: sha256 differs from the pin {pin}")
        diag["files"][path] = s_
        return b

    cache = {}

    def folder(fol):
        if fol not in cache:
            man = cfg["folders"][fol][row["sub"]]
            fe = json.loads(load(man["path"], man["sha256"]))["files"]
            ent = fe[row["file"]]
            d = man["path"].replace("/work/", "").rsplit("/", 1)[0]
            z = dict(np.load(io.BytesIO(load(f"{d}/{row['file']}", ent if isinstance(ent, str) else ent["sha256"])), allow_pickle=False))
            cache[fol] = z
        return cache[fol]

    fold_cell = cfg["cell_fold"]

    def take(z, m):  # (key, score, fold, score_by_model) rows under mask m, gated against r36
        k, s, f = z["key"].astype(str)[m], z["score"].astype(np.float64)[m], z["fold"].astype(np.int64)[m]
        exp = np.array([fold_cell[x.split("|", 1)[1]] for x in k], np.int64)
        if not np.array_equal(f, exp):
            stop("unit folds differ from r36's fold of the cell's main")
        if not np.isfinite(s[f >= 0]).all():
            stop("non-finite designated scores on mapped units")
        sb = np.full((len(k), 5), np.nan)
        if (f < 0).any():
            if "score_by_model" not in z:
                stop("fold -1 units without score_by_model")
            sb[f < 0] = z["score_by_model"].astype(np.float64)[m][f < 0]
            if not np.isfinite(sb[f < 0]).all():
                stop("non-finite score_by_model on fold -1 units")
        return k, s, f, sb

    parts, owners = [], set()
    for fol in cfg["H_B"]["folders"]:
        z = folder(fol)
        m = np.isin(z["key"].astype(str), sorted(cfg["H_B"]["keys"]))
        ks_ = set(z["key"].astype(str)[m].tolist())
        if ks_ & owners:
            stop("H_B keys in two folders")
        owners |= ks_
        parts.append(take(z, m))
    if owners != set(cfg["H_B"]["keys"]):
        stop("H_B keys missing")
    HB = tuple(np.concatenate([p_[i] for p_ in parts]) for i in range(4))
    parts, seen = [], set()
    for rt in cfg["H_sim"]["roots"]:
        ent = rt[row["sub"]]
        z = dict(np.load(io.BytesIO(root_bytes(load, ent, row["file"])), allow_pickle=False))
        m = np.ones(len(z["key"]), bool)
        ks_ = set(z["key"].astype(str).tolist())
        if {x.split("|", 1)[1] for x in ks_} & set(cfg.get("s4_cells_excluded") or ()):
            stop(f"H_sim {rt['root']}: S4 cells in H_sim under scope (a)")
        if ks_ & seen:
            stop("a sim-honest run appears in two roots")
        seen |= ks_
        if "score_by_model" not in z:
            z["score_by_model"] = np.full((len(z["key"]), 5), np.nan)
        parts.append(take(z, m))
    HS = tuple(np.concatenate([p_[i] for p_ in parts]) for i in range(4))

    def fit(P):
        k, s, f, sb = P
        mu, sd = np.zeros(5), np.ones(5)
        for g in range(5):
            x = s[f == g]
            if not (len(x) >= 2 and x.std() > 0):
                stop(f"pool fold {g} empty or constant")
            mu[g], sd[g] = x.mean(), x.std()
        neg = f < 0
        mm = ((sb[neg] - mu) / sd).mean(1)
        if neg.any() and not (neg.sum() >= 2 and mm.std() > 0):
            stop("pool fold -1 units too few or constant")
        mu0, sd0 = (mm.mean(), mm.std()) if neg.any() else (np.nan, np.nan)
        return mu, sd, mu0, sd0

    def tr(p, s, f, sb):
        mu, sd, mu0, sd0 = p
        z = np.empty(len(s))
        mp = f >= 0
        z[mp] = (s[mp] - mu[f[mp]]) / sd[f[mp]]
        if (~mp).any():
            if not np.isfinite(mu0):
                stop("a fold -1 unit needs mu_0/sd_0 but the pool has no fold -1 units")
            z[~mp] = (((sb[~mp] - mu) / sd).mean(1) - mu0) / sd0
        return z

    pars, lines = {}, {}
    for nm, P in (("H_B", HB), ("H_sim", HS)):
        pars[nm] = fit(P)
        z = tr(pars[nm], P[1], P[2], P[3])
        zs = np.sort(z)[::-1]
        inf = {"n_units": int(len(z)), "n_runs": int(len(set(P[0].tolist()))), "fold_minus1_units": int((P[2] < 0).sum()),
               "fold_minus1_runs": int(len(set(P[0][P[2] < 0].tolist())))}
        for a in cfg["points"]:
            kk = int(np.floor(a * len(z) + 1e-9))
            T = float(zs[min(kk, len(z) - 1)])
            if kk >= 1 and T != fc.thr(z, np.ones((1, len(z))), a)[0]:
                stop("draw-0 threshold differs from fc.thr")
            ab = int((z > T).sum())
            inf[f"{a:g}"] = {"k": kk, "T": T, "units_above": ab, "estimable": kk >= 1, "shaded": ab < 10}
        lines[nm] = inf
    cells = cfg["cells"]
    runmax = {}
    for arm in ARMS:
        for c in cells:
            fol, key = cfg["arm_rows"][arm][c]
            z = folder(fol)
            ix = z["key"].astype(str) == key
            if not ix.any():
                stop(f"{arm} {key}: no rows")
            k, s, f, sb = take(z, ix)
            runmax[(arm, c)] = {nm: float(tr(pars[nm], s, f, sb).max()) for nm in pars}
    hits = {}
    for a in cfg["points"]:
        for ln in ("real", "separate"):
            H = np.zeros((len(cells), len(ARMS)), np.int8)
            for j, arm in enumerate(ARMS):
                nm = "H_sim" if (ln == "separate" and arm in SIM_ARMS) else "H_B"
                H[:, j] = np.array([runmax[(arm, c)][nm] > lines[nm][f"{a:g}"]["T"] for c in cells], np.int8)
            hits[f"{cfg['set']}|{ln}|{a:g}|{lab}"] = H
    diag["fold_minus1_cells"] = [c for c in cells if fold_cell.get(c, -1) < 0]
    buf = io.BytesIO()
    np.savez_compressed(buf, **hits)
    return {"label": lab, "npz": buf.getvalue(), "lines": lines, "diag": diag}


@app.function(image=image, cpu=(2, 2), memory=(8192, 8192), timeout=1200, retries=0, volumes={"/conv": conv.read_only()})
def leakage_addendum(cfg_nm: dict, folders_311: list, inp: dict) -> dict:  # noqa: C901
    """09's gate c6023b805: the SAME overlap code as deviation()'s leakage check, run (1) on the nm360 pools and arms and (2) on the +311 mains as the
    positive control (expected per fold model k: the +311 mains whose r36 fold is neither k nor -1), plus (3) the by-key provenance of Sol's layer and
    readout selection and fitted preprocessing (the T196 runs of the avg_v1 selection, split by r36 fold)."""
    import numpy as np

    out = {"files": {}}

    def load(path):
        b = (Path("/conv") / path.lstrip("/").removeprefix("work/")).read_bytes()
        out["files"][path] = hashlib.sha256(b).hexdigest()
        return b

    def train_from(folder_dir, probe):  # training mains per fold model + the manifest's own pin for each file
        man = json.loads(load(f"{folder_dir}/manifest_sol.json"))
        tm, pins = {}, {"manifest_training_manifest_sha256_by_fold": man.get("training_manifest_sha256_by_fold" if probe == "a_tight"
                                                                            else "training_manifest_sha256_by_fold_c_asis")}
        for k in range(5):
            b = load(f"{folder_dir}/training_mains__{probe}__f{k}.json")
            tm[k] = sorted(json.loads(b)["training_mains"])
            pins[f"training_mains_file_f{k}_sha256"] = hashlib.sha256(b).hexdigest()
        return tm, pins

    def overlap(mains, tm):  # the SAME rule as deviation(): |set(mains) & set(training mains of model k)| per k
        return {f"fold_model_{k}": len(set(mains) & set(tm[k])) for k in range(5)} | {"n_mains": len(set(mains))}

    nm_main = {c["key"]: c["main"] for c in inp["nm360_cells"]}
    p311_main = inp["cell_main_311"]
    r36 = {}
    for r in inp["r36"]:
        r36.setdefault(r["main_task"], r["fold"])
    res = {}
    for probe in ("a_tight", "c_asis_tight"):
        tms, pinsets = [], []
        for d in [f"scaleup/{fo}/sol" for fo in cfg_nm["H_B"]["folders"]] + [f"scaleup/{fo}/sol" for fo in folders_311]:
            tm, pins = train_from(d, probe)
            tms.append(tm)
            pinsets.append((d, pins))
        if any(t != tms[0] for t in tms):
            raise SystemExit(f"{probe}: training mains differ across folders")
        tm = tms[0]
        # (1) nm360 pools and arms, by key
        hb = sorted({nm_main[k.split("|", 1)[1]] for k in cfg_nm["H_B"]["keys"]})
        hs = set()
        for rt in cfg_nm["H_sim"]["roots"]:
            ent = rt["sol"]
            z = np.load(io.BytesIO(load(f"{ent['path'].strip('/')}/{probe}__avg_top2.npz")), allow_pickle=False)
            hs |= {nm_main[k.split("|", 1)[1]] for k in z["key"].astype(str)}
        arms = sorted({nm_main[c] for c in cfg_nm["cells"]})
        # (2) +311 positive control
        all311 = sorted(set(p311_main.values()))
        expect = {f"fold_model_{k}": sum(1 for m in all311 if r36.get(m, -1) not in (k, -1)) for k in range(5)}
        # (3) selection / preprocessing provenance: avg_v1 selection used T196 runs; for outer fold f, every fit (outer and inner) used runs with fold != f
        t196 = sorted({c["main"] for c in inp["t196_cells"]})
        sel_f = {k: sorted(m for m in t196 if r36.get(m, -1) not in (k, -1)) for k in range(5)}
        res[probe] = {
            "training_mains_n": {f"fold_model_{k}": len(tm[k]) for k in range(5)},
            "training_file_pins": dict(pinsets),
            "nm360": {"H_B": overlap(hb, tm), "H_sim": overlap(sorted(hs), tm), "attack_arms": overlap(arms, tm),
                      "H_A": "covered through H_sim's mains: H_A = honest_a on exactly H_sim's cells (25f)"},
            "positive_control_plus311": {"overlap": overlap(all311, tm), "expected_from_r36": expect,
                                         "match": all(overlap(all311, tm)[k] == v for k, v in expect.items())},
            "selection_and_preprocessing": {
                "t196_mains_n": len(t196),
                "selection_fold_set_subset_of_training_mains": {f"fold_model_{k}": set(sel_f[k]) <= set(tm[k]) for k in range(5)},
                "t196_mains_unmapped_in_r36": sorted(m for m in t196 if r36.get(m, -1) < 0),
                "nm360_mains_in_t196": len(set(nm_main.values()) & set(t196)),
                "nm360_mains_in_r36": len(set(nm_main.values()) & set(r36))}}
    out["result"] = res
    return out
