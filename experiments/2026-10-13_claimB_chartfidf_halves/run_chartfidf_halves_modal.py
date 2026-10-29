"""POST HOC (RULE.md in this folder; Aswin about 00:13 PT 10-13 "yes", and 0d's "(A)", via b7): character-only TF-IDF
through the halves design. A copy of run_tfidf_halves_modal.py @ faafd24c; only the vectorizers, the reference gate, the
unit checks and arm W's text are added. One Modal CPU container:
  1. gate 1: gate (a)'s level1 dev_to_test rows reproduced exactly (faafd24c's step, unchanged);
  2. the halves' text, exactly as faafd24c built it (evening in both arms, level 1, plant-2 copies);
  3. gate 2: faafd24c's word + character TF-IDF refit per direction and comparison, every run score equal to the
     authenticated reference (passed in by the launcher) within 1e-9, same keys and metadata, same 1% alarm decisions;
     then gate 3, unit checks: q1's char_ngrams keeps whitespace runs and case; the arm-W statement on a known string;
  4. variants (a checkpoint on the Volume after each): character-only 1-5 and 2-5 on the halves' text; character-only 1-5 on arm W's text (scrub.py's level-3
     whitespace statement on every kept output's level-1 text).
Returns and writes to the Volume run scores and counts only (keys, arms, numbers; no text). analyze_chartfidf_halves.py
applies the rule.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with pyyaml modal deploy \\
      experiments/2026-10-13_claimB_chartfidf_halves/run_chartfidf_halves_modal.py
  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with pyyaml python \\
      experiments/2026-10-13_claimB_chartfidf_halves/run_chartfidf_halves_modal.py launch <gate_a_main_2026-10-09_185451.json> \\
      <tfidf_halves_run_scores_2026-10-12_111241.json.gz>
  (if the client is lost: `... run_chartfidf_halves_modal.py collect <stamp>` reads that invocation's Volume copy)
"""

import gzip
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import modal

HERE = Path(__file__).parent
LOCAL = modal.is_local()
FIXED = HERE.parent / "2026-10-09_fixed_sim" if LOCAL else Path("/root")
sys.path.insert(0, str(FIXED))
import run_gate_a_modal as G  # noqa: E402  (gate (a)'s code, unchanged)
import run_tfidf_slips_evened_modal as S  # noqa: E402  (the evening, unchanged)

APP = "probe-shift_aswin-surya_sub-claimb-chartfidf_kill-2026-11-12"
SCRIPT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
OUT = f"claimB/chartfidf_halves_v1/{SCRIPT_SHA[:12]}"
PACK_RUNS = "claimB/level1/packed_39f842231a46/runs.L63.json"  # step A's pack: its plant-2 arm names the 150 test cells
DEV_CELLS = "claimB/slips_evened_dev_v1/cells.json"  # B-dev: its 100 dev planted cells
SIM, E1, E2, PL2 = "sim-attack", "real-attack-e1", "real-attack-e2", "sim-attack-plant2"
GATE_ROWS = ("fixed_sim_vs_e1", "e2_vs_e1")
REF_SHA = "344b331ce5b34c625b248164430e280b4b340aa8b2a2ee7ff051aef117db049d"  # faafd24c result key `inputs`
REF_VOLUME = "claimB/tfidf_halves_v1/745179164150/2026-10-12_111241/run_scores.json"  # for information only
VARIANTS = (("char1_5", (1, 5), "level1"), ("char2_5", (2, 5), "level1"), ("char1_5_W", (1, 5), "W"))  # RULE.md

app = modal.App(APP)
if LOCAL:
    MC, SWAP = HERE.parent / "2026-10-10_claimB_meaning_control", HERE.parent / "2026-10-07_q2_rendering_swap"
    image = (G.image.add_local_file(FIXED / "run_gate_a_modal.py", "/root/run_gate_a_modal.py")
             .add_local_file(FIXED / "run_tfidf_slips_evened_modal.py", "/root/run_tfidf_slips_evened_modal.py")
             .add_local_file(MC / "undo_r2.py", "/root/undo_r2.py")
             .add_local_file(SWAP / "rerender.py", "/root/swap/rerender.py")
             .add_local_file(SWAP / "str_replace_tells.py", "/root/swap/str_replace_tells.py"))
else:
    image = G.image


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def code_paths(root: Path | None = None) -> dict:
    """The code the job runs, by name: local paths, or (root given) the container's copies."""
    import probe_shift

    ps = Path(probe_shift.__file__).parent
    if root is not None:
        return {"run_gate_a_modal.py": root / "run_gate_a_modal.py", "run_tfidf_slips_evened_modal.py": root / "run_tfidf_slips_evened_modal.py",
                "analyze_tfidf.py": root / "analyze_tfidf.py", "undo_r2.py": root / "undo_r2.py", "rerender.py": root / "swap" / "rerender.py",
                "str_replace_tells.py": root / "swap" / "str_replace_tells.py", "scrub.py": ps / "scrub.py"}
    mc, swap = HERE.parent / "2026-10-10_claimB_meaning_control", HERE.parent / "2026-10-07_q2_rendering_swap"
    return {"run_gate_a_modal.py": FIXED / "run_gate_a_modal.py", "run_tfidf_slips_evened_modal.py": FIXED / "run_tfidf_slips_evened_modal.py",
            "analyze_tfidf.py": G.Q1 / "analyze_tfidf.py", "undo_r2.py": mc / "undo_r2.py", "rerender.py": swap / "rerender.py",
            "str_replace_tells.py": swap / "str_replace_tells.py", "scrub.py": ps / "scrub.py"}


@app.function(image=image, volumes={"/acts": G.acts}, cpu=16, memory=65536, timeout=4200, retries=0)
def job(cfg: dict, split: dict, folds: dict, archived: dict, expect: dict, prov: dict, reference: list) -> dict:
    import numpy as np

    import analyze_tfidf as A
    from probe_shift.scrub import BLANK_RUNS, TRAILING_SPACE
    from undo_r2 import undo_r2

    def ws(t):  # src/probe_shift/scrub.py line 93, the level-3 whitespace statement alone (arm W; RULE.md)
        return BLANK_RUNS.sub("\n\n", TRAILING_SPACE.sub("", t)).rstrip()

    t0 = time.time()
    code = {k: sha(v) for k, v in code_paths(Path("/root")).items()}
    if SCRIPT_SHA != expect["script_sha256"] or code != expect["code_sha256"]:
        raise SystemExit("the deployed code differs from the launcher's: redeploy from the commit being launched")
    out = {"script_sha256": SCRIPT_SHA, "code_sha256": code, "provenance": prov, "timing": {}, "counts": {}}
    dst = Path("/acts") / OUT / prov["stamp"]
    dst.mkdir(parents=True, exist_ok=False)  # one folder per invocation

    # 1. gate: gate (a)'s level1 dev_to_test rows, reproduced exactly
    d, X, env, _, _ = G.prepare(cfg, "level1", split, folds)
    test = np.isin(env, split["test"])
    gate = {}
    for comp in GATE_ROWS:
        neg, pos = cfg["comparisons"][comp]
        R = np.isin(d["arm"], [neg, pos])
        y = (d["arm"] == pos).astype(int)
        s = G.score_jobs(X, y, G.design_jobs("dev_to_test", R, env, d, split), cfg)
        a = G.run_auroc(s, y, d, env, R & test, split["test"], cfg, boot=False)["auroc"]
        gate[comp] = {"auroc": a, "archived": archived[comp], "match": abs(a - archived[comp]) <= 1e-9}
    out["gate"] = gate | {"passed": all(g["match"] for g in gate.values())}
    out["timing"]["gate"] = round(time.time() - t0, 1)
    del X
    print("gate", json.dumps(out["gate"]), flush=True)
    if not out["gate"]["passed"]:
        (dst / "run_scores.json").write_text(json.dumps(out))
        G.acts.commit()
        return out

    # 2. the halves' text: evening in both arms on raw text, then level 1; plant-2 copies of the planted cells
    d = A.load(Path("/acts") / cfg["run"] / "examples", folds["environment_folds"]["environment_to_fold"], fold_key="environment")
    rows = d["rows"]
    pack = json.loads((Path("/acts") / PACK_RUNS).read_text())
    plant_test = sorted({r["cell"] for r in pack if r["arm"] == "sim-attack-fixed-plant2"})
    dev_cells = json.loads((Path("/acts") / DEV_CELLS).read_text())
    plant_dev = sorted(dev_cells["planted"])
    env_of = {r["cell"]: r["environment"] for r in rows}
    if len(plant_test) != 150 or len(plant_dev) != 100 or set(plant_test) & set(plant_dev) \
            or any(env_of[c] not in split["test"] for c in plant_test) or any(env_of[c] not in split["dev"] for c in plant_dev):
        raise SystemExit("plant-2 cells are not 150 test and 100 dev cells")
    planted = set(plant_test) | set(plant_dev)
    cols = {"arm": [], "key": [], "cell": [], "env": [], "task": []}
    texts, changed, sites = [], Counter(), Counter()

    pairs = {}  # planted cell -> [(fixed-sim row, its plant-2 copy's row)]

    def add(r, arm, key, text):
        for k, v in (("arm", arm), ("key", key), ("cell", r["cell"]), ("env", r["environment"]), ("task", r["main_task"])):
            cols[k].append(v)
        texts.append(A.scrub(text, 1))
        return len(texts) - 1

    for r, cmd in zip(rows, d["editor_command"]):
        raw = r[cfg["text"]]
        if S.RANGE in raw:
            changed[f"{r['arm']}|range_dropped"] += 1
            continue
        t, hit = S.even(raw, cmd, r["tool"])
        changed.update(f"{r['arm']}|{h}" for h in hit)
        i = add(r, r["arm"], r["key"], t)
        if r["arm"] == SIM and r["cell"] in planted:
            u = undo_r2(t) if r["tool"] == "text_editor" and cmd == "str_replace" else None
            sites[r["cell"]] += bool(u)
            pairs.setdefault(r["cell"], []).append((i, add(r, PL2, r["key"] + "__plant2", u or t)))
    arm, key, cell, env, task = (np.array(cols[k]) for k in ("arm", "key", "cell", "env", "task"))
    keys_of = {}
    for c, a, k in zip(cell.tolist(), arm.tolist(), key.tolist()):
        keys_of.setdefault((c, a), set()).add(k)
    cells3 = {c for c in set(cell.tolist()) if all((c, a) in keys_of for a in (SIM, E1, E2))}
    if len(cells3) != 939 or len(set(cell.tolist())) != 939 or any(len(v) != 1 for v in keys_of.values()) \
            or any((c, PL2) not in keys_of for c in planted):
        raise SystemExit("cells: not 939 cells with one run in each arm and every planted cell's copy")
    out["counts"] = {"kept_outputs": dict(Counter(arm.tolist())), "evening": dict(changed),
                     "plant_cells": {"test": len(plant_test), "dev": len(plant_dev)},
                     "plant_pairs_with_a_changed_output": sum(sites[c] > 0 for c in planted), "plant_changed_outputs": int(sum(sites.values())),
                     "plant_changed_outputs_by_cell": {c: int(sites[c]) for c in sorted(planted)},
                     "runs": {a: int(len(set(key[arm == a]))) for a in (SIM, E1, E2, PL2)},
                     "inputs_sha256": {PACK_RUNS: sha(Path("/acts") / PACK_RUNS), DEV_CELLS: sha(Path("/acts") / DEV_CELLS)}}
    dirs = {"dev_to_test": (split["dev"], split["test"]), "test_to_dev": (split["test"], split["dev"])}
    comps = {"fixed_vs_e1": (E1, SIM, [SIM, E1, E2]), "e2_vs_e1": (E1, E2, [E1, E2]), "plant2": (SIM, PL2, [SIM, PL2])}
    in_plant = np.isin(cell, sorted(planted))

    def fits(X, variant, cv):  # faafd24c's fits per direction, unchanged, with the variant named in each record
        recs = []
        for comp, (neg, pos, scored) in comps.items():
            t1 = time.time()
            y = (arm == pos).astype(int)
            base = np.isin(arm, [neg, pos]) & (in_plant if comp == "plant2" else True)
            jobs = [(base & np.isin(env, tr), np.isin(arm, scored) & np.isin(env, te) & (in_plant if comp == "plant2" else True))
                    for tr, te in dirs.values()]
            s = G.score_jobs(X, y, jobs, cv)
            if not all(np.isfinite(s[te]).all() for _, te in jobs):
                raise SystemExit(f"{variant} {comp}: a scored row has no finite score")
            for dname, (_, te) in zip(dirs, jobs):
                run_score, first = A.run_means(s, key, te)
                n = Counter(key[te].tolist())
                recs += [{"variant": variant, "direction": dname, "comparison": comp, "key": str(key[i]), "arm": str(arm[i]), "cell": str(cell[i]),
                          "environment": str(env[i]), "main_task": str(task[i]), "score": float(v), "rows": n[str(key[i])]}
                         for i, v in zip(first, run_score)]
            out["timing"][f"{variant}|{comp}"] = round(time.time() - t1, 1)
            print(variant, comp, "fitted in", out["timing"][f"{variant}|{comp}"], "s", flush=True)
        return recs

    # 3. gate 2: faafd24c's word + character TF-IDF refit, every run score equal to the authenticated reference
    t1 = time.time()
    X = A.featurize(texts, cfg)
    out["timing"]["featurize|wordchar"] = round(time.time() - t1, 1)
    print("featurized", X.shape, json.dumps(out["counts"]["runs"]), flush=True)
    records = fits(X, "wordchar", cfg)
    del X
    meta = ("arm", "cell", "environment", "main_task", "rows")
    ref = {(r["direction"], r["comparison"], r["key"]): r for r in reference}
    mine = {(r["direction"], r["comparison"], r["key"]): r for r in records}
    diffs = [abs(mine[k]["score"] - ref[k]["score"]) for k in ref if k in mine]

    def flagged(src, d):  # the 1% alarm on held-out e1 (RULE.md), flagged = strictly above
        rr = [r for (dd, c, _), r in src.items() if dd == d and c == "fixed_vs_e1"]
        e1 = np.sort([r["score"] for r in rr if r["arm"] == E1])[::-1]
        t = e1[int(np.floor(0.01 * len(e1)))]
        return sorted(r["key"] for r in rr if r["arm"] in (SIM, E2) and r["score"] > t)

    g2 = {"reference_records": len(reference), "reference_unique": len(ref), "refit_records": len(records), "refit_unique": len(mine),
          "same_keys": set(ref) == set(mine), "all_finite": all(np.isfinite(r["score"]) for r in records + list(reference)),
          "metadata_equal": all(all(mine[k][m] == ref[k][m] for m in meta) for k in ref if k in mine),
          "max_abs_diff": float(max(diffs)) if diffs else None}
    g2["alarm_decisions_equal"] = g2["same_keys"] and all(flagged(mine, d) == flagged(ref, d) for d in dirs)
    g2["passed"] = (g2["reference_records"] == g2["reference_unique"] == g2["refit_records"] == g2["refit_unique"] and g2["same_keys"]
                    and g2["all_finite"] and g2["metadata_equal"] and g2["max_abs_diff"] is not None and g2["max_abs_diff"] <= 1e-9
                    and g2["alarm_decisions_equal"])
    vol_copy = Path("/acts") / REF_VOLUME
    g2["volume_copy_sha256"] = sha(vol_copy) if vol_copy.exists() else None  # information only (RULE.md)
    out["gate2"] = g2
    out["timing"]["gate2"] = round(time.time() - t1, 1)
    print("gate 2", json.dumps(g2), flush=True)
    if not g2["passed"]:
        (dst / "run_scores.json").write_text(json.dumps(out))
        G.acts.commit()
        return out

    # gate 3: unit checks (RULE.md), after gate 2 and before any new-variant fit
    grams = set(A.char_ngrams("A  b\n\n\nc ", 1, 5))
    unit = {"char_keeps_space_run": "  " in grams, "char_keeps_newline_run": "\n\n\n" in grams, "char_keeps_trailing": "c " in grams,
            "char_keeps_case": "A" in grams and "a" not in grams, "w_statement": ws("x  \n\n\n\ny  ") == "x\n\ny"}
    out["unit_checks"] = unit | {"passed": all(unit.values())}
    print("unit checks", json.dumps(out["unit_checks"]), flush=True)
    if not out["unit_checks"]["passed"]:
        (dst / "run_scores.json").write_text(json.dumps(out))
        G.acts.commit()
        return out

    # 4. variants (RULE.md): character-only 1-5 and 2-5 on the halves' text; character-only 1-5 on arm W's text
    texts_w = [ws(t) for t in texts]
    out["counts"]["arm_W"] = {"changed_outputs": dict(Counter(str(arm[i]) for i, (t, u) in enumerate(zip(texts, texts_w)) if t != u)),
                              "plant_pairs_differing_level1": sum(any(texts[i] != texts[j] for i, j in pairs[c]) for c in planted),
                              "plant_pairs_differing_after_W": sum(any(texts_w[i] != texts_w[j] for i, j in pairs[c]) for c in planted)}
    print("arm W", json.dumps(out["counts"]["arm_W"]), flush=True)
    for name, (lo, hi), level in VARIANTS:
        cv = dict(cfg, word_ngrams=None, char_ngrams=[lo, hi])
        t1 = time.time()
        X = A.featurize(texts_w if level == "W" else texts, cv)
        out["timing"][f"featurize|{name}"] = round(time.time() - t1, 1)
        print("featurized", name, X.shape, flush=True)
        records += fits(X, name, cv)
        del X
        (dst / f"partial_{name}.json").write_text(json.dumps(out | {"records": records}))  # checkpoint (Codex, 10-13)
        G.acts.commit()
    out["records"] = records
    out["timing"]["total"] = round(time.time() - t0, 1)
    (dst / "run_scores.json").write_text(json.dumps(out))
    G.acts.commit()
    return out


def save(res: dict) -> Path:
    (HERE / "data").mkdir(exist_ok=True)
    p = HERE / "data" / f"chartfidf_halves_run_scores_{res['provenance']['stamp']}.json.gz"
    with open(p, "xb") as fh:  # keys and scores, no text; provenance is the launch's, recorded inside the job
        fh.write(gzip.compress(json.dumps(res).encode(), mtime=0))
    print(json.dumps({k: res.get(k) for k in ("gate", "unit_checks", "gate2", "counts", "timing")}, indent=1))
    print("->", p)
    return p


def main(mode: str, arg: str = "", ref_path: str = ""):
    from zoneinfo import ZoneInfo

    if mode == "collect":  # arg = the launch's stamp
        vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations")
        return save(json.loads(b"".join(vol.read_file(f"{OUT}/{arg}/run_scores.json"))))
    archived_path = arg
    raw_ref = Path(ref_path).read_bytes()
    if hashlib.sha256(raw_ref).hexdigest() != REF_SHA:
        raise SystemExit("the reference run scores do not match the sha256 faafd24c's result records for them")
    fields = ("direction", "comparison", "key", "arm", "cell", "environment", "main_task", "score", "rows")
    reference = [{f: r[f] for f in fields} for r in json.loads(gzip.decompress(raw_ref))["records"]]
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    if subprocess.run(["git", "status", "--porcelain", str(Path(__file__))], capture_output=True, text=True, check=True, cwd=HERE).stdout:
        raise SystemExit(f"commit {Path(__file__).name} first: the result records the commit that ran")
    arch = json.loads(Path(archived_path).read_bytes())
    want = json.loads((FIXED / "results_manifest_v1.json").read_text())["files"]["experiments/2026-10-09_fixed_sim/results/gate_a_main_2026-10-09_185451.json"]["sha256"]
    if sha(Path(archived_path)) != want:
        raise SystemExit("the archived gate (a) result does not match its manifest sha256")
    cfg, split = arch["config"], arch["split"]
    folds = json.loads((HERE.parents[1] / "splits" / cfg["folds_file"]).read_text())
    archived = {c: arch["rows"][f"{c}|dev_to_test|level1"]["test_envs"]["auroc"] for c in GATE_ROWS}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    expect = {"script_sha256": SCRIPT_SHA, "code_sha256": {k: sha(v) for k, v in code_paths().items()}}
    prov = {"stamp": stamp, "launch_git": git, "archive_sha256": want, "reference_sha256": REF_SHA,
            "reference_decompressed_sha256": hashlib.sha256(gzip.decompress(raw_ref)).hexdigest(), "folds_file": cfg["folds_file"],
            "folds_sha256": sha(HERE.parents[1] / "splits" / cfg["folds_file"]), "config": cfg, "split": split}
    call = modal.Function.from_name(APP, "job").spawn(cfg, split, folds, archived, expect, prov, reference)
    print("function call", call.object_id, flush=True)
    while True:
        try:
            return save(call.get(timeout=60))
        except TimeoutError:  # only the polling timeout; Modal's own timeout and expiry errors propagate
            print(time.strftime("%H:%M:%S"), "running", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
