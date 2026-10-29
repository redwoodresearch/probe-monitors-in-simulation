"""POST HOC (RULE.md in this folder; Aswin about 10:48 PT 10-12, via 0d): gate (a)'s raw-text TF-IDF through the halves
design, so the halves figure's 1-in-100 alarm has a text-classifier comparison. One Modal CPU container:
  1. gate: gate (a)'s variant level1, design dev_to_test, both comparisons, with run_gate_a_modal.py's prepare,
     design_jobs, score_jobs and run_auroc unchanged and the archived run's resolved config and split; the
     test-environment run AUROCs must equal the archived rows to 1e-9, or nothing else runs;
  2. the halves' text: gate (a)'s examples (939 cells x 3 arms), B-dev's evening in both arms on raw text
     (run_tfidf_slips_evened_modal.even; "Invalid `view_range`" outputs dropped), then the level-1 scrub; plant 2 copies
     (undo_r2 on every accepted text_editor str_replace output of the evened fixed-sim run) for step A's 150 test plant-2
     cells and B-dev's 100 dev planted cells;
  3. per direction (dev -> test, test -> dev): fixed sim vs e1 (scores held-out sim, e1 and e2: the direct yardstick),
     e2 vs e1 (the separate yardstick), plant 2 vs its fixed-sim twins (pairs).
Returns and writes to the Volume run scores and counts only (keys, arms, numbers; no text). analyze_tfidf_halves.py
applies the rule.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with pyyaml modal deploy \\
      experiments/2026-10-12_claimB_tfidf_halves/run_tfidf_halves_modal.py
  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with pyyaml python \\
      experiments/2026-10-12_claimB_tfidf_halves/run_tfidf_halves_modal.py launch <gate_a_main_2026-10-09_185451.json>
  (if the client is lost: `... run_tfidf_halves_modal.py collect <stamp>` reads that invocation's Volume copy)
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

APP = "probe-shift_aswin-surya_sub-claimb-tfidf-halves_kill-2026-11-12"
SCRIPT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
OUT = f"claimB/tfidf_halves_v1/{SCRIPT_SHA[:12]}"
PACK_RUNS = "claimB/level1/packed_39f842231a46/runs.L63.json"  # step A's pack: its plant-2 arm names the 150 test cells
DEV_CELLS = "claimB/slips_evened_dev_v1/cells.json"  # B-dev: its 100 dev planted cells
SIM, E1, E2, PL2 = "sim-attack", "real-attack-e1", "real-attack-e2", "sim-attack-plant2"
GATE_ROWS = ("fixed_sim_vs_e1", "e2_vs_e1")

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


@app.function(image=image, volumes={"/acts": G.acts}, cpu=16, memory=65536, timeout=1800, retries=0)
def job(cfg: dict, split: dict, folds: dict, archived: dict, expect: dict, prov: dict) -> dict:
    import numpy as np

    import analyze_tfidf as A
    from undo_r2 import undo_r2

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

    def add(r, arm, key, text):
        for k, v in (("arm", arm), ("key", key), ("cell", r["cell"]), ("env", r["environment"]), ("task", r["main_task"])):
            cols[k].append(v)
        texts.append(A.scrub(text, 1))

    for r, cmd in zip(rows, d["editor_command"]):
        raw = r[cfg["text"]]
        if S.RANGE in raw:
            changed[f"{r['arm']}|range_dropped"] += 1
            continue
        t, hit = S.even(raw, cmd, r["tool"])
        changed.update(f"{r['arm']}|{h}" for h in hit)
        add(r, r["arm"], r["key"], t)
        if r["arm"] == SIM and r["cell"] in planted:
            u = undo_r2(t) if r["tool"] == "text_editor" and cmd == "str_replace" else None
            sites[r["cell"]] += bool(u)
            add(r, PL2, r["key"] + "__plant2", u or t)
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
    t1 = time.time()
    X = A.featurize(texts, cfg)
    out["timing"]["featurize"] = round(time.time() - t1, 1)
    print("featurized", X.shape, json.dumps(out["counts"]["runs"]), flush=True)

    # 3. fits per direction
    dirs = {"dev_to_test": (split["dev"], split["test"]), "test_to_dev": (split["test"], split["dev"])}
    comps = {"fixed_vs_e1": (E1, SIM, [SIM, E1, E2]), "e2_vs_e1": (E1, E2, [E1, E2]), "plant2": (SIM, PL2, [SIM, PL2])}
    in_plant = np.isin(cell, sorted(planted))
    records = []
    for comp, (neg, pos, scored) in comps.items():
        t1 = time.time()
        y = (arm == pos).astype(int)
        base = np.isin(arm, [neg, pos]) & (in_plant if comp == "plant2" else True)
        jobs = [(base & np.isin(env, tr), np.isin(arm, scored) & np.isin(env, te) & (in_plant if comp == "plant2" else True))
                for tr, te in dirs.values()]
        s = G.score_jobs(X, y, jobs, cfg)
        if not all(np.isfinite(s[te]).all() for _, te in jobs):
            raise SystemExit(f"{comp}: a scored row has no finite score")
        for dname, (_, te) in zip(dirs, jobs):
            run_score, first = A.run_means(s, key, te)
            n = Counter(key[te].tolist())
            records += [{"direction": dname, "comparison": comp, "key": str(key[i]), "arm": str(arm[i]), "cell": str(cell[i]),
                         "environment": str(env[i]), "main_task": str(task[i]), "score": float(v), "rows": n[str(key[i])]}
                        for i, v in zip(first, run_score)]
        out["timing"][comp] = round(time.time() - t1, 1)
        print(comp, "fitted in", out["timing"][comp], "s", flush=True)
    out["records"] = records
    out["timing"]["total"] = round(time.time() - t0, 1)
    (dst / "run_scores.json").write_text(json.dumps(out))
    G.acts.commit()
    return out


def save(res: dict) -> Path:
    (HERE / "data").mkdir(exist_ok=True)
    p = HERE / "data" / f"tfidf_halves_run_scores_{res['provenance']['stamp']}.json.gz"
    with open(p, "xb") as fh:  # keys and scores, no text; provenance is the launch's, recorded inside the job
        fh.write(gzip.compress(json.dumps(res).encode(), mtime=0))
    print(json.dumps({k: res.get(k) for k in ("gate", "counts", "timing")}, indent=1))
    print("->", p)
    return p


def main(mode: str, arg: str = ""):
    from zoneinfo import ZoneInfo

    if mode == "collect":  # arg = the launch's stamp
        vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations")
        return save(json.loads(b"".join(vol.read_file(f"{OUT}/{arg}/run_scores.json"))))
    archived_path = arg
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
    prov = {"stamp": stamp, "launch_git": git, "archive_sha256": want, "folds_file": cfg["folds_file"],
            "folds_sha256": sha(HERE.parents[1] / "splits" / cfg["folds_file"]), "config": cfg, "split": split}
    call = modal.Function.from_name(APP, "job").spawn(cfg, split, folds, archived, expect, prov)
    print("function call", call.object_id, flush=True)
    while True:
        try:
            return save(call.get(timeout=60))
        except TimeoutError:  # only the polling timeout; Modal's own timeout and expiry errors propagate
            print(time.strftime("%H:%M:%S"), "running", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
