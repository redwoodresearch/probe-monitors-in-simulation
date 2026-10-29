"""Claim B's probe step, part 1: merged per-layer feature packs (Modal CPU, pennies). Fixed-sim runs from the new
extraction joined to Q1's real epoch-1 and epoch-2 rows, so the probe sees exactly the real rows Q1 used.

Per level and layer: Q1's packed output_end matrix (`<q1 run>__q1gpu/packed/d0c46e0e5516/`, its manifest checked by
digest) gives the real-attack-e1 and real-attack-e2 rows; the new all-layers extraction gives every kept output_end row
of the new arms (`by_layer/L<l>/<chunk>.npy`, `meta/<arm>/`), as `analyze_q1_gpu.py::pack_layer` reads them. Rows go in
`q1_eval.step_table` order over the merged run list. Gate, before anything is written: every `real-attack-e1-repro` run
of the new extraction must match Q1's row for row (same positions, cosine >= 0.9999 at every layer packed); repro runs
are checked, then left out. Writes `claimB/<level>/packed_<digest>/L<l>_output_end.npy` (float16) and `runs.L<l>.json` on
the activations Volume.

`--dry`: Q1's own extraction stands in for the new one (sim-attack read as sim-attack-fixed, 20 seeded real e1 runs as
the repro arm), to test the path before the new dumps land.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with pyyaml modal run \
      experiments/2026-10-10_claimB_probe/run_claimB_pack.py::main [--dry]
"""

import hashlib
import json
from pathlib import Path

import modal

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
APP = "probe-shift_aswin-surya_sub-claimb-probe_pack_kill-2026-11-10"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
Q1 = ROOT / "experiments/2026-10-06_q1_segment_end"
Q1_PACKED = "d0c46e0e5516"
POS = "output_end"
LAYERS = [19, 32, 41, 44, 47, 52, 58, 63]
REAL = ["real-attack-e1", "real-attack-e2"]
LEVELS = {  # Q1 config (its extraction is the real rows' source) and the new extraction, per level
    "level0": {"q1_config": "q1_alllayers.yaml", "new_run": "2026-10-10_la_fixed-sim-v2_qwen3-32b_alllayers_v1",
               "new_arms": ["sim-attack-fixed"]},
    "level1": {"q1_config": "q1_alllayers_scrub1.yaml", "new_run": "2026-10-10_la_fixed-sim-v2-scrub1_qwen3-32b_alllayers_v1",
               "new_arms": ["sim-attack-fixed", "sim-attack-fixed-plant1", "sim-attack-fixed-plant2"]}}
REPRO, COS_MIN, DRY_REPRO = "real-attack-e1-repro", 0.9999, 20

app = modal.App(APP)
vol = modal.Volume.from_name(ACTS, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy").add_local_python_source("probe_shift")


def q1_cfg(level: str) -> dict:
    import yaml

    cfg = yaml.safe_load((Q1 / "config" / LEVELS[level]["q1_config"]).read_text())
    folds = json.loads((ROOT / "splits" / cfg["folds_file"]).read_text())
    ex = json.loads((ROOT / "splits" / cfg["exclusions_file"]).read_text())
    return {"extraction_run": cfg["extraction_run"], "meta_dir": cfg["meta_dir"], "paired_cells": ex["paired_attack_cells"]["cells"],
            "main_task_to_fold": folds["task_folds"]["main_task_to_fold"], "environment_to_fold": folds["environment_folds"]["environment_to_fold"]}


@app.function(image=image, cpu=16, memory=65536, volumes={"/acts": vol}, timeout=3600, retries=0)
def pack(cfg: dict, level: str, layer: int, new_run: str, new_arms: list[str], dry: bool) -> dict:
    import numpy as np

    from probe_shift.q1_eval import load_runs, step_table

    q1_dir = Path("/acts") / f"{cfg['extraction_run']}__q1gpu" / "packed" / Q1_PACKED
    man = json.loads((q1_dir / f"manifest.L{layer}.json").read_text())
    if hashlib.sha1(json.dumps(man, sort_keys=True).encode()).hexdigest()[:12] != Q1_PACKED:
        raise ValueError("Q1 packed manifest digest mismatch")
    q1_runs = load_runs(cfg, man)
    q1_t = step_table(q1_runs, POS)
    q1_X = np.load(q1_dir / f"L{layer}_{POS}.npy", mmap_mode="r")
    if len(q1_X) != len(q1_t["run"]):
        raise ValueError("Q1 packed rows do not match its step table")
    real_ix = [i for i, m in enumerate(q1_runs) if m["arm"] in REAL]
    e1_by_key = {q1_runs[i]["key"]: i for i in real_ix if q1_runs[i]["arm"] == "real-attack-e1"}

    def q1_rows(i):
        return np.asarray(q1_X[np.flatnonzero(q1_t["run"] == i)], np.float32)

    # The new extraction's runs (dry: Q1's own sim-attack as sim-attack-fixed, seeded e1 runs as the repro arm)
    src = Path("/acts") / new_run
    arm_map = {"sim-attack": "sim-attack-fixed"} if dry else {a: a for a in new_arms}
    if dry:
        keys = sorted(e1_by_key)
        repro_keys = set(np.random.default_rng(20261010).choice(keys, DRY_REPRO, replace=False).tolist())
    new_runs, repro = [], []
    for src_arm, arm in list(arm_map.items()) + [("real-attack-e1" if dry else REPRO, REPRO)]:
        for f in sorted((src / cfg["meta_dir"] / src_arm).glob("*.json")):
            m = json.loads(f.read_text())
            if arm == REPRO and dry and m["key"] not in repro_keys:
                continue
            if m["cell"] not in set(cfg["paired_cells"]):
                continue
            m = m | {"arm": arm, "fold": cfg["main_task_to_fold"][m["main_task"]]}
            (repro if arm == REPRO else new_runs).append(m)
    chunks = {}

    def new_rows(m):
        if m["chunk"] not in chunks:
            chunks[m["chunk"]] = np.load(src / "by_layer" / f"L{layer}" / f"{m['chunk']}.npy", mmap_mode="r")
        rows = [p["row"] for p in m["positions"] if p["kind"] == POS and p["kept"]]
        return np.asarray(chunks[m["chunk"]][m["row_start"] + np.array(rows, int)], np.float32)

    # Gate: the repro runs reproduce Q1's real e1 rows
    cos_min, n_repro = 1.0, 0
    ident = lambda m: [(p["step"], p["action"], p["token"]) for p in m["positions"] if p["kind"] == POS and p["kept"]]
    for m in repro:
        q1m = q1_runs[e1_by_key[m["key"]]]
        if ident(m) != ident(q1m):  # the same readout positions (step, action, token index), in order
            raise ValueError(f"repro {m['key']}: readout positions differ from Q1's")
        a, b = new_rows(m), q1_rows(e1_by_key[m["key"]])
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        if a.shape != b.shape or not np.isfinite(cos).all():
            raise ValueError(f"repro {m['key']}: {a.shape} rows vs Q1's {b.shape}, or non-finite cosines")
        cos_min, n_repro = min(cos_min, float(cos.min())), n_repro + 1
    if n_repro == 0 or cos_min < COS_MIN:
        raise ValueError(f"reproduction gate failed: {n_repro} repro runs, min cosine {cos_min:.6f} (need >= {COS_MIN})")

    runs = [q1_runs[i] for i in real_ix] + new_runs
    X = np.concatenate([np.asarray(q1_X[np.isin(q1_t["run"], real_ix)], np.float16)] + [new_rows(m).astype(np.float16) for m in new_runs])
    t = step_table(runs, POS)
    if len(X) != len(t["run"]) or not np.isfinite(X).all():
        raise ValueError(f"merged rows {len(X)} vs step table {len(t['run'])}, or non-finite values")
    slim = [{k: m[k] for k in ("arm", "key", "cell", "main_task", "environment", "fold", "positions")} for m in runs]
    digest = hashlib.sha1(json.dumps({"q1": Q1_PACKED, "new_run": new_run, "dry": dry,
                                      "runs": [(m["arm"], m["key"]) for m in slim]}, sort_keys=True).encode()).hexdigest()[:12]
    out = Path("/acts") / "claimB" / level / f"packed_{digest}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f".L{layer}.tmp", "wb") as fh:
        np.save(fh, X)
    (out / f".L{layer}.tmp").replace(out / f"L{layer}_{POS}.npy")
    (out / f".runs.L{layer}.tmp").write_text(json.dumps(slim))  # one file per layer worker: no shared temp file
    (out / f".runs.L{layer}.tmp").replace(out / f"runs.L{layer}.json")
    vol.commit()
    arms = {a: sum(m["arm"] == a for m in runs) for a in sorted({m["arm"] for m in runs})}
    return {"level": level, "layer": layer, "packed": str(out), "rows": len(X), "runs_by_arm": arms,
            "repro": {"runs": n_repro, "min_cosine": round(cos_min, 6)}}


@app.local_entrypoint()
def main(dry: bool = False, levels: str = "level0,level1"):
    calls = []
    for level in levels.split(","):
        cfg = q1_cfg(level)
        new_run = cfg["extraction_run"] if dry else LEVELS[level]["new_run"]
        calls += [pack.spawn(cfg, level, l, new_run, LEVELS[level]["new_arms"], dry) for l in LAYERS]
    for c in calls:
        print(json.dumps(c.get()), flush=True)
