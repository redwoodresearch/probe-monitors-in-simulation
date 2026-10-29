"""Follow-up (c), the probe's packs (Modal CPU, pennies): per level and layer, every kept output_end row of the Qwen3.5-27B
dumps (run_extract_qwen35.py with config/onpolicy_september_l{0,1}.yaml; Volume `…qwen35_acts…`) in q1_eval.step_table
order, written as `claimB/<level>/packedB_<digest>/L<l>_output_end.npy` and `runs.L<l>.json` on the la-activations Volume,
the layout d9's run_claimB_b_grid.py (aswin/q3-claimb-probe @ 857cb3cc) reads, so it runs unchanged with
`--readouts output_end`. Arms are renamed to that grid's comparisons (ARM_MAP; each run keeps its own name in `arm_own`):
fixed_vs_e1 = fixed sim vs real run 1, e2_vs_e1 = real run 2 vs run 1, plant1 / plant2 = the planted checks. Folds: the
claim B splits (folds_v2: environments outer, main tasks inner; all 190 main tasks are in it). Gates, before anything is
written: every staged transcript extracted (per arm, against the staging manifest), every run with at least one kept
output_end, no non-finite value. The digest hashes the dump, the readout, this script and the full run list (folds and
positions included), so a changed pack never reuses an old pack's name or the grid's caches.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy modal run \
      experiments/2026-10-10_claimB_onpolicy/run_onpolicy_pack.py [--levels level0,level1]
"""

import hashlib
import json
from pathlib import Path

import modal

HERE = Path(__file__).parent
ROOT = HERE.parents[1] if modal.is_local() else HERE  # the repo root locally; unused in the container
APP = "probe-shift_aswin-surya_sub-claimb_onpol-pack_kill-2026-11-10"
LAYERS, POS = [19, 32, 41, 44, 47, 52, 58, 63], "output_end"
DUMPS = {"level0": ("2026-10-10_la_september-onpolicy-l0_qwen35-27b_alllayers_v1", "2026-10-10_la_september-onpolicy-l0_v1"),
         "level1": ("2026-10-10_la_september-onpolicy-l1_qwen35-27b_alllayers_v1", "2026-10-10_la_september-onpolicy-l1_v1")}
ARM_MAP = {"real-attack-r1": "real-attack-e1", "real-attack-r2": "real-attack-e2", "sim-attack-sept-fixed": "sim-attack-fixed",
           "sim-attack-sept-fixed-plant1": "sim-attack-fixed-plant1", "sim-attack-sept-fixed-plant2": "sim-attack-fixed-plant2"}

app = modal.App(APP)
acts = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
q35 = modal.Volume.from_name("probe-shift_aswin-surya_sub-qwen35_acts_kill-2026-11-06", version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy").add_local_python_source("probe_shift")


@app.function(image=image, cpu=8, memory=32768, volumes={"/acts": acts, "/q35": q35.read_only()}, timeout=1800, retries=0)
def pack(level: str, layer: int, dump: str, staged: str, task_fold: dict, code: str) -> dict:
    import numpy as np

    from probe_shift.q1_eval import step_table

    want = json.loads((Path("/acts") / staged / "manifest.json").read_text())["arms"]
    src = Path("/q35") / dump
    runs = []
    for own, arm in ARM_MAP.items():
        for f in sorted((src / "meta" / own).glob("*.json")) if (src / "meta" / own).exists() else []:
            m = json.loads(f.read_text())
            runs.append(m | {"arm": arm, "arm_own": own, "fold": task_fold[m["main_task"]]})
    got = {a: sum(m["arm_own"] == a for m in runs) for a in want}
    if got != want:
        raise ValueError(f"{level}: extracted runs per arm {got} differ from the staged {want}")
    empty = [m["key"] for m in runs if not any(p["kind"] == POS and p["kept"] for p in m["positions"])]
    if empty:
        raise ValueError(f"{level}: {len(empty)} runs have no kept {POS}, e.g. {empty[:3]}")
    runs.sort(key=lambda m: (m["arm"], m["key"]))
    chunks, parts = {}, []
    for m in runs:  # whole chunk files, read once each (row-indexed memory maps read page by page from a Volume)
        if m["chunk"] not in chunks:
            chunks[m["chunk"]] = np.load(src / "by_layer" / f"L{layer}" / f"{m['chunk']}.npy")
        rows = [p["row"] for p in m["positions"] if p["kind"] == POS and p["kept"]]
        parts.append(chunks[m["chunk"]][m["row_start"] + np.array(rows, int)].astype(np.float16))
    slim = [{k: m[k] for k in ("arm", "arm_own", "key", "cell", "main_task", "environment", "fold")}
            | {"positions": [p for p in m["positions"] if p["kind"] == POS]} for m in runs]
    t = step_table(slim, POS)
    X = np.concatenate(parts)
    if len(X) != len(t["run"]) or not np.isfinite(X).all():
        raise ValueError(f"{level} L{layer}: {len(X)} rows vs step table {len(t['run'])}, or non-finite values")
    for i, m in enumerate(slim):  # rows renumbered in step_table order (the grid reads row by position in the table)
        k = iter(range(int((t["run"] < i).sum()), int((t["run"] <= i).sum())))
        m["positions"] = [p | {"row": next(k)} if p["kept"] else p for p in m["positions"]]
    digest = hashlib.sha1(json.dumps({"dump": dump, "readout": POS, "code": code, "runs": slim}, sort_keys=True).encode()).hexdigest()[:12]  # any change: a new pack
    out = Path("/acts") / "claimB" / level / f"packedB_{digest}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f".L{layer}_{POS}.tmp", "wb") as fh:
        np.save(fh, X)
    (out / f".L{layer}_{POS}.tmp").replace(out / f"L{layer}_{POS}.npy")
    (out / f".runs.L{layer}.tmp").write_text(json.dumps(slim))
    (out / f".runs.L{layer}.tmp").replace(out / f"runs.L{layer}.json")
    acts.commit()
    return {"level": level, "layer": layer, "pack": digest, "rows": len(X), "runs_by_arm": got}


@app.local_entrypoint()
def main(levels: str = "level0,level1"):
    folds = json.loads((ROOT / "splits" / "folds_v2.json").read_text())["task_folds"]["main_task_to_fold"]
    code = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    calls = [pack.spawn(lv, l, *DUMPS[lv], folds, code) for lv in levels.split(",") for l in LAYERS]
    res = [c.get() for c in calls]
    for r in res:
        print(json.dumps(r), flush=True)
    print("packs:", sorted({(r["level"], r["pack"]) for r in res}))
