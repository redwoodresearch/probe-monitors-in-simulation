"""Max-over-tokens version (a) (PLAN.md amendment 7), part 1: per-layer packs of a pooled readout (Modal CPU, pennies).
d9's run_claimB_b_pack.py (aswin/q3-claimb-probe @ 857cb3cc) copied with one change: the pooled kinds come from --pooled
(default output_end_max, from the max dump) instead of B's three; A's output_end is packed beside them as in B, though
amendment 7's grid selects over output_end_max alone (beside B's existing rows). The digest hashes the readouts, so a max
pack never collides with B's.

Per level and layer: A's pack (`claimB/<level>/packed_<A digest>/`, the runs and output_end rows probe step A used) fixes
the runs, in A's order; B's dump (one run per level, every arm; `meta/<arm>/`, `by_layer/L<l>/<chunk>.npy`, pooled rows
only) gives the pooled rows. Each run keeps the outputs present in both: A's kept output_end positions and all three
pooled kinds, keyed by (step, action), so every readout is scored on the same outputs. Gates, before anything is
written: every A run is in B's dump; at most 1% of A's outputs are missing from B; B's pooled span ends at A's
output_end token (the same text and tokens); a mismatched output is dropped from every readout, at most 0.1%. Writes
`claimB/<level>/packedB_<digest>/L<l>_<readout>.npy` (float16, rows in q1_eval.step_table order) and `runs.L<l>.json`.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' modal run \
      experiments/2026-10-09_fixed_sim/run_claimB_max_pack.py [--pooled output_end_max]
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

import modal

APP = "probe-shift_aswin-surya_sub-claimb_maxpack_kill-2026-11-10"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
LAYERS = [19, 32, 41, 44, 47, 52, 58, 63]
LEVELS = {"level1": {"a_pack": "39f842231a46", "b_run": "2026-10-10_la_fixed-sim-v2-max-scrub1_qwen3-32b_l8_v1"}}
MISSING_MAX, TOKEN_MISMATCH_MAX = 0.01, 0.001

app = modal.App(APP)
vol = modal.Volume.from_name(ACTS, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy").add_local_python_source("probe_shift")


@app.function(image=image, cpu=8, memory=65536, volumes={"/acts": vol}, timeout=3600, retries=0)
def pack(level: str, layer: int, a_pack: str, b_run: str, POOLED: list[str]) -> dict:
    import numpy as np

    from probe_shift.q1_eval import step_table

    a_dir = Path("/acts") / "claimB" / level / f"packed_{a_pack}"
    a_runs = json.loads((a_dir / f"runs.L{layer}.json").read_text())
    a_t = step_table(a_runs, "output_end")
    a_X = np.load(a_dir / f"L{layer}_output_end.npy")
    if len(a_X) != len(a_t["run"]):
        raise ValueError("A's packed rows do not match its step table")
    a_rows, a_pos = {}, {}
    for k, (i, s, ac) in enumerate(zip(a_t["run"], a_t["step"], a_t["action"])):
        a_rows.setdefault(int(i), {})[(int(s), int(ac))] = k
    for i, m in enumerate(a_runs):
        a_pos[i] = {(p["step"], p["action"]): p for p in m["positions"] if p["kind"] == "output_end" and p["kept"]}

    src = Path("/acts") / b_run
    b_meta = {}
    for arm in sorted({m["arm"] for m in a_runs}):
        for f in sorted((src / "meta" / arm).glob("*.json")):
            m = json.loads(f.read_text())
            b_meta[(arm, m["key"])] = m
    missing = [(m["arm"], m["key"]) for m in a_runs if (m["arm"], m["key"]) not in b_meta]
    if missing:
        raise ValueError(f"{len(missing)} of A's {len(a_runs)} runs are not in B's dump, e.g. {missing[:3]}")

    n, chunk, chunk_name, parts = Counter(), None, None, {}
    for i in sorted(range(len(a_runs)), key=lambda i: b_meta[(a_runs[i]["arm"], a_runs[i]["key"])]["chunk"]):
        m = b_meta[(a_runs[i]["arm"], a_runs[i]["key"])]
        pooled = {kd: {(p["step"], p["action"]): p for p in m["positions"] if p["kind"] == kd and p["kept"]} for kd in POOLED}
        a_o, b_o = set(a_rows.get(i, {})), set.intersection(*(set(v) for v in pooled.values()))
        both = a_o & b_o
        bad_tok = {o for o in both if pooled[POOLED[0]][o]["token"] != a_pos[i][o]["token"]}
        common = sorted(both - bad_tok)  # an output whose tokens disagree is dropped from every readout
        n["A outputs"] += len(a_o)
        n["token mismatches (dropped)"] += len(bad_tok)
        n["A outputs missing from B"] += len(a_o - b_o)
        n["B outputs not in A (dropped)"] += len(b_o - a_o)
        n["runs without a common output"] += not common
        if m["chunk"] != chunk_name:  # whole-file reads: row-indexed memory maps fetch page by page from the Volume
            chunk, chunk_name = np.load(src / "by_layer" / f"L{layer}" / f"{m['chunk']}.npy"), m["chunk"]
        parts[i] = {kd: chunk[m["row_start"] + np.array([pooled[kd][o]["row"] for o in common], int)].astype(np.float16) for kd in POOLED}
        parts[i]["output_end"] = a_X[[a_rows[i][o] for o in common]].astype(np.float16)
        parts[i]["positions"] = [{"kind": "output_end", "step": o[0], "action": o[1], "tool": a_pos[i][o]["tool"], "token": a_pos[i][o]["token"],
                                  "row": j, "kept": True, "n_tokens": int(pooled[POOLED[0]][o]["span"][1] - pooled[POOLED[0]][o]["span"][0])}
                                 for j, o in enumerate(common)]  # n_tokens: the pooled span's length, for the length baseline
    if n["runs without a common output"] or n["A outputs missing from B"] > MISSING_MAX * n["A outputs"] \
            or n["token mismatches (dropped)"] > TOKEN_MISMATCH_MAX * n["A outputs"]:
        raise ValueError(f"alignment gate failed: {dict(n)}")

    READOUTS = POOLED + ["output_end"]
    runs = [{k: m[k] for k in ("arm", "key", "cell", "main_task", "environment", "fold")} | {"positions": parts[i]["positions"]}
            for i, m in enumerate(a_runs)]
    t = step_table(runs, "output_end")
    digest = hashlib.sha1(json.dumps({"a_pack": a_pack, "b_run": b_run, "readouts": READOUTS,
                                      "runs": [(m["arm"], m["key"]) for m in runs]}, sort_keys=True).encode()).hexdigest()[:12]
    out = Path("/acts") / "claimB" / level / f"packedB_{digest}"
    out.mkdir(parents=True, exist_ok=True)
    for kd in READOUTS:
        X = np.concatenate([parts[i][kd] for i in range(len(runs))])
        if len(X) != len(t["run"]) or not np.isfinite(X).all():
            raise ValueError(f"{kd}: {len(X)} rows vs step table {len(t['run'])}, or non-finite values")
        with open(out / f".L{layer}_{kd}.tmp", "wb") as fh:
            np.save(fh, X)
        (out / f".L{layer}_{kd}.tmp").replace(out / f"L{layer}_{kd}.npy")
    (out / f".runs.L{layer}.tmp").write_text(json.dumps(runs))
    (out / f".runs.L{layer}.tmp").replace(out / f"runs.L{layer}.json")
    vol.commit()
    return {"level": level, "layer": layer, "packed": str(out), "rows": len(t["run"]), "runs": len(runs), "counts": dict(n),
            "runs_by_arm": dict(Counter(m["arm"] for m in runs))}


@app.local_entrypoint()
def main(levels: str = "level1", layers: str = "", pooled: str = "output_end_max"):
    ls = [int(x) for x in layers.split(",")] if layers else LAYERS
    calls = [pack.spawn(lv, l, LEVELS[lv]["a_pack"], LEVELS[lv]["b_run"], pooled.split(",")) for lv in levels.split(",") for l in ls]
    for c in calls:
        print(json.dumps(c.get()), flush=True)
