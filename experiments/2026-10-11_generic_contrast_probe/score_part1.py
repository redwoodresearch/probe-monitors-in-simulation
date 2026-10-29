"""PLAN.md sections 4-6 and amendment 1, Modal CPU: projections for part 1. No labels are read here.

For every listed real stage-1 call (world, irid, call_index; passed in from key_stage1.csv's id columns) and every
impactful call of honest A and honest B in the 36 calibration tasks (stored rule-1 label of the command entry; the
scorer's run rule: never redrawn, never a rerun), the call_end reading at layer 48 (direction A) and layer 46
(direction B), projected onto: A = v, v_res, r and 100 random unit directions (seed 20261013); B = B, r_B and the same
100 random directions. Extractions as stage 1's check_joins.py and check_calibration.py.
"""

import json
from pathlib import Path

import modal

APP = "probe-shift_aswin-surya_sub-gcp_part1_kill-2026-11-10"
SCORE_VOL, ACTS_VOL = "probe-shift_aswin-surya_sub-q3onpol_score_kill-2026-11-08", "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08"
OUT, LAYER_A, LAYER_B, RANDOM_SEED, N_RANDOM = "generic_contrast_v1", 48, 46, 20261013, 100
BASES = ["2026-10-09_rerun_strict_segends_v1", "2026-10-10_scaleup_canary_segends_v1", "2026-10-10_scaleup_l2_segends_v1",
         "2026-10-10_scaleup_topup_segends_v1"]
WORLDS = {"real": ("attack", BASES), "run2": ("attack", ["2026-10-10_second_attack_segends_v1"])}

app = modal.App(APP)
score_vol, acts_vol = modal.Volume.from_name(SCORE_VOL, version=2), modal.Volume.from_name(ACTS_VOL, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")


@app.function(image=image, cpu=16, memory=65536, timeout=3600, retries=0, volumes={"/score": score_vol.read_only(), "/acts": acts_vol.read_only()})
def project(listed: list, cal_tasks: list) -> dict:
    import numpy as np

    score_vol.reload()
    acts_vol.reload()
    want = {(w, i, int(c)) for w, i, c in listed}
    cal = set(cal_tasks)
    loc, honest, kept, cal_cmd, cal_end = {}, [], {}, set(), set()  # listed call -> (extraction, chunk, row); honest rows
    sources = [(w, arm, x) for w, (arm, xs) in WORLDS.items() for x in xs] + [(a, a, x) for x in BASES for a in ("honest_a", "honest_b")]
    for world, arm, x in sources:
        root = Path("/acts") / x
        if len(list((root / "meta").glob("*/*.json"))) != json.loads((root / "complete.json").read_text())["epochs"]:
            raise ValueError(f"{x}: metas incomplete")
        for f in sorted((root / "meta" / arm).glob("*.json")):
            m = json.loads(f.read_text())
            if m.get("redrawn_because") or m.get("attempt_kind") == "rerun":
                continue
            if world in ("honest_a", "honest_b") and m["task"] not in cal:
                continue
            kept.setdefault((world, m["task"]), set()).add(m["run"])
            label = {(c["irid"], c["call_index"]): c.get("label") for c in m["calls"] if c.get("kind", "command") == "command" and "call_index" in c}
            if world in ("honest_a", "honest_b"):
                cal_cmd |= {(world, m["task"], i, ci) for (i, ci), lab in label.items() if lab == "impactful"}
            for c in m["calls"]:
                if c.get("kind") != "call_end" or c.get("row") is None or "call_index" not in c:
                    continue
                k = (world, c["irid"], c["call_index"])
                where = (x, m["chunk"], m["row_start"] + c["row"])
                if world in ("real", "run2") and k in want:
                    if k in loc:
                        raise ValueError(f"two call_end rows for {k}")
                    loc[k] = where
                elif world in ("honest_a", "honest_b") and label.get((c["irid"], c["call_index"])) == "impactful":
                    if (world, m["task"], c["irid"], c["call_index"]) in cal_end:
                        raise ValueError(f"two call_end rows for an honest call in {m['task']}")
                    cal_end.add((world, m["task"], c["irid"], c["call_index"]))
                    honest.append({"arm": world, "task": m["task"], "irid": c["irid"], "call_index": c["call_index"], "where": where})
    if any(len(v) > 1 for v in kept.values()):
        raise ValueError("more than one kept run for a world and task")
    missing = want - set(loc)
    if missing:
        raise ValueError(f"{len(missing)} listed calls have no call_end row")
    if cal_cmd != cal_end:  # every impactful calibration command has exactly one call_end reading
        raise ValueError(f"{len(cal_cmd - cal_end)} impactful honest calibration commands lack a call_end row")
    if {t for (_, t) in kept if _ in ("honest_a", "honest_b")} - cal:
        raise ValueError("honest runs outside the calibration tasks")
    rows = [loc[k] for k in sorted(want)] + [h["where"] for h in honest]
    rnd = np.random.default_rng(RANDOM_SEED).normal(size=(N_RANDOM, 5120)).astype(np.float32)
    rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
    out = Path("/score") / OUT
    v, r = np.load(out / "v.npy")[LAYER_A].astype(np.float32), np.load(out / "r.npy")[LAYER_A].astype(np.float32)
    r_hat = r / np.linalg.norm(r)
    v_res = v - float(v @ r_hat) * r_hat
    B, rB = np.load(out / "B.npy")[LAYER_B].astype(np.float32), np.load(out / "rB.npy")[LAYER_B].astype(np.float32)
    dirs = {LAYER_A: np.stack([v, v_res, r, *rnd], 1), LAYER_B: np.stack([B, rB, *rnd], 1)}
    import hashlib
    dsha = {k: hashlib.sha256(np.ascontiguousarray(a, np.float32).tobytes()).hexdigest() for k, a in (("v48", v), ("r48", r), ("B46", B), ("rB46", rB))}
    proj = {}
    for layer, D in dirs.items():
        X = np.empty((len(rows), 5120), np.float32)
        for (x, chunk) in sorted({(w[0], w[1]) for w in rows}):
            a = np.load(Path("/acts") / x / "by_layer" / f"L{layer}" / f"{chunk}.npy")  # whole file, then index
            idx = [i for i, w in enumerate(rows) if (w[0], w[1]) == (x, chunk)]
            X[idx] = a[[rows[i][2] for i in idx]]
        if not np.isfinite(X).all():
            raise ValueError(f"non-finite readings at layer {layer}")
        proj[str(layer)] = (X @ D).tolist()
    n = len(want)
    return {"listed": [list(k) for k in sorted(want)], "honest": [{k: h[k] for k in ("arm", "task", "irid", "call_index")} for h in honest],
            "columns": {str(LAYER_A): ["v", "v_res", "r"] + [f"random_{i}" for i in range(N_RANDOM)],
                        str(LAYER_B): ["B", "rB"] + [f"random_{i}" for i in range(N_RANDOM)]},
            "proj_listed": {L: p[:n] for L, p in proj.items()}, "proj_honest": {L: p[n:] for L, p in proj.items()},
            "counts": {"listed": n, "honest_a": sum(h["arm"] == "honest_a" for h in honest), "honest_b": sum(h["arm"] == "honest_b" for h in honest),
                       "calibration_tasks_with_runs": len({t for (w, t) in kept if w in ("honest_a", "honest_b")})},
            "direction_sha256": dsha}
