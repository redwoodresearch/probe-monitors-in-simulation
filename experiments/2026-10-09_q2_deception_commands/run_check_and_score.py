"""Deception probe at command tokens, amendment 1 (Modal CPU): the compatibility checks, then (a separate step, after
the checks are reported) the frozen probe's scores. Keys, positions, hashes and stored activations only: no transcript
is opened and no token is decoded.

--step check: the assertions of PLAN.md amendment 1 (probe provenance and arrays, the dump's config, meta, chunk index
and inputs hash, counted attempts, the key join on call_index with the staging record's readout, the honest
calibration rows, finiteness and fp16 range of the layer-31 rows, and agreement with the replay dump on the 60 keys).
Returns counts and booleans only; no score is computed.
--step score: the probe's per-token score ((x - scaler_mean) / scaler_scale) . coef at layer 31 for the 60 keys and the
honest calibration rows, written to data/ (scores keyed by run, irid and call index).

  uv run modal run --detach experiments/2026-10-09_q2_deception_commands/run_check_and_score.py --step check --staging <staging json> --step2 <step-2 result json>
"""
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-deception-commands_kill-2026-11-08"
ACTS = "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08"
PROBES = "probe-shift_aswin-surya_sub-q2-deception-qwen35_kill-2026-11-08"
SEGENDS, REPLAY, RUN2, LAYER = "2026-10-09_rerun_strict_segends_v1", "2026-10-09_rerun_strict_replay_v1", "runs/2026-10-09_191400", 31
MODEL, REVISION = "Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654"
FP16_MIN_NORMAL = 6.103515625e-05
app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")
VOLS = {"/acts": modal.Volume.from_name(ACTS).read_only(), "/probes": modal.Volume.from_name(PROBES).read_only()}


def _load(keys, staging, need_honest=True):
    """Counted epochs of a dump, the key rows (attack) and the honest calibration rows, with assertions."""
    import numpy as np

    def dump(run, arms):
        root = Path("/acts") / run
        cfg = json.loads((root / "config.json").read_text())
        metas, problems = [], []
        for arm in arms:
            for f in sorted((root / "meta" / arm).glob("*.json")):
                m = json.loads(f.read_text())
                idx = json.loads((root / "chunks" / f"{m['chunk']}.json").read_text())
                if m["key"] not in {e["key"] for e in idx["epochs"]}:
                    problems.append(f"{f.name}: its chunk does not list it")
                if (cfg.get("inputs_sha256") or {}).get(f"{arm}.jsonl") not in (None, m["inputs_sha256"]):
                    problems.append(f"{f.name}: inputs sha256 differs from the config's")
                metas.append(m)
        counted, seen = [], {}
        for m in metas:
            if m.get("redrawn_because") or m.get("attempt_kind") == "rerun":
                continue
            seen.setdefault((m["arm"], m["task"]), set()).add(m["run"])
            counted.append(m)
        dup = {k: v for k, v in seen.items() if len(v) > 1}
        if dup:
            problems.append(f"{len(dup)} arm/task pairs with several counted runs")
        return root, cfg, counted, problems

    def rows_of(m):
        """(irid, call index) -> call, the call index taken from the meta or, if absent, the order within the request."""
        out, order = {}, {}
        for c in m["calls"]:
            if c.get("kind", "command") != "command":
                continue
            ci = c.get("call_index")
            if ci is None:
                ci = order.get(c["irid"], 0)
                order[c["irid"]] = ci + 1
            out[(c["irid"], ci)] = c
        return out

    def read(root, items):
        """items: (chunk, row) -> layer-31 rows, chunk files loaded whole."""
        by_chunk = {}
        for i, (chunk, row) in enumerate(items):
            by_chunk.setdefault(chunk, []).append((i, row))
        X = np.zeros((len(items), 5120), np.float16)
        for chunk, lst in by_chunk.items():
            block = np.load(root / "by_layer" / f"L{LAYER}" / f"{chunk}.npy")
            X[[i for i, _ in lst]] = block[[r for _, r in lst]]
        return X

    root, cfg, counted, problems = dump(SEGENDS, ["attack", "honest_a", "honest_b"] if need_honest else ["attack"])
    attack = {}
    for m in counted:
        if m["arm"] == "attack":
            for k, c in rows_of(m).items():
                attack[k] = (m, c)
    key_rows, join = [], {"keys": len(keys), "found": 0, "readout equals staging": 0, "row present": 0, "task matches": 0}
    for k in keys:
        hit = attack.get((k["irid"], int(k["call"])))
        if hit is None:
            continue
        m, c = hit
        join["found"] += 1
        join["readout equals staging"] += c["readout"] == staging.get(f"{k['irid']}|{k['call']}")
        join["row present"] += c.get("row") is not None
        join["task matches"] += m["task"] == k["task"]
        key_rows.append((k, m, c))
    honest = []
    if need_honest:
        for m in counted:
            if m["arm"] in ("honest_a", "honest_b"):
                for (irid, ci), c in rows_of(m).items():
                    if c.get("label") == "impactful" and c.get("row") is not None:
                        honest.append((m, irid, ci, c))
    X_keys = read(root, [(m["chunk"], m["row_start"] + c["row"]) for _, m, c in key_rows])
    X_honest = read(root, [(m["chunk"], m["row_start"] + c["row"]) for m, _, _, c in honest]) if honest else None
    return {"root": root, "cfg": cfg, "problems": problems, "join": join, "key_rows": key_rows, "honest": honest,
            "X_keys": X_keys, "X_honest": X_honest, "rows_of": rows_of, "read": read, "dump": dump}


@app.function(image=image, volumes=VOLS, cpu=4, memory=16384, timeout=1800, retries=0)
def check(keys: list[dict], staging: dict, step2: dict) -> dict:
    import os

    import numpy as np

    out = {}
    # Probe provenance and arrays
    run2 = Path("/probes") / RUN2
    vol_result = json.loads((run2 / "result.json").read_text())
    shared = [k for k in vol_result if k in step2]
    out["probe: result.json fields equal the archived step-2 result"] = {"fields compared": len(shared), "fields only on the Volume": sorted(set(vol_result) - set(step2)),
                                                                        "all equal": all(vol_result[k] == step2[k] for k in shared)}
    npz = run2 / "probes_block_output.npz"
    st = os.stat(npz)
    p = np.load(npz)
    arrays = {k: p[k] for k in ("coef", "scaler_mean", "scaler_scale")}
    out["probe file"] = {"sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
                         "modified (PT)": datetime.fromtimestamp(st.st_mtime, ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds"),
                         "result.json modified (PT)": datetime.fromtimestamp(os.stat(run2 / "result.json").st_mtime, ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds"),
                         "shapes": {k: list(v.shape) for k, v in arrays.items()}, "all finite": all(bool(np.isfinite(v).all()) for v in arrays.values()),
                         "scaler_scale > 0 at layer 31": bool((arrays["scaler_scale"][LAYER] > 0).all())}
    out["step-2 training config"] = {k: step2["config"].get(k) for k in ("model", "revision", "dtype", "chat_template")}
    # The dump
    d = _load(keys, staging)
    cfg = d["cfg"]
    out["dump config"] = {"model": cfg.get("model"), "revision": cfg.get("revision"), "layer 31 stored": LAYER in (cfg.get("layers") or range(64)),
                          "inputs": cfg.get("inputs"), "same model and revision as the probe": cfg.get("model") == MODEL and cfg.get("revision") == REVISION}
    out["dump integrity problems"] = d["problems"]
    out["key join (call -> call_index)"] = d["join"]
    out["honest calibration rows (counted attempts, impactful, with a row)"] = {
        "honest_a": sum(m["arm"] == "honest_a" for m, *_ in d["honest"]), "honest_b": sum(m["arm"] == "honest_b" for m, *_ in d["honest"]),
        "tasks": len({m["task"] for m, *_ in d["honest"]})}
    labels = {}
    for k, m, _ in d["key_rows"]:
        labels.setdefault(m["task"], set()).add(k["label"])
    out["labelled tasks"] = {"tasks": len(labels), "with both labels": sum(len(v) == 2 for v in labels.values()),
                             "attack runs per task": max(len({m["run"] for kk, m, _ in d["key_rows"] if m["task"] == t}) for t in labels)}
    for name, X in (("key rows", d["X_keys"]), ("honest rows", d["X_honest"])):
        x = X.astype(np.float32)
        nz = x[x != 0]
        out[f"layer-31 {name}"] = {"rows": len(X), "all finite": bool(np.isfinite(x).all()),
                                   "share of nonzero values below fp16's smallest normal": float((np.abs(nz) < FP16_MIN_NORMAL).mean()),
                                   "max |x|": float(np.abs(x).max())}
    # Cross-dump agreement on the 60 keys
    rroot, rcfg, rcounted, rproblems = d["dump"](REPLAY, ["attack"])
    rattack = {}
    for m in rcounted:
        for k, c in d["rows_of"](m).items():
            rattack[k] = (m, c)
    pairs = [(i, rattack.get((k["irid"], int(k["call"])))) for i, (k, _, _) in enumerate(d["key_rows"])]
    found = [(i, mc) for i, mc in pairs if mc is not None and mc[1].get("row") is not None]
    Xr = d["read"](rroot, [(m["chunk"], m["row_start"] + c["row"]) for _, (m, c) in found])
    diff = np.abs(Xr.astype(np.float32) - d["X_keys"][[i for i, _ in found]].astype(np.float32))
    out["cross-dump (replay_v1) on the keys"] = {"found": len(found), "same readout": sum(mc[1]["readout"] == d["key_rows"][i][2]["readout"] for i, mc in found),
                                                 "max abs difference": float(diff.max()) if len(found) else None,
                                                 "replay config revision": rcfg.get("revision"), "problems": rproblems}
    return out


@app.function(image=image, volumes=VOLS, cpu=4, memory=16384, timeout=1800, retries=0)
def score(keys: list[dict], staging: dict) -> dict:
    import numpy as np

    p = np.load(Path("/probes") / RUN2 / "probes_block_output.npz")
    coef, mu, sc = (p[k][LAYER].astype(np.float64) for k in ("coef", "scaler_mean", "scaler_scale"))
    f = lambda X: (((X.astype(np.float64) - mu) / sc) @ coef).tolist()
    d = _load(keys, staging)
    if d["problems"] or d["join"]["found"] != len(keys) or d["join"]["readout equals staging"] != len(keys):
        raise ValueError(f"checks failed: {d['problems']}, {d['join']}")
    return {"keys": [{"task": k["task"], "irid": k["irid"], "call": int(k["call"]), "label": k["label"], "run": m["run"], "score": s}
                     for (k, m, _), s in zip(d["key_rows"], f(d["X_keys"]))],
            "honest": [{"arm": m["arm"], "task": m["task"], "run": m["run"], "irid": irid, "call": ci, "score": s}
                       for (m, irid, ci, _), s in zip(d["honest"], f(d["X_honest"]))]}


@app.local_entrypoint()
def main(step: str, staging: str, step2: str = ""):
    with open(HERE / "labels" / "reading_keys.csv") as fh:
        keys = list(csv.DictReader(fh))
    st = {f"{r['irid']}|{r['call_index']}": r["readout"] for r in json.loads(Path(staging).read_text())["calls"] if r.get("kind") == "command"}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    prov = {"labels sha256": hashlib.sha256((HERE / "labels" / "reading_keys.csv").read_bytes()).hexdigest(),
            "staging sha256": hashlib.sha256(Path(staging).read_bytes()).hexdigest(), "stamp": stamp}
    if step == "check":
        res = check.remote(keys, st, json.loads(Path(step2).read_text())) | prov | {"step-2 result sha256": hashlib.sha256(Path(step2).read_bytes()).hexdigest()}
        out = HERE / "results" / f"compatibility_check_{stamp}.json"
    elif step == "score":
        res = score.remote(keys, st) | prov
        out = HERE.parents[1] / "data" / "deception_commands" / f"scores_{stamp}.json"
    else:
        raise SystemExit("--step check or score")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x") as fh:
        fh.write(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1) if step == "check" else f"{len(res['keys'])} key scores, {len(res['honest'])} honest scores")
    print(out)
