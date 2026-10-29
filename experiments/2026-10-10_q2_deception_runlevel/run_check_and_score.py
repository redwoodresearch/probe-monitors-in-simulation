"""Run-level deception test (Modal CPU; PLAN.md): the checks, then, only if every check passes, the frozen probe's
per-call scores at the rule-1 command token of every rule-impactful call in six dumps. Keys, positions, hashes and
stored activations only: no transcript is opened and no token is decoded.

Checks (any failure: no score is computed, the report says why):
- the probe file: sha256 equal to amendment 1's check (c4cba16e…), arrays finite, scales positive;
- each dump: config names the model and revision; every meta's chunk index lists it; its inputs hash equals the
  config's; counted attempts only, one run per arm and task;
- every row to be scored: its readout position equals its dump's staging record, by (irid, call_index), and lies
  inside the completion (at or after the staged prompt length); call keys are unique;
- every layer-31 row read: finite.
The job writes its output to its own Volume before returning (the local entrypoint spawns it and saves the call id).

  uv run modal run --detach experiments/2026-10-10_q2_deception_runlevel/run_check_and_score.py --staging-dir <folder of staging records>
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-deception-runs_kill-2026-11-08"
ACTS = "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08"
PROBES = "probe-shift_aswin-surya_sub-q2-deception-qwen35_kill-2026-11-08"
PROBE_FILE, PROBE_SHA, LAYER = "runs/2026-10-09_191400/probes_block_output.npz", "c4cba16e0eb74925a946a46d17271cb8e8ccc3df2a28e652b9bd8a2d3bce74bd", 31
MODEL, REVISION = "Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654"
ARMS4 = ["attack", "honest_a", "honest_b", "sim_attack@september"]
DUMPS = [  # (dump, staging record, arms read, name given to the "attack" arm)
    ("2026-10-09_rerun_strict_segends_v1", "replay_readouts_rerun_strict_segends_2026-10-09_135600.json", ARMS4, "attack"),
    ("2026-10-10_scaleup_canary_segends_v1", "replay_readouts_scaleup_canary_segends_2026-10-10_014918.json", ARMS4, "attack"),
    ("2026-10-10_scaleup_l2_segends_v1", "replay_readouts_scaleup_l2_segends_2026-10-10_014046.json", ARMS4, "attack"),
    ("2026-10-10_scaleup_topup_segends_v1", "replay_readouts_scaleup_topup_segends_2026-10-10_023740.json", ARMS4, "attack"),
    ("2026-10-10_second_attack_segends_v1", "replay_readouts_second_attack_segends_2026-10-10_024517.json", ["attack"], "attack_run2"),
]
app = modal.App(APP)
out_vol = modal.Volume.from_name(APP, create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")


@app.function(image=image, cpu=8, memory=32768, timeout=3600, retries=0,
              volumes={"/acts": modal.Volume.from_name(ACTS).read_only(), "/probes": modal.Volume.from_name(PROBES).read_only(), "/out": out_vol})
def run(staging: dict, stamp: str) -> dict:
    import time

    import numpy as np

    t0, report, problems = time.time(), {}, []
    npz = Path("/probes") / PROBE_FILE
    p = np.load(npz)
    arrays = {k: p[k] for k in ("coef", "scaler_mean", "scaler_scale")}
    sha = hashlib.sha256(npz.read_bytes()).hexdigest()
    report["probe"] = {"sha256 as expected": sha == PROBE_SHA, "all finite": all(bool(np.isfinite(v).all()) for v in arrays.values()),
                       "scales positive at layer 31": bool((arrays["scaler_scale"][LAYER] > 0).all())}
    if not all(report["probe"].values()):
        problems.append("probe")
    plan = []  # (dump, arm name, meta, call) to score
    for dump, _, arms, attack_name in DUMPS:
        root = Path("/acts") / dump
        cfg = json.loads((root / "config.json").read_text())
        r = {"model and revision": cfg.get("model") == MODEL and cfg.get("revision") == REVISION, "meta problems": 0, "runs": {}, "rows": 0,
             "readout matched": 0, "readout missing from staging": 0, "readout unequal": 0, "readout before the completion": 0, "duplicate counted runs": 0,
             "counted runs with no scored row": {}}
        st = staging[dump]
        r["complete.json present"] = (root / "complete.json").exists()
        r["arms present"] = all(any((root / "meta" / a).glob("*.json")) for a in arms)
        seen, per_run = {}, {}
        for arm in arms:
            for f in sorted((root / "meta" / arm).glob("*.json")):
                m = json.loads(f.read_text())
                idx = json.loads((root / "chunks" / f"{m['chunk']}.json").read_text())
                if m["key"] not in {e["key"] for e in idx["epochs"]} or (cfg.get("inputs_sha256") or {}).get(f"{arm}.jsonl") not in (None, m["inputs_sha256"]):
                    r["meta problems"] += 1
                if m.get("redrawn_because") or m.get("attempt_kind") == "rerun":
                    continue
                seen.setdefault((arm, m["task"]), set()).add(m["run"])
                name = attack_name if arm == "attack" else arm
                n_before = len(plan)
                per_run.setdefault((name, m["run"]), 0)
                for c in m["calls"]:
                    if c.get("kind") != "command" or c.get("label") != "impactful" or c.get("row") is None:
                        continue
                    s = st.get(f"{c['irid']}|{c['call_index']}")
                    r["readout matched" if s and s[0] == c["readout"] else "readout missing from staging" if s is None else "readout unequal"] += 1
                    r["readout before the completion"] += bool(s) and c["readout"] < s[1]
                    plan.append((dump, name, m, c))
                    r["rows"] += 1
                per_run[(name, m["run"])] += len(plan) - n_before
        r["duplicate counted runs"] = sum(len(v) > 1 for v in seen.values())
        for (name, _), k in per_run.items():  # aggregated by run, not by epoch
            if k == 0:
                r["counted runs with no scored row"][name] = r["counted runs with no scored row"].get(name, 0) + 1
        r["runs"] = {a: sum(1 for (arm, _), v in seen.items() if arm == a) for a in arms}
        if (not r["model and revision"] or not r["complete.json present"] or not r["arms present"] or r["meta problems"] or r["readout missing from staging"] or r["readout unequal"]
                or r["readout before the completion"] or r["duplicate counted runs"]):
            problems.append(dump)
        report[dump] = r
    keys = [(dump, m["run"], c["irid"], c["call_index"]) for dump, _, m, c in plan]
    report["unique call keys"] = len(set(keys)) == len(keys)
    if not report["unique call keys"]:
        problems.append("duplicate call keys")
    report["problems"] = problems
    if problems:
        (Path("/out") / f"{stamp}_report.json").write_text(json.dumps(report)); out_vol.commit()
        return {"report": report, "scores": None}
    # Every check passed: read layer 31 and score
    coef, mu, sc = (arrays[k][LAYER].astype(np.float64) for k in ("coef", "scaler_mean", "scaler_scale"))
    by_chunk = {}
    for i, (dump, _, m, c) in enumerate(plan):
        by_chunk.setdefault((dump, m["chunk"]), []).append((i, m["row_start"] + c["row"]))
    X = np.zeros((len(plan), 5120), np.float16)
    for (dump, chunk), items in by_chunk.items():
        block = np.load(Path("/acts") / dump / "by_layer" / f"L{LAYER}" / f"{chunk}.npy")
        X[[i for i, _ in items]] = block[[r for _, r in items]]
    finite = bool(np.isfinite(X.astype(np.float32)).all())
    report["layer-31 rows finite"] = finite
    if not finite:
        return {"report": report, "scores": None}
    s = ((X.astype(np.float64) - mu) / sc) @ coef
    if not bool(np.isfinite(s).all()):
        report["scores finite"] = False
        return {"report": report, "scores": None}
    scores = [{"dump": dump, "arm": name, "task": m["task"], "run": m["run"], "epoch": m.get("epoch"), "turn": c.get("turn"), "irid": c["irid"],
               "call_index": c["call_index"], "score": float(v)}
              for (dump, name, m, c), v in zip(plan, s)]
    res = {"report": report, "scores": scores, "seconds": round(time.time() - t0)}
    (Path("/out") / f"{stamp}.json").write_text(json.dumps(res))  # durable before returning
    out_vol.commit()
    return res


@app.local_entrypoint()
def main(staging_dir: str):
    staging, shas = {}, {}
    for dump, fname, _, _ in DUMPS:
        f = Path(staging_dir) / fname
        shas[fname] = hashlib.sha256(f.read_bytes()).hexdigest()
        staging[dump] = {f"{r['irid']}|{r['call_index']}": [r["readout"], r["prompt_len"]] for r in json.loads(f.read_text())["calls"] if r.get("kind") == "command"}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    data = HERE.parents[1] / "data" / "deception_runlevel"
    data.mkdir(parents=True, exist_ok=True)
    call = run.spawn(staging, stamp)
    (data / f"launch_{stamp}.json").write_text(json.dumps({"stamp": stamp, "call": call.object_id, "volume": f"{APP}:/{stamp}.json"}) + "\n")
    print(f"stamp {stamp}, call {call.object_id}", flush=True)
    res = call.get() | {"staging sha256": shas, "stamp": stamp}
    with open(data / f"scores_{stamp}.json", "x") as fh:
        fh.write(json.dumps(res) + "\n")
    out = HERE / "results" / f"check_{stamp}.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "x") as fh:  # the check report alone (no scores)
        fh.write(json.dumps({"report": res["report"], "staging sha256": shas, "stamp": stamp, "scored rows": len(res["scores"] or [])}, indent=1) + "\n")
    print(json.dumps(res["report"], indent=1))
    print(out)
