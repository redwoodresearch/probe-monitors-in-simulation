"""Score the frozen Q3 detector on the five-arm v4 extraction, which adds 437 simulated honest runs (Modal CPU).

No refit. The trained probes of run 2026-10-06_181548 (runs/<run>/probes_L<layer>.npz on the work Volume: w, b and the
standardization mean and std of each outer and nested model) score every run's calls at each readout's chosen layer.
The features come from the v4 all-layers extraction, which matches the probes' training features:
  - same model revision, original text, 32k cut and positions (probe_shift.reader.segment_end_positions);
  - same layer hook (the output of decoder layer l, before the final norm), fp16.
A run's score per model is the mean over the readout's calls, as in run_q3_attack.probe_layer: outer model f scores
fold f, nested model (f, v) scores folds f and v. analyze_sim_honest.py computes the rates; it first checks that these
scores reproduce the committed numbers for the real arms.

  uv run modal run experiments/2026-10-06_q3_attack_detector/run_score_sim_honest.py
"""

import hashlib
import json
import socket
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REPO = HERE.parent.parent
APP = "probe-shift_aswin-surya_sub-q3attack_simhonest_kill-2026-11-06"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
WORK = "probe-shift_aswin-surya_sub-q3attack_work_kill-2026-11-06"
EXTRACTION = "2026-10-06_la_segment-end-v4_qwen3-32b_alllayers_v1"  # five arms, original text, all 64 layers
STAGED = "2026-10-06_la_segment-end-v4_qwen3-32b_v1"                # its staged reader messages (call arguments)
BASE_RUN = "2026-10-06_181548"                                      # the detector's trained probes and records
READOUTS = {"call_end_content": "call_end", "call_end": "call_end", "output_end": "output_end"}  # readout -> position

app = modal.App(APP)
acts_vol = modal.Volume.from_name(ACTS, version=2)
work_vol = modal.Volume.from_name(WORK, version=2)
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("torch==2.8.0", "numpy==2.3.3", "scikit-learn==1.7.2", "scipy==1.16.2")
         .add_local_python_source("probe_shift"))


@app.function(image=image, volumes={"/acts": acts_vol, "/work": work_vol}, cpu=8, memory=65536, timeout=3600)
def score(runs: list[dict], layers: dict, stamp: str, extraction: str = EXTRACTION, staged: str = STAGED,
          base_run: str = BASE_RUN, kinds_from: dict | None = None) -> dict:
    """Run scores [n] (outer) and [5, 5, n] (nested) per readout, from the frozen probes, plus row-count checks."""
    import numpy as np

    from probe_shift.attack_eval import CALL_END_KINDS, call_end_kind, original_kinds, pool

    t0 = time.time()
    ext, n = Path("/acts") / extraction, len(runs)
    want = {(r["arm"], r["key"]): i for i, r in enumerate(runs)}
    meta = [None] * n
    paths = [p for arm in sorted({r["arm"] for r in runs}) for p in (ext / "meta" / arm).glob("*.json")]
    with ThreadPoolExecutor(32) as ex:
        for m in ex.map(lambda p: json.loads(p.read_text()), paths):
            i = want.get((m["arm"], m["key"]))
            if i is not None:
                meta[i] = m
    missing = [runs[i]["key"] for i, m in enumerate(meta) if m is None]
    if missing:
        raise ValueError(f"{len(missing)} runs are not in {extraction}, e.g. {missing[:3]}")
    calls = {}
    for shard in sorted((Path("/acts") / staged / "inputs").glob("shard_*.jsonl")):
        for line in shard.read_text().splitlines():
            it = json.loads(line)
            if (it["arm"], it["key"]) in want:
                turns = [x for x in it["messages"] if x["role"] == "assistant"]
                calls[(it["arm"], it["key"])] = {(s, a): c["function"] for s, t in enumerate(turns)
                                                 for a, c in enumerate(t.get("tool_calls") or [])}
    if len(calls) != n:
        raise ValueError(f"{n - len(calls)} runs missing from the staged inputs")
    print(f"meta and inputs for {n} runs at {time.time() - t0:.0f} s", flush=True)
    # as in the detector run: each call's kind as it ends in original text (3: past the original cut)
    kinds0 = original_kinds(Path("/acts"), kinds_from, runs) if kinds_from else None

    # Rows per readout, grouped by run in run order: (run, chunk, row in the chunk file).
    rows = {r: [] for r in READOUTS}
    kinds = Counter()
    for i, m in enumerate(meta):
        for p in m["positions"]:
            if not p["kept"] or p["kind"] not in ("call_end", "output_end"):
                continue
            at = (i, m["chunk"], m["row_start"] + p["row"])
            if p["kind"] == "output_end":
                rows["output_end"].append(at)
                continue
            c = calls[(m["arm"], m["key"])][(p["step"], p["action"])]
            kind = (call_end_kind(c["arguments"], p["token_text"]) if kinds0 is None
                    else kinds0.get((i, p["step"], p["action"]), 3))
            kinds[(m["arm"], CALL_END_KINDS[kind])] += 1
            rows["call_end"].append(at)
            if kind == 0:
                rows["call_end_content"].append(at)
    out, info = {}, {"rows": {r: len(x) for r, x in rows.items()}, "call_end_kinds": {f"{a}|{k}": v for (a, k), v in kinds.items()},
                     "rows_per_run": {}, "feature_sha256": {}, "probes_sha256": {}}
    for readout, pos in READOUTS.items():
        layer = layers[readout]
        rr = rows[readout]
        run_of = np.array([x[0] for x in rr])
        info["rows_per_run"][readout] = np.bincount(run_of, minlength=n).tolist()
        X = np.zeros((len(rr), 5120), np.float16)
        by_chunk = {}
        for j, (_, chunk, row) in enumerate(rr):
            by_chunk.setdefault(chunk, []).append((j, row))
        for chunk, items in by_chunk.items():
            block = np.load(ext / "by_layer" / f"L{layer}" / f"{chunk}.npy", mmap_mode="r")
            X[[j for j, _ in items]] = block[[r for _, r in items]]
        if not np.isfinite(X).all():
            raise ValueError(f"{readout} L{layer}: non-finite features")
        info["feature_sha256"][readout] = hashlib.sha256(X.tobytes()).hexdigest()
        X = X.astype(np.float32)
        probes_path = Path("/work/runs") / base_run / f"probes_L{layer}.npz"
        info["probes_sha256"][f"L{layer}"] = hashlib.sha256(probes_path.read_bytes()).hexdigest()
        z = np.load(probes_path)
        fold = np.array([r["fold"] for r in runs])
        outer, nested = np.full(n, np.nan), np.full((5, 5, n), np.nan)

        def run_scores(model, scored_runs):
            """As probe_layer: standardize with the training rows' mean and std, logit = x . w + b, mean per run."""
            key = f"{readout}__{model}"
            sel = np.flatnonzero(scored_runs[run_of])
            xs = (X[sel] - z[f"{key}__mean"]) / z[f"{key}__std"]
            s = (xs @ z[f"{key}__w"] + np.float32(z[f"{key}__b"])).astype(np.float64)
            return pool(run_of[sel], s, n, "mean")

        for f in range(5):
            outer[fold == f] = run_scores(f"task_outer_{f}", fold == f)[fold == f]
            for v in range(5):
                if v != f:
                    sc = (fold == f) | (fold == v)
                    nested[f, v][sc] = run_scores(f"task_nested_{f}_{v}", sc)[sc]
        out[readout] = {"layer": layer, "outer": outer, "nested": nested}
        print(f"{readout} L{layer}: {len(rr)} rows scored at {time.time() - t0:.0f} s", flush=True)
    dest = Path("/work/simhonest") / stamp
    dest.mkdir(parents=True, exist_ok=True)
    np.savez(dest / "scores.npz", **{f"{r}__{k}": v[k] for r, v in out.items() for k in ("outer", "nested")})
    (dest / "info.json").write_text(json.dumps(info))
    work_vol.commit()
    info["seconds"] = round(time.time() - t0)
    return {"scores": {r: {k: v[k] for k in ("layer", "outer", "nested")} for r, v in out.items()}, "info": info}


def wait(call):
    for _ in range(120):  # our network: retry for up to an hour; remote errors are raised
        try:
            return call.get()
        except (socket.gaierror, ConnectionError, TimeoutError, modal.exception.ConnectionError,
                modal.exception.ServiceError) as e:
            print(f"lost the connection ({e!r}); retrying in 30 s", flush=True)
            time.sleep(30)
    raise TimeoutError("no connection to Modal for an hour; the scores are on the work Volume")


@app.local_entrypoint()
def main(extraction: str = EXTRACTION, staged: str = STAGED, base_run: str = BASE_RUN):  # kinds_from: the base run's config
    """Defaults: the original-text detector. For the level-7 one: --extraction <v4-scrub7 all-layers> --staged
    <its staged inputs> --base-run <the level-7 run>."""
    import numpy as np

    res = HERE / "results"
    record = json.loads((res / f"run_{base_run}.json").read_text())
    metrics = json.loads(sorted(res.glob(f"metrics_{base_run}__*.json"))[-1].read_text())
    layers = {}
    for readout in READOUTS:  # each readout's chosen layer: one layer in every fold, or this frozen scoring is ambiguous
        chosen = set(metrics["readouts"][f"probe_{readout}_mean"]["task"]["chosen_layer_by_fold"].values())
        if len(chosen) != 1:
            raise ValueError(f"{readout}: the committed analysis chose different layers by fold ({chosen})")
        layers[readout] = int(chosen.pop())
    runs = json.loads((res / f"run_scores_{base_run}" / "runs.json").read_text())
    ex = json.loads((REPO / "splits" / "exclusions_v4.json").read_text())
    folds = json.loads((REPO / "splits" / "folds_v4.json").read_text())
    tasks = set(ex["paired_honest_cells"]["cells"])
    m2f, e2f = folds["task_folds"]["main_task_to_fold"], folds["environment_folds"]["environment_to_fold"]
    sim_honest = [{k: s[k] for k in ("arm", "key", "cell", "main_task", "side_task", "environment", "sim_reference",
                                     "tokens", "truncated")} | {"fold": m2f[s["main_task"]], "env_fold": e2f[s["environment"]]}
                  for s in ex["samples"] if s["arm"] == "sim-honest" and not s["reason"] and s["main_task"] in tasks]
    if len(sim_honest) != len(tasks):
        raise ValueError(f"{len(sim_honest)} valid simulated honest runs for {len(tasks)} paired honest tasks")
    # Real honest runs of these tasks that the base run left out (their tasks have no paired attack cell), so every
    # task has its real honest runs. They are scored, but never enter the frozen (committed) thresholds.
    base_keys = {(r["arm"], r["key"]) for r in runs}
    extra_real = [{k: s[k] for k in ("arm", "key", "cell", "main_task", "environment", "truncated")}
                  | {"fold": m2f[s["main_task"]], "env_fold": e2f[s["environment"]]}
                  for s in ex["samples"] if s["arm"] == "real-honest" and not s["reason"] and s["main_task"] in tasks
                  and (s["arm"], s["key"]) not in base_keys]
    fields = ("arm", "key", "cell", "main_task", "environment", "fold", "env_fold", "truncated")
    all_runs = ([{k: r[k] for k in fields} | {"in_base": True} for r in runs]
                + [{k: r[k] for k in fields} | {"in_base": False} for r in extra_real + sim_honest])
    missing = tasks - {r["main_task"] for r in all_runs if r["arm"] == "real-honest"}
    if missing:
        raise ValueError(f"{len(missing)} paired honest tasks have no valid real honest run: {sorted(missing)[:3]}")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=REPO).stdout.strip())
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    print(f"scoring {len(all_runs)} runs ({len(sim_honest)} simulated honest, {len(extra_real)} real honest outside the "
          f"base run) with the frozen probes of {base_run} on {extraction}; "
          f"layers {layers}; commit {head[:8]}{' (dirty)' if dirty else ''}", flush=True)
    started = time.time()
    kinds_from = record["config"].get("kinds_from")  # the same call kinds as the detector run
    out = wait(score.spawn(all_runs, layers, stamp, extraction, staged, base_run, kinds_from))
    d = res / f"sim_honest_scores_{stamp}"
    d.mkdir(exist_ok=False)
    np.savez(d / "scores.npz", **{f"{r}__{k}": v[k] for r, v in out["scores"].items() for k in ("outer", "nested")})
    (d / "runs.json").write_text(json.dumps(all_runs))
    record_out = {"stamp": stamp, "git": head, "git_dirty": dirty, "base_run": base_run, "extraction": extraction,
                  "staged_inputs": staged, "kinds_from": kinds_from, "layers": layers, "n_runs": len(all_runs),
                  "modal": {"app": APP, "cpu": 8, "memory_mb": 65536, "raw": f"{WORK}/simhonest/{stamp}"},
                  "wall_seconds": round(time.time() - started), "info": out["info"]}
    (res / f"sim_honest_scoring_{stamp}.json").write_text(json.dumps(record_out, indent=1))
    print(f"scores -> {d}; record -> sim_honest_scoring_{stamp}.json; {out['info']['seconds']} s in the container")
