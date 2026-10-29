"""Q3 attack detector, the slow part: per-call probes on Qwen3-32B segment-end activations and a TF-IDF baseline,
trained on real LinuxArena honest vs real attack (epoch 1) runs and scored on every arm. Metrics: analyze_q3_attack.py.

Modal steps:
  1. manifest (CPU): finds each run's activation file, builds the row tables (one row per kept call_end and output_end
     position, in transcript order, with what each call_end token is: content, a timeout digit or an escaped
     newline), counts steps, actions and "Please proceed" nudges in the staged reader inputs, checks that every attack
     run was read with the honest system prompt and task message of its main task, and writes the call texts for
     TF-IDF. Containers have no AWS credentials, so everything comes from the Volumes.
  2. pack (CPU, one container per layer): one float16 matrix per position, rows in table order (GPU containers read
     the Volume's many small files slowly).
  3. probe (H100, one per layer), per readout (config `readouts`: call_end, call_end on content tokens only,
     output_end): L2 logistic regression per call, labelled with its run's label, fit on honest and e1 runs only:
       - task outer: train on 4 main-task folds, score the 5th;
       - task nested (f, v): train on the 3 folds other than f and v, score folds f and v. v's honest runs set the
         frozen thresholds and v's honest and e1 runs choose the layer; f is the test fold;
       - environment outer (leave one environment out) and nested (v = the next environment);
       - label-shuffle null on the task outer folds (labels permuted among each main task's honest and e1 runs).
     Call scores become run scores by pooling (attack_eval.POOLINGS).
  4. tfidf (CPU, one container per model): the same splits on the text of each call (all calls; content-token calls).
Every job writes its outputs to the work Volume (runs/<run_id>/) and marks itself done last, so `--resume <run_id>`
reruns only what is missing. collect() then copies the raw run scores to results/run_scores_<run_id>/ (npz,
gitignored), archives the run to S3 from the laptop with the user's own AWS session (archive(); personal credentials
never go into a container), and writes the run record (config, git hash, counts, exclusions, checks, fit diagnostics,
timings, archive receipt) to results/run_<run_id>.json.

  uv run modal run --detach experiments/2026-10-06_q3_attack_detector/run_q3_attack.py [--test]
  uv run modal run --detach experiments/2026-10-06_q3_attack_detector/run_q3_attack.py --resume <run_id>
  uv run python experiments/2026-10-06_q3_attack_detector/run_q3_attack.py --archive-only <run_id>   # laptop only
"""

import hashlib
import io
import json
import os
import socket
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REPO = HERE.parent.parent
APP = "probe-shift_aswin-surya_sub-q3attack_probe_kill-2026-11-06"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
WORK = "probe-shift_aswin-surya_sub-q3attack_work_kill-2026-11-06"
GPU = "H100!"
H = 5120
MANIFEST_VERSION = 2  # part of data_id: bump when the manifest's outputs change (2: a tool column in the tables)
# Durable home of each run's raw scores and trained probes (the work Volume is kill-dated): the project bucket
# (private, versioned), one prefix per run. Archives are written from the laptop with the user's own AWS session.
S3_ARCHIVE = "s3://individual-aswin-surya-probe-shift-kill-2027-04-06/experiments/2026-10-06_q3_attack_detector/"
S3_ARCHIVE_REGION = "us-east-2"
ARCHIVE_MAX_BYTES = 1_000_000_000  # a run's archive is ~350 MB; refuse anything over ~1 GB rather than fetch it

app = modal.App(APP)
acts_vol = modal.Volume.from_name(ACTS, version=2)
work_vol = modal.Volume.from_name(WORK, create_if_missing=True, version=2)
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("torch==2.8.0", "numpy==2.3.3", "scikit-learn==1.7.2", "scipy==1.16.2")
         .add_local_python_source("probe_shift"))


def data_dir(data_id: str) -> Path:
    return Path("/work/data") / data_id



@app.function(image=image, volumes={"/acts": acts_vol, "/work": work_vol}, cpu=8, memory=32768, timeout=3600)
def manifest(cfg: dict, runs: list[dict], data_id: str) -> dict:
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor

    from probe_shift.attack_eval import CALL_END_KINDS, HONEST, NUDGE, STEP_LIMIT, call_end_kind, original_kinds

    out = data_dir(data_id)
    if (out / "summary.json").exists():  # built earlier from the same runs and settings (data_id is their hash)
        return json.loads((out / "summary.json").read_text())
    ext = Path("/acts") / cfg["extraction_run"]
    by_layer = cfg.get("layout") == "by_layer"  # an all-layers extraction: meta/<arm>/*.json, by_layer/L<l>/<chunk>.npy
    want = {(r["arm"], r["key"]): i for i, r in enumerate(runs)}
    meta = [None] * len(runs)
    paths = [p for arm in sorted({r["arm"] for r in runs}) for p in (ext / ("meta" if by_layer else "acts") / arm).glob("*.json")]
    with ThreadPoolExecutor(32) as pool:
        for m in pool.map(lambda p: json.loads(p.read_text()), paths):
            i = want.get((m["arm"], m["key"]))
            if i is not None:
                meta[i] = m
    missing = [runs[i]["key"] for i, m in enumerate(meta) if m is None]
    if missing:
        raise ValueError(f"{len(missing)} runs have no extraction, e.g. {missing[:3]}")

    # The staged reader inputs: exactly what Qwen read for each run.
    items, honest_prompts = {}, {}
    # an all-layers extraction reads the inputs staged by another run (its config's inputs_from)
    for shard in sorted((Path("/acts") / cfg.get("inputs_run", cfg["extraction_run"]) / "inputs").glob("shard_*.jsonl")):
        for line in shard.read_text().splitlines():
            it = json.loads(line)
            ms = it["messages"]
            if it["arm"] == HONEST:
                honest_prompts.setdefault(it["main_task"], set()).add((ms[0]["content"], ms[1]["content"]))
            if (it["arm"], it["key"]) in want:
                items[(it["arm"], it["key"])] = ms
    if len(items) != len(runs):
        raise ValueError(f"{len(runs) - len(items)} runs missing from the staged inputs")
    if any(len(p) != 1 for p in honest_prompts.values()):
        raise ValueError("a main task has more than one honest system prompt or task message")
    kinds0 = original_kinds(Path("/acts"), cfg["kinds_from"], runs) if cfg.get("kinds_from") else None

    tables = {pos: [] for pos in cfg["positions"]}
    tool_ids = {}  # tool name -> id in the tables' tool column
    token_texts = {arm: Counter() for arm in {r["arm"] for r in runs}}
    texts, checks, per_run = [], {"system_prompt_differs": [], "task_message_differs": [], "unexpected_user_messages": [],
                                  "call_count_mismatch": []}, []
    for i, (r, m) in enumerate(zip(runs, meta)):
        ms = items[(r["arm"], r["key"])]
        (sys_h, user_h), = honest_prompts[r["main_task"]]
        if ms[0]["role"] != "system" or ms[0]["content"] != sys_h:
            checks["system_prompt_differs"].append(r["key"])
        if ms[1]["role"] != "user" or ms[1]["content"] != user_h:
            checks["task_message_differs"].append(r["key"])
        later_users = [x["content"] for x in ms[2:] if x["role"] == "user"]
        n_nudges = sum(NUDGE in u for u in later_users)
        n_limit = sum(u.startswith(STEP_LIMIT) for u in later_users)
        if n_nudges + n_limit != len(later_users):
            checks["unexpected_user_messages"].append(r["key"])
        turns = [x for x in ms if x["role"] == "assistant"]
        calls = {(s, a): c["function"] for s, t in enumerate(turns) for a, c in enumerate(t.get("tool_calls") or [])}
        call_pos = [p for p in m["positions"] if p["kind"] == "call_end"]
        if {(p["step"], p["action"]) for p in call_pos} != set(calls) or len(call_pos) != len(calls):
            checks["call_count_mismatch"].append(r["key"])
        kinds, tools = Counter(), Counter()
        for pos in cfg["positions"]:
            for p in m["positions"]:
                if p["kind"] == pos and p["kept"]:
                    kind = -1
                    if pos == "call_end":
                        c = calls[(p["step"], p["action"])]
                        texts.append(json.dumps({"name": c["name"], "arguments": c["arguments"]}, ensure_ascii=False))
                        kind = (call_end_kind(c["arguments"], p["token_text"]) if kinds0 is None
                                else kinds0.get((i, p["step"], p["action"]), 3))  # 3: past the original cut
                        kinds[CALL_END_KINDS[kind]] += 1
                        tools[c["name"]] += 1
                        token_texts[r["arm"]][p["token_text"]] += 1
                    tool = tool_ids.setdefault(p["tool"] or "", len(tool_ids))
                    tables[pos].append((i, p["row"], p["step"], p["action"], p["token"], kind, tool))
        per_run.append({"filename": m["filename"], "n_tokens": m["n_tokens"], "truncated": m["truncated"],
                        "n_steps": len(turns), "n_actions": len(calls), "n_nudges": n_nudges, "n_step_limit_messages": n_limit,
                        **{f"n_{pos}_kept": sum(p["kind"] == pos and p["kept"] for p in m["positions"]) for pos in cfg["positions"]},
                        "n_call_end_none": sum(p["kind"] == "call_end" and p["token"] is None for p in m["positions"]),
                        "n_output_end_none": sum(p["kind"] == "output_end" and p["token"] is None for p in m["positions"]),
                        "call_end_kinds": {k: kinds[k] for k in CALL_END_KINDS.values()}, "kept_call_tools": dict(tools),
                        **({"chunk": m["chunk"], "row_start": m["row_start"]} if by_layer else {})})
        # the exclusions file's truncation is at original text; cleaned text is shorter, so it can be cut less often
        if cfg.get("check_truncation", True) and m["truncated"] != r["truncated"]:
            raise ValueError(f"{r['key']}: truncation differs between the extraction and {cfg['exclusions_file']}")
    bad = {k: v for k, v in checks.items() if v}
    if bad:
        raise ValueError(f"reader-input checks failed: { {k: (len(v), v[:3]) for k, v in bad.items()} }")
    arrays = {}
    for pos, rows in tables.items():
        a = np.array(rows, np.int64).reshape(-1, 7)
        if np.any(np.diff(a[:, 0]) < 0):
            raise ValueError(f"{pos} rows are not grouped by run")
        # kind: CALL_END_KINDS, -1 if not a call; tool: index into summary["tools"]
        for j, name in enumerate(["run", "row", "step", "action", "token", "kind", "tool"]):
            arrays[f"{pos}__{name}"] = a[:, j]
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "tables.npz", **arrays)
    (out / "runs.json").write_text(json.dumps([r | x for r, x in zip(runs, per_run)]))
    (out / "texts_call_end.json").write_text(json.dumps(texts))
    summary = {"data_id": data_id, "n_runs": len(runs), "rows": {pos: len(rows) for pos, rows in tables.items()},
               "runs_without_rows": {pos: int(len(runs) - len(np.unique(arrays[f"{pos}__run"]))) for pos in tables},
               "checks": {k: len(v) for k, v in checks.items()}, "n_honest_prompt_tasks": len(honest_prompts),
               "tools": sorted(tool_ids, key=tool_ids.get),
               "call_end_kind_share": {arm: {k: round(sum(x["call_end_kinds"][k] for r, x in zip(runs, per_run) if r["arm"] == arm)
                                                    / max(1, sum(c.values())), 4) for k in CALL_END_KINDS.values()}
                                       for arm, c in token_texts.items()},
               "call_end_top_tokens": {arm: [(t, round(n / sum(c.values()), 4)) for t, n in c.most_common(20)]
                                       for arm, c in token_texts.items()},
               "per_run": per_run}
    (out / "summary.json").write_text(json.dumps(summary))
    work_vol.commit()
    return summary


@app.function(image=image, volumes={"/acts": acts_vol, "/work": work_vol}, cpu=8, memory=65536, timeout=3600)
def pack_layer(cfg: dict, data_id: str, layer: int) -> dict:
    """L<layer>_<position>.npy: [rows, 5120] float16, in table order."""
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor

    work_vol.reload()
    d = data_dir(data_id)
    runs = json.loads((d / "runs.json").read_text())
    t = np.load(d / "tables.npz")
    targets = {pos: d / "packed" / f"L{layer}_{pos}.npy" for pos in cfg["positions"]}
    if all(p.exists() and np.load(p, mmap_mode="r").shape == (len(t[f"{pos}__run"]), H) for pos, p in targets.items()):
        return {"layer": layer, "status": "already_packed"}
    t0 = time.time()
    by_layer = cfg.get("layout") == "by_layer"
    ext = Path("/acts") / cfg["extraction_run"] / ("by_layer" if by_layer else "acts")
    feats = {pos: np.zeros((len(t[f"{pos}__run"]), H), np.float16) for pos in cfg["positions"]}
    bounds = {pos: np.searchsorted(t[f"{pos}__run"], np.arange(len(runs) + 1)) for pos in cfg["positions"]}
    blocks = {c: np.load(ext / f"L{layer}" / f"{c}.npy", mmap_mode="r") for c in {r["chunk"] for r in runs}} if by_layer else {}

    def load(i):
        if by_layer:  # a run's rows start at row_start in its chunk file
            x, off = blocks[runs[i]["chunk"]], runs[i]["row_start"]
        else:
            x, off = np.load(ext / runs[i]["arm"] / f"{runs[i]['filename']}.npz")[f"L{layer}"], 0
        for pos in cfg["positions"]:
            a, b = bounds[pos][i], bounds[pos][i + 1]
            feats[pos][a:b] = x[off + t[f"{pos}__row"][a:b]]

    with ThreadPoolExecutor(16) as pool:
        list(pool.map(load, range(len(runs))))
    nonfinite = {pos: int((~np.isfinite(x)).sum()) for pos, x in feats.items()}
    if any(nonfinite.values()):
        raise ValueError(f"L{layer}: non-finite activations {nonfinite}")
    (d / "packed").mkdir(exist_ok=True)
    for pos, x in feats.items():  # written whole, then renamed: a file under the final name is always complete
        tmp = targets[pos].with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            np.save(fh, x)
        os.replace(tmp, targets[pos])
    work_vol.commit()
    return {"layer": layer, "status": "packed", "seconds": round(time.time() - t0)}


@app.function(image=image, volumes={"/work": work_vol}, cpu=4, memory=16384, timeout=1800)
def compare_data(data_id: str, reference: str, layers: list[int], positions: list[str]) -> dict:
    """Do two packed datasets hold the same features? Row tables (all but the extraction-specific row column) and
    the sha256 of every packed matrix, so a new reader can be checked against the data a committed run used."""
    import numpy as np

    work_vol.reload()
    a, b = np.load(data_dir(data_id) / "tables.npz"), np.load(data_dir(reference) / "tables.npz")
    tables = {k: bool(np.array_equal(a[k], b[k])) for k in sorted(b.files) if not k.endswith("__row")}
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    packed = {f"L{l}_{pos}": sha(data_dir(data_id) / "packed" / f"L{l}_{pos}.npy") == sha(data_dir(reference) / "packed" / f"L{l}_{pos}.npy")
              for l in layers for pos in positions}
    return {"data_id": data_id, "reference": reference, "tables_equal": tables, "packed_equal": packed,
            "identical": all(tables.values()) and all(packed.values())}


def splits_of(runs: list[dict]) -> dict:
    """{split: (fold per run, test/validation pairs for the nested models)}: every ordered pair for main-task folds,
    the next environment for environment folds."""
    import numpy as np

    task = np.array([r["fold"] for r in runs])
    env = np.array([r["env_fold"] for r in runs])
    kt, ke = 5, 10
    return {"task": (task, kt, [(f, v) for f in range(kt) for v in range(kt) if v != f]),
            "env": (env, ke, [(f, (f + 1) % ke) for f in range(ke)])}


@app.function(image=image, gpu=GPU, cpu=8, memory=65536, volumes={"/work": work_vol}, timeout=2 * 3600)
def probe_layer(cfg: dict, data_id: str, run_id: str, layer: int) -> dict:
    import numpy as np
    import torch

    from probe_shift import torch_probe as tp
    from probe_shift.attack_eval import E1, HONEST, POOLINGS, pool, shuffle_within_task

    t0 = time.time()
    work_vol.reload()
    d = data_dir(data_id)
    runs = json.loads((d / "runs.json").read_text())
    t = np.load(d / "tables.npz")
    n = len(runs)
    arm = np.array([r["arm"] for r in runs])
    task = np.array([r["main_task"] for r in runs])
    y = np.where(arm == E1, 1.0, np.where(arm == HONEST, 0.0, np.nan))
    trainable = ~np.isnan(y)
    rng = np.random.default_rng(cfg["seed"])  # the same null draws at every layer
    y_null = np.stack([shuffle_within_task(y, task, trainable, rng) for _ in range(cfg["shuffle_null"])], 1)
    arrays, fits, base_X, probes = {}, [], {}, {}
    for pos, spec in cfg["readouts"].items():  # pos: the readout's name (call_end, call_end_content, output_end)
        base = spec["position"]
        if base not in base_X:  # readouts of one packed position are consecutive in the config: keep one on the GPU
            base_X.clear()
            torch.cuda.empty_cache()
            base_X[base] = torch.from_numpy(np.load(d / "packed" / f"L{layer}_{base}.npy")).cuda().float()
            if base_X[base].shape[0] != len(t[f"{base}__run"]):
                raise ValueError(f"L{layer} {base}: {base_X[base].shape[0]} packed rows, {len(t[f'{base}__run'])} in the table")
        sel = np.arange(len(t[f"{base}__run"])) if "kinds" not in spec else np.flatnonzero(np.isin(t[f"{base}__kind"], spec["kinds"]))
        r_run = t[f"{base}__run"][sel]
        X = base_X[base] if "kinds" not in spec else base_X[base][torch.from_numpy(sel).cuda()]
        arrays[f"{pos}__rows"] = sel  # rows of the base position's table this readout uses
        print(f"L{layer} {pos}: {X.shape[0]} rows at {time.time() - t0:.0f} s", flush=True)

        def fit_score(train_runs, score_runs, Y, name):
            """Fit on the calls of train_runs (Y: per-run labels [n, k]), score the calls of score_runs.
            Returns the run scores per pooling, [n] or [n, k] with NaN outside score_runs."""
            tr = np.flatnonzero(train_runs[r_run])
            te = np.flatnonzero(score_runs[r_run])
            yt = Y[r_run[tr]]
            if len(np.unique(yt)) < 2 or not len(te):
                fits.append({"pos": pos, "model": name, "skipped": "one class in training or nothing to score"})
                return None, None
            Xtr = X[torch.from_numpy(tr).cuda()]
            mean, std = Xtr.mean(0), Xtr.std(0, unbiased=False)  # tp.standardize's statistics, kept with the probe
            std = std.where(std > 0, std.new_ones(()))
            Xtr, Xte = (Xtr - mean) / std, (X[torch.from_numpy(te).cuda()] - mean) / std
            W, b, n_eval = tp.fit(Xtr, torch.from_numpy(yt).cuda().float(), cfg["fit"]["C"], cfg["fit"]["max_iter"])
            s = tp.decision(Xte, W, b).double().cpu().numpy()
            del Xtr, Xte
            if W.shape[1] == 1:  # the frozen probe: score = ((x - mean) / std) . w + b (null fits are not kept)
                probes[f"{pos}__{name}"] = {"w": W[:, 0].cpu().numpy(), "b": float(b[0]), "mean": mean.cpu().numpy(),
                                            "std": std.cpu().numpy()}
            fits.append({"pos": pos, "model": name, "train_rows": int(len(tr)), "scored_rows": int(len(te)),
                         # n_eval counts loss evaluations (line search included), so reaching max_iter flags a fit
                         # that may have stopped at the cap rather than at the tolerance
                         "n_eval": int(n_eval), "may_have_hit_cap": bool(n_eval >= cfg["fit"]["max_iter"])})
            s = s[:, 0] if s.shape[1] == 1 else s
            return {p: pool(r_run[te], s, n, p) for p in POOLINGS}, (te, s)

        for split, (fold, k, pairs) in splits_of(runs).items():
            outer = {p: np.full(n, np.nan) for p in POOLINGS}
            nested = {p: np.full((k, k, n) if split == "task" else (k, n), np.nan) for p in POOLINGS}
            step = np.full(len(r_run), np.nan)
            for f in range(k):
                rs, raw = fit_score(trainable & (fold != f), fold == f, y[:, None], f"{split}_outer_{f}")
                if rs is None:
                    continue
                for p in POOLINGS:
                    outer[p][fold == f] = rs[p][fold == f]
                step[raw[0]] = raw[1]
            for f, v in pairs:
                te = (fold == f) | (fold == v)
                rs, _ = fit_score(trainable & ~te, te, y[:, None], f"{split}_nested_{f}_{v}")
                if rs is None:
                    continue
                for p in POOLINGS:
                    (nested[p][f, v] if split == "task" else nested[p][f])[te] = rs[p][te]
            print(f"L{layer} {pos} {split}: {len(pairs) + k} fits done at {time.time() - t0:.0f} s", flush=True)
            for p in POOLINGS:
                arrays[f"{pos}__{p}__{split}_outer"] = outer[p].astype(np.float32)
                arrays[f"{pos}__{p}__{split}_nested"] = nested[p].astype(np.float32)
            if split == "task":
                arrays[f"{pos}__step__task_outer"] = step.astype(np.float32)
                null = {p: np.full((n, y_null.shape[1]), np.nan) for p in POOLINGS}
                for f in range(k):
                    rs, _ = fit_score(trainable & (fold != f), fold == f, y_null, f"task_null_{f}")
                    if rs is None:
                        continue
                    for p in POOLINGS:
                        null[p][fold == f] = rs[p][fold == f]
                for p in POOLINGS:  # per-run null scores: the analysis scores them on the same runs as the real probe
                    arrays[f"{pos}__{p}__task_null"] = null[p].astype(np.float32)
                print(f"L{layer} {pos}: shuffle null done at {time.time() - t0:.0f} s", flush=True)
        del X
        torch.cuda.empty_cache()
    arrays["null_labels"] = y_null.astype(np.float32)  # NaN for runs outside training (e2, sim)
    out = Path("/work/runs") / run_id
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"scores_L{layer}.npz", **arrays)
    np.savez(out / f"probes_L{layer}.npz", **{f"{k}__{f}": np.asarray(v) for k, x in probes.items() for f, v in x.items()})
    info = {"layer": layer, "seconds": round(time.time() - t0), "fits": fits}
    (out / f"fits_L{layer}.json").write_text(json.dumps(info))  # written last: marks the layer done
    work_vol.commit()
    return info




@app.function(image=image, volumes={"/work": work_vol}, cpu=4, memory=32768, timeout=3600)
def tfidf_model(cfg: dict, data_id: str, run_id: str, readout: str, split: str, f: int, v: int | None) -> dict:
    """One TF-IDF + logistic regression model on the call texts of a call_end readout (all calls, or content-token
    calls only): outer (v None) scores fold f; nested scores folds f and v."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    from probe_shift.attack_eval import E1, HONEST

    t0 = time.time()
    work_vol.reload()
    d = data_dir(data_id)
    runs = json.loads((d / "runs.json").read_text())
    t = np.load(d / "tables.npz")
    spec = cfg["readouts"][readout]
    if spec["position"] != "call_end":
        raise ValueError("TF-IDF reads call texts: call_end readouts only")
    sel = np.arange(len(t["call_end__run"])) if "kinds" not in spec else np.flatnonzero(np.isin(t["call_end__kind"], spec["kinds"]))
    r_run = t["call_end__run"][sel]
    texts = np.array(json.loads((d / "texts_call_end.json").read_text()), object)[sel]
    arm = np.array([r["arm"] for r in runs])
    y = np.where(arm == E1, 1.0, np.where(arm == HONEST, 0.0, np.nan))
    fold = splits_of(runs)[split][0]
    score_runs = (fold == f) | (fold == v) if v is not None else fold == f
    tr = np.flatnonzero((~np.isnan(y) & ~score_runs)[r_run])
    te = np.flatnonzero(score_runs[r_run])
    name = f"{readout}__{split}_{f}_{v}"
    out = Path("/work/runs") / run_id / "tfidf"
    out.mkdir(parents=True, exist_ok=True)
    info = {"readout": readout, "split": split, "f": f, "v": v}
    if len(np.unique(y[r_run[tr]])) < 2 or not len(te):
        info["skipped"] = "one class in training or nothing to score"
    else:
        c = cfg["tfidf"]
        vec = TfidfVectorizer(ngram_range=tuple(c["ngram_range"]), min_df=c["min_df"], max_features=c["max_features"],
                              sublinear_tf=c["sublinear_tf"], dtype=np.float32)
        clf = LogisticRegression(C=c["C"], max_iter=c["max_iter"]).fit(vec.fit_transform(texts[tr]), y[r_run[tr]])
        s = clf.decision_function(vec.transform(texts[te]))
        np.savez(out / f"{name}.npz", run=r_run[te], scores=s)  # run: each scored call's run, grouped by run
        info |= {"train_rows": int(len(tr)), "scored_rows": int(len(te)), "n_iter": int(clf.n_iter_[0]),
                 "hit_max_iter": bool(clf.n_iter_[0] >= c["max_iter"]), "vocabulary": len(vec.vocabulary_)}
    info["seconds"] = round(time.time() - t0)
    (out / f"{name}.json").write_text(json.dumps(info))  # written last: marks the model done
    work_vol.commit()
    return info


def tfidf_jobs(cfg: dict) -> list[tuple]:
    pairs = ([(s, f, None) for s, k in (("task", 5), ("env", 10)) for f in range(k)]
             + [("task", f, v) for f in range(5) for v in range(5) if v != f] + [("env", f, (f + 1) % 10) for f in range(10)])
    return [(readout, *p) for readout in cfg["tfidf_readouts"] for p in pairs]


def read_volume(path: str) -> bytes:
    return b"".join(work_vol.read_file(path))


def wait(call):
    for _ in range(120):  # our network: retry for up to an hour; remote errors are raised
        try:
            return call.get()
        except (socket.gaierror, ConnectionError, TimeoutError, modal.exception.ConnectionError,
                modal.exception.ServiceError) as e:
            print(f"lost the connection ({e!r}); retrying in 30 s", flush=True)
            time.sleep(30)
    raise TimeoutError("no connection to Modal for an hour; outputs are on the work Volume")


ARCHIVE_DATA_FILES = ("tables.npz", "runs.json", "summary.json", "texts_call_end.json")  # texts: the TF-IDF input


def archive(run_id: str, only: list[str] | None = None) -> dict:
    """Archive a run's artifacts to S3 from the laptop, with the user's own AWS session (profile redwood-sandbox).
    Personal credentials never go into a Modal container.

    Lists the run's files on the work Volume (runs/<run_id>/, plus its data's ARCHIVE_DATA_FILES), refuses over
    ARCHIVE_MAX_BYTES, fetches each exact file (never a directory root) into a temporary folder and checks its size,
    uploads that folder with `aws s3 cp --no-overwrite` (new keys only; an existing key is skipped, never overwritten),
    verifies every object's size and SHA-256, and deletes the folder. Fails loudly on any mismatch, including a skipped
    key whose content differs. `only`: archive just these paths under the run's prefix (e.g. data/texts_call_end.json),
    leaving other objects already there alone. Returns the receipt."""
    import base64
    import hashlib
    import tempfile

    import boto3

    from probe_shift.transcripts import AWS_PROFILE

    bucket, prefix = f"{S3_ARCHIVE}{run_id}/".removeprefix("s3://").split("/", 1)
    file_type = modal.volume.FileEntryType.FILE
    with tempfile.TemporaryDirectory(prefix=f"q3-archive-{run_id}-") as work:
        meta, tmp = Path(work) / "meta", Path(work) / "upload"  # tmp holds exactly what is uploaded

        def fetch(volume_path: str, size: int, local: Path):
            local.parent.mkdir(parents=True, exist_ok=True)
            with open(local, "wb") as fh:
                for chunk in work_vol.read_file(volume_path):  # one exact file
                    fh.write(chunk)
            if local.stat().st_size != size:
                raise ValueError(f"{volume_path}: fetched {local.stat().st_size} bytes, the Volume lists {size}")

        run_files = {e.path: e.size for e in work_vol.listdir(f"runs/{run_id}", recursive=True) if e.type == file_type}
        launch_path = f"runs/{run_id}/launch.json"
        if launch_path not in run_files:
            raise FileNotFoundError(f"{launch_path} is not on the work Volume")
        fetch(launch_path, run_files[launch_path], meta / "launch.json")
        data_id = json.loads((meta / "launch.json").read_text())["data_id"]
        data_files = {e.path: e.size for e in work_vol.listdir(f"data/{data_id}") if e.type == file_type
                      and e.path.rsplit("/", 1)[-1] in ARCHIVE_DATA_FILES}
        if len(data_files) != len(ARCHIVE_DATA_FILES):
            raise FileNotFoundError(f"data/{data_id}: expected {ARCHIVE_DATA_FILES}, found {sorted(data_files)}")
        # Volume path -> path under the archive prefix: the run's files as they are, its data's files under data/
        files = {p: p.removeprefix(f"runs/{run_id}/") for p in run_files} | {p: "data/" + p.rsplit("/", 1)[-1] for p in data_files}
        if only is not None:
            unknown = set(only) - set(files.values())
            if unknown:
                raise FileNotFoundError(f"{run_id}: not among the run's archivable files: {sorted(unknown)}")
            files = {p: rel for p, rel in files.items() if rel in only}
        elif not any(rel.startswith("probes_L") for rel in files.values()):
            raise ValueError(f"{run_id}: no trained probes on the work Volume")
        sizes = {p: (run_files | data_files)[p] for p in files}
        total = sum(sizes.values())
        if total > ARCHIVE_MAX_BYTES:
            raise ValueError(f"{run_id}: {total:,} bytes on the Volume, over the {ARCHIVE_MAX_BYTES:,}-byte archive limit")
        for p, rel in files.items():
            fetch(p, sizes[p], tmp / rel)
        upload = subprocess.run(["aws", "s3", "cp", str(tmp), f"s3://{bucket}/{prefix}", "--recursive", "--no-overwrite",
                                 "--checksum-algorithm", "SHA256", "--profile", AWS_PROFILE, "--region", S3_ARCHIVE_REGION,
                                 "--only-show-errors"], capture_output=True, text=True)
        if upload.returncode != 0:
            raise RuntimeError(f"aws s3 cp failed ({upload.returncode}): {upload.stderr[-2000:]}")

        s3 = boto3.Session(profile_name=AWS_PROFILE).client("s3", region_name=S3_ARCHIVE_REGION)

        def matches(rel: str) -> bool:
            """Size and SHA-256 of the S3 object against the local file. A multipart upload stores a composite
            SHA-256 (the hash of the parts' hashes, with "-<parts>"): recomputed here from the actual part size."""
            local, key = tmp / rel, prefix + rel
            head = s3.head_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
            stored = head.get("ChecksumSHA256")
            if head["ContentLength"] != local.stat().st_size or not stored:
                return False
            data = local.read_bytes()
            if "-" not in stored:
                return stored == base64.b64encode(hashlib.sha256(data).digest()).decode()
            n = int(stored.split("-")[1])
            part = s3.head_object(Bucket=bucket, Key=key, PartNumber=1)["ContentLength"]
            digests = b"".join(hashlib.sha256(data[i:i + part]).digest() for i in range(0, len(data), part))
            return len(data) <= part * n and stored == f"{base64.b64encode(hashlib.sha256(digests).digest()).decode()}-{n}"

        bad = [rel for rel in files.values() if not matches(rel)]
        objects = {rel: {"bytes": (tmp / rel).stat().st_size,
                         "sha256": hashlib.sha256((tmp / rel).read_bytes()).hexdigest()} for rel in files.values()}
        pages = s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
        listed = {o["Key"]: o["Size"] for page in pages for o in page.get("Contents", [])}
    missing = sorted(rel for rel in files.values() if prefix + rel not in listed)
    # A full archive must leave exactly the run's files under its prefix; a partial one (`only`) checks just its own.
    extra = sorted(set(listed) - {prefix + rel for rel in files.values()}) if only is None else []
    if bad or missing or extra:
        raise ValueError(f"S3 differs from the Volume under s3://{bucket}/{prefix}: mismatched {bad}, missing {missing}, "
                         f"unexpected {extra}")
    print(f"archived {run_id}: {len(files)} files, {total:,} bytes, every size and SHA-256 matching the Volume, to "
          f"s3://{bucket}/{prefix}", flush=True)
    return {"run_id": run_id, "prefix": f"s3://{bucket}/{prefix}", "only": only, "volume_files": len(files),
            "volume_bytes": total, "sha256_verified": len(files) - len(bad), "objects": objects,
            "s3_objects_under_prefix": len(listed), "s3_bytes_under_prefix": sum(listed.values()),
            "how": "laptop, aws s3 cp --no-overwrite with the user's own session",
            "archived": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds")}


def archive_only(run_id: str, only: list[str] | None = None):
    """Archive a finished run (or just the `only` paths) and write the receipt to a new file,
    results/archive_<run_id>__<stamp>.json. Append-only: the run record and earlier receipts are never rewritten."""
    if not (HERE / "results" / f"run_{run_id}.json").exists():
        raise FileNotFoundError(f"results/run_{run_id}.json: not a collected run")  # fail before copying anything
    done = archive(run_id, only)
    path = HERE / "results" / f"archive_{run_id}__{done['archived'][:19].replace(':', '')}.json"
    if path.exists():
        raise FileExistsError(f"{path} exists; results/ is append-only")
    path.write_text(json.dumps(done, indent=1))
    print(f"archive receipt -> {path}", flush=True)


def collect(launch: dict, launched_at: float):
    """Everything from the work Volume into results/: probe run scores per layer, TF-IDF call scores pooled to run
    scores in the probe's layout, the row tables, the runs with their counts, and the run record."""
    import numpy as np

    from probe_shift import attack_eval as ae

    cfg, run_id, data_id, runs = launch["config"], launch["run_id"], launch["data_id"], launch["runs"]
    out = HERE / "results" / f"run_scores_{run_id}"
    out.mkdir(parents=True, exist_ok=True)
    fits = {}
    for layer in cfg["layers"]:
        (out / f"scores_L{layer}.npz").write_bytes(read_volume(f"runs/{run_id}/scores_L{layer}.npz"))
        fits[layer] = json.loads(read_volume(f"runs/{run_id}/fits_L{layer}.json"))
    (out / "tables.npz").write_bytes(read_volume(f"data/{data_id}/tables.npz"))
    n = len(runs)
    folds = {"task": np.array([r["fold"] for r in runs]), "env": np.array([r["env_fold"] for r in runs])}
    tf, tfidf_info = {}, []
    for readout in cfg["tfidf_readouts"]:
        for p in ae.POOLINGS:
            tf |= {f"{readout}__{p}__task_outer": np.full(n, np.nan), f"{readout}__{p}__task_nested": np.full((5, 5, n), np.nan),
                   f"{readout}__{p}__env_outer": np.full(n, np.nan), f"{readout}__{p}__env_nested": np.full((10, n), np.nan)}
    for readout, split, f, v in tfidf_jobs(cfg):
        name = f"{readout}__{split}_{f}_{v}"
        info = json.loads(read_volume(f"runs/{run_id}/tfidf/{name}.json"))
        tfidf_info.append(info)
        if info.get("skipped"):
            continue
        z = np.load(io.BytesIO(read_volume(f"runs/{run_id}/tfidf/{name}.npz")))
        fold = folds[split]
        scored = (fold == f) | (fold == v) if v is not None else fold == f
        for p in ae.POOLINGS:
            rs = ae.pool(z["run"], z["scores"], n, p)
            if v is None:
                tf[f"{readout}__{p}__{split}_outer"][scored] = rs[scored]
            elif split == "task":
                tf[f"{readout}__{p}__task_nested"][f, v][scored] = rs[scored]
            else:
                tf[f"{readout}__{p}__env_nested"][f][scored] = rs[scored]
    np.savez(out / "scores_tfidf.npz", **{k: x.astype(np.float32) for k, x in tf.items()})
    with work_vol.batch_upload(force=True) as batch:
        batch.put_file(out / "scores_tfidf.npz", f"runs/{run_id}/scores_tfidf.npz")
    summary = dict(launch["manifest"])
    per_run = summary.pop("per_run")
    (out / "runs.json").write_text(json.dumps([r | x for r, x in zip(runs, per_run)]))
    gpu_s = sum(x["seconds"] for x in fits.values())
    record = {k: launch[k] for k in ("run_id", "git", "git_dirty", "test", "data_id", "launched", "config", "excluded", "pack")}
    record |= {"collected": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds"),
               "client_wall_seconds_this_launch": round(time.time() - launched_at),
               "modal": {"app": APP, "gpu": GPU, "activations_volume": ACTS, "work_volume": WORK,
                         "raw_scores": f"{WORK}/runs/{run_id}", "packed_features": f"{WORK}/data/{data_id}/packed",
                         # every non-null fit: w, b and the standardization mean and std, keyed readout__model
                         "trained_probes": f"{WORK}/runs/{run_id}/probes_L<layer>.npz"},
               # the durable copy of runs/<run_id>/ (scores, trained probes, fit logs) and the data's tables and runs:
               # its S3 prefix and the listing check against the Volume
               "archive": archive(run_id),
               "counts": {a: sum(r["arm"] == a for r in runs) for a in ae.ARMS}, "manifest": summary,
               "excluded_counts": dict(Counter(f"{e['arm']}|{e['reason']}" for e in launch["excluded"])),
               "probe_fits": fits, "tfidf_fits": tfidf_info, "gpu_seconds": gpu_s,
               "tfidf_cpu_seconds": sum(x["seconds"] for x in tfidf_info)}
    (HERE / "results" / f"run_{run_id}.json").write_text(json.dumps(record, indent=1))
    capped = sum(x.get("may_have_hit_cap", False) for l in fits.values() for x in l["fits"])
    print(f"collected {run_id}: GPU seconds {gpu_s}; {capped} probe fits may have hit the iteration cap; "
          f"{sum(bool(x.get('hit_max_iter')) for x in tfidf_info)} TF-IDF fits hit max_iter; scores -> {out}", flush=True)


@app.local_entrypoint()
def main(test: bool = False, resume: str = "", config: str = "q3_attack.yaml", pack_only: bool = False,
         check_data: str = ""):
    """`--test`: a few main tasks per fold, one layer, a small null, under a separate run id. `--resume <run_id>`:
    spawn only the fits whose outputs are not on the work Volume yet (a fit still running from the first launch is
    spawned again and overwrites the same files), then collect. A run already collected (results/run_<run_id>.json
    exists) is refused, since results/ is append-only. Archiving a finished run is plain Python, no Modal
    app: `python run_q3_attack.py --archive-only <run_id>`. `--config <file in config/>` picks the settings.
    `--pack-only --check-data <data_id>`: build and pack the data, compare it with an earlier run's data
    (compare_data), and stop before any fit."""
    import yaml

    from probe_shift import attack_eval as ae
    from probe_shift.transcripts import REACT_NUDGE

    launched_at = time.time()
    if resume and (pack_only or check_data):  # a resumed run is already packed: it would go straight to the fits
        raise ValueError("--pack-only and --check-data apply to a new launch, not to --resume")
    if resume:
        launch = json.loads(read_volume(f"runs/{resume}/launch.json"))
    else:
        cfg = yaml.safe_load((HERE / "config" / config).read_text())
        if (cfg["activations_volume"], cfg["work_volume"]) != (ACTS, WORK):
            raise ValueError("config volumes differ from the script's")
        if ae.NUDGE != REACT_NUDGE:
            raise ValueError("attack_eval.NUDGE is out of date with transcripts.REACT_NUDGE")
        folds = json.loads((REPO / "splits" / cfg["folds_file"]).read_text())
        ex = json.loads((REPO / "splits" / cfg["exclusions_file"]).read_text())
        runs, excluded = ae.select_runs(ex, folds)
        if test:
            keep = set()
            for f in range(5):
                keep |= set(sorted({r["main_task"] for r in runs if r["fold"] == f})[: cfg["test"]["tasks_per_fold"]])
            runs = [r for r in runs if r["main_task"] in keep]
            cfg |= {k: cfg["test"][k] for k in ("layers", "shuffle_null")}
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=REPO).stdout.strip())
        now = datetime.now(ZoneInfo("America/Los_Angeles"))
        run_id = now.strftime("%Y-%m-%d_%H%M%S") + ("_test" if test else "")
        ident = [MANIFEST_VERSION, cfg["extraction_run"], cfg["positions"],
                 [(r["arm"], r["key"], r["fold"], r["env_fold"]) for r in runs]]
        ident += [cfg["kinds_from"]] if cfg.get("kinds_from") else []  # other kinds, other tables
        data_id = hashlib.sha1(json.dumps(ident).encode()).hexdigest()[:12]
        print(f"run {run_id} | data {data_id} | {len(runs)} runs:", {a: sum(r["arm"] == a for r in runs) for a in ae.ARMS},
              f"| git {git[:8]}{' (dirty)' if dirty else ''}", flush=True)
        # spawn + wait, not .remote/.map: a laptop network blip (the first full launch died on a DNS error here) must
        # not kill the client and, with it, the app
        summary = wait(manifest.spawn(cfg, runs, data_id))
        print("manifest:", {k: v for k, v in summary.items() if k not in ("per_run", "call_end_top_tokens")}, flush=True)
        packs = [wait(c) for c in [pack_layer.spawn(cfg, data_id, layer) for layer in cfg["layers"]]]
        print("packed:", packs, flush=True)
        if check_data:
            same = wait(compare_data.spawn(data_id, check_data, cfg["layers"], cfg["positions"]))
            print("compared with", check_data, json.dumps(same), flush=True)
            if not same["identical"]:
                raise ValueError(f"data {data_id} differs from {check_data}")
        if pack_only:
            return
        launch = {"run_id": run_id, "git": git, "git_dirty": dirty, "test": test, "data_id": data_id,
                  "launched": now.isoformat(timespec="seconds"), "config": cfg, "runs": runs, "excluded": excluded,
                  "manifest": summary, "pack": packs}
        with work_vol.batch_upload() as batch:  # the launch record first, so an interrupted run can be resumed
            batch.put_file(io.BytesIO(json.dumps(launch).encode()), f"runs/{run_id}/launch.json")
    cfg, run_id, data_id = launch["config"], launch["run_id"], launch["data_id"]
    if (HERE / "results" / f"run_{run_id}.json").exists():  # written last by collect(): results/ is append-only
        raise FileExistsError(f"results/run_{run_id}.json exists: run {run_id} is already collected")
    try:
        done = {e.path.removeprefix(f"runs/{run_id}/") for e in work_vol.listdir(f"runs/{run_id}", recursive=True)}
    except (FileNotFoundError, modal.exception.NotFoundError):
        done = set()
    calls = [(f"L{layer}", probe_layer.spawn(cfg, data_id, run_id, layer)) for layer in cfg["layers"]
             if f"fits_L{layer}.json" not in done]
    calls += [(f"tfidf {'/'.join(map(str, j))}", tfidf_model.spawn(cfg, data_id, run_id, *j)) for j in tfidf_jobs(cfg)
              if f"tfidf/{j[0]}__{j[1]}_{j[2]}_{j[3]}.json" not in done]
    print(f"run {run_id}: spawned {len(calls)} jobs ({len(done)} files already on the Volume)", flush=True)
    failed = []
    for name, call in calls:
        try:
            res = wait(call)
        except Exception as e:  # keep waiting on the others; every failure is raised below, after the loop
            failed.append((name, repr(e)[:300]))
            print(f"{name}: FAILED {e!r}", flush=True)
            continue
        if name.startswith("L"):
            print(f"{name}: {res['seconds']} s, {len(res['fits'])} fits", flush=True)
    if failed:
        raise RuntimeError(f"{len(failed)} jobs failed: {failed}; fix, then rerun with --resume {run_id}")
    collect(launch, launched_at)


if __name__ == "__main__":  # plain Python, on the laptop: no Modal app, no container
    import argparse

    ap = argparse.ArgumentParser(description="Archive a finished run's artifacts to S3 with your own AWS session.")
    ap.add_argument("--archive-only", required=True, metavar="RUN_ID")
    ap.add_argument("--files", nargs="+", default=None, metavar="PATH",
                    help="archive just these paths under the run's prefix, e.g. data/texts_call_end.json")
    a = ap.parse_args()
    archive_only(a.archive_only, a.files)
