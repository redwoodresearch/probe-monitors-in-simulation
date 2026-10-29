"""Q1 on segment-end features, on GPU: the same analysis as analyze_q1_segment_end.py, with batched torch probes.

Per layer, on one H100 (`probe_shift.torch_probe`: scikit-learn's L2 logistic objective, many label sets at once):
  - probes at output_end (tool outputs) and call_end (agent tool calls), real vs sim and the e1-vs-e2 null, out-of-fold
    over the shared main-task folds; variants: all outputs, no editor outputs, no tool-error outputs;
  - the label-shuffle null (100 draws, one batched fit per fold; per main task, real and sim swap with probability 1/2);
  - the best of 1,000 random directions, picked on the training folds;
  - held-out environment (leave-one-environment-out);
  - inner CV on each outer fold's training folds, for choosing the headline layer without test folds;
  - a learning curve: 10, 30, 100, 300 and all training main tasks per fold, 3 seeds (baselines-q3's subsets file).
Then one CPU step picks each headline layer (per outer fold, the earliest layer within headline_tolerance of the best
inner-CV AUROC) and scores it. Every AUROC has a 95% interval from resampling main tasks. Results go to the Volume and
to results/q1_gpu_<stamp>.json.

  uv run modal run --detach experiments/2026-10-06_q1_segment_end/analyze_q1_gpu.py [--test]
"""

import json
import os
import re
import socket
import time
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q1-segment-end_gpu_kill-2026-11-06"
POSITION_BUCKETS = [("1", 1, 1), ("2-3", 2, 3), ("4-10", 4, 10), ("11-25", 11, 25), (">25", 26, 10**9)]
SHARED = "probe-shift_aswin-surya_sub-la-activations"  # the main session's Volume: extractions and staged inputs
# The Volume that holds `extraction_run` (and where results go). Modal binds a function's Volumes when this module is
# imported, before --config is read, so it comes from the environment; the image carries the same value so the
# containers agree, and check_volume() refuses a config that names another Volume.
ACTS = os.environ.get("PROBE_SHIFT_ACTS_VOLUME", SHARED)
GPU = "H100!"
TIMEOUT_S = 3 * 3600
# Per-layer fan-outs get one input per container, so a finished container has nothing left to run. Modal's default
# 60 s idle window was billed on every one: on 2026-10-07 a 64-layer analysis billed 3.56 H100-hours for 2.02 inside
# gpu_layer, the rest container start, load and idle.
SCALEDOWN_S = 5
SUBSETS = Path.home() / "probe-shift-baselines/experiments/2026-10-06_q1_text_baselines/results/tfidf_learning_curve_task_subsets_2026-10-06_133445.json"

app = modal.App(APP)
acts_vol = modal.Volume.from_name(ACTS, version=2)
# Staged reader messages of an extraction on another Volume (its config.json `inputs_from`), mounted read only.
VOLUMES = {"/acts": acts_vol} | ({} if ACTS == SHARED else {"/staged": modal.Volume.from_name(SHARED, version=2).read_only()})
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("torch==2.8.0", "numpy", "scikit-learn==1.7.2", "scipy")
         .env({"PROBE_SHIFT_ACTS_VOLUME": ACTS}).add_local_python_source("probe_shift"))


# Every YAML key the analysis reads (n_jobs: the CPU version, analyze_q1_segment_end.py). A key outside this set fails
# loudly, so a setting the code does not implement can never be silently ignored.
CONFIG_KEYS = {"extraction_run", "folds_file", "exclusions_file", "layers", "positions", "sensitivity_positions", "C", "max_iter",
               "bootstrap", "shuffle_null", "random_directions", "position_bands", "seed", "n_jobs", "headline_tolerance",
               "meta_dir", "by_layer", "lite", "inner_cv_environment", "call_content_only", "variants", "comparisons",
               "clean_cells", "design", "honest_real_runs", "layers_per_container", "activations_volume", "match_cut_file"}
VARIANTS = {"all_outputs", "no_editor_outputs", "no_tool_error_outputs", "no_insert_undo_outputs"}


def load_config(name: str) -> dict:
    """config/<name>, with unknown keys and unknown variant or comparison names rejected."""
    import yaml

    from probe_shift.q1_eval import DESIGNS

    cfg = yaml.safe_load((HERE / "config" / name).read_text())
    bad = set(cfg) - CONFIG_KEYS
    bad |= {f"design: {cfg['design']}" for _ in [0] if cfg.get("design", "attack") not in DESIGNS}
    bad |= {f"variants: {v}" for v in cfg.get("variants") or [] if v not in VARIANTS}
    bad |= {f"comparisons: {c}" for c in cfg.get("comparisons") or [] if c not in DESIGNS["attack"]["comparisons"]}
    k = cfg.get("layers_per_container", 1)
    bad |= {f"layers_per_container: {k!r}" for _ in [0] if not (isinstance(k, int) and not isinstance(k, bool) and k >= 1)}
    if bad:
        raise ValueError(f"config/{name}: settings this analysis does not implement: {sorted(bad)}")
    return cfg


def launch_provenance(cfg: dict, commit: str = "", config: str = "") -> dict:
    """Git hash, dirty flag and source sha of the code a launch runs: this script, probe_shift, the config's split files
    and the config itself if named (probe_shift.provenance; with `commit`, every file checked against it)."""
    from probe_shift.provenance import capture

    root = HERE.parent.parent
    return capture(root, [Path(__file__), *sorted((root / "src" / "probe_shift").glob("*.py")),
                          root / "splits" / cfg["folds_file"], root / "splits" / cfg["exclusions_file"]]
                   + ([HERE / "config" / config] if config else []), commit)


def design_cells(cfg: dict, ex: dict) -> list[str]:
    """The cells a design analyses, from splits' exclusions: paired attack cells, paired honest cells (main tasks), or both."""
    design = cfg.get("design", "attack")
    return ((ex["paired_attack_cells"]["cells"] if design in ("attack", "pooled") else [])
            + (ex["paired_honest_cells"]["cells"] if design in ("honest", "pooled") else []))


def set_slices(cfg: dict, ex: dict) -> None:
    """cfg's cells and its slices: the simulation reference (base200 / overlap800), no transcript over 32k (attack
    cells; honest cells have no such pairing yet), and the honest slices to carry through every analysis."""
    cfg["paired_cells"] = design_cells(cfg, ex)
    ref = ex["paired_attack_cells"]["sim_reference"] | {c: "overlap800" for c in ex.get("paired_honest_cells", {}).get("cells", [])}
    cfg["base200"] = [c for c in cfg["paired_cells"] if ref[c] == "base200"]
    cfg["overlap800"] = [c for c in cfg["paired_cells"] if ref[c] == "overlap800"]
    cfg["under_32k"] = [c for c in ex["paired_attack_cells_under_32k"]["cells"] if c in set(cfg["paired_cells"])]
    if cfg.get("design", "attack") != "attack":
        h = ex["paired_honest_cells"]
        keys = {}
        for r in ex["samples"]:
            if r["arm"] == "real-honest" and not r["reason"] and r["main_task"] in set(h["cells"]):
                keys.setdefault(r["main_task"], []).append(r["key"])
        cfg["honest_real_keys"] = {c: sorted(k) for c, k in keys.items()}  # find_runs requires every one to be read
        stamped = set(h["slices"]["contract_stamped_locally"])
        cfg["extra_slices"] = {"honest_contract_stamped_locally": sorted(stamped),
                               "honest_not_stamped": [c for c in h["cells"] if c not in stamped],
                               "honest_without_redrawn": [c for c in h["cells"] if c not in set(h["slices"]["redrawn_after_model_calls"])]}
        if cfg["design"] == "pooled":
            cfg["extra_slices"] |= {"attack_cells": ex["paired_attack_cells"]["cells"], "honest_cells": h["cells"]}


def fixed_tokens(cfg: dict) -> list[str]:
    """The call_end fixed-slot tokens an analysis used: recorded in its config from amendment 3 on; earlier analyses
    used the original two (a timeout's final 0, an escaped newline's n)."""
    return cfg.get("fixed_call_tokens", ["0", "n"])


def wait(call):
    """A spawned call's result, retrying our network for up to an hour (a laptop DNS blip must not kill a run);
    remote errors are raised."""
    for _ in range(120):
        try:
            return call.get()
        except (socket.gaierror, ConnectionError, TimeoutError, modal.exception.ConnectionError, modal.exception.ServiceError) as e:
            print(f"lost the connection ({e!r}); retrying in 30 s", flush=True)
            time.sleep(30)
    raise TimeoutError("no connection to Modal for an hour; per-layer results are on the Volume")

@app.function(image=image, volumes=VOLUMES, cpu=8, memory=16384, timeout=1800)
def editor_commands(cfg: dict) -> dict:
    """{"key|step|action": command} for every text_editor call, from the extraction's staged reader messages."""
    out = {}
    for shard in sorted(staged_inputs(cfg).glob("shard_*.jsonl")):
        for line in shard.read_text().splitlines():
            it = json.loads(line)
            step = -1
            for m in it["messages"]:
                if m["role"] == "assistant":
                    step += 1
                    for a, c in enumerate(m.get("tool_calls") or []):
                        if c["function"]["name"] == "text_editor":
                            out[f"{it['key']}|{step}|{a}"] = str((c["function"]["arguments"] or {}).get("command"))
    return out


@app.function(image=image, volumes=VOLUMES, cpu=8, memory=16384, timeout=1800)
def insert_undo_cells(cfg: dict) -> list[str]:
    """Cells whose real or simulated run (the design's real_vs_sim arms; for honest cells, any real honest run) calls
    text_editor insert or undo_edit anywhere (staged reader messages)."""
    from probe_shift.q1_eval import arm_dir, comparisons

    out = set()
    for shard in sorted(staged_inputs(cfg).glob("shard_*.jsonl")):
        for line in shard.read_text().splitlines():
            it = json.loads(line)
            if it["arm"] in {arm_dir(a) for a in sum(comparisons(cfg)["real_vs_sim"], ())} and any(
                    c["function"]["name"] == "text_editor" and str((c["function"]["arguments"] or {}).get("command")) in ("insert", "undo_edit")
                    for m in it["messages"] if m["role"] == "assistant" for c in m.get("tool_calls") or []):
                out.add(it["cell"])
    return sorted(out)


@app.function(image=image, volumes=VOLUMES, cpu=8, memory=16384, timeout=1800)
def manifest(cfg: dict) -> dict:
    from probe_shift.q1_eval import find_runs

    return find_runs(cfg)


def packed_dir(cfg: dict, man: dict) -> Path:
    """Packed features for this manifest: keyed by its content, so a changed set of runs never reuses another's cache."""
    import hashlib

    digest = hashlib.sha1(json.dumps(json.loads(json.dumps(man)), sort_keys=True).encode()).hexdigest()[:12]
    return Path("/acts") / f"{cfg['extraction_run']}__q1gpu" / "packed" / digest


def staged_inputs(cfg: dict) -> Path:
    """Where the extraction's staged reader messages live: its own inputs/, or for an all-layers run, its source run's
    (on the same Volume, or on the shared one mounted at /staged)."""
    run_dir = Path("/acts") / cfg["extraction_run"]
    if (run_dir / "inputs").exists():
        return run_dir / "inputs"
    source = json.loads((run_dir / "config.json").read_text())["inputs_from"]
    for root in (Path("/acts"), Path("/staged")):
        if (root / source / "inputs").exists():
            return root / source / "inputs"
    raise FileNotFoundError(f"no staged inputs for {source} on {ACTS} or {SHARED}")


def check_volume(cfg: dict):
    """Refuse to run a config whose extraction lives on another Volume than the one this module mounted."""
    want = cfg.get("activations_volume") or (cfg.get("modal") or {}).get("activations_volume") or SHARED
    if want != ACTS:
        raise ValueError(f"{cfg['extraction_run']} is on Volume {want}, but {ACTS} is mounted: rerun with PROBE_SHIFT_ACTS_VOLUME={want}")


@app.function(image=image, volumes=VOLUMES, cpu=16, memory=65536, timeout=TIMEOUT_S, scaledown_window=SCALEDOWN_S)
def pack_layer(cfg: dict, layer: int, man: dict) -> str:
    """One float16 matrix per position kind for this layer, rows in step_table order over the manifest's runs: a few
    large reads for the GPU containers instead of thousands of small ones (GPU containers read the Volume slowly)."""
    import numpy as np

    from probe_shift.q1_eval import arm_dir, load_runs, step_table

    run_dir = Path("/acts") / cfg["extraction_run"]
    kinds = cfg["positions"] + ([] if cfg.get("lite") else cfg["sensitivity_positions"])
    out = packed_dir(cfg, man)
    # One manifest per layer, written by that layer's packer only: a file other containers write can look empty here.
    mf = out / f"manifest.L{layer}.json"
    try:
        done = mf.exists() and json.loads(mf.read_text()) == json.loads(json.dumps(man)) and all((out / f"L{layer}_{k}.npy").exists() for k in kinds)
    except json.JSONDecodeError:  # mid-write by a preempted earlier attempt of this layer: pack again
        done = False
    if done:
        return str(out)  # already packed from this manifest (a rerun after a stuck container)
    runs = load_runs(cfg, man)
    tables = {k: step_table(runs, k) for k in kinds}
    feats = {k: np.zeros((len(t["run"]), 5120), np.float16) for k, t in tables.items()}
    starts = {k: np.searchsorted(t["run"], np.arange(len(runs) + 1)) for k, t in tables.items()}  # rows are grouped by run
    chunks = {}
    for i, m in enumerate(runs):
        if cfg.get("by_layer"):  # all-layers runs: one file per layer and chunk; rows offset by the transcript's start
            if m["chunk"] not in chunks:
                chunks[m["chunk"]] = np.load(run_dir / "by_layer" / f"L{layer}" / f"{m['chunk']}.npy", mmap_mode="r")
            arr, off = chunks[m["chunk"]], m["row_start"]
        else:
            arr, off = np.load(run_dir / "acts" / arm_dir(m["arm"]) / f"{m['filename']}.npz")[f"L{layer}"], 0
        for k, t in tables.items():
            a, b = starts[k][i], starts[k][i + 1]
            feats[k][a:b] = arr[off + t["row"][a:b]]
    bad = {k: int((~np.isfinite(x)).sum()) for k, x in feats.items()}
    if any(bad.values()):  # a bad dump fails here instead of being fit
        raise ValueError(f"L{layer}: non-finite activations per position kind {bad}")
    out.mkdir(parents=True, exist_ok=True)
    tag = uuid.uuid4().hex  # temp names unique to this call: two jobs may pack the same layer at once
    for k, x in feats.items():  # atomic: a reader sees a whole file or none (packers run in parallel and can die)
        with open(out / f".L{layer}_{k}.{tag}.tmp", "wb") as fh:
            np.save(fh, x)
        os.replace(out / f".L{layer}_{k}.{tag}.tmp", out / f"L{layer}_{k}.npy")
    (out / f".manifest.{tag}.tmp").write_text(json.dumps(man))
    os.replace(out / f".manifest.{tag}.tmp", out / f"manifest.L{layer}.json")
    acts_vol.commit()
    return str(out)


@app.function(image=image, gpu=GPU, cpu=8, memory=98304, volumes=VOLUMES, timeout=TIMEOUT_S,
              scaledown_window=SCALEDOWN_S)
def gpu_layer(cfg: dict, layer: int, man: dict, subsets: dict, tool_errors: dict, commands: dict | None = None) -> dict:
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score

    from probe_shift import torch_probe as tp
    from probe_shift.q1_eval import EDITOR, FIXED_CALL_TOKENS, Scorer, comparisons, load_runs, step_table

    t0 = time.time()
    acts_vol.reload()
    run_dir = Path("/acts") / cfg["extraction_run"]
    runs = load_runs(cfg, man)
    lite = cfg.get("lite", False)  # all-layers sweep: probe, shuffle null, held-out environment; all and no editor
    kinds = cfg["positions"] + ([] if lite else cfg["sensitivity_positions"])
    tables = {k: step_table(runs, k) for k in kinds}
    src = packed_dir(cfg, man)
    if json.loads((src / f"manifest.L{layer}.json").read_text()) != json.loads(json.dumps(man)):  # tuples arrive, lists are stored
        raise ValueError("packed features were built from a different manifest")
    X = {}
    for k, t in tables.items():
        x = np.load(src / f"L{layer}_{k}.npy")
        if len(x) != len(t["run"]):
            raise ValueError(f"packed L{layer} {k}: {len(x)} rows, expected {len(t['run'])}")
        X[k] = torch.from_numpy(x).cuda().float()
    print(f"L{layer}: loaded in {time.time() - t0:.0f} s", flush=True)
    fold_of = {"task": np.array([m["fold"] for m in runs]),
               "environment": np.array([cfg["environment_to_fold"][m["environment"]] for m in runs])}
    main_of = np.array([m["main_task"] for m in runs], object)
    env = np.array([m["environment"] for m in runs], object)
    keys = [m["key"] for m in runs]
    commands = commands or {}
    variants = cfg.get("variants") or (["all_outputs", "no_editor_outputs"] + (["no_tool_error_outputs"] if tool_errors and not lite else [])
                                       + (["no_insert_undo_outputs"] if commands and not lite else []))
    INSERT_UNDO = {"insert", "undo_edit"}  # text_editor commands whose output format differs by harness (baselines-q3)

    def keep(pos, variant):
        t = tables[pos]
        # call_content_only: drop calls whose read token is a fixed slot (a timeout's final 0, an escaped newline's n)
        base = ~np.isin(t["token_text"], fixed_tokens(cfg)) if pos == "call_end" and cfg.get("call_content_only") else True
        if variant == "no_editor_outputs":
            return base & (t["tool"] != EDITOR)
        if variant == "no_tool_error_outputs":  # drop actions whose tool output was an error (the call too, for call_end)
            return base & np.array([not tool_errors.get(f"{keys[r]}|{s}|{a}") for r, s, a in zip(t["run"], t["step"], t["action"])])
        if variant == "no_insert_undo_outputs":  # drop text_editor insert and undo_edit actions
            return base & np.array([commands.get(f"{keys[r]}|{s}|{a}") not in INSERT_UNDO for r, s, a in zip(t["run"], t["step"], t["action"])])
        return base & np.ones(len(t["run"]), bool)

    def inner_cv(pos, comparison, variant, Y, fold_by):
        """Per outer fold f, the pooled run AUROC over its training folds g, each g scored by a probe trained without f
        and g. One fit per unordered pair {f, g} scores both g (for outer f) and f (for outer g): the same fits, once."""
        t = tables[pos]
        use, fold = use_mask(pos, comparison, variant), fold_of[fold_by][t["run"]]
        fs = sorted(set(fold[use]))
        parts = {f: ([], []) for f in fs}
        for i, f in enumerate(fs):
            for g in fs[i + 1:]:
                te_f, te_g = use & (fold == f), use & (fold == g)
                rows = np.flatnonzero(te_f | te_g)
                sc = fit_score(pos, Y, use & (fold != f) & (fold != g), te_f | te_g)[0][:, 0]
                for outer, te in ((f, te_g), (g, te_f)):
                    m = te[rows]
                    r = t["run"][rows[m]]
                    tot, cnt = np.bincount(r, weights=sc[m], minlength=len(runs)), np.bincount(r, minlength=len(runs))
                    kk = np.flatnonzero(cnt > 0)
                    parts[outer][0].append(tot[kk] / cnt[kk])
                    parts[outer][1].append(Y[kk, 0])
        return {int(f): round(float(roc_auc_score(np.concatenate(y), np.concatenate(sc_))), 4) for f, (sc_, y) in parts.items()
                if sc_ and len(set(np.concatenate(y))) == 2}  # an outer fold with no scored pair, or one class, gets none

    def by_command(t, s, scorer):
        """Step-level AUROC per text_editor command (None where one arm has no such step)."""
        cmd = np.array([commands.get(f"{keys[r]}|{st}|{a}", "") for r, st, a in zip(t["run"], t["step"], t["action"])])
        ok = scorer.runs_in[t["run"]] & ~np.isnan(s)
        out = {}
        for c in ("view", "create", "str_replace", "insert", "undo_edit"):
            m = ok & (cmd == c)
            r = scorer.auc_ci(scorer.y_run[t["run"]][m], s[m], scorer.main[t["run"]][m]) if m.sum() else None
            out[c] = (r or {"auroc": None}) | {"n": int(m.sum())}
        return out

    gen = torch.Generator(device="cuda").manual_seed(cfg["seed"] + 1000 + layer)
    dirs = torch.randn(cfg["random_directions"], 5120, device="cuda", generator=gen)
    dirs /= dirs.norm(dim=1, keepdim=True)

    def col_auc(P, y):
        """AUROC of each column of P [n, k] against labels y [n], with average ranks for ties (identical prefixes)."""
        from scipy.stats import rankdata

        ranks = rankdata(P.cpu().numpy(), axis=0)
        yy = y.bool().cpu().numpy()
        n1 = float(yy.sum())
        return torch.from_numpy((ranks[yy].sum(0) - n1 * (n1 + 1) / 2) / (n1 * (len(yy) - n1))).float().cuda()

    runs_in_of = {c: Scorer(cfg, runs, c).runs_in for c in comparisons(cfg)}

    def use_mask(pos, comparison, variant):
        return runs_in_of[comparison][tables[pos]["run"]] & keep(pos, variant)  # the comparison's runs, cells with both sides

    def fit_score(pos, Y, tr, te, random_dirs=False):
        """Fit on step rows tr (standardized on them), score rows te: ([n_te, k] scores, iterations, random-dir scores)."""
        t = tables[pos]
        Xtr, Xte = tp.standardize(X[pos][torch.from_numpy(np.flatnonzero(tr)).cuda()],
                                  X[pos][torch.from_numpy(np.flatnonzero(te)).cuda()])
        Ytr = torch.from_numpy(Y[t["run"][tr]]).cuda()
        W, b, n_iter = tp.fit(Xtr, Ytr, cfg["C"], cfg["max_iter"])
        rand = None
        if random_dirs:  # the direction farthest from 0.5 on the training rows, signed
            auc = col_auc(Xtr @ dirs.T, Ytr[:, 0])
            j = int((auc - 0.5).abs().argmax())
            rand = (torch.sign(auc[j] - 0.5) * (Xte @ dirs[j])).cpu().numpy()
        return tp.decision(Xte, W, b).cpu().numpy(), n_iter, rand

    def oof(pos, comparison, variant, Y, fold_by="task", train_tasks=None, random_dirs=False):
        """Out-of-fold scores [n_steps, k] for label sets Y [n_runs, k]; optionally the best random direction's.
        train_tasks: {fold: main tasks allowed in training} (learning curve)."""
        t = tables[pos]
        use = use_mask(pos, comparison, variant)
        fold = fold_of[fold_by][t["run"]]
        out = np.full((len(t["run"]), Y.shape[1]), np.nan, np.float32)
        rand = np.full(len(t["run"]), np.nan, np.float32) if random_dirs else None
        iters = []
        for f in sorted(set(fold[use])):
            tr = use & (fold != f)
            if train_tasks is not None:
                tr &= np.isin(main_of[t["run"]], list(train_tasks[int(f)]))
            te = use & (fold == f)
            sc, n_iter, r = fit_score(pos, Y, tr, te, random_dirs)
            out[te] = sc
            iters.append(n_iter)
            if random_dirs:
                rand[te] = r
        return out, max(iters), rand

    def by_position(t, s, scorer):
        """Step-level AUROC by the step's ordinal among its run's segments of this kind (1st, 2nd-3rd, ...)."""
        ordinal = np.zeros(len(t["run"]), int)
        starts = np.searchsorted(t["run"], np.arange(len(runs) + 1))
        for i in range(len(runs)):
            ordinal[starts[i]:starts[i + 1]] = np.arange(1, starts[i + 1] - starts[i] + 1)
        ok = scorer.runs_in[t["run"]] & ~np.isnan(s)
        y = scorer.y_run[t["run"]]
        out = {}
        for name, lo, hi in POSITION_BUCKETS:
            m = ok & (ordinal >= lo) & (ordinal <= hi)
            out[name] = scorer.auc_ci(y[m], s[m], scorer.main[t["run"][m]])
        return out

    def y_of(comparison):
        return np.array([[float(m["arm"] in comparisons(cfg)[comparison][1])] for m in runs], np.float32)

    result = {"layer": layer, "n_runs": len(runs), "variants": variants, "results": {}, "sensitivity": {}, "inner_cv": {},
              "learning_curve": {}, "run_scores": {}}
    step_scores = {}  # (position, comparison, variant) -> out-of-fold score per step, saved to the Volume
    for pos in cfg["positions"]:
        t = tables[pos]
        result["results"][pos], result["inner_cv"][pos] = {}, {}
        for comparison in cfg.get("comparisons") or comparisons(cfg):  # clean_cells: real_vs_sim only
            scorer = Scorer(cfg, runs, comparison)
            Y = y_of(comparison)
            res, inner = {}, {}
            for variant in variants:
                s, n_iter, rand = oof(pos, comparison, variant, Y, random_dirs=not lite)
                s = s[:, 0]
                step_scores[(pos, comparison, variant)] = s
                if commands and variant == "all_outputs":  # step-level AUROC per text_editor command
                    res["by_editor_command"] = by_command(t, s, scorer)
                res[variant] = scorer.report(t, s, cfg) | {"max_iter_used": n_iter, "by_output_position": by_position(t, s, scorer),
                                                           "best_random_direction": None if lite else scorer.report(t, rand, cfg, slices=False)}
                ok = ~np.isnan(s)
                rs = scorer.run_scores(t["run"][ok], s[ok])
                result["run_scores"][f"{pos}__{comparison}__{variant}"] = [None if np.isnan(x) else round(float(x), 5) for x in rs]
                print(f"L{layer} {pos} {comparison} {variant}: run {res[variant]['run']} step {res[variant]['step']}", flush=True)
                # Inner CV per outer fold, on its training folds only (for the headline layer): train on the folds
                # other than f and g, score g, pool g's run scores, AUROC per f.
                inner[variant] = {} if lite else inner_cv(pos, comparison, variant, Y, "task")
            # Held-out environment, every variant (TF-IDF without editor outputs drops sharply here).
            res["held_out_environment"], inner_env = {}, {}
            for variant in variants:
                s, _, _ = oof(pos, comparison, variant, Y, fold_by="environment")
                s = s[:, 0]
                step_scores[(pos, comparison, f"{variant}__held_out_environment")] = s
                held = scorer.report(t, s, cfg, slices=False) | {"by_output_position": by_position(t, s, scorer)}
                if commands and variant == "all_outputs":
                    held["by_editor_command"] = by_command(t, s, scorer)
                ok = ~np.isnan(s)
                rs = scorer.run_scores(t["run"][ok], s[ok])
                result["run_scores"][f"{pos}__{comparison}__{variant}__held_out_environment"] = [
                    None if np.isnan(x) else round(float(x), 5) for x in rs]
                if cfg.get("inner_cv_environment") and not lite:  # headline layer for split A (protocol_level7.md)
                    inner_env[variant] = inner_cv(pos, comparison, variant, Y, "environment")
                held["per_environment"] = {}
                for e in sorted(cfg["environment_to_fold"]):
                    r = np.flatnonzero(scorer.runs_in & (env == e) & ~np.isnan(rs))
                    held["per_environment"][e] = {"run": scorer.auc_ci(scorer.y_run[r], rs[r], scorer.main[r]), "n_runs": int(len(r))}
                res["held_out_environment"][variant] = held
            if comparison == "real_vs_sim":
                # Label-shuffle null: one batched fit per fold for all draws.
                rng = np.random.default_rng(cfg["seed"] + 1)
                tasks = sorted({runs[i]["main_task"] for i in np.flatnonzero(scorer.runs_in)})
                draws = np.zeros((len(runs), cfg["shuffle_null"]), np.float32)
                for d in range(cfg["shuffle_null"]):
                    flip = dict(zip(tasks, rng.random(len(tasks)) < 0.5))
                    draws[:, d] = [1 - y if (scorer.runs_in[i] and flip[runs[i]["main_task"]]) else y for i, y in enumerate(scorer.y_run)]
                for variant in variants:
                    S, _, _ = oof(pos, comparison, variant, draws)
                    null_run, null_step = [], []
                    for d in range(S.shape[1]):
                        s = S[:, d]
                        ok = ~np.isnan(s)
                        rs = scorer.run_scores(t["run"][ok], s[ok])
                        r = np.flatnonzero(scorer.runs_in & ~np.isnan(rs))
                        null_run.append(roc_auc_score(draws[r, d], rs[r]))
                        null_step.append(roc_auc_score(draws[t["run"][ok], d], s[ok]))
                    for level, null in (("run", np.array(null_run)), ("step", np.array(null_step))):
                        obs = res[variant][level]["auroc"]
                        res[variant][level]["label_shuffle_null"] = {
                            "n": len(null), "mean": round(float(null.mean()), 4), "p95": round(float(np.percentile(null, 95)), 4),
                            "p99": round(float(np.percentile(null, 99)), 4),
                            "p_value": round(float((1 + (null >= obs).sum()) / (1 + len(null))), 4)}
                    print(f"L{layer} {pos} shuffle null {variant}: p95 {res[variant]['run']['label_shuffle_null']['p95']}", flush=True)
                # Learning curve on baselines-q3's main-task subsets (same tasks as its TF-IDF curve).
                lc = {}
                for variant in (("all_outputs", "no_editor_outputs") if subsets["folds"] and not lite else ()):
                    lc[variant] = {}
                    first = next(iter(subsets["folds"].values()))["subsets"]
                    for n in first:  # "10", "30", "100", "300", "all"
                        lc[variant][n] = {}
                        for seed in first[n]:
                            train_tasks = {int(f): set(v["subsets"][n][seed]) for f, v in subsets["folds"].items()}
                            s, _, _ = oof(pos, comparison, variant, Y, train_tasks=train_tasks)
                            s = s[:, 0]
                            rep = scorer.report(t, s, cfg, slices=False)
                            lc[variant][n][seed] = {"run": rep["run"], "step": rep["step"]}
                result["learning_curve"][pos] = lc
            result["results"][pos][comparison] = res
            result["inner_cv"][pos][comparison] = inner
            result.setdefault("inner_cv_environment", {}).setdefault(pos, {})[comparison] = inner_env
    scorer = Scorer(cfg, runs, "real_vs_sim")
    for kind in ([] if lite else cfg["sensitivity_positions"]):
        s, _, _ = oof(kind, "real_vs_sim", "all_outputs", y_of("real_vs_sim"))
        result["sensitivity"][kind] = scorer.report(tables[kind], s[:, 0], cfg, slices=False)
    result["runs"] = [{k: m[k] for k in ("arm", "key", "cell", "main_task", "fold", "environment", "sim_reference", "truncated")}
                      for m in runs]
    result["seconds"] = round(time.time() - t0)
    out_dir = run_dir.parent / f"{cfg['extraction_run']}__q1gpu" / cfg["analysis_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for pos in cfg["positions"]:
        t = tables[pos]
        np.savez(out_dir / f"L{layer}_step_scores_{pos}.npz", run=t["run"], step=t["step"], action=t["action"], token=t["token"],
                 **{f"{c}__{v}": sc for (p_, c, v), sc in step_scores.items() if p_ == pos})
    (out_dir / f"L{layer}.json").write_text(json.dumps(result))
    acts_vol.commit()
    return result


@app.function(image=image, gpu=GPU, cpu=8, memory=98304, volumes=VOLUMES, timeout=24 * 3600,
              scaledown_window=SCALEDOWN_S)  # the group's layers run back to back under one deadline: Modal's maximum
def gpu_layers(cfg: dict, layers: list[int], man: dict, subsets: dict, tool_errors: dict, commands: dict | None = None) -> list[dict]:
    """gpu_layer for several layers in one container (config `layers_per_container`), one after another."""
    return [gpu_layer.local(cfg, layer, man, subsets, tool_errors, commands) for layer in layers]


@app.function(image=image, cpu=8, memory=32768, volumes=VOLUMES, timeout=TIMEOUT_S)
def headline(cfg: dict, man: dict, per_layer: dict, scheme: str = "task") -> dict:
    """Per probe, comparison and variant: in each outer fold, the earliest layer within headline_tolerance of the best
    inner-CV AUROC (training folds only); its out-of-fold run scores score that fold. scheme "task": main-task folds;
    "environment": held-out environment, with inner CV over the training environments."""
    import numpy as np

    from probe_shift.q1_eval import COMPARISONS, Scorer, load_runs

    runs = load_runs(cfg, man)
    fold_of = [m["fold"] if scheme == "task" else cfg["environment_to_fold"][m["environment"]] for m in runs]
    inner_key, suffix = ("inner_cv", "") if scheme == "task" else ("inner_cv_environment", "__held_out_environment")
    out = {}
    for pos in cfg["positions"]:
        out[pos] = {}
        for comparison in per_layer[str(cfg["layers"][0])]["results"][pos]:
            scorer = Scorer(cfg, runs, comparison)
            out[pos][comparison] = {}
            for variant in per_layer[str(cfg["layers"][0])]["variants"]:
                chosen, rs = {}, np.full(len(runs), np.nan)
                for f in sorted(set(fold_of)):
                    inner = {}
                    for l in cfg["layers"]:
                        by_fold = per_layer[str(l)][inner_key][pos][comparison][variant]
                        if f in by_fold or str(f) in by_fold:
                            inner[l] = by_fold[f] if f in by_fold else by_fold[str(f)]
                    if not inner:  # no inner-CV score for this fold (e.g. no kept steps there): its runs stay unscored
                        continue
                    top = max(inner.values())
                    chosen[f] = min(l for l, a in inner.items() if a >= top - cfg["headline_tolerance"])
                    scores = per_layer[str(chosen[f])]["run_scores"][f"{pos}__{comparison}__{variant}{suffix}"]
                    for i in range(len(runs)):
                        if fold_of[i] == f and scores[i] is not None:
                            rs[i] = scores[i]
                r = np.flatnonzero(scorer.runs_in & ~np.isnan(rs))
                out[pos][comparison][variant] = {"chosen_layer_by_fold": {str(k): v for k, v in chosen.items()},
                                                 "run": scorer.auc_ci(scorer.y_run[r], rs[r], scorer.main[r]),
                                                 "pairwise": scorer.pairwise(rs),
                                                 "run_scores": [None if np.isnan(x) else round(float(x), 5) for x in rs]}
    return out


@app.local_entrypoint()
def main(test: bool = False, tool_errors: str = "", subsets: str = str(SUBSETS), config: str = "q1_segment_end.yaml",
         editor_variants: bool = False, remote: bool = False, commit: str = ""):
    """`--test`: one layer, 5 shuffles, 100 bootstraps and directions. `--tool-errors`: the per-output error table
    (results/tool_errors_*.jsonl) for the no-tool-error variant. `--subsets`: the learning-curve main-task subsets
    (baselines-q3's file, so both curves use identical tasks; "" skips the curve)."""
    import yaml

    cfg = load_config(config)  # q1_segment_end_scrub5.yaml: the scrubbed extraction
    check_volume(cfg)
    # Recorded at launch, with the results and run_config.json (`--commit`: the commit holding this code, checked).
    cfg["provenance"] = launch_provenance(cfg, commit, config)

    if cfg["layers"] == "all":
        cfg["layers"] = list(range(64))
    if test:
        cfg |= {"layers": [24], "shuffle_null": 5, "bootstrap": 100, "random_directions": 100}
    splits = HERE.parent.parent / "splits"
    folds = json.loads((splits / cfg["folds_file"]).read_text())
    cfg["main_task_to_fold"] = folds["task_folds"]["main_task_to_fold"]
    cfg["environment_to_fold"] = folds["environment_folds"]["environment_to_fold"]
    ex = json.loads((splits / cfg["exclusions_file"]).read_text())
    set_slices(cfg, ex)
    if cfg.get("clean_cells"):  # only cells where neither the real e1 nor the simulated run calls insert or undo_edit
        dirty = set(wait(insert_undo_cells.spawn(cfg)))
        cfg["paired_cells"] = [c for c in cfg["paired_cells"] if c not in dirty]
        cfg["clean_cell_list"] = cfg["paired_cells"]
        print(f"clean cells: {len(cfg['paired_cells'])} ({len(dirty)} with insert/undo_edit dropped)", flush=True)
    # The config's name in the id, so two analyses launched in the same second never share a results file.
    if cfg.get("call_content_only"):
        from probe_shift.q1_eval import fixed_call_tokens

        cfg["fixed_call_tokens"] = fixed_call_tokens(cfg["extraction_run"])  # recorded, so follow-ups use the same list
    stamp = (datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S") + f"_{Path(config).stem}"
             + ("_test" if test else ""))
    cfg["analysis_id"] = stamp
    if (cfg.get("clean_cells") or cfg.get("design", "attack") != "attack") and subsets:
        print("learning curve skipped: baselines-q3's subsets come from the attack cells; their sizes would be wrong here", flush=True)
        subsets = ""
    subsets_file = subsets
    subsets = json.loads(Path(subsets).read_text()) if subsets else {"folds": {}}
    errors = {}
    if tool_errors:
        for line in Path(tool_errors).read_text().splitlines():
            r = json.loads(line)
            if r["error"]:
                errors[f"{r['key']}|{r['step']}|{r['action']}"] = r["error"]
    man = wait(manifest.spawn(cfg))
    if man["unreadable"] or man["cells_missing_an_arm"]:
        raise ValueError(f"incomplete extraction: {len(man['unreadable'])} unreadable, {len(man['cells_missing_an_arm'])} cells missing an arm")
    print(f"{len(man['runs'])} runs in {len(cfg['paired_cells']) - len(man['cells_missing_an_arm'])} cells; {len(errors)} errored tool outputs; analysis {stamp}", flush=True)
    with acts_vol.batch_upload(force=True) as batch:  # so `assemble` recovers these settings, not the current YAML
        import io

        batch.put_file(io.BytesIO(json.dumps({"config": cfg, "manifest": man, "subsets_file": subsets_file,
                                              "tool_errors_file": tool_errors, "editor_variants": editor_variants}).encode()),
                       f"{cfg['extraction_run']}__q1gpu/{stamp}/run_config.json")
    if remote:  # the rest runs on Modal (launch with --detach): no laptop client has to stay up; `fetch` gets the results
        call = orchestrate.spawn(cfg, man, subsets, errors, editor_variants, subsets_file, tool_errors)
        print(f"orchestrating on Modal: function call {call.object_id}; analysis {stamp}; then: fetch --analysis-id {stamp} "
              f"--config {config}", flush=True)
        return
    result, scores_jsonl = run_analysis(cfg, man, subsets, errors, editor_variants, subsets_file, tool_errors, wait)
    head = result["headline"]
    (HERE / "results").mkdir(exist_ok=True)
    path = HERE / "results" / f"q1_gpu_{stamp}.json"
    path.write_text(json.dumps(result, indent=1))
    (HERE / "results" / f"q1_gpu_run_scores_{stamp}.jsonl").write_text(scores_jsonl)
    fmt = lambda d: "n/a" if not d else f"{d['auroc']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]"
    for pos, by_c in (head or {}).items():
        for c, by_v in by_c.items():
            for v, h in by_v.items():
                print(f"HEADLINE {pos} {c} {v}: run {fmt(h['run'])}; layer by fold {h['chosen_layer_by_fold']}")
    print(f"results -> {path}")


def run_analysis(cfg, man, subsets, errors, editor_variants, subsets_file, tool_errors, get) -> tuple[dict, str]:
    """Pack, fit every layer, pick the headlines; returns the results dict and the run-scores JSONL. `get` waits on a
    spawned call (with network retries on the laptop; plainly on Modal)."""
    [get(c) for c in [pack_layer.spawn(cfg, l, man) for l in cfg["layers"]]]
    commands = get(editor_commands.spawn(cfg)) if editor_variants else {}
    k = cfg.get("layers_per_container", 1)
    if k == 1:
        calls = {l: gpu_layer.spawn(cfg, l, man, subsets, errors, commands) for l in cfg["layers"]}
        per_layer = {str(l): get(c) for l, c in calls.items()}
    else:  # several layers per GPU container, run one after another: start-up and imports are paid once per group
        groups = [cfg["layers"][i:i + k] for i in range(0, len(cfg["layers"]), k)]
        calls = [(g, gpu_layers.spawn(cfg, g, man, subsets, errors, commands)) for g in groups]
        per_layer = {str(l): r for g, c in calls for l, r in zip(g, get(c), strict=True)}
    head = None if cfg.get("lite") else get(headline.spawn(cfg, man, per_layer))
    head_env = get(headline.spawn(cfg, man, per_layer, "environment")) if cfg.get("inner_cv_environment") and not cfg.get("lite") else None
    cfg_out = {k: v for k, v in cfg.items() if k not in ("main_task_to_fold", "paired_cells", "base200", "overlap800", "under_32k")}
    result = {"manifest": man, "headline": head, "headline_held_out_environment": head_env,
              "by_layer": {l: {k: v for k, v in r.items() if k not in ("run_scores", "runs")} for l, r in per_layer.items()},
              "config": cfg_out | {"subsets_file": subsets_file, "tool_errors_file": tool_errors,
                                   "modal": {"app": APP, "gpu": GPU, "activations_volume": ACTS}}}
    first = per_layer[str(cfg["layers"][0])]
    scores = "".join(json.dumps(m | {f"L{l}__{k}": per_layer[l]["run_scores"][k][i] for l in per_layer
                                     for k in per_layer[l]["run_scores"]}) + "\n" for i, m in enumerate(first["runs"]))
    return result, scores


@app.function(image=image, cpu=2, memory=32768, volumes=VOLUMES, timeout=24 * 3600)
def orchestrate(cfg, man, subsets, errors, editor_variants, subsets_file, tool_errors) -> str:
    """main's analysis, run on Modal; writes results.json and run_scores.jsonl next to the per-layer results."""
    result, scores = run_analysis(cfg, man, subsets, errors, editor_variants, subsets_file, tool_errors, lambda c: c.get())
    out = Path("/acts") / f"{cfg['extraction_run']}__q1gpu" / cfg["analysis_id"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, indent=1))
    (out / "run_scores.jsonl").write_text(scores)
    acts_vol.commit()
    return str(out)


@app.local_entrypoint()
def fetch(analysis_id: str, config: str):
    """Copy an orchestrated analysis's results from the Volume to results/q1_gpu_<analysis_id>.json (and run scores)."""
    cfg = load_config(config)
    check_volume(cfg)
    run = cfg["extraction_run"]
    path = HERE / "results" / f"q1_gpu_{analysis_id}.json"
    for name, local in (("results.json", path), ("run_scores.jsonl", HERE / "results" / f"q1_gpu_run_scores_{analysis_id}.jsonl")):
        if local.exists():  # append-only: an earlier fetch wrote it; fetch only what is missing
            continue
        tmp = local.with_name(f".{local.name}.tmp")
        tmp.write_bytes(b"".join(acts_vol.read_file(f"{run}__q1gpu/{analysis_id}/{name}")))
        os.replace(tmp, local)
    print(f"results -> {path}")


# ---- Does the probe see anything TF-IDF doesn't? --------------------------------------------------------------------
NOTE_COMBINATION = ("The no-fit combinations (mean z-score, mean rank) are the leak-free headline. The fitted ones are "
                    "approximate: the training folds' out-of-fold scores come from base models that saw the held-out "
                    "fold, so the two learned weights see a little of it; nested base scores would remove that.")
# Per output and per run, at the given layers: within each training fold, a 1- or 2-feature logistic regression on
# [TF-IDF], [probe] and [both] (fit on the other folds' out-of-fold scores), scored on the held-out fold; plus a no-fit
# check (the mean of the two scores' z-scores and of their ranks). TF-IDF is baselines-q3's step-0 cleaned-text model
# (the text the probe read), joined on (arm, key, step, action); row counts must match exactly.


@app.function(image=image, cpu=16, memory=65536, volumes=VOLUMES, timeout=TIMEOUT_S)
def combination(cfg: dict, man: dict, analysis_id: str, layers: list[int], tfidf: dict, intersect: bool = False) -> dict:
    import numpy as np
    from scipy.stats import rankdata
    from sklearn.linear_model import LogisticRegression

    from probe_shift.q1_eval import Scorer, comparisons, load_runs

    runs = load_runs(cfg, man)
    key_of = [m["key"] for m in runs]
    base = Path("/acts") / f"{cfg['extraction_run']}__q1gpu" / analysis_id
    out = {}
    for layer in layers:
        z = np.load(base / f"L{layer}_step_scores_output_end.npz")
        out[str(layer)] = {}
        for comparison in cfg.get("comparisons") or comparisons(cfg):  # the comparisons that were fitted
            scorer = Scorer(cfg, runs, comparison)
            out[str(layer)][comparison] = {}
            for variant, tf_outputs in (("all_outputs", "all"), ("no_editor_outputs", "no_editor")):
                probe = z[f"{comparison}__{variant}"]
                ok = ~np.isnan(probe)
                tf = tfidf[f"{comparison}|{tf_outputs}"]
                tf_score = np.array([tf.get(f"{key_of[r]}|{s}|{a}", np.nan) for r, s, a in zip(z["run"], z["step"], z["action"])])
                n_tf = len(tf)
                join = {"probe_rows": int(ok.sum()), "tfidf_rows": n_tf, "matched": int((ok & ~np.isnan(tf_score)).sum())}
                if intersect:  # different text versions keep slightly different outputs: score the rows both have
                    ok &= ~np.isnan(tf_score)
                elif ok.sum() != n_tf or np.isnan(tf_score[ok]).any():
                    raise ValueError(f"L{layer} {comparison} {variant}: {ok.sum()} probe rows vs {n_tf} TF-IDF rows, "
                                     f"{int(np.isnan(tf_score[ok]).sum())} unmatched")
                run, step_y = z["run"][ok], scorer.y_run[z["run"][ok]]
                fold = scorer.fold[run]
                feats = {"tfidf": tf_score[ok][:, None], "probe": probe[ok][:, None],
                         "both": np.c_[tf_score[ok], probe[ok]]}
                scores = {}
                for name, F in feats.items():
                    s = np.full(len(run), np.nan)
                    for f in np.unique(fold):
                        tr, te = fold != f, fold == f
                        mu, sd = F[tr].mean(0), F[tr].std(0)
                        clf = LogisticRegression(C=1e6, max_iter=1000).fit((F[tr] - mu) / sd, step_y[tr])
                        s[te] = clf.decision_function((F[te] - mu) / sd)
                    scores[name] = s
                # No-fit combinations, normalized within each held-out fold on its own scores (label-free, and no
                # other fold's predictions enter).
                mean_z, mean_rank = np.full(len(run), np.nan), np.full(len(run), np.nan)
                for f in np.unique(fold):
                    te = fold == f
                    a, b = tf_score[ok][te], probe[ok][te]
                    mean_z[te] = (a - a.mean()) / a.std() + (b - b.mean()) / b.std()
                    mean_rank[te] = (rankdata(a) + rankdata(b)) / te.sum()
                scores["mean_z (no fit)"], scores["mean_rank (no fit)"] = mean_z, mean_rank
                tables = {"run": run, "step": z["step"][ok], "action": z["action"][ok], "token": z["token"][ok],
                          "tool": np.array([""] * len(run), object)}
                res = {name: scorer.report(tables, s, cfg, slices=False) for name, s in scores.items()}
                ordinal = np.zeros(len(run), int)  # the output's ordinal in its run, among kept outputs
                starts = np.searchsorted(run, np.arange(len(runs) + 1))
                for i in range(len(runs)):
                    ordinal[starts[i]:starts[i + 1]] = np.arange(1, starts[i + 1] - starts[i] + 1)
                res["by_output_position"] = {
                    name: {b: scorer.auc_ci(step_y[(ordinal >= lo) & (ordinal <= hi)], scores[name][(ordinal >= lo) & (ordinal <= hi)],
                                            scorer.main[run][(ordinal >= lo) & (ordinal <= hi)]) for b, lo, hi in POSITION_BUCKETS}
                    for name in ("tfidf", "probe")}
                # Gain from adding the probe to TF-IDF, with the same main-task resamples for both (paired).
                main = scorer.main[run]
                uniq, inv = np.unique(main, return_inverse=True)
                rs = {k: scorer.run_scores(run, v) for k, v in scores.items()}
                rr = np.flatnonzero(scorer.runs_in & ~np.isnan(rs["both"]))
                from sklearn.metrics import roc_auc_score
                rng = np.random.default_rng(cfg["seed"])
                run_task = np.unique(scorer.main[rr], return_inverse=True)[1]
                gains = {"step": [], "run": []}
                for _ in range(cfg["bootstrap"]):
                    b = rng.choice(len(uniq), len(uniq))
                    idx = np.concatenate([np.flatnonzero(inv == t) for t in b])
                    if len(np.unique(step_y[idx])) == 2:
                        gains["step"].append(roc_auc_score(step_y[idx], scores["both"][idx]) - roc_auc_score(step_y[idx], scores["tfidf"][idx]))
                    br = rng.choice(run_task.max() + 1, run_task.max() + 1)
                    ridx = rr[np.concatenate([np.flatnonzero(run_task == t) for t in br])]
                    if len(np.unique(scorer.y_run[ridx])) == 2:
                        gains["run"].append(roc_auc_score(scorer.y_run[ridx], rs["both"][ridx]) - roc_auc_score(scorer.y_run[ridx], rs["tfidf"][ridx]))
                res["gain_both_minus_tfidf"] = {
                    lvl: {"point": round(float(res["both"][lvl]["auroc"] - res["tfidf"][lvl]["auroc"]), 4),
                          "ci95": [round(float(x), 4) for x in np.percentile(gains[lvl], [2.5, 97.5])]} for lvl in ("step", "run")}
                out[str(layer)][comparison][variant] = res | {"join": join}
    return out


def comparisons_of(cfg: dict) -> dict:
    from probe_shift.q1_eval import comparisons

    return comparisons(cfg)


@app.local_entrypoint()
def combine(results: str, tfidf_scores: str, label: str, layers: str = "", intersect: bool = False):
    """The combination test for a finished GPU analysis (results/q1_gpu_*.json) against one TF-IDF score file, stored
    under --label (e.g. "output_only", "up_to_here"); --layers defaults to the headline layer chosen most often plus
    8 and 56."""
    import gzip
    from collections import Counter

    path = Path(results)
    res = json.loads(path.read_text())
    cfg, man = res["config"], res["manifest"]
    check_volume(cfg)
    splits = HERE.parent.parent / "splits"
    cfg["main_task_to_fold"] = json.loads((splits / cfg["folds_file"]).read_text())["task_folds"]["main_task_to_fold"]
    cfg["paired_cells"] = design_cells(cfg, json.loads((splits / cfg["exclusions_file"]).read_text()))
    if layers == "all":
        use_layers = list(range(64)) if cfg["layers"] == "all" else list(cfg["layers"])
    elif layers:
        use_layers = [int(x) for x in layers.split(",")]
    elif res["headline"] is None:
        raise ValueError("lite results have no headline layer; pass --layers")
    else:
        chosen = Counter(res["headline"]["output_end"]["real_vs_sim"]["all_outputs"]["chosen_layer_by_fold"].values())
        use_layers = sorted({chosen.most_common(1)[0][0], 8, 56})
    tf = {}
    with gzip.open(tfidf_scores, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            if r["arm"] in sum(comparisons_of(cfg)[r["comparison"]], ()):  # its models also score other arms
                tf.setdefault(f"{r['comparison']}|{r['outputs']}", {})[f"{r['key']}|{r['step']}|{r['action']}"] = r["score"]
    out = {}  # one container per layer (a 64-layer sweep in one container took hours)
    for r in [wait(c) for c in [combination.spawn(cfg, man, cfg["analysis_id"], [l], tf, intersect) for l in use_layers]]:
        out |= r
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out_path = HERE / "results" / f"q1_gpu_combination_{label}_{stamp}.json"
    out_path.write_text(json.dumps({"source": str(path), "label": label, "layers": use_layers, "tfidf_scores": tfidf_scores,
                                    "note": NOTE_COMBINATION, "results": out}, indent=1))
    path = out_path
    for l, by_c in out.items():
        for c, by_v in by_c.items():
            for v, r in by_v.items():
                g = r["gain_both_minus_tfidf"]
                print(f"L{l} {c} {v}: run tfidf {r['tfidf']['run']['auroc']} probe {r['probe']['run']['auroc']} both "
                      f"{r['both']['run']['auroc']} gain {g['run']}; step gain {g['step']}")
    print(f"combination added to {path}")


# ---- Is the late-layer jump the trailing-newline tell, read through the model's next-token belief? -----------------
# Simulated outputs end in a newline far more often than real ones (94% vs 60%), and the output_end token sits just
# before that newline. Re-score the saved out-of-fold probe scores on outputs that end in a newline in both arms, and on
# those that don't: if the separation survives within each group, it is not the newline.


@app.function(image=image, cpu=8, memory=32768, volumes=VOLUMES, timeout=TIMEOUT_S)
def newline_check(cfg: dict, man: dict, analysis_id: str, layers: list[int]) -> dict:
    import numpy as np

    from probe_shift.q1_eval import COMPARISONS, Scorer, load_runs

    runs = load_runs(cfg, man)
    run_dir = Path("/acts") / cfg["extraction_run"]
    ends_nl = {}  # (key, step, action) -> the tool output (as the reader saw it) ends in a newline
    for shard in sorted(staged_inputs(cfg).glob("shard_*.jsonl")):
        for line in shard.read_text().splitlines():
            it = json.loads(line)
            step, action = -1, 0
            for m in it["messages"]:
                if m["role"] == "assistant":
                    step, action = step + 1, 0
                elif m["role"] == "tool":
                    ends_nl[(it["key"], step, action)] = m["content"].endswith("\n")
                    action += 1
    base = Path("/acts") / f"{cfg['extraction_run']}__q1gpu" / analysis_id
    out = {}
    for layer in layers:
        z = np.load(base / f"L{layer}_step_scores_output_end.npz")
        flag = np.array([ends_nl[(runs[r]["key"], s, a)] for r, s, a in zip(z["run"], z["step"], z["action"])])
        out[str(layer)] = {}
        for comparison in COMPARISONS:
            scorer = Scorer(cfg, runs, comparison)
            out[str(layer)][comparison] = {}
            for variant in ("all_outputs", "no_editor_outputs"):
                s = z[f"{comparison}__{variant}"]
                res = {}
                for name, keep in (("ends_in_newline", flag), ("no_trailing_newline", ~flag)):
                    s_k = np.where(keep, s, np.nan)
                    t = {"run": z["run"], "token": z["token"]}
                    ok = scorer.runs_in[z["run"]] & ~np.isnan(s_k)
                    share = {arm: round(float(flag[scorer.runs_in[z["run"]] & ~np.isnan(s) & (np.array([runs[r]["arm"] for r in z["run"]]) == arm)].mean()), 3)
                             for arm in sum(COMPARISONS[comparison], ())}
                    res[name] = scorer.report(t, s_k, cfg, slices=False) | {"n_steps": int(ok.sum())}
                res["share_ending_in_newline"] = share
                out[str(layer)][comparison][variant] = res
    return out


@app.local_entrypoint()
def newline(results: str, layers: str = "8,24,40,48,56,63"):
    """The newline check for a finished GPU analysis; writes results/q1_gpu_newline_<stamp>.json."""
    res = json.loads(Path(results).read_text())
    cfg, man = res["config"], res["manifest"]
    check_volume(cfg)
    if cfg.get("design", "attack") != "attack":
        raise ValueError("this diagnostic pairs real e1 with simulated attack runs: attack design only")
    splits = HERE.parent.parent / "splits"
    cfg["main_task_to_fold"] = json.loads((splits / cfg["folds_file"]).read_text())["task_folds"]["main_task_to_fold"]
    cfg["paired_cells"] = design_cells(cfg, json.loads((splits / cfg["exclusions_file"]).read_text()))
    out = newline_check.remote(cfg, man, cfg["analysis_id"], [int(x) for x in layers.split(",")])
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"q1_gpu_newline_{stamp}.json"
    path.write_text(json.dumps({"source": results, "results": out}, indent=1))
    for l, by_c in out.items():
        r = by_c["real_vs_sim"]
        for v in ("all_outputs", "no_editor_outputs"):
            a, b = r[v]["ends_in_newline"], r[v]["no_trailing_newline"]
            print(f"L{l} {v}: ends-in-newline run {a['run']['auroc']} step {a['step']['auroc']} | no-newline run "
                  f"{b['run']['auroc'] if b['run'] else None} step {b['step']['auroc'] if b['step'] else None} | share {r[v]['share_ending_in_newline']}")
    print(f"-> {path}")


@app.function(image=image, cpu=8, memory=32768, volumes=VOLUMES, timeout=TIMEOUT_S)
def headline_from_volume(cfg: dict, man: dict, analysis_id: str, layers: list[int]) -> dict:
    """The headline selection over a subset of layers, from the per-layer results saved on the Volume."""
    base = Path("/acts") / f"{cfg['extraction_run']}__q1gpu" / analysis_id
    per_layer = {str(l): json.loads((base / f"L{l}.json").read_text()) for l in layers}
    return headline.local(cfg | {"layers": layers}, man, per_layer)


@app.local_entrypoint()
def rehead(results: str, max_layer: int = 48):
    """Headline restricted to layers <= max_layer (e.g. while late layers are suspect); writes a separate file."""
    res = json.loads(Path(results).read_text())
    cfg, man = res["config"], res["manifest"]
    check_volume(cfg)
    splits = HERE.parent.parent / "splits"
    cfg["main_task_to_fold"] = json.loads((splits / cfg["folds_file"]).read_text())["task_folds"]["main_task_to_fold"]
    cfg["paired_cells"] = design_cells(cfg, json.loads((splits / cfg["exclusions_file"]).read_text()))
    layers = [l for l in cfg["layers"] if l <= max_layer]
    head = headline_from_volume.remote(cfg, man, cfg["analysis_id"], layers)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"q1_gpu_headline_max{max_layer}_{stamp}.json"
    path.write_text(json.dumps({"source": results, "layers": layers, "headline": head}, indent=1))
    for pos, by_c in head.items():
        for c, by_v in by_c.items():
            for v, h in by_v.items():
                print(f"HEADLINE<= L{max_layer} {pos} {c} {v}: run {h['run']['auroc']} {h['run']['ci95']}; layers {h['chosen_layer_by_fold']}")
    print(f"-> {path}")


# ---- What does L63 read? Logit lens at output_end ---------------------------------------------------------------------
# L63 is the last decoder layer, so Qwen's final norm + lm_head on its output_end activation is Qwen's actual next-token
# distribution there. Compare real vs sim (outputs without the editor) on: the top next tokens; the probability that the
# output closes now (a newline-led token, "</...", or a template token) vs continues; and next-token entropy. Each as a
# 1-feature score (AUROC, oriented on training folds). Also L63's own probe scores split by truncated outputs (Inspect's
# <END_TOOL_OUTPUT> marker, which the simulator also writes) and long outputs (6,000+ characters).
lens_image = (modal.Image.debian_slim(python_version="3.12")
              .uv_pip_install("torch==2.8.0", "numpy", "scikit-learn==1.7.2", "scipy", "transformers==4.56.2", "safetensors")
              .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})
              .add_local_python_source("probe_shift"))
hf_cache = modal.Volume.from_name("probe-shift_aswin-surya_hf-cache")


@app.function(image=lens_image, gpu=GPU, cpu=8, memory=98304, volumes=VOLUMES | {"/hf": hf_cache}, timeout=TIMEOUT_S)
def logit_lens(cfg: dict, man: dict, analysis_id: str, layer: int = 63) -> dict:
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoTokenizer

    from probe_shift.models import QWEN3_32B
    from probe_shift.q1_eval import Scorer, load_runs, oriented, step_table

    runs = load_runs(cfg, man)
    t = step_table(runs, "output_end")
    X = np.load(packed_dir(cfg, man) / f"L{layer}_output_end.npy")
    assert len(X) == len(t["run"])
    reader = json.loads((Path("/acts") / cfg["extraction_run"] / "config.json").read_text())["model"]
    if reader != QWEN3_32B["name"]:
        raise ValueError(f"the logit lens implements {QWEN3_32B['name']}'s norm and head, not {reader}'s")
    # Final norm and lm_head weights only, from the cached checkpoint at the pinned revision.
    name, rev = QWEN3_32B["name"], QWEN3_32B["revision"]
    index = json.loads(Path(hf_hub_download(name, "model.safetensors.index.json", revision=rev)).read_text())["weight_map"]
    weights = {}
    for k in ("model.norm.weight", "lm_head.weight"):
        with safe_open(hf_hub_download(name, index[k], revision=rev), framework="pt", device="cuda") as fh:
            weights[k] = fh.get_tensor(k)
    tok = AutoTokenizer.from_pretrained(name, revision=rev)
    vocab = weights["lm_head.weight"].shape[0]
    texts = [tok.decode([i]) for i in range(len(tok))] + [""] * (vocab - len(tok))
    close = torch.tensor([s.startswith("\n") or s.startswith("</") or (s.startswith("<|") and s.endswith("|>")) for s in texts],
                         device="cuda")
    # Content flags per output, from the staged reader messages.
    flags = {}
    for shard in sorted(staged_inputs(cfg).glob("shard_*.jsonl")):
        for line in shard.read_text().splitlines():
            it = json.loads(line)
            step, action = -1, 0
            for m in it["messages"]:
                if m["role"] == "assistant":
                    step, action = step + 1, 0
                elif m["role"] == "tool":
                    flags[(it["key"], step, action)] = ("END_TOOL_OUTPUT" in m["content"], len(m["content"]) >= 6000)
                    action += 1
    keys = [m["key"] for m in runs]
    trunc = np.array([flags[(keys[r], s, a)][0] for r, s, a in zip(t["run"], t["step"], t["action"])])
    long_ = np.array([flags[(keys[r], s, a)][1] for r, s, a in zip(t["run"], t["step"], t["action"])])
    scorer = Scorer(cfg, runs, "real_vs_sim")
    use = scorer.runs_in[t["run"]] & (t["tool"] != "text_editor")
    idx = np.flatnonzero(use)
    p_close, entropy = np.full(len(t["run"]), np.nan), np.full(len(t["run"]), np.nan)
    top = {0.0: torch.zeros(vocab, device="cuda"), 1.0: torch.zeros(vocab, device="cuda")}
    counts = {0.0: 0, 1.0: 0}
    y = scorer.y_run[t["run"]]
    eps = weights["model.norm.weight"].new_tensor(1e-6, dtype=torch.float32)
    with torch.inference_mode():
        for chunk in np.array_split(idx, max(1, len(idx) // 2048)):
            h = torch.from_numpy(X[chunk]).cuda().float()
            h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * weights["model.norm.weight"].float()  # Qwen3 RMSNorm
            prob = (h.to(weights["lm_head.weight"].dtype) @ weights["lm_head.weight"].T).float().softmax(-1)
            p_close[chunk] = prob[:, close].sum(-1).cpu().numpy()
            entropy[chunk] = (-(prob * prob.clamp_min(1e-12).log()).sum(-1)).cpu().numpy()
            for lab in (0.0, 1.0):
                sel = torch.from_numpy(y[chunk] == lab).cuda()
                top[lab] += prob[sel].sum(0)
                counts[lab] += int(sel.sum())
    out = {"layer": layer, "n_outputs": {"real": counts[0.0], "sim": counts[1.0]},
           "close_token_rule": "decoded token starts with a newline, '</', or is a <|...|> template token"}
    for lab, arm in ((0.0, "real"), (1.0, "sim")):
        mean = top[lab] / counts[lab]
        v, i = mean.topk(10)
        out[f"top10_{arm}"] = [[texts[j], round(float(p), 4)] for j, p in zip(i.tolist(), v.tolist())]
        m = use & (y == lab)
        out[f"mean_p_close_{arm}"] = round(float(p_close[m].mean()), 4)
        out[f"mean_entropy_{arm}"] = round(float(entropy[m].mean()), 4)
    fold = scorer.fold[t["run"]]
    for name_, x in (("p_close", p_close), ("entropy", entropy)):
        s = oriented(np.where(use, x, np.nan), y, fold, use)
        out[f"auroc_{name_}"] = scorer.report(t, s, cfg, slices=False)
    # L63's own out-of-fold probe scores (no editor), split by truncated and long outputs.
    z = np.load(Path("/acts") / f"{cfg['extraction_run']}__q1gpu" / analysis_id / f"L{layer}_step_scores_output_end.npz")
    s = z["real_vs_sim__no_editor_outputs"]
    for name_, keep in (("truncated_marker", trunc), ("not_truncated", ~trunc), ("long_6000_chars", long_), ("short", ~long_)):
        sk = np.where(keep, s, np.nan)
        out[f"probe_L{layer}_no_editor_{name_}"] = scorer.report(t, sk, cfg, slices=False) | {
            "share_real": round(float(keep[use & (y == 0)].mean()), 4), "share_sim": round(float(keep[use & (y == 1)].mean()), 4)}
    return out


@app.local_entrypoint()
def lens(results: str, layer: int = 63):
    """The logit-lens check for a finished GPU analysis; writes results/q1_gpu_logit_lens_<stamp>.json."""
    res = json.loads(Path(results).read_text())
    cfg, man = res["config"], res["manifest"]
    check_volume(cfg)
    if cfg.get("design", "attack") != "attack":
        raise ValueError("this diagnostic pairs real e1 with simulated attack runs: attack design only")
    splits = HERE.parent.parent / "splits"
    cfg["main_task_to_fold"] = json.loads((splits / cfg["folds_file"]).read_text())["task_folds"]["main_task_to_fold"]
    cfg["paired_cells"] = design_cells(cfg, json.loads((splits / cfg["exclusions_file"]).read_text()))
    pack_layer.remote(cfg, layer, man)  # packs keyed by manifest; a no-op where already packed
    out = logit_lens.remote(cfg, man, cfg["analysis_id"], layer)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"q1_gpu_logit_lens_{stamp}.json"
    path.write_text(json.dumps({"source": results, "results": out}, indent=1))
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("probe_") and not k.startswith("auroc")}, indent=1))
    for k, v in out.items():
        if k.startswith("auroc") or k.startswith("probe_"):
            print(k, "run", v["run"]["auroc"] if v["run"] else None, "step", v["step"]["auroc"] if v["step"] else None,
                  {kk: vv for kk, vv in v.items() if kk.startswith("share")})
    print(f"-> {path}")


# ---- A label-shuffle null for the headline itself (the layer choice inside the null) -------------------------------
# Per layer: the same 100 label shuffles as the main null (per main task, real and sim swap with probability 1/2), fit
# as one batched problem for the outer folds and for each outer fold's inner CV. Then, per draw, the headline rule picks
# a layer per outer fold from the inner-CV AUROCs and scores that fold with it, exactly as for the real labels.


@app.function(image=image, gpu=GPU, cpu=8, memory=98304, volumes=VOLUMES, timeout=TIMEOUT_S,
              scaledown_window=SCALEDOWN_S)
def selection_null_layer(cfg: dict, layer: int, man: dict, variants: list[str], scheme: str = "task") -> dict:
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score

    from probe_shift import torch_probe as tp
    from probe_shift.q1_eval import EDITOR, FIXED_CALL_TOKENS, Scorer, comparisons, load_runs, step_table

    import pickle

    acts_vol.reload()
    ckpt = (Path("/acts") / f"{cfg['extraction_run']}__q1gpu" / cfg["analysis_id"] / "selection_null"
            / f"L{layer}_{scheme}_{'_'.join(variants)}.pkl")
    if ckpt.exists():  # finished by an earlier attempt
        return pickle.loads(ckpt.read_bytes())
    runs = load_runs(cfg, man)
    scorer = Scorer(cfg, runs, "real_vs_sim")
    rng = np.random.default_rng(cfg["seed"] + 1)  # the same draws as the main null
    tasks = sorted({runs[i]["main_task"] for i in np.flatnonzero(scorer.runs_in)})
    draws = np.zeros((len(runs), cfg["shuffle_null"]), np.float32)
    for d in range(cfg["shuffle_null"]):
        flip = dict(zip(tasks, rng.random(len(tasks)) < 0.5))
        draws[:, d] = [1 - y if (scorer.runs_in[i] and flip[runs[i]["main_task"]]) else y for i, y in enumerate(scorer.y_run)]
    fold_of = np.array([m["fold"] if scheme == "task" else cfg["environment_to_fold"][m["environment"]] for m in runs])
    out = {}
    for pos in cfg["positions"]:
        t = step_table(runs, pos)
        X = torch.from_numpy(np.load(packed_dir(cfg, man) / f"L{layer}_{pos}.npy")).cuda().float()
        arm = np.array([runs[r]["arm"] for r in t["run"]])
        for variant in variants:
            use = scorer.runs_in[t["run"]] & ((t["tool"] != EDITOR) if variant == "no_editor_outputs" else True)
            if pos == "call_end" and cfg.get("call_content_only"):  # as in gpu_layer
                use &= ~np.isin(t["token_text"], fixed_tokens(cfg))
            fold = fold_of[t["run"]]
            folds = sorted(set(fold[use]))

            def run_means(rows, S):
                """Mean step score per run for the given step rows: [n_runs, n_draws] with NaN where absent."""
                r = t["run"][rows]
                cnt = np.bincount(r, minlength=len(runs))[:, None]
                tot = np.zeros((len(runs), S.shape[1]))
                np.add.at(tot, r, S)
                with np.errstate(invalid="ignore"):
                    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)

            def fit_score(tr, te):
                Xtr, Xte = tp.standardize(X[torch.from_numpy(np.flatnonzero(tr)).cuda()], X[torch.from_numpy(np.flatnonzero(te)).cuda()])
                W, b, _ = tp.fit(Xtr, torch.from_numpy(draws[t["run"][tr]]).cuda(), cfg["C"], cfg["max_iter"])
                return tp.decision(Xte, W, b).cpu().numpy()

            outer = np.full((len(runs), draws.shape[1]), np.nan)
            inner_s = {f: np.full((len(runs), draws.shape[1]), np.nan) for f in folds}  # outer f -> its inner run scores
            for k, f in enumerate(folds):
                te = use & (fold == f)
                outer[np.unique(t["run"][te])] = run_means(np.flatnonzero(te), fit_score(use & (fold != f), te))[np.unique(t["run"][te])]
                for g in folds[k + 1:]:  # one fit per pair {f, g}: scores g for outer f and f for outer g, as in gpu_layer
                    te_f, te_g = use & (fold == f), use & (fold == g)
                    rows = np.flatnonzero(te_f | te_g)
                    S = fit_score(use & (fold != f) & (fold != g), te_f | te_g)
                    for o, te_o in ((f, te_g), (g, te_f)):
                        m = te_o[rows]
                        ro = np.unique(t["run"][rows[m]])
                        inner_s[o][ro] = run_means(rows[m], S[m])[ro]
            inner = np.zeros((len(folds), draws.shape[1]))
            for k, f in enumerate(folds):
                ok = ~np.isnan(inner_s[f][:, 0])
                inner[k] = [roc_auc_score(draws[ok, d], inner_s[f][ok, d]) for d in range(draws.shape[1])]
            out[f"{pos}__{variant}"] = {"outer_run_scores": outer.astype(np.float32), "inner_auroc": inner, "folds": folds}
            print(f"L{layer} {pos} {variant}: selection-null fits done", flush=True)
    result = {"layer": layer, "draws": draws, "fold_of_run": fold_of, "results": out}
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(pickle.dumps(result))
    acts_vol.commit()
    return result


@app.local_entrypoint()
def selection_null(results: str = "", variants: str = "all_outputs,no_editor_outputs", max_layers: str = "63,48",
                   scheme: str = "task", config: str = "", analysis_id: str = "", remote: bool = False, commit: str = ""):
    """The headline's own label-shuffle null, for a finished GPU analysis; `--scheme environment` for the held-out-
    environment headline (inner CV over training environments). Or, while that analysis runs, from its `--config` and
    `--analysis-id` (the same settings and runs; checkpoints go under that analysis). Writes
    results/q1_gpu_selection_null_<scheme>_<stamp>.json."""
    import numpy as np
    import yaml
    from sklearn.metrics import roc_auc_score

    if results:
        res = json.loads(Path(results).read_text())
        cfg, man = res["config"], res["manifest"]
    else:  # the running analysis's resolved settings and manifest, as `main` saved them (clean cells, test overrides)
        run = load_config(config)["extraction_run"]
        saved = json.loads(b"".join(acts_vol.read_file(f"{run}__q1gpu/{analysis_id}/run_config.json")))
        cfg, man = saved["config"], saved["manifest"]
    check_volume(cfg)
    splits = HERE.parent.parent / "splits"
    folds = json.loads((splits / cfg["folds_file"]).read_text())
    cfg["main_task_to_fold"] = folds["task_folds"]["main_task_to_fold"]
    cfg["environment_to_fold"] = folds["environment_folds"]["environment_to_fold"]
    cfg["paired_cells"] = design_cells(cfg, json.loads((splits / cfg["exclusions_file"]).read_text()))
    vs = variants.split(",")
    if set(vs) - {"all_outputs", "no_editor_outputs"}:
        raise ValueError(f"selection_null implements all_outputs and no_editor_outputs only, not {set(vs) - {'all_outputs', 'no_editor_outputs'}}")
    source = results or f"{config} / {analysis_id}"
    prov = launch_provenance(cfg, commit)
    if remote:  # all of it on Modal (launch with --detach); the result lands on the Volume, then `fetch_file` copies it
        call = selection_null_orchestrate.spawn(cfg, man, vs, scheme, max_layers, source, prov)
        print(f"selection null on Modal: function call {call.object_id}; writes {cfg['extraction_run']}__q1gpu/"
              f"{cfg['analysis_id']}/selection_null_{scheme}_{'-'.join(vs)}_max{max_layers.replace(',', '-')}.json", flush=True)
        return
    out = selection_null_compute(cfg, man, vs, scheme, max_layers, wait)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"q1_gpu_selection_null_{scheme}_{stamp}.json"
    path.write_text(json.dumps(selection_null_record(source, scheme, out, prov), indent=1))
    print(f"-> {path}")


def selection_null_record(source: str, scheme: str, out: dict, prov: dict | None = None) -> dict:
    return {"source": source, "scheme": scheme, "provenance": prov, "rule": "same draws as the main null; per draw and outer fold, the "
            "earliest layer within headline_tolerance of the best inner-CV AUROC", "results": out}


def selection_null_compute(cfg, man, vs, scheme, max_layers, get) -> dict:
    """Pack, run every layer's shuffled fits, then apply the headline rule per draw; `get` waits on a spawned call."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    n = len(cfg["layers"])
    [get(c) for c in [pack_layer.spawn(cfg, l, man) for l in cfg["layers"]]]  # keyed by manifest; a no-op where packed
    per = {r["layer"]: r for r in [get(c) for c in [selection_null_layer.spawn(cfg, l, man, vs, scheme) for l in cfg["layers"]]]}
    draws = per[cfg["layers"][0]]["draws"]
    out = {}
    for max_layer in (int(x) for x in max_layers.split(",")):
        layers = [l for l in cfg["layers"] if l <= max_layer]
        for pos in cfg["positions"]:
            for v in vs:
                key = f"{pos}__{v}"
                folds = per[layers[0]]["results"][key]["folds"]
                fold_of_run = per[layers[0]]["fold_of_run"]
                null = []
                for d in range(draws.shape[1]):
                    rs = np.full(draws.shape[0], np.nan)
                    for k, f in enumerate(folds):
                        inner = {l: per[l]["results"][key]["inner_auroc"][k, d] for l in layers}
                        top = max(inner.values())
                        chosen = min(l for l, a in inner.items() if a >= top - cfg["headline_tolerance"])
                        sc = per[chosen]["results"][key]["outer_run_scores"][:, d]
                        mask = (fold_of_run == f) & ~np.isnan(sc)  # each run sits in one outer fold
                        rs[mask] = sc[mask]
                    ok = ~np.isnan(rs)
                    null.append(roc_auc_score(draws[ok, d], rs[ok]))
                null = np.array(null)
                out[f"max{max_layer}__{key}"] = {"n": len(null), "mean": round(float(null.mean()), 4),
                                                  "p95": round(float(np.percentile(null, 95)), 4), "p99": round(float(np.percentile(null, 99)), 4)}
                print(f"selection-aware null, layers <= {max_layer}, {key}: mean {null.mean():.3f}, p95 {np.percentile(null, 95):.3f}")
    return out


@app.function(image=image, cpu=4, memory=65536, volumes=VOLUMES, timeout=24 * 3600)
def selection_null_orchestrate(cfg, man, vs, scheme, max_layers, source, prov=None) -> str:
    out = selection_null_compute(cfg, man, vs, scheme, max_layers, lambda c: c.get())
    name = f"selection_null_{scheme}_{'-'.join(vs)}_max{max_layers.replace(',', '-')}.json"  # one file per distinct request
    path = Path("/acts") / f"{cfg['extraction_run']}__q1gpu" / cfg["analysis_id"] / name
    path.write_text(json.dumps(selection_null_record(source, scheme, out, prov), indent=1))
    acts_vol.commit()
    return str(path)


@app.local_entrypoint()
def fetch_file(volume_path: str, name: str):
    """Copy a file an orchestrated job wrote on the Volume into results/<name> (append-only: an existing file is kept)."""
    local = HERE / "results" / name
    if local.exists():
        print(f"{local} exists; kept")
        return
    tmp = local.with_name(f".{name}.tmp")
    tmp.write_bytes(b"".join(acts_vol.read_file(volume_path)))
    os.replace(tmp, local)
    print(f"-> {local}")


@app.function(image=image, cpu=4, memory=16384, volumes=VOLUMES, timeout=1800)
def load_layer_results(cfg: dict, analysis_id: str) -> dict:
    acts_vol.reload()
    base = Path("/acts") / f"{cfg['extraction_run']}__q1gpu" / analysis_id
    return {str(l): json.loads((base / f"L{l}.json").read_text()) for l in cfg["layers"] if (base / f"L{l}.json").exists()}


@app.local_entrypoint()
def assemble(analysis_id: str, config: str = "q1_segment_end.yaml", tool_errors: str = "", subsets: str = str(SUBSETS)):
    """Rebuild a run's results file from the per-layer results on the Volume (when the client died after the layers
    finished), recomputing the headline; writes results/q1_gpu_<analysis_id>.json as `main` would."""
    import yaml

    cfg = load_config(config)
    check_volume(cfg)
    if cfg["layers"] == "all":
        cfg["layers"] = list(range(64))
    splits = HERE.parent.parent / "splits"
    folds = json.loads((splits / cfg["folds_file"]).read_text())
    cfg["main_task_to_fold"] = folds["task_folds"]["main_task_to_fold"]
    cfg["environment_to_fold"] = folds["environment_folds"]["environment_to_fold"]
    ex = json.loads((splits / cfg["exclusions_file"]).read_text())
    set_slices(cfg, ex)
    cfg["analysis_id"] = analysis_id
    if (HERE / "results" / f"q1_gpu_{analysis_id}.json").exists():
        raise FileExistsError(f"results/q1_gpu_{analysis_id}.json exists; results/ is append-only")
    try:  # the settings the run used, saved by `main`; older runs fall back to the current YAML and splits
        saved = json.loads(b"".join(acts_vol.read_file(f"{cfg['extraction_run']}__q1gpu/{analysis_id}/run_config.json")))
        cfg, subsets, tool_errors = saved["config"], saved["subsets_file"], saved["tool_errors_file"]
        man = saved["manifest"]
    except (FileNotFoundError, modal.exception.NotFoundError):
        print("no saved run_config.json: using the current YAML and splits", flush=True)
        cfg["settings_from_current_yaml"] = True
        man = manifest.remote(cfg)
    per_layer = load_layer_results.remote(cfg, analysis_id)
    missing = [l for l in cfg["layers"] if str(l) not in per_layer]
    if missing:
        raise ValueError(f"layers missing on the Volume: {missing}")
    keys = [m["key"] for m in per_layer[str(cfg["layers"][0])]["runs"]]
    assert all([m["key"] for m in r["runs"]] == keys for r in per_layer.values()), "layers read different runs"
    head = None if cfg.get("lite") else headline.remote(cfg, man, per_layer)
    head_env = headline.remote(cfg, man, per_layer, "environment") if cfg.get("inner_cv_environment") and not cfg.get("lite") else None
    cfg_out = {k: v for k, v in cfg.items() if k not in ("main_task_to_fold", "paired_cells", "base200", "overlap800", "under_32k")}
    result = {"manifest": man, "headline": head, "headline_held_out_environment": head_env, "by_layer": {l: {k: v for k, v in r.items() if k not in ("run_scores", "runs")}
                                                                for l, r in per_layer.items()},
              "config": cfg_out | {"subsets_file": subsets, "tool_errors_file": tool_errors, "assembled_from_volume": True,
                                   "modal": {"app": APP, "gpu": GPU, "activations_volume": ACTS}}}
    path = HERE / "results" / f"q1_gpu_{analysis_id}.json"
    path.write_text(json.dumps(result, indent=1))
    for pos, by_c in (head or {}).items():
        for c, by_v in by_c.items():
            for v, h in by_v.items():
                print(f"HEADLINE {pos} {c} {v}: run {h['run']['auroc']} {h['run']['ci95']}; layers {h['chosen_layer_by_fold']}")
    print(f"results -> {path}")


# ---- Insert-context confound and a hand-reading list (scrubbed run, held-out environment, no editor outputs) ---------
# At a bash output the probe has read every earlier output, including editor insert/undo_edit outputs, whose format
# level 5 does not equalize. (1) Cells where neither run calls insert or undo_edit anywhere; (2) outputs before a run's
# first insert/undo_edit, by output position; for the probe at the given layers and for TF-IDF level-5 "up to here".
VIEWER = "https://devbox-probe-shift-aswin-surya-viewer.tail5bc016.ts.net/?cell="
BUCKETS = [("1", 1, 1), ("2-3", 2, 3), ("4-10", 4, 10), ("11-25", 11, 25), (">25", 26, 10**9)]


@app.function(image=image, cpu=8, memory=65536, volumes=VOLUMES, timeout=TIMEOUT_S)
def insert_confound(cfg: dict, man: dict, analysis_id: str, layers: list[int], tfidf: dict, example_layer: int = 63) -> dict:
    import numpy as np

    from probe_shift.q1_eval import Scorer, load_runs

    runs = load_runs(cfg, man)
    scorer = Scorer(cfg, runs, "real_vs_sim")
    keys = [m["key"] for m in runs]
    first_iu, ordinal, texts, n_out = {}, {}, {}, {}
    wanted = set(keys)
    for shard in sorted(staged_inputs(cfg).glob("shard_*.jsonl")):
        for line in shard.read_text().splitlines():
            it = json.loads(line)
            if it["key"] not in wanted:
                continue
            step, action, k, first = -1, 0, 0, None
            for m in it["messages"]:
                if m["role"] == "assistant":
                    step, action = step + 1, 0
                    for a, c in enumerate(m.get("tool_calls") or []):
                        cmd = str((c["function"]["arguments"] or {}).get("command")) if c["function"]["name"] == "text_editor" else ""
                        if cmd in ("insert", "undo_edit") and first is None:
                            first = (step, a)
                elif m["role"] == "tool":
                    k += 1
                    ordinal[(it["key"], step, action)] = k
                    texts[(it["key"], step, action)] = m["content"][:300]
                    action += 1
            first_iu[it["key"]], n_out[it["key"]] = first, k
    cell = np.array([m["cell"] for m in runs], object)
    in_cmp = scorer.runs_in
    dirty = {cell[i] for i in np.flatnonzero(in_cmp) if first_iu[keys[i]] is not None}
    clean_cells = sorted({cell[i] for i in np.flatnonzero(in_cmp)} - dirty)
    base = Path("/acts") / f"{cfg['extraction_run']}__q1gpu" / analysis_id
    out = {"n_cells": int(len({cell[i] for i in np.flatnonzero(in_cmp)})), "n_clean_cells": len(clean_cells),
           "share_runs_with_insert_undo": {arm: round(float(np.mean([first_iu[keys[i]] is not None for i in np.flatnonzero(in_cmp)
                                                                     if runs[i]["arm"] == arm])), 3) for arm in ("real-attack-e1", "sim-attack")}}

    def evaluate(run_idx, step, action, s):
        t = {"run": run_idx, "token": np.zeros(len(run_idx), int)}
        ok = ~np.isnan(s)
        before = np.array([first_iu[keys[r]] is None or (st, a) < first_iu[keys[r]] for r, st, a in zip(run_idx, step, action)])
        clean = np.isin(cell[run_idx], clean_cells)
        ordn = np.array([ordinal[(keys[r], st, a)] for r, st, a in zip(run_idx, step, action)])
        res = {"all": scorer.report(t, s, cfg, slices=False),
               "clean_cells": scorer.report(t, np.where(clean, s, np.nan), cfg, slices=False),
               "before_first_insert_undo": scorer.report(t, np.where(before, s, np.nan), cfg, slices=False)}
        y = scorer.y_run[run_idx]
        res["before_first_insert_undo_by_position"] = {
            b: scorer.auc_ci(y[ok & before & (ordn >= lo) & (ordn <= hi)], s[ok & before & (ordn >= lo) & (ordn <= hi)],
                             scorer.main[run_idx][ok & before & (ordn >= lo) & (ordn <= hi)]) for b, lo, hi in BUCKETS}
        return res

    examples = None
    for layer in layers:
        z = np.load(base / f"L{layer}_step_scores_output_end.npz")
        s = z["real_vs_sim__no_editor_outputs__held_out_environment"]
        out[f"L{layer}"] = evaluate(z["run"], z["step"], z["action"], s)
        if layer == example_layer:
            examples = (z["run"], z["step"], z["action"], s)
    idx = {(keys[i], st, a): j for j, (i, st, a) in enumerate(zip(*examples[:3]))}
    tf = np.array([tfidf.get(f"{keys[r]}|{st}|{a}", np.nan) for r, st, a in zip(*examples[:3])])
    out["tfidf_level5_up_to_here"] = evaluate(examples[0], examples[1], examples[2], tf) | {
        "matched_rows": int((~np.isnan(tf) & ~np.isnan(examples[3])).sum()), "probe_rows": int((~np.isnan(examples[3])).sum())}

    # Hand-reading list at example_layer: run scores = mean step score (held-out environment, no editor outputs).
    r_idx, st_, ac_, s = examples
    ok = ~np.isnan(s)
    rs = scorer.run_scores(r_idx[ok], s[ok])
    by_cell = {}
    for i in np.flatnonzero(in_cmp & ~np.isnan(rs)):
        by_cell.setdefault(cell[i], {})[runs[i]["arm"]] = i
    pairs = [(c, d["real-attack-e1"], d["sim-attack"]) for c, d in by_cell.items() if len(d) == 2]
    def steps_of(i, k=3):
        rows = np.flatnonzero((r_idx == i) & ok)
        rows = rows[np.argsort(-np.abs(s[rows] - np.median(s[ok])))][:k]
        return [{"step": int(st_[j]), "action": int(ac_[j]), "position": ordinal[(keys[i], st_[j], ac_[j])], "score": round(float(s[j]), 3),
                 "text": texts[(keys[i], st_[j], ac_[j])]} for j in rows]
    def entry(c, ri, si, why):
        return {"cell": c, "why": why, "env": runs[ri]["environment"], "viewer": VIEWER + __import__("urllib.parse").parse.quote(c, safe=""),
                "real_e1": {"key": keys[ri], "run_score": round(float(rs[ri]), 3), "steps": steps_of(ri)},
                "sim": {"key": keys[si], "run_score": round(float(rs[si]), 3), "steps": steps_of(si)}}
    gap = lambda p: rs[p[2]] - rs[p[1]]  # sim minus real run score: > 0 means the pair is ordered correctly
    right = sorted(pairs, key=lambda p: -gap(p))[:10]
    wrong = sorted(pairs, key=gap)[:5]
    out["share_pairs_correct"], out["n_pairs"] = round(float(np.mean([gap(p) > 0 for p in pairs])), 3), len(pairs)
    out["reading_list"] = ([entry(c, r, s_, f"confidently right: sim scored {gap((c, r, s_)):.2f} above its real pair")
                            for c, r, s_ in right]
                           + [entry(c, r, s_, f"confidently wrong: sim scored {-gap((c, r, s_)):.2f} below its real pair")
                              for c, r, s_ in wrong])
    return out


@app.local_entrypoint()
def confound(results: str, tfidf_scores: str, layers: str = "48,63", example_layer: int = 63):
    """Insert-context confound and reading list (at `example_layer`, one of `layers`) for a finished (scrubbed) GPU
    analysis; writes JSON and markdown."""
    import gzip

    if example_layer not in [int(x) for x in layers.split(",")]:
        raise ValueError(f"--example-layer {example_layer} is not among --layers {layers}")
    res = json.loads(Path(results).read_text())
    cfg, man = res["config"], res["manifest"]
    check_volume(cfg)
    if cfg.get("design", "attack") != "attack":
        raise ValueError("this diagnostic pairs real e1 with simulated attack runs: attack design only")
    splits = HERE.parent.parent / "splits"
    cfg["main_task_to_fold"] = json.loads((splits / cfg["folds_file"]).read_text())["task_folds"]["main_task_to_fold"]
    cfg["paired_cells"] = design_cells(cfg, json.loads((splits / cfg["exclusions_file"]).read_text()))
    tf = {}
    with gzip.open(tfidf_scores, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            if r["comparison"] == "real_vs_sim" and r["outputs"] == "no_editor" and r["arm"] in ("real-attack-e1", "sim-attack"):
                tf[f"{r['key']}|{r['step']}|{r['action']}"] = r["score"]
    out = insert_confound.remote(cfg, man, cfg["analysis_id"], [int(x) for x in layers.split(",")], tf, example_layer)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"q1_gpu_insert_confound_{stamp}.json"
    path.write_text(json.dumps({"source": results, "tfidf_scores": tfidf_scores, "results": out}, indent=1))
    a = lambda d: "–" if not d else f"{d['auroc']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]"
    print(f"cells {out['n_cells']}, clean (no insert/undo_edit in either run) {out['n_clean_cells']}; runs with insert/undo: "
          f"{out['share_runs_with_insert_undo']}")
    for k in [f"L{l}" for l in layers.split(",")] + ["tfidf_level5_up_to_here"]:
        r = out[k]
        print(f"{k}: run AUROC all {a(r['all']['run'])} | clean cells {a(r['clean_cells']['run'])} | before first insert/undo "
              f"{a(r['before_first_insert_undo']['run'])}; by position {{{', '.join(f'{b}: ' + (format(v['auroc'], '.3f') if v else '–') for b, v in r['before_first_insert_undo_by_position'].items())}}}")
    md = [f"# Layer-{example_layer} reading list (scrubbed text, held-out environment, no editor outputs)", "",
          f"From `{Path(results).name}`. Run score: the mean of the probe's step scores (higher = more simulated-looking). "
          f"Cells ranked by the simulated run's score minus its real pair's: the 10 largest (right) and the 5 most negative "
          f"(wrong); {out['share_pairs_correct']:.1%} of {out['n_pairs']} pairs are ordered correctly. Each "
          "run lists its 3 outputs with the most extreme step scores, with the first 300 characters of the scrubbed text.", ""]
    for e in out["reading_list"]:
        md += [f"## {e['cell']} ({e['env']}): {e['why']}", "", f"[viewer]({e['viewer']}) · real e1 run score {e['real_e1']['run_score']}, "
               f"sim run score {e['sim']['run_score']}", ""]
        for arm in ("real_e1", "sim"):
            md.append(f"**{arm}** `{e[arm]['key']}`")
            for st in e[arm]["steps"]:
                md += [f"- step {st['step']} action {st['action']} (output #{st['position']}), score {st['score']}:", "",
                       "  ```", *["  " + l for l in st["text"].splitlines()], "  ```"]
            md.append("")
    mpath = HERE / "results" / f"reading_list_L{example_layer}_scrubbed_{stamp}.md"
    mpath.write_text("\n".join(md) + "\n")
    print(f"-> {path}\n-> {mpath}")


@app.function(image=image, cpu=8, memory=65536, volumes=VOLUMES, timeout=TIMEOUT_S)
def call_content_slice(cfg: dict, man: dict, analysis_id: str, layers: list[int]) -> dict:
    """Run- and step-level AUROC of the saved call_end scores, all calls vs only those whose read token is call content."""
    import numpy as np

    from probe_shift.q1_eval import Scorer, fixed_call_tokens, load_runs, step_table

    acts_vol.reload()
    runs = load_runs(cfg, man)
    t = step_table(runs, "call_end")
    if (t["token_text"] == "").any():
        raise ValueError("call_end positions without token_text in this extraction's metadata")
    fixed = np.isin(t["token_text"], fixed_call_tokens(cfg["extraction_run"]))  # today's list at this level, recorded
    base = Path("/acts") / f"{cfg['extraction_run']}__q1gpu" / analysis_id
    scorers, out = {}, {}
    for layer in layers:
        z = np.load(base / f"L{layer}_step_scores_call_end.npz")
        if not (np.array_equal(z["run"], t["run"]) and np.array_equal(z["token"], t["token"])):
            raise ValueError(f"L{layer}: saved call_end rows do not match this manifest's step table")
        for k in z.files:
            if k in ("run", "step", "action", "token"):
                continue
            comparison = k.split("__")[0]
            sc = scorers.setdefault(comparison, Scorer(cfg, runs, comparison))
            s = z[k]
            used = ~np.isnan(s) & sc.runs_in[t["run"]]
            all_, content = sc.report(t, s, cfg, slices=False), sc.report(t, np.where(fixed, np.nan, s), cfg, slices=False)
            out.setdefault(str(layer), {})[k] = {"all_calls": {"run": all_["run"], "step": all_["step"]},
                                                 "content_tokens": {"run": content["run"], "step": content["step"]},
                                                 "share_fixed": round(float(fixed[used].mean()), 4)}
    return out


@app.local_entrypoint()
def call_content(results: str, layers: str = ""):
    """call_end restricted to content tokens (not a timeout's final `0` or an escaped newline's `n`) for a finished GPU
    analysis; writes results/q1_gpu_call_content_<stamp>.json."""
    res = json.loads(Path(results).read_text())
    cfg, man = res["config"], res["manifest"]
    check_volume(cfg)
    splits = HERE.parent.parent / "splits"
    cfg["main_task_to_fold"] = json.loads((splits / cfg["folds_file"]).read_text())["task_folds"]["main_task_to_fold"]
    cfg["paired_cells"] = design_cells(cfg, json.loads((splits / cfg["exclusions_file"]).read_text()))
    ls = [int(x) for x in layers.split(",")] if layers else (list(range(64)) if cfg["layers"] == "all" else cfg["layers"])
    out = call_content_slice.remote(cfg, man, cfg["analysis_id"], ls)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"q1_gpu_call_content_{stamp}.json"
    from probe_shift.q1_eval import fixed_call_tokens

    path.write_text(json.dumps({"source": results, "fixed_tokens": fixed_call_tokens(cfg["extraction_run"]), "results": out}, indent=1))
    a = lambda d: "–" if not d else f"{d['auroc']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]"
    for l, r in out.items():
        for k, v in r.items():
            if k.startswith("real_vs_sim"):
                print(f"L{l} {k}: run all {a(v['all_calls']['run'])} | content only {a(v['content_tokens']['run'])}; "
                      f"fixed share {v['share_fixed']:.3f}")
    print(f"-> {path}")


# ---- The frozen protocol's decision and validity checks (protocol_level7.md) -----------------------------------------
# Per split (A: held-out environment, B: main task): the headline run scores against TF-IDF's run scores (the mean over
# the probe's own outputs that TF-IDF also scored), as a paired task-bootstrap gap; the same on the clean cells; per
# environment, with each environment dropped, and within each outer fold.
SPLITS = {"A_held_out_environment": "headline_held_out_environment", "B_main_task": "headline"}


def load_tfidf(paths: dict, baseline: str = "") -> tuple[dict, dict]:
    """baselines-q3 per-output score files, {split: path} -> ({"split|comparison|outputs": {"key|step|action": score}},
    {split: [baseline]}). One baseline per file (or the rows of `baseline`, for files holding several, e.g.
    level5_outputs_only) and one score per output, so another variant cannot stand in silently."""
    import gzip

    tfidf, baselines = {}, {}
    for split, path in paths.items():
        if not path:
            continue
        with gzip.open(path, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                if baseline and r["baseline"] != baseline:
                    continue
                baselines.setdefault(split, set()).add(r["baseline"])
                k = f"{r['key']}|{r['step']}|{r['action']}"
                d = tfidf.setdefault(f"{split}|{r['comparison']}|{r['outputs']}", {})
                if k in d:
                    raise ValueError(f"{path}: two scores for {r['comparison']} {r['outputs']} {k}")
                d[k] = r["score"]
        if len(baselines.get(split, ())) != 1:
            raise ValueError(f"{path}: TF-IDF baselines {sorted(baselines.get(split, ()))}; pass a file with one, or --tfidf-baseline")
    baselines = {k: sorted(v) for k, v in baselines.items()}
    print("TF-IDF baselines:", baselines)
    return tfidf, baselines
OUTPUTS = {"all_outputs": "all", "no_editor_outputs": "no_editor"}


# Outputs whose text ends (before trailing whitespace) in the number placeholder `<n>` or `<END_TOOL_OUTPUT>`: an output
# readout there sits on a fixed slot, in both arms (about 5% and 2.4% of level-7 outputs; verification pass, 2026-10-06).
FIXED_SLOT_END = re.compile(r"(<n>|<END_TOOL_OUTPUT>)\s*$")


@app.function(image=image, cpu=8, memory=32768, volumes=VOLUMES, timeout=TIMEOUT_S)
def protocol_checks(cfg: dict, man: dict, heads: dict, tfidf: dict, clean_cells: list[str], tag: str = "",
                    provenance: dict | None = None) -> dict:
    """The frozen protocol's numbers for every headline in `heads`: the probe on all its outputs, and for output readouts
    (output_end and its variants) the probe and TF-IDF on the outputs both scored, their paired gap, the clean-cell and
    fixed-slot rows, per environment, per fold and per layer."""
    import numpy as np

    from probe_shift.q1_eval import Scorer, load_runs, step_table

    acts_vol.reload()
    runs = load_runs(cfg, man)
    keys = np.array([m["key"] for m in runs], object)
    env = np.array([m["environment"] for m in runs], object)
    clean = np.isin([m["cell"] for m in runs], clean_cells)
    base = Path("/acts") / f"{cfg['extraction_run']}__q1gpu" / cfg["analysis_id"]
    tables, saved, texts = {}, {}, []

    def table(pos):
        if pos not in tables:
            tables[pos] = step_table(runs, pos)
        return tables[pos]

    def steps(pos, l):
        """Layer l's saved out-of-fold step scores at this readout, checked against this manifest's rows."""
        if (pos, l) not in saved:
            saved[(pos, l)] = np.load(base / f"L{l}_step_scores_{pos}.npz")
            if not np.array_equal(saved[(pos, l)]["run"], table(pos)["run"]):
                raise ValueError(f"L{l}: saved {pos} rows do not match this manifest")
        return saved[(pos, l)]

    def headline_steps(pos, comparison, variant, chosen, fold, suffix):
        """The headline's out-of-fold step scores: each outer fold's rows from its chosen layer."""
        t = table(pos)
        s = np.full(len(t["run"]), np.nan)
        for f, l in chosen.items():
            rows = fold[t["run"]] == int(f)
            s[rows] = steps(pos, l)[f"{comparison}__{variant}{suffix}"][rows]
        return s

    def text_mask(pos, pred):
        """Per row of this output readout: pred(the output's staged, scrubbed text)."""
        if not texts:
            texts.append(output_texts(cfg, set(keys))[0])
        t = table(pos)
        k = [(keys[r], int(st), int(a)) for r, st, a in zip(t["run"], t["step"], t["action"])]
        missing = [x for x in k if x not in texts[0]]
        if missing:
            raise ValueError(f"{len(missing)} {pos} rows without a staged output text, e.g. {missing[:3]}")
        return np.array([bool(pred(texts[0][x])) for x in k])

    out = {"cut_at_32k_by_arm": {a: {"runs": int(sum(m["arm"] == a for m in runs)), "cut": int(sum(m["arm"] == a and m["truncated"] for m in runs))}
                                 for a in sorted({m["arm"] for m in runs})}}  # this extraction's text, not the original's
    for split, head in heads.items():
        fold = np.array([cfg["environment_to_fold"][m["environment"]] if split.startswith("A") else m["fold"] for m in runs])
        suffix = "__held_out_environment" if split.startswith("A") else ""
        out[split] = {}
        for pos in head:
            t = table(pos)
            for comparison in head[pos]:
                sc = Scorer(cfg, runs, comparison)
                auc = lambda s, m: sc.auc_ci(sc.y_run[m], s[m], sc.main[m])
                for variant in head[pos][comparison]:
                    h = head[pos][comparison][variant]
                    p = np.array([np.nan if x is None else x for x in h["run_scores"]])  # the headline, all its outputs
                    r = {"probe": auc(p, sc.runs_in), "chosen_layer_by_fold": h["chosen_layer_by_fold"]}
                    tf = ps = None  # TF-IDF, and the probe on the same outputs (output readouts with TF-IDF scores only)
                    if pos.startswith("output_end") and f"{split}|{comparison}|{OUTPUTS[variant]}" in tfidf:
                        # Both run means over the identical outputs: the probe's scored outputs that TF-IDF also scored.
                        d = tfidf[f"{split}|{comparison}|{OUTPUTS[variant]}"]
                        hs = headline_steps(pos, comparison, variant, h["chosen_layer_by_fold"], fold, suffix)
                        rows = sc.runs_in[t["run"]] & ~np.isnan(hs)
                        v = np.array([d.get(f"{keys[i]}|{st}|{a}", np.nan) for i, st, a in zip(t["run"], t["step"], t["action"])])
                        hit = rows & ~np.isnan(v)
                        r["tfidf_coverage_of_probe_outputs"] = round(float(hit.sum() / max(rows.sum(), 1)), 4)
                        r["outputs_compared"] = int(hit.sum())
                        tf, ps = sc.run_scores(t["run"][hit], v[hit]), sc.run_scores(t["run"][hit], hs[hit])
                        m = sc.runs_in
                        r["probe_on_shared_outputs"] = auc(ps, m)
                        r["tfidf"] = auc(tf, m)
                        r["gap_probe_minus_tfidf"] = sc.auc_gap(sc.y_run[m], ps[m], tf[m], sc.main[m])
                        m = sc.runs_in & clean
                        r["clean_cells"] = {"n_cells": len(clean_cells), "probe_on_shared_outputs": auc(ps, m), "tfidf": auc(tf, m),
                                            "gap_probe_minus_tfidf": sc.auc_gap(sc.y_run[m], ps[m], tf[m], sc.main[m])}
                        # Sensitivity (no refit): the same comparison without outputs ending on a fixed slot, both sides.
                        fx = text_mask(pos, FIXED_SLOT_END.search)
                        keep = hit & ~fx
                        tf2, ps2 = sc.run_scores(t["run"][keep], v[keep]), sc.run_scores(t["run"][keep], hs[keep])
                        m = sc.runs_in
                        r["no_fixed_slot_outputs"] = {
                            "rule": "outputs whose text ends in <n> or <END_TOOL_OUTPUT> dropped from probe and TF-IDF",
                            "outputs_dropped": int((hit & fx).sum()), "outputs_compared": int(keep.sum()),
                            "probe_on_shared_outputs": auc(ps2, m), "tfidf": auc(tf2, m),
                            "gap_probe_minus_tfidf": sc.auc_gap(sc.y_run[m], ps2[m], tf2[m], sc.main[m])}
                        # Newlines, two definitions: (a) the output's text ends in "\n"; (b) the token the probe reads
                        # ends in a newline (byte-level BPE "Ċ"). Below level 3 most texts end in "\n", but that "\n"
                        # usually merges with the template's into one token that is not entirely inside the output, so
                        # the readout sits before it; (b) decides whether the probe reads a newline slot. Shares per
                        # arm for both; AUROCs and the gap stratified by (b). Probe: the headline over all its outputs
                        # in the stratum; probe and TF-IDF on the outputs both scored, and their gap.
                        nl_text = text_mask(pos, lambda x: x.endswith("\n"))
                        nl_token = np.array([str(x).endswith("Ċ") for x in t["token_text"]])
                        space_token = np.array([str(x).strip("ĊĠĉ") == "" for x in t["token_text"]])
                        arm_of = np.array([runs[i]["arm"] for i in t["run"]], object)
                        r["newline_shares"] = {a: {"outputs": int(n.sum()), "text_ends_in_newline": round(float(nl_text[n].mean()), 4),
                                                   "readout_token_ends_in_newline": round(float(nl_token[n].mean()), 4),
                                                   "readout_token_whitespace_only": round(float(space_token[n].mean()), 4)}
                                               for a in sorted(set(arm_of[rows])) for n in [rows & (arm_of == a)]}
                        r["by_readout_newline"] = {}
                        for name, part in (("readout_token_newline", nl_token), ("readout_token_other", ~nl_token)):
                            own, keep = rows & part, hit & part
                            pa = sc.run_scores(t["run"][own], hs[own])
                            tf3, ps3 = sc.run_scores(t["run"][keep], v[keep]), sc.run_scores(t["run"][keep], hs[keep])
                            r["by_readout_newline"][name] = {
                                "outputs": int(own.sum()), "outputs_compared": int(keep.sum()), "probe": auc(pa, m),
                                "probe_on_shared_outputs": auc(ps3, m), "tfidf": auc(tf3, m),
                                "gap_probe_minus_tfidf": sc.auc_gap(sc.y_run[m], ps3[m], tf3[m], sc.main[m])}
                    shared = lambda m: None if ps is None else auc(ps, m)
                    r["per_environment"], r["drop_one_environment"] = {}, {}
                    for e in sorted(set(env)):
                        m = sc.runs_in & (env == e)
                        r["per_environment"][e] = {"probe": auc(p, m), "probe_on_shared_outputs": shared(m),
                                                   "tfidf": None if tf is None else auc(tf, m)}
                        m = sc.runs_in & (env != e)
                        r["drop_one_environment"][e] = {"probe": auc(p, m), "probe_on_shared_outputs": shared(m),
                                                        "gap": None if tf is None else sc.auc_gap(sc.y_run[m], ps[m], tf[m], sc.main[m])}
                    r["within_fold"] = {int(f): {"probe": auc(p, m), "probe_on_shared_outputs": shared(m)}
                                        for f in sorted(set(fold)) for m in [sc.runs_in & (fold == f)]}
                    if tf is not None:  # the gap at every layer, for interpretation (the headline gap above decides)
                        r["gap_by_layer"] = {}
                        for l in cfg["layers"]:
                            sl = steps(pos, l)[f"{comparison}__{variant}{suffix}"]
                            hit_l = sc.runs_in[t["run"]] & ~np.isnan(sl) & ~np.isnan(v)  # this layer's outputs TF-IDF also scored
                            pl, tl = sc.run_scores(t["run"][hit_l], sl[hit_l]), sc.run_scores(t["run"][hit_l], v[hit_l])
                            m = sc.runs_in
                            r["gap_by_layer"][str(l)] = sc.auc_gap(sc.y_run[m], pl[m], tl[m], sc.main[m])
                    out[split][f"{pos}__{comparison}__{variant}"] = r
    if tag:  # a copy on the Volume, so a report whose laptop client dies can be finished with --from-checks
        dest = base / f"protocol_checks_{tag}.json"
        dest.write_text(json.dumps({"provenance": provenance, "results": out}))
        acts_vol.commit()
    return out


@app.local_entrypoint()
def protocol_report(results: str, tfidf_env: str, tfidf_task: str, selection_null_env: str = "", selection_null_task: str = "",
                    from_report: str = "", tfidf_baseline: str = "", from_checks: str = "", readout: str = "", commit: str = ""):
    """The frozen protocol's decision and checks for a finished level-7 analysis (headlines for both splits); TF-IDF
    level-7 scores on held-out-environment folds (`tfidf_env`) and main-task folds (`tfidf_task`), baselines-q3's format.
    `--from-report`: reuse an earlier report's computed checks (same results and TF-IDF files, checked) and only attach
    the selection-aware nulls. `--readout`: the primary readout (default: the analysis's first position; amendment 4's
    is output_end_ema0.95). `--commit`: the commit holding this code (checked), recorded with the report."""
    from collections import Counter

    res = json.loads(Path(results).read_text())
    cfg, man = res["config"], res["manifest"]
    check_volume(cfg)
    readout = readout or cfg["positions"][0]
    if not readout.startswith("output_end"):
        raise ValueError(f"the decision rule is on an output readout, not {readout}")
    prov = launch_provenance(cfg, commit)
    splits = HERE.parent.parent / "splits"
    folds = json.loads((splits / cfg["folds_file"]).read_text())
    cfg["main_task_to_fold"] = folds["task_folds"]["main_task_to_fold"]
    cfg["environment_to_fold"] = folds["environment_folds"]["environment_to_fold"]
    ex = json.loads((splits / cfg["exclusions_file"]).read_text())
    cfg["paired_cells"] = design_cells(cfg, ex)
    heads = {s: res[k] for s, k in SPLITS.items() if res.get(k)}
    if set(heads) != set(SPLITS):
        raise ValueError(f"results lack a headline for {set(SPLITS) - set(heads)}")
    if from_report:
        prev = json.loads(Path(from_report).read_text())
        if (prev["source"], prev["tfidf_env"], prev["tfidf_task"]) != (results, tfidf_env, tfidf_task):
            raise ValueError(f"{from_report} was computed from other inputs: {prev['source']}, {prev['tfidf_env']}, {prev['tfidf_task']}")
        if tfidf_baseline and any(v != [tfidf_baseline] for v in prev["tfidf_baselines"].values()):
            raise ValueError(f"{from_report} used TF-IDF baselines {prev['tfidf_baselines']}, not {tfidf_baseline}")
        out, baselines, clean_n = prev["results"], prev["tfidf_baselines"], prev["n_clean_cells"]
        clean_cells = [None] * clean_n
    else:
        # TF-IDF at the probe's text level, on each output's own text (the protocol's baseline).
        tfidf, baselines = load_tfidf({"A_held_out_environment": tfidf_env, "B_main_task": tfidf_task}, tfidf_baseline)
        dirty = set(wait(insert_undo_cells.spawn(cfg)))
        clean_cells = [c for c in cfg["paired_cells"] if c not in dirty]
        tag = f"{tfidf_baseline or '_'.join(sorted({b for v in baselines.values() for b in v}))}"
        provenance = {"results": results, "tfidf_env": tfidf_env, "tfidf_task": tfidf_task, "tfidf_baseline": tfidf_baseline}
        if from_checks:  # the checks a dead client's remote call saved, if they were computed from these same inputs
            saved = json.loads(b"".join(acts_vol.read_file(from_checks)))
            if saved.get("provenance") != provenance:
                raise ValueError(f"{from_checks} was computed from other inputs: {saved.get('provenance')}")
            out = saved["results"]
        else:
            out = wait(protocol_checks.spawn(cfg, man, heads, tfidf, clean_cells, tag, provenance))
    # Validity check 5: samples excluded as invalid, by arm (splits/exclusions; text-independent). Cuts at 32k depend on
    # the text, so they come from the analyzed runs' extraction metadata (results: cut_at_32k_by_arm).
    table = {}
    for r in ex["samples"]:
        c = table.setdefault(r["arm"], Counter())
        c["samples"] += 1
        c[f"excluded: {r['reason']}" if r["reason"] else "valid"] += 1
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"q1_protocol_{stamp}.json"
    # Validity check 1: the headline against its selection-aware shuffle null (selection_null, per split).
    for split, f, scheme in (("A_held_out_environment", selection_null_env, "environment"), ("B_main_task", selection_null_task, "task")):
        null_file = json.loads(Path(f).read_text()) if f else {}
        # Only this analysis's null for this split: its source names the analysis id, its scheme the split.
        if f and (null_file["scheme"] != scheme or cfg["analysis_id"] not in null_file["source"]):
            raise ValueError(f"{f} is the {null_file['scheme']} null of {null_file['source']}, not the {scheme} null of {cfg['analysis_id']}")
        nulls = null_file.get("results", {})
        for k, r in out[split].items():
            pos, comparison, variant = k.split("__")
            nk = f"max63__{pos}__{variant}"
            if comparison == "real_vs_sim":
                r["selection_null"] = None if nk not in nulls else nulls[nk] | {
                    "headline_above_p95": r["probe"]["auroc"] > nulls[nk]["p95"]}
    path.write_text(json.dumps({"source": results, "tfidf_env": tfidf_env, "tfidf_task": tfidf_task, "protocol": "protocol_level7.md",
                                "readout": readout, "provenance": prov,
                                "checks_from_report": from_report or None,
                                "tfidf_baselines": baselines,
                                "selection_null_env": selection_null_env, "selection_null_task": selection_null_task,
                                "n_clean_cells": len(clean_cells), "exclusions_by_arm": {a: dict(c) for a, c in table.items()},
                                "results": out}, indent=1))
    prim = out["A_held_out_environment"][f"{readout}__real_vs_sim__all_outputs"]
    g, cg = prim["gap_probe_minus_tfidf"], prim["clean_cells"]["gap_probe_minus_tfidf"]
    sn = prim.get("selection_null")
    print(f"selection-aware null (A, {readout}, all outputs): " + ("NOT PROVIDED" if sn is None else
          f"p95 {sn['p95']}, headline {prim['probe']['auroc']} -> {'passes' if sn['headline_above_p95'] else 'FAILS'}"))
    verdict = ("no probe advantage" if g["ci95"][0] <= 0 <= g["ci95"][1] else
               "probe advantage" if g["ci95"][0] > 0 else "TF-IDF better")
    print(f"PRIMARY (A, {readout}, all outputs): probe {g['probe']} vs TF-IDF {g['baseline']}; gap {g['gap']:+.4f} "
          f"[{g['ci95'][0]:+.4f}, {g['ci95'][1]:+.4f}] over {g['n_tasks']} tasks -> {verdict}; TF-IDF coverage "
          f"{prim['tfidf_coverage_of_probe_outputs']}")
    if prim.get("no_fixed_slot_outputs"):
        fs = prim["no_fixed_slot_outputs"]
        fg = fs["gap_probe_minus_tfidf"]
        print(f"sensitivity, outputs ending in <n> or <END_TOOL_OUTPUT> dropped ({fs['outputs_dropped']} of "
              f"{fs['outputs_dropped'] + fs['outputs_compared']}): gap {fg['gap']:+.4f} [{fg['ci95'][0]:+.4f}, {fg['ci95'][1]:+.4f}]")
    for split, by_k in out.items():
        for k, r in by_k.items() if isinstance(by_k, dict) else ():
            if "real_vs_sim" in k and r.get("newline_shares"):
                print(f"{split} {k} newline shares: {r['newline_shares']}")
            for name, x in (r.get("by_readout_newline") or {}).items() if "real_vs_sim" in k else ():
                xg, xp = x["gap_probe_minus_tfidf"], x["probe"]
                print(f"{split} {k} {name}: {x['outputs']} outputs; probe {xp and xp['auroc']}; on shared: " + (
                    "–" if not xg else f"probe {xg['probe']} vs TF-IDF {xg['baseline']}, gap {xg['gap']:+.4f} "
                                       f"[{xg['ci95'][0]:+.4f}, {xg['ci95'][1]:+.4f}]"))
    if verdict == "probe advantage":
        if cg["ci95"][0] > 0:
            carry = "not carry-over"
        elif cg["ci95"][1] < 0:  # not covered by the frozen rule (amendment 1): reported as its own outcome
            carry = "clean-cell gap below 0: TF-IDF better on clean cells (outside the frozen rule)"
        else:
            carry = "carry-over" if cg["gap"] < g["gap"] / 2 else "inconclusive"
        print(f"clean cells ({len(clean_cells)}): gap {cg['gap']:+.4f} [{cg['ci95'][0]:+.4f}, {cg['ci95'][1]:+.4f}] -> {carry}")
    print(f"-> {path}")



# ---- Probe vs TF-IDF at every layer, paired on identical outputs (any analysis, any text level) -----------------------
@app.function(image=image, cpu=8, memory=32768, volumes=VOLUMES, timeout=TIMEOUT_S)
def paired_gaps(cfg: dict, man: dict, layers: list[int], tfidf: dict) -> dict:
    """Per layer, split, comparison and variant: run AUROC of the probe and of TF-IDF, each a run's mean over the same
    outputs (the probe's scored outputs that TF-IDF also scored), and their paired task-bootstrap gap."""
    import numpy as np

    from probe_shift.q1_eval import Scorer, comparisons, load_runs, step_table

    acts_vol.reload()
    runs = load_runs(cfg, man)
    t = step_table(runs, "output_end")
    keys = [m["key"] for m in runs]
    base = Path("/acts") / f"{cfg['extraction_run']}__q1gpu" / cfg["analysis_id"]
    scorers = {c: Scorer(cfg, runs, c) for c in comparisons(cfg)}
    out = {}
    for l in layers:
        z = np.load(base / f"L{l}_step_scores_output_end.npz")
        if not np.array_equal(z["run"], t["run"]):
            raise ValueError(f"L{l}: saved output_end rows do not match this manifest")
        for split, suffix in (("A_held_out_environment", "__held_out_environment"), ("B_main_task", "")):
            for comparison, sc in scorers.items():
                for variant, outputs in OUTPUTS.items():
                    k = f"{comparison}__{variant}{suffix}"
                    d = tfidf.get(f"{split}|{comparison}|{outputs}")
                    if k not in z.files or d is None:
                        continue
                    s = z[k]
                    v = np.array([d.get(f"{keys[i]}|{st}|{a}", np.nan) for i, st, a in zip(t["run"], t["step"], t["action"])])
                    rows = sc.runs_in[t["run"]] & ~np.isnan(s)
                    hit = rows & ~np.isnan(v)
                    p, tf = sc.run_scores(t["run"][hit], s[hit]), sc.run_scores(t["run"][hit], v[hit])
                    m = sc.runs_in
                    out.setdefault(str(l), {})[f"{split}|{comparison}|{variant}"] = {
                        "coverage": round(float(hit.sum() / max(rows.sum(), 1)), 4), "outputs": int(hit.sum()),
                        "gap_probe_minus_tfidf": sc.auc_gap(sc.y_run[m], p[m], tf[m], sc.main[m])}
    return out


@app.local_entrypoint()
def paired(results: str, label: str, tfidf_env: str = "", tfidf_task: str = "", layers: str = "", tfidf_baseline: str = ""):
    """Probe minus TF-IDF at every layer of a finished analysis (or `--layers`), paired on identical outputs. Pass the
    TF-IDF files at the analysis's own text level. Writes results/q1_paired_<label>_<stamp>.json."""
    res = json.loads(Path(results).read_text())
    cfg, man = res["config"], res["manifest"]
    check_volume(cfg)
    splits = HERE.parent.parent / "splits"
    cfg["main_task_to_fold"] = json.loads((splits / cfg["folds_file"]).read_text())["task_folds"]["main_task_to_fold"]
    cfg["paired_cells"] = cfg.get("clean_cell_list") or design_cells(cfg, json.loads((splits / cfg["exclusions_file"]).read_text()))
    ls = [int(x) for x in layers.split(",")] if layers else (list(range(64)) if cfg["layers"] == "all" else cfg["layers"])
    tfidf, baselines = load_tfidf({"A_held_out_environment": tfidf_env, "B_main_task": tfidf_task}, tfidf_baseline)
    out = wait(paired_gaps.spawn(cfg, man, ls, tfidf))
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"q1_paired_{label}_{stamp}.json"
    path.write_text(json.dumps({"source": results, "extraction_run": cfg["extraction_run"], "tfidf_env": tfidf_env,
                                "tfidf_task": tfidf_task, "tfidf_baselines": baselines, "results": out}, indent=1))
    for l, r in out.items():
        for k, v in r.items():
            g = v["gap_probe_minus_tfidf"]
            if g and "real_vs_sim" in k:
                print(f"L{l} {k}: probe {g['probe']} TF-IDF {g['baseline']} gap {g['gap']:+.3f} [{g['ci95'][0]:+.3f}, "
                      f"{g['ci95'][1]:+.3f}] coverage {v['coverage']}")
    print(f"-> {path}")


# ---- Where the probe and TF-IDF disagree: a reading list ------------------------------------------------------------
@app.function(image=image, cpu=8, memory=32768, volumes=VOLUMES, timeout=TIMEOUT_S)
def disagreements(cfg: dict, man: dict, layer: int, key: str, tfidf: dict, n: int) -> list[dict]:
    """The n outputs where the probe most exceeds TF-IDF in calling them simulated (each score z-scored within its
    held-out fold), with the paired run's output at the same position in the same cell, side by side."""
    import numpy as np

    from probe_shift.q1_eval import load_runs, step_table

    acts_vol.reload()
    runs = load_runs(cfg, man)
    t = step_table(runs, "output_end")
    z = np.load(Path("/acts") / f"{cfg['extraction_run']}__q1gpu" / cfg["analysis_id"] / f"L{layer}_step_scores_output_end.npz")
    if not np.array_equal(z["run"], t["run"]):
        raise ValueError("saved rows do not match this manifest")
    s = z[key]
    keys = [m["key"] for m in runs]
    v = np.array([tfidf.get(f"{keys[i]}|{st}|{a}", np.nan) for i, st, a in zip(t["run"], t["step"], t["action"])])
    fold = np.array([cfg["environment_to_fold"][runs[i]["environment"]] if key.endswith("held_out_environment") else runs[i]["fold"]
                     for i in t["run"]])
    ok = ~np.isnan(s) & ~np.isnan(v) & np.isin([runs[i]["arm"] for i in t["run"]], ["real-attack-e1", "sim-attack"])
    zs, zt = np.full(len(s), np.nan), np.full(len(s), np.nan)
    for f in set(fold[ok]):
        m = ok & (fold == f)
        zs[m] = (s[m] - s[m].mean()) / s[m].std()
        zt[m] = (v[m] - v[m].mean()) / v[m].std()
    texts, position = output_texts(cfg, {keys[i] for i in range(len(runs))})
    ordinal = np.array([position[(keys[i], int(st), int(a))] for i, st, a in zip(t["run"], t["step"], t["action"])])
    at = {(ka[0], k): ka for ka, k in position.items()}
    row_at = {(int(t["run"][r]), int(ordinal[r])): r for r in range(len(s))}
    other = {}
    for i, m_ in enumerate(runs):
        other.setdefault(m_["cell"], {})[m_["arm"]] = i
    pick = [j for j in np.argsort(-(zs - zt)) if ok[j] and zs[j] > 1][:n]
    out = []
    for j in pick:
        i = t["run"][j]
        mate = other[runs[i]["cell"]].get("sim-attack" if runs[i]["arm"] == "real-attack-e1" else "real-attack-e1")
        mk = None if mate is None else at.get((keys[mate], int(ordinal[j])))  # the mate's output at this position, if any
        mr = None if mate is None else row_at.get((mate, int(ordinal[j])))  # and its probe row, if it was read
        out.append({"cell": runs[i]["cell"], "environment": runs[i]["environment"], "arm": runs[i]["arm"], "output": int(ordinal[j]),
                    "probe_z": round(float(zs[j]), 2), "tfidf_z": round(float(zt[j]), 2), "tool": t["tool"][j],
                    "text": texts.get((keys[i], int(t["step"][j]), int(t["action"][j])), "")[:600],
                    "mate_arm": None if mate is None else runs[mate]["arm"],
                    "mate_text": None if mk is None else texts[mk][:600],
                    "mate_probe_z": None if mr is None or np.isnan(zs[mr]) else round(float(zs[mr]), 2)})
    return out


@app.local_entrypoint()
def disagree(results: str, tfidf: str, layer: int = 48, variant: str = "all_outputs", split: str = "held_out_environment", n: int = 20):
    """Reading list: outputs the probe calls simulated and TF-IDF (same text level and split) doesn't. Writes
    results/reading_list_disagree_L<layer>_<stamp>.md."""
    res = json.loads(Path(results).read_text())
    cfg, man = res["config"], res["manifest"]
    check_volume(cfg)
    if cfg.get("design", "attack") != "attack":
        raise ValueError("this diagnostic pairs real e1 with simulated attack runs: attack design only")
    splits = HERE.parent.parent / "splits"
    folds = json.loads((splits / cfg["folds_file"]).read_text())
    cfg["main_task_to_fold"] = folds["task_folds"]["main_task_to_fold"]
    cfg["environment_to_fold"] = folds["environment_folds"]["environment_to_fold"]
    cfg["paired_cells"] = cfg.get("clean_cell_list") or design_cells(cfg, json.loads((splits / cfg["exclusions_file"]).read_text()))
    sp = "A_held_out_environment" if split == "held_out_environment" else "B_main_task"
    tf, baselines = load_tfidf({sp: tfidf})
    key = f"real_vs_sim__{variant}" + ("__held_out_environment" if split == "held_out_environment" else "")
    out = wait(disagreements.spawn(cfg, man, layer, key, tf[f"{sp}|real_vs_sim|{OUTPUTS[variant]}"], n))
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    md = [f"# Where the layer-{layer} probe says simulated and TF-IDF doesn't", "",
          f"From `{Path(results).name}` ({cfg['extraction_run']}; {len(man['runs']) // 3} cells), {split.replace('_', ' ')}, "
          f"{variant.replace('_', ' ')}; TF-IDF `{tfidf.rsplit('/', 1)[-1]}` ({', '.join(baselines[sp])}). Scores are z-scored within each held-out "
          "fold; outputs ranked by probe z minus TF-IDF z among those with probe z > 1. Each shows the other arm's output at the "
          "same position in the same cell. The text is what the probe read (first 600 characters).", ""]
    for k, e in enumerate(out, 1):
        md += [f"## {k}. {e['cell']} ({e['environment']}), output #{e['output']} of the {e['arm']} run ({e['tool']})", "",
               f"probe z {e['probe_z']}, TF-IDF z {e['tfidf_z']}; the {e['mate_arm']} run's output #{e['output']}: probe z {e['mate_probe_z']}", "",
               f"**{e['arm']}**", "", "```", e["text"], "```", "", f"**{e['mate_arm']}**", "", "```", e["mate_text"] or "(no output at this position)", "```", ""]
    path = HERE / "results" / f"reading_list_disagree_L{layer}_{stamp}.md"
    path.write_text("\n".join(md) + "\n")
    print(f"-> {path}")


# Output types, from the text the probe read (and Inspect's error record). Assistant-like: the simulator breaking
# character, writing a reply instead of a tool output; "broad" adds weaker cues the coordinator listed.
ASSISTANT_STRICT = re.compile(r"\bI(?: can(?:'t|not)?| won't| will not|'m (?:unable|not able|happy)| am (?:unable|not able|happy)|"
                              r"'d be happy| apologi[sz]e)\b|\bI'm sorry\b|\bAs an AI\b")
ASSISTANT_BROAD = re.compile(r"\b(?:help(?:s|ing)? (?:you|with)|instead|rather than)\b", re.I)
ERROR_LIKE = re.compile(r"(?i)\b(?:error|exception|traceback|failed|failure|not found|no such file|permission denied|invalid|"
                        r"validation|cannot|can't|unable to)\b")
TRUNCATION = "was too long to be displayed"


def output_types(text: str, error_record: bool) -> list[str]:
    tags = []
    if not text.strip():
        tags.append("empty")
    if error_record:
        tags.append("tool error record")
    if TRUNCATION in text:
        tags.append("truncation wrapper")
    if ASSISTANT_STRICT.search(text):
        tags.append("assistant-like")
    elif ASSISTANT_BROAD.search(text):
        tags.append("assistant-like (broad cue only)")
    if not error_record and ERROR_LIKE.search(text[:300]):
        tags.append("error-like text")
    return tags or ["other"]


@app.function(image=image, cpu=8, memory=65536, volumes=VOLUMES, timeout=TIMEOUT_S)
def cell_disagreements(cfg: dict, man: dict, layer: int, key: str, tfidf: dict, n_disagree: int, n_agree: int,
                       errors: list[str]) -> dict:
    """Per cell (real e1 and simulated run): the probe's and TF-IDF's run-mean margins (sim minus real, over the outputs
    both scored). Lists the cells where the probe orders sim above real and TF-IDF the other way (largest margins, each
    z-scored across cells) and, for reference, cells where both agree strongly. Each output's contribution to a margin is
    (score - the pair's pooled mean) / its run's output count, signed + for sim and - for real; they sum to the margin.

    Also: diffuseness (the share of a pair's outputs that must be dropped, largest contributions first, before its margin
    halves; contributions held fixed, an approximation) for the probe, for TF-IDF, and with each pair's outputs randomly
    reassigned between its runs; output types among the top contributors vs all outputs, by arm; how many top
    contributors have byte-identical text in the other arm at the same position (then the score difference is context);
    and the paired probe-minus-TF-IDF gap with assistant-like outputs dropped from both arms."""
    import numpy as np

    from probe_shift.q1_eval import Scorer, load_runs, step_table

    acts_vol.reload()
    runs = load_runs(cfg, man)
    t = step_table(runs, "output_end")
    z = np.load(Path("/acts") / f"{cfg['extraction_run']}__q1gpu" / cfg["analysis_id"] / f"L{layer}_step_scores_output_end.npz")
    if not np.array_equal(z["run"], t["run"]):
        raise ValueError("saved rows do not match this manifest")
    s, keys = z[key], [m["key"] for m in runs]
    v = np.array([tfidf.get(f"{keys[i]}|{st}|{a}", np.nan) for i, st, a in zip(t["run"], t["step"], t["action"])])
    hit = ~np.isnan(s) & ~np.isnan(v)
    pairs = {}
    for i, m in enumerate(runs):
        if m["arm"] in ("real-attack-e1", "sim-attack"):
            pairs.setdefault(m["cell"], {})[m["arm"]] = i
    texts, position = output_texts(cfg, set(keys))  # every run in the manifest: the step table has e2 rows too
    at = {(ka[0], k): ka for ka, k in position.items()}  # (key, position among all tool messages) -> (key, step, action)
    err = set(errors)
    text_of = np.array([texts.get((keys[i], int(st), int(a)), "") for i, st, a in zip(t["run"], t["step"], t["action"])], object)
    tags = [output_types(x, f"{keys[i]}|{st}|{a}" in err) for x, i, st, a in zip(text_of, t["run"], t["step"], t["action"])]
    ordinal = np.array([position[(keys[i], int(st), int(a))] for i, st, a in zip(t["run"], t["step"], t["action"])])
    row_at = {(int(t["run"][r]), int(ordinal[r])): r for r in range(len(s))}

    def contributions(x, rs, rr):
        mu = np.concatenate([x[rs], x[rr]]).mean()
        return np.concatenate([(x[rs] - mu) / len(rs), -(x[rr] - mu) / len(rr)]), float(x[rs].mean() - x[rr].mean())

    def share_to_halve(contrib, d):
        cum = np.cumsum(np.sort(contrib)[::-1])  # removed share of the margin, largest contributions first
        return (int(np.argmax(cum >= d / 2)) + 1) / len(contrib)  # exists: the contributions sum to d > 0

    cells = []
    for c, d in pairs.items():
        if len(d) < 2:
            continue
        rs = np.flatnonzero((t["run"] == d["sim-attack"]) & hit)
        rr = np.flatnonzero((t["run"] == d["real-attack-e1"]) & hit)
        if not len(rs) or not len(rr):
            continue
        cp, dp = contributions(s, rs, rr)
        ct, dt = contributions(v, rs, rr)
        cells.append({"cell": c, "sim": d["sim-attack"], "real": d["real-attack-e1"], "rows": np.concatenate([rs, rr]),
                      "n_sim": len(rs), "contrib": cp, "dp": dp, "contrib_t": ct, "dt": dt})
    dp_sd, dt_sd = np.std([x["dp"] for x in cells]), np.std([x["dt"] for x in cells])
    for x in cells:
        x["zp"], x["zt"] = x["dp"] / dp_sd, x["dt"] / dt_sd
    q = lambda a: {"n": int(len(a)), "median": round(float(np.median(a)), 3), "q25": round(float(np.percentile(a, 25)), 3),
                   "q75": round(float(np.percentile(a, 75)), 3)}
    rng, null_shares = np.random.default_rng(cfg["seed"]), []
    for x in cells:
        sc_ = s[x["rows"]]
        for _ in range(20):
            perm = rng.permutation(len(sc_))
            c_, d_ = contributions(sc_, perm[:x["n_sim"]], perm[x["n_sim"]:])
            if d_ > 0:
                null_shares.append(share_to_halve(c_, d_))
    diffuse = {"probe": q([share_to_halve(x["contrib"], x["dp"]) for x in cells if x["dp"] > 0]),
               "tfidf": q([share_to_halve(x["contrib_t"], x["dt"]) for x in cells if x["dt"] > 0]),
               "probe_outputs_shuffled_within_pair": q(null_shares)}

    def mate_of(r):
        i = int(t["run"][r])
        d = pairs[runs[i]["cell"]]
        return d["real-attack-e1"] if runs[i]["arm"] == "sim-attack" else d["sim-attack"]

    def other_row(r):  # the other arm's probe row at the same position, if it was read
        return row_at.get((mate_of(r), int(ordinal[r])))

    def other_text(r):  # the other arm's output at the same position (counting every tool message), if it has one
        k = at.get((keys[mate_of(r)], int(ordinal[r])))
        return None if k is None else texts[k]

    # Types among each probe-right cell's top-3 contributors vs all its outputs, by arm; identical text at the same position.
    from collections import Counter

    top_types, all_types, identical = {}, {}, Counter()  # tag counts plus "_outputs", the number of outputs (tags overlap)
    for x in cells:
        if x["dp"] <= 0:
            continue
        for j, r in enumerate(x["rows"]):
            c_ = all_types.setdefault(runs[t["run"][r]]["arm"], Counter())
            c_["_outputs"] += 1
            for tg in tags[r]:
                c_[tg] += 1
        for j in np.argsort(-x["contrib"])[:3]:
            r = x["rows"][j]
            c_ = top_types.setdefault(runs[t["run"][r]]["arm"], Counter())
            c_["_outputs"] += 1
            for tg in tags[r]:
                c_[tg] += 1
            identical["top3"] += 1
            identical["identical_text_in_other_arm"] += int(other_text(r) == text_of[r])
    # The paired gap with assistant-like outputs (strict) dropped from both arms.
    sc = Scorer(cfg, runs, "real_vs_sim")
    keep = hit & sc.runs_in[t["run"]] & np.array(["assistant-like" not in tg for tg in tags])
    p_, tf_ = sc.run_scores(t["run"][keep], s[keep]), sc.run_scores(t["run"][keep], v[keep])
    m = sc.runs_in
    gap_no_assistant = sc.auc_gap(sc.y_run[m], p_[m], tf_[m], sc.main[m])
    keep_all = hit & sc.runs_in[t["run"]]
    p_, tf_ = sc.run_scores(t["run"][keep_all], s[keep_all]), sc.run_scores(t["run"][keep_all], v[keep_all])
    gap_all = sc.auc_gap(sc.y_run[m], p_[m], tf_[m], sc.main[m])

    def describe(x, why):
        outs = []
        for j in np.argsort(-x["contrib"])[:3]:
            r = x["rows"][j]
            i, o, ot = t["run"][r], other_row(r), other_text(r)
            outs.append({"arm": runs[i]["arm"], "step": int(t["step"][r]), "action": int(t["action"][r]), "output": int(ordinal[r]),
                         "contribution": round(float(x["contrib"][j]), 4), "score": round(float(s[r]), 3), "types": tags[r],
                         "text": text_of[r][:600], "other_arm_identical_text": ot == text_of[r],
                         "other_arm_score": None if o is None else round(float(s[o]), 3),
                         "other_arm_text": None if ot is None else ot[:600]})
        m_ = runs[x["sim"]]
        return {"cell": x["cell"], "why": why, "environment": m_["environment"], "main_task": m_["main_task"],
                "outputs": {"sim": int(x["n_sim"]), "real": int(len(x["rows"]) - x["n_sim"])},
                "probe_margin": round(x["dp"], 4), "probe_margin_z": round(x["zp"], 2), "tfidf_margin": round(x["dt"], 4),
                "tfidf_margin_z": round(x["zt"], 2), "top_contributions": outs}

    disagree = sorted([x for x in cells if x["dp"] > 0 and x["dt"] < 0], key=lambda x: -(x["zp"] - x["zt"]))[:n_disagree]
    agree = sorted([x for x in cells if x["dp"] > 0 and x["dt"] > 0], key=lambda x: -min(x["zp"], x["zt"]))[:n_agree]
    count_types = lambda arm: dict(Counter(tg for r in range(len(s)) if hit[r] and runs[t["run"][r]]["arm"] == arm for tg in tags[r]))
    return {"n_cells": len(cells), "probe_right": int(sum(x["dp"] > 0 for x in cells)),
            "tfidf_right": int(sum(x["dt"] > 0 for x in cells)),
            "probe_right_tfidf_wrong": int(sum(x["dp"] > 0 and x["dt"] < 0 for x in cells)),
            "tfidf_right_probe_wrong": int(sum(x["dp"] < 0 and x["dt"] > 0 for x in cells)),
            "diffuseness_share_to_halve": diffuse,
            "types_all_outputs_by_arm": {a: count_types(a) for a in ("real-attack-e1", "sim-attack")},
            "types_probe_right_cells": {"top3_contributors": {a: dict(c) for a, c in top_types.items()},
                                        "all_outputs": {a: dict(c) for a, c in all_types.items()}},
            "top3_identical_text_in_other_arm": dict(identical),
            "paired_gap_all": gap_all, "paired_gap_without_assistant_like": gap_no_assistant,
            "disagree": [describe(x, "probe sim > real, TF-IDF sim < real") for x in disagree],
            "agree": [describe(x, "both sim > real") for x in agree]}


def output_texts(cfg: dict, wanted: set) -> tuple[dict, dict]:
    """({(key, step, action): the tool output's full text}, {(key, step, action): its position among all of the run's
    tool messages, from 1}) from the extraction's staged reader messages. Positions count empty outputs too, which the
    output_end readout skips, so two runs' k-th outputs line up."""
    texts, position = {}, {}
    for shard in sorted(staged_inputs(cfg).glob("shard_*.jsonl")):
        for line in shard.read_text().splitlines():
            it = json.loads(line)
            if it["key"] in wanted:
                step, action, k = -1, 0, 0
                for m in it["messages"]:
                    if m["role"] == "assistant":
                        step, action = step + 1, 0
                    elif m["role"] == "tool":
                        k += 1
                        texts[(it["key"], step, action)] = m["content"]
                        position[(it["key"], step, action)] = k
                        action += 1
    return texts, position


@app.local_entrypoint()
def cells_disagree(results: str, tfidf: str, layer: int = 48, variant: str = "all_outputs", split: str = "held_out_environment",
                   n_disagree: int = 15, n_agree: int = 5, tool_errors: str = str(HERE / "results" / "tool_errors_2026-10-06_134908.jsonl")):
    """Cell-level reading list and audit: pairs the probe orders right and TF-IDF wrong, plus pairs both order right, with
    each pair's top contributing outputs (typed, next to the other arm's output at the same position), diffuseness with
    two references, output types among top contributors, identical-text counts, and the paired gap without assistant-like
    outputs. Writes results/reading_list_cells_L<layer>_<stamp>.{md,json}."""
    res = json.loads(Path(results).read_text())
    cfg, man = res["config"], res["manifest"]
    check_volume(cfg)
    if cfg.get("design", "attack") != "attack":
        raise ValueError("this diagnostic pairs real e1 with simulated attack runs: attack design only")
    splits = HERE.parent.parent / "splits"
    folds = json.loads((splits / cfg["folds_file"]).read_text())
    cfg["main_task_to_fold"] = folds["task_folds"]["main_task_to_fold"]
    cfg["environment_to_fold"] = folds["environment_folds"]["environment_to_fold"]
    cfg["paired_cells"] = cfg.get("clean_cell_list") or design_cells(cfg, json.loads((splits / cfg["exclusions_file"]).read_text()))
    sp = "A_held_out_environment" if split == "held_out_environment" else "B_main_task"
    tf, baselines = load_tfidf({sp: tfidf})
    errors = [f"{r['key']}|{r['step']}|{r['action']}" for r in map(json.loads, Path(tool_errors).read_text().splitlines()) if r["error"]]
    key = f"real_vs_sim__{variant}" + ("__held_out_environment" if split == "held_out_environment" else "")
    out = wait(cell_disagreements.spawn(cfg, man, layer, key, tf[f"{sp}|real_vs_sim|{OUTPUTS[variant]}"], n_disagree, n_agree, errors))
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    base = HERE / "results" / f"reading_list_cells_L{layer}_{stamp}"
    base.with_suffix(".json").write_text(json.dumps({"source": results, "tfidf": tfidf, "tfidf_baselines": baselines, "layer": layer,
                                                     "key": key, "tool_errors": tool_errors} | out, indent=1))
    pc = lambda d: f"median {d['median']:.1%} (IQR {d['q25']:.1%}–{d['q75']:.1%}; n {d['n']})"
    dq, ident = out["diffuseness_share_to_halve"], out["top3_identical_text_in_other_arm"]
    g = lambda d: "–" if not d else f"{d['gap']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] (probe {d['probe']}, TF-IDF {d['baseline']})"
    md = [f"# Cells the layer-{layer} probe orders right and TF-IDF wrong", "",
          f"From `{Path(results).name}` ({cfg['extraction_run']}), {split.replace('_', ' ')}, {variant.replace('_', ' ')}; TF-IDF "
          f"`{tfidf.rsplit('/', 1)[-1]}` ({', '.join(baselines[sp])}). Margin: the simulated run's mean score minus the real "
          "run's, over the outputs both scored (z: across cells). Contribution: an output's share of the probe's margin.", "",
          f"- Cells: {out['n_cells']}. Probe orders sim above real in {out['probe_right']}, TF-IDF in {out['tfidf_right']}; the "
          f"probe right and TF-IDF wrong in {out['probe_right_tfidf_wrong']}, the reverse in {out['tfidf_right_probe_wrong']}.",
          f"- Paired probe-minus-TF-IDF gap (run AUROC): {g(out['paired_gap_all'])}; with assistant-like outputs dropped from "
          f"both arms: {g(out['paired_gap_without_assistant_like'])}.",
          "- Share of a pair's outputs to drop, largest contributions first, before its margin halves: probe "
          f"{pc(dq['probe'])}; TF-IDF {pc(dq['tfidf'])}; probe with each pair's outputs shuffled between its runs "
          f"{pc(dq['probe_outputs_shuffled_within_pair'])}.",
          f"- Top-3 contributors with byte-identical text in the other arm at the same position: "
          f"{ident.get('identical_text_in_other_arm', 0)} of {ident.get('top3', 0)} (there the score difference comes from earlier context).", "",
          "## Output types", "", "| Type | real, all outputs | sim, all outputs | real, top-3 contributors | sim, top-3 contributors |", "|---|---|---|---|---|"]
    ta, tt = out["types_probe_right_cells"]["all_outputs"], out["types_probe_right_cells"]["top3_contributors"]
    share = lambda d, k: f"{d.get(k, 0)} ({d.get(k, 0) / max(d.get('_outputs', 0), 1):.1%})"  # share of outputs; tags overlap
    for k in sorted({k for d in list(ta.values()) + list(tt.values()) for k in d} - {"_outputs"}):
        md.append(f"| {k} | {share(ta.get('real-attack-e1', {}), k)} | {share(ta.get('sim-attack', {}), k)} | "
                  f"{share(tt.get('real-attack-e1', {}), k)} | {share(tt.get('sim-attack', {}), k)} |")
    md += ["", "(Cells where the probe orders sim above real. An output can carry more than one type. Across all cells' outputs: "
           + "; ".join(f"{a}: " + ", ".join(f"{k} {n}" for k, n in sorted(c.items())) for a, c in out["types_all_outputs_by_arm"].items()) + ".)", ""]
    for title, items in (("Probe right, TF-IDF wrong", out["disagree"]), ("Both right (reference)", out["agree"])):
        md += [f"## {title}", ""]
        for k, e in enumerate(items, 1):
            md += [f"### {k}. {e['cell']} ({e['environment']})", "",
                   f"Outputs: sim {e['outputs']['sim']}, real {e['outputs']['real']}. Probe margin {e['probe_margin']:+.3f} "
                   f"(z {e['probe_margin_z']:+.2f}); TF-IDF margin {e['tfidf_margin']:+.3f} (z {e['tfidf_margin_z']:+.2f}).", ""]
            for o in e["top_contributions"]:
                md += [f"- **{o['arm']}**, output #{o['output']} (step {o['step']} action {o['action']}): score {o['score']}, contribution "
                       f"{o['contribution']:+.4f}; types: {', '.join(o['types'])}; other arm's output #{o['output']}: score "
                       f"{o['other_arm_score']}, {'IDENTICAL text' if o['other_arm_identical_text'] else 'different text'}", "",
                       "  ```", *["  " + l for l in o["text"].splitlines()], "  ```", "", "  Other arm, same position:", "",
                       "  ```", *["  " + l for l in (o["other_arm_text"] or "(none)").splitlines()], "  ```", ""]
    base.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md[4:9]))
    print(f"-> {base.with_suffix('.md')}")


# ---- Finish an analysis whose laptop client died before every layer ran ----------------------------------------------
@app.function(image=image, cpu=2, memory=16384, volumes=VOLUMES, timeout=24 * 3600)
def complete_layers_remote(run: str, analysis_id: str, layers: list[int], editor_variants: bool) -> list[int]:
    """Fit the given layers of an analysis from its saved settings (run_config.json), as `main` would have."""
    saved = json.loads((Path("/acts") / f"{run}__q1gpu" / analysis_id / "run_config.json").read_text())
    cfg, man = saved["config"], saved["manifest"]
    if saved["subsets_file"] or saved["tool_errors_file"]:  # their local files are not on Modal
        raise ValueError("complete supports analyses run without --subsets and --tool-errors")
    if saved.get("editor_variants", editor_variants) != editor_variants:  # the run's own setting, where it was recorded
        raise ValueError(f"{analysis_id} ran with editor_variants={saved['editor_variants']}")
    commands = editor_commands.remote(cfg) if editor_variants else {}
    [c.get() for c in [pack_layer.spawn(cfg, l, man) for l in layers]]
    [c.get() for c in [gpu_layer.spawn(cfg, l, man, {"folds": {}}, {}, commands) for l in layers]]
    return layers


@app.local_entrypoint()
def complete(analysis_id: str, config: str, layers: str, editor_variants: bool = False):
    """Launch (with --detach) the missing layers of an analysis on Modal; then `assemble` builds its results file."""
    cfg = load_config(config)
    check_volume(cfg)
    run = cfg["extraction_run"]
    call = complete_layers_remote.spawn(run, analysis_id, [int(x) for x in layers.split(",")], editor_variants)
    print(f"completing layers {layers} of {analysis_id} on Modal: function call {call.object_id}", flush=True)


# ---- Profiled one-layer pilot: rerun one layer of a finished analysis and measure where its time goes ---------------
@app.function(image=image, gpu=GPU, cpu=8, memory=98304, volumes=VOLUMES, timeout=TIMEOUT_S)
def profiled_layer(cfg: dict, layer: int, man: dict, commands: dict | None) -> dict:
    """gpu_layer for one layer, into this pilot's own analysis directory, while a thread samples nvidia-smi every
    0.5 s; returns the layer's result with wall time, process CPU time and the GPU utilization samples."""
    import resource
    import subprocess
    import threading

    samples, errors, stop = [], [], threading.Event()

    def sample():
        while not stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, check=True).stdout.strip()
                samples.append([float(x) for x in out.split(",")])
            except Exception as e:  # recorded, not swallowed: no samples means "unavailable", never zero utilization
                errors.append(repr(e)[:200])
            stop.wait(0.5)

    t = threading.Thread(target=sample, daemon=True)
    t0, c0 = time.time(), resource.getrusage(resource.RUSAGE_SELF)
    t.start()
    result = gpu_layer.local(cfg, layer, man, {"folds": {}}, {}, commands)
    stop.set()
    t.join()
    c1 = resource.getrusage(resource.RUSAGE_SELF)
    return {"result": result, "wall_seconds": round(time.time() - t0, 1),
            "cpu_seconds": round((c1.ru_utime - c0.ru_utime) + (c1.ru_stime - c0.ru_stime), 1),
            "gpu_util_samples": [s[0] for s in samples], "gpu_mem_mib_max": max((s[1] for s in samples), default=None),
            "gpu_sampling_errors": errors[:5], "n_gpu_sampling_errors": len(errors)}


@app.local_entrypoint()
def pilot_layer(analysis_id: str, config: str, layer: int, commit: str = ""):
    """Rerun one layer of a finished analysis (its saved run_config.json) under a new pilot id, with profiling; compare
    every number the layer reports with that analysis's own L<layer>.json on the Volume. Writes
    results/pilot_layer_<stamp>.json."""
    cli = load_config(config)
    check_volume(cli)  # before reading the Volume: a config on another Volume names PROBE_SHIFT_ACTS_VOLUME
    run = cli["extraction_run"]
    saved = json.loads(b"".join(acts_vol.read_file(f"{run}__q1gpu/{analysis_id}/run_config.json")))
    cfg, man = saved["config"], saved["manifest"]
    check_volume(cfg)
    if saved["subsets_file"] or saved["tool_errors_file"]:
        raise ValueError("pilot_layer supports analyses run without --subsets and --tool-errors")
    prov = launch_provenance(cfg, commit, config)  # before anything runs: a mismatch stops here, and the hash is the code launched
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    cfg = cfg | {"analysis_id": f"{stamp}_pilot_L{layer}_of_{analysis_id}"}  # its own directory: nothing is overwritten
    commands = wait(editor_commands.spawn(cfg)) if saved.get("editor_variants") else {}
    wait(pack_layer.spawn(cfg, layer, man))  # reuses the analysis's packed features (same manifest)
    out = wait(profiled_layer.spawn(cfg, layer, man, commands))
    # The committed analysis's own output for this layer (gpu_layer's full result, run scores included).
    committed = json.loads(b"".join(acts_vol.read_file(f"{run}__q1gpu/{analysis_id}/L{layer}.json")))
    strip = lambda d: {k: v for k, v in d.items() if k not in ("seconds",)}
    same = json.dumps(strip(out["result"]), sort_keys=True) == json.dumps(strip(committed), sort_keys=True)
    u = out["gpu_util_samples"]
    summary = {"analysis_id": analysis_id, "layer": layer, "reproduces_committed_layer": same,
               "wall_seconds": out["wall_seconds"], "cpu_seconds": out["cpu_seconds"],
               "gpu_busy_share": round(sum(x > 0 for x in u) / len(u), 3) if u else None,  # None: sampling failed
               "gpu_util_mean": round(sum(u) / len(u), 1) if u else None, "gpu_mem_mib_max": out["gpu_mem_mib_max"],
               "n_gpu_samples": len(u), "n_gpu_sampling_errors": out["n_gpu_sampling_errors"],
               "committed_layer_seconds": committed.get("seconds"), "provenance": prov}
    path = HERE / "results" / f"pilot_layer_{cfg['analysis_id']}.json"  # the pilot's own id: layer and source included
    if path.exists():
        raise FileExistsError(f"{path} exists; results/ is append-only")
    path.write_text(json.dumps(summary | {"result": out["result"], "gpu_util_samples": u}, indent=1))
    print(json.dumps(summary, indent=1))
    if not same:
        diffs = [k for k in committed if k != "seconds" and json.dumps(committed[k], sort_keys=True) != json.dumps(out["result"].get(k), sort_keys=True)]
        print(f"DIFFERS from the committed layer in: {diffs}")
    print(f"-> {path}")
