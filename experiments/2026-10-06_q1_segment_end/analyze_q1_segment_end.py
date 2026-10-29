"""Q1 on segment-end features: can a linear probe on Qwen3-32B activations tell real from simulated attack runs?

Reads the segment-end extraction (Modal Volume) and runs, per layer, on CPU in Modal:
  - real_vs_sim: real attack epoch 1 vs simulated attack, on the paired cells of splits/exclusions_v3.json.
  - null_e1_vs_e2: real epoch 1 vs real epoch 2 (same tasks, both real); should be ~0.5.
Two separately trained probes, as the project doc scores agent tokens and tool-output tokens separately:
  - output_end: one example per tool output, read at its last content token ("do the outputs look simulated?");
  - call_end: one example per tool call, read at the call's last content token ("does the agent's own text read
    differently after it has seen simulated outputs?").
Each: standardized features, L2 logistic regression, out-of-fold over the shared main-task folds; a run's score is the
mean of its step scores.
Every AUROC gets a 95% interval from resampling main tasks. Repeated with editor-tool (`text_editor`) outputs removed.
  - Label-shuffle null: 100 refits; in each, every main task's runs swap labels (real <-> sim) with probability 1/2.
  - Slices: base200 vs overlap800 (which honest run the simulator was shown), cells with no run over 32k,
    and step-level AUROC by the step's token position (0-8k, 8-16k, 16-32k) next to a position-only baseline.
  - Sensitivity: the same probe at output_marker and call_marker (template tokens, where the BashArena
    end-of-sequence artifact could reappear, and where the output's trailing newline is visible).
  - Random-direction baseline: 1,000 random unit directions in the standardized feature space; per test fold, the one
    whose training-fold AUROC is farthest from 0.5 (signed) scores the test fold.
  - Held-out environment: the probe trained on nine environments and tested on the tenth (leave-one-environment-out
    folds from the folds file), pooled and per environment, for real vs sim and for the e1-vs-e2 null.
Layer-free baselines, scored the same way: output length, tool-output perplexity (from the saved log-probs) and
token position. Writes results JSON and run-level scores locally; step-level scores go to the Volume.

  uv run modal run --detach experiments/2026-10-06_q1_segment_end/analyze_q1_segment_end.py::main [--test]
  uv run modal run --detach experiments/2026-10-06_q1_segment_end/analyze_q1_segment_end.py::headline   # after main

All computation runs in Modal CPU containers; each layer's result is also written to the Volume, so a lost laptop
connection loses nothing (the entrypoint retries network errors while it waits).
"""

import json
import socket
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q1-segment-end_cpu_kill-2026-11-06"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
TIMEOUT_S = 3 * 3600
ARMS = ["real-attack-e1", "real-attack-e2", "sim-attack"]
COMPARISONS = {"real_vs_sim": ("real-attack-e1", "sim-attack"), "null_e1_vs_e2": ("real-attack-e1", "real-attack-e2")}
VARIANTS = ["all_outputs", "no_editor_outputs"]
EDITOR = "text_editor"

app = modal.App(APP)
acts_vol = modal.Volume.from_name(ACTS, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy", "scikit-learn==1.7.2", "joblib", "threadpoolctl")


@app.function(image=image, volumes={"/acts": acts_vol}, cpu=8, memory=16384, timeout=1800)
def manifest(cfg: dict) -> dict:
    """The frozen list of runs every worker reads: (arm, filename) for each paired cell read in all three arms."""
    run_dir = Path("/acts") / cfg["extraction_run"]
    cells = set(cfg["paired_cells"])
    found, unreadable = {}, []
    for arm in ARMS:
        for p in sorted((run_dir / "acts" / arm).glob("*.json")):
            try:
                m = json.loads(p.read_text())
            except json.JSONDecodeError:  # a file mid-commit while the extraction is still running
                unreadable.append(f"{arm}/{p.name}")
                continue
            if m["cell"] in cells:
                if (arm, m["cell"]) in found:
                    raise ValueError(f"two {arm} runs for cell {m['cell']}")
                found[(arm, m["cell"])] = m["filename"]
    complete = sorted(c for c in cells if all((a, c) in found for a in ARMS))
    return {"runs": [(a, found[(a, c)]) for c in complete for a in ARMS], "cells_missing_an_arm": sorted(cells - set(complete)),
            "unreadable": unreadable}


def load_runs(cfg: dict, man: dict) -> list[dict]:
    """Metadata of the manifest's runs, in its order."""
    run_dir = Path("/acts") / cfg["extraction_run"]
    runs = []
    for arm, name in man["runs"]:
        m = json.loads((run_dir / "acts" / arm / f"{name}.json").read_text())
        runs.append(m | {"fold": cfg["main_task_to_fold"][m["main_task"]]})
    return runs


def step_table(runs: list[dict], kind: str):
    """One row per kept position of this kind: run index, npz row, step, action, tool, token position."""
    import numpy as np

    rows = [(i, p["row"], p["step"], p["action"], p["tool"] or "", p["token"])
            for i, m in enumerate(runs) for p in m["positions"] if p["kind"] == kind and p["kept"]]
    cols = list(zip(*rows)) if rows else [[]] * 6
    return {"run": np.array(cols[0], int), "row": np.array(cols[1], int), "step": np.array(cols[2], int),
            "action": np.array(cols[3], int), "tool": np.array(cols[4], object), "token": np.array(cols[5], int)}


class Scorer:
    """AUROCs with main-task bootstrap intervals, at step and run level, for one comparison."""

    def __init__(self, cfg, runs, comparison):
        import numpy as np

        neg, pos = COMPARISONS[comparison]
        self.np = np
        if not runs:
            raise ValueError("no paired cell has all three attack arms extracted yet")
        self.runs_in = np.array([m["arm"] in (neg, pos) for m in runs], bool)
        self.y_run = np.array([float(m["arm"] == pos) for m in runs])
        self.main = np.array([m["main_task"] for m in runs], object)
        self.cell = np.array([m["cell"] for m in runs], object)
        self.fold = np.array([m["fold"] for m in runs], int)
        tasks = sorted(set(self.main[self.runs_in]))
        self.seed, self.n_boot = cfg["seed"], cfg["bootstrap"]
        self.task_id = {t: i for i, t in enumerate(tasks)}
        self.neg, self.pos = neg, pos

    def draws(self, n_tasks):
        """Bootstrap resamples of n_tasks main tasks (the same draws for every metric on the same task set)."""
        rng = self.np.random.default_rng(self.seed)
        return [rng.choice(n_tasks, n_tasks) for _ in range(self.n_boot)]

    def auc_ci(self, y, s, task):
        """AUROC and 95% interval, resampling the main tasks present in this subset (task: one id per example)."""
        from sklearn.metrics import roc_auc_score

        np = self.np
        ok = ~np.isnan(s)
        y, s, task = y[ok], s[ok], task[ok]
        if len(np.unique(y)) < 2:
            return None
        uniq, inv = np.unique(task, return_inverse=True)
        order = np.argsort(inv, kind="stable")
        bounds = np.searchsorted(inv[order], np.arange(len(uniq) + 1))
        vals = []
        for b in self.draws(len(uniq)):
            idx = np.concatenate([order[bounds[t]:bounds[t + 1]] for t in b])
            if len(np.unique(y[idx])) == 2:
                vals.append(roc_auc_score(y[idx], s[idx]))
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return {"auroc": round(float(roc_auc_score(y, s)), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)], "n": int(len(y))}

    def run_scores(self, step_run, step_score):
        """Mean step score per run (NaN for a run with no steps)."""
        np = self.np
        tot = np.bincount(step_run, weights=step_score, minlength=len(self.y_run))
        cnt = np.bincount(step_run, minlength=len(self.y_run))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)

    def pairwise(self, run_score, keep_cells=None):
        """Share of cells whose positive run outscores its negative run (ties count half), with a task bootstrap."""
        np = self.np
        by_cell = {}
        for i in np.flatnonzero(self.runs_in):
            by_cell.setdefault(self.cell[i], {})[self.y_run[i]] = (run_score[i], self.task_id[self.main[i]])
        wins, tasks = [], []
        for c, d in by_cell.items():
            if (keep_cells is not None and c not in keep_cells) or len(d) < 2 or np.isnan(d[1.0][0]) or np.isnan(d[0.0][0]):
                continue
            wins.append(1.0 if d[1.0][0] > d[0.0][0] else 0.5 if d[1.0][0] == d[0.0][0] else 0.0)
            tasks.append(d[1.0][1])
        if not wins:
            return None
        wins = np.array(wins)
        uniq, tasks = np.unique(np.array(tasks, int), return_inverse=True)
        vals = []
        counts = np.bincount(tasks, minlength=len(uniq))
        sums = np.bincount(tasks, weights=wins, minlength=len(uniq))
        for b in self.draws(len(uniq)):
            n = counts[b].sum()
            if n:
                vals.append(sums[b].sum() / n)
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return {"accuracy": round(float(wins.mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)], "n_pairs": int(len(wins))}

    def report(self, st, step_score, cfg, slices=True):
        """Step- and run-level AUROC (and pairwise accuracy) for out-of-fold step scores; with slices if asked."""
        np = self.np
        in_cmp = self.runs_in[st["run"]] & ~np.isnan(step_score)
        y_step = self.y_run[st["run"]]
        task_step = np.array([self.task_id.get(t, -1) for t in self.main[st["run"]]])
        rs = self.run_scores(st["run"][in_cmp], step_score[in_cmp])
        runs = np.flatnonzero(self.runs_in & ~np.isnan(rs))
        task_run = np.array([self.task_id[t] for t in self.main[runs]])
        out = {"step": self.auc_ci(y_step[in_cmp], step_score[in_cmp], task_step[in_cmp]),
               "run": self.auc_ci(self.y_run[runs], rs[runs], task_run),
               "pairwise": self.pairwise(rs)}
        if not slices:
            return out
        groups = {"base200": set(cfg["base200"]), "overlap800": set(cfg["overlap800"]), "under_32k": set(cfg["under_32k"])}
        for name, cells in groups.items():
            r = runs[np.isin(self.cell[runs], list(cells))]
            s = in_cmp & np.isin(self.cell[st["run"]], list(cells))
            out[name] = {"step": self.auc_ci(y_step[s], step_score[s], task_step[s]),
                         "run": self.auc_ci(self.y_run[r], rs[r], np.array([self.task_id[t] for t in self.main[r]])),
                         "pairwise": self.pairwise(rs, cells)}
        for lo, hi in cfg["position_bands"]:
            s = in_cmp & (st["token"] >= lo) & (st["token"] < hi)
            pos_score = oriented(st["token"].astype(float), y_step, self.fold[st["run"]], in_cmp)
            out[f"band_{lo}_{hi}"] = {"step": self.auc_ci(y_step[s], step_score[s], task_step[s]),
                                      "position_only": self.auc_ci(y_step[s], pos_score[s], task_step[s])}
        return out


def oriented(x, y, fold, use):
    """A one-number baseline, signed per test fold by its AUROC on the training folds."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    out = np.full(len(x), np.nan)
    for f in np.unique(fold[use]):
        tr, te = use & (fold != f) & ~np.isnan(x), use & (fold == f)
        sign = 1.0 if roc_auc_score(y[tr], x[tr]) >= 0.5 else -1.0
        out[te] = sign * x[te]
    return out


def fit_fold(path, f, y_tr, C, max_iter):
    """Fit on one fold's standardized training matrix (memory-mapped) and score its test matrix."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits

    with threadpool_limits(1):
        Xtr = np.load(f"{path}/tr_{f}.npy", mmap_mode="r")
        Xte = np.load(f"{path}/te_{f}.npy", mmap_mode="r")
        clf = LogisticRegression(C=C, max_iter=max_iter).fit(Xtr, y_tr)
        return clf.decision_function(Xte), int(clf.n_iter_[0])


@app.function(image=image, volumes={"/acts": acts_vol}, cpu=32, memory=131072, timeout=TIMEOUT_S)
def analyze_layer(cfg: dict, layer: int, man: dict) -> dict:
    import shutil
    import tempfile

    import numpy as np
    from joblib import Parallel, delayed
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    run_dir = Path("/acts") / cfg["extraction_run"]
    runs = load_runs(cfg, man)
    kinds = cfg["positions"] + cfg["sensitivity_positions"]
    tables = {k: step_table(runs, k) for k in kinds}
    feats = {k: np.zeros((len(t["run"]), 5120), np.float32) for k, t in tables.items()}
    for i, m in enumerate(runs):
        arr = np.load(run_dir / "acts" / m["arm"] / f"{m['filename']}.npz")[f"L{layer}"]
        for k, t in tables.items():
            sel = np.flatnonzero(t["run"] == i)
            feats[k][sel] = arr[t["row"][sel]]
    nonfinite = {k: int((~np.isfinite(x)).sum()) for k, x in feats.items()}
    fold_of = {"task": np.array([m["fold"] for m in runs]),
               "environment": np.array([cfg["environment_to_fold"][m["environment"]] for m in runs])}
    rng_dirs = np.random.default_rng(cfg["seed"] + 1000 + layer)
    dirs = rng_dirs.standard_normal((cfg["random_directions"], 5120)).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    scratch = tempfile.mkdtemp()

    def best_random_direction(path, f, y_tr):
        """Project onto every random direction, pick the one whose training-fold AUROC is farthest from 0.5 (signed),
        and score the test fold with it."""
        from scipy.stats import rankdata

        Xtr = np.load(f"{path}/tr_{f}.npy", mmap_mode="r")
        Xte = np.load(f"{path}/te_{f}.npy", mmap_mode="r")
        ranks = rankdata(np.asarray(Xtr) @ dirs.T, axis=0)
        n1 = y_tr.sum()
        auc = (ranks[y_tr == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * (len(y_tr) - n1))
        j = int(np.argmax(np.abs(auc - 0.5)))
        return np.sign(auc[j] - 0.5) * (np.asarray(Xte) @ dirs[j])

    def oof(kind, comparison, variant, draws=None, fold_by="task", random_dirs=False):
        """Out-of-fold step scores (folds over main tasks, or leave-one-environment-out). With draws, the label-shuffle
        null: one score vector per draw. With random_dirs, also the best-of-random-directions scores."""
        t, X = tables[kind], feats[kind]
        neg, pos = COMPARISONS[comparison]
        arm = np.array([runs[r]["arm"] for r in t["run"]])
        use = np.isin(arm, [neg, pos]) & ((t["tool"] != EDITOR) if variant == "no_editor_outputs" else True)
        fold = fold_of[fold_by][t["run"]]
        folds = sorted(set(fold[use]))
        path = f"{scratch}/{kind}_{comparison}_{variant}_{fold_by}"
        Path(path).mkdir(exist_ok=True)
        for f in folds:  # standardize once per fold (labels are not involved), reused by every draw
            tr, te = use & (fold != f), use & (fold == f)
            sc = StandardScaler().fit(X[tr])
            np.save(f"{path}/tr_{f}.npy", sc.transform(X[tr]).astype(np.float32))
            np.save(f"{path}/te_{f}.npy", sc.transform(X[te]).astype(np.float32))
        y_sets = draws if draws is not None else [np.array([float(m["arm"] == pos) for m in runs])]
        jobs = [(d, f) for d in range(len(y_sets)) for f in folds]
        res = Parallel(n_jobs=cfg["n_jobs"])(delayed(fit_fold)(path, f, y_sets[d][t["run"]][use & (fold != f)], cfg["C"], cfg["max_iter"])
                                             for d, f in jobs)
        out = [np.full(len(t["run"]), np.nan) for _ in y_sets]
        iters = []
        for (d, f), (score, n_iter) in zip(jobs, res):
            out[d][use & (fold == f)] = score
            iters.append(n_iter)
        rand = None
        if random_dirs:
            rand = np.full(len(t["run"]), np.nan)
            for f in folds:
                rand[use & (fold == f)] = best_random_direction(path, f, y_sets[0][t["run"]][use & (fold != f)])
        shutil.rmtree(path)
        return out, max(iters), rand

    result = {"layer": layer, "n_runs": len(runs), "nonfinite": nonfinite, "results": {}, "sensitivity": {}}
    step_scores = {}
    env = np.array([m["environment"] for m in runs], object)
    for pos in cfg["positions"]:  # separately trained probes: tool outputs (output_end) and agent tool calls (call_end)
        t = tables[pos]
        result["results"][pos] = {}
        for comparison in COMPARISONS:
            scorer = Scorer(cfg, runs, comparison)
            res = {}
            for variant in VARIANTS:
                (s,), iters, rand = oof(pos, comparison, variant, random_dirs=True)
                res[variant] = scorer.report(t, s, cfg) | {"max_iter_used": iters,
                                                           "best_random_direction": scorer.report(t, rand, cfg, slices=False)}
                step_scores[(pos, comparison, variant)] = s
                step_scores[(pos, comparison, f"{variant}__random_direction")] = rand
                print(f"L{layer} {pos} {comparison} {variant}: run {res[variant]['run']}", flush=True)
            # Held-out environment: train on nine environments, test on the tenth; pooled over all held-out predictions,
            # and per environment (run level, resampling that environment's main tasks).
            (s,), _, _ = oof(pos, comparison, "all_outputs", fold_by="environment")
            step_scores[(pos, comparison, "all_outputs__held_out_environment")] = s
            held = scorer.report(t, s, cfg, slices=False)
            rs = scorer.run_scores(t["run"][~np.isnan(s)], s[~np.isnan(s)])
            held["per_environment"] = {}
            for e in sorted(cfg["environment_to_fold"]):
                r = np.flatnonzero(scorer.runs_in & (env == e) & ~np.isnan(rs))
                held["per_environment"][e] = {"run": scorer.auc_ci(scorer.y_run[r], rs[r], scorer.main[r]), "n_runs": int(len(r))}
            res["held_out_environment"] = held
            print(f"L{layer} {pos} {comparison} held-out environment: run {held['run']}", flush=True)
            if comparison == "real_vs_sim":
                # Label-shuffle null: per main task, with probability 1/2 every one of its cells swaps which run is called
                # simulated (one flip per main task, as the folds and the bootstrap group by main task); steps follow their run.
                rng = np.random.default_rng(cfg["seed"] + 1)
                tasks = sorted({runs[i]["main_task"] for i in np.flatnonzero(scorer.runs_in)})
                draws = []
                for _ in range(cfg["shuffle_null"]):
                    flip = dict(zip(tasks, rng.random(len(tasks)) < 0.5))
                    draws.append(np.array([1 - y if (scorer.runs_in[i] and flip[runs[i]["main_task"]]) else y
                                           for i, y in enumerate(scorer.y_run)]))
                for variant in VARIANTS:
                    null_scores, _, _ = oof(pos, comparison, variant, draws=draws)
                    null_run, null_step = [], []
                    for d, s in zip(draws, null_scores):
                        ok = ~np.isnan(s)
                        rs = scorer.run_scores(t["run"][ok], s[ok])
                        r = np.flatnonzero(scorer.runs_in & ~np.isnan(rs))
                        null_run.append(roc_auc_score(d[r], rs[r]))
                        null_step.append(roc_auc_score(d[t["run"][ok]], s[ok]))
                    for level, null in (("run", np.array(null_run)), ("step", np.array(null_step))):
                        obs = res[variant][level]["auroc"]
                        res[variant][level]["label_shuffle_null"] = {
                            "n": len(null), "mean": round(float(null.mean()), 4), "p95": round(float(np.percentile(null, 95)), 4),
                            "p99": round(float(np.percentile(null, 99)), 4), "p_value": round(float((1 + (null >= obs).sum()) / (1 + len(null))), 4)}
                    print(f"L{layer} {pos} shuffle null {variant} done", flush=True)
            result["results"][pos][comparison] = res
    scorer = Scorer(cfg, runs, "real_vs_sim")
    for kind in cfg["sensitivity_positions"]:  # template markers: the end-of-sequence-style check
        (s,), _, _ = oof(kind, "real_vs_sim", "all_outputs")
        result["sensitivity"][kind] = scorer.report(tables[kind], s, cfg, slices=False)

    # Step-level scores to the Volume (too big for git): one row per position, for the viewer and checks.
    out_dir = run_dir.parent / f"{cfg['extraction_run']}__q1" / cfg["analysis_id"] / f"L{layer}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for pos in cfg["positions"]:
        t = tables[pos]
        np.savez(out_dir / f"step_scores_{pos}.npz", run=t["run"], step=t["step"], action=t["action"], token=t["token"],
                 **{f"{c}__{v}": s for (p_, c, v), s in step_scores.items() if p_ == pos})
    (out_dir / "runs.json").write_text(json.dumps([{k: m[k] for k in ("arm", "key", "cell", "main_task", "fold")} for m in runs]))
    shutil.rmtree(scratch)
    # Run-level scores come back for the local results file.
    result["run_scores"] = {}
    for (pos, c, v), s in step_scores.items():
        sc = Scorer(cfg, runs, c)
        ok = ~np.isnan(s)
        rs = sc.run_scores(tables[pos]["run"][ok], s[ok])
        result["run_scores"][f"{pos}__{c}__{v}"] = [None if np.isnan(x) else round(float(x), 5) for x in rs]
    result["runs"] = [{k: m[k] for k in ("arm", "key", "cell", "main_task", "fold", "sim_reference", "truncated")} for m in runs]
    (out_dir / "result.json").write_text(json.dumps(result))
    acts_vol.commit()
    return result


@app.function(image=image, volumes={"/acts": acts_vol}, cpu=8, memory=32768, timeout=TIMEOUT_S)
def baselines(cfg: dict, man: dict) -> dict:
    """Layer-free baselines, scored like the probe: output length, tool-output perplexity, token position."""
    import numpy as np

    run_dir = Path("/acts") / cfg["extraction_run"]
    runs = load_runs(cfg, man)
    t = step_table(runs, "output_end")  # layer-free baselines are about tool outputs
    n_tok, nll = np.zeros(len(t["run"])), np.zeros(len(t["run"]))
    # Run-level totals over every tool-output token read, including an output cut by the 32k limit (whose
    # output_end step is not kept): per variant, total tokens and summed negative log-prob.
    run_tok = {v: np.zeros(len(runs)) for v in VARIANTS}
    run_nll = {v: np.zeros(len(runs)) for v in VARIANTS}
    for i, m in enumerate(runs):
        lp = np.load(run_dir / "acts" / m["arm"] / f"{m['filename']}.npz")["logprobs"]
        outs = {(o["step"], o["action"]): o for o in m["outputs"]}
        for o in m["outputs"]:
            seg = lp[o["lp_start"]:o["lp_end"]]
            for v in VARIANTS:
                if v == "all_outputs" or o["tool"] != EDITOR:
                    run_tok[v][i] += len(seg)
                    run_nll[v][i] -= seg.sum()
        for j in np.flatnonzero(t["run"] == i):
            o = outs[(t["step"][j], t["action"][j])]
            seg = lp[o["lp_start"]:o["lp_end"]]
            n_tok[j], nll[j] = len(seg), (-seg.mean() if len(seg) else np.nan)
    out = {"results": {}}
    for comparison in COMPARISONS:
        scorer = Scorer(cfg, runs, comparison)
        neg, pos = COMPARISONS[comparison]
        arm = np.array([runs[r]["arm"] for r in t["run"]])
        y = scorer.y_run[t["run"]]
        fold = scorer.fold[t["run"]]
        res = {}
        for variant in VARIANTS:
            use = np.isin(arm, [neg, pos]) & ((t["tool"] != EDITOR) if variant == "no_editor_outputs" else True)
            r = {}
            for name, x in (("output_length", n_tok), ("perplexity", nll), ("position", t["token"].astype(float))):
                s = oriented(np.where(use, x, np.nan), y, fold, use & ~np.isnan(x))
                r[name] = scorer.report(t, s, cfg, slices=(name != "position"))
            # Run-level totals, as in the pooled pilot: total tool-output tokens, and mean NLL over all output tokens.
            tot = run_tok[variant]
            nll_tok = np.where(tot > 0, run_nll[variant] / np.maximum(tot, 1), np.nan)
            rr = np.flatnonzero(scorer.runs_in)
            task_run = np.array([scorer.task_id[x] for x in scorer.main[rr]])
            for name, x in (("total_output_tokens", tot), ("run_perplexity", nll_tok)):
                s = oriented(x[rr], scorer.y_run[rr], scorer.fold[rr], np.ones(len(rr), bool))
                r[name] = {"run": scorer.auc_ci(scorer.y_run[rr], s, task_run)}
            res[variant] = r
        out["results"][comparison] = res
    out_dir = run_dir.parent / f"{cfg['extraction_run']}__q1" / cfg["analysis_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baselines.json").write_text(json.dumps(out))
    acts_vol.commit()
    return out


@app.local_entrypoint()
def main(test: bool = False):
    """`--test` runs one layer with 5 shuffles and 100 bootstrap draws on whatever is extracted so far."""
    import yaml

    cfg = yaml.safe_load((HERE / "config" / "q1_segment_end.yaml").read_text())
    if test:
        cfg |= {"layers": [24], "shuffle_null": 5, "bootstrap": 100, "random_directions": 100}
    splits = HERE.parent.parent / "splits"
    folds = json.loads((splits / cfg["folds_file"]).read_text())
    cfg["main_task_to_fold"] = folds["task_folds"]["main_task_to_fold"]
    cfg["environment_to_fold"] = folds["environment_folds"]["environment_to_fold"]
    ex = json.loads((splits / cfg["exclusions_file"]).read_text())
    cfg["paired_cells"] = ex["paired_attack_cells"]["cells"]
    ref = ex["paired_attack_cells"]["sim_reference"]
    cfg["base200"] = [c for c in cfg["paired_cells"] if ref[c] == "base200"]
    cfg["overlap800"] = [c for c in cfg["paired_cells"] if ref[c] == "overlap800"]
    cfg["under_32k"] = ex["paired_attack_cells_under_32k"]["cells"]
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    cfg["analysis_id"] = stamp + ("_test" if test else "")  # step-level scores: <extraction_run>__q1/<analysis_id>/
    man = manifest.remote(cfg)
    if not man["runs"]:
        raise SystemExit("no paired cell has all three attack arms extracted yet; rerun when the extraction is further along")
    print(f"{len(man['runs'])} runs in {len(man['runs']) // 3} paired cells; {len(man['cells_missing_an_arm'])} cells missing an arm; "
          f"{len(man['unreadable'])} unreadable metadata files")
    if (man["unreadable"] or man["cells_missing_an_arm"]) and not test:  # as the GPU runner: never a silently smaller cohort
        raise ValueError(f"incomplete extraction: {len(man['unreadable'])} unreadable, {len(man['cells_missing_an_arm'])} "
                         f"cells missing an arm (e.g. {man['cells_missing_an_arm'][:3]})")
    calls = {"baselines": baselines.spawn(cfg, man)} | {l: analyze_layer.spawn(cfg, l, man) for l in cfg["layers"]}
    print(f"analysis {cfg['analysis_id']}: spawned", {k: c.object_id for k, c in calls.items()}, flush=True)

    def wait(call):
        for _ in range(120):  # network trouble on our side: retry for up to an hour; remote errors are raised
            try:
                return call.get()
            except (socket.gaierror, ConnectionError, TimeoutError, modal.exception.ConnectionError, modal.exception.ServiceError) as e:
                print(f"lost the connection ({e!r}); retrying in 30 s", flush=True)
                time.sleep(30)
        raise TimeoutError("no connection to Modal for an hour; the results are on the Volume")

    base = wait(calls["baselines"])
    by_layer = {l: wait(calls[l]) for l in cfg["layers"]}
    keys = [m["key"] for m in by_layer[cfg["layers"][0]]["runs"]]
    assert all([m["key"] for m in r["runs"]] == keys for r in by_layer.values()), "layers read different runs"
    result = {"manifest": man, "baselines": base, "by_layer": {l: {k: v for k, v in r.items() if k not in ("run_scores", "runs")}
                                                     for l, r in by_layer.items()}}
    (HERE / "results").mkdir(exist_ok=True)
    tag = "_test" if test else ""
    path = HERE / "results" / f"q1_segment_end_{stamp}{tag}.json"
    cfg_out = {k: v for k, v in cfg.items() if k not in ("main_task_to_fold", "paired_cells", "base200", "overlap800", "under_32k")}
    path.write_text(json.dumps(result | {"config": cfg_out | {"modal": {"app": APP, "activations_volume": ACTS}}}, indent=1))
    first = by_layer[cfg["layers"][0]]
    with open(HERE / "results" / f"run_scores_{stamp}{tag}.jsonl", "w") as f:
        for i, m in enumerate(first["runs"]):
            row = m | {f"L{l}__{k}": by_layer[l]["run_scores"][k][i] for l in cfg["layers"] for k in by_layer[l]["run_scores"]}
            f.write(json.dumps(row) + "\n")
    fmt = lambda d: "n/a" if not d else f"{d['auroc']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]"
    for pos, c in [(p, c) for p in cfg["positions"] for c in COMPARISONS]:
        print(f"\n=== {pos} {c}")
        for v in VARIANTS:
            for l in cfg["layers"]:
                r = result["by_layer"][l]["results"][pos][c][v]
                null = r["run"].get("label_shuffle_null") if r["run"] else None
                extra = f"  shuffle p95 {null['p95']:.3f} p={null['p_value']:.3f}" if null else ""
                print(f"  [{v}] L{l:<2} run {fmt(r['run'])}  step {fmt(r['step'])}{extra}")
    print(f"\nresults -> {path}")


# ---- Headline layer, chosen without test folds -------------------------------------------------------------------
# Within each outer fold, an inner CV over its training folds scores every layer; the earliest layer within
# `headline_tolerance` of that fold's best inner run-level AUROC is the one whose out-of-fold predictions score the
# outer fold. Every comparison (the e1-vs-e2 null included) goes through the same selection.


def inner_fit(path, tr, te, y_tr, C, max_iter):
    """Standardize on tr rows of the memory-mapped matrix, fit, and score te rows."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    with threadpool_limits(1):
        X = np.load(path, mmap_mode="r")
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, max_iter=max_iter).fit(sc.transform(X[tr]), y_tr)
        return clf.decision_function(sc.transform(X[te]))


@app.function(image=image, volumes={"/acts": acts_vol}, cpu=32, memory=131072, timeout=TIMEOUT_S)
def inner_cv_layer(cfg: dict, layer: int, man: dict) -> dict:
    """{position: {comparison: {variant: {outer fold: inner run-level AUROC}}}} for one layer."""
    import tempfile

    import numpy as np
    from joblib import Parallel, delayed
    from sklearn.metrics import roc_auc_score

    run_dir = Path("/acts") / cfg["extraction_run"]
    runs = load_runs(cfg, man)
    tables = {k: step_table(runs, k) for k in cfg["positions"]}
    feats = {k: np.zeros((len(t["run"]), 5120), np.float32) for k, t in tables.items()}
    for i, m in enumerate(runs):
        arr = np.load(run_dir / "acts" / m["arm"] / f"{m['filename']}.npz")[f"L{layer}"]
        for k, t in tables.items():
            sel = np.flatnonzero(t["run"] == i)
            feats[k][sel] = arr[t["row"][sel]]
    fold_of_run = np.array([m["fold"] for m in runs])
    scratch = tempfile.mkdtemp()
    out = {}
    for pos, t in tables.items():
        path = f"{scratch}/{pos}.npy"
        np.save(path, feats.pop(pos))
        out[pos] = {}
        for comparison, (neg, posarm) in COMPARISONS.items():
            y_run = np.array([float(m["arm"] == posarm) for m in runs])
            arm = np.array([runs[r]["arm"] for r in t["run"]])
            out[pos][comparison] = {}
            for variant in VARIANTS:
                use = np.isin(arm, [neg, posarm]) & ((t["tool"] != EDITOR) if variant == "no_editor_outputs" else True)
                fold = fold_of_run[t["run"]]
                folds = sorted(set(fold[use]))
                jobs = [(f, g) for f in folds for g in folds if g != f]
                trs = {(f, g): np.flatnonzero(use & (fold != f) & (fold != g)) for f, g in jobs}
                tes = {(f, g): np.flatnonzero(use & (fold == g)) for f, g in jobs}
                scores = Parallel(n_jobs=16)(delayed(inner_fit)(path, trs[j], tes[j], y_run[t["run"][trs[j]]], cfg["C"], cfg["max_iter"])
                                             for j in jobs)
                res = {}
                for f in folds:
                    run_s, run_y = [], []
                    for j, s in zip(jobs, scores):
                        if j[0] != f:
                            continue
                        r = t["run"][tes[j]]
                        tot = np.bincount(r, weights=s, minlength=len(runs))
                        cnt = np.bincount(r, minlength=len(runs))
                        keep = np.flatnonzero(cnt > 0)
                        run_s.append(tot[keep] / cnt[keep])
                        run_y.append(y_run[keep])
                    res[int(f)] = round(float(roc_auc_score(np.concatenate(run_y), np.concatenate(run_s))), 4)
                out[pos][comparison][variant] = res
                print(f"L{layer} {pos} {comparison} {variant}: inner AUROC by outer fold {res}", flush=True)
    return out


@app.function(image=image, volumes={"/acts": acts_vol}, cpu=8, memory=32768, timeout=TIMEOUT_S)
def headline_scores(cfg: dict, man: dict, inner: dict) -> dict:
    """Assemble the headline out-of-fold run scores from each outer fold's chosen layer and score them."""
    import numpy as np

    acts_vol.reload()
    runs = load_runs(cfg, man)
    base = Path("/acts") / f"{cfg['extraction_run']}__q1" / cfg["analysis_id"]
    per_layer = {l: json.loads((base / f"L{l}" / "result.json").read_text()) for l in cfg["layers"]}
    keys = [m["key"] for m in runs]
    for l, r in per_layer.items():
        if [m["key"] for m in r["runs"]] != keys:
            raise ValueError(f"layer {l} read different runs than the manifest")
    out = {}
    for pos in cfg["positions"]:
        out[pos] = {}
        for comparison in COMPARISONS:
            scorer = Scorer(cfg, runs, comparison)
            out[pos][comparison] = {}
            for variant in VARIANTS:
                chosen, rs = {}, np.full(len(runs), np.nan)
                for f in sorted({m["fold"] for m in runs}):
                    by_layer = {l: inner[str(l)][pos][comparison][variant][f] for l in cfg["layers"]}
                    top = max(by_layer.values())
                    chosen[f] = min(l for l, a in by_layer.items() if a >= top - cfg["headline_tolerance"])
                    scores = per_layer[chosen[f]]["run_scores"][f"{pos}__{comparison}__{variant}"]
                    for i, m in enumerate(runs):
                        if m["fold"] == f and scores[i] is not None:
                            rs[i] = scores[i]
                r = np.flatnonzero(scorer.runs_in & ~np.isnan(rs))
                out[pos][comparison][variant] = {
                    "chosen_layer_by_fold": chosen, "inner_auroc": {str(l): inner[str(l)][pos][comparison][variant] for l in cfg["layers"]},
                    "run": scorer.auc_ci(scorer.y_run[r], rs[r], scorer.main[r]), "pairwise": scorer.pairwise(rs)}
    return out


@app.local_entrypoint()
def headline(results: str = ""):
    """Choose each probe's headline layer by inner CV on the training folds, for a finished analysis
    (default: the newest non-test results file)."""
    full = [p for p in sorted((HERE / "results").glob("q1_segment_end_*.json")) if not p.stem.endswith("_test")]
    path = Path(results) if results else full[-1]
    res = json.loads(path.read_text())
    cfg, man = res["config"], res["manifest"]
    splits = HERE.parent.parent / "splits"
    cfg["main_task_to_fold"] = json.loads((splits / cfg["folds_file"]).read_text())["task_folds"]["main_task_to_fold"]
    ex = json.loads((splits / cfg["exclusions_file"]).read_text())
    cfg["paired_cells"] = ex["paired_attack_cells"]["cells"]
    cfg.setdefault("headline_tolerance", 0.01)
    inner = {str(l): r for l, r in zip(cfg["layers"], inner_cv_layer.map([cfg] * len(cfg["layers"]), cfg["layers"],
                                                                         [man] * len(cfg["layers"])))}
    head = headline_scores.remote(cfg, man, inner)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = HERE / "results" / f"q1_headline_{stamp}.json"
    out.write_text(json.dumps({"source": str(path), "headline": head,
                               "rule": f"per outer fold, the earliest layer within {cfg['headline_tolerance']} of the best "
                                       "inner-CV run-level AUROC on that fold's training folds"}, indent=1))
    for pos, by_c in head.items():
        for c, by_v in by_c.items():
            for v, h in by_v.items():
                print(f"{pos} {c} {v}: headline run AUROC {h['run']}; layers by fold {h['chosen_layer_by_fold']}")
    print(f"headline -> {out}")
