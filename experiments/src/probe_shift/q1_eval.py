"""Shared Q1 evaluation: which runs and steps are scored, and AUROCs with main-task bootstrap intervals.

Used by the GPU analysis (experiments/2026-10-06_q1_segment_end/analyze_q1_gpu.py). numpy and scikit-learn are
imported lazily; activations are read from /acts (the Modal Volume mount).
"""

import json
from pathlib import Path

ARMS = ["real-attack-e1", "real-attack-e2", "sim-attack"]
EDITOR = "text_editor"
# Which runs a cell holds and what is compared, per design (cfg "design", default attack). The comparison keys stay the
# same in every design so downstream code and other readers (Qwen3.5) share them: "real_vs_sim" is the question and
# "null_e1_vs_e2" the real-vs-real control. Honest cells are main tasks: "real-honest" is a task's first real honest
# run by key and "real-honest-2" its second (the control, only where a task has two). Pooled holds both kinds of cell.
# Each comparison is (negative arms, positive arms).
_ATTACK = {"real_vs_sim": (("real-attack-e1",), ("sim-attack",)), "null_e1_vs_e2": (("real-attack-e1",), ("real-attack-e2",))}
_HONEST = {"real_vs_sim": (("real-honest",), ("sim-honest",)), "null_e1_vs_e2": (("real-honest",), ("real-honest-2",))}
DESIGNS = {
    "attack": {"comparisons": _ATTACK},
    "honest": {"comparisons": _HONEST},
    "pooled": {"comparisons": {k: (_ATTACK[k][0] + _HONEST[k][0], _ATTACK[k][1] + _HONEST[k][1]) for k in _ATTACK}},
}
COMPARISONS = _ATTACK  # the attack design's, for code written before designs


def comparisons(cfg: dict) -> dict:
    return DESIGNS[cfg.get("design", "attack")]["comparisons"]


def arm_dir(arm: str) -> str:
    """The extraction's folder for an arm label ("real-honest-2" is a second real-honest run)."""
    return "real-honest" if arm == "real-honest-2" else arm


def find_runs(cfg: dict) -> dict:
    """The frozen list of runs every worker reads, as (arm, filename): attack cells need e1, e2 and sim; honest cells
    need real-honest and sim-honest, plus real-honest-2 where the task has a second real honest run."""
    run_dir = Path("/acts") / cfg["extraction_run"]
    design = cfg.get("design", "attack")
    cells = set(cfg["paired_cells"])
    found, unreadable, honest_real = {}, [], {}
    dirs = (ARMS if design in ("attack", "pooled") else []) + (["real-honest", "sim-honest"] if design in ("honest", "pooled") else [])
    for arm in dirs:
        for p in sorted((run_dir / cfg.get("meta_dir", "acts") / arm).glob("*.json")):  # meta/ for all-layers runs
            try:
                m = json.loads(p.read_text())
            except json.JSONDecodeError:  # a file mid-commit while the extraction is still running
                unreadable.append(f"{arm}/{p.name}")
                continue
            if m["cell"] not in cells:
                continue
            if arm == "real-honest":  # several per task: ordered by key below
                honest_real.setdefault(m["cell"], []).append((m["key"], m["filename"]))
                continue
            if (arm, m["cell"]) in found:
                raise ValueError(f"two {arm} runs for cell {m['cell']}")
            found[(arm, m["cell"])] = m["filename"]
    every = cfg.get("honest_real_runs", "first") == "all"  # sensitivity: every real honest run of a task, no control
    expected = cfg.get("honest_real_keys", {})  # each honest cell's valid real-honest keys (splits): all must be read
    for c, rs in list(honest_real.items()):
        rs.sort()
        if sorted(k for k, _ in rs) != expected.get(c, sorted(k for k, _ in rs)):
            del honest_real[c]  # incomplete: the cell is reported missing, not silently re-paired
            continue
        found[("real-honest", c)] = rs[0][1]
        if len(rs) > 1 and not every:
            found[("real-honest-2", c)] = rs[1][1]
    runs, complete = [], []
    for c in sorted(cells):
        need = ARMS if ("real-attack-e1", c) in found or ("sim-attack", c) in found or ("real-attack-e2", c) in found \
            else ["real-honest", "sim-honest"]
        if all((a, c) in found for a in need):
            complete.append(c)
            runs += [(a, found[(a, c)]) for a in need + (["real-honest-2"] if need[0] == "real-honest" and ("real-honest-2", c) in found else [])]
            if every and need[0] == "real-honest":
                runs += [("real-honest", f) for _, f in honest_real[c][1:]]
    out = {"runs": runs, "cells_missing_an_arm": sorted(cells - set(complete)), "unreadable": unreadable}
    if cfg.get("match_cut_file"):  # part of the manifest, so packed features for a matched read are never reused unmatched
        out["match_cut_file"] = cfg["match_cut_file"]
    return out


def load_runs(cfg: dict, man: dict) -> list[dict]:
    """Metadata of the manifest's runs, in its order. With the manifest's `match_cut_file` ({"read": {"<arm>|<key>":
    [[kind, step, action], ...]}, "truncated": {"<arm>|<key>": bool}}: the positions another reader read, and whether it
    cut the transcript), every other position counts as not kept and `truncated` is that reader's."""
    run_dir = Path("/acts") / cfg["extraction_run"]
    match = json.loads((Path("/acts") / man["match_cut_file"]).read_text()) if man.get("match_cut_file") else None
    runs = []
    for arm, name in man["runs"]:
        m = json.loads((run_dir / cfg.get("meta_dir", "acts") / arm_dir(arm) / f"{name}.json").read_text())
        if match is not None:
            k = f"{m['arm']}|{m['key']}"  # the transcript's own arm (real-honest-2 is a real-honest run); a missing one raises
            read = {tuple(x) for x in match["read"][k]}
            m["positions"] = [p | {"kept": p["kept"] and (p["kind"], p["step"], p["action"]) in read} for p in m["positions"]]
            m["truncated"] = match["truncated"][k]
        runs.append(m | {"arm": arm, "fold": cfg["main_task_to_fold"][m["main_task"]]})
    return runs



def step_table(runs: list[dict], kind: str):
    """One row per kept position of this kind: run index, npz row, step, action, tool, token position, token text."""
    import numpy as np

    rows = [(i, p["row"], p["step"], p["action"], p["tool"] or "", p["token"], p.get("token_text") or "")
            for i, m in enumerate(runs) for p in m["positions"] if p["kind"] == kind and p["kept"]]
    cols = list(zip(*rows)) if rows else [[]] * 7
    return {"run": np.array(cols[0], int), "row": np.array(cols[1], int), "step": np.array(cols[2], int),
            "action": np.array(cols[3], int), "tool": np.array(cols[4], object), "token": np.array(cols[5], int),
            "token_text": np.array(cols[6], object)}


# call_end tokens that sit at a fixed slot rather than content: the final digit of `"timeout": 120` and the `n` of an
# escaped newline (viewer session, 2026-10-06), and from scrub level 2 on the `>` of the `<n>` placeholder a number in a
# tool-call argument becomes (protocol amendment 3).
FIXED_CALL_TOKENS = ("0", "n", ">")


def fixed_call_tokens(extraction_run: str) -> list[str]:
    """FIXED_CALL_TOKENS at an extraction's text level (`scrub<N>` in its run name; none means the original text): `>`
    is the `<n>` placeholder's only from level 2 on, and before that can be a call's real last character."""
    import re

    m = re.search(r"scrub(\d+)", extraction_run)
    return [t for t in FIXED_CALL_TOKENS if t != ">" or (m is not None and int(m.group(1)) >= 2)]



def boot_aucs(y, scores, task, draws):
    """For each bootstrap draw (main tasks sampled with replacement), the AUROC of each score vector on the resample,
    all draws at once: an example's weight is how often its task was drawn, and the AUROC is the weighted Mann-Whitney
    statistic with ties counted half, which is what roc_auc_score computes on the concatenated resample. Returns
    (valid [B]: the resample holds both classes, aucs [len(scores), B]). Runs on the GPU when there is one, else on all
    CPU cores; float64 throughout."""
    import numpy as np
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _, inv = np.unique(task, return_inverse=True)
    d = torch.as_tensor(np.stack(draws), device=dev)  # [B, n_tasks], the resampled task ids in draw order
    counts = torch.zeros(d.shape, dtype=torch.float64, device=dev).scatter_add_(1, d, torch.ones(d.shape, dtype=torch.float64, device=dev))
    w = counts[:, torch.as_tensor(inv, device=dev)]  # [B, n]: each example's multiplicity in each resample
    pos = torch.as_tensor(np.asarray(y) == 1, device=dev)
    wp, wn = w * pos, w * ~pos
    n_pos, n_neg = wp.sum(1), wn.sum(1)
    out = []
    for sc in scores:
        _, g = np.unique(sc, return_inverse=True)  # tie groups in ascending score order
        g = torch.as_tensor(g, device=dev)
        p = torch.zeros(w.shape[0], int(g.max()) + 1, dtype=torch.float64, device=dev).index_add_(1, g, wp)
        q = torch.zeros_like(p).index_add_(1, g, wn)
        u = (p * (q.cumsum(1) - q + 0.5 * q)).sum(1)  # positives over negatives scored below, plus half the ties
        out.append((u / (n_pos * n_neg)).cpu().numpy())
    return ((n_pos > 0) & (n_neg > 0)).cpu().numpy(), np.stack(out)


class Scorer:
    """AUROCs with main-task bootstrap intervals, at step and run level, for one comparison."""

    def __init__(self, cfg, runs, comparison):
        import numpy as np

        neg, pos = comparisons(cfg)[comparison]
        self.np = np
        if not runs:
            raise ValueError("no paired cell has all its arms extracted yet")
        self.runs_in = np.array([m["arm"] in neg + pos for m in runs], bool)
        self.y_run = np.array([float(m["arm"] in pos) for m in runs])
        # Only cells holding both sides (honest tasks with one real run have no real-vs-real control).
        sides = {}
        for i in np.flatnonzero(self.runs_in):
            sides.setdefault(runs[i]["cell"], set()).add(self.y_run[i])
        self.runs_in &= np.array([len(sides.get(m["cell"], ())) == 2 for m in runs], bool)
        self.main = np.array([m["main_task"] for m in runs], object)
        self.cell = np.array([m["cell"] for m in runs], object)
        self.fold = np.array([m["fold"] for m in runs], int)
        tasks = sorted(set(self.main[self.runs_in]))
        self.seed, self.n_boot = cfg["seed"], cfg["bootstrap"]
        self.task_id = {t: i for i, t in enumerate(tasks)}
        self.neg, self.pos = neg, pos
        # With every real honest run kept, a cell has several negatives: pairwise accuracy is not defined there.
        self.no_pairwise = cfg.get("honest_real_runs", "first") == "all"

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
        valid, a = boot_aucs(y, [s], task, self.draws(len(np.unique(task))))
        lo, hi = np.percentile(a[0][valid], [2.5, 97.5])
        return {"auroc": round(float(roc_auc_score(y, s)), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)], "n": int(len(y))}

    def auc_gap(self, y, s1, s2, task):
        """AUROC(s1) - AUROC(s2) on the same rows, with a paired bootstrap over main tasks (both AUROCs on each resample)."""
        from sklearn.metrics import roc_auc_score

        np = self.np
        ok = ~np.isnan(s1) & ~np.isnan(s2)
        y, s1, s2, task = y[ok], s1[ok], s2[ok], task[ok]
        if len(np.unique(y)) < 2:
            return None
        n_tasks = len(np.unique(task))
        valid, a = boot_aucs(y, [s1, s2], task, self.draws(n_tasks))
        lo, hi = np.percentile((a[0] - a[1])[valid], [2.5, 97.5])
        a1, a2 = roc_auc_score(y, s1), roc_auc_score(y, s2)
        return {"probe": round(float(a1), 4), "baseline": round(float(a2), 4), "gap": round(float(a1 - a2), 4),
                "ci95": [round(float(lo), 4), round(float(hi), 4)], "n": int(len(y)), "n_tasks": int(n_tasks)}

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
        if self.no_pairwise:
            return None
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
        groups = {"base200": set(cfg["base200"]), "overlap800": set(cfg["overlap800"]), "under_32k": set(cfg["under_32k"])} \
            | {k: set(v) for k, v in cfg.get("extra_slices", {}).items()}  # e.g. honest: contract_stamped_locally
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
