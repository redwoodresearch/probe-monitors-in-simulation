"""Behaviour-only fake-vs-real test on honest runs (RULE.md, frozen at c5a9051e). $0, local.

Reads the two honest cohorts from the local eval-log cache, computes per-run behaviour features by script, and fits a
standardized logistic regression with task-grouped 5-fold CV, a 200-draw within-task shuffled-label null, and a
real-run-1 vs real-run-2 control. Only numbers, task ids and sample ids are written; no command or reply text is
printed or saved. `uv run python experiments/2026-10-13_behaviour_cue/analyze_behaviour_cue.py`
"""

import collections
import hashlib
import itertools
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from probe_shift import cohorts
from probe_shift import transcripts as T
from probe_shift.reader import command_param

HERE = Path(__file__).parent
CACHE = T.CACHE / "rr-sce-sim-transcripts" / "transcripts"
ROOTS = {  # the local cache copies of la_public10_real_honest_n1000 and la_public10_sim_astra_honest_n437
    "real": CACHE / "2026-09-18_la-public10_real-honest-solmedium-step150-overlapaware-n1000",
    "fake": CACHE / "2026-10-06_la-public10_sim-astra-medium-honest-origtraj-read-nocritic-fork0",
}
CV_SEED, NULL_SEED, PAIR_SEED, ONE_SEED, N_NULL = 20261013, 20261014, 20261015, 20261016, 200
ERR = re.compile(r"Traceback|Error|error:|No such file|command not found|Permission denied")
FINE = ["bash", "view", "create", "str_replace", "insert", "other"]
COARSE = {"bash": "bash", "view": "view", "create": "edit", "str_replace": "edit", "insert": "edit", "other": "other"}
KINDS = ["bash", "view", "edit", "other"]
B_FEATURES = ["error_field_share", "error_like_share", "retry_after_error_rate"]


def kind(tc) -> str:
    """bash and the per-container `*_bash` tools are bash; text_editor by its subcommand; anything else is other."""
    name = tc["function"]
    if name == "bash" or name.endswith("_bash"):
        return "bash"
    cmd = (tc.get("arguments") or {}).get("command") if name == "text_editor" else None
    return cmd if cmd in FINE[1:5] else "other"


def features(sample) -> dict:
    calls, turns, outputs = [], 0, {}
    for m in sample["messages"]:
        if m["role"] == "assistant":
            cs = [tc for tc in m.get("tool_calls") or [] if tc["function"] != T.RUNNER_SUBMIT]
            if cs or T.visible_text(m["content"]).strip():
                turns += 1
            calls += [(tc, turns - 1) for tc in cs]
        elif m["role"] == "tool" and m.get("function") != T.RUNNER_SUBMIT:
            assert m.get("tool_call_id") and m["tool_call_id"] not in outputs, f"{sample['id']}: missing or repeated output id"
            outputs[m["tool_call_id"]] = (bool(m.get("error")), bool(ERR.search(T.tool_result_text(m))))
    kinds = [kind(tc) for tc, _ in calls]
    n = len(calls)
    missing = [tc.get("id") for tc, _ in calls if tc.get("id") not in outputs]
    assert not missing, f"{sample['id']}: {len(missing)} calls without a tool output"
    lens = [len(tc["arguments"][k]) for tc, _ in calls if (k := command_param(tc["function"], tc.get("arguments")))]
    assert lens, f"{sample['id']}: no call has a command parameter"
    edit_turns = [t for (tc, t), k in zip(calls, kinds) if COARSE[k] == "edit"]
    pairs = list(itertools.pairwise(kinds))
    f = {"n_calls": n, "n_turns": turns, "calls_per_turn": n / turns}
    f |= {f"share_{k}": kinds.count(k) / n for k in FINE}
    f |= {f"bigram_{a}_{b}": sum((COARSE[x], COARSE[y]) == (a, b) for x, y in pairs) / max(len(pairs), 1)
          for a in KINDS for b in KINDS}
    f |= {"cmd_len_mean": np.mean(lens), "cmd_len_median": np.median(lens), "cmd_len_max": max(lens),
          "cmd_len_log_total": np.log1p(sum(lens)), "repeat_rate": sum(x == y for x, y in pairs) / max(len(pairs), 1),
          "first_edit_frac": edit_turns[0] / turns if edit_turns else 1.0, "any_edit": float(bool(edit_turns)),
          "log_view_edit": np.log((kinds.count("view") + 1) / (len(edit_turns) + 1))}
    flags = [outputs[tc.get("id")] for tc, _ in calls]
    # A retry: the call after an error-like output, issued in a later turn (so after the output was seen), same kind.
    retries = sum(flags[i][1] and calls[i + 1][1] > calls[i][1] and kinds[i] == kinds[i + 1] for i in range(n - 1))
    f |= {"error_field_share": sum(e for e, _ in flags) / n, "error_like_share": sum(r for _, r in flags) / n,
          "retry_after_error_rate": retries / n}
    return f


def load():
    rows = []
    for world, root in ROOTS.items():
        for file in cohorts.eval_files(str(root)):
            for s in cohorts.file_samples(str(root), file):
                rows.append({"world": world, "file": file, "id": s["id"], "task": T.main_task_id(s),
                             "invalid": T.invalidity(s), **({} if T.invalidity(s) else features(s))})
    return rows


def oof_auroc(X, y, groups) -> float:
    oof = np.full(len(y), np.nan)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=CV_SEED).split(X, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000))
        oof[te] = clf.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    assert not np.isnan(oof).any()
    return float(roc_auc_score(y, oof))


def within_task_shuffle(y, groups, seed):
    rng, out = np.random.default_rng(seed), y.copy()
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        out[idx] = rng.permutation(y[idx])
    return out


def test(X, y, groups) -> dict:
    null = Parallel(n_jobs=-1)(delayed(oof_auroc)(X, within_task_shuffle(y, groups, NULL_SEED + i), groups)
                               for i in range(N_NULL))
    obs = oof_auroc(X, y, groups)
    return {"auroc": obs, "null_p95": float(np.percentile(null, 95)), "null_median": float(np.median(null)),
            "null_ge_observed": int(sum(v >= obs for v in null)), "n_null": N_NULL,
            "n": len(y), "n_pos": int(y.sum()), "n_tasks": len(np.unique(groups))}


def main():
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", str(HERE / Path(__file__).name)], capture_output=True,
                           text=True, check=True).stdout.strip()
    assert not dirty, "commit the script before running it"
    rows = load()
    invalid = collections.Counter((r["world"], r["invalid"]) for r in rows if r["invalid"])
    valid = [r for r in rows if not r["invalid"]]
    shared = {r["task"] for r in valid if r["world"] == "fake"} & {r["task"] for r in valid if r["world"] == "real"}
    keep = [r for r in valid if r["task"] in shared]
    names = [k for k in keep[0] if k not in ("world", "file", "id", "task", "invalid")]
    a_names = [k for k in names if k not in B_FEATURES]
    X = np.array([[r[k] for k in names] for r in keep], dtype=float)
    assert np.isfinite(X).all()
    y = np.array([r["world"] == "fake" for r in keep], dtype=int)
    groups = np.array([r["task"] for r in keep])
    ia = [names.index(k) for k in a_names]
    batch = lambda r: r["file"].split("_", 1)[0]  # real: base200 or overlap800
    out = {"git": git, "rule": "RULE.md (frozen c5a9051e; clarifications in this run's commit)", "roots": {k: str(v) for k, v in ROOTS.items()},
           "loaded": dict(collections.Counter(r["world"] for r in rows)),
           "invalid": {f"{w}:{why}": c for (w, why), c in invalid.items()},
           "kept": dict(collections.Counter(f"{r['world']}:{batch(r)}" for r in keep)), "shared_tasks": len(shared),
           "features_a": a_names, "features_b": B_FEATURES,
           "a_only": test(X[:, ia], y, groups), "a_plus_b": test(X, y, groups)}
    by_task = collections.defaultdict(list)
    for r in sorted((r for r in keep if r["world"] == "real"), key=lambda r: (r["task"], r["file"], r["id"])):
        by_task[r["task"]].append(r)
    rng = np.random.default_rng(PAIR_SEED)  # per task with 2+ real runs: two at random, in random order (label 0, 1)
    pairs = [rs[i] for t in sorted(by_task) if len(rs := by_task[t]) >= 2 for i in rng.choice(len(rs), 2, replace=False)]
    Xc = np.array([[r[k] for k in names] for r in pairs], dtype=float)
    yc, gc = np.arange(len(pairs)) % 2, np.array([r["task"] for r in pairs])
    out["control_pairs"] = {"tasks": len(pairs) // 2, "cross_batch": sum(batch(a) != batch(b) for a, b in
                                                                       zip(pairs[::2], pairs[1::2]))}
    rng = np.random.default_rng(ONE_SEED)  # task-balanced: one real run per task, at random, against its fake run
    at = {(r["file"], r["id"]): i for i, r in enumerate(keep)}
    assert len(at) == len(keep), "run keys must be unique"
    one = [at[(r["file"], r["id"])] for t in sorted(by_task) for r in [by_task[t][rng.integers(len(by_task[t]))]]]
    one = sorted(one + [i for i, r in enumerate(keep) if r["world"] == "fake"])
    out["task_balanced_a_only"] = test(X[np.ix_(one, ia)], y[one], groups[one])
    out["control_a_only"] = test(Xc[:, ia], yc, gc)
    out["control_a_plus_b"] = test(Xc, yc, gc)
    uni = [(k, float(roc_auc_score(y, X[:, j])), float(X[y == 1, j].mean()), float(X[y == 0, j].mean()))
           for j, k in enumerate(names)]
    out["univariate_top10"] = [{"feature": k, "auroc_fake_high": a, "mean_fake": mf, "mean_real": mr}
                               for k, a, mf, mr in sorted(uni, key=lambda t: -abs(t[1] - 0.5))[:10]]
    ts = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rows_path = HERE.parents[1] / ".run_cache" / "behaviour_cue" / f"rows_{ts}.jsonl"
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows_path.write_text("".join(json.dumps({k: r[k] for k in ("world", "file", "id", "task", *names)}) + "\n"
                                 for r in keep))
    out["rows"] = {"path": str(rows_path), "sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest()}
    path = HERE / "results" / f"behaviour_cue_{ts}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(path, file=sys.stderr)


if __name__ == "__main__":
    main()
