"""Real vs simulated discrimination on the on-policy strict rerun, TF-IDF on each run's text (local, free).

Unit: one run of the strict rerun (2026-10-09_000523_rerun_strict; Qwen3.5-27B acting; 20 tasks). Text: what the
episode produced, in order: the agent's visible text, each tool call (function and arguments) and each tool output;
the system prompt and the first user message (the task statement) are left out; a task is dropped from a comparison
if either of its runs has no episode text (a run that did not start). Comparisons, one run per class per
task: real honest_b vs sim_honest@september and @today; real attack vs sim_attack@september and @today; real honest_a
vs honest_b (the chance reference). Folds: q3's five main-task folds. Model: Q2's command_text settings (TF-IDF char_wb
2-5, min_df 2, at most 200,000 features, sublinear tf; logistic regression C=1), fit on four folds, scored on the
fifth; AUROC pools the out-of-fold scores. Interval: 2,000 task resamples of the fixed scores (seed 0). Null: 200
label swaps within task pairs, refitting the classifiers (the vectorizer is label-free and fit once per fold); p =
(1 + draws >= AUROC) / 201. Planted check (Aswin's rule): one fixed line appended to every tool output of one world
(the simulated one; honest_b in the real-vs-real reference), then the same pipeline.

  uv run --no-project --python 3.14 --with numpy --with scikit-learn python \
      experiments/2026-10-09_q2_onpolicy_discrimination/run_tfidf_discrimination.py <rerun folder> <q3 per-call json>
"""
import hashlib
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
PAIRS = {"real honest_b vs sim_honest@september": ("honest_b", "sim_honest@september"),
         "real honest_b vs sim_honest@today": ("honest_b", "sim_honest@today"),
         "real attack vs sim_attack@september": ("attack", "sim_attack@september"),
         "real attack vs sim_attack@today": ("attack", "sim_attack@today"),
         "real honest_a vs real honest_b (reference)": ("honest_a", "honest_b")}
TELL = "[harness] tool output checksum verified"
VEC = {"analyzer": "char_wb", "ngram_range": (2, 5), "min_df": 2, "max_features": 200000, "sublinear_tf": True}


def seen(m: dict, text) -> str:
    """A tool message as the policy model read it: Inspect's OpenAI-style providers send f"Error: {error.message}" instead
    of the content when the call failed (real runs record failures there; simulated runs inline them as text)."""
    return f"Error: {m['error']['message']}" if m.get("error") else text(m["content"])


def run_text(path: Path, tell: bool = False) -> str:
    with zipfile.ZipFile(path) as z:
        (name,) = [x for x in z.namelist() if x.startswith("samples/") and x.endswith(".json")]
        s = json.loads(z.read(name))
    att, parts, first_user = s.get("attachments") or {}, [], True
    res = lambda v: att.get(v.removeprefix("attachment://"), v) if isinstance(v, str) and v.startswith("attachment://") else v
    text = lambda c: res(c) if isinstance(c, str) else "\n".join(res(x.get("text") or "") for x in c or [] if x.get("type") == "text")
    for m in s["messages"]:
        if m["role"] == "system" or (m["role"] == "user" and first_user):
            first_user = first_user and m["role"] != "user"
            continue
        if m["role"] == "assistant":
            parts.append(text(m["content"]))
            parts += [f"{tc['function']} {json.dumps({k: res(v) for k, v in (tc.get('arguments') or {}).items()}, ensure_ascii=False)}" for tc in m.get("tool_calls") or []]
        elif m["role"] == "tool":  # what the model saw: Inspect sends "Error: <message>" in place of a failed call's content
            parts.append(seen(m, text) + (f"\n{TELL}" if tell else ""))
        else:
            parts.append(text(m["content"]))
    return "\n".join(parts)


def oof_scores(docs, y, fold, labels_list):
    """Out-of-fold scores for each label vector in labels_list (the vectorizer fit once per fold, label-free)."""
    out = [np.zeros(len(y)) for _ in labels_list]
    for k in range(5):
        tr, te = fold != k, fold == k
        vec = TfidfVectorizer(**VEC).fit([d for d, t in zip(docs, tr) if t])
        Xtr, Xte = vec.transform([d for d, t in zip(docs, tr) if t]), vec.transform([d for d, t in zip(docs, te) if t])
        for o, lab in zip(out, labels_list):
            o[te] = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, lab[tr]).decision_function(Xte)
    return out


def evaluate(docs, y, fold, task, rng):
    perms = [y]
    tasks = np.unique(task)
    for _ in range(200):
        swap = {t: rng.random() < 0.5 for t in tasks}
        perms.append(np.array([1 - v if swap[t] else v for v, t in zip(y, task)]))
    scores = oof_scores(docs, y, fold, perms)
    auc = roc_auc_score(y, scores[0])
    null = [roc_auc_score(p, s) for p, s in zip(perms[1:], scores[1:])]
    by = {t: np.flatnonzero(task == t) for t in tasks}
    boot = []
    for _ in range(2000):
        idx = np.concatenate([by[t] for t in rng.choice(tasks, len(tasks))])
        if len(set(y[idx])) == 2:
            boot.append(roc_auc_score(y[idx], scores[0][idx]))
    return {"AUROC": round(float(auc), 4), "95% (task resamples)": [round(float(x), 4) for x in np.percentile(boot, [2.5, 97.5])],
            "null mean / max": [round(float(np.mean(null)), 4), round(float(np.max(null)), 4)],
            "p": round((1 + sum(n >= auc for n in null)) / (1 + len(null)), 4)}


def main(run_dir: Path, percall_p: Path):
    folds = json.loads(percall_p.read_text())["folds"]
    rng, out = np.random.default_rng(0), {"inputs": {"run folder": run_dir.name, "q3 per-call sha256": hashlib.sha256(percall_p.read_bytes()).hexdigest()},
                                          "tell line": TELL, "comparisons": {}}
    for name, (a, b) in PAIRS.items():
        files = {arm: {f.parent.name: f for f in sorted((run_dir / arm).glob("*/*.eval"))} for arm in (a, b)}
        both = sorted(set(files[a]) & set(files[b]))
        empty = sorted(t for t in both if not run_text(files[a][t]).strip() or not run_text(files[b][t]).strip())
        tasks = [t for t in both if t not in empty]  # a run with no episode text (did not start) drops its task
        rows = [(t, 0, files[a][t]) for t in tasks] + [(t, 1, files[b][t]) for t in tasks]
        y, task = np.array([r[1] for r in rows]), np.array([r[0] for r in rows])
        fold = np.array([folds[t] for t in task])
        docs = [run_text(r[2]) for r in rows]
        planted = [run_text(r[2], tell=r[1] == 1) for r in rows]
        out["comparisons"][name] = {"tasks": len(tasks), "runs": len(rows), "tasks dropped (a run with no text)": empty, "TF-IDF": evaluate(docs, y, fold, task, rng),
                                    "TF-IDF, planted line in the second world": evaluate(planted, y, fold, task, rng)}
        print(name, json.dumps(out["comparisons"][name]), flush=True)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"tfidf_discrimination_{stamp}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(path)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
