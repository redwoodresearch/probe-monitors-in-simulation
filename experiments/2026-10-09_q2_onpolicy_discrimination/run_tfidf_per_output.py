"""Real vs simulated discrimination, per tool output (local, free). Added after the run-level TF-IDF failed its planted
check (a fixed line in every tool output of one world reached only 0.78-0.91 AUROC with 38-40 runs); not pre-registered.

Same runs, comparisons and folds as run_tfidf_discrimination.py, but each tool output is a sample (its text only),
labelled by its run's world. TF-IDF char_wb 2-5 (min_df 2, at most 200,000 features, sublinear tf) and logistic
regression C=1, fit on four folds' outputs, scored on the fifth. Reported: output-level AUROC, and run-level AUROC
from each run's mean output score. Intervals: 2,000 task resamples of the fixed scores (seed 0). Null: 50 label swaps
within task pairs (a whole run's outputs swap together), refitting the classifier; p = (1 + draws >= AUROC) / 51.
Planted check: the same fixed line appended to every tool output of the second world.

  uv run --no-project --python 3.14 --with numpy --with scikit-learn python \
      experiments/2026-10-09_q2_onpolicy_discrimination/run_tfidf_per_output.py <rerun folder> <q3 per-call json>
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

sys.path.insert(0, str(Path(__file__).parent))
from run_tfidf_discrimination import PAIRS, TELL, VEC, seen

HERE = Path(__file__).parent


def outputs(path: Path, tell: bool) -> list[str]:
    with zipfile.ZipFile(path) as z:
        (name,) = [x for x in z.namelist() if x.startswith("samples/") and x.endswith(".json")]
        s = json.loads(z.read(name))
    att = s.get("attachments") or {}
    res = lambda v: att.get(v.removeprefix("attachment://"), v) if isinstance(v, str) and v.startswith("attachment://") else v
    text = lambda c: res(c) if isinstance(c, str) else "\n".join(res(x.get("text") or "") for x in c or [] if x.get("type") == "text")
    return [seen(m, text) + (f"\n{TELL}" if tell else "") for m in s["messages"] if m["role"] == "tool"]


def evaluate(docs, y, run, task, fold, rng, n_null=50):
    runs = np.unique(run)
    run_y = {r: y[run == r][0] for r in runs}
    run_task = {r: task[run == r][0] for r in runs}
    tasks = np.unique(task)
    perms = [y]
    for _ in range(n_null):
        swap = {t: rng.random() < 0.5 for t in tasks}
        perms.append(np.array([1 - v if swap[t] else v for v, t in zip(y, task)]))
    scores = [np.zeros(len(y)) for _ in perms]
    for k in range(5):
        tr, te = fold != k, fold == k
        vec = TfidfVectorizer(**VEC).fit([d for d, t in zip(docs, tr) if t])
        Xtr, Xte = vec.transform([d for d, t in zip(docs, tr) if t]), vec.transform([d for d, t in zip(docs, te) if t])
        for sc, lab in zip(scores, perms):
            sc[te] = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, lab[tr]).decision_function(Xte)

    def run_auc(sc, lab):
        r_s = {r: sc[run == r].mean() for r in runs}
        r_y = {r: lab[run == r][0] for r in runs}
        return roc_auc_score([r_y[r] for r in runs], [r_s[r] for r in runs])
    out_auc, r_auc = roc_auc_score(y, scores[0]), run_auc(scores[0], y)
    null = [run_auc(sc, lab) for sc, lab in zip(scores[1:], perms[1:])]
    by = {t: [r for r in runs if run_task[r] == t] for t in tasks}
    rs = {r: scores[0][run == r].mean() for r in runs}
    boot = []
    for _ in range(2000):
        rr = [r for t in rng.choice(tasks, len(tasks)) for r in by[t]]
        if len({run_y[r] for r in rr}) == 2:
            boot.append(roc_auc_score([run_y[r] for r in rr], [rs[r] for r in rr]))
    return {"output-level AUROC": round(float(out_auc), 4), "run-level AUROC": round(float(r_auc), 4),
            "run-level 95% (task resamples)": [round(float(x), 4) for x in np.percentile(boot, [2.5, 97.5])],
            "run-level null mean / max": [round(float(np.mean(null)), 4), round(float(np.max(null)), 4)],
            "p": round((1 + sum(n >= r_auc for n in null)) / (1 + len(null)), 4), "outputs": len(y)}


def main(run_dir: Path, percall_p: Path):
    folds = json.loads(percall_p.read_text())["folds"]
    rng, out = np.random.default_rng(0), {"inputs": {"run folder": run_dir.name, "q3 per-call sha256": hashlib.sha256(percall_p.read_bytes()).hexdigest()},
                                          "tell line": TELL, "comparisons": {}}
    for name, (a, b) in PAIRS.items():
        files = {arm: {f.parent.name: f for f in sorted((run_dir / arm).glob("*/*.eval"))} for arm in (a, b)}
        res = {}
        for label, tell in (("TF-IDF per output", False), ("TF-IDF per output, planted line in the second world", True)):
            docs, y, run, task = [], [], [], []
            for t in sorted(set(files[a]) & set(files[b])):
                per = [outputs(files[a][t], False), outputs(files[b][t], tell)]
                if not per[0] or not per[1]:
                    continue  # a run with no tool outputs (did not start) drops its task
                for cls, o in enumerate(per):
                    docs += o; y += [cls] * len(o); run += [f"{t}|{cls}"] * len(o); task += [t] * len(o)
            y, run, task = np.array(y), np.array(run), np.array(task)
            res[label] = evaluate(docs, y, run, task, np.array([folds[t] for t in task]), rng)
            res["tasks"] = len(np.unique(task))
        out["comparisons"][name] = res
        print(name, json.dumps(res), flush=True)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"tfidf_per_output_{stamp}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(path)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
