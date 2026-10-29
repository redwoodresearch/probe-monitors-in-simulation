"""PLAN.md amendment 6, step 1: gate (a)'s TF-IDF with four leftover slips evened out on both sides (Aswin, 14:12 PT
10-10, "Option 1"; the slips are exploratory findings of the viewer session's transcript verifier, gathered past his
10-09 stop on new rendering checks). One Modal CPU container reruns gate (a)'s leave-one-environment-out fits over all 10
environments (run_gate_a_modal.py and analyze_tfidf.py unchanged, config/gate_a.yaml), on text where, in BOTH arms:
  1. create   every text_editor create output loses its trailing newlines;
  2. view     every text_editor view output loses an empty last numbered line (`%6d\\t` with nothing after it), keeping
              whatever trailing newline it had;
  3. order    in every non-editor output holding an error-style line (a whole line matching scrub.py's STDERR_LINE, the
              verifier's definition), those lines move to the front, each group keeping its order (real runs mostly
              print them first, the fixed sim often last);
  4. range    every output holding "Invalid `view_range`" is dropped from both arms (gate (a)'s secondary).
Evening runs on the raw text, then gate (a)'s scrub (level 0 raw, level 1 times masked). Comparisons fixed sim vs real e1
and real e2 vs e1. Returns run scores (no text) and the per-arm counts of outputs each evening changed; locally, run-level
AUROC on the 5 test environments and all 10, per environment, and a classifier-fixed null (1,000 draws, one coin per main
task swapping the two runs of every cell of that task), beside gate (a)'s archived rows. Free: one container, ~$0.3.

  PYTHONPATH=src uv run --no-project --python 3.13 --with 'modal>=1.4.3' --with pyyaml --with numpy --with scikit-learn \\
      modal run experiments/2026-10-09_fixed_sim/run_tfidf_slips_evened_modal.py
"""

import gzip
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REPO = HERE.parents[1] if modal.is_local() else Path("/root")
sys.path.insert(0, str(HERE) if modal.is_local() else "/root")
import run_gate_a_modal as G  # noqa: E402  (gate (a)'s code, unchanged)

app = modal.App("probe-shift_aswin-surya_sub-fixed-sim_tfidf-slips_kill-2026-11-10")
image = G.image.add_local_file(HERE / "run_gate_a_modal.py", "/root/run_gate_a_modal.py")
LEVELS, ARCHIVED, RANGE = {"level0": 0, "level1": 1}, HERE / "results" / "gate_a_main_2026-10-09_185451.json", "Invalid `view_range`"
EMPTY_LAST = re.compile(r"\n[ \t]*\d+\t?\Z")
ERROR_LINE = None  # scrub.STDERR_LINE as the verifier compiled it: whole lines, case-insensitive (set in even_order)


def even_create(text: str) -> str:
    return text.rstrip("\n")


def even_view(text: str) -> str:
    body = text.rstrip("\n")
    return EMPTY_LAST.sub("", body) + text[len(body):]


def even_order(text: str) -> str:
    global ERROR_LINE
    if ERROR_LINE is None:
        from probe_shift.scrub import STDERR_LINE

        ERROR_LINE = re.compile(STDERR_LINE.pattern.replace(r")\n?", ")"), re.I)
    lines = text.split("\n")
    err = [ERROR_LINE.fullmatch(l) is not None for l in lines]
    if not any(err) or all(err):
        return text
    return "\n".join([l for l, e in zip(lines, err) if e] + [l for l, e in zip(lines, err) if not e])


def even(text: str, command: str, tool: str) -> tuple[str, list[str]]:
    """The evened text and which evenings changed it."""
    out, hit = text, []
    for name, f, applies in (("create", even_create, command == "create"), ("view", even_view, command == "view"),
                             ("order", even_order, tool != "text_editor")):
        if applies and (t := f(out)) != out:
            out, hit = t, hit + [name]
    return out, hit


@app.function(image=image, volumes={"/acts": G.acts}, cpu=16, memory=65536, timeout=1500, retries=0)
def job(cfg: dict, split: dict, folds: dict) -> dict:
    import numpy as np

    import analyze_tfidf as A

    res = {"runs": {}, "changed": {}}
    for level, scrub in LEVELS.items():
        d = A.load(Path("/acts") / cfg["run"] / "examples", folds["environment_folds"]["environment_to_fold"], fold_key="environment")
        raw = [r[cfg["text"]] for r in d["rows"]]
        keep = np.array([RANGE not in t for t in raw])
        changed = Counter()
        texts = []
        for r, t, c, k in zip(d["rows"], raw, d["editor_command"], keep):
            if k:
                t, hit = even(t, c, r["tool"])
                changed.update(f"{r['arm']}|{h}" for h in hit)
                texts.append(A.scrub(t, scrub))
        changed.update(f"{a}|range_dropped" for a in d["arm"][~keep])
        res["changed"][level] = dict(changed)
        for k in ("arm", "key", "cell", "main_task"):
            d[k] = d[k][keep]
        env = np.array([r["environment"] for r, k in zip(d["rows"], keep) if k])
        X = A.featurize(texts, cfg)
        for comp, (neg, pos) in cfg["comparisons"].items():
            R = np.isin(d["arm"], [neg, pos])
            y = (d["arm"] == pos).astype(int)
            s = G.score_jobs(X, y, G.design_jobs("loeo", R, env, d, split, all_envs=True), cfg)
            run_score, first = A.run_means(s, d["key"], R)
            res["runs"][f"{comp}|{level}"] = [{"key": str(d["key"][i]), "arm": str(d["arm"][i]), "environment": str(env[i]), "cell": str(d["cell"][i]),
                                               "main_task": str(d["main_task"][i]), "score": float(v)} for i, v in zip(first, run_score)]
            print(comp, level, len(first), "runs", flush=True)
        del X
    return res


def summarize(runs: list[dict], pos: str, test: list[str], rng) -> dict:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    def null_of(rr):
        y = np.array([r["arm"] == pos for r in rr])
        s = np.array([r["score"] for r in rr])
        cells, tasks = {}, {}
        for i, r in enumerate(rr):
            cells.setdefault(r["cell"], []).append(i)
        pairs = [g for g in cells.values() if len(g) == 2 and y[g].sum() == 1]
        t_of = sorted({rr[g[0]]["main_task"] for g in pairs})
        null = []
        for _ in range(1000):
            flip = dict(zip(t_of, rng.random(len(t_of)) < 0.5))
            yy = y.copy()
            for g in pairs:
                if flip[rr[g[0]]["main_task"]]:
                    yy[g] = yy[g[::-1]]
            null.append(roc_auc_score(yy, s))
        obs = roc_auc_score(y, s)
        return {"auroc": round(float(obs), 4), "null_p95": round(float(np.percentile(null, 95)), 4),
                "p": round(float((1 + (np.array(null) >= obs).sum()) / 1001), 4), "runs": len(rr)}

    out = {"test_envs": null_of([r for r in runs if r["environment"] in test]), "all_envs": null_of(runs)}
    out["per_environment"] = {e: null_of([r for r in runs if r["environment"] == e]) for e in sorted({r["environment"] for r in runs})}
    return out


@app.local_entrypoint()
def main():
    import numpy as np
    import yaml

    if subprocess.run(["git", "status", "--porcelain", str(Path(__file__))], capture_output=True, text=True, check=True).stdout:
        raise SystemExit(f"commit {Path(__file__).name} first: the result records the commit that ran")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    cfg = yaml.safe_load((HERE / "config" / "gate_a.yaml").read_text())
    split = json.loads((HERE / cfg["split_file"]).read_text())
    folds = json.loads((REPO / "splits" / cfg["folds_file"]).read_text())
    res = job.remote(cfg, split, folds)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(HERE / "data" / f"tfidf_slips_run_scores_{stamp}.json.gz", "xb") as fh:  # keys and scores, no text
        fh.write(gzip.compress(json.dumps({"git": git, "runs": res["runs"]}).encode(), mtime=0))
    archived, rng = json.loads(ARCHIVED.read_text())["rows"], np.random.default_rng(0)
    out = {"git": git, "changed_outputs": res["changed"], "rows": {}}
    for k, runs in res["runs"].items():
        comp, level = k.split("|")
        out["rows"][k] = summarize(runs, cfg["comparisons"][comp][1], split["test"], rng)
        before = archived.get(f"{comp}|loeo|{'all' if level == 'level0' else 'level1'}", {}).get("test_envs", {}).get("auroc")
        out["rows"][k]["gate_a_before_evening"] = before
        print(k, "evened", out["rows"][k]["test_envs"], "| gate (a) before:", before)
    print("outputs changed by each evening:", res["changed"])
    with open(HERE / "results" / f"tfidf_slips_evened_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print("->", HERE / "results" / f"tfidf_slips_evened_{stamp}.json")
