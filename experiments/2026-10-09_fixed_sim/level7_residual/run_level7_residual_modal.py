"""Level-7 residual, real e1 vs simulated attack runs: what text TF-IDF still leans on when it has seen the environment.

One Modal CPU container (16 cores) on the Volume's staging `examples_2026-10-06_142532` (decompressed sha256 checked)
and the saved out-of-fold level-7 scores on main-task folds (`level7_scores_2026-10-06_190549`). It uses
experiments/2026-10-06_q1_text_baselines/analyze_tfidf.py unchanged (scrub level 7, featurize, fit, name_columns) with
config/tfidf.yaml's settings.

  1. Top features: one fit on all kept real-e1 and simulated outputs (for reading, as analyze_tfidf_tells.py does).
     For the top 30 each way: weight, outputs and runs containing the n-gram per arm, its most common environment, and
     one level-7 quote per arm with run key, step and tool.
  2. Confident runs: run score = mean out-of-fold output score (as the AUROCs use). Of the 25 most confident correct
     runs per arm, a seeded 5 each (seed 0), each with its outputs' level-7 text, for reading.
  3. Planted check: the token PLANT appended to (a) every output of 10 seeded simulated runs, (b) one output of each
     of 10 other seeded simulated runs; refit on all outputs; the token's word column's rank toward "simulated" and
     whether the top-30 listing names it.
  Lengths per arm (characters per output, outputs per run) are reported too.

  cd <worktree> && uv run modal run <this file>
"""

import gzip
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import re

import modal

# q1's config folder (the run read config/tfidf.yaml from a scratch worktree of main at 0d7c10ac; the same file is in the
# repo); analyze_tfidf_2df992a6.py comes from get_analyze_tfidf_2df992a6.sh
EXP = Path(__file__).resolve().parents[2] / "2026-10-06_q1_text_baselines"
OUT = Path(__file__).parent
NAME = "probe-shift_aswin-surya_sub-baselines_tfidf_kill-2026-10-20"
EXAMPLES = "examples_2026-10-06_142532"
OUTPUTS_SHA256 = "3c3a08126034f5ecd3451b171dea164ebc0744663c59e00026da6f461472813b"
SCORES = "level7_scores_2026-10-06_190549.jsonl.gz"
PLANT = "qzxplantedtoken"
N_TOP, N_CONFIDENT, N_READ, N_PLANT, SEED = 30, 25, 5, 10, 0

app = modal.App(NAME)
vol = modal.Volume.from_name(NAME)
image = (modal.Image.debian_slim(python_version="3.13")
         .uv_pip_install("numpy", "scipy", "scikit-learn==1.9.1", "joblib==1.6.0", "pyyaml")
         .env({"JOBLIB_TEMP_FOLDER": "/tmp"})
         .add_local_python_source("probe_shift")
         # analyze_tfidf.py as of 2df992a6: its scrub is byte-identical to bbf355e2's, the commit the saved scores
         # record (main's later 051d3352 changed the ID rule), and its vectorizers and column naming are equivalent
         .add_local_file(OUT / "analyze_tfidf_2df992a6.py", "/root/analyze_tfidf.py")
         .add_local_file(EXP / "config" / "tfidf.yaml", "/root/tfidf.yaml"))


def matcher(gram: str, word: bool):
    """Where an n-gram occurs in a text. A character n-gram is a literal substring (whitespace untouched). A word
    n-gram is tokens joined by spaces (token pattern: a word or one punctuation mark), so it matches its tokens in
    order with only whitespace between: at least one between two words, any amount around punctuation."""
    if not word:
        return lambda t: (i := t.find(gram), i + len(gram)) if gram in t else None
    toks = gram.split(" ")
    parts = []
    for k, t in enumerate(toks):
        if k:
            parts.append(r"\s+" if re.fullmatch(r"\w+", toks[k - 1]) and re.fullmatch(r"\w+", t) else r"\s*")
        parts.append(rf"(?<!\w){re.escape(t)}(?!\w)" if re.fullmatch(r"\w+", t) else re.escape(t))
    rx = re.compile("".join(parts))
    return lambda t: (m.start(), m.end()) if (m := rx.search(t)) else None


def quote(text: str, span: tuple[int, int], width: int = 70) -> str:
    i, j = span
    a, b = max(0, i - width), min(len(text), j + width)
    return ("…" if a else "") + text[a:b].replace("\n", "⏎") + ("…" if b < len(text) else "")


@app.function(image=image, volumes={"/data": vol}, cpu=16, memory=65536, timeout=2400, retries=0)
def analyze() -> dict:
    sys.path.insert(0, "/root")
    import numpy as np
    import yaml
    from sklearn.utils import murmurhash3_32

    import analyze_tfidf as A

    cfg = yaml.safe_load(open("/root/tfidf.yaml")) | {"n_jobs": 16}
    raw = gzip.open(f"/data/{EXAMPLES}/outputs.jsonl.gz").read()
    assert hashlib.sha256(raw).hexdigest() == OUTPUTS_SHA256, "Volume staging differs from the README's sha256"
    rows = [r for r in map(json.loads, raw.decode().splitlines()) if r["kept"] and r["arm"] in (A.E1, A.SIM)]
    texts = [A.scrub(r["readout_text"], 7) for r in rows]
    arm = np.array([r["arm"] for r in rows])
    key = np.array([r["key"] for r in rows])
    env = np.array([r["environment"] for r in rows])
    y = (arm == A.SIM).astype(int)
    out = {"analyzer_sha256": hashlib.sha256(open("/root/analyze_tfidf.py", "rb").read()).hexdigest(),
           "n_outputs": {a: int((arm == a).sum()) for a in (A.E1, A.SIM)},
           "n_runs": {a: len(set(key[arm == a])) for a in (A.E1, A.SIM)}}
    lens = np.array([len(t) for t in texts])
    per_run = Counter(key)
    out["lengths"] = {a: {"median_chars_per_output": float(np.median(lens[arm == a])),
                          "mean_chars_per_output": round(float(lens[arm == a].mean()), 1),
                          "median_outputs_per_run": float(np.median([per_run[k] for k in set(key[arm == a])])),
                          "median_chars_per_run": float(np.median([lens[key == k].sum() for k in set(key[arm == a])]))}
                      for a in (A.E1, A.SIM)}

    # Planted variants: (a) every output of 10 seeded sim runs, (b) one output of each of 10 other seeded sim runs.
    rng = random.Random(SEED)
    sim_runs = sorted(set(key[arm == A.SIM]))
    picked = rng.sample(sim_runs, 2 * N_PLANT)
    every, one = set(picked[:N_PLANT]), set(picked[N_PLANT:])
    first_of = {}
    for i, k in enumerate(key):
        if k in one and k not in first_of:
            first_of[k] = i
    planted = {"every_output": [i for i, k in enumerate(key) if k in every], "one_output": sorted(first_of.values())}

    import scipy.sparse as sp

    X = A.featurize(texts, cfg)
    variants, planted_rows = {"full": X}, {}
    for name, idx in planted.items():
        new_texts = [texts[i] + " " + PLANT for i in idx]
        new = A.featurize(new_texts, cfg)
        # replace the planted rows: X + S @ (new - X[idx]), S putting row j of the delta at row idx[j]
        S = sp.csr_matrix((np.ones(len(idx)), (idx, np.arange(len(idx)))), shape=(X.shape[0], len(idx)))
        variants[name] = (X + S @ (new - X[idx])).tocsr()
        planted_rows[name] = (new, new_texts)
    fits = {name: A.fit(M, y, cfg, cfg["C"])[1].coef_[0] for name, M in variants.items()}  # one at a time (memory)

    # 1. top features of the full fit
    coef = fits["full"]
    order = np.argsort(coef)
    sides = {"toward simulated": order[::-1][:N_TOP], "toward real": order[:N_TOP]}
    names = A.name_columns(X, texts, np.concatenate(list(sides.values())), cfg)
    out["top"] = {}
    for side, cols in sides.items():
        listed = []
        for rank, c in enumerate(cols, 1):
            grams = names[int(c)]
            g = grams[0] if grams else None
            kind = "word" if c < cfg["n_features"] else "char"
            item = {"rank": rank, "kind": kind, "ngrams": grams, "weight": round(float(coef[c]), 3)}
            if g:
                find = matcher(g, kind == "word")
                spans = [find(t) for t in texts]
                hit = np.array([sp_ is not None for sp_ in spans])
                item["outputs"] = {a: int((hit & (arm == a)).sum()) for a in (A.E1, A.SIM)}
                item["runs"] = {a: len(set(key[hit & (arm == a)])) for a in (A.E1, A.SIM)}
                envs = Counter(env[hit])
                item["top_environment"] = [envs.most_common(1)[0][0], round(envs.most_common(1)[0][1] / hit.sum(), 3)] if hit.any() else None
                item["examples"] = {}
                for a in (A.E1, A.SIM):
                    js = np.flatnonzero(hit & (arm == a))
                    if len(js):
                        j = int(js[0])
                        item["examples"][a] = {"key": rows[j]["key"], "step": rows[j]["step"], "tool": rows[j]["tool"],
                                               "environment": rows[j]["environment"], "quote": quote(texts[j], spans[j])}
            listed.append(item)
        out["top"][side] = listed

    # 3. planted check
    nf = cfg["n_features"]
    word_col = abs(murmurhash3_32(PLANT, seed=0)) % nf  # word block comes first
    out["planted"] = {"token": PLANT, "word_column": int(word_col)}
    for name, idx in planted.items():
        c2 = fits[name]
        rank = int((c2 > c2[word_col]).sum()) + 1
        top_cols = np.argsort(c2)[::-1][:N_TOP]
        named = A.name_columns(*planted_rows[name], [word_col], cfg)  # read back from the planted rows themselves
        out["planted"][name] = {"outputs_planted": len(idx), "runs_planted": N_PLANT, "weight": round(float(c2[word_col]), 3),
                                "rank_toward_simulated": rank, "in_top_30": bool(word_col in set(top_cols.tolist())),
                                "column_named_as": named[int(word_col)],
                                "planted_runs": sorted(every if name == "every_output" else one)}

    # 2. confident runs from the saved out-of-fold scores on main-task folds
    sc = [json.loads(l) for l in gzip.open(f"/data/results/{SCORES}", "rt")]
    sc = [s for s in sc if s["comparison"] == "real_vs_sim" and s["outputs"] == "all" and s["arm"] in (A.E1, A.SIM)]
    run_score, run_arm = {}, {}
    for s in sc:
        run_score.setdefault(s["key"], []).append(s["score"])
        run_arm[s["key"]] = s["arm"]
    mean = {k: float(np.mean(v)) for k, v in run_score.items()}
    # scores are decision values: positive leans simulated. Most confident and on the right side.
    sim_conf = sorted((k for k in mean if run_arm[k] == A.SIM and mean[k] > 0), key=lambda k: -mean[k])[:N_CONFIDENT]
    real_conf = sorted((k for k in mean if run_arm[k] == A.E1 and mean[k] < 0), key=lambda k: mean[k])[:N_CONFIDENT]
    assert len(sim_conf) == len(real_conf) == N_CONFIDENT
    r2 = random.Random(SEED)
    chosen = {A.SIM: r2.sample(sim_conf, N_READ), A.E1: r2.sample(real_conf, N_READ)}
    oof = {(s["key"], s["step"], s["action"]): s["score"] for s in sc}
    out["confident"] = {}
    for a, ks in chosen.items():
        out["confident"][a] = []
        for k in ks:
            idx = [i for i in range(len(rows)) if rows[i]["key"] == k]
            outs = [{"step": rows[i]["step"], "action": rows[i]["action"], "tool": rows[i]["tool"],
                     "oof": oof.get((k, int(rows[i]["step"]), int(rows[i]["action"]))), "text": texts[i]} for i in idx]  # full text
            out["confident"][a].append({"key": k, "environment": rows[idx[0]]["environment"], "cell": rows[idx[0]]["cell"],
                                        "run_score": round(mean[k], 4), "n_outputs": len(idx),
                                        "chars": int(sum(len(texts[i]) for i in idx)), "outputs": outs})
    out["confident_pool"] = {"sim_top25_min_score": round(mean[sim_conf[-1]], 4),
                             "real_top25_max_score": round(mean[real_conf[-1]], 4)}
    return out


@app.local_entrypoint()
def main():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    out = analyze.remote()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (OUT / f"level7_residual_{stamp}.json").write_text(json.dumps(out, indent=1))
    lines = [f"# Level-7 residual, real e1 vs simulated, main-task folds ({stamp})", "",
             f"outputs {out['n_outputs']}, runs {out['n_runs']}", f"lengths {json.dumps(out['lengths'])}", ""]
    for side, items in out["top"].items():
        lines.append(f"## Top {N_TOP} {side} (full fit; outputs and runs containing the first n-gram, e1 / sim)")
        for it in items:
            o, r, te = it.get("outputs", {}), it.get("runs", {}), it.get("top_environment")
            lines.append(f"{it['rank']:>2}. {' | '.join(repr(g) for g in it['ngrams'])}  w={it['weight']:+.3f}  "
                         f"outputs {o.get('real-attack-e1')}/{o.get('sim-attack')}  runs {r.get('real-attack-e1')}/{r.get('sim-attack')}  "
                         f"top env {te}")
            for a, ex in it.get("examples", {}).items():
                lines.append(f"      {a[:8]} {ex['key'][:60]} step {ex['step']} {ex['tool']}: {ex['quote'][:220]}")
        lines.append("")
    lines.append("## Planted check")
    lines.append(json.dumps(out["planted"], indent=1))
    lines.append("")
    lines.append(f"## Confident runs read (seeded {N_READ} of the {N_CONFIDENT} most confident per arm) {out['confident_pool']}")
    for a, runs in out["confident"].items():
        for r in runs:
            lines.append(f"- {a} {r['key'][:70]} env {r['environment']} run score {r['run_score']} outputs {r['n_outputs']} chars {r['chars']}")
    text = "\n".join(lines) + "\n"
    (OUT / f"level7_residual_{stamp}.md").write_text(text)
    print(text)
