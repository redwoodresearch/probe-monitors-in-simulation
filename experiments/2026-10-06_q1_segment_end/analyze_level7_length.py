"""Is the level-7 verdict just length? (2026-10-08; preliminary; saved scores and extraction metadata only, $0)

At text level 7 the frozen protocol (protocol_level7.md) has the probe at 0.541 and TF-IDF at 0.555, against an
original-text output-length baseline of 0.556. Length was never measured at level 7. This measures it, on the frozen
protocol's runs and its primary split: real-attack-e1 vs sim-attack, run AUROC, held-out environment (10 folds).

Length per run, from the level-7 Qwen3-32B extraction's metadata (the text and 32k cut both readers' matched analyses
read): tool-output tokens read (the total and the mean over outputs, from each kept output_end span), tokens read, and
whole-transcript tokens before the cut.
  1. Length alone: each feature's AUROC, its direction taken from the training folds of each held-out environment.
  2. Length regressed out, a descriptive adjustment: within each fold, the run scores are regressed on log total
     tool-output tokens and log whole-transcript tokens. The fit uses the training folds' runs of both arms; the test
     runs are scored by their residuals, for the probe and for TF-IDF. It is not a held-out estimate: the training
     folds' saved scores come from probes (and layer choices) that trained on the held-out environment's runs.
  3. The same for Qwen3.5's matched level-7 probe (analysis 2026-10-06_190337).
Run scores: each probe's saved held-out-environment output_end run scores; TF-IDF's per-output scores averaged over
the outputs the probe read. Their run AUROCs must reproduce the protocol reports (checks). Intervals: 1000 main-task
draws (seed 0); the change from regressing out length gets a paired interval.

  uv run --with pandas python experiments/2026-10-06_q1_segment_end/analyze_level7_length.py
"""

import gzip
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
VOL, EXTRACTION = "probe-shift_aswin-surya_sub-la-activations", "2026-10-06_la_segment-end-scrub7_qwen3-32b_alllayers_v1"
PROBES = {"qwen3-32b": (ROOT / ".run_cache" / "qwen3_32b_sources" / "q1_gpu_2026-10-06_181139.json", 0.5408),
          "qwen3.5 (matched)": (ROOT / "experiments" / "2026-10-06_qwen35_replication" / "results" / "q1" / "q1_gpu_2026-10-06_190337.json", 0.5721)}
TFIDF = (Path.home() / "probe-shift-baselines" / "experiments" / "2026-10-06_q1_text_baselines" / "results"
         / "harness_normalized_editor_format_scores_2026-10-06_172551.jsonl.gz", 0.5551)
ARMS, N_BOOT, SEED = ("real-attack-e1", "sim-attack"), 1000, 0  # sim scores high: AUROCs are sim over real


def short_key(key: str) -> str:
    """The extraction's run key: keys over 240 bytes are cut to 150 characters plus a hash (as the Q1 manifest has them)."""
    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


def wauroc(y, s, W):
    """AUROC (ties half) of y=1 over y=0 per row of task weights W (draws x runs)."""
    o = np.argsort(s, kind="mergesort")
    ss, Wo, yo = s[o], W[:, o], y[o]
    starts = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])
    P, N = np.add.reduceat(Wo * yo, starts, axis=1), np.add.reduceat(Wo * (1 - yo), starts, axis=1)
    return (P * (np.cumsum(N, 1) - N + 0.5 * N)).sum(1) / (W[:, y == 1].sum(1) * W[:, y == 0].sum(1))


def lengths(keys: set) -> dict:
    """(arm, key) -> tool-output tokens read (total, mean), tokens read, whole-transcript tokens, kept outputs."""
    import modal

    vol = modal.Volume.from_name(VOL, version=2)
    paths = [e.path for arm in ARMS for e in vol.listdir(f"{EXTRACTION}/meta/{arm}")]

    def one(p):
        m = json.loads(b"".join(vol.read_file(p)))
        ends = [q for q in m["positions"] if q["kind"] == "output_end" and q["kept"]]
        spans = [q["span"][1] - q["span"][0] for q in ends]
        return (m["arm"], short_key(m["key"])), {"output tokens read": sum(spans), "mean output tokens": float(np.mean(spans)) if spans else 0.0,
                                      "tokens read": m["kept_tokens"], "whole-transcript tokens": m["n_tokens"],
                                      "environment": m["environment"], "read": {(q["step"], q["action"]) for q in ends}}

    with ThreadPoolExecutor(32) as pool:
        out = dict(pool.map(one, paths))
    missing = keys - set(out)
    if missing:
        raise ValueError(f"{len(missing)} runs without level-7 metadata")
    return out


def main():
    first = json.loads(PROBES["qwen3-32b"][0].read_text())
    runs = [tuple(r) for r in first["manifest"]["runs"]]
    env_fold = first["config"]["environment_to_fold"]
    score = {}
    for name, (path, want) in PROBES.items():
        d = json.loads(path.read_text())
        if [tuple(r) for r in d["manifest"]["runs"]] != runs:
            raise ValueError(f"{name}: a different run list")
        score[name] = np.array(d["headline_held_out_environment"]["output_end"]["real_vs_sim"]["all_outputs"]["run_scores"], float)
    keep = np.array([a in ARMS for a, _ in runs]) & np.isfinite(score["qwen3-32b"]) & np.isfinite(score["qwen3.5 (matched)"])
    idx = np.flatnonzero(keep)
    sel = [runs[i] for i in idx]
    L = lengths(set(sel))
    tf, task = {}, {}  # TF-IDF run score: the mean over the outputs the probe read (the protocol's shared outputs)
    with gzip.open(TFIDF[0], "rt") as fh:
        for r in map(json.loads, fh):
            k = (r["arm"], short_key(r["key"]))
            if r["comparison"] == "real_vs_sim" and r["outputs"] == "all" and k in L and (r["step"], r["action"]) in L[k]["read"]:
                tf.setdefault(k, []).append(r["score"])
                task[k] = r["main_task"]
    y = np.array([a == "sim-attack" for a, _ in sel], int)
    s = {name: score[name][idx] for name in PROBES}
    s["tf-idf"] = np.array([np.mean(tf[r]) if r in tf else np.nan for r in sel])
    if np.isnan(s["tf-idf"]).any():
        raise ValueError(f"{int(np.isnan(s['tf-idf']).sum())} runs without TF-IDF scores")
    main_task = np.array([task[r] for r in sel])
    f = np.array([env_fold[L[r]["environment"]] for r in sel])  # the protocol's held-out-environment folds
    feats = {k: np.array([L[r][k] for r in sel], float) for k in ("output tokens read", "mean output tokens", "tokens read", "whole-transcript tokens")}
    X = np.c_[np.log1p(feats["output tokens read"]), np.log1p(feats["whole-transcript tokens"]), np.ones(len(sel))]
    tasks, inv = np.unique(main_task, return_inverse=True)
    rng = np.random.default_rng(SEED)
    W = np.vstack([np.ones(len(tasks)), np.stack([np.bincount(rng.integers(0, len(tasks), len(tasks)), minlength=len(tasks)) for _ in range(N_BOOT)])])[:, inv].astype(float)
    S = lambda a: {"value": round(float(a[0]), 4), "ci95": [round(float(v), 4) for v in np.percentile(a[1:], [2.5, 97.5])]}
    out = {"runs": len(sel), "folds": int(len(np.unique(f))), "checks": {}, "length alone": {}, "length regressed out": {}}
    for name, want in [(n, w) for n, (_, w) in PROBES.items()] + [("tf-idf", TFIDF[1])]:
        got = float(wauroc(y, s[name], W[:1])[0])
        out["checks"][f"{name} run AUROC (protocol {want})"] = round(got, 4)
        if abs(got - want) > 0.002:
            raise ValueError(f"{name}: run AUROC {got:.4f} does not reproduce the protocol's {want}")
    for k, v in feats.items():  # direction from the training folds of each held-out environment
        oriented = np.empty(len(v))
        for e in np.unique(f):
            tr, te = f != e, f == e
            oriented[te] = v[te] * np.sign(v[tr & (y == 1)].mean() - v[tr & (y == 0)].mean())
        out["length alone"][k] = {"oriented": S(wauroc(y, oriented, W)), "raw (sim over real)": round(float(wauroc(y, v, W[:1])[0]), 4)}
    resids, betas = {}, {}
    for name in s:
        resid = np.empty(len(sel))
        for e in np.unique(f):
            tr, te = f != e, f == e
            beta = np.linalg.lstsq(X[tr], s[name][tr], rcond=None)[0]
            resid[te] = s[name][te] - X[te] @ beta
            betas.setdefault(name, {})[int(e)] = [round(float(b), 6) for b in beta]  # log output tokens, log transcript tokens, intercept
        resids[name] = resid
        a0, a1 = wauroc(y, s[name], W), wauroc(y, resid, W)
        out["length regressed out"][name] = {"original": S(a0), "length regressed out": S(a1), "change": S(a1 - a0),
                                            "correlation of score with log output tokens read": round(float(np.corrcoef(s[name], X[:, 0])[0, 1]), 3)}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    per_run = ROOT / ".run_cache" / "level7_length" / f"level7_length_runs_{stamp}.jsonl.gz"  # for S3: every run's inputs and residuals
    per_run.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(per_run, "wt") as fh:
        for i, (arm, key) in enumerate(sel):
            fh.write(json.dumps({"arm": arm, "key": key, "main_task": main_task[i], "environment": L[(arm, key)]["environment"],
                                 "fold": int(f[i]), **{k: float(v[i]) for k, v in feats.items()},
                                 **{f"score {n}": float(s[n][i]) for n in s}, **{f"residual {n}": float(resids[n][i]) for n in s}}) + "\n")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    path = HERE / "results" / f"level7_length_{stamp}.json"
    res = {"preliminary": True, "git": git, "per_run": per_run.name, "per_run_sha256": hashlib.sha256(per_run.read_bytes()).hexdigest(),
           "tfidf": TFIDF[0].name, "tfidf_sha256": hashlib.sha256(TFIDF[0].read_bytes()).hexdigest(), "extraction": EXTRACTION,
           "probes": {k: v[0].name for k, v in PROBES.items()}, "split": "held-out environment", "n_boot": N_BOOT, "seed": SEED,
           **out, "coefficients by fold (log output tokens, log transcript tokens, intercept)": betas}
    path.write_text("{\n" + ",\n".join(f" {json.dumps(k)}: {json.dumps(v)}" for k, v in res.items()) + "\n}\n")  # one key per line
    print(json.dumps(out, indent=1))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
