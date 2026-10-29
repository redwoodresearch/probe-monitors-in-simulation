"""Item (a), a stronger text baseline (Aswin, about 14:12 PT 10-10: "(3) (a) now"): logistic regression on a small language
model's text embeddings, on exactly gate (a)'s data, folds, comparisons and null, beside TF-IDF and the probe. One Modal
L4 container.

  data     gate (a)'s kept outputs per variant, from its own code and config (run_gate_a_modal.prepare, config/gate_a.yaml;
           its TF-IDF featurizing skipped): "all" (level 0), "level1" (times masked), "planted" (gate (a)'s positive
           control, `INFO: cache warm` added to one seeded output per simulated run; must reach planted_min, 0.90)
  embedder Qwen/Qwen3-Embedding-0.6B at a pinned revision, bf16, sdpa; each tool output embedded on its own with no
           instruction (the model card's document side), left padding, last-token pooling, L2-normalized, cut at 8,192
           tokens (the count of cut outputs is reported); each distinct text embedded once across variants
  probe    per held-out environment (gate (a)'s loeo design over all 10; headline on the 5 test environments, all 10
           alongside): standardized embeddings, L2 logistic regression at C = 1 (TF-IDF's C; no selection), the
           project's GPU fitter (torch_probe.fit), outputs labelled by their run's arm; run score = mean of its outputs'
           scores (gate (a)'s run_means); a per-column gradient check (<= 1e-4; refits at 4x iterations up to 80,000); an
           unconverged real-label fit, or any null column unconverged in a test-environment fold, means no verdict
  null     200 shuffled-label columns in the same fits (a full refit null): draw k flips every row of a main task with
           probability 1/2, rng seed + 1 + k (gate (a)'s rule); 2.5–97.5% range, p95 and p over the test environments
  interval 95%, resampling whole test environments (gate (a)'s run_auroc)
Writes the result and per-run scores (no text) to the Volume first, then results/embed_baseline_<stamp>.json and
data/embed_run_scores_<stamp>.json.gz.

  PYTHONPATH=src uv run --no-project --python 3.13 --with 'modal>=1.4.3' --with numpy --with pyyaml --with scikit-learn \
      --with scipy modal run --detach experiments/2026-10-09_fixed_sim/run_embed_baseline_modal.py
"""

import gzip
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REPO = HERE.parents[1] if modal.is_local() else Path("/root")
sys.path.insert(0, str(HERE) if modal.is_local() else "/root")
import run_gate_a_modal as G  # noqa: E402  (gate (a)'s code, unchanged)

APP = "probe-shift_aswin-surya_sub-fixed-sim_embed-base_kill-2026-11-10"
MODEL, REVISION = "Qwen/Qwen3-Embedding-0.6B", "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
MAX_TOKENS, BATCH_TOKENS, ATTN_PAIRS, C, MAX_ITER, MAX_ITER_CAP, TOL, DRAWS = 8192, 65536, 2**27, 1.0, 5000, 80000, 1e-4, 200
VARIANTS = {"all": ["fixed_sim_vs_e1", "e2_vs_e1"], "level1": ["fixed_sim_vs_e1", "e2_vs_e1"], "planted": ["fixed_sim_vs_e1"]}
ARCHIVED = HERE / "results" / "gate_a_main_2026-10-09_185451.json"

app = modal.App(APP)
hf = modal.Volume.from_name("probe-shift_aswin-surya_hf-cache")
image = (modal.Image.debian_slim(python_version="3.13")
         .uv_pip_install("numpy", "scipy", "scikit-learn==1.9.1", "joblib==1.6.0", "pyyaml", "torch==2.8.0", "transformers==4.56.2")
         .env({"JOBLIB_TEMP_FOLDER": "/tmp", "HF_HOME": "/hf"})
         .add_local_python_source("probe_shift")
         .add_local_file(G.Q1 / "analyze_tfidf.py", "/root/analyze_tfidf.py")
         .add_local_file(HERE / "run_gate_a_modal.py", "/root/run_gate_a_modal.py"))


def score_core(E, y, task, env, seed: int, draws: int, device: str, c: float | None = None):
    """Out-of-fold output scores [n, 1 + draws] for embeddings E [n, d] and labels y (0/1 per output): one fit per
    held-out environment with the real labels and `draws` shuffled columns (gate (a)'s per-main-task flips). Returns the
    label matrix, the scores and, per environment, each column's convergence."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    from probe_shift import torch_probe as tp

    c = C if c is None else c
    tasks_u, t_idx = np.unique(task, return_inverse=True)
    Y = np.stack([y] + [np.where((np.random.default_rng(seed + 1 + k).random(len(tasks_u)) < 0.5)[t_idx], 1 - y, y)
                        for k in range(draws)], 1).astype(np.float32)
    S, conv = np.full(Y.shape, np.nan), {}
    X = torch.from_numpy(np.asarray(E, np.float32)).to(device)
    for e in sorted(set(env)):
        tr, te = env != e, env == e
        Xa, Xb = tp.standardize(X[torch.from_numpy(tr).to(device)], X[torch.from_numpy(te).to(device)])
        Ya = torch.from_numpy(Y[tr]).to(device)
        mi = MAX_ITER
        while True:  # d9's refit rule: a fit that used all its iterations is refit with 4x as many, up to the cap
            W, b, evals = tp.fit(Xa, Ya, c, mi, tol=TOL / 10)
            Wg, bg = W.clone().requires_grad_(True), b.clone().requires_grad_(True)
            loss = (c * F.binary_cross_entropy_with_logits(Xa @ Wg + bg, Ya, reduction="sum") + 0.5 * (Wg * Wg).sum()) / len(Xa)
            loss.backward()
            g = torch.maximum(Wg.grad.abs().amax(0), bg.grad.abs()).cpu().numpy()
            if g.max() <= TOL or evals < mi or mi * 4 > MAX_ITER_CAP:
                break
            mi *= 4
        conv[str(e)] = {"evals": int(evals), "max_iter": mi, "converged": (g <= TOL).tolist()}
        S[te] = tp.decision(Xb, W, b).detach().cpu().numpy()
    return Y, S, conv


def summarize(Y, S, conv, d, env, split, cfg) -> dict:
    """Gate (a)'s run-level AUROC (run_auroc: run means, environment-resampled interval) on the 5 test environments and on
    all 10 for the real labels, and the null over the converged shuffled columns."""
    import numpy as np

    test = np.isin(env, split["test"])
    every = np.ones(len(env), bool)
    ok = np.all([c["converged"] for e, c in conv.items() if e in split["test"]], axis=0)  # the folds the test rows use
    real_ok = bool(np.all([c["converged"][0] for c in conv.values()]))  # every fold: all_envs uses them all
    out = {"test_envs": G.run_auroc(S[:, 0], Y[:, 0].astype(int), d, env, test, split["test"], cfg),
           "all_envs": G.run_auroc(S[:, 0], Y[:, 0].astype(int), d, env, every, split["dev"] + split["test"], cfg),
           "real_labels_converged": real_ok, "fits": {e: [c["evals"], c["max_iter"]] for e, c in conv.items()}}
    null = [G.run_auroc(S[:, k], Y[:, k].astype(int), d, env, test, split["test"], cfg, boot=False)["auroc"]
            for k in range(1, Y.shape[1]) if ok[k]]
    obs = out["test_envs"]["auroc"]
    out["null"] = {"draws_used": len(null), "dropped_unconverged": int((~ok[1:]).sum()),
                   "range": [float(np.quantile(null, 0.025)), float(np.quantile(null, 0.975))] if null else None,
                   "p95": float(np.quantile(null, 0.95)) if null else None,
                   "p": float((1 + sum(v >= obs for v in null)) / (1 + len(null))),
                   "complete": len(null) == Y.shape[1] - 1}  # gate (a)'s rule: every draw, or no verdict
    out["verdict_valid"] = real_ok and out["null"]["complete"]
    return out


@app.function(image=image, gpu="L4", cpu=4, memory=32768, volumes={"/acts": G.acts, "/hf": hf}, timeout=9000, retries=0)
def job(cfg: dict, split: dict, folds: dict, git: str) -> dict:
    import hashlib

    import numpy as np
    import torch
    import torch.nn.functional as F
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    import analyze_tfidf as A

    A.featurize = lambda texts, cfg: None  # prepare() featurizes for TF-IDF; only its texts are needed here
    t0, timing = time.time(), {}
    data = {v: G.prepare(cfg, v, split, folds) for v in VARIANTS}
    timing["prepare_s"] = round(time.time() - t0)
    texts = sorted({t for d in data.values() for t in d[3]})
    index = {t: i for i, t in enumerate(texts)}
    path = snapshot_download(MODEL, revision=REVISION)
    hf.commit()
    tok = AutoTokenizer.from_pretrained(path, padding_side="left")
    model = AutoModel.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    lens = np.array([len(x) for x in tok(texts)["input_ids"]])
    order, E, i = np.argsort(-lens, kind="stable"), np.zeros((len(texts), model.config.hidden_size), np.float32), 0
    t1 = time.time()
    while i < len(order):  # longest first; a batch holds about BATCH_TOKENS tokens and ATTN_PAIRS attention pairs
        L = int(min(lens[order[i]], MAX_TOKENS))
        n = max(1, min(BATCH_TOKENS // L, ATTN_PAIRS // (L * L)))
        idx = order[i:i + n]
        bd = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            h = model(**bd, use_cache=False).last_hidden_state[:, -1]  # left padding: the last position is each text's last token
        E[idx] = F.normalize(h.float(), p=2, dim=1).cpu().numpy()
        if i // 2000 != (i + n) // 2000:  # progress, to check the speed in the first minutes
            done = int(np.minimum(lens[order[:i + n]], MAX_TOKENS).sum())
            print(f"embedded {i + n}/{len(order)} texts, {done:,} tokens, {done / (time.time() - t1):,.0f} tokens/s", flush=True)
        i += n
    timing["embed_s"], timing["tokens"] = round(time.time() - t1), int(np.minimum(lens, MAX_TOKENS).sum())
    out_dir = Path("/acts/claimB") / f"embed_baseline_{git[:8]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings_fp16.npy", E.astype(np.float16))  # rows follow text_sha256.json; no text stored
    (out_dir / "text_sha256.json").write_text(json.dumps([hashlib.sha256(t.encode()).hexdigest() for t in texts]))
    G.acts.commit()
    res = {"model": MODEL, "revision": REVISION, "git": git, "distinct_texts": len(texts), "cut_at_8192": int((lens > MAX_TOKENS).sum()),
           "rows": {}, "run_scores": {}, "counts": {v: d[4] for v, d in data.items()}}
    t2 = time.time()
    fit_rows(res, data, E, index, cfg, split, None)
    timing["fit_s"], timing["total_s"] = round(time.time() - t2), round(time.time() - t0)
    return finish(res, timing, cfg, out_dir)


def fit_rows(res, data, E, index, cfg, split, c):
    """Every comparison x variant row: out-of-fold scores, the summary and run scores, into res."""
    import numpy as np

    import analyze_tfidf as A

    for variant, comps in VARIANTS.items():
        d, _, env, vtexts, _ = data[variant]
        for comp in comps:
            neg, pos = cfg["comparisons"][comp]
            R = np.isin(d["arm"], [neg, pos])
            rows = np.array([index[t] for t, r in zip(vtexts, R) if r])
            dd = {"key": d["key"][R]}
            Y, S, conv = score_core(E[rows], (d["arm"][R] == pos).astype(int), d["main_task"][R], env[R], cfg["seed"], DRAWS, "cuda", c)
            res["rows"][f"{comp}|loeo|{variant}"] = summarize(Y, S, conv, dd, env[R], split, cfg)
            means, first = A.run_means(S[:, 0], dd["key"], np.ones(R.sum(), bool))
            res["run_scores"][f"{comp}|{variant}"] = [{"key": str(dd["key"][j]), "arm": str(d["arm"][R][j]), "environment": str(env[R][j]),
                                                       "score": float(s)} for j, s in zip(first, means)]
            print(comp, variant, json.dumps({k: v for k, v in res["rows"][f"{comp}|loeo|{variant}"].items() if k != "fits"}), flush=True)


def finish(res, timing, cfg, out_dir, name="result.json"):
    res["timing"] = timing
    res["planted_check"] = {"auroc": res["rows"]["fixed_sim_vs_e1|loeo|planted"]["test_envs"]["auroc"], "minimum": cfg["planted_min"]}
    res["planted_check"]["passes"] = (res["planted_check"]["auroc"] >= cfg["planted_min"]
                                      if res["rows"]["fixed_sim_vs_e1|loeo|planted"]["real_labels_converged"] else "incomplete: unconverged fit")
    (out_dir / name).write_text(json.dumps(res, default=str))  # scores only; a copy in case the laptop client dies
    G.acts.commit()
    return json.loads(json.dumps(res, default=str))  # plain Python types only


@app.function(image=image, gpu="L4", cpu=4, memory=32768, volumes={"/acts": G.acts}, timeout=1500, retries=0)
def refit(cfg: dict, split: dict, folds: dict, git: str, src: str, c: float) -> dict:
    """Exploratory refit after the first fit's convergence failure (C chosen after seeing it): the saved embeddings of
    `claimB/embed_baseline_<src>/` (no re-embedding), rows mapped by text sha256, every row refit at C = c."""
    import hashlib

    import numpy as np

    import analyze_tfidf as A

    A.featurize = lambda texts, cfg: None
    t0 = time.time()
    data = {v: G.prepare(cfg, v, split, folds) for v in VARIANTS}
    base = Path("/acts/claimB") / f"embed_baseline_{src}"
    E = np.load(base / "embeddings_fp16.npy").astype(np.float32)
    index = {h: i for i, h in enumerate(json.loads((base / "text_sha256.json").read_text()))}
    index = {t: index[hashlib.sha256(t.encode()).hexdigest()] for d in data.values() for t in d[3]}
    res = {"model": MODEL, "revision": REVISION, "git": git, "refit_of": src, "C": c, "label": "refit after convergence failure; C chosen after "
           "seeing the first fit (exploratory)", "rows": {}, "run_scores": {}, "counts": {v: d[4] for v, d in data.items()}}
    fit_rows(res, data, E, index, cfg, split, c)
    return finish(res, {"total_s": round(time.time() - t0)}, cfg, base, f"refit_C{c:g}_{git[:8]}.json")


@app.local_entrypoint()
def main(refit_from: str = "", c: float = 0.01, no_wait: bool = False):
    """`--refit-from <first run's commit[:8]> [--c 0.01]`: the exploratory refit of the saved embeddings, no re-embedding.
    `--no-wait` (with --detach): spawn and return; the result is the worker's Volume file."""
    import numpy  # noqa: F401  (fail before launching if the client environment is incomplete)
    import yaml

    if subprocess.run(["git", "status", "--porcelain", str(Path(__file__))], capture_output=True, text=True, check=True).stdout:
        raise SystemExit(f"commit {Path(__file__).name} first: the result records the commit that ran")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    cfg = yaml.safe_load((HERE / "config" / "gate_a.yaml").read_text())
    split = json.loads((HERE / cfg["split_file"]).read_text())
    folds = json.loads((REPO / "splits" / cfg["folds_file"]).read_text())
    if no_wait:
        call = refit.spawn(cfg, split, folds, git, refit_from, c) if refit_from else job.spawn(cfg, split, folds, git)
        print(f"spawned {call.object_id}; the result goes to the Volume (claimB/embed_baseline_*/)", flush=True)
        return
    res = refit.remote(cfg, split, folds, git, refit_from, c) if refit_from else job.remote(cfg, split, folds, git)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "data").mkdir(exist_ok=True)
    with open(HERE / "data" / f"embed_run_scores_{stamp}.json.gz", "xb") as fh:  # keys and scores, no text
        fh.write(gzip.compress(json.dumps(res.pop("run_scores")).encode(), mtime=0))
    with open(HERE / "results" / f"embed_baseline_{stamp}.json", "x") as fh:
        fh.write(json.dumps(res, indent=1))
    tfidf = json.loads(ARCHIVED.read_text())["rows"]
    for row, v in res["rows"].items():
        t, n = v["test_envs"], v["null"]
        print(f"{row}: embeddings {t['auroc']:.3f} [{t['ci95_env_resample'][0]:.3f}, {t['ci95_env_resample'][1]:.3f}], null "
              f"{n['range']} (p {n['p']:.3f}, {n['draws_used']} draws) | TF-IDF {tfidf.get(row, {}).get('test_envs', {}).get('auroc')}")
    print("planted check", res["planted_check"], "| timing", res["timing"], "->", HERE / "results" / f"embed_baseline_{stamp}.json")
