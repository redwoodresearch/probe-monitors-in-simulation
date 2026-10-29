"""Row 2, step 4 (RULE.md here): B2, realism-probe separation, run after the predictions commits. For each configuration, a
logistic regression on each run's mean output-readout vector at that layer, real e2 vs simulated, over the configuration's
retained cells (manifest part 2: all 939 for every configuration). The vector is the mean of the run's kept output_end rows
(the reader's dump: Qwen3-32B 2026-10-06_la_segment-end_qwen3-32b_alllayers_v1, Olmo-3.1-32B and Llama-3.3-70B their part-1
dumps). Standardization is fold-local; 5 folds grouped by main task: the cells' main tasks sorted, permuted by
numpy.random.default_rng(20261012), and dealt round-robin; C = 0.01 fixed (scikit-learn LogisticRegression, L2, lbfgs). The
out-of-fold run-level AUROC, simulated as the positive class. It loads no detector weights, thresholds or nested catch scores:
only run metadata and activations. One Modal CPU container per reader; activations stay on Modal; a run with no meta or no kept
output_end row fails loudly.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with numpy modal run experiments/2026-10-12_plantslip_heldout/q2_b2.py \\
      --qwen-runs <#36 runs.json> --olmo-runs <Olmo runs.json> --llama-runs <Llama runs.json>
"""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-heldout_b2_kill-2026-11-12"
SEED, C, FOLDS = 20261012, 0.01, 5
READERS = {"qwen3-32b": ("probe-shift_aswin-surya_sub-la-activations", "2026-10-06_la_segment-end_qwen3-32b_alllayers_v1", [20, 28, 36, 44, 52, 60]),
           "olmo": ("probe-shift_aswin-surya_sub-q2-olmo_acts_kill-2026-11-12", "2026-10-12_la_output-end-r2_olmo31-32b_alllayers_v1", [16, 24, 48, 63]),
           "llama": ("probe-shift_aswin-surya_sub-q2-llama_acts_kill-2026-11-12", "2026-10-12_la_output-end-r2j_llama33-70b_alllayers_v1", [20, 30, 40, 50, 79])}
RUNS_SHA = {"qwen3-32b": "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96",  # #36's runs.json (fix939's record)
            "olmo": "e2c86cf48983e6c93bb4ade799e47e08b2aaca0201f4a7244a5642b5de3c8784",       # run 2026-10-12_110549's data/runs.json
            "llama": "ad10475afccc285bde39a389a17cb8b278a15d670c5d673933a094c6ffbc5254"}      # run 2026-10-12_143129's data/runs.json
ANCESTORS = ("c5a2fd2f", "434b1f2c")
PREDICTIONS = {"predictions_2026-10-12_204016.json": "15d4bcfef8c6575b4a6f2e8d271b592bff18a51c44f06de06dbd259d95a0cad8",
               "predictions_2026-10-12_212633.json": "5b617fcca97bae442f7b9ac1dda407cbd17c415161a65aad1b52caf676486ee3"}
MANIFEST2 = ("manifest_part2_2026-10-12_212356.json", "0911c962646e08f60097b2e1123b12240d3309849927bb3dfcb9f92c29d64f68")

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "scikit-learn==1.7.2")
VOLS = {f"/{r}": modal.Volume.from_name(v, version=2).read_only() for r, (v, _, _) in READERS.items()}


def filename(key: str) -> str:
    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


@app.function(image=image, volumes=VOLS, cpu=8, memory=32768, timeout=3600, retries=0)
def b2(reader: str, items: list[dict], fold: list[int]) -> dict:
    """items: {key, filename, arm} for the configuration cells' real e2 and simulated runs; fold: each item's fold."""
    try:
        return _b2(reader, items, fold)
    except Exception as e:  # noqa: BLE001 — sanitized and re-raised, never swallowed
        raise RuntimeError(f"b2 failed: {type(e).__name__}") from None


def _b2(reader, items, fold):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from concurrent.futures import ThreadPoolExecutor
    _, run, layers = READERS[reader]
    root = Path(f"/{reader}") / run
    metas = []
    for it in items:
        m = json.loads((root / "meta" / it["arm"] / f"{filename(it['key'])}.json").read_text())
        rows = [m["row_start"] + q["row"] for q in m["positions"] if q["kind"] == "output_end" and q["kept"]]
        if m["key"] != it["key"] or not rows:
            raise ValueError("a run with no meta match or no kept output_end row")
        metas.append((m["chunk"], rows))
    y = np.array([1 if it["arm"] == "sim-attack" else 0 for it in items]); fold = np.array(fold)
    by_chunk = {}
    for j, (c, rows) in enumerate(metas):
        by_chunk.setdefault(c, []).append((j, rows))

    def one_layer(layer):
        X = None
        for c, js in by_chunk.items():
            A = np.load(root / "by_layer" / f"L{layer}" / f"{c}.npy")  # whole file: row-indexed mmap is slow on a Volume
            if X is None:
                X = np.zeros((len(items), A.shape[1]), np.float32)
            for j, rows in js:
                X[j] = A[rows].astype(np.float32).mean(0)
        if not np.isfinite(X).all():
            raise ValueError("a non-finite mean vector")
        oof, iters = np.full(len(items), np.nan), []
        for f in range(FOLDS):
            tr, te = fold != f, fold == f
            sc = StandardScaler().fit(X[tr])
            lr = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(X[tr]), y[tr])
            oof[te] = lr.decision_function(sc.transform(X[te])); iters.append(int(lr.n_iter_[0]))
        pos, neg = oof[y == 1], oof[y == 0]
        s = np.r_[pos, neg]; r = s.argsort(kind="mergesort").argsort(kind="mergesort").astype(float)
        u, inv = np.unique(s, return_inverse=True); rr = np.zeros(len(u)); np.add.at(rr, inv, r); r = (rr / np.bincount(inv))[inv]
        auc = float((r[: len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))
        return str(layer), {"auroc (simulated positive, out of fold)": auc, "runs": len(items), "simulated": int(y.sum()), "real e2": int((1 - y).sum()),
                            "iterations per fold": iters, "any fold at max_iter": any(i >= 5000 for i in iters), "oof": oof.tolist()}
    with ThreadPoolExecutor(len(layers)) as pool:
        return dict(pool.map(one_layer, layers))


@app.local_entrypoint()
def main(qwen_runs: str, olmo_runs: str, llama_runs: str):
    import numpy as np
    for f in (Path(__file__).resolve(), HERE / "RULE.md"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    for a in ANCESTORS:  # step 4 runs only after the predictions commits
        if subprocess.run(["git", "merge-base", "--is-ancestor", a, "HEAD"], cwd=HERE).returncode:
            raise SystemExit(f"{a} is not an ancestor of HEAD")
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    for n, h in PREDICTIONS.items():
        if sha(HERE / "results" / n) != h:
            raise SystemExit(f"{n} differs from its recorded sha256")
    if sha(HERE / "results" / MANIFEST2[0]) != MANIFEST2[1]:
        raise SystemExit("not manifest part 2's supplement")
    gates = json.loads((HERE / "results" / MANIFEST2[0]).read_text())["gate outcomes"]["false-alarm and convergence gates, eligibility, masks"]
    paths = {"qwen3-32b": qwen_runs, "olmo": olmo_runs, "llama": llama_runs}
    calls, meta = {}, {}
    for reader, (_, _, layers) in READERS.items():
        if sha(paths[reader]) != RUNS_SHA[reader]:
            raise SystemExit(f"{reader}: runs.json differs from its pin")
        if any(gates[f"{reader} @ L{l}"]["retained cells"] != 939 or not gates[f"{reader} @ L{l}"]["masks final"] for l in layers):
            raise SystemExit(f"{reader}: a configuration does not retain all 939 cells")
        runs = json.loads(Path(paths[reader]).read_text())
        e2 = {r["cell"]: r for r in runs if r["arm"] == "real-attack-e2"}; sim = {r["cell"]: r for r in runs if r["arm"] == "sim-attack"}
        cells = sorted(e2)
        if len(cells) != 939 or set(sim) != set(cells):
            raise SystemExit(f"{reader}: not 939 paired cells")
        items = [{"key": e2[c]["key"], "filename": e2[c]["filename"], "arm": "real-attack-e2"} for c in cells] + \
                [{"key": sim[c]["key"], "filename": sim[c]["filename"], "arm": "sim-attack"} for c in cells]
        tasks = sorted({e2[c]["main_task"] for c in cells})
        perm = np.random.default_rng(SEED).permutation(len(tasks)); fold_of = {tasks[p]: i % FOLDS for i, p in enumerate(perm)}
        fold = [fold_of[e2[c]["main_task"]] for c in cells] * 2
        meta[reader] = {"tasks": len(tasks), "keys": [it["key"] for it in items], "arms": [it["arm"] for it in items], "fold": fold}
        calls[reader] = b2.spawn(reader, items, fold)
    res = {r: c.get() for r, c in calls.items()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    npz = HERE / "results" / f"b2_oof_{stamp}.npz"
    np.savez(npz, **{f"{r}__L{l}": np.array(v["oof"]) for r, d in res.items() for l, v in d.items()},
             **{f"{r}__keys": np.array(meta[r]["keys"]) for r in res}, **{f"{r}__arms": np.array(meta[r]["arms"]) for r in res}, **{f"{r}__fold": np.array(meta[r]["fold"]) for r in res})
    out = {"rule": "RULE.md step 4, B2 (this folder)", "git": head, "seed": SEED, "C": C, "folds": FOLDS,
           "fold rule": "the cells' main tasks sorted, permuted by numpy.random.default_rng(20261012), dealt round-robin into 5 folds",
           "inputs sha256": {**PREDICTIONS, MANIFEST2[0]: MANIFEST2[1], **{f"{r} runs.json": RUNS_SHA[r] for r in READERS}},
           "dumps": {r: f"{v}:{run}" for r, (v, run, _) in READERS.items()},
           "configurations": {f"{r} @ L{l}": {k: v for k, v in d.items() if k != "oof"} for r, dd in res.items() for l, d in dd.items()},
           "oof scores file": {"file": npz.name, "sha256": sha(npz)}}
    with open(dest := HERE / "results" / f"b2_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    for k, v in out["configurations"].items():
        print(k, round(v["auroc (simulated positive, out of fold)"], 4), v["iterations per fold"], "max_iter hit" if v["any fold at max_iter"] else "")
    print(dest)
