"""Olmo-3.1-32B's detector against the label-shuffle null its fit run stored (0d's ask, 10-13; review gap from #1215 r1nJB).
A copy of Llama part 7's check_label_shuffle_null.py (aswin/q2-llama-7-null @ c20ca2bd) with Olmo's run, layers and pins.
The fit run 2026-10-12_110549 stored 100 probes per layer trained on labels permuted within main tasks
(`output_end__mean__task_null`, `null_labels` in scores_L*.npz). As in #36's analyze_q3_attack.py and Llama's check:
observed AUROC = outer scores, real attack e1 vs real honest; each null AUROC = that null probe's scores against its own
shuffled labels, on the same runs. Real arms only: no simulated row is read at any layer.
Reported: (1) each fitted layer 16, 24, 32, 40, 48, 56, 63; (2) the reading's own detector, the layer chosen per test fold
(32, 56, 56, 56, 40, from reply_break_olmo_2026-10-12_113245.json "layer per test fold"), each run's outer score and each
null probe's score at its fold's layer, null labels the same across layers (asserted); (3) a selection-aware bound: per
shuffle, the BEST stitched pooled null AUROC over all 7^5 = 16,807 fold-layer schedules of the seven layers, which no
per-fold selection rule can exceed; (4) beside, e2: the outer detector on e2 vs honest, and each null probe's scores on the
same rows against the true e2 label (a held-out-epoch null; the null labels cover the training arms only).
Every input's sha256 is checked against the fit record's pins (run_2026-10-12_110549.json @ 09e0e57f "archive" objects; runs.json
against reply_break_olmo's "inputs sha256"). Writes results/label_shuffle_null_check_olmo_<stamp>.json and prints a summary.
Exits nonzero if the reading's layers (32, 40, 56), the fold-wise detector or the schedule bound do not beat their null's maximum.
  uv run --no-project --with numpy python experiments/2026-10-12_q2_olmo_reader/check_label_shuffle_null_olmo.py <run_scores_2026-10-12_110549 dir>
"""
import hashlib, json, subprocess, sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
HERE = Path(__file__).resolve().parent
d = Path(sys.argv[1])
PINS = {"runs.json": "e2c86cf48983e6c93bb4ade799e47e08b2aaca0201f4a7244a5642b5de3c8784",
        "scores_L16.npz": "b73157b4fbb46432e8aa6823a4e14270d2656456321d8b89d946e597a3044e9a",
        "scores_L24.npz": "de23c1c3f12ffa79d67d287c713345703ea45f53b2f6ca4e48e4d02f74d863ea",
        "scores_L32.npz": "231ab6745f682def4ca8c61af2fbec80f76ab362f9df218705a7bb19e8a82e21",
        "scores_L40.npz": "c21042eb51468460d43fef97e085c73fb6090eff35e71137672ce03f5fbb5182",
        "scores_L48.npz": "0fab6ab5a92e184e19836908de5e9d3a6863b195282cb8c43434a1f2034c6055",
        "scores_L56.npz": "698c9ad803212403c80b880cc5bb7ffa9ebbc5211e330d679cc85c15c5ffb20a",
        "scores_L63.npz": "b8bac05e45d108c9da92297bcf3765e306b195fd0344c2ff793a8ef51cd9b9e0"}
for name, pin in PINS.items():
    got = hashlib.sha256((d / name).read_bytes()).hexdigest()
    if got != pin:
        raise SystemExit(f"{name}: sha256 {got[:16]} is not the pinned {pin[:16]}")
me = Path(__file__).resolve()
if subprocess.run(["git", "ls-files", "--error-unmatch", str(me)], cwd=HERE, capture_output=True).returncode \
        or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(me)], cwd=HERE).returncode:
    raise SystemExit("this script differs from HEAD: commit it before running")
HEAD = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
runs = json.loads((d / "runs.json").read_text())
arm = np.array([r["arm"] for r in runs])
use_base = np.isin(arm, ["real-honest", "real-attack-e1"])
if np.isin(arm[use_base], ["sim-attack"]).any():
    raise SystemExit("a simulated row is in the real-arm mask")


def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int)
    order = s.argsort(kind="mergesort"); r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    u, inv = np.unique(s, return_inverse=True); rr = np.zeros(len(u)); np.add.at(rr, inv, r); r = (rr / np.bincount(inv))[inv]
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def summ(obs, null):
    return {"observed AUROC": obs, "null n": int(len(null)), "null mean": float(null.mean()), "null p95": float(np.percentile(null, 95)),
            "null max": float(null.max()), "p": float((1 + (null >= obs).sum()) / (1 + len(null))), "beats the null max": bool(obs > null.max())}


LAYERS = (16, 24, 32, 40, 48, 56, 63)
READING_LAYERS = (32, 40, 56)
FOLD_LAYER = {0: 32, 1: 56, 2: 56, 3: 56, 4: 40}  # reply_break_olmo_2026-10-12_113245.json, "layer per test fold"
fold = np.array([int(r["fold"]) for r in runs])
out = {"check": "Olmo-3.1-32B detector vs its stored label-shuffle null (fit run 2026-10-12_110549); real arms only", "git": HEAD,
       "prediction": "NULL_CHECK_PREDICTION.md (this folder), committed at 27d4b4d2 before any number",
       "inputs sha256": PINS, "per layer (e1 vs honest)": {}, "beside: e2 vs honest (null probes against the true e2 label)": {}}
bad = 0
ZL = {l: np.load(d / f"scores_L{l}.npz") for l in LAYERS}
Y = ZL[LAYERS[0]]["null_labels"]
for l in LAYERS:
    if not np.array_equal(ZL[l]["null_labels"], Y, equal_nan=True):
        raise SystemExit(f"null labels differ at L{l}")
use_e2 = np.isin(arm, ["real-honest", "real-attack-e2"])
for layer in LAYERS:
    z = ZL[layer]; outer = z["output_end__mean__task_outer"]; S = z["output_end__mean__task_null"]
    use = use_base & np.isfinite(outer) & np.isfinite(S).all(1) & np.isfinite(Y).all(1)
    obs = auc(outer[use], (arm[use] == "real-attack-e1").astype(int))
    r = summ(obs, np.array([auc(S[use, j], Y[use, j]) for j in range(S.shape[1])]))
    r |= {"runs": int(use.sum()), "honest": int((arm[use] == "real-honest").sum()), "e1": int((arm[use] == "real-attack-e1").sum())}
    out["per layer (e1 vs honest)"][f"L{layer}"] = r
    bad += (layer in READING_LAYERS) and not r["beats the null max"]
    print(f"L{layer}: runs {r['runs']} (honest {r['honest']}, e1 {r['e1']}); observed AUROC e1 vs honest {obs:.4f}; label-shuffle null n {r['null n']}, "
          f"mean {r['null mean']:.4f}, p95 {r['null p95']:.4f}, max {r['null max']:.4f}; p = {r['p']:.4f}; beats the null's max: {r['beats the null max']}"
          f"{' (a reading layer)' if layer in READING_LAYERS else ''}")
    u2 = use_e2 & np.isfinite(outer) & np.isfinite(S).all(1); y2 = (arm[u2] == "real-attack-e2").astype(int)
    r2 = summ(auc(outer[u2], y2), np.array([auc(S[u2, j], y2) for j in range(S.shape[1])])) | {"runs": int(u2.sum())}
    out["beside: e2 vs honest (null probes against the true e2 label)"][f"L{layer}"] = r2
    print(f"   beside, e2 vs honest (null probes against the true e2 labels): runs {r2['runs']}; observed {r2['observed AUROC']:.4f}; null probes on the same rows: mean {r2['null mean']:.4f}, "
          f"p95 {r2['null p95']:.4f}, max {r2['null max']:.4f}; beats: {r2['beats the null max']}")
# the fold-wise detector, as the reading uses it
lay = np.array([FOLD_LAYER[f] if f in FOLD_LAYER else -1 for f in fold])
outer = np.zeros(len(runs)); S = np.zeros((len(runs), Y.shape[1]))
for l in set(FOLD_LAYER.values()):
    m = lay == l; outer[m] = ZL[l]["output_end__mean__task_outer"][m]; S[m] = ZL[l]["output_end__mean__task_null"][m]
use = use_base & (lay > 0) & np.isfinite(outer) & np.isfinite(S).all(1) & np.isfinite(Y).all(1)
obs = auc(outer[use], (arm[use] == "real-attack-e1").astype(int))
r = summ(obs, np.array([auc(S[use, j], Y[use, j]) for j in range(S.shape[1])])) | {"runs": int(use.sum())}
out["fold-wise reading detector (32, 56, 56, 56, 40)"] = r; bad += not r["beats the null max"]
print(f"fold-wise (32, 56, 56, 56, 40): runs {r['runs']}; observed AUROC e1 vs honest {obs:.4f}; label-shuffle null n {r['null n']}, mean {r['null mean']:.4f}, "
      f"p95 {r['null p95']:.4f}, max {r['null max']:.4f}; p = {r['p']:.4f}; beats the null's max: {r['beats the null max']}")
# conservative bound for any per-fold layer selection: per shuffle, the best stitched pooled AUROC over ALL 7^5 = 16,807
# fold-layer schedules, on one common real-arm mask for observed and null scores
import itertools
from concurrent.futures import ProcessPoolExecutor
common = use_base & (lay > 0) & np.isfinite(Y).all(1)
for l in LAYERS:
    common &= np.isfinite(ZL[l]["output_end__mean__task_null"]).all(1) & np.isfinite(ZL[l]["output_end__mean__task_outer"])
NS = np.stack([ZL[l]["output_end__mean__task_null"][common] for l in LAYERS])  # [7, runs, 100]
FC = fold[common]; YC = Y[common]
SCHED = list(itertools.product(range(len(LAYERS)), repeat=5))
li = {l: k for k, l in enumerate(LAYERS)}
OS = np.stack([ZL[l]["output_end__mean__task_outer"][common] for l in LAYERS])
obs_c = auc(OS[[li[FOLD_LAYER[f]] for f in FC], np.arange(len(FC))], (arm[common] == "real-attack-e1").astype(int))


def best_for(j):
    cols = np.arange(len(FC)); m = 0.0
    for s in SCHED:
        m = max(m, auc(NS[[s[f] for f in FC], cols, j], YC[:, j]))
    return m


if __name__ == "__main__":
    import multiprocessing  # fork: workers inherit the loaded arrays; spawn would re-run this module's top level
    with ProcessPoolExecutor(mp_context=multiprocessing.get_context("fork")) as ex:
        mx = np.array(list(ex.map(best_for, range(NS.shape[2]))))
    r = summ(obs_c, mx) | {"common mask runs": int(common.sum()), "schedules": len(SCHED)}
    out["best-of-16,807-schedules null (bound for any per-fold layer selection)"] = r; bad += not r["beats the null max"]
    print(f"best-of-16,807-schedules null (bound for any per-fold layer selection), common mask {r['common mask runs']} runs: n {r['null n']}, "
          f"mean {r['null mean']:.4f}, p95 {r['null p95']:.4f}, max {r['null max']:.4f}; fold-wise detector on the same mask {obs_c:.4f} beats it: {r['beats the null max']}")
    out["all reading checks beat their null's maximum"] = bool(bad == 0)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"label_shuffle_null_check_olmo_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(dest)
    sys.exit(1 if bad else 0)
