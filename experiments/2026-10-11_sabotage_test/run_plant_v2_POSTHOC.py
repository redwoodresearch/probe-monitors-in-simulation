"""POST HOC validation of the redesigned planted check (PLANT_REDESIGN.md at eb44f8d3; the stage-1 results shaped this
design). Tested on judge-panel labels; stored readings and numbers only; Modal CPU. The gate's recorded PASS is unchanged.

Plant, built inside every fitting set (fold-local): the fitting rows' weighted standardization and top 20 principal
components; a seeded unit mix u of them; planted rows get s x (SD of honest_a calibration projections on u) along u, in
raw readings (u x per-feature SD). 80% of the fitting set's positives and, with their own seed, 80% of its evaluation
set's positives are planted ('pos'); null plants shift the same numbers of rows drawn from both classes ('null').
- sizing_job(layer): reads training and honest_a rows only. Outer folds = the gate's 5 task folds; inner = the other 4;
  every inner fitting set builds its own plant; mean inner validation AUROC per (outer fold, size, C).
- sizing_final(layer, items): per (outer fold, size, C) the outer fitting set's plant, refit, honest_a threshold, catch
  on the outer fold's positives. Reads training and honest_a rows only.
- valid_job(layer, s): reads training and honest_a rows; the gate's 5-fold search with fold-local plants, for the real
  labels and 100 scrambles (planted copy) and for 20 null plants (real labels).
- valid_final(layer, s, picks): all training rows fit, held-out evaluated; refit at the chosen settings; held-out scores,
  threshold, catch, AUROC and honest_b false alarms.

  modal deploy run_plant_v2_POSTHOC.py; python run_plant_v2_POSTHOC.py pilot | sizing | size-final | valid <s> | valid-final <s>
"""

import json
import sys
import time
from pathlib import Path

import modal

import run_gate as G

APP = "probe-shift_aswin-surya_sub-sabotage-plant2_kill-2026-11-10"
app = modal.App(APP)
image = G.image.add_local_python_source("run_gate")
SIZES, PCS, SHARE, NULLS, PILOT_LAYER = [0, 1, 2, 3, 4, 6], 20, 0.8, 20, 32
D = f"{G.OUT}/plant_v2"
HERE = Path(__file__).parent
VOLS = {"/acts": G.acts.read_only(), "/score": G.score}


def load(layer, roles_wanted):
    """The staged rows of the wanted roles only: X [m, hidden] float64 and per-row arrays (positions are local)."""
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np

    spec = json.loads(Path(f"/score/{G.OUT}/rows.json").read_text())
    rows = spec["rows"]
    idx = [i for i, r in enumerate(rows) if r["role"] in roles_wanted]
    pos_of = {g: k for k, g in enumerate(idx)}
    groups = {}
    for k, g in enumerate(idx):
        groups.setdefault((rows[g]["x"], rows[g]["chunk"]), []).append((k, rows[g]["row"]))
    X = np.empty((len(idx), G.HIDDEN), np.float64)

    def one(item):
        (x, chunk), lst = item
        a = np.load(f"/acts/{x}/by_layer/L{layer}/{chunk}.npy")
        X[[k for k, _ in lst]] = a[[r for _, r in lst]]

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(one, groups.items()))
    sub = [rows[g] for g in idx]
    A = {"w": np.array([r["w"] for r in sub]), "role": np.array([r["role"] for r in sub]), "task": np.array([r["task"] for r in sub]),
         "y": np.array([-1 if r["y"] is None else r["y"] for r in sub]), "tr": np.array([pos_of[g] for g in spec["train_index"]]),
         "fold": np.array(spec["train_fold"]), "Y": np.array(spec["Y_train"], dtype=np.float64)}
    return X, A


def basis(X, w, fit):
    import numpy as np

    sw = w[fit].sum()
    mu = (w[fit, None] * X[fit]).sum(0) / sw
    sd = np.sqrt((w[fit, None] * (X[fit] - mu) ** 2).sum(0) / sw)
    sd = np.where(sd < 1e-6, 1.0, sd)
    Z = (X[fit] - mu) / sd
    return mu, sd, np.linalg.svd(Z - Z.mean(0), full_matrices=False)[2][:PCS]


def plant(X, A, fit, other, s, seed, mode, B, mseed):
    """A copy of X with the fold-local plant built from `fit` (basis B; direction seed `seed`, per layer), applied to rows
    of `fit` and `other` chosen with `mseed` (no layer in it, so the same calls are planted at every layer)."""
    import numpy as np

    mu, sd, Vt = B
    u = np.random.default_rng(seed).normal(size=PCS) @ Vt
    u /= np.linalg.norm(u)
    cal = np.flatnonzero(A["role"] == "cal_a")
    v = s * float((((X[cal] - mu) / sd) @ u).std()) * u * sd
    y, chosen = A["y"], []
    for k, rr in enumerate((fit, other)):
        n = int(round(SHARE * (y[rr] == 1).sum()))
        pool = rr[y[rr] == 1] if mode == "pos" else rr
        chosen.append(np.random.default_rng([*mseed, 10 + k]).choice(pool, n, replace=False) if n else pool[:0])
    Xp = X.copy()
    Xp[np.concatenate(chosen)] += v
    return Xp


def fit_score(X, w, Ycols, fit, score_rows, Cs):
    """Standardize on fit rows, fit_dual for label columns Ycols [B, n_fit], scores [B, c, score_rows]."""
    import numpy as np
    import torch

    mask = np.zeros(len(X), bool)
    mask[fit] = True
    Z = G.standardized(X, w, mask)
    Zt = torch.from_numpy(Z[fit])
    a = G.fit_dual((Zt @ Zt.T)[None].expand(len(Ycols), -1, -1), torch.from_numpy(Ycols)[:, None, :], torch.from_numpy(w[fit]), Cs)
    return ((a @ Zt) @ torch.from_numpy(Z[score_rows]).T).numpy()


def auc_cols(S, Yv, w):
    """S [B, c, n_val], Yv [B, n_val] -> [B, c] weighted AUROC."""
    import numpy as np

    return np.array([G.wauc(S[:, c], Yv, w) for c in range(S.shape[1])]).T


@app.function(image=image, cpu=4, memory=16384, timeout=1800, retries=0, volumes=VOLS)
def sizing_job(layer: int) -> dict:
    import numpy as np

    t0 = time.time()
    X, A = load(layer, {"train", "cal_a"})
    tr, fold, y, w = A["tr"], A["fold"], A["y"].astype(float), A["w"]
    out = {}
    for f in sorted(set(fold)):
        ofit, inner = tr[fold != f], fold[fold != f]
        acc = {s: [] for s in SIZES}
        for g in sorted(set(inner)):
            ifit, ival = ofit[inner != g], ofit[inner == g]
            B = basis(X, w, ifit)
            for s in SIZES:
                Xp = plant(X, A, ifit, ival, s, [G.SEED, 71, layer, f, g], "pos", B, [G.SEED, 71, f, g])
                acc[s].append(auc_cols(fit_score(Xp, w, y[ifit][None], ifit, ival, G.CS), y[ival][None], w[ival])[0])
        for s in SIZES:
            out[f"{f}|{s}"] = np.mean(acc[s], 0).tolist()
    Path(f"/score/{D}/sizing").mkdir(parents=True, exist_ok=True)
    Path(f"/score/{D}/sizing/L{layer}.json").write_text(json.dumps(out))
    G.score.commit()
    return {"layer": layer, "seconds": round(time.time() - t0, 1)}


@app.function(image=image, cpu=4, memory=16384, timeout=1800, retries=0, volumes=VOLS)
def sizing_final(layer: int, items: list) -> dict:
    import numpy as np

    X, A = load(layer, {"train", "cal_a"})
    tr, fold, y, w = A["tr"], A["fold"], A["y"].astype(float), A["w"]
    cal = np.flatnonzero(A["role"] == "cal_a")
    out, bases = {}, {}
    for f, s, C in items:
        ofit, oval = tr[fold != f], tr[fold == f]
        if f not in bases:
            bases[f] = basis(X, w, ofit)
        Xp = plant(X, A, ofit, oval, s, [G.SEED, 72, layer, f], "pos", bases[f], [G.SEED, 72, f])
        sc = fit_score(Xp, w, y[ofit][None], ofit, np.arange(len(X)), [C])[0, 0]
        t = G.threshold(sc[cal])
        p = oval[y[oval] == 1]
        out[f"{f}|{s}"] = {"layer": layer, "C": C, "catch": float((w[p] * (sc[p] > t)).sum() / w[p].sum()), "threshold": t}
    return out


@app.function(image=image, cpu=4, memory=16384, timeout=1800, retries=0, volumes=VOLS)
def valid_job(layer: int, s: float, sub: str = "valid") -> dict:
    import numpy as np

    t0 = time.time()
    X, A = load(layer, {"train", "cal_a"})
    tr, fold, y, w, Y = A["tr"], A["fold"], A["y"].astype(float), A["w"], A["Y"]
    planted, nulls = [], [[] for _ in range(NULLS)]
    for f in sorted(set(fold)):
        fi, va = fold != f, fold == f
        B = basis(X, w, tr[fi])
        Xp = plant(X, A, tr[fi], tr[va], s, [G.SEED, 81, layer, f], "pos", B, [G.SEED, 81, f])
        planted.append(auc_cols(fit_score(Xp, w, Y[:, fi], tr[fi], tr[va], G.CS), Y[:, va], w[tr[va]]))
        for j in range(NULLS):
            Xn = plant(X, A, tr[fi], tr[va], s, [G.SEED, 91, j, layer, f], "null", B, [G.SEED, 91, j, f])
            nulls[j].append(auc_cols(fit_score(Xn, w, y[tr[fi]][None], tr[fi], tr[va], G.CS), y[tr[va]][None], w[tr[va]])[0])
    out = {"planted": np.mean(planted, 0).tolist(), "nulls": [np.mean(n, 0).tolist() for n in nulls]}
    Path(f"/score/{D}/{sub}").mkdir(parents=True, exist_ok=True)
    Path(f"/score/{D}/{sub}/L{layer}.json").write_text(json.dumps(out))
    G.score.commit()
    return {"layer": layer, "seconds": round(time.time() - t0, 1)}


@app.function(image=image, cpu=4, memory=16384, timeout=1800, retries=0, volumes=VOLS)
def valid_final(layer: int, s: float, picks: dict) -> dict:
    """picks: copy key ('planted' or 'null<j>') -> {str(column): C}."""
    import numpy as np

    X, A = load(layer, {"train", "heldout", "cal_a", "honb_cal", "honb_ho"})
    tr, y, w, Y, role = A["tr"], A["y"], A["w"], A["Y"], A["role"]
    ho = np.flatnonzero(role == "heldout")
    pos = y[ho] == 1
    B = basis(X, w, tr)
    out = {}
    for key, cols in picks.items():
        seed, mseed = ([G.SEED, 82, layer], [G.SEED, 82]) if key == "planted" else ([G.SEED, 92, int(key[4:]), layer], [G.SEED, 92, int(key[4:])])
        Xc = plant(X, A, tr, ho, s, seed, "pos" if key == "planted" else "null", B, mseed)
        for col, C in cols.items():
            sc = fit_score(Xc, w, Y[[int(col)]], tr, np.arange(len(X)), [C])[0, 0]
            t = G.threshold(sc[role == "cal_a"])
            out[f"{key}|{col}"] = {"layer": layer, "C": C, "threshold": t, "auroc": float(G.wauc(sc[ho], y[ho], w[ho])[0]),
                                   "catch": float((w[ho][pos] * (sc[ho][pos] > t)).sum() / w[ho][pos].sum()),
                                   "fa_heldout_neg": float((w[ho][~pos] * (sc[ho][~pos] > t)).sum() / w[ho][~pos].sum()),
                                   "fa_cal_a": float((sc[role == "cal_a"] > t).mean()), "fa_honb_cal": float((sc[role == "honb_cal"] > t).mean()),
                                   "fa_honb_ho": float((sc[role == "honb_ho"] > t).mean()), **({"heldout_scores": sc[ho].tolist()} if col == "0" else {})}
    return out


def choose(A):
    """A [layer, c]: (layer, C) with the best value; ties lower layer, smaller C."""
    import numpy as np

    flat = np.asarray(A)
    layer, c = divmod(int(np.flatnonzero(flat.ravel() == flat.max())[0]), flat.shape[1])
    return layer, G.CS[c]


def driver(cmd, *args):
    get = lambda name: modal.Function.from_name(APP, name)  # noqa: E731
    read = lambda p: json.loads(b"".join(G.score.read_file(p)))  # noqa: E731
    stamp = time.strftime("%H%M%S")
    if cmd == "pilot":
        print(json.dumps([get("sizing_job").remote(PILOT_LAYER), get("valid_job").remote(PILOT_LAYER, 3.0, "pilot")]))
    elif cmd == "sizing":
        calls = {l: get("sizing_job").spawn(l).object_id for l in range(G.LAYERS) if l != PILOT_LAYER}
        (HERE / "results" / f"plant2_sizing_calls_{stamp}.json").write_text(json.dumps(calls))
    elif cmd == "size-final":
        cv = {l: read(f"{D}/sizing/L{l}.json") for l in range(G.LAYERS)}
        by_layer = {}
        for key in cv[0]:
            layer, C = choose([cv[l][key] for l in range(G.LAYERS)])
            f, s = (int(x) for x in key.split("|"))
            by_layer.setdefault(layer, []).append([f, s, C])
        res = {}
        for c in [get("sizing_final").spawn(l, it) for l, it in by_layer.items()]:
            res |= c.get(timeout=1800)
        (HERE / "results" / f"plant2_sizing_{stamp}.json").write_text(json.dumps(res))
    elif cmd == "valid":
        calls = {l: get("valid_job").spawn(l, float(args[0])).object_id for l in range(G.LAYERS)}
        (HERE / "results" / f"plant2_valid_calls_{stamp}.json").write_text(json.dumps({"s": float(args[0]), "calls": calls}))
    elif cmd == "valid-final":
        s = float(args[0])
        cv = {l: read(f"{D}/valid/L{l}.json") for l in range(G.LAYERS)}
        picks, choice = {}, {}
        for col in range(101):
            layer, C = choose([cv[l]["planted"][col] for l in range(G.LAYERS)])
            picks.setdefault(layer, {}).setdefault("planted", {})[str(col)] = C
            choice[f"planted|{col}"] = [layer, C]
        for j in range(NULLS):
            layer, C = choose([cv[l]["nulls"][j] for l in range(G.LAYERS)])
            picks.setdefault(layer, {}).setdefault(f"null{j}", {})["0"] = C
            choice[f"null{j}|0"] = [layer, C]
        res = {}
        for c in [get("valid_final").spawn(l, s, p) for l, p in picks.items()]:
            res |= c.get(timeout=1800)
        (HERE / "results" / f"plant2_valid_{stamp}.json").write_text(json.dumps({"s": s, "choice": choice, "final": res}))


if __name__ == "__main__":
    driver(*sys.argv[1:])
