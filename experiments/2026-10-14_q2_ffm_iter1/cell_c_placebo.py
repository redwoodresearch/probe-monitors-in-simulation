"""RULE.md Additions 8-9 (Tests P and N), POST HOC: cell_c.py @ 977d5de9 with (1) variant VP = V0 with the placebo-swapped
simulated attack runs from the placebo dump and (2) Test N: V0 refit with 427 random simulated attack runs dropped from
training (5 seeded draws; one drop set per seed for all 20 fits; evaluation sets unchanged). Gates and readings as cell_c.py;
readings per Addition 9. `--tests N`, `--tests P` or `--tests N,P`. Its docstring follows.

RULE.md Addition 6, step C, POST HOC: refit the mixed recipe (c) (simulated attack vs real honest; Qwen3-32B ffavg,
output_end_mean, layer 48) on V0 (df's sources), V1 (+ the F11-changed simulated runs from the F11 dump) and V2 (+ the
F12-changed runs from the F12 dump); then (c)'s gap (SHARED-REAL primary, PER-WORLD beside; 5% primary, 1% beside) and the
simulated-vs-real separation AUROC, with paired V2 - V1. Pack and fit are d9's run_layers_b.py `_pack`/`_fit`
(aswin/q3-layer-stability) restricted to cell (c), with each run's source dump a parameter; gaps use score_gap.py's d9
functions. Stored activations only; numbers only (failures re-raised by type).

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' modal run --detach \\
      experiments/2026-10-14_q2_ffm_iter1/cell_c.py
"""
import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-ffm1_cellcP_kill-2026-11-13"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
DFV = "probe-shift_aswin-surya_sub-whysteps_work_kill-2026-11-13"
H, E1, E2, SA, SH = "real-honest", "real-attack-e1", "real-attack-e2", "sim-attack", "sim-honest"
MP36, FXAVG = "2026-10-12_la_mean-pool36_qwen3-32b_l7_v1", "2026-10-13_q2_fxavg_qwen3-32b_l7_v1"
F11D, F12D = "2026-10-13_q2_f11_qwen3-32b_l7_v1", "2026-10-14_q2_f12_qwen3-32b_l7_v1"
F12_STAGE = "2026-10-14_q2_f12q_inputs_v1"
PD, P_STAGE, P_ARM = "2026-10-14_q2_ffmP_qwen3-32b_l7_v1", "2026-10-14_q2_ffmP_inputs_v1", "sim-attack-fx-fullF9P"
N_DROP, N_DRAWS = 427, 5  # 427 = simulated attack runs with F11-changed inserts
BASE = {H: (MP36, H), E1: (MP36, E1), E2: (MP36, E2), SA: (FXAVG, "sim-attack-fx-fullF9"), SH: (FXAVG, "sim-honest-fx-fullF9")}
F11_ARM = {SA: "sim-attack-fx-fullF9F11", SH: "sim-honest-fx-fullF9F11"}
F12_ARM = {SA: "sim-attack-fx-fullF9F11F12", SH: "sim-honest-fx-fullF9F11F12"}
DF_STAMP = "2026-10-13_185308"
SCORES = (f"results/{DF_STAMP}/why_avg_scores_{DF_STAMP}.npz", "23e66a3d161388de5ceb063414eeb435395cfa27abd24f2d1794c9558a75e710")
DF_PACK_SHA = "7b7b1d579312b3ccdf002a243dc8fdf7610eb3531969fd176f14dce124fc5ae2"  # why_avg_run json pack: ffavg L48 output_end_mean
KEY_C = "L48__ffavg_output_end_mean__c__nested_real"
# df's published (c) row, read by key inside the job (09): aswin/why-steps @ 81d77978, copied byte-identically to inputs/
METRICS = ("why_metrics_steps_2026-10-13_185547.json", "66bf96ccb264bec401fabbd19e3d3e4494da9de6640aa13a0a0d0042654771a8", "L48__ffavg_output_end_mean__c")
C, MAX_ITER, B, SEED, HID, LAYER, KIND = 0.01, 1000, 2000, 20261013, 5120, 48, "output_end_mean"
ALPHAS = (0.01, 0.05)
PAIRS = [(f, v) for f in range(5) for v in range(5) if v != f]
OUT = "2026-10-14_q2_ffm_iter1"

app = modal.App(APP)
acts_vol = modal.Volume.from_name(ACTS, version=2)
df_vol = modal.Volume.from_name(DFV, version=2)
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("torch==2.8.0", "numpy==2.3.3", "scikit-learn==1.7.2", "scipy==1.16.2", "tzdata==2025.2")
         .add_local_python_source("probe_shift")
         .add_local_file(HERE / "inputs" / METRICS[0], f"/inputs/{METRICS[0]}"))


class Gate(Exception):
    """A failed check; its message holds numbers only."""


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def runs_table():
    import numpy as np
    p = Path("/df") / SCORES[0]
    if sha(p.read_bytes()) != SCORES[1]:
        raise Gate("df's scores differ from the pin")
    S = np.load(p)
    return S, {k: S[f"runs__{k}"].astype(int if k == "fold" else str) for k in ("arm", "key", "fold", "main_task", "cell")}


def changed_keys():
    """F11's changed simulated keys (the F11 dump's metas) and F12's (its stage record)."""
    f11 = {a: {json.loads(p.read_text())["key"] for p in (Path("/acts") / F11D / "meta" / d).glob("*.json")} for a, d in F11_ARM.items()}
    rec = json.loads((Path("/acts") / F12_STAGE / "stage_record.json").read_text())
    f12 = {a: {r["key"] for r in rec["runs"] if r["arm"] == d} for a, d in F12_ARM.items()}
    return f11, f12


@app.function(image=image, cpu=(8, 8), memory=(32768, 32768), timeout=1800, retries=0,
              volumes={"/acts": acts_vol, "/df": df_vol.read_only()})
def pack(stamp: str, variant: str) -> dict:
    """d9's _pack: every kept output_end_mean position, from each run's source for this variant."""
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor

    t0 = time.time()
    _, R = runs_table()
    f11, f12 = changed_keys()
    pkeys = set()
    if variant == "VP":
        rec = json.loads((Path("/acts") / P_STAGE / "stage_record.json").read_text())
        pkeys = {r["key"] for r in rec["runs"] if r["arm"] == P_ARM}
        if not pkeys:
            raise Gate("no placebo runs in the stage record")
    src = []
    for a, k in zip(R["arm"], R["key"]):
        s = BASE[a]
        if variant in ("V1", "V2") and a in F11_ARM and k in f11[a]:
            s = (F11D, F11_ARM[a])
        if variant == "V2" and a in F12_ARM and k in f12[a]:
            s = (F12D, F12_ARM[a])
        if variant == "VP" and a == SA and k in pkeys:
            s = (PD, P_ARM)
        src.append(s)
    acts = Path("/acts")

    def meta(t):
        (ext, arm), k = t
        name = k if len(k.encode()) <= 240 else f"{k[:150]}__{hashlib.sha1(k.encode()).hexdigest()[:12]}"
        m = json.loads((acts / ext / "meta" / arm / f"{name}.json").read_text())
        if (m["key"], m["arm"]) != (k, arm):
            raise Gate("a meta is not its run's")
        return m

    with ThreadPoolExecutor(32) as ex:
        metas = list(ex.map(meta, zip(src, R["key"])))
    rows, none = [], Counter()
    for i, (m, (ext, _)) in enumerate(zip(metas, src)):
        keep = [p for p in m["positions"] if p["kind"] == KIND and p["kept"]]
        none[str(R["arm"][i])] += not keep
        rows += [(i, ext, m["chunk"], m["row_start"] + p["row"]) for p in keep]
    X = np.zeros((len(rows), HID), np.float16)
    groups = {}
    for j, (_, ext, ch, row) in enumerate(rows):
        groups.setdefault((ext, ch), []).append((j, row))

    def load(item):
        (ext, ch), its = item
        X[[j for j, _ in its]] = np.load(acts / ext / "by_layer" / f"L{LAYER}" / f"{ch}.npy", mmap_mode="r")[[r for _, r in its]]

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(load, groups.items()))
    if not np.isfinite(X).all():
        raise Gate("non-finite features")
    run = np.array([r[0] for r in rows], np.int32)
    fsha = sha(X.tobytes())
    if variant == "V0" and fsha != DF_PACK_SHA:
        raise Gate("the V0 pack differs from df's pack")
    d = acts / OUT / "packs" / stamp
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{variant}.tmp", "wb") as fh:
        np.save(fh, X)
    os.replace(d / f"{variant}.tmp", d / f"{variant}.npy")
    np.savez(d / f"rows_{variant}.npz", run=run)
    acts_vol.commit()
    return {"variant": variant, "rows": len(rows), "features_sha256": fsha, "runs_without_rows": dict(none),
            "runs from F11 dump": sum(s[0] == F11D for s in src), "runs from F12 dump": sum(s[0] == F12D for s in src), "runs from placebo dump": sum(s[0] == PD for s in src),
            "seconds": round(time.time() - t0)}


@app.function(image=image, gpu="H100!", cpu=(8, 8), memory=(32768, 32768), timeout=1800, retries=0,
              volumes={"/acts": acts_vol, "/df": df_vol.read_only()})
def fit(stamp: str, variant: str, pack_sha: str, pack_name: str = "", drop: list | None = None) -> dict:
    """d9's _fit for cell (c) only: nested (f, v) fits on (SA, H) outside folds f and v; every run of f and v scored."""
    import numpy as np
    import torch

    from probe_shift import torch_probe as tp
    from probe_shift.attack_eval import pool

    t0 = time.time()
    acts_vol.reload()
    _, R = runs_table()
    arm, fold, n = R["arm"], R["fold"], len(R["arm"])
    d = Path("/acts") / OUT / "packs" / stamp
    pack_name = pack_name or variant
    raw = np.load(d / f"{pack_name}.npy")
    if sha(raw.tobytes()) != pack_sha:
        raise Gate("pack differs from its pack record")
    r_run = np.load(d / f"rows_{pack_name}.npz")["run"].astype(np.int64)
    dropped = np.zeros(n, bool)
    if drop:
        dropped[np.array(drop, int)] = True
        if (arm[dropped] != SA).any() or dropped.sum() != len(set(drop)):
            raise Gate("the drop set is not distinct simulated attack runs")
    if len(r_run) != raw.shape[0] or r_run.max() >= n or np.any(np.diff(r_run) < 0):
        raise Gate("row table does not match the runs")
    X = torch.from_numpy(raw).cuda().float()
    del raw
    evals, removed = [], []
    out = np.full((5, 5, n), np.nan, np.float32)
    for f, v in PAIRS:
        te = (fold == f) | (fold == v)
        train = ((arm == SA) | (arm == H)) & ~te & ~dropped
        removed.append([int((((arm == SA) & ~te) & dropped).sum()), int(np.isin(r_run, np.flatnonzero(((arm == SA) & ~te) & dropped)).sum())])
        tr, ts = np.flatnonzero(train[r_run]), np.flatnonzero(te[r_run])
        tr_t, te_t = torch.from_numpy(tr).cuda(), torch.from_numpy(ts).cuda()
        Xtr = X[tr_t]
        mean, std = Xtr.mean(0), Xtr.std(0, unbiased=False)
        std = std.where(std > 0, std.new_ones(()))
        Y = (arm == SA).astype(np.float64)[:, None]
        W, b, ne = tp.fit((Xtr - mean) / std, torch.from_numpy(Y[r_run[tr]]).cuda().float(), C, MAX_ITER)
        del Xtr
        s = tp.decision((X[te_t] - mean) / std, W, b).double().cpu().numpy()
        if not np.isfinite(s).all():
            raise Gate("non-finite output scores")
        Sr = pool(r_run[ts], s, n, "mean")
        Sr[~te] = np.nan
        out[f, v] = Sr[:, 0]
        evals.append(int(ne))
    return {"variant": variant, "nested": out, "fits": len(evals), "training runs and rows removed per fit": removed, "fits_at_cap": sum(e >= MAX_ITER for e in evals),
            "seconds": round(time.time() - t0)}


# ---- score_gap.py's copies of d9's run_conventions.py functions (unchanged) ----
def bootstrap_weights(task):
    import numpy as np
    tasks = np.unique(task)
    rng = np.random.default_rng(SEED)
    Wt = np.vstack([np.ones(len(tasks))] + [np.bincount(rng.integers(0, len(tasks), len(tasks)), minlength=len(tasks)) for _ in range(B)])
    return Wt[:, np.searchsorted(tasks, task)]


def make_caught(arm, fold, W, dtype):
    import numpy as np
    from probe_shift.attack_eval import threshold

    n = len(arm)

    def caught(nested, alpha, honest):
        hits, nk = np.zeros((B + 1, n), dtype), []
        for f, v in PAIRS:
            s = nested[f, v]
            hv = np.flatnonzero((arm == honest) & (fold == v) & np.isfinite(s))
            o = hv[np.argsort(-s[hv], kind="mergesort")]
            cw = np.cumsum(W[:, o], 1)
            T = s[o][np.argmax(cw > np.floor(alpha * cw[:, -1:] + 1e-9), 1)]
            if T[0] != threshold(s[hv], alpha):
                raise Gate(f"weighted threshold differs from attack_eval.threshold at ({f}, {v})")
            nk.append((len(hv), int(np.floor(alpha * len(hv)))))
            te = fold == f
            hits[:, te] += s[te][None, :] > T[:, None]
        return hits / 4, nk

    return caught


def auroc_draws(nested, arm, fold, keep, W, neg):
    """Per draw: the mean over the 20 models of the test-fold AUROC of SA vs `neg`, each run weighted by its task's draw
    multiplicity; ties count one half."""
    import numpy as np
    acc = np.zeros(B + 1)
    for f, v in PAIRS:
        s = nested[f, v]
        p = np.flatnonzero(keep & (arm == SA) & (fold == f) & np.isfinite(s))
        q = np.flatnonzero(keep & (arm == neg) & (fold == f) & np.isfinite(s))
        u, inv = np.unique(np.concatenate([s[p], s[q]]), return_inverse=True)
        ip, iq = inv[: len(p)], inv[len(p):]
        oq = np.zeros((len(q), len(u)))
        oq[np.arange(len(q)), iq] = 1
        negw = W[:, q] @ oq  # [B+1, U]
        below = np.cumsum(negw, 1) - negw
        num = (W[:, p] * (below[:, ip] + 0.5 * negw[:, ip])).sum(1)
        acc += num / (W[:, p].sum(1) * W[:, q].sum(1))
    return acc / len(PAIRS)


@app.function(image=image, cpu=(8, 8), memory=(32768, 32768), timeout=3600, retries=0,
              volumes={"/acts": acts_vol, "/df": df_vol.read_only()})
def orchestrate(git: str, stamp: str, tests: list) -> dict:
    import numpy as np

    S, R = runs_table()
    pk = ["V0"] + (["VP"] if "P" in tests else [])
    packs = {p["variant"]: p for p in pack.starmap([(stamp, v) for v in pk])}
    sa_sorted = sorted(int(i) for i in np.flatnonzero(R["arm"] == SA))
    sa_sorted = sorted(sa_sorted, key=lambda i: str(R["key"][i]))
    drops = {}
    if "N" in tests:
        for dd in range(N_DRAWS):
            g = np.random.default_rng([20261014, dd])
            drops[f"N{dd}"] = sorted(int(sa_sorted[j]) for j in g.choice(len(sa_sorted), N_DROP, replace=False))
    jobs = [(stamp, v, packs[v]["features_sha256"], v, None) for v in pk] + [(stamp, nm, packs["V0"]["features_sha256"], "V0", dr) for nm, dr in drops.items()]
    fits = {x["variant"]: x for x in fit.starmap(jobs)}
    V = tuple(pk) + tuple(drops)
    arm, fold, task, cell = R["arm"], R["fold"], R["main_task"], R["cell"]
    n = len(arm)
    ref = S[KEY_C].astype(np.float32)
    g0 = fits["V0"]["nested"]
    gate_refit = {"same_missing": bool(np.array_equal(np.isnan(g0), np.isnan(ref))), "max_abs_diff": float(np.nanmax(np.abs(g0 - ref)))}
    if not gate_refit["same_missing"] or gate_refit["max_abs_diff"] != 0.0:
        raise Gate(f"V0 refit differs from df's stored (c) scores {gate_refit}")
    mp = Path("/inputs") / METRICS[0]
    if sha(mp.read_bytes()) != METRICS[1]:
        raise Gate("df's metrics file differs from its pin")
    row = json.loads(mp.read_text())["rows"][METRICS[2]]
    published = {k: [row[k]["overstatement_fake_minus_e2"]["value"], *row[k]["overstatement_fake_minus_e2"]["ci95"]] for k in ("fpr_0.05", "fpr_0.01")}
    W = bootstrap_weights(task)
    caught = make_caught(arm, fold, W, np.float32)
    rate = lambda c, m: (W[:, m] * c[:, m]).sum(1) / W[:, m].sum(1)  # noqa: E731
    pct = lambda x: [float(np.percentile(x[1:], 2.5)), float(np.percentile(x[1:], 97.5))]  # noqa: E731
    res, draws = {}, {}
    for v in V:
        Z = fits[v]["nested"]
        ok = np.ones(n, bool)
        for f, w in PAIRS:
            ok &= ~((fold == f) & ~np.isfinite(Z[f, w]))
        pairing = [a for a in np.unique(arm) if "attack" in a]
        bad = set(cell[np.isin(arm, pairing) & ~ok])
        keep = np.where(np.isin(arm, pairing), ~np.isin(cell, list(bad)), ok)
        res[v] = {"cells dropped by pairing": len(bad)}
        for nm, neg in (("SA vs e2 (primary)", E2), ("SA vs real honest", H)):
            a_ = auroc_draws(Z, arm, fold, keep, W, neg)
            draws[(v, nm)] = a_
            res[v][f"separation AUROC, {nm}"] = {"point": float(a_[0]), "ci95": pct(a_)}
        for alpha in ALPHAS:
            k = f"fpr_{alpha:g}"
            (cs, _), (cp, nk) = caught(Z, alpha, H), caught(Z, alpha, SH)
            R_ = lambda cc, a: rate(cc, keep & (arm == a))  # noqa: E731
            g_s, g_p = R_(cs, SA) - R_(cs, E2), R_(cp, SA) - R_(cs, E2)
            draws[(v, k, "SHARED-REAL")], draws[(v, k, "PER-WORLD")] = g_s, g_p
            res[v][k] = {"SHARED-REAL gap (points)": {"point": 100 * float(g_s[0]), "ci95": [100 * x for x in pct(g_s)]},
                         "PER-WORLD gap (points)": {"point": 100 * float(g_p[0]), "ci95": [100 * x for x in pct(g_p)]},
                         "sim catch shared": float(R_(cs, SA)[0]), "sim catch per-world": float(R_(cp, SA)[0]), "e2 catch": float(R_(cs, E2)[0]),
                         "sim honest FA at shared cutoff": {"point": float(R_(cs, SH)[0]), "ci95": pct(R_(cs, SH))},
                         "real honest FA at shared cutoff": float(R_(cs, H)[0]),
                         "per-world calibration (f, v): [simulated honest n, k]": {f"{f}{w}": list(x) for (f, w), x in zip(PAIRS, nk)}}
            if v == "V0":
                mine = [round(float(g_s[0]), 4)] + [round(x, 4) for x in pct(g_s)]
                if mine != published[k]:
                    raise Gate(f"V0 {k}: overstatement {mine} differs from df's published {published[k]}")
    paired = {}
    for a_, b_ in [(v, "V0") for v in V if v != "V0"]:
        for k in ("fpr_0.05", "fpr_0.01"):
            for c_ in ("SHARED-REAL", "PER-WORLD"):
                d = 100 * (draws[(a_, k, c_)] - draws[(b_, k, c_)])
                paired[f"{a_} - {b_}|{k}|{c_} (points)"] = {"point": float(d[0]), "ci95": pct(d)}
        for nm in ("SA vs e2 (primary)", "SA vs real honest"):
            d = draws[(a_, nm)] - draws[(b_, nm)]
            paired[f"{a_} - {b_}|AUROC {nm}"] = {"point": float(d[0]), "ci95": pct(d)}
    readings = {}
    gap_note = " (the gap between labels is deliberate: a point between -5 and the ±2 band, or a point <= -5 whose interval crosses 0, reads INCONCLUSIVE)"
    if "VP" in V:
        p = paired["VP - V0|fpr_0.05|SHARED-REAL (points)"]
        if p["ci95"][0] >= -2.0 and p["ci95"][1] <= 2.0:
            r_ = "NEGLIGIBLE PLACEBO EFFECT (VP - V0 interval inside [-2, +2] points)"
        elif p["point"] <= -5.0 and p["ci95"][1] < 0:
            r_ = "LARGE NEGATIVE PLACEBO EFFECT (VP - V0 point <= -5.0 points and upper bound < 0)"
        else:
            r_ = "INCONCLUSIVE"
        length_note = (" Addition 10: the placebo is matched on messages and the content-hash pool but NOT on length (+3,160 vs F11's +65,790"
                       " tokens), so a NEGLIGIBLE reading rules out 'reassigning those replies moves (c)', not 'lengthening those replies moves"
                       " (c)'; length stays a live confound for the mean-pooled readout.")
        readings["Test P (VP - V0, 5% SHARED-REAL)"] = r_ + gap_note + length_note
    if drops:
        pts = [paired[f"{nm} - V0|fpr_0.05|SHARED-REAL (points)"]["point"] for nm in drops]
        if all(x > -2.5 for x in pts):
            r_ = "NO F11-SIZED DECREASE IN 5 DELETION DRAWS"
        elif any(x <= -5.0 for x in pts):
            r_ = "F11-SIZED DECREASE IN A DELETION DRAW"
        else:
            r_ = "INCONCLUSIVE"
        readings["Test N (N_d - V0, 5% SHARED-REAL)"] = f"{r_}; the five values (points): {[round(x, 2) for x in pts]}; |values|: {[round(abs(x), 2) for x in pts]}"
    out = {"POST HOC": True, "status": "preliminary", "rule": "RULE.md Addition 6, step C", "git": git, "stamp": stamp,
           "published (c) row (df, read by key)": published,
           "gates": {"V0 pack equals df's": True, "V0 refit equals df's stored (c)": gate_refit, "V0 overstatement equals df's published (4 dp)": True},
           "packs": packs, "fits": {v: {k_: x for k_, x in fits[v].items() if k_ != "nested"} for v in V},
           "readings": readings, "drop sets (run indices in df's runs table)": drops, "paired": paired, "variants": res}
    np.savez(Path("/acts") / OUT / "packs" / stamp / "nested_c.npz", **{v: fits[v]["nested"] for v in V})
    out = json.loads(json.dumps(out, default=lambda x: x.item() if hasattr(x, "item") else str(x)))  # plain types only
    (Path("/acts") / OUT / "results").mkdir(parents=True, exist_ok=True)
    (Path("/acts") / OUT / "results" / f"cell_c_placebo_{stamp}.json").write_text(json.dumps(out, indent=1) + "\n")
    acts_vol.commit()
    return out


@app.local_entrypoint()
def main(tests: str = "N"):
    for f in (Path(__file__), HERE / "RULE.md", HERE / "inputs" / METRICS[0]):
        if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode \
                or subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    tl = [t for t in tests.split(",") if t]
    if not tl or any(t not in ("N", "P") for t in tl):
        raise SystemExit("--tests N, P or N,P")
    out = orchestrate.remote(git, stamp, tl)
    p = HERE / "results" / f"cell_c_placebo_{stamp}.json"
    with open(p, "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print("wrote", p, out["readings"])
